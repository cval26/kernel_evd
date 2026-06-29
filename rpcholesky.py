import functools
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as Pspec

def kernel_gaussian(ui, u, sqnorm_i, sqnorms, eps):
    """Compute a column of the Gaussian kernel matrix
    K[:, i] = exp(-||ui - u[:, :]||^2 / eps),
    using the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a^T b.

    Inputs:
      ui       : (Nq,) data point
      u        : (Nq, N) data matrix
      sqnorm_i : scalar, ||ui||^2
      sqnorms  : (N,) column-wise squared norms of u
      eps      : kernel bandwidth
    Output:
      (N,) kernel column

    Inside shard_map, this is invoked per-device with u and sqnorms being
    the local shards (shape (Nq, N_local) and (N_local,)) and ui, sqnorm_i
    replicated; it then returns the device's local slice of the kernel column.
    """
    temp = sqnorm_i + sqnorms - 2.0 * (ui @ u)
    temp = jnp.maximum(temp, 0.0)
    return jnp.exp(-temp / eps)

def kernel_gaussian_block(uS, u, sqnorms_S, sqnorms, eps):
    """Compute a block of rows of the Gaussian kernel matrix K[S, :]
    of shape (|S|, N),
    with K[l, j] = exp(-||uS[:, l] - u[:, j]||^2 / eps),
    using the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a^T b.

    Inputs:
      uS        : (Nq, |S|) data points indexing the rows of the block
      u         : (Nq, N)  data matrix
      sqnorms_S : (|S|,)   squared norms of uS columns
      sqnorms   : (N,)     column-wise squared norms of u
      eps       : kernel bandwidth

    Output:
      (|S|, N) kernel block

    Inside shard_map, when u and sqnorms are local shards (last axis sharded)
    and uS, sqnorms_S are replicated, the returned block is sharded along its
    last axis.
    When called with u = uS, sqnorms = sqnorms_S (both replicated)
    the returned (|S|, |S|) block is computed locally and replicated.
    """
    inner = uS.T @ u
    sqdists = sqnorms_S[:, None] + sqnorms[None, :] - 2.0 * inner
    sqdists = jnp.maximum(sqdists, 0.0)
    return jnp.exp(-sqdists / eps)

@functools.partial(jax.jit, static_argnames=('Nr', 'b', 'mesh'))
def arpc(u, eps, Nr, b, key, mesh):
    """Accelerated randomly pivoted Cholesky algorithm, sharded across
    multiple GPUs along the data point axis.

    Rejected slots are filled with zero rows of F (so F^T F is
    unaffected) and the corresponding ind entries are set to -1.

    Input:
      u    : (Nq, N) data matrix, sharded as Pspec(None, 'x')
      eps  : kernel bandwidth
      Nr   : maximum approximation rank, must be divisible by b
      b    : block size
      key  : JAX PRNG key (replicated)
      mesh : 1-D device mesh with axis name 'x'

    Output:
      F   : (Nr, N) Cholesky factor, sharded as Pspec(None, 'x'); rows
            corresponding to rejected pivots are zero
      ind : (Nr,) selected pivot indices, replicated; -1 at rejected slots
      err : (t,)  per-round residual trace history, replicated; t = Nr / b
    """
    N = u.shape[1]
    Ndevs = mesh.shape['x']

    assert N % Ndevs == 0, f"N={N} must be divisible by number of devices Ndevs={Ndevs}"
    assert Nr % b == 0, f"Nr={Nr} must be divisible by block size b={b}"

    t = Nr // b # number of rounds

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(Pspec(None, 'x'), Pspec()),
        out_specs=(Pspec(None, 'x'), Pspec(), Pspec()),
        check_vma=False,
    )
    def _body(u_local, key):
        # Per-device shapes:
        #   u_local       : (Nq, N_local)
        #   sqnorms_local : (N_local,)
        #   dvec_local    : (N_local,)
        #   F_local       : (Nr, N_local)
        #   ind, err      : (Nr,), (t,) -- replicated, identical on every device
        N_local = u_local.shape[1]
        device_id = jax.lax.axis_index('x')

        F_local = jnp.zeros((Nr, N_local), dtype=jnp.float32)
        ind = jnp.full((Nr,), -1, dtype=jnp.int32)
        err = jnp.zeros((t,), dtype=jnp.float32)

        # Squared norms of all data points (locally).
        sqnorms_local = jnp.sum(u_local**2, axis=0)

        # Initial residual diagonal: K(j, j) = 1 for the Gaussian kernel.
        dvec_local = jnp.ones((N_local,), dtype=jnp.float32)

        def gather_columns(arr_local, pvts):
            """Conditional gather of arr_local[..., pvts]
            where pvts is a (b,) vector of global column indices, replicated.
            Each device looks up its locally-owned entries; non-owners
            contribute zeros; psum assembles the (..., b) result,
            replicated across all devices.

            arr_local must have its column axis as the last axis; works for
            shapes (Nq, N_local), (Nr, N_local), and (N_local,).
            """
            local_pvts = pvts - device_id * N_local              # (b,)
            is_owner = (pvts // N_local) == device_id            # (b,)
            local_pvts_safe = jnp.clip(local_pvts, 0, N_local - 1)
            values = jnp.take(arr_local, local_pvts_safe, axis=-1)
            values = jnp.where(is_owner, values, jnp.zeros_like(values))
            return jax.lax.psum(values, 'x')

        def rejection_sample(H, u_init, rand_vals):
            """Rejection-sampling loop of Algorithm 2.1 on the b x b
            residual submatrix H, with proposal-distribution diagonal u_init
            and per-step uniform draws rand_vals (both shape (b,)).

            Returns:
              L           : (b, b) lower-triangular factor with diagonal 1 at
                            rejected slots (so it is invertible). On the
                            accepted x accepted block, L L^* = A(S, S).
              accept_mask : (b,) boolean mask of accepted proposals.
            """
            L0 = jnp.zeros((b, b), dtype=H.dtype)
            accept_mask0 = jnp.zeros((b,), dtype=bool)
            idx = jnp.arange(b)

            def step(i, carry):
                H, L, accept_mask = carry
                h_ii = H[i, i]
                # Acceptance test: rand * u_init[i] < h_ii.
                accept = rand_vals[i] * u_init[i] < h_ii

                sqrt_h = jnp.sqrt(jnp.maximum(h_ii, 0.0))
                inv_sqrt_h = jnp.where(sqrt_h > 0, 1.0 / sqrt_h, 0.0)

                # Cholesky column: H[i:, i] / sqrt(h_ii), zero above i.
                col = jnp.where(idx >= i, H[:, i] * inv_sqrt_h, 0.0)

                # Mask the entire column to zero on rejection.
                col_masked = col * accept.astype(col.dtype)

                # Write column i of L; override the diagonal to 1
                # on rejection, so L stays invertible.
                L = L.at[:, i].set(col_masked)
                L = L.at[i, i].set(jnp.where(accept, sqrt_h, 1.0))

                # Cholesky update of the trailing block.
                H = H - jnp.outer(col_masked, col_masked)
                accept_mask = accept_mask.at[i].set(accept)
                return (H, L, accept_mask)

            _, L, accept_mask = jax.lax.fori_loop(0, b, step, (H, L0, accept_mask0))
            return L, accept_mask

        def round_body(round_idx, carry):
            F_local, ind, err, dvec_local, key = carry

            # Propose b iid pivots from the current residual diagonal.
            # Replicated cumsum + searchsorted using a replicated key;
            # every device produces the same Sprime.
            key, subkey_p = jax.random.split(key)
            dvec_full = jax.lax.all_gather(dvec_local, 'x', tiled=True)  # (N,)
            dsum = jnp.sum(dvec_full)
            r_vals = jax.random.uniform(subkey_p, (b,), dtype=jnp.float32) * dsum
            cum = jnp.cumsum(dvec_full)
            Sprime = jnp.searchsorted(cum, r_vals).astype(jnp.int32)     # (b,)

            # Gather data points and current factor at proposed pivots.
            u_at_pvts = gather_columns(u_local, Sprime)         # (Nq, b),  replicated
            sqnorms_at_pvts = jnp.sum(u_at_pvts**2, axis=0)     # (b,),     replicated
            F_at_pvts = gather_columns(F_local, Sprime)         # (Nr, b),  replicated

            # Build H = A(S', S') - F(S', :) F(S', :)^*
            # F is stored as the paper's F^*, so paper's
            # F(S', :) F(S', :)^* = F_at_pvts^T @ F_at_pvts.
            K_pp = kernel_gaussian_block(
                u_at_pvts, u_at_pvts, sqnorms_at_pvts, sqnorms_at_pvts, eps)  # (b, b)
            H = K_pp - F_at_pvts.T @ F_at_pvts                  # (b, b),   replicated

            # Rejection sampling on H.
            # Replicated subkey so every device computes the same L and
            # accept_mask without further collectives.
            key, subkey_r = jax.random.split(key)
            rand_vals = jax.random.uniform(subkey_r, (b,), dtype=jnp.float32)
            u_init = jnp.diag(H) # (b,)
            L, accept_mask = rejection_sample(H, u_init, rand_vals)

            # Residual block at the proposed pivots
            # R has shape (b, N_local) (rows are pivots, sharded along
            # data points), and equals K(S', :) - F(:, S')^T @ F in our
            # storage convention.
            K_at_pvts = kernel_gaussian_block(
                u_at_pvts, u_local, sqnorms_at_pvts, sqnorms_local, eps)  # (b, N_local)
            R_local = K_at_pvts - F_at_pvts.T @ F_local                    # (b, N_local)

            # Solve L G = R, then mask rejected rows.
            # Paper's G = R^paper L^{-*} of shape (N, b); in our convention
            # this becomes G = L^{-1} R, of shape (b, N_local).
            # With our padded L, the rows of G at accepted slots equal
            # the columns of paper's G; rows at rejected slots come back
            # equal to R[rejected, :], which we discard via the accept mask.
            G_local = jax.scipy.linalg.solve_triangular(L, R_local, lower=True)
            mask = accept_mask.astype(G_local.dtype)            # (b,)
            G_local = G_local * mask[:, None]

            # Write G into F at rows [round_idx*b, (round_idx+1)*b)
            F_local = jax.lax.dynamic_update_slice(
                F_local, G_local, (round_idx * b, 0))

            # Update the residual diagonal.
            # Rejected rows of G are zero, so they contribute nothing.
            # In our convention, paper's SquaredRowNorms(G) is the
            # column-wise sum of squares of our G.
            dvec_local = jnp.maximum(
                dvec_local - jnp.sum(G_local**2, axis=0), 0.0)

            # Record indices and per-round error.
            ind_block = jnp.where(accept_mask, Sprime, -1).astype(jnp.int32)
            ind = jax.lax.dynamic_update_slice(ind, ind_block, (round_idx * b,))

            dsum_after = jax.lax.psum(jnp.sum(dvec_local), 'x')
            err = err.at[round_idx].set(dsum_after / N)

            return (F_local, ind, err, dvec_local, key)

        F_local, ind, err, dvec_local, key = jax.lax.fori_loop(
            0, t, round_body, (F_local, ind, err, dvec_local, key))

        return F_local, ind, err

    return _body(u, key)

@functools.partial(jax.jit, static_argnames=('Nr', 'mesh'))
def rpc(u, eps, Nr, key, mesh, prepvts=None):
    """Compute the partial Cholesky factor of a kernel matrix
    using the randomly pivoted Cholesky algorithm, sharded across
    multiple GPUs along the data-point axis.

    Input:
      u       : (Nq, N) data matrix, sharded as Pspec(None, 'x')
      eps     : kernel bandwidth
      Nr      : target rank
      key     : JAX PRNG key (replicated)
      mesh    : 1-D device mesh with axis name 'x'
      prepvts : (Nr,) pre-sampled pivots, replicated (optional)

    Output:
      F    : (Nr, N) Cholesky factor, sharded as Pspec(None, 'x')
      ind  : (Nr,) selected pivot indices, replicated
      err  : (Nr,) residual trace history, replicated
    """
    N = u.shape[1]
    Ndevs = mesh.shape['x']
    assert N % Ndevs == 0, f"N={N} must be divisible by number of devices Ndevs={Ndevs}"

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(Pspec(None, 'x'), Pspec(), Pspec()),
        out_specs=(Pspec(None, 'x'), Pspec(), Pspec()),
        check_vma=False,)
    def _body(u_local, key, prepvts=None):
        # Per-device shapes:
        #   u_local      : (Nq, N_local)
        #   sqnorms_local: (N_local,)
        #   dvec_local   : (N_local,)
        #   F_local      : (Nr, N_local)
        #   ind, err     : (Nr,) -- replicated, identical on every device
        N_local = u_local.shape[1]
        device_id = jax.lax.axis_index('x')

        F_local = jnp.zeros((Nr, N_local), dtype=jnp.float32)
        ind = jnp.zeros((Nr,), dtype=jnp.int32)
        err = jnp.zeros((Nr,), dtype=jnp.float32)

        # Precompute squared norms of all data points (locally).
        sqnorms_local = jnp.sum(u_local**2, axis=0)

        dvec_local = jnp.ones((N_local,), dtype=jnp.float32)

        # Range vector used to build the per-iteration mask.
        rows = jnp.arange(Nr)

        def gather_at(arr_local, pvt):
            """Conditional gather of arr_local[..., pvt] where pvt is a global
            column index. The owning device reads its local entry; non-owners
            contribute zeros; psum broadcasts the result so every device ends
            up with the same value (replicated).

            arr_local must have its column axis as the last axis; works for
            shapes (Nq, N_local), (Nr, N_local), and (N_local,).
            """
            local_pvt = pvt - device_id * N_local
            is_owner = (pvt // N_local) == device_id

            local_pvt_safe = jnp.clip(local_pvt, 0, N_local - 1)
            value = jnp.take(arr_local, local_pvt_safe, axis=-1)
            value = jnp.where(is_owner, value, jnp.zeros_like(value))

            return jax.lax.psum(value, 'x')

        def body_loop(i, carry):
            F_local, ind, err, dvec_local, key, prepvts = carry

            # Sample pvt by inverse-CDF on the residual diagonal dvec.
            # Gather dvec to every device, then run an identical local cumsum
            # and searchsorted. Since `key` is replicated, `r` and therefore
            # `pvt` are identical on every device.
            key, subkey = jax.random.split(key)

            dvec_full = jax.lax.all_gather(dvec_local, 'x', tiled=True)
            dsum = jnp.sum(dvec_full)

            r = jax.random.uniform(subkey, dtype=jnp.float32) * dsum
            pvt = jnp.searchsorted(jnp.cumsum(dvec_full), r)

            # If presampled pivots available, use those instead
            if prepvts is not None:
                pvt = prepvts[i]

            ind = ind.at[i].set(pvt)

            # Conditional gather for the pivot's column data.
            ui = gather_at(u_local, pvt) # (Nq,), replicated

            # ui is replicated, so its squared norm can be computed locally
            # on every device without a collective.
            sqnorm_i = ui @ ui # scalar, replicated

            # Each device computes its slice of the kernel column locally.
            col_local = kernel_gaussian(
                ui, u_local, sqnorm_i, sqnorms_local, eps) # (N_local,)

            # Cholesky update. The contraction is along the (replicated) Nr
            # axis, so the matmul is local: (Nr,) @ (Nr, N_local) -> (N_local,).
            f_pvt = gather_at(F_local, pvt)  # (Nr,), replicated
            mask = (rows < i).astype(F_local.dtype)
            col_local = col_local - (f_pvt * mask) @ F_local

            # Normalize by the square root of the residual diagonal
            # at the pivot.
            col_pvt = gather_at(col_local, pvt)  # scalar, replicated
            sqrt_norm = jnp.sqrt(jnp.maximum(col_pvt, 0.0))
            isqrtnorm = jnp.where(sqrt_norm > 0, 1.0 / sqrt_norm, 0.0)
            col_local = col_local * isqrtnorm

            F_local = F_local.at[i, :].set(col_local)

            # Update the residual diagonal locally.
            dvec_local = jnp.maximum(dvec_local - col_local**2, 0.0)

            # Record the global trace error after this iteration.
            dsum = jax.lax.psum(jnp.sum(dvec_local), 'x')
            err = err.at[i].set(dsum / N)

            return (F_local, ind, err, dvec_local, key, prepvts)

        F_local, ind, err, dvec_local, key, prepvts = jax.lax.fori_loop(
            0, Nr, body_loop, (F_local, ind, err, dvec_local, key, prepvts))

        return F_local, ind, err

    return _body(u, key, prepvts)
