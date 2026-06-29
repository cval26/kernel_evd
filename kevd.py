import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import functools
import numpy as np

import jax
jax.config.update('jax_default_matmul_precision', 'highest')
jax.config.update("jax_compilation_cache_dir", "data/cache")

import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec as Pspec, NamedSharding

from rpcholesky import arpc

def main():
    Lfact = 7 # domain length factor
    Nq = 64 # delays
    N = 2048 # temporal space dim
    M = 64 # spatial space dim
    NM = N * M # product space dim

    Nr = 16384  # approximation rank
    bsize = 64 # sampling block size

    L = Nr # spectral resolution
    eps = 32 # kernel bandwidth
    seed = 0 # RNG seed

    # GPU IDs based on those seen by JAX, which are controlled via the
    # CUDA_VISIBLE_DEVICES environment variable
    device_ids = [0, 1]

    # GPU IDs and 1D device mesh
    selected = [jax.devices('gpu')[i] for i in device_ids]
    Ndevs = len(selected)
    mesh = Mesh(np.array(selected).reshape(Ndevs), ('x',))

    # Read input data and place it on the mesh
    # u_dev is (Nq, NM), sharded along NM
    u = np.load(f"data/ks_train_{NM}_{Nq}_7_udelay32.npy")
    assert u.shape[0] == Nq and u.shape[1] == NM

    u_sharding = NamedSharding(mesh, Pspec(None, 'x'))
    u_dev = jax.device_put(jnp.asarray(u), u_sharding)

    key = jax.random.PRNGKey(seed)

    t0 = time.time()

    # Compute the partial Cholesky factorization
    # F is (Nr, NM), sharded along NM
    # ind and err are (Nr,) replicated
    F_dev, ind_dev, err_dev = arpc(u_dev, eps, Nr, bsize, key, mesh)
    F_dev.block_until_ready()

    t1 = time.time()
    tdelta = np.round(t1-t0, decimals=3)
    print(f"ARPC: {tdelta} sec.")

    # Approximate EVD
    Phi_dev, eig_dev = approx_evd(F_dev, mesh) 
    Phi_dev.block_until_ready()

    t2 = time.time()
    tdelta = np.round(t2-t1, decimals=3)
    print(f"EVD: {tdelta} sec.")

    tdelta = np.round(t2-t0, decimals=3)
    print(f"Total: {tdelta} sec.")

    err = np.asarray(err_dev)
    np.savetxt(f"data/arpc_err_{NM}_{Nr}.dat", err, fmt="%.8e")

    ind = np.asarray(ind_dev)
    print(ind[ind != -1].size)

    Phi = np.asarray(Phi_dev) # Nr x NM
    eig = np.asarray(eig_dev) # Nr

    np.save(f"data/ks_phi_{NM}_{L}_{Lfact}_{eps}.npy", Phi)
    np.save(f"data/ks_eig_{NM}_{L}_{Lfact}_{eps}.npy", eig)
    return None

def cholesky_qr2(M_local, axis_name):
    """Two-pass CholeskyQR of a sharded matrix.
    Must be called from inside shard_map; issues psum collectives
    over the input axis name.

    Input:
      M_local   : (Nr, N_local) local shard of a global matrix (Nr, N),
                  sharded along N.
      axis_name : mesh axis name to reduce over.

    Output:
      Q_local   : (Nr, N_local) shard of global Q, Q @ Q.T = I.
      R         : (Nr, Nr) upper triangular factor, replicated.

    Such that global M = R.T @ Q (in row major order).
    """
    Nr = M_local.shape[0]
    eye = jnp.eye(Nr, dtype=M_local.dtype)

    # Pass 1
    G1 = jax.lax.psum(M_local @ M_local.T, axis_name)
    jitter1 = 1e-7 * jnp.trace(G1) / Nr
    L1 = jnp.linalg.cholesky(G1 + jitter1 * eye)
    Q_tmp = jax.scipy.linalg.solve_triangular(L1, M_local, lower=True)

    # Pass 2
    G2 = jax.lax.psum(Q_tmp @ Q_tmp.T, axis_name)
    jitter2 = 1e-7 * jnp.trace(G2) / Nr
    L2 = jnp.linalg.cholesky(G2 + jitter2 * eye)
    Q_local = jax.scipy.linalg.solve_triangular(L2, Q_tmp, lower=True)

    R = (L1 @ L2).T
    return Q_local, R

@functools.partial(jax.jit, static_argnames=('mesh',))
def approx_evd(F, mesh):
    """Compute the eigenvalue decomposition of the bistochastic normalized
    kernel P from a partial Cholesky factorization K = F.T @ F.

    Input:
      F    : (Nr, N) sharded as Pspec(None, 'x').
      mesh : 1D device mesh.

    Output:
      Phi    : (Nr, N) sharded as Pspec(None, 'x').
      Lambda : (Nr,) in descending order, replicated.
    """
    Nr, N = F.shape

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(Pspec(None, 'x'),),
        out_specs=(Pspec(None, 'x'), Pspec()),
        check_vma=False,
    )
    def _body(F_local):
        # Diagonal normalization D = diag(K 1_N).
        t = jax.lax.psum(jnp.sum(F_local, axis=1), 'x') # (Nr,) replicated
        D_local = t @ F_local # (N_local,) sharded
        D_inv_local = 1.0 / D_local

        # Diagonal normalization Q = diag(K D^-1 1_N).
        s = jax.lax.psum(F_local @ D_inv_local, 'x') # (Nr,) replicated
        Q_local = s @ F_local # (N_local,) sharded
        Q_inv_local = 1.0 / Q_local

        # B^T = F D^{-1} (column scaling of F by 1/D).
        # (Nr, N_local) sharded
        B_T_local = F_local * D_inv_local[None, :]

        # C = F Q^{-1} F^T, (Nr, Nr) replicated
        C = jax.lax.psum((F_local * Q_inv_local[None, :]) @ F_local.T, 'x')

        # CholeskyQR2 of B^T gives Q row-orthonormal (globally)
        # and R upper triangular such that B^T = R^T @ Q.
        Q_local, R = cholesky_qr2(B_T_local, 'x')

        # Small inner EVD of R C R^T.
        inner = R @ C @ R.T # (Nr, Nr) replicated
        Lambda, U_inner = jnp.linalg.eigh(inner)

        # Reverse to descending order.
        Lambda = Lambda[::-1]
        U_inner = U_inner[:, ::-1]

        # Guard against small negative values.
        Lambda = jnp.maximum(Lambda, 0.0)

        # Eigenvectors of P (row major order): Phi^T = U_inner^T @ Q.
        # (Nr, N_local) sharded
        Phi_local = U_inner.T @ Q_local
        return Phi_local, Lambda

    return _body(F)

if __name__ == '__main__':
    main()
    main()
