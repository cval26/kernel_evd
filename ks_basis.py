import time
import numpy as np
import scipy as scp

def main():
    Lfact = 7 # domain length factor
    Nq = 64 # delays
    N = 500 # temporal space dim
    M = 64 # spatial space dim
    NM = N * M # product space dim

    Nr = 2048 # kernel matrix approximation rank
    L = Nr # spectral resolution
    eps = 50 # kernel bandwidth

    # Read the delay embedded samples.
    u = np.empty((NM, Nq), dtype=np.float64)
    u = np.load(f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay.npy")

    # Compute the partial Cholesky factorization.
    ind = np.empty((Nr,), dtype=int) # sampled indices
    F = np.empty((NM, Nr), dtype=np.float64) # partial Cholesky factor
    rpcholesky(F, ind, u, eps)
    np.savetxt(f"data/ks_samples_{NM}_{Nr}opt_{eps}.dat", ind, fmt="%d")

    # 1 -- Dilution
    Phi1 = np.empty((NM, L), dtype=np.float64)
    eig1 = np.empty((L,), dtype=np.float64)
    basis_dilute(Phi1, eig1, F)
    np.save(f"data/ks_phi1_{NM}_{L}_{Lfact}_{eps}.npy", Phi1)
    np.save(f"data/ks_eig1_{NM}_{L}_{Lfact}_{eps}.npy", eig1)
    del Phi1, eig1, F

    # 2 -- Subsampling
    usamp = np.empty((Nr, Nq), dtype=np.float64) # sampled data
    usamp = u[ind, :].copy()
    assert usamp.shape[0] == Nr and usamp.shape[1] == Nq

    Phi2 = np.empty((NM, L), dtype=np.float64)
    eig2 = np.empty((L,), dtype=np.float64)
    basis_subsample(Phi2, eig2, u, usamp, ind, eps)
    np.save(f"data/ks_phi2_{NM}_{L}_{Lfact}_{eps}.npy", Phi2)
    np.save(f"data/ks_eig2_{NM}_{L}_{Lfact}_{eps}.npy", eig2)
    return None

def basis_subsample(Phi, eig, x, xsamp, ind, eps):
    N = Phi.shape[0] # total samples number
    L = Phi.shape[1] # spectral resolution
    Nr = xsamp.shape[0] # partial Cholesky factor rank
    Nq = x.shape[1]
    assert x.shape[0] == N
    assert xsamp.shape[1] == Nq
    assert Nr == L

    # Compute the reduced basis matrix based only on sampled states.
    K = np.empty((Nr, Nr), dtype=np.float64)
    compute_kernel_mat(K, xsamp, eps)

    oneNr = np.ones((Nr,), dtype=np.float64)
    dvec = (1/Nr) * K @ oneNr
    Dm1 = np.diag(1.0 / dvec)

    qvec = (1/Nr) * K @ Dm1 @ oneNr
    Qm12 = np.diag(1.0 / np.sqrt(qvec))

    # Dividing by Nr**2 instead of Nr so that P is
    # bistochastic wrt the 2-norm, not the L2 norm.
    # In this way the max eigval is 1.0, not Nr.
    Ktilde = Dm1 @ K @ Qm12
    P = (1/Nr**2) * Ktilde @ Ktilde.T

    w, V = scp.linalg.eigh(P)

    # Reduced basis matrix (eigenfunctions based on sampled data)
    # and corresponding eigenvalues.
    Phir = V[:, ::-1] # Nr x L
    eig[:] = w[::-1] # L
    del Ktilde, P, w, V

    # Extend the eigenfunctions using Nystroem extension.
    dm1 = (1.0 / dvec).reshape(1, Nr)
    qm1 = (1.0 / qvec).reshape(Nr, 1)
    Kqd = qm1 * (dm1 * K) # Nr x Nr

    xi = np.empty((Nq,), dtype=np.float64)
    kvec = np.empty((Nr, 1), dtype=np.float64)
    pvec = np.empty((Nr, 1), dtype=np.float64)

    # Extended basis matrix.
    Phiex = np.empty((N, L), dtype=np.float64)
    Phiex[:, 0] = Phir[0, 0] # 1st eigenvector is constant
    Phiex[ind, 1:] = Phir[:, 1:].copy()

    factor = 1 / eig[1:] # not dividing by Nr

    k = 0
    for i in range(N):
        if i == ind[k]:
            k = (k + 1) % Nr
            continue

        xi[:] = x[i, :]
        kvec[:, 0] = np.exp(-np.sum((xi-xsamp)**2, axis=-1) / eps)

        dxi = (1/Nr) * np.sum(kvec)
        itemp = 1 / (Nr**2 * dxi) # dividing by Nr**2 as earlier
        pvec[:, 0] = itemp * np.sum(kvec * Kqd, axis=0)

        Phiex[i, 1:] = factor * np.sum(pvec * Phir[:, 1:], axis=0)

    # Orthonormalize the extended basis.
    Phi[:, :], _ = np.linalg.qr(Phiex, mode="reduced")
    return None

def basis_dilute(Phi, eig, F):
    N = Phi.shape[0] # total samples number
    L = Phi.shape[1] # spectral resolution
    Nr = F.shape[1] # partial Cholesky factor rank
    assert F.shape[0] == N
    assert Nr == L

    oneN = np.ones((N,), dtype=np.float64)
    dvec = F @ F.T @ oneN
    assert dvec[dvec<=0].size == 0
    Dm1 = np.diag(1.0 / dvec)

    qvec = F @ F.T @ Dm1 @ oneN
    Qm12 = np.diag(1.0 / np.sqrt(qvec))

    Khat = F.T @ F # Nr x Nr
    s2vec, V = scp.linalg.eigh(Khat, subset_by_index=[Nr-L, Nr-1]) # L x L
    S2 = np.diag(1.0 / np.sqrt(s2vec[::-1])) # L x L
    U = F @ V[:, ::-1] @ S2 # N x L

    Uresc = Dm1 @ U # N x L
    Vresc = Qm12 @ U # N x L
    S2 = np.diag(s2vec[::-1]) # L x L

    Q1, R1 = np.linalg.qr(Uresc, mode="reduced")
    _, R2 = np.linalg.qr(Vresc, mode="reduced")

    A1 = R1 @ S2 @ R2.T
    U1, s1vec, _ = np.linalg.svd(A1, full_matrices=False)

    eig[:] = s1vec**2

    LSV = Q1 @ U1 # N x L
    LSV[:, 0] = 1.0 / np.sqrt(N)
    Phi[:, :], _ = np.linalg.qr(LSV, mode="reduced") # N x L
    return None

def rpcholesky(F, ind, x, eps):
    """Compute the partial Cholesky factor of the kernel matrix
    using the randomly pivoted Cholesky algorithm.
    """
    N = x.shape[0]
    Nq = x.shape[1]
    Nr = F.shape[1]
    assert F.shape[0] == N
    assert ind.size == Nr

    xi = np.empty((Nq,), dtype=np.float64)

    dvec = np.ones((N,), dtype=np.float64)
    pvec = dvec / np.sum(dvec)
    rng = np.random.default_rng()

    # First iteration.
    pvt = rng.choice(np.arange(N), p=pvec)
    ind[0] = pvt

    xi[:] = x[pvt, :]

    F[:, 0] = np.exp(-np.sum((xi-x)**2, axis=-1) / eps)
    isqrtnorm = 1.0 / np.sqrt(F[pvt, 0])
    F[:, 0] *= isqrtnorm

    dvec -= F[:, 0]**2
    dvec = np.maximum(dvec, 0)
    pvec = dvec / np.sum(dvec)

    # Subsequent iterations.
    for i in range(1, Nr):
        pvt = rng.choice(np.arange(N), p=pvec)
        ind[i] = pvt

        xi[:] = x[pvt, :]

        F[:, i] = np.exp(-np.sum((xi-x)**2, axis=-1) / eps)
        F[:, i] -= np.sum(F[:, :i] * F[pvt, :i], axis=1)
        isqrtnorm = 1.0 / np.sqrt(F[pvt, i])
        F[:, i] *= isqrtnorm

        dvec -= F[:, i]**2
        dvec = np.maximum(dvec, 0)
        pvec = dvec / np.sum(dvec)

    ind.sort()

    print(f"Cholesky trace error: {np.sum(dvec)/N}")
    return None

def compute_kernel_mat(K, x, eps):
    """Compute the kernel matrix.
    """
    x1 = x[:, np.newaxis, :] # reshape to N x 1 x Nq
    x2 = x[np.newaxis, :, :] # reshape to 1 x N x Nq

    K[:, :] = np.exp(-np.sum((x1-x2)**2, axis=-1) / eps)
    return None

if __name__ == '__main__':
    t0 = time.time()
    main()
    tdelta = np.round(time.time()-t0, decimals=3)
    print(f"Elapsed time: {tdelta} sec.")
