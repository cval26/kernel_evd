import time
import numpy as np

def main():
    Lfact = 7 # domain length factor
    Nq = 64 # delays
    N = 500 # temporal space dim
    M = 64 # spatial space dim
    NM = N * M # product space dim

    Nsamp = 8192 # number of samples to identify
    eps = 157.60884 # kernel bandwidth

    ind = np.empty((Nsamp,), dtype=int) # sampled indices array

    # Read the delay embedded samples.
    u = np.empty((NM, Nq), dtype=np.float32)
    u = np.load(f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay32.npy")

    rpc_subsample(ind, u, Nsamp, eps)
    np.savetxt(f"data/ks_samples_{Nsamp}.dat", ind, fmt="%d")

    usamp = np.empty((Nsamp, Nq), dtype=np.float32)
    usamp = u[ind, :].copy()
    assert usamp.shape[0] == Nsamp and usamp.shape[1] == Nq
    np.save(f"data/ks_train_{Nsamp}_{Nq}_{Lfact}_usub32.npy", usamp)
    #del usamp, u

    ## Store the subsampled data in double precision.
    #u = np.empty((NM, Nq), dtype=np.float64)
    #u = np.load(f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay.npy")

    #usamp = np.empty((Nsamp, Nq), dtype=np.float64)
    #usamp = u[ind, :].copy()
    #assert usamp.shape[0] == Nsamp and usamp.shape[1] == Nq
    #np.save(f"data/ks_train_{Nsamp}_{Nq}_{Lfact}_usub.npy", usamp)

    return None

def rpc_subsample(ind, x, Nr, eps):
    """Identify a subset of the given data to be sampled
    using the randomly pivoted Cholesky algorithm.
    """
    N = x.shape[0]
    Nq = x.shape[1]
    assert ind.size == Nr

    F = np.empty((N, Nr), dtype=np.float32)
    xi = np.empty((Nq,), dtype=np.float32)

    dvec = np.ones((N,), dtype=np.float32)
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

if __name__ == '__main__':
    t0 = time.time()
    main()
    tdelta = np.round(time.time()-t0, decimals=3)
    print(f"Elapsed time: {tdelta} sec.")
