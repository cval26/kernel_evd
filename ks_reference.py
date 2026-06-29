import time
import numpy as np
import scipy as scp

def main():
    Lfact = 7 # domain length factor
    Nq = 64 # delays
    N = 512 # temporal space dim
    M = 64 # spatial space dim
    NM = N * M # product space dim

    eps = 32 # kernel bandwidth

    # Read the delay embedded samples.
    u = np.empty((NM, Nq), dtype=np.float64)
    u = np.load(f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay.npy")

    K = np.empty((NM, NM), dtype=np.float64)
    compute_kernel_mat(K, u, eps)

    onevec = np.ones((NM,), dtype=np.float64)
    dvec = (1/NM) * K @ onevec
    Dm1 = np.diag(1.0 / dvec)

    qvec = (1/NM) * K @ Dm1 @ onevec
    Qm12 = np.diag(1.0 / np.sqrt(qvec))

    # Dividing by NM**2 instead of NM so that P is
    # bistochastic wrt the 2-norm, not the L2 norm.
    # In this way the max eigenvalue is 1.0, not NM.
    Ktilde = Dm1 @ K @ Qm12
    P = (1/NM**2) * Ktilde @ Ktilde.T

    w, V = scp.linalg.eigh(P)

    eig = w[::-1]
    Phi = V[:, ::-1]

    np.save(f"data/ks_eigref_{NM}_{Lfact}_{eps}.npy", eig)
    np.save(f"data/ks_phiref_{NM}_{Lfact}_{eps}.npy", Phi)
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
