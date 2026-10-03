import time
import h5py
import numpy as np
import scipy as scp

def main():
    Lfact = 7 # domain length factor
    Nq = 64 # delays
    N = 512 # temporal space dim
    M = 64 # spatial space dim
    NM = N * M # product space dim

    eps = 0.8 # kernel bandwidth

    # Read the delay embedded samples.
    u = np.empty((NM, Nq), dtype=np.float64)
    fname = f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay.h5"
    with h5py.File(fname, "r") as f:
        u[:] = f["u"][:].reshape((NM, Nq), order="F")

    K = np.empty((NM, NM), dtype=np.float64)
    compute_kernel_mat(K, u, Nq, eps)

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

    fnamer = f"data/evd_ref_{N}_{M}_{Nq}_{Lfact}.h5"
    with h5py.File(fnamer, "w") as f:
        f.create_dataset("eigvals", shape=(NM,), dtype=np.float64,
                         data=eig)
        f.create_dataset("eigvecs", shape=(NM*NM,), dtype=np.float64,
                         data=Phi.flatten(order="F"))
    return None


def compute_kernel_mat(K, x, Nq, eps):
    """Compute the kernel matrix.
    """
    x1 = x[:, np.newaxis, :] # reshape to N x 1 x Nq
    x2 = x[np.newaxis, :, :] # reshape to 1 x N x Nq

    K[:, :] = np.exp(-np.sum((x1-x2)**2, axis=-1) / (Nq * eps))
    return None

if __name__ == '__main__':
    t0 = time.time()
    main()
    tdelta = np.round(time.time()-t0, decimals=3)
    print(f"Elapsed time: {tdelta} sec.")
