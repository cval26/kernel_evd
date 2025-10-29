import time
import numpy as np

def main():
    Lfact = 7 # domain length factor
    Nq = 64 # delays
    N = 8192 # total number of samples

    # Read the identified data subset.
    u = np.empty((N, Nq), dtype=np.float32)
    u = np.load(f"data/ks_train_{N}_{Nq}_{Lfact}_usub32.npy")

    u1 = u[:, np.newaxis, :] # reshape to N x 1 x Nq
    u2 = u[np.newaxis, :, :] # reshape to 1 x N x Nq

    ndmat = -np.sum((u1 - u2)**2, axis=-1) # N x N negative dist. matrix

    # Candidate bandwidth values.
    # Clustered around the median rule bandwidth.
    evec = np.concatenate((np.arange(1, 10, 1, dtype=np.float32),
                           np.arange(10, 100, 5, dtype=np.float32),
                           np.arange(100, 300, 10, dtype=np.float32),
                           np.arange(300, 1000, 50, dtype=np.float32)))

    Ke = np.empty((N, N), dtype=np.float32)
    logSe = np.empty((evec.size,), dtype=np.float32)
    for i in range(evec.size):
        Ke[:, :] = np.exp(ndmat / evec[i])
        logSe[i] = np.log(np.sum(Ke) / N**2)

    loge = np.log(evec)
    dlogSe = np.empty((loge.size-2,), dtype=np.float32)
    for i in range(1, loge.size-1):
        dlogSe[i-1] = (logSe[i+1] - logSe[i-1]) / (loge[i+1] - loge[i-1])

    idmax = np.argmax(dlogSe)
    eopt = evec[idmax+1]
    mdim = 2*dlogSe[idmax]

    print(evec)
    print(dlogSe)
    print(eopt)
    print(mdim)
    return None

if __name__ == '__main__':
    t0 = time.time()
    main()
    tdelta = np.round(time.time()-t0, decimals=3)
    print(f"Elapsed time: {tdelta} sec.")
