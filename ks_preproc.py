import time
import numpy as np

def main():
    Lfact = 7 # domain length factor
    Ntrue = 2111 # training samples
    Nq = 64 # delays
    N = Ntrue - Nq + 1 # temporal space dim
    M = 64 # spatial space dim
    NM = N * M # product space dim

    # Read the true state samples.
    utrue = np.empty((Ntrue, M), dtype=np.float64)
    utrue = np.load(f"data/ks_true_{Ntrue}_{M}_{Lfact}_u.npy")

    # Transform into delay form.
    # Each row contains the delay embedded data for the selected
    # gridpoint at the selected snapshot.
    udelay = np.empty((NM, Nq), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            udelay[i*M+j, :] = utrue[i:i+Nq, j]

    #np.save(f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay.npy", udelay)
    udelay32 = np.transpose(udelay.astype(np.float32))
    np.save(f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay32.npy", udelay32)
    return None

if __name__ == '__main__':
    t0 = time.time()
    main()
    tdelta = np.round(time.time()-t0, decimals=3)
    print(f"Elapsed time: {tdelta} sec.")
