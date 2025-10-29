import time
import numpy as np

def main():
    Lfact = 7 # domain length factor
    Nq = 64 # delays
    N = 500 # temporal space dim
    M = 64 # spatial space dim
    NM = N * M # product space dim

    # Read the delay embedded samples.
    u = np.empty((NM, Nq), dtype=np.float32)
    u = np.load(f"data/ks_train_{NM}_{Nq}_{Lfact}_udelay32.npy")

    u1 = u[:, np.newaxis, :] # NM x 1 x Nq
    u2 = u[np.newaxis, :, :] # 1 x NM x Nq

    dmat = np.sum((u1 - u2)**2, axis=-1) # NM x NM distance matrix

    emed = np.median(dmat) # bandwidth by median rule
    print(emed)

    return None

if __name__ == '__main__':
    t0 = time.time()
    main()
    tdelta = np.round(time.time()-t0, decimals=3)
    print(f"Elapsed time: {tdelta} sec.")
