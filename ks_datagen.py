import time
import h5py
import numpy as np

def main():
    Lfact = 7 # domain legnth factor
    L = Lfact * np.pi # domain length
    M = 64 # gridpoints
    M2 = int(M//2 + 1) # dim of the real FFT output
    Mext = int(3*M/2) # used for 3/2 dealiasing

    dt = 0.25 # timestep
    Nq = 64 # delays
    Nsave = 1 # save solution every Nsave timesteps
    Nequil = 10000 # timesteps to approach equilibrium

    Nfinal = 512 # final amount of delay embedded snapshots
    Ntrue = Nfinal + Nq - 1 # true amount of snapshots
    Nt = int(Nsave * (Ntrue - 1)) # total timesteps

    xvec = L * np.arange(-int(M/2), int(M/2))/M # grid vector

    # Fourier coeffs vector & initial condition.
    ufou = np.zeros((M2,), dtype=np.complex128)
    ufou[:4] = 0.6

    freq = 2*np.pi/L * np.arange(M2) # Fourier wavenumbers
    gg = -0.5 * 1j * Mext/M * freq 

    ldiag = freq**2 - freq**4

    # ETDRK4 timestepping scheme quantities.
    e1 = np.exp(dt * ldiag)
    e2 = np.exp(dt/2 * ldiag)

    J = 16 # points for the contour integration
    rvec = np.exp(1j*np.pi*np.linspace(0.5/J, (J-0.5)/J, num=J))
    Zmat = dt*ldiag.reshape(M2, 1) + rvec.reshape(1, J) # M2 x J
    
    # All f vectors have shape (M2,)
    f0 = dt*np.real(np.mean((np.exp(0.5*Zmat)-1)/Zmat, axis=-1))
    f1 = dt*np.real(np.mean((-4-Zmat+np.exp(Zmat)*(
                             4-3*Zmat+Zmat**2))/Zmat**3,
                            axis=-1 ))
    f2 = dt*np.real(np.mean(2*(2+Zmat+np.exp(Zmat)*(-2+Zmat))/Zmat**3,
                            axis=-1))
    f3 = dt*np.real(np.mean((-4-3*Zmat-Zmat**2+np.exp(Zmat)*(4-Zmat))/Zmat**3,
                            axis=-1))

    # Solution and time storage arrays.
    usave = np.empty((Ntrue, M), dtype=np.float64)
    tsave = np.empty((Ntrue,), dtype=np.float64)
    tsave[0] = 0.0

    # Timestepping used to reach equilibrium.
    for n in range(Nequil):
        # FFT(u*u_x); uses 3/2 dealiasing.
        Nu = gg * np.fft.rfft(np.fft.irfft(ufou, n=Mext)**2)[:M2]
        a = e2*ufou + f0*Nu
        Na = gg * np.fft.rfft(np.fft.irfft(a, n=Mext)**2)[:M2]
        b = e2*ufou + f0*Na
        Nb = gg * np.fft.rfft(np.fft.irfft(b, n=Mext)**2)[:M2]
        c = e2*a + f0*(2*Nb - Nu)
        Nc = gg * np.fft.rfft(np.fft.irfft(c, n=Mext)**2)[:M2]
        
        ufou = e1*ufou + f1*Nu + f2*(Na+Nb) + f3*Nc # FFT(u_{n+1})

    # Initial condition after equilibrium has been reached.
    usave[0, :] = np.fft.irfft(ufou)
    tsave[0] = Nequil * dt

    # Main timestepping loop.
    nsave = 1
    for n in range(Nt):
        # FFT(u*u_x); uses 3/2 dealiasing.
        Nu = gg * np.fft.rfft(np.fft.irfft(ufou, n=Mext)**2)[:M2]
        a = e2*ufou + f0*Nu
        Na = gg * np.fft.rfft(np.fft.irfft(a, n=Mext)**2)[:M2]
        b = e2*ufou + f0*Na
        Nb = gg * np.fft.rfft(np.fft.irfft(b, n=Mext)**2)[:M2]
        c = e2*a + f0*(2*Nb - Nu)
        Nc = gg * np.fft.rfft(np.fft.irfft(c, n=Mext)**2)[:M2]
        
        ufou = e1*ufou + f1*Nu + f2*(Na+Nb) + f3*Nc # FFT(u_{n+1})

        if (n+1) % Nsave == 0:
            usave[nsave, :] = np.fft.irfft(ufou)
            tsave[nsave] = (Nequil+n+1) * dt
            nsave += 1

    fname = f"data/ks_true_{Ntrue}_{M}_{Lfact}.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("u", shape=(Ntrue*M,), dtype=np.float64,
                         data=usave.flatten(order="F"))
        f.create_dataset("t", shape=(Ntrue,), dtype=np.float64,
                         data=tsave)

    fname32 = f"data/ks_true_{Ntrue}_{M}_{Lfact}_u32.h5"
    with h5py.File(fname32, "w") as f:
        f.create_dataset("u", shape=(Ntrue*M,), dtype=np.float32,
                         data=usave.astype(np.float32).flatten(order="F"))
        f.create_dataset("t", shape=(Ntrue,), dtype=np.float64,
                         data=tsave)

    return None

if __name__ == '__main__':
    t0 = time.time()
    main()
    tdelta = np.round(time.time()-t0, decimals=3)
    print(f"Elapsed time: {tdelta} sec.")
