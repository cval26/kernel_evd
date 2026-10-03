//nvcc ks_arpclm.cu -ccbin mpicxx -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lcurand -lcublas -lcublasLt -lcusparse -lcusolver -lnccl -lhdf5 --use_fast_math -o cuarpclm
//nvcc ks_arpclm.cu -ccbin mpicxx -g -G -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lcurand -lcublas -lcublasLt -lcusparse -lcusolver -lnccl -lhdf5 -o cuarpclm
//run: NCCL_P2P_LEVEL=SYS mpirun -np 4 ./cuarpclm
//run: CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ./cuarpclm

// Multi-GPU low-memory accelerated RPCholesky (MPI + NCCL).
// One MPI process (rank) per GPU.
//
// This is the memory-frugal sibling of arpc.cu:
// it stores only the rmax x rmax inverse Cholesky factor L^{-1} and
// regenerates kernel matrix entries on demand, rather than storing the
// NM x rmax factor Fmat.
//
// Distribution (Ndevs devices, device idev owns global rows
// [idev*Nrows, (idev+1)*Nrows), Nrows = NM/Ndevs, and global columns
// [idev*cblk, (idev+1)*cblk) of L^{-1}, cblk = rmax/Ndevs):
//
//   - u, bw, the full dvec (NM), Spiv, and the b x b dense buffers
//     are replicated on every device;
//   - the inverse Cholesky factor Linv (rmax x cblk per device) is split
//     along its columns;
//   - Step-3 row work is split along NM, each device owning Nrows rows.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <mpi.h>
#include <hdf5.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <nccl.h>

int constexpr BLK = 512; // RPC block size

#define CUDA_CHECK(val) cuda_check((val), __FILE__, __LINE__)
inline void cuda_check(cudaError_t err, char const* file, int const line)
{
    if (err != cudaSuccess) {
        printf("CUDA error: %s:%i %s: %s\n", file, line,
                cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

#define NCCL_CHECK(val) nccl_check((val), __FILE__, __LINE__)
inline void nccl_check(ncclResult_t res, char const* file, int const line)
{
    if (res != ncclSuccess) {
        printf("NCCL error: %s:%i %s\n", file, line,
                ncclGetErrorString(res));
        exit(1);
    }
}

#define CUBLAS_CHECK(val) cublas_check((val), __FILE__, __LINE__)
inline void cublas_check(cublasStatus_t st, char const* file,
        int const line)
{
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %s:%i status %d\n", file, line, (int) st);
        exit(1);
    }
}

#define MPI_CHECK(val) mpi_check((val), __FILE__, __LINE__)
inline void mpi_check(int const code, char const* file, int const line)
{
    if (code != MPI_SUCCESS) {
        char str[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(code, str, &len);
        printf("MPI error: %s:%i %s\n", file, line, str);
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}

int h5read(float* data, char const* filename, char const* dsetname,
        size_t const rank, size_t const N)
{
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file != H5I_INVALID_HID);

    hid_t dset = H5Dopen2(file, dsetname, H5P_DEFAULT);
    assert(dset != H5I_INVALID_HID);

    hid_t dspace = H5Dget_space(dset);
    assert(dspace != H5I_INVALID_HID);

    hsize_t* dims = NULL;
    dims = (hsize_t*) malloc(rank * sizeof(hsize_t));
    assert(dims != NULL);

    herr_t h5stat = H5Sget_simple_extent_dims(dspace, dims, NULL);
    assert(h5stat >= 0);

    size_t datasize = 1;
    for (size_t i = 0; i < rank; ++i) {
        datasize *= dims[i];
    }
    assert(datasize == N);

    assert(data != NULL);
    h5stat = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, data);
    assert(h5stat >= 0);

    H5Sclose(dspace);
    H5Dclose(dset);
    H5Fclose(file);
    return 0;
}

// Write a 1-D array of N size_t values to a fresh HDF5 file.
// The dataset is created with the platform native size_t type;
// an existing file of the same name is overwritten.
int h5write_sizet(size_t const* data, char const* filename,
        char const* dsetname, size_t const N)
{
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT,
            H5P_DEFAULT);
    assert(file != H5I_INVALID_HID);

    hsize_t dims[1] = { (hsize_t) N };
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    assert(dspace != H5I_INVALID_HID);

    hid_t dset = H5Dcreate2(file, dsetname, H5T_NATIVE_ULONG, dspace,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset != H5I_INVALID_HID);

    assert(data != NULL);
    herr_t h5stat = H5Dwrite(dset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, data);
    assert(h5stat >= 0);

    H5Dclose(dset);
    H5Sclose(dspace);
    H5Fclose(file);
    return 0;
}

// Write a 1-D array of N double values to a fresh HDF5 file.
// An existing file of the same name is overwritten.
int h5write_double(double const* data, char const* filename,
        char const* dsetname, size_t const N)
{
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT,
            H5P_DEFAULT);
    assert(file != H5I_INVALID_HID);

    hsize_t dims[1] = { (hsize_t) N };
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    assert(dspace != H5I_INVALID_HID);

    hid_t dset = H5Dcreate2(file, dsetname, H5T_NATIVE_DOUBLE, dspace,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset != H5I_INVALID_HID);

    assert(data != NULL);
    herr_t h5stat = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, data);
    assert(h5stat >= 0);

    H5Dclose(dset);
    H5Sclose(dspace);
    H5Fclose(file);
    return 0;
}

__global__
void initialize_rstate(curandState* state)
{
    // seed, sequence number, offset, state
    curand_init(1234, threadIdx.x, 0, &state[threadIdx.x]);
}

__global__
void initialize_dvec(float* dvec, float* dsum, size_t const NM)
{
    for (size_t i = threadIdx.x; i < NM; i += blockDim.x) {
        dvec[i] = 1.0;
    }
    if (threadIdx.x == 0) {
        *dsum = NM;
    }
}

// Step 1: propose a block of b pivots, drawn iid from u = dvec.
// One thread per proposal; each performs an inverse-CDF scan.
// Run redundantly on every device over the replicated dvec, so the
// result is identical across devices with no communication.
__global__
void propose_block(size_t* Sprime, float const* dvec,
        float const* dsum_ptr, curandState* rstate, size_t const NM)
{
    int const t = threadIdx.x; // blockDim.x == BLK
    float const dsum = *dsum_ptr;

    // Uniform random number in (0,1].
    float rand = curand_uniform(&rstate[t]);

    size_t flag = 0;
    float sum = 0.0;

    // TODO: use Kahan compensation.
    for (size_t i = 0; i < NM; ++i)
    {
        sum += dvec[i] / dsum;
        if (sum >= rand)
        {
            Sprime[t] = i;
            flag = 1;
            break;
        }
    }
    if (flag == 0) {
        Sprime[t] = NM - 1;
    }
}

// Kernel-value exp() in the matching precision: double for the
// proposal/H path, float for the per-row Step-3 path.
__device__ inline double kexp_t(double x) { return exp(x); }
__device__ inline float kexp_t(float x) { return expf(x); }

// Unified kernel-value block evaluator (replaces compute_Hmat_kval and
// compute_Gblock_kval of arpc.cu).
// Writes a row-major block out[r*ld + c] = K(row_glob(r), col_glob(c))
// for r in [0,nrows), c in [0, npiv+ntail).
// Rows: rowidx[r] if rowidx != NULL, else the contiguous row_offset + r
// (proposal/pivot rows use the index array; Step-3 row blocks use the
// contiguous form).
// Cols: Spiv[c] for c < npiv (the pivots S), else tail[c - npiv] (the
// proposals S' or accepted pivots S_i).
// Accumulation and exp() are carried in T (double or float) to match
// the precision of the consuming path.
template <typename T>
__global__
void compute_kblock(T* out, size_t const ld,
        size_t const* rowidx, size_t const row_offset,
        size_t const nrows,
        size_t const* Spiv, size_t const npiv,
        size_t const* tail, size_t const ntail,
        float const* u, float const bw,
        size_t const N, size_t const Nq)
{
    size_t const Ndata = N + Nq - 1;
    size_t const ncol = npiv + ntail;

    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;

    for (size_t idx = tid; idx < nrows * ncol; idx += stride)
    {
        size_t const r = idx / ncol;
        size_t const c = idx % ncol;

        size_t const ig =
                (rowidx != NULL) ? rowidx[r] : row_offset + r;
        size_t const jg = (c < npiv) ? Spiv[c] : tail[c - npiv];

        size_t const idata = (ig / N) * Ndata + ig % N;
        size_t const jdata = (jg / N) * Ndata + jg % N;

        // Kernel value K(ig, jg), accumulated in T.
        T sum1 = 0;
        for (size_t j = 0; j < Nq; ++j)
        {
            T const d1 = (T) u[idata + j] - (T) u[jdata + j];
            sum1 += d1 * d1;
        }
        T const arg = -sum1 / (T) Nq / (T) bw;
        out[r * ld + c] = kexp_t(arg);
    }
}

// Extract the b x b block K(S',S') from the proposal block Kprop.
// Kprop is row major with leading dim rmax; its columns [Nr, Nr+b) are
// the proposals S'.  Hmat is the row-major b x b residual submatrix
// (leading dim BLK) that the DSYRK overlap and rejection_sample read.
__global__
void extract_H(double* Hmat, double const* Kprop, size_t const Nr,
        size_t const ld)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < (size_t) BLK * BLK; idx += stride)
    {
        size_t const p = idx / BLK;
        size_t const q = idx % BLK;
        Hmat[p * BLK + q] = Kprop[p * ld + Nr + q];
    }
}

// Algorithm 2.1 rejection sampling on the b x b residual submatrix H,
// followed by inversion of the compacted Cholesky factor.
// Ported verbatim from arpc.cu: H is consumed in place, the accepted
// proposals land in Sacc/accpos/nacc, and the inverse compacted factor
// L_i^{-1} is emitted in double (Minv) and float (Mf).
__global__
void rejection_sample(double* Hmat, double* Lmat, double* Minv,
        float* Mf, size_t const* Sprime, size_t* Sacc, size_t* nacc_dev,
        curandState* rstate, float const tol, int* accpos_dev)
{
    __shared__ double u_sh[BLK];
    __shared__ int accpos[BLK];
    __shared__ int mcount;
    __shared__ int decision;

    int const t = threadIdx.x; // blockDim.x == BLK

    // u <- diag(H), captured before any Schur updates.
    u_sh[t] = Hmat[t * BLK + t];
    if (t == 0) {
        mcount = 0;
    }
    __syncthreads();

    for (int i = 0; i < BLK; ++i)
    {
        // Accept or reject the i-th proposal (thread 0 decides).
        if (t == 0)
        {
            double rnd = curand_uniform_double(&rstate[0]);
            double hii = Hmat[i * BLK + i];
            int acc = (u_sh[i] * rnd < hii) && (hii > (double) tol);
            decision = acc;
            if (acc) {
                accpos[mcount] = i;
                mcount++;
            }
        }
        __syncthreads();

        // decision is uniform across the block.
        if (decision)
        {
            double const factor = 1.0 / sqrt(Hmat[i * BLK + i]);

            // Cholesky column: L(i:b, i) <- H(i:b, i) / sqrt(H(i,i)).
            for (int j = i + t; j < BLK; j += blockDim.x) {
                Lmat[j * BLK + i] = Hmat[j * BLK + i] * factor;
            }
            __syncthreads();

            // Schur complement on the lower triangle of the trailing
            // submatrix: H(j,k) -= L(j,i) L(k,i), j >= k > i.
            int const w = BLK - (i + 1);
            for (int idx = t; idx < w * w; idx += blockDim.x)
            {
                int const jj = i + 1 + idx / w;
                int const kk = i + 1 + idx % w;
                if (jj >= kk) {
                    Hmat[jj * BLK + kk] -=
                            Lmat[jj * BLK + i] * Lmat[kk * BLK + i];
                }
            }
            __syncthreads();
        }
    }
    __syncthreads();

    int const nacc = mcount;
    if (t == 0) {
        *nacc_dev = (size_t) nacc;
    }

    // Accepted pivot indices and their proposal positions.
    if (t < nacc) {
        Sacc[t] = Sprime[accpos[t]];
        accpos_dev[t] = accpos[t];
    }

    // Invert the compacted lower-triangular factor Lc = L(T,T).
    // Columns are independent: thread t solves column t.
    if (t < nacc)
    {
        int const ac = accpos[t];
        Minv[t * BLK + t] = 1.0 / Lmat[ac * BLK + ac];
        for (int a = t + 1; a < nacc; ++a)
        {
            int const aa = accpos[a];
            double s = 0.0;
            for (int e = t; e < a; ++e) {
                s += Lmat[aa * BLK + accpos[e]] * Minv[e * BLK + t];
            }
            Minv[a * BLK + t] = -s / Lmat[aa * BLK + aa];
        }
    }
    __syncthreads();

    // Zero Mf, then cast the leading nacc x nacc lower triangle to float.
    for (int idx = t; idx < BLK * BLK; idx += blockDim.x) {
        Mf[idx] = 0.0;
    }
    __syncthreads();
    if (t < nacc)
    {
        for (int a = t; a < nacc; ++a) {
            Mf[a * BLK + t] = (float) Minv[a * BLK + t];
        }
    }
}

// Step 3: gather the accepted columns of P into Q = P[:,accpos].
// Both are column-major with leading dim Nr; column c of Q is column
// accpos[c] of P.  Since S_i subset S_i', Q = L^{-1} K(S, S_i) needs no
// new product -- it is exactly the accepted columns of the P already
// assembled for the H overlap.
__global__
void gather_cols(double* Q, double const* P, int const* accpos,
        size_t const nacc, size_t const Nr)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < Nr * nacc; idx += stride)
    {
        size_t const c = idx / Nr;
        size_t const r = idx % Nr;
        Q[c * Nr + r] = P[(size_t) accpos[c] * Nr + r];
    }
}

// Step 3: repack the AllGathered M row-blocks into a contiguous
// column-major Nr x nacc float matrix.
// ncclAllGather concatenates each device's cblk x nacc block (column major,
// leading dim cblk) by rank, so the gathered buffer is "blocked":
//   Mgath[dblk*cblk*nacc + c*cblk + lc] = M[dblk*cblk + lc, c].
// This converts it to the strided column-major layout (leading dim Nr,
// global row p = dblk*cblk + lc) that the blocck GEMM consumes, casting to
// float.  Rows p >= Nr are never read, so stale tail rows of a partially
// filled block are ignored.
__global__
void repack_M(float* Mfull_f, double const* Mgath, size_t const Nr,
        size_t const nacc, size_t const cblk)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < Nr * nacc; idx += stride)
    {
        size_t const c = idx / Nr;
        size_t const p = idx % Nr;
        size_t const dblk = p / cblk;
        size_t const lc = p % cblk;
        Mfull_f[c * Nr + p] =
                (float) Mgath[dblk * cblk * nacc + c * cblk + lc];
    }
}

// Step 3: residual diagonal update for one row block.
// Subtract the squared row norms of Gblk (nrblk x nacc, row major,
// leading dim ldg) from the replicated dvec at global row row_offset + L,
// clamped at zero.
__global__
void update_diag_block(float const* Gblk, float* dvec,
        size_t const nacc, size_t const nrblk, size_t const row_offset,
        size_t const ldg)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t L = tid; L < nrblk; L += stride)
    {
        size_t const base = L * ldg;
        float s = 0.0;
        for (size_t q = 0; q < nacc; ++q) {
            float const v = Gblk[base + q];
            s += v * v;
        }
        size_t const gi = row_offset + L;
        float const dd = dvec[gi] - s;
        dvec[gi] = dd >= 0.0 ? dd : 0.0;
    }
}

// Step 3 (L^{-1} update), overlap block: write -L_i^{-1} M^* into the
// new block-row [Nr, Nr+nacc) of L^{-1}, columns [0,Nr).
// Device d owns global columns [d*cblk, d*cblk+kloc) of L^{-1}, whose M
// rows are exactly this device's Mrow (kloc x nacc, column major, ld cblk).
// Entry (a, lc): L^{-1}[Nr+a, d*cblk+lc]
//   = -sum_{e<=a} Minv[a,e] Mrow[lc,e]   (L_i^{-1} is lower triangular,
// so only e <= a contribute -- this also avoids the stale upper triangle
// of Minv, which rejection_sample never zeroes).
// Written column-major into Linv (ld rmax), local column lc.
__global__
void append_Linv_overlap(double* Linv, double const* Minv,
        double const* Mrow, size_t const Nr, size_t const nacc,
        size_t const kloc, size_t const cblk, size_t const rmax)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < nacc * kloc; idx += stride)
    {
        size_t const lc = idx / nacc;
        size_t const a = idx % nacc;
        double s = 0.0;
        for (size_t e = 0; e <= a; ++e) {
            s += Minv[a * BLK + e] * Mrow[e * cblk + lc];
        }
        Linv[(Nr + a) + lc * rmax] = -s;
    }
}

// Step 3 (L^{-1} update), diagonal block: write L_i^{-1} into the new
// pivot columns [Nr, Nr+nacc).
// Global column Nr+e is owned by this device iff it lies in
// [col_start, col_start+cblk); only the lower triangle (a >= e) is written,
// the upper part staying zero from the initial memset.
__global__
void append_Linv_diag(double* Linv, double const* Minv,
        size_t const Nr, size_t const nacc, size_t const col_start,
        size_t const cblk, size_t const rmax)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < nacc * nacc; idx += stride)
    {
        size_t const e = idx / nacc; // column within L_i^{-1}
        size_t const a = idx % nacc; // row within L_i^{-1}
        if (a < e) {
            continue; // lower triangle only
        }
        size_t const gc = Nr + e; // global column
        if (gc < col_start || gc >= col_start + cblk) {
            continue;
        }
        size_t const lc = gc - col_start;
        Linv[(Nr + a) + lc * rmax] = Minv[a * BLK + e];
    }
}

// Sum the full replicated dvec and record the error.
// Run redundantly on every device.
__global__
void compute_dsum(float* dsum, float* err, float const* dvec,
        size_t const round, size_t const NM)
{
    extern __shared__ float tsum[]; // size blockDim.x

    tsum[threadIdx.x] = 0.0;

    // TODO: use Kahan compensation.
    for (size_t i = threadIdx.x; i < NM; i += blockDim.x) {
        tsum[threadIdx.x] += dvec[i];
    }
    __syncthreads();

    // Single-block sum reduction.
    // MUST be launched with one block.
    if (threadIdx.x == 0)
    {
        float sum = 0.0;
        // TODO: use Kahan compensation.
        for (size_t i = 0; i < blockDim.x; ++i) {
            sum += tsum[i];
        }
        *dsum = sum;
        err[round] = sum / NM;
    }
}

int main(int argc, char** argv)
{
    // SPMD MPI harness: one rank per GPU.
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size = 0, rank = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // Node-local rank -> GPU binding.
    // Splitting by shared memory gives the intra-node rank,
    // which selects this rank's device.
    MPI_Comm shmcomm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                rank, MPI_INFO_NULL, &shmcomm));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(shmcomm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&shmcomm));

    time_t t0 = time(NULL);

    size_t constexpr Lfact = 30; // spatial domain factor
    size_t constexpr M = 128; // spatial samples

    size_t constexpr Ndata = 32831; // data time samples
    size_t constexpr Nq = 64; // delays
    size_t constexpr N = Ndata - Nq + 1; // net time samples

    size_t constexpr NM = N * M; // net product samples
    size_t constexpr NMdata = Ndata * M; // training data samples

    size_t constexpr Nrounds = 64; // RPC rounds
    size_t constexpr rmax = Nrounds * BLK; // max rank

    float constexpr tol = 1e-10; // RPC rejection tolerance

    float constexpr bw = 0.8; // kernel bandwidth

    size_t constexpr Ndevs = 4; // GPU devices
    size_t constexpr Nrows = NM / Ndevs; // local rows per device
    size_t constexpr cblk = rmax / Ndevs; // local L^{-1} cols per device
    size_t constexpr rblk = 65536; // step-3 row block size

    // CUDA kernel launch block and grid sizes.
    int constexpr tpb = 512;
    int constexpr dsum_tpb = (int) (NM < 1024 ? NM : 1024);
    int constexpr grid_extract =
            (int) (((size_t) BLK * BLK + tpb - 1) / tpb);

    // Ndevs must divide NM, M, and rmax.
    static_assert(NM % Ndevs == 0, "Ndevs must divide NM");
    static_assert(M % Ndevs == 0, "Ndevs must divide M");
    static_assert(rmax % Ndevs == 0, "Ndevs must divide rmax");

    // The static distribution above fixes the process count: launch with
    // exactly Ndevs ranks (mpirun -np Ndevs).
    if (world_size != (int) Ndevs) {
        if (rank == 0) {
            printf("Error: launched with %d ranks, need Ndevs = %lu\n",
                    world_size, Ndevs);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Load the state training data.
    // Data is stored with the time index varying first.
    char fname[128];
    sprintf(fname, "data/ks_true_%lu_%lu_%lu_u32.h5", Ndata, M, Lfact);
    char constexpr dset[] = "/u";

    char linv_file[128];
    sprintf(linv_file, "data/linv_%lu_%lu_b%i_r%lu_%lu.h5",
            N, M, BLK, Nrounds, Lfact);
    char constexpr linv_dset[] = "/linv";

    char spiv_file[128];
    sprintf(spiv_file, "data/spiv_%lu_%lu_b%i_r%lu_%lu.h5",
            N, M, BLK, Nrounds, Lfact);
    char constexpr spiv_dset[] = "/spiv";

    if (rank == 0) {
        printf("Data file: %s\nData set: %s\n", fname, &dset[1]);
        printf("Block size: %d  Devices: %lu  Rows/device: %lu  "
                "Cols/device: %lu\n\n", BLK, Ndevs, Nrows, cblk);
    }

    // Host buffer for the state field.
    float* u_host = (float*) malloc(NMdata * sizeof(float));
    assert(u_host != NULL);
    h5read(u_host, fname, dset, 1, NMdata);

    // Bootstrap NCCL: rank 0 creates the unique id, broadcasts it, and
    // every rank joins the communicator via ncclCommInitRank below.
    ncclUniqueId ncclid;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&ncclid));
    }
    MPI_CHECK(MPI_Bcast(&ncclid, sizeof(ncclid), MPI_BYTE, 0,
                MPI_COMM_WORLD));

    // Per-rank host variables, replicated.
    // Spiv_host accumulates the accepted pivots across all rounds.
    size_t* Spiv_host = (size_t*) malloc(rmax * sizeof(size_t));
    size_t* hm = (size_t*) malloc(Nrounds * sizeof(size_t));
    float* err = (float*) malloc(Nrounds * sizeof(float));
    assert(Spiv_host != NULL && hm != NULL && err != NULL);

    size_t Nr = 0; // realized rank

    // This rank's device index equals its global rank; each rank drives
    // its own GPU with a private stream and NCCL communicator.
    CUDA_CHECK(cudaSetDevice(local_rank));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, ncclid, rank));

    // Per-device handles and buffers.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // cuBLAS handle bound to the device stream.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Input state data, replicated.  The bandwidth bw is a compile-time
    // scalar passed by value to the kernel evaluator.
    float* u_dev;
    CUDA_CHECK(cudaMalloc(&u_dev, NMdata * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(u_dev, u_host,
                NMdata * sizeof(float), cudaMemcpyHostToDevice));

    // Distributed inverse Cholesky factor.
    // Col-major rmax x cblk block; device d owns global columns
    // [d*cblk, (d+1)*cblk).
    // Upper-triangular zeros are stored.
    double* Linv_dev;
    CUDA_CHECK(cudaMalloc(&Linv_dev, rmax * cblk * sizeof(double)));
    CUDA_CHECK(cudaMemset(Linv_dev, 0, rmax * cblk * sizeof(double)));

    // Replicated pivot indices (appended each round).
    size_t* Spiv_dev;
    CUDA_CHECK(cudaMalloc(&Spiv_dev, rmax * sizeof(size_t)));
    CUDA_CHECK(cudaMemset(Spiv_dev, 0, rmax * sizeof(size_t)));

    // Replicated residual diagonal and its sum.
    float* dvec_dev;
    CUDA_CHECK(cudaMalloc(&dvec_dev, NM * sizeof(float)));

    float* dsum_dev;
    CUDA_CHECK(cudaMalloc(&dsum_dev, sizeof(float)));

    // Per-round buffers, replicated.
    size_t* Sprime_dev;
    CUDA_CHECK(cudaMalloc(&Sprime_dev, BLK * sizeof(size_t)));

    size_t* Sacc_dev;
    CUDA_CHECK(cudaMalloc(&Sacc_dev, BLK * sizeof(size_t)));

    size_t* nacc_dev;
    CUDA_CHECK(cudaMalloc(&nacc_dev, sizeof(size_t)));

    int* accpos_dev;
    CUDA_CHECK(cudaMalloc(&accpos_dev, BLK * sizeof(int)));

    // Proposal kernel block K(S',S u S') (replicated, double).
    double* Kprop_dev;
    CUDA_CHECK(cudaMalloc(&Kprop_dev, BLK * rmax * sizeof(double)));

    double* Hmat_dev;
    CUDA_CHECK(cudaMalloc(&Hmat_dev, BLK * BLK * sizeof(double)));

    double* Lmat_dev;
    CUDA_CHECK(cudaMalloc(&Lmat_dev, BLK * BLK * sizeof(double)));

    double* Minv_dev; // L_i^{-1} (double)
    CUDA_CHECK(cudaMalloc(&Minv_dev, BLK * BLK * sizeof(double)));

    float* Mf_dev; // L_i^{-1} (float)
    CUDA_CHECK(cudaMalloc(&Mf_dev, BLK * BLK * sizeof(float)));

    // L^{-1} products (replicated after collectives).
    double* P_dev; // P = L^{-1} K(S,S') (col-major rmax x b)
    CUDA_CHECK(cudaMalloc(&P_dev, rmax * BLK * sizeof(double)));

    double* Q_dev; // Q = P[:,accpos] (col-major rmax x b)
    CUDA_CHECK(cudaMalloc(&Q_dev, rmax * BLK * sizeof(double)));

    double* Mrow_dev; // this device's M row-block (cblk x b)
    CUDA_CHECK(cudaMalloc(&Mrow_dev, cblk * BLK * sizeof(double)));

    double* Mfull_dev; // M after AllGather (rmax x b)
    CUDA_CHECK(cudaMalloc(&Mfull_dev, rmax * BLK * sizeof(double)));

    float* Mfull_f_dev; // float cast of Mfull (rmax x b)
    CUDA_CHECK(cudaMalloc(&Mfull_f_dev, rmax * BLK * sizeof(float)));

    // Step-3 row-block work buffers (row-major float).
    float* Kblk_dev; // K(R, S u S_i): rblk x (Nr+nacc)
    CUDA_CHECK(cudaMalloc(&Kblk_dev, rblk * rmax * sizeof(float)));

    float* Tblk_dev; // K(R,S_i) - K(R,S) M: rblk x nacc
    CUDA_CHECK(cudaMalloc(&Tblk_dev, rblk * BLK * sizeof(float)));

    float* Gblk_dev; // G = T L_i^{-*}: rblk x nacc
    CUDA_CHECK(cudaMalloc(&Gblk_dev, rblk * BLK * sizeof(float)));

    // Per-round error vector (replicated; published by master).
    float* err_dev;
    CUDA_CHECK(cudaMalloc(&err_dev, Nrounds * sizeof(float)));
    CUDA_CHECK(cudaMemset(err_dev, 0, Nrounds * sizeof(float)));

    // cuRAND states, initialized identically on every device.
    curandState* rstate_dev;
    CUDA_CHECK(cudaMalloc(&rstate_dev, BLK * sizeof(curandState)));
    initialize_rstate<<<1, BLK, 0, stream>>>(rstate_dev);
    CUDA_CHECK(cudaGetLastError());

    // Replicated dvec = 1, dsum = NM.
    initialize_dvec<<<1, 32, 0, stream>>>(dvec_dev, dsum_dev, NM);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    size_t Nr_local = 0; // local realized rank

    for (size_t i = 0; i < Nrounds; ++i)
    {
        // Step 1: propose a block of b pivots (replicated).
        propose_block<<<1, BLK, 0, stream>>>
                (Sprime_dev, dvec_dev, dsum_dev, rstate_dev, NM);
        CUDA_CHECK(cudaGetLastError());

        // Step 2: residual submatrix, rejection, inversion.
        // Form the proposal kernel block K(S', S u S') (replicated,
        // double).  Columns [0,Nr) are the pivots S (from Spiv);
        // columns [Nr,Nr+b) are the proposals S' (from Sprime).
        // The leading dim is rmax, so the column-major
        // reinterpretation of columns [0,Nr) is exactly K(S,S'),
        // which the P overlap below consumes directly.
        size_t const ncol = Nr_local + BLK;
        int const grid_kblk =
                (int) (((size_t) BLK * ncol + tpb - 1) / tpb);
        compute_kblock<double><<<grid_kblk, tpb, 0, stream>>>
                (Kprop_dev, rmax,
                 Sprime_dev, 0, BLK,
                 Spiv_dev, Nr_local,
                 Sprime_dev, BLK,
                 u_dev, bw, N, Nq);
        CUDA_CHECK(cudaGetLastError());

        // Hmat <- K(S',S') (the proposal columns [Nr,Nr+b) of
        // Kprop, row major into the b x b row-major Hmat).
        extract_H<<<grid_extract, tpb, 0, stream>>>
                (Hmat_dev, Kprop_dev, Nr_local, rmax);
        CUDA_CHECK(cudaGetLastError());

        // This device's owned, currently-active columns of L^{-1}:
        // global columns [col_start, col_start+kloc) intersect with
        // the pivots S = [0, Nr).  Used by both the H overlap and the
        // M / L^{-1}-append products below.
        size_t const col_start = (size_t) rank * cblk;
        size_t const kloc = (col_start >= Nr_local) ? 0
                : ((Nr_local - col_start < cblk)
                        ? Nr_local - col_start : cblk);

        // Overlap: Hmat <- Hmat - P^* P with P = L^{-1} K(S,S')
        // (Nr x b, col major, leading dim Nr).  Each device holds a
        // contiguous column-block of L^{-1}, so it forms the partial
        // product over its owned columns and the AllReduce sums them.
        // The column-major reinterpretation of Kprop's columns [0,Nr)
        // (leading dim rmax) is exactly K(S,S') (Nr x b), so the
        // device's owned columns [col_start, col_start+kloc) of
        // K(S,S') are the kloc rows at row offset col_start.
        if (Nr_local > 0) {
            if (kloc > 0) {
                double const one = 1.0, zero = 0.0;
                CUBLAS_CHECK(cublasDgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        (int) Nr_local, BLK, (int) kloc,
                        &one,
                        Linv_dev, (int) rmax,
                        Kprop_dev + col_start, (int) rmax,
                        &zero,
                        P_dev, (int) Nr_local));
            } else {
                // This device owns no active columns; contribute 0.
                CUDA_CHECK(cudaMemsetAsync(P_dev, 0,
                            Nr_local * BLK * sizeof(double), stream));
            }

            NCCL_CHECK(ncclAllReduce(P_dev, P_dev,
                        Nr_local * BLK, ncclDouble, ncclSum,
                        comm, stream));

            // Hmat <- Hmat - P^* P (DSYRK, UPPER/OP_T, double).
            // Row-major Hmat is reinterpreted column major; UPPER
            // fills the column-major upper triangle, i.e. the
            // row-major lower triangle rejection_sample reads.
            double const alpha = -1.0, beta = 1.0;
            CUBLAS_CHECK(cublasDsyrk(handle,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    BLK, (int) Nr_local,
                    &alpha,
                    P_dev, (int) Nr_local,
                    &beta,
                    Hmat_dev, BLK));
        }

        // Algorithm 2.1 thinning + inversion (replicated).
        rejection_sample<<<1, BLK, 0, stream>>>
                (Hmat_dev, Lmat_dev, Minv_dev, Mf_dev, Sprime_dev,
                 Sacc_dev, nacc_dev, rstate_dev, tol, accpos_dev);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Read the accepted count (replicated, identical per device).
        size_t nacc = 0;
        CUDA_CHECK(cudaMemcpy(&nacc, nacc_dev, sizeof(size_t),
                    cudaMemcpyDeviceToHost));

        // Step 3: update the residual diagonal and append to L^{-1}.
        if (nacc > 0)
        {
            // M = L^{-*} L^{-1} K(S, S_i) = L^{-*} Q, with
            // Q = P[:,accpos] = L^{-1} K(S, S_i) (Nr x nacc).
            // Each device forms its own row-block Mrow of M from its
            // owned L^{-1} columns; the AllGather then replicates M.
            if (Nr_local > 0)
            {
                int const grid_q = (int)
                        ((Nr_local * nacc + tpb - 1) / tpb);
                gather_cols<<<grid_q, tpb, 0, stream>>>
                        (Q_dev, P_dev, accpos_dev, nacc, Nr_local);
                CUDA_CHECK(cudaGetLastError());

                // Mrow = (Linv[:,Cd])^* Q (kloc x nacc), this
                // device's row-block of M; col-major leading dim cblk.
                if (kloc > 0) {
                    double const one = 1.0, zero = 0.0;
                    CUBLAS_CHECK(cublasDgemm(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            (int) kloc, (int) nacc, (int) Nr_local,
                            &one,
                            Linv_dev, (int) rmax,
                            Q_dev, (int) Nr_local,
                            &zero,
                            Mrow_dev, (int) cblk));
                }

                // AllGather the cblk x nacc row-blocks (blocked layout),
                // then repack into the strided Nr x nacc float M.
                NCCL_CHECK(ncclAllGather(Mrow_dev, Mfull_dev,
                            cblk * nacc, ncclDouble, comm, stream));

                int const grid_rp = (int)
                        ((Nr_local * nacc + tpb - 1) / tpb);
                repack_M<<<grid_rp, tpb, 0, stream>>>
                        (Mfull_f_dev, Mfull_dev, Nr_local, nacc, cblk);
                CUDA_CHECK(cudaGetLastError());
            }

            // Row-block loop over this device's Nrows local rows.
            // Regenerate K(R, S u S_i), remove the overlap to form
            // Tblk = K(R,S_i) - K(R,S) M, apply L_i^{-*} to get
            // Gblk, and subtract its squared row norms from dvec.
            for (size_t irow = 0; irow < Nrows; irow += rblk)
            {
                size_t const nrblk =
                        (Nrows - irow < rblk) ? Nrows - irow : rblk;
                size_t const row_off = (size_t) rank * Nrows + irow;

                // Kblk = K(R, S u S_i): cols [0,Nr) are S (Spiv),
                // [Nr,Nr+nacc) are S_i (Sacc); row major, ld rmax.
                size_t const ablk_cols = Nr_local + nacc;
                int const grid_ab = (int)
                        ((nrblk * ablk_cols + tpb - 1) / tpb);
                compute_kblock<float><<<grid_ab, tpb, 0, stream>>>
                        (Kblk_dev, rmax,
                         NULL, row_off, nrblk,
                         Spiv_dev, Nr_local,
                         Sacc_dev, nacc,
                         u_dev, bw, N, Nq);
                CUDA_CHECK(cudaGetLastError());

                // Tblk <- K(R,S_i) (Kblk cols [Nr,Nr+nacc)).
                CUDA_CHECK(cudaMemcpy2DAsync(
                            Tblk_dev, BLK * sizeof(float),
                            Kblk_dev + Nr_local,
                            rmax * sizeof(float),
                            nacc * sizeof(float), nrblk,
                            cudaMemcpyDeviceToDevice, stream));

                // Tblk <- Tblk - K(R,S) M (nrblk x nacc).
                // Row-major buffers reinterpreted column major; the
                // transpose flags select (Tblk)^T = M^T (K(R,S))^T.
                if (Nr_local > 0) {
                    float const negone = -1.0f, one = 1.0f;
                    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            (int) nacc, (int) nrblk, (int) Nr_local,
                            &negone,
                            Mfull_f_dev, (int) Nr_local,
                            Kblk_dev, (int) rmax,
                            &one,
                            Tblk_dev, BLK));
                }

                // Gblk <- Tblk L_i^{-*} (nrblk x nacc).
                // Row-major reinterpreted column major; OP_T on Mf
                // selects (Gblk)^T = Mf (Tblk)^T.
                float const one = 1.0f, zero = 0.0f;
                CUBLAS_CHECK(cublasSgemm(handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int) nacc, (int) nrblk, (int) nacc,
                        &one,
                        Mf_dev, BLK,
                        Tblk_dev, BLK,
                        &zero,
                        Gblk_dev, BLK));

                // dvec(R) <- max{dvec(R) - rowSqNorms(Gblk), 0}.
                int const grid_ud =
                        (int) ((nrblk + tpb - 1) / tpb);
                update_diag_block<<<grid_ud, tpb, 0, stream>>>
                        (Gblk_dev, dvec_dev, nacc, nrblk, row_off,
                         BLK);
                CUDA_CHECK(cudaGetLastError());
            }

            // Append the new block-row [ -L_i^{-1} M^* | L_i^{-1} ]
            // to L^{-1}.  Each device writes only its owned columns:
            // the overlap part uses this device's local Mrow, the
            // diagonal part writes whichever new columns it owns.
            if (kloc > 0) {
                int const grid_ao =
                        (int) ((nacc * kloc + tpb - 1) / tpb);
                append_Linv_overlap<<<grid_ao, tpb, 0, stream>>>
                        (Linv_dev, Minv_dev, Mrow_dev, Nr_local,
                         nacc, kloc, cblk, rmax);
                CUDA_CHECK(cudaGetLastError());
            }
            int const grid_ad =
                    (int) ((nacc * nacc + tpb - 1) / tpb);
            append_Linv_diag<<<grid_ad, tpb, 0, stream>>>
                    (Linv_dev, Minv_dev, Nr_local, nacc, col_start,
                     cblk, rmax);
            CUDA_CHECK(cudaGetLastError());

            // S <- S u S_i: append the accepted pivots to Spiv.
            CUDA_CHECK(cudaMemcpyAsync(Spiv_dev + Nr_local, Sacc_dev,
                        nacc * sizeof(size_t),
                        cudaMemcpyDeviceToDevice, stream));
        }

        // Step 3c: re-replicate the updated dvec, then re-sum it and
        // record the error.  The AllGather is a no-op when nacc == 0
        // (no device touched its slice), but is called uniformly to
        // stay in NCCL lock-step.
        NCCL_CHECK(ncclAllGather(dvec_dev + (size_t) rank * Nrows,
                    dvec_dev, Nrows, ncclFloat, comm, stream));

        compute_dsum<<<1, dsum_tpb,
                dsum_tpb * sizeof(float), stream>>>
                (dsum_dev, err_dev, dvec_dev, i, NM);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Rank 0 copies the accepted pivots to host.
        if (rank == 0) {
            if (nacc > 0) {
                CUDA_CHECK(cudaMemcpy(&Spiv_host[Nr_local], Sacc_dev,
                            nacc * sizeof(size_t),
                            cudaMemcpyDeviceToHost));
            }
            hm[i] = nacc;
        }
        Nr_local += nacc;
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Rank 0 records the realized rank and error for the report.
    if (rank == 0) {
        Nr = Nr_local;
        CUDA_CHECK(cudaMemcpy(err, err_dev, Nrounds * sizeof(float),
                    cudaMemcpyDeviceToHost));
    }

    // Assemble the distributed L^{-1} on rank 0 and write it.
    // Each rank owns the col-major block of columns
    // [rank*cblk, (rank+1)*cblk) with leading dim rmax.
    // MPI_Gather concatenates the blocks by rank, reconstructing
    // the full rmax x rmax col-major matrix.
    double* Linv_blk = (double*) malloc(rmax * cblk * sizeof(double));
    assert(Linv_blk != NULL);
    CUDA_CHECK(cudaMemcpy(Linv_blk, Linv_dev,
                rmax * cblk * sizeof(double), cudaMemcpyDeviceToHost));

    double* Linv_full = NULL;
    if (rank == 0) {
        Linv_full = (double*) malloc(rmax * rmax * sizeof(double));
        assert(Linv_full != NULL);
    }

    // A two-level datatype (cblk columns of rmax doubles) lets the
    // gather use unit count, avoiding the need to fit rmax*cblk
    // into an int.
    MPI_Datatype col_type, blk_type;
    MPI_CHECK(MPI_Type_contiguous((int) rmax, MPI_DOUBLE, &col_type));
    MPI_CHECK(MPI_Type_contiguous((int) cblk, col_type, &blk_type));
    MPI_CHECK(MPI_Type_commit(&blk_type));

    MPI_CHECK(MPI_Gather(Linv_blk, 1, blk_type,
                Linv_full, 1, blk_type, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_free(&blk_type));
    MPI_CHECK(MPI_Type_free(&col_type));

    // Compact the meaningful top left Nr x Nr block in place,
    // then write it to disk.
    if (rank == 0) {
        for (size_t g = 1; g < Nr_local; ++g) {
            memmove(Linv_full + g * Nr_local, Linv_full + g * rmax,
                    Nr_local * sizeof(double));
        }
        // 1D flattened, column major.
        h5write_double(Linv_full, linv_file, linv_dset,
                Nr_local * Nr_local);
        free(Linv_full);
    }
    free(Linv_blk);

    // Cleanup of this device's resources.
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(rstate_dev);
    cudaFree(err_dev);
    cudaFree(Gblk_dev);
    cudaFree(Tblk_dev);
    cudaFree(Kblk_dev);
    cudaFree(Mfull_f_dev);
    cudaFree(Mfull_dev);
    cudaFree(Mrow_dev);
    cudaFree(Q_dev);
    cudaFree(P_dev);
    cudaFree(Mf_dev);
    cudaFree(Minv_dev);
    cudaFree(Lmat_dev);
    cudaFree(Hmat_dev);
    cudaFree(Kprop_dev);
    cudaFree(accpos_dev);
    cudaFree(nacc_dev);
    cudaFree(Sacc_dev);
    cudaFree(Sprime_dev);
    cudaFree(dsum_dev);
    cudaFree(dvec_dev);
    cudaFree(Spiv_dev);
    cudaFree(Linv_dev);
    cudaFree(u_dev);
    ncclCommDestroy(comm);
    free(u_host);

    // Every rank holds the same replicated result; rank 0 reports it.
    if (rank == 0) {
        printf("\nAccepted rank: %lu in %lu rounds\n", Nr, Nrounds);
        printf("# round accepted cumrank relerr\n");
        size_t cum = 0;
        for (size_t i = 0; i < Nrounds; ++i) {
            cum += hm[i];
            printf("%lu %lu %lu %.5e\n", i, hm[i], cum, err[i]);
        }

        // Write the accepted pivot set S (length Nr).
        h5write_sizet(Spiv_host, spiv_file, spiv_dset, Nr);
    }
    free(err);
    free(hm);
    free(Spiv_host);

    time_t t1 = time(NULL);
    if (rank == 0) {
        printf("Total time: %.2f s\n", difftime(t1, t0));
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
