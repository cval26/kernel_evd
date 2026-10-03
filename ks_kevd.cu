//nvcc ks_kevd.cu -ccbin mpicxx -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lcublas -lcublasmp -lcusolverMp -lnccl -lhdf5 --use_fast_math -o cukevd
//nvcc ks_kevd.cu -ccbin mpicxx -g -G -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lcublas -lcublasmp -lcusolverMp -lnccl -lhdf5 -o cukevd
//run: NCCL_P2P_LEVEL=SYS mpirun -np 4 ./cukevd
//run: CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ./cukevd

// Approximate EVD post-processing after ARPC (MPI + NCCL).
// One MPI process (rank) per GPU.
//
// This consumes the arpclm.cu outputs -- the r x r inverse Cholesky
// factor L^{-1} and the length-r pivot set S, read from HDF5 -- and
// approximates the EVD of the bistochastic normalization of the rank-r
// kernel approximation.
// It never forms the NM x NM kernel or the NM x r factor F;
// it regenerates kernel entries K(R,S) in row blocks on demand,
// exactly as arpclm.cu does.

#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <mpi.h>
#include <hdf5.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasmp.h>
#include <cusolverMp.h>
#include <nccl.h>

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

#define CUBLASMP_CHECK(val) cublasmp_check((val), __FILE__, __LINE__)
inline void cublasmp_check(cublasMpStatus_t st, char const* file,
        int const line)
{
    if (st != CUBLASMP_STATUS_SUCCESS) {
        printf("cuBLASMp error: %s:%i status %d\n", file, line,
                (int) st);
        exit(1);
    }
}

#define CUSOLVERMP_CHECK(val) cusolvermp_check((val), __FILE__, __LINE__)
inline void cusolvermp_check(cusolverStatus_t st, char const* file,
        int const line)
{
    if (st != CUSOLVER_STATUS_SUCCESS) {
        printf("cuSOLVERMp error: %s:%i status %d\n", file, line,
                (int) st);
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

// Report root's GPU memory use at a labeled checkpoint.
void mem_probe(char const* label, int const rank)
{
    if (rank != 0) { return; }
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t freeb = 0, totalb = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
    double const gib = 1024.0 * 1024.0 * 1024.0;
    printf("MEM[%-10s] rank %d: used %.3f GiB  free %.3f GiB\n",
            label, rank, (double) (totalb - freeb) / gib,
            (double) freeb / gib);
    fflush(stdout);
}

// Return the total number of elements of a dataset.
size_t h5_num_elements(char const* filename, char const* dsetname)
{
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file != H5I_INVALID_HID);

    hid_t dset = H5Dopen2(file, dsetname, H5P_DEFAULT);
    assert(dset != H5I_INVALID_HID);

    hid_t dspace = H5Dget_space(dset);
    assert(dspace != H5I_INVALID_HID);

    hssize_t npoints = H5Sget_simple_extent_npoints(dspace);
    assert(npoints > 0);

    H5Sclose(dspace);
    H5Dclose(dset);
    H5Fclose(file);
    return (size_t) npoints;
}

// Read N float values from a dataset.
int h5read_float(float* data, char const* filename, char const* dsetname,
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
    free(dims);

    assert(data != NULL);
    h5stat = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, data);
    assert(h5stat >= 0);

    H5Sclose(dspace);
    H5Dclose(dset);
    H5Fclose(file);
    return 0;
}

// Read N double values from a dataset.
int h5read_double(double* data, char const* filename,
        char const* dsetname, size_t const rank, size_t const N)
{
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file != H5I_INVALID_HID);

    hid_t dset = H5Dopen2(file, dsetname, H5P_DEFAULT);
    assert(dset != H5I_INVALID_HID);

    hid_t dspace = H5Dget_space(dset);
    assert(dspace != H5I_INVALID_HID);

    hsize_t* dims = (hsize_t*) malloc(rank * sizeof(hsize_t));
    assert(dims != NULL);

    herr_t h5stat = H5Sget_simple_extent_dims(dspace, dims, NULL);
    assert(h5stat >= 0);

    size_t datasize = 1;
    for (size_t i = 0; i < rank; ++i) {
        datasize *= dims[i];
    }
    assert(datasize == N);
    free(dims);

    assert(data != NULL);
    h5stat = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, data);
    assert(h5stat >= 0);

    H5Sclose(dspace);
    H5Dclose(dset);
    H5Fclose(file);
    return 0;
}

// Read N size_t values from a dataset.
// The native unsigned long type matches the
// h5write_sizet writer in arpclm.cu.
int h5read_sizet(size_t* data, char const* filename,
        char const* dsetname, size_t const rank, size_t const N)
{
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file != H5I_INVALID_HID);

    hid_t dset = H5Dopen2(file, dsetname, H5P_DEFAULT);
    assert(dset != H5I_INVALID_HID);

    hid_t dspace = H5Dget_space(dset);
    assert(dspace != H5I_INVALID_HID);

    hsize_t* dims = (hsize_t*) malloc(rank * sizeof(hsize_t));
    assert(dims != NULL);

    herr_t h5stat = H5Sget_simple_extent_dims(dspace, dims, NULL);
    assert(h5stat >= 0);

    size_t datasize = 1;
    for (size_t i = 0; i < rank; ++i) {
        datasize *= dims[i];
    }
    assert(datasize == N);
    free(dims);

    assert(data != NULL);
    h5stat = H5Dread(dset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, data);
    assert(h5stat >= 0);

    H5Sclose(dspace);
    H5Dclose(dset);
    H5Fclose(file);
    return 0;
}

// Append one 1D double dataset to an already-open HDF5 file.
// Matrices are written flat in their native column-major order.
void h5add_double(hid_t file, char const* dsetname,
        double const* data, size_t const N)
{
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
}

// Float twin of h5add_double, for the pass-4 output U.
// U is streamed in float and dumped flat in its row-major order.
void h5add_float(hid_t file, char const* dsetname,
        float const* data, size_t const N)
{
    hsize_t dims[1] = { (hsize_t) N };
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    assert(dspace != H5I_INVALID_HID);

    hid_t dset = H5Dcreate2(file, dsetname, H5T_NATIVE_FLOAT, dspace,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset != H5I_INVALID_HID);

    assert(data != NULL);
    herr_t h5stat = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, data);
    assert(h5stat >= 0);

    H5Dclose(dset);
    H5Sclose(dspace);
}

__device__ inline double kexp_t(double x) { return exp(x); }
__device__ inline float kexp_t(float x) { return expf(x); }

// Kernel-value block evaluator.
// Writes a row-major block out[r*ld + c] = K(row_glob(r), col_glob(c))
// for r in [0,nrows), c in [0, npiv+ntail).
// Rows: rowidx[r] if rowidx != NULL, else the contiguous row_offset + r
// (the streaming row blocks use the contiguous form).
// Cols: Spiv[c] for c < npiv (the pivots S), else tail[c - npiv].
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

// Accumulate a panel's float column sums into the double c1 vector.
// The per-panel sums are float (the panel is float, and each is a sum
// of only rblk kernel values), but c1 runs over all NM rows, so the
// cross-panel accumulation is double per the long-sum convention.
__global__
void accum_colsums(double* c1, float const* csum, size_t const Nr)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t j = tid; j < Nr; j += stride) {
        c1[j] += (double) csum[j];
    }
}

// Elementwise reciprocal dinv = 1 / d over n entries (float).
// Used for the pass-2 row scaling diag(dtil^{-1}) and the cq reduction
// K(R,S)^T dtil^{-1}.
__global__
void recip_float(float* dinv, float const* d, size_t const n)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) {
        dinv[i] = 1.0f / d[i];
    }
}

// Elementwise reciprocal square root qinv = 1 / sqrt(q) over n entries.
// Used for the pass-3 row scaling diag(qtil^{-1/2}): the symmetric split
// of the Btil weighting, so Khat^T Khat = K(S,:) diag(qtil^{-1}) K(:,S).
__global__
void rsqrt_float(float* qinv, float const* q, size_t const n)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) {
        qinv[i] = rsqrtf(q[i]);
    }
}

// Build the double, column-major syrk source panel from the float
// kernel panel and the per-row reciprocal dinv.
// K is row major with leading dimension Nr (K[i*Nr + s]); src is the
// column-major (nrblk x Nr) panel with leading dimension nrblk
// (src[i + s*nrblk]), the layout the P x 1 source descriptor expects.
// The row scaling diag(dtil^{-1}) K is carried in float (the streaming
// precision), then cast to the double source.
__global__
void scale_cast_panel(double* src, float const* K, float const* dinv,
        size_t const nrblk, size_t const Nr)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < nrblk * Nr; idx += stride) {
        size_t const i = idx % nrblk;
        size_t const s = idx / nrblk;
        src[idx] = (double) (K[i * Nr + s] * dinv[i]);
    }
}

// Apply the pass-4 row scaling diag(dtil^{-1}) to the U panel in place.
// U is the column-major k_out x nrblk buffer (K(R,S) Z)^T, so its
// logical row i (one output row) is the contiguous chunk
// U[i*k_out .. (i+1)*k_out); scaling that chunk by dinv[i] gives
// U(R) = diag(dtil(R)^{-1}) K(R,S) Z in the same row-major layout.
__global__
void scale_rows_rm(float* U, float const* dinv,
        size_t const nrblk, size_t const k_out)
{
    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < nrblk * k_out; idx += stride) {
        size_t const i = idx / k_out;
        U[idx] *= dinv[i];
    }
}

int main(int argc, char** argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_size = 0, rank = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    // Node-local rank for GPU binding.
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

    size_t constexpr Nr = 32122; // realized rank

    float constexpr bw = 0.8; // kernel bandwidth

    constexpr size_t Ndevs = 4; // GPU devices
    size_t constexpr Nrows = NM / Ndevs; // local rows per device

    // Ndevs must divide NM (1D row-block decomposition).
    static_assert(NM % Ndevs == 0, "Ndevs must divide NM");

    // Streaming panel height; chosen on memory grounds, but must
    // divide Nrows.
    // Each K(R,S) panel is rblk x Nr float.
    size_t constexpr rblk = 16384;
    static_assert(Nrows % rblk == 0, "rblk must divide Nrows");

    int constexpr tpb = 512;

    // Block size of the 2-D block-cyclic r x r matrices.
    // The design's rule is the largest nb leaving >= ~8 blocks per
    // dimension, which is 512 at Nr = 4084 (8 per dimension, 4 per rank
    // on a 2 x 2 grid).
    // Square blocks (mb == nb) are required by cusolverMpSyevd.
    // See docs/DESIGN_evd2.md, "Choosing the block size nb".
    size_t constexpr nb = 4096;

    // Number of leading eigenvectors kept (the trailing k_out columns of
    // V, since syevd returns eigenvalues ascending).  "A few hundred"
    // per the design; a placeholder for the test regime, tunable here.
    // Matches scripts/evd_oracle.py's default --kout 256 so the oracle
    // Z / U checks line up without a flag.  Must be <= Nr (asserted at
    // the dense-2 Z step once Nr is known).
    size_t constexpr k_out = 256;

    // Dump the intermediate results for testing.
    bool constexpr dump_h5 = false;

    // Probe GPU memory usage.
    bool constexpr mem_check = false;

    // EVD output file.
    char out_file[256];
    snprintf(out_file, sizeof(out_file),
            "data/evd_%lu_%lu_%lu_%lu_%lu.h5",
            N, M, Nq, k_out, Lfact);
    char constexpr out_dset1[] = "/lam";
    char constexpr out_dset2[] = "/U";

    if (rank == 0) {
        hid_t file = H5Fcreate(out_file, H5F_ACC_TRUNC,
                H5P_DEFAULT, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        H5Fclose(file);
    }

    // State training data file.
    // Data is stored with the time index varying first.
    char fname[128];
    sprintf(fname, "data/ks_true_%lu_%lu_%lu_u32.h5", Ndata, M, Lfact);
    char constexpr dset[] = "/u";

    // ARPC files.
    char linv_file[128];
    sprintf(linv_file, "data/linv_%lu_%lu_b512_r64_%lu.h5",
            N, M, Lfact);
    char constexpr linv_dset[] = "/linv";

    char spiv_file[128];
    sprintf(spiv_file, "data/spiv_%lu_%lu_b512_r64_%lu.h5",
            N, M, Lfact);
    char constexpr spiv_dset[] = "/spiv";

    // Discover the realized rank r and validate Nr.
    {
        size_t const nr = h5_num_elements(spiv_file, spiv_dset);
        assert(nr == Nr);
    }
    if (rank == 0) {
        size_t const linv_len = h5_num_elements(linv_file, linv_dset);
        assert(linv_len == Nr * Nr);
    }

    // Create the dump file.
    // Every dump then opens it and appends.
    char dump_file[128];
    sprintf(dump_file, "data/kevd_dump_%lu_%lu_%lu.h5",
            N, M, Lfact);

    if (dump_h5 && rank == 0) {
        hid_t file = H5Fcreate(dump_file, H5F_ACC_TRUNC, H5P_DEFAULT,
                H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        H5Fclose(file);
    }

    if (world_size != (int) Ndevs) {
        if (rank == 0) {
            printf("Error: launched with %d ranks, need Ndevs = %lu\n",
                    world_size, Ndevs);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Process grid for the 2-D block-cyclic r x r matrices.
    // Square-ish factorization, column major (rank = myrow +
    // mycol * nprow), matching kevd_wsprobe.cu and bench_pxsyrk.cu.
    size_t nprow = (size_t) llround(sqrt((double) world_size));
    while (nprow > 1 && world_size % (int) nprow != 0) {
        --nprow;
    }
    size_t const npcol = (size_t) world_size / nprow;
    size_t const myrow = (size_t) rank % nprow;
    size_t const mycol = (size_t) rank / nprow;

    if (rank == 0) {
        printf("ARPC inputs: %s (%s), %s (%s)\n",
                spiv_file, &spiv_dset[1], linv_file, &linv_dset[1]);
        printf("Data file: %s\nData set: %s\n", fname, &dset[1]);
        printf("Realized rank r = %lu\n", Nr);
        printf("Devices: %lu  NM: %lu  Rows/device: %lu\n",
                Ndevs, NM, Nrows);
        printf("Panel: %lu rows  Panels/device: %lu\n",
                rblk, (Nrows + rblk - 1) / rblk);
        printf("Grid: %lu x %lu (col major)  nb: %lu\n\n",
                nprow, npcol, nb);
    }

    // Replicated; every rank reads independently.
    // Every rank needs the whole pivot set: compute_kblock regenerates
    // this rank's own rows against all Nr pivot columns.
    size_t* Spiv_host = (size_t*) malloc(Nr * sizeof(size_t));
    assert(Spiv_host != NULL);
    h5read_sizet(Spiv_host, spiv_file, spiv_dset, 1, Nr);

    // L^{-1} is read on rank 0 alone: it is the scatter root,
    // and the block-cyclic tiles are placed by
    // cusolverMpMatrixScatterH2D.
    // Column major, lower triangular; the flat Nr*Nr dataset is
    // read straight into the scatter's host source buffer.
    double* Linv_host = NULL;
    if (rank == 0) {
        Linv_host = (double*) malloc(Nr * Nr * sizeof(double));
        assert(Linv_host != NULL);
        h5read_double(Linv_host, linv_file, linv_dset, 1, Nr * Nr);
    }

    float* u_host = (float*) malloc(NMdata * sizeof(float));
    assert(u_host != NULL);
    h5read_float(u_host, fname, dset, 1, NMdata);

    // Rank 0 creates the unique NCCL ID and broadcasts it.
    // Every rank then joins the communicator.
    ncclUniqueId ncclid;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&ncclid));
    }
    MPI_CHECK(MPI_Bcast(&ncclid, sizeof(ncclid), MPI_BYTE, 0,
                MPI_COMM_WORLD));

    // Each rank drives its own GPU with a private stream,
    // cuBLAS handle, and NCCL communicator.
    CUDA_CHECK(cudaSetDevice(local_rank));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, ncclid, rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Both distributed libraries take the NCCL communicator.
    // Each needs its own grid object over the same communicator.
    cusolverMpHandle_t shandle;
    CUSOLVERMP_CHECK(cusolverMpCreate(&shandle, local_rank, stream));

    cublasMpHandle_t bhandle;
    CUBLASMP_CHECK(cublasMpCreate(&bhandle, stream));

    cusolverMpGrid_t sgrid;
    CUSOLVERMP_CHECK(cusolverMpCreateDeviceGrid(shandle, &sgrid, comm,
                (int32_t) nprow, (int32_t) npcol,
                CUSOLVERMP_GRID_MAPPING_COL_MAJOR));

    cublasMpGrid_t bgrid;
    CUBLASMP_CHECK(cublasMpGridCreate((int64_t) nprow, (int64_t) npcol,
                CUBLASMP_GRID_LAYOUT_COL_MAJOR, comm, &bgrid));

    // r-independent memory baseline (context, libraries, NCCL).
    if (mem_check) { mem_probe("post-init", rank); }

    // Device storage buffers.
    // State data, replicated.
    float* u_dev;
    CUDA_CHECK(cudaMalloc(&u_dev, NMdata * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(u_dev, u_host,
                NMdata * sizeof(float), cudaMemcpyHostToDevice));

    // Pivot set S, replicated.
    size_t* Spiv_dev;
    CUDA_CHECK(cudaMalloc(&Spiv_dev, Nr * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(Spiv_dev, Spiv_host,
                Nr * sizeof(size_t), cudaMemcpyHostToDevice));

    // Inverse Cholesky factor L^{-1}: Nr x Nr, column major,
    // lower triangular, 2D block cyclic over the nprow x npcol grid.
    // cublasMpNumroc sets the extent of the local blocks.
    size_t const loc_m = (size_t) cublasMpNumroc((int64_t) Nr,
            (int64_t) nb, (uint32_t) myrow, 0, (uint32_t) nprow);
    size_t const loc_n = (size_t) cublasMpNumroc((int64_t) Nr,
            (int64_t) nb, (uint32_t) mycol, 0, (uint32_t) npcol);
    size_t const lld = (loc_m > 0) ? loc_m : 1;

    // A rank owning no block of a dimension still needs a positive
    // lld and a non-null pointer, hence the floors.
    size_t const tile_elems = (loc_n > 0) ? lld * loc_n : 1;

    double* Linv_bc;
    CUDA_CHECK(cudaMalloc(&Linv_bc, tile_elems * sizeof(double)));

    // Descriptors for the same tile in both libraries.
    cusolverMpMatrixDescriptor_t sdescLinv;
    CUSOLVERMP_CHECK(cusolverMpCreateMatrixDesc(&sdescLinv, sgrid,
                CUDA_R_64F, (int64_t) Nr, (int64_t) Nr,
                (int64_t) nb, (int64_t) nb, 0, 0, (int64_t) lld));

    cublasMpMatrixDescriptor_t bdescLinv;
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
                (int64_t) Nr, (int64_t) Nr, (int64_t) nb, (int64_t) nb,
                0, 0, (int64_t) lld, CUDA_R_64F, bgrid, &bdescLinv));

    // Scatter L^{-1} from rank 0's host buffer to the block-cyclic tiles.
    // Every rank calls it, each passing its own tile; only the
    // root's source is read.
    // The descriptor carries the layout, so no block-cyclic index
    // arithmetic is required.
    // Documented as a utility routine, not a fast path.
    CUSOLVERMP_CHECK(cusolverMpMatrixScatterH2D(shandle,
                (int64_t) Nr, (int64_t) Nr,
                Linv_bc, 1, 1, sdescLinv,
                0, Linv_host, (int64_t) Nr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Zero-fill the strict upper triangle of L^{-1}.
    // Applying uplo = UPPER with alpha = beta = 0 to the (Nr-1)x(Nr-1)
    // submatrix at (1,2) zeroes that block's upper triangle *including
    // its own diagonal*, which is exactly the parent's strict upper
    // triangle; the parent diagonal and lower triangle are untouched.
    // Laset constrains IA, JA only to be >= 1, so this is legal at any
    // nb and needs no index arithmetic.
    int* dinfo_dev;
    CUDA_CHECK(cudaMalloc(&dinfo_dev, sizeof(int)));
    {
        double const zero_d = 0.0;
        CUSOLVERMP_CHECK(cusolverMpLaset(shandle,
                    CUBLAS_FILL_MODE_UPPER,
                    (int64_t) (Nr - 1), (int64_t) (Nr - 1),
                    &zero_d, &zero_d,
                    Linv_bc, 1, 2, sdescLinv, dinfo_dev));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int info = 0;
        CUDA_CHECK(cudaMemcpy(&info, dinfo_dev, sizeof(int),
                    cudaMemcpyDeviceToHost));
        assert(info == 0);
    }

    // Baseline after the first distributed calls (scatter + Laset) have
    // lazy-loaded their library modules; plus the one Linv_bc tile.
    if (mem_check) { mem_probe("post-laset", rank); }

    // Pass-1 buffers.
    // Kblk holds one regenerated panel K(R,S): rblk x Nr, row major,
    // leading dimension Nr.
    float* Kblk_dev;
    CUDA_CHECK(cudaMalloc(&Kblk_dev, rblk * Nr * sizeof(float)));

    // The all-ones vector the panel column sums contract against.
    float* ones_dev;
    CUDA_CHECK(cudaMalloc(&ones_dev, rblk * sizeof(float)));
    {
        float* ones_host = (float*) malloc(rblk * sizeof(float));
        assert(ones_host != NULL);
        for (size_t i = 0; i < rblk; ++i) {
            ones_host[i] = 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(ones_dev, ones_host,
                    rblk * sizeof(float), cudaMemcpyHostToDevice));
        free(ones_host);
    }

    // Per-panel column sums and the c1 accumulator.
    float* csum_dev;
    CUDA_CHECK(cudaMalloc(&csum_dev, Nr * sizeof(float)));

    double* c1_dev;
    CUDA_CHECK(cudaMalloc(&c1_dev, Nr * sizeof(double)));
    CUDA_CHECK(cudaMemset(c1_dev, 0, Nr * sizeof(double)));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (mem_check) { mem_probe("pre-pass1", rank); }

    // Pass 1 -- c1 = K(:,S)^T 1.
    // Sweep this device's Nrows local rows in rblk-row panels,
    // regenerating K(R,S) and reducing its column sums into c1.
    // The rows partition across devices, so the pass ends in one
    // AllReduce.
    for (size_t irow = 0; irow < Nrows; irow += rblk)
    {
        size_t const nrblk =
                (Nrows - irow < rblk) ? Nrows - irow : rblk;
        size_t const row_off = (size_t) rank * Nrows + irow;

        // Kblk = K(R,S): row major, ld Nr, no tail columns.
        int const grid_kblk =
                (int) ((nrblk * Nr + tpb - 1) / tpb);
        compute_kblock<float><<<grid_kblk, tpb, 0, stream>>>
                (Kblk_dev, Nr,
                 NULL, row_off, nrblk,
                 Spiv_dev, Nr,
                 NULL, 0,
                 u_dev, bw, N, Nq);
        CUDA_CHECK(cudaGetLastError());

        // csum <- K(R,S)^T 1 (length Nr).
        // The row-major panel reinterpreted column major is the
        // Nr x nrblk matrix K(R,S)^T, so OP_N against the ones vector
        // sums each pivot column over the panel's rows.
        float const one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                    (int) Nr, (int) nrblk,
                    &one,
                    Kblk_dev, (int) Nr,
                    ones_dev, 1,
                    &zero,
                    csum_dev, 1));

        // c1 <- c1 + csum, accumulating in double.
        int const grid_ac = (int) ((Nr + tpb - 1) / tpb);
        accum_colsums<<<grid_ac, tpb, 0, stream>>>
                (c1_dev, csum_dev, Nr);
        CUDA_CHECK(cudaGetLastError());
    }

    // Sum the per-device partials -> c1 replicated on every rank.
    NCCL_CHECK(ncclAllReduce(c1_dev, c1_dev, Nr, ncclDouble, ncclSum,
                comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy the reduced c1 to host: it is the p1 scatter source below and
    // is dumped here (fully known on every rank after the AllReduce).
    double* c1_host = (double*) malloc(Nr * sizeof(double));
    assert(c1_host != NULL);
    CUDA_CHECK(cudaMemcpy(c1_host, c1_dev, Nr * sizeof(double),
                cudaMemcpyDeviceToHost));

    if (dump_h5 && rank == 0) {
        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, "c1", c1_host, Nr);
        H5Fclose(file);
        printf("Pass 1: dumped c1\n");
    }

    // Dense infrastructure shared by p1 and pq.
    // p1 = L^{-T}(L^{-1} c1) and pq = L^{-T}(L^{-1} cq) are the same two
    // triangular matvecs against the 2D block cyclic L^{-1}.
    // The descriptors, buffers and workspace below are built once
    // and reused by the inline p1 and pq solves below.
    // The r x 1 operands carry L^{-1}'s row blocking, as trmm requires;
    // with n = 1 < nb only process column 0 owns anything, so loc_n_v
    // is 1 there and 0 elsewhere.
    size_t const loc_n_v = (size_t) cublasMpNumroc(1, (int64_t) nb,
            (uint32_t) mycol, 0, (uint32_t) npcol);
    size_t const vec_elems = (loc_n_v > 0) ? lld * loc_n_v : 1;

    double* vec_a; // rhs scattered, then the result p
    double* vec_b; // L^{-1} rhs
    CUDA_CHECK(cudaMalloc(&vec_a, vec_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vec_b, vec_elems * sizeof(double)));

    cusolverMpMatrixDescriptor_t sdescVec;
    CUSOLVERMP_CHECK(cusolverMpCreateMatrixDesc(&sdescVec, sgrid,
                CUDA_R_64F, (int64_t) Nr, 1,
                (int64_t) nb, (int64_t) nb, 0, 0, (int64_t) lld));

    cublasMpMatrixDescriptor_t bdescVec;
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
                (int64_t) Nr, 1, (int64_t) nb, (int64_t) nb,
                0, 0, (int64_t) lld, CUDA_R_64F, bgrid, &bdescVec));

    // cuBLASMp has no trmv/gemv-class entry point, so each triangular
    // matvec is a trmm with a single-column RHS.
    // trmm is out-of-place (C distinct from B), hence the two buffers.
    // One workspace pair serves both the OP_N and OP_T calls and both
    // solves.
    double const one_d = 1.0;
    size_t trmm_dsz = 0, trmm_hsz = 0;
    {
        size_t dsz_n = 0, hsz_n = 0, dsz_t = 0, hsz_t = 0;
        CUBLASMP_CHECK(cublasMpTrmm_bufferSize(bhandle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    (int64_t) Nr, 1, &one_d,
                    Linv_bc, 1, 1, bdescLinv,
                    vec_a, 1, 1, bdescVec,
                    vec_b, 1, 1, bdescVec,
                    CUBLAS_COMPUTE_64F, &dsz_n, &hsz_n));
        CUBLASMP_CHECK(cublasMpTrmm_bufferSize(bhandle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    (int64_t) Nr, 1, &one_d,
                    Linv_bc, 1, 1, bdescLinv,
                    vec_b, 1, 1, bdescVec,
                    vec_a, 1, 1, bdescVec,
                    CUBLAS_COMPUTE_64F, &dsz_t, &hsz_t));
        trmm_dsz = (dsz_n > dsz_t) ? dsz_n : dsz_t;
        trmm_hsz = (hsz_n > hsz_t) ? hsz_n : hsz_t;
    }

    void* trmm_dwork = NULL;
    if (trmm_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&trmm_dwork, trmm_dsz));
    }
    void* trmm_hwork = NULL;
    if (trmm_hsz > 0) {
        trmm_hwork = malloc(trmm_hsz);
        assert(trmm_hwork != NULL);
    }

    // p1 = L^{-T}(L^{-1} c1), computed block-cyclic in vec_a, then
    // replicated on every device as float for the pass-2 dtil matvec.
    // The solve: scatter the replicated rhs (rank 0's host copy is the
    // source, the other ranks' are ignored) to the r x 1 block-cyclic
    // column, then apply L^{-1} and L^{-T} by trmm.  cuBLASMp has no trmv,
    // so each is a single-column trmm, out-of-place (vec_b != vec_a).
    CUSOLVERMP_CHECK(cusolverMpMatrixScatterH2D(shandle,
                (int64_t) Nr, 1,
                vec_a, 1, 1, sdescVec,
                0, c1_host, (int64_t) Nr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // vec_b <- L^{-1} c1.
    CUBLASMP_CHECK(cublasMpTrmm(bhandle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, 1, &one_d,
                Linv_bc, 1, 1, bdescLinv,
                vec_a, 1, 1, bdescVec,
                vec_b, 1, 1, bdescVec,
                CUBLAS_COMPUTE_64F, trmm_dwork, trmm_dsz,
                trmm_hwork, trmm_hsz));

    // vec_a <- L^{-T} vec_b = p1.
    CUBLASMP_CHECK(cublasMpTrmm(bhandle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, 1, &one_d,
                Linv_bc, 1, 1, bdescLinv,
                vec_b, 1, 1, bdescVec,
                vec_a, 1, 1, bdescVec,
                CUBLAS_COMPUTE_64F, trmm_dwork, trmm_dsz,
                trmm_hwork, trmm_hsz));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Replicate p1 as float on every device (mirrors the Z step): gather
    // the block-cyclic result to root, dump the double, cast on root,
    // broadcast the float, upload.  A plain ncclAllGather would return the
    // block-cyclic row blocks interleaved in the wrong order, so the root
    // bridge (GatherD2H + Bcast) is kept.  vec_a is not freed -- pq's
    // solve reuses it.
    double* p1d_host = NULL;
    if (rank == 0) {
        p1d_host = (double*) malloc(Nr * sizeof(double));
        assert(p1d_host != NULL);
    }
    CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                (int64_t) Nr, 1,
                vec_a, 1, 1, sdescVec,
                0, p1d_host, (int64_t) Nr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (dump_h5 && rank == 0) {
        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, "p1", p1d_host, Nr);
        H5Fclose(file);
        printf("Pass 1: dumped p1\n");
    }

    float* p1f_host = (float*) malloc(Nr * sizeof(float));
    assert(p1f_host != NULL);
    if (rank == 0) {
        for (size_t j = 0; j < Nr; ++j) {
            p1f_host[j] = (float) p1d_host[j];
        }
        free(p1d_host);
        p1d_host = NULL;
    }
    MPI_CHECK(MPI_Bcast(p1f_host, (int) Nr, MPI_FLOAT, 0,
                MPI_COMM_WORLD));

    // p1 replicated on device as float; it lives until the pass-2 dtil
    // matvec consumes it.
    float* p1f_dev;
    CUDA_CHECK(cudaMalloc(&p1f_dev, Nr * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(p1f_dev, p1f_host, Nr * sizeof(float),
                cudaMemcpyHostToDevice));
    free(p1f_host);

    // Pass 2 -- dtil, cq, Atil.
    // Sweep this device's Nrows local rows in rblk-row panels,
    // synchronized across ranks.
    // Every rank has the same Nrows and rblk, so the panel boundaries
    // and the collective Gemr2D and Syrk calls line up.
    // For each panel:
    // dtil(R) = K(R,S) p1 (local)
    // assert and report min(dtil) > 0
    // reduce cq += K(R,S)^T dtil^{-1}
    // pre-scale Ktil = diag(dtil^{-1}) K(R,S)
    // redistribute the panel to 2-D block cyclic
    // accumulate Atil += Ktil^T Ktil via pXsyrk(beta=1).

    // The synchronized sweep requires uniform panel heights,
    // so rblk must divide Nrows.
    // Every panel is the full rblk rows;
    // the global panel is Npanel = Ndevs * rblk rows;
    // the descriptors and workspaces are sized once.
    size_t constexpr Npanel = Ndevs * rblk;

    double const one_pd = 1.0, zero_pd = 0.0;
    float const one_f = 1.0f, zero_f = 0.0f;

    // Source panel: 1D row block over an Ndevs x 1 grid;
    // each rank owns one contiguous rblk-row block with all Nr
    // columns local -- the streaming layout.
    // The redistribute below bridges it to the 2D block-cyclic syrk
    // operand; the syrk contracts the Npanel rows, so the panel-global
    // row identity is irrelevant and each rank simply fills its own
    // kernel rows.
    cublasMpGrid_t bgrid_src;
    CUBLASMP_CHECK(cublasMpGridCreate((int64_t) Ndevs, 1,
                CUBLASMP_GRID_LAYOUT_COL_MAJOR, comm, &bgrid_src));

    // loc_m_s is rblk (one block per rank);
    // the single wide column block keeps all Nr columns local.
    size_t const loc_m_s = (size_t) cublasMpNumroc((int64_t) Npanel,
            (int64_t) rblk, (uint32_t) rank, 0, (uint32_t) Ndevs);
    size_t const lld_s = (loc_m_s > 0) ? loc_m_s : 1;

    // Ktil(R,S) as double, column major (rblk x Nr), the Gemr2D source.
    double* srcpanel_dev;
    CUDA_CHECK(cudaMalloc(&srcpanel_dev, rblk * Nr * sizeof(double)));

    cublasMpMatrixDescriptor_t bdescPanelSrc;
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
                (int64_t) Npanel, (int64_t) Nr,
                (int64_t) rblk, (int64_t) Nr,
                0, 0, (int64_t) lld_s, CUDA_R_64F, bgrid_src,
                &bdescPanelSrc));

    // Destination panel: same Npanel x Nr, 2D block cyclic,
    // the syrk operand.
    size_t const loc_m_p = (size_t) cublasMpNumroc((int64_t) Npanel,
            (int64_t) nb, (uint32_t) myrow, 0, (uint32_t) nprow);
    size_t const loc_n_p = (size_t) cublasMpNumroc((int64_t) Nr,
            (int64_t) nb, (uint32_t) mycol, 0, (uint32_t) npcol);
    size_t const lld_p = (loc_m_p > 0) ? loc_m_p : 1;
    size_t const panel_elems =
            (loc_m_p * loc_n_p > 0) ? loc_m_p * loc_n_p : 1;

    double* dstpanel_dev;
    CUDA_CHECK(cudaMalloc(&dstpanel_dev, panel_elems * sizeof(double)));

    cublasMpMatrixDescriptor_t bdescPanelDst;
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
                (int64_t) Npanel, (int64_t) Nr,
                (int64_t) nb, (int64_t) nb,
                0, 0, (int64_t) lld_p, CUDA_R_64F, bgrid, &bdescPanelDst));

    // Atil (Nr x Nr, SPD), 2D block cyclic.
    // Same shape and blocking as L^{-1}, so it reuses the local extents
    // loc_m / loc_n / lld and the tile_elems allocation size.
    double* Atil_dev;
    CUDA_CHECK(cudaMalloc(&Atil_dev, tile_elems * sizeof(double)));

    cublasMpMatrixDescriptor_t bdescAtil;
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
                (int64_t) Nr, (int64_t) Nr, (int64_t) nb, (int64_t) nb,
                0, 0, (int64_t) lld, CUDA_R_64F, bgrid, &bdescAtil));

    // Per-panel reciprocal dtil^{-1}, this rank's contiguous dtil, and
    // the replicated dtil after the AllGather.
    float* dinv_dev;
    CUDA_CHECK(cudaMalloc(&dinv_dev, rblk * sizeof(float)));
    float* dtil_loc_dev;
    CUDA_CHECK(cudaMalloc(&dtil_loc_dev, Nrows * sizeof(float)));
    float* dtil_full_dev;
    CUDA_CHECK(cudaMalloc(&dtil_full_dev, NM * sizeof(float)));

    // cq accumulator (double, a long sum over NM rows, as c1).
    double* cq_dev;
    CUDA_CHECK(cudaMalloc(&cq_dev, Nr * sizeof(double)));
    CUDA_CHECK(cudaMemset(cq_dev, 0, Nr * sizeof(double)));

    // Host scratch for the dtil / qtil positivity checks.
    float* dtilp_host = (float*) malloc(Nrows * sizeof(float));
    assert(dtilp_host != NULL);

    // Redistribute and syrk workspaces (sizes fixed across panels).
    size_t gem_dsz = 0, gem_hsz = 0;
    CUBLASMP_CHECK(cublasMpGemr2D_bufferSize(bhandle,
                (int64_t) Npanel, (int64_t) Nr,
                srcpanel_dev, 1, 1, bdescPanelSrc,
                dstpanel_dev, 1, 1, bdescPanelDst,
                &gem_dsz, &gem_hsz, comm));
    void* gem_dwork = NULL;
    if (gem_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&gem_dwork, gem_dsz));
    }
    void* gem_hwork = (gem_hsz > 0) ? malloc(gem_hsz) : NULL;

    size_t syrk_dsz = 0, syrk_hsz = 0;
    CUBLASMP_CHECK(cublasMpSyrk_bufferSize(bhandle,
                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                (int64_t) Nr, (int64_t) Npanel,
                &one_pd, dstpanel_dev, 1, 1, bdescPanelDst,
                &zero_pd, Atil_dev, 1, 1, bdescAtil,
                CUBLAS_COMPUTE_64F, &syrk_dsz, &syrk_hsz));
    void* syrk_dwork = NULL;
    if (syrk_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&syrk_dwork, syrk_dsz));
    }
    void* syrk_hwork = (syrk_hsz > 0) ? malloc(syrk_hsz) : NULL;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    for (size_t irow = 0; irow < Nrows; irow += rblk)
    {
        size_t const row_off = (size_t) rank * Nrows + irow;

        // Kblk = K(R,S): row major, ld Nr, no tail columns.
        int const grid_kblk = (int) ((rblk * Nr + tpb - 1) / tpb);
        compute_kblock<float><<<grid_kblk, tpb, 0, stream>>>
                (Kblk_dev, Nr,
                 NULL, row_off, rblk,
                 Spiv_dev, Nr,
                 NULL, 0,
                 u_dev, bw, N, Nq);
        CUDA_CHECK(cudaGetLastError());

        // dtil(R) = K(R,S) p1 (float).
        // Positivity is checked after the streaming loop.
        // The row-major panel reinterpreted column major is the
        // Nr x rblk matrix K(R,S)^T, so OP_T applies K(R,S) to p1; the
        // result lands in this panel's slice of dtil_loc.
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                    (int) Nr, (int) rblk,
                    &one_f, Kblk_dev, (int) Nr,
                    p1f_dev, 1, &zero_f,
                    dtil_loc_dev + irow, 1));

        // dinv = 1 / dtil (float).
        int const grid_rc = (int) ((rblk + tpb - 1) / tpb);
        recip_float<<<grid_rc, tpb, 0, stream>>>
                (dinv_dev, dtil_loc_dev + irow, rblk);
        CUDA_CHECK(cudaGetLastError());

        // cq += K(R,S)^T dtil^{-1}, accumulating in double.
        // Same OP_N contraction as pass 1's c1,
        // with dinv in place of the ones vector.
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                    (int) Nr, (int) rblk,
                    &one_f, Kblk_dev, (int) Nr,
                    dinv_dev, 1, &zero_f,
                    csum_dev, 1));
        int const grid_ac = (int) ((Nr + tpb - 1) / tpb);
        accum_colsums<<<grid_ac, tpb, 0, stream>>>
                (cq_dev, csum_dev, Nr);
        CUDA_CHECK(cudaGetLastError());

        // Ktil = diag(dtil^{-1}) K(R,S)
        // scaled in float then cast into the double col-major
        // source panel (lld rblk).
        int const grid_sc = (int) ((rblk * Nr + tpb - 1) / tpb);
        scale_cast_panel<<<grid_sc, tpb, 0, stream>>>
                (srcpanel_dev, Kblk_dev, dinv_dev, rblk, Nr);
        CUDA_CHECK(cudaGetLastError());

        // Redistribute the row-complete panel into 2D block cyclic.
        CUBLASMP_CHECK(cublasMpGemr2D(bhandle,
                    (int64_t) Npanel, (int64_t) Nr,
                    srcpanel_dev, 1, 1, bdescPanelSrc,
                    dstpanel_dev, 1, 1, bdescPanelDst,
                    gem_dwork, gem_dsz, gem_hwork, gem_hsz, comm));

        // Atil += Ktil^T Ktil (beta = 0 on the first panel initializes).
        double const* beta = (irow == 0) ? &zero_pd : &one_pd;
        CUBLASMP_CHECK(cublasMpSyrk(bhandle,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    (int64_t) Nr, (int64_t) Npanel,
                    &one_pd, dstpanel_dev, 1, 1, bdescPanelDst,
                    beta, Atil_dev, 1, 1, bdescAtil,
                    CUBLAS_COMPUTE_64F,
                    syrk_dwork, syrk_dsz, syrk_hwork, syrk_hsz));
    }

    // End of pass 2: replicate dtil, reduce cq, pq from cq.
    // dtil_loc holds this rank's contiguous rows, so one
    // rank-major AllGather is sufficient.
    NCCL_CHECK(ncclAllGather(dtil_loc_dev, dtil_full_dev, Nrows,
                ncclFloat, comm, stream));
    NCCL_CHECK(ncclAllReduce(cq_dev, cq_dev, Nr, ncclDouble, ncclSum,
                comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Check dtil > 0.
    CUDA_CHECK(cudaMemcpy(dtilp_host, dtil_loc_dev,
                Nrows * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < Nrows; ++i) {
        double const d = (double) dtilp_host[i];
        if (!(d > 0.0)) {
            printf("Pass 2: nonpositive dtil on rank %d, "
                    "global row %lu: %.6e\n",
                    rank, (size_t) rank * Nrows + i, d);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    // Dump dtil on root (float on device; dumped as double).
    if (dump_h5 && rank == 0) {
        float* dtil_hf = (float*) malloc(NM * sizeof(float));
        assert(dtil_hf != NULL);
        CUDA_CHECK(cudaMemcpy(dtil_hf, dtil_full_dev,
                    NM * sizeof(float), cudaMemcpyDeviceToHost));

        double* dtil_hd = (double*) malloc(NM * sizeof(double));
        assert(dtil_hd != NULL);
        for (size_t i = 0; i < NM; ++i) {
            dtil_hd[i] = (double) dtil_hf[i];
        }

        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, "dtil", dtil_hd, NM);
        H5Fclose(file);
        printf("Pass 2: dumped dtil\n");

        free(dtil_hd);
        free(dtil_hf);
    }

    // Copy to host and dump the reduced cq.
    double* cq_host = (double*) malloc(Nr * sizeof(double));
    assert(cq_host != NULL);
    CUDA_CHECK(cudaMemcpy(cq_host, cq_dev, Nr * sizeof(double),
                cudaMemcpyDeviceToHost));

    if (dump_h5 && rank == 0) {
        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, "cq", cq_host, Nr);
        H5Fclose(file);
        printf("Pass 2: dumped cq\n");
    }

    // pq = L^{-T}(L^{-1} cq), computed block-cyclic in vec_a, then
    // replicated on every device as float for the pass-3 qtil matvec.
    // Same inline solve as p1 (scatter + two single-column trmm).
    CUSOLVERMP_CHECK(cusolverMpMatrixScatterH2D(shandle,
                (int64_t) Nr, 1,
                vec_a, 1, 1, sdescVec,
                0, cq_host, (int64_t) Nr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // vec_b <- L^{-1} cq.
    CUBLASMP_CHECK(cublasMpTrmm(bhandle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, 1, &one_d,
                Linv_bc, 1, 1, bdescLinv,
                vec_a, 1, 1, bdescVec,
                vec_b, 1, 1, bdescVec,
                CUBLAS_COMPUTE_64F, trmm_dwork, trmm_dsz,
                trmm_hwork, trmm_hsz));

    // vec_a <- L^{-T} vec_b = pq.
    CUBLASMP_CHECK(cublasMpTrmm(bhandle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, 1, &one_d,
                Linv_bc, 1, 1, bdescLinv,
                vec_b, 1, 1, bdescVec,
                vec_a, 1, 1, bdescVec,
                CUBLAS_COMPUTE_64F, trmm_dwork, trmm_dsz,
                trmm_hwork, trmm_hsz));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // This was the last solve, so the trmm workspace is now dead.  Free
    // it here rather than at cleanup so it does not sit resident through
    // dense 1, pass 3, and the dense-2 syevd peak.  Dense 2's Z step
    // applies L^{-1}/R to a k_out-column RHS, a different shape, so it
    // queries and allocates its own workspace rather than reviving this.
    if (trmm_dwork != NULL) { cudaFree(trmm_dwork); trmm_dwork = NULL; }
    if (trmm_hwork != NULL) { free(trmm_hwork); trmm_hwork = NULL; }

    // Replicate pq as float on every device: gather to root,
    // dump the double, cast on root, broadcast the float, upload.
    // vec_a is dead after this (no further solve).
    double* pqd_host = NULL;
    if (rank == 0) {
        pqd_host = (double*) malloc(Nr * sizeof(double));
        assert(pqd_host != NULL);
    }
    CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                (int64_t) Nr, 1,
                vec_a, 1, 1, sdescVec,
                0, pqd_host, (int64_t) Nr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (dump_h5 && rank == 0) {
        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, "pq", pqd_host, Nr);
        H5Fclose(file);
        printf("Pass 2: dumped pq\n");
    }

    float* pqf_host = (float*) malloc(Nr * sizeof(float));
    assert(pqf_host != NULL);
    if (rank == 0) {
        for (size_t j = 0; j < Nr; ++j) {
            pqf_host[j] = (float) pqd_host[j];
        }
        free(pqd_host);
        pqd_host = NULL;
    }
    MPI_CHECK(MPI_Bcast(pqf_host, (int) Nr, MPI_FLOAT, 0,
                MPI_COMM_WORLD));

    // pq replicated on device as float; it lives across dense 1 until
    // pass 3 consumes it.
    float* pqf_dev;
    CUDA_CHECK(cudaMalloc(&pqf_dev, Nr * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(pqf_dev, pqf_host, Nr * sizeof(float),
                cudaMemcpyHostToDevice));
    free(pqf_host);

    if (mem_check) { mem_probe("pre-dense1", rank); }

    // --- Dense phase 1 (between passes 2 and 3). ----------------------
    // On the 2-D block-cyclic tiles, form A = L^{-1} Atil L^{-T}, factor
    // R = chol(A + sI) (trace-shifted CholeskyQR guard), and build
    // G = R L^{-1}.  L^{-1}'s strict upper triangle was zero-filled
    // above (Laset), so it is a valid general operand.
    // Three physical tiles carry the phase, the design's structural
    // floor (symm/trmm are out-of-place):
    //   buf_L = Linv_bc   holds L^{-1}
    //   buf_A = Atil_dev  holds Atil, then A, then R
    //   buf_T = tmp_dev   holds T, then the identity I, then G
    // side=RIGHT keeps the Gram in the symmetric slot and applies
    // L^{-T} from the right without materializing a transpose.
    // The spill of L^{-1} and R is deferred to the pass-3 increment,
    // whose Btil/W reuse their slots; here all three stay resident.
    // Descriptors are layout-only, so the Nr x Nr tmp tile reuses
    // bdescAtil / sdescLinv (identical Nr x Nr / nb / lld layout).
    double* tmp_dev;
    CUDA_CHECK(cudaMalloc(&tmp_dev, tile_elems * sizeof(double)));

    // Dense-phase-1 workspace.
    // One device and one host buffer sized to the max over the five
    // routines with a workspace (symm, the two trmms, geadd, potrf);
    // the calls are sequential, so they share it.  Laset needs none.
    size_t d1_dsz = 0, d1_hsz = 0;
    {
        size_t d = 0, h = 0;
        // symm(RIGHT, LOWER): T = L^{-1} Atil.
        CUBLASMP_CHECK(cublasMpSymm_bufferSize(bhandle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    (int64_t) Nr, (int64_t) Nr, &one_pd,
                    Atil_dev, 1, 1, bdescAtil,
                    Linv_bc, 1, 1, bdescLinv, &zero_pd,
                    tmp_dev, 1, 1, bdescAtil,
                    CUBLAS_COMPUTE_64F, &d, &h));
        d1_dsz = (d > d1_dsz) ? d : d1_dsz;
        d1_hsz = (h > d1_hsz) ? h : d1_hsz;
        // trmm(RIGHT, LOWER, OP_T): A = T L^{-T}.
        CUBLASMP_CHECK(cublasMpTrmm_bufferSize(bhandle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    (int64_t) Nr, (int64_t) Nr, &one_pd,
                    Linv_bc, 1, 1, bdescLinv,
                    tmp_dev, 1, 1, bdescAtil,
                    Atil_dev, 1, 1, bdescAtil,
                    CUBLAS_COMPUTE_64F, &d, &h));
        d1_dsz = (d > d1_dsz) ? d : d1_dsz;
        d1_hsz = (h > d1_hsz) ? h : d1_hsz;
        // geadd(OP_N): A <- A + s I.
        CUBLASMP_CHECK(cublasMpGeadd_bufferSize(bhandle, CUBLAS_OP_N,
                    (int64_t) Nr, (int64_t) Nr, &one_pd,
                    tmp_dev, 1, 1, bdescAtil, &one_pd,
                    Atil_dev, 1, 1, bdescAtil, &d, &h));
        d1_dsz = (d > d1_dsz) ? d : d1_dsz;
        d1_hsz = (h > d1_hsz) ? h : d1_hsz;
        // trmm(LEFT, UPPER, OP_N): G = R L^{-1}.
        CUBLASMP_CHECK(cublasMpTrmm_bufferSize(bhandle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    (int64_t) Nr, (int64_t) Nr, &one_pd,
                    Atil_dev, 1, 1, bdescAtil,
                    Linv_bc, 1, 1, bdescLinv,
                    tmp_dev, 1, 1, bdescAtil,
                    CUBLAS_COMPUTE_64F, &d, &h));
        d1_dsz = (d > d1_dsz) ? d : d1_dsz;
        d1_hsz = (h > d1_hsz) ? h : d1_hsz;
        // potrf(UPPER): R = chol(A + sI), in place.
        size_t pd = 0, ph = 0;
        CUSOLVERMP_CHECK(cusolverMpPotrf_bufferSize(shandle,
                    CUBLAS_FILL_MODE_UPPER, (int64_t) Nr,
                    Atil_dev, 1, 1, sdescLinv, CUDA_R_64F, &pd, &ph));
        d1_dsz = (pd > d1_dsz) ? pd : d1_dsz;
        d1_hsz = (ph > d1_hsz) ? ph : d1_hsz;
    }
    void* d1_dwork = NULL;
    if (d1_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&d1_dwork, d1_dsz));
    }
    void* d1_hwork = (d1_hsz > 0) ? malloc(d1_hsz) : NULL;

    // Validation dump of the pass-1/2 and dense-phase-1 intermediates
    // for the full-array oracle (scripts/evd_oracle.py --gpu-dump).
    // Vectors go out length Nr (NM for dtil); r x r matrices go out flat
    // in native column-major order, read back with order='F'.  Atil is
    // lower-only (pXsyrk fills LOWER); A is full symmetric; R is the
    // upper factor (its strict lower is zeroed below, since potrf leaves
    // stale entries there).  The gathers are O(r^2) root-bridge copies,
    // paid once here; Atil is gathered before symm destroys it.
    double* Atil_host = NULL;
    double* A_host = NULL;
    double* R_host = NULL;
    double* G_host = NULL;
    if (dump_h5 && rank == 0) {
        Atil_host = (double*) malloc(Nr * Nr * sizeof(double));
        A_host = (double*) malloc(Nr * Nr * sizeof(double));
        R_host = (double*) malloc(Nr * Nr * sizeof(double));
        G_host = (double*) malloc(Nr * Nr * sizeof(double));
        assert(Atil_host != NULL && A_host != NULL
                && R_host != NULL && G_host != NULL);
    }

    // Gather Atil to root.
    if (dump_h5) {
        CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                    (int64_t) Nr, (int64_t) Nr,
                    Atil_dev, 1, 1, sdescLinv,
                    0, Atil_host, (int64_t) Nr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // T = L^{-1} Atil.
    // symm(side=RIGHT): C = alpha B A_sym, so A_sym = Atil (symmetric,
    // lower) and B = L^{-1}; only Atil's valid lower triangle is read.
    CUBLASMP_CHECK(cublasMpSymm(bhandle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                (int64_t) Nr, (int64_t) Nr, &one_pd,
                Atil_dev, 1, 1, bdescAtil,
                Linv_bc, 1, 1, bdescLinv, &zero_pd,
                tmp_dev, 1, 1, bdescAtil, CUBLAS_COMPUTE_64F,
                d1_dwork, d1_dsz, d1_hwork, d1_hsz));

    // A = T L^{-T}.
    // trmm(side=RIGHT, trans=OP_T): C = alpha B op(A_tri), so A_tri =
    // L^{-1} (lower) and OP_T applies L^{-T}; result overwrites Atil.
    // A is a full general matrix (mathematically symmetric).
    CUBLASMP_CHECK(cublasMpTrmm(bhandle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, (int64_t) Nr, &one_pd,
                Linv_bc, 1, 1, bdescLinv,
                tmp_dev, 1, 1, bdescAtil,
                Atil_dev, 1, 1, bdescAtil, CUBLAS_COMPUTE_64F,
                d1_dwork, d1_dsz, d1_hwork, d1_hsz));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Gather the unshifted A.
    if (dump_h5) {
        CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                    (int64_t) Nr, (int64_t) Nr,
                    Atil_dev, 1, 1, sdescLinv,
                    0, A_host, (int64_t) Nr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Trace-based shift guard for chol(A) (CholeskyQR squares cond(A)).
    // Build an identity in the freed T slot and reuse it for both the
    // trace and the add: tr(A) = <A, I>_F is a plain local Ddot of the A
    // tile against the identity tile (identical block-cyclic layout, so
    // local element k is the same global (i,j) in both, and only the
    // diagonal survives), reduced across ranks.  No block-cyclic index
    // arithmetic.
    {
        double const zero_d = 0.0, one_id = 1.0;
        CUSOLVERMP_CHECK(cusolverMpLaset(shandle, CUBLAS_FILL_MODE_FULL,
                    (int64_t) Nr, (int64_t) Nr, &zero_d, &one_id,
                    tmp_dev, 1, 1, sdescLinv, dinfo_dev));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int linfo = 0;
        CUDA_CHECK(cudaMemcpy(&linfo, dinfo_dev, sizeof(int),
                    cudaMemcpyDeviceToHost));
        assert(linfo == 0);

        // Local diagonal partial via the Frobenius dot A : I, then a
        // SUM AllReduce over the grid.
        double tr_loc = 0.0;
        size_t const n_loc =
                (loc_m > 0 && loc_n > 0) ? loc_m * loc_n : 0;
        if (n_loc > 0) {
            // The 64-bit dot: a single tile's element count can exceed
            // INT_MAX at production r, and n_loc is already size_t.
            CUBLAS_CHECK(cublasDdot_64(handle, (int64_t) n_loc,
                        Atil_dev, 1, tmp_dev, 1, &tr_loc));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        double trA = 0.0;
        MPI_CHECK(MPI_Allreduce(&tr_loc, &trA, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD));

        // s = c u tr(A): the shifted-CholeskyQR shift (Fukaya et al.).
        // u = DBL_EPSILON, c = 1; tr(A) >= ||A||_2 already dominates the
        // O(u ||A||_2) indefiniteness, so this guarantees potrf with
        // margin while perturbing R only at the ~u cond(A) level.
        double constexpr c_shift = 1.0;
        double const s_shift = c_shift * DBL_EPSILON * trA;
        if (rank == 0) {
            printf("Dense 1: tr(A) %.6e  shift s %.6e\n", trA, s_shift);
        }

        // A <- A + s I.  geadd accumulates into C in place; the identity
        // operand is the distinct tmp tile, so no fourth tile is needed.
        CUBLASMP_CHECK(cublasMpGeadd(bhandle, CUBLAS_OP_N,
                    (int64_t) Nr, (int64_t) Nr, &s_shift,
                    tmp_dev, 1, 1, bdescAtil, &one_pd,
                    Atil_dev, 1, 1, bdescAtil,
                    d1_dwork, d1_dsz, d1_hwork, d1_hsz));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // R = chol(A + sI), upper (A_shift = R^T R), in place: A -> R.
    // potrf(UPPER) leaves stale A entries in the strict lower triangle,
    // which is fine: every later use of R is as a triangular operand
    // that reads the upper triangle only.
    CUSOLVERMP_CHECK(cusolverMpPotrf(shandle, CUBLAS_FILL_MODE_UPPER,
                (int64_t) Nr, Atil_dev, 1, 1, sdescLinv,
                CUDA_R_64F, d1_dwork, d1_dsz, d1_hwork, d1_hsz,
                dinfo_dev));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        int pinfo = 0;
        CUDA_CHECK(cudaMemcpy(&pinfo, dinfo_dev, sizeof(int),
                    cudaMemcpyDeviceToHost));
        if (pinfo != 0) {
            printf("Dense 1: potrf failed on rank %d, info %d "
                    "(leading minor not PD; shift too small)\n",
                    rank, pinfo);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    // Gather R and zero its strict lower host-side.
    if (dump_h5) {
        CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                    (int64_t) Nr, (int64_t) Nr,
                    Atil_dev, 1, 1, sdescLinv,
                    0, R_host, (int64_t) Nr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (rank == 0) {
            for (size_t j = 0; j < Nr; ++j) {
                for (size_t i = j + 1; i < Nr; ++i) {
                    R_host[i + j * Nr] = 0.0;
                }
            }
        }
    }

    // G = R L^{-1}.
    // trmm(side=LEFT, uplo=UPPER, trans=OP_N): C = alpha op(A_tri) B,
    // A_tri = R (upper), B = L^{-1}; result into the freed T slot.
    CUBLASMP_CHECK(cublasMpTrmm(bhandle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, (int64_t) Nr, &one_pd,
                Atil_dev, 1, 1, bdescAtil,
                Linv_bc, 1, 1, bdescLinv,
                tmp_dev, 1, 1, bdescAtil, CUBLAS_COMPUTE_64F,
                d1_dwork, d1_dsz, d1_hwork, d1_hsz));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Gather G (general; upper R times lower L^{-1} is full).
    // The oracle has no G check yet -- G is validated downstream through
    // W = G Btil G^T -- but it is dumped for the record.
    if (dump_h5) {
        CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                    (int64_t) Nr, (int64_t) Nr,
                    tmp_dev, 1, 1, sdescLinv,
                    0, G_host, (int64_t) Nr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Write the dump on root.
    if (dump_h5 && rank == 0) {
        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, "Atil", Atil_host, Nr * Nr);
        h5add_double(file, "A", A_host, Nr * Nr);
        h5add_double(file, "R", R_host, Nr * Nr);
        h5add_double(file, "G", G_host, Nr * Nr);
        H5Fclose(file);
        printf("Dense 1: dumped Atil, A, R, G\n");
        free(G_host);
        free(R_host);
        free(A_host);
        free(Atil_host);
    }

    // Dense phase 1 is done; free symm/trmm/geadd/potrf workspace.
    if (d1_dwork != NULL) { cudaFree(d1_dwork); d1_dwork = NULL; }
    if (d1_hwork != NULL) { free(d1_hwork); d1_hwork = NULL; }

    // --- Spill L^{-1} and R to host (end of dense 1). -----------
    // Move each rank's contiguous block-cyclic tile of L^{-1} and R
    // to its own host RAM, so pass 3 / dense 2 can reuse the two tile
    // slots for Btil / T / W.
    // This is a per-rank D2H copy; no collectives or root gather.
    // Pageable staging: the transfer is one-time O(r^2/P), off the
    // O(r^3) critical path.
    // The device allocations are not freed; they are repurposed later.
    // L^{-1} keeps its zeroed strict-upper and R its stale strict-lower
    // in the spilled bytes.
    // The stream was synchronized by the G gather above, so the tiles
    // are ready for the blocking copies.
    size_t const tile_bytes = tile_elems * sizeof(double);
    double* Linv_stage = (double*) malloc(tile_bytes);
    double* R_stage = (double*) malloc(tile_bytes);
    assert(Linv_stage != NULL && R_stage != NULL);
    CUDA_CHECK(cudaMemcpy(Linv_stage, Linv_bc, tile_bytes,
                cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(R_stage, Atil_dev, tile_bytes,
                cudaMemcpyDeviceToHost));

    if (mem_check) { mem_probe("pre-pass3", rank); }

    // --- Pass 3 -- qtil, Btil. ---------------------------------------
    // One more streaming sweep, the near-twin of pass 2: for each panel
    // compute qtil(R) = K(R,S) pq (local), assert min(qtil) > 0, pre-scale
    // Khat = diag(qtil^{-1/2}) K(R,S), redistribute to 2-D block cyclic,
    // and accumulate Btil += Khat^T Khat via pXsyrk(beta=1).  Btil lands
    // in the Atil_dev tile freed by the spill; the panel path
    // (srcpanel/dstpanel, Gemr2D, syrk and their workspaces, descriptors)
    // is reused verbatim from pass 2.
    // qtil is then AllGathered to replicated.
    double* Btil_dev = Atil_dev; // reused tile; bdescAtil / sdescLinv fit

    // qtil this rank's contiguous rows, then the replicated qtil.
    // dtil_full is untouched (pass 4 still needs it),
    // so qtil gets its own buffers.
    float* qtil_loc_dev;
    CUDA_CHECK(cudaMalloc(&qtil_loc_dev, Nrows * sizeof(float)));
    float* qtil_full_dev;
    CUDA_CHECK(cudaMalloc(&qtil_full_dev, NM * sizeof(float)));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    for (size_t irow = 0; irow < Nrows; irow += rblk)
    {
        size_t const row_off = (size_t) rank * Nrows + irow;

        // Kblk = K(R,S): row major, ld Nr, no tail columns.
        int const grid_kblk = (int) ((rblk * Nr + tpb - 1) / tpb);
        compute_kblock<float><<<grid_kblk, tpb, 0, stream>>>
                (Kblk_dev, Nr,
                 NULL, row_off, rblk,
                 Spiv_dev, Nr,
                 NULL, 0,
                 u_dev, bw, N, Nq);
        CUDA_CHECK(cudaGetLastError());

        // qtil(R) = K(R,S) pq (float).
        // Positivity is checked after the streaming loop.
        // The row-major panel reinterpreted column major is the
        // Nr x rblk matrix K(R,S)^T, so OP_T applies K(R,S) to pq;
        // the result lands in this panel's slice of qtil_loc.
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                    (int) Nr, (int) rblk,
                    &one_f, Kblk_dev, (int) Nr,
                    pqf_dev, 1, &zero_f,
                    qtil_loc_dev + irow, 1));

        // qinv2 = qtil^{-1/2} (float), into the reused dinv buffer.
        int const grid_rc = (int) ((rblk + tpb - 1) / tpb);
        rsqrt_float<<<grid_rc, tpb, 0, stream>>>
                (dinv_dev, qtil_loc_dev + irow, rblk);
        CUDA_CHECK(cudaGetLastError());

        // Khat = diag(qtil^{-1/2}) K(R,S)
        // scaled in float then cast into the double col-major source
        // panel (lld rblk).
        // scale_cast_panel multiplies K row i by its per-row factor.
        // qtil^{-1/2} in place of dtil^{-1} gives Khat.
        int const grid_sc = (int) ((rblk * Nr + tpb - 1) / tpb);
        scale_cast_panel<<<grid_sc, tpb, 0, stream>>>
                (srcpanel_dev, Kblk_dev, dinv_dev, rblk, Nr);
        CUDA_CHECK(cudaGetLastError());

        // Redistribute the row-complete panel into 2D block cyclic.
        CUBLASMP_CHECK(cublasMpGemr2D(bhandle,
                    (int64_t) Npanel, (int64_t) Nr,
                    srcpanel_dev, 1, 1, bdescPanelSrc,
                    dstpanel_dev, 1, 1, bdescPanelDst,
                    gem_dwork, gem_dsz, gem_hwork, gem_hsz, comm));

        // Btil += Khat^T Khat (beta = 0 on the first panel initializes,
        // overwriting the restored R that shared this tile).
        double const* beta = (irow == 0) ? &zero_pd : &one_pd;
        CUBLASMP_CHECK(cublasMpSyrk(bhandle,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    (int64_t) Nr, (int64_t) Npanel,
                    &one_pd, dstpanel_dev, 1, 1, bdescPanelDst,
                    beta, Btil_dev, 1, 1, bdescAtil,
                    CUBLAS_COMPUTE_64F,
                    syrk_dwork, syrk_dsz, syrk_hwork, syrk_hsz));
    }

    // Replicate qtil.
    NCCL_CHECK(ncclAllGather(qtil_loc_dev, qtil_full_dev, Nrows,
                ncclFloat, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Check qtil > 0.
    CUDA_CHECK(cudaMemcpy(dtilp_host, qtil_loc_dev,
                Nrows * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < Nrows; ++i) {
        double const q = (double) dtilp_host[i];
        if (!(q > 0.0)) {
            printf("Pass 3: nonpositive qtil on rank %d, "
                    "global row %lu: %.6e\n",
                    rank, (size_t) rank * Nrows + i, q);
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    // Append qtil and Btil to the oracle dump.
    // Btil is gathered to root like Atil.
    // qtil is D2H'd and cast to double.
    if (dump_h5) {
        double* Btil_host = NULL;
        if (rank == 0) {
            Btil_host = (double*) malloc(Nr * Nr * sizeof(double));
            assert(Btil_host != NULL);
        }
        CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                    (int64_t) Nr, (int64_t) Nr,
                    Btil_dev, 1, 1, sdescLinv,
                    0, Btil_host, (int64_t) Nr));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (rank == 0) {
            // qtil is float on device; dump it as double.
            float* qtil_hf = (float*) malloc(NM * sizeof(float));
            assert(qtil_hf != NULL);
            CUDA_CHECK(cudaMemcpy(qtil_hf, qtil_full_dev,
                        NM * sizeof(float), cudaMemcpyDeviceToHost));

            double* qtil_hd = (double*) malloc(NM * sizeof(double));
            assert(qtil_hd != NULL);
            for (size_t i = 0; i < NM; ++i) {
                qtil_hd[i] = (double) qtil_hf[i];
            }

            hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
            assert(file != H5I_INVALID_HID);
            h5add_double(file, "qtil", qtil_hd, NM);
            h5add_double(file, "Btil", Btil_host, Nr * Nr);
            H5Fclose(file);
            printf("Pass 3: dumped qtil, Btil\n");

            free(qtil_hd);
            free(qtil_hf);
            free(Btil_host);
        }
    }

    // Pass 3 was the last user of the streaming-sweep buffers: the Gram
    // machinery (the panel Gemr2D + pXsyrk) and the Kblk kernel panel.
    // Retire them here, before dense 2, so none sits resident through the
    // syevd peak.  The Gram machinery is not needed downstream (dense 2
    // and the deferred Z / pass 4 do no redistribute or syrk).
    cudaFree(dstpanel_dev); dstpanel_dev = NULL;
    cudaFree(srcpanel_dev); srcpanel_dev = NULL;
    if (gem_dwork != NULL) {cudaFree(gem_dwork); gem_dwork = NULL;}
    if (gem_hwork != NULL) {free(gem_hwork); gem_hwork = NULL;}
    if (syrk_dwork != NULL) {cudaFree(syrk_dwork); syrk_dwork = NULL;}
    if (syrk_hwork != NULL) {free(syrk_hwork); syrk_hwork = NULL;}

    // Kblk (the rblk x Nr K(R,S) streaming panel) is last used by pass
    // 3's sweep above; free it so it is not resident through the dense-2
    // syevd peak.  Pass 4 reallocates it to stream
    // U(R) = diag(dtil^{-1}) K(R,S) Z out to HDF5.
    cudaFree(Kblk_dev); Kblk_dev = NULL;

    // --- Dense phase 2 (part): W = G Btil G^T, then syevd(W). ---------
    // Form T = G Btil (symm), W = T G^T (gemm), then V, Lambda =
    // syevd(W).  This increment stops once V and Lambda exist and are
    // dumped; extracting the trailing k_out columns of V and the Z step
    // are deferred.
    // Tile schedule (no new r x r allocations for T/W/V):
    //   Linv_bc : L^{-1} -> T -> V   (L^{-1} is dead here; it lives in
    //                                 Linv_stage for the deferred Z)
    //   Atil_dev: Btil   -> W        (read-only through syevd)
    //   tmp_dev : G      -> freed before syevd
    // Btil is consumed by the symm, so W overwrites its slot; G is read
    // by both products, then freed.  Freeing the tmp_dev tile and the
    // product workspace before the syevd workspace holds the syevd peak
    // at the design's 2 live tiles + 4 syevd-workspace tiles.

    // Product workspace, sized to the max over symm and gemm (both
    // Nr x Nr and sequential, so they share one buffer).
    size_t d2_dsz = 0, d2_hsz = 0;
    {
        size_t d = 0, h = 0;
        // symm(RIGHT, LOWER): T = G Btil.
        CUBLASMP_CHECK(cublasMpSymm_bufferSize(bhandle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    (int64_t) Nr, (int64_t) Nr, &one_pd,
                    Atil_dev, 1, 1, bdescAtil,
                    tmp_dev, 1, 1, bdescAtil, &zero_pd,
                    Linv_bc, 1, 1, bdescLinv,
                    CUBLAS_COMPUTE_64F, &d, &h));
        d2_dsz = (d > d2_dsz) ? d : d2_dsz;
        d2_hsz = (h > d2_hsz) ? h : d2_hsz;
        // gemm(OP_N, OP_T): W = T G^T.
        CUBLASMP_CHECK(cublasMpGemm_bufferSize(bhandle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    (int64_t) Nr, (int64_t) Nr, (int64_t) Nr, &one_pd,
                    Linv_bc, 1, 1, bdescLinv,
                    tmp_dev, 1, 1, bdescAtil, &zero_pd,
                    Atil_dev, 1, 1, bdescAtil,
                    CUBLAS_COMPUTE_64F, &d, &h));
        d2_dsz = (d > d2_dsz) ? d : d2_dsz;
        d2_hsz = (h > d2_hsz) ? h : d2_hsz;
    }
    void* d2_dwork = NULL;
    if (d2_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&d2_dwork, d2_dsz));
    }
    void* d2_hwork = (d2_hsz > 0) ? malloc(d2_hsz) : NULL;

    // T = G Btil.
    // symm(side=RIGHT): C = alpha B A_sym, A_sym = Btil (symmetric,
    // lower; pass 3 filled LOWER), B = G; only Btil's valid lower
    // triangle is read.  Result T overwrites the dead L^{-1}.
    CUBLASMP_CHECK(cublasMpSymm(bhandle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                (int64_t) Nr, (int64_t) Nr, &one_pd,
                Atil_dev, 1, 1, bdescAtil,
                tmp_dev, 1, 1, bdescAtil, &zero_pd,
                Linv_bc, 1, 1, bdescLinv, CUBLAS_COMPUTE_64F,
                d2_dwork, d2_dsz, d2_hwork, d2_hsz));

    // W = T G^T.
    // gemm(OP_N, OP_T): C = alpha op(A) op(B), A = T, B = G; result W
    // overwrites Btil (consumed by the symm above).  W comes out full
    // and mathematically symmetric.
    CUBLASMP_CHECK(cublasMpGemm(bhandle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                (int64_t) Nr, (int64_t) Nr, (int64_t) Nr, &one_pd,
                Linv_bc, 1, 1, bdescLinv,
                tmp_dev, 1, 1, bdescAtil, &zero_pd,
                Atil_dev, 1, 1, bdescAtil, CUBLAS_COMPUTE_64F,
                d2_dwork, d2_dsz, d2_hwork, d2_hsz));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The product workspace is dead; free it before the syevd workspace
    // so the two never coexist.
    if (d2_dwork != NULL) { cudaFree(d2_dwork); d2_dwork = NULL; }
    if (d2_hwork != NULL) { free(d2_hwork); d2_hwork = NULL; }

    // Free one r x r tile (G, no longer read) to hold the syevd peak at
    // the design's 2 live + 4 workspace tiles.
    cudaFree(tmp_dev);
    tmp_dev = NULL;

    // W is in Atil_dev; V lands in the freed T slot Linv_bc.  Both use
    // sdescLinv (identical layout, so the MB_A == MB_Q / IA == IQ
    // constraint holds by construction).
    double* W_dev = Atil_dev;
    double* V_dev = Linv_bc;

    // W_host holds the pre-syevd copy of W for the dump.
    double* W_host = NULL;

    // Capture W for the oracle dump before syevd:
    // cusolverMpSyevd overwrites its input matrix A.
    if (dump_h5) {
        if (rank == 0) {
            W_host = (double*) malloc(Nr * Nr * sizeof(double));
            assert(W_host != NULL);
        }
        CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                    (int64_t) Nr, (int64_t) Nr,
                    W_dev, 1, 1, sdescLinv,
                    0, W_host, (int64_t) Nr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Eigenvalues: a replicated length-Nr device vector (small, not an
    // r x r tile), ascending after syevd.
    double* lam_dev;
    CUDA_CHECK(cudaMalloc(&lam_dev, Nr * sizeof(double)));

    // syevd workspace (~4 r x r tiles on device; a few MB on host).
    char jobz[] = "V"; // eigenvalues and eigenvectors
    size_t sy_dsz = 0, sy_hsz = 0;
    CUSOLVERMP_CHECK(cusolverMpSyevd_bufferSize(shandle, jobz,
                CUBLAS_FILL_MODE_LOWER, (int64_t) Nr,
                W_dev, 1, 1, sdescLinv,
                lam_dev, V_dev, 1, 1, sdescLinv,
                CUDA_R_64F, &sy_dsz, &sy_hsz));
    void* sy_dwork = NULL;
    if (sy_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&sy_dwork, sy_dsz));
    }
    void* sy_hwork = (sy_hsz > 0) ? malloc(sy_hsz) : NULL;

    // Design-predicted peak: 2 live tiles (W, V) + the syevd device
    // workspace + baseline.  Compare against nvtop's max to expose any
    // syevd-internal allocation (invisible to a host-side probe during
    // the blocking call).
    if (mem_check) { mem_probe("pre-syevd", rank); }

    // V, Lambda = syevd(W).  W (in W_dev) is consumed and overwritten
    // here -- it was captured above for the dump.  V (eigenvectors) is a
    // separate output; eigenvalues land ascending, replicated in lam_dev.
    CUSOLVERMP_CHECK(cusolverMpSyevd(shandle, jobz,
                CUBLAS_FILL_MODE_LOWER, (int64_t) Nr,
                W_dev, 1, 1, sdescLinv,
                lam_dev, V_dev, 1, 1, sdescLinv,
                CUDA_R_64F, sy_dwork, sy_dsz, sy_hwork, sy_hsz,
                dinfo_dev));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        int sinfo = 0;
        CUDA_CHECK(cudaMemcpy(&sinfo, dinfo_dev, sizeof(int),
                    cudaMemcpyDeviceToHost));
        if (sinfo != 0) {
            printf("Dense 2: syevd failed on rank %d, info %d\n",
                    rank, sinfo);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
    }

    // Free the syevd workspace: now exactly two r x r tiles remain,
    // Atil_dev (now syevd scratch; W was captured before) and V_dev
    // (Linv_bc, the eigenvectors).
    if (sy_dwork != NULL) { cudaFree(sy_dwork); sy_dwork = NULL; }
    if (sy_hwork != NULL) { free(sy_hwork); sy_hwork = NULL; }

    if (mem_check) { mem_probe("post-syevd", rank); }

    // Copy the replicated eigenvalues to root. 
    double* lam_host = NULL;
    if (rank == 0) {
        lam_host = (double*) malloc(Nr * sizeof(double));
        assert(lam_host != NULL);
        CUDA_CHECK(cudaMemcpy(lam_host, lam_dev, Nr * sizeof(double),
                    cudaMemcpyDeviceToHost));
    }

    // Write the eigenvalues (Nr, double, ascending)
    // to the EVD output file.
    if (rank == 0) {
        hid_t file = H5Fopen(out_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, out_dset1, lam_host, Nr);
        H5Fclose(file);
        printf("Dense 2: appended lam to %s\n", out_file);
    }

    // Append W (captured before syevd), V, and Lambda to the oracle
    // dump.  W and V are full r x r (gathered flat column-major, read
    // back order='F'); Lambda is length Nr.
    if (dump_h5) {
        double* V_host = NULL;
        if (rank == 0) {
            V_host = (double*) malloc(Nr * Nr * sizeof(double));
            assert(V_host != NULL);
        }
        CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                    (int64_t) Nr, (int64_t) Nr,
                    V_dev, 1, 1, sdescLinv,
                    0, V_host, (int64_t) Nr));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (rank == 0) {
            hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
            assert(file != H5I_INVALID_HID);
            h5add_double(file, "W", W_host, Nr * Nr);
            h5add_double(file, "V", V_host, Nr * Nr);
            h5add_double(file, "lam", lam_host, Nr);
            H5Fclose(file);
            printf("Dense 2: appended W, V, lam\n");

            free(V_host);
            free(W_host);
        }
    }
    if (rank == 0) {free(lam_host);}
    cudaFree(lam_dev); lam_dev = NULL;

    // --- Dense phase 2 (Z step): Z = L^{-T} R^{-1} Vk. ---------------
    // Vk is the trailing k_out columns of V (the largest eigenvalues,
    // since syevd returns them ascending).  Its start column
    // jvk = Nr - k_out + 1 is in general *not* block-aligned, and
    // cuBLASMp's Trsm/Trmm require sub-matrix offsets jb-1 to be a
    // multiple of nb.  So Vk cannot be referenced in place as a column
    // strip of V; it is extracted first with cublasMpGemr2D, whose
    // contract permits arbitrary (unaligned) 1-based offsets, into a
    // fresh Nr x k_out tile Vk_dev whose columns start at global column 1
    // (aligned).  The Trsm/Trmm then run on Vk_dev at (1,1), all offsets
    // block-aligned.  This is the design's "W and V freed once Vk is
    // extracted"; see docs/DESIGN_evd2.md.
    //
    // trsm (uses R) and trmm (uses L^{-1}) run sequentially and never
    // co-need R and L^{-1}, so both restore into the single freed W tile
    // (Atil_dev): restore R, trsm in place on Vk_dev, then restore L^{-1}
    // over the dead R, then trmm into Z_dev.  Live sizeable tiles peak at
    // 2 r x r (V during the extraction, then the restore tile), plus the
    // two small r x k_out tiles Vk and Z -- no third r x r allocation.
    // (The syevd peak of 6 tiles already passed; its workspace is freed,
    // so these tiles sit far under it.)
    assert((size_t) Nr >= k_out);

    // 1-based global column where Vk begins inside V (may be unaligned;
    // only the Gemr2D source addressing below uses it).
    int64_t const jvk = (int64_t) Nr - (int64_t) k_out + 1;

    // Vk and Z tiles: Nr x k_out, 2-D block cyclic on the same grid / nb
    // as V, columns starting at global column 1 (aligned).  Both share
    // the Nr x k_out layout, so one descriptor pair (bdescZ / sdescZ)
    // serves both; only Z needs the cuSOLVERMp descriptor for the gather.
    // Rows carry V's blocking, so lld is unchanged.
    size_t const loc_n_z = (size_t) cublasMpNumroc((int64_t) k_out,
            (int64_t) nb, (uint32_t) mycol, 0, (uint32_t) npcol);
    size_t const z_elems = (loc_n_z > 0) ? lld * loc_n_z : 1;
    double* Vk_dev;
    CUDA_CHECK(cudaMalloc(&Vk_dev, z_elems * sizeof(double)));
    double* Z_dev;
    CUDA_CHECK(cudaMalloc(&Z_dev, z_elems * sizeof(double)));

    cusolverMpMatrixDescriptor_t sdescZ;
    CUSOLVERMP_CHECK(cusolverMpCreateMatrixDesc(&sdescZ, sgrid,
                CUDA_R_64F, (int64_t) Nr, (int64_t) k_out,
                (int64_t) nb, (int64_t) nb, 0, 0, (int64_t) lld));
    cublasMpMatrixDescriptor_t bdescZ;
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
                (int64_t) Nr, (int64_t) k_out, (int64_t) nb, (int64_t) nb,
                0, 0, (int64_t) lld, CUDA_R_64F, bgrid, &bdescZ));

    // Extract Vk = V(:, jvk : Nr) into Vk_dev(:, 1 : k_out).  Gemr2D is
    // a device-to-device redistribute over NCCL (the source strip lives
    // on the last process column, the dest on process column 0); its
    // contract allows the unaligned source offset jvk that Trsm/Trmm
    // reject.  One-shot O(Nr k_out) traffic, negligible vs the O(r^3)
    // dense phase.  V is dead once this returns.
    size_t vk_dsz = 0, vk_hsz = 0;
    CUBLASMP_CHECK(cublasMpGemr2D_bufferSize(bhandle,
                (int64_t) Nr, (int64_t) k_out,
                V_dev, 1, jvk, bdescLinv,
                Vk_dev, 1, 1, bdescZ,
                &vk_dsz, &vk_hsz, comm));
    void* vk_dwork = NULL;
    if (vk_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&vk_dwork, vk_dsz));
    }
    void* vk_hwork = (vk_hsz > 0) ? malloc(vk_hsz) : NULL;
    CUBLASMP_CHECK(cublasMpGemr2D(bhandle,
                (int64_t) Nr, (int64_t) k_out,
                V_dev, 1, jvk, bdescLinv,
                Vk_dev, 1, 1, bdescZ,
                vk_dwork, vk_dsz, vk_hwork, vk_hsz, comm));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The extract is done; free its workspace and retire V.
    if (vk_dwork != NULL) { cudaFree(vk_dwork); vk_dwork = NULL; }
    if (vk_hwork != NULL) { free(vk_hwork); vk_hwork = NULL; }
    cudaFree(V_dev);    Linv_bc = NULL; V_dev = NULL;

    // Z-step workspace, sized to the max over trsm and trmm (sequential,
    // so they share one buffer).  trsm is in-place (A = R, B = Vk); trmm
    // is out-of-place (A = L^{-1}, B = Vk, C = Z).  Both operands are now
    // at (1,1), block-aligned.
    size_t z_dsz = 0, z_hsz = 0;
    {
        size_t d = 0, h = 0;
        // trsm(LEFT, UPPER, OP_N): R X = Vk, X overwrites Vk.
        CUBLASMP_CHECK(cublasMpTrsm_bufferSize(bhandle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    (int64_t) Nr, (int64_t) k_out, &one_pd,
                    Atil_dev, 1, 1, bdescLinv,
                    Vk_dev, 1, 1, bdescZ,
                    CUBLAS_COMPUTE_64F, &d, &h));
        z_dsz = (d > z_dsz) ? d : z_dsz;
        z_hsz = (h > z_hsz) ? h : z_hsz;
        // trmm(LEFT, LOWER, OP_T): Z = L^{-T} (R^{-1} Vk).
        CUBLASMP_CHECK(cublasMpTrmm_bufferSize(bhandle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    (int64_t) Nr, (int64_t) k_out, &one_pd,
                    Atil_dev, 1, 1, bdescLinv,
                    Vk_dev, 1, 1, bdescZ,
                    Z_dev, 1, 1, bdescZ,
                    CUBLAS_COMPUTE_64F, &d, &h));
        z_dsz = (d > z_dsz) ? d : z_dsz;
        z_hsz = (h > z_hsz) ? h : z_hsz;
    }
    void* z_dwork = NULL;
    if (z_dsz > 0) {
        CUDA_CHECK(cudaMalloc(&z_dwork, z_dsz));
    }
    void* z_hwork = (z_hsz > 0) ? malloc(z_hsz) : NULL;

    // Restore R into the freed W tile.  The tile is idle here: syevd, the
    // dump, and the Gemr2D above are done and the stream was
    // synchronized.  Per-rank H2D copy of R's block-cyclic tile
    // (bit-exact, keeps R's stale strict-lower, which trsm(UPPER) never
    // reads).
    CUDA_CHECK(cudaMemcpy(Atil_dev, R_stage, tile_bytes,
                cudaMemcpyHostToDevice));
    free(R_stage);
    R_stage = NULL;

    // trsm: solve R X = Vk, X overwrites Vk_dev in place.
    CUBLASMP_CHECK(cublasMpTrsm(bhandle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, (int64_t) k_out, &one_pd,
                Atil_dev, 1, 1, bdescLinv,
                Vk_dev, 1, 1, bdescZ,
                CUBLAS_COMPUTE_64F, z_dwork, z_dsz, z_hwork, z_hsz));

    // Wait for trsm to finish reading R before overwriting the tile with
    // L^{-1} (a WAR hazard on Atil_dev across the restore memcpy).
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Restore L^{-1} into the same tile, over the now-dead R.  Its zeroed
    // strict-upper is irrelevant: trmm(LOWER, OP_T) reads only the lower
    // triangle.
    CUDA_CHECK(cudaMemcpy(Atil_dev, Linv_stage, tile_bytes,
                cudaMemcpyHostToDevice));
    free(Linv_stage);
    Linv_stage = NULL;

    // trmm: Z = L^{-T} (R^{-1} Vk), out-of-place into the Z tile.
    CUBLASMP_CHECK(cublasMpTrmm(bhandle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                (int64_t) Nr, (int64_t) k_out, &one_pd,
                Atil_dev, 1, 1, bdescLinv,
                Vk_dev, 1, 1, bdescZ,
                Z_dev, 1, 1, bdescZ,
                CUBLAS_COMPUTE_64F, z_dwork, z_dsz, z_hwork, z_hsz));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The Z-step workspace, Vk (consumed by trmm), and the restore tile
    // (dead L^{-1}) are all done; free at last use.  Only the small
    // distributed Z remains sizeable.
    if (z_dwork != NULL) { cudaFree(z_dwork); z_dwork = NULL; }
    if (z_hwork != NULL) { free(z_hwork); z_hwork = NULL; }
    cudaFree(Vk_dev);
    cudaFree(Atil_dev); Atil_dev = NULL;

    if (mem_check) { mem_probe("post-Z", rank); }

    // Gather Z to root: the outbound half of the root bridge, the same
    // GatherD2H that returns p1 / pq.  Z is r x k_out, gathered flat
    // column-major (lld = Nr), so the whole matrix lands as one
    // contiguous block on root -- no block-cyclic unshuffle.  The double
    // Z_host is root-only: it serves the oracle dump and the float cast
    // below, nothing on the other ranks.  All ranks call the collective
    // gather; only root passes a destination.
    double* Z_host = NULL;
    if (rank == 0) {
        Z_host = (double*) malloc(Nr * k_out * sizeof(double));
        assert(Z_host != NULL);
    }
    CUSOLVERMP_CHECK(cusolverMpMatrixGatherD2H(shandle,
                (int64_t) Nr, (int64_t) k_out,
                Z_dev, 1, 1, sdescZ,
                0, Z_host, (int64_t) Nr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The block-cyclic Z tile is dead once gathered; free it before the
    // float replica is built so the two never co-reside.
    cudaFree(Z_dev);
    Z_dev = NULL;

    // Append the double Z to the oracle dump (read back order='F').  Like
    // V, Z is sign-ambiguous column-wise -- validate by subspace angles,
    // or defer to the gauge-robust U (pass 4).
    if (dump_h5 && rank == 0) {
        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_double(file, "Z", Z_host, Nr * k_out);
        H5Fclose(file);
        printf("Dense 2: appended Z (r x %lu)\n", k_out);
    }

    // Replicate Z on every device as float, to match pass 4's float
    // K(R,S) streaming matvec U(R) = diag(dtil^{-1}) K(R,S) Z.
    // This mirrors the p1 / pq bridge (GatherD2H -> Bcast -> upload) with
    // one deliberate change: root casts to float *before* the broadcast,
    // rather than broadcasting the double and casting per rank.  p1 / pq
    // broadcast the double because their double host copy is itself
    // replicated and dumped; Z's double is root-only (the dump above), so
    // casting once on root keeps the double off the wire and off the
    // other ranks, and only half the bytes cross.
    size_t const z_count = (size_t) Nr * k_out;
    assert(z_count <= (size_t) INT_MAX);   // MPI_Bcast count is int.
    float* Zf_host = (float*) malloc(z_count * sizeof(float));
    assert(Zf_host != NULL);
    if (rank == 0) {
        for (size_t j = 0; j < z_count; ++j) {
            Zf_host[j] = (float) Z_host[j];
        }
        free(Z_host);
        Z_host = NULL;
    }
    MPI_CHECK(MPI_Bcast(Zf_host, (int) z_count, MPI_FLOAT, 0,
                MPI_COMM_WORLD));

    // Upload the replicated float Z to this rank's device.  Zf_rep_dev is
    // the r x k_out float Z that pass 4 consumes; it survives to cleanup.
    float* Zf_rep_dev;
    CUDA_CHECK(cudaMalloc(&Zf_rep_dev, z_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(Zf_rep_dev, Zf_host, z_count * sizeof(float),
                cudaMemcpyHostToDevice));
    free(Zf_host);

    // --- Pass 4 -- emit U. -------------------------------------------
    // One final streaming sweep, collective-free like pass 1: for each
    // panel regenerate K(R,S) and form
    //   U(R) = diag(dtil(R)^{-1}) K(R,S) Z   (rblk x k_out, float),
    // then spill it to this rank's host.  Z is replicated (Zf_rep_dev),
    // dtil is this rank's own rows (dtil_loc_dev), so the whole pass is
    // local.  The row scaling is applied after the matmul: it is the
    // same diag(dtil^{-1}) but on the small k_out-wide product rather
    // than the wide K panel, and diag(d^{-1})(K Z) = (diag(d^{-1}) K) Z.
    if (mem_check) { mem_probe("pre-pass4", rank); }

    // Reallocate the streaming panel Kblk (freed at the end of pass 3):
    // rblk x Nr float, row major, leading dimension Nr, as in passes 1-3.
    CUDA_CHECK(cudaMalloc(&Kblk_dev, rblk * Nr * sizeof(float)));

    // The U panel (K(R,S) Z)^T: column-major k_out x rblk, which is the
    // row-major rblk x k_out output panel (logical row i contiguous).
    float* Upanel_dev;
    CUDA_CHECK(cudaMalloc(&Upanel_dev, k_out * rblk * sizeof(float)));

    // This rank's contiguous slab of U, assembled on host in row-major
    // order (row i = local row i, k_out contiguous), then gathered to
    // root in rank order (= global row order) for the flat dump.
    float* U_loc_host = (float*) malloc(Nrows * k_out * sizeof(float));
    assert(U_loc_host != NULL);

    for (size_t irow = 0; irow < Nrows; irow += rblk)
    {
        size_t const nrblk =
                (Nrows - irow < rblk) ? Nrows - irow : rblk;
        size_t const row_off = (size_t) rank * Nrows + irow;

        // Kblk = K(R,S): row major, ld Nr, no tail columns.
        int const grid_kblk =
                (int) ((nrblk * Nr + tpb - 1) / tpb);
        compute_kblock<float><<<grid_kblk, tpb, 0, stream>>>
                (Kblk_dev, Nr,
                 NULL, row_off, nrblk,
                 Spiv_dev, Nr,
                 NULL, 0,
                 u_dev, bw, N, Nq);
        CUDA_CHECK(cudaGetLastError());

        // Upanel = (K(R,S) Z)^T, column major k_out x nrblk.
        // Kblk row major (nrblk x Nr) reinterpreted column major is
        // K(R,S)^T (Nr x nrblk); Zf_rep is column major Nr x k_out.
        // gemm(OP_T, OP_N): C = Z^T K^T = (K Z)^T, so C column-major
        // k_out x nrblk is the row-major nrblk x k_out panel K(R,S) Z.
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int) k_out, (int) nrblk, (int) Nr,
                    &one_f, Zf_rep_dev, (int) Nr,
                    Kblk_dev, (int) Nr,
                    &zero_f, Upanel_dev, (int) k_out));

        // dinv = 1 / dtil (float), this panel's rows.  dtil_loc holds
        // this rank's own dtil from pass 2; its rows align with the
        // panel's output rows.
        int const grid_rc = (int) ((nrblk + tpb - 1) / tpb);
        recip_float<<<grid_rc, tpb, 0, stream>>>
                (dinv_dev, dtil_loc_dev + irow, nrblk);
        CUDA_CHECK(cudaGetLastError());

        // Scale each output row i by dinv[i], in place on the panel.
        int const grid_sr = (int) ((nrblk * k_out + tpb - 1) / tpb);
        scale_rows_rm<<<grid_sr, tpb, 0, stream>>>
                (Upanel_dev, dinv_dev, nrblk, k_out);
        CUDA_CHECK(cudaGetLastError());

        // Spill the row-major panel to host.
        CUDA_CHECK(cudaMemcpy(U_loc_host + irow * k_out, Upanel_dev,
                    nrblk * k_out * sizeof(float),
                    cudaMemcpyDeviceToHost));
    }

    // Kblk, the U panel, and the replicated Z are done; free at last use
    // before the gather.
    cudaFree(Upanel_dev);
    cudaFree(Kblk_dev); Kblk_dev = NULL;
    cudaFree(Zf_rep_dev); Zf_rep_dev = NULL;

    // Assemble the full U on root, then dump it.  Every rank owns the
    // contiguous global rows [rank*Nrows, (rank+1)*Nrows), all equal in
    // size, so an MPI_Gather concatenates the slabs in rank order = global
    // row order.  This is the outbound root bridge once more (the pass-4
    // twin of the p1 / pq / Z gathers), but over host buffers: each rank
    // spilled its slab above, and root assembles the whole NM x k_out.
    size_t const u_loc_count = Nrows * k_out;
    assert(u_loc_count <= (size_t) INT_MAX); // MPI_Gather count is int.
    float* U_full_host = NULL;
    if (rank == 0) {
        U_full_host = (float*) malloc(NM * k_out * sizeof(float));
        assert(U_full_host != NULL);
    }
    MPI_CHECK(MPI_Gather(U_loc_host, (int) u_loc_count, MPI_FLOAT,
                U_full_host, (int) u_loc_count, MPI_FLOAT, 0,
                MPI_COMM_WORLD));
    free(U_loc_host);

    // Write U in the EVD output file (flat, row major, float).
    if (rank == 0) {
        hid_t file = H5Fopen(out_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_float(file, out_dset2, U_full_host, NM * k_out);
        H5Fclose(file);
        printf("Pass 4: appended U (NM x %lu) to %s\n", k_out, out_file);
    }

    // Write U in the dump file (flat, row major, float).
    if (dump_h5 && rank == 0) {
        hid_t file = H5Fopen(dump_file, H5F_ACC_RDWR, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID);
        h5add_float(file, "U", U_full_host, NM * k_out);
        H5Fclose(file);
        printf("Pass 4: dumped U (NM x %lu)\n", k_out);
    }
    if (rank == 0) {free(U_full_host);}

    if (mem_check) { mem_probe("post-pass4", rank); }

    // Cleanup of this rank's resources.
    // Buffers are freed where they fall dead, not here: the trmm
    // workspace at the end of pass 2, the Gram machinery (srcpanel /
    // dstpanel, gem / syrk workspaces) at the end of pass 3, the dense-1
    // workspace at the end of dense phase 1, and the dense-2 / syevd
    // workspaces within dense phase 2.  Only the descriptors and the
    // still-live buffers are released here.  tmp_dev was freed inside
    // dense phase 2 (set NULL there), so its cudaFree below is a
    // NULL-guarded no-op.
    if (tmp_dev != NULL) { cudaFree(tmp_dev); }
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(bdescAtil));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(bdescPanelDst));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(bdescPanelSrc));
    CUBLASMP_CHECK(cublasMpGridDestroy(bgrid_src));

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(bdescZ));
    CUSOLVERMP_CHECK(cusolverMpDestroyMatrixDesc(sdescZ));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(bdescVec));
    CUSOLVERMP_CHECK(cusolverMpDestroyMatrixDesc(sdescVec));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(bdescLinv));
    CUSOLVERMP_CHECK(cusolverMpDestroyMatrixDesc(sdescLinv));
    CUBLASMP_CHECK(cublasMpGridDestroy(bgrid));
    CUSOLVERMP_CHECK(cusolverMpDestroyGrid(sgrid));
    CUBLASMP_CHECK(cublasMpDestroy(bhandle));
    CUSOLVERMP_CHECK(cusolverMpDestroy(shandle));

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    // lam_dev (after its D2H), Atil_dev / Linv_bc (the W and V tiles,
    // Atil_dev holding L^{-1} and Linv_bc = V_dev at the end), and the
    // block-cyclic Z_dev (freed once gathered) were all freed at last use
    // in the dense-2 Z step and set NULL; the guards here are no-ops until
    // anything reallocates them.  Zf_rep_dev, the replicated r x k_out
    // float Z, was consumed and freed at the end of pass 4 (set NULL), so
    // its guard here is a no-op too.
    if (Z_dev != NULL) { cudaFree(Z_dev); }
    if (Zf_rep_dev != NULL) { cudaFree(Zf_rep_dev); }
    if (lam_dev != NULL) { cudaFree(lam_dev); }
    if (Atil_dev != NULL) { cudaFree(Atil_dev); }
    cudaFree(qtil_full_dev);
    cudaFree(qtil_loc_dev);
    cudaFree(cq_dev);
    cudaFree(dtil_full_dev);
    cudaFree(dtil_loc_dev);
    cudaFree(dinv_dev);
    cudaFree(pqf_dev);
    cudaFree(p1f_dev);
    cudaFree(vec_b);
    cudaFree(vec_a);
    cudaFree(dinfo_dev);
    cudaFree(c1_dev);
    cudaFree(csum_dev);
    cudaFree(ones_dev);
    // Kblk_dev was reallocated in pass 4 and freed at last use there (set
    // NULL); the guard here is a no-op.
    if (Kblk_dev != NULL) { cudaFree(Kblk_dev); }
    if (Linv_bc != NULL) { cudaFree(Linv_bc); }
    cudaFree(Spiv_dev);
    cudaFree(u_dev);
    ncclCommDestroy(comm);

    free(dtilp_host);
    // The spill stage buffers were consumed and freed in the dense-2 Z
    // step (set NULL there); these guards are no-ops.
    if (R_stage != NULL) { free(R_stage); }
    if (Linv_stage != NULL) { free(Linv_stage); }
    // p1 / pq host doubles are gone -- they are now root-only temporaries
    // (p1d_host / pqd_host), freed right after their float cast.
    free(cq_host);
    free(c1_host);
    free(u_host);
    free(Linv_host);
    free(Spiv_host);

    time_t t1 = time(NULL);
    if (rank == 0) {
        printf("Total time: %.2f s\n", difftime(t1, t0));
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
