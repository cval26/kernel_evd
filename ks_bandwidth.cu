//nvcc ks_bandwidth.cu -ccbin mpicxx -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lhdf5 --use_fast_math -o cuband
//nvcc ks_bandwidth.cu -ccbin mpicxx -g -G -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudart -lhdf5 -o cuband
//run: mpirun -np 4 ./cuband

// Multi-GPU kernel bandwidth calibration (MPI).
// One MPI process (rank) per GPU.
//
// Each rank owns a contiguous range of the strictly lower triangular
// part of the distance matrix and accumulates its own partial kernel
// sums over the candidate bandwidth values.
// The ranges are derived from compile time constants, so the work
// partition needs no communication.
// The only communication is the final reduction of the Nbw partial
// sums onto rank 0, which does the post-processing.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>

#include <mpi.h>
#include <hdf5.h>

#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(val) cuda_check((val), __FILE__, __LINE__)
__host__ __device__
inline void cuda_check(cudaError_t err, char const* file, int const line)
{
    if (err != cudaSuccess) {
        printf("CUDA error: %s:%i %s: %s\n", file, line,
                cudaGetErrorName(err), cudaGetErrorString(err));
        assert(0);
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

// Fill out with the n values start, start + delta, ..., and return
// the number of values written.
size_t regspace(float* out, size_t const cap, float const start,
        float const delta, size_t const n)
{
    assert(out != NULL);
    assert(n <= cap);

    for (size_t i = 0; i < n; ++i) {
        out[i] = start + (float) i * delta;
    }
    return n;
}

// Number of entries of the strictly lower triangular part of an
// N x N matrix that lie in the columns before column k.
// The product is always even, so the division is exact.
__host__ __device__
inline size_t tri_count(size_t const k, size_t const N)
{
    return k * (2 * N - 1 - k) / 2;
}

// Invert the flattened index p of the strictly lower triangular
// part of an N x N matrix into its row and column indices.
// The column is the largest k with tri_count(k, N) <= p, obtained
// by solving that quadratic and correcting for the round-off
// error of the square root.
__host__ __device__
inline void tri_ind(size_t* row, size_t* col, size_t const p,
        size_t const N)
{
    double const a = 2.0 * (double) N - 1.0;
    double const disc = a * a - 8.0 * (double) p;

    long int k = (long int) ((a - sqrt(disc > 0.0 ? disc : 0.0)) / 2.0);

    if (k < 0) {
        k = 0;
    }
    while (k > 0 && tri_count((size_t) k, N) > p) {
        --k;
    }
    while (tri_count((size_t) k + 1, N) <= p) {
        ++k;
    }

    *col = (size_t) k;
    *row = *col + 1 + p - tri_count(*col, N);
}

__global__
void dmat(float* out, float const* udata,
        size_t const* i0_ptr,
        size_t const* Nentries_ptr,
        size_t const* NM_ptr,
        size_t const* N_ptr,
        size_t const* Nq_ptr)
{
    size_t const Nentries = *Nentries_ptr;
    size_t const NM = *NM_ptr;

    size_t const N = *N_ptr;
    size_t const Nq = *Nq_ptr;
    size_t const Ndata = N + Nq - 1;

    size_t const p0 = *i0_ptr;

    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < Nentries; i += stride)
    {
        // Compute the distance matrix row and column indices.
        size_t row = 0;
        size_t col = 0;
        tri_ind(&row, &col, p0 + i, NM);

        // Row and column indices for the state data.
        // They are different from the above because of the
        // time delay embedding.
        size_t drow = (row / N) * Ndata + row % N;
        size_t dcol = (col / N) * Ndata + col % N;

        float sum = 0.0;
        for (size_t j = 0; j < Nq; ++j)
        {
            sum += (udata[drow+j] - udata[dcol+j]) *
                (udata[drow+j] - udata[dcol+j]) / Nq;
        }
        out[i] = sum;
    }
}

__global__
void dmat_comp(float* out, float const* udata,
        size_t const* i0_ptr,
        size_t const* Nentries_ptr,
        size_t const* NM_ptr,
        size_t const* N_ptr,
        size_t const* Nq_ptr)
{
    size_t const Nentries = *Nentries_ptr;
    size_t const NM = *NM_ptr;

    size_t const N = *N_ptr;
    size_t const Nq = *Nq_ptr;
    size_t const Ndata = N + Nq - 1;

    size_t const p0 = *i0_ptr;

    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < Nentries; i += stride)
    {
        // Compute the distance matrix row and column indices.
        size_t row = 0;
        size_t col = 0;
        tri_ind(&row, &col, p0 + i, NM);

        // Row and column indices for the state data.
        // They are different from the above because of the
        // time delay embedding.
        size_t drow = (row / N) * Ndata + row % N;
        size_t dcol = (col / N) * Ndata + col % N;

        float sum = 0.0;
        float err = 0.0;

        // Evaluate the distance matrix entry.
        // Uses Kahan compensated summation.
        for (size_t j = 0; j < Nq; ++j)
        {
            float summand = (udata[drow+j] - udata[dcol+j]) *
                    (udata[drow+j] - udata[dcol+j]) / Nq - err;

            float volatile temp = sum + summand;
            float volatile diff = temp - sum;
            err = diff - summand;

            sum = temp;
        }
        out[i] = sum;
    }
}

__global__
void kernelsum(float* out, float const* dmat, float const* bw,
        size_t const* Nentries_ptr,
        size_t const* N_ptr)
{
    extern __shared__ float thread_sum[]; // size blockDim.x
    thread_sum[threadIdx.x] = 0.0;

    size_t const Nentries = *Nentries_ptr;
    size_t const N = *N_ptr;

    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < Nentries; i += stride)
    {
        thread_sum[threadIdx.x] += exp(-dmat[i] / *bw) / N;
    }
    __syncthreads();

    // Reduce the thread sums within the block using a binary tree.
    // Starting from the largest power of two below blockDim.x
    // handles a block size that is not a power of two.
    unsigned int s = 1;
    while (2 * s < blockDim.x) {
        s *= 2;
    }

    for (; s > 0; s /= 2)
    {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            thread_sum[threadIdx.x] += thread_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(out, thread_sum[0]); // each block updates global memory
    }
}

__global__
void kernelsum_comp(float* out, float const* dmat, float const* bw,
        size_t const* Nentries_ptr,
        size_t const* N_ptr)
{
    extern __shared__ float thread_sum[]; // size blockDim.x
    thread_sum[threadIdx.x] = 0.0;

    size_t const Nentries = *Nentries_ptr;
    size_t const N = *N_ptr;

    size_t const tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = gridDim.x * blockDim.x;

    float err = 0.0;

    // Uses Kahan compensated summation.
    for (size_t i = tid; i < Nentries; i += stride)
    {
        float summand = exp(-dmat[i] / *bw) / N - err;

        float volatile temp = thread_sum[threadIdx.x] + summand;
        float volatile diff = temp - thread_sum[threadIdx.x];
        err = diff - summand;

        thread_sum[threadIdx.x] = temp;
    }
    __syncthreads();

    // Reduce the thread sums within the block using a binary tree.
    // Starting from the largest power of two below blockDim.x
    // handles a block size that is not a power of two.
    unsigned int s = 1;
    while (2 * s < blockDim.x) {
        s *= 2;
    }

    for (; s > 0; s /= 2)
    {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            thread_sum[threadIdx.x] += thread_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(out, thread_sum[0]); // each block updates global memory
    }
}

// Dynamic shared memory required by the kernelsum kernels for
// a given block size.
// Used to query the occupancy of those kernels.
struct ksum_smem
{
    __host__ __device__
    size_t operator()(int const block) const
    {
        return (size_t) block * sizeof(float);
    }
};

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

    size_t constexpr Ndata = 4607; // data time samples
    size_t constexpr Nq = 512; // delays
    size_t constexpr N = Ndata - Nq + 1; // net time samples

    size_t constexpr NM = N * M; // net product samples

    // NM * (NM - 1) must fit in a size_t.
    static_assert(NM - 1 <= SIZE_MAX / NM,
            "NM too large: NM * (NM - 1) overflows size_t");

    // Distance matrix entries to compute, given by the lower
    // triangular part of the distance matrix, exlc. the diagonal.
    size_t constexpr NMdmat = NM * (NM - 1) / 2;

    size_t constexpr Ndev = 4; // number of devices (GPUs)

    if (world_size != (int) Ndev) {
        if (rank == 0) {
            printf("Error: launched with %d ranks, need Ndev = %lu\n",
                    world_size, Ndev);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    {
        int ndevices;
        CUDA_CHECK(cudaGetDeviceCount(&ndevices));
        assert(local_rank < ndevices);
    }

    // Range of distance matrix indices this rank computes.
    // Every rank derives its own range from the constants above,
    // so the partition needs no communication.
    // The remainder falls to the last rank.
    size_t const ibeg = (size_t) rank * (NMdmat / Ndev);
    size_t const iend = rank == (int) Ndev - 1 ? NMdmat :
            ibeg + NMdmat / Ndev;

    // Number of state training data samples.
    size_t constexpr NMdata = Ndata * M;

    // Load the state training data samples.
    // Data is stored with the time index varying first.
    char fname[128];
    sprintf(fname, "data/ks_true_%lu_%lu_%lu_u32.h5", Ndata, M, Lfact);
    char constexpr dset[] = "/u";

    if (rank == 0) {
        printf("Data file: %s\nData set: %s\n", fname, &dset[1]);
    }

    float* udata = NULL;
    udata = (float*) malloc(NMdata * sizeof(float));
    assert(udata != NULL);
    h5read(udata, fname, dset, 1, NMdata);

    // Candidate bandwidth values.
    size_t constexpr Nbwmax = 256;
    float bwvec[Nbwmax] = {};

    size_t Nbw = 0;
    Nbw += regspace(&bwvec[Nbw], Nbwmax - Nbw, 1e-3f, 1e-3f, 9);
    Nbw += regspace(&bwvec[Nbw], Nbwmax - Nbw, 1e-2f, 1e-2f, 9);
    Nbw += regspace(&bwvec[Nbw], Nbwmax - Nbw, 1e-1f, 1e-1f, 9);
    Nbw += regspace(&bwvec[Nbw], Nbwmax - Nbw, 1e+0f, 1e+0f, 9);

    // The centered difference needs at least three values.
    assert(Nbw >= 3);

    CUDA_CHECK(cudaSetDevice(local_rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Number of streaming multiprocessors of this device.
    int nsm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm,
            cudaDevAttrMultiProcessorCount, local_rank));

    // Block sizes that maximize the occupancy of each kernel,
    // and the number of blocks each SM then holds.
    // Both kernels use a grid stride loop, so the grid only
    // needs to be large enough to fill the device.
    int gridmin = 0;

    int dmat_block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&gridmin,
            &dmat_block, dmat_comp, 0, 0));

    int dmat_bpsm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &dmat_bpsm, dmat_comp, dmat_block, 0));

    // The shared memory of kernelsum_comp scales with the block
    // size, so the query takes the mapping between the two
    // instead of a fixed size.
    int ksum_block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
            &gridmin, &ksum_block, kernelsum_comp,
            ksum_smem(), 0));

    size_t const ksum_shmem = ksum_smem()(ksum_block);

    int ksum_bpsm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ksum_bpsm, kernelsum_comp, ksum_block, ksum_shmem));

    // Number of blocks needed to fill the device once.
    size_t const dmat_wave = (size_t) nsm * dmat_bpsm;
    size_t const ksum_wave = (size_t) nsm * ksum_bpsm;

    printf("(%d) dmat: %d x %lu, kernelsum: %d x %lu\n", rank,
            dmat_block, dmat_wave, ksum_block, ksum_wave);

    // Copy data to the device memory.
    float* udata_dev = NULL;
    CUDA_CHECK(cudaMalloc((float**) &udata_dev,
                NMdata * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(udata_dev, udata, NMdata * sizeof(float),
            cudaMemcpyHostToDevice));
    free(udata);

    float* bwvec_dev = NULL;
    CUDA_CHECK(cudaMalloc((float**) &bwvec_dev, Nbw * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(bwvec_dev, bwvec, Nbw * sizeof(float),
            cudaMemcpyHostToDevice));

    size_t* NM_dev = NULL;
    CUDA_CHECK(cudaMalloc((size_t**) &NM_dev, sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(NM_dev, &NM, sizeof(size_t),
            cudaMemcpyHostToDevice));

    size_t* N_dev = NULL;
    CUDA_CHECK(cudaMalloc((size_t**) &N_dev, sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(N_dev, &N, sizeof(size_t),
            cudaMemcpyHostToDevice));

    size_t* Nq_dev = NULL;
    CUDA_CHECK(cudaMalloc((size_t**) &Nq_dev, sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(Nq_dev, &Nq, sizeof(size_t),
            cudaMemcpyHostToDevice));

    // Storage for the kernel sums, initialized to zero.
    float* ksum_dev = NULL;
    CUDA_CHECK(cudaMalloc((float**) &ksum_dev, Nbw * sizeof(float)));
    CUDA_CHECK(cudaMemset(ksum_dev, 0, Nbw * sizeof(float)));

    // Total number of distance matrix entries to compute.
    size_t const Ntotal = iend - ibeg;

    // Size of each batch of distance matrix entries.
    // Upper bound is 9e+9 floats (approx. 36 GiB).
    size_t constexpr Nlimit = 9000000000ull;
    size_t Nentries = Nlimit < Ntotal ? Nlimit : Ntotal;

    size_t* Nentries_dev = NULL;
    CUDA_CHECK(cudaMalloc((size_t**) &Nentries_dev, sizeof(size_t)));

    // Number of batches needed to compute all entries.
    size_t const Nbatch = (Ntotal + Nentries - 1) / Nentries;
    printf("(%d) Number of batches: %lu\n", rank, Nbatch);

    // Storage for each batch of distance matrix entries.
    float* dmat_dev = NULL;
    CUDA_CHECK(cudaMalloc((float**) &dmat_dev, Nentries * sizeof(float)));

    // Index of the first distance matrix entry of the current
    // batch, when viewing the matrix as a flattened 1D array.
    // The row and column indices it corresponds to are
    // recovered on the device by tri_ind.
    size_t i0 = ibeg;

    size_t* i0_dev = NULL;
    CUDA_CHECK(cudaMalloc((size_t**) &i0_dev, sizeof(size_t)));

    for (size_t i = 0; i < Nbatch; ++i)
    {
        // Update i0 and Nentries in the device memory.
        CUDA_CHECK(cudaMemcpy(i0_dev, &i0, sizeof(size_t),
                    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(Nentries_dev, &Nentries, sizeof(size_t),
                cudaMemcpyHostToDevice));

        printf("(%d) Batch %lu\n", rank, i);

        // Grid sizes for this batch: one wave that fills the
        // device, capped by the number of entries to compute.
        size_t const dmat_nb = (Nentries + dmat_block - 1) /
                dmat_block;
        size_t const dmat_grid = dmat_nb < dmat_wave ? dmat_nb :
                dmat_wave;

        size_t const ksum_nb = (Nentries + ksum_block - 1) /
                ksum_block;
        size_t const ksum_grid = ksum_nb < ksum_wave ? ksum_nb :
                ksum_wave;

        // Compute the next batch of distance matrix entries.
        dmat_comp<<<dmat_grid, dmat_block, 0, stream>>>(dmat_dev,
                udata_dev, i0_dev, Nentries_dev, NM_dev, N_dev,
                Nq_dev);
        CUDA_CHECK(cudaGetLastError());

        // Exponentiate and sum for each bandwidth value.
        for (size_t j = 0; j < Nbw; ++j)
        {
            kernelsum_comp<<<ksum_grid, ksum_block, ksum_shmem, stream>>>
                    (&ksum_dev[j], dmat_dev, &bwvec_dev[j],
                    Nentries_dev, NM_dev);
            CUDA_CHECK(cudaGetLastError());
        }

        // Update the flattened index and batch size.
        i0 += Nentries;
        Nentries = i0 + Nlimit < iend ? Nlimit : iend - i0;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Store the kernel sums of this rank.
    float* ksum = NULL;
    ksum = (float*) malloc(Nbw * sizeof(float));
    assert(ksum != NULL);

    CUDA_CHECK(cudaMemcpy(ksum, ksum_dev, Nbw * sizeof(float),
            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(ksum_dev));

    cudaFree(i0_dev);
    cudaFree(dmat_dev);
    cudaFree(Nentries_dev);
    cudaFree(Nq_dev);
    cudaFree(N_dev);
    cudaFree(NM_dev);
    cudaFree(bwvec_dev);
    cudaFree(udata_dev);
    cudaStreamDestroy(stream);

    // Reduce the partial kernel sums onto rank 0.
    float* ksum_all = NULL;
    if (rank == 0) {
        ksum_all = (float*) malloc(Nbw * sizeof(float));
        assert(ksum_all != NULL);
    }
    MPI_CHECK(MPI_Reduce(ksum, ksum_all, (int) Nbw, MPI_FLOAT,
            MPI_SUM, 0, MPI_COMM_WORLD));
    free(ksum);

    // Rank 0 holds the total kernel sums and reports the results.
    if (rank == 0)
    {
        // Take the log of the kernel sum results.
        // Adding the log(NM) term is optional, since
        // it does not affect the derivative computation.
        for (size_t i = 0; i < Nbw; ++i)
        {
            ksum_all[i] = log(2 * ksum_all[i] + 1); // + log(NM);
        }

        float logbw[Nbwmax] = {};
        for (size_t i = 0; i < Nbw; ++i)
        {
            logbw[i] = log(bwvec[i]);
        }

        float dlogS[Nbwmax] = {};
        for (size_t i = 1; i < Nbw-1; ++i)
        {
            dlogS[i-1] = (ksum_all[i+1] - ksum_all[i-1]) /
                    (logbw[i+1] - logbw[i-1]);
        }

        // Locate the first maximum of the slope.
        size_t idmax = 0;
        for (size_t i = 1; i < Nbw-2; ++i)
        {
            if (dlogS[i] > dlogS[idmax]) {
                idmax = i;
            }
        }
        float eopt = bwvec[idmax+1];
        float mdim = 2.0 * dlogS[idmax];

        // Write the results.
        FILE* outfile = NULL;
        outfile = fopen("bw0.out", "w");
        assert(outfile != NULL);

        for (size_t i = 1; i < Nbw-1; ++i)
        {
            fprintf(outfile, "%.5e %.5e %.5e\n", bwvec[i], ksum_all[i],
                    dlogS[i-1]);
        }
        fprintf(outfile, "\n%.5e\n%.5e\n", eopt, mdim);
        fclose(outfile);

        free(ksum_all);
    }

    time_t t1 = time(NULL);
    if (rank == 0) {
        printf("Total time: %.2f s\n", difftime(t1, t0));
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
