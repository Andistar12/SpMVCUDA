/* This program first reads in a sparse matrix stored in matrix market format
   (mtx). It generates a set of arrays - row_ind, col_ind, and val, which stores
   the row/column index and the value for the non-zero elements in the matrix,
   respectively. This is also known as the co-ordinate format.
   A vector will also be read in from another file.

   Then, it should convert this matrix stored in COO format to the
   compressed sparse row (CSR) format and ELLPACK (ELL) format.

   Then, the program runs CPU and GPU kernels for each (4 total routines)
   to calculate a matrix-vector product with the vector from before
   (i.e. calculate the sparse matrix vector multiply, or SpMV).

   The GPU and CPU computations are validated against each other with margin
   of error specified by the MARGIN_OF_ERROR #define

   The resulting vector should then be stored in a file, one value per line,
   whose name was specified as an input to the program.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "utils.c"
#include "mmio.c"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Constants to store timing data for each method
#define NUM_TIMERS 13
#define CHECK_TIMER 12
#define SPMV_ELL_GPU_TIME 11
#define CONVERT_ELL_TIME 9
#define SPMV_ELL_TIME 10
#define VERIFY_TIME 8
#define COPYBACK_TIME 7
#define LOAD_TIME 0
#define LOAD_VEC_TIME 4
#define LOAD_GPU 5
#define SPMV_GPU_TIME 6
#define CONVERT_TIME 1
#define SPMV_TIME 2
#define STORE_TIME 3

// The number of repetitions to do for calculating SpMV
#define REPETITIONS 500

// The number of parallel accumulators to use
#define ACCUMULATORS 32

// The allowed margin of error between CPU and GPU calculations
#define MARGIN_OF_ERROR 0.00000001


/* This function performs SpMV on a matrix in CSR format
   input parameters:
       int*             csr_row_ptr The row indices of a CSR matrix
       int*             csr_col_ptr The column indices of a CSR matrix
       double*          csr_vals    The values associated with csr_row_ptr and csr_col_ptr
       int              m           The number of rows in the matrix
       int              n           The number of columns in the matrix
       int              nnz         The number of non-zero entries in the matrix
       double*          vector_x    The input vector to multiply against, length n
       double*          res         The resulting vector to store the product
   return paramters:
       none
*/
void spmv_csr_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
    double* csr_vals, int m, int n, int nnz,
    double* vector_x, double* res)
{

    for (unsigned int i = 0; i < m; i++) res[i] = 0.0;

    for (unsigned int i = 0; i < m; i++) {
        unsigned int row_begin = csr_row_ptr[i];
        unsigned int row_end = csr_row_ptr[i + 1];
        for (unsigned int j = row_begin; j < row_end; j++) {
            res[i] += csr_vals[j] * vector_x[csr_col_ind[j]];
        }
    }
}

/* This function performs SpMV on a matrix in ELL format
   input parameters:
       int*             ell_data_ptr    The data array of the ELL matrix
       int*             ell_col_ind     The column indices of the ELL matrix
       int              max_row_length  The maximum length of any one row in the ELL
       int              m               The number of rows in the COO matrix
       int              n               The number of columns in the COO matrix
       int              nnz             The number of non-zero entries in the COO
       double*          vector_x        The input vector to multiply against, length n
       double*          res             The resulting vector to store the product
   return paramters:
       none
*/
void spmv_ell_cpu(double* ell_data_ptr, unsigned int* ell_col_ind,
    int max_row_length, int m, int n, int nnz,
    double* vector_x, double* res) {
    for (unsigned int i = 0; i < m; i++) res[i] = 0.0;
    
    int idx;
    for (unsigned int row = 0; row < m; row++) {
        for (unsigned int j = 0; j < max_row_length; j++) {
            idx = j * m + row;
            res[row] += ell_data_ptr[idx] * vector_x[ell_col_ind[idx]];
        }
    }
}

/* This function prints out profiling stats for each module
   input parameters:
       none
    return parameters:
       none
*/
void print_time(double timer[])
{
    fprintf(stdout, "%15s  %s\n", "Module", "Time (s)");
    fprintf(stdout, "%15s  %f\n", "Load Matrix", timer[LOAD_TIME]);
    fprintf(stdout, "%15s  %f\n", "Load Vector", timer[LOAD_VEC_TIME]);
    fprintf(stdout, "%15s  %f\n", "Convert CSR", timer[CONVERT_TIME]);
    fprintf(stdout, "%15s  %f\n", "Convert ELL", timer[CONVERT_ELL_TIME]);
    fprintf(stdout, "%15s  %f\n", "Load GPU", timer[LOAD_GPU]);
    fprintf(stdout, "%15s  %f\n", "SpMV CSR CPU", timer[SPMV_TIME]);
    fprintf(stdout, "%15s  %f\n", "SpMV CSR GPU", timer[SPMV_GPU_TIME]);
    fprintf(stdout, "%15s  %f\n", "SpMV ELL CPU", timer[SPMV_ELL_TIME]);
    fprintf(stdout, "%15s  %f\n", "SpMV ELL GPU", timer[SPMV_ELL_GPU_TIME]);
    fprintf(stdout, "%15s  %f\n", "Copyback GPU", timer[COPYBACK_TIME]);
    fprintf(stdout, "%15s  %f\n", "Verification", timer[VERIFY_TIME]);
    fprintf(stdout, "%15s  %f\n", "Store", timer[STORE_TIME]);
    fprintf(stdout, "%15s  %f\n", "Check time", timer[CHECK_TIMER]);
}

/* This function prints out an error and crashes if a CUDA error was detected
   input parameters:
       int          cudaStatus  The CUDA status returned by CUDA methods
       const char*  error       An error string to display before crashing
    return parameters:
       none
*/
void check_cuda(cudaError_t cudaStatus, const char* error) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", error, cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
}


/* This function performs parallel SpMV and stores it in the scratchpad
   input parameters:
       int*             row_ind     The row indices of a COO matrix
       int*             col_ind     The column indices of a COO matrix
       double*          val         The values associated with row_ind and col_ind
       int              m           The number of rows in the COO matrix
       int              n           The number of columns in the COO matrix
       int              nzz         The number of non-zero entries in the COO
       double*          vector_x    The input vector to multiply against, length n
       double*          scratchpad  The scratchpad to store accumulators in
       int              size        The length of the scratchpad
   return paramters:
       none
*/
__global__ void spmv_csr_gpu_ma(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
    double* csr_vals, int m, int n, int nnz,
    double* vector_x, double* scratchpad, int size) {

    // This method runs per parallel accumulator
    // blockIdx is current row, threadIdx is parallel accumulator index
    // blockDim is how many parallel accumulators

    unsigned int scpIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (scpIdx < size) {
        scratchpad[scpIdx] = 0.0;
        unsigned int row_begin = csr_row_ptr[blockIdx.x];
        unsigned int row_end = csr_row_ptr[blockIdx.x + 1];

        for (unsigned int j = row_begin + threadIdx.x; j < row_end; j += blockDim.x) {
            scratchpad[scpIdx] += csr_vals[j] * vector_x[csr_col_ind[j]];
        }
    }
}


/* This function performs a summation of each row in the scratchpad
   input parameters:
       int*             row_ind     The row indices of a COO matrix
       int*             col_ind     The column indices of a COO matrix
       double*          val         The values associated with row_ind and col_ind
       int              m           The number of rows in the COO matrix
       int              n           The number of columns in the COO matrix
       int              nzz         The number of non-zero entries in the COO
       double*          vector_x    The input vector to multiply against, length n
       double*          scratchpad  The scratchpad to store accumulators in
       int              size        The length of the scratchpad
       int              acc         The number of parallel accumulators available
   return paramters:
       none
*/
__global__ void spmv_csr_gpu_parallel_reduction(double* res, int m, double* scratchpad, int size, int acc) {
    // This method runs once per row (blockIdx is row)

    unsigned int res_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx;
    if (res_idx < m) {
        res[res_idx] = 0.0;

        for (int i = 0; i < acc; i++) {
            idx = res_idx * acc + i;
            if (idx >= size) break;
            res[res_idx] += scratchpad[idx];
        }
    }
}


/* This function performs SpMV on a matrix in CSR format on the GPU
   All data arrays should be allocated on the GPU already
   input parameters:
       int*             row_ind     The row indices of a COO matrix
       int*             col_ind     The column indices of a COO matrix
       double*          val         The values associated with row_ind and col_ind
       int              m           The number of rows in the COO matrix
       int              n           The number of columns in the COO matrix
       int              nzz         The number of non-zero entries in the COO
       int              acc         The number of parallel accumulators to use
       double*          vector_x    The input vector to multiply against, length n
       double*          res         The resulting vector to store the product
       double*          scratchpad  The multiply-accumulate scratchpad, length m * 32
       int              size        The length of the scratchpad array
   return paramters:
       none
*/
void spmv_csr_gpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
    double* csr_vals, int m, int n, int nnz, int acc, 
    double* vector_x, double* res, double* scratchpad, int size) {
    // Blocks = m, per block = acc = 32 (exactly one warp)

    // Perform multiply-accumulate
    spmv_csr_gpu_ma << <m, acc>> > (csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, vector_x, scratchpad, size);
    check_cuda(cudaGetLastError(), "spmv_gpu_ma launch");
    check_cuda(cudaDeviceSynchronize(), "spmv_gpu_ma execute");

    // Perform reduction
    spmv_csr_gpu_parallel_reduction << <m/acc + 1, acc>> > (res, m, scratchpad, size, acc);
    check_cuda(cudaGetLastError(), "spmv_gpu_parallel_reduction launch");
    check_cuda(cudaDeviceSynchronize(), "spmv_gpu_parallel_reduction execute");
}


/* This function performs SpMV on a matrix in ELL format
   input parameters:
       int*             ell_data_ptr    The data array of the ELL matrix
       int*             ell_col_ind     The column indices of the ELL matrix
       int              max_row_length  The maximum length of any one row in the ELL
       int              m               The number of rows in the COO matrix
       int              n               The number of columns in the COO matrix
       int              nnz             The number of non-zero entries in the COO
       double*          vector_x        The input vector to multiply against, length n
       double*          res             The resulting vector to store the product
   return paramters:
       none
*/
__global__ void spmv_ell_gpu_kernel(double* ell_data_ptr, unsigned int* ell_col_ind,
    int max_row_length, int m, int n, int nnz,
        double* vector_x, double* res) {

    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        res[row] = 0.0;
        int idx;
        for (unsigned int j = 0; j < max_row_length; j++) {
            idx = j * m + row;
            res[row] += ell_data_ptr[idx] * vector_x[ell_col_ind[idx]];
        }
    }
}


/* This function performs SpMV on a matrix in ELL format on the GPU
   input parameters:
       int*             ell_data_ptr    The data array of the ELL matrix
       int*             ell_col_ind     The column indices of the ELL matrix
       int              max_row_length  The maximum length of any one row in the ELL
       int              m               The number of rows in the COO matrix
       int              n               The number of columns in the COO matrix
       int              nnz             The number of non-zero entries in the COO
       int              acc             The number of parallel accumulators to use
       double*          vector_x        The input vector to multiply against, length n
       double*          res             The resulting vector to store the product
   return paramters:
       none
*/
void spmv_ell_gpu(double* ell_data_ptr, unsigned int* ell_col_ind,
    int max_row_length, int m, int n, int nnz, int acc,
    double* vector_x, double* res) {

    spmv_ell_gpu_kernel << <m/acc + 1, acc >> > (ell_data_ptr, ell_col_ind, max_row_length,
        m, n, nnz, vector_x, res);
    check_cuda(cudaGetLastError(), "spmv_ell_gpu_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "spmv_ell_gpu_kernel execute");
}

/* This function checks the number of input parameters to the program to make
   sure it is correct. If the number of input parameters is incorrect, it
   prints out a message on how to properly use the program.
   input parameters:
       int    argc
       char** argv
    return parameters:
       none
 */
void usage(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <matrix> <vector> <result>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
}



int main(int argc, char** argv)
{
    usage(argc, argv);

    // Initialize timers
    double timer[NUM_TIMERS];
    uint64_t t0;
    for (unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();

    // Load the vector file
    char vectorName[MAX_FILENAME_SIZE];
    strcpy(vectorName, argv[2]);
    fprintf(stdout, "Vector file name: %s ... ", vectorName);

    double* vector_x;
    unsigned int vector_size;

    t0 = ReadTSC();
    read_vector(vectorName, &vector_x, &vector_size);
    timer[LOAD_VEC_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "file loaded\n");

    // Read the sparse matrix file name
    char matrixName[MAX_FILENAME_SIZE];
    strcpy(matrixName, argv[1]);
    //read_info(matrixName);
    fprintf(stdout, "Matrix file name: %s ... ", matrixName);

    // Read the sparse matrix and store it in row_ind, col_ind, and val,
    // also known as co-ordinate format.
    int ret; // Status code
    MM_typecode matcode; // ??
    int m; // Rows
    int n; // Columns
    int nnz; // Nonzeros
    int* row_ind; // Row ptr array
    int* col_ind; // Col ind array
    double* val;  // Val array

    t0 = ReadTSC();
    ret = mm_read_mtx_crd(matrixName, &m, &n, &nnz, &row_ind, &col_ind, &val,
        &matcode);
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);
    assert(n == vector_size);
    check_mm_ret(ret);

    // Convert co-ordinate format to CSR format
    fprintf(stdout, "Converting COO to CSR...");
    unsigned int* csr_row_ptr = NULL;
    unsigned int* csr_col_ind = NULL;
    double* csr_vals = NULL;
    t0 = ReadTSC();
    convert_coo_to_csr(row_ind, col_ind, val, m, n, nnz,
        &csr_row_ptr, &csr_col_ind, &csr_vals);
    timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");

    // Convert co-ordinate format to ELL format
    fprintf(stdout, "Converting COO to ELL...");
    double* ell_data_ptr = NULL;
    unsigned int* ell_col_ind = NULL;
    int max_row_length = 0;
    t0 = ReadTSC();
    convert_coo_to_ell(row_ind, col_ind, val, m, n, nnz,
        &ell_data_ptr, &ell_col_ind, &max_row_length);
    timer[CONVERT_ELL_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");

    // Setup GPU buffers
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    unsigned int* d_row_ind;
    unsigned int* d_col_ind;
    double* d_val;
    double* d_vector_x;
    double* d_sol;
    double* d_sol2;
    double* d_scratchpad;
    double* d_ell_data_ptr;
    unsigned int* d_ell_col_ind;
    int size = ACCUMULATORS * m;
    cudaError_t status;

    fprintf(stdout, "Loading GPU...");
    t0 = ReadTSC();
    status = cudaMalloc((void**)&d_row_ind, sizeof(unsigned int) * (m + 1));
    check_cuda(status, "cudaMalloc row_ind");
    status = cudaMalloc((void**)&d_col_ind, sizeof(unsigned int) * nnz);
    check_cuda(status, "cudaMalloc col_ind");
    status = cudaMalloc((void**)&d_val, sizeof(double) * nnz);
    check_cuda(status, "cudaMalloc vals");
    status = cudaMalloc((void**)&d_sol, sizeof(double) * m);
    check_cuda(status, "cudaMalloc solution");
    status = cudaMalloc((void**)&d_sol2, sizeof(double) * m);
    check_cuda(status, "cudaMalloc solution2");
    status = cudaMalloc((void**)&d_vector_x, sizeof(double) * n);
    check_cuda(status, "cudaMalloc vector_x");
    status = cudaMalloc((void**)&d_scratchpad, sizeof(double) * size);
    check_cuda(status, "cudaMalloc scratchpad");
    status = cudaMalloc((void**)&d_ell_data_ptr, sizeof(double) * max_row_length * m);
    check_cuda(status, "cudaMalloc d_ell_data_ptr");
    status = cudaMalloc((void**)&d_ell_col_ind, sizeof(unsigned int) * max_row_length * m);
    check_cuda(status, "cudaMalloc d_ell_col_ind");

    // Memcpy is dst, then src, then size, then type
    status = cudaMemcpy(d_row_ind, csr_row_ptr, (m + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    check_cuda(status, "cudaMemcpy row_ind");
    status = cudaMemcpy(d_col_ind, csr_col_ind, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    check_cuda(status, "cudaMemcpy col_ind");
    status = cudaMemcpy(d_val, csr_vals, nnz * sizeof(double), cudaMemcpyHostToDevice);
    check_cuda(status, "cudaMemcpy vals");
    status = cudaMemcpy(d_vector_x, vector_x, m * sizeof(double), cudaMemcpyHostToDevice);
    check_cuda(status, "cudaMemcpy vector_x");
    status = cudaMemcpy(d_ell_data_ptr, ell_data_ptr, sizeof(double) * m * max_row_length, cudaMemcpyHostToDevice);
    check_cuda(status, "cudaMemcpy ell_data_ptr");
    status = cudaMemcpy(d_ell_col_ind, ell_col_ind, sizeof(unsigned int) * m * max_row_length, cudaMemcpyHostToDevice);
    check_cuda(status, "cudaMemcpy ell_col_ind");
    timer[LOAD_GPU] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");


    fprintf(stdout, "Allocating solution memory...");
    double* res = (double*)malloc(sizeof(double) * m);;
    assert(res);
    double* res2 = (double*)malloc(sizeof(double) * m);;
    assert(res2);
    double* verify = (double*)malloc(sizeof(double) * m);;
    assert(verify);
    double* verify2 = (double*)malloc(sizeof(double) * m);;
    assert(verify2);
    fprintf(stdout, "done\n");

    // Calculate SpMV CSR on cpu
    fprintf(stdout, "Calculating SpMV CSR CPU... ");
    t0 = ReadTSC();
    for (unsigned int i = 0; i < REPETITIONS; i++) {
        spmv_csr_cpu(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, vector_x, res);
    }
    timer[SPMV_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");

    // Calcualte SpMV ELL on cpu
    fprintf(stdout, "Calculating SpMV ELL CPU... ");
    t0 = ReadTSC();
    for (unsigned int i = 0; i < REPETITIONS; i++) {
        spmv_ell_cpu(ell_data_ptr, ell_col_ind, max_row_length, m, n, nnz, vector_x, res2);
    }
    timer[SPMV_ELL_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");

    // Calculate SpMV CSR on gpu
    fprintf(stdout, "Calculating SpMV CSR GPU ... ");
    t0 = ReadTSC();
    for (unsigned int i = 0; i < REPETITIONS; i++) {
        spmv_csr_gpu(d_row_ind, d_col_ind, d_val, m, n, nnz, ACCUMULATORS, d_vector_x, d_sol, d_scratchpad, size);
    }
    timer[SPMV_GPU_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");

    // Calculate SpMV ELL on gpu
    fprintf(stdout, "Calculating SpMV ELL GPU ... ");
    t0 = ReadTSC();
    for (unsigned int i = 0; i < REPETITIONS; i++) {
        spmv_ell_gpu(d_ell_data_ptr, d_ell_col_ind, max_row_length, m, n, nnz, ACCUMULATORS, d_vector_x, d_sol2);
    }
    timer[SPMV_ELL_GPU_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");

    // Copy back
    t0 = ReadTSC();
    status = cudaMemcpy(verify, d_sol, m * sizeof(double), cudaMemcpyDeviceToHost);
    check_cuda(status, "cudaMemcpy verify");
    status = cudaMemcpy(verify2, d_sol2, m * sizeof(double), cudaMemcpyDeviceToHost);
    check_cuda(status, "cudaMemcpy verify2");
    timer[COPYBACK_TIME] += ElapsedTime(ReadTSC() - t0);

    // Verify equal
    double diff;
    t0 = ReadTSC();
    // Compare CSR CPU vs GPU
    for (int i = 0; i < m; i++) {
        diff = res[i] - verify[i];
        if (diff < 0) diff = -diff;
        if (diff > MARGIN_OF_ERROR) {
            fprintf(stdout, "Verification fail: CPU CSR %f vs GPU CSR %f\n", res[i], verify[i]);
            exit(EXIT_FAILURE);
        }
    }
    // Compare ELL CPU vs GPU
    for (int i = 0; i < m; i++) {
        diff = res2[i] - verify2[i];
        if (diff < 0) diff = -diff;
        if (diff > MARGIN_OF_ERROR) {
            fprintf(stdout, "Verification fail: CPU ELL %f vs GPU ELL %f\n", res2[i], verify2[i]);
            exit(EXIT_FAILURE);
        }
    }
    timer[VERIFY_TIME] = ElapsedTime(ReadTSC() - t0);

    // Store the calculated vector in a file, one element per line.
    char resName[MAX_FILENAME_SIZE];
    strcpy(resName, argv[3]);
    fprintf(stdout, "Result file name: %s ... ", resName);

    t0 = ReadTSC();
    store_result(resName, res, m);
    timer[STORE_TIME] += ElapsedTime(ReadTSC() - t0);

    fprintf(stdout, "file saved\n");

    fprintf(stdout, "Timer check: sleep 1 second...");
    t0 = ReadTSC();
    sleep(1);
    timer[CHECK_TIMER] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n\n");

    print_time(timer);

    // Free memory

    cudaFree(d_row_ind);
    cudaFree(d_col_ind);
    cudaFree(d_val);
    cudaFree(d_vector_x);
    cudaFree(d_sol);
    cudaFree(d_sol2);
    cudaFree(d_scratchpad);
    cudaFree(d_ell_col_ind);
    cudaFree(d_ell_data_ptr);

    free(csr_row_ptr);
    free(csr_col_ind);
    free(csr_vals);

    free(vector_x);
    free(res);
    free(res2);
    free(verify);
    free(verify2);

    free(row_ind);
    free(col_ind);
    free(val);
    free(ell_data_ptr);
    free(ell_col_ind);

    return 0;
}
