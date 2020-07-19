#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "utils.h" // Documentation is stored in here
#include <string.h>


static double g_ticks_persecond = 0.0;


void InitTSC(void)
{
#ifdef _WIN32
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    g_ticks_persecond = (double)frequency.QuadPart;
#else
    uint64_t start_tick = ReadTSC();
    usleep(1 * 1000000);   // usleep takes sleep time in us (1 millionth of a second)
    uint64_t end_tick = ReadTSC();
    g_ticks_persecond = (double)(end_tick - start_tick);
#endif

    //fprintf(stderr, "%e ticks per second.\n", g_ticks_persecond);
}


double ElapsedTime(uint64_t ticks)
{
    if (g_ticks_persecond == 0.0) {
        fprintf(stderr, "TSC timer has not been initialized.\n");
        return 0.0;
    }
    else {
        return (ticks / g_ticks_persecond);
    }
}


uint64_t ReadTSC(void)
{
#ifdef _WIN32
    LARGE_INTEGER time;
    QueryPerformanceCounter(&time);
    return (uint64_t)time.QuadPart * 1000000; // Convert to microsec   
#elif defined(__i386__)

    uint64_t x;
    __asm__ __volatile__(".byte 0x0f, 0x31":"=A"(x));
    return x;

#elif defined(__x86_64__)

    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc":"=a"(lo), "=d"(hi));
    return ((uint64_t)lo) | (((uint64_t)hi) << 32);

#elif defined(__powerpc__)

    uint64_t result = 0;
    uint64_t upper, lower, tmp;
    __asm__ __volatile__("0:                  \n"
        "\tmftbu   %0           \n"
        "\tmftb    %1           \n"
        "\tmftbu   %2           \n"
        "\tcmpw    %2,%0        \n"
        "\tbne     0b         \n":"=r"(upper), "=r"(lower),
        "=r"(tmp)
    );
    result = upper;
    result = result << 32;
    result = result | lower;
    return result;

#endif // defined(__i386__)
}


void check_mm_ret(int ret)
{
    switch (ret)
    {
    case MM_COULD_NOT_READ_FILE:
        fprintf(stderr, "Error reading file.\n");
        exit(EXIT_FAILURE);
        break;
    case MM_PREMATURE_EOF:
        fprintf(stderr, "Premature EOF (not enough values in a line).\n");
        exit(EXIT_FAILURE);
        break;
    case MM_NOT_MTX:
        fprintf(stderr, "Not Matrix Market format.\n");
        exit(EXIT_FAILURE);
        break;
    case MM_NO_HEADER:
        fprintf(stderr, "No header information.\n");
        exit(EXIT_FAILURE);
        break;
    case MM_UNSUPPORTED_TYPE:
        fprintf(stderr, "Unsupported type (not a matrix).\n");
        exit(EXIT_FAILURE);
        break;
    case MM_LINE_TOO_LONG:
        fprintf(stderr, "Too many values in a line.\n");
        exit(EXIT_FAILURE);
        break;
    case MM_COULD_NOT_WRITE_FILE:
        fprintf(stderr, "Error writing to a file.\n");
        exit(EXIT_FAILURE);
        break;
    case 0:
        fprintf(stdout, "file loaded.\n");
        break;
    default:
        fprintf(stdout, "Error - should not be here.\n");
        exit(EXIT_FAILURE);
        break;

    }
}


void read_info(char* fileName)
{
    FILE* fp;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;

    if ((fp = fopen(fileName, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if (mm_read_banner(fp, &matcode) != 0)
    {
        fprintf(stderr, "Error processing Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    }

    if (mm_read_mtx_crd_size(fp, &m, &n, &nnz) != 0) {
        fprintf(stderr, "Error reading size.\n");
        exit(EXIT_FAILURE);
    }

    print_matrix_info(fileName, matcode, m, n, nnz);

    fclose(fp);
}


void print_matrix_info(char* fileName, MM_typecode matcode,
    int m, int n, int nnz)
{
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Matrix name:     %s\n", fileName);
    fprintf(stdout, "Matrix size:     %d x %d => %d\n", m, n, nnz);
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is matrix:       %d\n", mm_is_matrix(matcode));
    fprintf(stdout, "Is sparse:       %d\n", mm_is_sparse(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is complex:      %d\n", mm_is_complex(matcode));
    fprintf(stdout, "Is real:         %d\n", mm_is_real(matcode));
    fprintf(stdout, "Is integer:      %d\n", mm_is_integer(matcode));
    fprintf(stdout, "Is pattern only: %d\n", mm_is_pattern(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is general:      %d\n", mm_is_general(matcode));
    fprintf(stdout, "Is symmetric:    %d\n", mm_is_symmetric(matcode));
    fprintf(stdout, "Is skewed:       %d\n", mm_is_skew(matcode));
    fprintf(stdout, "Is hermitian:    %d\n", mm_is_hermitian(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");

}


void convert_coo_to_csr(int* row_ind, int* col_ind, double* val,
    int m, int n, int nnz,
    unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
    double** csr_vals)
{
    // Temporary pointers
    unsigned int* row_ptr_;
    unsigned int* col_ind_;
    double* vals_;

    // We now how large the data structures should be
    // csr_row_ptr -> m + 1
    // csr_col_ind -> nnz
    // csr_vals    -> nnz
    row_ptr_ = (unsigned int*)malloc(sizeof(unsigned int) * (m + 1));
    assert(row_ptr_);
    col_ind_ = (unsigned int*)malloc(sizeof(unsigned int) * nnz);
    assert(col_ind_);
    vals_ = (double*)malloc(sizeof(double) * nnz);
    assert(vals_);

    // Now determine how many non-zero elements are in each row
    // Use a histogram to do this
    unsigned int* buckets = (unsigned int*) malloc(sizeof(unsigned int) * m);
    assert(buckets);
    memset(buckets, 0, sizeof(unsigned int) * m);

    for (unsigned int i = 0; i < nnz; i++) {
        // row_ind[i] - 1 because index in mtx format starts from 1 (not 0)
        buckets[row_ind[i] - 1]++;
    }

    // Now use a cumulative sum to determine the starting position of each
    // row in csr_col_ind and csr_vals - this information is also what is
    // stored in csr_row_ptr
    for (unsigned int i = 1; i < m; i++) {
        buckets[i] = buckets[i] + buckets[i - 1];
    }
    // Copy this to csr_row_ptr
    row_ptr_[0] = 0;
    for (unsigned int i = 0; i < m; i++) {
        row_ptr_[i + 1] = buckets[i];
    }

    // We can use row_ptr_ to copy the column indices and vals to the 
    // correct positions in csr_col_ind and csr_vals
    unsigned int* tmp_row_ptr = (unsigned int*)malloc(sizeof(unsigned int) *
        m);
    assert(tmp_row_ptr);
    memcpy(tmp_row_ptr, row_ptr_, sizeof(unsigned int) * m);

    // Now go through each non-zero and copy it to its appropriate position
    for (unsigned int i = 0; i < nnz; i++) {
        col_ind_[tmp_row_ptr[row_ind[i] - 1]] = col_ind[i] - 1;
        vals_[tmp_row_ptr[row_ind[i] - 1]] = val[i];
        tmp_row_ptr[row_ind[i] - 1]++;
    }

    // Copy the memory address to the input parameters
    *csr_row_ptr = row_ptr_;
    *csr_col_ind = col_ind_;
    *csr_vals = vals_;

    // Free memory
    free(tmp_row_ptr);
    free(buckets);
}


/* This function converts a matrix expressed in coordinate (COO)
   format to ELLPACK (ELL) format.
   input parameters
       int*             row_ind         The row indices of a COO matrix
       int*             col_ind         The column indices of a COO matrix
       double*          val             The values associated with row_ind and col_ind
       int              m               The number of rows in the COO matrix
       int              n               The number of columns in the COO matrix
       int              nnz             The number of non-zero entries in the COO
       double**         ell_data_ptr    The array to store ELL data (column-major) in
       unsigned int**   ell_col_ind     The array to store ELL column indices in
       int*             max_row_length  The max number of non-zero entries in any row
   return paramters:
       none
*/
void convert_coo_to_ell(int* row_ind, int* col_ind, double* val,
    int m, int n, int nnz,
    double** ell_data_ptr, unsigned int** ell_col_ind, int* max_row_length) {

    // Temporary pointers
    double* data;
    unsigned int* e_col_ind;

    // Now determine how many non-zero elements are in each row
    // Use a histogram to do this
    unsigned int max_row = 0;
    unsigned int* buckets = (unsigned int*)malloc(sizeof(unsigned int) * m);
    assert(buckets);
    memset(buckets, 0, sizeof(unsigned int) * m);

    for (unsigned int i = 0; i < nnz; i++) {
        // row_ind[i] - 1 because index in mtx format starts from 1 (not 0)
        buckets[row_ind[i] - 1]++;
        if (buckets[row_ind[i] - 1] > max_row) max_row = buckets[row_ind[i] - 1];
    }

    // We now know how big data and col_ind should be
    data = (double*) malloc(sizeof(double) * m * max_row);
    assert(data);
    e_col_ind = (unsigned int*) malloc(sizeof(unsigned int) * m * max_row);
    assert(e_col_ind);

    // Zero out histogram again
    memset(buckets, 0, sizeof(unsigned int) * m);

    // Iterate through every non-zero value in array
    unsigned int index;
    for (unsigned int i = 0; i < nnz; i++) {
        index = buckets[row_ind[i] - 1] * m + row_ind[i] - 1;
        data[index] = val[i];
        e_col_ind[index] = col_ind[i] - 1;
        buckets[row_ind[i] - 1]++;
    }

    // Pad rest of arrays
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = buckets[i]; j < max_row; j++) {
            index = j * m + i;
            data[index] = 0.0;
            e_col_ind[index] = 0;
        }
    }

    // Copy the memory address to the input parameters
    *ell_data_ptr = data;
    *ell_col_ind = e_col_ind;
    *max_row_length = max_row;

    // Free memory
    free(buckets);
}


void read_vector(char* fileName, double** vector, unsigned int* vecSize)
{
    FILE* fp = fopen(fileName, "r");
    assert(fp);
    char line[MAX_NUM_LENGTH];
    fgets(line, MAX_NUM_LENGTH, fp);
    fclose(fp);

    unsigned int vector_size = atoi(line);
    double* vector_ = (double*)malloc(sizeof(double) * vector_size);

    fp = fopen(fileName, "r");
    assert(fp);
    // first read the first line to get the # elements
    fgets(line, MAX_NUM_LENGTH, fp);

    unsigned int index = 0;
    while (fgets(line, MAX_NUM_LENGTH, fp) != NULL) {
        vector_[index] = atof(line);
        index++;
    }

    fclose(fp);
    assert(index == vector_size);

    *vector = vector_;
    *vecSize = vector_size;
}


void store_result(char* fileName, double* res, int m)
{
    FILE* fp = fopen(fileName, "w");
    assert(fp);

    fprintf(fp, "%d\n", m);
    for (int i = 0; i < m; i++) {
        fprintf(fp, "%0.10f\n", res[i]);
    }

    fclose(fp);
}
