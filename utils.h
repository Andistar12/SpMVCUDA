#pragma once
#include <stdint.h>
#include "mmio.h"
#ifdef _WIN32
#include <Windows.h>
#include <profileapi.h>
#else
#include <unistd.h>
#endif

/* The maximum allowed file name size
*/
#define MAX_FILENAME_SIZE 256


/* The maximum allowed line length within each file
*/
#define MAX_NUM_LENGTH 100


/* This function initiates the tick second counter
   input parameters:
       none
   return parameters:
       none
*/
void InitTSC(void);


/* This function calculates the elapsed time given the total ticks elapsed
   input parameters:
       uint64_t ticks   the number of ticks that have elapsed in total
   return parameters:
       double   the total elapsed time in seconds
*/
double ElapsedTime(uint64_t ticks);


/* This function returns an arbritrary value representing the current time.
   The difference of two calls to this function should be calculate along with
   the ElapsedTime(ticks) function to calculate total elapsed time
   input parameters:
       none
   return parameters:
       uint64_t the current time in tick count
*/
uint64_t ReadTSC(void);


/* This function checks the return value from the matrix read function,
   mm_read_mtx_crd(), and provides descriptive information.
   input parameters:
       int ret    return value from the mm_read_mtx_crd() function
   return paramters:
       none
 */
void check_mm_ret(int ret);


/* This function prints out information about a sparse matrix
   input parameters:
       char*       fileName    name of the sparse matrix file
       MM_typecode matcode     matrix information
       int         m           # of rows
       int         n           # of columns
       int         nnz         # of non-zeros
   return paramters:
       none
 */
void print_matrix_info(char* fileName, MM_typecode matcode,
    int m, int n, int nnz);


/* This function reads information about a sparse matrix using the
   mm_read_banner() function and printsout information using the
   print_matrix_info() function.
   input parameters:
       char*       fileName    name of the sparse matrix file
   return paramters:
       none
 */
void read_info(char* fileName);


/* This function converts a matrix expressed in coordinate (COO)
   format to column-sparse (CSR) format.
   input parameters
       int*             row_ind     The row indices of a COO matrix 
       int*             col_ind     The column indices of a COO matrix
       double*          val         The values associated with row_ind and col_ind
       int              m           The number of rows in the COO matrix
       int              n           The number of columns in the COO matrix
       int              nnz         The number of non-zero entries in the COO
       unsigned int**   csr_row_ptr The array to store CSR row pointers in
       unsigned int**   csr_col_ind The array to store CSR column indices in
       double**         csr_vals    The array to store CSR matrix values in
   return paramters:
       none
*/
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val,
    int m, int n, int nnz,
    unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
    double** csr_vals);



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
    double** ell_data_ptr, unsigned int** ell_col_ind, int* max_row_length);


/* This function reads in a vector from a file. The first line should be the
   number of elements in the vector, and there should be that many lines following
   the first, each line containing one double.
   input parameters:
       char*    fileName    The file to read the vector from
       double** vector      The array to store the vector in
       u int*   vecSize     A pointer to the size of the vector (to be overwritten)
   return parameters:
       none
 */
void read_vector(char* fileName, double** vector, unsigned int* vecSize);


/* This function stores the vector result of SpMV into the given file
   input parameters:
       char*    fileName    The file to store the result in
       double*  res         The vector array
       int      m           The length of the vector array
*/
void store_result(char* fileName, double* res, int m);
