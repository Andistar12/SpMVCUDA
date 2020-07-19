# SpMV CUDA

This program implements the sparse matrix-vector multiply (spmv) routine for CSR and ELLPACK matrix formats using CUDA. 

## Usage

The program takes three CLI arguments: `matrix`, `vector`, and `result`. `matrix` is a path to a file stored in matrix market format. `vector` is a file with one float per line except the very first line, which contains an integer for the number of lines following it (i.e. the length of the vector). `result` is a path to where the resulting vector should be stored on disk.

## Citations

* T. G. Kolda, B. W. Bader. Tensor Decompositions and Applications. SIAM Review, Vol. 51, No. 3, pp. 455-500, 2009. https://doi.org/10.1137/07070111X
* Jee W. Choi, Amik Singh, and Richard W. Vuduc. 2010. Model-driven autotuning of sparse matrix-vector multiply on GPUs. SIGPLAN Not. 45, 5 (May 2010), 115–126. DOI:https://doi.org/10.1145/1837853.1693471
