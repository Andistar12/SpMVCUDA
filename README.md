# SpMV CUDA

This program implements the sparse matrix-vector multiply (spmv) routine for CSR and ELLPACK matrix formats using CUDA. 

## Usage

The program takes three CLI arguments: `matrix`, `vector`, and `result`. `matrix` is a path to a file stored in matrix market format. `vector` is a file with one float per line except the very first line, which contains an integer for the number of lines following it (i.e. the length of the vector). `result` is a path to where the resulting vector should be stored on disk.

## Citations

* N. Bell and M. Garland, "Implementing sparse matrix-vector multiplication on throughput-oriented processors," Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis, Portland, OR, 2009, pp. 1-11, doi: 10.1145/1654059.1654078.
* Jee W. Choi, Amik Singh, and Richard W. Vuduc. 2010. Model-driven autotuning of sparse matrix-vector multiply on GPUs. SIGPLAN Not. 45, 5 (May 2010), 115–126. DOI:https://doi.org/10.1145/1837853.1693471
