cc = /usr/local/cuda/bin/nvcc
sources = $(wildcard *.cu)
objects = $(addsuffix .o, $(basename $(sources)))
flags = -DDEBUG=1 -std=c++14
target = spmv 

$(target) : $(objects)
	$(cc) -o $(target) $(objects)

%.o : %.cu
	$(cc) -c $(flags) $< -o $@

clean:
	rm -f $(target) $(objects)
