NVCC=nvcc
CFLAGS=-fPIC -Wall -Wextra
ARCH=-gencode arch=compute_50,code=sm_50 \
     -gencode arch=compute_61,code=sm_61 \
     -gencode arch=compute_75,code=sm_75 \
	 -gencode arch=compute_35,code=sm_35
NVCCFLAGS=--shared $(ARCH) --expt-extended-lambda -std=c++11 --compiler-options '$(CFLAGS)' -O3 --verbose
OBJ=build/libphylogenetics.so
SRC= src/main/cpp/*.cu

all: $(OBJ)

$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(OBJ) $(SRC)

.PHONY: clean

clean:
	rm -f $(OBJ)
