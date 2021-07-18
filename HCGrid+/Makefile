# CC          := clang
# CXX         := gcc
AR          := ar -rc
CUDA_PATH   ?=/usr/local/cuda-10.2
NVCC        :=$(CUDA_PATH)/bin/nvcc
INCLUDE     := -I/home/summit/.local/include\
			  -I/usr/local/hdf5/include\
			  -I./
LIBRARIES   :=-L/usr/local/cuda-10.2/lib64 -lcudart -lcufft\
			  -L/home/summit/.local/lib\
			  -L/usr/local/hdf5/lib   
CUDA_ARCH =	-gencode arch=compute_60,code=sm_60
	    
NVCC_FLAGS := -Wno-deprecated-gpu-targets -dc
CXX_FLAGS := -std=c++11 -Xcompiler -fopenmp
FITS_FLAGS := -lm -lcfitsio -lwcs -lhdf5 

HCGrid: helpers.o healpix.o gmap.o gridding.o HCGrid.o
	@ echo ./$@ $+
	$(NVCC) -O3 -ccbin $(CXX) -Wno-deprecated-gpu-targets $(INCLUDE) -o $@ $+ $(CUDA_ARCH) $(LIBRARIES) $(FITS_FLAGS) $(CXX_FLAGS)
	
helpers.o: helpers.cu
#	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) --ptxas-options=-v $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(INCLUDE) -o $@ -c $<

healpix.o: healpix.cu
#	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) --ptxas-options=-v $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(INCLUDE) -o $@ -c $<

gmap.o: gmap.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

gridding.o: gridding-omp.cu
#	$(NVCC) -O3 -ccbin gcc $(NVCC_FLAGS) --ptxas-options=-v $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

HCGrid.o: HCGrid.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

clean:
	rm -rf *.o HCGrid
