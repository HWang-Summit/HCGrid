CC          := clang
CXX         := clang++
AR          := ar -rc
CUDA_PATH   ?=/usr/local/cuda-8.0
NVCC        :=$(CUDA_PATH)/bin/nvcc
INCLUDE     :=-I/usr/local/cuda-8.0/include\
			-I/usr/local/cuda-8.0/samples/common/inc\
			-I/usr/include/c++\
			-I/home/summit/HCGrid\
			-I./
LIBRARIES   :=-L/usr/local/cuda-8.0/lib64 -lcudart -lcufft
CUDA_ARCH=-gencode arch=compute_35,code=sm_35 \
		  -gencode arch=compute_50,code=sm_50 \
		  -gencode arch=compute_60,code=sm_60
	    
NVCC_FLAGS := -Wno-deprecated-gpu-targets -dc
CXX_FLAGS := -std=c++11
FITS_FLAGS := -lm -lcfitsio -lwcs

helpers.o: helpers.cu
#	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) --ptxas-options=-v $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(INCLUDE) -o $@ -c $<

healpix.o: healpix.cu
#	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) --ptxas-options=-v $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(INCLUDE) -o $@ -c $<

gmap.o: gmap.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

gridding.o: gridding.cu
#	$(NVCC) -O3 -ccbin gcc-5 $(NVCC_FLAGS) --ptxas-options=-v $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<
	$(NVCC) -O3 -ccbin gcc-5 $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

HCGrid.o: HCGrid.cpp
	$(NVCC) -O3 -ccbin $(CXX) $(NVCC_FLAGS) $(CXX_FLAGS) $(INCLUDE) -o $@ -c $<

HCGrid: helpers.o healpix.o gmap.o gridding.o HCGrid.o
	@ echo ./$@ $+
	$(NVCC) -O3 -ccbin $(CXX) -Wno-deprecated-gpu-targets $(INCLUDE) -o $@ $+ $(CUDA_ARCH) $(LIBRARIES) $(FITS_FLAGS) $(CXX_FLAGS)

run_gauss1d_grid:
	./HCGrid --fits_path /home/summit/HCGrid/data/ --input_file go --target_file target --sorted_file sort --output_file out  --fits_id 100 --beam_size 300 --order_arg 1 --block_num 96

create_maps:
	./createGo.py -p /home/summit/hygrid/data/ -i go -t target -n 100 -s 10000

show_maps:
	python ./showImage.py -p /home/summit/hygrid/data/ -i go -o out -n 100

clear_test:
	rm -rf *.o HCGrid
