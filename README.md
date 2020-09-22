# HCGrid
- *Version:* 1.0.0
- *Authors:*

## Introduction
Gridding refers to map the non-uniform sampled data to a uniform grid, which is one of the key steps in the data processing pipeline of the radio telescope. The computation performance is the main bottleneck of gridding, which affects the whole performance of the radio data reduction pipeline and even determines the release speed of available astronomical scientific data. For the single-dish radio telescope, the representative solution for this problem is implemented on the multi-CPU platform and that mainly based on the convolution gridding algorithm. Such methods can achieve a good performance through parallel threads on multi-CPU architecture, however, its performance is still limited to a certain extent by the homogeneous multi-core hardware architecture.
HCGrid is a convolution-based gridding framework for radio astronomy in CPU/GPU heterogeneous platform, which can efficiently re-sample raw non-uniform sample data into a uniform grid of arbitrary resolution, generate Data Cube and save the processing result as FITS file. 

## implementation
<P align="center"><img src=pic/HCGrid.png width="50%"></img></p>
The main work of HCGrid include three-part:

- **Initialization module:** This module mainly initializes some parameters involved in the calculation process, such as setting the size of the sampling space, output resolution and other parameters.
- **Gridding module:** The core functional modules of the HCGrid. The key to improving gridding performance is to increase the speed of convolution calculations. First, in order to reduce the search space of the original sampling points, we use a parallel ordering algorithm to pre-order the sampling points based on HEALPix on the CPU platform and propose an efficient two-level lookup table to speed up the acquisition of sampling points. Then, accelerating convolution by utilizing the high parallelism of GPU and through related performance optimization strategies based on CUDA architecture to further improve the gridding performance.
- **Results Process module:** Responsible for verifying the accuracy of gridding results; Save processing results as FITS files and visualize the computing results.

## Features
- Supports WCS projection system as target.
- High performance.
- Scales well in CPU/GPU heterogeneous platforms.

## Build
- The Makefile is very simple, you can easily adapt it to any Unix-like OS.
- Change the path of CUDA to match your server, in general, the default installation path for CUDA is /usr/local/cuda-xx.
- If using profiling, it is necessary to use the compile option --ptxas-options=-v to view all static resources identified by the kernel during the compilation phase, such as register resources, shared memory resources, etc.

## Dependencies
We kept the dependencies as minimal as possible. The following packages are required:
- cfitsio-3.47 or later
- wcslib-5.16 or later
- CUDA Toolkit

 All of these packages can be found in "Dependencies" directory or get from follow address:

- cfitsio: https://heasarc.gsfc.nasa.gov/fitsio/
- wcslib: https://www.atnf.csiro.au/people/Mark.Calabretta/WCS/
- CUDA: https://developer.nvidia.com/cuda-toolkit-archive

**Note:** The installation paths of these dependent libraries need to be updated to the makefile for subsequent compilation work.

## Usage
### DEMO
Using HCGrid is extremely simple. Just define a FITS header(with valid WCS), define gridding kernel, pre-sorting sample points and run the gridding function.
1. Define a FITS header and define gridding kernel:
``` python
# define target FITS/WCS header
header = {
	'NAXIS': 3,
	'NAXIS1': dnaxis1,
	'NAXIS2': dnaxis2,
	'NAXIS3': 1,  
	'CTYPE1': 'RA---SIN',
	'CTYPE2': 'DEC--SIN',
	'CUNIT1': 'deg',
	'CUNIT2': 'deg',
	'CDELT1': -pixsize,
	'CDELT2': pixsize,
	'CRPIX1': dnaxis1 / 2.,
	'CRPIX2': dnaxis2 / 2.,
	'CRVAL1': mapcenter[0],
	'CRVAL2': mapcenter[1],
	}
```
```c++
/*Set kernel*/
kernel_type = GAUSS1D;
kernelsize_fwhm = 300./3600.;
kernelsize_sigma = 0.2;
kernel_params[0] = kernelsize_sigma;
sphere_radius = 3.*kernelsize_sigma;
hpx_max_resolution=kernelsize_sigma/2;
_prepare_grid_kernel(
	kernel_type, 
	kernel_params, 
	sphere_radius, 
	hpx_max_resolution
	);
```

2. Select a suitable pre-sorting interface to pre-sort sampling points. Our program provides a variety of pre-sorting interfaces to pre-order the sampling points. such as "BLOCK_INDIRECT_SORT", "PARALLEL_STABLE_SORT", etc. Through a series of experiments, we demonstrated that the BlockIndirectSort based on CPU multi-thread could achieve the best performance when dealing with large-scale data. So, we set BlockIndirectSort as the default pre-sorting interface in our program.

``` C++
/* 
 * func: Sort input points with CPU
 * sort_param:
 * - BLOCK_INDIRECT_SORT
 * - PARALLEL_STABLE_SORT
 * - STL_SORT
 *   */
void init_input_with_cpu(const int &sort_param) {
    double iTime1 = cpuSecond();
    uint32_t data_shape = h_GMaps.data_shape;
    std::vector<HPX_IDX> V(data_shape);
    V.reserve(data_shape);

    // Sort input points by param
    double iTime2 = cpuSecond();
    if (sort_param == BLOCK_INDIRECT_SORT) {
        boost::sort::block_indirect_sort(V.begin(), V.end());
    } else if (sort_param == PARALLEL_STABLE_SORT) {
        boost::sort::parallel_stable_sort(V.begin(), V.end());
    } else if (sort_param == STL_SORT) {
        std::sort(V.begin(), V.end());
    }
}

/* 
 * func: Sort input points with Thrust
 * sort_param:
 * - THRUST
 * */
void init_input_with_thrust(const int &sort_param) {
    double iTime1 = cpuSecond();
    uint32_t data_shape = h_GMaps.data_shape;

    // Sort input points by param
    double iTime2 = cpuSecond();
    if (sort_param == THRUST) {
        thrust::sort_by_key(h_hpx_idx, h_hpx_idx + data_shape, in_inx);
    }
}
```
3. Do the gridding.
``` C++
/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int& param, const int &bDim) {
    
    // Read input points.
    read_input_map(infile);

    // Read output map.
    read_output_map(tarfile);

    // Set wcs for output pixels.
    set_WCS();

    // Initialize output spectrals and weights.
    init_output();

    // Block Indirect Sort input points by their healpix indexes.
    if (param == THRUST) {
        init_input_with_thrust(param);
    } else {
        init_input_with_cpu(param);
    }

    // Alloc data for GPU.
    data_alloc();

    // Send data from CPU to GPU.
    data_h2d();
    printf("h_zyx[1]=%d, h_zyx[2]=%d, ", h_zyx[1], h_zyx[2]);

    // Set block and thread.
    dim3 block(bDim);
    dim3 grid((h_GMaps.block_warp_num * h_zyx[1] - 1) / (block.x / 32) + 1);
    printf("grid.x=%d, block.x=%d, ", grid.x, block.x);

    // Call device kernel.
    hcgrid<<< grid, block >>>(d_lons, d_lats, d_data, d_weights, d_xwcs, d_ywcs, d_datacube, d_weightscube, d_hpx_idx);

    // Send data from GPU to CPU
    data_d2h();

    // Write output FITS file
    write_output_map(outfile);

    // Write sorted input FITS file
    if (sortfile) {
        write_ordered_map(infile, sortfile);
    }

    // Release data
    data_free();
}
```
### Minimal example
In the terminal, change directory to HCGrid, then:
- make HCGrid (generate an executable file: HCGrid).
- Type "**./HCGrid -h**" to get the detail parameter guide.
- ./HCGrid [options]. The options include the following parameter:

| Parameter | Description |
| :----------| :-----------------------------------|
| fits_path  | Absolute path of FITS file          |
| input_file | Name of unsorted input FITS file    |
| target_file| Name of target FITS file            |
| output_file| Name of output FITS file            |
| sorted_file| Name of sorted input FITS file      |
| fits_id    | ID of FITS file                     |
| beam_size  | Beam size of FITS file              |
| ord_arg    | Select the preorder function        |
| block_num  | The number of thread in each block  |
| coarsening_factor| The value of coarsening factor|

After setting the relevant parameters (for example, sphere_radius in HCGrid.cpp, etc), perform gridding operation according to the following steps:

1. Create the target map file:

```shell
$ python Creat_target_file.py -p /home/summit/Project/HCGrid/data/ -t target -n 1 -b 300
```

**Note:** You need to set the relevant parameters of target_map according to the coverage sky area and beam width of the sampled data. For details, please refer to "Creat_target_file.py " file.

2. Compile HCGrid:

```shell
$ make clean
$ make HCGrid
```

3. Utilizing  HCGrid do the gridding:

For the thread organization configuration, the architecture of the GPU and the number of SPs in the
SM should be carefully adjusted, in order to select the most appropriate scheme to improve the performance of GPU parallelization. For mainstream GPU architectures by NVIDIA, including Turing,
Volta, Pascal, Kepler, Fermi, and Maxwell, the minimum number of SPs in each SM equals to 32 (for Fermi architecture). When taking thread configuration into consideration only, we get the empirical equation as follows:  


$$
T_{max} = (Register\_{num}) / 184
$$

$$
blockdim.x = \left\{
\begin{array}{rcl}
SP        &      & {32      <      SP <\frac {1}{2}T_{max}}\\
T_{max}   &      & {SP >= \frac {1}{2}T_{max}}\\
other    &      & {Based\ on\ astual\ test\ results}\\
\end{array} \right.
$$

$Register\_{num}$ represents the total number of registers available for each thread block of the GPU, and $T\_{max}$is the maximum number of threads that each thread block can execute simultaneously when running HCGrid. $SP$ is the number of SPs in each SM of the GPU.   

When no specific performance analysis is performed for the actual application environment, users can use the following methods for gridding, and better performance may be obtained at this time.

```shell
$ ./HCGrid --fits_path /home/summit/Project/cygrid/ --input_file input_testB_ --target_file target_testB_ --output_file output_testB_ --fits_id 1 --beam_size 300 --register_num 64 --sp_num 64 --order_arg 1
```

 While, if you want to obtain further performance improvement, you need to perform relevant analysis according to the actual situation and organize threads reasonably to achieve the best performance. 

```shell
$ ./HCGrid --fits_path /home/summit/HCGrid/data/ --input_file in --target_file target --sorted_file sort --output_file out --fits_id 100 --beam_size 300 --order_arg 1 --block_num 64
```

***Notice:***

 1. fits_path represents the absolute path to all FITS files (including input files, target map files, and output files).
 2. The gridding framework not only supports storing the data after the gridding task is completed as a FITS file but also supporting storing the CPU multi-threaded sorted data as a FITS file. The reason for this is that can use the sorted data to analyze the performance of GPU-based convolution computation accurately. When typing the parameter of "sorted_file", the sorted data will be stored in FITS,otherwise, it will not save.
 3. The parameter "block_num" represents the number of thread in each block. Changing the value of it will also change the number of block in the grid to realize the reasonable thread organization configuration. The best value of block_num has relationship with the register of GPU. For example, For Tesla K40, the total number of registers available per block is *64K*. And the compilation report shows that the kernel of HCGrid calls a total of 184 registers, because the kernel does not use shared memory to store parameters, so it is expected that each thread block can execute about 64K/184 $\approx$ 356 threads concurrently. So, the better value of block_num should close to 356.
 4. Parameter "coarsening_factor" represents the value of coarsening factor $\gamma$. When applying thread coarsening strategy in practice, the factor $\gamma$ should be reasonable setting according to the resolution of the output grid. Through experiments, we found that a large $\gamma$ would reduce the accuracy of gridding, so we suggested that the selection range of $\gamma$ should be $\gamma=1,2,3$.

### Supplementary explanation
For ease of testing and resulting verification, we provide "Create_input_file.py" and "Visualize.py" to generate test data and visualize the gridding results. In the terminal, in the path where the file is located, you can type "python Create_input.py -h" and "python Visyalize.py -h" separately to get the detail use guide. The figure below is an example of HCGrid gridding results.

<P align="center"><img src=pic/T_B.png width="100%"></img></p>

### Community Contribution and Advice

If you have any question or ideas, please don't skimp on your suggestions and welcome make a pull request. Moreover, you can contact us through the follow address.

- imwh@tju.edu.cn

- jianxiao@tju.edu.cn