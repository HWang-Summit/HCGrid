# HCGrid
- *Version:* 1.0.0
- *Authors:*

## Introduction
Gridding refers to map the non-uniform sampled data to a uniform grid, which is one of the key steps in the data processing pipeline of the radio telescope. The computation performance is the main bottleneck of gridding, which affects the whole performance of the radio data reduction pipeline and even determines the release speed of available astronomical scientific data. For the single-dish radio telescope, the representative solution for this problem is implemented on the multi-CPU platform and that mainly based on the convolution gridding algorithm. Such methods can achieve a good performance through parallel threads on multi-CPU architecture, however, its performance is still limited to a certain extent by the homogeneous multi-core hardware architecture.
HCGrid is a convolution-based gridding framework for radio astronomy on CPU/GPU heterogeneous platform, which can efficiently resample raw non-uniform sample data into a uniform grid of arbitrary resolution, generate DataCube and save the processing result as FITS file. 

## implementation
<P align="center"><img src=pic/HCGrid.png width="50%"></img></p>
The main work of HCGrid include three-part:

- **Initialization module:** This module mainly initializes some parameters involved in the calculation process, such as setting the size of the sampling space, output resolution and other parameters.
- **Gridding module:** The core functional modules of the HCGrid. The key to improving gridding performance is to increase the speed of convolution calculations. First, in order to reduce the search space of the original sampling points, we use a parallel ordering algorithm to preorder the sampling points based on HEALPix on the CPU platform and propose an efficient two-level lookup table to speed up the acquisition of sampling points. Then, accelerating convolution by utilizing the high parallelism of GPU and through related performance optimization strategies based on CUDA architecture to further improve the gridding performance.
- **Results Process module:** Responsible for verifying the accuracy of gridding results; Save processing results as FITS files and visualize the computing results.

## Features
- Supports WCS projection system as target.
- High performance.
- Scales well on CPU/GPU heterogeneous platforms.

## Build
- The Makefile is very simple, you can easily adapt it to any Unix-like OS.
- Change the path of CUDA to match your server, in general, the default installation path for CUDA is /usr/local/cuda-xx.
- If using profiling, it is necessary to use the compile option --ptxas-options=-v to view all static resources identified by the kernel during the compilation phase, such as register resources, shared memory resources, etc. 

