# HCGrid
- *Version:* 1.0.0
- *Authors:*

## Introduction
Gridding refers to map the non-uniform sampled data to a uniform grid, which is one of the key steps in the data processing pipeline of the radio telescope. The computation performance is the main bottleneck of gridding, which affects the whole performance of the radio data reduction pipeline and even determines the release speed of available astronomical scientific data.
HCGrid is a convolution-based gridding framework for radio astronomy on CPU/GPU heterogeneous platform, which can efficiently resample raw non-uniform sample data into a uniform grid of arbitrary resolution, generate DataCube and save the processing result as FITS file. 

## implementation

<img src=pic/HCGrid.png width="65%">
