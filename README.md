# HCGrid
- *Version:* 1.0.0
- *Authors:*

## Introduction
Gridding refers to map the non-uniform sampled data to a uniform grid, which is one of the key steps in the data processing pipeline of the radio telescope. The computation performance is the main bottleneck of gridding, which affects the whole performance of the radio data reduction pipeline and even determines the release speed of available astronomical scientific data. For the single-dish radio telescope, the representative solution for this problem is implemented on the multi-CPU platform and that mainly based on the convolution gridding algorithm. Such methods can achieve a good performance through parallel threads on multi-CPU architecture, however, its performance is still limited to a certain extent by the homogeneous multi-core hardware architecture.
HCGrid is a convolution-based gridding framework for radio astronomy on CPU/GPU heterogeneous platform, which can efficiently resample raw non-uniform sample data into a uniform grid of arbitrary resolution, generate DataCube and save the processing result as FITS file. 

## implementation

<P align="center"><img src=pic/HCGrid.png height="300px"></img></p>
