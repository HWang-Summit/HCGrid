# HCGrid
- *Version:* 1.0.0
- *Authors:*

## Purpose
Gridding refers to map the non-uniform sampled data to a uniform grid, which is one of the key steps in the data processing pipeline of the radio telescope. The computation performance is the main bottleneck of gridding, which affects the whole performance of the radio data reduction pipeline and even determines the release speed of available astronomical scientific data.
HCGrid is a convolution-based gridding framework for radio astronomy on CPU/GPU heterogeneous platform, which can efficiently resample raw non-uniform sample data into a uniform grid of arbitrary resolution, generate DataCube and save the processing result as FITS file. HCGrid supports WCS standard, has strong scalability, and can be adapted to many application scenarios.

## implementation

![Architure1](0A0711E4577B4D09BF9E3824A3520A49)