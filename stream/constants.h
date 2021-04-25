// --------------------------------------------------------------------
//
// title                  :constants.h
// description            :Some constants to be used.
// author                 :
//
// --------------------------------------------------------------------

#ifndef CONSTANTS_H
#define CONSTANTS_H



#include <iostream>
#include <unistd.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <algorithm>
#include <cuda.h>
#include <hdf5.h>
#include <H5Cpp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

#ifdef PI
#undef PI
#endif

#define MAX_Y (1073741824)  // 2^30
#define MAX_DIM (1024)

// Constants needed for Sort Function
#define BLOCK_INDIRECT_SORT 1
#define PARALLEL_STABLE_SORT 2
#define THRUST 3
#define STL_SORT 4

// Constants needed for GRID KERNEL
#define GAUSS1D 1
#define GAUSS2D 2
#define TAPERED_SINC 3

// Constants needed for HPX
#define NESTED 1
#define RING 2

#define PI (3.1415926535897932384626433832)
#define TWOTHIRD (2.0 / 3.0)
#define TWOPI (2 * PI)
#define HALFPI (PI / 2.)
#define INV_TWOPI (1.0 / TWOPI)
#define INV_HALFPI (1. / HALFPI)
#define NORTH_B (20)
#define SOUTH_B (160)
#define DEG2RAD (PI / 180.)
#define RAD2DEG (180. / PI)

#endif  // CONSTANTS_H
