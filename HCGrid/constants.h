// --------------------------------------------------------------------
//
// title                  :constants.h
// description            :Some constants to be used.
// author                 :
//
// --------------------------------------------------------------------
// Copyright (C) 2010+ by Hao Wang, Qi Luo
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// Note: Some HEALPix related helper functions and convolution algorithm 
// code are adopted from (HEALPix C++) library (Copyright (C) 2003-2012
//  Max-Planck-Society; author Martin Reinecke) and Cygrid software 
// (Copyright (C) 2010+ by Benjamin Winkel, Lars Flöer & Daniel Lenz),
// respectively.
// 
// For more information about HEALPix, see http://healpix.sourceforge.net
// Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
// and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
// (DLR).
// 
// For more information about Cygrid, see https://github.com/bwinkel/cygrid.
// Cygrid was developed in the framework of the Effelsberg-Bonn H i Survey (EBHIS).
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
#define DEG2RAD (PI / 180.)
#define RAD2DEG (180. / PI)

#endif  // CONSTANTS_H
