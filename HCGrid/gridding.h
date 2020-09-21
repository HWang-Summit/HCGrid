// --------------------------------------------------------------------
//
// title                  :gridding.h
// description            :Sort and Gridding process.
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

#ifndef SORTGRID_H
#define SORTGRID_H

#include "gmap.h"

/* Block Indirect Sort input points by their healpix indexes;
 * Reorder lons, lats and data of input points by their healpix index.;
 * Pre-process by h_hpx_idx*/
void init_input();

/* 
 * func: Sort input points with CPU
 * sort_param:
 * - BLOCK_INDIRECT_SORT
 * - PARALLEL_STABLE_SORT
 * - STL_SORT
 *   */
void init_input_with_cpu(const int &sort_param);

/* 
 * func: Sort input points with Thrust
 * sort_param:
 * - THRUST
 * */
void init_input_with_thrust(const int &sort_param);

/* Initialize output spectrals and weights. */
void init_output();

/* Sinc function with simple singularity check. */
__device__ double sinc(double x);

/* Grid-kernel definitions. */
__device__ double kernel_func_ptr(double distance, double bearing);

/* Binary searcha key in hpx_idx array. */
__host__ __device__ uint32_t searchLastPosLessThan(uint64_t *values, uint32_t left, uint32_t right, uint64_t _key);

/* Gridding on GPU. */
__global__ void hcgrid (
        double *d_lons,
        double *d_lats,
        double *d_data,
        double *d_weights,
        double *d_xwcs,
        double *d_ywcs,
        double *d_datacube,
        double *d_weightscube,
        uint64_t *d_hpx_idx);

/* Alloc data for GPU. */
void data_alloc();

/* Send data from CPU to GPU. */
void data_h2d();

/* Send data from GPU to CPU. */
void data_d2h();

/* Release data. */
void data_free();

/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int& param, const int &bDim);

/* Print a array pair. */
extern void print_double_array(double *A, double *B, uint32_t num);

#endif  //SORTGRID_H
