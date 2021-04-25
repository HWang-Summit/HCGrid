// --------------------------------------------------------------------
//
// title                  :gridding.h
// description            :Sort and Gridding process.
// author                 :
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
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int& param, const int &bDim, int argc, char **argv);

/* Print a array pair. */
extern void print_double_array(double *A, double *B, uint32_t num);

#endif  //SORTGRID_H
