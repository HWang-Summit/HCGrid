// --------------------------------------------------------------------
//
// title                  :gridding.cu
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

#include <boost/sort/sort.hpp>
#include <thrust/sort.h>
#include "gridding.h"
#include "/usr/local/mpich-3.4.1/include/mpi.h"
#include "omp.h"

using boost::sort::block_indirect_sort;

#define stream_size 10
#define max_group_size 24
cudaStream_t stream[stream_size];

double *d_lons;
double *d_lats;
double *d_data;
double *d_weights;
uint64_t *d_hpx_idx;
uint32_t *d_start_ring;
texture<uint32_t> tex_start_ring;
__constant__ uint32_t d_const_zyx[3];
uint32_t *d_zyx;
double *d_xwcs;
double *d_ywcs;
double **d_datacube;
double **d_weightscube;
double **tempArray;
__constant__ double d_const_kernel_params[3];
__constant__ GMaps d_const_GMaps;
size_t pitch_d,pitch_h, pitch_r;
double coordinate_order_time = 0;
double data_order_time = 0;
double pre_order_time = 0;

void init_input_with_cpu(const int &sort_param) {
    double iTime1 = cpuSecond();
    uint32_t data_shape = h_GMaps.data_shape;
    std::vector<HPX_IDX> V(data_shape);
    V.reserve(data_shape);

    // Get healpix index and input index of each input point.
    for(int i=0; i < data_shape; ++i) {
        double theta = HALFPI - DEG2RAD * h_lats[i];
        double phi = DEG2RAD * h_lons[i];
        uint64_t hpx = h_ang2pix(theta, phi);
        V[i] = HPX_IDX(hpx, i);  
    }

    // Sort input points by param
    double iTime2 = cpuSecond();
    if (sort_param == BLOCK_INDIRECT_SORT) {
        boost::sort::block_indirect_sort(V.begin(), V.end());
    } else if (sort_param == PARALLEL_STABLE_SORT) {
        boost::sort::parallel_stable_sort(V.begin(), V.end());
    } else if (sort_param == STL_SORT) {
        std::sort(V.begin(), V.end());
    }
    double iTime3 = cpuSecond();

    // Copy the healpixes, lons, lats and data for sorted input points
    h_hpx_idx = RALLOC(uint64_t, data_shape);
    h_inx_idx = RALLOC(uint32_t, data_shape);
    for(int i=0; i < data_shape; ++i){
        h_hpx_idx[i] = V[i].hpx;
        h_inx_idx[i] = V[i].inx;
    }
    h_hpx_idx[data_shape] = h_Healpix._npix;
    double *tmpArray = RALLOC(double, data_shape);
    for(int i=0; i < data_shape; ++i){
        tmpArray[i] = h_lons[V[i].inx];
    }
    swap(h_lons, tmpArray);
    for(int i=0; i < data_shape; ++i){
        tmpArray[i] = h_lats[V[i].inx];
    }
    swap(h_lats, tmpArray);
    DEALLOC(tmpArray);

    // Pre-process by h_hpx_idx
    double iTime4 = cpuSecond();
    uint64_t first_ring = h_pix2ring(h_hpx_idx[0]);
    uint32_t temp_count = (uint32_t)(1 + h_pix2ring(h_hpx_idx[data_shape - 1]) - first_ring);
    h_Healpix.firstring = first_ring;
    h_Healpix.usedrings = temp_count;
    h_start_ring = RALLOC(uint32_t, temp_count + 1);
    h_start_ring[0] = 0;
    uint64_t startpix, num_pix_in_ring;
    uint32_t ring_idx = 0;
    bool shifted;
    for(uint64_t cnt_ring = 1; cnt_ring < temp_count; ++cnt_ring) {
        h_get_ring_info_small(cnt_ring + first_ring, startpix, num_pix_in_ring, shifted);
        uint32_t cnt_ring_idx = searchLastPosLessThan(h_hpx_idx, ring_idx, data_shape, startpix);
        if (cnt_ring_idx == data_shape) {
            cnt_ring_idx = ring_idx - 1;
        }
        ring_idx = cnt_ring_idx + 1;
        h_start_ring[cnt_ring] = ring_idx;
    }
    h_start_ring[temp_count] = data_shape;
    double iTime5 = cpuSecond();

    // Release
    vector<HPX_IDX> vtTemp;
    vtTemp.swap(V);

    // Get runtime
    double iTime6 = cpuSecond();
    coordinate_order_time = iTime6 - iTime1;
    //printf("Coords sort time = %f, ", (iTime6 - iTime1) * 1000.);
    printf("%f ", (iTime6 - iTime1) * 1000.);
}
void pre_order_data(const int dim){
    uint32_t data_shape = h_GMaps.data_shape;
    int i;
    // double *tempArray = RALLOC(double, data_shape);
    // omp_set_num_threads(8);
#pragma omp parallel for shared(tempArray, h_data) private(i)
    for(i=0; i < data_shape; ++i){
        tempArray[dim][i] = h_data[dim][h_inx_idx[i]];
    }
    swap(h_data[dim], tempArray[dim]);
    // DEALLOC(tempArray);
}
void init_input_with_thrust(const int &sort_param) {
    double iTime1 = cpuSecond();
    uint32_t data_shape = h_GMaps.data_shape;

    // Get healpix index and input index of each input point.
    h_hpx_idx = RALLOC(uint64_t, data_shape + 1);
    uint32_t *in_inx = RALLOC(uint32_t, data_shape);
    for(int i=0; i < data_shape; ++i) {
        double theta = HALFPI - DEG2RAD * h_lats[i];
        double phi = DEG2RAD * h_lons[i];
        h_hpx_idx[i] = h_ang2pix(theta, phi);
        in_inx[i] = i;
    }

    // Sort input points by param
    double iTime2 = cpuSecond();
    if (sort_param == THRUST) {
        thrust::sort_by_key(h_hpx_idx, h_hpx_idx + data_shape, in_inx);
    }
    double iTime3 = cpuSecond();

    // Copy the lons, lats and data for sorted input points
    double *tempArray = RALLOC(double, data_shape);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lons[in_inx[i]];
    }
    swap(h_lons, tempArray);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lats[in_inx[i]];
    }
    swap(h_lats, tempArray);
    // for(int i=0; i < data_shape; ++i){
    //     tempArray[i] = h_data[in_inx[i]];
    // }
    // swap(h_data, tempArray);
    DEALLOC(tempArray);

    // Pre-process by h_hpx_idx
    double iTime4 = cpuSecond();
    uint64_t first_ring = h_pix2ring(h_hpx_idx[0]);
    uint32_t temp_count = (uint32_t)(1 + h_pix2ring(h_hpx_idx[data_shape - 1]) - first_ring);
    h_Healpix.firstring = first_ring;
    h_Healpix.usedrings = temp_count;
    h_start_ring = RALLOC(uint32_t, temp_count + 1);
    h_start_ring[0] = 0;
    uint64_t startpix, num_pix_in_ring;
    uint32_t ring_idx = 0;
    bool shifted;
    for(uint64_t cnt_ring = 1; cnt_ring < temp_count; ++cnt_ring) {
        h_get_ring_info_small(cnt_ring + first_ring, startpix, num_pix_in_ring, shifted);
        uint32_t cnt_ring_idx = searchLastPosLessThan(h_hpx_idx, ring_idx, data_shape, startpix);
        if (cnt_ring_idx == data_shape) {
            cnt_ring_idx = ring_idx - 1;
        }
        ring_idx = cnt_ring_idx + 1;
        h_start_ring[cnt_ring] = ring_idx;
    }
    h_start_ring[temp_count] = data_shape;
    double iTime5 = cpuSecond();

    // Release
    DEALLOC(in_inx);

    // Get runtime
    double iTime6 = cpuSecond();
    printf("%f", (iTime6 - iTime1) * 1000.);
}

/* Initialize output spectrals and weights. */
void init_output(){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t channels = h_GMaps.spec_dim;
    HANDLE_ERROR(cudaMallocHost((void**)& h_datacube, sizeof(double*)*channels));
    HANDLE_ERROR(cudaMallocHost((void**)& h_weightscube, sizeof(double*)*channels));
    // h_datacube = RALLOC(double*, channels);
    // h_weightscube = RALLOC(double* , channels);
    for(int i = 0; i < channels; i++){
        // h_datacube[i] = RALLOC(double, num);
        HANDLE_ERROR(cudaMallocHost((void**)& h_datacube[i], sizeof(double)*num)); 
        HANDLE_ERROR(cudaMallocHost((void**)& h_weightscube[i], sizeof(double)*num));         
        // cudaMemsetHost(h_datacube[i], 0, num*sizeof(double));
        for(int j = 0; j < num; j++){
            h_datacube[i][j] = 0;
            h_weightscube[i][j] = 0;
        }
    }
}

/* Sinc function with simple singularity check. */
__device__ double sinc(double x){
    if(fabs(x) < 1.e-10)
        return 1.;
    else
        return sin(x) / x;
}

/* Grid-kernel definitions. */
__device__ double kernel_func_ptr(double distance, double bearing){
    if(d_const_GMaps.kernel_type == GAUSS1D){   // GAUSS1D
        return exp(-distance * distance * d_const_kernel_params[0]);
    }
    else if(d_const_GMaps.kernel_type == GAUSS2D){  // GAUSS2D
        double ellarg = (\
                pow(d_const_kernel_params[0], 2.0)\
                    * pow(sin(bearing - d_const_kernel_params[2]), 2.0)\
                + pow(d_const_kernel_params[1], 2.0)\
                    * pow(cos(bearing - d_const_kernel_params[2]), 2.0));
        double Earg = pow(distance / d_const_kernel_params[0] /\
                       d_const_kernel_params[1], 2.0) / 2. * ellarg;
        return exp(-Earg);
    }
    else if(d_const_GMaps.kernel_type == TAPERED_SINC){ // TAPERED_SINC
        double arg = PI * distance / d_const_kernel_params[0];
        return sinc(arg / d_const_kernel_params[2])\
            * exp(pow(-(arg / d_const_kernel_params[1]), 2.0));
    }
}

/* Binary searcha key in hpx_idx array. */
__host__ __device__ uint32_t searchLastPosLessThan(uint64_t *values, uint32_t left, uint32_t right, uint64_t _key){
    if(right <= left)
        return right;
    uint32_t low = left, mid, up = right - 1;
    while (low < up){
        mid = low + ((up - low + 1) >> 1);
        if (values[mid] < _key)
            low = mid;
        else
            up = mid - 1;
    }
    if(values[low] < _key)
        return low;
    else
        return right;
}

__global__ void hcgrid (
    double *d_lons,
    double *d_lats,
    double *d_data,
    double *d_weights,
    double *d_xwcs,
    double *d_ywcs,
    double *d_datacube,
    double *d_weightscube,
    uint64_t *d_hpx_idx) {
uint32_t warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
uint32_t thread_id = ((warp_id % d_const_GMaps.block_warp_num) * 32 + threadIdx.x % 32) * d_const_GMaps.factor;
if (thread_id < d_const_zyx[1]) {
    uint32_t left = thread_id;
    uint32_t right = left + d_const_GMaps.factor - 1;
    if (right >= d_const_zyx[1]) {
        right = d_const_zyx[1] - 1;
    }
    uint32_t step = (warp_id / d_const_GMaps.block_warp_num) * d_const_zyx[1];
    left = left + step;
    right = right + step;
    double temp_weights[3], temp_data[3], l1[3], b1[3];
    for (thread_id = left; thread_id <= right; ++thread_id) {
        temp_weights[thread_id-left] = d_weightscube[thread_id];
        temp_data[thread_id-left] = d_datacube[thread_id];
        l1[thread_id-left] = d_xwcs[thread_id] * DEG2RAD;
        b1[thread_id-left] = d_ywcs[thread_id] * DEG2RAD;
    }

    // get northeast ring and southeast ring
    double disc_theta = HALFPI - b1[0];
    double disc_phi = l1[0];
    double utheta = disc_theta - d_const_GMaps.disc_size;
    if (utheta * RAD2DEG < 0){
        utheta = 0;
    }
    uint64_t upix = d_ang2pix(utheta, disc_phi);
    uint64_t uring = d_pix2ring(upix);
    if (uring < d_const_Healpix.firstring){
        uring = d_const_Healpix.firstring;
    }
    utheta = disc_theta + d_const_GMaps.disc_size;
    upix = d_ang2pix(utheta, disc_phi);
    uint64_t dring = d_pix2ring(upix);
    if (dring >= d_const_Healpix.firstring + d_const_Healpix.usedrings){
        dring = d_const_Healpix.firstring + d_const_Healpix.usedrings - 1;
    }

    // Go from the Northeast ring to the Southeast one
    uint32_t start_int = tex1Dfetch(tex_start_ring, uring-d_const_Healpix.firstring);
    while (uring <= dring) {
        // get ring info
        uint32_t end_int = tex1Dfetch(tex_start_ring, uring-d_const_Healpix.firstring+1);
        uint64_t startpix, num_pix_in_ring;
        bool shifted;
        d_get_ring_info_small(uring, startpix, num_pix_in_ring, shifted);
        double utheta, uphi;
        d_pix2ang(startpix, utheta, uphi);

        // get lpix and rpix
        disc_theta = HALFPI - b1[0];
        disc_phi = l1[0];
        uphi = disc_phi - d_const_GMaps.disc_size;
        uint64_t lpix = d_ang2pix(utheta, uphi);
        if (disc_theta * RAD2DEG <= NORTH_B || disc_theta * RAD2DEG >= SOUTH_B){
            lpix = startpix;
        } else{
            lpix = lpix;
        }
        if (!(lpix >= startpix && lpix < startpix+num_pix_in_ring)) {
            start_int = end_int;
            continue;
        }
        uphi = disc_phi + d_const_GMaps.disc_size;
        uint64_t rpix = d_ang2pix(utheta, uphi);
        if (disc_theta * RAD2DEG <= NORTH_B || disc_theta * RAD2DEG >= SOUTH_B){
            rpix = startpix + num_pix_in_ring - 1;
        } else{
            rpix = rpix;
        }
        if (!(rpix >= startpix && rpix < startpix+num_pix_in_ring)) {
            start_int = end_int;
            continue;
        }

        // find position of lpix
        uint32_t upix_idx = searchLastPosLessThan(d_hpx_idx, start_int - 1, end_int, lpix);
        ++upix_idx;
        if (upix_idx > end_int) {
            upix_idx = d_const_GMaps.data_shape;
        }

        // Gridding
        while(upix_idx < d_const_GMaps.data_shape){
            double l2 = d_lons[upix_idx] * DEG2RAD;
            double b2 = d_lats[upix_idx] * DEG2RAD;
            upix = d_ang2pix(HALFPI - b2, l2);
            if (upix > rpix) {
                break;
            }
            double in_weights = d_weights[upix_idx];
            double in_data = d_data[upix_idx];

            for (thread_id = left; thread_id <= right; ++thread_id) {
                double sdist = true_angular_distance(l1[thread_id-left], b1[thread_id-left], l2, b2) * RAD2DEG;
                double sbear = 0.;
                if (d_const_GMaps.bearing_needed) {
                    sbear = great_circle_bearing(l1[thread_id-left], b1[thread_id-left], l2, b2);
                }
                if (sdist < d_const_GMaps.sphere_radius) {
                    double sweight = kernel_func_ptr(sdist, sbear);
                    double tweight = in_weights * sweight;
                    temp_data[thread_id-left] += in_data * tweight;
                    temp_weights[thread_id-left] += tweight;
                }
                d_datacube[thread_id] = temp_data[thread_id - left];
                d_weightscube[thread_id] = temp_weights[thread_id - left];
            }
            ++upix_idx;
        }

        start_int = end_int;
        ++uring;
    }
}
__syncthreads();
}

/* Alloc data for GPU. */
void data_alloc(){
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t channels = h_GMaps.spec_dim;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t usedrings = h_Healpix.usedrings;

    HANDLE_ERROR(cudaMalloc((void**)& d_lons, sizeof(double)*data_shape));
    HANDLE_ERROR(cudaMalloc((void**)& d_lats, sizeof(double)*data_shape));

    HANDLE_ERROR(cudaMallocHost((void**)& h_data, sizeof(double*)*channels));
    for(int i = 0; i < channels; i++){
        HANDLE_ERROR(cudaMallocHost((void**)& h_data[i], sizeof(double)*data_shape));
    }

    HANDLE_ERROR(cudaMallocPitch((void**)& d_data, &pitch_d, sizeof(double)*data_shape, channels));
    HANDLE_ERROR(cudaMallocPitch((void**)& d_datacube, &pitch_r, sizeof(double)*num, channels));
    HANDLE_ERROR(cudaMallocPitch((void**)& d_weightscube, &pitch_r, sizeof(double)*num, channels));

    HANDLE_ERROR(cudaMalloc((void**)& d_weights, sizeof(double)*data_shape));
    HANDLE_ERROR(cudaMalloc((void**)& d_xwcs, sizeof(double)*num));
    HANDLE_ERROR(cudaMalloc((void**)& d_ywcs, sizeof(double)*num));
    HANDLE_ERROR(cudaMalloc((void**)& d_hpx_idx, sizeof(uint64_t)*(data_shape+1)));
    HANDLE_ERROR(cudaMalloc((void**)& d_start_ring, sizeof(uint32_t)*(usedrings+1)));
    HANDLE_ERROR(cudaBindTexture(NULL, tex_start_ring, d_start_ring, sizeof(uint32_t)*(usedrings+1)));
}

/* Send data from CPU to GPU. */
void data_h2d(){
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t usedrings = h_Healpix.usedrings;

    // Copy constants memory
    HANDLE_ERROR(cudaMemcpy(d_lons, h_lons, sizeof(double)*data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lats, h_lats, sizeof(double)*data_shape, cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(d_data, hh_data[dim_index], sizeof(double)*data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weights, h_weights, sizeof(double)*data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_xwcs, h_xwcs, sizeof(double)*num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ywcs,h_ywcs, sizeof(double)*num, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(d_datacube, h_datacube, sizeof(double)*num, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(d_weightscube, h_weightscube, sizeof(double)*num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_hpx_idx, h_hpx_idx, sizeof(uint64_t)*(data_shape+1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_start_ring, h_start_ring, sizeof(uint32_t)*(usedrings+1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_kernel_params, h_kernel_params, sizeof(double)*3));    
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_zyx, h_zyx, sizeof(uint32_t)*3));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_Healpix, &h_Healpix, sizeof(Healpix)));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_GMaps, &h_GMaps, sizeof(GMaps)));
}

/* Send data from GPU to CPU. */
void data_d2h(uint32_t s_index, uint32_t dim){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    HANDLE_ERROR(cudaMemcpyAsync(h_datacube[dim], (double*)((char*)d_datacube+dim*pitch_r), sizeof(double)*num, cudaMemcpyDeviceToHost,stream[s_index]));
    HANDLE_ERROR(cudaMemcpyAsync(h_weightscube[dim],(double*)((char*)d_weightscube+dim*pitch_r), sizeof(double)*num, cudaMemcpyDeviceToHost,stream[s_index]));
}

/* Release data. */
void data_free(){
    DEALLOC(h_lons);
    HANDLE_ERROR( cudaFree(d_lons) );
    DEALLOC(h_lats);
    HANDLE_ERROR( cudaFree(d_lats) );
    // DEALLOC(h_data);
    for(int i = 0; i < h_GMaps.spec_dim; i++){
        // HANDLE_ERROR( cudaFreeHost(h_data[i]));
        cudaFreeHost(h_data[i]);
    }
    HANDLE_ERROR(cudaFreeHost(h_data));
    
    HANDLE_ERROR(cudaFree(d_data));
    DEALLOC(h_weights);
    HANDLE_ERROR( cudaFree(d_weights) );
    DEALLOC(h_xwcs);
    HANDLE_ERROR( cudaFree(d_xwcs) );
    DEALLOC(h_ywcs);
    HANDLE_ERROR( cudaFree(d_ywcs) );

    for(int i = 0; i < h_GMaps.spec_dim; i++){
        cudaFreeHost(h_datacube[i]);
    }
    cudaFreeHost(h_datacube);
    for(int i = 0; i < h_GMaps.spec_dim; i++){
        cudaFreeHost(h_weightscube[i]);
    }
    cudaFreeHost(h_weightscube);
 
    HANDLE_ERROR(cudaFree(d_datacube));
    HANDLE_ERROR(cudaFree(d_weightscube));
    DEALLOC(h_hpx_idx);
    HANDLE_ERROR( cudaFree(d_hpx_idx) );
    DEALLOC(h_start_ring);
    HANDLE_ERROR( cudaUnbindTexture(tex_start_ring) );
    HANDLE_ERROR( cudaFree(d_start_ring) );
    DEALLOC(h_header);
    DEALLOC(h_zyx);
    DEALLOC(h_kernel_params);
    // for(int i = 0; i < h_GMaps.spec_dim; i++){
    //     printf("after release %d\n", i);
    //     printf("%f\n", tempArray[i][0]);
    //     DEALLOC(tempArray[i]);
    // }
    DEALLOC(tempArray);
}

/*mpi read&pre-order input data*/
void MallocTempArray(){
    //read&pre-order input data
    uint32_t channels = h_GMaps.spec_dim;
    uint32_t data_shape = h_GMaps.data_shape; 
    tempArray = RALLOC(double*, channels);
    for(int i = 0; i < channels; i++){
        tempArray[i] = RALLOC(double, data_shape);
    }
//     read_input_data(infile);

//     double iTime7 = cpuSecond();
//     // omp_set_num_threads(2);
// #pragma omp parallel for 
//     for(int i = 0; i < channels; i++){
//             pre_order_data(i);           
//     }

//     double iTime8 = cpuSecond();
//     data_order_time = iTime8 - iTime7;
//     // printf("Data sort time = %f\n", (iTime8 - iTime7) * 1000.);
//     printf("%f ", data_order_time * 1000.);
}

/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int& param, const int &bDim, int argc, char **argv) {
    // Read input coordinate.
    read_input_coordinate(infile);

    // Read output map.
    read_output_map(tarfile);

    // Set wcs for output pixels.
    set_WCS();

    // Initialize output spectrals and weights.
    init_output(); 

    double iTime1 = cpuSecond();
    // Block Indirect Sort input points by their healpix indexes.
    if (param == THRUST) {
        init_input_with_thrust(param);
    } else {
        init_input_with_cpu(param);
    } 


    double iTime2 = cpuSecond();
    // Alloc data for GPU.
    data_alloc();    
    double iTime3 = cpuSecond();
    
    // Send data from CPU to GPU.
    data_h2d();
    double iTime4 = cpuSecond();
    // printf("h_zyx[1]=%d, h_zyx[2]=%d, ", h_zyx[1], h_zyx[2]);

    // read_and_sort(argc, argv, infile);
    // pre_order_time = coordinate_order_time + data_order_time;
    // // printf("pre_order_time = %f\n", pre_order_time * 1000.);
    // printf("%f ", pre_order_time * 1000.);

    // read input data.
    read_input_data(infile);
    MallocTempArray();

    double iTime5 =cpuSecond();

    //create stream
    for(int i=0;i<stream_size;i++){
        cudaStreamCreate(&stream[i]);
    }

    // printf("h_GMaps.spec_dim=%d\n",h_GMaps.spec_dim);
    uint32_t channels = h_GMaps.spec_dim;
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    dim3 block(bDim);
    dim3 grid((h_GMaps.block_warp_num * h_zyx[1] - 1)/ (block.x / 32) + 1);
    // printf("grid.x=%d, block.x=%d, ", grid.x, block.x);
    // printf("grid.x=%d, block.x=%d, ", grid.x, block.x);
    // printf("h_zyx[1]=%d, h_zyx[2]=%d, data_shape=%d, data_spec=%d, sphere_radius=%f\n", h_zyx[1], h_zyx[2], h_GMaps.data_shape, h_GMaps.spec_dim,h_GMaps.sphere_radius);

    // Get start time.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    
    omp_set_num_threads(stream_size);
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        pre_order_data(i);
        HANDLE_ERROR(cudaMemcpyAsync((double*)((char*)d_data+i*pitch_d), h_data[i], sizeof(double)*data_shape, cudaMemcpyHostToDevice, stream[i]));
        HANDLE_ERROR(cudaMemcpyAsync((double*)((char*)d_datacube+i*pitch_r), h_datacube[i], sizeof(double)*num, cudaMemcpyHostToDevice, stream[i]));
        HANDLE_ERROR(cudaMemcpyAsync((double*)((char*)d_weightscube+i*pitch_r), h_weightscube[i], sizeof(double)*num, cudaMemcpyHostToDevice,stream[i]));
        hcgrid<<< grid, block, 0, stream[i] >>>(d_lons, d_lats, (double*)((char*)d_data+i*pitch_d), d_weights, d_xwcs, d_ywcs, (double*)((char*)d_datacube+i*pitch_r), (double*)((char*)d_weightscube+i*pitch_r), d_hpx_idx);
        data_d2h(i, i);
    }

    // for(int i = 0; i < channels; i++){
    //     int j = i % stream_size;
    //     HANDLE_ERROR(cudaMemcpyAsync((double*)((char*)d_data+i*pitch_d), h_data[i], sizeof(double)*data_shape, cudaMemcpyHostToDevice, stream[j]));
    //     HANDLE_ERROR(cudaMemcpyAsync((double*)((char*)d_datacube+i*pitch_r), h_datacube[i], sizeof(double)*num, cudaMemcpyHostToDevice, stream[j]));
    //     HANDLE_ERROR(cudaMemcpyAsync((double*)((char*)d_weightscube+i*pitch_r), h_weightscube[i], sizeof(double)*num, cudaMemcpyHostToDevice,stream[j]));
    //     hcgrid<<< grid, block, 0, stream[j] >>>(d_lons, d_lats, (double*)((char*)d_data+i*pitch_d), d_weights, d_xwcs, d_ywcs, (double*)((char*)d_datacube+i*pitch_r), (double*)((char*)d_weightscube+i*pitch_r), d_hpx_idx);
    // }
    // for(int i = 0; i < channels; i++){
    //     data_d2h(i % stream_size, i);
    // }

    // Get stop time.
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    // printf("stop!");
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    // printf("kernel elapsed time=%f, \n", elapsedTime);
    printf("%f ", elapsedTime);

    cudaDeviceSynchronize();
    
    //destroy stream
    for(int i=0;i<stream_size;i++){
        cudaStreamDestroy(stream[i]);
    }

    // Send data from GPU to CPU
    // data_d2h();

    // Write output FITS file
    write_output_map(outfile);

    // Write sorted input FITS file
    if (sortfile) {
        write_ordered_map(infile, sortfile);
    }

    // Release data
    // printf("before Release data\n");
    data_free();
 
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );
    HANDLE_ERROR( cudaDeviceReset() );

    double iTime6 = cpuSecond();
    double iElaps = ((iTime4 - iTime1) + (iTime6 - iTime5)) * 1000.;
    // printf("solving_gridding time=%f\n", iElaps);
    printf("%f\n", iElaps);
}

