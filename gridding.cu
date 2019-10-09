// --------------------------------------------------------------------
//
// title                  :gridding.cu
// description            :Sort and Gridding process.
// author                 :
//
// --------------------------------------------------------------------

#include <boost/sort/sort.hpp>
#include <thrust/sort.h>
#include "gridding.h"
using boost::sort::block_indirect_sort;

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
double *d_datacube;
double *d_weightscube;
__constant__ double d_const_kernel_params[3];
__constant__ GMaps d_const_GMaps;

/* Print a array pair. */
void print_double_array(double *A, double *B, uint32_t num){
    printf("Array (A, B) = <<<\n");
    for(int i=0; i<10; ++i){
        printf("(%f, %f), ", A[i], B[i]);
    }
    printf("\n..., \n");
    for(int i=num-11; i<num; ++i){
        printf("(%f, %f), ", A[i], B[i]);
    }
    printf("\n>>>\n\n");
}

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
    h_hpx_idx = RALLOC(uint64_t, data_shape + 1);
    for(int i=0; i < data_shape; ++i){
        h_hpx_idx[i] = V[i].hpx;
    }
    h_hpx_idx[data_shape] = h_Healpix._npix;
    double *tempArray = RALLOC(double, data_shape);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lons[V[i].inx];
    }
    swap(h_lons, tempArray);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lats[V[i].inx];
    }
    swap(h_lats, tempArray);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_data[V[i].inx];
    }
    swap(h_data, tempArray);
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
    vector<HPX_IDX> vtTemp;
    vtTemp.swap(V);

    // Get runtime
    double iTime6 = cpuSecond();
    printf("%f, ", (iTime6 - iTime1) * 1000.);
//    printf("%f, %f, %f\n", (iTime3 - iTime2) * 1000., (iTime5 - iTime4) * 1000., (iTime6 - iTime1) * 1000.);
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
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_data[in_inx[i]];
    }
    swap(h_data, tempArray);
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
    printf("%f, ", (iTime6 - iTime1) * 1000.);
//    printf("%f, %f, %f\n", (iTime3 - iTime2) * 1000., (iTime5 - iTime4) * 1000., (iTime6 - iTime1) * 1000.);
}

/* Initialize output spectrals and weights. */
void init_output(){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    h_datacube = RALLOC(double, num);
    h_weightscube = RALLOC(double, num);
    for(uint32_t i = 0; i < num; ++i){
        h_datacube[i] = 0.;
        h_weightscube[i] = 0.;
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
    uint32_t tid = ((warp_id % d_const_GMaps.block_warp_num) * 32 + threadIdx.x % 32) * d_const_GMaps.factor;
    if (tid < d_const_zyx[1]) {
        uint32_t left = tid;
        uint32_t right = left + d_const_GMaps.factor - 1;
        if (right >= d_const_zyx[1]) {
            right = d_const_zyx[1] - 1;
        }
        tid = (warp_id / d_const_GMaps.block_warp_num) * d_const_zyx[1];
        left = left + tid;
        right = right + tid;
        double temp_weights[3], temp_data[3], l1[3], b1[3];
        for (tid = left; tid <= right; ++tid) {
            temp_weights[tid-left] = d_weightscube[tid];
            temp_data[tid-left] = d_datacube[tid];
            l1[tid-left] = d_xwcs[tid] * DEG2RAD;
            b1[tid-left] = d_ywcs[tid] * DEG2RAD;
        }

        // get northeast ring and southeast ring
        uint64_t upix = d_ang2pix(HALFPI-b1[0], l1[0]);
        double disc_theta, disc_phi;
        d_pix2ang(upix, disc_theta, disc_phi);
        double utheta = disc_theta - d_const_GMaps.disc_size;
        upix = d_ang2pix(utheta, disc_phi);
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
            upix = d_ang2pix(HALFPI-b1[0], l1[0]);
            d_pix2ang(upix, disc_theta, disc_phi);
            uphi = disc_phi - d_const_GMaps.disc_size;
            uint64_t lpix = d_ang2pix(utheta, uphi);
            if (!(lpix >= startpix && lpix < startpix+num_pix_in_ring)) {
                start_int = end_int;
                continue;
            }
            uphi = disc_phi + d_const_GMaps.disc_size;
            uint64_t rpix = d_ang2pix(utheta, uphi);
            if (!(rpix >= startpix && rpix < startpix+num_pix_in_ring)) {
                start_int = end_int;
                continue;
            }

            // find position of lpix
            uint32_t upix_idx = searchLastPosLessThan(d_hpx_idx, start_int, end_int, lpix);
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

                for (tid = left; tid <= right; ++tid) {
                    double sdist = true_angular_distance(l1[tid-left], b1[tid-left], l2, b2) * RAD2DEG;
                    double sbear = 0.;
                    if (d_const_GMaps.bearing_needed) {
                        sbear = great_circle_bearing(l1[tid-left], b1[tid-left], l2, b2);
                    }
                    if (sdist < d_const_GMaps.sphere_radius) {
                        double sweight = kernel_func_ptr(sdist, sbear);
                        double tweight = in_weights * sweight;
                        temp_data[tid-left] += in_data * tweight;
                        temp_weights[tid-left] += tweight;
                    }
                }
                ++upix_idx;
            }

            start_int = end_int;
            ++uring;
        }
        for (tid = left; tid <= right; ++tid) {
            d_datacube[tid] = temp_data[tid-left];
            d_weightscube[tid] = temp_weights[tid-left];
        }
    }
    __syncthreads();
}

/* Alloc data for GPU. */
void data_alloc(){
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t usedrings = h_Healpix.usedrings;

    HANDLE_ERROR(cudaMalloc((void**)& d_lons, sizeof(double)*data_shape));
    HANDLE_ERROR(cudaMalloc((void**)& d_lats, sizeof(double)*data_shape));
    HANDLE_ERROR(cudaMalloc((void**)& d_data, sizeof(double)*data_shape));
    HANDLE_ERROR(cudaMalloc((void**)& d_weights, sizeof(double)*data_shape));
    HANDLE_ERROR(cudaMalloc((void**)& d_xwcs, sizeof(double)*num));
    HANDLE_ERROR(cudaMalloc((void**)& d_ywcs, sizeof(double)*num));
    HANDLE_ERROR(cudaMalloc((void**)& d_datacube, sizeof(double)*num));
    HANDLE_ERROR(cudaMalloc((void**)& d_weightscube, sizeof(double)*num));
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
    HANDLE_ERROR(cudaMemcpy(d_data, h_data, sizeof(double)*data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weights, h_weights, sizeof(double)*data_shape, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_xwcs, h_xwcs, sizeof(double)*num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ywcs,h_ywcs, sizeof(double)*num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_datacube, h_datacube, sizeof(double)*num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weightscube, h_weightscube, sizeof(double)*num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_hpx_idx, h_hpx_idx, sizeof(uint64_t)*(data_shape+1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_start_ring, h_start_ring, sizeof(uint32_t)*(usedrings+1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_kernel_params, h_kernel_params, sizeof(double)*3));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_zyx, h_zyx, sizeof(uint32_t)*3));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_Healpix, &h_Healpix, sizeof(Healpix)));
    HANDLE_ERROR(cudaMemcpyToSymbol(d_const_GMaps, &h_GMaps, sizeof(GMaps)));
}

/* Send data from GPU to CPU. */
void data_d2h(){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    HANDLE_ERROR(cudaMemcpy(h_datacube, d_datacube, sizeof(double)*num, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_weightscube, d_weightscube, sizeof(double)*num, cudaMemcpyDeviceToHost));
}

/* Release data. */
void data_free(){
    DEALLOC(h_lons);
    HANDLE_ERROR( cudaFree(d_lons) );
    DEALLOC(h_lats);
    HANDLE_ERROR( cudaFree(d_lats) );
    DEALLOC(h_data);
    HANDLE_ERROR( cudaFree(d_data) );
    DEALLOC(h_weights);
    HANDLE_ERROR( cudaFree(d_weights) );
    DEALLOC(h_xwcs);
    HANDLE_ERROR( cudaFree(d_xwcs) );
    DEALLOC(h_ywcs);
    HANDLE_ERROR( cudaFree(d_ywcs) );
    DEALLOC(h_datacube);
    HANDLE_ERROR( cudaFree(d_datacube) );
    DEALLOC(h_weightscube);
    HANDLE_ERROR( cudaFree(d_weightscube) );
    DEALLOC(h_hpx_idx);
    HANDLE_ERROR( cudaFree(d_hpx_idx) );
    DEALLOC(h_start_ring);
    HANDLE_ERROR( cudaUnbindTexture(tex_start_ring) );

    HANDLE_ERROR( cudaFree(d_start_ring) );
    DEALLOC(h_header);
    DEALLOC(h_zyx);
    DEALLOC(h_kernel_params);
}

/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int& param, const int &bDim) {
    double iTime1 = cpuSecond();
    // Read input points.
    read_input_map(infile);

    // Read output map.
    read_output_map(tarfile);

    // Set wcs for output pixels.
    set_WCS();

    // Initialize output spectrals and weights.
    init_output();

//    iTime2 = cpuSecond();
    // Block Indirect Sort input points by their healpix indexes.
    if (param == THRUST) {
        init_input_with_thrust(param);
    } else {
        init_input_with_cpu(param);
    }

    double iTime3 = cpuSecond();
    // Alloc data for GPU.
    data_alloc();

    double iTime4 = cpuSecond();
    // Send data from CPU to GPU.
    data_h2d();
    printf("h_zyx[1]=%d, h_zyx[2]=%d, ", h_zyx[1], h_zyx[2]);

    // Set block and thread.
    dim3 block(bDim);
    dim3 grid((h_GMaps.block_warp_num * h_zyx[1] - 1) / (block.x / 32) + 1);
    printf("grid.x=%d, block.x=%d, ", grid.x, block.x);

    // Get start time.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Call device kernel.
    hcgrid<<< grid, block >>>(d_lons, d_lats, d_data, d_weights, d_xwcs, d_ywcs, d_datacube, d_weightscube, d_hpx_idx);

    // Get stop time.
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("kernel elapsed time=%f, ", elapsedTime);

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
    HANDLE_ERROR( cudaEventDestroy(start) );
    HANDLE_ERROR( cudaEventDestroy(stop) );
    HANDLE_ERROR( cudaDeviceReset() );

    double iTime5 = cpuSecond();
    double iElaps = (iTime5 - iTime1) * 1000.;
    printf("solving_gridding time=%f\n", iElaps);
}

