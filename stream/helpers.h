// --------------------------------------------------------------------
//
// title                  :helpers.h
// description            :Helper functions for Gridding.
// author                 :
//
// --------------------------------------------------------------------

#ifndef HELPERS_H
#define HELPERS_H

#include "constants.h"

#if defined (__GNUC__)
#define UTIL_FUNC_NAME__ __func__
#else
#define UTIL_FUNC_NAME__ "unknown"
#endif
#define UTIL_ASSERT(cond,msg) if(!(cond)) util_fail_(__FILE__,__LINE__,UTIL_FUNC_NAME__,msg)
#define UTIL_FAIL(msg) util_fail_(__FILE__,__LINE__,UTIL_FUNC_NAME__,msg)
#define RALLOC(type,num) ((type *)util_malloc_((num)*sizeof(type)))
#define DEALLOC(ptr) do { util_free_(ptr); (ptr)=NULL; } while(0)
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/* Print error message. */
static void util_fail_(const char *file, int line, const char *func, const char *msg){
    fprintf(stderr, "%s, %i (%s):\n%s\n", file, line, func, msg);
    exit(1);
}

/* Malloc memory for pointer. */
static void *util_malloc_ (size_t sz){    
    if (sz==0)
        return NULL;
    void *res = malloc(sz);
    UTIL_ASSERT(res, "malloc() failed");
    return res;
}

/* Free memory for pointer. */
static void util_free_ (void *ptr){
    if ((ptr)!=NULL)
        free(ptr);
}

/* Print cuda error message. */
static void HandleError(cudaError_t err, const char *file, int line){
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit( EXIT_FAILURE );
    }
}

/* Compute integer base-2 logarithm. */
__host__ __device__ uint64_t ilog2(uint64_t arg);

/* Compute integer square root. */
__host__ __device__ uint64_t isqrt(uint64_t arg);

/* Integer modulo for C. */
//__host__ __device__ int64_t imod(int64_t a, int64_t b);

/* Compute remainder of division v1 / v2. */
__host__ __device__ double fmodulo (double v1, double v2);

__device__ double radlat2lon(double lat1, double lat2, double dis);

/* Calculate true angular distance between two points on the sphere. */
__host__ __device__ double true_angular_distance(double l1, double b1, double l2, double b2);

/* Calculate great circle bearing of (l2, b2) w.r.t. (l1, b1). */
__device__ double great_circle_bearing(double l1, double b1, double l2, double b2);

/* Set the HEALPix nside such that HPX resolution is less than target_res. */
uint64_t set_optimal_nside(double target_res);

/* Compute HEALPix order from nside. */
__host__ __device__ uint64_t nside_to_order(uint64_t nside);//MAX_Y

/* Compute HEALPix nside from npix. */
//__host__ __device__ uint64_t npix_to_nside(uint64_t npix);//MAX_Y

/* Calculate current CPU time. */
double cpuSecond();

#endif  // HELPERS_H
