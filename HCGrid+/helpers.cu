// --------------------------------------------------------------------
//
// title                  :helpers.cu
// description            :Helper functions for Gridding.
// author                 :
//
// --------------------------------------------------------------------

#include <sys/time.h>
#include "helpers.h"

/* Compute integer base-2 logarithm. */
__host__ __device__ uint64_t ilog2(uint64_t arg){
	uint64_t r = 0;

	while(arg > 0x0000FFFF){
		r += 16;
		arg >>= 16;
	}
	if(arg > 0x000000FF){
		r |= 8;
		arg >>= 8;
	}
	if(arg > 0x0000000F){
		r |= 4;
		arg >>= 4;
	}
	if(arg > 0x00000003){
		r |= 2;
		arg >>= 2;
	}
	if(arg > 0x00000001){
		r |= 1;
		arg >>= 1;
	}
	return r;
}

/* Compute integer square root. */
__host__ __device__ uint64_t isqrt(uint64_t arg){
	return ((uint64_t) sqrt(arg + 0.5));
}

/* Integer modulo for C. */
/*
__host__ __device__ int64_t imod(int64_t a, int64_t b){
	int64_t r = a % b;
	if(r < 0)
		return r + b;
	else
		return r;
}
*/

/* Compute remainder of division v1 / v2. */
__host__ __device__ double fmodulo (double v1, double v2){
	if(v1 >= 0){
		if(v1 < v2)
			return v1;
		else
			return fmod(v1, v2);
	}
	double tmp = fmod(v1, v2) + v2;
	if(tmp == v2)
		return 0.;
	else
		return tmp;
}

/* Set the HEALPix nside such that HPX resolution is less than target_res. */
uint64_t set_optimal_nside(double target_res){
    uint64_t nside = (uint64_t)(sqrt(PI/3.) / target_res + 0.5);
    nside = 1 << (ilog2(nside) + 1);
    return nside;
}

/* Compute HEALPix order from nside. */
__host__ __device__ uint64_t nside_to_order(uint64_t nside){
	if(nside <= 0)
        return MAX_Y;
//		throw "Invalid value for nside (must be positive).";
	if(nside & (nside-1))
		return -1;
		// invalid value
	else
		return ilog2(nside);
}

/* Compute HEALPix nside from npix. */
/*
__host__ __device__ uint64_t npix_to_nside(uint64_t npix){
	uint64_t res = isqrt((uint64_t) (floor(npix / 12.0)));
	if (npix != res * res * 12)
        return MAX_Y;
//		throw "Invalid value for npix.";
	return res;
}
*/

/* Calculate true angular distance between two points on the sphere. */
__host__ __device__ double true_angular_distance(double l1, double b1, double l2, double b2){
    return 2 * asin(sqrt(pow(sin((b1 - b2) / 2.), 2) + cos(b1) * cos(b2) *  pow(sin((l1 - l2) / 2.), 2)));
}

/* Calculate great circle bearing of (l2, b2) w.r.t. (l1, b1). */
__device__ double great_circle_bearing(double l1, double b1, double l2, double b2){
	double l_diff_rad = l2 - l1;
	double cos_b2 = cos(b2);
	return atan2(cos_b2 * sin(l_diff_rad), cos(b1) * sin(b2) - sin(b1) * cos_b2 * cos(l_diff_rad));
}

__device__ double radlat2lon(double lat1, double lat2, double dis){
    lat1 *= RAD2DEG;
    lat2 *= RAD2DEG;
    double temp = (cos(dis)-sin(lat1)*sin(lat2))/(cos(lat1)*cos(lat2));
    temp = acos(temp);
    temp *= DEG2RAD;
    return temp;
}

/* Calculate current CPU time. */
double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
