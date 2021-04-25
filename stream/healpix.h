// --------------------------------------------------------------------
//
// title                  :healpix.h
// description            :Healpix helper class.
// author                 :
//
// --------------------------------------------------------------------

#ifndef HEALPIX_H
#define HEALPIX_H

#include "helpers.h"

struct Healpix{
    uint64_t _nside, _order, _nrings, usedrings, firstring;
    uint64_t _max_npix_per_ring;
    uint64_t _npface, _ncap, _npix;
    double _fact1, _fact2, _resolution, _omega;
    uint32_t _scheme;
    bool _params_dirty;
};
extern __constant__ Healpix d_const_Healpix;
extern Healpix h_Healpix;

/* Initialize Healpix parameters based on nside and scheme. */
void _Healpix_init(uint64_t nside, uint32_t scheme);

/* Update HEALPix parameters if necessary (_params_dirty is True). */
void _update_params();

/* Return start index and number of pixels per healpix ring. */
__device__ void d_get_ring_info_small(uint64_t ring, uint64_t &startpix, uint64_t &num_pix_in_ring, bool &shifted);
void h_get_ring_info_small(uint64_t ring, uint64_t &startpix, uint64_t &num_pix_in_ring, bool &shifted);

/* Return ring index of hpx pixel. */
__device__ uint64_t d_pix2ring(uint64_t pix);
uint64_t h_pix2ring(uint64_t pix);

/* Convert location (z, phi) to HEALPix Index. */
__device__ uint64_t d_loc2pix(double z, double phi, double sin_theta, bool have_sin_theta);
uint64_t h_loc2pix(double z, double phi, double sin_theta, bool have_sin_theta);

/* Return the pixel index containing the angular coordinates (phi, theta). */
__device__ uint64_t d_ang2pix(double theta, double phi);//MAX_Y
uint64_t h_ang2pix(double theta, double phi);//MAX_Y

/* Convert HEALPix Index to location (z, phi). */
__device__ void d_pix2loc(uint64_t pix, double &z, double &phi, double &sin_theta, bool &have_sin_theta);
void h_pix2loc(uint64_t pix, double &z, double &phi, double &sin_theta, bool &have_sin_theta);

/* Convert HEALPix Index to angular coordinates (phi, theta). */
__device__ void d_pix2ang(uint64_t pix, double &theta, double &phi);
void h_pix2ang(uint64_t pix, double &theta, double &phi);

#endif  // HEALPIX_H
