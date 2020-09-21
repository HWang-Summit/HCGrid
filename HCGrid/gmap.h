// --------------------------------------------------------------------
//
// title                  :gmap.h
// description            :\
//                          set grid kernel & healpix lookup table;
//                          read input points;
//                          read output map;
//                          set wcs for output pixels;
//                          write output map.
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

#ifndef GMAP_H
#define GMAP_H

#include "healpix.h"

// Some parameters needed in device
class GMaps{
 public:
    uint32_t block_warp_num;
    uint32_t factor;
    uint32_t data_shape;    // Number of input points.
    uint32_t spec_dim;      // The spectral dimension of input points.
    uint32_t kernel_type;   // Grid kernel type:
    // 1.'GAUSS1D', params=(kernel_sigma,);
    // 2.'GAUSS2D', params=(kernel_sigma_maj, kernel_sigma_min, PA);
    // 3.'TAPERED_SINC', params=(kernel_sigma, param_a, param_b);
    double disc_size;
    double sphere_radius;   // Kernel sphere radius:
                            // determine which distance the kernel is computed for.
    bool bearing_needed;
};
extern __constant__ GMaps d_const_GMaps;
extern GMaps h_GMaps;

// Coordinates for each input point.
extern double *d_lons;
extern double *d_lats;
extern double *h_lons;
extern double *h_lats;

// Spectra and weight for each input point.
extern double *d_data;
extern double *d_weights;
extern double *h_data;
extern double *h_weights;

// Healpix index and input index for sorted input points.
extern uint64_t *d_hpx_idx;
extern uint32_t *d_start_ring;
extern uint64_t *h_hpx_idx;
extern uint32_t *h_start_ring;

// Header of output FITS file
extern char *h_header;

// Dimension for output pixels.
extern __constant__ uint32_t d_const_zyx[3];
extern uint32_t *h_zyx;

// Coordinates for each output pixel.
extern double *d_xwcs;
extern double *d_ywcs;
extern double *h_xwcs;
extern double *h_ywcs;

// Spectra and weight for each output pixel.
extern double *d_datacube;
extern double *d_weightscube;
extern double *h_datacube;
extern double *h_weightscube;

// Parameters used in grid kernel
extern __constant__ double d_const_kernel_params[3];
extern double *h_kernel_params;

extern double last_sphere_radius;  // Record last sphere radius
extern double last_hpxmaxres;  // Record last max healpix resolution

/* (healpix_index, input_index) key-value pair used for sorting input data. */
class HPX_IDX{
 public:
    uint64_t hpx;
    uint32_t inx;
    HPX_IDX(const uint64_t& _hpx = 0, const uint32_t& _inx = 0): hpx(_hpx), inx(_inx){};
    bool operator< (const HPX_IDX& A) const {
        return (hpx < A.hpx);
    };
};

/* Set grid kernel & healpix lookup table. */
void _prepare_grid_kernel(uint32_t kernel_type, double *kernel_params, double sphere_radius, double hpx_max_res);

/* Read input points. */
void read_input_map(const char *infile);

/* Read output map. */
void read_output_map(const char *infile);

/* Set wcs for output pixels. */
int set_WCS();

/* Write output map. */
void write_output_map(const char *infile);

/* Write ordered input map. */
void write_ordered_map(const char *infile, const char *outfile);

/* Print out cfitsio error messages */
static void printfitserror (int status);

#endif // GMAP_H
