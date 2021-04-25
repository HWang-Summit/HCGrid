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
extern double **h_data;
extern double *h_weights;

// Healpix index and input index for sorted input points.
extern uint64_t *d_hpx_idx;
extern uint32_t *d_start_ring;
extern uint64_t *h_hpx_idx;
extern uint32_t *h_inx_idx;
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
extern double **d_datacube;
extern double **d_weightscube;
extern double **h_datacube;
extern double **h_weightscube;
extern double **tempArray;
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
// void read_input_map(const char *infile);
void read_input_map_hdf5(const char *infile);
void read_input_coordinate(const char *infile);
void read_input_data(const char *infile);
void read_and_sort(int argc, char **argv, const char *infile);

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
