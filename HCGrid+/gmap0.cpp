// --------------------------------------------------------------------
//
// title                  :gmap.cpp
// description            :\
//                          set grid kernel & healpix;
//                          read input points;
//                          read output map;
//                          set wcs for output pixels.
//                          write output map.
//                          write reordered map.
// author                 :
//
// --------------------------------------------------------------------

#include <fitsio.h>
#include <wcslib/wcslib.h>
#include "gmap.h"

double *h_lons;
double *h_lats;
double *h_data;
double *h_weights;
uint64_t *h_hpx_idx;
uint32_t *h_start_ring;
char *h_header;
uint32_t *h_zyx;
double *h_xwcs;
double *h_ywcs;
double *h_datacube;
double *h_weightscube;
double *h_kernel_params;
double last_sphere_radius = -1;
double last_hpxmaxres = -1;
GMaps h_GMaps;

/* Set grid kernel & healpix lookup table. */
void _prepare_grid_kernel(uint32_t kernel_type, double *kernel_params, double sphere_radius, double hpx_max_res){
    uint64_t num_params;    // Number of parameters

    // Set kernel type & parameters
    if(kernel_type == GAUSS1D){// params = ('0.5 / kernel_sigma ** 2',)
        num_params = 1;
        kernel_params[0] = pow(0.5 / kernel_params[0], 2.);
        h_GMaps.bearing_needed = false;
    }
    else if(kernel_type == GAUSS2D){// params = ('kernel_sigma_maj', 'kernel_sigma_min', 'PA')
        num_params = 3;
        h_GMaps.bearing_needed = true;
    }
    else if(kernel_type == TAPERED_SINC){// params = ('kernel_sigma', 'param_a', 'param_b')
        num_params = 3;
        h_GMaps.bearing_needed = false;
    }

    if (kernel_params+num_params-1 == NULL)
        UTIL_FAIL("Number Kernel Paramters Error.");

    h_GMaps.kernel_type = kernel_type;
    h_kernel_params = kernel_params;

    // Recompute healpix lookup table in case kernel sphere has changed
    if(fabs(last_sphere_radius-sphere_radius) > 3e-5 || fabs(last_hpxmaxres-hpx_max_res) > 3e-5){
        uint64_t nside = set_optimal_nside(D2R * hpx_max_res);
        _Healpix_init(nside, RING);
        last_sphere_radius = sphere_radius;
        last_hpxmaxres = hpx_max_res;
//        printf("* Recompute healpix lookup table success!\n");
    }

    h_GMaps.sphere_radius = sphere_radius;
    h_GMaps.disc_size = (D2R * sphere_radius + h_Healpix._resolution);
}

/* Read input points. */
void read_input_map(const char *infile){
    long naxes, *naxis, z, y, x;
    int status=0, hdutype, nfound, anynul;
    double nulval;
    fitsfile *fptr;

    // Open FITS file
    fits_open_file(&fptr, infile, READONLY, &status);
    printfitserror(status);

    // Read the dimension
    fits_read_key_lng(fptr, "NAXIS", &naxes, NULL, &status);
    printfitserror(status);
    naxis = RALLOC(long, naxes);
    fits_read_keys_lng(fptr, "NAXIS", 1, naxes, naxis, &nfound, &status);
    printfitserror(status);
    UTIL_ASSERT(nfound==naxes, "nfound!=naxes");
    y = (uint32_t) naxis[0];
    x = (uint32_t) naxis[1];
    z = (uint32_t) (naxes == 3) ? naxis[2] : 1;
    h_GMaps.data_shape = y * x;
    h_GMaps.spec_dim = z;

    // Read the data
    h_data = RALLOC(double, y*x*z);
    fits_read_img(fptr, TDOUBLE, 1, y*x*z, &nulval, h_data, &anynul, &status);
    printfitserror(status);

    // Read the coordinates
    fits_movabs_hdu(fptr, 2, &hdutype, &status);
    h_lons = RALLOC(double, y*x);
    fits_read_img(fptr, TDOUBLE, 1, y*x, &nulval, h_lons, &anynul, &status);
    printfitserror(status);
    h_lats = RALLOC(double, y*x);
    fits_read_img(fptr, TDOUBLE, y*x+1, y*x, &nulval, h_lats, &anynul, &status);
    printfitserror(status);

    // Intial the weight
    h_weights = RALLOC(double, z*y*x);
    for(int i=0; i<z*y*x; ++i)
        h_weights[i] = 1.;

    // Release
    DEALLOC(naxis);
    fits_close_file(fptr, &status);
    printfitserror(status);
}

/* 3. Read output map. */
void read_output_map(const char *infile){
    long naxes, *naxis;
    int status=0, nkeyrec, nfound;
    fitsfile *fptr;

    // Open FITS file
    fits_open_file(&fptr, infile, READONLY, &status);
    printfitserror(status);

    // Read FITS header
    fits_hdr2str(fptr, 0, NULL, 0, &h_header, &nkeyrec, &status);
    printfitserror(status);

    // Read the dimension
    fits_read_key_lng(fptr, "NAXIS", &naxes, NULL, &status);
    printfitserror(status);
    naxis = RALLOC(long, naxes);
    fits_read_keys_lng(fptr, "NAXIS", 1, naxes, naxis, &nfound, &status);
    printfitserror(status);
    UTIL_ASSERT(nfound==naxes, "nfound!=naxes");
    h_zyx = RALLOC(uint32_t, naxes);
    for(int i=0; i<naxes; ++i)
        h_zyx[i] = (uint32_t) naxis[naxes-i-1];
    h_GMaps.block_warp_num = (h_zyx[1] - 1) / (32 * h_GMaps.factor) + 1;

    // Release
    DEALLOC(naxis);
    fits_close_file(fptr, &status);
    printfitserror(status);
}

/* Set wcs for output pixels. */
int set_WCS(){
    int naxes = 3;
    int status=0, nkeyrec, nreject, nwcs, tempInd;
    struct wcsprm *wcs;
    int *stat;
    double *pixcrd, *imgcrd, *phi, *theta, *world;

    // Get the dimension
    uint32_t z = h_zyx[0];
    uint32_t y = h_zyx[1];
    uint32_t x = h_zyx[2];
    uint64_t ncoords = x * y;

    // Parse the primary header of the FITS file
    nkeyrec = strlen(h_header) / 80;
    if((status = wcspih(h_header, nkeyrec, WCSHDR_all, 2, &nreject, &nwcs, &wcs))){
        fprintf(stderr, "wcspih ERROR %d: %s.\n", status, wcshdr_errmsg[status]);
    }

    // Initialize the wcsprm struct
    if ((status = wcsset(wcs))) {
        fprintf(stderr, "wcsset ERROR %d: %s.\n", status, wcs_errmsg[status]);
        return 0;
    }

    // Set target map pixels.
    pixcrd = RALLOC(double, ncoords*naxes);
    tempInd = 0;
    for(int yy=0; yy<y; ++yy){
        for(int xx=0; xx<x; ++xx){
            pixcrd[tempInd] = xx+1;
            pixcrd[tempInd+1] = yy+1;
            pixcrd[tempInd+2] = z;
            tempInd += naxes;
        }
    }

    // Transfrom target pixels to sky coordinates
    imgcrd = RALLOC(double, ncoords*naxes);
    world = RALLOC(double, ncoords*naxes);
    phi = RALLOC(double, ncoords);
    theta = RALLOC(double, ncoords);
    stat = RALLOC(int, ncoords);
    h_xwcs = RALLOC(double, ncoords);
    h_ywcs = RALLOC(double, ncoords);
                         
    if ((status = wcsp2s(wcs, ncoords, naxes, pixcrd, imgcrd, phi, theta, world, stat))){
        fprintf(stderr, "\n\nwcsp2s ERROR %d: %s.\n", status, wcs_errmsg[status]);
    }

    tempInd = 0;
    for(int i=0; i<x*y; ++i) {
        h_xwcs[i] = world[tempInd];
        h_ywcs[i] = world[tempInd + 1];
        tempInd += naxes;
    }

    // Release
    DEALLOC(pixcrd);
    DEALLOC(imgcrd);
    DEALLOC(phi);
    DEALLOC(theta);
    DEALLOC(stat);
    DEALLOC(world);
    status = wcsvfree(&nwcs, &wcs);

    return 1;
}

/* Write output map. */
void write_output_map(const char *infile){
    fitsfile *fptr;
    int naxis, ncoords, status = 0, nkeyrec, i, j, k, first;
    long *naxes;
    bool end;
    char keyrec[81];

    // Create output FITS file, delete any pre-existing file
    fits_create_file(&fptr, infile, &status);
    printfitserror(status);

    // Get dimension
    naxis = 3;
    naxes = RALLOC(long, naxis);
    naxes[0] = h_zyx[1];
    naxes[1] = h_zyx[2];
    naxes[2] = h_zyx[0];
    ncoords = naxes[0] * naxes[1];

    // Explicitly create new image
    fits_create_img(fptr, DOUBLE_IMG, naxis, naxes, &status);
    printfitserror(status);

    // Convert header keyrecords to FITS.
    nkeyrec = 0;
    end = false;
    k = 80 * 6;
    while(true){
        strncpy(keyrec, h_header + k, 80);
        k += 80;
        ++nkeyrec;

        // An END keyrecord was read, break while.
        if (strncmp(keyrec, "END       ", 10) == 0)
            end = true;

        // Ignore meta-comments (copyright information, etc.)
        if (keyrec[0] == '#')
            continue;

        // Strip off the newline
        i = strlen(keyrec) - 1;
        if (keyrec[i] == '\n')
            keyrec[i] = '\0';
                  
        fits_write_record(fptr, keyrec, &status);
        printfitserror(status);
                                    
        if(end)
            break;
    }

    // Get weighted data cube
    i = 0;
    for(k=0; k < naxes[0]; ++k){
        for(j=0; j < naxes[1]; ++j){
            h_datacube[i] /= h_weightscube[i];
            ++i;
        }
    }

    // Write image data to the output FITS file
    first = 1;
    fits_write_img(fptr, TDOUBLE, first, ncoords, h_datacube, &status);
    printfitserror(status);

    // Release
    DEALLOC(naxes);
    fits_close_file(fptr, &status);
    printfitserror(status);
}

/* Write ordered input map. */
void write_ordered_map(const char *infile, const char *outfile) {
    // Open original FITS file
    fitsfile *fptr;
    int status;
    fits_open_file(&fptr, infile, READONLY, &status);
    printfitserror(status);

    // Read the dimension
    long naxis;
    fits_read_key_lng(fptr, "NAXIS", &naxis, NULL, &status);
    printfitserror(status);
    long *naxes = RALLOC(long, naxis);
    int nfound;
    fits_read_keys_lng(fptr, "NAXIS", 1, naxis, naxes, &nfound, &status);
    printfitserror(status);
    UTIL_ASSERT(nfound==naxis, "nfound!=naxis");
    long y = (uint32_t) naxes[0];
    long x = (uint32_t) naxes[1];
    long z = (uint32_t) (naxis == 3) ? naxes[2] : 1;

    // Create ordered input FITS file, delete any pre-existing file
    fitsfile *fptr1;
    fits_create_file(&fptr1, outfile, &status);
    printfitserror(status);

    // Copy primary header from original to ordered
    fits_copy_header(fptr, fptr1, &status);
    printfitserror(status);

    // Write image data for ordered
    fits_write_img(fptr1, TDOUBLE, 1, y*x*z, h_data, &status);
    printfitserror(status);

    // Copy secondary header from original to ordered
    fits_movabs_hdu(fptr, 2, NULL, &status);
    printfitserror(status);
    fits_copy_header(fptr, fptr1, &status);
    printfitserror(status);

    // Write image coordinates for ordered
    fits_write_col(fptr1, TDOUBLE, 1, 1, 1, y*x, h_lons, &status);
    printfitserror(status);
    fits_write_col(fptr1, TDOUBLE, 2, 1, y*x+1, y*x, h_lats, &status);
    printfitserror(status);

    // Close FITS files
    fits_close_file(fptr, &status);
    printfitserror(status);
    fits_close_file(fptr1, &status);
    printfitserror(status);

    // Release
    DEALLOC(naxes);
}

/* Print out cfitsio error messages */
static void printfitserror (int status){
    if (status==0)
        return;
    fits_report_error(stderr, status);  // print error report
    UTIL_FAIL("FITS error");
}
