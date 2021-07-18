// --------------------------------------------------------------------
//
// title                  :HCGrid.cpp
// description            :Grid data points to map
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

#include "HCGrid.h"

int main(int argc, char **argv){
    // Get FITS files from command
    char *path = NULL, *ifile = NULL, *tfile = NULL, *ofile = NULL, *sfile = NULL, *num = NULL, *beam = NULL, *order = NULL, *bDim = NULL, *factor = NULL;
    char pcl;
    int option_index = 0;
    static const struct option long_options[] = {
        {"helparg", no_argument, NULL, 'h'},
        {"fits_path", required_argument, NULL, 'p'},            // absolute path of FITS file
        {"input_file", required_argument, NULL, 'i'},           // name of unsorted input FITS file (it will call sort function)
        {"target_file", required_argument, NULL, 't'},          // name of target FITS file
        {"output_file", required_argument, NULL, 'o'},          // name of output FITS file
        {"sorted_file", required_argument, NULL, 's'},          // name of sorted input FITS file (it won't call sort function)
        {"fits_id", required_argument, NULL, 'n'},              // ID of FITS file
        {"beam_size", required_argument, NULL, 'b'},            // beam size of FITS file
        {"order_arg", required_argument, NULL, 'd'},            // sort parameter
        {"block_num", required_argument, NULL, 'a'},            // the number of thread in each block
        {"coarsening_factor", required_argument, NULL, 'f'},    // the value of coarsening factor
        {0, 0, 0, 0}
    };

    while((pcl = getopt_long_only (argc, argv, "hp:i:t:o:s:n:b:d:a:", long_options, \
                    &option_index)) != EOF){
        switch(pcl){
            case 'h':
                fprintf(stderr, "useage: ./HCGrid --fits _path <absolute path> --input_file <input file> --target_file <target file> "
                "--sorted_file <sorted file> --output_file <output file>--fits_id <number> --beam_size <beam> --order_arg <order> --block_num <num>\n");
                return 1;
            case 'p':
                path = optarg;
                break;
            case 'i':
                ifile = optarg;
                break;
            case 't':
                tfile = optarg;
                break;
            case 'o':
                ofile = optarg;
                break;
            case 's':
                sfile = optarg;
                break;
            case 'n':
                num = optarg;
                break;
            case 'b':
                beam = optarg;
                break;
            case 'd':
                order = optarg;
                break;
            case 'a':
                bDim = optarg;
                break;
            case 'f':
                factor = optarg;
                break;
            case '?':
                fprintf (stderr, "Unknown option `-%c'.\n", (char)optopt);
                break;
            default:
                return 1;
        }
    }

    char infile[180] = "", tarfile[180] = "", outfile[180] = "!", sortfile[180] = "!";
    strcat(infile, path);
    strcat(infile, ifile);
    strcat(infile, num);
    // strcat(infile, ".fits");
    strcat(infile, ".hdf5");
    strcat(tarfile, path);
    strcat(tarfile, tfile);
    strcat(tarfile, num);
    strcat(tarfile, ".fits");
    strcat(outfile, path);
    strcat(outfile, ofile);
    strcat(outfile, num);
    strcat(outfile, ".fits");
    if (sfile) {
        strcat(sortfile, path);
        strcat(sortfile, sfile);
        strcat(sortfile, num);
        strcat(sortfile, ".fits");
    }
    // printf("order: %s, num: %s, ", order, num);

    // Initialize healpix
    _Healpix_init(1, RING);

    // Set kernel
    uint32_t kernel_type = GAUSS1D;
    double kernelsize_fwhm = 300. / 3600.;
    if (beam) {
        double kernelsize_fwhm = atoi(beam) / 3600.;
    }
    double kernelsize_sigma = kernelsize_fwhm / sqrt(8*log(2));
    double *kernel_params;
    kernel_params = RALLOC(double, 3);
    kernel_params[0] = kernelsize_sigma;
    double sphere_radius = 5. * kernelsize_sigma;
    double hpx_max_resolution = kernelsize_sigma / 2.;
    _prepare_grid_kernel(kernel_type, kernel_params, sphere_radius, hpx_max_resolution);

    // Gridding process
    h_GMaps.factor = 1;
    if (factor) {
        h_GMaps.factor = atoi(factor);
    }
    // printf("h_GMaps.factor=%d, \n", h_GMaps.factor);
    if (sfile) {
        if (bDim)
            solve_gridding(infile, tarfile, outfile, sortfile, atoi(order), atoi(bDim), argc, argv);
        else
            solve_gridding(infile, tarfile, outfile, sortfile, atoi(order), 96, argc, argv);
    } else {
        
        if (bDim)
            solve_gridding(infile, tarfile, outfile, NULL, atoi(order), atoi(bDim), argc, argv);
        else
            solve_gridding(infile, tarfile, outfile, NULL, atoi(order), 96, argc, argv);
    }

    return 0;
}
