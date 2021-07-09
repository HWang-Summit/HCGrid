# --------------------------------------------------------------------
#
# title                  :Create_input_file.py
# description            :Produce random data for test
# author                 :
#
# --------------------------------------------------------------------

# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.io import fits
# from astropy.coordinates import SkyCoord
from astropy import wcs
import numpy as np
import sys, getopt
import os
import math

# get input / target files path from arguments
path = ""
ifile = ""
tfile = ""
num = ""
num_samples = 0
beam_size = 0
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hp:i:t:n:s:b:',
                               ['path=', 'input=', 'target=', 'num=', 'sample=', 'beam='])
except getopt.GetoptError:
    print(
        'python Create_input_file.py -p <absolute path> -i <inputfile> -t <targetfile> -n <number> -s <number samples> -b <beam size>')
    print(
        'For example\n\tinput file: /home/summit/testfits/input1.fits\n\ttarget file: /home/summit/testfits/target1.fits\n\tCommand: python Create_input_file.py -p /home/summit/testfits/ -i input -t target -n 1 -s 10000 -b 300')
    sys.exit(2)
for opt, arg in opts:
    if opt in ["-h", "--help"]:
        print(
            "usage: python Create_input_file.py -p <absolute path> -i <inputfile> -t <targetfile> -n <number> -s <number samples> -b <beam size>")
        print(
            'For example\n\tinput file: /home/summit/testfits/input1.fits\n\ttarget file: /home/summit/testfits/target1.fits\n\tCommand: python Create_input_file.py -p /home/summit/testfits/ -i input -t target -n 1 -s 10000 -b 300')
        sys.exit()
    elif opt in ["-p", "--path"]:
        path = arg
    elif opt in ["-i", "--input"]:
        ifile = arg
    elif opt in ["-t", "--target"]:
        tfile = arg
    elif opt in ["-n", "--num"]:
        num = arg
    elif opt in ["-s", "--numsam"]:
        num_samples = int(arg)
        print(num_samples)
    elif opt in ["-b", "--beamarg"]:
        beam_size = int(arg)
        print(beam_size)

print("input file is ", path + ifile + num + '.fits')
print("target file is ", path + tfile + num + '.fits')
infile = os.path.join(wcs.__path__[0], path + ifile + num + '.fits')
targetfile = os.path.join(wcs.__path__[0], path + tfile + num + '.fits')


# generate input data and input coords
def setup_data(mapcenter, mapsize, beamsize_fwhm, num_samples, num_sources):
    lon_scale = np.cos(np.radians(mapcenter[1]))
    scale_size = 1.0
    map_l, map_r = (
        mapcenter[0] - scale_size * mapsize[0] / 2. / lon_scale,
        mapcenter[0] + scale_size * mapsize[0] / 2. / lon_scale
    )
    map_b, map_t = mapcenter[1] - scale_size * mapsize[1] / 2., mapcenter[1] + scale_size * mapsize[1] / 2.

    # coordinates are drawn from a uniform distribution
    xcoords = np.random.uniform(map_l, map_r, num_samples).astype(np.float32)
    ycoords = np.random.uniform(map_b, map_t, num_samples).astype(np.float32)

    # add Gaussian noise
    signal = np.random.normal(0., 1., len(xcoords)).astype(np.float32)

    beamsize_sigma = beamsize_fwhm / np.sqrt(8 * np.log(2))

    # put in artifical point source, with random amplitudes
    # we'll assume a Gaussian-shaped PSF

    def gauss2d(x, y, x0, y0, A, s):
        return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2. / s ** 2)

    for _ in range(num_sources):
        sou_x = np.random.uniform(map_l, map_r, 1).astype(np.float32)
        sou_y = np.random.uniform(map_b, map_t, 1).astype(np.float32)
        A = np.random.uniform(0, 100, 1).astype(np.float32)
        signal += gauss2d(xcoords, ycoords, sou_x, sou_y, A, beamsize_sigma)

    signal = signal[:, np.newaxis]  # need dummy spectral axis
    #print(signal)
    return xcoords, ycoords, signal


# set input data and input coords parameters
mapcenter = 60., 30.  # all in degrees
map_size = 5.
mapsize = map_size, map_size
beamsize_fwhm = 2 * beam_size / 3600.
# num_samples = 100000000
num_sources = 20

# get data and coords
xcoords, ycoords, signal = setup_data(
    mapcenter, mapsize, beamsize_fwhm, num_samples, num_sources
)

# produce a FITS header that contains the input coords
hdu = fits.ImageHDU(data=[xcoords, ycoords])

# write input fits file
fits.writeto(infile, data=signal, overwrite=True, checksum=True)
hdulist = fits.open(infile)
hdulist.insert(1, hdu)
hdulist.writeto(infile, overwrite=True, checksum=True)
hdulist.close()


# check input fits file
# print(fits.open(infile)[1].header)

# Produce a FITS header that contains the target field.
def setup_header(mapcenter, mapsize, beamsize_fwhm):
    # define target grid (via fits header according to WCS convention)
    # a good pixel size is a third of the FWHM of the PSF (avoids aliasing)
    pixsize = beamsize_fwhm / 3.
    dnaxis1 = int(mapsize[0] / pixsize)
    dnaxis2 = int(mapsize[1] / pixsize)
    print(pixsize, dnaxis1, dnaxis2)

    header = {
        'NAXIS': 3,
        'NAXIS1': dnaxis1,
        'NAXIS2': dnaxis2,
        'NAXIS3': 2,  # need dummy spectral axis
        'CTYPE1': 'RA---SIN',
        'CTYPE2': 'DEC--SIN',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CDELT1': -pixsize,
        'CDELT2': pixsize,
        'CRPIX1': dnaxis1 / 2.,
        'CRPIX2': dnaxis2 / 2.,
        'CRVAL1': mapcenter[0],
        'CRVAL2': mapcenter[1],
    }
    return header


# initialize target data
def setup_target_data(my_header, signal):
    yzx_shape = (my_header['NAXIS3'], my_header['NAXIS2'], my_header['NAXIS1'])
    datacube = np.zeros(yzx_shape, dtype=np.float32)
    index = 0
    for zz in range(yzx_shape[0]):
        index = 0
        for yy in range(yzx_shape[1]):
            for xx in range(yzx_shape[2]):
                datacube[zz, yy, xx] = signal[index]
                index += 1
    return datacube


# get target header
my_header = setup_header(mapcenter, mapsize, beamsize_fwhm)
my_wcs = wcs.WCS(my_header, naxis=[wcs.WCSSUB_CELESTIAL])
my_wcs = wcs.WCS(my_header)
target_header = my_wcs.to_header()

# get target data
datacube = setup_target_data(my_header, signal)
print(datacube[0])
print(datacube[1])
# write target fits file
fits.writeto(targetfile, data=datacube, header=target_header, overwrite=True, checksum=True)
