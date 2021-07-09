# --------------------------------------------------------------------
#
# title                  :Visualize.py
# description            :Visualize output map
# author                 :Hao Wang, Jian Xiao, Ce Yu
#
# --------------------------------------------------------------------

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from astropy.io import fits
from astropy import wcs
import numpy as np
import os
import sys, getopt
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style

# get input / output fits file from arguments
path = ""
ifile = ""
ofile = ""
num = ""
try:
	opts, args = getopt.getopt(sys.argv[1:], 'hp:i:o:n:', ['path=', 'input=', 'output=', 'num='])
except getopt.GetoptError:
	print('python Visualize.py -p <absolute path> -i <inputfile> -o <outputfile> -n <number>')
	print('For example\n\tinput file: /home/summit/testfits/input1.fits\n\toutput file: /home/summit/testfits/out1.fits\n\tCommand: python Visualize.py -p /home/summit/testfits/ -i input -o out -n 1')
	sys.exit()
for opt, arg in opts:
	if opt in ["-h", "--help"]:
		print("usage: python Visualize.py -p <absolute path> -i <inputfile> -o <outputfile> -n <number>'")
		print('For example\n\tinput file: /home/summit/testfits/input1.fits\n\toutput file: /home/summit/testfits/out1.fits\n\tCommand: python Visualize.py -p /home/summit/testfits/ -i input -o out -n 1')
		sys.exit()
	elif opt in ["-p", "--path"]:
		path = arg
	elif opt in ["-i", "--input"]:
		ifile = arg
	elif opt in ["-o", "--output"]:
		ofile = arg
	elif opt in ["-n", "--num"]:
		num = arg
print("input file is ", path+ifile+num+'.fits')
print("output file is ", path+ofile+num+'.fits')
infile = os.path.join(wcs.__path__[0], path+ifile+num+'.fits')
outfile = os.path.join(wcs.__path__[0], path+ofile+num+'.fits')

# get input data
hdulist = fits.open(infile)
in_data = hdulist[0].data
in_max = np.max(in_data)
in_min = np.min(in_data)
in_cabs = np.max(np.abs(in_data))
xcoords, ycoords = hdulist[1].data
print(in_data)
print(xcoords)
print(ycoords)
hdulist.close()

# show input image
plt.style.use(astropy_mpl_style)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
gci = ax.scatter(xcoords, ycoords, c=np.squeeze(in_data), cmap='jet', edgecolor='none', vmin=in_min, vmax=in_max)
cbar = plt.colorbar(gci)
ax.set_xlabel('DEC [deg]', fontsize=15)
ax.set_ylabel('RA [deg]', fontsize=15)
ax.set_xlim(np.amax(xcoords), np.amin(xcoords))
ax.set_ylim(np.amax(ycoords), np.amin(ycoords))
##plt.show()
fig.savefig(path+'input'+num+'.png')

# get output data
hdulist = fits.open(outfile)
out_header = hdulist[0].header
out_data = hdulist[0].data
out_data = np.nan_to_num(out_data)
out_data = np.reshape(out_data, (out_header['NAXIS1'], out_header['NAXIS2']))
out_min = np.min(out_data)
out_max = np.max(out_data)
out_cabs = np.max(np.abs(out_data))
out_wcs = wcs.WCS(header=out_header, naxis=[wcs.WCSSUB_CELESTIAL])
hdulist.close()

# show output image
plt.style.use(astropy_mpl_style)
fig = plt.figure(figsize=(20, 20))
# ax = fig.add_subplot(111, projection=out_wcs.celestial)
ax = fig.add_subplot(projection=out_wcs.celestial)
im = ax.imshow(out_data, cmap='viridis', vmin=out_min, vmax=out_max)

lon, lat = ax.coords
lon.set_major_formatter('dd:mm')
lat.set_major_formatter('dd:mm')
lon.set_axislabel('R.A. [deg]')
lat.set_axislabel('Dec [deg]')

ax = plt.gca()
# ax.invert_xaxis()  #x轴反向
cbarr = plt.colorbar(im)
plt.show()
fig.savefig(path+'output'+num+'.png')

