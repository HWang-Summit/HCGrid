// --------------------------------------------------------------------
//
// title                  :healpix.cu
// description            :Healpix helper class.
// author                 :
//
// --------------------------------------------------------------------

#include "healpix.h"

__constant__ Healpix d_const_Healpix;
Healpix h_Healpix;

/* Initialize Healpix parameters based on nside and scheme. */
void _Healpix_init(uint64_t nside, uint32_t scheme){
    h_Healpix._nside = nside;
    if(h_Healpix._scheme == NESTED)
        UTIL_FAIL("Currently, only RING scheme is supported.");
    h_Healpix._scheme = scheme;
    h_Healpix._params_dirty = true;
    _update_params();
}

/* Update HEALPix parameters if necessary (_params_dirty is True). */
void _update_params(){
    if (h_Healpix._params_dirty){
		h_Healpix._order = nside_to_order(h_Healpix._nside);
		h_Healpix._nrings = 4 * h_Healpix._nside - 1;
		h_Healpix._max_npix_per_ring = 4 * h_Healpix._nside;
		h_Healpix._npface = h_Healpix._nside * h_Healpix._nside;
		h_Healpix._ncap = (h_Healpix._npface - h_Healpix._nside) << 1;
		h_Healpix._npix = 12 * h_Healpix._npface;
		h_Healpix._fact2 = 4. / h_Healpix._npix;
        h_Healpix._fact1 = (h_Healpix._nside << 1) * h_Healpix._fact2;
        h_Healpix._omega = 4. * PI / h_Healpix._npix;
        h_Healpix._resolution = sqrt(h_Healpix._omega);
	}
}

/* Return start index and number of pixels per healpix ring. */
__device__ void d_get_ring_info_small(uint64_t ring, uint64_t &startpix, uint64_t &num_pix_in_ring, bool &shifted){
	uint64_t nr;
    if(ring < d_const_Healpix._nside){
        shifted = true;
        num_pix_in_ring = 4 * ring;
        startpix = 2 * ring * (ring - 1);
    }
    else if(ring < 3 * d_const_Healpix._nside){
        shifted = (bool)(((ring - d_const_Healpix._nside) & 1) == 0);
        num_pix_in_ring = 4 * d_const_Healpix._nside;
        startpix = d_const_Healpix._ncap + (ring - d_const_Healpix._nside) * num_pix_in_ring;
    }
    else{
        shifted = true;
        nr = 4 * d_const_Healpix._nside - ring;
        num_pix_in_ring = 4 * nr;
        startpix = d_const_Healpix._npix - 2 * nr * (nr + 1);
    }
}
void h_get_ring_info_small(uint64_t ring, uint64_t &startpix, uint64_t &num_pix_in_ring, bool &shifted){
    uint64_t nr;
    if(ring < h_Healpix._nside){
        shifted = true;
        num_pix_in_ring = 4 * ring;
        startpix = 2 * ring * (ring - 1);
    }
    else if(ring < 3 * h_Healpix._nside){
        shifted = (bool)(((ring - h_Healpix._nside) & 1) == 0);
        num_pix_in_ring = 4 * h_Healpix._nside;
        startpix = h_Healpix._ncap + (ring - h_Healpix._nside) * num_pix_in_ring;
    }
    else{
        shifted = true;
        nr = 4 * h_Healpix._nside - ring;
        num_pix_in_ring = 4 * nr;
        startpix = h_Healpix._npix - 2 * nr * (nr + 1);
    }
}

/* Return ring index of hpx pixel. */
__device__ uint64_t d_pix2ring(uint64_t pix){
    if(pix < d_const_Healpix._ncap)
        return (1 + isqrt(1 + 2 * pix)) >> 1;
    else if(pix < d_const_Healpix._npix - d_const_Healpix._ncap)
        return (pix - d_const_Healpix._ncap) / (4 * d_const_Healpix._nside) + d_const_Healpix._nside;
    else
        return 4 * d_const_Healpix._nside - ((1 + (uint64_t)(isqrt(2 * (d_const_Healpix._npix - pix) - 1))) >> 1);
}
uint64_t h_pix2ring(uint64_t pix){
    if(pix < h_Healpix._ncap)
        return (1 + isqrt(1 + 2 * pix)) >> 1;
    else if(pix < h_Healpix._npix - h_Healpix._ncap)
        return (pix - h_Healpix._ncap) / (4 * h_Healpix._nside) + h_Healpix._nside;
    else
        return 4 * h_Healpix._nside - ((1 + (uint64_t)(isqrt(2 * (h_Healpix._npix - pix) - 1))) >> 1);
}

/* Convert location (z, phi) to HEALPix Index. */
__device__ uint64_t d_loc2pix(double z, double phi, double sin_theta, bool have_sin_theta){
    double temp1, temp2, tp, tmp;
    double za = fabs(z);
    double tt = fmodulo(phi * INV_HALFPI, 4.0);
    uint64_t nl4, jp, jm, ir, kshift, t1, ip;

	if(za <= TWOTHIRD){
        nl4 = 4 * d_const_Healpix._nside;
        temp1 = d_const_Healpix._nside * (0.5 + tt);
        temp2 = d_const_Healpix._nside * z * 0.75;
        jp = (uint64_t) (temp1 - temp2);
        jm = (uint64_t) (temp1 + temp2);
        ir = d_const_Healpix._nside + 1 + jp - jm;
        kshift = 1 - (ir & 1);
        t1 = jp + jm + kshift + 1 + nl4 + nl4 - d_const_Healpix._nside;
        if(d_const_Healpix._order > 0)
            ip = (t1 >> 1) & (nl4 - 1);
        else
            ip = (t1 >> 1) % nl4;

        return d_const_Healpix._ncap + (ir - 1) * nl4 + ip;
    }
	else{
		tp = tt - ((uint64_t) tt);
        if ((za < 0.99) || (!have_sin_theta))
            tmp = d_const_Healpix._nside * sqrt(3. * (1. - za));
        else
            tmp = d_const_Healpix._nside * sin_theta / sqrt((1. + za) / 3.);

        jp = (uint64_t) (tp * tmp);
        jm = (uint64_t) ((1.0 - tp) * tmp);
        ir = jp + jm + 1;
        ip = (uint64_t) (tt * ir);

        if(z > 0)
            return 2 * ir * (ir - 1) + ip;
        else
            return d_const_Healpix._npix - 2 * ir * (ir + 1) + ip;
	}
}
uint64_t h_loc2pix(double z, double phi, double sin_theta, bool have_sin_theta){
    double temp1, temp2, tp, tmp;
    double za = fabs(z);
    double tt = fmodulo(phi * INV_HALFPI, 4.0);
    uint64_t nl4, jp, jm, ir, kshift, t1, ip;

    if(za <= TWOTHIRD){
        nl4 = 4 * h_Healpix._nside;
        temp1 = h_Healpix._nside * (0.5 + tt);
        temp2 = h_Healpix._nside * z * 0.75;
        jp = (uint64_t) (temp1 - temp2);
        jm = (uint64_t) (temp1 + temp2);
        ir = h_Healpix._nside + 1 + jp - jm;
        kshift = 1 - (ir & 1);
        t1 = jp + jm + kshift + 1 + nl4 + nl4 - h_Healpix._nside;
        if(h_Healpix._order > 0)
            ip = (t1 >> 1) & (nl4 - 1);
        else
            ip = (t1 >> 1) % nl4;

        return h_Healpix._ncap + (ir - 1) * nl4 + ip;
    }
    else{
        tp = tt - ((uint64_t) tt);
        if ((za < 0.99) || (!have_sin_theta))
            tmp = h_Healpix._nside * sqrt(3. * (1. - za));
        else
            tmp = h_Healpix._nside * sin_theta / sqrt((1. + za) / 3.);

        jp = (uint64_t) (tp * tmp);
        jm = (uint64_t) ((1.0 - tp) * tmp);
        ir = jp + jm + 1;
        ip = (uint64_t) (tt * ir);

        if(z > 0)
            return 2 * ir * (ir - 1) + ip;
        else
            return h_Healpix._npix - 2 * ir * (ir + 1) + ip;
    }
}

/* Return the pixel index containing the angular coordinates (phi, theta). */
__device__ uint64_t d_ang2pix(double theta, double phi){
    if(!(theta >= 0 && theta <= PI))
        return MAX_Y;
//		throw "Invalid theta value.";
    if ((theta < 0.01) || (theta > PI - 0.01))
        return d_loc2pix(cos(theta), phi, sin(theta), (bool)(true));
    else
        return d_loc2pix(cos(theta), phi, 0., (bool)(false));
}
uint64_t h_ang2pix(double theta, double phi){
    if(!(theta >= 0 && theta <= PI))
        return MAX_Y;
    if ((theta < 0.01) || (theta > PI - 0.01))
        return h_loc2pix(cos(theta), phi, sin(theta), (bool)(true));
    else
        return h_loc2pix(cos(theta), phi, 0., (bool)(false));
}

/* Convert HEALPix Index to location (z, phi). */
__device__ void d_pix2loc(uint64_t pix, double &z, double &phi, double &sin_theta, bool &have_sin_theta){
    double tmp, fodd;
    uint64_t ip, itmp, nl4, iphi, iring;

    have_sin_theta = false;
    if(pix < d_const_Healpix._ncap){
        iring = (1 + (uint64_t)(isqrt(1+2*pix))) >> 1;
        iphi = (pix + 1) - 2 * iring * (iring - 1);
        tmp = (iring * iring) * d_const_Healpix._fact2;
        z = 1.0 - tmp;
        if (z > 0.99){
            sin_theta = sqrt(tmp * (2. - tmp));
            have_sin_theta = true;
        }
        phi = (iphi - 0.5) * HALFPI / iring;
    }
    else if(pix < d_const_Healpix._npix - d_const_Healpix._ncap){
        nl4 = 4 * d_const_Healpix._nside;
        ip = pix - d_const_Healpix._ncap;
        if(d_const_Healpix._order >= 0)
            itmp = ip >> (d_const_Healpix._order + 2);
        else
            itmp = ip / nl4;

        iring = itmp + d_const_Healpix._nside;
        iphi = ip - nl4 * itmp + 1;
        if (((iring + d_const_Healpix._nside) & 1) > 0)
            fodd = 1.0;
        else
            fodd = 0.5;

        z = (2. * d_const_Healpix._nside - iring) * d_const_Healpix._fact1;
        phi = (iphi - fodd) * PI * 0.75 * d_const_Healpix._fact1;
    }
    else{
        ip = d_const_Healpix._npix - pix;
        iring = (1 + (uint64_t)(isqrt(2 * ip - 1))) >> 1;
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        tmp = (iring * iring) * d_const_Healpix._fact2;
        z = tmp - 1.0;
        if (z < -0.99){
            sin_theta = sqrt(tmp * (2. - tmp));
            have_sin_theta = true;
        }
        phi = (iphi - 0.5) * HALFPI / iring;
    }
}
void h_pix2loc(uint64_t pix, double &z, double &phi, double &sin_theta, bool &have_sin_theta){
    double tmp, fodd;
    uint64_t ip, itmp, nl4, iphi, iring;

    have_sin_theta = false;
    if(pix < h_Healpix._ncap){
        iring = (1 + (uint64_t)(isqrt(1+2*pix))) >> 1;
        iphi = (pix + 1) - 2 * iring * (iring - 1);
        tmp = (iring * iring) * h_Healpix._fact2;
        z = 1.0 - tmp;
        if (z > 0.99){
            sin_theta = sqrt(tmp * (2. - tmp));
            have_sin_theta = true;
        }
        phi = (iphi - 0.5) * HALFPI / iring;
    }
    else if(pix < h_Healpix._npix - h_Healpix._ncap){
        nl4 = 4 * h_Healpix._nside;
        ip = pix - h_Healpix._ncap;
        if(h_Healpix._order >= 0)
            itmp = ip >> (h_Healpix._order + 2);
        else
            itmp = ip / nl4;

        iring = itmp + h_Healpix._nside;
        iphi = ip - nl4 * itmp + 1;
        if (((iring + h_Healpix._nside) & 1) > 0)
            fodd = 1.0;
        else
            fodd = 0.5;

        z = (2. * h_Healpix._nside - iring) * h_Healpix._fact1;
        phi = (iphi - fodd) * PI * 0.75 * h_Healpix._fact1;
    }
    else{
        ip = h_Healpix._npix - pix;
        iring = (1 + (uint64_t)(isqrt(2 * ip - 1))) >> 1;
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        tmp = (iring * iring) * h_Healpix._fact2;
        z = tmp - 1.0;
        if (z < -0.99){
            sin_theta = sqrt(tmp * (2. - tmp));
            have_sin_theta = true;
        }
        phi = (iphi - 0.5) * HALFPI / iring;
    }
}

/* Convert HEALPix Index to angular coordinates (phi, theta). */
__device__ void d_pix2ang(uint64_t pix, double &theta, double &phi){
    double _z, _sin_theta;
    bool _have_sin_theta;

    d_pix2loc(pix, _z, phi, _sin_theta, _have_sin_theta);

    if (_have_sin_theta){
        theta = atan2(_sin_theta, _z);
    }
    else{
        theta = acos(_z);
    }
}
void h_pix2ang(uint64_t pix, double &theta, double &phi){
    double _z, _sin_theta;
    bool _have_sin_theta;

    h_pix2loc(pix, _z, phi, _sin_theta, _have_sin_theta);

    if (_have_sin_theta){
        theta = atan2(_sin_theta, _z);
    }
    else{
        theta = acos(_z);
    }
}
