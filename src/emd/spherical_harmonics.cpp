/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#include "spherical_harmonics.h"

#include <cmath>
extern "C" {
// Legendre polynomials
#include <gsl/gsl_sf_legendre.h>
}

complex spherical_harmonics(int l, int m, double cth, double phi) {
  /* Calculate value of spherical harmonic Y_{lm} = N_{lm} P_{lm} (\cos \theta) e^{i m \phi} */
  complex ylm;

  if(m<0) { // GSL doesn't handle m<0
    //    printf("m<0 ");
    ylm=spherical_harmonics(l,-m,cth,phi);
    ylm=cscale(ylm,pow(-1.0,m));
    return cconj(ylm);
  }

  //  printf("Calculating spherical harmonic Y_{%i,%i} at (%f,%f), cth=%f:",l,m,acos(cth),phi,cth);
  //  fflush(stdout);
  // First, calculate the exponential
  ylm.re=0.0;
  ylm.im=m*phi;
  ylm=cexp(ylm);
  
  //  Then add in the normalized associated Legendre polynomial
  ylm=cscale(ylm,gsl_sf_legendre_sphPlm(l,m,cth));
  //   printf(" (%e, %e)\n",ylm.re,ylm.im);
 
  return ylm;
}
