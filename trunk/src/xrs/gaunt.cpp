/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#include <cstdio>
#include <cmath>
#include "gaunt.h"
#include "lmgrid.h"

extern "C" {
  // 3j symbols
#include <gsl/gsl_sf_coupling.h>
}

double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp) {
  // First, compute square root factor
  double res=sqrt((2*L+1)*(2*l+1)*(2*lp+1)/(4.0*M_PI));
  // Plug in (l1 l2 l3 | 0 0 0), GSL uses half units
  res*=gsl_sf_coupling_3j(2*L,2*l,2*lp,0,0,0);
  // Plug in (l1 l2 l3 | m1 m2 m3)
  res*=gsl_sf_coupling_3j(2*L,2*l,2*lp,-2*M,2*m,2*mp);
  // and the phase factor
  res*=pow(-1.0,M);

  return res;
}

Gaunt::Gaunt() {
}

Gaunt::Gaunt(int Lmax, int lmax, int lpmax) {
  // Allocate storage
  table=arma::cube(lmind(Lmax,Lmax)+1,lmind(lmax,lmax)+1,lmind(lpmax,lpmax)+1);

  // Compute coefficients
  for(int L=0;L<=Lmax;L++)
    for(int M=-L;M<=L;M++)

      for(int l=0;l<=lmax;l++)
	for(int m=-l;m<=l;m++)

	  for(int lp=0;lp<=lpmax;lp++)
	    for(int mp=-lp;mp<=lp;mp++)
	      table(lmind(L,M),lmind(l,m),lmind(lp,mp))=gaunt_coefficient(L,M,l,m,lp,mp);
}

Gaunt::~Gaunt() {
}

double Gaunt::coeff(int L, int M, int l, int m, int lp, int mp) const {
  return table(lmind(L,M),lmind(l,m),lmind(lp,mp));
}
