/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#include "eriscreen.h"
#include "basis.h"
#include "mathf.h"

#include <cstdio>
// For exceptions
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Print out screening table?
//#define PRINTOUT


#ifdef CHECKFILL
size_t idx(size_t i, size_t j, size_t k, size_t l) {
  if(i<j)
    std::swap(i,j);
  if(k<l)
    std::swap(k,l);

  size_t ij=(i*(i+1))/2+j;
  size_t kl=(k*(k+1))/2+l;

  if(ij<kl)
    std::swap(ij,kl);

  return (ij*(ij+1))/2+kl;
}
#endif


ERIscreen::ERIscreen() {
  omega=0.0;
  alpha=1.0;
  beta=0.0;
}

ERIscreen::~ERIscreen() {
}

void ERIscreen::set_range_separation(double w, double a, double b) {
  omega=w;
  alpha=a;
  beta=b;
}

void ERIscreen::get_range_separation(double & w, double & a, double & b) {
  w=omega;
  a=alpha;
  b=beta;
}

void ERIscreen::fill(const BasisSet * basisv, double shtol, bool verbose) {
  // Form screening table.

  if(basisv==NULL)
    return;

  basp=basisv;

  // Form index helper
  iidx=i_idx(basp->get_Nbf());

  // Screening matrix and pairs
  shpairs=basp->get_eripairs(screen,shtol,verbose);

  // Get minimum of screening matrix
  //  double minval=min(min(screen));
  //  printf("done.\nScreening is effective for cutoffs larger than %e.\n",minval*minval);

#ifdef PRINTOUT
  FILE *out=fopen("screening.dat","w");

  for(size_t i=0;i<Nsh;i++)
    for(size_t j=0;j<=i;j++)
      fprintf(out,"%e\n",screen(i,j));
  fflush(out);
  fclose(out);
#endif

}
