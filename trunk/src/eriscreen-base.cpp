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
}

ERIscreen::~ERIscreen() {
}

void ERIscreen::fill(const BasisSet * basisv) {
  // Form screening table.

  if(basisv==NULL)
    return;

  basp=basisv;

  // Amount of shells in basis set is
  size_t Nsh=basp->get_Nshells();

  // Form index helper
  iidx=i_idx(basp->get_Nbf());

  // Allocate storage
  screen=arma::mat(Nsh,Nsh);

  // Get list of unique shell pairs
  std::vector<shellpair_t> pairs=basp->get_unique_shellpairs();
  // Get number of shell pairs
  const size_t Npairs=pairs.size();

  std::vector<GaussianShell> shells=basp->get_shells();

  // Loop over shell pairs
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker eri(basp->get_max_am(),basp->get_max_Ncontr());
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<Npairs;ip++) {
      size_t i=pairs[ip].is;
      size_t j=pairs[ip].js;

      // Compute integrals
      eri.compute(&shells[i],&shells[j],&shells[i],&shells[j]);
      erip=eri.getp();
      // Get maximum value
      double m=0.0;
      for(size_t k=0;k<(*erip).size();k++)
        if(fabs((*erip)[k])>m)
          m=fabs((*erip)[k]);
      m=sqrt(m);
      screen(i,j)=m;
      screen(j,i)=m;
    }
  }

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
