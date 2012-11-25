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

  // Loop over shell pairs
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(size_t ip=0;ip<Npairs;ip++) {
    size_t i=pairs[ip].is;
    size_t j=pairs[ip].js;
    
    // Compute integrals
    std::vector<double> ints=basp->ERI(i,j,i,j);
    // Get maximum value
    double m=0.0;
    for(size_t k=0;k<ints.size();k++)
      if(fabs(ints[k])>m)
	m=fabs(ints[k]);
    m=sqrt(m);
    screen(i,j)=m;
    screen(j,i)=m;
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

void ERIscreen::integral_symmetry(size_t i, size_t j, size_t k, size_t l, size_t iarr[], size_t jarr[], size_t karr[], size_t larr[], size_t & nid) const {
  // Forms list of identical integrals

  bool nonid_ij, nonid_kl;

  nid=0;
  
  // Initialize table of identical integrals.
  // By default we have at least one integral, (ij|kl).
  iarr[nid]=i;
  jarr[nid]=j;
  karr[nid]=k;
  larr[nid]=l;
  nid++;
  
  // Check permutation symmetries.
  if(i!=j) { 
    // (ij|kl) = (ji|kl).
    nonid_ij=1;
    iarr[nid]=j;
    jarr[nid]=i;
    karr[nid]=k;
    larr[nid]=l;
    nid++;
  } else
    nonid_ij=0;
  
  if(k!=l) { // (ij|kl) = (ij|lk)
    nonid_kl=1;
    iarr[nid]=i;
    jarr[nid]=j;
    karr[nid]=l;
    larr[nid]=k;
    nid++;
  } else
    nonid_kl=0;
  
  if(nonid_ij && nonid_kl) { // (ij|kl) = (ji|lk)
    iarr[nid]=j;
    jarr[nid]=i;
    karr[nid]=l;
    larr[nid]=k;
    nid++;
  }
  
  // (ij|kl) = (kl|ij)
  if( !( ((i==k) && (j==l)) || ((i==l) && (j==k)) )) {
    // The number of identical pairs so far
    size_t nido=nid;
    
    // Fill in the rest of the permutations
    for(size_t inid=0;inid<nido;inid++) {
      iarr[nid]=karr[inid];
      jarr[nid]=larr[inid];
      karr[nid]=iarr[inid];
      larr[nid]=jarr[inid];
      nid++;
    }
  }

  /*
    printf("The (%i %i | %i %i) has the identical permutations: ",i,j,k,l);
    for(int i=1;i<nid;i++)
    printf(" (%i %i | %i %i)",iarr[i],jarr[i],karr[i],larr[i]);
    printf("\n");
  */
}
