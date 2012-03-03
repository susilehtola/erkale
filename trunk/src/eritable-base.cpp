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



#include "eritable.h"
#include "integrals.h"
#include "mathf.h"
#include "stringutil.h"

#include <algorithm>
#include <cfloat>
// For exceptions
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// To check that every nonequivalent integral is computed exactly once
//#define CHECKFILL


ERItable::ERItable() {
}

ERItable::~ERItable() {
}

size_t ERItable::N_ints(const BasisSet * basp) const {
  // Number of basis functions
  size_t Nbas=basp->get_Nbf();
  // Number of symmetry irreducible integrals
  size_t N=Nbas*(Nbas+1)*(Nbas*Nbas+Nbas+2)/8;

  return N;
}

size_t ERItable::memory_estimate(const BasisSet * basp) const {
  // Compute size of table
  return sizeof(double)*N_ints(basp);
}

size_t ERItable::idx(size_t i, size_t j, size_t k, size_t l) const {
  if(i<j)
    std::swap(i,j);
  if(k<l)
    std::swap(k,l);

  size_t ij=iidx[i]+j;
  size_t kl=iidx[k]+l;

  if(ij<kl)
    std::swap(ij,kl);

  return iidx[ij]+kl;
}

void ERItable::print() const {
  for(size_t i=0;i<ints.size();i++)
    printf("%i\t%e\n",(int) i,ints[i]);
  printf("\n");
}

size_t ERItable::count_nonzero() const {
  size_t n=0;
  for(size_t i=0;i<ints.size();i++)
    //    if(ints[i]>DBL_EPSILON)
    if(ints[i]>DBL_MIN)
      n++;
  return n;
}

double ERItable::getERI(size_t i, size_t j, size_t k, size_t l) const {
  return ints[idx(i,j,k,l)];
}

std::vector<double> & ERItable::get() {
  return ints;
}

arma::mat ERItable::calcJ(const arma::mat & R) const {
  // Calculate Coulomb matrix

  // Size of basis set
  size_t N=R.n_cols;

  // Returned matrix
  arma::mat J(N,N);
  J.zeros();

  // Index helpers
  size_t i, j;
  // The (ij) element in the J array
  double tmp;
  
  // Loop over matrix elements
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(i,j,tmp)
#endif
  for(size_t ip=0;ip<pairs.size();ip++) {
    // The relevant indices are
    i=pairs[ip].i;
    j=pairs[ip].j;

    // Loop over density matrix
    tmp=0.0;
    for(size_t k=0;k<N;k++)
      for(size_t l=0;l<N;l++) {
	tmp+=R(k,l)*getERI(i,j,k,l);
	//	  printf("J(%i,%i) += %e * %e\t(%i %i %i %i)\n",i,j,R(k,l),getERI(i,j,k,l),i,j,k,l);
      }
    
    // Store result 
    J(i,j)=tmp;
    J(j,i)=tmp;
    //      J[i,j]=tmp;
  }
  
  return J;
}

arma::mat ERItable::calcK(const arma::mat & R) const {
  // Calculate exchange matrix

  // Size of basis set
  size_t N=R.n_cols;

  // Returned matrix
  arma::mat K(N,N);
  K.zeros();

  // Helpers
  size_t i, l;
  double tmp;

  // Loop over matrix elements
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(i,l,tmp)
#endif
  for(size_t ip=0;ip<pairs.size();ip++) {
    // The relevant indices are
    i=pairs[ip].i;
    l=pairs[ip].j;
    
    // The (il) element in the K array
    tmp=0.0;
    
    // Loop over density matrix
    for(size_t j=0;j<N;j++)
      for(size_t k=0;k<N;k++) {
	tmp+=R(j,k)*getERI(i,j,k,l);
      }
    
    // Store result
    K(i,l)=tmp;
    K(l,i)=tmp;
  }
  
  return K;
}
