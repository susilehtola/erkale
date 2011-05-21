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
#define CHECKFILL


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

  // Loop over matrix elements
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<=i;j++) {
      // The (ij) element in the J array
      double tmp=0.0;

      // Loop over density matrix
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

  // Loop over matrix elements
  for(size_t i=0;i<N;i++)
    for(size_t l=0;l<=i;l++) {
      // The (ij) element in the J array
      double tmp=0.0;

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

void ERItable::calcJK(const arma::mat & R, arma::mat & J, arma::mat & K) const {
  // Calculate Coulomb and exchange at the same time

  // Size of basis set
  const size_t Rc=R.n_cols, Rr=R.n_rows;

  if(Rc!=Rr) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Density matrix not square!\n";
    throw std::runtime_error(oss.str());
  }

  const size_t N=R.n_cols;

  J=arma::mat(N,N);
  K=arma::mat(N,N);

  // Initialize output
  J.zeros();
  K.zeros();

  double res;

  // Loops
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<N;j++) 
      for(size_t k=0;k<N;k++)
	for(size_t l=0;l<N;l++) {
	  // Get integral
	  res=getERI(i,j,k,l);
	  // Increment Coulomb and exchange
	  //J[i,j]+=R[k,l]*res;
	  //K[i,l]+=R[j,k]*res;
	  J(i,j)+=R(k,l)*res;
	  K(i,l)+=R(j,k)*res;
	}
}


void ERItable::calcJK(const arma::mat & Ra, const arma::mat & Rb, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const {
  // Calculate Coulomb and exchange at the same time

  // Size of basis set
  const size_t Rc=Ra.n_cols, Rr=Ra.n_rows;

  if(Rc!=Rr) {
    ERROR_INFO();
    throw std::runtime_error("Density matrix not square!\n");
  }

  if(Rb.n_cols != Rc || Rb.n_rows != Rc) {
    ERROR_INFO();
    throw std::runtime_error("Alpha and beta density matrices are not of the same size!\n");
  }

  const size_t N=Ra.n_cols;

  if(J.n_cols!=N || J.n_rows!=N)
    J=arma::mat(N,N);
  if(Ka.n_cols!=N || Ka.n_rows!=N)
    Ka=arma::mat(N,N);
  if(Kb.n_cols!=N || Kb.n_rows!=N)
    Kb=arma::mat(N,N);

  // Initialize output
  J.zeros();
  Ka.zeros();
  Kb.zeros();

  // Compute total density matrix
  arma::mat R=Ra+Rb;

  double res;

  // Loops
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<N;j++) 
      for(size_t k=0;k<N;k++)
	for(size_t l=0;l<N;l++) {
	  // Get integral
	  res=getERI(i,j,k,l);
	  // Increment Coulomb and exchange
	  //J[i,j]+=R[k,l]*res;
	  //K[i,l]+=R[j,k]*res;
	  J(i,j)+=R(k,l)*res;
	  Ka(i,l)+=Ra(j,k)*res;
	  Kb(i,l)+=Rb(j,k)*res;
	}
}
