/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "atomtable.h"
#include "../stringutil.h"
#include "../timer.h"

AtomTable::AtomTable() {
  Nbf=0;
}

size_t AtomTable::idx(size_t i, size_t j, size_t k, size_t l) const {
  return (((i*Nbf+j)*Nbf+k)*Nbf+l);
}

void AtomTable::fill(const std::vector<bf_t> & basis, bool verbose) {
  // Amount of basis functions is
  Nbf=basis.size();

  // Amount of integrals is (complex functions, so symmetry is different..)
  size_t N=Nbf*Nbf*Nbf*Nbf;

  // Make pairs helper
  pairs.clear();
  for(size_t i=0;i<Nbf;i++)
    for(size_t j=0;j<=i;j++) {
      bfpair_t tmp;
      tmp.i=i;
      tmp.j=j;
      pairs.push_back(tmp);
    }

  try {
    ints.reserve(N);
    ints.resize(N);
  } catch(std::bad_alloc & err) {
    std::ostringstream oss;

    ERROR_INFO();
    oss << "Was unable to reserve " << memory_size(N*sizeof(double)) << " of memory.\n";
    throw std::runtime_error(oss.str());
  }
  // Initialize with zeros
  for(size_t i=0;i<N;i++)
    ints[i]=0.0;

  Timer t;
  if(verbose) {
    printf("Filling table of integrals ... ");
    fflush(stdout);
  }

  // Fill integrals table
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<Nbf;i++)
    for(size_t j=0;j<Nbf;j++)
      for(size_t k=0;k<Nbf;k++)
	for(size_t l=0;l<Nbf;l++) {
	  ints[idx(i,j,k,l)]=ERI(basis[i],basis[j],basis[k],basis[l]);
	}

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

double AtomTable::getERI(size_t i, size_t j, size_t k, size_t l) const {
  return ints[idx(i,j,k,l)];
}

AtomTable::~AtomTable() {
}

arma::mat AtomTable::calcJ(const arma::mat & R) const {
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

arma::mat AtomTable::calcK(const arma::mat & R) const {
  // Calculate exchange matrix

  // Size of basis set
  size_t N=R.n_cols;

  // Returned matrix
  arma::mat K(N,N);
  K.zeros();

  // Loop over matrix elements
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<pairs.size();ip++) {
    // The relevant indices are
    size_t i=pairs[ip].i;
    size_t j=pairs[ip].j;

    // The (ij) element in the K array
    double el=0.0;

    // Loop over density matrix
    for(size_t k=0;k<N;k++)
      for(size_t l=0;l<N;l++) {
	el+=R(k,l)*getERI(i,k,j,l);
      }

    // Store result
    K(i,j)=el;
    K(j,i)=el;
  }

  return K;
}

