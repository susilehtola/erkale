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



#include "eritable.h"
#include "eri_digest.h"
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
  omega=0.0;
  alpha=1.0;
  beta=0.0;
}

ERItable::~ERItable() {
}

void ERItable::set_range_separation(double w, double a, double b) {
  omega=w;
  alpha=a;
  beta=b;
}

void ERItable::get_range_separation(double & w, double & a, double & b) {
  w=omega;
  a=alpha;
  b=beta;
}

size_t ERItable::N_ints(const BasisSet * basp, double thr) {
  // Get ERI pairs
  shpairs=basp->get_eripairs(screen, thr, omega, alpha, beta);
  
  // Form offset table and calculate amount of integrals
  size_t N=0;
  shoff.resize(shpairs.size());

  shoff[0]=0;
  for(size_t ip=0;ip<shpairs.size()-1;ip++) {
    size_t Nij=shpairs[ip].Ni*shpairs[ip].Nj;
    for(size_t jp=0;jp<=ip;jp++) {
      N+=Nij*shpairs[jp].Ni*shpairs[jp].Nj;
    }
    shoff[ip+1]=N;
  }

  // Contribution from last shell (no importance to offset)
  size_t ip=shpairs.size()-1;
  size_t Nij=shpairs[ip].Ni*shpairs[ip].Nj;
  for(size_t jp=0;jp<=ip;jp++) {
    N+=Nij*shpairs[jp].Ni*shpairs[jp].Nj;
  }
  
  return N;
}

size_t ERItable::get_N() const {
  return ints.size();
}

arma::mat ERItable::calcJ(const arma::mat & P) const {
  arma::mat J(P);
  J.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::mat Jwrk(J);

#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t ip=0;ip<shpairs.size();ip++) {
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++) {
	// Shells on first pair
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;
	// and those on the second pair
	size_t ks=shpairs[jp].is;
	size_t ls=shpairs[jp].js;

        // Calculate offset in integrals table
	size_t ioff0(shoff[ip]);
	size_t Nij=shpairs[ip].Ni*shpairs[ip].Nj;
	for(size_t jj=0;jj<jp;jj++)
	  ioff0+=Nij*shpairs[jj].Ni*shpairs[jj].Nj;
	
	// Digest integral block
	digest_J(shpairs,ip,jp,P,ints,ioff0,Jwrk);
      }
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    J+=Jwrk;
  }

  return J;
}

arma::mat ERItable::calcK(const arma::mat & P) const {
  arma::mat K(P);
  K.zeros();
  
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::mat Kwrk(K);
    
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t ip=0;ip<shpairs.size();ip++) {
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++) {
	// Shells on first pair
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;
	// and those on the second pair
	size_t ks=shpairs[jp].is;
	size_t ls=shpairs[jp].js;
	
	// Calculate offset in integrals table
	size_t ioff0(shoff[ip]);
	size_t Nij=shpairs[ip].Ni*shpairs[ip].Nj;
	for(size_t jj=0;jj<jp;jj++)
	  ioff0+=Nij*shpairs[jj].Ni*shpairs[jj].Nj;
	
	// Digest integral block
	digest_K(shpairs,ip,jp,double,P,ints,ioff0,Kwrk);
      }
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    K+=Kwrk;
  }

  return K;
}

arma::cx_mat ERItable::calcK(const arma::cx_mat & P) const {
  arma::cx_mat K(P);
  K.zeros();
  
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::cx_mat Kwrk(K);
    
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t ip=0;ip<shpairs.size();ip++) {
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++) {
	// Shells on first pair
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;
	// and those on the second pair
	size_t ks=shpairs[jp].is;
	size_t ls=shpairs[jp].js;

	// Calculate offset in integrals table
	size_t ioff0(shoff[ip]);
	size_t Nij=shpairs[ip].Ni*shpairs[ip].Nj;
	for(size_t jj=0;jj<jp;jj++)
	  ioff0+=Nij*shpairs[jj].Ni*shpairs[jj].Nj;
	
	// Digest integral block
	digest_K(shpairs,ip,jp,std::complex<double>,P,ints,ioff0,Kwrk);
      }
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    K+=Kwrk;
  }

  return K;
}
