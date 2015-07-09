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
#include "eriworker.h"
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

void ERItable::get_range_separation(double & w, double & a, double & b) const {
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

size_t ERItable::offset(size_t ip, size_t jp) const {
  // Calculate offset in integrals table
  size_t ioff(shoff[ip]); 
  size_t Nij=shpairs[ip].Ni*shpairs[ip].Nj;
  for(size_t jj=0;jj<jp;jj++)
    ioff+=Nij*shpairs[jj].Ni*shpairs[jj].Nj;

  return ioff;
}

arma::mat ERItable::calcJ(const arma::mat & P) const {
  arma::mat J(P);
  J.zeros();
  
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Integral digestor
    JDigestor dig(P);
    
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t ip=0;ip<shpairs.size();ip++)
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++)
	dig.digest(shpairs,ip,jp,ints,offset(ip,jp));
    
#ifdef _OPENMP
#pragma omp critical
#endif
    J+=dig.get_J();
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
    // Integral digestor
    KDigestor dig(P);
    
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t ip=0;ip<shpairs.size();ip++)
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++)
	dig.digest(shpairs,ip,jp,ints,offset(ip,jp));
    
#ifdef _OPENMP
#pragma omp critical
#endif
    K+=dig.get_K();
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
    
    // Integral digestor
    cxKDigestor dig(P);
    
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t ip=0;ip<shpairs.size();ip++)
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++)
	dig.digest(shpairs,ip,jp,ints,offset(ip,jp));
    
#ifdef _OPENMP
#pragma omp critical
#endif
    K+=dig.get_K();
  }

  return K;
}

size_t ERItable::fill(const BasisSet * basp, double tol) {
  // Shells
  std::vector<GaussianShell> shells=basp->get_shells();

  // Compute memory requirements
  size_t N;
  N=N_ints(basp,tol);
  
  // Don't do DOS
  if(N*sizeof(double)>14*1e9) {
    ERROR_INFO();
    throw std::out_of_range("Cowardly refusing to allocate more than 14 gigs of memory.\n");
  }
  
  try {
    ints.resize(N);
  } catch(std::bad_alloc err) {
    std::ostringstream oss;
    
    ERROR_INFO();
    oss << "Was unable to reserve " << memory_size(N*sizeof(double)) << " of memory.\n";
    throw std::runtime_error(oss.str());
  }

  // Get number of shell pairs
  const size_t Npairs=shpairs.size();

#ifdef _OPENMP
#pragma omp parallel
#endif // ifdef _OPENMP
  {
    // ERI worker
    ERIWorker *eri = (omega==0.0 && alpha==1.0 && beta==0.0) ? new ERIWorker(basp->get_max_am(),basp->get_max_Ncontr()) : new ERIWorker_srlr(basp->get_max_am(),basp->get_max_Ncontr(),omega,alpha,beta);

    // Integral array
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<Npairs;ip++) {
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++) {
	// Shells on first pair
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;
	// and those on the second pair
	size_t ks=shpairs[jp].is;
	size_t ls=shpairs[jp].js;
	
	// Amount of functions on the first pair
	size_t Ni=shpairs[ip].Ni;
	size_t Nj=shpairs[ip].Nj;
	// and on the second
	size_t Nk=shpairs[jp].Ni;
	size_t Nl=shpairs[jp].Nj;
	// Amount of integrals is
	size_t Nints=Ni*Nj*Nk*Nl;
	
	// Initialize table
	size_t ioff(offset(ip,jp));
	for(size_t i=0;i<Nints;i++)
	  ints[ioff+i]=0.0;

	// Maximum value of the 2-electron integrals on this shell pair
	double intmax=screen(is,js)*screen(ks,ls);
	if(intmax<tol) {
	  // Skip due to small value of integral. Because the
	  // integrals have been ordered, all the next ones will be
	  // small as well!
	  break;
	}

	// Compute integrals
	eri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	erip=eri->getp();

	// Store integrals
	for(size_t ii=0;ii<Nints;ii++)
	  ints[ioff+ii]=(*erip)[ii];
      }
    }
  }

  return shpairs.size();
}

