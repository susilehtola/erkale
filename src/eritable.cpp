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
  omega_=0.0;
  alpha_=1.0;
  beta_=0.0;
}

ERItable::~ERItable() {
}

void ERItable::set_range_separation(double w, double a, double b) {
  omega_=w;
  alpha_=a;
  beta_=b;
}

void ERItable::range_separation(double & w, double & a, double & b) const {
  w=omega_;
  a=alpha_;
  b=beta_;
}

size_t ERItable::N_ints(const BasisSet * basp, double thr) {
  // Get ERI pairs
  ScreeningData s = basp->compute_screening(thr, omega_, alpha_, beta_);
  Q_ = std::move(s.Q);
  M_ = std::move(s.M);
  shpairs_ = std::move(s.shpairs);

  // Form offset table and calculate amount of integrals
  size_t N=0;
  shoff_.resize(shpairs_.size());

  shoff_[0]=0;
  for(size_t ip=0;ip<shpairs_.size()-1;ip++) {
    size_t Nij=shpairs_[ip].Ni*shpairs_[ip].Nj;
    for(size_t jp=0;jp<=ip;jp++) {
      N+=Nij*shpairs_[jp].Ni*shpairs_[jp].Nj;
    }
    shoff_[ip+1]=N;
  }

  // Contribution from last shell (no importance to offset)
  size_t ip=shpairs_.size()-1;
  size_t Nij=shpairs_[ip].Ni*shpairs_[ip].Nj;
  for(size_t jp=0;jp<=ip;jp++) {
    N+=Nij*shpairs_[jp].Ni*shpairs_[jp].Nj;
  }

  return N;
}

size_t ERItable::N() const {
  return ints_.size();
}

size_t ERItable::offset(size_t ip, size_t jp) const {
  // Calculate offset in integrals table
  size_t ioff(shoff_[ip]);
  size_t Nij=shpairs_[ip].Ni*shpairs_[ip].Nj;
  for(size_t jj=0;jj<jp;jj++)
    ioff+=Nij*shpairs_[jj].Ni*shpairs_[jj].Nj;

  return ioff;
}

arma::mat ERItable::calcJ(const arma::mat & P) const {
  if(P.n_rows != Nbf_ || P.n_cols != Nbf_) {
    std::ostringstream oss;
    oss << "Error in ERItable: Nbf = " << Nbf_ << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

  arma::mat J(P);
  J.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Integral digestor
    JDigestor dig(P);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shpairs_.size();ip++)
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++)
	dig.digest(shpairs_,ip,jp,ints_,offset(ip,jp));

#ifdef _OPENMP
#pragma omp critical
#endif
    J+=dig.J();
  }

  return J;
}

arma::mat ERItable::calcK(const arma::mat & P) const {
  if(P.n_rows != Nbf_ || P.n_cols != Nbf_) {
    std::ostringstream oss;
    oss << "Error in ERItable: Nbf = " << Nbf_ << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

  arma::mat K(P);
  K.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Integral digestor
    KDigestor dig(P);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shpairs_.size();ip++)
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++)
	dig.digest(shpairs_,ip,jp,ints_,offset(ip,jp));

#ifdef _OPENMP
#pragma omp critical
#endif
    K+=dig.K();
  }

  return K;
}

arma::cx_mat ERItable::calcK(const arma::cx_mat & P) const {
  if(P.n_rows != Nbf_ || P.n_cols != Nbf_) {
    std::ostringstream oss;
    oss << "Error in ERItable: Nbf = " << Nbf_ << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

  arma::cx_mat K(P);
  K.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Integral digestor
    cxKDigestor dig(P);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shpairs_.size();ip++)
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++)
	dig.digest(shpairs_,ip,jp,ints_,offset(ip,jp));

#ifdef _OPENMP
#pragma omp critical
#endif
    K+=dig.K();
  }

  return K;
}

size_t ERItable::fill(const BasisSet * basp, double tol) {
  Nbf_=basp->get_Nbf();

  // libcint description of the basis, shared by the workers
  CintEnv cenv(*basp);

  // Shells
  const std::vector<GaussianShell> & shells=basp->get_shells_ref();

  // Compute memory requirements
  size_t N;
  N=N_ints(basp,tol);

  // Don't do DOS
  if(N*sizeof(double)>14*1e9) {
    ERROR_INFO();
    throw std::out_of_range("Cowardly refusing to allocate more than 14 gigs of memory.\n");
  }

  try {
    ints_.assign(N,0.0);
  } catch(std::bad_alloc &) {
    std::ostringstream oss;

    ERROR_INFO();
    oss << "Was unable to reserve " << memory_size(N*sizeof(double)) << " of memory.\n";
    throw std::runtime_error(oss.str());
  }

  // Get number of shell pairs
  const size_t Npairs=shpairs_.size();

#ifdef _OPENMP
#pragma omp parallel
#endif // ifdef _OPENMP
  {
    // ERI worker
    auto eri = make_eri_worker(cenv, omega_, alpha_, beta_);

    // Integral array
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<Npairs;ip++) {
      // Loop over second pairs
      for(size_t jp=0;jp<=ip;jp++) {
	// Shells on first pair
	size_t is=shpairs_[ip].is;
	size_t js=shpairs_[ip].js;
	// and those on the second pair
	size_t ks=shpairs_[jp].is;
	size_t ls=shpairs_[jp].js;

	// Amount of functions on the first pair
	size_t Ni=shpairs_[ip].Ni;
	size_t Nj=shpairs_[ip].Nj;
	// and on the second
	size_t Nk=shpairs_[jp].Ni;
	size_t Nl=shpairs_[jp].Nj;
	// Amount of integrals is
	size_t Nints=Ni*Nj*Nk*Nl;

	// Initialize table
	size_t ioff(offset(ip,jp));
	for(size_t i=0;i<Nints;i++)
	  ints_[ioff+i]=0.0;

        // Schwarz screening estimate
        double QQ=Q_(is,js)*Q_(ks,ls);
        if(QQ<tol) {
          // Skip due to small value of integral. Because the
          // integrals have been ordered wrt Q, all the next ones
          // will be small as well!
          break;
        }

        // Distance screening estimate
        double MM1=M_(is,ks)*M_(js,ls);
        double MM2=M_(is,ls)*M_(js,ks);
        if(MM1<tol || MM2<tol) {
          // This pair is negligible
          continue;
        }

	// Compute integrals
	eri->compute(is,js,ks,ls);
	erip=eri->getp();

	// Store integrals
	for(size_t ii=0;ii<Nints;ii++)
	  ints_[ioff+ii]=(*erip)[ii];
      }
    }
  }

  return shpairs_.size();
}
