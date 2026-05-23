
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
#include "eri_digest.h"
#include "eriworker.h"
#include "mathf.h"
#include "timer.h"

#include <algorithm>
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
  basp=NULL;
  Nbf=0;
  screen_thresh_=0.0;
}

ERIscreen::~ERIscreen() {
}

size_t ERIscreen::get_N() const {
  return Nbf;
}

void ERIscreen::set_range_separation(double w, double a, double b) {
  omega=w;
  alpha=a;
  beta=b;
  // Workers in the pool were built against the previous omega /
  // alpha / beta -- drop them so the next acquire_eri / acquire_deri
  // rebuilds with the new parameters. Keep the pools sized to the
  // thread count: acquire_* indexes by thread number, so the slots
  // must exist even if no fill() intervenes before the next calc.
#ifdef _OPENMP
  const int nth=omp_get_max_threads();
#else
  const int nth=1;
#endif
  eri_pool_.clear();
  eri_pool_.resize(nth);
  deri_pool_.clear();
  deri_pool_.resize(nth);
}

void ERIscreen::get_range_separation(double & w, double & a, double & b) const {
  w=omega;
  a=alpha;
  b=beta;
}

size_t ERIscreen::fill(const BasisSet * basisv, double shtol, bool verbose) {
  // Form screening table.
  if(basisv==NULL)
    return 0;

  basp=basisv;
  Nbf=basisv->get_Nbf();

  // Form index helper
  iidx=i_idx(Nbf);

  // Shell-pair list
  ScreeningData s = basp->compute_screening(shtol,omega,alpha,beta,verbose);
  Q = std::move(s.Q);
  M = std::move(s.M);
  shpairs = std::move(s.shpairs);

  // Worker pools are tied to (basis, omega/alpha/beta). Basis just
  // changed, so reset; size to omp_get_max_threads() so threads can
  // index by omp_get_thread_num() without locking.
#ifdef _OPENMP
  const int nth = omp_get_max_threads();
#else
  const int nth = 1;
#endif
  eri_pool_.clear();
  eri_pool_.resize(nth);
  deri_pool_.clear();
  deri_pool_.resize(nth);

  return shpairs.size();
}

ERIWorker * ERIscreen::acquire_eri(int ith) const {
  if(!eri_pool_[ith]) {
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri_pool_[ith].reset(new ERIWorker(basp->get_max_am(), basp->get_max_Ncontr()));
    else
      eri_pool_[ith].reset(new ERIWorker_srlr(basp->get_max_am(), basp->get_max_Ncontr(), omega, alpha, beta));
  }
  return eri_pool_[ith].get();
}

dERIWorker * ERIscreen::acquire_deri(int ith) const {
  if(!deri_pool_[ith]) {
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      deri_pool_[ith].reset(new dERIWorker(basp->get_max_am(), basp->get_max_Ncontr()));
    else
      deri_pool_[ith].reset(new dERIWorker_srlr(basp->get_max_am(), basp->get_max_Ncontr(), omega, alpha, beta));
  }
  return deri_pool_[ith].get();
}

arma::mat ERIscreen::density_bounds(const arma::mat & P) const {
  // D(i,j) = max |P(mu,nu)| over the (shell i, shell j) block. Used to
  // bound each shell-quartet's contribution to J/K (the integral times
  // the largest coupled density element).
  const std::vector<GaussianShell> & shells=basp->get_shells_ref();
  const size_t Nsh=shells.size();

  // Per-shell function-index vectors. Built once and reused for the
  // Nsh^2 block reductions below.
  std::vector<arma::uvec> idx(Nsh);
  for(size_t i=0;i<Nsh;i++) {
    const size_t i0=shells[i].get_first_ind();
    const size_t Ni=shells[i].get_Nbf();
    idx[i]=arma::regspace<arma::uvec>(i0, i0+Ni-1);
  }

  arma::mat D(Nsh,Nsh);
  for(size_t i=0;i<Nsh;i++)
    for(size_t j=0;j<Nsh;j++)
      D(i,j)=arma::abs(P(idx[i], idx[j])).max();
  return D;
}

void ERIscreen::calculate(std::vector< std::vector<IntegralDigestor *> > & digest, const arma::mat & D, double tol) const {
  // Shells in basis set
  const std::vector<GaussianShell> & shells=basp->get_shells_ref();
  // Get number of shell pairs
  const size_t Npairs=shpairs.size();

  // Global density bound. D(a,b) <= Dmax for every shell pair, so the
  // sorted-Q early-out can use QQ*Dmax: once that drops below tol,
  // every remaining (smaller-Q) quartet is below threshold too.
  const double Dmax = D.n_elem ? D.max() : 0.0;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith(omp_get_thread_num());
#endif
    // ERI worker (per-thread pool; lazily built on first acquire).
    ERIWorker *eri = acquire_eri(ith);

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

        // Schwarz bound on |(ij|kl)|.
        double QQ=Q(is,js)*Q(ks,ls);
        // Integral threshold: break if every remaining quartet has
        // |(ij|kl)| < tol. The shellpair list is sorted by Q so all
        // later (smaller-QQ) pairs are below threshold as well.
        if(QQ<tol)
          break;
        // Density-weighted threshold: break if every remaining Fock
        // contribution QQ*D drops below screen_thresh_. D(.,.)<=Dmax
        // for every pair so this is a valid early-out too.
        if(screen_thresh_>0.0 && QQ*Dmax<screen_thresh_)
          break;

        // Two product-basis Cauchy-Schwarz bounds on |(ij|kl)| via the
        // (i,k)-(j,l) and (i,l)-(j,k) groupings of the four shells.
        // Both bound the same integral so the tightest is their min;
        // skip when that drops below the threshold.
        const double MM1=M(is,ks)*M(js,ls);
        const double MM2=M(is,ls)*M(js,ks);
        const double MM=std::min(MM1,MM2);
        // Integral threshold (per-quartet).
        if(MM<tol)
          continue;

        if(screen_thresh_>0.0) {
          // Density-weighted screening. The contribution of (ij|kl) to
          // J/K is bounded by the integral times the largest density-
          // matrix element over the blocks it couples -- (ij) and (kl)
          // for Coulomb, the four cross blocks for exchange. Screen on
          // that product so screen_thresh_ is a threshold on the actual
          // Fock contribution rather than on the bare integral.
          double Dq=D(is,js);
          Dq=std::max(Dq,D(ks,ls));
          Dq=std::max(Dq,D(is,ks));
          Dq=std::max(Dq,D(is,ls));
          Dq=std::max(Dq,D(js,ks));
          Dq=std::max(Dq,D(js,ls));
          if(QQ*Dq<screen_thresh_)
            continue;
          if(MM*Dq<screen_thresh_)
            continue;
        }

	// Compute integrals
	eri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	erip=eri->getp();

	// Digest the integrals
	for(size_t i=0;i<digest[ith].size();i++)
	  digest[ith][i]->digest(shpairs,ip,jp,*erip,0);
      }
    }
    // eri is owned by eri_pool_ -- do not delete.
  }
}

arma::vec ERIscreen::calculate_force(std::vector< std::vector<ForceDigestor *> > & digest, double tol) const {
  // Shells
  const std::vector<GaussianShell> & shells=basp->get_shells_ref();
  // Get number of shell pairs
  const size_t Npairs=shpairs.size();

  // Forces
  arma::vec F(3*basp->get_Nnuc());
  F.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith(omp_get_thread_num());
    arma::vec Fwrk(F);
#endif
    /// ERI derivative worker (per-thread pool, lazily built).
    dERIWorker *deri = acquire_deri(ith);

    // Shell centers
    size_t inuc, jnuc, knuc, lnuc;

    // Per-quartet force accumulator; hoisted out of the loop so it is
    // re-zeroed rather than reallocated on every shell quartet.
    arma::vec f(12);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<Npairs;ip++) {
      for(size_t jp=0;jp<=ip;jp++) {
	// Shells on first pair
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;
	// and those on the second pair
	size_t ks=shpairs[jp].is;
	size_t ls=shpairs[jp].js;

	// Shell centers
	inuc=shells[is].get_center_ind();
	jnuc=shells[js].get_center_ind();
	knuc=shells[ks].get_center_ind();
	lnuc=shells[ls].get_center_ind();

	// Skip when all functions are on the same nucleus - force will vanish
	if(inuc==jnuc && jnuc==knuc && knuc==lnuc)
	  continue;

        // Schwarz screening estimate
        double QQ=Q(is,js)*Q(ks,ls);
        if(QQ<tol) {
          // Skip due to small value of integral. Because the
          // integrals have been ordered wrt Q, all the next ones
          // will be small as well!
          break;
        }

        // Two product-basis Cauchy-Schwarz bounds on |(ij|kl)| via the
        // (i,k)-(j,l) and (i,l)-(j,k) groupings; take the tightest.
        const double MM1=M(is,ks)*M(js,ls);
        const double MM2=M(is,ls)*M(js,ks);
        if(std::min(MM1,MM2)<tol)
          continue;

	// Compute the derivatives.
	deri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);

	// Digest the forces on the nuclei
	f.zeros();

	// Digest the integrals
	for(size_t i=0;i<digest[ith].size();i++)
	  digest[ith][i]->digest(shpairs,ip,jp,*deri,f);

	// Increment forces
#ifdef _OPENMP
	Fwrk.subvec(3*inuc,3*inuc+2)+=f.subvec(0,2);
	Fwrk.subvec(3*jnuc,3*jnuc+2)+=f.subvec(3,5);
	Fwrk.subvec(3*knuc,3*knuc+2)+=f.subvec(6,8);
	Fwrk.subvec(3*lnuc,3*lnuc+2)+=f.subvec(9,11);
#else
	F.subvec(3*inuc,3*inuc+2)+=f.subvec(0,2);
	F.subvec(3*jnuc,3*jnuc+2)+=f.subvec(3,5);
	F.subvec(3*knuc,3*knuc+2)+=f.subvec(6,8);
	F.subvec(3*lnuc,3*lnuc+2)+=f.subvec(9,11);
#endif
      }
    }

    // Collect results
#ifdef _OPENMP
#pragma omp critical
    F+=Fwrk;
#endif
    // deri is owned by deri_pool_ -- do not delete.
  }

  return F;
}

arma::mat ERIscreen::calcJ(const arma::mat & P, double tol) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(1);
    p[i][0]=new JDigestor(P);
  }

  // Do calculation
  arma::mat D=density_bounds(P);
  calculate(p,D,tol);

  // Collect results
  arma::mat J(((JDigestor *) p[0][0])->get_J());
  for(int i=1;i<nth;i++)
    J+=((JDigestor *) p[i][0])->get_J();

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return J;
}

arma::mat ERIscreen::calcK(const arma::mat & P, double tol) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(1);
    p[i][0]=new KDigestor(P);
  }

  // Do calculation
  arma::mat D=density_bounds(P);
  calculate(p,D,tol);

  // Collect results
  arma::mat K(((KDigestor *) p[0][0])->get_K());
  for(int i=1;i<nth;i++)
    K+=((KDigestor *) p[i][0])->get_K();

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return K;
}

arma::cx_mat ERIscreen::calcK(const arma::cx_mat & P, double tol) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(1);
    p[i][0]=new cxKDigestor(P);
  }

  // Do calculation
  arma::mat D=density_bounds(arma::abs(P));
  calculate(p,D,tol);

  // Collect results
  arma::cx_mat K(((cxKDigestor *) p[0][0])->get_K());
  for(int i=1;i<nth;i++)
    K+=((cxKDigestor *) p[i][0])->get_K();

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return K;
}

void ERIscreen::calcK(const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ka, arma::mat & Kb, double tol) const {
  if(Pa.n_rows != Nbf || Pa.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pa.n_rows = " << Pa.n_rows << ", Pa.n_cols = " << Pa.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(Pb.n_rows != Nbf || Pb.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pb.n_rows = " << Pb.n_rows << ", Pb.n_cols = " << Pb.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(2);
    p[i][0]=new KDigestor(Pa);
    p[i][1]=new KDigestor(Pb);
  }

  // Do calculation
  arma::mat D=density_bounds(arma::abs(Pa)+arma::abs(Pb));
  calculate(p,D,tol);

  // Collect results
  Ka=((KDigestor *) p[0][0])->get_K();
  Kb=((KDigestor *) p[0][1])->get_K();
  for(int i=1;i<nth;i++) {
    Ka+=((KDigestor *) p[i][0])->get_K();
    Kb+=((KDigestor *) p[i][1])->get_K();
  }

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];
}

void ERIscreen::calcK(const arma::cx_mat & Pa, const arma::cx_mat & Pb, arma::cx_mat & Ka, arma::cx_mat & Kb, double tol) const {
  if(Pa.n_rows != Nbf || Pa.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pa.n_rows = " << Pa.n_rows << ", Pa.n_cols = " << Pa.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(Pb.n_rows != Nbf || Pb.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pb.n_rows = " << Pb.n_rows << ", Pb.n_cols = " << Pb.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(2);
    p[i][0]=new cxKDigestor(Pa);
    p[i][1]=new cxKDigestor(Pb);
  }

  // Do calculation
  arma::mat D=density_bounds(arma::abs(Pa)+arma::abs(Pb));
  calculate(p,D,tol);

  // Collect results
  Ka=((cxKDigestor *) p[0][0])->get_K();
  Kb=((cxKDigestor *) p[0][1])->get_K();
  for(int i=1;i<nth;i++) {
    Ka+=((cxKDigestor *) p[i][0])->get_K();
    Kb+=((cxKDigestor *) p[i][1])->get_K();
  }

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];
}

void ERIscreen::calcJK(const arma::mat & P, arma::mat & J, arma::mat & K, double tol) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(J.n_rows != Nbf || J.n_cols != Nbf) {
    J.zeros(Nbf,Nbf);
  }
  if(K.n_rows != Nbf || K.n_cols != Nbf) {
    K.zeros(Nbf,Nbf);
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(2);
    p[i][0]=new JDigestor(P);
    p[i][1]=new KDigestor(P);
  }

  // Do calculation
  arma::mat D=density_bounds(P);
  calculate(p,D,tol);

  // Collect results
  J=((JDigestor *) p[0][0])->get_J();
  K=((KDigestor *) p[0][1])->get_K();
  for(int i=1;i<nth;i++) {
    J+=((JDigestor *) p[i][0])->get_J();
    K+=((KDigestor *) p[i][1])->get_K();
  }

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];
}

void ERIscreen::calcJK(const arma::cx_mat & P, arma::mat & J, arma::cx_mat & K, double tol) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(J.n_rows != Nbf || J.n_cols != Nbf) {
    J.zeros(Nbf,Nbf);
  }
  if(K.n_rows != Nbf || K.n_cols != Nbf) {
    K.zeros(Nbf,Nbf);
  }


#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Real part of the density, hoisted so the JDigestor holds a
  // reference rather than each thread copying a temporary.
  arma::mat Preal(arma::real(P));

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(2);
    p[i][0]=new JDigestor(Preal);
    p[i][1]=new cxKDigestor(P);
  }

  // Do calculation
  arma::mat D=density_bounds(arma::abs(P));
  calculate(p,D,tol);

  // Collect results
  J=((JDigestor *) p[0][0])->get_J();
  K=((cxKDigestor *) p[0][1])->get_K();
  for(int i=1;i<nth;i++) {
    J+=((JDigestor *) p[i][0])->get_J();
    K+=((cxKDigestor *) p[i][1])->get_K();
  }

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];
}

void ERIscreen::calcJK(const arma::mat & Pa, const arma::mat & Pb, arma::mat & J, arma::mat & Ka, arma::mat & Kb, double tol) const {
  if(Pa.n_rows != Nbf || Pa.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pa.n_rows = " << Pa.n_rows << ", Pa.n_cols = " << Pa.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(Pb.n_rows != Nbf || Pb.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pb.n_rows = " << Pb.n_rows << ", Pb.n_cols = " << Pb.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(J.n_rows != Nbf || J.n_cols != Nbf) {
    J.zeros(Nbf,Nbf);
  }
  if(Ka.n_rows != Nbf || Ka.n_cols != Nbf) {
    Ka.zeros(Nbf,Nbf);
  }
  if(Kb.n_rows != Nbf || Kb.n_cols != Nbf) {
    Kb.zeros(Nbf,Nbf);
  }


#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Total density, hoisted so the JDigestor holds a reference rather
  // than each thread copying a temporary.
  arma::mat Psum(Pa+Pb);

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(3);
    p[i][0]=new JDigestor(Psum);
    p[i][1]=new KDigestor(Pa);
    p[i][2]=new KDigestor(Pb);
  }

  // Do calculation
  arma::mat D=density_bounds(arma::abs(Pa)+arma::abs(Pb));
  calculate(p,D,tol);

  // Collect results
  J=((JDigestor *) p[0][0])->get_J();
  Ka=((KDigestor *) p[0][1])->get_K();
  Kb=((KDigestor *) p[0][2])->get_K();
  for(int i=1;i<nth;i++) {
    J+=((JDigestor *) p[i][0])->get_J();
    Ka+=((KDigestor *) p[i][1])->get_K();
    Kb+=((KDigestor *) p[i][2])->get_K();
  }

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];
}

void ERIscreen::calcJK(const arma::cx_mat & Pa, const arma::cx_mat & Pb, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb, double tol) const {
  if(Pa.n_rows != Nbf || Pa.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pa.n_rows = " << Pa.n_rows << ", Pa.n_cols = " << Pa.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(Pb.n_rows != Nbf || Pb.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pb.n_rows = " << Pb.n_rows << ", Pb.n_cols = " << Pb.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(J.n_rows != Nbf || J.n_cols != Nbf) {
    J.zeros(Nbf,Nbf);
  }
  if(Ka.n_rows != Nbf || Ka.n_cols != Nbf) {
    Ka.zeros(Nbf,Nbf);
  }
  if(Kb.n_rows != Nbf || Kb.n_cols != Nbf) {
    Kb.zeros(Nbf,Nbf);
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Real part of the total density, hoisted so the JDigestor holds a
  // reference rather than each thread copying a temporary.
  arma::mat Preal(arma::real(Pa+Pb));

  // Get workers
  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(3);
    p[i][0]=new JDigestor(Preal);
    p[i][1]=new cxKDigestor(Pa);
    p[i][2]=new cxKDigestor(Pb);
  }

  // Do calculation
  arma::mat D=density_bounds(arma::abs(Pa)+arma::abs(Pb));
  calculate(p,D,tol);

  // Collect results
  J=((JDigestor *) p[0][0])->get_J();
  Ka=((cxKDigestor *) p[0][1])->get_K();
  Kb=((cxKDigestor *) p[0][2])->get_K();
  for(int i=1;i<nth;i++) {
    J+=((JDigestor *) p[i][0])->get_J();
    Ka+=((cxKDigestor *) p[i][1])->get_K();
    Kb+=((cxKDigestor *) p[i][2])->get_K();
  }

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];
}

std::vector<arma::cx_mat> ERIscreen::calcJK(const std::vector<arma::cx_mat> & P, double jfrac, double kfrac, double tol) const {
  for(size_t i=0;i<P.size();i++) {
    if(P[i].n_rows != Nbf || P[i].n_cols != Nbf) {
      std::ostringstream oss;
      oss << "Error in ERIscreen: Nbf = " << Nbf << ", P[" << i << "].n_rows = " << P[i].n_rows << ", P[" << i << "].n_cols = " << P[i].n_cols << "!\n";
      throw std::logic_error(oss.str());
    }
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  bool doj(jfrac!=0.0);
  bool dok(kfrac!=0.0);

  // Get workers
  // Real parts of the densities, hoisted so the JDigestors hold
  // references rather than each thread copying temporaries.
  std::vector<arma::mat> Preal;
  if(doj) {
    Preal.resize(P.size());
    for(size_t j=0;j<P.size();j++)
      Preal[j]=arma::real(P[j]);
  }

  std::vector< std::vector<IntegralDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    if(doj) {
      for(size_t j=0;j<P.size();j++)
	p[i].push_back(new JDigestor(Preal[j]));
    }
    if(dok) {
      for(size_t j=0;j<P.size();j++)
	p[i].push_back(new cxKDigestor(P[j]));
    }
  }

  // Do calculation. Density bound covers all input densities: |P[j]|
  // bounds each cxKDigestor and dominates each |real(P[j])|, so the
  // sum of moduli bounds the union.
  arma::mat D;
  if(!P.empty()) {
    arma::mat Pabs=arma::abs(P[0]);
    for(size_t j=1;j<P.size();j++)
      Pabs+=arma::abs(P[j]);
    D=density_bounds(Pabs);
  }
  calculate(p,D,tol);

  // Collect results
  std::vector<arma::cx_mat> JK(P.size());
  for(size_t i=0;i<JK.size();i++)
    JK[i].zeros(P[i].n_rows,P[i].n_cols);

  size_t joff=0;
  if(doj) {
    for(size_t j=0;j<P.size();j++)
      for(int i=0;i<nth;i++)
	JK[j]+=jfrac*((JDigestor *) p[i][j+joff])->get_J()*COMPLEX1;
    joff+=P.size();
  }
  // Exchange contribution
  if(dok) {
    for(size_t j=0;j<P.size();j++)
      for(int i=0;i<nth;i++)
	JK[j]-=kfrac*((cxKDigestor *) p[i][j+joff])->get_K();
    joff+=P.size();
  }

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return JK;
}

arma::vec ERIscreen::forceJ(const arma::mat & P, double tol) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<ForceDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(1);
    p[i][0]=new JFDigestor(P);
  }

  // Do calculation
  arma::vec f=calculate_force(p,tol);

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return f;
}

arma::vec ERIscreen::forceK(const arma::mat & P, double tol, double kfrac) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<ForceDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(1);
    p[i][0]=new KFDigestor(P,kfrac,true);
  }

  // Do calculation
  arma::vec f=calculate_force(p,tol);

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return f;
}

arma::vec ERIscreen::forceK(const arma::mat & Pa, const arma::mat & Pb, double tol, double kfrac) const {
  if(Pa.n_rows != Nbf || Pa.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pa.n_rows = " << Pa.n_rows << ", Pa.n_cols = " << Pa.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(Pb.n_rows != Nbf || Pb.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pb.n_rows = " << Pb.n_rows << ", Pb.n_cols = " << Pb.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Total density, hoisted so the JFDigestor holds a reference rather
  // than each thread copying a temporary.
  arma::mat Psum(Pa+Pb);

  // Get workers
  std::vector< std::vector<ForceDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(3);
    p[i][0]=new JFDigestor(Psum);
    p[i][1]=new KFDigestor(Pa,kfrac,false);
    p[i][2]=new KFDigestor(Pb,kfrac,false);
  }

  // Do calculation
  arma::vec f=calculate_force(p,tol);

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return f;
}

arma::vec ERIscreen::forceJK(const arma::mat & P, double tol, double kfrac) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector< std::vector<ForceDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(2);
    p[i][0]=new JFDigestor(P);
    p[i][1]=new KFDigestor(P,kfrac,true);
  }

  // Do calculation
  arma::vec f=calculate_force(p,tol);

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return f;
}

arma::vec ERIscreen::forceJK(const arma::mat & Pa, const arma::mat & Pb, double tol, double kfrac) const {
  if(Pa.n_rows != Nbf || Pa.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pa.n_rows = " << Pa.n_rows << ", Pa.n_cols = " << Pa.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
  if(Pb.n_rows != Nbf || Pb.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in ERIscreen: Nbf = " << Nbf << ", Pb.n_rows = " << Pb.n_rows << ", Pb.n_cols = " << Pb.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Total density, hoisted so the JFDigestor holds a reference rather
  // than each thread copying a temporary.
  arma::mat Psum(Pa+Pb);

  // Get workers
  std::vector< std::vector<ForceDigestor *> > p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++) {
    p[i].resize(3);
    p[i][0]=new JFDigestor(Psum);
    p[i][1]=new KFDigestor(Pa,kfrac,false);
    p[i][2]=new KFDigestor(Pb,kfrac,false);
  }

  // Do calculation
  arma::vec f=calculate_force(p,tol);

  // Free memory
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<p[i].size();j++)
      delete p[i][j];

  return f;
}
