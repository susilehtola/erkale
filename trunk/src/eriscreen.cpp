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

size_t ERIscreen::get_N() const {
  return basp ? basp->get_Nbf() : 0;
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

size_t ERIscreen::fill(const BasisSet * basisv, double shtol, bool verbose) {
  // Form screening table.
  if(basisv==NULL)
    return 0;

  basp=basisv;

  // Form index helper
  iidx=i_idx(basp->get_Nbf());

  // Screening matrix and pairs
  shpairs=basp->get_eripairs(screen,shtol,omega,alpha,beta,verbose);

  return shpairs.size();
}

void ERIscreen::calculate(std::vector<IntegralDigestor *> digest, double tol) const {
  // Shells in basis set
  std::vector<GaussianShell> shells=basp->get_shells();
  // Get number of shell pairs
  const size_t Npairs=shpairs.size();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // ERI worker
    ERIWorker *eri = (omega==0.0 && alpha==1.0 && beta==0.0) ? new ERIWorker(basp->get_max_am(),basp->get_max_Ncontr()) : new ERIWorker_srlr(basp->get_max_am(),basp->get_max_Ncontr(),omega,alpha,beta);
    
    // Integral array
    const std::vector<double> * erip;

#ifndef _OPENMP
    int ith=0;
#else
    int ith(omp_get_thread_num());
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

	{
	  // Maximum value of the 2-electron integrals on this shell pair
	  double intmax=screen(is,js)*screen(ks,ls);
	  if(intmax<tol) {
	    // Skip due to small value of integral. Because the
	    // integrals have been ordered, all the next ones will be
	    // small as well!
	    break;
	  }
	}

	// Compute integrals
	eri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	erip=eri->getp();

	// Digest the integrals
	digest[ith]->digest(shpairs,ip,jp,*erip,0);
      }
    }

    delete eri;
  }
}

void ERIscreen::calculate_force(std::vector<IntegralDigestor *> digest, double tol) const {
  // Shells
  std::vector<GaussianShell> shells=basp->get_shells();
  // Get number of shell pairs
  const size_t Npairs=shpairs.size();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    /// ERI derivative worker
    dERIWorker *deri = (omega==0.0 && alpha==1.0 && beta==0.0) ? new dERIWorker(basp->get_max_am(),basp->get_max_Ncontr()) : new dERIWorker_srlr(basp->get_max_am(),basp->get_max_Ncontr(),omega,alpha,beta);

    /// Integral array
    //    const std::vector<double> * erip;
    (void) digest;

    // Shell centers
    size_t inuc, jnuc, knuc, lnuc;

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
	
	{
	  // Maximum value of the 2-electron integrals on this shell pair
	  double intmax=screen(is,js)*screen(ks,ls);
	  if(intmax<tol) {
	    // Skip due to small value of integral. Because the
	    // integrals have been ordered, all the next ones will be
	    // small as well!
	    break;
	  }
	}
	
	// Compute the derivatives.
	deri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);

	// Forces on the nuclei
	arma::vec f(12);
	f.zeros();
	// Digest into forces
	
      }
    }

    delete deri;
  }
}

arma::mat ERIscreen::calcJ(const arma::mat & P, double tol) const {
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector<IntegralDigestor *> p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++)
    p[i]=new JDigestor(P);

  // Do calculation
  calculate(p,tol);

  // Collect results
  arma::mat J(((JDigestor *) p[0])->get_J());
  for(int i=1;i<nth;i++)
    J+=((JDigestor *) p[i])->get_J();

  // Free memory
  for(int i=0;i<nth;i++)
    delete p[i];
  
  return J;
}

arma::mat ERIscreen::calcK(const arma::mat & P, double tol) const {
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector<IntegralDigestor *> p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++)
    p[i]=new KDigestor(P);

  // Do calculation
  calculate(p,tol);

  // Collect results
  arma::mat K(((KDigestor *) p[0])->get_K());
  for(int i=1;i<nth;i++)
    K+=((KDigestor *) p[i])->get_K();

  // Free memory
  for(int i=0;i<nth;i++)
    delete p[i];
  
  return K;
}

void ERIscreen::calcK(const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ka, arma::mat & Kb, double tol) const {
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector<IntegralDigestor *> p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++)
    p[i]=new KabDigestor(Pa,Pb);

  // Do calculation
  calculate(p,tol);

  // Collect results
  Ka=((KabDigestor *) p[0])->get_Ka();
  Kb=((KabDigestor *) p[0])->get_Kb();
  for(int i=1;i<nth;i++) {
    Ka+=((KabDigestor *) p[i])->get_Ka();
    Kb+=((KabDigestor *) p[i])->get_Kb();
  }

  // Free memory
  for(int i=0;i<nth;i++)
    delete p[i];
}

void ERIscreen::calcJK(const arma::mat & P, arma::mat & J, arma::mat & K, double tol) const {
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector<IntegralDigestor *> p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++)
    p[i]=new JKDigestor(P);

  // Do calculation
  calculate(p,tol);

  // Collect results
  J=((JKDigestor *) p[0])->get_J();
  K=((JKDigestor *) p[0])->get_K();
  for(int i=1;i<nth;i++) {
    J+=((JKDigestor *) p[i])->get_J();
    K+=((JKDigestor *) p[i])->get_K();
  }

  // Free memory
  for(int i=0;i<nth;i++)
    delete p[i];
}

void ERIscreen::calcJK(const arma::mat & Pa, const arma::mat & Pb, arma::mat & J, arma::mat & Ka, arma::mat & Kb, double tol) const {
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector<IntegralDigestor *> p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++)
    p[i]=new JKabDigestor(Pa,Pb);

  // Do calculation
  calculate(p,tol);

  // Collect results
  J=((JKabDigestor *) p[0])->get_J();
  Ka=((JKabDigestor *) p[0])->get_Ka();
  Kb=((JKabDigestor *) p[0])->get_Kb();
  for(int i=1;i<nth;i++) {
    J+=((JKabDigestor *) p[i])->get_J();
    Ka+=((JKabDigestor *) p[i])->get_Ka();
    Kb+=((JKabDigestor *) p[i])->get_Kb();
  }

  // Free memory
  for(int i=0;i<nth;i++)
    delete p[i];
}

std::vector<arma::mat> ERIscreen::calcJ(const std::vector<arma::mat> & P, double tol) const {
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  // Get workers
  std::vector<IntegralDigestor *> p(nth);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0;i<nth;i++)
    p[i]=new JvDigestor(P);

  // Do calculation
  calculate(p,tol);

  // Collect results
  std::vector<arma::mat> J(((JvDigestor *) p[0])->get_J());
  for(int i=1;i<nth;i++) {
    std::vector<arma::mat> Jt(((JvDigestor *) p[i])->get_J());
    for(size_t j=0;j<J.size();j++)
      J[j]+=Jt[j];
  }

  // Free memory
  for(int i=0;i<nth;i++)
    delete p[i];
  
  return J;
}

arma::vec ERIscreen::forceJ(const arma::mat & P, double tol) const {
  throw std::runtime_error("Force temporarily not available!\n");
  (void) P;
  (void) tol;
  
  arma::vec f;
  return f;
}

arma::vec ERIscreen::forceK(const arma::mat & P, double tol) const {
  throw std::runtime_error("Force temporarily not available!\n");
  (void) P;
  (void) tol;

  arma::vec f;
  return f;
}

void ERIscreen::forceK(const arma::mat & Pa, const arma::mat & Pb, arma::vec & fKa, arma::vec & fKb, double tol) const {
  throw std::runtime_error("Force temporarily not available!\n");
  (void) Pa;
  (void) Pb;
  (void) fKa;
  (void) fKb;
  (void) tol;
}

void ERIscreen::forceJK(const arma::mat & P, arma::vec & fJ, arma::vec & fK, double tol) const {
  throw std::runtime_error("Force temporarily not available!\n");
  (void) P;
  (void) fJ;
  (void) fK;
  (void) tol;
}

void ERIscreen::forceJK(const arma::mat & Pa, const arma::mat & Pb, arma::vec & fJ, arma::vec & fKa, arma::vec & fKb, double tol) const {
  throw std::runtime_error("Force temporarily not available!\n");
  (void) Pa;
  (void) Pb;
  (void) fJ;
  (void) fKa;
  (void) fKb;
  (void) tol;
}
