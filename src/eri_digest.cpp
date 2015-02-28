/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "eri_digest.h"
#include "eriworker.h"

IntegralDigestor::IntegralDigestor() {
}

IntegralDigestor::~IntegralDigestor() {
}

JDigestor::JDigestor(const arma::mat & P_) : P(P_) {
  J.zeros(P.n_rows,P.n_cols);
}

JDigestor::~JDigestor() {
}

void JDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  // Shells in quartet are
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;

  // Amount of functions on the first pair is
  size_t Ni=shpairs[ip].Ni;
  size_t Nj=shpairs[ip].Nj;
  // and on second pair is
  size_t Nk=shpairs[jp].Ni;
  size_t Nl=shpairs[jp].Nj;

  // First functions on the first pair is
  size_t i0=shpairs[ip].i0;
  size_t j0=shpairs[ip].j0;
  // Second pair is
  size_t k0=shpairs[jp].i0;
  size_t l0=shpairs[jp].j0;

  // J_ij = (ij|kl) P_kl
  {
    // Work matrix
    arma::mat Jij(Ni,Nj);
    Jij.zeros();
    arma::mat Pkl=P.submat(k0,l0,k0+Nk-1,l0+Nl-1);

    // Degeneracy factor
    double fac=1.0;
    if(ks!=ls)
      fac=2.0;

    // Increment matrix
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t jj=0;jj<Nj;jj++) {

	// Matrix element
	double el=0.0;
	for(size_t kk=0;kk<Nk;kk++)
	  for(size_t ll=0;ll<Nl;ll++)
	    el+=Pkl(kk,ll)*ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll];

	// Set the element
	Jij(ii,jj)+=fac*el;
      }

    // Store the matrix element
    J.submat(i0,j0,i0+Ni-1,j0+Nj-1)+=Jij;
    if(is!=js)
      J.submat(j0,i0,j0+Nj-1,i0+Ni-1)+=arma::trans(Jij);
  }

  // Permutation: J_kl = (ij|kl) P_ij
  if(ip!=jp) {
    // Work matrix
    arma::mat Jkl(Nk,Nl);
    Jkl.zeros();
    arma::mat Pij=P.submat(i0,j0,i0+Ni-1,j0+Nj-1);

    // Degeneracy factor
    double fac=1.0;
    if(is!=js)
      fac=2.0;

    // Increment matrix
    for(size_t kk=0;kk<Nk;kk++)
      for(size_t ll=0;ll<Nl;ll++) {

	// Matrix element
	double el=0.0;
	for(size_t ii=0;ii<Ni;ii++) {
	  for(size_t jj=0;jj<Nj;jj++) {
	    el+=Pij(ii,jj)*ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll];
	  }
	}

	// Set the element
	Jkl(kk,ll)+=fac*el;
      }

    // Store the matrix element
    J.submat(k0,l0,k0+Nk-1,l0+Nl-1)+=Jkl;
    if(ks!=ls)
      J.submat(l0,k0,l0+Nl-1,k0+Nk-1)+=arma::trans(Jkl);
  }
}

arma::mat JDigestor::get_J() const {
  return J;
}

KDigestor::KDigestor(const arma::mat & P_) : P(P_) {
  K.zeros(P.n_rows,P.n_cols);
}

KDigestor::~KDigestor() {
}

void KDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;

  // Amount of functions on the first pair is
  size_t Ni=shpairs[ip].Ni;
  size_t Nj=shpairs[ip].Nj;
  // and on second pair is
  size_t Nk=shpairs[jp].Ni;
  size_t Nl=shpairs[jp].Nj;

  // First functions on the first pair is
  size_t i0=shpairs[ip].i0;
  size_t j0=shpairs[ip].j0;
  // Second pair is
  size_t k0=shpairs[jp].i0;
  size_t l0=shpairs[jp].j0;

  /*
    When all indices are different, the
    following integrals are equivalent:
    (ij|kl) (ij|lk) (ji|kl) (ji|lk)
    (kl|ij) (kl|ji) (lk|ij) (lk|ji)

    This translates to

    K(i,k) += (ij|kl) P(j,l) // always
    K(j,k) += (ij|kl) P(i,l) // if (is!=js)
    K(i,l) += (ij|kl) P(j,k) // if (ls!=ks)
    K(j,l) += (ij|kl) P(i,k) // if (is!=js) and (ls!=ks)

    and for ij != kl

    K(k,i) += (ij|kl) P(j,l) // always
    K(k,j) += (ij|kl) P(i,l) // if (is!=js)
    K(l,i) += (ij|kl) P(j,k) // if (ks!=ls)
    K(l,j) += (ij|kl) P(i,k) // if (is!=js) and (ks!=ls)

    However, the latter four permutations just make the
    exchange matrix symmetric. So the only thing we need to do
    is do the first four permutations, and at the end we sum up
    K_ij and K_ji for j>i and set K_ij and K_ji to this
    value. This makes things a *lot* easier. So:
    We just need to check if the shells are different, in which
    case K will get extra increments.
  */

  // First, do the ik part: K(i,k) += (ij|kl) P(j,l)
  {
    arma::mat Kik(Ni,Nk);
    Kik.zeros();
    arma::mat Pjl =P.submat(j0,l0,j0+Nj-1,l0+Nl-1);

    // Increment Kik
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t jj=0;jj<Nj;jj++)
	    Kik (ii,kk)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pjl (jj,ll);

    // Set elements
    K.submat(i0,k0,i0+Ni-1,k0+Nk-1)+=Kik;
    // Symmetrize if necessary
    if(ip!=jp)
      K.submat(k0,i0,k0+Nk-1,i0+Ni-1)+=arma::trans(Kik);
  }

  // Then, the second part: K(j,k) += (ij|kl) P(i,l)
  if(is!=js) {
    arma::mat Kjk(Nj,Nk);
    Kjk.zeros();
    arma::mat Pil=P.submat(i0,l0,i0+Ni-1,l0+Nl-1);

    // Increment Kjk
    for(size_t jj=0;jj<Nj;jj++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t ii=0;ii<Ni;ii++)
	    Kjk(jj,kk)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pil(ii,ll);

    // Set elements
    K.submat(j0,k0,j0+Nj-1,k0+Nk-1)+=Kjk;
    // Symmetrize if necessary (take care about possible overlap with next routine)
    if(ip!=jp) {
      K.submat(k0,j0,k0+Nk-1,j0+Nj-1)+=arma::trans(Kjk);
    }
  }

  // Third part: K(i,l) += (ij|kl) P(j,k)
  if(ks!=ls) {
    arma::mat Kil(Ni,Nl);
    Kil.zeros();
    arma::mat Pjk=P.submat(j0,k0,j0+Nj-1,k0+Nk-1);

    // Increment Kil
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t jj=0;jj<Nj;jj++)
	  for(size_t kk=0;kk<Nk;kk++) {
	    Kil(ii,ll)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pjk(jj,kk);
	  }

    // Set elements
    K.submat(i0,l0,i0+Ni-1,l0+Nl-1)+=Kil;
    // Symmetrize if necessary
    if(ip!=jp)
      K.submat(l0,i0,l0+Nl-1,i0+Ni-1)+=arma::trans(Kil);
  }

  // Last permutation: K(j,l) += (ij|kl) P(i,k)
  if(is!=js && ks!=ls) {
    arma::mat Kjl(Nj,Nl);
    Kjl.zeros();
    arma::mat Pik=P.submat(i0,k0,i0+Ni-1,k0+Nk-1);

    // Increment Kjl
    for(size_t jj=0;jj<Nj;jj++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t kk=0;kk<Nk;kk++) {
	    Kjl(jj,ll)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pik(ii,kk);
	  }

    // Set elements
    K.submat(j0,l0,j0+Nj-1,l0+Nl-1)+=Kjl;
    // Symmetrize if necessary
    if (ip!=jp)
      K.submat(l0,j0,l0+Nl-1,j0+Nj-1)+=arma::trans(Kjl);
  }
}

arma::mat KDigestor::get_K() const {
  return K;
}

cxKDigestor::cxKDigestor(const arma::cx_mat & P_) : P(P_) {
  K.zeros(P.n_rows,P.n_cols);
}

cxKDigestor::~cxKDigestor() {
}

void cxKDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;

  // Amount of functions on the first pair is
  size_t Ni=shpairs[ip].Ni;
  size_t Nj=shpairs[ip].Nj;
  // and on second pair is
  size_t Nk=shpairs[jp].Ni;
  size_t Nl=shpairs[jp].Nj;

  // First functions on the first pair is
  size_t i0=shpairs[ip].i0;
  size_t j0=shpairs[ip].j0;
  // Second pair is
  size_t k0=shpairs[jp].i0;
  size_t l0=shpairs[jp].j0;

  /*
    When all indices are different, the
    following integrals are equivalent:
    (ij|kl) (ij|lk) (ji|kl) (ji|lk)
    (kl|ij) (kl|ji) (lk|ij) (lk|ji)

    This translates to

    K(i,k) += (ij|kl) P(j,l) // always
    K(j,k) += (ij|kl) P(i,l) // if (is!=js)
    K(i,l) += (ij|kl) P(j,k) // if (ls!=ks)
    K(j,l) += (ij|kl) P(i,k) // if (is!=js) and (ls!=ks)

    and for ij != kl

    K(k,i) += (ij|kl) P(j,l) // always
    K(k,j) += (ij|kl) P(i,l) // if (is!=js)
    K(l,i) += (ij|kl) P(j,k) // if (ks!=ls)
    K(l,j) += (ij|kl) P(i,k) // if (is!=js) and (ks!=ls)

    However, the latter four permutations just make the
    exchange matrix symmetric. So the only thing we need to do
    is do the first four permutations, and at the end we sum up
    K_ij and K_ji for j>i and set K_ij and K_ji to this
    value. This makes things a *lot* easier. So:
    We just need to check if the shells are different, in which
    case K will get extra increments.
  */

  // First, do the ik part: K(i,k) += (ij|kl) P(j,l)
  {
    arma::cx_mat Kik(Ni,Nk);
    Kik.zeros();
    arma::cx_mat Pjl =P.submat(j0,l0,j0+Nj-1,l0+Nl-1);

    // Increment Kik
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t jj=0;jj<Nj;jj++)
	    Kik (ii,kk)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pjl (jj,ll);

    // Set elements
    K.submat(i0,k0,i0+Ni-1,k0+Nk-1)+=Kik;
    // Symmetrize if necessary
    if(ip!=jp)
      K.submat(k0,i0,k0+Nk-1,i0+Ni-1)+=arma::trans(Kik);
  }

  // Then, the second part: K(j,k) += (ij|kl) P(i,l)
  if(is!=js) {
    arma::cx_mat Kjk(Nj,Nk);
    Kjk.zeros();
    arma::cx_mat Pil=P.submat(i0,l0,i0+Ni-1,l0+Nl-1);

    // Increment Kjk
    for(size_t jj=0;jj<Nj;jj++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t ii=0;ii<Ni;ii++)
	    Kjk(jj,kk)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pil(ii,ll);

    // Set elements
    K.submat(j0,k0,j0+Nj-1,k0+Nk-1)+=Kjk;
    // Symmetrize if necessary (take care about possible overlap with next routine)
    if(ip!=jp) {
      K.submat(k0,j0,k0+Nk-1,j0+Nj-1)+=arma::trans(Kjk);
    }
  }

  // Third part: K(i,l) += (ij|kl) P(j,k)
  if(ks!=ls) {
    arma::cx_mat Kil(Ni,Nl);
    Kil.zeros();
    arma::cx_mat Pjk=P.submat(j0,k0,j0+Nj-1,k0+Nk-1);

    // Increment Kil
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t jj=0;jj<Nj;jj++)
	  for(size_t kk=0;kk<Nk;kk++) {
	    Kil(ii,ll)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pjk(jj,kk);
	  }

    // Set elements
    K.submat(i0,l0,i0+Ni-1,l0+Nl-1)+=Kil;
    // Symmetrize if necessary
    if(ip!=jp)
      K.submat(l0,i0,l0+Nl-1,i0+Ni-1)+=arma::trans(Kil);
  }

  // Last permutation: K(j,l) += (ij|kl) P(i,k)
  if(is!=js && ks!=ls) {
    arma::cx_mat Kjl(Nj,Nl);
    Kjl.zeros();
    arma::cx_mat Pik=P.submat(i0,k0,i0+Ni-1,k0+Nk-1);

    // Increment Kjl
    for(size_t jj=0;jj<Nj;jj++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t kk=0;kk<Nk;kk++) {
	    Kjl(jj,ll)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pik(ii,kk);
	  }

    // Set elements
    K.submat(j0,l0,j0+Nj-1,l0+Nl-1)+=Kjl;
    // Symmetrize if necessary
    if (ip!=jp)
      K.submat(l0,j0,l0+Nl-1,j0+Nj-1)+=arma::trans(Kjl);
  }
}

arma::cx_mat cxKDigestor::get_K() const {
  return K;
}

ForceDigestor::ForceDigestor() {
}

ForceDigestor::~ForceDigestor() {
}

JFDigestor::JFDigestor(const arma::mat & P_) : P(P_) {
}

JFDigestor::~JFDigestor() {
}

void JFDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, dERIWorker & deriw, arma::vec & f) {
  // Shells in question are
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;

  // Amount of functions on the first pair is
  size_t Ni=shpairs[ip].Ni;
  size_t Nj=shpairs[ip].Nj;
  // and on second pair is
  size_t Nk=shpairs[jp].Ni;
  size_t Nl=shpairs[jp].Nj;

  // First functions on the first pair is
  size_t i0=shpairs[ip].i0;
  size_t j0=shpairs[ip].j0;
  // Second pair is
  size_t k0=shpairs[jp].i0;
  size_t l0=shpairs[jp].j0;

  // E_J = P_ij (ij|kl) P_kl. Work matrices
  arma::mat Pij=P.submat(i0,j0,i0+Ni-1,j0+Nj-1);
  arma::mat Pkl=P.submat(k0,l0,k0+Nk-1,l0+Nl-1);

  // Degeneracy factor
  double Jfac=-0.5;
  if(is!=js)
    Jfac*=2.0;
  if(ks!=ls)
    Jfac*=2.0;
  if(ip!=jp)
    Jfac*=2.0;

  // Increment the forces.
  for(int idx=0;idx<12;idx++) {
    // Get the integral derivatives
    const std::vector<double> *erip=deriw.getp(idx);

    // E_J = P_ij (ij|kl) P_kl
    double el=0.0;
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t jj=0;jj<Nj;jj++)
	for(size_t kk=0;kk<Nk;kk++)
	  for(size_t ll=0;ll<Nl;ll++)
	    el+=Pij(ii,jj)*Pkl(kk,ll)*(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
    
    // Increment the element
    f(idx)+=Jfac*el;
  }
}

KFDigestor::KFDigestor(const arma::mat & P_, double kfrac_, bool restr) : P(P_), kfrac(kfrac_) {
  fac = restr ? 0.5 : 1.0;
}

KFDigestor::~KFDigestor() {
}

void KFDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, dERIWorker & deriw, arma::vec & f) {
  // Shells on quartet are
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;

  // Amount of functions on the first pair is
  size_t Ni=shpairs[ip].Ni;
  size_t Nj=shpairs[ip].Nj;
  // and on second pair is
  size_t Nk=shpairs[jp].Ni;
  size_t Nl=shpairs[jp].Nj;

  // First functions on the first pair is
  size_t i0=shpairs[ip].i0;
  size_t j0=shpairs[ip].j0;
  // Second pair is
  size_t k0=shpairs[jp].i0;
  size_t l0=shpairs[jp].j0;

  // E_K = P_ik (ij|kl) P_jl
  arma::mat Pik=P.submat(i0,k0,i0+Ni-1,k0+Nk-1);
  arma::mat Pjl=P.submat(j0,l0,j0+Nj-1,l0+Nl-1);
  //     + P_jk (ij|kl) P_il
  arma::mat Pjk=P.submat(j0,k0,j0+Nj-1,k0+Nk-1);
  arma::mat Pil=P.submat(i0,l0,i0+Ni-1,l0+Nl-1);
  double K1fac, K2fac;
  if(is!=js && ks!=ls) {
    // Get both twice.
    K1fac=1.0;
    K2fac=1.0;
  } else if(is==js && ks==ls) {
    // Only get the first one, once.
    K1fac=0.5;
    K2fac=0.0;
  } else {
    // Get both once.
    K1fac=0.5;
    K2fac=0.5;
  }
  // Switch symmetry
  if(ip!=jp) {
    K1fac*=2.0;
    K2fac*=2.0;
  }
  // Restricted calculation?
  K1fac*=fac*kfrac;
  K2fac*=fac*kfrac;

  // Increment the forces.
  for(int idx=0;idx<12;idx++) {
    // Get the integral derivatives
    const std::vector<double> * erip=deriw.getp(idx);

    // E_K = P_ik (ij|kl) P_jl
    double el=0.0;
    // Increment matrix
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t jj=0;jj<Nj;jj++)
	for(size_t kk=0;kk<Nk;kk++)
	  for(size_t ll=0;ll<Nl;ll++)
	    el+=Pik(ii,kk)*Pjl(jj,ll)*(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
    
    // Increment the element
    f(idx)+=K1fac*el;
    
    // Second contribution
    if(K2fac!=0.0) {
      el=0.0;
      // Increment matrix
      for(size_t ii=0;ii<Ni;ii++)
	for(size_t jj=0;jj<Nj;jj++)
	  for(size_t kk=0;kk<Nk;kk++)
	    for(size_t ll=0;ll<Nl;ll++)
	      el+=Pjk(jj,kk)*Pil(ii,ll)*(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
      
      // Increment the element
      f(idx)+=K2fac*el;
    }
  }
}
