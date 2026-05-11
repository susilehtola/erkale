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

  // The (μν|λσ) integrals are stored in C-order with σ varying
  // fastest:  ints[ioff + ((ii*Nj+jj)*Nk+kk)*Nl+ll]. Viewed as a
  // column-major matrix with row index r = (kk*Nl+ll) and column
  // index c = (ii*Nj+jj), this matches the offset c*(Nk*Nl) + r,
  // so we can wrap it as an Nk*Nl × Ni*Nj armadillo matrix and let
  // BLAS GEMV do the contractions.
  const arma::mat ints_view(const_cast<double*>(&ints[ioff]), Nk*Nl, Ni*Nj, false, true);

  // J_ij = (ij|kl) P_kl. Reuses the digestor-member scratch
  // buffers; set_size grows them monotonically. Replaces the
  // previous arma::mat Pkl submat copy + vectorise(Pkl.t()) +
  // arma::vec rv + arma::reshape(rv, ...).t() + arma::mat Jij
  // chain, which allocated ~5 arma containers per quartet.
  {
    const double fac = (ks!=ls) ? 2.0 : 1.0;

    // Row-major flat of Pkl: scratch_Pflat(kk*Nl+ll) = P(k0+kk, l0+ll).
    // ints_view.t() * scratch_Pflat then produces rv(ii*Nj+jj) =
    // Σ_kl (ij|kl) P(k,l) via BLAS GEMV into pre-sized scratch_rv.
    scratch_Pflat.set_size(Nk * Nl);
    for(size_t kk=0; kk<Nk; kk++)
      for(size_t ll=0; ll<Nl; ll++)
        scratch_Pflat(kk*Nl + ll) = P(k0+kk, l0+ll);
    scratch_rv.set_size(Ni * Nj);
    scratch_rv = ints_view.t() * scratch_Pflat;

    // Direct element-wise accumulate into J -- avoids the reshape
    // + transpose + Jij temp + J.submat(...) += Jij chain.
    for(size_t ii=0; ii<Ni; ii++)
      for(size_t jj=0; jj<Nj; jj++) {
        const double v = fac * scratch_rv(ii*Nj + jj);
        J(i0+ii, j0+jj) += v;
        if(is!=js)
          J(j0+jj, i0+ii) += v;
      }
  }

  // J_kl = (ij|kl) P_ij; same trick, contracting over (ij).
  if(ip!=jp) {
    const double fac = (is!=js) ? 2.0 : 1.0;

    scratch_Pflat.set_size(Ni * Nj);
    for(size_t ii=0; ii<Ni; ii++)
      for(size_t jj=0; jj<Nj; jj++)
        scratch_Pflat(ii*Nj + jj) = P(i0+ii, j0+jj);
    scratch_rv.set_size(Nk * Nl);
    scratch_rv = ints_view * scratch_Pflat;

    for(size_t kk=0; kk<Nk; kk++)
      for(size_t ll=0; ll<Nl; ll++) {
        const double v = fac * scratch_rv(kk*Nl + ll);
        J(k0+kk, l0+ll) += v;
        if(ks!=ls)
          J(l0+ll, k0+kk) += v;
      }
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

  // K(i,k) += (ij|kl) P(j,l). The Kik accumulator and the Pjl
  // density slice both go through member-owned scratch storage
  // (set_size grows monotonically; no allocation after the first
  // largest shellpair has been seen). The Pjl materialisation gives
  // the inner loop contiguous-array access rather than going through
  // an arma::subview operator() per element.
  {
    scratch_Kik.set_size(Ni, Nk);
    scratch_Kik.zeros();
    scratch_Pjl.set_size(Nj, Nl);
    scratch_Pjl = P.submat(j0, l0, j0+Nj-1, l0+Nl-1);

    for(size_t ii=0;ii<Ni;ii++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t jj=0;jj<Nj;jj++)
	    scratch_Kik(ii,kk) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pjl(jj,ll);

    K.submat(i0,k0,i0+Ni-1,k0+Nk-1) += scratch_Kik;
    if(ip!=jp)
      K.submat(k0,i0,k0+Nk-1,i0+Ni-1) += arma::trans(scratch_Kik);
  }

  // K(j,k) += (ij|kl) P(i,l)
  if(is!=js) {
    scratch_Kjk.set_size(Nj, Nk);
    scratch_Kjk.zeros();
    scratch_Pil.set_size(Ni, Nl);
    scratch_Pil = P.submat(i0, l0, i0+Ni-1, l0+Nl-1);

    for(size_t jj=0;jj<Nj;jj++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t ii=0;ii<Ni;ii++)
	    scratch_Kjk(jj,kk) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pil(ii,ll);

    K.submat(j0,k0,j0+Nj-1,k0+Nk-1) += scratch_Kjk;
    if(ip!=jp)
      K.submat(k0,j0,k0+Nk-1,j0+Nj-1) += arma::trans(scratch_Kjk);
  }

  // K(i,l) += (ij|kl) P(j,k)
  if(ks!=ls) {
    scratch_Kil.set_size(Ni, Nl);
    scratch_Kil.zeros();
    scratch_Pjk.set_size(Nj, Nk);
    scratch_Pjk = P.submat(j0, k0, j0+Nj-1, k0+Nk-1);

    for(size_t ii=0;ii<Ni;ii++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t jj=0;jj<Nj;jj++)
	  for(size_t kk=0;kk<Nk;kk++)
	    scratch_Kil(ii,ll) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pjk(jj,kk);

    K.submat(i0,l0,i0+Ni-1,l0+Nl-1) += scratch_Kil;
    if(ip!=jp)
      K.submat(l0,i0,l0+Nl-1,i0+Ni-1) += arma::trans(scratch_Kil);
  }

  // K(j,l) += (ij|kl) P(i,k)
  if(is!=js && ks!=ls) {
    scratch_Kjl.set_size(Nj, Nl);
    scratch_Kjl.zeros();
    scratch_Pik.set_size(Ni, Nk);
    scratch_Pik = P.submat(i0, k0, i0+Ni-1, k0+Nk-1);

    for(size_t jj=0;jj<Nj;jj++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t kk=0;kk<Nk;kk++)
	    scratch_Kjl(jj,ll) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pik(ii,kk);

    K.submat(j0,l0,j0+Nj-1,l0+Nl-1) += scratch_Kjl;
    if (ip!=jp)
      K.submat(l0,j0,l0+Nl-1,j0+Nj-1) += arma::trans(scratch_Kjl);
  }
}

arma::mat KDigestor::get_K() const {
  return K;
}

cxKDigestor::cxKDigestor(const arma::cx_mat & P_) : P(P_) {
  // The symmetrisation in digest() uses arma::trans (Hermitian
  // conjugate) on the off-diagonal Kij blocks; this is correct only
  // when P is Hermitian, which is the case for HF/hybrid exchange in
  // ERKALE today. If you ever need to feed a non-Hermitian P here,
  // the K(k,i)/K(k,j)/K(l,i)/K(l,j) blocks must be computed directly
  // rather than recovered from arma::trans of the corresponding K(i,*)
  // block.
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

  // K(i,k) += (ij|kl) P(j,l). Same K/P scratch pattern as KDigestor.
  {
    scratch_Kik.set_size(Ni, Nk);
    scratch_Kik.zeros();
    scratch_Pjl.set_size(Nj, Nl);
    scratch_Pjl = P.submat(j0, l0, j0+Nj-1, l0+Nl-1);

    for(size_t ii=0;ii<Ni;ii++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t jj=0;jj<Nj;jj++)
	    scratch_Kik(ii,kk) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pjl(jj,ll);

    K.submat(i0,k0,i0+Ni-1,k0+Nk-1) += scratch_Kik;
    if(ip!=jp)
      K.submat(k0,i0,k0+Nk-1,i0+Ni-1) += arma::trans(scratch_Kik);
  }

  // K(j,k) += (ij|kl) P(i,l)
  if(is!=js) {
    scratch_Kjk.set_size(Nj, Nk);
    scratch_Kjk.zeros();
    scratch_Pil.set_size(Ni, Nl);
    scratch_Pil = P.submat(i0, l0, i0+Ni-1, l0+Nl-1);

    for(size_t jj=0;jj<Nj;jj++)
      for(size_t kk=0;kk<Nk;kk++)
	for(size_t ll=0;ll<Nl;ll++)
	  for(size_t ii=0;ii<Ni;ii++)
	    scratch_Kjk(jj,kk) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pil(ii,ll);

    K.submat(j0,k0,j0+Nj-1,k0+Nk-1) += scratch_Kjk;
    if(ip!=jp)
      K.submat(k0,j0,k0+Nk-1,j0+Nj-1) += arma::trans(scratch_Kjk);
  }

  // K(i,l) += (ij|kl) P(j,k)
  if(ks!=ls) {
    scratch_Kil.set_size(Ni, Nl);
    scratch_Kil.zeros();
    scratch_Pjk.set_size(Nj, Nk);
    scratch_Pjk = P.submat(j0, k0, j0+Nj-1, k0+Nk-1);

    for(size_t ii=0;ii<Ni;ii++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t jj=0;jj<Nj;jj++)
	  for(size_t kk=0;kk<Nk;kk++)
	    scratch_Kil(ii,ll) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pjk(jj,kk);

    K.submat(i0,l0,i0+Ni-1,l0+Nl-1) += scratch_Kil;
    if(ip!=jp)
      K.submat(l0,i0,l0+Nl-1,i0+Ni-1) += arma::trans(scratch_Kil);
  }

  // K(j,l) += (ij|kl) P(i,k)
  if(is!=js && ks!=ls) {
    scratch_Kjl.set_size(Nj, Nl);
    scratch_Kjl.zeros();
    scratch_Pik.set_size(Ni, Nk);
    scratch_Pik = P.submat(i0, k0, i0+Ni-1, k0+Nk-1);

    for(size_t jj=0;jj<Nj;jj++)
      for(size_t ll=0;ll<Nl;ll++)
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t kk=0;kk<Nk;kk++)
	    scratch_Kjl(jj,ll) += ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll] * scratch_Pik(ii,kk);

    K.submat(j0,l0,j0+Nj-1,l0+Nl-1) += scratch_Kjl;
    if (ip!=jp)
      K.submat(l0,j0,l0+Nl-1,j0+Nj-1) += arma::trans(scratch_Kjl);
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

  // E_J = P_ij (ij|kl) P_kl. Materialise the P submatrices into
  // member-owned scratch (set_size grows monotonically) -- avoids
  // two fresh arma::mat allocations per shellpair quartet.
  scratch_Pij.set_size(Ni, Nj);
  for(size_t jj=0; jj<Nj; jj++)
    for(size_t ii=0; ii<Ni; ii++)
      scratch_Pij(ii, jj) = P(i0+ii, j0+jj);
  scratch_Pkl.set_size(Nk, Nl);
  for(size_t ll=0; ll<Nl; ll++)
    for(size_t kk=0; kk<Nk; kk++)
      scratch_Pkl(kk, ll) = P(k0+kk, l0+ll);

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
	    el+=scratch_Pij(ii,jj)*scratch_Pkl(kk,ll)*(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];

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

  // E_K = P_ik (ij|kl) P_jl  +  P_jk (ij|kl) P_il.
  // Materialise the four P submatrices into member-owned scratch
  // (set_size grows monotonically) -- avoids four fresh arma::mat
  // allocations per shellpair quartet.
  scratch_Pik.set_size(Ni, Nk);
  for(size_t kk=0; kk<Nk; kk++)
    for(size_t ii=0; ii<Ni; ii++)
      scratch_Pik(ii, kk) = P(i0+ii, k0+kk);
  scratch_Pjl.set_size(Nj, Nl);
  for(size_t ll=0; ll<Nl; ll++)
    for(size_t jj=0; jj<Nj; jj++)
      scratch_Pjl(jj, ll) = P(j0+jj, l0+ll);
  scratch_Pjk.set_size(Nj, Nk);
  for(size_t kk=0; kk<Nk; kk++)
    for(size_t jj=0; jj<Nj; jj++)
      scratch_Pjk(jj, kk) = P(j0+jj, k0+kk);
  scratch_Pil.set_size(Ni, Nl);
  for(size_t ll=0; ll<Nl; ll++)
    for(size_t ii=0; ii<Ni; ii++)
      scratch_Pil(ii, ll) = P(i0+ii, l0+ll);
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
	    el+=scratch_Pik(ii,kk)*scratch_Pjl(jj,ll)*(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];

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
	      el+=scratch_Pjk(jj,kk)*scratch_Pil(ii,ll)*(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];

      // Increment the element
      f(idx)+=K2fac*el;
    }
  }
}
