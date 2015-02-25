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
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;
  digest_J(shpairs,ip,jp,ints,ioff,P,J);
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
  digest_K(shpairs,ip,jp,ints,ioff,double,P,K);
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
  digest_K(shpairs,ip,jp,ints,ioff,std::complex<double>,P,K);
}

arma::cx_mat cxKDigestor::get_K() const {
  return K;
}
