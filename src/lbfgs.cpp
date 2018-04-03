/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "lbfgs.h"

LBFGS::LBFGS(size_t nmax_) : nmax(nmax_) {
}

LBFGS::~LBFGS() {
}

void LBFGS::update(const arma::vec & x, const arma::vec & g) {
  xk.push_back(x);
  gk.push_back(g);

  if(xk.size()>nmax) {
    xk.erase(xk.begin());
    gk.erase(gk.begin());
  }
}

arma::vec LBFGS::apply_diagonal_hessian(const arma::vec & q) const {
  if(xk.size()>=2) {
    arma::vec s=xk[xk.size()-1]-xk[xk.size()-2];
    arma::vec y=gk[gk.size()-1]-gk[gk.size()-2];

    return arma::dot(s,y)/arma::dot(y,y)*q;

  } else
    // Unit diagonal Hessian
    return q;
}

arma::vec LBFGS::solve() const {
  // Algorithm 9.1 in Nocedal's book
  size_t k=gk.size()-1;
  arma::vec q(gk[k]);

  std::vector<arma::vec> sk(k);
  for(size_t i=0;i<k;i++)
    sk[i]=xk[i+1]-xk[i];
  std::vector<arma::vec> yk(k);
  for(size_t i=0;i<k;i++)
    yk[i]=gk[i+1]-gk[i];

  // Alpha_i
  std::vector<double> alphai(k);

  // First part
  for(size_t i=k-1;i<k;i--) {
    // Alpha_i
    alphai[i]=arma::dot(sk[i],q)/arma::dot(yk[i],sk[i]);
    // Update q
    q-=alphai[i]*yk[i];
  }

  // Apply diagonal Hessian
  arma::vec r(apply_diagonal_hessian(q));

  // Second part
  for(size_t i=0;i<k;i++) {
    // Beta
    double beta=arma::dot(yk[i],r)/arma::dot(yk[i],sk[i]);
    // Update r
    r+=sk[i]*(alphai[i]-beta);
  }

  return r;
}

void LBFGS::clear() {
  xk.clear();
  gk.clear();
}


