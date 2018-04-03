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
#include "gdm.h"

GDM::GDM(size_t nmax_) : nmax(nmax_) {
}

GDM::~GDM() {
}

arma::vec GDM::solve() {
  LBFGS helper(nmax);

  // Convert steps and gradients into energy weighted coordinates (eqn
  // 29) in GDM paper
  for(size_t i=0;i<xk.size();i++)
    helper.update(xk[i]%arma::sqrt(h), gk[i]/arma::sqrt(h));

  // Solve the parameters and do the back-transform
  return helper.solve()/arma::sqrt(h);
}

void GDM::update(const arma::vec & x, const arma::vec & g, const arma::vec & h_) {
  xk.push_back(x);
  gk.push_back(g);
  h=h_;

  if(xk.size()>nmax) {
    xk.erase(xk.begin());
    gk.erase(gk.begin());
  }
}

void GDM::clear() {
  xk.clear();
  gk.clear();
}
