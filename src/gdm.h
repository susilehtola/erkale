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

#ifndef ERKALE_GDM
#define ERKALE_GDM

#include <armadillo>

class GDM {
  /// Maximum number of matrices
  size_t nmax;
  /// Coordinates
  std::vector<arma::vec> xk;
  /// Gradients
  std::vector<arma::vec> gk;
  /// Diagonal Hessian
  arma::vec h;

 public:
  /// Constructor
  GDM(size_t nmax=10);
  /// Destructor
  ~GDM();

  /// Update
  void update(const arma::vec & x, const arma::vec & g, const arma::vec & h);
  /// Solve the problem
  arma::vec solve();
  /// Clear out history
  void clear();
};

#endif
