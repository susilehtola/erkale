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
  std::vector<arma::mat> xk;
  /// Gradients
  std::vector<arma::mat> gk;

  /// Diagonal Hessian
  arma::mat h;

 public:
  /// Constructor
  GDM(size_t nmax=10);
  /// Destructor
  ~GDM();

  /// Update
  void update(const arma::mat & x, const arma::mat & g, const arma::mat & h);
  /// Solve the problem
  arma::mat solve();
  /// Apply the parallel transport tV = exp(-Delta/2) V exp(Delta/2)
  void parallel_transport(const arma::mat & exphdelta);
  /// Clear out history
  void clear();
};

#endif
