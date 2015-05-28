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

#ifndef ERKALE_LBFGS
#define ERKALE_LBFGS

#include "global.h"
#include <armadillo>

class LBFGS {
 protected:
  /// Maximum number of matrices
  size_t nmax;
  
  /// Coordinates
  std::vector<arma::vec> xk;
  /// Gradients
  std::vector<arma::vec> gk;
  
  /// Apply diagonal Hessian: r = H_0 q
  virtual arma::vec diagonal_hessian(const arma::vec & q) const;

 public:
  /// Constructor
  LBFGS(size_t nmax=5);
  /// Destructor
  virtual ~LBFGS();

  /// Update
  void update(const arma::vec & x, const arma::vec & g);
  /// Solve for new x
  arma::vec solve() const;
};

#endif
