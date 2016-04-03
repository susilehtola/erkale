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

#ifndef ERKALE_FDHESSIAN
#define ERKALE_FDHESSIAN

#include "global.h"
#include <armadillo>
#include "timer.h"

class FDHessian {
 protected:
  /// Finite difference derivative step size
  double ss_fd;
  /// Line search step size
  double ss_ls;

  /// Print optimization status
  virtual void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;

 public:
  /// Constructor
  FDHessian();
  /// Destructor
  virtual ~FDHessian();

  /// Get amount of parameters
  virtual size_t count_params() const=0;
  /// Evaluate function
  virtual double eval(const arma::vec & x)=0;
  /// Update solution
  virtual void update(const arma::vec & x);

  /// Evaluate finite difference gradient
  virtual arma::vec gradient();
  /// Evaluate finite difference gradient at point x
  virtual arma::vec gradient(const arma::vec & x);
  /// Evaluate finite difference Hessian
  virtual arma::mat hessian();

  /// Run optimization
  virtual double optimize(size_t maxiter=1000, double gthr=1e-4, bool max=false);
};

#endif
