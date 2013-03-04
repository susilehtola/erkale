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


/**
 * \class Broyden
 *
 * \brief Broyden accelerator.
 * 
 * This class contains a Broyden convergence accelerator, as described in
 *
 * K. Baarman, T. Eirola and V. Havu, "Robust acceleration of self
 * consistent field calculations for density functional theory",
 * J. Chem. Phys. 134 (2011), 134109.
 *
 * The dimension of the problem is Nbas^2. The jacobian would thus be
 * Nbas^4, so we need to work with a memory limited algorithm.
 *
 * \author Susi Lehtola
 * \date 2011/05/10 15:39
 *
 */


#ifndef ERKALE_BROYDEN
#define ERKALE_BROYDEN

#include "global.h"

#include <armadillo>
#include <vector>

/// Broyden convergence accelerator
class Broyden {
  /// Stack of approximate solutions
  std::vector<arma::vec> x;
  /// Stack of differences from SCF solution
  std::vector<arma::vec> f;

  /// Difficulties encountered? (Halve mixing parameter for this step)
  bool difficult;
  /// Verbose operation? (Complain about bad updates?)
  bool verbose;

  /// Number of matrices to keep in memory
  size_t m;
  /// Damping parameter
  double beta;
  /// Sigma parameter
  double sigma;

 public:
  /// Construct accelerator storing m iterations and with mixing parameters beta and sigma.
  Broyden(bool verbose=true, size_t m=10, double beta=0.8, double sigma=0.25);
  /// Destructor
  ~Broyden();

  /// Add solutions of SCF equation to stack
  void push_x(const arma::vec & x);
  /// Add difference from SCF solution to stack
  void push_f(const arma::vec & f);

  /// Clean old matrices from memory
  void clear();

  /// Get estimate for solution
  arma::vec update_x();
  /// Operate on vector with estimated inverse Jacobian
  arma::vec operate_G(const arma::vec & v, size_t ind) const;
};

#endif
  
