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

#ifndef ERKALE_ADIIS
#define ERKALE_ADIIS

#include "global.h"

#include <vector>
#include <armadillo>
#include <gsl/gsl_multimin.h>

/**
 * \class ADIIS
 *
 * \brief This class contains the ADIIS convergence accelerator.
 *
 * The ADIIS algorithm is described in
 * X. Hu and W. Yang, "Accelerating self-consistent field convergence
 * with the augmented Roothaan–Hall energy function",
 * J. Chem. Phys. 132 (2010), 054109.
 *
 * \author Susi Lehtola
 * \date 2011/05/08 19:32
 *
 */

/// Stack entry
typedef struct {
  /// Energy
  double E;
  /// Density matrix
  arma::mat P;
  /// Fock matrix
  arma::mat F;
} adiis_t;

class ADIIS {
  /// Stack of entries
  std::vector<adiis_t> stack;

  /// Maximum number of matrices to keep in memory
  size_t max;

  // Helpers for speeding up evaluation
  /// < P_i - P_n | F(D_n) >
  arma::vec PiF;
  /// < P_i - P_n | F(D_j) - F(D_n) >
  arma::mat PiFj;

 public:
  /// Constructor, keep max matrices in memory
  ADIIS(size_t max=6);
  /// Destructor
  ~ADIIS();

  /// Add new matrices to stacks
  void push(double E, const arma::mat & P, const arma::mat & F);
  /// Drop everything in memory
  void clear();

  /// Compute new estimate for density matrix
  arma::mat get_P() const;
  /// Compute new estimate for Fock operator
  arma::mat get_H() const;

  /// Solve coefficients
  arma::vec get_c() const;

  /// Compute energy and its derivative with contraction coefficients \f$ c_i = x_i^2 / \left[ \sum_j x_j^2 \right] \f$
  double get_E(const gsl_vector * x) const;
  /// Compute derivative wrt contraction coefficients
  void get_dEdx(const gsl_vector * x, gsl_vector * dEdx) const;
  /// Compute energy and derivative wrt contraction coefficients
  void get_E_dEdx(const gsl_vector * x, double * E, gsl_vector * dEdx) const;
};

namespace adiis {
  /// Compute weights
  arma::vec compute_c(const gsl_vector * x);
  /// Compute jacobian
  arma::mat compute_jac(const gsl_vector * x);


  /// Compute energy
  double min_f(const gsl_vector * x, void * params);
  /// Compute derivative
  void min_df(const gsl_vector * x, void * params, gsl_vector * g);
  /// Compute energy and derivative
  void min_fdf(const gsl_vector * x, void * params, double * f, gsl_vector * g);
};

#endif