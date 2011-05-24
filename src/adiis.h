/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
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
 * \author Jussi Lehtola
 * \date 2011/05/08 19:32
 *
 */


class ADIIS {
  /// Energy stack
  std::vector<double> E;
  /// Density matrices
  std::vector<arma::mat> D;
  /// Fock matrices
  std::vector<arma::mat> F;

  /// Maximum number of matrices to keep in memory
  size_t max;

 public:
  /// Constructor, keep max matrices in memory
  ADIIS(size_t max=6);
  /// Destructor
  ~ADIIS();

  /// Add new matrices to stacks
  void push(double E, const arma::mat & D, const arma::mat & F);
  /// Drop everything in memory
  void clear();

  /// Compute new estimate for density matrix
  arma::mat get_D() const;
  /// Compute new estimate for Fock operator
  arma::mat get_H() const;

  /// Solve coefficients
  std::vector<double> get_c() const;

  /// Compute energy and its derivative with contraction coefficients \f$ c_i = x_i^2 / \left[ \sum_j x_j^2 \right] \f$
  double get_E(const gsl_vector * x) const;
  /// Compute derivative wrt contraction coefficients
  void get_dEdx(const gsl_vector * x, gsl_vector * dEdx) const;
  /// Compute energy and derivative wrt contraction coefficients
  void get_E_dEdx(const gsl_vector * x, double * E, gsl_vector * dEdx) const;
};

/// Compute weights
std::vector<double> compute_c(const gsl_vector * x);
/// Compute jacobian
arma::mat compute_jac(const gsl_vector * x);


/// Compute energy
double min_f(const gsl_vector * x, void * params);
/// Compute derivative
void min_df(const gsl_vector * x, void * params, gsl_vector * g);
/// Compute energy and derivative
void min_fdf(const gsl_vector * x, void * params, double * f, gsl_vector * g);

#endif
