/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_EDIIS
#define ERKALE_EDIIS

#include "global.h"

#include <vector>
#include <armadillo>
#include <gsl/gsl_multimin.h>

/**
 * \class EDIIS
 *
 * \brief This class contains the EDIIS convergence accelerator.
 *
 * The EDIIS algorithm is described in K. N. Kudin, G. E. Scuseria and
 * E. Cancès, "A black-box self-consistent field convergence
 * algorithm: One step closer", J. Chem. Phys. 116, 8255 (2002).
 *
 * \author Susi Lehtola
 * \date 2011/05/08 19:32
 *
 */

/// Helper for sorts
typedef struct {
  /// Energy
  double E;
  /// Density matrix
  arma::mat P;
  /// Fock matrix
  arma::mat F;
} ediis_t;

/// Helper for sorts
bool operator<(const ediis_t & lhs, const ediis_t & rhs);

class EDIIS {
  /// Stack of entries
  std::vector<ediis_t> stack;

  /// Trace matrix
  arma::mat FDtr;

  /// Maximum number of matrices to keep in memory
  size_t max;

 public:
  /// Constructor, keep max matrices in memory
  EDIIS(size_t max=5);
  /// Destructor
  ~EDIIS();

  /// Add new matrices to stack
  void push(double E, const arma::mat & P, const arma::mat & F);
  /// Drop everything in memory
  void clear();

  /// Compute new estimate for density matrix
  arma::mat get_P() const;
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

namespace ediis {  
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
};

#endif
