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

#ifndef ERKALE_TRDSM
#define ERKALE_TRDSM

#include "global.h"

#include <vector>
#include <armadillo>
#include <gsl/gsl_multimin.h>

/**
 * \class TRDSM
 *
 * \brief This class contains the trust-region SCF algorithm as described in
 *
 * L. Thøgersen et al, "The trust-region self-consistent field method:
 * Towards a black-box optimization in Hartree-Fock and Kohn-Sham
 * theories", JCP 121, 16 (2004).
 *
 * L. Thøgersen et al, "The trust-region self-consistent field method
 * in Kohn-Sham density-functional theory", JCP 123, 074103 (2005).
 *
 * \author Susi Lehtola
 * \date 2011/05/08 19:32
 *
 */


class TRDSM {
  /// Energy stack
  std::vector<double> Es;
  /// Density matrices
  std::vector<arma::mat> Ds;
  /// Fock matrices
  std::vector<arma::mat> Fs;  
  /// Maximum number of matrices to keep in memory
  size_t max;

  /// Overlap matrix
  arma::mat S;

  /// Index of entry with smallest energy
  size_t minind;

  /// Get the M matrix
  arma::mat get_M() const;

  /// Get the average density matrix
  arma::mat get_Dbar(const arma::vec & c) const;
  /// Get the average Fock matrix
  arma::mat get_Fbar(const arma::vec & c) const;

  /// Get the stack of difference matrices
  std::vector<arma::mat> get_Dn0() const;
  /// Get the stack of difference matrices
  std::vector<arma::mat> get_Fn0() const;

  /// Compute the gradient
  arma::vec get_gradient(const arma::vec & c) const;
  /// Compute the gradient and hessian, JCP 121 eqn 49
  void get_gradient_hessian(const arma::vec & c, arma::vec & g, arma::mat & H) const;

  /// Solve for the coefficients using line search
  arma::vec solve_c_ls() const;

  /// Solve for the coefficients
  arma::vec solve_c() const;
  /// Helper routine
  void solve_c(arma::vec & c, double mu, arma::vec & Hval, arma::mat & Hvec, const arma::mat & M) const;

  /// Compute length of step
  double step_len(const arma::vec & c, const arma::mat & M) const;

  /// Update the minimal index
  void update_minind();
  /// Clean bad density matrices
  void clean_density();

 public:
  /// Constructor, keep max matrices in memory
  TRDSM(const arma::mat & S, size_t max=7);
  /// Destructor
  ~TRDSM();

  /// Add new matrices to stacks
  void push(double E, const arma::mat & D, const arma::mat & F);

  /// Drop everything in memory
  void clear();

  /// Get the new Fock matrix
  arma::mat solve() const;

  /// Get the energy estimate
  double E_DSM(const arma::vec & c) const;
  /// Get the estimated change in energy
  double dE_DSM(const arma::vec & g, const arma::mat & H, const arma::vec & c) const;
};


#endif
