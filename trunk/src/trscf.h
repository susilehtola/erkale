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

#ifndef ERKALE_TRSCF
#define ERKALE_TRSCF

#include "global.h"

#include <vector>
#include <armadillo>
#include <gsl/gsl_multimin.h>

/**
 * \class TRSCF
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
 * \author Jussi Lehtola
 * \date 2011/05/08 19:32
 *
 */


class TRSCF {
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

  /// Predicted energies
  std::vector<double> Epreds;

  /// Trust radius
  double h;
  
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

  /// Compute the gradient, JCP 121 eqn 49
  arma::vec get_g(const arma::vec & c) const;
  /// Compute the Hessian, JCP 121 eqn 49
  arma::mat get_H(const arma::vec & c) const;

  /// Solve for the coefficients
  arma::vec solve_c() const;
  /// Helper routine
  void solve_c(arma::vec & c, double mu, arma::vec & Hval, arma::mat & Hvec, const arma::mat & M) const;

  /// Compute length of step
  double step_len(const arma::vec & c, const arma::mat & M) const;

 public:
  /// Constructor, keep max matrices in memory
  TRSCF(const arma::mat & S, size_t max=6);
  /// Destructor
  ~TRSCF();

  /// Add new matrices to stacks
  void push(double E, const arma::mat & D, const arma::mat & F);
  /// Drop everything in memory
  void clear();

  /// Update the minimal index
  void update_minind();

  /// Get the new Fock matrix
  arma::mat solve() const;

  /// Get the energy estimate
  double E_DSM(const arma::vec & c) const;
  /// Get the estimated change in energy
  double dE(const arma::vec & g, const arma::mat & H, const arma::vec & c) const;
};


#endif
