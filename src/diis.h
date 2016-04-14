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


#ifndef ERKALE_DIIS
#define ERKALE_DIIS

#include "global.h"
#include <armadillo>
#include <vector>

/// Spin-polarized entry
typedef struct {
  /// Alpha density matrix
  arma::mat Pa;
  /// Alpha Fock matrix
  arma::mat Fa;
  /// Beta density matrix
  arma::mat Pb;
  /// Beta Fock matrix
  arma::mat Fb;
  /// Energy
  double E;

  /// DIIS error matrix
  arma::vec err;
} diis_pol_entry_t;

/// Spin-unpolarized entry
typedef struct {
  /// Density matrix
  arma::mat P;
  /// Fock matrix
  arma::mat F;
  /// Energy
  double E;

  /// DIIS error matrix
  arma::vec err;
} diis_unpol_entry_t;

/// Helper for sort
bool operator<(const diis_pol_entry_t & lhs, const diis_pol_entry_t & rhs);
/// Helper for sort
bool operator<(const diis_unpol_entry_t & lhs, const diis_unpol_entry_t & rhs);

/**
 * \class DIIS
 *
 * \brief DIIS - Direct Inversion in the Iterative Subspace and ADIIS
 *
 * This class contains the DIIS and ADIIS convergence accelerators.
 *
 *
 * The original DIIS (C1-DIIS) is based on the articles
 *
 * P. Pulay, "Convergence acceleration of iterative sequences. The
 * case of SCF iteration", Chem. Phys. Lett. 73 (1980), pp. 393 - 398
 *
 * and
 *
 * P. Pulay, "Improved SCF Convergence Acceleration", J. Comp. Chem. 3
 * (1982), pp. 556 - 560.
 *
 * Using C1-DIIS is, however, not recommended. What is used by
 * default, instead, is C2-DIIS, which is documented in the article
 *
 * H. Sellers, "The C2-DIIS convergence acceleration algorithm",
 * Int. J. Quant. Chem. 45 (1993), pp. 31 - 41
 *
 *
 * The ADIIS algorithm is described in
 *
 * X. Hu and W. Yang, "Accelerating self-consistent field convergence
 * with the augmented Roothaan–Hall energy function",
 * J. Chem. Phys. 132 (2010), 054109.
 *
 * \author Susi Lehtola
 * \date 2011/05/08 19:32
 */

class DIIS {
 protected:
  /// Overlap matrix
  arma::mat S;
  /// Half-inverse overlap matrix
  arma::mat Sinvh;

  /// Use DIIS?
  bool usediis;
  /// C1-DIIS?
  bool c1diis;
  /// Use ADIIS?
  bool useadiis;
  /// Verbose operation?
  bool verbose;

  /// When to start using DIIS weights
  double diiseps;
  /// When to start using DIIS exclusively
  double diisthr;
  /// Counter for not using DIIS
  int cooloff;
  
  /// Maximum amount of matrices to store
  size_t imax;
  /// Get energies
  virtual arma::vec get_energies() const=0;
  /// Get errors
  virtual arma::mat get_diis_error() const=0;
  /// Reduce size of stack by one
  virtual void erase_last()=0;

  // Helpers for speeding up ADIIS evaluation
  /// < P_i - P_n | F(D_n) >   or   < Pa_i - Pa_n | Fa(P_n) > + < Pb_i - Pb_n | Fb(P_n) >
  arma::vec PiF;
  /// < P_i - P_n | F(D_j) - F(D_n) >   or    < Pa_i - Pa_n | Fa(P_j) - Fa(P_n) > + < Pb_i - Pb_n | Fb(P_j) - Fb(P_n) >
  arma::mat PiFj;

  /// Compute weights
  arma::vec get_w();
  /// Compute DIIS weights
  arma::vec get_w_diis() const;
  /// Compute DIIS weights, worker routine
  arma::vec get_w_diis_wrk(const arma::mat & err) const;
  /// Compute ADIIS weights
  arma::vec get_w_adiis() const;

  /// Solve coefficients
  arma::vec get_c_adiis(bool verbose=false) const;
  
 public:
  /// Constructor
  DIIS(const arma::mat & S, const arma::mat & Sinvh, bool usediis, bool c1diis, double diiseps, double diisthr, bool useadiis, bool verbose, size_t imax);
  /// Destructor
  virtual ~DIIS();

  /// Clear Fock matrices and errors
  virtual void clear()=0;

  /// Compute energy with contraction coefficients \f$ c_i = x_i^2 / \left[ \sum_j x_j^2 \right] \f$
  double get_E_adiis(const arma::vec & x) const;
  /// Compute derivative of energy wrt contraction coefficients
  arma::vec get_dEdx_adiis(const arma::vec & x) const;
};

/// Spin-restricted DIIS
class rDIIS: protected DIIS {
  /// Fock matrices in AO basis
  std::vector<diis_unpol_entry_t> stack;

  /// Get energies
  arma::vec get_energies() const;
  /// Get errors
  arma::mat get_diis_error() const;
  /// Reduce size of stack by one
  void erase_last();
  /// ADIIS update
  void PiF_update();
  
 public:
  /// Constructor
  rDIIS(const arma::mat & S, const arma::mat & Sinvh, bool usediis, bool c1diis, double diiseps, double diisthr, bool useadiis, bool verbose, size_t imax);
  /// Destructor
  ~rDIIS();

  /// Add matrices to stack
  void update(const arma::mat & F, const arma::mat & P, double E, double & error);

  /// Compute new Fock matrix, use C1-DIIS if wanted
  void solve_F(arma::mat & F);

  /// Compute new density matrix, use C1-DIIS if wanted
  void solve_P(arma::mat & P);

  /// Clear Fock matrices and errors
  void clear();
};

/// Spin-unrestricted DIIS
class uDIIS: protected DIIS {
  /// Fock matrices in AO basis - spin polarized
  std::vector<diis_pol_entry_t> stack;

  /// Get energies
  arma::vec get_energies() const;
  /// Get errors
  arma::mat get_diis_error() const;
  /// Reduce size of stack by one
  void erase_last();
  /// ADIIS update
  void PiF_update();
  
 public:
  /// Constructor
  uDIIS(const arma::mat & S, const arma::mat & Sinvh, bool usediis, bool c1diis, double diiseps, double diisthr, bool useadiis, bool verbose, size_t imax);
  /// Destructor
  ~uDIIS();

  /// Add matrices to stack
  void update(const arma::mat & Fa, const arma::mat & Fb, const arma::mat & Pa, const arma::mat & Pb, double E, double & error);

  /// Compute new Fock matrix, use C1-DIIS if wanted
  void solve_F(arma::mat & Fa, arma::mat & Fb);

  /// Compute new density matrix, use C1-DIIS if wanted
  void solve_P(arma::mat & Pa, arma::mat & Pb);

  /// Clear Fock matrices and errors
  void clear();
};

#endif
