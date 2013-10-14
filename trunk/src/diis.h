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


#include "global.h"

#ifndef ERKALE_DIIS
#define ERKALE_DIIS

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
 * \brief DIIS - Direct Inversion in the Iterative Subspace
 *
 * This class contains the DIIS convergence accelerator.
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
 *
 * Using C1-DIIS is, however, not recommended. What is used by
 * default, instead, is C2-DIIS, which is documented in the article
 *
 * H. Sellers, "The C2-DIIS convergence acceleration algorithm",
 * Int. J. Quant. Chem. 45 (1993), pp. 31 - 41
 *
 * \author Susi Lehtola
 * \date 2011/04/20 15:37
 */

class DIIS {
 protected:
  /// Overlap matrix
  arma::mat S;
  /// Half-inverse overlap matrix
  arma::mat Sinvh;

  /// Maximum amount of matrices to store
  size_t imax;

  /// Get errors
  virtual std::vector<arma::vec> get_error() const=0;
  /// Reduce size of stack by one
  virtual void erase_last()=0;

  /// Compute weights, use C1-DIIS if wanted
  arma::vec get_weights(bool verbose, bool c1_diis);

 public:
  /// Constructor
  DIIS(const arma::mat & S, const arma::mat & Sinvh, size_t imax=20);
  /// Destructor
  ~DIIS();

  /// Clear Fock matrices and errors
  virtual void clear()=0;
};

/// Spin-restricted DIIS
class rDIIS: protected DIIS {
  /// Fock matrices in AO basis
  std::vector<diis_unpol_entry_t> stack;

  /// Get errors
  std::vector<arma::vec> get_error() const;
  /// Reduce size of stack by one
  void erase_last();

 public:
  /// Constructor
  rDIIS(const arma::mat & S, const arma::mat & Sinvh, size_t imax=20);
  /// Destructor
  ~rDIIS();

  /// Add matrices to stack
  void update(const arma::mat & F, const arma::mat & P, double E, double & error);

  /// Compute new Fock matrix, use C1-DIIS if wanted
  void solve_F(arma::mat & F, bool verbose=true, bool c1_diis=false);

  /// Compute new density matrix, use C1-DIIS if wanted
  void solve_P(arma::mat & P, bool verbose=true, bool c1_diis=false);

  /// Clear Fock matrices and errors
  void clear();
};

/// Spin-unrestricted DIIS
class uDIIS: protected DIIS {
  /// Fock matrices in AO basis - spin polarized
  std::vector<diis_pol_entry_t> stack;

  /// Get errors
  std::vector<arma::vec> get_error() const;
  /// Reduce size of stack by one
  void erase_last();

 public:
  /// Constructor
  uDIIS(const arma::mat & S, const arma::mat & Sinvh, size_t imax=20);
  /// Destructor
  ~uDIIS();

  /// Add matrices to stack
  void update(const arma::mat & Fa, const arma::mat & Fb, const arma::mat & Pa, const arma::mat & Pb, double E, double & error);

  /// Compute new Fock matrix, use C1-DIIS if wanted
  void solve_F(arma::mat & Fa, arma::mat & Fb, bool verbose=true, bool c1_diis=false);

  /// Compute new density matrix, use C1-DIIS if wanted
  void solve_P(arma::mat & Pa, arma::mat & Pb, bool verbose=true, bool c1_diis=false);

  /// Clear Fock matrices and errors
  void clear();
};


#endif
