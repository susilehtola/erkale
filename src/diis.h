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


#include "global.h"

#ifndef ERKALE_DIIS
#define ERKALE_DIIS

#include <armadillo>
#include <vector>

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
 * \author Jussi Lehtola
 * \date 2011/04/20 15:37
 */

class DIIS {
  /// Fock matrices in AO basis
  std::vector<arma::mat> Fs;
  /// Errors
  std::vector<arma::vec> errs;
  /// Overlap matrix
  arma::mat S;

  /// Index of current iteration
  int icur;
  /// Maximum amount of matrices to store
  int imax;

 public:
  /// Constructor
  DIIS(const arma::mat & S, size_t imax=5);
  /// Destructor
  ~DIIS();

  /// Clear Fock matrices and errors
  void clear();

  /// Add matrix to stack
  void update(const arma::mat & F, const arma::mat & D, double & error);
  
  /// Compute new Fock matrix, use C1-DIIS if wanted
  void solve(arma::mat & F, bool c1_diis=0) const;
};

#endif
