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

#ifndef ERKALE_ERICHOL
#define ERKALE_ERICHOL

#include "global.h"
#include "basis.h"

/// Cholesky decomposition of ERIs
class ERIchol {
  /// Amount of basis functions
  size_t Nbf;
  /// Map of product indices in full space (for getting density subvector)
  arma::uvec prodidx;
  /// Map to function indices, 2 x Nprod
  arma::umat invmap;
  /// Map to product index
  arma::umat prodmap;
  /// List of off-diagonal products
  arma::uvec odiagidx;
  /// Cholesky vectors, L x Nprod
  arma::mat B;

  /// Range separation constant
  double omega;
  /// Fraction of full-range Coulomb
  double alpha;
  /// Fraction of short-range Coulomb
  double beta;
  
 public:
  /// Constructor
  ERIchol();
  /// Destructor
  ~ERIchol();

  /// Set range separation
  void set_range_separation(double w, double a, double b);
  void get_range_separation(double & w, double & a, double & b) const;

  /// Load B matrix
  void load();
  /// Save B matrix
  void save() const;
    
  /// Fill matrix, returns amount of significant pairs.
  size_t fill(const BasisSet & basis, double cholesky_tol, double shell_reuse_thr, double shell_screen_tol, bool verbose);

  /// Get amount of vectors
  size_t get_Naux() const;
  /// Get basis set size
  size_t get_Nbf() const;
  /// Get basis set size
  size_t get_Npairs() const;
  
  /// Get the matrix
  arma::mat get() const;

  /// Form Coulomb matrix
  arma::mat calcJ(const arma::mat & P) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::vec & C) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::mat & C, const std::vector<double> & occs) const;

  /// Form exchange matrix
  arma::cx_mat calcK(const arma::cx_vec & C) const;
  /// Form exchange matrix
  arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occs) const;

  /// Get full B matrix
  void B_matrix(arma::mat & B) const;
  /// Get transformed B matrix
  arma::mat B_transform(const arma::mat & Cl, const arma::mat & Cr, bool verbose=false) const;
};

#endif
