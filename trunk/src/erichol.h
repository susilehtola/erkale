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
  /// Cholesky vectors, L x (Nbf x Nbf)
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
    
  /// Fill matrix
  void fill(const BasisSet & basis, double tol, double shthr, double shtol, bool verbose);

  /// Get the matrix
  arma::mat get() const;
};

#endif
