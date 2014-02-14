/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_HIRSHFELDI
#define ERKALE_HIRSHFELDI

#include "hirshfeld.h"

/**
 * Iterative Hirshfeld atoms.
 *
 * P. Bultinck, Ch. Van Alsenoy, P. W. Ayers, and R. Carbó-Dorca,
 * "Critical analysis and extension of the Hirshfeld atoms in
 * molecules", J. Chem. Phys. 126, 144111 (2007).
 */
class HirshfeldI {
  /// Individual atomic densities
  std::vector< std::vector< std::vector<double> > > atoms;
  /// Charges
  std::vector< std::vector<int> > atQ;
  /// Atomic centers
  std::vector<coords_t> cen;

  /// Current iteration for individual atomic densities
  Hirshfeld ISA;
  /// Grid spacing
  double dr;

  /// Solve the charges
  void solve(const BasisSet & basis, const arma::mat & P, double tol, bool verbose);
  /// Iteratively refine charges
  void iterate(const BasisSet & basis, const arma::mat & P, arma::vec & q, double tol, bool verbose);
  /// Get new Hirshfeld composition
  Hirshfeld get(const arma::vec & Q);

 public:
  /// Dummy constructor
  HirshfeldI();
  /// Destructor
  ~HirshfeldI();

  /// Compute decomposition. Tolerance for change in the integral \f$ \int_0^\infty r^2 | \rho_n(r) - \rho_o(r) | dr \f$, grid spacing, and change in charge species to compute.
  void compute(const BasisSet & basis, const arma::mat & P, std::string method="HF", double tol=1e-5, double dr=0.001, int dq=2, bool verbose=true);
  /// Compute decomposition, but load atomic densities from files.
  void compute_load(const BasisSet & basis, const arma::mat & P, double tol=1e-5, double dr=0.001, int dq=2, bool verbose=true);

  /// Get the decomposition
  Hirshfeld get() const;
};

#endif
