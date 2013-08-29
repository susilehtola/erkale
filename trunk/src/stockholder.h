/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_STOCKHOLDER
#define ERKALE_STOCKHOLDER

#include "hirshfeld.h"
#include "lebedev.h"
#include "basis.h"

/**
 * Iterative Stockholder atoms.
 *
 * T. C. Lillestolen and R. J. Wheatley, "Atomic charge densities
 * generated using an iterative stockholder procedure",
 * J. Chem. Phys. 131, 144101 (2009).
 */
class StockholderAtom {
  /// Atom index
  size_t atind;
  /// List of molecular densities
  std::vector< std::vector<double> > rho;
  /// and weights
  std::vector< std::vector<double> > weights;
  /// and grid points
  std::vector< std::vector<coords_t> > grid;

 public:
  /// Constructor
  StockholderAtom();
  /// Destructor
  ~StockholderAtom();

  /// Compute the density
  void fill(const BasisSet & basis, const arma::mat & P, size_t atind, double dr, int nrad, int lmax);

  /// Compute a new radial density
  void update(const Hirshfeld & hirsh, std::vector<double> & rho);
};

/// Stockholder grid
class Stockholder {
  /// Atomic grids
  std::vector<StockholderAtom> atoms;
  /// Atom centers
  std::vector<coords_t> cen;

  /// Spherical atoms
  Hirshfeld ISA;

 public:
  /// Constructor. Tolerance for change in the integral \f$ \int_0^\infty r^2 | \rho_n(r) - \rho_o(r) | dr \f$
  Stockholder(const BasisSet & basis, const arma::mat & P, double tol=1e-6, double dr=0.01, int nrad=1001, int lmax=35, bool verbose=true);
  /// Destructor
  ~Stockholder();

  /// Get the decomposition
  Hirshfeld get() const;
};

#endif
