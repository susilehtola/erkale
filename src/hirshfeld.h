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

#ifndef ERKALE_HIRSHFELD
#define ERKALE_HIRSHFELD

#include "basis.h"

extern "C" {
  // For spline interpolation
#include <gsl/gsl_spline.h>
}

/// Hirshfeld atomic density
class HirshfeldAtom {
  /// Grid spacing
  static const double dr=0.001;
  /// Densities
  std::vector<double> rho;

 public:
  /// Dummy constructor
  HirshfeldAtom();
  /// Constructor
  HirshfeldAtom(const BasisSet & basis, const arma::mat & P);
  /// Destructor
  ~HirshfeldAtom();

  /// Evaluate density at r
  double get(double r) const;

  /// Get grid spacing
  double get_spacing() const;
  /// Get densities
  std::vector<double> get_rho() const;
};

/// Hirshfeld atomic densities
class Hirshfeld {
  /// The actual atoms
  std::vector<HirshfeldAtom> atomstorage;
  /// List of identical nuclei
  std::vector< std::vector<size_t> > idnuc;

  /// List of atom pointers
  std::vector<HirshfeldAtom *> atoms;
  /// Centers
  std::vector<coords_t> cen;

 public:
  /// Dummy constructor
  Hirshfeld();
  /// Destructor
  ~Hirshfeld();

  /// Compute
  void compute(const BasisSet & basis, std::string method);
  /// Update pointers
  void update_pointers();

  /// Evaluate density at r
  double get_density(size_t inuc, const coords_t & r) const;
  /// Evaluate weight at r
  double get_weight(size_t inuc, const coords_t & r) const;
};

#endif
