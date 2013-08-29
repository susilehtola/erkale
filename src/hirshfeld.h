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
  double dr;
  /// Densities
  std::vector<double> rho;

 public:
  /// Dummy constructor
  HirshfeldAtom();
  /// Constructor
  HirshfeldAtom(const BasisSet & basis, const arma::mat & P, double dr=0.05);
  /// Constructor, given input density
  HirshfeldAtom(double dr, const std::vector<double> & rho);
  /// Destructor
  ~HirshfeldAtom();

  /// Evaluate density at r
  double get(double r) const;

  /// Get grid spacing
  double get_spacing() const;
  /// Get densities
  std::vector<double> get_rho() const;

  /// Get the range of the atom
  double get_range() const;
};

/// Hirshfeld atomic densities
class Hirshfeld {
 protected:
  /// List of atoms
  std::vector<HirshfeldAtom> atoms;
  /// Centers
  std::vector<coords_t> cen;

 public:
  /// Dummy constructor
  Hirshfeld();
  /// Destructor
  ~Hirshfeld();

  /// Set atoms
  void set(const std::vector<coords_t> & cen, double dr, const std::vector< std::vector<double> > & rho);
  /// Get atomic densities
  std::vector< std::vector<double> > get_rho() const;

  /// Compute
  void compute(const BasisSet & basis, std::string method);

  /// Evaluate density at r
  double get_density(size_t inuc, const coords_t & r) const;
  /// Evaluate weight at r
  double get_weight(size_t inuc, const coords_t & r) const;
  /// Get range of atom
  double get_range(size_t inuc) const;

  /// Print densities
  void print_densities() const;
};

#endif
