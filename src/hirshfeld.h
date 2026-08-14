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
  double dr_;
  /// Densities
  std::vector<double> rho_;

 public:
  /// Dummy constructor
  HirshfeldAtom();
  /// Constructor
  HirshfeldAtom(const BasisSet & basis, const arma::mat & P, double dr=0.001);
  /// Constructor, given input density
  HirshfeldAtom(double dr, const std::vector<double> & rho);
  /// Destructor
  ~HirshfeldAtom();

  /// Evaluate density at r
  double density(double r) const;

  /// Grid spacing
  double spacing() const;
  /// Densities
  std::vector<double> rho() const;

  /// The range of the atom
  double range() const;
  /// Calculate expectation values of radius (already includes r^2 factor)
  double moment(int n) const;
};

/// Hirshfeld atomic densities
class Hirshfeld {
 protected:
  /// List of atoms
  std::vector<HirshfeldAtom> atoms_;
  /// Centers
  std::vector<coords_t> cen_;

 public:
  /// Dummy constructor
  Hirshfeld();
  /// Destructor
  ~Hirshfeld();

  /// Set the atoms from precomputed centers, spacing and radial densities
  void set_atoms(const std::vector<coords_t> & cen, double dr, const std::vector< std::vector<double> > & rho);
  /// Atomic densities
  std::vector< std::vector<double> > rho() const;

  /// Compute
  void compute(const BasisSet & basis, std::string method);
  /// Load from checkpoints
  void load(const BasisSet & basis);

  /// Evaluate density at r
  double density(size_t inuc, const coords_t & r) const;
  /// Evaluate weight at r
  double weight(size_t inuc, const coords_t & r) const;
  /// Range of atom
  double range(size_t inuc) const;
  /// Calculate expectation values of radius (already includes r^2 factor)
  double moment(size_t inuc, int n) const;

  /// Print densities
  void print_densities() const;
};

#endif
