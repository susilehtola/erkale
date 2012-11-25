/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#ifndef ERKALE_LMGRID
#define ERKALE_LMGRID

#include <complex>
#include <vector>
#include "basis.h"
#include "chebyshev.h"
#include "lobatto.h"

/// Index of (l,m) in tables: l^2 + l + m
#define lmind(l,m) ( ((size_t) (l))*(size_t (l)) + (size_t) (l) + (size_t) (m))

/// Structure for radial integration
typedef struct {
  /// Radius
  double r;
  /// Radial weight
  double w;
} radial_grid_t;

/// Form radial grid
std::vector<radial_grid_t> form_radial_grid(int nrad);

/// Structure for angular integration
typedef struct {
  /// Coordinate on the (unit) sphere
  coords_t r;
  /// Weight
  double w;
} angular_grid_t;

/// Form angular grid
std::vector<angular_grid_t> form_angular_grid(int lmax);
/// Compute values of spherical harmonics Ylm[ngrid][l,m]
std::vector< std::vector< std::complex<double> > > compute_spherical_harmonics(const std::vector<angular_grid_t> & grid, int lmax);

/// Expansion of orbitals
typedef struct {
  /// Radial grid
  std::vector<radial_grid_t> grid;
  /// Expansion coefficients of orbitals clm[norbs][l,m][nrad]
  std::vector< std::vector< std::vector< std::complex<double> > > > clm;
} expansion_t;

/// Compute expansion of orbitals around cen, return clm[norbs][l,m][nrad] up to lmax. Quadrature uses lquad:th order
expansion_t expand_orbitals(const arma::mat & C, const BasisSet & bas, const coords_t & cen, size_t Nrad, int lmax, int lquad);

#endif
