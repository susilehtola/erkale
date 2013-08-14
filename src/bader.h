/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2013
 * Copyright (c) 2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_BADER
#define ERKALE_BADER

#include "global.h"
#include "basis.h"

/// Helper for tracking points in current path
typedef struct {
  /// x index
  size_t ix;
  /// y index
  size_t iy;
  /// z index
  size_t iz;
} baderpoint_t;

/**
 * Performs Bader analysis of the electron density.
 *
 * The routine is based on the article
 *
 * W. Tang, E. Sanville and G. Henkelman, "A grid-based Bader analysis
 * algorithm without lattice bias", J. Phys.: Condens. Matter 21
 * (2009) 084204.
 */
class Bader {
  /// List of nuclear coordinates
  arma::mat nuclei;

  /// Density array
  arma::cube dens;
  /// Which Bader region the points belong to
  arma::icube region;
  /// Index for next Bader region
  arma::sword index;

  /// Starting point of grid
  arma::vec start;
  /// Grid spacing
  arma::vec spacing;

  /// Check that point is in the cube
  bool in_cube(const arma::ivec & p) const;
  /// Are the neighbors of the point assigned?
  bool neighbors_assigned(const arma::ivec & p) const;
  /// Is the point a local maximum
  bool local_maximum(const arma::ivec & p) const;
  /// Is the point on a Bader region boundary?
  bool on_boundary(const arma::ivec & p) const;

  /// Determine nuclear regions
  arma::ivec nuclear_regions() const;

  /// Run classification on the grid point p. Returns a list of points
  /// on the trajectory.
  std::vector<arma::ivec> classify(arma::ivec p);
  
 public:
  /// Constructor
  Bader();
  /// Destructor
  ~Bader();

  /// Fill grid
  void fill(const BasisSet & basis, const arma::mat & P, double spacing=0.1*ANGSTROMINBOHR, double padding=2.5*ANGSTROMINBOHR);

  /// Perform Bader analysis
  void analysis();

  /// Get Bader charges
  arma::vec charges() const;
};

#endif
