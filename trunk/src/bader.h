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
  /// List of nuclei
  std::vector<nucleus_t> nuclei;
  /// List of nuclear coordinates
  arma::mat nucc;

  /// Density array
  arma::cube dens;
  /// Which Bader region the points belong to
  arma::icube region;
  /// Amount of Bader regions
  arma::sword Nregions;
  /// Array size
  arma::ivec array_size;

  /// Starting point of grid
  arma::vec start;
  /// Grid spacing
  arma::vec spacing;

  /// Verbose operation?
  bool verbose;

  /// Check that all points have been classified
  void check_regions(std::string msg="") const;

  /// Check that point is in the cube
  bool in_cube(const arma::ivec & p) const;
  /// Is the point on an edge of the cube?
  bool on_edge(const arma::ivec & p) const;

  /// Are the neighbors of the point assigned? nnei gives the maximum displacement to look at
  bool neighbors_assigned(const arma::ivec & p, int nnei=1) const;
  /// Is the point a local maximum
  bool local_maximum(const arma::ivec & p) const;
  /// Is the point on a Bader region boundary?
  bool on_boundary(const arma::ivec & p, int nnei=1) const;

  /// Compute gradient
  arma::vec gradient(const arma::ivec & p) const;

  /// Run classification on the grid point p. Returns a list of points
  /// on the trajectory.
  std::vector<arma::ivec> classify(arma::ivec p) const;

  /// Reorder regions to nuclear order
  void reorder();

  /// Print neighbors of point
  void print_neighbors(const arma::ivec & p) const;

  /// Perform Bader analysis
  void analysis();

 public:
  /// Constructor
  Bader(bool verbose=true);
  /// Destructor
  ~Bader();

  /// Fill grid and analyse
  void analyse(const BasisSet & basis, const arma::mat & P, double spacing=0.025*ANGSTROMINBOHR, double padding=5.0*ANGSTROMINBOHR);

  /// Determine nuclear regions
  arma::ivec nuclear_regions() const;

  /// Get charges in the Bader regions
  arma::vec regional_charges() const;
  /// Get nuclear charges
  arma::vec nuclear_charges() const;

  /// Write out Bader regions
  void print_regions() const;
  /// Write out individual Bader regions
  void print_individual_regions() const;

  /// Compute Bader charges using existing grid but input density
  arma::vec regional_charges(const BasisSet & basis, const arma::mat & P) const;
  /// Get nuclear charges
  arma::vec nuclear_charges(const BasisSet & basis, const arma::mat & P) const;
  
  /// Compute Bader regional overlap matrices
  std::vector<arma::mat> regional_overlap(const BasisSet & basis) const;
};

#endif
