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

#ifndef ERKALE_BADERGRID
#define ERKALE_BADERGRID

#include "global.h"
#include "basis.h"
#include "dftgrid.h"

/**
 * The algorithms here are original work, but actually are very much
 * similar to the work in
 *
 * J. I. Rodríguez, A. M. Köster, P. W. Ayers, A. Santos-Valle,
 * A. Vela, and G. Merino, "An efficient grid-based scheme to compute
 * QTAIM atomic properties without explicit calculation of zero-flux
 * surfaces", J. Comp. Chem. 30, 1082 (2009).
 */

/**
 * Integration over Bader regions
 */
class BaderGrid {
  /// Basis set
  const BasisSet *basp;
  /// Grid worker
  AngularGrid wrk;

  /// Locations of maxima
  std::vector<coords_t> maxima;
  /// Grid points corresponding to the regions
  std::vector< std::vector<gridpoint_t> > reggrid;
  /// Amount of nuclei
  size_t Nnuc;

  /// Verbose operation?
  bool verbose;
  /// Print maxima
  void print_maxima() const;

 public:
  /// Constructor
  BaderGrid();
  /// Destructor
  ~BaderGrid();

  /// Set parameters
  void set(const BasisSet & basis, bool verbose=true, bool lobatto=false);

  /// Construct grid with AO overlap matrix threshold thr and classify points into regions with P
  void construct_bader(const arma::mat & P, double thr);
  /// Construct grid with AO overlap matrix threshold thr and classify points into Voronoi regions
  void construct_voronoi(double tol);
  /// Get amount of regions
  size_t get_Nmax() const;

  /// Compute regional charges
  arma::vec regional_charges(const arma::mat & P);
  /// Compute nuclear charges
  arma::vec nuclear_charges(const arma::mat & P);

  /// Compute regional overlap matrices
  std::vector<arma::mat> regional_overlap();
  /// Compute regional overlap matrix
  arma::mat regional_overlap(size_t ireg);
};

/// Track point to maximum
coords_t track_to_maximum(const BasisSet & basis, const arma::mat & P, const coords_t r0, size_t & nd, size_t & ng);


#endif
