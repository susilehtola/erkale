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
class BaderAtom: public AtomGrid {
 public:
  /// Constructor
  BaderAtom(bool lobatto, double tol=1e-4);
  /// Destructor
  ~BaderAtom();
  
  /// Classify points into regions. Returns region list
  std::vector<arma::sword> classify(const BasisSet & basis, const arma::mat & P, std::vector<coords_t> & maxima, size_t & ndens, size_t & ngrad);
  /// Classify points into Voronoi regions. Returns region list
  std::vector<arma::sword> classify_voronoi(const BasisSet & basis);

  /// Integrate charge
  void charge(const BasisSet & basis, const arma::mat & P, const std::vector<arma::sword> & regions, arma::vec & q) const;

  /// Calculate regional overlap matrices
  void regional_overlap(const std::vector<arma::sword> & regions, std::vector<arma::mat> & stack) const;
  /// Calculate regional overlap matrix
  void regional_overlap(const std::vector<arma::sword> & regions, size_t ireg, arma::mat & Sat) const;
};

/**
 * Integration over Bader regions
 */
class BaderGrid {
  /// Work grid
  std::vector<BaderAtom> wrk;
  /// Atomic grids
  std::vector<atomgrid_t> grids;

  /// Locations of maxima
  std::vector<coords_t> maxima;
  /// Classifications of grid points
  std::vector< std::vector<arma::sword> > regions;

  /// Basis set
  const BasisSet * basp;
  /// Verbose operation?
  bool verbose;
  /// Use Lobatto quadrature?
  bool use_lobatto;

  /// Print maxima
  void print_maxima() const;
  
 public:
  /// Dummy constructor
  BaderGrid();
  /// Constructor
  BaderGrid(const BasisSet * bas, bool verbose=true, bool lobatto=false);
  /// Destructor
  ~BaderGrid();

  /// Create grid for Bader charges (optimize overlap matrix)
  void construct(double tol);

  /// Run classification
  void classify(const arma::mat & P);
  /// Run Voronoi classification
  void classify_voronoi();
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
