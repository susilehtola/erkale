/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/**
 * This file contains the necessary stuff needed to evaluate the TDLDA
 * integrals in Casida formalism.
 */

#ifndef ERKALE_CASIDAGRID
#define ERKALE_CASIDAGRID

#include "casida.h"
#include "dftgrid.h"

/// Perform integral over atomic grid
class CasidaAtom : public AtomGrid {
  /// Stack of values of orbitals at grid points: orbs[nspin][ngrid][norb]
  std::vector< std::vector< std::vector<double> > > orbs;

  /// Values of the exchange part of \f$ \frac {\delta^2 E_{xc} [\rho_\uparrow,\rho_\downarrow]} {\delta \rho_\sigma ({\bf r}) \delta \rho_\tau ({\bf r})} \f$
  std::vector<double> fx;
  /// Values of the correlation part
  std::vector<double> fc;

 public:
  /// Constructor. Need to set tolerance as well before using constructor!
  CasidaAtom(bool lobatto=false, double tol=1e-4);
  /// Destructor
  ~CasidaAtom();

  /// Evaluate values of orbitals at grid points
  void compute_orbs(const std::vector<arma::mat> & C);

  /// Evaluate fx and fc
  void eval_fxc(int x_func, int c_func);

  /// Evaluate Kxc
  void Kxc(const std::vector< std::vector<states_pair_t> > & pairs, arma::mat & K) const;

  void free();
};

/// Perform integral over molecular grid
class CasidaGrid {
  /// Work grids
  std::vector<CasidaAtom> wrk;
  /// Atomic grids
  std::vector<atomgrid_t> grids;

  /// Basis set
  const BasisSet * basp;
  /// Verbose operation?
  bool verbose;
  /// Use Lobatto quadrature?
  bool use_lobatto;

  /// Construct grid
  void construct(const std::vector<arma::mat> & P, double tol, int x_func, int c_func);

 public:
  CasidaGrid(const BasisSet * bas, bool verbose=false, bool lobatto=false);
  ~CasidaGrid();

  /// Evaluate Kxc
  void Kxc(const std::vector<arma::mat> & P, double tol, int x_func, int c_func, const std::vector<arma::mat> & C, const std::vector < std::vector<states_pair_t> > & pairs, arma::mat & Kx);
};

#endif
