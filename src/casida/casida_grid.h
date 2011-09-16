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

  /// Values of the exchange part of \lf$ \frac {\delta^2 E_{xc} [\rho_\uparrow,\rho_\downarrow]} {\delta \rho_\sigma ({\bf r}) \delta \rho_\tau ({\bf r})} \lf$
  std::vector<double> fx;
  /// Values of the correlation part
  std::vector<double> fc;

 public:
  /// Dummy constructor
  CasidaAtom();
  /// Construct unpolarized atomic grid
  CasidaAtom(const BasisSet & bas, const arma::mat & P, size_t cenind, double toler, int x_func, int c_func, bool lobatto, bool verbose);
  /// Construct polarized atomic grid
  CasidaAtom(const BasisSet & bas, const arma::mat & Pa, const arma::mat & Pb, size_t cenind, double toler, int x_func, int c_func, bool lobatto, bool verbose);
  /// Destructor
  ~CasidaAtom();

  /// Evaluate values of orbitals at grid points
  void compute_orbs(const std::vector<arma::mat> & C);

  /// Evaluate fx and fc
  void eval_fxc(int x_func, int c_func);

  /// Evaluate Kxc
  arma::mat Kxc(const std::vector<states_pair_t> & pairs, bool spin) const;

  void free();
};

/// Perform integral over molecular grid
class CasidaGrid {
  /// Atomic grids
  std::vector<CasidaAtom> atoms;
  /// Basis set
  const BasisSet * basp;
  /// Direct calculation?
  bool direct;
  /// Verbose operation?
  bool verbose;
  /// Use Lobatto quadrature?
  bool use_lobatto;

  /// Construct grid
  void construct(const std::vector<arma::mat> & P, double tol, int x_func, int c_func);

 public:
  CasidaGrid(const BasisSet * bas, bool dir=0, bool ver=0, bool lobatto=0);
  ~CasidaGrid();

  /// Evaluate Kxc
  void Kxc(const std::vector<arma::mat> & P, double tol, int x_func, int c_func, const std::vector<arma::mat> & C, const std::vector < std::vector<states_pair_t> > & pairs, std::vector<arma::mat> & Kx);
};

#endif
