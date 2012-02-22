/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_ATOMGUESS
#define ERKALE_ATOMGUESS

#include "global.h"
#include "basis.h"

#include <vector>
#include <armadillo>

/**
 * Form starting guess density matrix from atomic ROHF densities
 *
 * "Starting SCF Calculations by Superposition of Atomic Densities"
 * by J. H. Van Lenthe et al., J. Comp. Chem. 27 (2006), pp. 926-932.
 */
void atomic_guess(const BasisSet & basis, arma::mat & C, arma::mat & E, bool verbose);

/// Determine list of identical nuclei, determined by nucleus and basis set
std::vector< std::vector<size_t> > identical_nuclei(const BasisSet & basis);

/// Electronic configuration
typedef struct {
  /// Primary quantum number
  int n;
  /// Angular quantum number
  int l;
} el_conf_t;

/// Sorting operator for determining occupations
bool operator<(const el_conf_t & lhs, const el_conf_t & rhs);

/// Get ordering of occupations
std::vector<el_conf_t> get_occ_order(int nmax);

/// Ground state configuration
typedef struct {
  /// Spin multiplicity 2S+1
  int mult;
  /// Angular momentum
  int L;
  /// Total angular momentum * 2
  int dJ;
} gs_conf_t;

/// Determine ground state symbol
gs_conf_t get_ground_state(int Z);


#endif
