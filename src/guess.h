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

#ifndef ERKALE_ATOMGUESS
#define ERKALE_ATOMGUESS

#include "global.h"
class BasisSet;

#include <vector>
#include <armadillo>

/**
 * Form starting guess density matrix from atomic densities
 *
 * "Starting SCF Calculations by Superposition of Atomic Densities"
 * by J. H. Van Lenthe et al., J. Comp. Chem. 27 (2006), pp. 926-932.
 *
 * However, contrary to the above paper, this version computes the
 * atomic electron density with the same method as the main
 * calculation instead of HF.
 *
 * Optional charge can be given as input.
 */
arma::mat sad_guess(const BasisSet & basis);

/// Same, but do SAP guess by projecting Fock matrices. Not as good as the real-space version
arma::mat sap_guess(const BasisSet & basis);

/**
 * Form starting guess from a Huckel type calculation. See
 *
 * "An assessment of initial guesses for self-consistent field
 * calculations. Superposition of Atomic Potentials: simple yet
 * efficient" by S. Lehtola, J. Chem. Theory Comput. (2019), 
 * DOI: 10.1021/acs.jctc.8b01089.
 *
 * "Phosphorescence parameters for platinum (II) organometallic
 * chromophores: A study at the non-collinear four-component Kohn–Sham
 * level of theory" by P. Norman and H. J. Aa. Jensen,
 * Chem. Phys. Lett. 531 (2012), pp. 229-235.
 */
arma::mat huckel_guess(const BasisSet & basis, double Kgwh);

/**
 * Forms a projection to a minimal atomic basis set.
 */
arma::mat minimal_basis_projection(const BasisSet & basis);

/**
 * Worker routine - perform guess for inuc:th atom in basis, using given method.
 *
 * Returns the shells on the atom, with shellidx containing the
 * original indices of the shells on the list obtained with get_funcs.
 *
 * Uses spherically averaged occupations.
 *
 * Optional charge given as input.
 */
void atomic_guess(const BasisSet & basis, size_t inuc, const std::string & method, std::vector<size_t> & shellidx, BasisSet & atbas, arma::vec & atE, arma::mat & atC, arma::mat & atP, arma::mat & atF, int Q);

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
