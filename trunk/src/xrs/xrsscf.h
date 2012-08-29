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


#ifndef ERKALE_XRSSCF
#define ERKALE_XRSSCF

#include "global.h"
#include <armadillo>
#include "scf.h"

enum xrs_method {
  // Transition potential: one half electron in excited initial state, system has net charge +0.5
  TP,
  // Full hole: no electron in excited initial state, system has net charge +1
  FCH,
  // XCH: Full hole, but excited electron placed on LUMO
  XCH
};

class XRSSCF : public SCF {
  /// Excite beta spin?
  bool spin;
  /// Number of alpha electrons
  int nocca;
  /// Number of beta electrons 
  int noccb;

 public:
  /// Constructor
  XRSSCF(const BasisSet & basis, const Settings & set, Checkpoint & chkpt, bool spin);
  /// Destructor
  ~XRSSCF();

  /// Compute 1st core-excited state
  size_t full_hole(size_t xcatom, uscf_t & sol, convergence_t conv, dft_t dft, bool xch) const;
  /// Compute TP solution
  size_t half_hole(size_t xcatom, uscf_t & sol, convergence_t conv, dft_t dft) const;
};

/// Get excited atom from atomlist
size_t get_excited_atom_idx(std::vector<atom_t> & at);

/// Compute center and rms width of orbitals.
arma::mat compute_center_width(const arma::mat & C, const BasisSet & basis, size_t nocc);

/// Find excited core orbital, located on atom idx
size_t find_excited_orb(const arma::mat & C, const BasisSet & basis, size_t atom_idx, size_t nocc);

/// Print information about orbitals
void print_info(const arma::mat & C, const arma::vec & E, const BasisSet & basis);

/// Aufbau occupation
std::vector<double> norm_occ(size_t nocc);
/// Set fractional occupation on excited orbital
std::vector<double> tp_occ(size_t excited, size_t nocc);
/// First excited state; core orbital is not occupied
std::vector<double> xch_occ(size_t excited, size_t nocc);
/// Full hole; core orbital is not occupied
std::vector<double> fch_occ(size_t excited, size_t nocc);

/// Localize orbitals, returns number of localized orbitals.
size_t localize(const BasisSet & basis, int nocc, size_t xcatom, arma::mat & C); 

#endif
