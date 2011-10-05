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

class XRSSCF : public SCF {
 public:
  XRSSCF(const BasisSet & basis, const Settings & set);
  ~XRSSCF();

  size_t full_hole(size_t xcatom, uscf_t & sol, convergence_t conv, dft_t dft) const;
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
std::vector<double> frac_occ(size_t excited, size_t nocc);
/// First excited state; core orbital is not occupied
std::vector<double> exc_occ(size_t excited, size_t nocc);

#endif
