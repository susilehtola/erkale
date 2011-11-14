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
  /// Excite beta spin?
  bool spin;
  /// Number of alpha electrons
  int nocca;
  /// Number of beta electrons 
  int noccb;

  /// List of frozen orbitals
  std::vector< std::vector<arma::vec> > freeze;
  
 public:
  /// Constructor
  XRSSCF(const BasisSet & basis, const Settings & set, Checkpoint & chkpt, bool spin);
  /// Destructor
  ~XRSSCF();

  /// Set frozen orbitals in ind:th symmetry group
  void set_frozen(const arma::mat & C, size_t ind);

  /// Compute 1st core-excited state
  size_t full_hole(size_t xcatom, uscf_t & sol, convergence_t conv, dft_t dft) const;
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
std::vector<double> frac_occ(size_t excited, size_t nocc);
/// First excited state; core orbital is not occupied
std::vector<double> exc_occ(size_t excited, size_t nocc);

/// Helper structure for localization 
typedef struct {
  /// Index of center
  size_t ind;
  /// Distance from excited nucleus
  double dist;
} locdist_t;

/// Operator for sorts
bool operator<(const locdist_t & lhs, const locdist_t & rhs);

/// Localize orbitals, returns number of localized orbitals.
size_t localize(const BasisSet & basis, int nocc, size_t xcatom, arma::mat & C); 

/// Get symmetry groups of orbitals
std::vector<int> symgroups(const arma::mat & C, const arma::mat & S, const std::vector< std::vector<arma::vec> > & freeze);

/// Freeze orbitals
void freeze_orbs(const std::vector< std::vector<arma::vec> > & freeze, const arma::mat & C, const arma::mat & S, arma::mat & H);

#endif
