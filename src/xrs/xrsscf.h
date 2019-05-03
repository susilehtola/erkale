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


#ifndef ERKALE_XRSSCF
#define ERKALE_XRSSCF

#include "../global.h"
#include <armadillo>
#include "../scf.h"

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

  /// Initial core hole orbital, used to identify the state by maximal overlap
  arma::vec coreorb;

 public:
  /// Constructor
  XRSSCF(const BasisSet & basis, Checkpoint & chkpt, bool spin);
  /// Destructor
  ~XRSSCF();

  /// Set core hole state [Iannuzzi and Hütter, PCCP 9, 1559 (2007)]
  void set_core(const arma::vec & c);
  /// Get core hole state
  arma::vec get_core() const;

  /// Compute 1st core-excited state
  size_t full_hole(uscf_t & sol, double convthr, dft_t dft, bool xch);
  /// Compute TP solution
  size_t half_hole(uscf_t & sol, double convthr, dft_t dft);

  /// Compute 1st core-excited state using line search (slow!)
  size_t full_hole_ls(size_t xcatom, uscf_t & sol, double convthr, dft_t dft, bool xch);
  /// Compute TP solution using line search (slow!)
  size_t half_hole_ls(size_t xcatom, uscf_t & sol, double convthr, dft_t dft);

  /// Get Fock operator for 1st core-excited state
  void Fock_full_hole(uscf_t & sol, dft_t dft, const std::vector<double> & occa, const std::vector<double> & occb, DFTGrid & grid, DFTGrid & nlgrid, bool xch) const;
  /// Get Fock operator for TP state
  void Fock_half_hole(uscf_t & sol, dft_t dft, const std::vector<double> & occa, const std::vector<double> & occb, DFTGrid & grid, DFTGrid & nlgrid) const;
};

/// Find excited core orbital
size_t find_excited_orb(const BasisSet & basis, const arma::vec & xco, const arma::mat & C, int nocc);

/// Get excited atom from atomlist
size_t get_excited_atom_idx(std::vector<atom_t> & at);

/// Construct list of atoms of the same type
std::vector<size_t> atom_list(const BasisSet & basis, size_t xcatom, bool verbose);

/// Aufbau occupation
std::vector<double> norm_occ(size_t nocc);
/// Set fractional occupation on excited orbital
std::vector<double> tp_occ(size_t excited, size_t nocc);
/// First excited state; core orbital is not occupied
std::vector<double> xch_occ(size_t excited, size_t nocc);
/// Full hole; core orbital is not occupied
std::vector<double> fch_occ(size_t excited, size_t nocc);

/// Localize orbitals on wanted atom. Return index of wanted core orbital.
size_t localize(const BasisSet & basis, int nocc, size_t xcatom, arma::mat & C, const std::string & state, int iorb);

#endif
