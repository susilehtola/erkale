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


#ifndef ERKALE_LMTRANS
#define ERKALE_LMTRANS

#include "../global.h"
#include <armadillo>

#include "../gaunt.h"
#include "../lmgrid.h"

/// Radial integrals stack
typedef struct {
  /// Initial state
  size_t i;
  /// Final state
  size_t f;
  /// Momentum transfer
  double q;

  /// Stack of radial integrals, for different values of l
  std::vector<arma::cx_mat> itg;
} rad_int_t;

/// Bessel function stack
typedef struct {
  /// Momentum transfer
  double q;
  /// Stack of Bessel functions, for different values l
  std::vector< std::vector<double> > jl;
} bessel_t;

/// Class for calculating XRS
class lmtrans {
  /// Orbital expansions
  expansion_t exp;
  /// Maximum angular momentum in expansion
  int lmax;
  /// Table of Gaunt coefficients
  Gaunt gaunt;

  /**
   * Compute radial integral between initial state i component (li,mi)
   * and final state f component (lf,mf). Uses tabled Bessel function
   * values for momentum transfer q.  Returns R[li,mi][lf,mf]
   */
  arma::cx_mat radial_integral(size_t i, size_t f, int l, const bessel_t & bes) const;

 public:
  /// Dummy constructor
  lmtrans();
  /// Generate expansion for given orbitals
  lmtrans(const arma::mat & C, const BasisSet & bas, const coords_t & cen, size_t Nrad=200, int lmax=5, int lquad=30);
  /// Destructor
  ~lmtrans();

  /**
   * Compute radial integrals in itg for transition from initial state
   * i to final state f, when momentum transfer is q.
   */
  rad_int_t compute_radial_integrals(size_t i, size_t f, const bessel_t & bes) const;

  /**
   * Compute Bessel function stack
   */
  bessel_t compute_bessel(double q) const;

  /**
   * Compute transition amplitude matrix from li component of initial
   * state i to lf component of final state f when momentum transfer
   * is (qx,qy,qz). Uses radial integrals from itg. Returns A(li,lf).
   */
  arma::cx_mat transition_amplitude(const rad_int_t & rad, double qx, double qy, double qz) const;

  /**
   * Compute transition velocity from initial state to final state f.
   * Returns a vector with first element giving the total transition
   * velocity, whereas the rest give the final state velocities as in
   * Sakko et al.
   */
  std::vector<double> transition_velocity(size_t i, size_t f, const bessel_t & bes) const;

  /// Print information about orbitals
  void print_info() const;

  /// Write radial distribution of orbital o into file
  void write_prob(size_t o, const std::string & fname) const;
};



#endif
