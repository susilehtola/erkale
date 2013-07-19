/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


/**
 * The routines in this file provide basis set agnostic tools for
 * evaluating the radial electron momentum density and its
 * moments. For the formalism see chapter 3.8 in Susi Lehtola's PhD
 * thesis "Computational Modeling of the Electron Momentum Density",
 * University of Helsinki 2013. 
 *
 * A full electronic version is available at
 * http://urn.fi/URN:ISBN:978-952-10-8091-3
 */

#ifndef ERKALE_RADEMD
#define ERKALE_RADEMD

#include "basis.h"
#include "global.h"
#include "../emd/spherical_expansion.h"

#include <complex>
#include <vector>
#include <armadillo>

/// Class for (basis set independent) normalized radial wfs
class RadialFourier {
 protected:
  /// l value
  int l;

 public:
  /// Constructor
  RadialFourier(int l);
  /// Destructor
  virtual ~RadialFourier();

  /// Get l value
  int getl() const;
  // Print expansion
  virtual void print() const = 0;

  /**
   * Calculate radial function at p. Must be overridden in the class
   * that implements the function.
   */
  virtual std::complex<double> get(double p) const = 0;
};

/// Coupling coefficient
typedef struct {
  /// l
  int l;
  /// l'
  int lp;

  // couple to

  /// L
  int L;
  /// M
  int M;

  /// The coupling coefficient
  std::complex<double> c;
} coupl_coeff_t;

/// Sorting operator
bool operator<(const coupl_coeff_t & lhs, const coupl_coeff_t & rhs);
/// Addition operator
bool operator==(const coupl_coeff_t & lhs, const coupl_coeff_t & rhs);

/// Coupling list
typedef struct {
  /// L
  int L;
  /// M
  int M;
  /// Coefficient
  std::complex<double> c;
} total_coupl_t;

/// Sorting operator
bool operator<(const total_coupl_t & lhs, const total_coupl_t & rhs);
/// Addition operator
bool operator==(const total_coupl_t & lhs, const total_coupl_t & rhs);

/// Add coupling to vector
void add_coupling_term(std::vector<total_coupl_t> & v, total_coupl_t & t);

/// Value of radial function
typedef struct {
  /// l
  int l;
  /// Value
  std::complex<double> f;
} radf_val_t;

/// List of identical functions
typedef struct {
  /// first index
  size_t i;
  /// second index
  size_t j;
} noneqradf_t;

/// Radial EMD evaluator
class EMDEvaluator {
  /**
   * Lists of identical functions (same radial and angular parts),
   * only difference comes from phase factor (different origins)
   */
  std::vector< std::vector<size_t> > idfuncs;
  /// The coupling coefficients of the nonequivalent functions
  std::vector< std::vector<coupl_coeff_t> > cc;

  /// The locations of the functions on the atoms (Nbas)
  std::vector<size_t> loc;

  /// The number of centers
  size_t Nat;
  /// The distances between the functions' origins (Nat x Nat)
  std::vector<double> dist;
  /// Spherical harmonics values, complex conjugated [Nat x Nat] [(L,M)]
  std::vector< std::vector< std::complex<double> > > YLM;

  /// The density matrix
  arma::mat P;

  /// Maximum value of L
  int Lmax;

  /// Computes the distance table
  void distance_table(const std::vector<coords_t> & coord);

  /// Computes the coupling coefficients
  void compute_coefficients(const std::vector< std::vector<ylmcoeff_t> > & clm);

  /// Add coupling coefficient
  void add_coupling(size_t ig, size_t jg, coupl_coeff_t c);

  /// Get the coupling constants for L=|l-lp|, ..., l+lp.
  std::vector<total_coupl_t> get_coupling(size_t ig, size_t jg, int l, int lp) const;

  /// Computes the ig:th radial function
  std::vector<radf_val_t> get_radial(size_t ig, double p) const;

  /// Get the total coupling (incl. radial function)
  std::vector<total_coupl_t> get_total_coupling(size_t ig, size_t jg, double p) const;

 protected:
  /**
   * Radial parts for each non-equivalent function. Needs a pointer
   * array, since otherwise the objects will be stripped to
   * RadialFourier.
   *
   * This array needs to be constructed in a basis-set specific subclass.
   */
  std::vector< std::vector<RadialFourier *> > rad;

 public:
  /// Dummy constructor
  EMDEvaluator();

  /**
   * Construct evaluator.
   *
   * idfuncs is the list of equivalent functions, idfuncs[ieq][1..N] = ibf
   * clm gives the lm expansion of the nonequivalent radial functions
   *
   * loc gives indices of centers of all of the basis functions
   * coord are coordinates of the individual atoms (centers)
   * P is the density matrix
   *
   * The radial functions need to be separately initialized.
   */
  EMDEvaluator(const std::vector< std::vector<size_t> > & idfuncsv, const std::vector< std::vector<ylmcoeff_t> > & clm, const std::vector<size_t> & locv, const std::vector<coords_t> & coord, const arma::mat & Pv);

  /// Destructor
  ~EMDEvaluator();

  /// Print the evaluator
  void print() const;
  /// Check norms of radial functions
  void check_norm() const;

  /// Evaluate radial EMD at p
  double get(double p) const;
};

/// Evaluate Bessel functions j_l(pr_i), return j(pr_i,l)
arma::mat bessel_array(const std::vector<double> & args, int lmax);


/// Structure for holding radial EMD
typedef struct {
  /// Radial momentum
  double p;
  /// Electron momentum density at p
  double d;
} emd_t;


/**
 * \class EMD
 *
 * \brief Functions for evaluating properties of the electron momentum density
 *
 * This class contains functions for computing moments of the electron
 * momentum density and Compton profiles. The adaptive grid algorithm
 * has been described in
 *
 * J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen, "Calculation of
 * isotropic Compton profiles with Gaussian basis sets",
 * Phys. Chem. Chem. Phys 13 (2011), pp. 5630 - 5641.
 *
 * \author Susi Lehtola
 * \date 2011/03/08 17:13
 */

class EMD {
  /// List of radial densities
  std::vector<emd_t> dens;
  /// Add 4 points at ind
  void add4(size_t ind);

 protected:
  /// Number of electrons
  int Nel;

  /// Evaluator
  const EMDEvaluator * eval;

 public:
  /// Constructor
  EMD(const EMDEvaluator * eval, int Nel);
  /// Destructor
  ~EMD();

  /// Initial filling of grid
  void initial_fill(bool verbose=true);
  /// Fill regions where density changes by huge amounts
  void complete_fill();

  /// Continue filling until number of electrons is reproduced within tolerance
  void find_electrons(bool verbose=true, double tol=1e-4);
  /// Optimize physical moments of EMD within tolerance
  void optimize_moments(bool verbose=true, double tol=1e-8);

  /// Get EMD
  std::vector<emd_t> get() const;

  /// Save values of momentum density
  void save(const char * fname) const;

  /// Calculate moments of momentum density
  arma::mat moments() const;
  /// Save moments of momentum density
  void moments(const char * fname) const;

  /// Calculate Compton profile
  arma::mat compton_profile() const;
  /// Save Compton profile in "raw" form
  void compton_profile(const char * raw) const;
  /// Save Compton profile in interpolated form
  void compton_profile_interp(const char * interp) const;
};

#endif
