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

#ifndef ERKALE_EMD
#define ERKALE_EMD

#include <cstdio>
#include <vector>
#include "../basis.h"

/// One-center term in electron momentum density
typedef struct {
  /// Expansion coefficient
  double c;
  /// p^pm
  int pm;
  /// exp(-z*p^2)
  double z;
} onecenter_t;

/// Two-center term in electron momentum density \f$ c_i j_l (p \Delta r_i) p^{m} exp(-z p^2) \f$
typedef struct {
  /// List of contraction coefficients \f$ c_i \f$
  std::vector<double> c;
  /// List of displacements \f$ \Delta r_i \f$
  std::vector<double> dr;
  /// j_l (p * dr_i)
  int l;
  /// p^pm
  int pm;
  /// exp(-z*p^2)
  double z;
} twocenter_t;

/// Comparison operator for one-center terms
bool operator<(const onecenter_t & lhs, const onecenter_t & rhs);
/// Equivalence operator for one-center term
bool operator==(const onecenter_t & lhs, const onecenter_t & rhs);

/// Comparison operator for two-center terms
bool operator<(const twocenter_t & lhs, const twocenter_t & rhs);
/// Equivalence operator for two-center term
bool operator==(const twocenter_t & lhs, const twocenter_t & rhs);

/// Find out basis functions of the same type
std::vector< std::vector<size_t> > find_identical_shells(const BasisSet & bas);

/**
 * \class EMDEvaluator
 *
 * \brief Functions for evaluating the momentum density.
 *
 * This class evaluates the momentum density from the given density
 * matrix, using the procedure described in
 *
 * J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen, "Calculation of
 * isotropic Compton profiles with Gaussian basis sets",
 * Phys. Chem. Chem. Phys 13 (2011), pp. 5630 - 5641.
 *
 * \author Jussi Lehtola
 * \date 2011/03/08 17:13
 */

class EMDEvaluator {
  /// List of two-center terms
  std::vector<twocenter_t> twoc;
  /// List of one-center terms
  std::vector<onecenter_t> onec;

  /// Add one-center term
  void add_1c(double c, int pm, double z);
  /// Add two-center term
  void add_2c(double c, double dr, int l, int pm, double z);
  /// Add contraction in two-center term
  void add_2c_contr(double c, double dr, size_t ind);

  /// Evaluate one-center terms at p
  double eval_onec(double p) const;
  /// Evaluate two-center terms at p
  double eval_twoc(double p) const;

 public:
  /// Dummy constructor
  EMDEvaluator();
  /// Constructor
  EMDEvaluator(const BasisSet & bas, const arma::mat & P);
  /// Destructor
  ~EMDEvaluator();

  /// Evaluate radial electron momentum density at p
  double eval(double p) const;
  
  /// Get amount of one-center terms
  size_t getN_onec() const;
  /// Get amount of two-center terms
  size_t getN_twoc() const;
  /// Get total amount of two-center terms
  size_t getN_twoc_total() const;

  /// Print info
  void print() const;
  
  /// Clean terms with zero contribution
  void clean();
};

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
 * momentum density and Compton profiles. The algorithm has been
 * described in
 *
 * J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen, "Calculation of
 * isotropic Compton profiles with Gaussian basis sets",
 * Phys. Chem. Chem. Phys 13 (2011), pp. 5630 - 5641.
 *
 * \author Jussi Lehtola
 * \date 2011/03/08 17:13
 */

class EMD {
  /// List of radial densities
  std::vector<emd_t> dens;
  /// Number of electrons
  int Nel;
  /// Norm of density matrix
  double dmnorm;

  /// Evaluator
  EMDEvaluator eval;

  /// Add 4 points at ind
  void add4(size_t ind);
 public:
  /// Constructor
  EMD(const BasisSet & bas, const arma::mat & P);
  /// Destructor
  ~EMD();

  /// Initial filling of grid
  void initial_fill();
  /// Continue filling until number of electrons is reproduced within tolerance
  void find_electrons(double tol=1e-4);
  /// Optimize physical moments of EMD within tolerance
  void optimize_moments(double tol=1e-10);

  /// Save values of momentum density
  void save(const char * fname) const;
  /// Save moments of momentum density
  void moments(const char * fname) const;
  /// Calculate Compton profile in "raw" and interpolated form, input are filenames
  void compton_profile(const char * raw, const char * interp) const;
};

#endif
