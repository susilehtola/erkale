/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/**
 * \class ERIscreen
 *
 * \brief On-the-fly calculation of electron repulsion integrals
 *
 * This class performs on-the-fly formations of the Hartree-Fock
 * exchange and Coulomb matrices, that are needed calculations of
 * large systems for which the integrals don't fit into memory.  The
 * integrals are screened adaptively by the use of the Schwarz
 * inequality.
 *
 * \author Susi Lehtola
 * \date 2011/05/12 19:44
*/

#ifndef ERKALE_ERISCREEN
#define ERKALE_ERISCREEN

class IntegralDigestor;
class ForceDigestor;
#include "global.h"
#include <armadillo>
#include <memory>
#include <vector>

#include "basis.h"
#include "eriworker.h"

/// Screening of electron repulsion integrals
class ERIscreen {
  /// Integral pairs sorted by value
  std::vector<eripair_t> shpairs;
  /// Shell-pair screening matrices
  arma::mat Q, M;
  /// Number of basis functions
  size_t Nbf;

  /// Pointer to the used basis set
  const BasisSet * basp;
  /// Index helper
  std::vector<size_t> iidx;

  /// Range separation parameter
  double omega;
  /// Fraction of long-range exchange
  double alpha;
  /// Fraction of short-range exchange
  double beta;

  /// Per-thread ERIWorker / dERIWorker pool. Sized to
  /// omp_get_max_threads() in fill(); each thread lazily constructs
  /// its slot on first use via acquire_eri / acquire_deri. Workers
  /// are reused across calcJ / calcK / calcJK / forceJ / forceK
  /// calls (workers' internal state is overwritten on every
  /// compute() so cross-call reuse is safe). Pools are cleared
  /// whenever set_range_separation or fill change the parameters
  /// the constructors baked in.
  ///
  /// mutable so the const calc* / force* members can populate.
  mutable std::vector< std::unique_ptr<ERIWorker> > eri_pool_;
  mutable std::vector< std::unique_ptr<dERIWorker> > deri_pool_;

  /// Lazily construct & return the per-thread ERI worker.
  ERIWorker * acquire_eri(int ith) const;
  /// Lazily construct & return the per-thread derivative ERI worker.
  dERIWorker * acquire_deri(int ith) const;

  /// Run calculation with given digestor
  void calculate(std::vector< std::vector<IntegralDigestor *> > & digest, double tol) const;
  /// Run force calculation with given digestor
  arma::vec calculate_force(std::vector< std::vector<ForceDigestor *> > & digest, double tol) const;

 public:
  /// Constructor
  ERIscreen();
  /// Destructor
  ~ERIscreen();

  /// Get amount of basis functions
  size_t get_N() const;

  /// Set range separation
  void set_range_separation(double omega, double alpha, double beta);
  void set_range_separation(const RangeSeparation & rs) { set_range_separation(rs.omega, rs.alpha, rs.beta); }
  /// Get range separation
  void get_range_separation(double & omega, double & alpha, double & beta) const;
  RangeSeparation get_range_separation() const { RangeSeparation rs; get_range_separation(rs.omega, rs.alpha, rs.beta); return rs; }

  /// Form screening matrix, return amount of significant shell pairs
  size_t fill(const BasisSet * basis, double shtol, bool verbose=true);

  /// Calculate Coulomb matrix with tolerance tol for integrals
  arma::mat calcJ(const arma::mat & P, double tol) const;
  /// Calculate set of Coulomb and exchange matrices with tolerance tol for integrals (for PZ-SIC)
  std::vector<arma::cx_mat> calcJK(const std::vector<arma::cx_mat> & P, double jfrac, double kfrac, double tol) const;

  /// Calculate exchange matrix with tolerance tol for integrals
  arma::mat calcK(const arma::mat & P, double tol) const;
    /// Calculate exchange matrix with tolerance tol for integrals
  arma::cx_mat calcK(const arma::cx_mat & P, double tol) const;
  /// Calculate  exchange matrices with tolerance tol for integrals
  void calcK(const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ka, arma::mat & Kb, double tol) const;
  /// Calculate  exchange matrices with tolerance tol for integrals
  void calcK(const arma::cx_mat & Pa, const arma::cx_mat & Pb, arma::cx_mat & Ka, arma::cx_mat & Kb, double tol) const;

  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals
  void calcJK(const arma::mat & P, arma::mat & J, arma::mat & K, double tol) const;
  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals
  void calcJK(const arma::cx_mat & P, arma::mat & J, arma::cx_mat & K, double tol) const;
  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals, unrestricted calculation
  void calcJK(const arma::mat & Pa, const arma::mat & Pb, arma::mat & J, arma::mat & Ka, arma::mat & Kb, double tol) const;
  /// Calculate Coulomb and exchange matrices at the same time with tolerance tol for integrals, unrestricted calculation
  void calcJK(const arma::cx_mat & Pa, const arma::cx_mat & Pb, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb, double tol) const;

  /* Force calculation */

  /// Calculate Coulomb force with tolerance tol for integrals
  arma::vec forceJ(const arma::mat & P, double tol) const;
  /// Calculate exchange force with tolerance tol for integrals
  arma::vec forceK(const arma::mat & P, double tol, double kfrac) const;
  /// Calculate Coulomb and exchange forces at the same time with tolerance tol for integrals, unrestricted calculation
  arma::vec forceK(const arma::mat & Pa, const arma::mat & Pb, double tol, double kfrac) const;
  /// Calculate Coulomb and exchange forces at the same time with tolerance tol for integrals
  arma::vec forceJK(const arma::mat & P, double tol, double kfrac) const;
  /// Calculate Coulomb and exchange forces at the same time with tolerance tol for integrals, unrestricted calculation
  arma::vec forceJK(const arma::mat & Pa, const arma::mat & Pb, double tol, double kfrac) const;
};

#endif
