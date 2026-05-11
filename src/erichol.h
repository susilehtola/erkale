/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_ERICHOL
#define ERKALE_ERICHOL

#include "global.h"
#include "basis.h"
#include <set>

/// Cholesky decomposition of ERIs
class ERIchol {
  /// Amount of basis functions
  size_t Nbf;
  /// Map of product indices in full space (for getting density subvector)
  arma::uvec prodidx;
  /// Map to function indices, 2 x Nprod
  arma::umat invmap;
  /// Map to product index
  arma::umat prodmap;
  /// List of off-diagonal products
  arma::uvec odiagidx;
  /// Cholesky vectors, Nprod x length
  arma::mat B;

  /// Range separation constant
  double omega;
  /// Fraction of full-range Coulomb
  double alpha;
  /// Fraction of short-range Coulomb
  double beta;

  /// Pivot indices
  arma::uvec pi;
  /// Pivot shell-pairs
  std::set< std::pair<size_t, size_t> > pivot_shellpairs;
  /// Get the pivot shell pairs
  void form_pivot_shellpairs(const BasisSet & Basis);

  /// Two-step CD state. Filled by fill_two_step and used by
  /// forceJ. Empty when populated via fill() (one-step CD).
  /// two_step_metric_(p, q) = (piv_p | piv_q) over selected pivot
  /// orbital pairs (size Nselected x Nselected).
  arma::mat two_step_metric_;
  /// two_step_metric_invh_ = PartialCholeskyOrth(metric, ...);
  /// satisfies two_step_metric_invh_^T M two_step_metric_invh_ = I
  /// (over the lindep-cleaned subspace). Used to map between raw
  /// gamma vectors (length Nselected) and the orthonormalised L
  /// indexing of B. Size (Nselected x Naux_eff).
  arma::mat two_step_metric_invh_;
  /// True iff this object was filled via fill_two_step (so the
  /// metric / metric_invh members are valid).
  bool two_step_;

 public:
  /// Constructor
  ERIchol();
  /// Destructor
  ~ERIchol();

  /// Set range separation
  void set_range_separation(double w, double a, double b);
  void set_range_separation(const RangeSeparation & rs) { set_range_separation(rs.omega, rs.alpha, rs.beta); }
  void get_range_separation(double & w, double & a, double & b) const;
  RangeSeparation get_range_separation() const { RangeSeparation rs; get_range_separation(rs.omega, rs.alpha, rs.beta); return rs; }

  /// Load B matrix
  void load();
  /// Save B matrix
  void save() const;

  /// Fill matrix via one-step pivoted Cholesky decomposition on the
  /// full molecular (uv|ls) ERI tensor. Returns amount of
  /// significant (uv) pairs.
  size_t fill(const BasisSet & basis, double cholesky_tol, double shell_reuse_thr, double shell_screen_tol, bool verbose);

  /// Fill matrix via two-step pivoted Cholesky decomposition. The
  /// pivot-selection phase is identical to fill() (shellpair-batched
  /// libint calls, shell_reuse_thr gating), so the kept pivots are
  /// the same orbital-pair set as one-step CD at the same threshold.
  /// The selected pivot columns of (mu nu | piv) are accumulated as
  /// we go; after selection we build the small (Naux, Naux) metric
  /// M[p, q] = (piv_p | piv_q) via libint, orthogonalize via
  /// PartialCholeskyOrth(M, fit_cholesky_thr) to drop linearly
  /// dependent pivots, and construct the final L vectors as
  /// L_J(mu nu) = (mu nu | piv) * M^{-1/2}. Output L vectors are
  /// stored in this object's standard layout (B / prodidx /
  /// invmap / prodmap / odiagidx) so all downstream consumers --
  /// calcJ / calcK / B_matrix / naf_transform / save / load -- work
  /// unchanged.
  ///
  /// Cf. F. Folkestad, E. F. Kjonstad and H. Koch,
  /// J. Chem. Phys. 150, 194112 (2019).
  size_t fill_two_step(const BasisSet & basis, double cholesky_tol, double shell_reuse_thr, double shell_screen_tol, double fit_cholesky_thr, bool verbose);

  /// Get the pivot vector
  arma::uvec get_pivot() const;
  /// Get the pivot shellpairs
  std::set< std::pair<size_t, size_t> > get_pivot_shellpairs() const;

  /// Perform natural auxiliary function transform [M. Kallay, JCP 141, 244113 (2014)]
  size_t naf_transform(double thr, bool verbose);

  /// Get amount of vectors
  size_t get_Naux() const;
  /// Get basis set size
  size_t get_Nbf() const;
  /// Get basis set size
  size_t get_Npairs() const;

  /// Get the matrix
  arma::mat get() const;
  /// Get basis function numbers
  arma::umat get_invmap() const;

  /// Form Coulomb matrix
  arma::mat calcJ(const arma::mat & P) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::vec & C) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::mat & C, const std::vector<double> & occs) const;
  /// Form exchange matrix
  arma::mat calcK(const arma::mat & C, const arma::vec & occs) const;

  /// Form exchange matrix
  arma::cx_mat calcK(const arma::cx_vec & C) const;
  /// Form exchange matrix
  arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occs) const;
  /// Form exchange matrix
  arma::cx_mat calcK(const arma::cx_mat & C, const arma::vec & occs) const;

  /// Get full B matrix
  void B_matrix(arma::mat & B) const;
  /// Get partial B matrix
  void B_matrix(arma::mat & B, arma::uword first, arma::uword last) const;

  /// Get transformed B matrix
  arma::mat B_transform(const arma::mat & Cl, const arma::mat & Cr, bool verbose=false) const;

  /// Compute Coulomb forces via two-step CD: f = 0.5 c (dM/dR) c -
  /// sum_munu P (d(mu nu|piv)/dR) c, where c is the expansion of
  /// the density on the pivot orbital pairs. Only valid when the
  /// object was filled via fill_two_step (i.e. is_force_capable());
  /// throws otherwise.
  arma::vec forceJ(const BasisSet & basis, const arma::mat & P) const;

  /// True iff forceJ is available on this object. One-step CD has
  /// no pivot metric and so cannot evaluate the gradient; two-step
  /// CD stashes the metric in fill_two_step. Callers should query
  /// this before dispatching to forceJ and fall back to the
  /// four-index path otherwise.
  bool is_force_capable() const { return two_step_; }
};

// ===========================================================================
// ERIfit namespace -- merged in from src/erifit.{h,cpp}. The CD-based aux
// basis construction in basislibrary.cpp uses the same Cholesky / fitting
// machinery as ERIchol, so the two are folded into one translation unit
// here. (Pure cosmetic merge; the API is unchanged.)
// ===========================================================================
namespace ERIfit {
  /// Basis function pair
  struct bf_pair_t {
    /// Index
    size_t idx;
    /// lh function
    size_t i;
    /// shell index
    size_t is;
    /// rh function
    size_t j;
    /// shell index
    size_t js;
  };

  /// Comparison operator
  bool operator<(const bf_pair_t & lhs, const bf_pair_t & rhs);

  /// Compute the exact repulsion integrals
  void compute_ERIs(const BasisSet & basis, arma::mat & eris);
  /// Compute the exact repulsion integrals
  void compute_ERIs(const ElementBasisSet & orbel, arma::mat & eris);
  /// Compute the exact diagonal repulsion integrals
  void compute_diag_ERIs(const ElementBasisSet & orbel, arma::mat & eris);

  /// Find unique exponent pairs
  void unique_exponent_pairs(const ElementBasisSet & orbel, int am1, int am2, std::vector< std::vector<shellpair_t> > & pairs, std::vector<double> & exps);
  /// Compute the T matrix needed for Cholesky decomposition
  void compute_cholesky_T(const ElementBasisSet & orbel, int am1, int am2, arma::mat & eris, arma::vec & exps);

  /// Compute fitting integrals
  void compute_fitint(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, arma::mat & fitint);

  /// Compute the fitted repulsion integrals using the supplied fitting integrals
  void compute_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri);
  /// Compute the diagonal fitted repulsion integrals using the supplied fitting integrals
  void compute_diag_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri);

  /// Compute the transformation matrix to orthonormal orbitals
  void orthonormal_ERI_trans(const ElementBasisSet & orbel, double linthr, arma::mat & trans);
}

#endif
