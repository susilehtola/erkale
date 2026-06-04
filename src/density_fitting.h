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
 * \class DensityFit
 *
 * \brief Density fitting / RI routines
 *
 * This class contains density fitting and resolution of the identity
 * routines used for the approximate calculation of the Coulomb and
 * exchange operators J and K.
 *
 * The RI-JK implementation is based on the procedure described in
 *
 * F. Weigend, "A fully direct RI-HF algorithm: Implementation,
 * optimised auxiliary basis sets, demonstration of accuracy and
 * efficiency", Phys. Chem. Chem. Phys. 4, 4285 (2002).
 *
 *
 * If only RI-J is necessary, then the procedure described in
 *
 * K. Eichkorn, O. Treutler, H. Öhm, M. Häser and R. Ahlrichs,
 * "Auxiliary basis sets to approximate Coulomb potentials",
 * Chem. Phys. Lett. 240 (1995), 283-290.
 *
 * is used.
 *
 * \author Susi Lehtola
 * \date 2012/08/22 23:53
 */




#ifndef ERKALE_DENSITYFIT
#define ERKALE_DENSITYFIT

#include "global.h"
#include "b_tensor.h"
#include "basis.h"
#include "eriworker.h"

#include <memory>
#include <set>
#include <utility>

/// Density fitting routines.
///
/// DensityFit holds the cached three-index integrals through a
/// shared_ptr so the surrounding objects (Edmiston etc.) that copy
/// a DensityFit by value end up sharing the heavy block storage
/// rather than duplicating it. The ptr-by-value fields and the
/// shared aux-metric matrices make a copy of *this O(handful of
/// scalars and shared_ptr atomics).
class DensityFit {
  /// Amount of orbital basis functions
  size_t Nbf;
  /// Amount of auxiliary basis functions
  size_t Naux;
  /// Direct calculation? (Compute three-center integrals on-the-fly)
  bool direct;

  /// Range separation constants
  double omega, alpha, beta;

  /// Amount of nuclei
  size_t Nnuc;
  /// Maximum angular momentum
  int maxam;
  /// Maximum contractions
  int maxcontr;

  /// Orbital shells
  std::vector<GaussianShell> orbshells;
  int maxorbam;
  size_t maxorbcontr;
  /// Density fitting shells
  std::vector<GaussianShell> auxshells;
  int maxauxam;
  size_t maxauxcontr;
  /// Dummy shell
  GaussianShell dummy;

  /// Index of dummy function
  size_t dummyind;

  /// List of unique orbital shell pairs
  std::vector<eripair_t> orbpairs;
  /// Three-index (alpha | mu nu) block source, indexed per orbital
  /// shellpair. In non-direct mode this is a CachedBlocks with the
  /// integrals precomputed and stored; in direct mode this is a
  /// DirectDFBlocks that computes blocks on demand via libint.
  /// Either way, the J/K kernels consume blocks via the same
  /// blocks->get_block(ip) interface. shared_ptr so DensityFit
  /// copies (e.g. Edmiston) share the storage / state.
  std::shared_ptr<BTensorBlocks> blocks;

  /// \f$ ( \alpha | \beta) \f$
  arma::mat ab;
  /// \f$ ( \alpha | \beta)^-1 \f$
  arma::mat ab_inv;
  /// \f$ ( \alpha | \beta)^-1/2 \f$
  arma::mat ab_invh;

  /// True when this object was filled via fill_cholesky. CD and DF
  /// share the same J/K machinery; the only thing this flag affects
  /// is which gradient path is available (forceJ for DF aux shells,
  /// forceJ_cholesky for pivot-orbital-pair "aux") and the pivot
  /// machinery exposed via get_pivot_shellpairs().
  bool cholesky_mode_;

  /// (Nbf x Nbf) lookup: (mu, nu) -> pivot rank in 0..Naux-1, or
  /// cd_pivot_sentinel_ for non-pivot pairs. Built in fill_cholesky
  /// and consumed by forceJ_cholesky for the dM/dR + d(mu nu | piv)/dR
  /// contractions.
  arma::umat cd_pivot_index_;
  /// Sentinel value used in cd_pivot_index_ (== Naux).
  arma::uword cd_pivot_sentinel_;
  /// Pivot shellpairs in lexicographic order; enumerated to drive
  /// the dM/dR sweep in forceJ_cholesky without re-sorting per call.
  std::vector<std::pair<size_t, size_t>> cd_pivot_shellpairs_vec_;

  /// Pivot shellpairs (set form) populated by fill_cholesky. Exposed
  /// via get_pivot_shellpairs() for basistool / basislibrary atom-CD
  /// aux basis construction.
  std::set<std::pair<size_t, size_t>> pivot_shellpairs_;

  /// Form screening matrix
  void form_screening();
  /// Two-center metric-derivative force contribution. Iterates the
  /// aux-shellpair (DF) or pivot-shellpair (CD) pair index space,
  /// computes the corresponding dERIWorker derivative integrals,
  /// and contracts each with M(ia, ib). The signed result is
  /// added to f.
  ///
  /// sign = +1 reproduces forceJ's "f += (1/2) c^T (dM/dR) c"
  /// (M_lookup = c(a)*c(b)); sign = -1 gives forceK's
  /// "f -= (1/2) G : dM/dR" (M_lookup = G(a, b)). The 1/2 enters
  /// via the symmetry factor that already lives in this loop.
  ///
  /// Definition in the .cpp -- template so the lookup lambda
  /// inlines and we avoid materialising the c-outer-product for
  /// the rank-1 forceJ case.
  template<typename M_lookup>
  void accumulate_2c_metric_force(arma::vec & f, M_lookup && M, double sign) const;

  /// Three-center derivative force contribution, DF aux dispatch.
  /// Iterates orbital shellpairs through DirectDFPerturbedBlocks;
  /// build_q(ip) returns the per-shellpair contraction matrix
  /// Q_ip of shape (Naux x Nmu*Nnu) with column index = inu*Nmu + imu
  /// (matching the value-side sub_block layout). For each
  /// perturbation block delivered by for_each_pert, the
  /// contribution to f at (pert.atom, pert.xyz) is
  ///   sign * <sub_block, Q_ip.rows(a0, a0+Na_shell-1)>_F.
  template<typename BuildQ>
  void accumulate_3c_force_DF(arma::vec & f, double sign, BuildQ && build_q) const;

  /// Three-center derivative force contribution, CD pivot dispatch.
  /// Iterates orbital shellpairs (outer) x pivot shellpairs
  /// (inner), computes 4-shell dERIWorker derivatives, and per
  /// component contracts the integrals with build_q(ip)(qidx, ii*Nj+jj)
  /// for each (ii, jj, kk, ll) on the (orb_shellpair, pivot_shellpair)
  /// quartet. build_q(ip) returns a per-orbital-shellpair matrix
  /// of shape (Naux x Ni*Nj) with column index = ii*Nj + jj.
  template<typename BuildQ>
  void accumulate_3c_force_CD(const BasisSet & basis, arma::vec & f, double sign, BuildQ && build_q) const;

  /// Two-step CD pivot selection (phases A-C: diagonal, pair
  /// enumeration, pivoted selection). Populates the by-ref output
  /// parameters with the pivoting machinery; b_raw_out is filled
  /// with the (mu nu | piv) column for each selected pivot when
  /// non-null, otherwise the columns are computed and discarded.
  size_t select_two_step_pivots(const BasisSet & basis,
                                double cholesky_tol,
                                double shell_reuse_thr,
                                double shell_screen_tol,
                                bool verbose,
                                arma::uvec & pi,
                                arma::umat & invmap,
                                arma::umat & prodmap,
                                arma::uvec & prodidx,
                                arma::uvec & odiagidx,
                                std::set<std::pair<size_t, size_t>> & piv_shellpairs,
                                arma::mat * b_raw_out) const;
  /// Compute shell in (a|uv) matrix
  arma::mat compute_a_munu(ERIWorker * eri, size_t ip, double * memptr = nullptr) const;
  /// Project P_munu onto the aux basis through one shellpair block:
  /// gamma_a += (a|mu nu) P_munu, restricted to the (mu, nu) range
  /// described by the block at index ip.
  void project_density_to_aux(const arma::mat & P, size_t ip, const arma::mat & amunu, arma::vec & gamma) const;
  /// Contract the aux-space expansion gamma back to J through one
  /// shellpair block: J_munu += (a|mu nu) gamma_a.
  void contract_aux_to_J(const arma::vec & gamma, size_t ip, const arma::mat & amunu, arma::mat & J) const;
  /// Build K by looping orbital shellpairs, half-transforming each
  /// (a|mu nu) block against the occupied MOs, and accumulating
  /// occ * aui^H aui (arma::trans is conjugate-transpose for complex
  /// orbitals, plain transpose for real). Backs onto
  /// BTensorBlocks::get_block, so in direct mode the blocks
  /// recompute on the fly per call.
  ///
  /// Templated on the orbital scalar type so the same code services
  /// the real (HF/DFT) and complex (PZ-SIC, complex-orbital
  /// guesses) paths. Definition lives in the .cpp; specializations
  /// for T = double and T = std::complex<double> are instantiated
  /// implicitly via the calcK call sites.
  template<typename T>
  void accumulate_K_from_blocks(const arma::Mat<T> & C, const arma::vec & occs, arma::Mat<T> & K) const;

 public:
  /// Constructor
  DensityFit();
  /// Destructor
  ~DensityFit();

  /// Set range separation constants
  void set_range_separation(double w, double a, double b);
  void set_range_separation(const RangeSeparation & rs) { set_range_separation(rs.omega, rs.alpha, rs.beta); }
  /// Get range separation constants
  void get_range_separation(double & w, double & a, double & b) const;
  RangeSeparation get_range_separation() const { RangeSeparation rs; get_range_separation(rs.omega, rs.alpha, rs.beta); return rs; }

  /**
   * Compute integrals, use given linear dependency threshold. The HF
   * flag here controls formation of (a|b)^{-1/2} and (a|b)^{-1}; the
   * HF routine should be more tolerant of linear dependencies in the basis.
   * Returns amount of significant orbital shell pairs.
   */
  size_t fill(const BasisSet & orbbas, const BasisSet & auxbas, bool direct, double erithr, double linthr, double cholthr);

  /// Fill the B tensor via two-step pivoted Cholesky decomposition
  /// (Folkestad/Kjonstad/Koch JCP 150, 194112 (2019)). Reshapes the
  /// orthonormal L vectors into the block-shellpair layout the DF
  /// J/K kernels consume; metric is identity (L vectors are
  /// orthonormal by construction) so the same kernels handle CD and
  /// DF transparently. The pivot metric is stashed so
  /// forceJ_cholesky has the algebraic gradient available. Range
  /// separation is honored from prior set_range_separation().
  /// Returns amount of significant orbital shell pairs.
  ///
  /// One-step CD (full pivoted CD on the molecular tensor) was
  /// retired here -- TwoStep is mathematically equivalent at the
  /// same threshold but cheaper to construct.
  size_t fill_cholesky(const BasisSet & basis,
                       bool direct,
                       double cholesky_tol,
                       double shell_reuse_thr,
                       double shell_screen_tol,
                       double fit_cholesky_thr,
                       bool verbose);

  /// True iff this object was filled via fill_cholesky (i.e. B holds
  /// orthonormal CD vectors and there is no genuine aux basis).
  bool is_cholesky() const { return cholesky_mode_; }

  /// Algebraic two-step CD gradient of the Coulomb energy. Requires
  /// fill_cholesky_twostep to have populated the pivot metric;
  /// throws otherwise. Returns f of size 3*Nnuc.
  arma::vec forceJ_cholesky(const BasisSet & basis, const arma::mat & P) const;

  /// Algebraic exchange gradient, closed-shell, scaled by kfrac.
  /// Works on both DF (aux Gaussian basis) and CD (pivot orbital
  /// products) modes; cholesky_mode_ selects the integral dispatch
  /// internally. C is (Nbf x Norb), occs has the same length as the
  /// columns of C (zero entries are filtered). Returns f of size
  /// 3*Nnuc.
  arma::vec forceK(const BasisSet & basis, const arma::mat & C, const std::vector<double> & occs, double kfrac) const;

  /// Pivot shellpairs that the CD picked. Populated by both
  /// fill_cholesky and fill_cholesky_twostep. Returns an empty set
  /// in non-cholesky mode.
  std::set<std::pair<size_t, size_t>> get_pivot_shellpairs() const { return pivot_shellpairs_; }

  /// Two-step CD pivot selection without building the metric or
  /// L vectors. Returns the pivot shellpair set selected by phases
  /// A-C of the two-step algorithm at the given threshold. Use when
  /// downstream only needs the pivot list (e.g. atom-CD aux basis
  /// construction in basislibrary.cpp). Range separation is honored
  /// from prior set_range_separation(). Static because it doesn't
  /// touch *this -- it's a pure utility on the basis.
  static std::set<std::pair<size_t, size_t>> find_cholesky_pivots(const BasisSet & basis,
                                                                   double cholesky_tol,
                                                                   double shell_reuse_thr,
                                                                   double shell_screen_tol,
                                                                   bool verbose);

  /// Compute estimate of necessary memory
  size_t memory_estimate(const BasisSet & orbbas, const BasisSet & auxbas, double erithr, bool direct) const;

  /// Compute expansion coefficients c
  arma::vec compute_expansion(const arma::mat & P) const;
  /// Compute expansion coefficients c
  std::vector<arma::vec> compute_expansion(const std::vector<arma::mat> & P) const;

  /// Get Coulomb matrix from P
  arma::mat calcJ(const arma::mat & P) const;
  /// Get Coulomb matrix from P
  std::vector<arma::mat> calcJ(const std::vector<arma::mat> & P) const;
  /// Digest J matrix from computed expansion
  arma::mat calcJ_vector(const arma::vec & gamma) const;

  /// Calculate force from P
  arma::vec forceJ(const arma::mat & P);

  /// Get exchange matrix from orbitals with occupation numbers occs
  arma::mat calcK(const arma::mat & C, const std::vector<double> & occs) const;
  /// Get exchange matrix from orbitals with occupation numbers occs
  arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occs) const;

  /// Get the number of orbital functions
  size_t get_Norb() const;
  /// Get the number of auxiliary functions
  size_t get_Naux() const;
  /// Get the number of linearly independent auxiliary functions
  size_t get_Naux_indep() const;
  /// Get ab_inv
  arma::mat get_ab() const;
  /// Get ab_inv
  arma::mat get_ab_inv() const;
  /// Get ab_invh
  arma::mat get_ab_invh() const;

  /// Get 3-center integrals (must have HF enabled)
  void three_center_integrals(arma::mat & B) const;
  /// Get B matrix (must have HF enabled)
  void B_matrix(arma::mat & B) const;
  /// Two-sided MO transform of the B tensor: returns Br with
  /// Br(P, r*Nl + l) = sum_{u,v} Cl(u,l) Cr(v,r) B_dense(u*Nbf+v, P).
  /// Used by post-HF consumers (moints.cpp); works equally for DF
  /// and CD-mode storage because it builds on B_matrix.
  arma::mat B_transform(const arma::mat & Cl, const arma::mat & Cr, bool verbose=false) const;

  /// Compute error in (AB|AB) type integrals
  double fitting_error() const;
};


#endif
