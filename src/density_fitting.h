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
#include "cintenv.h"
#include "eriworker.h"

#include <memory>
#include <set>
#include <utility>

/// Density fitting routines.
///
/// DensityFit holds the cached three-index integrals through a
/// shared_ptr, so objects that copy a DensityFit by value (e.g.
/// Edmiston) share the heavy block storage rather than duplicating
/// it. The metric matrices (X, and ab in DF mode) are still
/// deep-copied, so a copy is O(Naux^2), not free -- cheap relative to
/// the block tensor, but not nothing.
///
/// Density fitting and two-step Cholesky store the same object, the
/// metric-baked B = X^T (a | mu nu) with X^T (a|b) X = 1, where a runs
/// over the auxiliary functions (DF) or the pivot orbital products (CD).
/// The J/K builds are thereby identical for both, and only the fill and
/// the derivative integrals of the force kernels differ. (Direct DF keeps
/// raw blocks and applies X^T to the contracted vectors instead.)
class DensityFit {
  /// Amount of orbital basis functions
  size_t Nbf_;
  /// Amount of auxiliary basis functions
  size_t Naux_;
  /// Direct calculation? (Compute three-center integrals on-the-fly)
  bool direct_;

  /// Range separation constants
  double omega_, alpha_, beta_;

  /// Amount of nuclei
  size_t Nnuc_;
  /// Maximum angular momentum
  int maxam_;
  /// Maximum contractions
  int maxcontr_;

  /// Orbital shells
  std::vector<GaussianShell> orbshells_;
  int maxorbam_;
  size_t maxorbcontr_;
  /// Density fitting shells
  std::vector<GaussianShell> auxshells_;
  int maxauxam_;
  size_t maxauxcontr_;
  /// libcint description of the orbital basis, followed by the
  /// auxiliary basis (in two-step CD there is no separate auxiliary
  /// basis, and the environment holds the orbital shells alone)
  CintEnv cenv_;

  /// List of unique orbital shell pairs
  std::vector<eripair_t> orbpairs_;
  /// Three-index (alpha | mu nu) block source, indexed per orbital
  /// shellpair. In non-direct mode this is a CachedBlocks with the
  /// integrals precomputed and stored; in direct mode this is a
  /// DirectDFBlocks that computes blocks on demand.
  /// Either way, the J/K kernels consume blocks via the same
  /// blocks->get_block(ip) interface. shared_ptr so DensityFit
  /// copies (e.g. Edmiston) share the storage / state.
  std::shared_ptr<BTensorBlocks> blocks_;

  /// \f$ ( \alpha | \beta) \f$ in DF mode, for the (a|b) accessor; empty in
  /// CD mode, where the metric is not retained
  arma::mat ab_;

  /// True when this object was filled via fill_cholesky. DF and CD store
  /// the same kind of B blocks and share the J/K kernels; this flag only
  /// selects the derivative integrals in the force kernels (Gaussian aux
  /// shells vs pivot orbital products).
  bool cholesky_mode_;

  /// Half-inverse X of the fitting metric M = (a|b) (DF) or (piv|piv) (CD),
  /// X^T M X = 1, of shape (Nfit x Naux_) where Nfit is the number of aux
  /// basis functions or pivots. The stored B blocks are
  /// B = X^T (a|mu nu), so the J/K kernels need no metric. Applying X once
  /// to the fixed three-index integrals instead of to the density in every
  /// SCF iteration keeps the SCF energy free of the roundoff the
  /// ill-conditioned metric would otherwise amplify. Direct DF, which
  /// recomputes the blocks, instead applies X^T to the contracted aux
  /// vectors (see BTensorBlocks::metric()). The force kernels map the
  /// B-space expansion d back to the aux space as c = X d.
  arma::mat X_;

  /// (Nbf x Nbf) lookup: (mu, nu) -> pivot rank in 0..Nfit-1, or
  /// cd_pivot_sentinel for non-pivot pairs. Built in fill_cholesky
  /// and consumed by the CD force kernels for the dM/dR +
  /// d(mu nu | piv)/dR contractions.
  arma::umat cd_pivot_index_;
  /// Sentinel value used in cd_pivot_index (== Naux).
  arma::uword cd_pivot_sentinel_;
  /// Pivot shellpairs in lexicographic order; enumerated to drive
  /// the CD dM/dR sweep without re-sorting per call.
  std::vector<std::pair<size_t, size_t>> cd_pivot_shellpairs_vec_;

  /// Pivot shellpairs (set form) populated by fill_cholesky via
  /// select_two_step_pivots; copied into cd_pivot_shellpairs_vec to
  /// drive the metric build and the force sweeps.
  std::set<std::pair<size_t, size_t>> pivot_shellpairs_;

  /// True when the metric half-inverse X was supplied by another fit
  /// (fill with an external X, or fill_cholesky_shared). This object's own
  /// metric then differs from the one X orthonormalizes, and for a shared
  /// pivot set cd_pivot_index is indexed over the pivot basis, not the
  /// orbital basis, so the gradient kernels must refuse to run.
  bool foreign_metric_ = false;

  /// Form screening matrix
  void form_screening();
  /// Throw std::logic_error unless P has the Nbf x Nbf density-matrix
  /// shape. Shared by the J / expansion entry points.
  void check_density_dims(const arma::mat & P) const;
  /// Set Nbf, Nnuc, direct, orbshells, maxorbam, maxorbcontr
  /// from the orbital basis. Shared by fill() and fill_cholesky();
  /// aux-side state (Naux, auxshells, maxauxam/contr, maxam/contr)
  /// and the cholesky_mode-specific bookkeeping are set by the
  /// caller after this returns.
  void init_orbital_state(const BasisSet & orbbas, bool dir);
  /// Set up the orbital and auxiliary state of a DF fill
  void init_df(const BasisSet & orbbas, const BasisSet & auxbas, bool dir, double erithr);
  /// Build the DF B blocks from X_; returns the number of significant
  /// orbital shell pairs
  size_t fill_df_blocks();
  /// Lay out the per-shellpair (shell pair, first-function pair, size
  /// pair) descriptor triple consumed by every BTensorBlocks
  /// constructor. The same descriptor is also passed to
  /// DirectDFPerturbedBlocks, so the helper is on the class to keep
  /// the layout in one place.
  void build_shellpair_descriptor(
      std::vector<std::pair<size_t, size_t>> & sp_pairs,
      std::vector<std::pair<size_t, size_t>> & sp_firsts,
      std::vector<std::pair<size_t, size_t>> & sp_sizes) const;
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

  /// Three-center derivative force contribution. build_q(is, js)
  /// returns the contraction matrix for the orbital shellpair (is, js),
  /// of shape (Nfit x Ni*Nj) with column index jj*Ni + ii (the B block
  /// layout), in the aux (DF) or pivot (CD) basis. DF iterates the
  /// shellpairs through DirectDFPerturbedBlocks and contracts each
  /// perturbation's (a | mu nu) derivative block with Q; CD iterates
  /// (orbital shellpair, pivot shellpair) quartets of 4-center
  /// derivatives. The result, scaled by sign, is added to f.
  template<typename BuildQ>
  void accumulate_3c_force(const BasisSet & basis, arma::vec & f, double sign, BuildQ && build_q) const;
  /// Throw unless the analytic gradient is available (it is not for a
  /// shared-pivot CD decomposition)
  void check_force_available() const;
  /// Number of rows of the blocks: Naux_ for metric-baked blocks, or
  /// the number of aux functions for raw direct DF blocks
  size_t block_rows() const;
  /// Map a vector over the blocks' aux index to the orthonormal fitting
  /// basis (X^T gamma for raw blocks, identity for baked ones)
  arma::vec apply_metric_t(const arma::vec & gamma) const;
  /// Map an expansion in the orthonormal fitting basis to the blocks'
  /// aux index (X d for raw blocks, identity for baked ones)
  arma::vec apply_metric(const arma::vec & d) const;
  /// Scatter the block of shellpair ip, of shape (Ncol x Nmu*Nnu), into
  /// the dense (Nbf*Nbf x Ncol) matrix ints
  void scatter_block(arma::mat & ints, size_t ip, const arma::mat & block) const;

  /// Compute the raw (a|uv) integrals of a shellpair (DF only), shape
  /// (Nfit x Nmu*Nnu)
  arma::mat compute_a_munu(ERIWorker * eri, size_t ip) const;
  /// Project P_munu onto the fitting basis through one shellpair block:
  /// gamma_Q += B_{Q,mu nu} P_munu, restricted to the (mu, nu) range
  /// described by the block at index ip.
  void project_density_to_aux(const arma::mat & P, size_t ip, const arma::mat & amunu, arma::vec & gamma) const;
  /// Contract the expansion gamma back to J through one shellpair
  /// block: J_munu += B_{Q,mu nu} gamma_Q.
  void contract_aux_to_J(const arma::vec & gamma, size_t ip, const arma::mat & amunu, arma::mat & J) const;
  /// Filter the input orbital matrix Corig (Nbf x Norb) and
  /// matching occupations occo down to the columns with non-zero
  /// occupation. Returns C_out (Nbf x Nmo) and occs_out (Nmo);
  /// shared by calcK / forceK so the wrappers stay trivial.
  template<typename T>
  void filter_occupied(const arma::Mat<T> & Corig, const std::vector<double> & occo,
                       arma::Mat<T> & C_out, arma::vec & occs_out) const;
  /// Templated calcK implementation; the real / complex public
  /// overloads forward here.
  template<typename T>
  arma::Mat<T> calcK_impl(const arma::Mat<T> & Corig, const std::vector<double> & occo) const;

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

  /// Half-transform a single occupied orbital io: fill aui (Naux x Nbf)
  /// with B_{Q,mu i} = sum_nu B_{Q,mu nu} C(nu,io), looping orbital
  /// shellpairs. The three scratch buffers are caller-owned per-thread
  /// workspace. Shared by accumulate_K_from_blocks (conventional RI-K),
  /// accumulate_KC_from_blocks (occ-RI-K) and forceK.
  template<typename T>
  void halftransform_orbital(const arma::Mat<T> & C, size_t io, arma::Mat<T> & aui,
                             arma::Mat<T> & ui_scratch, arma::Mat<T> & vi_scratch,
                             arma::mat & anumu_scratch) const;

  /// occ-RI-K assembly: accumulate only the occupied columns of the
  /// exchange matrix, KC(mu,k) += sum_i occs[i] sum_a B^a_{mu,i} B^a_{k,i}
  /// = (K C)_{mu,k}, where C holds the (already occupied-filtered)
  /// orbitals. Costs O(Nocc^2 Nbf Naux) instead of conventional RI-K's
  /// O(Nocc Nbf^2 Naux), at the price of only producing K on the
  /// occupied space (see calcK_occ_impl for the full-matrix recovery).
  template<typename T>
  void accumulate_KC_from_blocks(const arma::Mat<T> & C, const arma::vec & occs, arma::Mat<T> & KC) const;

  /// Templated occ-RI-K implementation; the real / complex public
  /// calcK_occ overloads forward here. Builds the occupied columns
  /// KC = K C_o and recovers a full symmetric/Hermitian AO exchange
  /// matrix that reproduces the exact (RI) occupied-occupied and
  /// occupied-virtual blocks; see calcK_occ for the algorithm.
  template<typename T>
  arma::Mat<T> calcK_occ_impl(const arma::Mat<T> & Corig, const std::vector<double> & occo, const arma::mat & S) const;

 public:
  /// Constructor
  DensityFit();
  /// Destructor
  ~DensityFit();

  /// Set range separation constants
  void set_range_separation(double w, double a, double b);
  void set_range_separation(const RangeSeparation & rs) { set_range_separation(rs.omega, rs.alpha, rs.beta); }
  /// Get range separation constants
  void range_separation(double & w, double & a, double & b) const;
  RangeSeparation range_separation() const { RangeSeparation rs; range_separation(rs.omega, rs.alpha, rs.beta); return rs; }

  /**
   * Compute the density-fitting integrals against the auxiliary basis
   * auxbas. linthr / cholthr are the linear-dependence and pivoted-
   * Cholesky thresholds for orthogonalising the (a|b) metric; erithr
   * screens the orbital shell pairs. Returns the number of
   * significant orbital shell pairs.
   */
  size_t fill(const BasisSet & orbbas, const BasisSet & auxbas, bool direct, double erithr, double linthr, double cholthr);

  /**
   * Compute the density-fitting integrals against the auxiliary basis
   * auxbas with the metric half-inverse X of another fit in the same
   * auxiliary basis (see metric_half_inverse), instead of this object's
   * own. Fits sharing X share the orthonormal fitting basis, so
   * compute_expansion of one can be passed to calcJ_vector of the other,
   * e.g. for a Coulomb interaction between two species. The analytic
   * gradient is not available on the result. Returns the number of
   * significant orbital shell pairs.
   */
  size_t fill(const BasisSet & orbbas, const BasisSet & auxbas, bool direct, double erithr, const arma::mat & X);

  /// Fill the B tensor via two-step pivoted Cholesky decomposition
  /// (Folkestad/Kjonstad/Koch JCP 150, 194112 (2019)). The selected
  /// pivot orbital products act as an auxiliary basis; the (piv|piv)
  /// metric is normalised, canonical-orthogonalised, and baked into
  /// the stored L blocks exactly as in DF mode, so the J/K kernels
  /// handle CD and DF transparently. The metric half-inverse X is kept
  /// for the force kernels. Range separation is
  /// honored from prior set_range_separation(). Returns the number
  /// of significant orbital shell pairs.
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

  /**
   * Fill the B tensor from an *externally supplied* pivot basis and metric
   * orthogonaliser, rather than selecting pivots from this object's own
   * orbital basis.
   *
   * This is what makes a multicomponent (NEO) decomposition possible: one
   * pivot set spanning the union of the electronic and protonic pair spaces,
   * with one metric M = (piv|piv) and one orthogonaliser X = M^{-1/2}, is
   * handed to a DensityFit per species. Each then holds
   * L = X^T (piv | mu nu) over its own orbital basis, so the two share a
   * common vector index P and the cross-species integrals come out as
   * (mu nu | a b) = sum_P B_e[P,mu,nu] B_p[P,a,b] -- exact to the Cholesky
   * threshold, rather than exact only when both species happen to be fitted
   * in the same auxiliary basis.
   *
   * piv_shells is the concatenated pivot basis with globally unique
   * first_ind; piv_index is indexed over it. piv_max_am / piv_max_contr must
   * cover it. The CD force kernels are unavailable on the result (the pivot
   * and orbital bases no longer coincide) and throw if called.
   *
   * Returns the number of significant orbital shell pairs.
   */
  size_t fill_cholesky_shared(const BasisSet & orbbas,
                              const std::vector<GaussianShell> & piv_shells,
                              const std::vector<std::pair<size_t, size_t>> & piv_shellpairs,
                              const arma::umat & piv_index,
                              arma::uword piv_sentinel,
                              const arma::mat & X,
                              bool direct,
                              double shell_screen_tol,
                              int piv_max_am, int piv_max_contr,
                              bool verbose);

  /// Two-step CD pivot selection (phases A-C: diagonal, pair
  /// enumeration, pivoted selection). Pivots-only: populates the
  /// by-ref outputs (pivot list, product map, pivot shellpairs); the
  /// (mu nu | piv) integrals are rebuilt per block by the cached/direct
  /// block builders, not returned here. Honors the instance's range
  /// separation (set_range_separation). Public so a multicomponent
  /// decomposition can select pivots per species and union them before
  /// building one shared metric (see contrib/neo_cholesky.h).
  size_t select_two_step_pivots(const BasisSet & basis,
                                double cholesky_tol,
                                double shell_reuse_thr,
                                double shell_screen_tol,
                                bool verbose,
                                arma::uvec & pi,
                                arma::umat & invmap,
                                std::set<std::pair<size_t, size_t>> & piv_shellpairs) const;

  /// True iff this object was filled via fill_cholesky (i.e. the
  /// blocks hold CD-derived L vectors, not a genuine aux basis).
  bool is_cholesky() const { return cholesky_mode_; }

  /// Algebraic gradient of the fitted Coulomb energy (1/2) tr(P J) of
  /// the density P, in DF and CD modes. basis is the orbital basis.
  /// Returns f of size 3*Nnuc.
  arma::vec forceJ(const BasisSet & basis, const arma::mat & P) const;

  /// Algebraic exchange gradient, closed-shell, scaled by kfrac.
  /// Works on both DF (aux Gaussian basis) and CD (pivot orbital
  /// products) modes; cholesky_mode selects the integral dispatch
  /// internally. C is (Nbf x Norb), occs has the same length as the
  /// columns of C (zero entries are filtered). Returns f of size
  /// 3*Nnuc.
  arma::vec forceK(const BasisSet & basis, const arma::mat & C, const std::vector<double> & occs, double kfrac) const;

  /// Two-step CD pivot selection without building the metric or
  /// L vectors. Returns the pivot shellpair set selected by phases
  /// A-C of the two-step algorithm at the given threshold. Use when
  /// downstream only needs the pivot list (e.g. atom-CD aux basis
  /// construction in basislibrary.cpp). Range separation is honored
  /// from this object's set_range_separation() (default: plain
  /// Coulomb).
  std::set<std::pair<size_t, size_t>> find_cholesky_pivots(const BasisSet & basis,
                                                           double cholesky_tol,
                                                           double shell_reuse_thr,
                                                           double shell_screen_tol,
                                                           bool verbose) const;

  /// Compute estimate of necessary memory
  size_t memory_estimate(const BasisSet & orbbas, const BasisSet & auxbas, double erithr, bool direct) const;

  /// Compute the expansion d_Q = sum_munu B_{Q,munu} P_munu of the density
  /// in the orthonormal fitting basis; the aux-basis coefficients are
  /// X d. Objects that share X (fill with an external X, or
  /// fill_cholesky_shared) can combine compute_expansion and calcJ_vector
  /// across orbital bases.
  arma::vec compute_expansion(const arma::mat & P) const;
  /// Compute the expansions of several densities
  std::vector<arma::vec> compute_expansion(const std::vector<arma::mat> & P) const;

  /// Get Coulomb matrix from P
  arma::mat calcJ(const arma::mat & P) const;
  /// Get Coulomb matrix from P
  std::vector<arma::mat> calcJ(const std::vector<arma::mat> & P) const;
  /// Digest J matrix from computed expansion
  arma::mat calcJ_vector(const arma::vec & gamma) const;

  /// Get exchange matrix from orbitals with occupation numbers occs
  arma::mat calcK(const arma::mat & C, const std::vector<double> & occs) const;
  /// Get exchange matrix from orbitals with occupation numbers occs
  arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occs) const;

  /// Exchange matrix via the occ-RI-K algorithm (Manzer, Horn,
  /// Mardirossian, Head-Gordon, J. Chem. Phys. 143, 024113 (2015)).
  ///
  /// Only the occupied columns KC = K C_o are assembled (cost
  /// O(Nocc^2 Nbf Naux), a factor ~Nbf/Nocc below conventional RI-K);
  /// the full AO matrix is then recovered by the symmetric/Hermitian
  /// reconstruction
  ///   K = KC (S C_o)^H + (S C_o) KC^H - (S C_o)(C_o^H K C_o)(S C_o)^H,
  /// which -- using C_o^H S C_o = I -- reproduces the occupied-occupied
  /// and occupied-virtual blocks of the RI exchange exactly, leaving
  /// only the virtual-virtual block approximate. The SCF energy,
  /// density and orbital gradient are therefore unchanged from
  /// conventional RI-K; virtual orbital energies are not. The occupied
  /// orbitals supplied in C must be S-orthonormal (as SCF eigenvectors
  /// are). S is the orbital-basis overlap.
  arma::mat calcK_occ(const arma::mat & C, const std::vector<double> & occs, const arma::mat & S) const;
  /// Complex-orbital occ-RI-K exchange; see calcK_occ.
  arma::cx_mat calcK_occ(const arma::cx_mat & C, const std::vector<double> & occs, const arma::mat & S) const;

  /// Get the number of auxiliary functions (DF), or of Cholesky vectors (CD)
  size_t Naux() const;
  /// Get the number of linearly independent fitting functions
  size_t Naux_indep() const;
  /// Get the (a|b) metric
  const arma::mat & ab() const;
  /// Get the metric half-inverse X, X^T (a|b) X = 1
  const arma::mat & metric_half_inverse() const { return X_; }

  /// Get the raw three-center integrals (mu nu|a), (Nbf*Nbf x Naux), in
  /// DF mode; in CD mode, where there is no auxiliary basis, the B matrix
  void three_center_integrals(arma::mat & B) const;
  /// Get the B matrix B(mu*Nbf+nu, Q), (Nbf*Nbf x Naux_indep)
  void B_matrix(arma::mat & B) const;
  /// Two-sided MO transform of the B tensor: returns Br with
  /// Br(P, r*Nl + l) = sum_{u,v} Cl(u,l) Cr(v,r) B_dense(u*Nbf+v, P).
  /// Used by post-HF consumers (moints.cpp); works equally for DF
  /// and CD-mode storage because it builds on B_matrix.
  arma::mat B_transform(const arma::mat & Cl, const arma::mat & Cr, bool verbose=false) const;

  /// Compute error in (AB|AB) type integrals
  double fitting_error() const;

  /// Save the (cached) DensityFit state to fname (HDF5, via the
  /// Checkpoint wrapper). Direct mode and uninitialised objects are
  /// rejected -- there's no precomputed integral storage to cache.
  /// Range-separated and plain entries can coexist in one file under
  /// distinct keys built from (omega, alpha, beta), so dfit and
  /// dfit_rs can share a CholeskyFile. Use-case: repeated SCF runs
  /// with different functionals on the same geometry / basis can
  /// load a previously cached set of integrals and skip the fill.
  void save(const std::string & fname) const;

  /// Try to load a DensityFit state matching the current
  /// (omega, alpha, beta) settings from fname. On success populates
  /// *this and returns true; on a missing file, missing key, or
  /// Nbf / Naux mismatch returns false and leaves *this unchanged.
  /// `auxbas` selects DF (non-null) vs CD (null) entries; in DF mode
  /// the loaded Naux is checked against auxbas.Nbf(). The basis
  /// is the orbital basis for the run; it must match Nbf and is used
  /// to repopulate orbpair / orbshell / aux-shell state.
  bool load(const BasisSet & basis, const BasisSet * auxbas, const std::string & fname);
};


#endif
