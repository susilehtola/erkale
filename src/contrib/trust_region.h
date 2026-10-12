/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_CONTRIB_TRUST_REGION
#define ERKALE_CONTRIB_TRUST_REGION

#include "openorbitaloptimizer/scfsolver.hpp"
#include "openorbitaloptimizer/armadillo_compat.hpp"
#include <opentrustregion.h>
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Second-order trust-region orbital optimization and stability analysis
 * with OpenTrustRegion (J. Greiner, I.-M. Hoyvik, S. Lehtola, and
 * J. J. Eriksen, J. Chem. Theory Comput. 22, 881 (2026)) for the
 * programs that build their Fock matrices for OpenOrbitalOptimizer
 * (erkale_neo, erkale_complex_orbs).
 *
 * The orbitals are given in blocks (particle types, spins, symmetries),
 * each in its own orthonormal basis. The orbital parameters are the
 * rotations between orbitals of different occupation within each block,
 * C_b -> C_b exp(K_b), so the block structure (and thereby the symmetry)
 * is preserved. The energy and the Fock matrices come from the program's
 * Fock builder. The gradient is g_ai = 2 (n_i - n_a) F_ai in the orbital
 * basis; Hessian-vector products are analytic given the Fock response,
 * and central differences of the gradient otherwise.
 *
 * The occupations are either fixed, or free to move between the blocks
 * of each particle type, as when an aufbau over symmetry blocks leaves a
 * shell fractionally occupied. Free occupations follow the second-order
 * optimizer of HelFEM (src/general/trustregion_scf.h):
 *
 * - Each block carries one continuous variable, its electron count q_b,
 *   and fills its orbitals in order, f_bi = clamp(q_b - i w_b, 0, w_b)
 *   with w_b the occupation of a filled orbital. A block thus holds at
 *   most one fractional orbital, k_b, and fills continuously past it.
 * - Only blocks whose k_b is fractional are free; a block at an exact
 *   shell closing sits on a kink of E(q), and an empty one on the bound
 *   q = 0. Whether such a block opens follows from the KKT conditions
 *   below.
 * - The electron count of each particle type is conserved, so the free
 *   q_b move in the sum-zero subspace, in an orthonormal (Helmert) basis
 *   that keeps the trust-region metric isotropic.
 * - dE/dq_b = F_kk (Janak). The uncoupled Hessian diagonal vanishes
 *   along every occupation direction, so the occupation block of the
 *   Hessian is built exactly at every accepted step and inverted exactly
 *   in the preconditioner, with its small curvatures floored: the
 *   minimum can be a valley of equivalent ensembles, such as charge
 *   moving between the complex m and -m orbitals in a pure functional.
 * - A step does not carry a free block past the filling or emptying of
 *   its fractional orbital: E(q) has a kink there, and the quadratic
 *   model does not hold across it. The occupation part of the step is cut
 *   short at the first kink, so that a step overshooting a shell closing
 *   does not leave a sliver of charge in the next orbital, to be drained
 *   back by the next step in an endless back and forth. When such a step
 *   lowers the energy, the solve stops there, and the parametrization is
 *   rebuilt with that block integer and frozen.
 * - A frozen block that violates the KKT conditions of the occupation
 *   problem (a full orbital above the Fermi level, an empty one below
 *   it, or with no free blocks, a full orbital above an empty one in
 *   another block) is released by moving a little charge, so that the
 *   solve does not stop at a stationary point of the problem with that
 *   block pinned. The orbital energies serve as the multipliers, so this
 *   is done only once the orbital gradient is small.
 * - A step that lowered the energy but that OpenTrustRegion declined is
 *   retaken, Hessian-vector products per macroiteration are capped, and
 *   convergence is judged from the gradient, since OpenTrustRegion
 *   reports a stalled line search as success.
 */
class TrustRegionSCF {
 public:
  using DensityMatrix = OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double>;
  using FockBuilder = OpenOrbitalOptimizer::Armadillo::FockBuilder<double, double>;
  /**
   * Linear response of the Fock matrices, for analytic Hessian-vector
   * products. Given the current orbitals and occupations and, for each
   * block, a transition density D_b = L_b R_b^T + R_b L_b^T (L and R in
   * the block's orthonormal basis, like the orbitals), it returns the
   * change of each block's Fock matrix to first order in D, in the same
   * basis as the Fock builder: dF_b = sum_c dF_b/dP_c D_c.
   */
  using ResponseBuilder = std::function<std::vector<arma::mat>(const DensityMatrix & dm, const std::vector<arma::mat> & L, const std::vector<arma::mat> & R)>;

  /// Block structure for free occupations
  struct FreeOccupations {
    /// Number of blocks of each particle type, in order
    std::vector<size_t> blocks_per_particle;
    /// Occupation of a filled orbital in each block
    std::vector<double> maxocc;
    /// Block labels for the output (optional)
    std::vector<std::string> descriptions;
  };

  /// Constructor for fixed occupations: Fock builder, the starting
  /// orbitals and occupations, and optionally the Fock response for
  /// analytic Hessian-vector products (without it, the products are
  /// finite differences of the gradient)
  TrustRegionSCF(const FockBuilder & fock, const DensityMatrix & dm, const ResponseBuilder & response=ResponseBuilder(), double fdstep=1e-4) : fock_(fock), response_(response), C_(dm.first), occ_(dm.second), fdstep_(fdstep) {
    // Rotations between orbitals of different occupation
    occupied_.resize(C_.size());
    for(size_t b=0;b<C_.size();b++) {
      std::vector<arma::uword> occd;
      for(size_t i=0;i<occ_[b].n_elem;i++) {
        bool is_occupied=false;
        for(size_t a=0;a<occ_[b].n_elem;a++)
          if(occ_[b](i)-occ_[b](a) > 1e-10) {
            pairs_.push_back({b, a, i});
            is_occupied=true;
          }
        if(is_occupied)
          occd.push_back(i);
      }
      occupied_[b]=arma::uvec(occd);
    }
    qoff_=nparam_=pairs_.size();
    evaluate(C_, occ_, E_, Fmo_);
    grad_=gradient(Fmo_, occ_);
  }

  /// Constructor for occupations that are free to move between the blocks
  /// of each particle type. The occupations are read as block electron
  /// counts; the orbitals must be in aufbau order within each block, as
  /// OpenOrbitalOptimizer hands them over.
  TrustRegionSCF(const FockBuilder & fock, const DensityMatrix & dm, const FreeOccupations & free, const ResponseBuilder & response=ResponseBuilder(), double fdstep=1e-4) : fock_(fock), response_(response), C_(dm.first), occ_(dm.second), fdstep_(fdstep), free_occ_(true), maxocc_(free.maxocc), labels_(free.descriptions) {
    size_t b=0;
    for(size_t n : free.blocks_per_particle) {
      std::vector<size_t> blocks;
      for(size_t i=0;i<n;i++)
        blocks.push_back(b++);
      pblocks_.push_back(blocks);
    }
    if(b!=C_.size() || maxocc_.size()!=C_.size() || occ_.size()!=C_.size())
      throw std::logic_error("TrustRegionSCF: the block structure does not match the orbitals.\n");
    if(labels_.size()!=C_.size()) {
      labels_.resize(C_.size());
      for(size_t c=0;c<C_.size();c++)
        labels_[c]="block " + std::to_string(c);
    }
    read_fillings();
    // The energy at the handover, from the occupations rebuilt from the
    // block electron counts
    evaluate(C_, occ_, E_, Fmo_);
    printf("Energy at the handover to the trust-region solver % .10f\n", E_);
    fflush(stdout);
    canonicalize();
    build_layout();
    derive();
  }

  /// Number of parameters
  size_t n_param() const { return nparam_; }
  /// Precondition the occupation directions with the exact occupation
  /// block (default)? Without it, OpenTrustRegion's diagonal
  /// preconditioner sees nothing along them: the uncoupled diagonal
  /// vanishes there. Only useful to measure what the exact block buys.
  void exact_preconditioner(bool exact) {
    exact_precond_=exact;
    if(!exact) {
      exact_val_.reset();
      exact_vec_.reset();
      if(free_occ_)
        hdiag_=hessian_diagonal(Fmo_);
    }
  }
  /// Current energy
  double energy() const { return E_; }
  /// Current orbitals and occupations
  DensityMatrix density_matrix() const { return std::make_pair(C_, occ_); }

  /**
   * Minimize the energy. With stability, OpenTrustRegion checks the
   * stability of the converged solution and follows any instability to
   * a lower solution. convthr is the convergence threshold for the
   * root-mean-square gradient. Returns whether the gradient converged.
   */
  bool optimize(double convthr, bool stability, int verbose) {
    if(free_occ_)
      return optimize_free(convthr, stability, verbose);
    if(!n_param())
      return true;
    solver_settings_type settings=solver_settings_init();
    settings.logger=&TrustRegionSCF::cb_logger;
    settings.stability=stability;
    settings.conv_tol=convthr;
    settings.verbose=verbose;
    Activate guard(this);
    const c_int err=solver(&TrustRegionSCF::cb_update, &TrustRegionSCF::cb_obj, (c_int) n_param(), settings);
    guard.rethrow();
    if(err!=0)
      throw std::runtime_error("OpenTrustRegion solver failed with error code " + std::to_string((long long) err) + ".\n");
    return true;
  }

  /**
   * Stability analysis of the current solution: is the lowest eigenvalue
   * of the Hessian positive? Only rotations that preserve the block
   * structure are examined, along with the occupation transfers if the
   * occupations are free.
   */
  bool is_stable(double convthr, int verbose) {
    if(!n_param())
      return true;
    stability_settings_type settings=stability_settings_init();
    settings.logger=&TrustRegionSCF::cb_logger;
    settings.conv_tol=convthr;
    settings.verbose=verbose;
    arma::vec h_diag(free_occ_ ? hdiag_ : hessian_diagonal(Fmo_));
    if(free_occ_ && exact_precond_)
      settings.precond=&TrustRegionSCF::cb_precond;
    arma::vec kappa(n_param(), arma::fill::zeros);
    c_bool stable=true;
    Activate guard(this);
    const c_int err=stability_check(h_diag.memptr(), &TrustRegionSCF::cb_hess_x, (c_int) n_param(), &stable, settings, kappa.memptr());
    guard.rethrow();
    if(err!=0)
      throw std::runtime_error("OpenTrustRegion stability check failed with error code " + std::to_string((long long) err) + ".\n");
    return stable;
  }

  /**
   * Check the gradient and the Hessian-vector products against central
   * differences of the energy, along the gradient, random directions, and
   * (with free occupations) pure rotation and pure occupation directions,
   * whose mixed product checks the rotation-occupation coupling. step is
   * the displacement; returns whether everything agrees to within tol
   * (relative) or the finite-difference noise.
   */
  bool check_derivatives(double step, double tol) const {
    if(!n_param())
      return true;
    auto energy_at=[&](const arma::vec & x) {
      std::vector<arma::mat> C;
      std::vector<arma::vec> occ;
      arma::vec q;
      displaced(x.memptr(), 1.0, C, occ, q);
      double E;
      std::vector<arma::mat> Fmo;
      evaluate(C, occ, E, Fmo);
      return E;
    };

    arma::arma_rng::set_seed(20260824);
    std::vector<arma::vec> D;
    std::vector<std::string> name;
    D.push_back(arma::normalise(grad_));
    name.push_back("gradient");
    D.push_back(arma::normalise(arma::vec(arma::randn(n_param()))));
    name.push_back("random");
    if(qoff_>0) {
      arma::vec d(n_param(), arma::fill::zeros);
      d.head(qoff_)=arma::randn(qoff_);
      D.push_back(arma::normalise(d));
      name.push_back("rotation");
    }
    if(nparam_>qoff_) {
      arma::vec d(n_param(), arma::fill::zeros);
      d.tail(nparam_-qoff_)=arma::randn(nparam_-qoff_);
      D.push_back(arma::normalise(d));
      name.push_back("occupation");
    }
    std::vector<arma::vec> HD(D.size());
    for(size_t i=0;i<D.size();i++)
      HD[i]=hessian_times(D[i].memptr());

    // The finite differences resolve no better than the roundoff of the
    // energy, eps |E| / h for the first and eps |E| / h^2 for the second
    // derivative, plus a truncation error of order h^2
    const double eps=std::numeric_limits<double>::epsilon();
    const double noise1=eps*std::abs(E_)/step, noise2=eps*std::abs(E_)/(step*step);
    bool ok=true;
    printf("\nFinite-difference check of the trust-region derivatives, step %.1e, %i parameters (%i occupation)\n", step, (int) n_param(), (int) (nparam_-qoff_));
    for(size_t i=0;i<D.size();i++) {
      const double num=(energy_at(step*D[i])-energy_at(-step*D[i]))/(2.0*step);
      const double ana=arma::dot(grad_, D[i]);
      const double atol=10.0*(noise1+step*step*std::abs(arma::dot(D[i], HD[i])));
      const bool pass=std::abs(num-ana) <= tol*std::max(std::abs(num), std::abs(ana)) + atol;
      printf("  g.d  %-10s  analytic % .10e  numerical % .10e  diff %.2e  noise %.1e%s\n", name[i].c_str(), ana, num, std::abs(num-ana), atol, pass ? "" : "  ***");
      ok=ok && pass;
    }
    // Bilinear form d1.H.d2 by four-point polarization
    for(size_t i=0;i<D.size();i++)
      for(size_t j=i;j<D.size();j++) {
        const arma::vec s(D[i]+D[j]), d(D[i]-D[j]);
        const double num=(energy_at(step*s)+energy_at(-step*s)-energy_at(step*d)-energy_at(-step*d))/(4.0*step*step);
        const double ana=arma::dot(D[i], HD[j]);
        const double atol=10.0*noise2;
        const bool pass=std::abs(num-ana) <= tol*std::max(std::abs(num), std::abs(ana)) + atol;
        printf("  d.H.d %-10s %-10s  analytic % .10e  numerical % .10e  diff %.2e  noise %.1e%s\n", name[i].c_str(), name[j].c_str(), ana, num, std::abs(num-ana), atol, pass ? "" : "  ***");
        ok=ok && pass;
      }
    // Symmetry of the analytic Hessian
    for(size_t i=0;i<D.size();i++)
      for(size_t j=i+1;j<D.size();j++) {
        const double ij=arma::dot(D[i], HD[j]), ji=arma::dot(D[j], HD[i]);
        const bool pass=std::abs(ij-ji) <= tol*std::max(std::abs(ij), std::abs(ji)) + 1e-12;
        printf("  symmetry %-10s %-10s  % .10e vs % .10e%s\n", name[i].c_str(), name[j].c_str(), ij, ji, pass ? "" : "  ***");
        ok=ok && pass;
      }
    printf("%s\n", ok ? "All derivatives agree." : "THE DERIVATIVES DISAGREE.");
    fflush(stdout);
    return ok;
  }

 private:
  /// A rotation between orbitals a and i of block b, n_i > n_a
  struct pair_t {
    size_t b, a, i;
  };

  FockBuilder fock_;
  /// Fock response (empty: finite-difference Hessian-vector products)
  ResponseBuilder response_;
  /// Orbitals and occupations by block
  std::vector<arma::mat> C_;
  std::vector<arma::vec> occ_;
  std::vector<pair_t> pairs_;
  /// Orbitals that lose occupation in some rotation, and the fractional
  /// orbital of each free block, by block
  std::vector<arma::uvec> occupied_;
  /// Finite-difference step for the Hessian-vector products
  double fdstep_;
  /// Energy, orbital-basis Fock matrices, and gradient at the current orbitals
  double E_;
  std::vector<arma::mat> Fmo_;
  arma::vec grad_;
  /// Number of parameters, and the offset of the occupation parameters
  /// (the rotations come first)
  size_t nparam_=0, qoff_=0;
  /// Exception thrown inside a callback, rethrown after the solver returns
  std::exception_ptr error_;

  /* Free occupations */
  bool free_occ_=false;
  /// Build the occupation block exactly for the preconditioner?
  bool exact_precond_=true;
  /// Occupation of a filled orbital, and the label of each block
  std::vector<double> maxocc_;
  std::vector<std::string> labels_;
  /// Blocks of each particle type, and those whose filling is free
  std::vector<std::vector<size_t>> pblocks_, free_;
  /// Orthonormal basis of the occupation parameters, n_blocks x n_occ:
  /// the sum-zero subspaces of the free fillings of each particle type
  /// (Helmert)
  arma::mat occbasis_;
  /// Electrons in each block, and the index of its fractional orbital:
  /// the first orbital that is not full
  arma::vec q_;
  std::vector<arma::uword> active_;
  /// Fractional orbitals the parametrization was built around, and
  /// whether the reference has left that pattern
  std::vector<arma::uword> entry_active_;
  bool pattern_changed_=false;
  /// Hessian diagonal: uncoupled for the rotations, exact for the
  /// occupations
  arma::vec hdiag_;
  /// Eigendecomposition of the exact occupation block of the Hessian
  arma::vec exact_val_;
  arma::mat exact_vec_;
  /// Ceiling on Hessian-vector products within one macroiteration, the
  /// count, and whether it was hit
  size_t hess_budget_=0, hess_this_macro_=0;
  bool budget_spent_=false;
  /// Has an objective evaluation stopped the solve at a kink, and the
  /// step that reached it
  bool kink_stop_=false;
  arma::vec kink_step_;
  /// The lowest-energy point any objective evaluation has reached since
  /// the last accepted step
  arma::vec best_step_;
  double best_energy_=0.0;
  /// Counters
  size_t n_update_=0, n_objective_=0, n_hessian_=0;

  /// An occupation this close to empty or full counts as integer
  static constexpr double occ_tol=1e-6;
  /// Floor on |lambda - mu| in the preconditioner, as in OpenTrustRegion
  static constexpr double precond_floor=1e-10;
  /// A departure from integer occupation up to this large, in a block
  /// that already has a fractional orbital, is taken as a residue of the
  /// first-order solution (natural occupations of an optimal-damping
  /// mixture) rather than as a second fractional orbital
  static constexpr double residue_tol=1e-4;
  /// Floor on the curvatures of the occupation block in the
  /// preconditioner, relative to the largest one
  static constexpr double valley_floor=1e-3;
  /// Electrons moved to release a block frozen against the KKT conditions
  static constexpr double kkt_nudge=1e-3;
  /// RMS orbital gradient below which the orbital energies are trusted as
  /// the multipliers of the KKT conditions
  static constexpr double kkt_gate=1e-4;
  /// Safety cap on the restarts of the solver after a pattern change, a
  /// declined step or a KKT release; a pattern that recurs without a
  /// lower energy stops the optimization before that
  static constexpr int max_restarts=100;

  /// The optimizer whose callbacks OpenTrustRegion calls; its C interface
  /// passes no user data.
  static TrustRegionSCF *& active() {
    static TrustRegionSCF * p=nullptr;
    return p;
  }
  /// Sets the active optimizer for the duration of a solver call
  struct Activate {
    TrustRegionSCF * self;
    explicit Activate(TrustRegionSCF * s) : self(s) { self->error_=nullptr; active()=s; }
    ~Activate() { active()=nullptr; }
    void rethrow() const { if(self->error_) std::rethrow_exception(self->error_); }
  };

  /// Occupation difference of a rotation pair
  static double dn(const pair_t & p, const std::vector<arma::vec> & occ) {
    return occ[p.b](p.i)-occ[p.b](p.a);
  }

  /// Block-antisymmetric generators from a parameter vector
  std::vector<arma::mat> generators(const double * x) const {
    std::vector<arma::mat> K(C_.size());
    for(size_t b=0;b<C_.size();b++)
      K[b].zeros(C_[b].n_cols, C_[b].n_cols);
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      K[p.b](p.a,p.i)=x[k];
      K[p.b](p.i,p.a)=-x[k];
    }
    return K;
  }
  /// Orbitals rotated by the parameters x, scaled by s
  std::vector<arma::mat> rotated(const double * x, double s=1.0) const {
    std::vector<arma::mat> K(generators(x));
    std::vector<arma::mat> C(C_);
    for(size_t b=0;b<C.size();b++)
      if(K[b].n_elem)
        C[b]=C[b]*arma::expmat(s*K[b]);
    return C;
  }
  /// Change of the block fillings for the occupation parameters of x
  arma::vec filling_change(const double * x) const {
    if(!occbasis_.n_cols)
      return arma::vec(C_.size(), arma::fill::zeros);
    return occbasis_*arma::vec(x+qoff_, occbasis_.n_cols);
  }
  /// Occupations of block b for the filling q, and its fractional orbital
  /// (the first one that is not full; found by scanning, since at an
  /// exact shell closing q/w can round to just below an integer)
  void fill(size_t b, double q, arma::vec & f, arma::uword & k) const {
    const arma::uword n=C_[b].n_cols;
    const double w=maxocc_[b];
    f.zeros(n);
    for(arma::uword i=0;i<n;i++)
      f(i)=std::min(std::max(q-i*w, 0.0), w);
    k=n ? n-1 : 0;
    for(arma::uword i=0;i<n;i++)
      if(f(i) < w-occ_tol) {
        k=i;
        break;
      }
  }
  /// Occupations for the fillings q
  std::vector<arma::vec> fillings(const arma::vec & q) const {
    std::vector<arma::vec> occ(C_.size());
    arma::uword k;
    for(size_t b=0;b<C_.size();b++)
      fill(b, q(b), occ[b], k);
    return occ;
  }
  /// Orbitals, occupations and fillings displaced from the reference by
  /// the parameters x, scaled by s. Only the free fillings move, so no
  /// charge leaks into a frozen block. The occupation part of the step is
  /// cut short where a free block's fractional orbital fills or empties,
  /// and that block is put exactly on the kink; kink says whether it was.
  void displaced(const double * x, double s, std::vector<arma::mat> & C, std::vector<arma::vec> & occ, arma::vec & q, bool * kink=nullptr) const {
    C=rotated(x, s);
    if(kink)
      *kink=false;
    if(!free_occ_) {
      occ=occ_;
      return;
    }
    q=q_+s*filling_change(x);
    for(size_t p=0;p<pblocks_.size();p++) {
      // The largest fraction t of the step that keeps every free block's
      // filling within the segment of its fractional orbital; scaling the
      // whole step keeps the electron count
      double t=1.0, qkink=0.0;
      size_t bkink=0;
      for(size_t b : free_[p]) {
        const double lo=active_[b]*maxocc_[b], hi=lo+maxocc_[b], d=q(b)-q_(b);
        const double bound=(d < 0.0) ? lo : hi;
        if((d < 0.0 && q(b) < lo) || (d > 0.0 && q(b) > hi)) {
          const double tb=(bound-q_(b))/d;
          if(tb < t) {
            t=tb;
            bkink=b;
            qkink=bound;
          }
        }
      }
      if(t < 1.0) {
        for(size_t b : free_[p])
          q(b)=q_(b)+t*(q(b)-q_(b));
        q(bkink)=qkink;
        if(kink)
          *kink=true;
      }
    }
    occ=fillings(q);
  }
  /// Energy and orbital-basis Fock matrices for the orbitals C
  void evaluate(const std::vector<arma::mat> & C, const std::vector<arma::vec> & occ, double & E, std::vector<arma::mat> & Fmo) const {
    const auto ret=fock_(std::make_pair(C, occ));
    E=ret.first;
    Fmo.resize(C.size());
    for(size_t b=0;b<C.size();b++)
      Fmo[b]=C[b].t()*ret.second[b]*C[b];
  }
  /// Gradient from the orbital-basis Fock matrices and the occupations
  arma::vec gradient(const std::vector<arma::mat> & Fmo, const std::vector<arma::vec> & occ) const {
    arma::vec g(nparam_);
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      g(k)=2.0*dn(p, occ)*Fmo[p.b](p.a,p.i);
    }
    // dE/dq_b is the energy of the fractional orbital (Janak)
    if(occbasis_.n_cols) {
      arma::vec gq(C_.size(), arma::fill::zeros);
      for(const auto & fp : free_)
        for(size_t b : fp)
          gq(b)=Fmo[b](active_[b],active_[b]);
      g.tail(occbasis_.n_cols)=occbasis_.t()*gq;
    }
    return g;
  }
  /// Uncoupled Hessian diagonal of the rotations; zero for the occupations
  arma::vec hessian_diagonal(const std::vector<arma::mat> & Fmo) const {
    arma::vec h(nparam_, arma::fill::zeros);
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      h(k)=2.0*dn(p, occ_)*(Fmo[p.b](p.a,p.a)-Fmo[p.b](p.i,p.i));
    }
    return h;
  }
  /// Hessian-vector product: analytic with a Fock response, else finite differences
  arma::vec hessian_times(const double * x) const {
    return response_ ? hessian_times_analytic(x) : hessian_times_fd(x);
  }

  /**
   * Analytic Hessian-vector product. With C -> C exp(K) and the
   * occupations N fixed, the energy to second order is
   *   E = E0 + tr(F [K,N]) + tr(F [K,[K,N]])/2 + tr([K,N] G([K,N]))/2
   * in the orbital basis, where F are the current Fock matrices and G(D)
   * the Fock response to the transition density D. Differentiating,
   *   (H x)_ai = dn [F,X]_ai + [F,M]_ai + 2 dn G(M)_ai,  M = [X,N].
   * A change dq_b of a free filling adds dq_b e_k e_k^T to M, with k the
   * fractional orbital; the rotation rows gain 2 F_ai d(dn) in all, and
   * the occupation rows are ([F,X] + G(M))_kk.
   * The transition density D = C M C^T is passed to the response as
   * L R^T + R L^T, with L the orbitals that lose occupation: M only has
   * rows and columns of those orbitals.
   */
  arma::vec hessian_times_analytic(const double * x) const {
    std::vector<arma::mat> X(generators(x));
    const arma::vec dq(free_occ_ ? filling_change(x) : arma::vec());
    std::vector<arma::mat> M(C_.size()), L(C_.size()), R(C_.size());
    for(size_t b=0;b<C_.size();b++) {
      const arma::mat N(arma::diagmat(occ_[b]));
      M[b]=X[b]*N-N*X[b];
      if(free_occ_ && dq(b)!=0.0)
        M[b](active_[b],active_[b])+=dq(b);
      const arma::uvec & I=occupied_[b];
      arma::mat MI(M[b].cols(I));
      MI.rows(I)-=0.5*M[b](I,I);
      L[b]=C_[b].cols(I);
      R[b]=C_[b]*MI;
    }
    const std::vector<arma::mat> dF(response_(std::make_pair(C_, occ_), L, R));

    arma::vec hx(nparam_);
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      const double dnp=dn(p, occ_);
      const arma::mat & F=Fmo_[p.b];
      const arma::mat & Xb=X[p.b];
      const arma::mat & Mb=M[p.b];
      const double FX=arma::dot(F.row(p.a), Xb.col(p.i))-arma::dot(Xb.row(p.a), F.col(p.i));
      const double FM=arma::dot(F.row(p.a), Mb.col(p.i))-arma::dot(Mb.row(p.a), F.col(p.i));
      const double G=arma::dot(C_[p.b].col(p.a), dF[p.b]*C_[p.b].col(p.i));
      hx(k)=dnp*FX+FM+2.0*dnp*G;
      if(free_occ_) {
        // The mixed term is 2 F_ai d(dn); [F,M] already carries one F_ai
        // d(dn), since M includes the change of the occupations
        const arma::uword kb=active_[p.b];
        const double ddn=(p.i==kb ? dq(p.b) : 0.0)-(p.a==kb ? dq(p.b) : 0.0);
        hx(k)+=F(p.a,p.i)*ddn;
      }
    }

    if(occbasis_.n_cols) {
      arma::vec hq(C_.size(), arma::fill::zeros);
      for(const auto & fp : free_)
        for(size_t b : fp) {
          const arma::uword kb=active_[b];
          const arma::mat & F=Fmo_[b];
          const double FX=arma::dot(F.row(kb), X[b].col(kb))-arma::dot(X[b].row(kb), F.col(kb));
          const arma::vec ck(C_[b].col(kb));
          hq(b)=FX+arma::dot(ck, dF[b]*ck);
        }
      hx.tail(occbasis_.n_cols)=occbasis_.t()*hq;
    }
    return hx;
  }

  /**
   * Finite-difference Hessian-vector product: central difference of the
   * gradient along x.
   * The gradients at the displaced orbitals are with respect to
   * rotations from those orbitals; converting them to the rotations from
   * the current orbitals adds [X, G], where X and G are the generators of
   * x and of the current gradient (G_ai = g_ai/2). This term vanishes at
   * convergence and makes the product that of the symmetric Hessian. The
   * fillings are additive, so the occupation rows need no such term.
   */
  arma::vec hessian_times_fd(const double * x) const {
    double E;
    std::vector<arma::mat> Cp, Cm, Fp, Fm;
    std::vector<arma::vec> occp, occm;
    arma::vec qp, qm;
    displaced(x, fdstep_, Cp, occp, qp);
    displaced(x, -fdstep_, Cm, occm, qm);
    evaluate(Cp, occp, E, Fp);
    evaluate(Cm, occm, E, Fm);
    arma::vec hx((gradient(Fp, occp)-gradient(Fm, occm))/(2.0*fdstep_));

    std::vector<arma::mat> X(generators(x));
    arma::vec halfg(0.5*grad_);
    std::vector<arma::mat> G(generators(halfg.memptr()));
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      const arma::mat & Xb=X[p.b];
      const arma::mat & Gb=G[p.b];
      hx(k)+=arma::dot(Xb.row(p.a), Gb.col(p.i))-arma::dot(Gb.row(p.a), Xb.col(p.i));
    }
    return hx;
  }

  /* Free occupations */

  /// Orthonormal (Helmert) basis of the sum-zero subspace of R^n
  static arma::mat helmert(size_t n) {
    arma::mat H(n, n ? n-1 : 0, arma::fill::zeros);
    for(size_t m=1;m<n;m++) {
      const double nrm=std::sqrt((double) m*(m+1));
      H.submat(0, m-1, m-1, m-1).fill(1.0/nrm);
      H(m, m-1)=-(double) m/nrm;
    }
    return H;
  }

  /// Read the block electron counts from the starting occupations, which
  /// must be in aufbau order with at most one fractional orbital per block
  void read_fillings() {
    q_.zeros(C_.size());
    active_.assign(C_.size(), 0);
    double dev=0.0;
    for(size_t b=0;b<C_.size();b++) {
      const arma::vec & n=occ_[b];
      const double w=maxocc_[b];
      if(n.n_elem!=C_[b].n_cols)
        throw std::logic_error("TrustRegionSCF: occupation and orbital counts differ in " + labels_[b] + ".\n");
      // Fractional orbitals, and among them those that are genuinely so
      // rather than a residue of an unconverged first-order solution
      int nfrac=0, nmajor=0;
      double residue=0.0;
      for(size_t i=0;i<n.n_elem;i++) {
        if(i>0 && n(i) > n(i-1)+occ_tol)
          throw std::runtime_error("The occupations of " + labels_[b] + " increase from orbital " + std::to_string(i-1) + " to " + std::to_string(i) + ". The trust-region solver fills each block in orbital order, so the orbitals must arrive sorted by energy.\n");
        const double d=std::min(n(i), w-n(i));
        if(d > occ_tol) {
          nfrac++;
          if(d > residue_tol)
            nmajor++;
          else
            residue=std::max(residue, d);
        }
      }
      if(nmajor > 1)
        throw std::runtime_error(labels_[b] + " has " + std::to_string(nmajor) + " fractionally occupied orbitals. The trust-region solver gives each block one electron count and so represents at most one fractional orbital per block; an ensemble of orbitals in the same block, such as 4s and 3d0 in the m=0 block of an atom, needs both. Fix the occupations with LinearOccupations, or leave TrustRegion off.\n");
      if(nfrac > nmajor && nfrac > 1)
        printf("%s: %i orbital(s) depart from integer occupation by at most %.2e, a residue of the first-order solution; the aufbau filling absorbs it.\n", labels_[b].c_str(), nfrac-nmajor, residue);
      q_(b)=arma::accu(n);
      arma::vec f;
      fill(b, q_(b), f, active_[b]);
      if(f.n_elem)
        dev=std::max(dev, arma::abs(f-n).max());
      occ_[b]=f;
    }
    printf("Occupations rebuilt from the block electron counts, largest change %.3e\n", dev);
    fflush(stdout);
  }

  /// Rotate the full and the empty orbitals of each block among
  /// themselves to diagonalize the Fock matrix there. This leaves the
  /// density and the energy unchanged, and makes the fill order the
  /// energy order, so that a block filling past its fractional orbital
  /// continues into its lowest empty one, and empties from its highest
  /// full one.
  void canonicalize() {
    for(size_t b=0;b<C_.size();b++) {
      const arma::uword n=C_[b].n_cols;
      const double w=maxocc_[b];
      const arma::uvec full(arma::find(occ_[b] >= w-occ_tol)), empty(arma::find(occ_[b] <= occ_tol));
      arma::mat W(n, n, arma::fill::eye);
      for(const arma::uvec & S : {full, empty}) {
        if(S.n_elem < 2)
          continue;
        arma::vec e;
        arma::mat U;
        arma::eig_sym(e, U, arma::symmatu(Fmo_[b](S,S)));
        W(S,S)=U;
      }
      C_[b]=C_[b]*W;
      Fmo_[b]=W.t()*Fmo_[b]*W;
    }
  }

  /// Build the parameter layout around the current occupation pattern
  void build_layout() {
    pairs_.clear();
    occupied_.assign(C_.size(), arma::uvec());
    free_.assign(pblocks_.size(), std::vector<size_t>());
    for(size_t p=0;p<pblocks_.size();p++)
      for(size_t b : pblocks_[p]) {
        const arma::uword n=C_[b].n_cols, k=active_[b];
        const bool is_free=n && occ_[b](k) > occ_tol && occ_[b](k) < maxocc_[b]-occ_tol;
        if(is_free)
          free_[p].push_back(b);
        // A rotation changes the density only between orbitals of
        // different occupation; pairs of equal occupation (two full or
        // two empty orbitals, or the empty orbital at a shell closing
        // with the empty ones above it) are redundant
        std::vector<arma::uword> occd;
        for(arma::uword i=0;i<n;i++) {
          bool is_occupied=false;
          for(arma::uword a=i+1;a<n;a++)
            if(occ_[b](i)-occ_[b](a) > occ_tol) {
              pairs_.push_back({b, a, i});
              is_occupied=true;
            }
          // The fractional orbital carries the occupation changes, so the
          // transition densities need it even when it is the last one
          if(is_occupied || (is_free && i==k))
            occd.push_back(i);
        }
        occupied_[b]=arma::uvec(occd);
      }
    qoff_=pairs_.size();
    size_t ny=0;
    for(const auto & fp : free_)
      ny+=fp.size() ? fp.size()-1 : 0;
    occbasis_.zeros(C_.size(), ny);
    size_t off=0;
    for(const auto & fp : free_) {
      if(fp.size() < 2)
        continue;
      const arma::mat H(helmert(fp.size()));
      occbasis_.submat(arma::conv_to<arma::uvec>::from(fp), arma::regspace<arma::uvec>(off, off+H.n_cols-1))=H;
      off+=H.n_cols;
    }
    nparam_=qoff_+occbasis_.n_cols;
    entry_active_=active_;
    pattern_changed_=false;
  }

  /// Has the reference left the occupation pattern the parametrization
  /// was built around: has a free block's fractional orbital filled or
  /// emptied, or has any block's fractional orbital moved?
  bool pattern_stale() const {
    for(size_t b=0;b<C_.size();b++)
      if(active_[b]!=entry_active_[b])
        return true;
    for(size_t p=0;p<free_.size();p++)
      for(size_t b : free_[p]) {
        const double fk=occ_[b](active_[b]);
        if(fk <= occ_tol || fk >= maxocc_[b]-occ_tol)
          return true;
      }
    return false;
  }

  /// Build the occupation block of the Hessian exactly, one product per
  /// occupation parameter, and diagonalize it
  void build_exact_block() {
    const size_t ny=nparam_-qoff_;
    exact_val_.reset();
    exact_vec_.reset();
    if(!ny)
      return;
    arma::mat Hs(ny, ny);
    for(size_t j=0;j<ny;j++) {
      arma::vec e(nparam_, arma::fill::zeros);
      e(qoff_+j)=1.0;
      Hs.col(j)=hessian_times(e.memptr()).tail(ny);
    }
    arma::eig_sym(exact_val_, exact_vec_, arma::mat(0.5*(Hs+Hs.t())));
  }

  /// Derived quantities at the reference: gradient, Hessian diagonal and
  /// the exact occupation block
  void derive() {
    grad_=gradient(Fmo_, occ_);
    hdiag_=hessian_diagonal(Fmo_);
    if(pattern_stale())
      // The parametrization no longer describes the reference, and the
      // solve will be stopped; the old occupation block will do until then
      pattern_changed_=true;
    else if(exact_precond_)
      build_exact_block();
    if(exact_val_.n_elem==nparam_-qoff_ && exact_val_.n_elem)
      hdiag_.tail(exact_val_.n_elem)=arma::diagvec(exact_vec_*arma::diagmat(exact_val_)*exact_vec_.t());
  }

  /// Adopt a point as the new reference
  void adopt(const std::vector<arma::mat> & C, const arma::vec & q) {
    C_=C;
    q_=q;
    for(size_t b=0;b<C_.size();b++)
      fill(b, q_(b), occ_[b], active_[b]);
    evaluate(C_, occ_, E_, Fmo_);
    derive();
    n_update_++;
    // A step was accepted, so the runaway counter starts over
    hess_this_macro_=0;
    best_energy_=E_;
    best_step_.reset();
  }

  /**
   * Preconditioner: the inverse of the diagonal shifted by mu, with the
   * occupation block inverted exactly in its eigenbasis. At mu = 0
   * OpenTrustRegion stands it in for its positive-definite preconditioner
   * (used in sqrt(x.Mx)), so absolute values are taken there; at a
   * nonzero shift it wants the signed inverse of H - mu.
   */
  void precondition(const double * r, double mu, double * out) const {
    const bool absolute=(mu==0.0);
    auto invert=[&](double d) {
      if(absolute)
        return 1.0/std::max(std::abs(d), precond_floor);
      if(std::abs(d) < precond_floor)
        d=(d < 0.0) ? -precond_floor : precond_floor;
      return 1.0/d;
    };
    for(size_t i=0;i<nparam_;i++)
      out[i]=r[i]*invert(hdiag_(i)-mu);
    const size_t ny=exact_val_.n_elem;
    if(!ny || ny!=nparam_-qoff_)
      return;
    arma::vec y(exact_vec_.t()*arma::vec(r+qoff_, ny));
    // An occupation direction whose curvature is small next to the others
    // is (or becomes, at convergence) a valley of equivalent ensembles, as
    // is moving charge between the complex m and -m orbitals in a pure
    // density functional. Inverting its curvature exactly sends the steps
    // along the valley, where the energy no longer changes, so floor it
    // relative to the largest one.
    const double vfloor=valley_floor*arma::abs(exact_val_).max();
    for(size_t i=0;i<ny;i++) {
      double d=exact_val_(i)-mu;
      if(std::abs(d) < vfloor)
        d=(d < 0.0 && !absolute) ? -vfloor : vfloor;
      y(i)*=invert(d);
    }
    const arma::vec z(exact_vec_*y);
    std::copy(z.begin(), z.end(), out+qoff_);
  }

  /// Fermi level of particle type p: the highest fractional-orbital
  /// energy of its free blocks
  double fermi_level(size_t p) const {
    double ef=-std::numeric_limits<double>::max();
    for(size_t b : free_[p])
      ef=std::max(ef, Fmo_[b](active_[b],active_[b]));
    return ef;
  }
  /// Energies of the highest full and lowest empty orbital of a frozen
  /// block (+-infinity where there is none)
  void frozen_edges(size_t b, double & etop, double & enext) const {
    const arma::uword n=C_[b].n_cols, k=active_[b];
    const bool all_full=occ_[b](k) >= maxocc_[b]-occ_tol;
    etop=-std::numeric_limits<double>::infinity();
    enext=std::numeric_limits<double>::infinity();
    if(all_full)
      etop=Fmo_[b](k,k);
    else {
      if(k>0)
        etop=Fmo_[b](k-1,k-1);
      if(k<n)
        enext=Fmo_[b](k,k);
    }
  }
  /// With every block of particle type p at an integer filling, the KKT
  /// conditions say no full orbital lies above an empty one. Is that
  /// violated across two blocks, where the orbital rotations cannot fix
  /// it? Gives the blocks with the highest full and the lowest empty
  /// orbital, and their energies.
  bool integer_aufbau_violation(size_t p, size_t & btop, size_t & bnext, double & emax, double & emin) const {
    btop=bnext=0;
    emax=-std::numeric_limits<double>::infinity();
    emin=std::numeric_limits<double>::infinity();
    for(size_t b : pblocks_[p]) {
      if(!C_[b].n_cols)
        continue;
      double etop, enext;
      frozen_edges(b, etop, enext);
      if(etop > emax) {
        emax=etop;
        btop=b;
      }
      if(enext < emin) {
        emin=enext;
        bnext=b;
      }
    }
    return btop!=bnext && emax > emin+1e-8;
  }
  bool is_free(size_t p, size_t b) const {
    return std::find(free_[p].begin(), free_[p].end(), b)!=free_[p].end();
  }

  /**
   * The KKT conditions of min E(n) s.t. 0 <= n <= w, sum n = N are, with
   * eps_i = dE/dn_i (Janak), eps_i = eps_F for fractional, eps_i <= eps_F
   * for full and eps_i >= eps_F for empty orbitals. A frozen block that
   * violates them is stationary only for the problem with it pinned, so
   * the gradient can be small while the energy is wrong. Move a little
   * charge the way the multipliers say, which makes the block free.
   * Returns the new fillings, or an empty vector if nothing was moved.
   */
  arma::vec release_kkt_violation() const {
    for(size_t p=0;p<pblocks_.size();p++) {
      if(free_[p].empty()) {
        // With every block at an integer filling, open the two blocks
        // between which the aufbau is violated
        size_t btop, bnext;
        double emax, emin;
        if(!integer_aufbau_violation(p, btop, bnext, emax, emin))
          continue;
        printf("\n%s is held full up to an orbital at % .6f, above the empty orbital of %s at % .6f:\nthe integer occupations are not aufbau. Moving charge between them.\n", labels_[btop].c_str(), emax, labels_[bnext].c_str(), emin);
        fflush(stdout);
        arma::vec q(q_);
        q(btop)-=kkt_nudge;
        q(bnext)+=kkt_nudge;
        return q;
      }
      const double ef=fermi_level(p);
      for(size_t b : pblocks_[p]) {
        if(!C_[b].n_cols || is_free(p, b))
          continue;
        double etop, enext;
        frozen_edges(b, etop, enext);
        arma::vec q(q_);
        if(etop > ef+1e-8) {
          // Full above the Fermi level: give charge to the free block
          // with the lowest fractional orbital
          size_t to=free_[p][0];
          for(size_t c : free_[p])
            if(Fmo_[c](active_[c],active_[c]) < Fmo_[to](active_[to],active_[to]))
              to=c;
          printf("\n%s is held full up to an orbital at % .6f, above the Fermi level % .6f:\nthe solution is stationary only with it pinned. Releasing charge to %s.\n", labels_[b].c_str(), etop, ef, labels_[to].c_str());
          fflush(stdout);
          q(b)-=kkt_nudge;
          q(to)+=kkt_nudge;
          return q;
        }
        if(enext < ef-1e-8) {
          // Its next orbital lies below the Fermi level: take charge from
          // the free block with the highest fractional orbital
          size_t from=free_[p][0];
          for(size_t c : free_[p])
            if(Fmo_[c](active_[c],active_[c]) > Fmo_[from](active_[from],active_[from]))
              from=c;
          if(q_(from) <= kkt_nudge)
            continue;
          printf("\n%s is held with its next orbital at % .6f, below the Fermi level % .6f:\nthe solution is stationary only with it pinned. Releasing charge from %s.\n", labels_[b].c_str(), enext, ef, labels_[from].c_str());
          fflush(stdout);
          q(b)+=kkt_nudge;
          q(from)-=kkt_nudge;
          return q;
        }
      }
    }
    return arma::vec();
  }

  /// Print the fractional occupations, the Fermi levels and whether the
  /// frozen blocks are aufbau-consistent with them
  void report_occupations() const {
    for(size_t p=0;p<pblocks_.size();p++) {
      if(free_[p].empty()) {
        size_t btop, bnext;
        double emax, emin;
        if(integer_aufbau_violation(p, btop, bnext, emax, emin))
          printf("WARNING: %s is held full up to an orbital at % .6f, above the empty orbital of %s at % .6f: the occupations are not aufbau.\n", labels_[btop].c_str(), emax, labels_[bnext].c_str(), emin);
        continue;
      }
      const double ef=fermi_level(p);
      printf("Particle type %i: Fermi level % .8f\n", (int) p, ef);
      for(size_t b : free_[p])
        printf("  %-12s orbital %3i occupation %.8f energy % .8f\n", labels_[b].c_str(), (int) active_[b], occ_[b](active_[b]), Fmo_[b](active_[b],active_[b]));
      for(size_t b : pblocks_[p]) {
        if(!C_[b].n_cols || is_free(p, b))
          continue;
        double etop, enext;
        frozen_edges(b, etop, enext);
        if(etop > ef+1e-8)
          printf("WARNING: %s is held full up to an orbital at % .6f, above the Fermi level: the occupations are not aufbau.\n", labels_[b].c_str(), etop);
        if(enext < ef-1e-8)
          printf("WARNING: %s is held with its next orbital at % .6f, below the Fermi level: the occupations are not aufbau.\n", labels_[b].c_str(), enext);
      }
    }
    if(exact_val_.n_elem) {
      printf("Eigenvalues of the occupation Hessian:");
      for(size_t i=0;i<exact_val_.n_elem;i++)
        printf(" % .6e", exact_val_(i));
      printf("\n");
    }
    fflush(stdout);
  }

  double gradient_rms() const {
    return nparam_ ? arma::norm(grad_)/std::sqrt((double) nparam_) : 0.0;
  }

  /// Optimize with free occupations: the solve runs in a loop that
  /// rebuilds the parametrization whenever the occupation pattern changes
  bool optimize_free(double convthr, bool stability, int verbose) {
    solver_settings_type settings=solver_settings_init();
    settings.logger=&TrustRegionSCF::cb_logger;
    settings.stability=stability;
    settings.conv_tol=convthr;
    settings.verbose=verbose;
    if(exact_precond_)
      settings.precond=&TrustRegionSCF::cb_precond;
    settings.conv_check=&TrustRegionSCF::cb_conv_check;
    // Tighter residual targets make OpenTrustRegion discard steps that
    // lowered the energy and landed on the minimum
    settings.global_red_factor=3e-1;
    settings.local_red_factor=3e-2;
    // OpenTrustRegion re-enters its microiterations until a step is
    // accepted, without resetting its subspace, so a rejection loop
    // grinds on; cap the products per macroiteration
    hess_budget_=5*(size_t) std::max<c_int>(1, settings.n_micro);
    n_update_=n_objective_=n_hessian_=0;

    int nrestart=0;
    // Occupation patterns visited, as the fractional orbital of each block
    // and whether the block is free, with the energy on entry
    std::vector<std::pair<std::vector<arma::uword>, double>> visited;
    for(int pass=0;;pass++) {
      // A block that fills or empties continues with its lowest empty or
      // highest full orbital, as the canonical orbitals say
      if(pass) {
        canonicalize();
        build_layout();
      }
      std::vector<arma::uword> pattern(active_);
      for(const auto & fp : free_)
        for(size_t b : fp)
          pattern.push_back(b);
      bool cycle=false;
      for(const auto & v : visited)
        if(v.first==pattern && E_ >= v.second-1e-10*std::abs(E_))
          cycle=true;
      if(cycle) {
        printf("\nThe occupation pattern recurs without a lower energy; stopping.\n");
        fflush(stdout);
        break;
      }
      visited.emplace_back(pattern, E_);
      size_t nfree=0;
      for(const auto & fp : free_)
        nfree+=fp.size();
      printf("\n%s over %i parameters: %i orbital rotations and %i occupation transfers among %i fractionally occupied blocks\n", pass ? "Restarting the trust-region optimization" : "Trust-region optimization", (int) nparam_, (int) qoff_, (int) (nparam_-qoff_), (int) nfree);
      fflush(stdout);
      if(!nparam_)
        break;

      budget_spent_=false;
      kink_stop_=false;
      c_int err;
      {
        Activate guard(this);
        err=solver(&TrustRegionSCF::cb_update, &TrustRegionSCF::cb_obj, (c_int) nparam_, settings);
        guard.rethrow();
      }
      if(err!=0 && !budget_spent_ && !kink_stop_)
        throw std::runtime_error("OpenTrustRegion solver failed with error code " + std::to_string((long long) err) + ".\n");
      if(budget_spent_) {
        printf("A macroiteration spent %i Hessian-vector products without accepting a step.\n", (int) hess_budget_);
        fflush(stdout);
      }
      if(pass >= max_restarts)
        break;

      // A step that reached a kink lowered the energy: restart from there,
      // with the block that reached it integer
      if(kink_stop_) {
        std::vector<arma::mat> C;
        std::vector<arma::vec> occ;
        arma::vec q;
        displaced(kink_step_.memptr(), 1.0, C, occ, q);
        adopt(C, q);
        printf("\nA step filled or emptied a fractional orbital, lowering the energy to % .10f; restarting at the shell closing.\n", E_);
        fflush(stdout);
        nrestart++;
        continue;
      }

      // OpenTrustRegion returns success on a stalled line search. If the
      // pass ends unconverged having declined a step that lowered the
      // energy, take that step and continue from there.
      if(gradient_rms() > convthr && best_step_.n_elem && best_energy_ < E_-1e-12*std::abs(E_)) {
        printf("\nThe solver declined a step that lowers the energy by %.3e; taking it and continuing.\n", E_-best_energy_);
        fflush(stdout);
        std::vector<arma::mat> C;
        std::vector<arma::vec> occ;
        arma::vec q;
        displaced(best_step_.memptr(), 1.0, C, occ, q);
        adopt(C, q);
        nrestart++;
        continue;
      }
      // Is the point a KKT point of the problem with free occupations? The
      // orbital energies are its multipliers once the orbitals are nearly
      // converged. Those of the full and the empty orbitals are the
      // eigenvalues of the Fock matrix within them: rotations among them
      // are redundant, so the optimizer leaves them mixed, and a diagonal
      // element can lie far from the edge of their spectrum.
      if(!pattern_changed_ && gradient_rms() <= std::max(convthr, kkt_gate)) {
        canonicalize();
        derive();
        const arma::vec q(release_kkt_violation());
        if(q.n_elem) {
          adopt(C_, q);
          nrestart++;
          continue;
        }
      }
      if(!pattern_changed_ || budget_spent_)
        break;
      printf("\nThe occupation pattern changed (a shell filled or emptied); rebuilding the parametrization around the new one.\n");
      fflush(stdout);
      nrestart++;
    }

    // Report convergence from the gradient: a zero return from
    // OpenTrustRegion does not mean the gradient came down
    const double grms=gradient_rms();
    const bool converged=grms <= convthr;
    printf("\nTrust-region optimization %s: energy % .10f, RMS gradient %.3e, %i restarts, %i updates, %i energy evaluations, %i Hessian-vector products\n", converged ? "converged" : "did NOT converge", E_, grms, nrestart, (int) n_update_, (int) n_objective_, (int) n_hessian_);
    fflush(stdout);
    if(!exact_precond_)
      build_exact_block();
    canonicalize();
    report_occupations();
    return converged;
  }

  /* OpenTrustRegion callbacks. Exceptions must not cross the Fortran
     library: they are stored and rethrown after it returns. */
  static c_int cb_update(const c_real * kappa, c_real * func, c_real * grad, c_real * h_diag, hess_x_fp * hess_x) {
    TrustRegionSCF * self=active();
    try {
      if(self->free_occ_) {
        std::vector<arma::mat> C;
        std::vector<arma::vec> occ;
        arma::vec q;
        self->displaced(kappa, 1.0, C, occ, q);
        self->adopt(C, q);
        std::copy(self->grad_.begin(), self->grad_.end(), grad);
        std::copy(self->hdiag_.begin(), self->hdiag_.end(), h_diag);
      } else {
        self->C_=self->rotated(kappa);
        self->evaluate(self->C_, self->occ_, self->E_, self->Fmo_);
        self->grad_=self->gradient(self->Fmo_, self->occ_);
        const arma::vec h(self->hessian_diagonal(self->Fmo_));
        for(size_t k=0;k<self->pairs_.size();k++) {
          grad[k]=self->grad_(k);
          h_diag[k]=h(k);
        }
      }
      *func=self->E_;
      *hess_x=&TrustRegionSCF::cb_hess_x;
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  static c_int cb_obj(const c_real * kappa, c_real * func) {
    TrustRegionSCF * self=active();
    try {
      std::vector<arma::mat> C, Fmo;
      std::vector<arma::vec> occ;
      arma::vec q;
      double E;
      bool kink;
      self->displaced(kappa, 1.0, C, occ, q, &kink);
      self->evaluate(C, occ, E, Fmo);
      *func=E;
      if(self->free_occ_) {
        self->n_objective_++;
        if(E < self->best_energy_) {
          self->best_energy_=E;
          self->best_step_=arma::vec(kappa, self->nparam_);
        }
        // The step was cut short at a kink and still lowered the energy:
        // stop the solve to restart from there. (Failing here is the clean
        // way out; optimize_free() treats it as a stop, not an error.)
        if(kink && E < self->E_) {
          self->kink_stop_=true;
          self->kink_step_=arma::vec(kappa, self->nparam_);
          return 1;
        }
      }
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  static c_int cb_hess_x(const c_real * x, c_real * hx) {
    TrustRegionSCF * self=active();
    try {
      if(self->free_occ_) {
        // Failing here is the clean way out of a runaway macroiteration;
        // optimize_free() treats it as a stop, not an error
        if(self->hess_budget_ && self->hess_this_macro_ >= self->hess_budget_) {
          self->budget_spent_=true;
          return 1;
        }
        self->hess_this_macro_++;
        self->n_hessian_++;
      }
      const arma::vec h(self->hessian_times(x));
      for(size_t k=0;k<self->nparam_;k++)
        hx[k]=h(k);
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  static c_int cb_precond(const c_real * residual, const c_real * mu, c_real * out) {
    TrustRegionSCF * self=active();
    try {
      self->precondition(residual, *mu, out);
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  /// Stops the solve when the occupation pattern has changed under it
  static c_int cb_conv_check(c_bool * converged) {
    *converged=active()->pattern_changed_;
    return 0;
  }
  static void cb_logger(const char * message) {
    printf("%s\n", message);
    fflush(stdout);
  }
};

#endif
