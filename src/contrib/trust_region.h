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
 * The occupations are either fixed, or free to move between the orbitals
 * of each particle type, as when an aufbau over symmetry blocks leaves a
 * shell fractionally occupied:
 *
 * - The occupation n_i of each orbital, 0 <= n_i <= w_b with w_b that of a
 *   filled orbital of its block, is a variable in its own right. The
 *   energy is smooth in the occupations, dE/dn_i = F_ii (Janak), so a
 *   block that fills past a shell closing or empties past one is no
 *   kink: the occupations of the orbitals on both sides move together.
 *   Nor is the number of fractional orbitals of a block limited.
 * - The free occupations are those of the fractional orbitals, and those
 *   of the orbitals at a bound that the KKT conditions of the occupation
 *   problem release: with eps_F the Fermi level (the mean energy of the
 *   fractional orbitals), a full orbital above it or an empty one below
 *   it; without fractional orbitals, the highest full orbital and the
 *   lowest empty one, if the former lies above the latter. Only the
 *   highest full and the lowest empty orbital of each block are
 *   candidates, in the canonical orbitals. This is checked whenever the
 *   parametrization is built, and again once the solver has converged.
 * - The electron count of each particle type is conserved, so the free
 *   occupations move in the sum-zero subspace, in an orthonormal
 *   (Helmert) basis that keeps the trust-region metric isotropic.
 * - The uncoupled Hessian diagonal vanishes along every occupation
 *   direction, so the occupation block of the Hessian is built exactly
 *   at every accepted step and inverted exactly in the preconditioner,
 *   with its small curvatures floored: the minimum can be a valley of
 *   equivalent ensembles, such as charge moving between the complex m and
 *   -m orbitals in a pure functional.
 * - A step that would carry a free occupation out of [0, w] is cut short
 *   where the first one reaches its bound, and a step that lowers the
 *   energy there stops the solve; the parametrization is rebuilt with
 *   that orbital at its bound, which the KKT conditions then keep or
 *   release. A free occupation at a bound whose orbital energy points
 *   out of it, relative to the mean of the others of its particle type,
 *   is held there: the occupation changes are projected onto the face
 *   that keeps it, the other free occupations making up the difference,
 *   and the gradient and the Hessian are those of the projected
 *   occupations, with a fixed curvature along the parameters that would
 *   only move held occupations. Whether it stays at the bound is then
 *   decided by the KKT conditions once the solver has converged.
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

  /// Constructor for occupations that are free to move between the
  /// orbitals of each particle type
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
    read_occupations();
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
    const solver_settings_type settings(solver_settings(convthr, stability, verbose));
    c_int err;
    {
      CallGuard guard(this);
      err=solver(&TrustRegionSCF::cb_update, &TrustRegionSCF::cb_obj, (c_int) n_param(), &settings, nullptr);
      guard.rethrow();
    }
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
    settings.context=this;
    settings.logger=&TrustRegionSCF::cb_logger;
    settings.conv_tol=convthr;
    settings.verbose=verbose;
    arma::vec h_diag(free_occ_ ? hdiag_ : hessian_diagonal(Fmo_));
    if(free_occ_ && exact_precond_)
      settings.precond=&TrustRegionSCF::cb_precond;
    arma::vec kappa(n_param(), arma::fill::zeros);
    c_bool stable=true;
    // The cap on Hessian-vector products is for the solver's
    // macroiterations, not for the eigensolver
    const size_t budget=hess_budget_;
    hess_budget_=0;
    c_int err;
    {
      CallGuard guard(this);
      err=stability_check(h_diag.memptr(), &TrustRegionSCF::cb_hess_x, (c_int) n_param(), &stable, &settings, nullptr, kappa.memptr());
      hess_budget_=budget;
      guard.rethrow();
    }
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
      displaced(x.memptr(), 1.0, C, occ);
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
  /// Blocks of each particle type
  std::vector<std::vector<size_t>> pblocks_;
  /// An orbital whose occupation is a variable
  struct var_t {
    size_t b;
    arma::uword i;
  };
  /// The free occupations, the particle type of each, which bound each
  /// was at when the parametrization was built (-1 empty, 0 fractional,
  /// +1 full), and a flag for the free orbitals of each block
  std::vector<var_t> vars_;
  std::vector<size_t> varp_;
  std::vector<int> entry_bound_;
  std::vector<arma::uvec> isvar_;
  /// The free occupations held at their bounds at the reference: those
  /// whose orbital energy points out of the bound, relative to the mean
  /// energy of the others of the particle type. The occupation changes
  /// are projected onto the face that keeps them, and the gradient and
  /// the Hessian are those of the projected occupations.
  std::vector<bool> held_;
  /// Orthonormal basis of the occupation parameters that the projection
  /// maps to no change, which get a curvature of held_stiffness so that
  /// the Hessian is not singular there; the gradient has no component
  /// along them, so the steps do not either
  arma::mat nullbasis_;
  /// Orthonormal basis of the occupation parameters, n_vars x n_occ: the
  /// sum-zero subspaces of the free occupations of each particle type
  /// (Helmert)
  arma::mat occbasis_;
  /// Has the reference left the bounds the parametrization was built
  /// around?
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
  /// Has an objective evaluation stopped the solve where a free
  /// occupation reached its bound, and the step that reached it
  bool bound_stop_=false;
  arma::vec bound_step_;
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
  /// A departure from integer occupation up to this large, in a particle
  /// type that has a genuinely fractional orbital, is taken as a residue
  /// of the first-order solution (natural occupations of an
  /// optimal-damping mixture) and snapped to the bound
  static constexpr double residue_tol=1e-4;
  /// Floor on the curvatures of the occupation block in the
  /// preconditioner, relative to the largest one
  static constexpr double valley_floor=1e-3;
  /// Violation of the KKT conditions, in orbital energy, that releases an
  /// orbital from its bound
  static constexpr double kkt_tol=1e-7;
  /// Floor on |n_i - n_a| in the Hessian diagonal of a rotation that
  /// involves a free occupation, relative to w: such occupations can be
  /// (nearly) equal, which leaves the rotation without curvature until
  /// they part
  static constexpr double dn_floor=1e-1;
  /// Curvature of the occupation parameters that move only held
  /// occupations
  static constexpr double held_stiffness=1.0;
  /// Safety cap on the restarts of the solver after a pattern change, a
  /// declined step or a KKT release; a pattern that recurs without a
  /// lower energy stops the optimization before that
  static constexpr int max_restarts=100;

  /// Clears the stored exception for the duration of a solver call, and
  /// rethrows one that a callback stored
  struct CallGuard {
    TrustRegionSCF * self;
    explicit CallGuard(TrustRegionSCF * s) : self(s) { self->error_=nullptr; }
    void rethrow() const { if(self->error_) std::rethrow_exception(self->error_); }
  };
  /// The settings of a solver call, with the callbacks reaching this
  /// object through the context
  solver_settings_type solver_settings(double convthr, bool stability, int verbose) {
    solver_settings_type settings=solver_settings_init();
    settings.context=this;
    settings.logger=&TrustRegionSCF::cb_logger;
    settings.stability=stability;
    settings.conv_tol=convthr;
    settings.verbose=verbose;
    settings.stability_settings.context=this;
    settings.stability_settings.logger=&TrustRegionSCF::cb_logger;
    return settings;
  }

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
  /// Which bound the occupation n of an orbital of block b is at: -1
  /// empty, +1 full, 0 neither
  int bound_state(size_t b, double n) const {
    if(n <= occ_tol)
      return -1;
    if(n >= maxocc_[b]-occ_tol)
      return 1;
    return 0;
  }
  /// Change of the free occupations for the occupation parameters of x
  arma::vec occupation_change(const double * x) const {
    if(!occbasis_.n_cols)
      return arma::vec(vars_.size(), arma::fill::zeros);
    return occbasis_*arma::vec(x+qoff_, occbasis_.n_cols);
  }
  /// The change dv of the free occupations by block
  std::vector<arma::vec> by_block(const arma::vec & dv) const {
    std::vector<arma::vec> d(C_.size());
    for(size_t b=0;b<C_.size();b++)
      d[b].zeros(C_[b].n_cols);
    for(size_t v=0;v<vars_.size();v++)
      d[vars_[v].b](vars_[v].i)=dv(v);
    return d;
  }
  /// The occupation change d with the held occupations kept, the others
  /// of their particle type making up the difference equally (with none
  /// left, the change of the particle type vanishes)
  arma::vec project_change(arma::vec d, const std::vector<bool> & held) const {
    for(size_t p=0;p<pblocks_.size();p++) {
      double excess=0.0;
      size_t nfree=0;
      for(size_t v=0;v<vars_.size();v++)
        if(varp_[v]==p) {
          if(held[v]) {
            excess+=d(v);
            d(v)=0.0;
          } else
            nfree++;
        }
      for(size_t v=0;v<vars_.size();v++)
        if(varp_[v]==p && !held[v])
          d(v) = nfree ? d(v)+excess/nfree : 0.0;
    }
    return d;
  }
  /// The transpose of project_change() for the held occupations of the
  /// reference: a held occupation takes the mean of the others of its
  /// particle type
  arma::vec project_derivative(arma::vec g) const {
    for(size_t p=0;p<pblocks_.size();p++) {
      double mean=0.0;
      size_t nfree=0;
      for(size_t v=0;v<vars_.size();v++)
        if(varp_[v]==p && !held_[v]) {
          mean+=g(v);
          nfree++;
        }
      mean = nfree ? mean/nfree : 0.0;
      for(size_t v=0;v<vars_.size();v++)
        if(varp_[v]==p && held_[v])
          g(v)=mean;
      if(!nfree)
        for(size_t v=0;v<vars_.size();v++)
          if(varp_[v]==p)
            g(v)=0.0;
    }
    return g;
  }
  /// Find the free occupations held at their bounds at the reference,
  /// and the occupation parameters that only move them
  void find_held() {
    held_.assign(vars_.size(), false);
    nullbasis_.reset();
    for(bool changed=true;changed;) {
      changed=false;
      for(size_t p=0;p<pblocks_.size();p++) {
        double mean=0.0;
        size_t nfree=0;
        for(size_t v=0;v<vars_.size();v++)
          if(varp_[v]==p && !held_[v]) {
            mean+=Fmo_[vars_[v].b](vars_[v].i,vars_[v].i);
            nfree++;
          }
        if(!nfree)
          continue;
        mean/=nfree;
        for(size_t v=0;v<vars_.size();v++) {
          if(varp_[v]!=p || held_[v])
            continue;
          const double e=Fmo_[vars_[v].b](vars_[v].i,vars_[v].i);
          const int st=bound_state(vars_[v].b, occ_[vars_[v].b](vars_[v].i));
          if((st>0 && e < mean-kkt_tol) || (st<0 && e > mean+kkt_tol)) {
            held_[v]=true;
            changed=true;
          }
        }
      }
    }
    if(occbasis_.n_cols && std::find(held_.begin(), held_.end(), true)!=held_.end()) {
      arma::mat M(vars_.size(), occbasis_.n_cols);
      for(size_t j=0;j<occbasis_.n_cols;j++)
        M.col(j)=project_change(occbasis_.col(j), held_);
      nullbasis_=arma::null(M);
    }
  }

  /// Orbitals and occupations displaced from the reference by the
  /// parameters x, scaled by s. The held occupations, and any other at a
  /// bound that the step would push further out, stay there, the other
  /// free occupations of their particle type making up the difference;
  /// the occupation part of the step is then cut short where the first
  /// free occupation reaches its bound, which is put exactly there.
  /// bound says whether that happened.
  void displaced(const double * x, double s, std::vector<arma::mat> & C, std::vector<arma::vec> & occ, bool * bound=nullptr) const {
    C=rotated(x, s);
    occ=occ_;
    if(bound)
      *bound=false;
    if(!free_occ_ || !occbasis_.n_cols)
      return;
    // Hold also the occupations that the step pushes out of their
    // bounds, until none is
    std::vector<bool> held(held_);
    arma::vec d(project_change(s*occupation_change(x), held));
    for(bool changed=true;changed;) {
      changed=false;
      for(size_t v=0;v<vars_.size();v++) {
        const int st=bound_state(vars_[v].b, occ_[vars_[v].b](vars_[v].i));
        if(!held[v] && ((st<0 && d(v)<0.0) || (st>0 && d(v)>0.0))) {
          held[v]=true;
          changed=true;
        }
      }
      if(changed)
        d=project_change(d, held);
    }
    // The largest fraction t of the step that keeps the occupations
    // within their bounds; scaling the whole step keeps the electron
    // counts
    double t=1.0, nbound=0.0;
    size_t vbound=0;
    for(size_t v=0;v<vars_.size();v++) {
      const double n=occ_[vars_[v].b](vars_[v].i), w=maxocc_[vars_[v].b];
      double tv=1.0, nb=0.0;
      if(n+d(v) < 0.0) {
        tv=std::max(0.0, -n/d(v));
        nb=0.0;
      } else if(n+d(v) > w) {
        tv=std::max(0.0, (w-n)/d(v));
        nb=w;
      }
      if(tv < t) {
        t=tv;
        vbound=v;
        nbound=nb;
      }
    }
    const std::vector<arma::vec> dn(by_block(t*d));
    for(size_t b=0;b<C_.size();b++)
      occ[b]+=dn[b];
    if(t < 1.0) {
      occ[vars_[vbound].b](vars_[vbound].i)=nbound;
      if(bound)
        *bound=true;
    }
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
    // dE/dn_i is the orbital energy (Janak)
    if(occbasis_.n_cols) {
      arma::vec gv(vars_.size());
      for(size_t v=0;v<vars_.size();v++)
        gv(v)=Fmo[vars_[v].b](vars_[v].i,vars_[v].i);
      g.tail(occbasis_.n_cols)=occbasis_.t()*project_derivative(gv);
    }
    return g;
  }
  /// Uncoupled Hessian diagonal of the rotations; zero for the
  /// occupations. A rotation that involves a free occupation has its
  /// occupation difference floored, as the two can meet.
  arma::vec hessian_diagonal(const std::vector<arma::mat> & Fmo) const {
    arma::vec h(nparam_, arma::fill::zeros);
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      double d=dn(p, occ_);
      if(free_occ_ && (isvar_[p.b](p.a) || isvar_[p.b](p.i)))
        d=std::max(d, dn_floor*maxocc_[p.b]);
      h(k)=2.0*d*(Fmo[p.b](p.a,p.a)-Fmo[p.b](p.i,p.i));
    }
    return h;
  }
  /// Hessian-vector product: analytic with a Fock response, else finite differences
  arma::vec hessian_times(const double * x) const {
    arma::vec hx(response_ ? hessian_times_analytic(x) : hessian_times_fd(x));
    if(nullbasis_.n_cols)
      hx.tail(nullbasis_.n_rows)+=held_stiffness*nullbasis_*(nullbasis_.t()*arma::vec(x+qoff_, nullbasis_.n_rows));
    return hx;
  }

  /**
   * Analytic Hessian-vector product. With C -> C exp(K) and the
   * occupations N fixed, the energy to second order is
   *   E = E0 + tr(F [K,N]) + tr(F [K,[K,N]])/2 + tr([K,N] G([K,N]))/2
   * in the orbital basis, where F are the current Fock matrices and G(D)
   * the Fock response to the transition density D. Differentiating,
   *   (H x)_ai = dn [F,X]_ai + [F,M]_ai + 2 dn G(M)_ai,  M = [X,N].
   * A change d_k of a free occupation adds d_k e_k e_k^T to M; the
   * rotation rows gain 2 F_ai d(dn) in all, and the occupation rows are
   * ([F,X] + G(M))_kk.
   * The transition density D = C M C^T is passed to the response as
   * L R^T + R L^T, with L the orbitals that lose occupation: M only has
   * rows and columns of those orbitals.
   */
  arma::vec hessian_times_analytic(const double * x) const {
    std::vector<arma::mat> X(generators(x));
    const std::vector<arma::vec> dv(free_occ_ ? by_block(project_change(occupation_change(x), held_)) : std::vector<arma::vec>());
    std::vector<arma::mat> M(C_.size()), L(C_.size()), R(C_.size());
    for(size_t b=0;b<C_.size();b++) {
      const arma::mat N(arma::diagmat(occ_[b]));
      M[b]=X[b]*N-N*X[b];
      if(free_occ_)
        M[b].diag()+=dv[b];
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
        hx(k)+=F(p.a,p.i)*(dv[p.b](p.i)-dv[p.b](p.a));
      }
    }

    if(occbasis_.n_cols) {
      arma::vec hv(vars_.size());
      for(size_t v=0;v<vars_.size();v++) {
        const size_t b=vars_[v].b;
        const arma::uword kb=vars_[v].i;
        const arma::mat & F=Fmo_[b];
        const double FX=arma::dot(F.row(kb), X[b].col(kb))-arma::dot(X[b].row(kb), F.col(kb));
        const arma::vec ck(C_[b].col(kb));
        hv(v)=FX+arma::dot(ck, dF[b]*ck);
      }
      hx.tail(occbasis_.n_cols)=occbasis_.t()*project_derivative(hv);
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
   * occupations are additive, so the occupation rows need no such term.
   */
  arma::vec hessian_times_fd(const double * x) const {
    double E;
    std::vector<arma::mat> Cp, Cm, Fp, Fm;
    std::vector<arma::vec> occp, occm;
    displaced(x, fdstep_, Cp, occp);
    displaced(x, -fdstep_, Cm, occm);
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

  /// Read the starting occupations. In a particle type with a genuinely
  /// fractional orbital, departures from integer occupation below
  /// residue_tol are residues of the first-order solution: they are
  /// snapped to the bound, the most fractional orbital taking up the
  /// difference.
  void read_occupations() {
    for(size_t b=0;b<C_.size();b++) {
      if(occ_[b].n_elem!=C_[b].n_cols)
        throw std::logic_error("TrustRegionSCF: occupation and orbital counts differ in " + labels_[b] + ".\n");
      if(occ_[b].n_elem && (occ_[b].min() < -occ_tol || occ_[b].max() > maxocc_[b]+occ_tol))
        throw std::runtime_error("The occupations of " + labels_[b] + " are not within [0, " + std::to_string(maxocc_[b]) + "].\n");
      occ_[b]=arma::clamp(occ_[b], 0.0, maxocc_[b]);
    }
    for(size_t p=0;p<pblocks_.size();p++) {
      // The most fractional orbital
      double dmax=0.0;
      size_t bmax=0;
      arma::uword imax=0;
      for(size_t b : pblocks_[p])
        for(arma::uword i=0;i<occ_[b].n_elem;i++) {
          const double d=std::min(occ_[b](i), maxocc_[b]-occ_[b](i));
          if(d > dmax) {
            dmax=d;
            bmax=b;
            imax=i;
          }
        }
      if(dmax <= residue_tol)
        continue;
      double moved=0.0, residue=0.0;
      int nres=0;
      for(size_t b : pblocks_[p])
        for(arma::uword i=0;i<occ_[b].n_elem;i++) {
          double & n=occ_[b](i);
          const double d=std::min(n, maxocc_[b]-n);
          if(d <= occ_tol || d > residue_tol)
            continue;
          const double snapped = (n < 0.5*maxocc_[b]) ? 0.0 : maxocc_[b];
          moved+=n-snapped;
          residue=std::max(residue, d);
          nres++;
          n=snapped;
        }
      occ_[bmax](imax)+=moved;
      if(nres) {
        printf("%i orbital(s) depart from integer occupation by at most %.2e, a residue of the first-order solution; snapped to integer, with %s taking up %.2e electrons.\n", nres, residue, labels_[bmax].c_str(), moved);
        fflush(stdout);
      }
    }
  }

  /// Rotate the orbitals of equal occupation in each block among
  /// themselves to diagonalize the Fock matrix there, which leaves the
  /// density and the energy unchanged, and sort the orbitals by
  /// decreasing occupation and then increasing energy. The diagonal of
  /// the Fock matrix then gives the edges of the spectra of the full and
  /// the empty orbitals, which the KKT conditions compare.
  void canonicalize() {
    for(size_t b=0;b<C_.size();b++) {
      const arma::uword n=C_[b].n_cols;
      if(!n)
        continue;
      const arma::vec & f=occ_[b];
      arma::uvec order(arma::sort_index(f, "descend"));
      arma::mat W(n, n, arma::fill::eye);
      for(arma::uword g0=0;g0<n;) {
        arma::uword g1=g0+1;
        while(g1<n && f(order(g0))-f(order(g1)) <= occ_tol)
          g1++;
        if(g1-g0 > 1) {
          const arma::uvec S(order.subvec(g0, g1-1));
          arma::vec e;
          arma::mat U;
          arma::eig_sym(e, U, arma::symmatu(Fmo_[b](S,S)));
          W(S,S)=U;
        }
        g0=g1;
      }
      C_[b]=C_[b]*W;
      Fmo_[b]=W.t()*Fmo_[b]*W;
      // Sort
      std::vector<arma::uword> perm(n);
      for(arma::uword i=0;i<n;i++)
        perm[i]=i;
      std::stable_sort(perm.begin(), perm.end(), [&](arma::uword i, arma::uword j) {
        if(std::abs(f(i)-f(j)) > occ_tol)
          return f(i) > f(j);
        return Fmo_[b](i,i) < Fmo_[b](j,j);
      });
      const arma::uvec P(arma::conv_to<arma::uvec>::from(perm));
      C_[b]=arma::mat(C_[b].cols(P));
      occ_[b]=arma::vec(occ_[b](P));
      Fmo_[b]=arma::mat(Fmo_[b](P,P));
    }
  }

  /// Fermi level of particle type p: the mean energy of its fractional
  /// orbitals; false if there are none
  bool fermi_level(size_t p, double & ef) const {
    ef=0.0;
    int nf=0;
    for(size_t b : pblocks_[p])
      for(arma::uword i=0;i<occ_[b].n_elem;i++)
        if(!bound_state(b, occ_[b](i))) {
          ef+=Fmo_[b](i,i);
          nf++;
        }
    if(nf)
      ef/=nf;
    return nf>0;
  }

  /**
   * The orbitals of particle type p at a bound that violate the KKT
   * conditions of min E(n) s.t. 0 <= n_i <= w, sum n_i = N: with
   * eps_i = dE/dn_i (Janak), eps_i = eps_F for fractional, eps_i <= eps_F
   * for full and eps_i >= eps_F for empty orbitals. Only the highest full
   * and the lowest empty orbital of each block are candidates; the
   * orbitals must be canonical. Without fractional orbitals, the
   * violation is a full orbital above an empty one, and both are given.
   */
  std::vector<var_t> kkt_violations(size_t p) const {
    // The highest full and the lowest empty orbital of each block
    std::vector<var_t> top, bottom;
    for(size_t b : pblocks_[p]) {
      bool havetop=false, havebottom=false;
      var_t t{b, 0}, u{b, 0};
      for(arma::uword i=0;i<occ_[b].n_elem;i++) {
        const int st=bound_state(b, occ_[b](i));
        if(st>0 && (!havetop || Fmo_[b](i,i) > Fmo_[b](t.i,t.i))) {
          t.i=i;
          havetop=true;
        }
        if(st<0 && (!havebottom || Fmo_[b](i,i) < Fmo_[b](u.i,u.i))) {
          u.i=i;
          havebottom=true;
        }
      }
      if(havetop)
        top.push_back(t);
      if(havebottom)
        bottom.push_back(u);
    }
    auto eps=[&](const var_t & v) { return Fmo_[v.b](v.i,v.i); };
    std::vector<var_t> viol;
    double ef;
    if(fermi_level(p, ef)) {
      for(const auto & v : top)
        if(eps(v) > ef+kkt_tol)
          viol.push_back(v);
      for(const auto & v : bottom)
        if(eps(v) < ef-kkt_tol)
          viol.push_back(v);
      return viol;
    }
    if(top.empty() || bottom.empty())
      return viol;
    var_t vt=top[0], vb=bottom[0];
    for(const auto & v : top)
      if(eps(v) > eps(vt))
        vt=v;
    for(const auto & v : bottom)
      if(eps(v) < eps(vb))
        vb=v;
    if(eps(vt) > eps(vb)+kkt_tol) {
      viol.push_back(vt);
      viol.push_back(vb);
    }
    return viol;
  }

  /// Build the parameter layout around the current occupations, freeing
  /// the fractional orbitals and those that the KKT conditions release.
  /// Returns the number of orbitals released from a bound.
  size_t build_layout() {
    pairs_.clear();
    occupied_.assign(C_.size(), arma::uvec());
    vars_.clear();
    varp_.clear();
    entry_bound_.clear();
    isvar_.assign(C_.size(), arma::uvec());
    for(size_t b=0;b<C_.size();b++)
      isvar_[b].zeros(C_[b].n_cols);
    size_t released=0;
    for(size_t p=0;p<pblocks_.size();p++) {
      std::vector<var_t> pv;
      for(size_t b : pblocks_[p])
        for(arma::uword i=0;i<occ_[b].n_elem;i++)
          if(!bound_state(b, occ_[b](i)))
            pv.push_back({b, i});
      const std::vector<var_t> viol(kkt_violations(p));
      for(const auto & v : viol) {
        printf("%s orbital %i at % .6f is %s, against the KKT conditions: releasing it.\n", labels_[v.b].c_str(), (int) v.i, Fmo_[v.b](v.i,v.i), occ_[v.b](v.i) > 0.5*maxocc_[v.b] ? "full" : "empty");
        fflush(stdout);
        pv.push_back(v);
      }
      released+=viol.size();
      for(const auto & v : pv) {
        vars_.push_back(v);
        varp_.push_back(p);
        entry_bound_.push_back(bound_state(v.b, occ_[v.b](v.i)));
        isvar_[v.b](v.i)=1;
      }
    }

    for(size_t b=0;b<C_.size();b++) {
      const arma::uword n=C_[b].n_cols;
      const arma::vec & f=occ_[b];
      // A rotation changes the density only between orbitals of
      // different occupation; pairs of equal occupation are redundant,
      // unless one of the occupations is free
      std::vector<arma::uword> occd;
      for(arma::uword i=0;i<n;i++) {
        for(arma::uword a=i+1;a<n;a++)
          if(std::abs(f(i)-f(a)) > occ_tol || isvar_[b](i) || isvar_[b](a)) {
            if(f(i) >= f(a))
              pairs_.push_back({b, a, i});
            else
              pairs_.push_back({b, i, a});
          }
        // The orbitals that carry occupation, or whose occupation can
        // change, enter the transition densities
        if(f(i) > occ_tol || isvar_[b](i))
          occd.push_back(i);
      }
      occupied_[b]=arma::uvec(occd);
    }
    qoff_=pairs_.size();

    size_t ny=0;
    for(size_t p=0;p<pblocks_.size();p++) {
      const size_t m=std::count(varp_.begin(), varp_.end(), p);
      ny+=m ? m-1 : 0;
    }
    occbasis_.zeros(vars_.size(), ny);
    size_t off=0;
    for(size_t p=0;p<pblocks_.size();p++) {
      std::vector<arma::uword> rows;
      for(size_t v=0;v<vars_.size();v++)
        if(varp_[v]==p)
          rows.push_back(v);
      if(rows.size() < 2)
        continue;
      const arma::mat H(helmert(rows.size()));
      occbasis_.submat(arma::conv_to<arma::uvec>::from(rows), arma::regspace<arma::uvec>(off, off+H.n_cols-1))=H;
      off+=H.n_cols;
    }
    nparam_=qoff_+occbasis_.n_cols;
    pattern_changed_=false;
    find_held();
    return released;
  }

  /// Has a free occupation reached a bound it was not at when the
  /// parametrization was built?
  bool pattern_stale() const {
    for(size_t v=0;v<vars_.size();v++) {
      const int st=bound_state(vars_[v].b, occ_[vars_[v].b](vars_[v].i));
      if(st && st!=entry_bound_[v])
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
    if(free_occ_)
      find_held();
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
  void adopt(const std::vector<arma::mat> & C, const std::vector<arma::vec> & occ) {
    C_=C;
    occ_=occ;
    evaluate(C_, occ_, E_, Fmo_);
    derive();
    n_update_++;
    // A step was accepted, so the runaway counter starts over
    hess_this_macro_=0;
    best_energy_=E_;
    best_step_.reset();
  }

  /// Move the reference by a step of the parameters
  void take_step(const double * kappa) {
    if(free_occ_) {
      std::vector<arma::mat> C;
      std::vector<arma::vec> occ;
      displaced(kappa, 1.0, C, occ);
      adopt(C, occ);
    } else {
      C_=rotated(kappa);
      evaluate(C_, occ_, E_, Fmo_);
      grad_=gradient(Fmo_, occ_);
    }
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

  /// Print the fractional occupations, the Fermi levels and whether the
  /// orbitals at their bounds are consistent with them
  void report_occupations() const {
    for(size_t p=0;p<pblocks_.size();p++) {
      double ef;
      if(fermi_level(p, ef)) {
        printf("Particle type %i: Fermi level % .8f\n", (int) p, ef);
        for(size_t b : pblocks_[p])
          for(arma::uword i=0;i<occ_[b].n_elem;i++)
            if(!bound_state(b, occ_[b](i)))
              printf("  %-12s orbital %3i occupation %.8f energy % .8f\n", labels_[b].c_str(), (int) i, occ_[b](i), Fmo_[b](i,i));
      }
      for(const auto & v : kkt_violations(p))
        printf("WARNING: %s orbital %i at % .6f is %s against the KKT conditions: the occupations are not aufbau.\n", labels_[v.b].c_str(), (int) v.i, Fmo_[v.b](v.i,v.i), occ_[v.b](v.i) > 0.5*maxocc_[v.b] ? "full" : "empty");
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
  /// rebuilds the parametrization whenever a free occupation reaches a
  /// bound, and once the solver has converged, until the KKT conditions
  /// release no orbital
  bool optimize_free(double convthr, bool stability, int verbose) {
    solver_settings_type settings(solver_settings(convthr, stability, verbose));
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
    // Did the last solve finish? Then a layout that releases nothing ends
    // the optimization.
    bool finished=false;
    // Free occupations visited, with the energy on entry
    std::vector<std::pair<std::vector<arma::uword>, double>> visited;
    for(int pass=0;;pass++) {
      if(pass) {
        canonicalize();
        const size_t released=build_layout();
        if(finished && !released)
          break;
        nrestart++;
      }
      std::vector<arma::uword> pattern;
      for(const auto & v : vars_) {
        pattern.push_back(v.b);
        pattern.push_back(v.i);
      }
      bool cycle=false;
      for(const auto & v : visited)
        if(v.first==pattern && E_ >= v.second-1e-10*std::abs(E_))
          cycle=true;
      if(cycle) {
        printf("\nThe free occupations recur without a lower energy; stopping.\n");
        fflush(stdout);
        break;
      }
      visited.emplace_back(pattern, E_);
      printf("\n%s over %i parameters: %i orbital rotations and %i occupation transfers among %i free occupations\n", pass ? "Restarting the trust-region optimization" : "Trust-region optimization", (int) nparam_, (int) qoff_, (int) (nparam_-qoff_), (int) vars_.size());
      fflush(stdout);
      if(!nparam_)
        break;

      finished=false;
      budget_spent_=false;
      bound_stop_=false;
      c_int err;
      {
        CallGuard guard(this);
        err=solver(&TrustRegionSCF::cb_update, &TrustRegionSCF::cb_obj, (c_int) nparam_, &settings, nullptr);
        guard.rethrow();
      }
      if(err!=0 && !budget_spent_ && !bound_stop_)
        throw std::runtime_error("OpenTrustRegion solver failed with error code " + std::to_string((long long) err) + ".\n");
      if(budget_spent_) {
        printf("A macroiteration spent %i Hessian-vector products without accepting a step.\n", (int) hess_budget_);
        fflush(stdout);
        break;
      }
      if(pass >= max_restarts)
        break;

      // A step that took a free occupation to its bound lowered the
      // energy: restart from there
      if(bound_stop_) {
        std::vector<arma::mat> C;
        std::vector<arma::vec> occ;
        displaced(bound_step_.memptr(), 1.0, C, occ);
        adopt(C, occ);
        printf("\nA step filled or emptied an orbital, lowering the energy to % .10f; restarting there.\n", E_);
        fflush(stdout);
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
        displaced(best_step_.memptr(), 1.0, C, occ);
        adopt(C, occ);
        continue;
      }
      if(pattern_changed_) {
        printf("\nAn orbital filled or emptied; rebuilding the parametrization.\n");
        fflush(stdout);
        continue;
      }
      // The solve finished: check the KKT conditions
      finished=true;
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
  static c_int cb_update(const c_real * kappa, c_real * func, c_real * grad, c_real * h_diag, hess_x_fp * hess_x, void * context) {
    TrustRegionSCF * self=static_cast<TrustRegionSCF *>(context);
    try {
      self->take_step(kappa);
      std::copy(self->grad_.begin(), self->grad_.end(), grad);
      if(self->free_occ_) {
        std::copy(self->hdiag_.begin(), self->hdiag_.end(), h_diag);
      } else {
        const arma::vec h(self->hessian_diagonal(self->Fmo_));
        std::copy(h.begin(), h.end(), h_diag);
      }
      *func=self->E_;
      *hess_x=&TrustRegionSCF::cb_hess_x;
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  static c_int cb_obj(const c_real * kappa, c_real * func, void * context) {
    TrustRegionSCF * self=static_cast<TrustRegionSCF *>(context);
    try {
      std::vector<arma::mat> C, Fmo;
      std::vector<arma::vec> occ;
      double E;
      bool bound;
      self->displaced(kappa, 1.0, C, occ, &bound);
      self->evaluate(C, occ, E, Fmo);
      *func=E;
      if(self->free_occ_) {
        self->n_objective_++;
        if(E < self->best_energy_) {
          self->best_energy_=E;
          self->best_step_=arma::vec(kappa, self->nparam_);
        }
        // The step was cut short where a free occupation reached its
        // bound and still lowered the energy: stop the solve to restart
        // from there. (Failing here is the clean way out; optimize_free()
        // treats it as a stop, not an error.)
        if(bound && E < self->E_) {
          self->bound_stop_=true;
          self->bound_step_=arma::vec(kappa, self->nparam_);
          return 1;
        }
      }
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  static c_int cb_hess_x(const c_real * x, c_real * hx, void * context) {
    TrustRegionSCF * self=static_cast<TrustRegionSCF *>(context);
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
  static c_int cb_precond(const c_real * residual, const c_real * mu, c_real * out, void * context) {
    TrustRegionSCF * self=static_cast<TrustRegionSCF *>(context);
    try {
      self->precondition(residual, *mu, out);
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  /// Stops the solve when a free occupation has reached a bound under it
  static c_int cb_conv_check(c_bool * converged, void * context) {
    *converged=static_cast<TrustRegionSCF *>(context)->pattern_changed_;
    return 0;
  }
  static void cb_logger(const char * message, void *) {
    printf("%s\n", message);
    fflush(stdout);
  }
};

#endif
