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
#include <cstdio>
#include <exception>
#include <functional>
#include <stdexcept>
#include <vector>

/**
 * Second-order trust-region orbital optimization and stability analysis
 * with OpenTrustRegion (J. Greiner, I.-M. Hoyvik, S. Lehtola, and
 * J. J. Eriksen, J. Chem. Theory Comput. 22, 881 (2026)) for the
 * programs that build their Fock matrices for OpenOrbitalOptimizer
 * (erkale_neo, erkale_complex_orbs).
 *
 * The orbitals are given in blocks (particle types, spins, symmetries),
 * each in its own orthonormal basis, with fixed occupations. The
 * parameters are the rotations between orbitals of different
 * occupation within each block, C_b -> C_b exp(K_b), so the block
 * structure (and thereby the symmetry) is preserved. The energy and the
 * Fock matrices come from the program's Fock builder. The gradient is
 * g_ai = 2 (n_i - n_a) F_ai in the orbital basis; Hessian-vector
 * products are central differences of the gradient.
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

  /// Constructor: Fock builder, the starting orbitals and occupations, and
  /// optionally the Fock response for analytic Hessian-vector products
  /// (without it, the products are finite differences of the gradient)
  TrustRegionSCF(const FockBuilder & fock, const DensityMatrix & dm, const ResponseBuilder & response=ResponseBuilder(), double fdstep=1e-4) : fock_(fock), response_(response), C_(dm.first), occ_(dm.second), fdstep_(fdstep) {
    // Rotations between orbitals of different occupation
    occupied_.resize(C_.size());
    for(size_t b=0;b<C_.size();b++) {
      std::vector<arma::uword> occd;
      for(size_t i=0;i<occ_[b].n_elem;i++) {
        bool is_occupied=false;
        for(size_t a=0;a<occ_[b].n_elem;a++)
          if(occ_[b](i)-occ_[b](a) > 1e-10) {
            pairs_.push_back({b, a, i, occ_[b](i)-occ_[b](a)});
            is_occupied=true;
          }
        if(is_occupied)
          occd.push_back(i);
      }
      occupied_[b]=arma::uvec(occd);
    }
    evaluate(C_, E_, Fmo_);
    grad_=gradient(Fmo_);
  }

  /// Number of parameters
  size_t n_param() const { return pairs_.size(); }
  /// Current energy
  double energy() const { return E_; }
  /// Current orbitals and occupations
  DensityMatrix density_matrix() const { return std::make_pair(C_, occ_); }

  /**
   * Minimize the energy. With stability, OpenTrustRegion checks the
   * stability of the converged solution and follows any instability to
   * a lower solution. convthr is the convergence threshold for the
   * root-mean-square gradient.
   */
  void optimize(double convthr, bool stability, int verbose) {
    if(!n_param())
      return;
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
  }

  /**
   * Stability analysis of the current solution: is the lowest eigenvalue
   * of the orbital Hessian positive? Only rotations that preserve the
   * block structure are examined.
   */
  bool is_stable(double convthr, int verbose) {
    if(!n_param())
      return true;
    stability_settings_type settings=stability_settings_init();
    settings.logger=&TrustRegionSCF::cb_logger;
    settings.conv_tol=convthr;
    settings.verbose=verbose;
    arma::vec h_diag(hessian_diagonal(Fmo_));
    arma::vec kappa(n_param(), arma::fill::zeros);
    c_bool stable=true;
    Activate guard(this);
    const c_int err=stability_check(h_diag.memptr(), &TrustRegionSCF::cb_hess_x, (c_int) n_param(), &stable, settings, kappa.memptr());
    guard.rethrow();
    if(err!=0)
      throw std::runtime_error("OpenTrustRegion stability check failed with error code " + std::to_string((long long) err) + ".\n");
    return stable;
  }

 private:
  /// A rotation between orbitals a and i of block b, n_i - n_a = dn > 0
  struct pair_t {
    size_t b, a, i;
    double dn;
  };

  FockBuilder fock_;
  /// Fock response (empty: finite-difference Hessian-vector products)
  ResponseBuilder response_;
  /// Orbitals and occupations by block
  std::vector<arma::mat> C_;
  std::vector<arma::vec> occ_;
  std::vector<pair_t> pairs_;
  /// Orbitals that lose occupation in some rotation, by block
  std::vector<arma::uvec> occupied_;
  /// Finite-difference step for the Hessian-vector products
  double fdstep_;
  /// Energy, orbital-basis Fock matrices, and gradient at the current orbitals
  double E_;
  std::vector<arma::mat> Fmo_;
  arma::vec grad_;
  /// Exception thrown inside a callback, rethrown after the solver returns
  std::exception_ptr error_;

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
  /// Energy and orbital-basis Fock matrices for the orbitals C
  void evaluate(const std::vector<arma::mat> & C, double & E, std::vector<arma::mat> & Fmo) const {
    const auto ret=fock_(std::make_pair(C, occ_));
    E=ret.first;
    Fmo.resize(C.size());
    for(size_t b=0;b<C.size();b++)
      Fmo[b]=C[b].t()*ret.second[b]*C[b];
  }
  /// Gradient from the orbital-basis Fock matrices
  arma::vec gradient(const std::vector<arma::mat> & Fmo) const {
    arma::vec g(pairs_.size());
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      g(k)=2.0*p.dn*Fmo[p.b](p.a,p.i);
    }
    return g;
  }
  /// Approximate diagonal of the Hessian
  arma::vec hessian_diagonal(const std::vector<arma::mat> & Fmo) const {
    arma::vec h(pairs_.size());
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      h(k)=2.0*p.dn*(Fmo[p.b](p.a,p.a)-Fmo[p.b](p.i,p.i));
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
   * The transition density D = C M C^T is passed to the response as
   * L R^T + R L^T, with L the orbitals that lose occupation: M only has
   * rows and columns of those orbitals.
   */
  arma::vec hessian_times_analytic(const double * x) const {
    std::vector<arma::mat> X(generators(x));
    std::vector<arma::mat> M(C_.size()), L(C_.size()), R(C_.size());
    for(size_t b=0;b<C_.size();b++) {
      const arma::mat N(arma::diagmat(occ_[b]));
      M[b]=X[b]*N-N*X[b];
      const arma::uvec & I=occupied_[b];
      arma::mat MI(M[b].cols(I));
      MI.rows(I)-=0.5*M[b](I,I);
      L[b]=C_[b].cols(I);
      R[b]=C_[b]*MI;
    }
    const std::vector<arma::mat> dF(response_(std::make_pair(C_, occ_), L, R));

    arma::vec hx(pairs_.size());
    for(size_t k=0;k<pairs_.size();k++) {
      const pair_t & p=pairs_[k];
      const arma::mat & F=Fmo_[p.b];
      const arma::mat & Xb=X[p.b];
      const arma::mat & Mb=M[p.b];
      const double FX=arma::dot(F.row(p.a), Xb.col(p.i))-arma::dot(Xb.row(p.a), F.col(p.i));
      const double FM=arma::dot(F.row(p.a), Mb.col(p.i))-arma::dot(Mb.row(p.a), F.col(p.i));
      const double G=arma::dot(C_[p.b].col(p.a), dF[p.b]*C_[p.b].col(p.i));
      hx(k)=p.dn*FX+FM+2.0*p.dn*G;
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
   * convergence and makes the product that of the symmetric Hessian.
   */
  arma::vec hessian_times_fd(const double * x) const {
    double E;
    std::vector<arma::mat> Fp, Fm;
    evaluate(rotated(x, fdstep_), E, Fp);
    evaluate(rotated(x, -fdstep_), E, Fm);
    arma::vec hx((gradient(Fp)-gradient(Fm))/(2.0*fdstep_));

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

  /* OpenTrustRegion callbacks. Exceptions must not cross the Fortran
     library: they are stored and rethrown after it returns. */
  static c_int cb_update(const c_real * kappa, c_real * func, c_real * grad, c_real * h_diag, hess_x_fp * hess_x) {
    TrustRegionSCF * self=active();
    try {
      self->C_=self->rotated(kappa);
      self->evaluate(self->C_, self->E_, self->Fmo_);
      self->grad_=self->gradient(self->Fmo_);
      const arma::vec h(self->hessian_diagonal(self->Fmo_));
      *func=self->E_;
      for(size_t k=0;k<self->pairs_.size();k++) {
        grad[k]=self->grad_(k);
        h_diag[k]=h(k);
      }
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
      std::vector<arma::mat> Fmo;
      double E;
      self->evaluate(self->rotated(kappa), E, Fmo);
      *func=E;
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  static c_int cb_hess_x(const c_real * x, c_real * hx) {
    TrustRegionSCF * self=active();
    try {
      const arma::vec h(self->hessian_times(x));
      for(size_t k=0;k<self->pairs_.size();k++)
        hx[k]=h(k);
      return 0;
    } catch(...) {
      self->error_=std::current_exception();
      return 1;
    }
  }
  static void cb_logger(const char * message) {
    printf("%s\n", message);
    fflush(stdout);
  }
};

#endif
