/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_JKBUILDER
#define ERKALE_JKBUILDER

#include "global.h"
#include "basis.h"
#include "eritable.h"
#include "eriscreen.h"
#include "density_fitting.h"

#include <armadillo>
#include <memory>
#include <string>
#include <vector>

class Settings;
/// Polymorphic J/K backend (one per integral engine); defined in jkbuilder.cpp.
class JKBackend;

/**
 * \class JKBuilder
 *
 * \brief Unified Coulomb / exchange (J/K) build driver.
 *
 * JKBuilder is a thin facade over a polymorphic \ref JKBackend chosen once at
 * \ref configure / \ref init time: the in-core four-index table (ERItable),
 * the direct four-index screener (ERIscreen) or the density-fitting / Cholesky
 * tensor (DensityFit), each with a range-separated twin. The build method,
 * the \c Direct modifier (on-the-fly integrals), \c DecFock (decontracted
 * direct build) and \c occ_rik (occ-RI-K exchange) select and configure the
 * backend; every J/K build, range-separated build and gradient then forwards
 * to it through this single interface, with no per-call method dispatch.
 *
 * The polymorphic backend hierarchy is an implementation detail living
 * entirely in jkbuilder.cpp; consumers only ever see this facade.
 */
class JKBuilder {
 public:
  /// Base J/K build method (orthogonal to the Direct modifier).
  enum class Method {
    FourIndex,        ///< Exact four-index ERIs (ERItable / ERIscreen)
    DensityFitting,   ///< RI with a Gaussian auxiliary basis
    CholeskyTwoStep,  ///< Two-step pivoted Cholesky (Folkestad et al)
    CDFit             ///< DF on a per-atom CD-derived auxiliary basis
  };

 private:
  /// Resolved build method
  Method method;
  /// The chosen backend (built in configure, filled in init). It owns the
  /// exact-exchange admixture (set via set_range_separation at fill time).
  std::unique_ptr<JKBackend> impl;

 public:
  JKBuilder();
  ~JKBuilder();

  /// Resolve the method, modifiers and thresholds from the settings and
  /// construct the backend (still empty; fill it with init).
  void configure(const Settings & set);
  /// Map a JKMethod string to a Method (static; used by configure and by
  /// consumers that want to validate the method before constructing).
  static Method resolve_method(const Settings & set);
  /// Human-readable method name.
  static std::string method_name(Method m);
  /// Override the density-fitting auxiliary basis (before init).
  void set_fitting(const BasisSet & fitbas);
  /// Build the integral engine(s) for the chosen method.
  void init(const BasisSet & basis, bool verbose);
  /// Lazily build the range-separated (short-range) integrals.
  void init_rs(double omega);

  bool is_densityfit() const;
  bool is_cholesky() const;
  bool is_direct() const;
  bool is_decfock() const;
  bool is_occ_rik() const;

  /// Density-fitted / Cholesky Coulomb matrix from the density P.
  arma::mat   calcJ(const arma::mat & P) const;
  /// Density-fitted / Cholesky exchange from the occupied orbitals C
  /// (occupations occ; S the orbital-basis overlap for occ-RI-K). Real and
  /// complex orbitals are both supported.
  arma::mat   calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const;
  arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const;
  /// Same, for the range-separated (short-range) exchange operator.
  arma::mat   calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const;
  arma::cx_mat calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const;

  /// Configure the exact-exchange admixture used by formJK and build the
  /// range-separated (short-range) integrals when omega != 0. Call once at fill
  /// time (SCF::set_range_separation): kfull / kshort are the full- and
  /// short-range exact-exchange fractions, omega the range-separation
  /// parameter. Defaults to Hartree-Fock (kfull=1, kshort=0, omega=0).
  void set_range_separation(double kfull, double kshort, double omega);
  /// Whether any exact exchange is present (kfull or kshort nonzero).
  bool has_exact_exchange() const;

  /// Unified Coulomb + exact-exchange build (restricted / unrestricted, real /
  /// complex). J is always the full Coulomb operator, built from the real total
  /// density Ptot. K is the combined exact exchange
  /// kfull*K_full + kshort*K_short(omega) for the admixture configured by
  /// set_range_separation (K is zero when there is no exact exchange).
  void formJK(const arma::mat & Ptot, const arma::mat & C, const std::vector<double> & occ,
              const arma::mat & S, arma::mat & J, arma::mat & K) const;
  void formJK(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat & cC,
              const std::vector<double> & occ, const arma::mat & S,
              arma::mat & J, arma::cx_mat & K) const;
  void formJK(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
              const arma::mat & Ca, const arma::mat & Cb,
              const std::vector<double> & occa, const std::vector<double> & occb,
              const arma::mat & S, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const;
  void formJK(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
              const arma::cx_mat & cCa, const arma::cx_mat & cCb,
              const std::vector<double> & occa, const std::vector<double> & occb,
              const arma::mat & S, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const;

  /// Coulomb + exact-exchange contribution to the force gradient: J always,
  /// plus kfull-scaled exchange, plus kshort-scaled short-range exchange when
  /// omega != 0. Restricted uses the total density P with orbitals C;
  /// unrestricted uses the total density Ptot for J and the spin channels for
  /// exchange. (For HF pass kfull=1, kshort=0, omega=0.)
  arma::vec formForce(const arma::mat & P, const arma::mat & C, const std::vector<double> & occ,
                      double kfull, double kshort, double omega, double tol) const;
  arma::vec formForce(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                      const arma::mat & Ca, const arma::mat & Cb,
                      const std::vector<double> & occa, const std::vector<double> & occb,
                      double kfull, double kshort, double omega, double tol) const;

  /// Engine accessors. Transitional: the force / PZ-SIC / contrib code reaches
  /// the owned engine through these (the backend that does not own the
  /// requested engine throws). They are removed as those call sites move to
  /// dedicated virtual builds.
  ERItable & eritable() const;
  ERItable & eritable_rs() const;
  ERIscreen & eriscreen() const;
  ERIscreen & eriscreen_rs() const;
  DensityFit & densityfit() const;
  DensityFit & densityfit_rs() const;
};

#endif
