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
#include <string>
#include <vector>

class Settings;

/**
 * \class JKBuilder
 *
 * \brief Unified Coulomb / exchange (J/K) build driver.
 *
 * Owns the integral engines -- the in-core four-index table (ERItable),
 * the direct four-index screener (ERIscreen) and the density-fitting /
 * Cholesky tensor (DensityFit), each with a range-separated twin -- and
 * the resolved build method, so the SCF no longer juggles the
 * DensityFitting / Cholesky / CholeskyAlgorithm / Direct / DecFock
 * booleans or a three-way Fock-dispatch. The method is chosen once
 * (configure) and the J/K builds, range-separated builds and gradients
 * dispatch through a single interface.
 *
 * The four base methods are orthogonal to the \c Direct modifier, which
 * selects on-the-fly integrals (ERIscreen vs ERItable for four-index;
 * cached vs recomputed blocks for DF/CD), and to \c occ_rik, which
 * switches density-fitted exchange to the occ-RI-K algorithm.
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
  /// On-the-fly integrals (orthogonal modifier)
  bool direct;
  /// Decontracted basis for direct four-index Fock builds
  bool decfock;
  /// occ-RI-K density-fitted exchange
  bool occ_rik;
  /// Verbose output (set in init)
  bool verbose;

  /// Integral screening threshold
  double intthr;
  /// DF linear-dependence threshold
  double fitthr;
  /// DF pivoted-Cholesky threshold
  double fitcholthr;
  /// Cholesky decomposition threshold
  double cholthr;
  /// Cholesky cache (shell) threshold
  double cholshthr;
  /// Cholesky integral-cache mode
  int cholmode;
  /// Cholesky integral-cache filename
  std::string cholfile;
  /// Density-fitting auxiliary basis name (or "Auto")
  std::string fittingbasis;

  /// Integral engines. Marked mutable: they are computational caches
  /// (integral tables / fitted tensors), so the const Fock and force
  /// builds reach them through the const accessors below, and the
  /// range-separated twins are lazily filled at build time.
  /// In-core four-index table
  mutable ERItable tab;
  /// In-core four-index table, range separation
  mutable ERItable tab_rs;
  /// Direct four-index screener
  mutable ERIscreen scr;
  /// Direct four-index screener, range separation
  mutable ERIscreen scr_rs;
  /// Density-fitting / Cholesky tensor
  mutable DensityFit dfit;
  /// Density-fitting / Cholesky tensor, range separation
  mutable DensityFit dfit_rs;

  /// Density-fitting auxiliary basis
  BasisSet dfitbas;
  /// Decontracted basis set (for decfock direct four-index)
  BasisSet decbas;
  /// Contracted->decontracted conversion matrix
  arma::mat decconv;

  /// Orbital basis (not owned)
  const BasisSet * basisp;

  /// True for the DF/CD methods that route through DensityFit.
  bool uses_dfit() const;

  /// decfock: map a contracted density to the decontracted basis (D P D^T).
  arma::mat    decontract(const arma::mat & P) const;
  arma::cx_mat decontract(const arma::cx_mat & P) const;
  /// decfock: map a decontracted matrix back to the contracted basis (D^T M D).
  arma::mat    recontract(const arma::mat & M) const;
  arma::cx_mat recontract(const arma::cx_mat & M) const;

 public:
  JKBuilder();
  ~JKBuilder();

  /// Resolve the method, modifiers and thresholds from the settings.
  void configure(const Settings & set);
  /// Resolve just the build method from the settings. Exposed so the
  /// non-SCF consumers (moints, complex_orbs, neo) share one mapping.
  static Method resolve_method(const Settings & set);
  /// Human-readable method name (for messages).
  static std::string method_name(Method m);

  /// Override the density-fitting basis (used by callers that build
  /// their own auxiliary basis before init()).
  void set_fitting(const BasisSet & fitbas);
  /// The density-fitting auxiliary basis.
  const BasisSet & get_fitting() const { return dfitbas; }

  /// Build the full-range engines for the orbital basis.
  void init(const BasisSet & basis, bool verbose);
  /// Build the range-separated engines for screening parameter omega.
  void init_rs(double omega);

  /// Resolved build method.
  Method get_method() const { return method; }
  /// True for DF / CD / CDFit (routes through DensityFit).
  bool is_densityfit() const { return uses_dfit(); }
  /// True iff the DensityFit tensor holds two-step CD vectors.
  bool is_cholesky() const;
  /// On-the-fly integral modifier.
  bool is_direct() const { return direct; }
  /// Decontracted-basis direct Fock build.
  bool is_decfock() const { return decfock; }
  /// occ-RI-K exchange selected.
  bool is_occ_rik() const { return occ_rik; }

  /// Contracted->decontracted conversion matrix (decfock).
  const arma::mat & decontraction() const { return decconv; }
  /// Decontracted basis (decfock).
  const BasisSet & decontracted_basis() const { return decbas; }

  /// Density-fitted / Cholesky Coulomb matrix from the density P.
  /// Symmetric with calcK: the builder owns the engine, so callers
  /// build J through it rather than reaching into densityfit().
  arma::mat   calcJ(const arma::mat & P) const;

  /// Density-fitted / Cholesky exchange matrix from the occupied
  /// orbitals C (occupations occ), for the full-range operator. Routes
  /// to occ-RI-K (DensityFit::calcK_occ) when OccRIK is set, otherwise
  /// to conventional RI-K (DensityFit::calcK). S is the orbital-basis
  /// overlap, needed by occ-RI-K's reconstruction. Real and complex
  /// (PZ-SIC / complex-orbital) orbitals are both supported.
  arma::mat   calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const;
  arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const;
  /// Same, for the range-separated (short-range) exchange operator.
  arma::mat   calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const;
  arma::cx_mat calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const;

  /// Unified full-range Coulomb + exchange build. Dispatches the resolved
  /// method: DF/CD (calcJ + orbital-based calcK above), direct four-index
  /// (single-pass ERIscreen::calcJK, with decfock decontraction handled
  /// internally) or in-core four-index (ERItable). J is always real --
  /// built from the real total density Ptot, since the imaginary density
  /// does not contribute to the Coulomb term -- and K is gated by want_K
  /// (false for pure DFT). The complex overloads return a complex K for the
  /// caller to split into real/imaginary parts.
  ///
  /// Restricted, real / complex density:
  void formJK(const arma::mat & Ptot, const arma::mat & C, const std::vector<double> & occ,
              const arma::mat & S, bool want_K, arma::mat & J, arma::mat & K) const;
  void formJK(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat & cC,
              const std::vector<double> & occ, const arma::mat & S, bool want_K,
              arma::mat & J, arma::cx_mat & K) const;
  /// Unrestricted, real / complex density (J from Ptot; Ka, Kb per channel):
  void formJK(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
              const arma::mat & Ca, const arma::mat & Cb,
              const std::vector<double> & occa, const std::vector<double> & occb,
              const arma::mat & S, bool want_K, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const;
  void formJK(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
              const arma::cx_mat & cCa, const arma::cx_mat & cCb,
              const std::vector<double> & occa, const std::vector<double> & occb,
              const arma::mat & S, bool want_K, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const;

  /// Short-range (range-separated) exchange only; J is unaffected by the
  /// range split. Same engine dispatch as formJK's K path, via the _rs twins.
  arma::mat   formKshort(const arma::mat & P, const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const;
  arma::cx_mat formKshort(const arma::cx_mat & cP, const arma::cx_mat & cC, const std::vector<double> & occ, const arma::mat & S) const;
  void formKshort(const arma::mat & Pa, const arma::mat & Pb, const arma::mat & Ca, const arma::mat & Cb,
                  const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                  arma::mat & Ka, arma::mat & Kb) const;
  void formKshort(const arma::cx_mat & cPa, const arma::cx_mat & cPb, const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                  const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                  arma::cx_mat & Ka, arma::cx_mat & Kb) const;

  /// Engine accessors. Transitional: the Fock / force dispatch reaches
  /// the owned engines through these until the build/force methods
  /// below subsume every call site.
  ERItable & eritable() const { return tab; }
  ERItable & eritable_rs() const { return tab_rs; }
  ERIscreen & eriscreen() const { return scr; }
  ERIscreen & eriscreen_rs() const { return scr_rs; }
  DensityFit & densityfit() const { return dfit; }
  DensityFit & densityfit_rs() const { return dfit_rs; }
};

#endif
