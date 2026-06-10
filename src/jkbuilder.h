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
