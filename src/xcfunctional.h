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

#ifndef ERKALE_XCFUNCTIONAL
#define ERKALE_XCFUNCTIONAL

#include <armadillo>
#include <cstddef>
#include <map>
#include <string>

extern "C" {
#include <xc.h>
// xc.h does not pull in the functional-id macros (XC_LDA_C_EPC17, ...).
#include <xc_funcs.h>
}

/**
 * \class XCFunctional
 *
 * \brief RAII wrapper around a single libxc functional.
 *
 * Owns the lifetime of one xc_func_type (init in the constructor, end in
 * the destructor), the density threshold and the external-parameter
 * overrides, and exposes its rung (LDA / GGA / meta-GGA) and a single
 * energy-density / potential evaluation. It encapsulates the raw libxc
 * plumbing that used to be inlined in AngularGrid::compute_xc.
 *
 * libxc gives the electron-proton correlation (EPC) functionals no
 * structural marker -- each is an ordinary nspin=2 LDA_C functional whose
 * two density channels mean (electron, proton) rather than (up, down) --
 * so is_epc() tags them for multicomponent (NEO) callers.
 */
class XCFunctional {
  /// libxc functional handle
  xc_func_type func;
  /// Functional id
  int id;
  /// Has the handle been initialised? (guards the destructor)
  bool initialized;
  /// Rung: plain GGA
  bool gga_;
  /// Rung: meta-GGA using the kinetic energy density tau
  bool mgga_tau_;
  /// Rung: meta-GGA using the density laplacian
  bool mgga_lapl_;

 public:
  /// Initialise functional func_id, polarized (nspin=2) or not.
  XCFunctional(int func_id, bool polarized);
  ~XCFunctional();
  /// Non-copyable: owns a libxc resource.
  XCFunctional(const XCFunctional &) = delete;
  XCFunctional & operator=(const XCFunctional &) = delete;

  /// Set the density threshold below which the functional is screened.
  void set_dens_threshold(double thr);
  /// Number of external parameters the functional expects.
  size_t n_ext_params() const;
  /// Override the functional's external parameters (e.g. DFTXpars /
  /// DFTCpars). A no-op for an empty vector; throws on a count mismatch.
  void set_ext_params(const arma::vec & pars);

  /// Plain GGA (or higher) reduced-gradient dependence.
  bool is_gga() const { return gga_; }
  /// Meta-GGA tau dependence.
  bool is_mgga_tau() const { return mgga_tau_; }
  /// Meta-GGA laplacian dependence.
  bool is_mgga_lapl() const { return mgga_lapl_; }
  /// Any meta-GGA dependence.
  bool is_mgga() const { return mgga_tau_ || mgga_lapl_; }
  /// Does the functional provide an energy density?
  bool has_exc() const;
  /// Does the functional provide second derivatives?
  bool has_fxc() const;
  /// Libxc's array dimensions (components per point) of this functional,
  /// for its family and spin treatment
  const xc_dimensions & dims() const { return func.dim; }
  /// Libxc's array dimensions for the spin treatment, of a meta-GGA,
  /// whose arrays cover every variable
  static const xc_dimensions & dimensions(bool polarized);
  /// Electron-proton correlation functional (LDA_C_EPC17/17_2/18_1/18_2)?
  bool is_epc() const;
  /// libxc functional id.
  int get_id() const { return id; }

  /**
   * Evaluate the energy density (if has_exc()) and, when pot is set, the
   * potential. Output pointers that are not needed for the functional's
   * rung / for the requested quantities should be passed as NULL by the
   * caller (matching the conditioning of lapl/tau/vlapl/vtau on the
   * meta-GGA flags). Inputs and outputs are libxc's flat layouts.
   */
  void eval(size_t N, const double * rho, const double * sigma,
            const double * lapl, const double * tau, bool pot,
            double * exc, double * vrho, double * vsigma,
            double * vlapl, double * vtau) const;
  /**
   * Evaluate the second derivatives. out receives Libxc's arrays of the
   * functional's family (all ten for a meta-GGA, zero in a variable the
   * functional does not depend on), keyed by their Libxc names (v2rho2,
   * v2rhosigma, ...), each stored ncomp x N with the spin component
   * fastest and ncomp as Libxc's dimensions give it: Libxc's flat
   * layout. Throws if the functional has no fxc.
   */
  void eval_fxc(size_t N, const double * rho, const double * sigma,
                const double * lapl, const double * tau,
                std::map<std::string, arma::mat> & out) const;
};

#endif
