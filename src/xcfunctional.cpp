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

#include "xcfunctional.h"
#include "dftfuncs.h"

#include <sstream>
#include <stdexcept>

XCFunctional::XCFunctional(int func_id, bool polarized) : id(func_id), initialized(false), polarized_(polarized) {
  // Determine the rung (reuse the dftfuncs classifier).
  is_gga_mgga(func_id, gga_, mgga_tau_, mgga_lapl_);

  const int nspin = polarized ? XC_POLARIZED : XC_UNPOLARIZED;
  if(xc_func_init(&func, func_id, nspin) != 0) {
    std::ostringstream oss;
    oss << "Functional " << func_id << " not found!";
    throw std::runtime_error(oss.str());
  }
  initialized = true;
}

XCFunctional::~XCFunctional() {
  if(initialized)
    xc_func_end(&func);
}

void XCFunctional::set_dens_threshold(double thr) {
  xc_func_set_dens_threshold(&func, thr);
}

size_t XCFunctional::n_ext_params() const {
  return xc_func_info_get_n_ext_params((xc_func_info_type *) func.info);
}

void XCFunctional::set_ext_params(const arma::vec & pars) {
  if(!pars.n_elem)
    return;
  const size_t npars = n_ext_params();
  if(npars != pars.n_elem) {
    std::ostringstream oss;
    oss << "Inconsistent number of parameters for functional " << id << ".\n";
    oss << "Expected " << npars << ", got " << pars.n_elem << ".\n";
    throw std::logic_error(oss.str());
  }
  xc_func_set_ext_params(&func, pars.memptr());
}

bool XCFunctional::has_exc() const {
  return xc_func_info_get_flags((xc_func_info_type *) func.info) & XC_FLAGS_HAVE_EXC;
}

bool XCFunctional::has_fxc() const {
  return xc_func_info_get_flags((xc_func_info_type *) func.info) & XC_FLAGS_HAVE_FXC;
}

bool XCFunctional::is_epc() const {
  // libxc gives EPC no structural marker -- match the known ids. The
  // EPC functionals came in libxc 7.
#ifdef XC_LDA_C_EPC17
  return id == XC_LDA_C_EPC17 || id == XC_LDA_C_EPC17_2 ||
         id == XC_LDA_C_EPC18_1 || id == XC_LDA_C_EPC18_2;
#else
  return false;
#endif
}

void XCFunctional::eval(size_t N, const double * rho, const double * sigma,
                        const double * lapl, const double * tau, bool pot,
                        double * exc, double * vrho, double * vsigma,
                        double * vlapl, double * vtau) const {
  const bool mgga = mgga_tau_ || mgga_lapl_;
  if(has_exc()) {
    if(pot) {
      if(mgga)
        xc_mgga_exc_vxc(&func, N, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau);
      else if(gga_)
        xc_gga_exc_vxc(&func, N, rho, sigma, exc, vrho, vsigma);
      else
        xc_lda_exc_vxc(&func, N, rho, exc, vrho);
    } else {
      if(mgga)
        xc_mgga_exc(&func, N, rho, sigma, lapl, tau, exc);
      else if(gga_)
        xc_gga_exc(&func, N, rho, sigma, exc);
      else
        xc_lda_exc(&func, N, rho, exc);
    }
  } else if(pot) {
    if(mgga)
      xc_mgga_vxc(&func, N, rho, sigma, lapl, tau, vrho, vsigma, vlapl, vtau);
    else if(gga_)
      xc_gga_vxc(&func, N, rho, sigma, vrho, vsigma);
    else
      xc_lda_vxc(&func, N, rho, vrho);
  }
}

void XCFunctional::eval_fxc(size_t N, const double * rho, const double * sigma,
                            const double * lapl, const double * tau,
                            std::map<std::string, arma::mat> & out) const {
  if(!has_fxc()) {
    std::ostringstream oss;
    oss << "Functional " << id << " provides no second derivatives in this libxc build.\n";
    throw std::runtime_error(oss.str());
  }

  // Number of spin components of each array: unpolarized arrays have
  // one; the polarized counts follow Libxc's packing of the symmetric
  // second derivatives.
  auto alloc = [&](const char * name, size_t upol, size_t pol) -> double * {
    arma::mat & m = out[name];
    m.zeros(polarized_ ? pol : upol, N);
    return m.memptr();
  };

  out.clear();
  const bool mgga = mgga_tau_ || mgga_lapl_;
  if(mgga) {
    double * v2rho2 = alloc("v2rho2", 1, 3);
    double * v2rhosigma = alloc("v2rhosigma", 1, 6);
    double * v2rholapl = alloc("v2rholapl", 1, 4);
    double * v2rhotau = alloc("v2rhotau", 1, 4);
    double * v2sigma2 = alloc("v2sigma2", 1, 6);
    double * v2sigmalapl = alloc("v2sigmalapl", 1, 6);
    double * v2sigmatau = alloc("v2sigmatau", 1, 6);
    double * v2lapl2 = alloc("v2lapl2", 1, 3);
    double * v2lapltau = alloc("v2lapltau", 1, 4);
    double * v2tau2 = alloc("v2tau2", 1, 3);
    xc_mgga_fxc(&func, N, rho, sigma, lapl, tau, v2rho2, v2rhosigma, v2rholapl,
                v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2);
    // Drop the arrays of the dependence the functional does not have
    if(!mgga_lapl_)
      for(const char * name : {"v2rholapl", "v2sigmalapl", "v2lapl2", "v2lapltau"})
        out.erase(name);
    if(!mgga_tau_)
      for(const char * name : {"v2rhotau", "v2sigmatau", "v2lapltau", "v2tau2"})
        out.erase(name);
  } else if(gga_) {
    double * v2rho2 = alloc("v2rho2", 1, 3);
    double * v2rhosigma = alloc("v2rhosigma", 1, 6);
    double * v2sigma2 = alloc("v2sigma2", 1, 6);
    xc_gga_fxc(&func, N, rho, sigma, v2rho2, v2rhosigma, v2sigma2);
  } else {
    xc_lda_fxc(&func, N, rho, alloc("v2rho2", 1, 3));
  }
}
