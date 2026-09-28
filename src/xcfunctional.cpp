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

XCFunctional::XCFunctional(int func_id, bool polarized) : id(func_id), initialized(false) {
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
