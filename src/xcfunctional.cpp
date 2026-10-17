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

namespace {
  /// The array dimensions Libxc gives a meta-GGA with this spin treatment
  xc_dimensions metagga_dimensions(bool polarized) {
    xc_func_type f;
    if(xc_func_init(&f, XC_MGGA_X_TPSS, polarized ? XC_POLARIZED : XC_UNPOLARIZED) != 0)
      throw std::runtime_error("Could not initialize a reference meta-GGA for the array dimensions.\n");
    const xc_dimensions d(f.dim);
    xc_func_end(&f);
    return d;
  }
}

const xc_dimensions & XCFunctional::dimensions(bool polarized) {
  static const xc_dimensions unpolarized(metagga_dimensions(false)), spinpolarized(metagga_dimensions(true));
  return polarized ? spinpolarized : unpolarized;
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

namespace {
  /// Energy density of the functional f, of any family
  arma::vec exc_of(const xc_func_type * f, size_t N, const double * rho, const double * sigma,
                   const double * lapl, const double * tau) {
    arma::vec zk(N, arma::fill::zeros);
    switch(f->info->family) {
    case XC_FAMILY_LDA:
    case XC_FAMILY_HYB_LDA:
      xc_lda_exc(f, N, rho, zk.memptr());
      break;
    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
      xc_gga_exc(f, N, rho, sigma, zk.memptr());
      break;
    case XC_FAMILY_MGGA:
    case XC_FAMILY_HYB_MGGA:
      xc_mgga_exc(f, N, rho, sigma, lapl, tau, zk.memptr());
      break;
    default:
      throw std::logic_error("Unknown functional family.\n");
    }
    return zk;
  }

  /**
   * Add the energy density of f times weight to the column of e for its
   * kind, and return the energy density. Libxc evaluates a functional's
   * own term, if any, and adds its mix of components; the components are
   * split in turn, and the own term is what remains of the energy density.
   */
  arma::vec split_exc(const xc_func_type * f, double weight, size_t N, const double * rho, const double * sigma,
                      const double * lapl, const double * tau, arma::mat & e) {
    const int col = f->info->kind == XC_EXCHANGE ? 0 : (f->info->kind == XC_CORRELATION ? 1 : 2);
    if(f->mix_coef == NULL) {
      const arma::vec zk(exc_of(f, N, rho, sigma, lapl, tau));
      e.col(col) += weight*zk;
      return zk;
    }
    arma::vec mix(N, arma::fill::zeros);
    for(int i=0; i<f->n_func_aux; i++)
      mix += f->mix_coef[i]*split_exc(f->func_aux[i], weight*f->mix_coef[i], N, rho, sigma, lapl, tau, e);
    const bool own = f->info->lda != NULL || f->info->gga != NULL || f->info->mgga != NULL;
    if(!own)
      return mix;
    const arma::vec zk(exc_of(f, N, rho, sigma, lapl, tau));
    e.col(col) += weight*(zk - mix);
    return zk;
  }
}

arma::mat XCFunctional::exc_split(size_t N, const double * rho, const double * sigma,
                                  const double * lapl, const double * tau) const {
  arma::mat e(N, 3, arma::fill::zeros);
  split_exc(&func, 1.0, N, rho, sigma, lapl, tau, e);
  return e;
}

void XCFunctional::eval_fxc(size_t N, const double * rho, const double * sigma,
                            const double * lapl, const double * tau,
                            std::map<std::string, arma::mat> & out) const {
  if(!has_fxc()) {
    std::ostringstream oss;
    oss << "Functional " << id << " provides no second derivatives in this libxc build.\n";
    throw std::runtime_error(oss.str());
  }

  // Libxc's second-derivative arrays of the functional's family, with
  // the numbers of components per point that Libxc gives for the spin
  // treatment. A meta-GGA has them all; those of a variable the
  // functional does not depend on stay zero.
  const xc_dimensions & dim(func.dim);
  out.clear();
  auto alloc = [&](const char * name, int ncomp) -> double * {
    arma::mat & m = out[name];
    m.zeros(ncomp, N);
    return m.memptr();
  };

  if(mgga_tau_ || mgga_lapl_) {
    double * v2rho2 = alloc("v2rho2", dim.v2rho2);
    double * v2rhosigma = alloc("v2rhosigma", dim.v2rhosigma);
    double * v2rholapl = alloc("v2rholapl", dim.v2rholapl);
    double * v2rhotau = alloc("v2rhotau", dim.v2rhotau);
    double * v2sigma2 = alloc("v2sigma2", dim.v2sigma2);
    double * v2sigmalapl = alloc("v2sigmalapl", dim.v2sigmalapl);
    double * v2sigmatau = alloc("v2sigmatau", dim.v2sigmatau);
    double * v2lapl2 = alloc("v2lapl2", dim.v2lapl2);
    double * v2lapltau = alloc("v2lapltau", dim.v2lapltau);
    double * v2tau2 = alloc("v2tau2", dim.v2tau2);
    xc_mgga_fxc(&func, N, rho, sigma, lapl, tau, v2rho2, v2rhosigma, v2rholapl,
                v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2);
  } else if(gga_) {
    double * v2rho2 = alloc("v2rho2", dim.v2rho2);
    double * v2rhosigma = alloc("v2rhosigma", dim.v2rhosigma);
    double * v2sigma2 = alloc("v2sigma2", dim.v2sigma2);
    xc_gga_fxc(&func, N, rho, sigma, v2rho2, v2rhosigma, v2sigma2);
  } else {
    xc_lda_fxc(&func, N, rho, alloc("v2rho2", dim.v2rho2));
  }
}
