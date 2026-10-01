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
#include "xckernel_dispatch.h"
#include "xckernel.h"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
  typedef int (*mat_fn_t)(int64_t, int64_t, const double *, const double * const *, double *);
  typedef int (*g1_fn_t)(int64_t, int64_t, const double *, const double *, const double * const *, double *);
  typedef int (*gg_fn_t)(int64_t, const double * const *, double *);

  template<typename F> struct kernel_t {
    const char * name;
    F fn;
    const char ** scal_names;
    const int * n_scal;
    const int * chi_order;
  };

  // The tables list every kernel of the vendored header, by signature
#define XCK_ENTRY(k) {#k, k, k##_scal_names, &k##_n_scal, &k##_chi_order}
  const kernel_t<mat_fn_t> mat_kernels[] = {
    XCK_ENTRY(xck_lda_r_o1),
    XCK_ENTRY(xck_lda_r_o2),
    XCK_ENTRY(xck_lda_r_o1_diag),
    XCK_ENTRY(xck_lda_ua_o1_diag),
    XCK_ENTRY(xck_lda_ub_o1_diag),
    XCK_ENTRY(xck_lda_ua_o1),
    XCK_ENTRY(xck_lda_ua_o2),
    XCK_ENTRY(xck_lda_ub_o1),
    XCK_ENTRY(xck_lda_ub_o2),
    XCK_ENTRY(xck_lda_st_o2_p),
    XCK_ENTRY(xck_lda_st_o2_m),
    XCK_ENTRY(xck_gga_r_o1),
    XCK_ENTRY(xck_gga_r_o2),
    XCK_ENTRY(xck_gga_r_o1_diag),
    XCK_ENTRY(xck_gga_ua_o1_diag),
    XCK_ENTRY(xck_gga_ub_o1_diag),
    XCK_ENTRY(xck_gga_ua_o1),
    XCK_ENTRY(xck_gga_ua_o2),
    XCK_ENTRY(xck_gga_ub_o1),
    XCK_ENTRY(xck_gga_ub_o2),
    XCK_ENTRY(xck_gga_st_o2_p),
    XCK_ENTRY(xck_gga_st_o2_m),
    XCK_ENTRY(xck_mgga_tau_r_o1),
    XCK_ENTRY(xck_mgga_tau_r_o2),
    XCK_ENTRY(xck_mgga_tau_r_o1_diag),
    XCK_ENTRY(xck_mgga_tau_ua_o1_diag),
    XCK_ENTRY(xck_mgga_tau_ub_o1_diag),
    XCK_ENTRY(xck_mgga_tau_ua_o1),
    XCK_ENTRY(xck_mgga_tau_ua_o2),
    XCK_ENTRY(xck_mgga_tau_ub_o1),
    XCK_ENTRY(xck_mgga_tau_ub_o2),
    XCK_ENTRY(xck_mgga_tau_st_o2_p),
    XCK_ENTRY(xck_mgga_tau_st_o2_m),
    XCK_ENTRY(xck_mgga_lapl_r_o1),
    XCK_ENTRY(xck_mgga_lapl_r_o2),
    XCK_ENTRY(xck_mgga_lapl_r_o1_diag),
    XCK_ENTRY(xck_mgga_lapl_ua_o1_diag),
    XCK_ENTRY(xck_mgga_lapl_ub_o1_diag),
    XCK_ENTRY(xck_mgga_lapl_ua_o1),
    XCK_ENTRY(xck_mgga_lapl_ua_o2),
    XCK_ENTRY(xck_mgga_lapl_ub_o1),
    XCK_ENTRY(xck_mgga_lapl_ub_o2),
    XCK_ENTRY(xck_mgga_lapl_st_o2_p),
    XCK_ENTRY(xck_mgga_lapl_st_o2_m),
    XCK_ENTRY(xck_mgga_r_o1),
    XCK_ENTRY(xck_mgga_r_o2),
    XCK_ENTRY(xck_mgga_r_o1_diag),
    XCK_ENTRY(xck_mgga_ua_o1_diag),
    XCK_ENTRY(xck_mgga_ub_o1_diag),
    XCK_ENTRY(xck_mgga_ua_o1),
    XCK_ENTRY(xck_mgga_ua_o2),
    XCK_ENTRY(xck_mgga_ub_o1),
    XCK_ENTRY(xck_mgga_ub_o2),
    XCK_ENTRY(xck_mgga_st_o2_p),
    XCK_ENTRY(xck_mgga_st_o2_m),
    XCK_ENTRY(xck_cmgga_tau_r_o1),
    XCK_ENTRY(xck_cmgga_tau_r_o2),
    XCK_ENTRY(xck_cmgga_tau_r_o1_diag),
    XCK_ENTRY(xck_cmgga_tau_ua_o1_diag),
    XCK_ENTRY(xck_cmgga_tau_ub_o1_diag),
    XCK_ENTRY(xck_cmgga_tau_ua_o1),
    XCK_ENTRY(xck_cmgga_tau_ua_o2),
    XCK_ENTRY(xck_cmgga_tau_ub_o1),
    XCK_ENTRY(xck_cmgga_tau_ub_o2),
    XCK_ENTRY(xck_cmgga_tau_st_o2_p),
    XCK_ENTRY(xck_cmgga_tau_st_o2_m),
  };
  const kernel_t<g1_fn_t> g1_kernels[] = {
    XCK_ENTRY(xck_lda_r_g1),
    XCK_ENTRY(xck_lda_ua_g1),
    XCK_ENTRY(xck_lda_ub_g1),
    XCK_ENTRY(xck_gga_r_g1),
    XCK_ENTRY(xck_gga_ua_g1),
    XCK_ENTRY(xck_gga_ub_g1),
    XCK_ENTRY(xck_mgga_tau_r_g1),
    XCK_ENTRY(xck_mgga_tau_ua_g1),
    XCK_ENTRY(xck_mgga_tau_ub_g1),
    XCK_ENTRY(xck_mgga_lapl_r_g1),
    XCK_ENTRY(xck_mgga_lapl_ua_g1),
    XCK_ENTRY(xck_mgga_lapl_ub_g1),
    XCK_ENTRY(xck_mgga_r_g1),
    XCK_ENTRY(xck_mgga_ua_g1),
    XCK_ENTRY(xck_mgga_ub_g1),
  };
#undef XCK_ENTRY
#define XCK_ENTRY(k) {#k, k, k##_scal_names, &k##_n_scal, nullptr}
  const kernel_t<gg_fn_t> gg_kernels[] = {
    XCK_ENTRY(xck_lda_r_gg),
    XCK_ENTRY(xck_lda_u_gg),
    XCK_ENTRY(xck_gga_r_gg),
    XCK_ENTRY(xck_gga_u_gg),
    XCK_ENTRY(xck_mgga_tau_r_gg),
    XCK_ENTRY(xck_mgga_tau_u_gg),
    XCK_ENTRY(xck_mgga_lapl_r_gg),
    XCK_ENTRY(xck_mgga_lapl_u_gg),
    XCK_ENTRY(xck_mgga_r_gg),
    XCK_ENTRY(xck_mgga_u_gg),
  };
#undef XCK_ENTRY

  template<typename F, size_t N> const kernel_t<F> * find(const kernel_t<F> (&table)[N], const std::string & name) {
    for(const kernel_t<F> & k : table)
      if(name == k.name)
        return &k;
    return nullptr;
  }

  template<typename F> const kernel_t<F> & require(const kernel_t<F> * k, const std::string & name) {
    if(!k)
      throw std::logic_error("libxckernel has no kernel " + name + ".\n");
    return *k;
  }

  /// The kernel's operands in its order
  template<typename F> std::vector<const double *> scal_array(const kernel_t<F> & k, const xckernel_dispatch::operands_t & ops) {
    std::vector<const double *> scal(*k.n_scal);
    for(int i=0;i<*k.n_scal;i++) {
      auto it = ops.find(k.scal_names[i]);
      if(it == ops.end()) {
        std::ostringstream oss;
        oss << "Kernel " << k.name << " needs the operand " << k.scal_names[i] << ".\n";
        throw std::logic_error(oss.str());
      }
      scal[i] = it->second;
    }
    return scal;
  }

  template<typename F> void check_tower(const kernel_t<F> & k, const xckernel_dispatch::tower_t & t, int order, const char * what) {
    if(!t.data || t.order < order) {
      std::ostringstream oss;
      oss << "Kernel " << k.name << " needs the " << what << " tower through order " << order << ".\n";
      throw std::logic_error(oss.str());
    }
  }

  void check_status(int status, const char * name) {
    if(status != 0)
      throw std::runtime_error(std::string("libxckernel kernel ") + name + " failed.\n");
  }
}

namespace xckernel_dispatch {
  bool has_kernel(const std::string & name) {
    return find(mat_kernels, name) || find(g1_kernels, name) || find(gg_kernels, name);
  }

  int chi_order(const std::string & name) {
    if(const kernel_t<mat_fn_t> * k = find(mat_kernels, name))
      return *k->chi_order;
    if(const kernel_t<g1_fn_t> * k = find(g1_kernels, name))
      return *k->chi_order;
    if(find(gg_kernels, name))
      return -1;
    throw std::logic_error("libxckernel has no kernel " + name + ".\n");
  }

  void contract(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                const operands_t & ops, double * out) {
    const kernel_t<mat_fn_t> & k(require(find(mat_kernels, name), name));
    check_tower(k, chi, *k.chi_order, "collocation");
    const std::vector<const double *> scal(scal_array(k, ops));
    check_status(k.fn(npts, nbf, chi.data, scal.data(), out), k.name);
  }

  void contract_g1(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                   const tower_t & Dchi, const operands_t & ops, double * out) {
    const kernel_t<g1_fn_t> & k(require(find(g1_kernels, name), name));
    // D chi is read one order below chi
    check_tower(k, chi, *k.chi_order, "collocation");
    check_tower(k, Dchi, *k.chi_order-1, "density-contracted collocation");
    const std::vector<const double *> scal(scal_array(k, ops));
    check_status(k.fn(npts, nbf, chi.data, Dchi.data, scal.data(), out), k.name);
  }

  void contract_gg(const std::string & name, int64_t npts, const operands_t & ops, double * out) {
    const kernel_t<gg_fn_t> & k(require(find(gg_kernels, name), name));
    const std::vector<const double *> scal(scal_array(k, ops));
    check_status(k.fn(npts, scal.data(), out), k.name);
  }
}
