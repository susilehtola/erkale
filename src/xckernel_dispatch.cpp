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

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
  typedef int (*kernel_fn_t)(int64_t, int64_t, const double *, const double *,
                             const double *, const double *,
                             const double * const *, double *);

  struct kernel_t {
    const char * name;
    kernel_fn_t fn;
    const char ** scal_names;
    const int * n_scal;
    const int * n_fields;
  };

#define XCK_ENTRY(k) {#k, k, k##_scal_names, &k##_n_scal, &k##_n_fields}
  const kernel_t kernels[] = {
    XCK_ENTRY(xck_lda_r_o1),
    XCK_ENTRY(xck_lda_ua_o1),
    XCK_ENTRY(xck_lda_ub_o1),
    XCK_ENTRY(xck_lda_r_o2),
    XCK_ENTRY(xck_lda_ua_o2),
    XCK_ENTRY(xck_lda_ub_o2),
    XCK_ENTRY(xck_lda_st_o2_p),
    XCK_ENTRY(xck_lda_st_o2_m),
    XCK_ENTRY(xck_gga_r_o1),
    XCK_ENTRY(xck_gga_ua_o1),
    XCK_ENTRY(xck_gga_ub_o1),
    XCK_ENTRY(xck_gga_r_o2),
    XCK_ENTRY(xck_gga_ua_o2),
    XCK_ENTRY(xck_gga_ub_o2),
    XCK_ENTRY(xck_gga_st_o2_p),
    XCK_ENTRY(xck_gga_st_o2_m),
    XCK_ENTRY(xck_mgga_tau_r_o1),
    XCK_ENTRY(xck_mgga_tau_ua_o1),
    XCK_ENTRY(xck_mgga_tau_ub_o1),
    XCK_ENTRY(xck_mgga_tau_r_o2),
    XCK_ENTRY(xck_mgga_tau_ua_o2),
    XCK_ENTRY(xck_mgga_tau_ub_o2),
    XCK_ENTRY(xck_mgga_tau_st_o2_p),
    XCK_ENTRY(xck_mgga_tau_st_o2_m),
    XCK_ENTRY(xck_mgga_lapl_r_o1),
    XCK_ENTRY(xck_mgga_lapl_ua_o1),
    XCK_ENTRY(xck_mgga_lapl_ub_o1),
    XCK_ENTRY(xck_mgga_lapl_r_o2),
    XCK_ENTRY(xck_mgga_lapl_ua_o2),
    XCK_ENTRY(xck_mgga_lapl_ub_o2),
    XCK_ENTRY(xck_mgga_lapl_st_o2_p),
    XCK_ENTRY(xck_mgga_lapl_st_o2_m),
    XCK_ENTRY(xck_mgga_r_o1),
    XCK_ENTRY(xck_mgga_ua_o1),
    XCK_ENTRY(xck_mgga_ub_o1),
    XCK_ENTRY(xck_mgga_r_o2),
    XCK_ENTRY(xck_mgga_ua_o2),
    XCK_ENTRY(xck_mgga_ub_o2),
    XCK_ENTRY(xck_mgga_st_o2_p),
    XCK_ENTRY(xck_mgga_st_o2_m),
    XCK_ENTRY(xck_cmgga_tau_r_o1),
    XCK_ENTRY(xck_cmgga_tau_ua_o1),
    XCK_ENTRY(xck_cmgga_tau_ub_o1),
    XCK_ENTRY(xck_cmgga_tau_r_o2),
    XCK_ENTRY(xck_cmgga_tau_ua_o2),
    XCK_ENTRY(xck_cmgga_tau_ub_o2),
    XCK_ENTRY(xck_cmgga_tau_st_o2_p),
    XCK_ENTRY(xck_cmgga_tau_st_o2_m),
  };
#undef XCK_ENTRY

  const kernel_t * find(const std::string & name) {
    for(const kernel_t & k : kernels)
      if(name == k.name)
        return &k;
    return nullptr;
  }
}

namespace xckernel_dispatch {
  bool has_kernel(const std::string & name) {
    return find(name) != nullptr;
  }

  void contract(const std::string & name, int64_t npts, int64_t nbf,
                const double * chi, const double * dchi, const double * lapl_chi,
                const operands_t & ops, double * out) {
    const kernel_t * k = find(name);
    if(!k)
      throw std::logic_error("libxckernel has no kernel " + name + ".\n");

    // Order the operands as the kernel declares them; absent derivative
    // arrays are zero.
    std::vector<double> zero;
    std::vector<const double *> scal(*k->n_scal);
    for(int i=0;i<*k->n_scal;i++) {
      auto it = ops.find(k->scal_names[i]);
      if(it != ops.end()) {
        scal[i] = it->second;
      } else if(i >= *k->n_fields) {
        if(zero.empty())
          zero.assign(npts, 0.0);
        scal[i] = zero.data();
      } else {
        std::ostringstream oss;
        oss << "Kernel " << name << " needs the operand " << k->scal_names[i] << ".\n";
        throw std::logic_error(oss.str());
      }
    }

    if(k->fn(npts, nbf, chi, dchi, lapl_chi, nullptr, scal.data(), out) != 0)
      throw std::runtime_error("libxckernel kernel " + name + " failed.\n");
  }
}
