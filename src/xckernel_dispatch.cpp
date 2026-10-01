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
  typedef int (*kernel_fn_t)(int64_t, int64_t, const double *, const double *,
                             const double *, const double *,
                             const double * const *, double *);

  struct kernel_t {
    const char * name;
    kernel_fn_t fn;
    const char ** scal_names;
    const int * n_scal;
    /// Does the kernel read the basis-function gradients / laplacians?
    bool dchi, lapl_chi;
  };

// The collocation each family reads: LDA the basis functions only, the
// other families also their gradients, and the laplacian families their
// laplacians; no kernel here uses the Hessian.
#define XCK_ENTRY(k, dchi, lapl) {#k, k, k##_scal_names, &k##_n_scal, dchi, lapl}
  const kernel_t kernels[] = {
    XCK_ENTRY(xck_lda_r_o1, false, false),
    XCK_ENTRY(xck_lda_ua_o1, false, false),
    XCK_ENTRY(xck_lda_ub_o1, false, false),
    XCK_ENTRY(xck_lda_r_o2, false, false),
    XCK_ENTRY(xck_lda_ua_o2, false, false),
    XCK_ENTRY(xck_lda_ub_o2, false, false),
    XCK_ENTRY(xck_lda_st_o2_p, false, false),
    XCK_ENTRY(xck_lda_st_o2_m, false, false),
    XCK_ENTRY(xck_gga_r_o1, true, false),
    XCK_ENTRY(xck_gga_ua_o1, true, false),
    XCK_ENTRY(xck_gga_ub_o1, true, false),
    XCK_ENTRY(xck_gga_r_o2, true, false),
    XCK_ENTRY(xck_gga_ua_o2, true, false),
    XCK_ENTRY(xck_gga_ub_o2, true, false),
    XCK_ENTRY(xck_gga_st_o2_p, true, false),
    XCK_ENTRY(xck_gga_st_o2_m, true, false),
    XCK_ENTRY(xck_mgga_tau_r_o1, true, false),
    XCK_ENTRY(xck_mgga_tau_ua_o1, true, false),
    XCK_ENTRY(xck_mgga_tau_ub_o1, true, false),
    XCK_ENTRY(xck_mgga_tau_r_o2, true, false),
    XCK_ENTRY(xck_mgga_tau_ua_o2, true, false),
    XCK_ENTRY(xck_mgga_tau_ub_o2, true, false),
    XCK_ENTRY(xck_mgga_tau_st_o2_p, true, false),
    XCK_ENTRY(xck_mgga_tau_st_o2_m, true, false),
    XCK_ENTRY(xck_mgga_lapl_r_o1, true, true),
    XCK_ENTRY(xck_mgga_lapl_ua_o1, true, true),
    XCK_ENTRY(xck_mgga_lapl_ub_o1, true, true),
    XCK_ENTRY(xck_mgga_lapl_r_o2, true, true),
    XCK_ENTRY(xck_mgga_lapl_ua_o2, true, true),
    XCK_ENTRY(xck_mgga_lapl_ub_o2, true, true),
    XCK_ENTRY(xck_mgga_lapl_st_o2_p, true, true),
    XCK_ENTRY(xck_mgga_lapl_st_o2_m, true, true),
    XCK_ENTRY(xck_mgga_r_o1, true, true),
    XCK_ENTRY(xck_mgga_ua_o1, true, true),
    XCK_ENTRY(xck_mgga_ub_o1, true, true),
    XCK_ENTRY(xck_mgga_r_o2, true, true),
    XCK_ENTRY(xck_mgga_ua_o2, true, true),
    XCK_ENTRY(xck_mgga_ub_o2, true, true),
    XCK_ENTRY(xck_mgga_st_o2_p, true, true),
    XCK_ENTRY(xck_mgga_st_o2_m, true, true),
    XCK_ENTRY(xck_cmgga_tau_r_o1, true, false),
    XCK_ENTRY(xck_cmgga_tau_ua_o1, true, false),
    XCK_ENTRY(xck_cmgga_tau_ub_o1, true, false),
    XCK_ENTRY(xck_cmgga_tau_r_o2, true, false),
    XCK_ENTRY(xck_cmgga_tau_ua_o2, true, false),
    XCK_ENTRY(xck_cmgga_tau_ub_o2, true, false),
    XCK_ENTRY(xck_cmgga_tau_st_o2_p, true, false),
    XCK_ENTRY(xck_cmgga_tau_st_o2_m, true, false),
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

    if(!chi || (k->dchi && !dchi) || (k->lapl_chi && !lapl_chi))
      throw std::logic_error("Kernel " + name + " is missing basis-function values or derivatives.\n");

    // Order the operands as the kernel declares them
    std::vector<const double *> scal(*k->n_scal);
    for(int i=0;i<*k->n_scal;i++) {
      auto it = ops.find(k->scal_names[i]);
      if(it == ops.end()) {
        std::ostringstream oss;
        oss << "Kernel " << name << " needs the operand " << k->scal_names[i] << ".\n";
        throw std::logic_error(oss.str());
      }
      scal[i] = it->second;
    }

    if(k->fn(npts, nbf, chi, dchi, lapl_chi, nullptr, scal.data(), out) != 0)
      throw std::runtime_error("libxckernel kernel " + name + " failed.\n");
  }
}
