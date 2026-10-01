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

#ifndef ERKALE_XCKERNEL_DISPATCH
#define ERKALE_XCKERNEL_DISPATCH

#include <cstdint>
#include <map>
#include <string>

/**
 * Name-based access to the vendored libxckernel contractions.
 *
 * A kernel is addressed by family (lda, gga, mgga_tau, mgga_lapl,
 * mgga, cmgga_tau), spin variant (r, ua, ub, st_o2_p/m are the
 * suffixes of the generated names) and order, e.g. "xck_gga_ua_o2".
 * Its per-point scalar operands are bound by the names the kernel
 * declares (w, grad_rho_x, rho_a_p1, vsigma_1, v2rhosigma_3, ...): the
 * caller fills an operand table and the dispatcher orders it into the
 * kernel's scal array. Missing functional-derivative arrays are taken
 * as zero (the kernels are linear in them); a missing field is an
 * error.
 */
namespace xckernel_dispatch {
  /// Operand table: scalar operand name -> contiguous array of npts values
  typedef std::map<std::string, const double *> operands_t;

  /// Does the kernel exist?
  bool has_kernel(const std::string & name);

  /**
   * Run kernel name, accumulating into out (nbf x nbf). chi is
   * (nbf, npts) and dchi (3, nbf, npts) with the grid index fastest;
   * lapl_chi is (nbf, npts) or NULL for families that do not use it.
   */
  void contract(const std::string & name, int64_t npts, int64_t nbf,
                const double * chi, const double * dchi, const double * lapl_chi,
                const operands_t & ops, double * out);
}

#endif
