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
 * Name-based access to the vendored libxckernel contractions (ABI v2).
 *
 * A kernel is addressed by its generated name, e.g. "xck_gga_ua_o2",
 * "xck_mgga_r_o1_diag", "xck_gga_r_g1" or "xck_gga_u_gg". Its
 * per-point operands are bound by the names the kernel declares (w,
 * rho_x, rho_a_p1_xy, tau_x, vsigma_1, v2rhosigma_3, ...): the caller
 * fills an operand table and the dispatcher orders it into the
 * kernel's scal array. The collocation is a Cartesian derivative
 * tower chi[k][u][g], components 1, x, y, z, xx, xy, ... with the grid
 * index fastest, through at least the order the kernel reads. Every
 * operand the kernel declares and the collocation it reads must be
 * supplied; anything missing is an error.
 */
namespace xckernel_dispatch {
  /// Operand table: scalar operand name -> contiguous array of npts values
  typedef std::map<std::string, const double *> operands_t;

  /// A Cartesian derivative tower (nbf functions, npts points) through order
  struct tower_t {
    const double * data;
    int order;
  };

  /// Number of tower components through the given order
  inline int tower_size(int order) { return (order+1)*(order+2)*(order+3)/6; }

  /// Does the kernel exist?
  bool has_kernel(const std::string & name);
  /// Collocation order the kernel reads
  int chi_order(const std::string & name);
  /// Order of the density-contracted collocation a gradient kernel reads
  int Dchi_order(const std::string & name);

  /// Matrix-valued kernels: o1, o2 (nbf x nbf), o1_diag (nbf), fg (3 x nbf x nbf)
  void contract(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                const operands_t & ops, double * out);
  /// Basis-class gradient rows (3 x nbf) of a symmetric density matrix
  /// D: chi and Dchi = D chi
  void contract_g1(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                   const tower_t & Dchi, const operands_t & ops, double * out);
  /// Basis-class gradient rows (3 x nbf) of a general density matrix M
  /// (complex orbitals in a real basis): Dchi = M chi and DTchi = M^T chi
  void contract_g1c(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                    const tower_t & Dchi, const tower_t & DTchi, const operands_t & ops, double * out);
  /// Grid-class gradient (3 x npts)
  void contract_gg(const std::string & name, int64_t npts, const operands_t & ops, double * out);
}

#endif
