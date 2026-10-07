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
  typedef int (*mat_fn_t)(int64_t, int64_t, const double *, const double * const *, double *);
  typedef int (*g1_fn_t)(int64_t, int64_t, const double *, const double *, const double * const *, double *);
  typedef int (*g1c_fn_t)(int64_t, int64_t, const double *, const double *, const double *, const double * const *, double *);
  typedef int (*gg_fn_t)(int64_t, const double * const *, double *);

  /// The kernel's entry in libxckernel's dispatch index
  const xck_kernel_info * find(const std::string & name) {
    for(int i=0;i<xckernel_n_kernels;i++)
      if(name == xckernel_kernels[i].name)
        return &xckernel_kernels[i];
    return nullptr;
  }

  /// The kernel's entry, which must be of one of the given ABI kinds
  const xck_kernel_info & require(const std::string & name, std::initializer_list<const char *> kinds) {
    const xck_kernel_info * k(find(name));
    if(!k)
      throw std::logic_error("libxckernel has no kernel " + name + ".\n");
    for(const char * kind : kinds)
      if(std::strcmp(k->kind, kind)==0)
        return *k;
    throw std::logic_error("libxckernel kernel " + name + " is of kind " + k->kind + ", not callable here.\n");
  }

  /// The kernel's operands in its order
  std::vector<const double *> scal_array(const xck_kernel_info & k, const xckernel_dispatch::operands_t & ops) {
    std::vector<const double *> scal(k.n_scal);
    for(int i=0;i<k.n_scal;i++) {
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

  void check_tower(const xck_kernel_info & k, const xckernel_dispatch::tower_t & t, int order, const char * what) {
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
    return find(name) != nullptr;
  }

  int chi_order(const std::string & name) {
    const xck_kernel_info * k(find(name));
    if(!k)
      throw std::logic_error("libxckernel has no kernel " + name + ".\n");
    return k->chi_order;
  }

  int Dchi_order(const std::string & name) {
    return require(name, {"g1", "g1c"}).dchi_order;
  }

  void contract(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                const operands_t & ops, double * out) {
    const xck_kernel_info & k(require(name, {"matrix", "diag"}));
    check_tower(k, chi, k.chi_order, "collocation");
    const std::vector<const double *> scal(scal_array(k, ops));
    check_status(reinterpret_cast<mat_fn_t>(k.fn)(npts, nbf, chi.data, scal.data(), out), k.name);
  }

  void contract_g1(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                   const tower_t & Dchi, const operands_t & ops, double * out) {
    const xck_kernel_info & k(require(name, {"g1"}));
    check_tower(k, chi, k.chi_order, "collocation");
    check_tower(k, Dchi, k.dchi_order, "density-contracted collocation");
    const std::vector<const double *> scal(scal_array(k, ops));
    check_status(reinterpret_cast<g1_fn_t>(k.fn)(npts, nbf, chi.data, Dchi.data, scal.data(), out), k.name);
  }

  void contract_g1c(const std::string & name, int64_t npts, int64_t nbf, const tower_t & chi,
                    const tower_t & Dchi, const tower_t & DTchi, const operands_t & ops, double * out) {
    const xck_kernel_info & k(require(name, {"g1c"}));
    check_tower(k, chi, k.chi_order, "collocation");
    check_tower(k, Dchi, k.dchi_order, "density-contracted collocation");
    check_tower(k, DTchi, k.dtchi_order, "transpose-density-contracted collocation");
    const std::vector<const double *> scal(scal_array(k, ops));
    check_status(reinterpret_cast<g1c_fn_t>(k.fn)(npts, nbf, chi.data, Dchi.data, DTchi.data, scal.data(), out), k.name);
  }

  void contract_gg(const std::string & name, int64_t npts, const operands_t & ops, double * out) {
    const xck_kernel_info & k(require(name, {"gg"}));
    const std::vector<const double *> scal(scal_array(k, ops));
    check_status(reinterpret_cast<gg_fn_t>(k.fn)(npts, scal.data(), out), k.name);
  }
}
