/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#ifndef ERIWORKER_H
#define ERIWORKER_H

// To debug derivatives
//#define DEBUGDERIV

#include <vector>
// Needed to define contr_t
#include "basis.h"

// The workers no longer use libint or libderiv, but the executables
// still call init_libint_base() / init_libderiv_base(); the includes
// go away together with those calls when libint is dropped for good.
#include <libint/libint.h>
#include <libderiv/libderiv.h>

/// Selector for the libcint two-electron integral kernels
typedef enum {
  CINT_ERI,      ///< int2e: plain electron repulsion integrals
  CINT_ERI_IP1,  ///< int2e_ip1: derivative wrt the first shell
  CINT_ERI_IP2   ///< int2e_ip2: derivative wrt the third shell
} cint_eri_t;

/// Worker for dealing with electron repulsion integrals and their derivatives
class IntegralWorker {
 protected:
  /// Storage arrays (operated through pointer)
  std::vector<double> arrone;
  /// Storage arrays (operated through pointer)
  std::vector<double> arrtwo;

  /// Input array
  std::vector<double> * input;
  /// Output array
  std::vector<double> * output;

  /// Range separation constant: the computed integrals are
  /// rs_alpha * full Coulomb + rs_beta * erfc(rs_omega r12)/r12
  double rs_omega;
  /// Weight of the full-Coulomb component
  double rs_alpha;
  /// Weight of the short-range (complementary error function) component
  double rs_beta;

  /// libcint atom table for the current shell quartet
  std::vector<int> cint_atm;
  /// libcint shell table for the current shell quartet
  std::vector<int> cint_bas;
  /// libcint data array: coordinates, exponents, contraction coefficients
  std::vector<double> cint_env;
  /// libcint integral output for the short-range component
  std::vector<double> cint_sr;
  /// libcint scratch memory
  std::vector<double> cint_cache;

  /// Set up the libcint environment for the given shell quartet
  void setup_cint_env(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Evaluate a two-electron kernel over the quartet set up in the
  /// environment, combining the range separation components:
  /// out = rs_alpha * full + rs_beta * erfc(rs_omega r12)/r12.
  /// N is the number of integrals per operator component.
  void cint_int2e(cint_eri_t kernel, int ncomp, const int *shls, size_t N, std::vector<double> & out);

  /// Do spherical transforms if necessary
  void spherical_transform(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Do spherical transform with respect to first index
  void transform_i(int am, size_t Nj, size_t Nk, size_t Nl);
  /// Do spherical transform with respect to second index
  void transform_j(int am, size_t Ni, size_t Nk, size_t Nl);
  /// Do spherical transform with respect to third index
  void transform_k(int am, size_t Ni, size_t Nj, size_t Nl);
  /// Do spherical transform with respect to fourth index
  void transform_l(int am, size_t Ni, size_t Nj, size_t Nk);

 public:
  IntegralWorker();
  virtual ~IntegralWorker();
};

/// Worker for computing electron repulsion integrals
class ERIWorker: public IntegralWorker {
  /// libcint integral output
  std::vector<double> cint_out;

  /// Compute the cartesian ERIs
  void compute_cartesian(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Compute the cartesian ERIs using Huzinaga routines
  void compute_cartesian_debug(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);

 public:
  /// Constructor
  ERIWorker(int maxam, int maxcontr);
  /// Destructor
  virtual ~ERIWorker();

  /// Compute eris
  void compute(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Compute eris using Huzinaga routines
  void compute_debug(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Get the eris
  std::vector<double> get() const;
  /// Get the eris
  std::vector<double> & rget() const;
  /// Get pointer to eris
  const std::vector<double> * getp() const;
};

/// Worker for computing electron repulsion integral derivatives
class dERIWorker: public IntegralWorker {
  // Pointers to the current shells
  /// 1st shell
  const GaussianShell *is;
  /// 2nd shell
  const GaussianShell *js;
  /// 3rd shell
  const GaussianShell *ks;
  /// 4th shell
  const GaussianShell *ls;

  /// Cartesian derivative components (x,y,z) with respect to the
  /// centers of the 2nd, 3rd and 4th shell, in ERKALE layout with the
  /// normalization factors included; the 1st shell derivative follows
  /// from translational invariance
  std::vector<double> cint_dJ, cint_dK, cint_dL;
  /// Scratch buffer for the K derivative layout remap
  std::vector<double> cint_scr;

  /// Compute the cartesian ERI derivatives
  void compute_cartesian();
  /// Get the idx'th derivative in the input array
  void get_idx(int idx);

 public:
  dERIWorker(int maxam, int maxcontr);
  virtual ~dERIWorker();

  /// Compute derivatives
  void compute(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Get the derivatives wrt index idx
  std::vector<double> get(int idx);
  /// Get the derivatives wrt index idx
  const std::vector<double> * getp(int idx);

  /// Compute the derivatives, debug version
  std::vector<double> get_debug(int idx);
};

/// Worker for computing short- and long-range electron repulsion integrals
class ERIWorker_srlr: public ERIWorker {
 public:
  /// Constructor
  ERIWorker_srlr(int maxam, int maxcontr, double omega, double alpha, double beta);
  /// Destructor
  ~ERIWorker_srlr();
};

/// Worker for computing short- and long-range electron repulsion integral derivatives
class dERIWorker_srlr: public dERIWorker {
 public:
  /// Constructor
  dERIWorker_srlr(int maxam, int maxcontr, double omega, double alpha, double beta);
  /// Destructor
  ~dERIWorker_srlr();
};

#include <memory>

/// Allocate an ERIWorker matching the given range-separation parameters.
/// (omega, alpha, beta) == (0, 1, 0) is the plain-Coulomb default and
/// gets a vanilla ERIWorker; everything else gets ERIWorker_srlr.
inline std::unique_ptr<ERIWorker>
make_eri_worker(int maxam, int maxcontr, double omega, double alpha, double beta) {
  if(omega == 0.0 && alpha == 1.0 && beta == 0.0)
    return std::unique_ptr<ERIWorker>(new ERIWorker(maxam, maxcontr));
  return std::unique_ptr<ERIWorker>(new ERIWorker_srlr(maxam, maxcontr, omega, alpha, beta));
}

inline std::unique_ptr<dERIWorker>
make_deri_worker(int maxam, int maxcontr, double omega, double alpha, double beta) {
  if(omega == 0.0 && alpha == 1.0 && beta == 0.0)
    return std::unique_ptr<dERIWorker>(new dERIWorker(maxam, maxcontr));
  return std::unique_ptr<dERIWorker>(new dERIWorker_srlr(maxam, maxcontr, omega, alpha, beta));
}

#endif
