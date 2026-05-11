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

#include <libint/libint.h>
#include <libderiv/libderiv.h>

/// Precursor data for ERIs
typedef struct {
  /// Distance between centers A and B
  arma::vec AB;
  /// Sum of exponents (Na,Nb)
  arma::mat zeta;
  /// Coordinates of center of product gaussian P, dimension (Na,Nb,3)
  arma::cube P;
  /// Distance between P and shell i center (Na,Nb,3)
  arma::cube PA;
  /// Distance between P and shell j center (Na,Nb,3)
  arma::cube PB;
  /// Contraction for first center (Na)
  std::vector<contr_t> ic;
  /// Array of exponents for second center (Nb)
  std::vector<contr_t> jc;
  /// Overlap of primitives on i and j (Na,Nb)
  arma::mat S;
} eri_precursor_t;

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

  /// Integral kernel (i.e. Boys' function for Coulomb integrals)
  arma::vec Gn;

  /// Per-position shellpair-precursor cache, keyed by shell pointer
  /// pair. Each compute_precursor call site identifies itself with a
  /// slot index (0 for the bra "ij" pair, 1 for the ket "kl" pair);
  /// the slots are independent so a hit in one never invalidates the
  /// other. Within a 4-index J/K build the outer (is,js) pair
  /// repeats for many (ks,ls) inner iterations -- slot 0 stays
  /// perfectly warm and slot 1 churns harmlessly. One worker per
  /// thread under OpenMP, so no locking. Geometry steps rebuild the
  /// basis and the workers, so stale shell pointers don't survive
  /// across iterations.
  const GaussianShell* cached_is_[2];
  const GaussianShell* cached_js_[2];
  eri_precursor_t cached_precursor_[2];

  /// Compute the integral kernel
  virtual void compute_G(double rho, double T, int nmax);

  /// Reorder integrals
  void reorder(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls, bool swap_ij, bool swap_kl, bool swap_ijkl);

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

  /// Get precursor for a shell pair, consulting the per-worker cache.
  /// `slot` (0 or 1) identifies the cache bucket: callers must use a
  /// distinct slot for each precursor that needs to coexist in one
  /// computation (slot 0 for the bra ij pair, slot 1 for the ket kl
  /// pair). The returned reference is valid until the next
  /// compute_precursor() call on the *same* slot.
  const eri_precursor_t & compute_precursor(const GaussianShell *is, const GaussianShell *js, int slot);
  /// Fill an eri_precursor_t for a shell pair (uncached helper).
  void fill_precursor(const GaussianShell *is, const GaussianShell *js, eri_precursor_t & r);

 public:
  IntegralWorker();
  virtual ~IntegralWorker();
};

/// Worker for computing electron repulsion integrals
class ERIWorker: public IntegralWorker {
  /// Libint worker
  Libint_t libint;

  /// Compute the cartesian ERIs
  void compute_cartesian(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Compute the cartesian ERIs using Huzinaga routines
  void compute_cartesian_debug(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Compute data for libint
  void compute_libint_data(const eri_precursor_t & ip, const eri_precursor_t & jp, int mmax);

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

/// Worker for computing electron repulsion integrals
class dERIWorker: public IntegralWorker {
  /// Libint worker
  Libderiv_t libderiv;

  /// Compute the cartesian ERI derivatives
  void compute_cartesian();
  /// Compute data for libderiv
  void compute_libderiv_data(const eri_precursor_t & ip, const eri_precursor_t & jp, int mmax);

  // Pointers to the current shells
  /// 1st shell
  const GaussianShell *is, *is_orig;
  /// 2nd shell
  const GaussianShell *js, *js_orig;
  /// 3rd shell
  const GaussianShell *ks, *ks_orig;
  /// 4th shell
  const GaussianShell *ls, *ls_orig;
  /// Swap?
  bool swap_ij, swap_kl, swap_ijkl;

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
  /// Compute the kernel
  void compute_G(double rho, double T, int nmax);

  /// Range separation constant
  double omega;
  /// Weight for long-range (i.e. normal HF) exchange
  double alpha;
  /// Weight for short-range exchange
  double beta;

  /// Short and long range Boys functions
  arma::vec bf_short, bf_long;

 public:
  /// Constructor
  ERIWorker_srlr(int maxam, int maxcontr, double omega, double alpha, double beta);
  /// Destructor
  ~ERIWorker_srlr();
};

/// Worker for computing short- and long-range electron repulsion integrals
class dERIWorker_srlr: public dERIWorker {
  /// Compute the kernel
  void compute_G(double rho, double T, int nmax);

  /// Range separation constant
  double omega;
  /// Factor of long-range exchange
  double alpha;
  /// Factor of short-range exchange
  double beta;

  /// Short and long range Boys functions
  arma::vec bf_short, bf_long;

 public:
  dERIWorker_srlr(int maxam, int maxcontr, double omega, double alpha, double beta);
  ~dERIWorker_srlr();
};


#endif
