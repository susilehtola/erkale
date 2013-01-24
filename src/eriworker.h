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

#include <vector>
#include "basis.h"

#include <libint/libint.h>

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

/// Worker for computing electron repulsion integrals
class ERIWorker {
  /// Input array
  std::vector<double> input;
  /// Output array
  std::vector<double> output;

  /// Libint worker
  Libint_t libint;

  /// Compute the cartesian ERIs
  void compute_cartesian(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Reorder integrals
  void reorder(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);

  /// Do spherical transform
  void spherical_transform(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  void transform_i(int am, size_t Nj, size_t Nk, size_t Nl);
  void transform_j(int am, size_t Ni, size_t Nk, size_t Nl);
  void transform_k(int am, size_t Ni, size_t Nj, size_t Nl);
  void transform_l(int am, size_t Ni, size_t Nj, size_t Nk);

 public:
  ERIWorker(int maxam, int maxcontr);
  ~ERIWorker();

  /// Compute eris
  void compute(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls, std::vector<double> & ints);
};

/// Compute precursor
eri_precursor_t compute_precursor(const GaussianShell *is, const GaussianShell *js);
/// Compute data for libint
void compute_libint_data(Libint_t & libint, const eri_precursor_t & ip, const eri_precursor_t &jp, int mmax);
/// Compute index of swapped integral
size_t get_swapped_ind(size_t i, size_t Ni, size_t j, size_t Nj, size_t k, size_t Nk, size_t l, size_t Nl, bool swap_ij, bool swap_kl, bool swap_ijkl);

#endif
