/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#ifndef ERKALE_BFPROD
#define ERKALE_BFPROD

#include <vector>
#include <cstdlib>
#include "basis.h"

/// Structure for contraction in 1d product
typedef struct {
  /// Exponent in Fourier transform
  int m;
  /// Expansion coefficient
  double c;
} prod_gaussian_1d_contr_t;

/// Comparison operator for sorts
bool operator<(const prod_gaussian_1d_contr_t & lhs, const prod_gaussian_1d_contr_t & rhs);
/// Comparison operator for addition
bool operator==(const prod_gaussian_1d_contr_t & lhs, const prod_gaussian_1d_contr_t & rhs);

/// Structure for 1d product
typedef struct {
  /// Center of product Gaussian
  double xp;
  /// Reduced exponent
  double zeta;
  /// Contraction
  std::vector<prod_gaussian_1d_contr_t> c;
} prod_gaussian_1d_t;

/// Comparison operator for sorting
bool operator<(const prod_gaussian_1d_t & lhs, const prod_gaussian_1d_t & rhs);
/// Comparison for adding terms
bool operator==(const prod_gaussian_1d_t & lhs, const prod_gaussian_1d_t & rhs);

/// One dimensional product
class prod_gaussian_1d {
  /// Product Gaussians
  std::vector<prod_gaussian_1d_t> p;

  void add_term(const prod_gaussian_1d_t & t);
  void add_contr(size_t ind, const prod_gaussian_1d_contr_t & t);

 public:
  prod_gaussian_1d(double xa, double xb, int la, int lb, double zetaa, double zetab);
  ~prod_gaussian_1d();

  prod_gaussian_1d operator+(const prod_gaussian_1d & rhs) const;
  prod_gaussian_1d & operator+=(const prod_gaussian_1d & rhs);

  void print() const;
  std::vector<prod_gaussian_1d_t> get() const;
};

/// Structure for contraction in 3d product
typedef struct {
  /// x exponent
  int l;
  /// y exponent
  int m;
  /// z exponent
  int n;

  /// Contraction coefficient
  double c;
} prod_gaussian_3d_contr_t;

/// Comparison operator for sorts
bool operator<(const prod_gaussian_3d_contr_t & lhs, const prod_gaussian_3d_contr_t & rhs);
/// Comparison operator for addition
bool operator==(const prod_gaussian_3d_contr_t & lhs, const prod_gaussian_3d_contr_t & rhs);

/// Structure for 3d product
typedef struct {
  /// x coordinate of center
  double xp;
  /// y coordinate of center
  double yp;
  /// z coordinate of center
  double zp;

  /// Exponent
  double zeta;

  /// Contraction
  std::vector<prod_gaussian_3d_contr_t> c;
} prod_gaussian_3d_t;

/// Comparison operator for sorting
bool operator<(const prod_gaussian_3d_t & lhs, const prod_gaussian_3d_t & rhs);
/// Comparison for adding terms
bool operator==(const prod_gaussian_3d_t & lhs, const prod_gaussian_3d_t & rhs);

/// Three dimensional product
class prod_gaussian_3d {
  /// Product Gaussians
  std::vector<prod_gaussian_3d_t> p;

  void add_term(const prod_gaussian_3d_t & t);
  void add_contr(size_t ind, const prod_gaussian_3d_contr_t & t);

 public:
  prod_gaussian_3d();
  ~prod_gaussian_3d();

  prod_gaussian_3d(double xa, double xb, double ya, double yb, double za, double zb, int la, int lb, int ma, int mb, int na, int nb, double zetaa, double zetab);

  prod_gaussian_3d operator+(const prod_gaussian_3d & rhs) const;
  prod_gaussian_3d & operator+=(const prod_gaussian_3d & rhs);

  prod_gaussian_3d operator*(double fac) const;

  /// Clean out terms with zero contribution
  void clean();

  /// Compute the integral over \f$ d^3r \f$. When \f$ \chi_\mu \chi_\nu \f$ has been computed, one can check this against \f$ \langle \mu | \nu \rangle \f$
  double integral() const;

  /// Get expansion
  std::vector<prod_gaussian_3d_t> get() const;

  void print() const;
};

// Compute product of bfs on shells i and j
std::vector<prod_gaussian_3d> compute_product(const BasisSet & bas, size_t is, size_t js);
// Transform computed product into spherical basis
std::vector<prod_gaussian_3d> spherical_transform(const BasisSet & bas, size_t is, size_t js, std::vector<prod_gaussian_3d> & res);

// Compute product bfs
std::vector<prod_gaussian_3d> compute_products(const BasisSet & bas);

#endif
