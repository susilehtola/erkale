/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#ifndef ERKALE_SPHEXP
#define ERKALE_SPHEXP

#include <vector>
#include "complex.h"

/// Coefficient of expansion in spherical harmonics \f$ \sum_{lm} c_{lm} Y_{lm} \f$
typedef struct {
  /// Angular number l of term in expansion
  int l;
  /// Angular number m of term in expansion
  int m;
  /// Expansion coefficient
  complex c;
} ylmcoeff_t;

/// Sorting operator
bool operator<(const ylmcoeff_t & lhs, const ylmcoeff_t & rhs);
/// Addition operator
bool operator==(const ylmcoeff_t & lhs, const ylmcoeff_t & rhs);

// Forward declaration
class SphericalExpansionMultiplicationTable;

/**
 * \class SphericalExpansion
 *
 * \brief Class for working with spherical harmonics expansions
 *
 * This class can be used for operating with linear combinations of
 * spherical harmonics. It can do, e.g., multiplication of spherical
 * harmonics using the closure relations.
 *
 * \author Jussi Lehtola
 * \date 2011/03/07 15:31
 */

class SphericalExpansion {
  /// Linear combination of spherical harmonics
  std::vector<ylmcoeff_t> comb;

  /// Add new Ylm with coefficient c to the linear combination
  void add(const ylmcoeff_t & c);

 public:
  /// Constructor
  SphericalExpansion();
  /// Destructor
  ~SphericalExpansion();

  /// Add new Ylm with coefficient c to the linear combination
  void addylm(int l, int m, complex c);
  /// Add new Ylm with coefficient c to the linear combination
  void addylm(int l, int m, double c);

  /// Clean out the expansion by removing any entries with zero coefficient
  void clean();
  /// Clear out everything
  void clear();
  /// Complex conjugate the expansion
  SphericalExpansion conjugate() const;
  /// Print out the expansion
  void print() const;
  /// Sort the combination in increasing l, increasing m
  void sort();

  /// Get amount of terms in the expansion
  size_t getN() const;
  /// Get i:th expansion coefficient
  ylmcoeff_t getcoeff(size_t i) const;
  /// Get expansion coefficients
  std::vector<ylmcoeff_t> getcoeffs() const;
  /// Get maximum value of l in expansion
  int getmaxl() const;

  /// Addition operator
  SphericalExpansion operator+(const SphericalExpansion & rhs) const;
  /// Increment operator
  SphericalExpansion & operator+=(const SphericalExpansion & rhs);

  /// Get negative of expansion
  SphericalExpansion operator-() const;

  /// Substraction operator
  SphericalExpansion operator-(const SphericalExpansion & rhs) const;
  /// Decrement operator
  SphericalExpansion & operator-=(const SphericalExpansion & rhs);

  /// Multiplication operator
  SphericalExpansion operator*(const SphericalExpansion & rhs) const;
  /// Multiplication operator
  SphericalExpansion & operator*=(const SphericalExpansion & rhs);
  /// Scale expansion by fac
  SphericalExpansion & operator*=(complex fac);
  /// Scale expansion by fac
  SphericalExpansion & operator*=(double fac);

  friend SphericalExpansion operator*(complex fac, const SphericalExpansion & func);
  friend SphericalExpansion operator*(double fac, const SphericalExpansion & func);
  friend class SphericalExpansionMultiplicationTable;
};

/// Scale expansion by fac
SphericalExpansion operator*(complex fac, const SphericalExpansion & func);
/// Scale expansion by fac
SphericalExpansion operator*(double fac, const SphericalExpansion & func);

// Forward declaration
class GTO_Fourier_Ylm;

/**
 * \class SphericalExpansionMultiplicationTable
 *
 * \brief Multiplication table of spherical harmonics.
 *
 * This class is used to speed up multiplication, since they only need
 * be computed once.
 *
 * \author Jussi Lehtola
 * \date 2011/03/07 15:31
 */

class SphericalExpansionMultiplicationTable {
  /// Multiplication table of spherical harmonics
  std::vector<SphericalExpansion> table;
  /// Maximum angular momentum supported
  int maxam;
 public:
  /// Construct multiplication table that supports spherical harmonics up to maxam
  SphericalExpansionMultiplicationTable(int maxam=6);
  /// Destructor
  ~SphericalExpansionMultiplicationTable();

  /// Print multiplication table
  void print() const;

  /// Multiplication operator
  SphericalExpansion mult(const SphericalExpansion & lhs, const SphericalExpansion & rhs) const;
  /// Multiplication operator
  GTO_Fourier_Ylm mult(const GTO_Fourier_Ylm & lhs, const GTO_Fourier_Ylm & rhs) const;
};

/// Spherical harmonics expansion of Fourier transform of GTO
typedef struct {
  /// Angular part
  SphericalExpansion ang;
  /// p^pm
  int pm;
  /// exp(-z*p^2);
  double z;
} GTO_Fourier_Ylm_t;


/**
 * \class GTO_Fourier_Ylm
 *
 * \brief Routines for expanding the Fourier transforms of GTOs in
 * spherical harmonics
 *
 * The expansion in spherical harmonics is used to compute the angular
 * integral over products of Gaussian Type Orbital basis functions.
 *
 * For a reference, see
 *
 * J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen, "Calculation of
 * isotropic Compton profiles with Gaussian basis sets",
 * Phys. Chem. Chem. Phys. 13 (2011), pp. 5630 - 5641.
 *
 * \author Jussi Lehtola
 * \date 2011/03/07 15:31
 */

class GTO_Fourier_Ylm {
  /// Spherical harmonics expansion of Fourier transform of GTO
  std::vector<GTO_Fourier_Ylm_t> sphexp;

 public:
  /// Constructor
  GTO_Fourier_Ylm();
  /// Compute Fourier transform of \f$ x^l y^m z^n \exp(-\zeta*r^2) \f$ and expand it in spherical harmonics
  GTO_Fourier_Ylm(int l, int m, int n, double zeta);
  /// Destructor
  ~GTO_Fourier_Ylm();

  /// Add term to expansion
  void addterm(const SphericalExpansion & ang, int pm, double z);
  /// Clean expansion by dropping terms with zero contribution
  void clean();
  /// Print out expansion
  void print() const;
 
  /// Get complex conjugate of expansion
  GTO_Fourier_Ylm conjugate() const;
  /// Get expansion in spherical harmonics
  std::vector<GTO_Fourier_Ylm_t> getexp() const;

  /// Multiplication operator
  GTO_Fourier_Ylm operator*(const GTO_Fourier_Ylm & rhs) const;
  /// Addition operator
  GTO_Fourier_Ylm operator+(const GTO_Fourier_Ylm & rhs) const;
  /// Increment operator
  GTO_Fourier_Ylm & operator+=(const GTO_Fourier_Ylm & rhs);

  friend GTO_Fourier_Ylm operator*(complex fac, const GTO_Fourier_Ylm & func);
  friend GTO_Fourier_Ylm operator*(double fac, const GTO_Fourier_Ylm & func);
  friend class SphericalExpansionMultiplicationTable;
};

/// Scale expansion by factor fac
GTO_Fourier_Ylm operator*(complex fac, const GTO_Fourier_Ylm & func);
/// Scale expansion by factor fac
GTO_Fourier_Ylm operator*(double fac, const GTO_Fourier_Ylm & func);

#endif
