/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#include "global.h"
#include "settings.h"

#ifndef ERKALE_LINALG
#define ERKALE_LINALG

#include <armadillo>

/**
 * Solve eigenvalues of symmetric matrix with guarantee
 * of eigenvalue ordering from smallest to biggest */
void eig_sym_ordered(arma::colvec & eigval, arma::mat & eigvec, const arma::mat & X);

/// Helper for sorts
typedef struct {
  /// Energy
  double E;
  /// Eigenvector
  arma::vec c;
} orbital_t;

/// Helper for sorts
bool operator<(const orbital_t & lhs, const orbital_t & rhs);

/// Helper for sorts
typedef struct {
  /// Energy
  double E;
  /// Eigenvector
  arma::cx_vec c;
} corbital_t;


/// Helper for sorts
bool operator<(const corbital_t & lhs, const corbital_t & rhs);

/// Sort vectors in order of increasing eigenvalue
void sort_eigvec(arma::colvec & eigval, arma::mat & eigvec);
/// Sort vectors in order of increasing eigenvalue
void sort_eigvec(arma::colvec & eigval, arma::cx_mat & eigvec);

/* Orthogonalization routines */

/// Cholesky orthogonalization of basis set
arma::mat CholeskyOrth(const arma::mat & S);
/// Symmetric orthogonalization of basis set
arma::mat SymmetricOrth(const arma::mat & S);
/// Same, but use computed decomoposition
arma::mat SymmetricOrth(const arma::mat & Svec, const arma::vec & Sval);
/// Canonical orthogonalization of basis set
arma::mat CanonicalOrth(const arma::mat & S, double cutoff=LINTHRES);
/// Same, but use computed decomposition
arma::mat CanonicalOrth(const arma::mat & Svec, const arma::vec & Sval, double cutoff);

/// Automatic orthonormalization.
arma::mat BasOrth(const arma::mat & S, bool verbose);
/// Orthogonalize basis
arma::mat BasOrth(const arma::mat & S, const Settings & set);

/// Form matrices S^1/2 and S^-1/2. By default matrices are computed in symmetric form, but canonical form is also available.
void S_half_invhalf(const arma::mat & S, arma::mat & Shalf, arma::mat & Sinvhalf, bool canonical=false, double cutoff=LINTHRES);

/// Transform matrix to vector
arma::vec MatToVec(const arma::mat & v);
/// Transform vector to matrix
arma::mat VecToMat(const arma::vec & v, size_t nrows, size_t ncols);

/// Get vector from cube: c(i,j,:)
arma::vec slicevec(const arma::cube & c, size_t i, size_t j);

/// Compute cosine of matrix
arma::mat cos(const arma::mat & U);
/// Compute sine of matrix
arma::mat sin(const arma::mat & U);
/// Compute sin(x)/x of matrix
arma::mat sinc(const arma::mat & U);
/// Compute square root of matrix
arma::mat sqrt(const arma::mat & M);

/// Orthogonalize
arma::mat orthogonalize(const arma::mat & M);
/// Unitarize
arma::cx_mat unitarize(const arma::cx_mat & M);

/// Orthonormalize vectors
arma::mat orthonormalize(const arma::mat & S, const arma::mat & C);

/// Incomplete Cholesky factorization of matrix M, use n vectors
arma::mat incomplete_cholesky(const arma::mat & M, size_t n);

#endif
