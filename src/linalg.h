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
class Settings;

#ifndef ERKALE_LINALG
#define ERKALE_LINALG

#include <armadillo>

/// Helper for eigenvector sorts
template<typename T> struct eigenvector {
  /// Energy
  double E;
  /// Eigenvector
  arma::Col<T> c;
};

/// Helper for sorts
template<typename T> inline bool operator<(const struct eigenvector<T> & lhs, const struct eigenvector<T> & rhs) {
  return lhs.E < rhs.E;
}

/// Sort vectors in order of increasing eigenvalue
template<typename T> inline void sort_eigvec_wrk(arma::vec & eigval, arma::Mat<T> & eigvec) {
  if(eigval.n_elem != eigvec.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Eigenvalue vector does not correspond to eigenvector matrix!\n");
  }

  // Helper
  std::vector< struct eigenvector<T> > orbs(eigval.n_elem);
  for(size_t io=0;io<eigval.n_elem;io++) {
    orbs[io].E=eigval(io);
    orbs[io].c=eigvec.col(io);
  }
  std::stable_sort(orbs.begin(),orbs.end());
  for(size_t io=0;io<eigval.n_elem;io++) {
    eigval(io)=orbs[io].E;
    eigvec.col(io)=orbs[io].c;
  }
}

/// Wrapper
void sort_eigvec(arma::vec & eigval, arma::mat & eigvec);
/// Sort vectors in order of increasing eigenvalue
void sort_eigvec(arma::vec & eigval, arma::cx_mat & eigvec);

/**
 * Solve eigenvalues of symmetric matrix with guarantee
 * of eigenvalue ordering from smallest to biggest */
template<typename T> inline void eig_sym_ordered_wrk(arma::vec & eigval, arma::Mat<T> & eigvec, const arma::Mat<T> & X) {
  /* Solve eigenvalues of symmetric matrix with guarantee
     of ordering of eigenvalues from smallest to biggest */

  // Solve eigenvalues and eigenvectors
  bool ok=arma::eig_sym(eigval,eigvec,X);
  if(!ok) {
    ERROR_INFO();
    printf("Unable to diagonalize matrix!\n");
    X.print("X");
    throw std::runtime_error("Error in eig_sym.\n");
  }

  // Sort vectors
  sort_eigvec_wrk<T>(eigval,eigvec);
}


/**
 * Solve eigenvalues of symmetric matrix with guarantee
 * of eigenvalue ordering from smallest to biggest */
void eig_sym_ordered(arma::vec & eigval, arma::mat & eigvec, const arma::mat & X);
/**
 * Solve eigenvalues of Hermitian matrix with guarantee
 * of eigenvalue ordering from smallest to biggest */
void eig_sym_ordered(arma::vec & eigval, arma::cx_mat & eigvec, const arma::cx_mat & X);

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
arma::mat cosmat(const arma::mat & M);
/// Compute sine of matrix
arma::mat sinmat(const arma::mat & M);
/// Compute sin(x)/x of matrix
arma::mat sincmat(const arma::mat & M);
/// Compute square root of matrix
arma::mat sqrtmat(const arma::mat & M);
/// Compute exponential of matrix
arma::mat expmat(const arma::mat & M);
/// Compute exponential of matrix
arma::cx_mat expmat(const arma::cx_mat & M);

/// Orthogonalize
arma::mat orthogonalize(const arma::mat & M);
/// Unitarize
arma::cx_mat unitarize(const arma::cx_mat & M);

/// Orthonormalize vectors
arma::mat orthonormalize(const arma::mat & S, const arma::mat & C);

/**
 * Find natural orbitals from P. Orbitals returned in decreasing occupation number.
 */
void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::mat & NO_to_AO, arma::vec & occs);

/**
 * Find natural orbitals from P. Orbitals returned in decreasing occupation number.
 */
void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::vec & occs);

/**
 * Does a pivoted Cholesky decomposition. The algorithm is adapted
 * from the reference
 *
 * H. Harbrecht, M. Peters, and R. Schneider, "On the low-rank
 * approximation by the pivoted Cholesky decomposition",
 * Appl. Num. Math. 62, 428 (2012).
 */
arma::mat pivoted_cholesky(const arma::mat & M, double thr, arma::uvec & pivot);
/// Same as above but doesn't return pivot
arma::mat pivoted_cholesky(const arma::mat & M, double thr);
/// Incomplete Cholesky factorization of matrix M, use n vectors
arma::mat incomplete_cholesky(const arma::mat & M, size_t n);

/// Transform B matrix to MO basis
arma::mat B_transform(arma::mat B, const arma::mat & Cl, const arma::mat & Cr);

/// Check thread safety of LAPACK library
void check_lapack_thread();

#endif
