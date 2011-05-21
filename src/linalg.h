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



#include "global.h"
#include "settings.h"

#ifndef ERKALE_LINALG
#define ERKALE_LINALG

#include <armadillo>

/**
 * Solve eigenvalues of symmetric matrix with guarantee
 *  of ordering of eigenvalues from smallest to biggest */
void eig_sym_ordered(arma::colvec & eigval, arma::mat & eigvec, const arma::mat & X);

/// Sort vectors in order of increasing eigenvalue
void sort_eigvec(arma::colvec & eigval, arma::mat & eigvec);

/* Orthogonalization routines */

/// Orthogonalize basis
arma::mat BasOrth(const arma::mat & S, const Settings & set, double & smallest);

/// Cholesky orthogonalization of basis set
arma::mat CholeskyOrth(const arma::mat & S, double *smallest=NULL);
/// Symmetric orthogonalization of basis set
arma::mat SymmetricOrth(const arma::mat & S, double *smallest=NULL);
/// Canonical orthogonalization of basis set
arma::mat CanonicalOrth(const arma::mat & S, double cutoff=1e-4, double *smallest=NULL);

/// Form symmetric matrices S^1/2 and S^-1/2
void S_half_invhalf(const arma::mat & S, arma::mat & Shalf, arma::mat & Sinvhalf, double cutoff);

/// Transform matrix to vector
arma::vec MatToVec(const arma::mat & v);
/// Transform vector to matrix
arma::mat VecToMat(const arma::vec & v, size_t nrows, size_t ncols);

/// Compute cosine of matrix
arma::mat cos(const arma::mat & U);
/// Compute sine of matrix
arma::mat sin(const arma::mat & U);
/// Compute sin(x)/x of matrix
arma::mat sinc(const arma::mat & U);
/// Compute square root of matrix
arma::mat sqrt(const arma::mat & M);

#endif
