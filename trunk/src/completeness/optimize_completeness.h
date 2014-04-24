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

#ifndef ERKALE_OPTCOMP
#define ERKALE_OPTCOMP

/// Minimum allowed value of deviation from completeness (for numerical stability)
#define MINTAU pow(10.0,-6.0)

#include "../global.h"
#include <armadillo>
#include <vector>

extern "C" {
#include <gsl/gsl_vector.h>
}

/// Parameters for completeness scan
typedef struct {
  /// Angular momentum of shell to optimize
  int am;
  /// Which moment to optimize
  int n;

  /// Scanning exponents to optimize against
  arma::vec scanexp;

  /// Add in exponent at the center?
  bool odd;
  /// Amount of even-tempered exponents near the center
  size_t neven;
  /// Amount of fully optimized exponents at the edges
  size_t nfull;
} completeness_scan_t;

/// Get exponents. x contains the natural logarithms
arma::vec get_exponents(const gsl_vector *x, const completeness_scan_t & p);

/// Compute self-overlap \f$ S_{ij} \f$
arma::mat self_overlap(const arma::vec & z, int am);

/// Compute completeness profile
arma::vec completeness_profile(const gsl_vector * x, void * params);

/// Evaluate measure of goodness
double compl_mog(const gsl_vector * x, void * params);

/// Evaluate gradient of measure of goodness
void compl_mog_df(const gsl_vector * x, void * params, gsl_vector * g);

/// Evaluate gradient and measure of goodness
void compl_mog_fdf(const gsl_vector * x, void * params, double * f, gsl_vector * df);

/**
 * Optimize completeness profile for angular momentum am in exponent
 * range from 10^{min} to 10^{max}. n gives the moment to optimize for
 * (1 for maximal area, 2 for minimal rms deviation from unity).
 *
 * By default, only the 4 outermost exponents on each end are fully
 * optimized, while the inner exponents are well described by an
 * even-tempered formula. If nfull>=Nf, a full optimization is performed.
 */
arma::vec optimize_completeness(int am, double min, double max, int Nf, int n=1, bool verbose=true, double *mog=NULL, int nfull=4);

/**
 * Optimize completeness profile for angular momentum am in exponent
 * range from 10^{min} to 10^{max}. n gives the moment to optimize for
 * (1 for maximal area, 2 for minimal rms deviation from unity).
 *
 * This routine uses Fletcher-Reeves conjugate gradients.
 *
 * By default, only the 4 outermost exponents on each end are fully
 * optimized, while the inner exponents are well described by an
 * even-tempered formula. If nfull>=Nf, a full optimization is performed.
 */
arma::vec optimize_completeness_cg(int am, double min, double max, int Nf, int n=1, bool verbose=true, double *mog=NULL, int nfull=4);

/**
 * Optimize completeness profile for angular momentum am in exponent
 * range from 10^{min} to 10^{max}. n gives the moment to optimize for
 * (1 for maximal area, 2 for minimal rms deviation from unity).
 *
 * This is the old version of the routine, which uses the Nead-Miller algorithm.
 *
 * By default, only the 4 outermost exponents on each end are fully
 * optimized, while the inner exponents are well described by an
 * even-tempered formula. If nfull>=Nf, a full optimization is
 * performed.
 */
arma::vec optimize_completeness_simplex(int am, double min, double max, int Nf, int n=1, bool verbose=true, double *mog=NULL, int nfull=4);

/// Calculate maximum width to obtain tolerance with given amount of exponents
double maxwidth(int am, double tol, int nexp, int n=1, int nfull=4);

/// Calculate exponents corresponding to maximum width to obtain tolerance with given amount of exponents
arma::vec maxwidth_exps(int am, double tol, int nexp, double *width, int n=1, int nfull=4);

/// Perform completeness-optimization of exponents
arma::vec get_exponents(int am, double start, double end, double tol, int n=1, bool verbose=false, int nfull=4);

#endif
