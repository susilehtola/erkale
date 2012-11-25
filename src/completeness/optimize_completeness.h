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
#define MINTAU pow(10.0,-4.8)

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
  std::vector<double> scanexp;
} completeness_scan_t;

/// Get exponents. x contains the natural logarithms
std::vector<double> get_exponents(const gsl_vector *x);

/// Compute self-overlap \f$ S_{ij} \f$
arma::mat self_overlap(const std::vector<double> & z, int am);

/// Compute derivative of inverse overlap
std::vector<arma::mat> self_inv_overlap_logder(const arma::mat & Sinv, const arma::mat & D);

/**
 * Compute overlap derivative matrix \f$ D_{\mu \nu} =
 * \partial_{\log \zeta_\mu} S_{\mu \nu } = \partial_{\log \zeta_\mu} S_{\nu \mu} \f$
 * (this also works for M)
 */
arma::mat overlap_logder(const std::vector<double> & z, const std::vector<double> & zp, const arma::mat & S, int am);

/// Compute completeness profile
std::vector<double> completeness_profile(const gsl_vector * x, void * params);

/// Compute derivative of completeness profile
std::vector< std::vector<double> > completeness_profile_logder(const gsl_vector * x, void * params);

/// Evaluate measure of goodness
double compl_mog(const gsl_vector * x, void * params);
/// Evaluate derivative of measure of goodness
void compl_mog_df(const gsl_vector * x, void * params, gsl_vector *g);
/// Evaluate mog and its derivative
void compl_mog_fdf(const gsl_vector * x, void * params, double *f, gsl_vector *g);

/**
 * Optimize completeness profile for angular momentum am in exponent
 * range from 10^{min} to 10^{max}. n gives the moment to optimize for
 * (1 for maximal area, 2 for minimal rms deviation from unity).
 *
 * This routine uses the Nead-Miller algorithm.
 */
std::vector<double> optimize_completeness(int am, double min, double max, int Nf, int n=1, bool verbose=true, double *mog=NULL);

/// Calculate maximum width to obtain tolerance with given amount of exponents
double maxwidth(int am, double tol, int nexp, int n=1);

/// Calculate exponents corresponding to maximum width to obtain tolerance with given amount of exponents
std::vector<double> maxwidth_exps(int am, double tol, int nexp, double *width, int n=1);

/// Perform completeness-optimization of exponents
std::vector<double> get_exponents(int am, double start, double end, double tol, int n=1, bool verbose=false);


/**
 * Optimize completeness profile for angular momentum am in exponent
 * range from 10^{min} to 10^{max}. n gives the moment to optimize for
 * (1 for maximal area, 2 for minimal rms deviation from unity).
 *
 * This routine uses steepest descent using quasianalytical
 * derivatives. However, this really doesn't seem to bring any added
 * benefit compared to the above, as first of all the derivatives are
 * expensive to calculate and GSL routines have trouble minimizing the
 * problem.
 */
std::vector<double> optimize_completeness_df(int am, double min, double max, int Nf, int n=2);

#endif
