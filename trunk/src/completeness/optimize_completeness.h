/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_OPTCOMP
#define ERKALE_OPTCOMP

#include <vector>

extern "C" {
#include <gsl/gsl_vector.h>
}

/// Parameters for completeness scan
typedef struct {
  /// Angular momentum of shell
  int am;
  /// Scanning exponents
  std::vector<double> scanexp;
} completeness_scan_t;

/// Helper function - evaluate completeness. v holds logarithms of exponents, params is pointer to completeness_scan_t
double evaluate_completeness(const gsl_vector *v, void *params);
/// Wrapper for the above
double evaluate_completeness(const std::vector<double> & v, completeness_scan_t p);

/// Find out exponents in completeness optimized basis set.
std::vector<double> optimize_completeness(int am, double min, double max, int Nf);
/// Same, using algorithms in GSL
std::vector<double> optimize_completeness_gsl(int am, double min, double max, int Nf);


#endif
