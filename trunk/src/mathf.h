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

#ifndef ERKALE_MATH
#define ERKALE_MATH

#include <armadillo>
#include <vector>

/// Computes the double factorial n!!
double doublefact(int n);
/// Computes the factorial n!
double fact(int n);
/// Computes the ratio i!/(r!*(i-2r)!)
double fact_ratio(int i, int r);

/// Compute sinc = sin(x)/x
double sinc(double x);

/// Computes the Boys function
double boysF(int m, double x);
/// Computes an array of Boys function
std::vector<double> boysF_arr(int mmax, double x);

/// Computes the binomial coefficient
double choose(int m, int n);

/// Find index of basis function with indices l, m, n
int getind(int l, int m, int n);

/// Compute RMS difference of matrices
double rms_diff(const arma::mat & R, const arma::mat & Rold);
/// Compute maximum difference of matrices
double max_diff(const arma::mat & R, const arma::mat & Rold);

/// Get maximum
template <class T> T max(const T & a, const T & b) {
  return (b<a ?a:b);
}

/// Get maximum
template <class T> T max(const T & a, const T & b, const T & c, const T & d) {
  return max(max(a,b),max(c,d));
}

/// Get maximum element of x
double max(std::vector<double> x);
/// Get element with maximum absolute value
double max_abs(const arma::mat & R);

/// Get minimum
template <class T> T min(const T & a, const T & b) {
  return (b>a ? a:b);
}

/// Zero array
void zero(std::vector<double> & x);

/// Reverse array
template <class T> void reverse(std::vector<T> & a) {
  size_t i;
  size_t n=a.size();
  T temp;
  for(i=0;i<n/2;i++) {
    temp=a[i];
    a[i]=a[n-1-i];
    a[n-1-i]=temp;
  }
}

#endif
