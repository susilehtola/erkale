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

/// Compute the Gamma function
double fgamma(double x);

/// Compute sinc = sin(x)/x
double sinc(double x);

/// Compute the spherical Bessel function
double bessel_jl(int l, double x);

/// Computes the Boys function
double boysF(int m, double x);
/// Computes an array of Boys function
std::vector<double> boysF_arr(int mmax, double x);

/// Computes the confluent hypergeometric function
double hyperg_1F1(double a, double b, double x);

/// Computes the binomial coefficient
double choose(int m, int n);

/// Find index of basis function with indices l, m, n
int getind(int l, int m, int n);

/// Get maximum element of x
template <class T> T max(const std::vector<T> & x) {
  if(x.size()==0) {
    ERROR_INFO();
    throw std::runtime_error("Trying to get maximum value of empty array!\n");
  }

  T m=x[0];
  for(size_t i=1;i<x.size();i++)
    if(m<x[i])
      m=x[i];
  return m;
}

/// Get maximum
template <class T> T max4(const T & a, const T & b, const T & c, const T & d) {
  return std::max(std::max(a,b),std::max(c,d));
}

/// Get element with maximum absolute value
double max_abs(const arma::mat & R);
/// Get element with maximum absolute value
double max_cabs(const arma::cx_mat & R);
/// Compute rms norm of matrix
double rms_norm(const arma::mat & R);
/// Compute rms norm of matrix
double rms_cnorm(const arma::cx_mat & R);

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

/// Sum over array
template <class T> T sum(const std::vector<T> & a) {
  T sum=0;
  for(size_t i=0;i<a.size();i++) {
    sum+=a[i];
  }
  return sum;
}

/// Check magnitude
template <class T> bool abscomp(const std::complex<T> & a, const std::complex<T> & b) {
  return std::abs(a) < std::abs(b);
}

/// Spline interpolate data (xt,yt) to points in x.
std::vector<double> spline_interpolation(const std::vector<double> & xt, const std::vector<double> & yt, const std::vector<double> & x);

/// Get random matrix
arma::mat randu_mat(size_t M, size_t N, unsigned long int seed=0);
/// Get random matrix
arma::mat randn_mat(size_t M, size_t N, unsigned long int seed=0);

/// Get complex unitary matrix of size N
arma::cx_mat complex_unitary(size_t N, unsigned long int seed=0);

#endif
