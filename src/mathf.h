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
#include <cfloat>
#include <string>

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
void boysF_arr(int mmax, double x, arma::vec & bf);

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

/// Find element
template <typename T> size_t find(const std::vector<T> & list, const T & val) {
  for(size_t i=0;i<list.size();i++)
    if(list[i]==val)
      return i;

  return std::string::npos;
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
/// Spline interpolate data (xt,yt) to points in x.
arma::vec spline_interpolation(const arma::vec & xt, const arma::vec & yt, const arma::vec & x);
/// Spline interpolate data (xt,yt) to x.
double spline_interpolation(const std::vector<double> & xt, const std::vector<double> & yt, double x);

/// Get random matrix
arma::mat randu_mat(size_t M, size_t N, unsigned long int seed=0);
/// Get random matrix
arma::mat randn_mat(size_t M, size_t N, unsigned long int seed=0);

/// Get random real orthogonal matrix of size N
arma::mat real_orthogonal(size_t N, unsigned long int seed=0);
/// Get random complex unitary matrix of size N
arma::cx_mat complex_unitary(size_t N, unsigned long int seed=0);

/// Round to n:th decimal
double round(double x, unsigned n);

/**
 * Find minima in input data, performing a running average over runave
 * nearest neighbors. Returns the locations of the minima.
 *
 * To further safeguard against spurious minima, the procedure screens
 * out minima with y values over the given threshold.
 */
arma::vec find_minima(const arma::vec & x, const arma::vec & y, size_t runave=0, double thr=DBL_MAX);

/// Sorted insertion
template<typename T> size_t sorted_insertion(std::vector<T> & v, T t) {
  // Get upper bound
  typename std::vector<T>::iterator high;
  high=std::upper_bound(v.begin(),v.end(),t);

  // Corresponding index is
  size_t ind=high-v.begin();

  if(ind>0 && v[ind-1]==t) {
    // Value already exists in vector - return position
    return ind-1;
  } else {
    // Value doesn't exist in vector - add it
    typename std::vector<T>::iterator pos=v.insert(high,t);
    // and return the position
    return pos-v.begin();
  }
}

/**
 * Get n point stencil for derivatives up to m:th order evaluated at
 * point z, with the values of the function known at points x
 *
 * This implementation is based on B. Fornberg, "Calculation of
 * weights in finite difference formulas", SIAM Rev. 40, 685
 * (1998).
 */
void stencil(double z, const arma::vec & x, arma::mat & w);

#endif
