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



#include "mathf.h"
#include <cmath>
#include <cfloat>

// For exceptions
#include <sstream>
#include <stdexcept>


extern "C" {
  // For factorials and so on
#include <gsl/gsl_sf_gamma.h>
  // For spline interpolation
#include <gsl/gsl_spline.h>
}

double doublefact(int n) {
  if(n<-1) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Trying to compute double factorial for n="<<n<<"!"; 
    throw std::runtime_error(oss.str());
  }

  if(n>=-1 && n<=1)
    return 1.0;
  else
    return gsl_sf_doublefact(n);
}

double fact(int n) {
  if(n<0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Trying to compute factorial for n="<<n<<"\
!";
    throw std::runtime_error(oss.str());
  }

  return gsl_sf_fact(n);
}

double fact_ratio(int i, int r) {
  return gsl_sf_fact(i)/(gsl_sf_fact(r)*gsl_sf_fact(i-2*r));
}

double sinc(double x) {
  if(fabs(x)<100*DBL_EPSILON) {
    double x2=x*x;
    return 1.0 - x2/6.0 + x2*x2/140.0;
  } else
    return sin(x)/x;
}

double boysF(int m, double x) {
  // Compute Boys' function

  double x2=x*x;

  // Check whether we are operating in the range where x is small
  if( x2/(2*m+3) < 100*DBL_EPSILON/(2*m+1)  ) {
    // For small values of x the numeric evaluation leads to trouble
    // due to x^(-m-0.5). Use the first few terms of the Taylor series
    // instead, which is accurate enough

    return 1.0/(2*m+1.0) - x2/(2*m+3.0) + 0.5*x2*x2/(2*m+5.0);
  } else
    return 0.5*gsl_sf_gamma(m+0.5)*pow(x,-m-0.5)*gsl_sf_gamma_inc_P(m+0.5,x);
}

std::vector<double> boysF_arr(int mmax, double x) {

  // Returned array
  std::vector<double> F;

  // Resize array
  F.reserve(mmax+1);
  F.resize(mmax+1);

  // Fill in highest value
  F[mmax]=boysF(mmax,x);
  // and fill in the rest with downward recursion
  double emx=exp(-x);

  for(int m=mmax-1;m>=0;m--)
    F[m]=(2*x*F[m+1]+emx)/(2*m+1);

  return F;
}

double choose(int m, int n) {
  if(m<0 || n<0) {
    ERROR_INFO();
    throw std::domain_error("Choose called with a negative argument!\n");
  }

  return gsl_sf_choose(m,n);
}

int getind(int l, int m, int n) {
  int am=l+m+n;

  int l1, m1, n1;
  int ii, jj;

  int ind=0;

  for(ii = 0; ii <= am; ii++) {
    l1 = am - ii;

    for(jj = 0; jj <= ii; jj++) {
      m1 = ii - jj;
      n1 = jj;

      if((l==l1) && (m==m1) && (n==n1))
	return ind;

      ind++;
    }
  }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "Could not find index of basis function!\n";
  throw std::runtime_error(oss.str());
  
  return 0;
}

double max_abs(const arma::mat & R) {
  // Find maximum absolute value of matrix
  double m=0;
  double tmp;
  for(size_t i=0;i<R.n_rows;i++)
    for(size_t j=0;j<R.n_cols;j++) {
      tmp=fabs(R(i,j));
      if(tmp>m)
	m=tmp;
    }
  return m;
}

double rms_norm(const arma::mat & R) {
  double rms=0.0;
  for(size_t i=0;i<R.n_rows;i++)
    for(size_t j=0;j<R.n_cols;j++) {
      rms+=R(i,j)*R(i,j);
    }
  return sqrt(rms/(R.n_rows*R.n_cols));
}

double normsq(const arma::vec & v) {
  double n=0;
  for(size_t i=0;i<v.n_elem;i++)
    n+=v(i)*v(i);
  return n;
}

double norm(const arma::vec & v) {
  return sqrt(normsq(v));
}

void zero(std::vector<double> & x) {
  for(size_t i=0;i<x.size();i++)
    x[i]=0;
}

std::vector<double> spline_interpolation(const std::vector<double> & xt, const std::vector<double> & yt, const std::vector<double> & x) {
  if(xt.size()!=yt.size()) {
    ERROR_INFO();
    throw std::runtime_error("xt and yt are of different lengths!\n");
  }

  // Returned data
  std::vector<double> y(x.size());

  // Index accelerator
  gsl_interp_accel *acc=gsl_interp_accel_alloc();
  // Interpolant
  gsl_interp *interp=gsl_interp_alloc(gsl_interp_cspline,xt.size());

  // Initialize interpolant
  gsl_interp_init(interp,&(xt[0]),&(yt[0]),xt.size());
  
  // Perform interpolation.
  for(size_t i=0;i<x.size();i++)
    y[i]=gsl_interp_eval(interp,&(xt[0]),&(yt[0]),x[i],acc);

  // Free memory
  gsl_interp_accel_free(acc);
  gsl_interp_free(interp);

  return y;
}
