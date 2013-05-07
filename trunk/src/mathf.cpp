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
  // For Bessel functions
#include <gsl/gsl_sf_bessel.h>
  // For hypergeometric functions
#include <gsl/gsl_sf_hyperg.h>
  // For trigonometric functions
#include <gsl/gsl_sf_trig.h>
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

double fgamma(double x) {
  return gsl_sf_gamma(x);
}

double sinc(double x) {
  return gsl_sf_sinc(x/M_PI);
}

double bessel_jl(int l, double x) {
  // Small x: use series expansion
  double series=pow(x,l)/doublefact(2*l+1);
  if(fabs(series)<sqrt(DBL_EPSILON))
    return series;

  // Large x: truncate due to incorrect GSL asymptotics
  // (GSL bug https://savannah.gnu.org/bugs/index.php?36152)
  if(x>1.0/DBL_EPSILON)
    return 0.0;

  // Default case: use GSL
  return gsl_sf_bessel_jl(l,x);
}

double boysF(int m, double x) {
  // Compute Boys' function

  // Check whether we are operating in the range where the Taylor series is OK
  if( x<=5.5 ) {

    // Taylor series uses -x
    x=-x;

    // (-x)^k
    double xk=1.0;
    // k!
    double kf=1.0;
    // Value of Boys' function
    double fm=0.0;
    // index k
    int k=0;

    while(k<=38) {
      // \f$ F_m(x) = \sum_{k=0}^\infty \frac {(-x)^k} { k! (2m+2k+1)} \f$
      fm+=xk/(kf*(2*(m+k)+1));

      k++;
      xk*=x;
      kf*=k;
    }

    return fm;

  } else if(x>=40.0) {
    // Use asymptotic expansion, which is precise to <1e-16 for F_n(x), n = 0 .. 60
    return doublefact(2*m-1)/pow(2,m+1)*sqrt(M_PI/pow(x,2*m+1));
  } else
    // Need to use the exact formula
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

double hyperg_1F1(double a, double b, double x) {
  // Handle possible underflows with a Kummer transformation
  if(x>=-500.0) {
    return gsl_sf_hyperg_1F1(a,b,x);
  } else {
    return exp(x)*gsl_sf_hyperg_1F1(b-a,b,-x);
  }
}

double choose(int m, int n) {
  if(m<0 || n<0) {
    ERROR_INFO();
    throw std::domain_error("Choose called with a negative argument!\n");
  }

  return gsl_sf_choose(m,n);
}

int getind(int l, int m, int n) {
  // Silence compiler warning
  (void) l;
  // In the loop the indices will be
  int ii=m+n;
  int jj=n;
  // so the corresponding index is
  return ii*(ii+1)/2 + jj;
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
  // Calculate \sum_ij R_ij^2
  double rms=arma::trace(arma::trans(R)*R)/(R.n_rows*R.n_cols);
  // and convert to rms
  rms=sqrt(rms/(R.n_rows*R.n_cols));

  return rms;
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
  for(size_t i=0;i<x.size();i++) {
    y[i]=gsl_interp_eval(interp,&(xt[0]),&(yt[0]),x[i],acc);
  }

  // Free memory
  gsl_interp_accel_free(acc);
  gsl_interp_free(interp);

  return y;
}
