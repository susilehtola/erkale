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
  // Random number generation
#include <gsl/gsl_rng.h>
  // Random number distributions
#include <gsl/gsl_randist.h>
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
  if( x<=1.0 ) {

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

    while(k<=15) {
      // \f$ F_m(x) = \sum_{k=0}^\infty \frac {(-x)^k} { k! (2m+2k+1)} \f$
      fm+=xk/(kf*(2*(m+k)+1));

      k++;
      xk*=x;
      kf*=k;
    }

    return fm;

  } else if(x>=38.0) {
    // Use asymptotic expansion, which is precise to <1e-16 for F_n(x), n = 0 .. 60
    return doublefact(2*m-1)/pow(2,m+1)*sqrt(M_PI/pow(x,2*m+1));

  } else
    // Need to use the exact formula
    return 0.5*gsl_sf_gamma(m+0.5)*pow(x,-m-0.5)*gsl_sf_gamma_inc_P(m+0.5,x);
}

void boysF_arr(int mmax, double x, arma::vec & F) {
  // Resize array
  F.zeros(mmax+1);
  // Exp(-x) for recursion
  double emx=exp(-x);

  if(x<mmax) {
    // Fill in highest value
    F[mmax]=boysF(mmax,x);
    // and fill in the rest with downward recursion
    for(int m=mmax-1;m>=0;m--)
      F[m]=(2*x*F[m+1]+emx)/(2*m+1);
  } else {
    // Fill in lowest value
    F[0]=boysF(0,x);
    // and fill in the rest with upward recursion
    for(int m=1;m<=mmax;m++)
      F[m]=((2*m-1)*F[m-1]-emx)/(2.0*x);
  }
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
  return arma::max(arma::max(arma::abs(R)));
}

double max_cabs(const arma::cx_mat & R) {
  return arma::max(arma::max(arma::abs(R)));
}

double rms_norm(const arma::mat & R) {
  // Calculate \sum_ij R_ij^2
  double rms=arma::trace(arma::trans(R)*R);
  // and convert to rms
  rms=sqrt(rms/(R.n_rows*R.n_cols));

  return rms;
}

double rms_cnorm(const arma::cx_mat & R) {
  // Calculate \sum_ij R_ij^2
  double rms=std::abs(arma::trace(arma::trans(R)*R));
  // and convert to rms
  rms=sqrt(rms/(R.n_rows*R.n_cols));

  return rms;
}

std::vector<double> spline_interpolation(const std::vector<double> & xt, const std::vector<double> & yt, const std::vector<double> & x) {
  if(xt.size()!=yt.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "xt and yt are of different lengths - " << xt.size() << " vs " << yt.size() << "!\n";
    throw std::runtime_error(oss.str());
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

arma::vec spline_interpolation(const arma::vec & xtv, const arma::vec & ytv, const arma::vec & xv) {
  return arma::conv_to<arma::colvec>::from(spline_interpolation(arma::conv_to< std::vector<double> >::from(xtv), arma::conv_to< std::vector<double> >::from(ytv), arma::conv_to< std::vector<double> >::from(xv)));
}

double spline_interpolation(const std::vector<double> & xt, const std::vector<double> & yt, double x) {
  if(xt.size()!=yt.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "xt and yt are of different lengths - " << xt.size() << " vs " << yt.size() << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Index accelerator
  gsl_interp_accel *acc=gsl_interp_accel_alloc();
  // Interpolant
  gsl_interp *interp=gsl_interp_alloc(gsl_interp_cspline,xt.size());

  // Initialize interpolant
  gsl_interp_init(interp,&(xt[0]),&(yt[0]),xt.size());

  // Perform interpolation.
  double y=gsl_interp_eval(interp,&(xt[0]),&(yt[0]),x,acc);

  // Free memory
  gsl_interp_accel_free(acc);
  gsl_interp_free(interp);

  return y;
}

arma::mat randu_mat(size_t M, size_t N, unsigned long int seed) {
  // Use Mersenne Twister algorithm
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  // Set seed
  gsl_rng_set(r,seed);

  // Matrix
  arma::mat mat(M,N);
  // Fill it
  for(size_t i=0;i<M;i++)
    for(size_t j=0;j<N;j++)
      mat(i,j)=gsl_rng_uniform(r);

  // Free rng
  gsl_rng_free(r);
  return mat;
}

arma::mat randn_mat(size_t M, size_t N, unsigned long int seed) {
  // Use Mersenne Twister algorithm
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  // Set seed
  gsl_rng_set(r,seed);

  // Matrix
  arma::mat mat(M,N);
  // Fill it
  for(size_t i=0;i<M;i++)
    for(size_t j=0;j<N;j++)
      mat(i,j)=gsl_ran_gaussian(r,1.0);

  // Free rng
  gsl_rng_free(r);
  return mat;
}

arma::mat real_orthogonal(size_t N, unsigned long int seed) {
  arma::mat U(N,N);
  U.zeros();

  // Generate totally random matrix
  arma::mat A=randn_mat(N,N,seed);

  // Perform QR decomposition on matrix
  arma::mat Q, R;
  bool ok=arma::qr(Q,R,A);
  if(!ok) {
    ERROR_INFO();
    throw std::runtime_error("QR decomposition failure in complex_unitary.\n");
  }

  // Check that Q is orthogonal
  arma::mat test=Q*arma::trans(Q);
  for(size_t i=0;i<test.n_cols;i++)
    test(i,i)-=1.0;
  double n=rms_norm(test);
  if(n>10*DBL_EPSILON) {
    ERROR_INFO();
    throw std::runtime_error("Generated matrix is not unitary!\n");
  }

  return Q;
}

arma::cx_mat complex_unitary(size_t N, unsigned long int seed) {
  arma::cx_mat U(N,N);
  U.zeros();

  // Generate totally random matrix
  arma::cx_mat A=std::complex<double>(1.0,0.0)*randn_mat(N,N,seed) + std::complex<double>(0.0,1.0)*randn_mat(N,N,seed+1);

  // Perform QR decomposition on matrix
  arma::cx_mat Q, R;
  bool ok=arma::qr(Q,R,A);
  if(!ok) {
    ERROR_INFO();
    throw std::runtime_error("QR decomposition failure in complex_unitary.\n");
  }

  // Check that Q is unitary
  arma::cx_mat test=Q*arma::trans(Q);
  for(size_t i=0;i<test.n_cols;i++)
    test(i,i)-=1.0;
  double n=rms_cnorm(test);
  if(n>10*DBL_EPSILON) {
    ERROR_INFO();
    throw std::runtime_error("Generated matrix is not unitary!\n");
  }

  return Q;
}

double round(double x, unsigned n) {
  double fac=pow(10.0,n);
  return round(fac*x)/fac;
}

arma::vec find_minima(const arma::vec & x, const arma::vec & y, size_t runave, double thr) {
  if(x.n_elem != y.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Input vectors are of inconsistent size!\n");
  }

  // Create averaged vectors
  arma::vec xave(x.n_elem-2*runave);
  arma::vec yave(y.n_elem-2*runave);
  if(runave==0) {
    xave=x;
    yave=y;
  } else {
    for(arma::uword i=runave;i<x.n_elem-runave;i++) {
      xave(i-runave)=arma::mean(x.subvec(i-runave,i+runave));
      yave(i-runave)=arma::mean(y.subvec(i-runave,i+runave));
    }
  }

  // Find minima
  std::vector<size_t> minloc;
  if(yave(0)<yave(1))
    minloc.push_back(0);

  for(arma::uword i=1;i<yave.n_elem-1;i++)
    if(yave(i)<yave(i-1) && yave(i)<yave(i+1))
      minloc.push_back(i);

  if(yave(yave.n_elem-1)<yave(yave.n_elem-2))
    minloc.push_back(yave.n_elem-1);

  // Check the minimum values
  for(size_t i=minloc.size()-1;i<minloc.size();i--)
    if(yave(minloc[i]) >= thr)
      minloc.erase(minloc.begin()+i);

  // Returned minima
  arma::vec ret(minloc.size());
  for(size_t i=0;i<minloc.size();i++)
    ret(i)=xave(minloc[i]);

  return ret;
}

void stencil(double z, const arma::vec & x, arma::mat & w) {
  /*
    z: location where approximations are wanted
    x: grid points at which value of function is known
    w: matrix holding weights for grid points for different orders
       of the derivatives: w(x, order)

       This implementation is based on B. Fornberg, "Calculation of
       weights in finite difference formulas", SIAM Rev. 40, 685
       (1998).
   */

  /* Extract variables */
  size_t n = w.n_rows - 1;
  size_t m = w.n_cols - 1;

  /* Sanity check */
  if(w.n_rows != x.n_elem)
    throw std::logic_error("Grid points and weight matrix sizes aren't compatible!\n");

  /* Initialize elements */
  double c1 = 1.0;
  double c4 = x(0)-z;
  w.zeros();
  w(0,0)=1.0;

  for(size_t i=1;i<=n;i++) {
    size_t mn = std::min(i,m);
    double c2 = 1.0;
    double c5 = c4;
    c4 = x(i) - z;

    for(size_t j=0;j<i;j++) {
      double c3 = x(i) - x(j);
      c2 *= c3;

      if(j == i-1) {
	for(size_t k=mn;k>0;k--)
	  w(i,k) = c1*(k*w(i-1,k-1) - c5*w(i-1,k))/c2;
	w(i,0)=-c1*c5*w(i-1,0)/c2;
      }
      for(size_t k=mn;k>0;k--)
	w(j,k)=(c4*w(j,k) - k*w(j,k-1))/c3;
      w(j,0)=c4*w(j,0)/c3;
    }
    c1 = c2;
  }
}
