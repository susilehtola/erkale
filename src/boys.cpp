#include "boys.h"
#include "mathf.h"
#include <cmath>

extern "C" {
  // For factorials and so on
#include <gsl/gsl_sf_gamma.h>
}

namespace BoysTable {
  /// Table holding Boys function values
  arma::mat data;
  /// Table holding the constant prefactor for the asymptotic formula
  arma::vec prefac;

  /// Maximum m value
  int mmax;
  /// Order of expansion
  int order;
  /// Tabulation interval
  double dx;
  /// Upper limit of table
  double xmax;
}

void BoysTable::fill(int mmax_, int order_, double dx_, double xmax_) {
  if(data.n_rows==(arma::uword) (mmax_+order_+1))
    return;
  
  // Set values
  mmax=mmax_;
  order=order_;
  dx=dx_;
  xmax=xmax_;
  
  // Calculate number of entries
  size_t N=ceil(xmax/dx+1);
  
  // Calculate prefactors
  prefac.zeros(mmax+1);
  for(int m=0;m<=mmax;m++)
    prefac(m)=doublefact(2*m-1)/pow(2.0,m+1)*sqrt(M_PI);
  
  // Allocate table
  data.zeros(mmax+order+1,N);
  
  // x=0
  for(int m=0;m<=mmax+order;m++)
    data(m,0)=1.0/(2*m+1);
  
  // Fill table
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t ix=1;ix<N;ix++) {
    // x value is
    double x=ix*dx;
    for(int m=0;m<=mmax+order;m++) {
      // We don't use any recursion here to avoid numeric instabilities in the reference values
      data(m,ix)=0.5*gsl_sf_gamma(m+0.5)*pow(x,-m-0.5)*gsl_sf_gamma_inc_P(m+0.5,x);
    }
  }
}

double BoysTable::eval(int m, double x) {
  if(x>=xmax)
    // Use asymptotic expansion
    return prefac(m)/(sqrt(x)*std::pow(x,m));

  // Calculate index
  size_t idx=round(x/dx);

  // Delta x
  double delta=x-idx*dx;

  double F=0.0;
  double kfac=1.0;
  double mdeltapk=1.0;
  for(int k=0;k<order;k++) {
    // Increment value
    F+=data(m+k,idx)*mdeltapk/kfac;
    // Increment -(delta x)^k
    mdeltapk*=-delta;
    // Increment k!
    kfac*=(k+1);
  }

  return F;
}

void BoysTable::eval(int mx, double x, arma::vec & F) {
  // Resize array
  F.zeros(mx+1);
  // Exp(-x) for recursion
  double emx=exp(-x);

  if(x<mx) {
    // Fill in highest value
    F[mx]=eval(mx,x);
    // and fill in the rest with downward recursion
    for(int m=mx-1;m>=0;m--)
      F[m]=(2*x*F[m+1]+emx)/(2*m+1);
  } else {
    // Fill in lowest value
    F[0]=eval(0,x);
    // and fill in the rest with upward recursion
    for(int m=1;m<=mx;m++)
      F[m]=((2*m-1)*F[m-1]-emx)/(2.0*x);
  }
}
