#include "boys.h"
#include "mathf.h"
#include <cmath>

BoysTable::BoysTable() {
}

BoysTable::~BoysTable() {
}

void BoysTable::fill(int mmax_, int order_, double dx_, double xmax_) {
  // Set values
  mmax=mmax_;
  order=order_;
  dx=dx_;
  xmax=xmax_;

  // Calculate number of entries
  size_t N=ceil(xmax/dx+1);

  // Calculate prefactors
  prefac.zeros(mmax);
  for(int m=0;m<=mmax;m++)
    prefac(m)=doublefact(2*m-1)/pow(2.0,m+1)*sqrt(M_PI);
    
  // Allocate table
  data.zeros(mmax+order+1,N);

  // Fill table
  for(size_t ix=0;ix<N;ix++) {
    // x value is
    double x=ix*dx;

    // Evaluate BoysTable function
    arma::vec bf;
    boysF_arr(mmax+order,x,bf);

    // Store values
    data.col(ix)=bf;
  }
}

double BoysTable::eval(int m, double x) const {
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

void BoysTable::eval(int mx, double x, arma::vec & F) const {
  // Resize array
  F.zeros(mx+1);

  // Fill in highest value
  F[mmax]=eval(mx,x);
  // and fill in the rest with downward recursion
  double emx=exp(-x);

  for(int m=mx-1;m>=0;m--)
    F[m]=(2*x*F[m+1]+emx)/(2*m+1);
}
