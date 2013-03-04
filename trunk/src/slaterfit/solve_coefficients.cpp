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



#include "solve_coefficients.h"

/* Hypergeometric functions */
#include <gsl/gsl_sf_hyperg.h>
/* Gamma functions */
#include <gsl/gsl_sf_gamma.h>


arma::vec form_P(const std::vector<double> & expns, double zeta, int l) {
  // Amount of exponents
  const size_t N=expns.size();
  // Returned vector
  arma::vec ret(N);

  // Form elements
  for(size_t i=0;i<expns.size();i++)
    ret(i)=pow(2.0,-l/2.0-1.25)*sqrt(gsl_sf_gamma(2*l+3)/gsl_sf_gamma(l+1.5))*pow(zeta,l+2.5)/pow(expns[i],l/2.0+1.25)*gsl_sf_hyperg_U(2.0+l,1.5,zeta*zeta/(4.0*expns[i]));
  return ret;
}

arma::mat form_S(const std::vector<double> & expns, int l) {
  // Amount of exponents
  const size_t N=expns.size();
  // Returned vector
  arma::mat ret(N,N);

  double zi, zj;

  // Form elements
  for(size_t i=0;i<expns.size();i++) {
    zi=expns[i];

    for(size_t j=0;j<=i;j++) {
      zj=expns[j];
      ret(i,j)=pow(4.0*zi*zj/((zi+zj)*(zi+zj)),l/2.0+3.0/4.0);
      ret(j,i)=ret(i,j);
    }
  }
  return ret;
}

arma::vec solve_coefficients(std::vector<double> expns, double zeta, int l) {
  // Solve coefficients of exponents.

  arma::vec P=form_P(expns,zeta,l);
  arma::mat S=form_S(expns,l);

  return inv(S)*P;
}

double compute_difference(std::vector<double> expns, double zeta, int l) {
  arma::vec P=form_P(expns,zeta,l);
  arma::mat S=form_S(expns,l);
  // Inverse overlap matrix
  arma::mat invS=arma::inv(S);

  // If inversion was not succesful, invS is empty
  if(invS.n_elem==0)
    return 1.0;

  arma::vec c=invS*P;

  // Compute difference from unity
  double delta=1.0-2.0*arma::dot(c,P);
  delta+=arma::as_scalar(arma::trans(c)*S*c);
  return delta;
}
