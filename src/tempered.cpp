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



#include "tempered.h"
#include <cmath>

extern "C" {
#include <gsl/gsl_sf_legendre.h>
}

// Construct even-tempered set of exponents
arma::vec eventempered_set(double alpha, double beta, int Nf) {
  arma::vec ret(Nf);

  ret(0)=alpha;
  for(int i=1;i<Nf;i++)
    ret(i)=ret(i-1)*beta;

  return ret;
}

// Construct well-tempered set of exponents
arma::vec welltempered_set(double alpha, double beta, double gamma, double delta, size_t Nf) {
  /*
  // WT-F1
  for(int i=0;i<Nf;i++)
    ret.push_back(alpha*pow(beta,i)*(1.0 + gamma*pow(i*1.0/Nf,delta)));
  */

  arma::vec ret(Nf);
  // WT-F2
  if(Nf>=1)
    ret(0)=alpha;
  if(Nf>=2)
    ret(1)=alpha*beta;
  for(size_t i=2;i<Nf;i++)
    ret(i)=ret(i-1)*beta*(1.0 + gamma*pow((i+1.0)/Nf,delta));
  
  return ret;
}

arma::mat legendre_P_mat(int Nprim, int kmax) {
  // Compute the Legendre polynomials
  arma::mat Pk(Nprim, kmax);
  for(int j=1;j<=Nprim;j++) { // Loop over exponents
    // Argument for Legendre polynomial is
    double arg=(j-1)*2.0/(Nprim-1) - 1.0;

    // Helper array
    double Plarr[kmax];
    gsl_sf_legendre_Pl_array(kmax-1, arg, Plarr);

    // Store values
    for(int k=0;k<kmax;k++)
      Pk(j-1,k)=Plarr[k];
  }

  return Pk;
}

arma::vec legendre_set(const arma::vec & A, int Nf) {
  // Get Pk matrix
  arma::mat Pk=legendre_P_mat(Nf,A.n_elem);
  // Exponents are
  arma::vec exps=arma::exp(Pk*A);

  return exps;
}
