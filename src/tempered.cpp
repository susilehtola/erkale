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
  // Maps the integer index j=1..Nprim to a Legendre argument in [-1, 1].
  // Requires Nprim >= 2 to avoid dividing by zero at the (j-1)*2/(Nprim-1)
  // step; a single-primitive Legendre expansion isn't well-defined here.
  if(Nprim<2) {
    ERROR_INFO();
    throw std::runtime_error("legendre_P_mat requires at least two primitives.\n");
  }

  // Compute the Legendre polynomials
  arma::mat Pk(Nprim, kmax);
  for(int j=1;j<=Nprim;j++) { // Loop over exponents
    // Argument for Legendre polynomial is
    double arg=(j-1)*2.0/(Nprim-1) - 1.0;

    // Store values P_0, P_1, ..., P_{kmax-1}
    for(int k=0;k<kmax;k++)
      Pk(j-1,k)=std::legendre(k, arg);
  }

  return Pk;
}

arma::vec legendre_set(const arma::vec & A, int Nf) {
  // Get Pk matrix
  arma::mat Pk=legendre_P_mat(Nf,A.n_elem);
  // Exponents are
  arma::vec exps=arma::exp10(Pk*A);

  return arma::sort(exps,"descend");
}

arma::vec legendre_pars(const arma::vec & z, int Np) {
  // Get Pk matrix
  arma::mat P(legendre_P_mat(z.n_elem,Np));
  // Parameters are
  arma::vec pars;
  bool ok=arma::solve(pars,P,arma::log10(z));
  if(!ok) {
    ERROR_INFO();
    throw std::runtime_error("Unable to solve set of Legendre parameters.\n");
  }

  return pars;
}

