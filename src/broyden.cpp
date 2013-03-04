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



#include "broyden.h"

Broyden::Broyden(bool verb, size_t max, double b, double s) {
  m=max;
  beta=b;
  sigma=s;
  difficult=0;
  verbose=verb;
}

Broyden::~Broyden() {
}

void Broyden::push_x(const arma::vec & xv) {
  x.push_back(xv);
}

void Broyden::push_f(const arma::vec & fv) {
  f.push_back(fv);

  if(f.size()>=2) {
    // Check if update was bad and |f| actually increased.

    double newnorm=norm(f[f.size()-1],2);
    double oldnorm=norm(f[f.size()-2],2);

    if( newnorm > oldnorm) {
      if(verbose)
	printf("Broyden: bad update detected - norm increased by %e from %e to %e.\n",newnorm-oldnorm,oldnorm,newnorm);
      
      //      x.erase(x.begin()+x.size()-1);
      //      f.erase(f.begin()+f.size()-1);
      difficult=1;
    }
  }

  if(f.size()>m) {
    // Drop older matrices from memory
    x.erase(x.begin());
    f.erase(f.begin());
  }
}

arma::vec Broyden::update_x() {
  size_t xi=x.size()-1;
  size_t fi=f.size()-1;

  if(xi!=fi) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "\nxi="<<xi<<" != fi="<<fi<<"!\n";
    throw std::runtime_error(oss.str());
  }

  // Compute approximate solution
  // \tilde{x}_{k+1}
  arma::vec xtilde;
  if(!difficult)
    // Normal case: \tilde{x}_{k+1} = x_k - G_k f_k
    xtilde=x[x.size()-1] - operate_G(f[f.size()-1],f.size()-1);
  else
    // Difficult case: \tilde{x}_{k+1} = x_{k-1} - G_k f_{k-1}
    xtilde=x[x.size()-2] - operate_G(f[f.size()-2],f.size()-1);

  // and damp it with
  // x_{k+1} = (1-\beta) x_k + \beta \tilde{x}_{k+1}
  if(!difficult)
    return (1.0-beta)*x[x.size()-1] + beta*xtilde;
  else {
    difficult=0;
    return (1.0-0.5*beta)*x[x.size()-2] + 0.5*beta*xtilde;
  }
}

arma::vec Broyden::operate_G(const arma::vec & v, size_t ind) const {
  if(ind==0)
    // G_0
    return sigma*v;
  else {
    // Compute \Delta f_{n-1}^T v / (\Delta f_{n-1}^T \Delta f_{n-1})
    arma::vec fnm1=f[ind]-f[ind-1];
    arma::vec xnm1=x[ind]-x[ind-1];

    double fnm1sq=arma::dot(fnm1,fnm1);
    double fv=arma::dot(fnm1,v);

    arma::vec Garg=v-fnm1*fv/fnm1sq;

    return xnm1*fv/fnm1sq + operate_G(Garg,ind-1);
  }
}

void Broyden::clear() {
  x.clear();
  f.clear();
}
