/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "completeness_profile.h"
#include "../basis.h"
#include "../linalg.h"

/// Compute overlap of normalized Gaussian primitives
arma::mat overlap(const std::vector<double> & iexps, const std::vector<double> & jexps, int am) {
  arma::mat S(iexps.size(),jexps.size());
  for(size_t i=0;i<iexps.size();i++)
    for(size_t j=0;j<jexps.size();j++) {
      // Sum of exponents
      double zeta=iexps[i] + jexps[j];
      // Helper
      double eta=4.0*iexps[i]*jexps[j]/(zeta*zeta);
      // Compute overlap
      S(i,j)=pow(eta,am/2.0+0.75);
    }
  return S;
}

/// Compute completeness profile for given element
compprof_t compute_completeness(const ElementBasisSet & bas, double min, double max, size_t Np) {
  // Returned completeness profile
  compprof_t ret;

  // Form scanning exponents
  std::vector<double> scan_exp(Np);
  double da=(max-min)/(Np-1);
  for(size_t i=0;i<Np;i++) {
    ret.lga.push_back(min+i*da);
    scan_exp[i]=pow(10.0,ret.lga[i]);
  }
  
  // Loop over angular momenta
  for(int am=0;am<=bas.get_max_am();am++) {
    // Get primitives and contraction coefficients
    std::vector<double> exps;
    arma::mat contr;
    bas.get_primitives(exps,contr,am);

    // Compute overlaps of scanning functions and primitives
    arma::mat scanov=overlap(exps,scan_exp,am);

    // Compute overlap matrix in used basis set
    arma::mat S;
    S=arma::trans(contr)*overlap(exps,exps,am)*contr;

    // Form Choleksky inverse of S
    arma::mat Sinvh=CholeskyOrth(S);

    // Helper matrix
    arma::mat K=contr*Sinvh;

    // Compute completeness overlaps
    arma::mat J=arma::trans(K)*scanov;

    // Compute completeness profile
    compprof_am_t profile;
    profile.am=am;

    // Loop over scanning exponents
    for(size_t ip=0;ip<Np;ip++) {
      double Y=0.0;
      // Loop over functions
      for(size_t ifunc=0;ifunc<J.n_rows;ifunc++)
	Y+=J(ifunc,ip)*J(ifunc,ip);

      profile.Y.push_back(Y);
    }
    
    ret.shells.push_back(profile);
  }

  return ret;
}
