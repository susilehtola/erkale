/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "completeness_profile.h"
#include "../basis.h"
#include "../linalg.h"

arma::vec get_scanning_exponents(double min, double max, size_t Np) {
  // Form scanning exponents
  arma::vec scan_exp(Np);
  double da=(max-min)/(Np-1);
  for(size_t i=0;i<Np;i++) {
    scan_exp(i)=pow(10.0,min+i*da);
  }

  return scan_exp;
}

/// Compute completeness profile for given element
compprof_t compute_completeness(const ElementBasisSet & bas, double min, double max, size_t Np, bool coulomb) {
  return compute_completeness(bas,get_scanning_exponents(min,max,Np),coulomb);
}


/// Compute completeness profile for given element
compprof_t compute_completeness(const ElementBasisSet & bas, const arma::vec & scan_exp, bool coulomb) {
  // Returned completeness profile
  compprof_t ret;
  ret.lga=arma::log10(scan_exp);

  // Loop over angular momenta
  for(int am=0;am<=bas.get_max_am();am++) {
    // Get primitives and contraction coefficients
    arma::vec exps;
    arma::mat contr;
    bas.get_primitives(exps,contr,am);

    // Do we need to calculate something?
    if(exps.n_elem) {

      int amval=am;
      if(coulomb)
	amval--;

      // Compute overlaps of scanning functions and primitives
      arma::mat scanov=overlap(exps,scan_exp,amval);

      // Compute overlap matrix in used basis set
      arma::mat S;
      S=arma::trans(contr)*overlap(exps,exps,amval)*contr;

      // Helper: scan overlaps of contracted basis functions
      arma::mat hlp=arma::trans(scanov)*contr;

      // Compute completeness profile. Only a single center is
      // involved, so the basis set is always well behaved.
      compprof_am_t profile;
      profile.am=am;
      profile.Y=arma::diagvec(hlp*arma::inv(S)*arma::trans(hlp));
      
      ret.shells.push_back(profile);

    } else {
      // No exponents, create dummy profile.
      compprof_am_t profile;
      profile.am=am;
      profile.Y.zeros(scan_exp.n_elem);
      ret.shells.push_back(profile);
    }
  }

  return ret;
}
