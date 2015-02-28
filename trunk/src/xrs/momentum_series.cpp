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


#include "momentum_series.h"
#include "../mathf.h"
#include "../basis.h"

momentum_transfer_series::momentum_transfer_series(const BasisSet * b) {
  bas=b;
  lmax=-1;
}

momentum_transfer_series::~momentum_transfer_series() {
}

arma::cx_mat momentum_transfer_series::get(const arma::vec & q, double rmstol, double maxtol) {
  // Get amount of basis functions.
  const size_t Nbf=bas->get_Nbf();

  // Returned array
  arma::cx_mat ret(Nbf,Nbf);
  ret.zeros();

  // The contributions for different values of l.
  std::vector<double> rmscontr; // RMS norm
  std::vector<double> maxcontr; // Maximum norm

  // Form array using series expansion
  int l=0;
  do {
    // Need to calculate more matrices?
    if(l>lmax) {
      stack.push_back(bas->moment(l));
      lmax++;
    }

    // Helper array: difference to previous result
    arma::mat diff(Nbf,Nbf);
    diff.zeros();

    // Loop over functions of this value of l
    int ind=0;
    for(int ii=0; ii<=l; ii++) {
      int lc=l - ii;
      for(int jj=0; jj<=ii; jj++) {
	int mc=ii - jj;
	int nc=jj;

	diff+=pow(q(0),lc)*pow(q(1),mc)*pow(q(2),nc)*stack[l][ind++];
      }
    }

    // Plug in factorial
    diff/=fact(l);

    // Determine sign. l = 0 real +, l = 1 imag +, l = 2 real -, l = 3 imag -
    if(l%4 == 2 || l%4 == 3)
      diff*=-1.0;

    // Compute contributions
    rmscontr.push_back(sqrt(arma::trace(arma::trans(diff)*diff)/(Nbf*Nbf)));
    maxcontr.push_back(norm(diff,"inf"));

    // Add in to total matrix
    if(l%2==0) // Real part
      for(size_t i=0;i<Nbf;i++)
	for(size_t j=0;j<Nbf;j++)
	  ret(i,j).real()+=diff(i,j);
    else // Imaginary part
      for(size_t i=0;i<Nbf;i++)
	for(size_t j=0;j<Nbf;j++)
	  ret(i,j).imag()+=diff(i,j);

    // Break now?
    if(l>=2)
      // Check two past contributions
      if( (rmscontr[l] <= rmstol && rmscontr[l-1] <= rmstol) && (maxcontr[l] <= maxtol && maxcontr[l-1] <= maxtol) )
	break;

    // Increment value of l
    l++;
  } while(true);

  printf("Contribution of l values:\n");
  for(int i=0;i<=l;i++)
    printf("%i\t%e\t%e\n",i,rmscontr[i],maxcontr[i]);

  return ret;
}
