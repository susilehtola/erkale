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

#include "properties.h"
#include "stringutil.h"

arma::mat mulliken_overlap(const BasisSet & basis, const arma::mat & P) {
  // Amount of nuclei in basis set
  size_t Nnuc=basis.get_Nnuc();

  arma::mat ret(Nnuc,Nnuc);
  ret.zeros();

  // Get overlap
  arma::mat S=basis.overlap();

  // Loop over nuclei
  for(size_t ii=0;ii<Nnuc;ii++) {
    // Get shells on nucleus
    std::vector<GaussianShell> ifuncs=basis.get_funcs(ii);

    // Loop over nuclei
    for(size_t jj=0;jj<=ii;jj++) {
      // Get shells on nucleus
      std::vector<GaussianShell> jfuncs=basis.get_funcs(jj);

      // Initialize output
      ret(ii,jj)=0.0;

      // Loop over shells
      for(size_t fi=0;fi<ifuncs.size();fi++) {
	// First function on shell is
	size_t ifirst=ifuncs[fi].get_first_ind();
	// Last function on shell is
	size_t ilast=ifuncs[fi].get_last_ind();

	// Loop over shells
	for(size_t fj=0;fj<jfuncs.size();fj++) {
	  size_t jfirst=jfuncs[fj].get_first_ind();
	  size_t jlast=jfuncs[fj].get_last_ind();

	  // Loop over functions
	  for(size_t i=ifirst;i<=ilast;i++)
	    for(size_t j=jfirst;j<=jlast;j++)
	      ret(ii,jj)+=P(i,j)*S(i,j);
	}
      }

      // Symmetricize
      if(ii!=jj)
	ret(jj,ii)=ret(ii,jj);
    }
  }

  return ret;
}

arma::mat bond_order(const BasisSet & basis, const arma::mat & P) {
  // Amount of nuclei in basis set
  size_t Nnuc=basis.get_Nnuc();

  arma::mat ret(Nnuc,Nnuc);
  ret.zeros();

  // Get overlap
  arma::mat S=basis.overlap();

  // Form PS
  arma::mat PS=P*S;

  // Loop over nuclei
  for(size_t ii=0;ii<Nnuc;ii++) {
    // Get shells on nucleus
    std::vector<GaussianShell> ifuncs=basis.get_funcs(ii);

    // Loop over nuclei
    for(size_t jj=0;jj<=ii;jj++) {
      // Get shells on nucleus
      std::vector<GaussianShell> jfuncs=basis.get_funcs(jj);

      // Initialize output
      ret(ii,jj)=0.0;

      // Loop over shells
      for(size_t fi=0;fi<ifuncs.size();fi++) {
	// First function on shell is
	size_t ifirst=ifuncs[fi].get_first_ind();
	// Last function on shell is
	size_t ilast=ifuncs[fi].get_last_ind();

	// Loop over shells
	for(size_t fj=0;fj<jfuncs.size();fj++) {
	  size_t jfirst=jfuncs[fj].get_first_ind();
	  size_t jlast=jfuncs[fj].get_last_ind();

	  // Loop over functions
	  for(size_t i=ifirst;i<=ilast;i++)
	    for(size_t j=jfirst;j<=jlast;j++)
	      ret(ii,jj)+=PS(i,j)*PS(j,i);
	}
      }

      // Symmetricize
      if(ii!=jj)
	ret(jj,ii)=ret(ii,jj);
    }
  }

  // The factor 1/2 seems necessary.
  return ret/2.0;
}

arma::mat bond_order(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  return bond_order(basis,Pa+Pb)+bond_order(basis,Pa-Pb);
}


arma::vec nuclear_density(const BasisSet & basis, const arma::mat & P) {
  arma::vec ret(basis.get_Nnuc());
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++)
    ret(inuc)=compute_density(P,basis,basis.get_coords(inuc));
  return ret;
}

void population_analysis(const BasisSet & basis, const arma::mat & P) {

  // Mulliken overlap
  arma::mat mulov=mulliken_overlap(basis,P);
  // Mulliken charges
  arma::mat mulq=-sum(mulov);
  for(size_t i=0;i<basis.get_Nnuc();i++) {
    nucleus_t nuc=basis.get_nucleus(i);
    if(!nuc.bsse)
      mulq(i)+=nuc.Z;
  }

  // Bond order
  arma::mat bord=bond_order(basis,P);

  // Electron density at nuclei
  arma::vec nucd=nuclear_density(basis,P);

  printf("\nMulliken charges\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol(i).c_str(), mulq(i));

  printf("\nElectron density at nuclei\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol(i).c_str(), nucd(i));
  printf("\n");

  // These generate a lot of output
  /*
  mulov.print("Mulliken overlap");
  bord.print("Bond order");
  */
}

void population_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  arma::mat P=Pa+Pb;

  // Mulliken overlap
  arma::mat mulova=mulliken_overlap(basis,Pa);
  arma::mat mulovb=mulliken_overlap(basis,Pb);
  // Mulliken charges
  arma::mat mulq(basis.get_Nnuc(),3);
  mulq.col(0)=-arma::trans(sum(mulova));
  mulq.col(1)=-arma::trans(sum(mulovb));
  mulq.col(2)=mulq.col(0)+mulq.col(1);
  for(size_t i=0;i<basis.get_Nnuc();i++) {
    nucleus_t nuc=basis.get_nucleus(i);
    if(!nuc.bsse) {
      mulq(i,2)+=nuc.Z;
    }
  }

  // Bond order
  arma::mat bord=bond_order(basis,Pa,Pb);

  // Electron density at nuclei
  arma::vec nucd_a=nuclear_density(basis,Pa);
  arma::vec nucd_b=nuclear_density(basis,Pb);

  // Total density
  arma::mat nucd(nucd_a.n_elem,3);
  nucd.col(0)=nucd_a;
  nucd.col(1)=nucd_b;
  nucd.col(2)=nucd_a+nucd_b;

  printf("\nMulliken charges: alpha, beta, total (incl. nucleus)\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol(i).c_str(), mulq(i,0), mulq(i,1), mulq(i,2));

  printf("\nElectron density at nuclei: alpha, beta, total\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol(i).c_str(), nucd(i,0), nucd(i,1), nucd(i,2));
  printf("\n");

  // These generate a lot of output
  /*
  mulov.print("Mulliken overlap");
  bord.print("Bond order");
  */
}


double darwin_1e(const BasisSet & basis, const arma::mat & P) {
  // Energy
  double E=0.0;
  nucleus_t nuc;

  // Loop over nuclei
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    // Get nucleus
    nuc=basis.get_nucleus(inuc);

    if(!nuc.bsse)
      // Don't do correction for BSSE nuclei
      E+=nuc.Z*compute_density(P,basis,nuc.r);
  }

  // Plug in the constant terms
  E*=0.5*M_PI*FINESTRUCT*FINESTRUCT;

  return E;
}