/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "emd_gto.h"
#include "../mathf.h"
#include <algorithm>

RadialGaussian::RadialGaussian(int lambdav, int lv) : RadialFourier(lv) {
  lambda=lambdav;
}

RadialGaussian::~RadialGaussian() {
}

void RadialGaussian::add_term(const contr_t & t) {
  if(c.size()==0) {
    c.push_back(t);
  } else {
    // Get upper bound
    std::vector<contr_t>::iterator high;
    high=std::upper_bound(c.begin(),c.end(),t);

    // Corresponding index is
    size_t ind=high-c.begin();

    if(ind>0 && c[ind-1].z==t.z)
      // Found it.
      c[ind-1].c+=t.c;
    else {
      // Term does not exist, add it
      c.insert(high,t);
    }
  }
}

void RadialGaussian::print() const {
  printf("l=%i, lambda=%i:",l,lambda);
  for(size_t i=0;i<c.size();i++)
    printf(" %+e (%e)\n",c[i].c,c[i].z);
}

std::complex<double> RadialGaussian::get(double p) const {
  std::complex<double> ret=0.0;

  if(lambda==l) {
    // Pure spherical harmonics.
    for(size_t i=0;i<c.size();i++)
      ret+=c[i].c*exp(-p*p/(4.0*c[i].z));
  } else {
    // Mixed set.
    for(size_t i=0;i<c.size();i++)
      ret+=c[i].c*hyperg_1F1((l+lambda)/2.0+1.5,l+1.5,-p*p/(4.0*c[i].z));

    ret*=pow(sqrt(2.0),lambda-l)*doublefact(l+lambda+1)/doublefact(2*l+1);
  }

  ret*=pow(std::complex<double>(0.0,-p),l)*pow(M_2_PI,1.0/4.0)/sqrt(doublefact(2*lambda+1));

  return ret;
}

std::vector< std::vector<size_t> > find_identical_functions(const BasisSet & bas) {
  // Get shells in basis set
  std::vector<GaussianShell> sh=bas.get_shells();
  // and the list of "identical" shells
  std::vector< std::vector<size_t> > idsh=bas.find_identical_shells();

  // The returned list is
  std::vector< std::vector<size_t> > ret;

  // Now we just add all of the functions to their separate lists.
  for(size_t iidsh=0;iidsh<idsh.size();iidsh++) {
    // The index of the first function in the return array is
    size_t first=ret.size();

    // Increase the size of the return array
    ret.resize(ret.size()+bas.get_Nbf(idsh[iidsh][0]));

    // Add the functions on all equivalent shells
    for(size_t ifunc=0;ifunc<bas.get_Nbf(idsh[iidsh][0]);ifunc++)
      for(size_t ish=0;ish<idsh[iidsh].size();ish++)
	ret[first+ifunc].push_back(bas.get_first_ind(idsh[iidsh][ish])+ifunc);
  }

  /*
  printf("Identical functions:\n");
  for(size_t ig=0;ig<ret.size();ig++) {
    printf("Group %i:",(int) ig);
    for(size_t i=0;i<ret[ig].size();i++)
      printf(" %i",(int) ret[ig][i]);
    printf("\n");
  }
  */

  return ret;
}

std::vector< std::vector<ylmcoeff_t> > form_clm(const BasisSet & bas) {
  // Get shells in basis set
  std::vector<GaussianShell> sh=bas.get_shells();
  // and the list of "identical" shells
  std::vector< std::vector<size_t> > idsh=bas.find_identical_shells();

  // Returned decompositions
  std::vector< std::vector<ylmcoeff_t> > ret;

  // Form cartesian expansions
  CartesianExpansion cart(bas.get_max_am());

  // Loop over shells
  for(size_t iid=0;iid<idsh.size();iid++) {
    // Angular momentum
    int l=bas.get_am(idsh[iid][0]);

    // Are spherical harmonics used?
    if(bas.lm_in_use(idsh[iid][0])) {
      // Easy job.
      for(int m=-l;m<=l;m++) {
	// The coefficients for current m
	std::vector<ylmcoeff_t> c;

	ylmcoeff_t tmp;
	tmp.l=l;

	if(m==0) {
	  tmp.m=0;
	  tmp.c=1.0;
	  c.push_back(tmp);
	} else if(m>0) {
	  // Y_{lm} = ( (-1)^m Y_l^m + Y_l^{-m} ) / sqrt(2)
	  tmp.m=m;
	  tmp.c=M_SQRT1_2*pow(-1,m);
	  c.push_back(tmp);

	  tmp.m=-m;
	  tmp.c=M_SQRT1_2;
	  c.push_back(tmp);
	} else {
	  // Y_{lm} = ( (-1)^m Y_l^{-m} - Y_l^{m} ) / [i sqrt(2)]

	  tmp.m=-m;
	  tmp.c=std::complex<double>(0.0,-M_SQRT1_2*pow(-1,m));
	  c.push_back(tmp);

	  tmp.m=m;
	  tmp.c=std::complex<double>(0.0,M_SQRT1_2);
	  c.push_back(tmp);
	}

	// Add function to stack
	ret.push_back(c);
      }
    } else {
      // Need to do cartesian decomposition. Loop over functions on shell.
      for(int i=0; i<=l; i++) {
	int nx = l - i;
	for(int j=0; j<=i; j++) {
	  int ny = i-j;
	  int nz = j;

	  // Get transform
	  SphericalExpansion expn=cart.get(nx,ny,nz);

	  // Get coefficients
	  std::vector<ylmcoeff_t> c=expn.getcoeffs();
	  // and normalize them
	  double n=0.0;
	  for(size_t ic=0;ic<c.size();ic++)
	    n+=norm(c[ic].c);
	  n=sqrt(n);
	  for(size_t ic=0;ic<c.size();ic++)
	    c[ic].c/=n;

	  // and add them to the stack
	  ret.push_back(c);
	}
      }
    }
  }

  /*
  for(size_t i=0;i<ret.size();i++) {
    printf("*** Function %3i ***\n",(int) i +1);
    for(size_t j=0;j<ret[i].size();j++)
      printf(" (% e,% e) Y_%i^%+i",ret[i][j].c.real(),ret[i][j].c.imag(),ret[i][j].l,ret[i][j].m);
    printf("\n");
  }
  */

  return ret;
}

std::vector< std::vector<RadialGaussian> > form_radial(const BasisSet & bas) {
  // Get shells in basis set
  std::vector<GaussianShell> sh=bas.get_shells();
  // and the list of "identical" shells
  std::vector< std::vector<size_t> > idsh=bas.find_identical_shells();

  // Returned functions
  std::vector< std::vector<RadialGaussian> > ret;

  // Form cartesian expansions
  CartesianExpansion cart(bas.get_max_am());

  // Loop over shells
  for(size_t iid=0;iid<idsh.size();iid++) {
    // Angular momentum
    int am=bas.get_am(idsh[iid][0]);

    // Normalized contraction for shell
    std::vector<contr_t> c=bas.get_contr_normalized(idsh[iid][0]);

    // The radial part for this shell
    std::vector<RadialGaussian> rad;

    // Are spherical harmonics used?
    if(bas.lm_in_use(idsh[iid][0])) {
      // Yes. We only get a single l value.
      RadialGaussian help(am,am);

      // Add the contractions.
      for(size_t i=0;i<c.size();i++) {
	contr_t term;
	term.z=c[i].z;
	term.c=c[i].c*pow(c[i].z,-am/2.0-3.0/4.0);
	help.add_term(term);
      }
      rad.push_back(help);

      // All functions on shell have the same radial part
      for(size_t ind=0;ind<bas.get_Nbf(idsh[iid][0]);ind++)
	ret.push_back(rad);

    } else {
      // No, we get multiple values of l.

      // Loop over possible l values
      for(int l=am;l>=0;l-=2) {
	// Construct the radial gaussian
	RadialGaussian help(am,l);

	// Add the contractions.
	for(size_t i=0;i<c.size();i++) {
	  contr_t term;
	  term.z=c[i].z;
	  term.c=c[i].c*pow(c[i].z,-l/2.0-3.0/4.0);
	  help.add_term(term);
	}
	// and add the term
	rad.push_back(help);
      }

      // All functions on shell have the same radial part
      for(size_t ind=0;ind<bas.get_Nbf(idsh[iid][0]);ind++)
	ret.push_back(rad);
    }
  }

  return ret;
}

GaussianEMDEvaluator::GaussianEMDEvaluator() {
}

GaussianEMDEvaluator::GaussianEMDEvaluator(const BasisSet & bas, const arma::mat & Pv) {
  // Check size of P
  if(Pv.n_cols!=Pv.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("P is not square matrix!\n");
  }
  if(Pv.n_cols!=bas.get_Nbf()) {
    ERROR_INFO();
    throw std::runtime_error("Density matrix does not correspond to basis!\n");
  }

  // Form radial functions
  radf=form_radial(bas);

  // Form identical functions
  std::vector< std::vector<size_t> > idf=find_identical_functions(bas);

  // Form Ylm expansion of functions
  std::vector< std::vector<ylmcoeff_t> > clm=form_clm(bas);

  // Form the index list of the centers of the functions
  std::vector<size_t> locv;
  for(size_t ish=0;ish<bas.get_Nshells();ish++)
    for(size_t ifunc=0;ifunc<bas.get_Nbf(ish);ifunc++)
      locv.push_back(bas.get_shell_center_ind(ish));

  /*
  printf("Functions centered on atoms:\n");
  for(size_t i=0;i<loc.size();i++)
    printf("%i: %i\n",(int) i+1, (int) loc[i]+1);
  */

  // Form the list of atomic coordinates
  std::vector<coords_t> coord;
  for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++)
    coord.push_back(bas.get_nuclear_coords(inuc));

  /*
  printf("Coordinates of atoms:\n");
  for(size_t i=0;i<bas.get_Nnuc();i++)
    printf("%3i % f % f % f\n",(int) i+1, coord[i].x, coord[i].y, coord[i].z);
  */

  *this=GaussianEMDEvaluator(radf,idf,clm,locv,coord,Pv);

  // Check norm of radial functions
  //  check_norm();
}

GaussianEMDEvaluator::GaussianEMDEvaluator(const std::vector< std::vector<RadialGaussian> > & radfv, const std::vector< std::vector<size_t> > & idfuncsv, const std::vector< std::vector<ylmcoeff_t> > & clm, const std::vector<size_t> & locv, const std::vector<coords_t> & coord, const arma::mat & Pv) : EMDEvaluator(idfuncsv,clm,locv,coord,Pv) {
  // Set the radial functions
  radf=radfv;
  // and assign the necessary pointers
  update_pointers();
}

GaussianEMDEvaluator::~GaussianEMDEvaluator() {
}


GaussianEMDEvaluator & GaussianEMDEvaluator::operator=(const GaussianEMDEvaluator & rhs) {
  // Assign superclass part
  EMDEvaluator::operator=(rhs);
  // Copy radial functions
  radf=rhs.radf;
  // Update the pointers
  update_pointers();

  return *this;
}

void GaussianEMDEvaluator::update_pointers() {
  rad.resize(radf.size());
  for(size_t i=0;i<radf.size();i++) {
    rad[i].resize(radf[i].size());
    for(size_t j=0;j<radf[i].size();j++)
      rad[i][j]=&radf[i][j];
  }
}
