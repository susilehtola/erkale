/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "erifit.h"
#include "basis.h"
#include "linalg.h"
#include "eriworker.h"

namespace ERIfit {

  bool operator<(const bf_pair_t & lhs, const bf_pair_t & rhs) {
    return lhs.idx < rhs.idx;
  }

  void get_basis(BasisSet & basis, const BasisSetLibrary & blib, const ElementBasisSet & orbel) {
    // Settings needed to form basis set
    Settings set;
    set.add_scf_settings();
    set.set_bool("BasisRotate", false);
    set.set_string("Decontract", "");
    set.set_bool("UseLM", true);

    // Atoms
    std::vector<atom_t> atoms(1);
    atoms[0].el=orbel.get_symbol();
    atoms[0].num=0;
    atoms[0].x=atoms[0].y=atoms[0].z=0.0;
    atoms[0].Q=0;

    // Form basis set
    construct_basis(basis,atoms,blib,set);
  }

  void compute_ERIs(const ElementBasisSet & orbel, arma::mat & eris) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);
    
    size_t Nbf(basis.get_Nbf());

    // Get shells in basis set
    std::vector<GaussianShell> shells(basis.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> shpairs(basis.get_unique_shellpairs());

    // Print basis
    basis.print(true);

    // Create pair list
    std::vector<bf_pair_t> list;
    for(size_t ip=0;ip<shpairs.size();ip++) {
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;
      size_t i0=shells[is].get_first_ind();
      size_t j0=shells[js].get_first_ind();
      size_t Ni=shells[is].get_Nbf();
      size_t Nj=shells[js].get_Nbf();

      // Loop over the functions
      for(size_t ii=0;ii<Ni;ii++)
	for(size_t jj=0;jj<Nj;jj++) {
	  bf_pair_t hlp;
	  hlp.i=i0+ii;
	  hlp.j=j0+jj;
	  hlp.is=is;
	  hlp.js=js;
	  hlp.idx=hlp.i*Nbf+hlp.j;
	  list.push_back(hlp);
	}
      if(is!=js)
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t jj=0;jj<Nj;jj++) {
	    bf_pair_t hlp;
	    hlp.j=i0+ii;
	    hlp.i=j0+jj;
	    hlp.js=is;
	    hlp.is=js;
	    hlp.idx=hlp.i*Nbf+hlp.j;
	    list.push_back(hlp);
	  }
    }
    std::stable_sort(list.begin(),list.end());
    
    printf("Basis function pairs:\n");
    for(size_t i=0;i<list.size();i++)
      printf("%4i: functions %3i and %3i on shells %2i and %2i\n",(int) list[i].idx, (int) list[i].i, (int) list[i].j, (int) list[i].is, (int) list[i].js);

    // Integral worker
    ERIWorker *eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());

    // Compute the integrals
    eris.zeros(Nbf*Nbf,Nbf*Nbf);
    for(size_t ip=0;ip<shpairs.size();ip++)
      for(size_t jp=0;jp<=ip;jp++) {
	// Shells are
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;
	size_t ks=shpairs[jp].is;
	size_t ls=shpairs[jp].js;

	// First functions on shells
	size_t i0=shells[is].get_first_ind();
	size_t j0=shells[js].get_first_ind();
	size_t k0=shells[ks].get_first_ind();
	size_t l0=shells[ls].get_first_ind();
	
	// Amount of functions
	size_t Ni=shells[is].get_Nbf();
	size_t Nj=shells[js].get_Nbf();
	size_t Nk=shells[ks].get_Nbf();
	size_t Nl=shells[ls].get_Nbf();
	
	// Compute integral block
	eri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	// Get array
	const std::vector<double> *erip=eri->getp();

	// Store integrals
	for(size_t ii=0;ii<Ni;ii++) {
	  size_t i=i0+ii;
	  for(size_t jj=0;jj<Nj;jj++) {
	    size_t j=j0+jj;
	    for(size_t kk=0;kk<Nk;kk++) {
	      size_t k=k0+kk;
	      for(size_t ll=0;ll<Nl;ll++) {
		size_t l=l0+ll;

		// Go through the 8 permutation symmetries
		double mel=(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
		eris(i*Nbf+j,k*Nbf+l)=mel;
		eris(j*Nbf+i,k*Nbf+l)=mel;
		eris(i*Nbf+j,l*Nbf+k)=mel;
		eris(j*Nbf+i,l*Nbf+k)=mel;
		eris(k*Nbf+l,i*Nbf+j)=mel;
		eris(k*Nbf+l,j*Nbf+i)=mel;
		eris(l*Nbf+k,i*Nbf+j)=mel;
		eris(l*Nbf+k,j*Nbf+i)=mel;
	      }
	    }
	  }
	}
      }
    
    // Free memory
    delete eri;
  }

  void compute_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, arma::mat & fitint, arma::mat & fiteri) {
    // Form orbital basis set library
    BasisSetLibrary orblib;
    orblib.add_element(orbel);
    
    // Form orbital basis set
    BasisSet orbbas;
    get_basis(orbbas,orblib,orbel);
    
    // and fitting basis set
    BasisSet fitbas;
    get_basis(fitbas,fitlib,orbel);

    // Coulomb normalize the fitting set
    fitbas.coulomb_normalize();

    // Get shells in basis sets
    std::vector<GaussianShell> orbsh(orbbas.get_shells());
    std::vector<GaussianShell> fitsh(fitbas.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> orbpairs(orbbas.get_unique_shellpairs());

    // Dummy shell
    GaussianShell dummy(dummyshell());

    // Integral worker
    int maxam=std::max(orbbas.get_max_am(),fitbas.get_max_am());
    ERIWorker *eri=new ERIWorker(maxam,orbbas.get_max_Ncontr());

    // Compute the fitting basis overlap
    size_t Nfit(fitbas.get_Nbf());
    arma::mat S(Nfit,Nfit);
    for(size_t i=0;i<fitsh.size();i++)
      for(size_t j=0;j<=i;j++) {
	// Compute integral block
	eri->compute(&fitsh[i],&dummy,&fitsh[j],&dummy);
	// Get array
	const std::vector<double> *erip=eri->getp();
	// Store integrals
	size_t i0=fitsh[i].get_first_ind();
	size_t j0=fitsh[j].get_first_ind();
	size_t Ni=fitsh[i].get_Nbf();
	size_t Nj=fitsh[j].get_Nbf();
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t jj=0;jj<Nj;jj++) {
	    double mel=(*erip)[ii*Nj+jj];
	    S(i0+ii,j0+jj)=mel;
	    S(j0+jj,i0+ii)=mel;
	  }
      }
    S.print("S");
    
    // Do the eigendecomposition
    arma::vec Sval;
    arma::mat Svec;
    eig_sym_ordered(Sval,Svec,S);

    // Count linearly independent vectors
    size_t Nind=0;
    for(size_t i=0;i<Sval.n_elem;i++)
      if(Sval(i)>=linthr)
	Nind++;
    // and drop the linearly dependent ones
    Sval=Sval.subvec(Sval.n_elem-Nind,Sval.n_elem-1);
    Svec=Svec.cols(Svec.n_cols-Nind,Svec.n_cols-1);
    
    // Form inverse overlap matrix
    arma::mat S_inv;
    S_inv.zeros(Svec.n_rows,Svec.n_rows);
    for(size_t i=0;i<Sval.n_elem;i++)
      S_inv+=Svec.col(i)*arma::trans(Svec.col(i))/Sval(i);

    // Next, compute the fitting integrals.
    size_t Norb=orbbas.get_Nbf();
    fitint.zeros(Norb*Norb,Nfit);
    for(size_t ip=0;ip<orbpairs.size();ip++)
      for(size_t as=0;as<fitsh.size();as++) {
	// Orbital shells are
	size_t is=orbpairs[ip].is;
	size_t js=orbpairs[ip].js;

	// First function is
	size_t i0=orbsh[is].get_first_ind();
	size_t j0=orbsh[js].get_first_ind();
	size_t a0=fitsh[as].get_first_ind();
	
	// Amount of functions
	size_t Ni=orbsh[is].get_Nbf();
	size_t Nj=orbsh[js].get_Nbf();
	size_t Na=fitsh[as].get_Nbf();

	// Compute integral block
	eri->compute(&orbsh[is],&orbsh[js],&dummy,&fitsh[as]);
	// Get array
	const std::vector<double> *erip=eri->getp();
	
	// Store integrals
	for(size_t ii=0;ii<Ni;ii++) {
	  size_t i=i0+ii;
	  for(size_t jj=0;jj<Nj;jj++) {
	    size_t j=j0+jj;
	    for(size_t aa=0;aa<Na;aa++) {
	      size_t a=a0+aa;

	      // Go through the two permutation symmetries
	      double mel=(*erip)[(ii*Nj+jj)*Na+aa];
	      fitint(i*Norb+j,a)=mel;
	      fitint(j*Norb+i,a)=mel;
	    }
	  }
	}
      }

    fitint.print("Fitint");

    //printf("fitint size is %i x %i\n",(int) fitint.n_rows, (int) fitint.n_cols);
    
    // Free memory
    delete eri;
    
    // Fitted ERIs are
    fiteri=fitint*S_inv*arma::trans(fitint);
  }
}

