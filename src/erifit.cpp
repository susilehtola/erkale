/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California 
 * All Rights Reserved 
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "erifit.h"
#include "basis.h"
#include "linalg.h"
#include "mathf.h"
#include "settings.h"
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

  void orthonormal_ERI_trans(const ElementBasisSet & orbel, double linthr, arma::mat & trans) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);
    
    // Get orthonormal orbitals
    arma::mat S(basis.overlap());
    arma::mat Sinvh(CanonicalOrth(S,linthr));

    // Sizes
    size_t Nbf(Sinvh.n_rows);
    size_t Nmo(Sinvh.n_cols);

    // Fill matrix
    trans.zeros(Nbf*Nbf,Nmo*Nmo);
    printf("Size of orthogonal transformation matrix is %i x %i\n",(int) trans.n_rows,(int) trans.n_cols);

    for(size_t iao=0;iao<Nbf;iao++)
      for(size_t jao=0;jao<Nbf;jao++)
	for(size_t imo=0;imo<Nmo;imo++)
	  for(size_t jmo=0;jmo<Nmo;jmo++)
	    trans(iao*Nbf+jao,imo*Nmo+jmo)=Sinvh(iao,imo)*Sinvh(jao,jmo);
  }

  void compute_ERIs(const BasisSet & basis, arma::mat & eris) {
    // Amount of functions
    size_t Nbf(basis.get_Nbf());

    // Get shells in basis set
    std::vector<GaussianShell> shells(basis.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> shpairs(basis.get_unique_shellpairs());

    // Print basis
    //    basis.print(true);

    // Allocate memory for the integrals
    eris.zeros(Nbf*Nbf,Nbf*Nbf);
    printf("Size of integral matrix is %i x %i\n",(int) eris.n_rows,(int) eris.n_cols);
    
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      ERIWorker *eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
      
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
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
		  if(js!=is)
		    eris(j*Nbf+i,k*Nbf+l)=mel;
		  if(ks!=ls)
		    eris(i*Nbf+j,l*Nbf+k)=mel;
		  if(is!=js && ks!=ls)
		    eris(j*Nbf+i,l*Nbf+k)=mel;
		  
		  if(ip!=jp) {
		    eris(k*Nbf+l,i*Nbf+j)=mel;
		    if(js!=is)
		      eris(k*Nbf+l,j*Nbf+i)=mel;
		    if(ks!=ls)
		      eris(l*Nbf+k,i*Nbf+j)=mel;
		    if(is!=js && ks!=ls)
		      eris(l*Nbf+k,j*Nbf+i)=mel;
		  }
		}
	      }
	    }
	  }
	}
      
      // Free memory
      delete eri;
    }
  }
  
  void compute_ERIs(const ElementBasisSet & orbel, arma::mat & eris) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);

    // Calculate
    compute_ERIs(basis,eris);
  }

  void compute_diag_ERIs(const ElementBasisSet & orbel, arma::mat & eris) {
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
    //    basis.print(true);

    // Allocate memory for the integrals
    eris.zeros(Nbf,Nbf);
    printf("Size of integral matrix is %i x %i\n",(int) eris.n_rows,(int) eris.n_cols);
    
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      ERIWorker *eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
      
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
      for(size_t ip=0;ip<shpairs.size();ip++) {
	// Shells are
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;

	// First functions on shells
	size_t i0=shells[is].get_first_ind();
	size_t j0=shells[js].get_first_ind();
	
	// Amount of functions
	size_t Ni=shells[is].get_Nbf();
	size_t Nj=shells[js].get_Nbf();
	
	// Compute integral block
	eri->compute(&shells[is],&shells[js],&shells[is],&shells[js]);
	// Get array
	const std::vector<double> *erip=eri->getp();
	
	// Store integrals
	for(size_t ii=0;ii<Ni;ii++) {
	  size_t i=i0+ii;
	  for(size_t jj=0;jj<Nj;jj++) {
	    size_t j=j0+jj;
	    eris(i,j)=(*erip)[((ii*Nj+jj)*Ni+ii)*Nj+jj];
	  }
	}
      }
      
      // Free memory
      delete eri;
    }
  }

  void unique_exponent_pairs(const ElementBasisSet & orbel, int am1, int am2, std::vector< std::vector<shellpair_t> > & pairs, std::vector<double> & exps) {
    // Form orbital basis set library
    BasisSetLibrary orblib;
    orblib.add_element(orbel);
    
    // Form orbital basis set
    BasisSet basis;
    get_basis(basis,orblib,orbel);

    // Get shells
    std::vector<GaussianShell> shells(basis.get_shells());
    // and list of unique shell pairs
    std::vector<shellpair_t> shpairs(basis.get_unique_shellpairs());

    // Create the exponent list
    exps.clear();
    for(size_t ip=0;ip<shpairs.size();ip++) {
      // Shells are
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;

      // Check am
      if(!( (shells[is].get_am()==am1 && shells[js].get_am()==am2) || (shells[is].get_am()==am2 && shells[js].get_am()==am1)))
	continue;
      
      // Check that shells aren't contracted
      if(shells[is].get_Ncontr()!=1 || shells[js].get_Ncontr()!=1) {
	ERROR_INFO();
	throw std::runtime_error("Must use primitive basis set!\n");
      }
      
      // Exponent value is
      double zeta=shells[is].get_contr()[0].z + shells[js].get_contr()[0].z;
      sorted_insertion<double>(exps,zeta);
    }

    // Create the pair list
    pairs.resize(exps.size());
    for(size_t ip=0;ip<shpairs.size();ip++) {
      // Shells are
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;

      // Check am
      if(!( (shells[is].get_am()==am1 && shells[js].get_am()==am2) || (shells[is].get_am()==am2 && shells[js].get_am()==am1)))
	continue;
      
      // Pair is
      double zeta=shells[is].get_contr()[0].z + shells[js].get_contr()[0].z;
      size_t pos=sorted_insertion<double>(exps,zeta);
      
      // Insert pair
      pairs[pos].push_back(shpairs[ip]);
    }
  }
  
  void compute_cholesky_T(const ElementBasisSet & orbel, int am1, int am2, arma::mat & eris, arma::vec & exps_) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);
    // Decontract the basis set
    blib.decontract();

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);

    // Get shells in basis sets
    std::vector<GaussianShell> shells(basis.get_shells());

    // Get list of unique exponent pairs
    std::vector< std::vector<shellpair_t> > upairs;
    std::vector<double> exps;
    unique_exponent_pairs(orbel,am1,am2,upairs,exps);

    // Store exponents
    exps_=arma::conv_to<arma::vec>::from(exps);
    
    // Allocate memory for the integrals
    eris.zeros(exps.size(),exps.size());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      ERIWorker *eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
      
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
      // Loop over unique exponent pairs
      for(size_t iip=0;iip<upairs.size();iip++)
	for(size_t jjp=0;jjp<=iip;jjp++) {
	  
	  // Loop over individual shell pairs in the group
	  for(size_t ip=0;ip<upairs[iip].size();ip++)
	    for(size_t jp=0;jp<upairs[jjp].size();jp++) {
	      // Shells are
	      size_t is=upairs[iip][ip].is;
	      size_t js=upairs[iip][ip].js;
	      size_t ks=upairs[jjp][jp].is;
	      size_t ls=upairs[jjp][jp].js;

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
	      for(size_t ii=0;ii<Ni;ii++)
		for(size_t jj=0;jj<Nj;jj++)
		  for(size_t kk=0;kk<Nk;kk++)
		    for(size_t ll=0;ll<Nl;ll++) {
		      double mel=std::abs((*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll]);
		      // mel*=mel;

		      eris(iip,jjp)+=mel;
		      if(iip!=jjp)
			eris(jjp,iip)+=mel;
		    }
	    }
	}
      
      // Free memory
      delete eri;
    }
  }
    
  void compute_fitint(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, arma::mat & fitint) {
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

    // Problem sizes
    const size_t Norb(orbbas.get_Nbf());
    const size_t Nfit(fitbas.get_Nbf());
    
    // Allocate memory
    fitint.zeros(Norb*Norb,Nfit);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      int maxam=std::max(orbbas.get_max_am(),fitbas.get_max_am());
      ERIWorker *eri=new ERIWorker(maxam,orbbas.get_max_Ncontr());

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
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
      
      // Free memory
      delete eri;
    }
  }
  
  void compute_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri) {
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

    if(fitint.n_cols != fitbas.get_Nbf())
      throw std::runtime_error("Need to supply fitting integrals for ERIfit!\n");

    // Get shells in basis sets
    std::vector<GaussianShell> orbsh(orbbas.get_shells());
    std::vector<GaussianShell> fitsh(fitbas.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> orbpairs(orbbas.get_unique_shellpairs());

    // Dummy shell
    GaussianShell dummy(dummyshell());

    // Problem size
    size_t Nfit(fitbas.get_Nbf());
    // Overlap matrix
    arma::mat S(Nfit,Nfit);

#ifdef _OPENMP
#pragma omp parallel
#endif
    { 
      // Integral worker
      int maxam=std::max(orbbas.get_max_am(),fitbas.get_max_am());
      ERIWorker *eri=new ERIWorker(maxam,orbbas.get_max_Ncontr());

      // Compute the fitting basis overlap
#ifdef _OPENMP
#pragma omp for
#endif
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
    
      // Free memory
      delete eri;
    }

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
    
    // Fitted ERIs are
    fiteri=fitint*S_inv*arma::trans(fitint);
  }

  void compute_diag_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri) {
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

    if(fitint.n_rows != orbbas.get_Nbf()*orbbas.get_Nbf())
      throw std::runtime_error("Need to supply fitting integrals for ERIfit!\n");
    if(fitint.n_cols != fitbas.get_Nbf())
      throw std::runtime_error("Need to supply fitting integrals for ERIfit!\n");

    // Get shells in basis sets
    std::vector<GaussianShell> orbsh(orbbas.get_shells());
    std::vector<GaussianShell> fitsh(fitbas.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> orbpairs(orbbas.get_unique_shellpairs());

    // Dummy shell
    GaussianShell dummy(dummyshell());

    // Problem size
    size_t Nfit(fitbas.get_Nbf());
    // Overlap matrix
    arma::mat S(Nfit,Nfit);

#ifdef _OPENMP
#pragma omp parallel
#endif
    { 
      // Integral worker
      int maxam=std::max(orbbas.get_max_am(),fitbas.get_max_am());
      ERIWorker *eri=new ERIWorker(maxam,orbbas.get_max_Ncontr());

      // Compute the fitting basis overlap
#ifdef _OPENMP
#pragma omp for
#endif
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
    
      // Free memory
      delete eri;
    }

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
    
    // Fitted ERIs are
    size_t Nbf(orbbas.get_Nbf());
    fiteri.zeros(Nbf,Nbf);
    for(size_t i=0;i<Nbf;i++)
      for(size_t j=0;j<=i;j++) {
	double el=arma::as_scalar(fitint.row(i*Nbf+j)*S_inv*arma::trans(fitint.row(i*Nbf+j)));
	fiteri(i,j)=el;
	fiteri(j,i)=el;
      }
  }
}
