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

#include "checkpoint.h"
#include "erichol.h"
#include "linalg.h"
#include "eriworker.h"
#include "mathf.h"
#include "stringutil.h"
#include "timer.h"

#include <cstdio>
// For exceptions
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CHOLFILE "cholesky.chk"

ERIchol::ERIchol() {
  Nbf=0;
  omega=0.0;
  alpha=1.0;
  beta=0.0;
}

ERIchol::~ERIchol() {
}

void ERIchol::set_range_separation(double w, double a, double b) {
  omega=w;
  alpha=a;
  beta=b;
}

void ERIchol::get_range_separation(double & w, double & a, double & b) const {
  w=omega;
  a=alpha;
  b=beta;
}

void ERIchol::load() {
  Checkpoint chkpt(CHOLFILE,false);

  // Suffix in checkpoint
  std::string suffix;
  if(omega!=1.0) {
    std::ostringstream oss;
    oss << "_" << omega;
    suffix=oss.str();
  }

  // Read B matrix
  chkpt.read("B"+suffix,B);

  // Read amount of basis functions
  {
    hsize_t Nbft;
    chkpt.read("Nbf",Nbft);
    Nbf=Nbft;
  }

  // Read product index
  {
    std::vector<hsize_t> prodidxv;
    chkpt.read("prodidx"+suffix,prodidxv);
    prodidx=arma::conv_to<arma::uvec>::from(prodidxv);
  }
  // and the off-diagonal index
  {
    std::vector<hsize_t> odiagidxv;
    chkpt.read("odiagidx"+suffix,odiagidxv);
    odiagidx=arma::conv_to<arma::uvec>::from(odiagidxv);
  }
  // the product map
  {
    std::vector<hsize_t> prodmapv;
    chkpt.read("prodmap"+suffix,prodmapv);
    prodmap=arma::reshape(arma::conv_to<arma::umat>::from(prodmapv),Nbf,Nbf);
  }
  // and the inverse map
  {
    std::vector<hsize_t> invmapv;
    chkpt.read("invmap"+suffix,invmapv);
    invmap=arma::reshape(arma::conv_to<arma::umat>::from(invmapv),2,prodidx.n_elem);
  }
}

void ERIchol::save() const {
  // Check consistency. Default is to not truncate
  bool trunc=false;
  if(file_exists(CHOLFILE)) {
    // Open in read-only mode and try to get Nbf
    Checkpoint chkpt(CHOLFILE,false);
    hsize_t Nbft;
    if(chkpt.exist("Nbf")) {
      chkpt.read("Nbf",Nbft);
      if(Nbf!=Nbft)
	trunc=true;
    } else
      trunc=true;
  }
  // Open in write mode
  Checkpoint chkpt(CHOLFILE,true,trunc);

  // Suffix in checkpoint
  std::string suffix;
  if(omega!=1.0) {
    std::ostringstream oss;
    oss << "_" << omega;
    suffix=oss.str();
  }

  // Write B matrix
  chkpt.write("B"+suffix,B);

  // Save amount of basis functions
  {
    hsize_t Nbft(Nbf);
    chkpt.write("Nbf",Nbft);
  }

  // Save product index
  {
    std::vector<hsize_t> prodidxv(arma::conv_to< std::vector<hsize_t> >::from(prodidx));
    chkpt.write("prodidx"+suffix,prodidxv);
  }
  // and the off-diagonal index
  {
    std::vector<hsize_t> odiagidxv(arma::conv_to< std::vector<hsize_t> >::from(odiagidx));
    chkpt.write("odiagidx"+suffix,odiagidxv);
  }
  // the product map
  {
    std::vector<hsize_t> prodmapv(arma::conv_to< std::vector<hsize_t> >::from(arma::vectorise(prodmap)));
    chkpt.write("prodmap"+suffix,prodmapv);
  }
  // and the inverse map
  {
    std::vector<hsize_t> invmapv(arma::conv_to< std::vector<hsize_t> >::from(arma::vectorise(invmap)));
    chkpt.write("invmap"+suffix,invmapv);
  }
}

size_t ERIchol::fill(const BasisSet & basis, double tol, double shthr, double shtol, bool verbose) {
  // Screening matrix and pairs
  arma::mat screen;
  std::vector<eripair_t> shpairs=basis.get_eripairs(screen,shtol,omega,alpha,beta,verbose);

  // Amount of basis functions
  Nbf=basis.get_Nbf();
  // Shells
  std::vector<GaussianShell> shells=basis.get_shells();

  Timer t;
  Timer ttot;

  // Integral time
  double t_int=0.0;
  // Cholesky time
  double t_chol=0.0;

  // Calculate diagonal element vector
  arma::vec d(Nbf*Nbf);
  d.zeros();
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker *eri;
    const std::vector<double> * erip;

    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
    else
      eri=new ERIWorker_srlr(basis.get_max_am(),basis.get_max_Ncontr(),omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shpairs.size();ip++) {
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;

      // Compute integrals
      eri->compute(&shells[is],&shells[js],&shells[is],&shells[js]);
      erip=eri->getp();

      // and store them
      size_t Ni(shells[is].get_Nbf());
      size_t Nj(shells[js].get_Nbf());
      size_t i0(shells[is].get_first_ind());
      size_t j0(shells[js].get_first_ind());

      for(size_t ii=0;ii<Ni;ii++)
	for(size_t jj=0;jj<Nj;jj++) {
	  size_t i=i0+ii;
	  size_t j=j0+jj;
	  d(i*Nbf+j)=(*erip)[((ii*Nj+jj)*Ni+ii)*Nj+jj];
	  d(j*Nbf+i)=d(i*Nbf+j);
	}
    }

    delete eri;
  }
  t_int+=t.get();
  t.set();

  // Amount of pairs surviving shell pair screening
  size_t Nshp=0;
  {
    prodmap.ones(Nbf,Nbf);
    prodmap*=-1; // Go to UINT_MAX

    size_t iprod=0;
    size_t iodiag=0;
    prodidx.resize(Nbf*Nbf);
    odiagidx.resize(Nbf*Nbf);
    invmap.zeros(2,Nbf*Nbf);
    for(size_t ip=0;ip<shpairs.size();ip++) {
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;
      size_t Ni(shells[is].get_Nbf());
      size_t Nj(shells[js].get_Nbf());
      size_t i0(shells[is].get_first_ind());
      size_t j0(shells[js].get_first_ind());
      if(is==js) {
	for(size_t i=i0;i<i0+Ni;i++) {
	  for(size_t j=j0;j<i;j++) {
	    // Function indices are
	    invmap(0,iprod)=i;
	    invmap(1,iprod)=j;
	    // Global product index is
	    prodidx(iprod)=i*Nbf+j;
	    // Off-diagonal product
	    odiagidx(iodiag)=iprod;

	    // Significant product?
	    Nshp++;
	    if(d(prodidx(iprod))>=shtol) {
	      // Product index mapping is
	      prodmap(i,j)=iprod;
	      prodmap(j,i)=iprod;
	      iprod++;
	      iodiag++;
	    }
	  }
	  // Function indices are
	  invmap(0,iprod)=i;
	  invmap(1,iprod)=i;
	  // Global product index is
	  prodidx(iprod)=i*Nbf+i;
	  // Significant product?
	  Nshp++;
	  if(d(prodidx(iprod))>=shtol) {
	    // Product index mapping is
	    prodmap(i,i)=iprod;

	    iprod++;
	  }
	}
      } else {
      	for(size_t i=i0;i<i0+Ni;i++)
	  for(size_t j=j0;j<j0+Nj;j++) {
	    // Function indices are
	    invmap(0,iprod)=i;
	    invmap(1,iprod)=j;
	    // Global product index is
	    prodidx(iprod)=i*Nbf+j;
	    // Off-diagonal product
	    odiagidx(iodiag)=iprod;
	    // Significant product?
	    Nshp++;
	    if(d(prodidx(iprod))>=shtol) {
	      // Product index mapping is
	      prodmap(i,j)=iprod;

	      iprod++;
	      iodiag++;
	    }
	  }
      }
    }
    // Resize prodidx
    prodidx.resize(iprod);
    odiagidx.resize(iodiag);
    if(iprod<invmap.n_cols-1)
      invmap.shed_cols(iprod,invmap.n_cols-1);
  }

  if(verbose) {
    printf("Screening by shell pairs and symmetry reduced dofs by factor %.2f.\n",d.n_elem*1.0/Nshp);
    printf("Individual screening reduced dofs by a total factor %.2f.\n",d.n_elem*1.0/prodidx.n_elem);
    printf("Computing Cholesky vectors. Estimated memory size is %s - %s.\n",memory_size(3*Nbf*prodidx.n_elem*sizeof(double),true).c_str(),memory_size(10*Nbf*prodidx.n_elem*sizeof(double),true).c_str());
  }

  // Drop unnecessary vectors
  d=d(prodidx);

  // Error is
  double error(arma::max(d));

  // Pivot index
  arma::uvec pi(arma::linspace<arma::uvec>(0,d.n_elem-1,d.n_elem));
  // Allocate memory
  B.zeros(100,prodidx.n_elem);
  // Loop index
  size_t m(0);

  while(error>tol && m<d.n_elem-1) {
    // Errors in pivoted order
    arma::vec errs(d(pi));
    // Sort the upcoming errors so that largest one is first
    arma::uvec idx=arma::stable_sort_index(errs.subvec(m,d.n_elem-1),"descend");

    // Update the pivot index
    {
      arma::uvec pisub(pi.subvec(m,d.n_elem-1));
      pi.subvec(m,d.n_elem-1)=pisub(idx);
    }

    // Pivot index
    size_t pim=pi(m);
    //printf("Pivot index is %4i, corresponding to product %i, with error %e, error is %e\n",(int) pim, (int) prodidx(pim), d(pim), error);

    // Off-diagonal elements: find out which shells the pivot index
    // belongs to. The relevant function indices are
    size_t max_k, max_l;
    // and they belong to the shells
    size_t max_ks, max_ls;
    // that have N functions
    size_t max_Nk, max_Nl;
    // where the first functions are
    size_t max_k0, max_l0;
    {
      // The corresponding functions are
      max_k=invmap(0,pim);
      max_l=invmap(1,pim);
      // which are on the shells
      max_ks=basis.find_shell_ind(max_k);
      max_ls=basis.find_shell_ind(max_l);
      // that have N functions
      max_Nk=basis.get_Nbf(max_ks);
      max_Nl=basis.get_Nbf(max_ls);
      // and the function indices are
      max_k0=basis.get_first_ind(max_ks);
      max_l0=basis.get_first_ind(max_ls);
    }
    //    printf("Pivot corresponds to functions %i and %i on shells %i and %i.\n",(int) max_k, (int) max_l, (int) max_ks, (int) max_ls);

    // Compute integrals on the rows
    arma::mat A(d.n_elem,max_Nk*max_Nl);
    A.zeros();
    t.set();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      ERIWorker *eri;
      const std::vector<double> * erip;

      if(omega==0.0 && alpha==1.0 && beta==0.0)
	eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
      else
	eri=new ERIWorker_srlr(basis.get_max_am(),basis.get_max_Ncontr(),omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for(size_t ipair=0;ipair<shpairs.size();ipair++) {
	size_t is=shpairs[ipair].is;
	size_t js=shpairs[ipair].js;

	// Do we need to compute the shell?
	if(screen(is,js)*screen(max_ks,max_ls)<shtol)
	  continue;

	// Compute integrals
	eri->compute(&shells[is],&shells[js],&shells[max_ks],&shells[max_ls]);
	erip=eri->getp();

	// and store them
	size_t Ni(shells[is].get_Nbf());
	size_t Nj(shells[js].get_Nbf());
	size_t i0(shells[is].get_first_ind());
	size_t j0(shells[js].get_first_ind());

	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t jj=0;jj<Nj;jj++) {
	    size_t i=i0+ii;
	    size_t j=j0+jj;

	    // Check if function pair is significant
	    if(prodmap(i,j)>Nbf*Nbf)
	      continue;

	    for(size_t kk=0;kk<max_Nk;kk++)
	      for(size_t ll=0;ll<max_Nl;ll++) {
		A(prodmap(i,j),kk*max_Nl+ll)=(*erip)[((ii*Nj+jj)*max_Nk+kk)*max_Nl+ll];
	      }
	  }
      }

      delete eri;
    }
    t_int+=t.get();
    t.set();

    size_t nb=0;
    size_t b0=m;
    while(true) {
      // Did we already treat everything in the block?
      if(nb==A.n_cols)
	break;
      // Find global largest error
      errs=d(pi);
      double errmax=arma::max(errs.subvec(m,d.n_elem-1));
      // and the largest error within the current block
      double blockerr=0;
      size_t blockind=0;
      size_t Aind=0;
      for(size_t kk=0;kk<max_Nk;kk++)
	for(size_t ll=0;ll<max_Nl;ll++) {
	  // Function indices are
	  size_t k=kk+max_k0;
	  size_t l=ll+max_l0;
	  // Corresponding index in the array is
	  size_t ind = prodmap(k,l);
	  if(ind > Nbf*Nbf)
	    continue;

	  if(d(ind)>blockerr) {
	    // Check that the index is not in the old pivots
	    bool found=false;
	    for(size_t i=0;i<m;i++)
	      if(pi(i)==ind)
		found=true;
	    if(!found) {
	      Aind=kk*max_Nl+ll;
	      blockind=ind;
	      blockerr=d(ind);
	    }
	  }
	}
      // Move to next block.
      if(blockerr<shthr*errmax) {
	//printf("Block error is %e compared to global error %e, stopping\n",blockerr,errmax);
	break;
      }

      // Increment amount of vectors in the block
      nb++;

      // Switch the pivot
      if(pi(m)!=blockind) {
	bool found=false;
	for(size_t i=m+1;i<pi.n_elem;i++)
	  if(pi(i)==blockind) {
	    found=true;
	    std::swap(pi(i),pi(m));
	    break;
	  }
	if(!found) {
	  pi.t().print("Pivot");
	  fflush(stdout);
	  std::ostringstream oss;
	  oss << "Pivot index " << blockind << " not found, m = " << m << " !\n";
	  throw std::logic_error(oss.str());
	}
      }

      pim=pi(m);

      // Insert new rows if necessary
      if(m>=B.n_rows)
	B.insert_rows(B.n_rows,100,true);

      // Compute diagonal element
      B(m,pim)=sqrt(d(pim));

      // Off-diagonal elements
      if(m==0) {
	// No B contribution here; avoid if clause in for loop
#ifdef _OPENMP
#pragma omp parallel for
#endif
     	for(size_t i=m+1;i<d.n_elem;i++) {
	  size_t pii=pi(i);
	  // Compute element
	  B(m,pii)=A(pii,Aind)/B(m,pim);
	  // Update d
	  d(pii)-=B(m,pii)*B(m,pii);
	}
      } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(size_t i=m+1;i<d.n_elem;i++) {
	  size_t pii=pi(i);
	  // Compute element
	  B(m,pii)=(A(pii,Aind) - arma::as_scalar(arma::trans(B.submat(0,pim,m-1,pim))*B.submat(0,pii,m-1,pii)))/B(m,pim);
	  // Update d
	  d(pii)-=B(m,pii)*B(m,pii);
	}
      }

      // Update error
      error=arma::max(d(pi.subvec(m+1,pi.n_elem-1)));
      // Increase m
      m++;
    }
    t_chol+=t.get();

    if(verbose) {
      printf("Cholesky vectors no %5i - %5i computed, error is %e (%s).\n",(int) b0, (int) (b0+nb-1),error,t.elapsed().c_str());
      fflush(stdout);
      t.set();
    }
  }

  if(verbose) {
    printf("Cholesky decomposition finished in %s. Realized memory size is %s.\n",ttot.elapsed().c_str(),memory_size(B.n_elem*sizeof(double)).c_str());
    printf("Time use: integrals %3.1f %%, linear algebra %3.1f %%.\n",100*t_int/(t_int+t_chol),100*t_chol/(t_int+t_chol));
  }

  // Transpose to get Cholesky vectors as columns
  arma::inplace_trans(B);

  // Drop any unnecessary columns
  if(m<B.n_cols)
    B.shed_cols(m,B.n_cols-1);

  return shpairs.size();
}

size_t ERIchol::get_Naux() const {
  return B.n_cols;
}

size_t ERIchol::get_Nbf() const {
  return Nbf;
}

size_t ERIchol::get_Npairs() const {
  return B.n_rows;
}

arma::mat ERIchol::get() const {
  return B;
}

arma::mat ERIchol::calcJ(const arma::mat & P) const {
  // Vectorize P
  arma::rowvec Pv(arma::trans(P(prodidx)));
  // Twice the off-diagonal contribution
  Pv(odiagidx)*=2.0;
  // Calculate expansion coefficients
  arma::rowvec g(Pv*B);
  // Form Coulomb matrix
  arma::vec Jv(B*arma::trans(g));
  // and restore it
  arma::mat J(P.n_rows,P.n_cols);
  J.zeros();
  for(size_t i=0;i<prodidx.size();i++)
    J(invmap(0,i),invmap(1,i))=Jv(i);
  for(size_t i=0;i<odiagidx.size();i++)
    J(invmap(1,odiagidx(i)),invmap(0,odiagidx(i)))=Jv(odiagidx(i));

  return J;
}

arma::mat ERIchol::calcK(const arma::vec & C) const {
  // K_uv = C_r C_s (ur|vs) = (L^P_ur C_r) (L^P_vs Cs)
  arma::mat v(C.n_elem,B.n_cols);
  v.zeros();

  // First part: diagonal and above diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t i=0;i<prodidx.size();i++)
      v(invmap(0,i),P)+=B(i,P)*C(invmap(1,i));
  // Below diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t ii=0;ii<odiagidx.n_elem;ii++) {
      size_t i=odiagidx(ii);
      v(invmap(1,i),P)+=B(i,P)*C(invmap(0,i));
    }

  return v*arma::trans(v);
}

arma::cx_mat ERIchol::calcK(const arma::cx_vec & C0) const {
  // Need to complex conjugate C
  arma::cx_vec C(arma::conj(C0));

  // K_uv = C_r C_s (ur|vs) = (L^P_ur C_r) (L^P_vs Cs)
  arma::cx_mat v(C.n_elem,B.n_cols);
  v.zeros();

  // First part: diagonal and above diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t i=0;i<prodidx.size();i++)
      v(invmap(0,i),P)+=B(i,P)*C(invmap(1,i));
  // Below diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t ii=0;ii<odiagidx.n_elem;ii++) {
      size_t i=odiagidx(ii);
      v(invmap(1,i),P)+=B(i,P)*C(invmap(0,i));
    }

  return v*arma::trans(v);

}

arma::mat ERIchol::calcK(const arma::mat & C, const std::vector<double> & occs) const {
  arma::mat K(C.n_rows,C.n_rows);
  K.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<occs.size();i++)
    if(occs[i]!=0.0) {
#ifndef _OPENMP
      K+=occs[i]*calcK(C.col(i));
#else
      arma::mat wK(occs[i]*calcK(C.col(i)));
#pragma omp critical
      K+=wK;
#endif
    }
  return K;
}

arma::cx_mat ERIchol::calcK(const arma::cx_mat & C, const std::vector<double> & occs) const {
  arma::cx_mat K(C.n_rows,C.n_rows);
  K.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<occs.size();i++)
    if(occs[i]!=0.0) {
#ifndef _OPENMP
      K+=occs[i]*calcK(C.col(i));
#else
      arma::cx_mat wK(occs[i]*calcK(C.col(i)));
#pragma omp critical
      K+=wK;
#endif
    }

  return K;
}

void ERIchol::B_matrix(arma::mat & Br) const {
  Br.zeros(Nbf*Nbf,B.n_cols);
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t i=0;i<prodidx.size();i++) {
      size_t u=invmap(0,i);
      size_t v=invmap(1,i);
      Br(u*Nbf+v,P)=B(i,P);
      Br(v*Nbf+u,P)=B(i,P);
    }
}

arma::mat ERIchol::B_transform(const arma::mat & Cl, const arma::mat & Cr, bool verbose) const {
  // Amount of basis and auxiliary functions
  if(Cl.n_rows != Nbf || Cr.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Orbital matrices don't match basis set! N = " << Nbf << ", N(Cl) = " << Cl.n_rows << ", N(Cr) = " << Cr.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  Timer t;

  // L_uv^P -> L_lv^P = L_uv^P C_lu
  arma::mat Ll(Cl.n_cols,B.n_cols*Nbf);
  Ll.zeros();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t i=0;i<prodidx.size();i++)
      for(size_t l=0;l<Cl.n_cols;l++)
	Ll(l,P*Nbf+invmap(0,i))+=B(i,P)*Cl(invmap(1,i),l);
  // Off-diagonal contribution
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t ii=0;ii<odiagidx.size();ii++) {
      size_t i=odiagidx(ii);
      for(size_t l=0;l<Cl.n_cols;l++)
	Ll(l,P*Nbf+invmap(1,i))+=B(i,P)*Cl(invmap(0,i),l);
    }

  if(verbose) {
    printf("First half-transform of B matrix done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  // Shuffle indices
  arma::mat Bs(Cl.n_cols*B.n_cols,Nbf);
  for(size_t mu=0;mu<Nbf;mu++)
    for(size_t P=0;P<B.n_cols;P++)
      for(size_t l=0;l<Cl.n_cols;l++)
	Bs(P*Cl.n_cols+l,mu)=Ll(l,P*Nbf+mu);

  if(verbose) {
    printf("Index shuffle done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  // Do RH transform
  Bs=Bs*Cr;

  if(verbose) {
    printf("Second half-transform done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  // Return array
  arma::mat Br(B.n_cols,Cl.n_cols*Cr.n_cols);
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t l=0;l<Cl.n_cols;l++)
      for(size_t r=0;r<Cr.n_cols;r++)
	Br(P,r*Cl.n_cols+l)=Bs(P*Cl.n_cols+l,r);

  if(verbose) {
    printf("Final index shuffle done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  return Br;
}
