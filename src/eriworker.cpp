/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "eriworker.h"
#include "linalg.h"
#include "mathf.h"

IntegralWorker::IntegralWorker() {
}

IntegralWorker::~IntegralWorker() {
}

void IntegralWorker::spherical_transform(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  const bool is_lm=is->lm_in_use();
  const bool js_lm=js->lm_in_use();
  const bool ks_lm=ks->lm_in_use();
  const bool ls_lm=ls->lm_in_use();

  const int am_i=is->get_am();
  const int am_j=js->get_am();
  const int am_k=ks->get_am();
  const int am_l=ls->get_am();

  const size_t Nj_tgt=js->get_Nbf();
  const size_t Nk_tgt=ks->get_Nbf();
  const size_t Nl_tgt=ls->get_Nbf();

  const size_t Ni_cart=is->get_Ncart();
  const size_t Nj_cart=js->get_Ncart();
  const size_t Nk_cart=ks->get_Ncart();

  // Do transforms
  if(ls_lm)
    transform_l(am_l,Ni_cart,Nj_cart,Nk_cart);
  if(ks_lm)
    transform_k(am_k,Ni_cart,Nj_cart,Nl_tgt);
  if(js_lm)
    transform_j(am_j,Ni_cart,Nk_tgt,Nl_tgt);
  if(is_lm)
    transform_i(am_i,Nj_tgt,Nk_tgt,Nl_tgt);
}

ERIWorker::ERIWorker(int maxam, int maxcontr) {
  if(maxam>=LIBINT_MAX_AM) {
    ERROR_INFO();
    throw std::domain_error("You need a version of LIBINT that supports larger angular momentum.\n");
  }

  // Initialize evaluator
  init_libint(&libint,maxam,pow(maxcontr,4));
}

ERIWorker::~ERIWorker() {
  free_libint(&libint);
}

dERIWorker::dERIWorker(int maxam, int maxcontr) {
  if(maxam>=LIBDERIV_MAX_AM1) {
    ERROR_INFO();
    throw std::domain_error("You need a version of LIBDERIV that supports larger angular momentum.\n");
  }

  // Maximum amount of basis functions is
  int nbf=(maxam+1)*(maxam+2)/2;

  // Initialize evaluator
  init_libderiv1(&libderiv,maxam,pow(maxcontr,4),pow(nbf,4));
}

dERIWorker::~dERIWorker() {
  free_libderiv(&libderiv);
}

void ERIWorker::compute_cartesian(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  eri_precursor_t ip=compute_precursor(is,js);
  eri_precursor_t jp=compute_precursor(ks,ls);

  // Compute shell of cartesian ERIs using libint

  // Libint computes (ab|cd) for
  // l(a)>=l(b), l(c)>=l(d) and l(a)+l(b)>=l(c)+l(d)
  // where l(a) is the angular momentum type of the a shell.
  // THIS ROUTINE ASSUMES THAT THE SHELLS ARE ALREADY IN CORRECT ORDER!

  // The sum of angular momenta is
  int mmax=is->get_am()+js->get_am()+ks->get_am()+ls->get_am();

  // Figure out the number of contractions
  size_t Ncomb=is->get_Ncontr()*js->get_Ncontr()*ks->get_Ncontr()*ls->get_Ncontr();

  // Check that all is in order
  if(is->get_am()<js->get_am()) {
    ERROR_INFO();
    throw std::runtime_error("lambda_i < lambda_j\n");
  }

  if(ks->get_am()<ls->get_am()) {
    ERROR_INFO();
    throw std::runtime_error("lambda_k < lambda_l\n");
  }

  if( (is->get_am()+js->get_am()) > (ks->get_am()+ls->get_am())) {
    ERROR_INFO();
    throw std::runtime_error("lambda_k + lambda_l < lambda_i + lambda_j\n");
  }

  // Compute data for LIBINT
  compute_libint_data(ip,jp,mmax);

  // Pointer to integrals table
  double *ints;

  // Special handling of (ss|ss) integrals:
  if(mmax==0) {
    double tmp=0.0;
    for(size_t i=0;i<Ncomb;i++)
      tmp+=libint.PrimQuartet[i].F[0];

    // Plug in normalizations
    tmp*=is->get_cart()[0].relnorm;
    tmp*=js->get_cart()[0].relnorm;
    tmp*=ks->get_cart()[0].relnorm;
    tmp*=ls->get_cart()[0].relnorm;

    input.resize(1);
    input[0]=tmp;
  } else {
    //    printf("Computing shell %i %i %i %i",shells[is].get_am(),shells[js].get_am(),shells[ks].get_am(),shells[ls].get_am());
    //    printf("which consists of basis functions (%i-%i)x(%i-%i)x(%i-%i)x(%i-%i).\n",(int) shells[is].get_first_ind(),(int) shells[is].get_last_ind(),(int) shells[js].get_first_ind(),(int) shells[js].get_last_ind(),(int) shells[ks].get_first_ind(),(int) shells[ks].get_last_ind(),(int) shells[ls].get_first_ind(),(int) shells[ls].get_last_ind());

    // Now we can compute the integrals using libint:
    ints=build_eri[is->get_am()][js->get_am()][ks->get_am()][ls->get_am()](&libint,Ncomb);

    // and collect the results, plugging in the normalization factors
    size_t ind_i, ind_ij, ind_ijk, ind;
    double norm_i, norm_ij, norm_ijk, norm;

    // Numbers of functions on each shell
    std::vector<shellf_t> ci=is->get_cart();
    std::vector<shellf_t> cj=js->get_cart();
    std::vector<shellf_t> ck=ks->get_cart();
    std::vector<shellf_t> cl=ls->get_cart();
    input.resize(ci.size()*cj.size()*ck.size()*cl.size());

    for(size_t ii=0;ii<ci.size();ii++) {
      ind_i=ii*cj.size();
      norm_i=ci[ii].relnorm;
      for(size_t ji=0;ji<cj.size();ji++) {
	ind_ij=(ind_i+ji)*ck.size();
	norm_ij=cj[ji].relnorm*norm_i;
	for(size_t ki=0;ki<ck.size();ki++) {
	  ind_ijk=(ind_ij+ki)*cl.size();
	  norm_ijk=ck[ki].relnorm*norm_ij;
	  for(size_t li=0;li<cl.size();li++) {
	    // Index in computed integrals table
	    ind=ind_ijk+li;
	    // Total norm factor
	    norm=cl[li].relnorm*norm_ijk;
	    // Compute output index
	    input[ind]=norm*ints[ind];
	  }
	}
      }
    }
  }
}

void dERIWorker::compute_cartesian() {
  eri_precursor_t ip=compute_precursor(is,js);
  eri_precursor_t jp=compute_precursor(ks,ls);

  // Compute shell of cartesian ERI derivatives using libderiv

  // Libint computes (ab|cd) for
  // l(a)>=l(b), l(c)>=l(d) and l(a)+l(b)>=l(c)+l(d)
  // where l(a) is the angular momentum type of the a shell.
  // THIS ROUTINE ASSUMES THAT THE SHELLS ARE ALREADY IN CORRECT ORDER!

  // The sum of angular momenta is
  int mmax=is->get_am()+js->get_am()+ks->get_am()+ls->get_am();

  // Figure out the number of contractions
  size_t Ncomb=is->get_Ncontr()*js->get_Ncontr()*ks->get_Ncontr()*ls->get_Ncontr();

  // Check that all is in order
  if(is->get_am()<js->get_am()) {
    ERROR_INFO();
    throw std::runtime_error("lambda_i < lambda_j\n");
  }

  if(ks->get_am()<ls->get_am()) {
    ERROR_INFO();
    throw std::runtime_error("lambda_k < lambda_l\n");
  }

  if( (is->get_am()+js->get_am()) > (ks->get_am()+ls->get_am())) {
    ERROR_INFO();
    throw std::runtime_error("lambda_k + lambda_l < lambda_i + lambda_j\n");
  }

  // Compute data for LIBINT
  compute_libderiv_data(ip,jp,mmax);

  // Now we can compute the integrals using libint:
  build_deriv1_eri[is->get_am()][js->get_am()][ks->get_am()][ls->get_am()](&libderiv,Ncomb);

  // and plug in the normalization factors.
  size_t ind_i, ind_ij, ind_ijk, ind;
  double norm_i, norm_ij, norm_ijk, norm;
  
  // Numbers of functions on each shell
  std::vector<shellf_t> ci=is->get_cart();
  std::vector<shellf_t> cj=js->get_cart();
  std::vector<shellf_t> ck=ks->get_cart();
  std::vector<shellf_t> cl=ls->get_cart();
  
  // Integrals computed by libderiv
  const int idx[]={0, 1, 2, 6, 7, 8, 9, 10, 11};

  for(size_t ii=0;ii<ci.size();ii++) {
    ind_i=ii*cj.size();
    norm_i=ci[ii].relnorm;
    for(size_t ji=0;ji<cj.size();ji++) {
      ind_ij=(ind_i+ji)*ck.size();
      norm_ij=cj[ji].relnorm*norm_i;
      for(size_t ki=0;ki<ck.size();ki++) {
	  ind_ijk=(ind_ij+ki)*cl.size();
	  norm_ijk=ck[ki].relnorm*norm_ij;
	  for(size_t li=0;li<cl.size();li++) {
	    // Index in computed integrals table
	    ind=ind_ijk+li;
	    // Total norm factor
	    norm=cl[li].relnorm*norm_ijk;
	    // Normalize integrals
	    for(size_t iidx=0;iidx<sizeof(idx)/sizeof(idx[0]);iidx++)
	      libderiv.ABCD[idx[iidx]][ind]*=norm;
	  }
      }
    }
  }
}


void dERIWorker::get(int idx, std::vector<double> & ints) {
  // First, copy data. Amount of integrals
  size_t N=(is->get_Ncart())*(js->get_Ncart())*(ks->get_Ncart())*(ls->get_Ncart());
  input.resize(N);

  if((idx>=0 && idx<3) || (idx>5 && idx<12)) {
    // Integrals have been computed explicitely.
    for(size_t i=0;i<N;i++)
      input[i]=libderiv.ABCD[idx][i];
  } else {
    // Derivate wrt B_i: d/dB_i = - d/dA_i - d/dC_i - d/dD_i
    for(size_t i=0;i<N;i++)
      input[i]= -libderiv.ABCD[idx-3][i] - libderiv.ABCD[idx+3][i] - libderiv.ABCD[idx+6][i];
  }
  
  // Reorder integrals
  reorder(is,js,ks,ls,swap_ij,swap_kl,swap_ijkl);
  // and transform them into the spherical basis
  spherical_transform(is_orig,js_orig,ks_orig,ls_orig);
  // and return them
  ints=input;
}

eri_precursor_t IntegralWorker::compute_precursor(const GaussianShell *is, const GaussianShell *js) {
  // Returned precursor
  eri_precursor_t r;

  // Initialize arrays
  r.AB.zeros(3);

  r.zeta.zeros(is->get_Ncontr(),js->get_Ncontr());
  r.P.zeros(is->get_Ncontr(),js->get_Ncontr(),3);
  r.PA.zeros(is->get_Ncontr(),js->get_Ncontr(),3);
  r.PB.zeros(is->get_Ncontr(),js->get_Ncontr(),3);
  r.S.zeros(is->get_Ncontr(),js->get_Ncontr());

  // Get data
  r.ic=is->get_contr();
  r.jc=js->get_contr();

  coords_t Ac=is->get_center();
  arma::vec A(3);
  A(0)=Ac.x;
  A(1)=Ac.y;
  A(2)=Ac.z;

  coords_t Bc=js->get_center();
  arma::vec B(3);
  B(0)=Bc.x;
  B(1)=Bc.y;
  B(2)=Bc.z;

  // Compute AB
  r.AB=A-B;
  double rabsq=arma::dot(r.AB,r.AB);

  // Compute zeta
  for(size_t i=0;i<r.ic.size();i++)
    for(size_t j=0;j<r.jc.size();j++)
      r.zeta(i,j)=r.ic[i].z+r.jc[j].z;

  // Form P
  for(size_t i=0;i<r.ic.size();i++)
    for(size_t j=0;j<r.jc.size();j++)
      for(int k=0;k<3;k++)
	r.P(i,j,k)=(r.ic[i].z*A(k) + r.jc[j].z*B(k))/r.zeta(i,j);

  // Compute PA
  for(size_t i=0;i<r.ic.size();i++)
    for(size_t j=0;j<r.jc.size();j++)
      for(int k=0;k<3;k++)
	r.PA(i,j,k)=r.P(i,j,k)-A(k);

  // Compute PB
  for(size_t i=0;i<r.ic.size();i++)
    for(size_t j=0;j<r.jc.size();j++)
      for(int k=0;k<3;k++)
	r.PB(i,j,k)=r.P(i,j,k)-B(k);

  // Compute S
  for(size_t i=0;i<r.ic.size();i++)
    for(size_t j=0;j<r.jc.size();j++)
      r.S(i,j)=r.ic[i].c*r.jc[j].c*(M_PI/r.zeta(i,j))*sqrt(M_PI/r.zeta(i,j))*exp(-r.ic[i].z*r.jc[j].z/r.zeta(i,j)*rabsq);

  return r;
}

void ERIWorker::compute_libint_data(const eri_precursor_t & ip, const eri_precursor_t &jp, int mmax) {
  // Store AB and CD
  for(int i=0;i<3;i++) {
    libint.AB[i]=ip.AB(i);
    libint.CD[i]=jp.AB(i);
  }

  size_t ind=0;

  // Compute primitive data
  for(size_t p=0;p<ip.ic.size();p++)
    for(size_t q=0;q<ip.jc.size();q++) {
      double zeta=ip.zeta(p,q);

      // Compute overlaps for auxiliary integrals
      double S12=ip.S(p,q);

      for(size_t r=0;r<jp.ic.size();r++)
	for(size_t s=0;s<jp.jc.size();s++) {
	  double eta=jp.zeta(r,s);

	  // Overlap for auxiliary integral
	  double S34=jp.S(r,s);

	  // Reduced exponent
          double rho=zeta*eta/(zeta+eta);

	  // Four-product center
	  arma::vec W=(zeta*slicevec(ip.P,p,q)+eta*slicevec(jp.P,r,s))/(zeta+eta);

	  arma::vec PQ=slicevec(ip.P,p,q)-slicevec(jp.P,r,s);
	  arma::vec WP=W-slicevec(ip.P,p,q);
	  arma::vec WQ=W-slicevec(jp.P,r,s);

	  double rpqsq=arma::dot(PQ,PQ);

	  // Helper variable
	  prim_data data;

          // Compute and store PA, QC, WP and WQ
          for(int i=0;i<3;i++) {
            data.U[0][i]=ip.PA(p,q,i); // PA
            data.U[2][i]=jp.PA(r,s,i); // QC
            data.U[4][i]=WP(i); // WP
            data.U[5][i]=WQ(i); // WQ
          }

	  // Store exponents
	  data.oo2z=0.5/zeta;
	  data.oo2n=0.5/eta;
	  data.oo2zn=0.5/(eta+zeta);
	  data.oo2p=0.5/rho;
	  data.poz=rho/zeta;
	  data.pon=rho/eta;

	  // Prefactor of Boys' function is
	  double prefac=2.0*sqrt(rho/M_PI)*S12*S34;
	  // and its argument is
	  double boysarg=rho*rpqsq;
	  // Evaluate Boys' function
	  std::vector<double> bf=boysF_arr(mmax,boysarg);

	  // Store auxiliary integrals
	  for(int i=0;i<=mmax;i++)
	    data.F[i]=prefac*bf[i];

	  // We have all necessary data; store quartet.
	  libint.PrimQuartet[ind++]=data;
	}
    }
}

void dERIWorker::compute_libderiv_data(const eri_precursor_t & ip, const eri_precursor_t &jp, int mmax) {
  // Store AB and CD
  for(int i=0;i<3;i++) {
    libderiv.AB[i]=ip.AB(i);
    libderiv.CD[i]=jp.AB(i);
  }

  size_t ind=0;

  // Compute primitive data
  for(size_t p=0;p<ip.ic.size();p++)
    for(size_t q=0;q<ip.jc.size();q++) {
      double zeta=ip.zeta(p,q);

      // Compute overlaps for auxiliary integrals
      double S12=ip.S(p,q);

      for(size_t r=0;r<jp.ic.size();r++)
	for(size_t s=0;s<jp.jc.size();s++) {
	  double eta=jp.zeta(r,s);

	  // Overlap for auxiliary integral
	  double S34=jp.S(r,s);

	  // Reduced exponent
          double rho=zeta*eta/(zeta+eta);

	  // Four-product center
	  arma::vec W=(zeta*slicevec(ip.P,p,q)+eta*slicevec(jp.P,r,s))/(zeta+eta);

	  arma::vec PQ=slicevec(ip.P,p,q)-slicevec(jp.P,r,s);
	  arma::vec WP=W-slicevec(ip.P,p,q);
	  arma::vec WQ=W-slicevec(jp.P,r,s);

	  double rpqsq=arma::dot(PQ,PQ);

	  // Helper variable
	  prim_data data;

          // Compute and store PA, QC, WP and WQ
          for(int i=0;i<3;i++) {
            data.U[0][i]=ip.PA(p,q,i); // PA
	    data.U[1][i]=ip.PB(p,q,i); // PB
            data.U[2][i]=jp.PA(r,s,i); // QC
	    data.U[3][i]=jp.PB(r,s,i); // QD
            data.U[4][i]=WP(i); // WP
            data.U[5][i]=WQ(i); // WQ
          }

	  // Store exponents
	  data.twozeta_a=2.0*ip.ic[p].z;
	  data.twozeta_b=2.0*ip.jc[q].z;
	  data.twozeta_c=2.0*jp.ic[r].z;
	  data.twozeta_d=2.0*jp.jc[s].z;

	  data.oo2z=0.5/zeta;
	  data.oo2n=0.5/eta;
	  data.oo2zn=0.5/(eta+zeta);
	  data.oo2p=0.5/rho;
	  data.poz=rho/zeta;
	  data.pon=rho/eta;

	  // Prefactor of Boys' function is
	  double prefac=2.0*sqrt(rho/M_PI)*S12*S34;
	  // and its argument is
	  double boysarg=rho*rpqsq;
	  // Evaluate Boys' function
	  std::vector<double> bf=boysF_arr(mmax+1,boysarg);

	  // Store auxiliary integrals
	  for(int i=0;i<=mmax+1;i++)
	    data.F[i]=prefac*bf[i];

	  // We have all necessary data; store quartet.
	  libderiv.PrimQuartet[ind++]=data;
	}
    }
}

void ERIWorker::compute(const GaussianShell *is_orig, const GaussianShell *js_orig, const GaussianShell *ks_orig, const GaussianShell *ls_orig, std::vector<double> & ints) {
  // Calculate ERIs and transform them to spherical harmonics basis, if necessary.

  // Use precursors. Helpers
  const GaussianShell *is=is_orig;
  const GaussianShell *js=js_orig;
  const GaussianShell *ks=ks_orig;
  const GaussianShell *ls=ls_orig;

  // Did we need to swap the indices?
  bool swap_ij=false;
  bool swap_kl=false;
  bool swap_ijkl=false;

  // Check order and swap shells if necessary
  if(is->get_am()<js->get_am()) {
    swap_ij=true;
    std::swap(is,js);
  }

  if(ks->get_am()<ls->get_am()) {
    swap_kl=true;
    std::swap(ks,ls);
  }

  if(is->get_am()+js->get_am() > ks->get_am() + ls->get_am()) {
    swap_ijkl=true;
    std::swap(is,ks);
    std::swap(js,ls);
  }

  // Get the cartesian ERIs
  compute_cartesian(is,js,ks,ls);
  // Reorder them
  reorder(is,js,ks,ls,swap_ij,swap_kl,swap_ijkl);
  // and transform them into the spherical basis
  spherical_transform(is_orig,js_orig,ks_orig,ls_orig);

  // Finally, return the integrals
  ints=input;
}

void dERIWorker::compute(const GaussianShell *is_origv, const GaussianShell *js_origv, const GaussianShell *ks_origv, const GaussianShell *ls_origv) {
  // Calculate derivative ERIs

  is_orig=is_origv;
  js_orig=js_origv;
  ks_orig=ks_origv;
  ls_orig=ls_origv;

  is=is_orig;
  js=js_orig;
  ks=ks_orig;
  ls=ls_orig;

  // Did we need to swap the indices?
  swap_ij=false;
  swap_kl=false;
  swap_ijkl=false;

  // Check order and swap shells if necessary
  if(is->get_am()<js->get_am()) {
    swap_ij=true;
    std::swap(is,js);
  }

  if(ks->get_am()<ls->get_am()) {
    swap_kl=true;
    std::swap(ks,ls);
  }

  if(is->get_am()+js->get_am() > ks->get_am() + ls->get_am()) {
    swap_ijkl=true;
    std::swap(is,ks);
    std::swap(js,ls);
  }

  // Get the cartesian ERIs
  compute_cartesian();
}

void IntegralWorker::reorder(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls, bool swap_ij, bool swap_kl, bool swap_ijkl) {
  // Integrals are now in the input array.
  // Get them in the original order
  if(swap_ij || swap_kl || swap_ijkl) {
    // Check that we have enough memory
    output.resize(input.size());
    
    // Indices
    size_t ind_i, ind_ij, ind_ijk, ind, indout;
    
    // Numbers of functions on each shell
    const size_t Ni=is->get_Ncart();
    const size_t Nj=js->get_Ncart();
    const size_t Nk=ks->get_Ncart();
    const size_t Nl=ls->get_Ncart();
    
    for(size_t ii=0;ii<Ni;ii++) {
      ind_i=ii*Nj;
      for(size_t ji=0;ji<Nj;ji++) {
	ind_ij=(ind_i+ji)*Nk;
	for(size_t ki=0;ki<Nk;ki++) {
	  ind_ijk=(ind_ij+ki)*Nl;
	  for(size_t li=0;li<Nl;li++) {
	    ind=ind_ijk+li;
	    indout=get_swapped_ind(ii,Ni,ji,Nj,ki,Nk,li,Nl,swap_ij,swap_kl,swap_ijkl);
	    output[indout]=input[ind];
	  }
	}
      }
    }
    
    // Swap arrays
    std::swap(input,output);
  }  
}

size_t IntegralWorker::get_swapped_ind(size_t i, size_t Ni, size_t j, size_t Nj, size_t k, size_t Nk, size_t l, size_t Nl, bool swap_ij, bool swap_kl, bool swap_ijkl) {
  // Compute indices of swapped integrals.

  // First, swap ij-kl if necessary.
  if(swap_ijkl) {
    std::swap(i,k);
    std::swap(Ni,Nk);
    std::swap(j,l);
    std::swap(Nj,Nl);
  }

  // Then, swap k-l if necessary.
  if(swap_kl) {
    std::swap(k,l);
    std::swap(Nk,Nl);
  }

  // Finally, swap i-j if necessary.
  if(swap_ij) {
    std::swap(i,j);
    std::swap(Ni,Nj);
  }

  // Now, compute the index
  return ((i * Nj + j) * Nk + k) * Nl + l;
}
