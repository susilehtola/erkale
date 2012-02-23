/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "trrh.h"
#include "linalg.h"
#include "mathf.h"

#include <cfloat>

/// Minimal allowed value for projection
#define MINA 0.98

/// Maximum allowed number of mu iterations
#define NMU 50

/// Mu increment
#define DELTAMU 0.5
/// Minimal allowed value of mu so that division by zero doesn't occur
#define EPSMU 0.01

/// Throw error if eigenvalues of K^2 are bigger than (should be negative!)
#define MAXNEGEIG 1e-8

void TRRH_update(const arma::mat & F_AO, const arma::mat & C, const arma::mat & S, arma::mat & Cnew, arma::vec & Enew, size_t nocc, bool verbose) {
  // Transform Fock matrix into MO basis
  const arma::mat F_MO=arma::trans(C)*F_AO*C;

  // Number of basis functions
  const size_t nbf=C.n_rows;
  // Number of orbitals
  const size_t norbs=C.n_cols;
  // Number of virtual orbitals
  const size_t nvirt=norbs-nocc;

  // Diagonalize occupied-occupied and virtual-virtual blocks of Fock matrix.
  arma::mat F_oo=F_MO.submat(0,0,nocc-1,nocc-1);
  arma::mat F_vv=F_MO.submat(nocc,nocc,norbs-1,norbs-1);

  arma::mat oo_vec;
  arma::vec oo_eig;
  eig_sym_ordered(oo_eig,oo_vec,F_oo);

  arma::mat vv_vec;
  arma::vec vv_eig;
  eig_sym_ordered(vv_eig,vv_vec,F_vv);

  // Transform orbitals into new basis
  arma::mat C_ov(C);
  C_ov.submat(0,0,nbf-1,nocc-1)=C.submat(0,0,nbf-1,nocc-1)*oo_vec;
  C_ov.submat(0,nocc,nbf-1,norbs-1)=C.submat(0,nocc,nbf-1,norbs-1)*vv_vec;

  // Compute Fock matrix in new basis
  arma::mat F_ov=arma::trans(C_ov)*F_AO*C_ov;
  // Force symmetrization of F_ov
  F_ov=(F_ov + arma::trans(F_ov))/2.0;

  // Save (pseudo-)orbital energies
  Enew.zeros(F_ov.n_rows);
  for(size_t i=0;i<F_ov.n_rows;i++)
    Enew(i)=F_ov(i,i);

  // Form gradient and Hessian, eqns (29a) and (29b)
  arma::mat grad(nvirt,nocc);
  grad.zeros();

  // Hessian is diagonal, so we only take the diagonal part
  arma::mat hess(nvirt,nocc);
  hess.zeros();

  // Fill the matrices, remembering the ordering of the full Fock matrix
  for(size_t a=0;a<nvirt;a++)
    for(size_t i=0;i<nocc;i++) {
      grad(a,i)=-4.0*F_ov(nocc+a,i);
      hess(a,i)=4.0*(F_ov(nocc+a,nocc+a)-F_ov(i,i));
    }

  // Get (approximation to) smallest negative eigenvalue
  double minhess=0.0;
  for(size_t a=0;a<nvirt;a++)
    for(size_t i=0;i<nocc;i++)
      if(hess(a,i)<minhess)
	minhess=hess(a,i);

  // Get the rotated coefficients
  size_t iit;
  for(iit=0;iit<NMU;iit++) {
    // Value of mu is
    double mu=EPSMU-minhess+iit*DELTAMU;

    // Get rotation parameters
    arma::mat kappa(nvirt,nocc);
    for(size_t a=0;a<nvirt;a++)
      for(size_t i=0;i<nocc;i++)
	kappa(a,i)=-grad(a,i)/(hess(a,i)+mu);

    // Get rotation matrix
    arma::mat expK=TRRH::make_expK(kappa);

    // New orbital coefficients are given by exp(-K)
    Cnew=C_ov*arma::trans(expK);
    
    // Calculate minimal projection
    double amin=DBL_MAX;
    arma::mat proj=arma::trans(C.submat(0,0,nbf-1,nocc-1))*S*Cnew.submat(0,0,nbf-1,nocc-1);
    for(size_t i=0;i<nocc;i++) {
      // Compute projection
      double a=0.0;
      for(size_t j=0;j<nocc;j++)
	a+=proj(i,j)*proj(i,j);

      if(a<amin)
	amin=a;
    }
    
    if(verbose)
      printf("\t%2i %e %e\n",(int) iit,mu,amin);
    
    if(amin>=MINA)
      break;
  }
  
  if(iit==NMU) {
    printf("Warning - wanted level shift not found.\n");
    fprintf(stderr,"Warning - wanted level shift not found.\n");
  } else if(verbose)
    printf("mu loop converged in %i iterations\n",(int) iit+1);
}

arma::mat TRRH::make_expK(const arma::mat & kappa) {
  // Amount of virtual orbitals
  const size_t nvirt=kappa.n_rows;
  // Amount of occupied orbitals
  const size_t nocc=kappa.n_cols;
  // Total amount of orbitals
  const size_t norbs=nocc+nvirt;

  // K^2 is block diagonal, so we can do the diagonalization in
  // subblocks.
  
  // Compute kkt and ktk
  arma::mat kkt=-kappa*arma::trans(kappa);
  arma::mat ktk=-arma::trans(kappa)*kappa;

  // Do eigendecomposition
  arma::vec kktval;
  arma::mat kktvec;
  eig_sym_ordered(kktval,kktvec,kkt);

  arma::vec ktkval;
  arma::mat ktkvec;
  eig_sym_ordered(ktkval,ktkvec,ktk);

  // Clean up eigenvalues
  for(size_t i=0;i<kktval.n_elem;i++) {
    // Check sanity
    if(kktval(i)>=MAXNEGEIG) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "kkt part of K^2 has eivenvalue " << kktval(i) << "!\n";
      throw std::runtime_error(oss.str());
    }

    if(kktval(i)>0.0)
      kktval(i)=0.0;
  }

  for(size_t i=0;i<ktkval.n_elem;i++) {
    // Check sanity
    if(ktkval(i)>=MAXNEGEIG) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "ktk part of K^2 has eivenvalue " << ktkval(i) << "!\n";
      throw std::runtime_error(oss.str());
    }

    if(ktkval(i)>0.0)
      ktkval(i)=0.0;
  }

  // Transform eigenvalues
  for(size_t i=0;i<kktval.n_elem;i++)
    kktval(i)=sqrt(-kktval(i));
  for(size_t i=0;i<ktkval.n_elem;i++)
    ktkval(i)=sqrt(-ktkval(i));

  // Form the cos and sinc matrices
  arma::mat coskkt(kkt);
  coskkt.zeros();
  arma::mat sinckkt(kkt);
  sinckkt.zeros();
  for(size_t i=0;i<kktval.n_elem;i++) {
    coskkt +=kktvec.col(i)*cos (kktval(i))*arma::trans(kktvec.col(i));
    sinckkt+=kktvec.col(i)*sinc(kktval(i))*arma::trans(kktvec.col(i));    
  }
  
  arma::mat cosktk(ktk);
  cosktk.zeros();
  arma::mat sincktk(ktk);
  sincktk.zeros();
  for(size_t i=0;i<ktkval.n_elem;i++) {
    cosktk +=ktkvec.col(i)*cos (ktkval(i))*arma::trans(ktkvec.col(i));
    sincktk+=ktkvec.col(i)*sinc(ktkval(i))*arma::trans(ktkvec.col(i));
  }

  // Form the total matrices
  arma::mat K(norbs,norbs);
  K.zeros();
  K.submat(nocc,0,norbs-1,nocc-1)=kappa;
  K.submat(0,nocc,nocc-1,norbs-1)=-arma::trans(kappa);

  arma::mat cosKsq(norbs,norbs);
  cosKsq.zeros();
  cosKsq.submat(0,0,nocc-1,nocc-1)=cosktk;
  cosKsq.submat(nocc,nocc,norbs-1,norbs-1)=coskkt;

  arma::mat sincKsq(norbs,norbs);
  sincKsq.zeros();
  sincKsq.submat(0,0,nocc-1,nocc-1)=sincktk;
  sincKsq.submat(nocc,nocc,norbs-1,norbs-1)=sinckkt;

  // K matrix is
  return cosKsq+sincKsq*K;
}
