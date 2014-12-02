/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "trrh.h"
#include "linalg.h"
#include "mathf.h"
#include "timer.h"
#include "basis.h"

#include <cfloat>

/// Minimal allowed value for projection
#define MINA 0.98

/// Maximum allowed number of mu iterations
#define NMU 50

/// Mu increment
#define DELTAMU 1.0
/// Minimal allowed value for level shift
#define EPSILONMU 1e-4

/// Throw error if eigenvalues of K^2 are bigger than (should be negative!)
#define MAXNEGEIG 1e-4

template<typename T> arma::Mat<T> make_expK(const arma::Mat<T> & kappa) {
  // Amount of virtual orbitals
  const size_t nvirt=kappa.n_rows;
  // Amount of occupied orbitals
  const size_t nocc=kappa.n_cols;
  // Total amount of orbitals
  const size_t norbs=nocc+nvirt;

  // K^2 is block diagonal, so we can do the diagonalization in
  // subblocks.

  // Compute kkt and ktk
  arma::Mat<T> kkt=-kappa*arma::trans(kappa);
  arma::Mat<T> ktk=-arma::trans(kappa)*kappa;

  // Do eigendecomposition
  arma::vec kktval;
  arma::Mat<T> kktvec;
  eig_sym_ordered_wrk<T>(kktval,kktvec,kkt);

  arma::vec ktkval;
  arma::Mat<T> ktkvec;
  eig_sym_ordered_wrk<T>(ktkval,ktkvec,ktk);

  // Clean up eigenvalues
  for(size_t i=0;i<kktval.n_elem;i++) {
    // Check sanity
    if(kktval(i)>=MAXNEGEIG) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "kkt part of K^2 has eigenvalue " << kktval(i) << "!\n";
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
  arma::Mat<T> coskkt(kkt);
  coskkt.zeros();
  arma::Mat<T> sinckkt(kkt);
  sinckkt.zeros();
  for(size_t i=0;i<kktval.n_elem;i++) {
    coskkt +=kktvec.col(i)*cos (kktval(i))*arma::trans(kktvec.col(i));
    sinckkt+=kktvec.col(i)*sinc(kktval(i))*arma::trans(kktvec.col(i));
  }

  arma::Mat<T> cosktk(ktk);
  cosktk.zeros();
  arma::Mat<T> sincktk(ktk);
  sincktk.zeros();
  for(size_t i=0;i<ktkval.n_elem;i++) {
    cosktk +=ktkvec.col(i)*cos (ktkval(i))*arma::trans(ktkvec.col(i));
    sincktk+=ktkvec.col(i)*sinc(ktkval(i))*arma::trans(ktkvec.col(i));
  }

  // Form the total matrices
  arma::Mat<T> K(norbs,norbs);
  K.zeros();
  K.submat(nocc,0,norbs-1,nocc-1)=kappa;
  K.submat(0,nocc,nocc-1,norbs-1)=-arma::trans(kappa);

  arma::Mat<T> cosKsq(norbs,norbs);
  cosKsq.zeros();
  cosKsq.submat(0,0,nocc-1,nocc-1)=cosktk;
  cosKsq.submat(nocc,nocc,norbs-1,norbs-1)=coskkt;

  arma::Mat<T> sincKsq(norbs,norbs);
  sincKsq.zeros();
  sincKsq.submat(0,0,nocc-1,nocc-1)=sincktk;
  sincKsq.submat(nocc,nocc,norbs-1,norbs-1)=sinckkt;

  // K matrix is
  return cosKsq+sincKsq*K;
}

template<typename T> void TRRH_update_wrk(const arma::Mat<T> & F_AO, const arma::Mat<T> & C, const arma::mat & S, arma::Mat<T> & Cnew, arma::vec & Enew, size_t nocc, double minshift, bool verbose) {
  // Transform Fock matrix into MO basis
  const arma::Mat<T> F_MO=arma::trans(C)*F_AO*C;

  // Number of orbitals
  const size_t norbs=C.n_cols;
  // Number of virtual orbitals
  const size_t nvirt=norbs-nocc;

  // Diagonalize occupied-occupied and virtual-virtual blocks of Fock matrix.
  arma::Mat<T> F_oo=F_MO.submat(0,0,nocc-1,nocc-1);
  arma::Mat<T> F_vv=F_MO.submat(nocc,nocc,norbs-1,norbs-1);

  arma::Mat<T> oo_vec;
  arma::vec oo_eig;
  eig_sym_ordered_wrk<T>(oo_eig,oo_vec,F_oo);

  arma::Mat<T> vv_vec;
  arma::vec vv_eig;
  eig_sym_ordered_wrk<T>(vv_eig,vv_vec,F_vv);

  // Transform orbitals into new basis
  arma::Mat<T> C_ov(C);
  C_ov.cols(0,nocc-1)=C.cols(0,nocc-1)*oo_vec;
  C_ov.cols(nocc,norbs-1)=C.cols(nocc,norbs-1)*vv_vec;

  // Compute Fock matrix in new basis
  arma::Mat<T> F_ov=arma::trans(C_ov)*F_AO*C_ov;
  // Force symmetricity of F_ov
  F_ov=(F_ov + arma::trans(F_ov))/2.0;

  // Save (pseudo-)orbital energies
  Enew=arma::diagvec(arma::real(F_ov));

  // Form gradient and Hessian, eqns (29a) and (29b)
  arma::Mat<T> grad(nvirt,nocc);
  grad.zeros();

  // Hessian is diagonal, so we only take the diagonal part
  arma::mat hess(nvirt,nocc);
  hess.zeros();

  // Fill the matrices, remembering the ordering of the full Fock matrix
  for(size_t a=0;a<nvirt;a++)
    for(size_t i=0;i<nocc;i++) {
      grad(a,i)=-4.0*F_ov(nocc+a,i); // eq (27b)
      hess(a,i)=4.0*(Enew(nocc+a)-Enew(i)); // eq (29b)
    }

  // Get (approximation to) smallest negative eigenvalue
  double minhess=hess.min();

  // Print legend
  if(verbose) {
    printf("\t%2s %12s %5s time\n","it","mu","Amin");
    fflush(stdout);
  }

  // Check shift value is sane
  if(minshift<EPSILONMU)
    minshift=EPSILONMU;

  // Increase mu until the change is small enough
  const double fac=2.0;
  double mu=minshift/fac;
  size_t iit=0;
  double amin;

  bool refine=false;
  double lmu=0.0;
  double rmu=0.0;
  
  while(true) {
    iit++;
    Timer t;
    
    // Value of mu is
    if(!refine)
      mu*=fac;
    else {
      mu=(lmu+rmu)/2.0;
    }

    // Get rotation parameters
    arma::Mat<T> kappa(nvirt,nocc);
    for(size_t a=0;a<nvirt;a++)
      for(size_t i=0;i<nocc;i++)
	kappa(a,i)=-grad(a,i)/(hess(a,i)-minhess+mu);

    // Get rotation matrix
    arma::Mat<T> expK=make_expK<T>(kappa);

    // New orbital coefficients are given by exp(-K)
    Cnew=C_ov*arma::trans(expK);

    // Calculate minimal projection.
    amin=DBL_MAX;
    arma::Mat<T> proj=arma::trans(C.cols(0,nocc-1))*S*Cnew.cols(0,nocc-1);
    for(size_t i=0;i<nocc;i++) {
      // Compute projection
      double a=arma::norm(proj.row(i),2);
      if(a<amin)
	amin=a;
    }

    if(verbose) {
      printf("\t%2i %e %.3f %s\n",(int) iit,mu,amin,t.elapsed().c_str());
      fflush(stdout);
    }

    // Have we reached the refine stage?
    if(amin>=MINA && !refine) {
      // Did we converge straight away?
      if(mu==minshift)
	break;

      refine=true;
      rmu=mu;
      lmu=mu/fac;
    } else if(refine) {
      // Determine what to do.
      if(amin<MINA)
	lmu=mu;
      else if(amin>MINA)
	rmu=mu;
      else
	break;

      // Have we determined the shift with the wanted accuracy?
      if(fabs(amin-MINA) <= 1e-4)
	break;
    }
  }

  if(verbose) {
    printf("mu loop converged in %i iterations\n",(int) iit);
    fflush(stdout);
  }

  // Make absolutely sure orbitals stay orthonormal.
  for(size_t i=0;i<norbs;i++) {
    for(size_t j=0;j<i;j++)
      Cnew.col(i)-=arma::as_scalar(arma::trans(Cnew.col(i))*S*Cnew.col(j))*Cnew.col(j);

    Cnew.col(i)/=sqrt(arma::as_scalar(arma::trans(Cnew.col(i))*S*Cnew.col(i)));
  }
}

void TRRH_update(const arma::mat & F_AO, const arma::mat & C, const arma::mat & S, arma::mat & Cnew, arma::vec & Enew, size_t nocc, double minshift, bool verbose) {
  TRRH_update_wrk<double>(F_AO,C,S,Cnew,Enew,nocc,minshift,verbose);

  // Debug
  try {
    check_orth(Cnew,S,false);
  } catch(std::runtime_error err) {
    ERROR_INFO();
    throw std::runtime_error("Orbitals updated by TRRH are not orthonormal!\n");
  }
}

void TRRH_update(const arma::cx_mat & F_AO, const arma::cx_mat & C, const arma::mat & S, arma::cx_mat & Cnew, arma::vec & Enew, size_t nocc, double minshift, bool verbose) {
  TRRH_update_wrk< std::complex<double> >(F_AO,C,S,Cnew,Enew,nocc,minshift,verbose);

  // Debug
  try {
    check_orth(Cnew,S,false);
  } catch(std::runtime_error err) {
    ERROR_INFO();
    throw std::runtime_error("Orbitals updated by TRRH are not orthonormal!\n");
  }
}

