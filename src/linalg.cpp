/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include <cfloat>
#include "linalg.h"
#include "mathf.h"
#include "settings.h"
#include "stringutil.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void sort_eigvec(arma::vec & eigval, arma::mat & eigvec) {
  sort_eigvec_wrk<double>(eigval,eigvec);
}

void sort_eigvec(arma::vec & eigval, arma::cx_mat & eigvec) {
  sort_eigvec_wrk< std::complex<double> >(eigval,eigvec);
}

void eig_sym_ordered(arma::vec & eigval, arma::mat & eigvec, const arma::mat & X) {
  eig_sym_ordered_wrk<double>(eigval,eigvec,X);
}

void eig_sym_ordered(arma::vec & eigval, arma::cx_mat & eigvec, const arma::cx_mat & X) {
  eig_sym_ordered_wrk< std::complex<double> >(eigval,eigvec,X);
}

arma::mat CholeskyOrth(const arma::mat & S) {
  // Cholesky orthogonalization
  return inv(chol(S));
}

extern Settings settings;

arma::mat SymmetricOrth(const arma::mat & Svec, const arma::vec & Sval) {
  // Compute inverse roots of eigenvalues
  arma::vec Sinvh(Sval);
  for(size_t i=0;i<Sinvh.n_elem;i++)
    if(Sinvh(i)>=settings.get_double("LinDepThresh"))
      Sinvh(i)=1/sqrt(Sinvh(i));
    else
      Sinvh(i)=0.0;

  // Returned matrix is
  return Svec*arma::diagmat(Sinvh)*trans(Svec);
}

arma::mat SymmetricOrth(const arma::mat & S) {
  // Symmetric orthogonalization

  // Eigendecomposition of S: eigenvalues and eigenvectors
  arma::vec Sval;
  arma::mat Svec;
  eig_sym_ordered(Sval,Svec,S);

  // Compute the decomposition
  return SymmetricOrth(Svec,Sval);
}

arma::mat CanonicalOrth(const arma::mat & Svec, const arma::vec & Sval, double cutoff) {
  // Count number of eigenvalues that are above cutoff
  const size_t Nbf=Svec.n_rows;

  size_t Nlin=0;
  for(size_t i=0;i<Nbf;i++)
    if(Sval(i)>=cutoff)
      Nlin++;
  // Number of linearly dependent basis functions
  size_t Ndep=Nbf-Nlin;

  // Form returned matrix
  arma::mat Sinvh(Nbf,Nlin);
  for(size_t i=0;i<Nlin;i++)
    Sinvh.col(i)=Svec.col(Ndep+i)/sqrt(Sval(Ndep+i));

  return Sinvh;
}

arma::mat CanonicalOrth(const arma::mat & S, double cutoff) {
  // Canonical orthogonalization

  if(S.n_cols != S.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Cannot orthogonalize non-square matrix!\n";
    throw std::runtime_error(oss.str());
  }

  // Eigendecomposition of S: eigenvalues and eigenvectors
  arma::vec Sval;
  arma::mat Svec;

  // Compute the decomposition
  eig_sym_ordered(Sval,Svec,S);
  return CanonicalOrth(Svec,Sval,cutoff);
}

arma::mat PartialCholeskyOrth(const arma::mat & S, double cholcut, double scut) {
  // Partial Cholesky orthogonalization
  if(S.n_cols != S.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Cannot orthogonalize non-square matrix!\n";
    throw std::runtime_error(oss.str());
  }

  // Off-diagonal S
  arma::mat odS(arma::abs(S));
  odS.diag().zeros();
  // Column sum
  arma::vec odSs(arma::sum(S).t());
  arma::uvec pivot = arma::stable_sort_index(odSs,"ascend");
  // Find suitable basis by partial Cholesky decomposition
  pivoted_cholesky(S,cholcut,pivot);

  // Canonical orthogonalization of subbasis
  arma::mat Ssub(S(pivot,pivot));
  arma::mat Xsub(CanonicalOrth(Ssub,scut));

  arma::mat X(S.n_rows,Xsub.n_cols,arma::fill::zeros);
  X.rows(pivot)=Xsub;
  return X;
}

arma::mat BasOrth(const arma::mat & S, bool verbose) {
  // Symmetric if possible, otherwise canonical. Default cutoff
  const double tol=settings.get_double("LinDepThresh");
  // Cholesky threshold
  const double chtol=settings.get_double("CholDepThresh");

  // Eigendecomposition of S: eigenvalues and eigenvectors
  arma::vec Sval;
  arma::mat Svec;
  // Compute the decomposition
  eig_sym_ordered(Sval,Svec,S);

  if(verbose) {
    printf("Smallest eigenvalue of overlap matrix is %.2e, reciprocal condition number is %.2e.\n",Sval(0),Sval(0)/Sval(Sval.n_elem-1));
  }

  // Check condition number
  if(Sval(0)/Sval(Sval.n_elem-1) <= DBL_EPSILON) {
    if(verbose) printf("Using partial Cholesky orthogonalization (doi: 10.1063/1.5139948).\n");

    return PartialCholeskyOrth(S,chtol,tol);

    // Check smallest eigenvalue.
  } else if(Sval(0)>=tol) {
    // OK to use symmetric
    if(verbose) printf("Using symmetric orthogonalization.\n");

    return SymmetricOrth(Svec,Sval);
  } else {
    if(verbose) printf("Using canonical orthogonalization.\n");

    // Have to drop eigenvectors. Use canonical.
    return CanonicalOrth(Svec,Sval,tol);
  }
}

arma::mat BasOrth(const arma::mat & S) {
  // Orthogonalize basis

  // Get wanted method
  std::string met=settings.get_string("BasisOrth");
  // Verbose operation?
  bool verbose=settings.get_bool("Verbose");

  if(stricmp(met,"auto")==0) {
    return BasOrth(S,verbose);
  } else if(stricmp(met,"Can")==0) {
    // Canonical orthogonalization
    double tol=settings.get_double("LinDepThresh");
    return CanonicalOrth(S,tol);
  } else if(stricmp(met,"Sym")==0) {
    // Symmetric orthogonalization
    return SymmetricOrth(S);
  } else if(stricmp(met,"Chol")==0) {
    return CholeskyOrth(S);
  } else {
    ERROR_INFO();
    std::ostringstream oss;
    oss << met << " is not a valid orthogonalization keyword.\n";
    throw std::domain_error(oss.str());
    return arma::mat();
  }
}

void S_half_invhalf(const arma::mat & S, arma::mat & Shalf, arma::mat & Sinvh, double cutoff) {
  if(S.n_cols != S.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Cannot orthogonalize non-square matrix!\n";
    throw std::runtime_error(oss.str());
  }

  // Size of basis
  const size_t Nbf=S.n_cols;

  // Eigendecomposition of S: eigenvalues and eigenvectors
  arma::vec Sval;
  arma::mat Svec;

  // Compute the decomposition
  eig_sym_ordered(Sval,Svec,S);

  // Count number of eigenvalues that are above cutoff
  size_t Nind=0;
  for(size_t i=0;i<Nbf;i++)
    if(Sval(i)>=cutoff)
      Nind++;
  // Number of linearly dependent basis functions
  size_t Ndep=Nbf-Nind;

  // Form Shalf and Sinvhalf
  Shalf.zeros(Nbf,Nbf);
  Sinvh.zeros(Nbf,Nbf);
  for(size_t i=0;i<Nind;i++) {
    size_t icol=Ndep+i;
    Sinvh+=Svec.col(icol)*arma::trans(Svec.col(icol))/sqrt(Sval(icol));
    Shalf+=Svec.col(icol)*arma::trans(Svec.col(icol))*sqrt(Sval(icol));
  }
}

arma::vec MatToVec(const arma::mat & m) {
  return arma::vectorise(m);
}

arma::mat VecToMat(const arma::vec & v, size_t nrows, size_t ncols) {
  return arma::reshape(v,nrows,ncols);
}

/// Get vector from cube: c(i,j,:)
arma::vec slicevec(const arma::cube & c, size_t i, size_t j) {
  arma::vec v(c.n_slices);
  for(size_t k=0;k<c.n_slices;k++)
    v(k)=c(i,j,k);
  return v;
}


arma::mat cosmat(const arma::mat & U) {
  // Compute eigendecomposition
  arma::vec evals;
  arma::mat evec;
  eig_sym_ordered(evals,evec,U);

  arma::mat cosU=U;

  // Check eigenvalues
  bool ok=0;
  for(size_t i=0;i<evals.n_elem;i++)
    if(fabs(evals(i))>DBL_EPSILON) {
      ok=1;
      break;
    }

  if(ok) {
    // OK to use general formula
    cosU.zeros();
    for(size_t i=0;i<evals.n_elem;i++)
      cosU+=std::cos(evals(i))*evec.col(i)*arma::trans(evec.col(i));
  } else {
    printf("Looks like U is singular. Using power expansion for cos.\n");

    // Zeroth order
    cosU.eye();
    // U^2
    arma::mat Usq=U*U;

    cosU+=0.5*Usq*(-1.0 + Usq*(1/12.0 - 1/360.0*Usq));
  }

  return cosU;
}

arma::mat sinmat(const arma::mat & U) {
  // Compute eigendecomposition
  arma::vec evals;
  arma::mat evec;
  eig_sym_ordered(evals,evec,U);

  arma::mat sinU=U;

  // Check eigenvalues
  bool ok=0;
  for(size_t i=0;i<evals.n_elem;i++)
    if(fabs(evals(i))>DBL_EPSILON) {
      ok=1;
      break;
    }

  if(ok) {
    // OK to use general formula
    sinU.zeros();
    for(size_t i=0;i<evals.n_elem;i++)
      sinU+=std::sin(evals(i))*evec.col(i)*arma::trans(evec.col(i));
  } else {
    printf("Looks like U is singular. Using power expansion for sin.\n");

    // U^2
    arma::mat Usq=U*U;

    sinU=U;
    sinU+=1.0/6.0*U*Usq*(-1.0 + 1.0/20.0*Usq*(1.0 - 1.0/42.0*Usq));
  }

  return sinU;
}

arma::mat sincmat(const arma::mat & U) {
  // Compute eigendecomposition
  arma::vec evals;
  arma::mat evec;
  eig_sym_ordered(evals,evec,U);

  arma::mat sincU=U;

  // Check eigenvalues
  bool ok=0;
  for(size_t i=0;i<evals.n_elem;i++)
    if(fabs(evals(i))>DBL_EPSILON) {
      ok=1;
      break;
    }

  if(ok) {
    // OK to use general formula
    sincU.zeros();
    for(size_t i=0;i<evals.n_elem;i++)
      sincU+=sinc(evals(i))*evec.col(i)*arma::trans(evec.col(i));
  } else {
    printf("Looks like U is singular. Using power expansion for sinc.\n");

    // U^2
    arma::mat Usq=U*U;

    sincU.eye();
    sincU+=1.0/6.0*Usq*(-1.0 + 1.0/20.0*Usq*(1.0 - 1.0/42.0*Usq));
  }

  return sincU;
}

arma::mat sqrtmat(const arma::mat & M) {
  arma::vec evals;
  arma::mat evec;
  eig_sym_ordered(evals,evec,M);

  arma::mat sqrtM=M;

  // Check eigenvalues
  if(evals(0)<0) {
    ERROR_INFO();
    throw std::runtime_error("Negative eigenvalue of matrix!\n");
  }

  sqrtM.zeros();
  for(size_t i=0;i<evals.n_elem;i++)
    sqrtM+=std::sqrt(evals(i))*evec.col(i)*arma::trans(evec.col(i));

  return sqrtM;
}

arma::mat expmat(const arma::mat & M) {
  arma::vec eval;
  arma::mat evec;
  eig_sym_ordered(eval,evec,M);

  return evec*arma::diagmat(arma::exp(eval))*arma::trans(evec);
}

arma::cx_mat expmat(const arma::cx_mat & M) {
  arma::vec eval;
  arma::cx_mat evec;
  eig_sym_ordered(eval,evec,M);

  return evec*arma::diagmat(arma::exp(eval))*arma::trans(evec);
}

void check_unitarity(const arma::cx_mat & W) {
  arma::cx_mat prod((arma::trans(W)*W)-arma::eye<arma::mat>(W.n_cols,W.n_cols));
  double norm=rms_cnorm(prod);

  if(norm>=sqrt(DBL_EPSILON)) {
    std::ostringstream oss;
    oss << "Matrix is not unitary: || W W^H -1 || = " << norm << "!\n";
    oss << prod;
    throw std::runtime_error(oss.str());
  }
}

void check_orthogonality(const arma::mat & W) {
  arma::mat prod((arma::trans(W)*W)-arma::eye<arma::mat>(W.n_cols,W.n_cols));
  double norm=rms_norm(prod);

  if(norm>=sqrt(DBL_EPSILON)) {
    std::ostringstream oss;
    oss << "Matrix is not orthogonal: || W W^T -1 || = " << norm << "!\n";
    throw std::runtime_error(oss.str());
  }
}

arma::mat orthogonalize(const arma::mat & M) {
  // Decomposition: M = U s V'
  arma::mat U;
  arma::vec s;
  arma::mat V;
  bool svdok=arma::svd(U,s,V,M);
  if(!svdok) {
    ERROR_INFO();
    M.print("M");
    throw std::runtime_error("SVD failed.\n");
  }

  // Return matrix with singular values set to unity
  return U*arma::trans(V);
}

arma::cx_mat unitarize(const arma::cx_mat & M) {
  // Decomposition: M = U s V'
  arma::cx_mat U;
  arma::vec s;
  arma::cx_mat V;
  bool svdok=arma::svd(U,s,V,M);
  if(!svdok) {
    ERROR_INFO();
    M.print("M");
    throw std::runtime_error("SVD failed.\n");
  }

  // Return matrix with singular values set to unity
  return U*arma::trans(V);
}

arma::mat orthonormalize(const arma::mat & S, const arma::mat & C) {
  // Compute MO overlap
  arma::mat MOovl=arma::trans(C)*S*C;

  // Perform eigendecomposition
  arma::vec oval;
  arma::mat ovec;
  eig_sym_ordered(oval,ovec,MOovl);

  // Orthogonalizing matrix
  arma::mat O(ovec*diagmat(arma::pow(oval,-0.5))*arma::trans(ovec));

  // Returned orbitals
  return C*O;
}

arma::cx_mat orthonormalize(const arma::mat & S, const arma::cx_mat & C) {
  // Compute MO overlap
  arma::cx_mat MOovl=arma::trans(C)*S*C;

  // Perform eigendecomposition
  arma::vec oval;
  arma::cx_mat ovec;
  eig_sym_ordered(oval,ovec,MOovl);

  // Orthogonalizing matrix
  arma::cx_mat O(ovec*diagmat(arma::pow(oval,-0.5))*arma::trans(ovec));

  // Returned orbitals
  return C*O;
}

void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::mat & NO_to_AO, arma::vec & occs) {
  // Get canonical half-overlap and half-inverse overlap matrices
  arma::mat Sh, Sinvh;
  S_half_invhalf(S,Sh,Sinvh,settings.get_double("LinDepThresh"));

  // P in orthonormal basis is
  arma::mat P_orth=arma::trans(Sh)*P*Sh;

  // Diagonalize P to get NOs in orthonormal basis.
  arma::vec Pval;
  arma::mat Pvec;
  eig_sym_ordered(Pval,Pvec,P_orth);

  // Reverse ordering to get decreasing eigenvalues
  occs.zeros(Pval.n_elem);
  arma::mat Pv(Pvec.n_rows,Pvec.n_cols);
  for(size_t i=0;i<Pval.n_elem;i++) {
    size_t idx=Pval.n_elem-1-i;
    occs(i)=Pval(idx);
    Pv.col(i)=Pvec.col(idx);
  }

  /* Get NOs in AO basis. The natural orbital is written in the
     orthonormal basis as

     |i> = x_ai |a> = x_ai s_ja |j>
     = s_ja x_ai |j>
  */

  // The matrix that takes us from AO to NO is
  AO_to_NO=Sinvh*Pv;
  // and the one that takes us from NO to AO is
  NO_to_AO=arma::trans(Sh*Pv);
}


void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::vec & occs) {
  arma::mat tmp;
  form_NOs(P,S,AO_to_NO,tmp,occs);
}

arma::mat pivoted_cholesky(const arma::mat & A, double eps, arma::uvec & pivot) {
  if(A.n_rows != A.n_cols)
    throw std::runtime_error("Pivoted Cholesky requires a square matrix!\n");

  // Returned matrix
  arma::mat L;
  L.zeros(A.n_rows,A.n_cols);

  // Loop index
  size_t m(0);
  // Diagonal element vector
  arma::vec d(arma::diagvec(A));
  // Error
  double error(arma::max(d));

  // Pivot index
  arma::uvec pi=pivot;

  while(error>eps && m<d.n_elem) {
    // Errors in pivoted order
    arma::vec errs(d(pi));
    // Sort the upcoming errors so that largest one is first
    arma::uvec idx=arma::stable_sort_index(errs.subvec(m,d.n_elem-1),"descend");

    // Update the pivot index
    arma::uvec pisub(pi.subvec(m,d.n_elem-1));
    pisub=pisub(idx);
    pi.subvec(m,d.n_elem-1)=pisub;

    // Pivot index
    size_t pim=pi(m);
    //printf("Pivot index is %4i with error %e, error is %e\n",(int) pim, d(pim), error);

    // Compute diagonal element
    L(m,pim)=sqrt(d(pim));

    // Off-diagonal elements
    for(size_t i=m+1;i<d.n_elem;i++) {
      size_t pii=pi(i);
      // Compute element
      L(m,pii)= (m>0) ? (A(pim,pii) - arma::dot(L.col(pim).subvec(0,m-1),L.col(pii).subvec(0,m-1)))/L(m,pim) : (A(pim,pii))/L(m,pim);
      // Update d
      d(pii)-=L(m,pii)*L(m,pii);
    }

    // Update error
    if(m+1<pi.n_elem)
      error=arma::max(d(pi.subvec(m+1,pi.n_elem-1)));
    // Increase m
    m++;
  }
  //printf("Final error is %e\n",error);

  // Transpose to get Cholesky vectors as columns
  arma::inplace_trans(L);

  // Drop unnecessary columns
  if(m<L.n_cols)
    L.shed_cols(m,L.n_cols-1);

  // Store pivot
  pivot=pi.subvec(0,m-1);

  return L;
}

arma::mat pivoted_cholesky(const arma::mat & A, double eps) {
  arma::uvec p;
  return pivoted_cholesky(A,eps,p);
}

// Same algorithm as above, only for a fixed amount of vectors
arma::mat incomplete_cholesky(const arma::mat & A, size_t n) {
  if(A.n_rows != A.n_cols)
    throw std::runtime_error("Pivoted Cholesky requires a square matrix!\n");

  // Returned matrix
  arma::mat L;
  L.zeros(A.n_rows,n);

  // Diagonal element vector
  arma::vec d(arma::diagvec(A));

  // Pivot index
  arma::uvec pi(arma::linspace<arma::uvec>(0,d.n_elem-1,d.n_elem));

  for(size_t m=0;m<n;m++) {
    // Errors in pivoted order
    arma::vec errs(d(pi));
    // Sort the upcoming errors so that largest one is first
    arma::uvec idx=arma::stable_sort_index(errs.subvec(m,d.n_elem-1),"descend");

    // Update the pivot index
    arma::uvec pisub(pi.subvec(m,d.n_elem-1));
    pisub=pisub(idx);
    pi.subvec(m,d.n_elem-1)=pisub;

    // Pivot index
    size_t pim=pi(m);

    // Compute diagonal element
    L(pim,m)=sqrt(d(pim));

    // Off-diagonal elements
    for(size_t i=m+1;i<d.n_elem;i++) {
      size_t pii=pi(i);
      // Compute element
      L(pii,m)=(A(pii,pim) - arma::dot(L.row(pim).subvec(0,m-1),L.row(pii).subvec(0,m-1)))/L(pim,m);
      // Update d
      d(pii)-=L(pii,m)*L(pii,m);
    }
  }

  return L;
}

arma::mat B_transform(arma::mat B, const arma::mat & Cl, const arma::mat & Cr) {
  if(Cl.n_rows != Cr.n_rows)
    throw std::logic_error("Orbital matrices aren't consistent!\n");
  if(B.n_rows != Cl.n_rows * Cr.n_rows)
    throw std::logic_error("B matrix does not correspond to orbital basis!\n");

  // Amount of basis and auxiliary functions
  size_t Nbf(Cl.n_rows);
  size_t Naux(B.n_cols);

  // Do LH transform
  B.reshape(Nbf,Nbf*Naux);
  B=arma::trans(Cl)*B;

  // Shuffle indices
  arma::mat Bs(Cl.n_cols*Naux,Nbf);
  for(size_t mu=0;mu<Nbf;mu++)
    for(size_t a=0;a<Naux;a++)
      for(size_t l=0;l<Cl.n_cols;l++)
	Bs(a*Cl.n_cols+l,mu)=B(l,a*Nbf+mu);

  // Do RH transform
  Bs=Bs*Cr;

  // Return array
  B.resize(Naux,Cl.n_cols*Cr.n_cols);
  for(size_t a=0;a<Naux;a++)
    for(size_t l=0;l<Cl.n_cols;l++)
      for(size_t r=0;r<Cr.n_cols;r++)
	B(a,r*Cl.n_cols+l)=Bs(a*Cl.n_cols+l,r);

  return B;
}


void check_lapack_thread() {
#ifdef _OPENMP
  // Size of problem
  size_t N=100;
  // Allocate memory
  arma::mat M(N,N);
  // Fill with random data
  M.randn();
  // Symmetrize
  M=(M+arma::trans(M))/2.0;

  // Eigendecomposition
  arma::vec eval;
  arma::mat evec;
  eig_sym_ordered(eval,evec,M);

  // Avoid possible degeneracies - reset eigenvalues
  for(size_t i=0;i<N;i++)
    eval(i)=i+1;
  // and recreate matrix
  M=evec*arma::diagmat(eval)*arma::trans(evec);
  // and rerun decomposition
  eig_sym_ordered(eval,evec,M);
  evec.save("seq.dat",arma::raw_ascii);

  // Difference matrix
  arma::mat dmat(omp_get_max_threads(),N+1);
  dmat.zeros();

  // Thread check
#pragma omp parallel
  {
    arma::vec thval;
    arma::mat thvec;
    eig_sym_ordered(thval,thvec,M);

    int ith=omp_get_thread_num();
    std::ostringstream oss;
    oss << "th_" << ith << ".dat";
    thvec.save(oss.str(),arma::raw_ascii);

    // Calculate eigenvector errors
    for(size_t i=0;i<N;i++)
      dmat(ith,i)=arma::norm(evec.col(i)-thvec.col(i),2);
    // Eigenvalues
    dmat(ith,N)=arma::norm(thval-eval,2);
  }

  //  dmat.print("RMS difference norms");

  if(arma::max(arma::max(dmat)) > 1e-8) {
    printf("Warning - LAPACK library doesn't seem to be thread safe!\n");
    printf("Max error in eigenvectors %e\n",arma::max(arma::max(dmat)));
    fprintf(stderr,"Warning - LAPACK library doesn't seem to be thread safe!\n");
  } else {
    printf("LAPACK checks out fine.\n");
  }
#endif
}
