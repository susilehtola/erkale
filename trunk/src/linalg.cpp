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



#include <cfloat>
#include "linalg.h"
#include "mathf.h"

void eig_sym_ordered(arma::colvec & eigval, arma::mat & eigvec, const arma::mat & X) {
  /* Solve eigenvalues of symmetric matrix with guarantee
     of ordering of eigenvalues from smallest to biggest */

  // Solve eigenvalues and eigenvectors
  arma::eig_sym(eigval,eigvec,X);
  
  // Sort vectors
  sort_eigvec(eigval,eigvec);
}

void sort_eigvec(arma::colvec & eigval, arma::mat & eigvec) {

  // Get number of elements
  const size_t N=eigval.n_elem;

  size_t incr=N/2;
  arma::colvec tmpvec;
  double tmpval;

  while(incr>0) {
    // Loop over eigenvalues
    for(size_t i=incr;i<N;i++) {
      tmpval=eigval(i);
      tmpvec=eigvec.col(i);

      size_t j=i;
      while((j>=incr) && (eigval[j-incr]>tmpval)) {
	printf("j=%i\n",(int) j);
	eigval(j)=eigval(j-incr);
	eigvec.col(j)=eigvec.col(j-incr);
	j=j-incr;
      }

      eigval(j)=tmpval;
      eigvec.col(j)=tmpvec;
    }

    incr=(int) (incr/2.2);
  }
}

arma::mat CholeskyOrth(const arma::mat & S, double *ratio) {
  // Cholesky orthogonalization

  // Dummy value.
  if(ratio!=NULL)
    *ratio=-1.0;

  //  printf("Using Cholesky orthogonalization.\n");

  return inv(chol(S));
}

arma::mat SymmetricOrth(const arma::mat & S, double *ratio) {
  // Symmetric orthogonalization

  if(S.n_cols != S.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Cannot orthogonalize non-square matrix!\n";
    throw std::runtime_error(oss.str());
  }

  //  printf("Using symmetric orthogonalization.\n");

  // Size of basis
  const size_t Nbf=S.n_cols;

  // Eigendecomposition of S: eigenvalues and eigenvectors
  arma::vec Sval;
  arma::mat Svec;

  // Compute the decomposition
  eig_sym_ordered(Sval,Svec,S);

  if(ratio!=NULL)
    *ratio=Sval(0)/Sval(Nbf-1);

  // Compute matrix filled with inverse eigenvalues
  arma::mat invval(Nbf,Nbf);
  invval.zeros();
  for(size_t i=0;i<Nbf;i++)
    invval(i,i)=1.0/sqrt(Sval(i));

  // Returned matrix is
  return Svec*invval*trans(Svec);
}
  

arma::mat CanonicalOrth(const arma::mat & S, double cutoff, double *ratio) {
  // Canonical orthogonalization

  if(S.n_cols != S.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Cannot orthogonalize non-square matrix!\n";
    throw std::runtime_error(oss.str());
  }

  //  printf("Using canonical orthogonalization with cutoff=%.2e.\n",cutoff);

  // Size of basis
  const size_t Nbf=S.n_cols;

  // Eigendecomposition of S: eigenvalues and eigenvectors
  arma::vec Sval;
  arma::mat Svec;

  // Compute the decomposition
  eig_sym_ordered(Sval,Svec,S);

  // Count number of eigenvalues that are above cutoff
  size_t Nlin=0;
  for(size_t i=0;i<Nbf;i++)
    if(Sval(i)>=cutoff)
      Nlin++;
  // Number of linearly dependent basis functions
  size_t Ndep=Nbf-Nlin;

  if(ratio!=NULL)
    *ratio=Sval(0)/Sval(Nbf-1);

  // Form returned matrix
  arma::mat Sinvh(Nbf,Nlin);
  for(size_t i=0;i<Nlin;i++)
    Sinvh.col(i)=Svec.col(Ndep+i)/sqrt(Sval(Ndep+i));

  return Sinvh;
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
  size_t Nlin=0;
  for(size_t i=0;i<Nbf;i++)
    if(Sval(i)>=cutoff)
      Nlin++;
  // Number of linearly dependent basis functions
  size_t Ndep=Nbf-Nlin;

  // Form Shalf and Sinvhalf
  Shalf=arma::mat(Nbf,Nlin);
  Sinvh=arma::mat(Nbf,Nlin);

  Shalf.zeros();
  Sinvh.zeros();
  for(size_t i=0;i<Nlin;i++) {
    Sinvh.col(i)=Svec.col(Ndep+i)/sqrt(Sval(Ndep+i));
    Shalf.col(i)=Svec.col(Ndep+i)*sqrt(Sval(Ndep+i));
  }
}

arma::mat BasOrth(const arma::mat & S, const Settings & set, double & ratio) {
  // Orthogonalize basis

  // Get wanted method
  std::string met=set.get_string("BasisOrth");

  if(met=="Can") {
    // Canonical orthogonalization
    double tol=set.get_double("BasisLinTol");
    return CanonicalOrth(S,tol,&ratio);
  } else if(met=="Sym") {
    // Symmetric orthogonalization
    return SymmetricOrth(S,&ratio);
  } else if(met=="Chol") {
    return CholeskyOrth(S,&ratio);
  } else {
    ERROR_INFO();
    std::ostringstream oss;
    oss << met << " is not a valid orthogonalization keyword.\n";
    throw std::domain_error(oss.str());
    return arma::mat();
  }
}

arma::vec MatToVec(const arma::mat & m) {
  // Size of vector to return
  size_t N=m.n_cols*m.n_rows;

  // Returned vector
  arma::vec ret(N);

  // Store matrix
  for(size_t i=0;i<m.n_rows;i++)
    for(size_t j=0;j<m.n_cols;j++)
      ret(i*m.n_cols+j)=m(i,j);

  return ret;
}

arma::mat VecToMat(const arma::vec & v, size_t nrows, size_t ncols) {

  // Check size consistency
  if(nrows*ncols!=v.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Cannot reshape a vector to a differently sized matrix.\n");
  }

  arma::mat m(nrows,ncols);
  m.zeros();

  for(size_t i=0;i<nrows;i++)
    for(size_t j=0;j<ncols;j++)
      m(i,j)=v(i*ncols+j);

  return m;
}

arma::mat cos(const arma::mat & U) {
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

arma::mat sin(const arma::mat & U) {
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

arma::mat sinc(const arma::mat & U) {
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

arma::mat sqrt(const arma::mat & M) {
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
  
