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



#include <cfloat>
#include "diis.h"
#include "linalg.h"
#include "mathf.h"

// Maximum allowed absolute weight for a Fockian.
#define MAXWEIGHT 10.0

DIIS::DIIS(const arma::mat &Sv, size_t imaxv) {
  S=Sv;
  
  // Get half-inverse
  arma::mat Sh;
  S_half_invhalf(S,Sh,Sinvh);

  imax=imaxv;
  icur=0;
}

DIIS::~DIIS() {
}

void DIIS::clear() {
  Fs.clear();
  errs.clear();
  icur=0;
}


void DIIS::update(const arma::mat & F, const arma::mat & D, double & error) {
  // Compute error matrix
  arma::mat err=F*D*S-S*D*F;
  // and transform it to the orthonormal basis (1982 paper, page 557)
  err=Sinvh*err*Sinvh;

  double maxerr=max_abs(err);
  error=maxerr;

  // Do we have accumulated enough matrices?
  if((int) Fs.size()<imax) { // No, add to stack
    icur=(int) Fs.size();
    Fs.push_back(F);
    errs.push_back(MatToVec(err));
  } else { // Yes, replace oldest
    // Index is
    icur=(icur+1)%imax;
    Fs[icur]=F;
    errs[icur]=MatToVec(err);
  }
}

void DIIS::solve(arma::mat & F, bool c1) {
  // Size of LA problem
  int N;
  if((int) Fs.size()<imax)
    N=Fs.size();
  else
    N=imax;

  arma::vec sol(N);
  sol.zeros();

  if(c1) {
    // Original, Pulay's DIIS (C1-DIIS)

    // Array holding the errors
    arma::mat B(N+1,N+1);
    B.zeros();
    // RHS vector
    arma::vec A(N+1);
    A.zeros();

    // Compute errors
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++) {
	B(i,j)=arma::dot(errs[i],errs[j]);
      }
    // Fill in the rest of B
    for(int i=0;i<N;i++) {
      B(i,N)=-1.0;
      B(N,i)=-1.0;
    }

    // Fill in A
    A(N)=-1.0;

    // Solve B*X = A
    arma::vec X;
    bool succ;

    succ=arma::solve(X,B,A);

    if(succ) {
      // Check that weights are within tolerance
      for(int i=0;i<N;i++)
	if(fabs(X(i))>=MAXWEIGHT) {
	  printf("Large coefficient produced by DIIS. Reducing to %i matrices.\n",(int) Fs.size()-1);
	  Fs.erase(Fs.begin());
	  solve(F,c1);
	  return;
	}

      // Solution is (last element of X is DIIS error)
      sol.zeros();
      for(int i=0;i<N;i++)
	sol(i)=X(i);
    }

    if(!succ) {
      // Failed to invert matrix. Use the two last iterations instead.
      printf("C1-DIIS was not succesful, mixing matrices instead.\n");
      // Index of last iteration
      int ilast=(icur+imax-1)%imax;

      sol.zeros();
      sol(icur)=0.5;
      sol(ilast)=0.5;
    }
  } else {
    // C2-DIIS

    // Array holding the errors
    arma::mat B(N,N);
    B.zeros();

    // Compute errors
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++) {
	B(i,j)=arma::dot(errs[i],errs[j]);
      }

    // Solve eigenvectors of B
    arma::mat Q;
    arma::vec lambda;
    eig_sym_ordered(lambda,Q,B);

    // Normalize weights
    for(int i=0;i<N;i++) {
      double s=0;
      s=sum(Q.col(i));
      Q.col(i)/=s;
    }

    // Choose solution by picking out solution with smallest error
    std::vector<double> errors(N);
    // Helper array
    arma::vec werr(errs[0].n_elem);
    for(int i=0;i<N;i++) {
      // Zero out helper
      werr.zeros();
      // Compute weighed error
      for(int j=0;j<N;j++)
	werr+=Q(j,i)*errs[j];
      // The error is
      errors[i]=arma::dot(werr,werr);
    }

    // Find minimal error
    double mine=DBL_MAX;
    int minloc=-1;
    for(int i=0;i<N;i++) {
      if(errors[i]<mine) {
	// Check weights
	bool ok=1;
	for(int j=0;j<N;j++)
	  if(fabs(Q(j,i))>=MAXWEIGHT)
	    ok=0;

	if(ok) {
	  mine=errors[i];
	  minloc=i;
	}
      }
    }

    if(minloc!=-1) {
      // Solution is
      sol=Q.col(minloc);
    } else {
      printf("C2-DIIS did not find a suitable solution. Mixing matrices instead.\n");

      int ilast=(icur+imax-1)%imax;
      sol.zeros();
      sol(icur)=0.5;
      sol(ilast)=0.5;
    }
  }

  /*
  printf("DIIS weights are\n");
  for(int i=0;i<N;i++)
    printf(" % e",sol(i));
  printf("\n");
  */

  // Form weighted Fockian
  F.zeros();
  for(int i=0;i<N;i++)
    F+=sol(i)*Fs[i];
}
