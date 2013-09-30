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

bool operator<(const diis_entry_t & lhs, const diis_entry_t & rhs) {
  return lhs.E < rhs.E;
}

DIIS::DIIS(const arma::mat & Sv, size_t imaxv) {
  S=Sv;
  
  // Get half-inverse
  arma::mat Sh;
  S_half_invhalf(S,Sh,Sinvh);

  imax=imaxv;
}

DIIS::~DIIS() {
}

void DIIS::clear() {
  stack.clear();
}


void DIIS::update(const arma::mat & F, const arma::mat & P, double E, double & error) {
  // New entry
  diis_entry_t hlp;
  hlp.F=F;
  hlp.P=P;
  hlp.E=E;

  // Compute error matrix
  arma::mat errmat=F*P*S-S*P*F;
  // and transform it to the orthonormal basis (1982 paper, page 557)
  errmat=Sinvh*errmat*Sinvh;
  // and store it
  hlp.err=MatToVec(errmat);

  // DIIS error is
  error=max_abs(errmat);

  // Is stack full?
  if(stack.size()==imax) {
    stack.erase(stack.begin()+stack.size()-1);
  }
  // Add to stack and resort
  stack.push_back(hlp);
  std::stable_sort(stack.begin(),stack.end());
}

arma::vec DIIS::get_weights(bool c1) {
  // Size of LA problem
  int N=(int) stack.size();

  // DIIS weights
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
	B(i,j)=arma::dot(stack[i].err,stack[j].err);
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
	  printf("Large coefficient produced by DIIS. Reducing to %i matrices.\n",(int) stack.size()-1);
	  stack.erase(stack.begin()+stack.size()-1);
	  return get_weights(c1);
	}

      // Solution is (last element of X is DIIS error)
      sol.zeros();
      for(int i=0;i<N;i++)
	sol(i)=X(i);
    }

    if(!succ) {
      // Failed to invert matrix. Use the two last iterations instead.
      printf("C1-DIIS was not succesful, mixing matrices instead.\n");
      sol.zeros();
      sol(0)=0.5;
      sol(1)=0.5;
    }
  } else {
    // C2-DIIS

    // Array holding the errors
    arma::mat B(N,N);
    B.zeros();

    // Compute errors
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++) {
	B(i,j)=arma::dot(stack[i].err,stack[j].err);
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
    arma::vec werr(stack[0].err.n_elem);
    for(int i=0;i<N;i++) {
      // Zero out helper
      werr.zeros();
      // Compute weighed error
      for(int j=0;j<N;j++)
	werr+=Q(j,i)*stack[j].err;
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

      sol.zeros();
      sol(0)=0.5;
      sol(1)=0.5;
    }
  }

  // arma::trans(sol).print("DIIS weights");

  return sol;
}

void DIIS::solve_F(arma::mat & F, bool c1) {
  arma::vec sol=get_weights(c1);
 
  // Form weighted Fock matrix
  F.zeros();
  for(size_t i=0;i<stack.size();i++)
    F+=sol(i)*stack[i].F;
}

void DIIS::solve_P(arma::mat & P, bool c1) {
  arma::vec sol=get_weights(c1);

  // Form weighted density matrix
  P.zeros();
  for(size_t i=0;i<stack.size();i++)
    P+=sol(i)*stack[i].P;
}
