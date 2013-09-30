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

bool operator<(const diis_pol_entry_t & lhs, const diis_pol_entry_t & rhs) {
  return lhs.E < rhs.E;
}

bool operator<(const diis_unpol_entry_t & lhs, const diis_unpol_entry_t & rhs) {
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
  pol.clear();
  unpol.clear();
}

void DIIS::update(const arma::mat & F, const arma::mat & P, double E, double & error) {
  if(pol.size()) {
    throw std::runtime_error("Trying to add spin-unpolarized entry to spin-polarized DIIS stack!\n");
  }

  // New entry
  diis_unpol_entry_t hlp;
  hlp.F=F;
  hlp.P=P;
  hlp.E=E;

  // Compute error matrix
  arma::mat errmat=F*P*S-S*P*F;
  // and transform it to the orthonormal basis (1982 paper, page 557)
  errmat=arma::trans(Sinvh)*errmat*Sinvh;
  // and store it
  hlp.err=MatToVec(errmat);

  // DIIS error is
  error=max_abs(errmat);

  // Is stack full?
  if(unpol.size()==imax) {
    unpol.erase(unpol.begin()+unpol.size()-1);
  }
  // Add to stack and resort
  unpol.push_back(hlp);
  std::stable_sort(unpol.begin(),unpol.end());
}

void DIIS::update(const arma::mat & Fa, const arma::mat & Fb, const arma::mat & Pa, const arma::mat & Pb, double E, double & error) {
  if(unpol.size()) {
    throw std::runtime_error("Trying to add spin-polarized entry to spin-unpolarized DIIS stack!\n");
  }

  // New entry
  diis_pol_entry_t hlp;
  hlp.Fa=Fa;
  hlp.Fb=Fb;
  hlp.Pa=Pa;
  hlp.Pb=Pb;
  hlp.E=E;

  // Compute error matrices
  arma::mat errmata=Fa*Pa*S-S*Pa*Fa;
  arma::mat errmatb=Fb*Pb*S-S*Pb*Fb;
  // and transform them to the orthonormal basis (1982 paper, page 557)
  arma::mat errmat=arma::trans(Sinvh)*(errmata+errmatb)*Sinvh;
  // and store it
  hlp.err=MatToVec(errmat);

  // DIIS error is
  error=max_abs(errmat);

  // Is stack full?
  if(pol.size()==imax) {
    pol.erase(pol.begin()+pol.size()-1);
  }
  // Add to stack and resort
  pol.push_back(hlp);
  std::stable_sort(pol.begin(),pol.end());
}

arma::vec DIIS::get_weights(bool c1) {
  // Collect error vectors
  std::vector<arma::vec> errs;
  if(unpol.size() && pol.size())
    throw std::runtime_error("Error - DIIS has both spin-polarized and spin-unpolarized error stacks!\n");
  if(unpol.size()) {
    errs.resize(unpol.size());
    for(size_t i=0;i<unpol.size();i++)
      errs[i]=unpol[i].err;
  } else if(pol.size()) {
    errs.resize(pol.size());
    for(size_t i=0;i<pol.size();i++)
      errs[i]=pol[i].err;
  }

  // Size of LA problem
  int N=(int) errs.size();

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
	  printf("Large coefficient produced by DIIS. Reducing to %i matrices.\n",N-1);
	  if(pol.size())
	    pol.erase(pol.begin()+pol.size()-1);
	  if(unpol.size())
	    unpol.erase(unpol.begin()+unpol.size()-1);
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

      sol.zeros();
      sol(0)=0.5;
      sol(1)=0.5;
    }
  }

  // arma::trans(sol).print("DIIS weights");

  return sol;
}

void DIIS::solve_F(arma::mat & F, bool c1) {
  if(!unpol.size())
    throw std::runtime_error("Trying to get DIIS-averaged spin-unpolarized Fock matrix while no entries exist!\n");
  
  arma::vec sol=get_weights(c1);
 
  // Form weighted Fock matrix
  F.zeros();
  for(size_t i=0;i<unpol.size();i++)
    F+=sol(i)*unpol[i].F;
}

void DIIS::solve_F(arma::mat & Fa, arma::mat & Fb, bool c1) {
  if(!pol.size())
    throw std::runtime_error("Trying to get DIIS-averaged spin-polarized Fock matrix while no entries exist!\n");
  
  arma::vec sol=get_weights(c1);
 
  // Form weighted Fock matrix
  Fa.zeros();
  Fb.zeros();
  for(size_t i=0;i<pol.size();i++) {
    Fa+=sol(i)*pol[i].Fa;
    Fb+=sol(i)*pol[i].Fb;
  }
}

void DIIS::solve_P(arma::mat & P, bool c1) {
  if(!unpol.size())
    throw std::runtime_error("Trying to get DIIS-averaged spin-unpolarized density matrix while no entries exist!\n");
  
  arma::vec sol=get_weights(c1);
 
  // Form weighted density matrix
  P.zeros();
  for(size_t i=0;i<unpol.size();i++)
    P+=sol(i)*unpol[i].P;
}

void DIIS::solve_P(arma::mat & Pa, arma::mat & Pb, bool c1) {
  if(!pol.size())
    throw std::runtime_error("Trying to get DIIS-averaged spin-polarized density matrix while no entries exist!\n");
  
  arma::vec sol=get_weights(c1);
 
  // Form weighted density matrix
  Pa.zeros();
  Pb.zeros();
  for(size_t i=0;i<pol.size();i++) {
    Pa+=sol(i)*pol[i].Pa;
    Pb+=sol(i)*pol[i].Pb;
  }
}


