/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "hirshfeldi.h"
#include "dftgrid.h"
#include "guess.h"
#include "elements.h"
#include "timer.h"

HirshfeldI::HirshfeldI(const BasisSet & basis, const arma::mat & P, std::string method, double tol, double drv, int dqmax, bool verbose) {
  // Store grid spacing
  dr=drv;

  Timer ttot;
  
  // Store atomic centers.
  cen.resize(basis.get_Nnuc());
  for(size_t i=0;i<cen.size();i++)
    cen[i]=basis.get_nucleus(i).r;

  // Reserve memory for atomic densities
  atoms.resize(basis.get_Nnuc());
  atQ.resize(basis.get_Nnuc());

  // Get list of identical nuclei
  std::vector< std::vector<size_t> > idnuc=identical_nuclei(basis);

  Timer t;
  if(verbose)
    printf("\t%4s %2s %6s %4s\n","atom","el","charge","time");

  // Loop over list of identical nuclei. (Can't parallellize here because of HDF)
  for(size_t i=0;i<idnuc.size();i++) {
    // Get the nucleus
    nucleus_t nuc=basis.get_nucleus(idnuc[i][0]);

    // Resize storage
    for(size_t j=0;j<idnuc[i].size();j++) {
      atoms[idnuc[i][j]].resize(2*dqmax+1);
      atQ[idnuc[i][j]].assign(2*dqmax+1,0.0);
    }
    
    // Loop over acceptable charge states
    for(int q=std::max(1,nuc.Z-dqmax);q<=nuc.Z+dqmax;q++) {
      Timer tatq;

      // Charge difference to neutral species is
      int dq=q-nuc.Z;

      // Perform guess
      arma::vec atE;
      arma::mat atP;
      BasisSet atbas;
      std::vector<size_t> shellidx;
      // Don't drop polarization shells, but do occupation smearing
      atomic_guess(basis,idnuc[i][0],method,shellidx,atbas,atE,atP,false,true,dq);
      
      // Construct atom
      HirshfeldAtom at(atbas,atP,dr);
      // and store it
      for(size_t j=0;j<idnuc[i].size();j++) {
	atoms[idnuc[i][j]][dq+dqmax]=at.get_rho();
	atQ[idnuc[i][j]][dq+dqmax]=q;
      }

      if(verbose) {
	printf("\t%4i %-2s %6i %s\n",(int) idnuc[i][0]+1, element_symbols[nuc.Z].c_str(),dq,tatq.elapsed().c_str());
      }
    }
  }

  if(verbose) {
    printf("Computed Hirshfeld atoms in %s.\n\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  // Starting guess: neutral species
  arma::vec q(cen.size());
  for(size_t i=0;i<cen.size();i++) {
    nucleus_t nuc=basis.get_nucleus(i);
    if(nuc.bsse)
      q[i]=0.0;
    else
      q[i]=nuc.Z;
  }
  ISA=get(q);

  //  arma::trans(q).print("Starting guess");

  // First iteration
  if(verbose) {
    printf("First iteration\n");
  }
  iterate(basis,P,q,tol,verbose);
  if(verbose) {
    printf("Converged in %s.\n\n",t.elapsed().c_str());
    t.set();
    printf("Second iteration\n");
  }
  // Second iteration
  iterate(basis,P,q,tol,verbose);
  if(verbose) {
    printf("Converged in %s.\n",t.elapsed().c_str());
    printf("Iterative Hirshfeld decomposition computed in %s.\n\n",ttot.elapsed().c_str());
  }
}

HirshfeldI::~HirshfeldI() {
}

void HirshfeldI::iterate(const BasisSet & basis, const arma::mat & P, arma::vec & q, double tol, bool verbose) {
  // Helper.
  DFTGrid intgrid(&basis,verbose);
  // Construct grid
  intgrid.construct_hirshfeld(ISA,tol);

  size_t iiter=0;
  if(verbose)
    printf("%5s  %12s  %12s %6s\n","iter","max","mean","t");

  while(true) {
    iiter++;
    Timer t;

    // Get new charges
    arma::vec newq=intgrid.compute_atomic_Nel(ISA,P);
    if(verbose) {
      printf("%5i  %e  %e %s\n",(int) iiter,arma::max(arma::abs(q-newq)),arma::mean(arma::abs(q-newq)),t.elapsed().c_str());
      fflush(stdout);
    }

    //    arma::trans(newq).print("Atomic charges");

    // Check convergence
    double dq=arma::max(arma::abs(q-newq));
    if(dq<tol)
      // Converged
      break;

    // Update atoms
    ISA=get(newq);
    // and the charges
    q=newq;
  }
}

Hirshfeld HirshfeldI::get(const arma::vec & Q) {
  // Atomic densities
  std::vector< std::vector<double> > rho(cen.size());

  // Loop over atoms
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<cen.size();i++) {
    if(Q[i]==0.0) {
      // No charge
      rho[i].push_back(0.0);
      continue;
    }

    // Find out upper limit
    size_t upl;
    for(upl=1;upl<atoms[i].size();upl++)
      if(atQ[i][upl] >= Q[i])
	break;

    if(Q[i]<atQ[i][0] || upl==atoms[i].size()) {
      printf("Charge on atom %i is %e.\n",(int) i+1, Q[i]);
      printf("Charge array:");
      for(size_t j=0;j<atQ[i].size();j++)
	printf(" %i",atQ[i][j]);
      printf("\n");

      if(upl==atoms[i].size())
	throw std::runtime_error("Overrun in Hirshfeld-I routine. Increase dq parameter.\n");
      else
	throw std::runtime_error("Underrun in Hirshfeld-I routine. Increase dq parameter.\n");
    }

    // Form linear combination
    double uc=Q[i]-atQ[i][upl-1];
    double lc=1.0-uc;

    // Initialize
    rho[i].resize(std::max(atoms[i][upl].size(),atoms[i][upl-1].size()));

    // Compute superposition.
    for(size_t j=0;j<std::min(atoms[i][upl].size(),atoms[i][upl-1].size());j++)
      // Both cover this range
      rho[i][j]=uc*atoms[i][upl][j] + lc*atoms[i][upl-1][j];
    if(atoms[i][upl].size()>atoms[i][upl-1].size()) {
      // Only upper limit exists here
      for(size_t j=atoms[i][upl-1].size();j<atoms[i][upl].size();j++)
	rho[i][j]=uc*atoms[i][upl][j];
    } else {
      // Only lower limit exists here
      for(size_t j=atoms[i][upl].size();j<atoms[i][upl-1].size();j++)
	rho[i][j]=lc*atoms[i][upl-1][j];
    }
  }

  // Returned object
  Hirshfeld hirsh;
  hirsh.set(cen,dr,rho);
  return hirsh;
}

Hirshfeld HirshfeldI::get() const {
  return ISA;
}
