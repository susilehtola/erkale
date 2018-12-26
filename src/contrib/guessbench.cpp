/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "basislibrary.h"
#include "basis.h"
#include "checkpoint.h"
#include "dftgrid.h"
#include "elements.h"
#include "find_molecules.h"
#include "guess.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
#include "sap.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"

// Needed for libint init
#include "eriworker.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

void print_header() {
#ifdef _OPENMP
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

void diag(arma::vec & E, arma::mat & C, const arma::mat & H, const arma::mat & Sinvh) {
  arma::eig_sym(E,C,Sinvh.t()*H*Sinvh);
  C=Sinvh*C;

  //E.print("Eigenvalues");
}

int main(int argc, char **argv) {
  print_header();

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_scf_settings();
  set.add_string("LoadChk","File to load old results from","");
  set.add_double("LinDepThr","Linear dependency threshold",1e-5);
  set.parse(std::string(argv[1]),true);
  set.print();

  // Basis set
  BasisSet basis;
  // Get checkpoint file
  std::string chkf(set.get_string("LoadChk"));
  Checkpoint chk(chkf,false);
  chk.read(basis);

  // Read in density matrix
  arma::mat Pa, Pb;
  chk.read("Pa",Pa);
  chk.read("Pb",Pb);
  
  // Number of electrons
  int Nela;
  chk.read("Nel-a",Nela);
  int Nelb;
  chk.read("Nel-b",Nelb);

  // Overlap matrix
  arma::mat S(basis.overlap());
  // and its half-inverse
  double linthr(set.get_double("LinDepThr"));
  arma::mat Sinvh(CanonicalOrth(S,linthr));

  // Guess energy and orbitals
  arma::vec E;
  arma::mat C;
  arma::mat Pag, Pbg;
  // Form core Hamiltonian
  arma::mat T(basis.kinetic());
  arma::mat V(basis.nuclear());
  arma::mat Hcore(T+V);

  // Compute guess density matrix
  std::string guess(set.get_string("Guess"));
  if(stricmp(guess,"core")==0) {
    // Diagonalize it
    diag(E,C,Hcore,Sinvh);
    Pag=C.cols(0,Nela-1)*C.cols(0,Nela-1).t();
    Pbg=C.cols(0,Nelb-1)*C.cols(0,Nelb-1).t();

  } else if(stricmp(guess,"sap")==0) {
    DFTGrid grid(&basis);

    // Use a (99,590) grid
    int nrad=99;
    int lmax=41;
    bool grad=false;
    bool tau=false;
    bool lapl=false;
    bool strict=false;
    bool nl=false;
    grid.construct(nrad,lmax,grad,tau,lapl,strict,nl);

    // Get SAP potential
    arma::mat Vsap(grid.eval_SAP());
    // Approximate Hamiltonian is
    diag(E,C,Hcore+Vsap,Sinvh);
    // Guess density is
    Pag=C.cols(0,Nela-1)*C.cols(0,Nela-1).t();
    Pbg=C.cols(0,Nelb-1)*C.cols(0,Nelb-1).t();

  } else if(stricmp(guess,"sad")==0 || stricmp(guess,"no")==0) {
    // Get SAD guess
    Pag=sad_guess(basis,set)/2.0;
    Pbg=Pag;
    
    if(stricmp(guess,"no")==0) {
      // Build Fock operator from SAD density matrix. Get natural orbitals
      arma::vec occ;
      form_NOs(Pag,S,C,occ);
      // Recreate guess
      Pag=C.cols(0,Nela-1)*C.cols(0,Nela-1).t();
      Pbg=C.cols(0,Nelb-1)*C.cols(0,Nelb-1).t();
    } else {
      if((Nela+Nelb)!=basis.Ztot()) {
        printf("Warning: SAD density doesn't integrate to wanted number of electrons.\n");
      }
      if(Nela!=Nelb) {
        printf("Warning: SAD density doesn't correspond to wanted spin state.\n");
      }
    }

  } else if(stricmp(guess,"gsap")==0) {
    // Get SAP guess
    diag(E,C,Hcore+sap_guess(basis,set),Sinvh);
    // Guess density is
    Pag=C.cols(0,Nela-1)*C.cols(0,Nela-1).t();
    Pbg=C.cols(0,Nelb-1)*C.cols(0,Nelb-1).t();
  } else
    throw std::logic_error("Unsupported guess!\n");

  // Calculate the projection
  double aproj(arma::trace(Pa*S*Pag*S));
  double bproj(arma::trace(Pb*S*Pbg*S));
  if(std::abs(aproj-bproj)>=sqrt(DBL_EPSILON)) {
    printf("Alpha projection of guess onto SCF density is %e i.e. %5.2f %%\n",aproj,aproj/Nela*100.0);
    printf("Beta  projection of guess onto SCF density is %e i.e. %5.2f %%\n",bproj,bproj/Nelb*100.0);
  }
  printf("Projection of guess onto SCF density is %e i.e. %5.2f %%\n",aproj+bproj,(aproj+bproj)/(Nela+Nelb)*100.0);

  printf("\nRunning program took %s.\n",t.elapsed().c_str());

  return 0;
}
