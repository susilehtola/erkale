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

inline arma::vec focc(const arma::vec & E, double B, double mu) {
  if(!E.size())
    throw std::logic_error("Can't do Fermi occupations without orbital energies!\n");

  arma::vec focc(E);
  for(size_t i=0;i<focc.n_elem;i++)
    focc(i)=1.0/(1.0 + exp(B*(E(i)-mu)));
  return focc;
}

arma::vec FermiON(const arma::vec & E, int N, double T) {
  if(!E.size())
    throw std::logic_error("Can't do Fermi occupations without orbital energies!\n");
  // Temperature factor: 1/(kB T) at T=1000K i.e.
  const double B(315775/T);

  arma::vec occ;
  double occsum;
  double Eleft, Eright;
  {
    int id=N-1;
    while(true) {
      occ=focc(E,B,E(id));
      occsum=arma::sum(occ);
      if(occsum>N)
        id--;
      else
        break;
    }
    Eleft=E(id);

    id=N;
    while(true) {
      occ=focc(E,B,E(id));
      occsum=arma::sum(occ);
      if(occsum<N)
        id++;
      else
        break;
    }
    Eright=E(id);
  }

  size_t it=0;
  do {
    double Efermi((Eleft+Eright)/2);
    occ=focc(E,B,Efermi);
    occsum=arma::sum(occ);

    if(occsum>N) {
      Eright=Efermi;
    } else if(occsum<N) {
      Eleft=Efermi;
    } else
      break;

    it++;

  } while(std::abs(occsum-N)>N*sqrt(DBL_EPSILON));

  // Rescale occupation numbers
  return N*occ/arma::sum(occ);
}

arma::vec pFermiON(const arma::vec & E, int N, double T) {
  if(!E.size())
    throw std::logic_error("Can't do Fermi occupations without orbital energies!\n");

  // Temperature factor: 1/(kB T) at T=1000K i.e.
  const double B(315775/T);

  // Pseudo-FON: chemical potential is
  double mu=0.5*(E(N)+E(N-1));

  // Get occupation numbers
  arma::vec occ=focc(E,B,mu);

  // Rescale occupation numbers
  return N*occ/arma::sum(occ);
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
  set.add_bool("FON","Fermi occupation numbers",false);
  set.add_string("FONscan","Fermi occupation scan","");
  set.add_double("T","Temperature in K",1000);
  set.add_int("nrad","Number of radial shells for SAP",99);
  set.add_int("lmax","Angular rule for SAP (defaults to l=41 i.e. 590 points)",41);
  set.parse(std::string(argv[1]),true);
  set.print();

  // SAP quadrature rule
  int nrad(set.get_int("nrad"));
  int lmax(set.get_int("lmax"));

  // Basis set
  BasisSet basis;
  // Get checkpoint file
  std::string chkf(set.get_string("LoadChk"));
  Checkpoint chk(chkf,false);
  chk.read(basis);

  int restr;
  chk.read("Restricted",restr);

  // Read in density matrix
  arma::mat Pa, Pb;
  if(restr) {
    chk.read("P",Pa);
    Pa/=2.0;
    Pb=Pa;
  } else {
    chk.read("Pa",Pa);
    chk.read("Pb",Pb);
  }

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
  arma::mat Hcore(basis.nuclear()+basis.kinetic());

  // Form density matrix from guess orbitals?
  bool formP=true;

  // Compute guess density matrix
  std::string guess(set.get_string("Guess"));
  bool FON(set.get_bool("FON"));
  std::string FONscan(set.get_string("FONscan"));
  double T(set.get_double("T"));
  if(stricmp(guess,"core")==0) {
    // Diagonalize it
    diag(E,C,Hcore,Sinvh);

  } else if(stricmp(guess,"sap")==0) {
    DFTGrid grid(&basis);
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

  } else if(stricmp(guess,"sad")==0 || stricmp(guess,"no")==0) {
    // Get SAD guess
    Pag=sad_guess(basis,set)/2.0;
    Pbg=Pag;

    if(stricmp(guess,"no")==0) {
      // Build Fock operator from SAD density matrix. Get natural orbitals
      arma::vec occ;
      form_NOs(Pag,S,C,occ);
    } else {
      if((Nela+Nelb)!=basis.Ztot()) {
        printf("Warning: SAD density doesn't integrate to wanted number of electrons.\n");
      }
      if(Nela!=Nelb) {
        printf("Warning: SAD density doesn't correspond to wanted spin state.\n");
      }
      formP=false;
    }

  } else if(stricmp(guess,"gsap")==0) {
    // Get SAP guess
    diag(E,C,Hcore+sap_guess(basis,set),Sinvh);

  } else if(stricmp(guess,"huckel")==0) {
    // Get Huckel guess
    diag(E,C,huckel_guess(basis,set),Sinvh);

  } else {
    throw std::logic_error("Unsupported guess!\n");
  }

  if(formP) {
    arma::vec occa(C.n_cols), occb(C.n_cols);
    if(FON) {
      if(!formP)
        throw std::logic_error("Can't do Fermi occupations without orbitals!\n");
      if(Nela)
        printf("Alpha gap is %e i.e. % 6.3f eV\n",(E(Nela)-E(Nela-1)),27.2114*(E(Nela)-E(Nela-1)));
      if(Nelb && Nela!=Nelb)
        printf("Beta  gap is %e i.e. % 6.3f eV\n",(E(Nelb)-E(Nelb-1)),27.2114*(E(Nelb)-E(Nelb-1)));
      occa=FermiON(E,Nela,T);
      occb=FermiON(E,Nelb,T);

      if(Nela)
        printf("Alpha NOON gap is %e\n",(occa(Nela-1)-occa(Nela)));
      if(Nelb && Nela!=Nelb)
        printf("Beta  NOON gap is %e\n",(occb(Nelb-1)-occb(Nelb)));
    } else {
      occa.zeros();
      occb.zeros();
      occa.subvec(0,Nela-1).ones();
      occb.subvec(0,Nelb-1).ones();
    }
    //E.print("Orbital energies");
    //occa.print("Alpha occupation");
    //occb.print("Beta  occupation");
    Pag=C*arma::diagmat(occa)*C.t();
    Pbg=C*arma::diagmat(occb)*C.t();
  }

  // Calculate the projection
  double aproj(arma::trace(Pa*S*Pag*S));
  double bproj(arma::trace(Pb*S*Pbg*S));
  if(std::abs(aproj-bproj)>=sqrt(DBL_EPSILON)) {
    printf("Alpha projection of guess onto SCF density is %e i.e. %5.2f %%\n",aproj,aproj/Nela*100.0);
    printf("Beta  projection of guess onto SCF density is %e i.e. %5.2f %%\n",bproj,bproj/Nelb*100.0);
  }
  printf("Projection of guess onto SCF density is %e i.e. %5.2f %%\n",aproj+bproj,(aproj+bproj)/(Nela+Nelb)*100.0);

  if(FONscan.size()) {
    if(!formP)
      throw std::logic_error("Can't do Fermi temperature scan without orbitals!\n");

    // Scan FONs
    arma::vec Ts(arma::linspace<arma::vec>(0,20000,101));

    // Data
    arma::mat fon(Ts.n_elem,2);
    fon.col(0)=Ts;
    for(size_t i=0;i<Ts.n_elem;i++) {
      arma::vec occa(FermiON(E,Nela,Ts(i)));
      arma::vec occb(FermiON(E,Nelb,Ts(i)));
      Pag=C*arma::diagmat(occa)*C.t();
      Pbg=C*arma::diagmat(occb)*C.t();
      aproj=arma::trace(Pa*S*Pag*S);
      bproj=arma::trace(Pb*S*Pbg*S);
      fon(i,1)=(aproj+bproj)/(Nela+Nelb)*100.0;
    }
    fon.save(FONscan,arma::raw_ascii);
    printf("FON scan saved in %s\n",FONscan.c_str());
  }

  printf("\nRunning program took %s.\n",t.elapsed().c_str());

  return 0;
}
