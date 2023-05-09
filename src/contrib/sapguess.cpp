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
#include "scf.h"
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

  // E.print("Eigenvalues");
}

double model_energy(const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib, const BasisSetLibrary & potlib) {
  // Construct basis set
  BasisSet basis;
  construct_basis(basis,atoms,baslib);

  // Construct model Hamiltonian
  arma::mat H(basis.kinetic()+basis.nuclear()+basis.sap_potential(potlib));
  arma::mat S(basis.overlap());
  arma::mat Sinvh(BasOrth(S,false));

  // Diagonalize it
  arma::vec E;
  arma::mat C;
  diag(E,C,H,Sinvh);

  // Compute number of electrons
  int Ztot=basis.Ztot();

  // Dummy number of electrons
  int nelb=Ztot/2;
  int nela=Ztot-nelb;
  //printf("Ztot = %i nela = %i nelb = %i\n",Ztot,nela,nelb);

  // Orbital weights
  arma::vec oweight(E.n_elem,arma::fill::zeros);
  oweight.subvec(0,nela-1)+=arma::ones<arma::vec>(nela);
  if(nelb)
    oweight.subvec(0,nelb-1)+=arma::ones<arma::vec>(nelb);
  //oweight.print("Orbital weights");

  // Return sum
  return arma::dot(oweight,E);
}

Settings settings;

int main_guarded(int argc, char **argv) {
  print_header();

  if(argc!=4) {
    printf("Usage: $ %s basis guessbasis geometry.xyz\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  settings.add_string("Decontract","","");
  settings.add_bool("BasisRotate","",false);
  settings.add_double("BasisCutoff","",0.0);
  settings.add_bool("UseLM","",true);
  settings.add_bool("OptLM","",false);
  settings.add_double("DFTBasisThr","",1e-10);
  settings.add_double("IntegralThresh","",1e-10);
  settings.add_double("LinDepThresh","",1e-5);
  settings.add_double("CholDepThresh","",1e-6);
  settings.add_bool("Verbose","",false);

  std::string basfile(argv[1]);
  std::string potfile(argv[2]);
  std::string geometry(argv[3]);

  // Read in basis set
  BasisSetLibrary baslib;
  baslib.load_basis(basfile);

  // Read in potential
  BasisSetLibrary potlib;
  potlib.load_basis(potfile);

  // Load atoms
  std::vector<atom_t> atoms=load_xyz(geometry,false);

  // Compute molecular energy
  double Emol = model_energy(atoms, baslib, potlib);

  printf("\nSum of occupied SAP orbital energies is %.9f\n",Emol);

  arma::vec E(1);
  E(0)=Emol;
  E.save("Emol.dat",arma::raw_ascii);

  return 0;
}

int main(int argc, char **argv) {
#ifdef CATCH_EXCEPTIONS
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
#else
  return main_guarded(argc, argv);
#endif
}
