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



#include "basislibrary.h"
#include "basis.h"
#include "elements.h"
#include "emd/emd.h"
#include "linalg.h"
#include "xyzutils.h"
#include "scf.h"
#include "settings.h"
#include "timer.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - HF from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");

  print_license();


  Timer t;
  if(argc<3) {
    printf("Usage: %s atoms.xyz basis.gbs (settings)\n",argv[0]);
    return 0;
  }

  // Read in atoms.
  std::vector<atom_t> atoms=load_xyz(std::string(argv[1]));
  // Read in basis set
  BasisSetLibrary baslib;
  baslib.load_gaussian94(argv[2]);
  // Parse settings
  Settings set(0); // HF settings
  if(argc>=4)
    set.parse(std::string(argv[3]));
  else
    printf("Using default settings.\n");

  // Print out settings
  if(set.get_bool("Verbose")) {
    set.print();
  }

  // Number of atoms is
  size_t Nat=atoms.size();

  // Create basis set
  BasisSet basis(Nat,set);
  // and add atoms to basis set
  for(size_t i=0;i<Nat;i++) {

    // Get center
    coords_t cen;
    cen.x=atoms[i].x;
    cen.y=atoms[i].y;
    cen.z=atoms[i].z;

    // Determine if nucleus is BSSE or not
    bool bsse=0;
    std::string el=atoms[i].el;

    if(el.size()>3 && el.substr(el.size()-3,3)=="-Bq") {
      // Yes, this is a BSSE nucleus
      bsse=1;
      el=el.substr(0,el.size()-3);
    }

    // Add basis functions
    basis.add_functions(i,cen,baslib.get_element(el));
    // and the nucleus
    basis.add_nucleus(i,cen,get_Z(el),el,bsse);
  }

  // Finalize basis set
  basis.finalize();

  // Solve problem. 
  // Number of electrons is

  int Nel=basis.Ztot()-set.get_int("Charge");

  // Solver
  SCF solver(basis,set);

  // Density matrix
  arma::mat P;

  if(set.get_int("Multiplicity")==1 && Nel%2==0) {
    // Closed shell case
    arma::mat C;
    arma::vec E;
    solver.RHF(C,E);

    // Form density matrix
    form_density(P,C,Nel/2);
    // All states are occupied by two electrons
    P*=2.0;
  } else {
    arma::mat Ca, Cb;
    arma::vec Ea, Eb;
    solver.UHF(Ca,Cb,Ea,Eb);

    // Form density matrix
    arma::mat Pa, Pb;
    form_density(Pa,Ca,solver.get_Nel_alpha());
    form_density(Pb,Cb,solver.get_Nel_beta());
    P=Pa+Pb;
  }    

  // Form momentum density
  if(set.get_bool("DoEMD")) {
    printf("\nCalculating EMD properties.\n");
    printf("Please read and cite the reference:\n%s\n%s\n%s\n",\
   "J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen",\
   "Calculation of isotropic Compton profiles with Gaussian basis sets",\
   "Phys. Chem. Chem. Phys 13 (2011), pp. 5630 - 5641.");

    Timer temd;
    EMD emd(basis,P);
    emd.initial_fill();
    emd.find_electrons();
    emd.optimize_moments();
    emd.save("emd.txt");
    emd.moments("moments.txt");
    emd.compton_profile("compton.txt","compton-interp.txt");
    
    if(set.get_bool("Verbose"))
      printf("Calculating EMD properties took %s.\n",temd.elapsed().c_str());
  }  

  if(set.get_bool("Verbose"))
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
  
  return 0;
}
