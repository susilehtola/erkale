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
#include "emd/emd.h"
#include "dftfuncs.h"
#include "elements.h"
#include "linalg.h"
#include "xyzutils.h"
#include "scf.h"
#include "settings.h"
#include "stringutil.h"
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
  printf("ERKALE - DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - DFT from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");

  print_license();


  Timer t;
  if(argc<3) {
    printf("Usage: %s atoms.xyz basis.gbs (settings) (fsam)\n",argv[0]);
    return 0;
  }

  // Read in atoms.
  std::vector<atom_t> atoms=load_xyz(std::string(argv[1]));
  // Read in basis set
  BasisSetLibrary baslib;
  baslib.load_gaussian94(argv[2]);


  // Parse settings
  Settings set;
  set.add_dft_settings();
  if(argc>=4)
    set.parse(std::string(argv[3]));
  else
    printf("Using default settings.\n");

  double fsam=1.5;
  if(argc>=5)
    fsam=atof(argv[4]);
  if(fsam<=1.0) {
    ERROR_INFO();
    throw std::runtime_error("Invalid value of fsam!\n");
  }

  // Print out settings
  if(set.get_bool("Verbose")) {
    set.print();
  }

  // Get exchange and correlation functionals
  int x_func, c_func;
  parse_xc_func(x_func,c_func,set.get_string("DFT_XC"));

  // Check consistency of parameters
  if(set.get_bool("DensityFitting") && exact_exchange(x_func)!=0.0) {
    printf("A hybrid functional is used, turning off density fitting.\n");
    set.set_bool("DensityFitting",0);
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

  // Number of electrons is
  int Nel=basis.Ztot()-set.get_int("Charge");

  // Get wanted initialization method
  int x_init, c_init;
  parse_xc_func(x_init,c_init,set.get_string("InitMethod"));

  // Density matrix
  arma::mat P;

  if(set.get_int("Multiplicity")==1 && Nel%2==0) {
    // Closed shell case
    arma::mat C;
    arma::vec E;

    if(x_init>0 || c_init>0) {
      // Initialize calculation
      Settings initset=set;
      initset.set_double("DFTInitialTol",1e-3);
      initset.set_double("DFTFinalTol",1e-3);

      initset.set_double("DeltaPrms",1e-6);
      initset.set_double("DeltaPmax",1e-4);
      initset.set_double("DeltaEmax",1e-4);

      printf("\nInitializing calculation with a DFT run.\n");
      print_info(x_init,c_init);
      printf("\n\n");

      SCF initsolver(basis,initset);
      initsolver.RDFT(C,E,x_init,c_init);

      printf("\nInitialization complete.\n\n");
    }

    // Print information about used functionals
    print_info(x_func,c_func);

    SCF solver(basis,set);
    solver.RDFT(C,E,x_func,c_func);

    form_density(P,C,Nel/2);
    P*=2.0;    
  } else {
    arma::mat Ca, Cb;
    arma::vec Ea, Eb;

    if(x_init>0 || c_init>0) {
      // Initialize calculation
      Settings initset=set;
      initset.set_double("DFTInitialTol",1e-3);
      initset.set_double("DFTFinalTol",1e-3);

      initset.set_double("DeltaPrms",1e-6);
      initset.set_double("DeltaPmax",1e-4);
      initset.set_double("DeltaEmax",1e-4);

      printf("\nInitializing calculation with a DFT run.\n");
      print_info(x_init,c_init);
      printf("\n\n");

      SCF initsolver(basis,initset);
      initsolver.UDFT(Ca,Cb,Ea,Eb,x_init,c_init);

      printf("\nInitialization complete.\n\n");
    }

    // Print information about used functionals
    print_info(x_func,c_func);
    
    SCF solver(basis,set);
    solver.UDFT(Ca,Cb,Ea,Eb,x_func,c_func);

    arma::mat Pa, Pb;
    form_density(Pa,Ca,solver.get_Nel_alpha());
    form_density(Pb,Cb,solver.get_Nel_beta());
    P=Pa+Pb;
  }    


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
