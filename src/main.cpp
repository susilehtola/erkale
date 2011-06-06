/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
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
#include "stringutil.h"
#include "timer.h"

#ifdef DFT_ENABLED
#include "dftfuncs.h"
#endif

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char **argv) {

#ifdef DFT_ENABLED
#ifdef _OPENMP
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
#else
#ifdef _OPENMP
  printf("ERKALE - HF from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF from Hel, serial version.\n");
#endif
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");

  print_license();

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 0;
  }

  Timer t;

  // Parse settings
  Settings set;
  set.parse(std::string(argv[1]));
  // Print out settings
  if(set.get_bool("Verbose")) {
    set.print();
  }

  // Read in atoms.
  std::vector<atom_t> atoms;
  std::string atomfile=set.get_string("System");
  atoms=load_xyz(atomfile);

  // Read in basis set
  BasisSetLibrary baslib;
  std::string basfile=set.get_string("Basis");
  baslib.load_gaussian94(basfile);

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

  // Do a plain Hartree-Fock calculation?
  bool hf= (stricmp(set.get_string("Method"),"HF")==0);

  // Initialize calculation?
  bool noinit=(stricmp(set.get_string("InitMethod"),"none")==0);
  noinit = noinit || ( stricmp(set.get_string("InitMethod"),"none-none")==0);
  bool init=!noinit;

  // Initialize with Hartree-Fock? (Even though there's not much sense in it)
  bool hfinit= (stricmp(set.get_string("InitMethod"),"HF")==0);

  // Settings used for initialization
  Settings initset=set;
  // Make initialization parameters more relaxed
  if(init) {
    double dPr=100.0*set.get_double("DeltaPrms");
    double dPm=100.0*set.get_double("DeltaPmax");
    double dEm=100.0*set.get_double("DeltaEmax");

    initset.set_double("DeltaPrms",dPr);
    initset.set_double("DeltaPmax",dPm);
    initset.set_double("DeltaEmax",dEm);
  }

  if(hfinit) {
    printf("\nHartree-Fock has been specified for initialization.\n");
#ifdef DFT_ENABLED
    printf("You might want to initialize with a pure DFT functional instead.\n");
#endif
    printf("\n");
  }

#ifdef DFT_ENABLED
  // Get exchange and correlation functionals
  int x_func=0, c_func=0;
  if(!hf)
    parse_xc_func(x_func,c_func,set.get_string("Method"));

  // Get wanted initialization method
  int x_init=0, c_init=0;
  if(init && !hfinit)
    parse_xc_func(x_init,c_init,set.get_string("InitMethod"));

  if(hf && !hfinit) {
    // Need to add DFT settings to initset
    printf("Adding DFT settings to initset.\n");
    initset.add_dft_settings();
  }

  if(!hf && hfinit) {
    // Need to remove DFT settings from initset
    printf("Removing DFT settings from initset.\n");
    initset.remove_dft_settings();
  }

  // Check consistency of parameters
  if(!hf && exact_exchange(x_func)!=0.0)
    if(set.get_bool("DFTFitting")) {
      printf("A hybrid functional is used, turning off density fitting.\n");
      set.set_bool("DFTFitting",0);
    }
  
  if(init && !hfinit && exact_exchange(x_init)!=0.0)
    if(initset.get_bool("DFTFitting")) {
      printf("A hybrid functional is used in initialization, turning off density fitting.\n");
      initset.set_bool("DFTFitting",0);
    }

  // Make initialization parameters more relaxed
  if(init && !hfinit) {
    initset.set_double("DFTInitialTol",1e-3);
    initset.set_double("DFTFinalTol",1e-3);
  }
#endif
  
  // Density matrix (for momentum density calculations)
  arma::mat P;
  
  if(set.get_int("Multiplicity")==1 && Nel%2==0) {
    // Closed shell case
    arma::mat C;
    arma::vec E;

    if(init) {
      // Initialize calculation

      SCF initsolver(basis,initset);

      if(hfinit) {
      	// Solve restricted Hartree-Fock
	initsolver.RHF(C,E);
      } else {
#ifdef DFT_ENABLED
	// Print information about used functionals
	print_info(x_init,c_init);
	// Solve restricted DFT problem
	initsolver.RDFT(C,E,x_init,c_init);
#else
	throw std::runtime_error("DFT support was not compiled in this version of ERKALE.\n");
#endif
      }

      printf("\nInitialization complete.\n\n\n\n");
    }

    // Solver
    SCF solver(basis,set);

    if(hf) {
      // Solve restricted Hartree-Fock
      solver.RHF(C,E);
    } else {
#ifdef DFT_ENABLED
      // Print information about used functionals
      print_info(x_func,c_func);
      // Solve restricted DFT problem
      solver.RDFT(C,E,x_func,c_func);
#else
      throw std::runtime_error("DFT support was not compiled in this version of ERKALE.\n");
#endif
    }


    // Form density matrix
    form_density(P,C,Nel/2);
    // All states are occupied by two electrons
    P*=2.0;
  } else {
    arma::mat Ca, Cb;
    arma::vec Ea, Eb;

    if(init) {
      // Initialize calculation

      SCF initsolver(basis,initset);

      if(hfinit) {
	// Solve restricted Hartree-Fock
	initsolver.UHF(Ca,Cb,Ea,Eb);
      } else {
#ifdef DFT_ENABLED
	// Print information about used functionals
	print_info(x_init,c_init);
	// Solve restricted DFT problem
	initsolver.UDFT(Ca,Cb,Ea,Eb,x_init,c_init);
#else
	throw std::runtime_error("DFT support was not compiled in this version of ERKALE.\n");
#endif
      }
      
      printf("\nInitialization complete.\n\n\n\n");
    }

    // Solver
    SCF solver(basis,set);

    if(hf) {
      // Solve restricted Hartree-Fock
      solver.UHF(Ca,Cb,Ea,Eb);
    } else {
#ifdef DFT_ENABLED
      // Print information about used functionals
      print_info(x_func,c_func);
      // Solve restricted DFT problem
      solver.UDFT(Ca,Cb,Ea,Eb,x_func,c_func);
#else
      throw std::runtime_error("DFT support was not compiled in this version of ERKALE.\n");
#endif
    }

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
