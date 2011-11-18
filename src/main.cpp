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
#include "checkpoint.h"
#include "dftfuncs.h"
#include "elements.h"
#include "emd/emd.h"
#include "find_molecules.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
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
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");

  print_license();

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
  set.add_string("SaveChk","File to use as checkpoint","erkale.chk");
  set.add_string("LoadChk","File to load old results from","");
  set.add_bool("ForcePol","Force polarized calculation",0);
  set.parse(std::string(argv[1]));

  // Checkpoint files to load and save
  std::string loadname=set.get_string("LoadChk");
  std::string savename=set.get_string("SaveChk");
  
  // Redirect output?
  std::string logfile=set.get_string("Logfile");
  if(stricmp(logfile,"stdout")!=0) {
    // Redirect stdout to file
    FILE *outstream=freopen(logfile.c_str(),"w",stdout);
    if(outstream==NULL) {
      ERROR_INFO();
      throw std::runtime_error("Unable to redirect output!\n");
    } else
      fprintf(stderr,"\n");
  }

  bool verbose=set.get_bool("Verbose");

  // Print out settings
  if(verbose)
    set.print();

  // Read in atoms.
  std::vector<atom_t> atoms;
  std::string atomfile=set.get_string("System");
  atoms=load_xyz(atomfile);

  // Read in basis set
  BasisSetLibrary baslib;
  std::string basfile=set.get_string("Basis");
  baslib.load_gaussian94(basfile);
  printf("\n");

  // Construct basis set
  BasisSet basis=construct_basis(atoms,baslib,set);

  // Number of electrons is
  int Nel=basis.Ztot()-set.get_int("Charge");

  // Do a plain Hartree-Fock calculation?
  bool hf= (stricmp(set.get_string("Method"),"HF")==0);
  bool rohf=(stricmp(set.get_string("Method"),"ROHF")==0);

  // Final convergence settings
  convergence_t conv;
  conv.deltaEmax=set.get_double("DeltaEmax");
  conv.deltaPmax=set.get_double("DeltaPmax");
  conv.deltaPrms=set.get_double("DeltaPrms");

  // Get exchange and correlation functionals
  dft_t dft;
  dft_t initdft;
  // Initial convergence settings
  convergence_t initconv;

  if(!hf && !rohf) {
    parse_xc_func(dft.x_func,dft.c_func,set.get_string("Method"));
    dft.gridtol=set.get_double("DFTFinalTol");

    initdft=dft;
    initdft.gridtol=set.get_double("DFTInitialTol");

    initconv=conv;
    initconv.deltaEmax*=set.get_double("DFTDelta");
    initconv.deltaPmax*=set.get_double("DFTDelta");
    initconv.deltaPrms*=set.get_double("DFTDelta");
  }

  // Check consistency of parameters
  if(!hf && !rohf && exact_exchange(dft.x_func)!=0.0)
    if(set.get_bool("DFTFitting")) {
      printf("A hybrid functional is used, turning off density fitting.\n");
      set.set_bool("DFTFitting",0);
    }

  // Write checkpoint.
  Checkpoint chkpt(savename,true);
  chkpt.write(basis);

  if(set.get_int("Multiplicity")==1 && Nel%2==0 && !set.get_bool("ForcePol")) {
    // Closed shell case
    rscf_t sol;

    // Load starting guess?
    if(stricmp(loadname,"")!=0) {
      Checkpoint load(loadname,false);

      // Basis set
      BasisSet oldbas;
      load.read(oldbas);
      
      // Restricted calculation?
      bool restr;
      load.read("Restricted",restr);

      if(restr) {
	// Load energies and orbitals
	arma::vec Eold;
	arma::mat Cold;
	load.read("C",Cold);
	load.read("E",Eold);

	// Project to new basis.
	basis.projectMOs(oldbas,Eold,Cold,sol.E,sol.C);
      } else {
	// Load old density matrix
	arma::vec Pold;
	load.read("P",Pold);
	// Find out natural orbitals
	arma::mat Cold;
	arma::mat hlp;
	form_NOs(Pold,oldbas.overlap(),Cold,hlp);

	arma::vec Eold;
	load.read("Ea",Eold);

	// Project natural orbitals to new basis
	basis.projectMOs(oldbas,Eold,Cold,sol.E,sol.C);
      }
    }	

    // Get orbital occupancies
    std::vector<double> occs=get_restricted_occupancy(set,basis);

    // Solver
    SCF solver(basis,set,chkpt);

    if(hf) {
      // Solve restricted Hartree-Fock
      solver.RHF(sol,occs,conv);
    } else {
      // Print information about used functionals
      print_info(dft.x_func,dft.c_func);
      // Solve restricted DFT problem first on a rough grid
      solver.RDFT(sol,occs,initconv,initdft);
      // .. and then on the final grid
      solver.RDFT(sol,occs,conv,dft);
    }

    // Do population analysis
    population_analysis(basis,sol.P);

  } else {
    uscf_t sol;

    // Load starting guess?
    if(stricmp(loadname,"")!=0) {
      Checkpoint load(loadname,false);

      // Basis set
      BasisSet oldbas;
      load.read(oldbas);
      
      // Restricted calculation?
      bool restr;
      load.read("Restricted",restr);

      if(restr) {
	// Load energies and orbitals
	arma::vec Eold;
	arma::mat Cold;
	load.read("C",Cold);
	load.read("E",Eold);

	// Project to new basis.
	basis.projectMOs(oldbas,Eold,Cold,sol.Ea,sol.Ca);
	sol.Eb=sol.Ea;
	sol.Cb=sol.Ca;
      } else {
	// Load energies and orbitals
	arma::vec Eaold, Ebold;
	arma::mat Caold, Cbold;
	load.read("Ca",Caold);
	load.read("Ea",Eaold);
	load.read("Cb",Cbold);
	load.read("Eb",Ebold);

	// Project to new basis.
	basis.projectMOs(oldbas,Eaold,Caold,sol.Ea,sol.Ca);
	basis.projectMOs(oldbas,Ebold,Cbold,sol.Eb,sol.Cb);
      }
    }

    // Get orbital occupancies
    std::vector<double> occa, occb;
    get_unrestricted_occupancy(set,basis,occa,occb);

    // Solver
    SCF solver(basis,set,chkpt);

    if(hf) {
      // Solve restricted Hartree-Fock
      solver.UHF(sol,occa,occb,conv);
    } else if(rohf) {
      // Solve restricted open-shell Hartree-Fock

      // Amount of occupied states
      int Nel_alpha;
      int Nel_beta;
      get_Nel_alpha_beta(basis.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);
      // Solve ROHF
      solver.ROHF(sol,Nel_alpha,Nel_beta,conv);

      // Set occupancies right
      get_unrestricted_occupancy(set,basis,occa,occb);
    } else {
      // Print information about used functionals
      print_info(dft.x_func,dft.c_func);
      // Solve restricted DFT problem first on a rough grid
      solver.UDFT(sol,occa,occb,initconv,initdft);
      // ... and then on the more accurate grid
      solver.UDFT(sol,occa,occb,conv,dft);
    }

    population_analysis(basis,sol.Pa,sol.Pb);
  }

  if(verbose) {
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
    t.print_time();
  }

  return 0;
}
