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
#include "casida.h"
#include "elements.h"
#include "find_molecules.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
#include "scf.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"

#include "dftfuncs.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

void print_spectrum(const std::string & fname, const arma::mat & m) {
  FILE *out=fopen(fname.c_str(),"w");
  for(size_t it=0; it<m.n_rows; it++)
    fprintf(out,"%e %e\n",m(it,0)*HARTREEINEV, m(it,1));
  fclose(out);
}

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

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_int("CasidaX","Exchange functional for Casida",1);
  set.add_int("CasidaC","Correlation functional for Casida",7);
  set.add_bool("CasidaPol","Perform polarized Casida calculation (when using restricted wf)",0);
  set.add_int("CasidaCoupling","Coupling mode: 0 for IPA, 1 for RPA and 2 for TDLDA",2);
  set.add_double("CasidaTol","Tolerance for Casida grid",1e-3);
  set.add_string("CasidaStates","States to include in Casida calculation, eg ""1,3-4,10,13"" ","");
  set.add_string("CasidaQval","Values of Q to compute spectrum for","");

  set.parse(std::string(argv[1]));
  
  // Get values of q to compute spectrum for
  std::vector<double> qvals=parse_range_double(set.get_string("CasidaQval"));

  // Settings used for initialization
  Settings initset=set;

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
  // Decontract basis set?
  if(set.get_bool("Decontract"))
    baslib.decontract();

  // Construct basis set
  BasisSet basis=construct_basis(atoms,baslib,set);

  // Number of electrons is
  int Nel=basis.Ztot()-set.get_int("Charge");

  // Do a plain Hartree-Fock calculation?
  bool hf= (stricmp(set.get_string("Method"),"HF")==0);
  bool rohf=(stricmp(set.get_string("Method"),"ROHF")==0);

  // Initialize calculation?
  bool noinit=(stricmp(set.get_string("InitMethod"),"none")==0);
  noinit = noinit || ( stricmp(set.get_string("InitMethod"),"none-none")==0);
  bool init=!noinit;

  // Initialize with Hartree-Fock? (Even though there's not much sense in it)
  bool hfinit= (stricmp(set.get_string("InitMethod"),"HF")==0);
  // Initialize by divide-and-conquer?
  bool dncinit= (stricmp(set.get_string("InitMethod"),"DnC")==0);
  // Initialize with DFT?
  bool dftinit= (!hfinit && !dncinit);

  // Final convergence settings
  convergence_t conv;
  // Make initialization parameters more relaxed
  conv.deltaEmax=set.get_double("DeltaEmax");
  conv.deltaPmax=set.get_double("DeltaPmax");
  conv.deltaPrms=set.get_double("DeltaPrms");  

  // Convergence settings for initialization
  convergence_t init_conv(conv);
  // Make initialization parameters more relaxed
  double initfac=set.get_double("DeltaInit");
  init_conv.deltaEmax*=initfac;
  init_conv.deltaPmax*=initfac;
  init_conv.deltaPrms*=initfac;


  if(hfinit) {
    printf("\nHartree-Fock has been specified for initialization.\n");
    printf("You might want to initialize with a pure DFT functional instead.\n");
    printf("\n");
  }

  // Get exchange and correlation functionals
  dft_t dft;
  if(!hf && !rohf) {
    parse_xc_func(dft.x_func,dft.c_func,set.get_string("Method"));
    dft.gridtol=set.get_double("DFTFinalTol");
  }  

  if(init && (hf||rohf) && dftinit) {
    // Need to add DFT settings to initset
    printf("Adding DFT settings to initset.\n");
    initset.add_dft_settings();
  }

  if(init && !hf && !rohf && hfinit) {
    // Need to remove DFT settings from initset
    printf("Removing DFT settings from initset.\n");
    initset.remove_dft_settings();
  }

  // Check consistency of parameters
  if(!hf && !rohf && exact_exchange(dft.x_func)!=0.0)
    if(set.get_bool("DFTFitting")) {
      printf("A hybrid functional is used, turning off density fitting.\n");
      set.set_bool("DFTFitting",0);
    }
  
  // Get wanted initialization method
  dft_t dft_init;
  if(init && dftinit) {
    parse_xc_func(dft_init.x_func,dft_init.c_func,set.get_string("InitMethod"));
    dft_init.gridtol=initset.get_double("DFTInitialTol");
  } else if(!hf && !rohf && dncinit) {
    dft_init=dft;
    dft_init.gridtol=set.get_double("DFTInitialTol");
  }

  if(init && dftinit && exact_exchange(dft_init.x_func)!=0.0)
    if(initset.get_bool("DFTFitting")) {
      printf("A hybrid functional is used in initialization, turning off density fitting.\n");
      initset.set_bool("DFTFitting",0);
    }

  // Density matrix (for momentum density calculations)
  arma::mat P;

  if(set.get_int("Multiplicity")==1 && Nel%2==0) {
    // Closed shell case
    rscf_t sol;

    // Get orbital occupancies
    std::vector<double> occs=get_restricted_occupancy(set,basis);

    if(init) {
      // Initialize calculation

      if(hfinit) {
	SCF initsolver(basis,initset);

      	// Solve restricted Hartree-Fock
	initsolver.RHF(sol,occs,init_conv);
      }

      if(dncinit) {
	// Initialize with divide-and-conquer algorithm.

	// Initialize C and E
	size_t Nbf=basis.get_Nbf();
	sol.C.zeros(Nbf,Nbf);
	sol.E.zeros(Nbf);

	// Non-verbose solution.
	initset.set_bool("Verbose",0);
	
	// First, find out molecules in input.
	std::vector< std::vector<size_t> > mols;
	mols=find_molecules(atoms);
	if(verbose)
	  printf("Found %i molecules in system. Performing divide-and-conquer.\n",(int) mols.size());

	size_t iorb=0;
	
	// Now, solve the states of the molecules.
	for(size_t imol=0;imol<mols.size();imol++) {
	  // Timer
	  Timer tmol;

	  // Get the atoms in the molecule
	  std::vector<atom_t> molat;
	  for(size_t iat=0;iat<mols[imol].size();iat++)
	    molat.push_back(atoms[mols[imol][iat]]);

	  if(verbose) {
	    printf("Molecule %3i contains the atoms: ",(int) imol+1);
	    for(size_t iat=0;iat<molat.size();iat++)
	      printf("%3i ",(int) molat[iat].num+1);
	    fflush(stdout);
	  }
	  
	  // Construct a basis set for the molecule.  
	  // Libint was already initialized above.
	  BasisSet molbas=construct_basis(molat,baslib,initset,1);

	  // Solve states
	  rscf_t molsol;

	  // Solver
	  SCF molsolver(molbas,initset);

	  // Make occupancies
	  std::vector<double> molocc=get_restricted_occupancy(initset,molbas);

	  if(hf) {
	    // Solve restricted Hartree-Fock
	    molsolver.RHF(molsol,molocc,init_conv);
	  } else {
	    // Solve restricted DFT problem
	    molsolver.RDFT(molsol,molocc,init_conv,dft_init);
	  }

	  // Now we should have the occupied states of the
	  // molecule. However, the basis set was different, so we
	  // need to project the occupied orbitals onto the full basis
	  // set.
	  arma::mat Cfull;
	  arma::vec Efull;
	  basis.projectMOs(molbas,molsol.E,molsol.C,Efull,Cfull);

	  // Now we have the orbitals, and the orbital energies, so we
	  // can just plant them in the initial guess.
	  for(int i=0;i<sum(molocc)/2;i++) {
	    // Orbital coefficients
	    sol.C.col(iorb)=Cfull.col(i);
	    // Orbital energy
	    sol.E(iorb)=Efull(i);
	    // Increment orbital number
	    iorb++;
	  }
	  if(verbose)
	    printf("done (%s)\n",tmol.elapsed().c_str());
	}

	// Sort orbitals and energies
	sort_eigvec(sol.E,sol.C);
      }
      
      if(dftinit) {	
	SCF initsolver(basis,initset);
	
	// Print information about used functionals
	print_info(dft_init.x_func,dft_init.c_func);
	// Solve restricted DFT problem
	initsolver.RDFT(sol,occs,init_conv,dft_init);
      }

      if(verbose) {
	printf("\nInitialization complete.\n");
	t.print_time();
	printf("\n\n\n");
      }
    }

    // Solver
    SCF solver(basis,set);


    if(hf) {
      // Solve restricted Hartree-Fock
      solver.RHF(sol,occs,conv);
    } else {
      // Print information about used functionals
      print_info(dft.x_func,dft.c_func);
      // Solve restricted DFT problem
      if(!init) {
	// Starting density was probably bad. Do an initial
	// calculation first with a low-density grid.
	solver.RDFT(sol,occs,init_conv,dft);
      }
      solver.RDFT(sol,occs,conv,dft);
    }

    // Get density matrix
    P=sol.P;

    // Do population analysis
    population_analysis(basis,P);

    Casida cas;
    if(set.get_bool("CasidaPol"))
      cas=Casida(set,basis,sol.E,sol.E,sol.C,sol.C,sol.P/2.0,sol.P/2.0);
    else
      cas=Casida(set,basis,sol.E,sol.C,sol.P);

    // Dipole transition
    print_spectrum("casida.dat",cas.dipole_transition(basis));
    // Q dependent stuff
    for(size_t iq=0;iq<qvals.size();iq++) {
      // File to save output
      char fname[80];
      sprintf(fname,"casida-%.2f.dat",qvals[iq]);
      
      print_spectrum(fname,cas.transition(basis,qvals[iq]));
    }

  } else {
    uscf_t sol;

    // Get orbital occupancies
    std::vector<double> occa, occb;
    get_unrestricted_occupancy(set,basis,occa,occb);

    if(init) {
      // Initialize calculation

      SCF initsolver(basis,initset);

      if(hfinit) {
	// Solve restricted Hartree-Fock
	initsolver.UHF(sol,occa,occb,init_conv);
      } else {
	// Print information about used functionals
	print_info(dft_init.x_func,dft_init.c_func);
	// Solve restricted DFT problem
	initsolver.UDFT(sol,occa,occb,init_conv,dft_init);
      }
      
      printf("\nInitialization complete.\n\n\n\n");
    }

    // Solver
    SCF solver(basis,set);

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
      // Solve restricted DFT problem
      if(!init) {
	// Starting density was probably bad. Do an initial
	// calculation first with a low-density grid.
	solver.UDFT(sol,occa,occb,init_conv,dft);
      }
      solver.UDFT(sol,occa,occb,conv,dft);
    }

    // Get density matrix
    P=sol.P;

    population_analysis(basis,sol.Pa,sol.Pb);

    Casida cas(set,basis,sol.Ea,sol.Eb,sol.Ca,sol.Cb,sol.Pa,sol.Pb);

    // Dipole transition
    print_spectrum("casida.dat",cas.dipole_transition(basis));
    // Q dependent stuff
    for(size_t iq=0;iq<qvals.size();iq++) {
      // File to save output
      char fname[80];
      sprintf(fname,"casida-%.2f.dat",qvals[iq]);
      
      print_spectrum(fname,cas.transition(basis,qvals[iq]));
    }
  }    
  
  if(verbose) {
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
    t.print_time();
  }
  
  return 0;
}
