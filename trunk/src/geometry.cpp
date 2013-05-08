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
  printf("ERKALE - Geometry optimization from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Geometry optimization from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();
  // Initialize libderiv
  init_libderiv_base();

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_scf_settings();
  set.add_string("SaveChk","File to use as checkpoint","erkale.chk");
  set.add_string("LoadChk","File to load old results from","");
  set.add_bool("ForcePol","Force polarized calculation",false);
  set.add_bool("FreezeCore","Freeze the atomic cores?",false);
  set.parse(std::string(argv[1]));

  // Do a plain Hartree-Fock calculation?
  bool hf= (stricmp(set.get_string("Method"),"HF")==0);
  bool rohf=(stricmp(set.get_string("Method"),"ROHF")==0);
  if(rohf) {
    ERROR_INFO();
    throw std::runtime_error("Optimize does not support ROHF calculations yet!\n");
  }

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

  // Read in atoms.
  std::string atomfile=set.get_string("System");
  const std::vector<atom_t> origgeom=load_xyz(atomfile);
  std::vector<atom_t> atoms(origgeom);

  // Read in basis set
  BasisSetLibrary baslib;
  std::string basfile=set.get_string("Basis");
  baslib.load_gaussian94(basfile);
  printf("\n");

  // Save to output
  save_xyz(atoms,"Initial configuration","optimize.xyz",false);

  // Energy of past iteration
  double Eold=0.0;
  // Projected energy change
  double dEproj;

  // Optimization loop
  for(size_t istep=1;istep<=100;istep++) {
    // Construct basis set
    BasisSet basis=construct_basis(atoms,baslib,set);

    // Initialize with old calculation?
    if(istep>1) {
      set.set_string("LoadChk",set.get_string("SaveChk"));
      set.set_bool("Verbose",false);
    }
    
    // Perform the electronic structure calculation
    calculate(basis,set);

    // Solution checkpoint
    Checkpoint solchk(set.get_string("SaveChk"),false);
        
    // Temporary file
    char *tmpnamep=tempnam("./",".chk");
    std::string tmpname(tmpnamep);
    free(tmpnamep);
    
    Checkpoint chkpt(tmpname,true);
    
    // "Solver"
    SCF solver(basis,set,chkpt);

    // Force
    arma::vec f;

    // Restricted run?
    bool restr=true;
    solchk.read("Restricted",restr);
    if(restr) {
      rscf_t sol;
      solchk.read("C",sol.C);
      solchk.read("E",sol.E);
      solchk.read("P",sol.P);
      solchk.read("H",sol.H);

      std::vector<double> occs;
      solchk.read("occs",occs);

      if(hf)
	f=solver.force_RHF(sol,occs,ROUGHTOL);
      else
	throw std::runtime_error("DFT not implemented\n");

    } else {
      uscf_t sol;
      solchk.read("Ca",sol.Ca);
      solchk.read("Cb",sol.Cb);
      solchk.read("Ea",sol.Ea);
      solchk.read("Eb",sol.Eb);
      solchk.read("Pa",sol.Pa);
      solchk.read("Pb",sol.Pb);
      solchk.read("P",sol.P);
      solchk.read("Ha",sol.Ha);
      solchk.read("Hb",sol.Hb);

      std::vector<double> occa, occb;
      solchk.read("occa",occa);
      solchk.read("occb",occb);

      if(hf)
	f=solver.force_UHF(sol,occa,occb,ROUGHTOL);
      else
	throw std::runtime_error("DFT not implemented\n");
    }

    // Force is
    arma::mat force=interpret_force(f);
    force.print("Forces");
    printf("\n\n");

    arma::vec fval(atoms.size());
    for(size_t inuc=0;inuc<atoms.size();inuc++) {
      fval(inuc)=arma::norm(force.row(inuc),2);
    }

    // Current energy is
    energy_t en;
    solchk.read(en);

    // Convergence criteria
    double Fmax=arma::max(fval);
    double Frms=rms_norm(fval);

    fprintf(stderr,"Step %4i: energy %.12f, max force %e, rms force %e.\n",(int) istep, en.E, Fmax, Frms);
    if(istep>1) {
      double dE=en.E-Eold;
      fprintf(stderr,"Projected change of energy %e, actual change %e, difference %e, ratio %e.\n",dEproj,dE,dEproj-dE,dE/dEproj);
    }
    
    // Save to output
    char comment[80];
    sprintf(comment,"Step %i",(int) istep);
    save_xyz(atoms,comment,"optimize.xyz",true);    

    // Check convergence
    if(Frms<1e-5 && Fmax<1.5e-5)
      break;

    // Move nuclei
    const std::vector<atom_t> oldgeom(atoms);
    for(size_t inuc=0;inuc<atoms.size();inuc++) {
      atoms[inuc].x+=f(3*inuc);
      atoms[inuc].y+=f(3*inuc+1);
      atoms[inuc].z+=f(3*inuc+2);
    }

    // Save old energy
    Eold=en.E;
    // and projected energy change
    dEproj=0.0;
    for(size_t inuc=0;inuc<atoms.size();inuc++) {
      dEproj-=f(3*inuc  )*(atoms[inuc].x-oldgeom[inuc].x);
      dEproj-=f(3*inuc+1)*(atoms[inuc].y-oldgeom[inuc].y);
      dEproj-=f(3*inuc+2)*(atoms[inuc].z-oldgeom[inuc].z);
    }
  }   


  if(set.get_bool("Verbose")) {
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
    t.print_time();
  }


  return 0;
}
