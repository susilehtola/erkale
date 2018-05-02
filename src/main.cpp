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
  set.add_string("SaveChk","File to use as checkpoint","erkale.chk");
  set.add_string("LoadChk","File to load old results from","");
  set.add_bool("ForcePol","Force polarized calculation",false);
  set.parse(std::string(argv[1]),true);
  set.print();

  // Redirect output?
  std::string logfile=set.get_string("Logfile");
  if(stricmp(logfile,"stdout")!=0) {
    // Redirect stdout to file
    FILE *outstream=freopen(logfile.c_str(),"w",stdout);
    if(outstream==NULL) {
      ERROR_INFO();
      throw std::runtime_error("Unable to redirect output!\n");
    }

    fprintf(stderr,"\n");
    // Print header to log file too
    print_header();
  }

  // Basis set
  BasisSet basis;
  std::string basfile(set.get_string("Basis"));
  if(stricmp(basfile,"Read")==0) {
    // Get checkpoint file
    std::string chkf(set.get_string("LoadChk"));
    if(!chkf.size())
      throw std::runtime_error("Must specify LoadChk for Basis Read\n");
    if(!file_exists(chkf))
      throw std::runtime_error("Can't find LoadChk!\n");
    Checkpoint chk(chkf,false);
    chk.read(basis);

    printf("Basis set read in from checkpoint.\n\n");

  } else {
    // Read in atoms.
    std::vector<atom_t> atoms;
    std::string atomfile=set.get_string("System");
    if(file_exists(atomfile))
      atoms=load_xyz(atomfile,!set.get_bool("InputBohr"));
    else {
      // Check if a directory has been set
      char * libloc=getenv("ERKALE_SYSDIR");
      if(libloc) {
	std::string filename=std::string(libloc)+"/"+atomfile;
	if(file_exists(filename))
	  atoms=load_xyz(filename,!set.get_bool("InputBohr"));
	else
	  throw std::runtime_error("Unable to open xyz input file!\n");
      } else
	throw std::runtime_error("Unable to open xyz input file!\n");
    }

    // Read in basis set
    BasisSetLibrary baslib;
    baslib.load_basis(basfile);

    // Construct basis set
    construct_basis(basis,atoms,baslib,set);
  }

  // Do the calculation
  calculate(basis,set);

  if(set.get_bool("Verbose")) {
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
    t.print_time();
  }

  return 0;
}
