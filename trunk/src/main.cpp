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
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
  printf("(c) Susi Lehtola, 2010-2012.\n");

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
  set.add_bool("ForcePol","Force polarized calculation",false);
  set.add_bool("FreezeCore","Freeze the atomic cores?",false);
  set.parse(std::string(argv[1]));

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

  // Do the calculation
  calculate(basis,set);

  if(set.get_bool("Verbose")) {
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
    t.print_time();
  }


  return 0;
}
