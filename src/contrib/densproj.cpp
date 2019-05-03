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

Settings settings;

int main_guarded(int argc, char **argv) {
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
  settings.add_string("LoadChk","File to load old results from","");
  settings.add_string("LoadChk2","File to load old results from","");
  settings.parse(std::string(argv[1]),true);
  settings.print();

  // Get checkpoint files
  std::string chk1f(settings.get_string("LoadChk"));
  Checkpoint chk1(chk1f,false);

  std::string chk2f(settings.get_string("LoadChk2"));
  Checkpoint chk2(chk2f,false);

  BasisSet basis1, basis2;
  chk1.read(basis1);
  chk2.read(basis2);

  int restr1, restr2;
  chk1.read("Restricted",restr1);
  chk2.read("Restricted",restr2);
  if(restr1 != restr2)
    throw std::logic_error("One is restricted, other is not!\n");

  int Nela1, Nela2;
  chk1.read("Nel-a",Nela1);
  chk2.read("Nel-a",Nela2);
  if(Nela1 != Nela2)
    throw std::logic_error("Number of alpha electrons doesn't match!\n");

  int Nelb1, Nelb2;
  chk1.read("Nel-b",Nelb1);
  chk2.read("Nel-b",Nelb2);
  if(Nelb1 != Nelb2)
    throw std::logic_error("Number of beta electrons doesn't match!\n");
  
  // Read in density matrix
  arma::mat Pa1, Pb1, Pa2, Pb2;
  if(restr1) {
    chk1.read("P",Pa1);
    chk2.read("P",Pa2);
    Pa1/=2.0;
    Pa2/=2.0;
    Pb1=Pa1;
    Pb2=Pa2;
  } else {
    chk1.read("Pa",Pa1);
    chk1.read("Pb",Pb1);
    chk2.read("Pa",Pa2);
    chk2.read("Pb",Pb2);
  }

  // Overlap matrix
  arma::mat S12(basis1.overlap(basis2));

  // Calculate the projection
  double aproj(arma::trace(Pa1*S12*Pa2*S12.t()));
  double bproj(arma::trace(Pb1*S12*Pb2*S12.t()));
  if(std::abs(aproj-bproj)>=sqrt(DBL_EPSILON)) {
    printf("Projection of alpha densities is %e i.e. %5.2f %%\n",aproj,aproj/Nela1*100.0);
    printf("Projection of beta  densities is %e i.e. %5.2f %%\n",bproj,bproj/Nelb1*100.0);
  }
  printf("Projection of density is %e i.e. %5.2f %%\n",aproj+bproj,(aproj+bproj)/(Nela1+Nelb1)*100.0);

  printf("\nRunning program took %s.\n",t.elapsed().c_str());

  return 0;
}

int main(int argc, char **argv) {
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
