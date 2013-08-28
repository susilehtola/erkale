/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "global.h"
#include "basis.h"
#include "checkpoint.h"
#include "stringutil.h"
#include "properties.h"
#include "bader.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - population from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - population from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();

  // Parse settings
  Settings set;
  set.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  set.add_bool("Bader", "Run Bader analysis?", false);
  set.add_bool("Becke", "Run Becke analysis?", false);
  set.add_bool("Mulliken", "Run Mulliken analysis?", false);
  set.add_bool("Lowdin", "Run Löwdin analysis?", false);
  set.add_bool("Hirshfeld", "Run Hirshfeld analysis?", false);
  set.add_double("Tol", "Grid tolerance to use for the charges", 1e-5);

  if(argc==2)
    set.parse(argv[1]);
  else
    printf("Using default settings.\n");

  // Print settings
  set.print();

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);
  
  // Restricted calculation?
  bool restr;
  chkpt.read("Restricted",restr);

  arma::mat P, Pa, Pb;
  chkpt.read("P",P);
  if(!restr) {
    chkpt.read("Pa",Pa);
    chkpt.read("Pb",Pb);
  }

  double tol=set.get_double("Tol");

  if(set.get_bool("Bader")) {
    if(restr)
      bader_analysis(basis,P,tol);
    else
      bader_analysis(basis,Pa,Pb,tol);
  }

  if(set.get_bool("Becke")) {
    if(restr)
      becke_analysis(basis,P,tol);
    else
      becke_analysis(basis,Pa,Pb,tol);
  }

  if(set.get_bool("Hirshfeld")) {
    if(restr)
      hirshfeld_analysis(basis,P,tol);
    else
      hirshfeld_analysis(basis,Pa,Pb,tol);
  }

  if(set.get_bool("Lowdin")) {
    if(restr)
      lowdin_analysis(basis,P);
    else
      lowdin_analysis(basis,Pa,Pb);
  }

  if(set.get_bool("Mulliken")) {
    if(restr)
      mulliken_analysis(basis,P);
    else
      mulliken_analysis(basis,Pa,Pb);
  }



  return 0;
}
