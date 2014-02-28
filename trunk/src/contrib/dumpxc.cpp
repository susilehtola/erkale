/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "checkpoint.h"
#include "dftgrid.h"
#include "dftfuncs.h"

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - xc data from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - xc data from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();

  if(argc!=1 && argc!=2) {
    printf("Usage: $ %s (runfile)\n",argv[0]);
    return 0;
  }

  Timer t;

  // Parse settings
  Settings set;
  set.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  set.add_string("Method","Functional to dump data for","mgga_x_tpss");
  set.add_double("GridTol","DFT grid tolerance to use",1e-5);
  if(argc==2)
    set.parse(argv[1]);
  else printf("Using default settings.\n\n");

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);
  
  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Load density matrix
  arma::mat P;
  chkpt.read("P",P);

  // Restricted calculation?
  int restr;
  chkpt.read("Restricted",restr);
  
  // Functional id
  int func_id=find_func(set.get_string("Method"));

  // DFT grid, verbose operation
  DFTGrid grid(&basis,true);
  // Tolerance
  double gridtol=set.get_double("GridTol");

  if(restr) {
    grid.construct(P,gridtol,func_id,0);
    grid.print_density_potential(func_id,P);

  } else {
    arma::mat Pa, Pb;
    chkpt.read("Pa",Pa);
    chkpt.read("Pb",Pb);

    grid.construct(Pa,Pb,gridtol,func_id,0);
    grid.print_density_potential(func_id,Pa,Pb);
  }

  return 0;
}
