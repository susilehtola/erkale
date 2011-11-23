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

#include "emd.h"
#include "checkpoint.h"
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
  printf("ERKALE - EMD from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - EMD from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");
  print_license();

  if(argc!=1 && argc!=2) {
    printf("Usage: $ %s (runfile)\n",argv[0]);
    return 0;
  }

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  set.add_bool("DoEMD", "Perform calculation of isotropic EMD (moments of EMD, Compton profile)", true);
  set.add_string("EMDCube", "Calculate EMD on a cube? e.g. -10:.3:10 -5:.2:4 -2:.1:3", "");

  if(argc==2)
    set.parse(argv[1]);
  else
    printf("Using default settings.\n");

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Load density matrix
  arma::mat P;
  chkpt.read("P",P);
  
  if(set.get_bool("DoEMD")) {
    t.print_time();
    
    printf("\nCalculating EMD properties.\n");
    printf("Please read and cite the reference:\n%s\n%s\n%s\n",	\
	   "J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen",		\
	   "Calculation of isotropic Compton profiles with Gaussian basis sets", \
	   "Phys. Chem. Chem. Phys 13 (2011), pp. 5630 - 5641.");
    
    Timer temd;
    EMD emd(basis,P);
    emd.initial_fill();
    emd.find_electrons();
    emd.optimize_moments();
    emd.save("emd.txt");
    emd.moments("moments.txt");
    emd.compton_profile("compton.txt","compton-interp.txt");
    
    printf("Calculating isotropic EMD properties took %s.\n",temd.elapsed().c_str());
  }
  // Do EMD on a cube?
  if(stricmp(set.get_string("EMDCube"),"")!=0) {
    t.print_time();
    Timer temd;
    
    // Form grid in p space.
    std::vector<double> px, py, pz;
    parse_cube(set.get_string("EMDCube"),px,py,pz);
    
    // Calculate EMD on cube
    emd_cube(basis,P,px,py,pz);
    
    printf("Calculating EMD on a cube took %s.\n",temd.elapsed().c_str());
  }

  return 0;
}
