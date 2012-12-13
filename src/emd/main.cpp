/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "emd.h"
#include "emd_gto.h"
#include "emdcube.h"
//include "gto_fourier_ylm.h"
#include "spherical_expansion.h"
#include "../solidharmonics.h"
#include "spherical_harmonics.h"
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
  set.add_bool("DoEMD", "Perform calculation of isotropic EMD (moments of EMD, Compton profile)", true);
  set.add_double("EMDTol", "Tolerance for the numerical integration of the radial EMD",1e-8);

  set.add_string("EMDCube", "Calculate EMD on a cube? e.g. -10:.3:10 -5:.2:4 -2:.1:3", "");
  set.add_string("EMDOrbitals", "Compute EMD of given orbitals, e.g. 1,2,4:6","");

  if(argc==2)
    set.parse(argv[1]);
  else
    printf("Using default settings.\n");

  // Get the tolerance
  double tol=set.get_double("EMDTol");

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Load density matrix
  arma::mat P;
  chkpt.read("P",P);

  // Compute orbital EMDs?
  if(set.get_string("EMDOrbitals")!="") {
    // Get orbitals
    std::vector<std::string> orbs=splitline(set.get_string("EMDOrbitals"));

    // Polarized calculation?
    bool restr;
    chkpt.read("Restricted",restr);
    if(restr!= (orbs.size()==1))
      throw std::runtime_error("Invalid occupancies for spin alpha and beta!\n");

    for(size_t ispin=0;ispin<orbs.size();ispin++) {
      // Indices of orbitals to include.
      std::vector<size_t> idx=parse_range(orbs[ispin]);
      // Change into C++ indexing
      for(size_t i=0;i<idx.size();i++)
	idx[i]--;

      // Read orbitals
      arma::mat C;
      if(restr)
	chkpt.read("C",C);
      else {
	if(ispin==0)
	  chkpt.read("Ca",C);
	else
	  chkpt.read("Cb",C);
      }
	
      for(size_t i=0;i<idx.size();i++) {
	// Names of output files
	char emdname[80];
	char momname[80];
	char Jname[80];
	char Jintname[80];

	if(restr) {
	  sprintf(emdname,"emd-%i.txt",(int) idx[i]+1);
	  sprintf(momname,"moments-%i.txt",(int) idx[i]+1);
	  sprintf(Jname,"compton-%i.txt",(int) idx[i]+1);
	  sprintf(Jintname,"compton-interp-%i.txt",(int) idx[i]+1);
	} else {
	  if(ispin==0) {
	    sprintf(emdname,"emd-a-%i.txt",(int) idx[i]+1);
	    sprintf(momname,"moments-a-%i.txt",(int) idx[i]+1);
	    sprintf(Jname,"compton-a-%i.txt",(int) idx[i]+1);
	    sprintf(Jintname,"compton-interp-a-%i.txt",(int) idx[i]+1);
	  } else {
	    sprintf(emdname,"emd-b-%i.txt",(int) idx[i]+1);
	    sprintf(momname,"moments-b-%i.txt",(int) idx[i]+1);
	    sprintf(Jname,"compton-b-%i.txt",(int) idx[i]+1);
	    sprintf(Jintname,"compton-interp-b-%i.txt",(int) idx[i]+1);
	  }
	}

	// Generate dummy density matrix
	arma::mat Pdum=C.col(idx[i])*arma::trans(C.col(idx[i]));

	Timer temd;
	GaussianEMDEvaluator eval(basis,Pdum);
	EMD emd(&eval, 1);
	emd.initial_fill();
	emd.find_electrons();
	emd.optimize_moments(true,tol);
	emd.save(emdname);
	emd.moments(momname);
	emd.compton_profile(Jname);
	emd.compton_profile_interp(Jintname);	  
      }
    }
  }
  
  if(set.get_bool("DoEMD")) {
    t.print_time();
    
    printf("\nCalculating EMD properties.\n");
    printf("Please read and cite the reference:\n%s\n%s\n%s\n",	\
	   "J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen",		\
	   "Calculation of isotropic Compton profiles with Gaussian basis sets", \
	   "Phys. Chem. Chem. Phys 13 (2011), pp. 5630 - 5641.");

    // Construct EMD evaluator
    Timer temd;
    GaussianEMDEvaluator eval(basis,P);
    //    eval.print();

    // and the EMD class
    int Nel;
    chkpt.read("Nel",Nel);

    temd.set();
    EMD emd(&eval, Nel);
    emd.initial_fill();
    emd.find_electrons();
    emd.optimize_moments(true,tol);
    emd.save("emd.txt");
    emd.moments("moments.txt");
    emd.compton_profile("compton.txt");
    emd.compton_profile_interp("compton-interp.txt");
    
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
