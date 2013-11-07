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

double compute_threshold(DFTGrid & intgrid, const arma::mat & Po, double thr) {
  // Binary search
  double lt=0.1;
  double rt=0.4;

  // Check lt is ok
  double lval;
  while((lval=intgrid.eval_dens_cutoff(Po,lt)) < thr) {
    rt=lt;
    lt/=2.0;
    //    printf("lt decreased, lval=%e\n",lval);
  }

  // Check rt is ok
  double rval;
  while((rval=intgrid.eval_dens_cutoff(Po,rt)) > thr) {
    lt=rt;
    rt*=2.0;
    //    printf("rt increased, rval=%e\n",rval);
  }

  // Binary search algorithm. Relative accuracy of 1e-3
  while(rt-lt > 1e-3*(rt+lt)) {
    // Middle value
    double mt=(rt+lt)/2.0;
    double mval=intgrid.eval_dens_cutoff(Po,mt);

    //    printf("mt = %e, mval = %e, acc = %e\n",mt,mval,(rt-lt)/(rt+mt));

    if(mval>thr)
      lt=mt;
    else if(mval<thr)
      rt=mt;
    else {
      rt=mt;
      lt=mt;
      break;
    }
  }

  // Convert to orbital value
  return sqrt((rt+lt)/2.0);
}


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
  set.add_bool("IAO", "Run Intrinsic Atomic Orbital analysis?", false);
  set.add_string("IAOBasis", "Minimal basis set for IAO analysis", "MINAO.gbs");
  set.add_bool("Hirshfeld", "Run Hirshfeld analysis?", false);
  set.add_bool("Stockholder", "Run Stockholder analysis?", false);
  set.add_bool("Voronoi", "Run Voronoi analysis?", false);
  set.add_double("Tol", "Grid tolerance to use for the charges", 1e-5);
  set.add_bool("OrbThr", "Compute orbital density thresholds", false);
  set.add_double("OrbThrVal", "Which density threshold to calculate", 0.85);
  set.add_double("OrbThrAcc", "Accuracy of orbital density integration grid", 1e-5);

  if(argc==2)
    set.parse(argv[1]);
  else
    printf("Using default settings.\n");

  // Print settings
  set.print();

  // Initialize libint
  init_libint_base();

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

  if(set.get_bool("IAO")) {
    std::string minbas=set.get_string("IAOBasis");

    if(restr) {
      // Get amount of occupied orbitals
      int Nela;
      chkpt.read("Nel-a",Nela);

      // Get orbital coefficients
      arma::mat C;
      chkpt.read("C",C);

      // Do analysis
      IAO_analysis(basis,C.submat(0,0,C.n_rows-1,Nela-1),P,minbas);
    } else {
      // Get amount of occupied orbitals
      int Nela, Nelb;
      chkpt.read("Nel-a",Nela);
      chkpt.read("Nel-b",Nelb);

      // Get orbital coefficients
      arma::mat Ca, Cb;
      chkpt.read("Ca",Ca);
      chkpt.read("Cb",Cb);

      // Do analysis
      IAO_analysis(basis,Ca.submat(0,0,Ca.n_rows-1,Nela-1),Cb.submat(0,0,Cb.n_rows-1,Nelb-1),Pa,Pb,minbas);
    }
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

  if(set.get_bool("Stockholder")) {
    if(restr)
      stockholder_analysis(basis,P,tol);
    else
      stockholder_analysis(basis,Pa,Pb,tol);
  }

  if(set.get_bool("Voronoi")) {
    if(restr)
      voronoi_analysis(basis,P,tol);
    else
      voronoi_analysis(basis,Pa,Pb,tol);
  }

  if(set.get_bool("OrbThr")) {
    // Calculate orbital density thresholds

    // Integration grid
    DFTGrid intgrid(&basis,true);
    intgrid.construct_becke(set.get_double("OrbThrAcc"));

    // Threshold is
    double thr=set.get_double("OrbThrVal");

    if(restr) {
      // Get amount of occupied orbitals
      int Nela;
      chkpt.read("Nel-a",Nela);

      // Get orbital coefficients
      arma::mat C;
      chkpt.read("C",C);

      printf("\n%4s %9s %8s\n","orb","thr","t (s)");
      for(int io=0;io<Nela;io++) {
	Timer t;
	
	// Orbital density matrix is
	arma::mat Po=C.col(io)*arma::trans(C.col(io));
	double val=compute_threshold(intgrid,Po,thr);

	// Print out orbital threshold
	printf("%4i %8.3e %8.3f\n", io+1, val, t.get());
      }

    } else {
      // Get amount of occupied orbitals
      int Nela, Nelb;
      chkpt.read("Nel-a",Nela);
      chkpt.read("Nel-b",Nelb);

      // Get orbital coefficients
      arma::mat Ca, Cb;
      chkpt.read("Ca",Ca);
      chkpt.read("Cb",Cb);

      printf("\n%4s %9s %9s %8s\n","orb","thr-a","thr-b","t (s)");
      for(int io=0;io<Nela;io++) {
	Timer t;

	// Orbital density matrix is
	arma::mat Po=Ca.col(io)*arma::trans(Ca.col(io));
	double vala=compute_threshold(intgrid,Po,thr);

	if(io<Nelb) {
	  Po=Cb.col(io)*arma::trans(Cb.col(io));
	  double valb=compute_threshold(intgrid,Po,thr);
	  
	  // Print out orbital threshold
	  printf("%4i %8.3e %8.3e %8.3f\n", io+1, vala, valb, t.get());

	} else 
	  printf("%4i %8.3e %9s %8.3f\n", io+1, vala, "****", t.get());
      }
      
    }
  }


  return 0;
}
