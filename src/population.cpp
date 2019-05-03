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
#include "dftgrid.h"
#include "checkpoint.h"
#include "settings.h"
#include "stringutil.h"
#include "properties.h"
#include "bader.h"
#include "timer.h"

// Needed for libint init
#include "eriworker.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

Settings settings;

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - population from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - population from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  // Parse settings
  settings.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  settings.add_bool("Bader", "Run Bader analysis?", false);
  settings.add_bool("Becke", "Run Becke analysis?", false);
  settings.add_bool("Mulliken", "Run Mulliken analysis?", false);
  settings.add_bool("Lowdin", "Run Löwdin analysis?", false);
  settings.add_bool("IAO", "Run Intrinsic Atomic Orbital analysis?", false);
  settings.add_string("IAOBasis", "Minimal basis set for IAO analysis", "MINAO.gbs");
  settings.add_bool("Hirshfeld", "Run Hirshfeld analysis?", false);
  settings.add_string("HirshfeldMethod", "Method to use for Hirshfeld(-I) analysis", "HF");
  settings.add_bool("IterativeHirshfeld", "Run iterative Hirshfeld analysis?", false);
  settings.add_bool("Stockholder", "Run Stockholder analysis?", false);
  settings.add_bool("Voronoi", "Run Voronoi analysis?", false);
  settings.add_double("Tol", "Grid tolerance to use for the charges", 1e-5);
  settings.add_bool("OrbThr", "Compute orbital density thresholds", false);
  settings.add_bool("VirtThr", "Also do the virtual orbitals?", false);
  settings.add_bool("SICThr", "Compute SIC orbital density thresholds", false);
  settings.add_bool("DensThr", "Compute total density thresholds", false);
  settings.add_double("OrbThrVal", "Which density threshold to calculate", 0.85);
  settings.add_double("OrbThrGrid", "Accuracy of orbital density threshold integration grid", 1e-3);

  // Parse settings
  settings.parse(argv[1]);

  // Print settings
  settings.print();

  // Initialize libint
  init_libint_base();

  // Load checkpoint
  Checkpoint chkpt(settings.get_string("LoadChk"),false);

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

  double tol=settings.get_double("Tol");

  if(settings.get_bool("Bader")) {
    if(restr)
      bader_analysis(basis,P,tol);
    else
      bader_analysis(basis,Pa,Pb,tol);
  }

  if(settings.get_bool("Becke")) {
    if(restr)
      becke_analysis(basis,P,tol);
    else
      becke_analysis(basis,Pa,Pb,tol);
  }

  std::string hmet=settings.get_string("HirshfeldMethod");

  if(settings.get_bool("Hirshfeld")) {
    if(restr)
      hirshfeld_analysis(basis,P,hmet,tol);
    else
      hirshfeld_analysis(basis,Pa,Pb,hmet,tol);
  }

  if(settings.get_bool("IterativeHirshfeld")) {
    if(restr)
      iterative_hirshfeld_analysis(basis,P,hmet,tol);
    else
      iterative_hirshfeld_analysis(basis,Pa,Pb,hmet,tol);
  }

  if(settings.get_bool("IAO")) {
    std::string minbas=settings.get_string("IAOBasis");

    if(restr) {
      // Get amount of occupied orbitals
      int Nela;
      chkpt.read("Nel-a",Nela);

      // Get orbital coefficients
      if(chkpt.exist("CW.re")) {
	arma::cx_mat CW;
	chkpt.cread("CW",CW);
	IAO_analysis(basis,CW.cols(0,Nela-1),minbas);
      } else {
	arma::mat C;
	chkpt.read("C",C);
	IAO_analysis(basis,C.cols(0,Nela-1),minbas);
      }

    } else {
      // Get amount of occupied orbitals
      int Nela, Nelb;
      chkpt.read("Nel-a",Nela);
      chkpt.read("Nel-b",Nelb);

      // Get orbital coefficients
      if(chkpt.exist("CWa.re")) {
	arma::cx_mat CWa, CWb;
	chkpt.cread("CWa",CWa);
	chkpt.cread("CWb",CWb);
	IAO_analysis(basis,CWa.cols(0,Nela-1),CWb.cols(0,Nelb-1),minbas);
      } else {
	arma::mat Ca, Cb;
	chkpt.read("Ca",Ca);
	chkpt.read("Cb",Cb);
	IAO_analysis(basis,Ca.cols(0,Nela-1),Cb.cols(0,Nelb-1),minbas);
      }
    }
  }

  if(settings.get_bool("Lowdin")) {
    if(restr)
      lowdin_analysis(basis,P);
    else
      lowdin_analysis(basis,Pa,Pb);
  }

  if(settings.get_bool("Mulliken")) {
    if(restr)
      mulliken_analysis(basis,P);
    else
      mulliken_analysis(basis,Pa,Pb);
  }

  if(settings.get_bool("Stockholder")) {
    if(restr)
      stockholder_analysis(basis,P,tol);
    else
      stockholder_analysis(basis,Pa,Pb,tol);
  }

  if(settings.get_bool("Voronoi")) {
    if(restr)
      voronoi_analysis(basis,P,tol);
    else
      voronoi_analysis(basis,Pa,Pb,tol);
  }

  if(settings.get_bool("OrbThr")) {
    // Calculate orbital density thresholds

    // Integration grid
    DFTGrid intgrid(&basis,true);
    intgrid.construct_becke(settings.get_double("OrbThrGrid"));

    // Threshold is
    double thr=settings.get_double("OrbThrVal");
    bool virt=settings.get_bool("VirtThr");

    if(restr) {
      // Get amount of occupied orbitals
      int Nela;
      chkpt.read("Nel-a",Nela);

      // Get orbital coefficients
      arma::mat C;
      chkpt.read("C",C);

      int Nmo = virt ? C.n_cols : Nela;

      printf("\n%4s %9s %8s\n","orb","thr","t (s)");
      for(int io=0;io<Nmo;io++) {
	Timer t;

	// Orbital density matrix is
	arma::mat Po=C.col(io)*arma::trans(C.col(io));
	double val=sqrt(intgrid.density_threshold(Po,thr));

	// Print out orbital threshold
	printf("%4i %8.3e %8.3f\n", io+1, val, t.get());
	fflush(stdout);
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

      int Nmo = virt ? Ca.n_cols : Nela;

      printf("\n%4s %9s %9s %8s\n","orb","thr-a","thr-b","t (s)");
      for(int io=0;io<Nmo;io++) {
	Timer t;

	// Orbital density matrix is
	arma::mat Po=Ca.col(io)*arma::trans(Ca.col(io));
	double vala=sqrt(intgrid.density_threshold(Po,thr));

	if(virt || (!virt && io<Nelb)) {
	  Po=Cb.col(io)*arma::trans(Cb.col(io));
	  double valb=sqrt(intgrid.density_threshold(Po,thr));

	  // Print out orbital threshold
	  printf("%4i %8.3e %8.3e %8.3f\n", io+1, vala, valb, t.get());
	  fflush(stdout);
	} else {
	  printf("%4i %8.3e %9s %8.3f\n", io+1, vala, "****", t.get());
	  fflush(stdout);
	}
      }

    }
  }

  if(settings.get_bool("SICThr")) {
    // Calculate SIC orbital density thresholds

    // Integration grid
    DFTGrid intgrid(&basis,true);
    intgrid.construct_becke(settings.get_double("OrbThrGrid"));

    // Threshold is
    double thr=settings.get_double("OrbThrVal");

    if(restr) {
      // Get orbital coefficients
      arma::cx_mat CW;
      chkpt.cread("CW",CW);

      printf("\n%4s %9s %8s\n","orb","thr","t (s)");
      for(size_t io=0;io<CW.n_cols;io++) {
	Timer t;

	// Orbital density matrix is
	arma::mat Po=arma::real(CW.col(io)*arma::trans(CW.col(io)));
	double val=sqrt(intgrid.density_threshold(Po,thr));

	// Print out orbital threshold
	printf("%4i %8.3e %8.3f\n", (int) io+1, val, t.get());
	fflush(stdout);
      }

    } else {
      // Get orbital coefficients
      arma::cx_mat CWa, CWb;
      chkpt.cread("CWa",CWa);
      chkpt.cread("CWb",CWb);

      printf("\n%4s %9s %9s %8s\n","orb","thr-a","thr-b","t (s)");
      for(size_t io=0;io<CWa.n_cols;io++) {
	Timer t;

	// Orbital density matrix is
	arma::mat Po=arma::real(CWa.col(io)*arma::trans(CWa.col(io)));
	double vala=sqrt(intgrid.density_threshold(Po,thr));

	if(io<CWb.n_cols) {
	  Po=arma::real(CWb.col(io)*arma::trans(CWb.col(io)));
	  double valb=sqrt(intgrid.density_threshold(Po,thr));

	  // Print out orbital threshold
	  printf("%4i %8.3e %8.3e %8.3f\n", (int) io+1, vala, valb, t.get());
	  fflush(stdout);
	} else {
	  printf("%4i %8.3e %9s %8.3f\n", (int) io+1, vala, "****", t.get());
	  fflush(stdout);
	}
      }
    }
  }

  if(settings.get_bool("DensThr")) {
    // Calculate total density thresholds

    // Integration grid
    DFTGrid intgrid(&basis,true);
    intgrid.construct_becke(settings.get_double("OrbThrGrid"));

    // Threshold is
    double thr=settings.get_double("OrbThrVal");

    Timer t;

    printf("\n%10s %9s %8s\n","density","thr","t (s)");
    if(!restr) {
      double aval=intgrid.density_threshold(Pa,thr*arma::trace(Pa*basis.overlap()));
      printf("%10s %8.3e %8.3f\n", "alpha", aval, t.get());
      fflush(stdout);
      t.set();

      double bval=intgrid.density_threshold(Pb,thr*arma::trace(Pb*basis.overlap()));
      printf("%10s %8.3e %8.3f\n", "beta", bval, t.get());
      fflush(stdout);
      t.set();
    }

    double val=intgrid.density_threshold(P,thr*arma::trace(P*basis.overlap()));
    printf("%10s %8.3e %8.3f\n", "total", val, t.get());
    fflush(stdout);
  }


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
