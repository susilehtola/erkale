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

#include <cstdio>
#include "optimize_completeness.h"
#include "../basislibrary.h"
#include "../global.h"
#include "../timer.h"
#include "../settings.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

Settings settings;

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Completeness optimization from Hel. OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Completeness optimization from Hel. Serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  settings.add_int("am","angular momentum of shell to optimize for",0);
  settings.add_int("n","moment to optimize for: 1 for maximal area, 2 for minimal rms deviation",0);
  settings.add_double("min","lower limit of exponent range in log10",-2);
  settings.add_double("tol","Optimize and add functions until tolerance is achieved",0.0);
  settings.add_int("nfunc","Fixed number of functions to optimize",0);
  settings.add_int("nfull","Number of functions at each side to fully optimize",4);
  settings.add_bool("coulomb","Use Coulomb metric? (Use only for RI basis sets)",false);
  settings.add_double("LinDepThresh","Basis set linear dependence threshold",1e-5);

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    settings.print();
    return 1;
  }
  settings.parse(std::string(argv[1]),true);
  settings.print();

  // Get parameters
  int am=settings.get_int("am");
  int n=settings.get_int("n");
  double min=settings.get_double("min");

  // Did we get a tolerance, or a number of functions?
  double tol=settings.get_double("tol");
  int nfunc=settings.get_int("nfunc");
  int nfull=settings.get_int("nfull");
  bool coulomb=settings.get_bool("coulomb");
  // The Coulomb metric is equivalent to the normal metric with am-1
  if(coulomb)
    am--;

  Timer t;

  // Form optimized set of primitives
  double w;
  arma::vec exps=move_exps(maxwidth_exps(am,tol,nfunc,w,n,nfull),min);

  // Return the original value if Coulomb metric was used
  if(coulomb)
    am++;

  // Create a basis set out of it. Print exponents in descending order
  ElementBasisSet el("El");
  for(size_t i=exps.size()-1;i<exps.size();i--) {
    // Create shell of functions
    FunctionShell tmp(am);
    tmp.add_exponent(1.0,exps[i]);
    // and add it to the basis set
    el.add_function(tmp);
  }

  BasisSetLibrary baslib;
  baslib.add_element(el);
  baslib.save_gaussian94("optimized.gbs");

  printf("Optimization performed in %s.\n",t.elapsed().c_str());
  printf("Completeness-optimized basis set saved to optimized.gbs.\n\n");

  printf("Width of profile is %.10e.\n",w);

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
