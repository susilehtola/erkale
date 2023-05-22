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
  settings.add_int("n","moment to optimize for: 1 for maximal area, 2 for minimal rms deviation",1);
  settings.add_double("min","lower limit of exponent range in log10",-2,true);
  settings.add_double("max","upper limit of exponent range in log10",6,true);
  settings.add_double("tol","Optimize and add functions until tolerance is achieved",0.0);
  settings.add_int("nfunc","Fixed number of functions to optimize",0);
  settings.add_int("nfull","Number of functions at each side to fully optimize",4);
  settings.add_bool("coulomb","Use Coulomb metric? (Use only for RI basis sets)",false);
  settings.add_double("LinDepThresh","Basis set linear dependence threshold",1e-5);
  settings.add_string("Output","Output file to use","optimized.gbs");

  if(argc!=2) {
    settings.print();
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }
  settings.parse(std::string(argv[1]),true);
  settings.print();

  // Get parameters
  int am=settings.get_int("am");
  int n=settings.get_int("n");
  double min=settings.get_double("min");
  double max=settings.get_double("max");

  // Form optimized set of primitives
  arma::vec exps;

  // Did we get a tolerance, or a number of functions?
  double tol=settings.get_double("tol");
  int nfunc=settings.get_int("nfunc");
  int nfull=settings.get_int("nfull");
  bool coulomb=settings.get_bool("coulomb");
  // The Coulomb metric is equivalent to the normal metric with am-1
  if(coulomb)
    am--;

  double tau;

  if(tol != 0.0 && nfunc != 0)
    throw std::logic_error("Can't specify both wanted tolerance and number of functions!\n");
  if(tol == 0.0 && nfunc == 0)
    throw std::logic_error("Neither wanted tolerance or number of functions was given!\n");

  if(tol!=0.0) {
    exps=get_exponents(am,min,max,tol,n,true,nfull);
  } else {
    // Number of functions given.
    exps=optimize_completeness(am,min,max,nfunc,n,true,&tau,nfull);
  }

  // Sort into ascending order
  std::sort(exps.begin(),exps.end());

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
  baslib.save_gaussian94(settings.get_string("Output"));

  printf("\nCompleteness-optimized basis set saved to %s.\n",settings.get_string("Output").c_str());

  return 0;
}

int main(int argc, char **argv) {
#ifdef CATCH_EXCEPTIONS
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
#else
  return main_guarded(argc, argv);
#endif
}
