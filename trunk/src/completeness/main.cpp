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

#ifdef _OPENMP
#include <omp.h>
#endif

/// Maximum allowed number of functions
#define NFMAX 40

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Completeness optimization from Hel. OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Completeness optimization from Hel. Serial version.\n");
#endif
  print_copyright();
  print_license();

  if(argc!=6 && argc!=7) {
    printf("Usage:   %s am n min max Nf/tol (coulomb)\n",argv[0]);
    printf("am:      angular momentum of shell to optimize for\n");
    printf("n:       moment to optimize for.\n");
    printf("         1 for maximal area, 2 for minimal rms deviation from unity.\n");
    printf("min:     lower limit of exponent range to optimize, in log10\n");
    printf("max:     upper limit of exponent range to optimize, in log10\n");
    printf("Nf/tol:  number of functions to place on shell, or wanted tolerance.\n");
    printf("coulomb: use Coulomb metric?\n");
    return 1;
  }

  // Get parameters
  int am=atoi(argv[1]);
  int n=atoi(argv[2]);
  double min=atof(argv[3]);
  double max=atof(argv[4]);

  // Form optimized set of primitives
  std::vector<double> exps;

  // Did we get a tolerance, or a number of functions?
  double tol=atof(argv[5]);
  bool coulomb=false;
  if(argc==7)
    coulomb=atoi(argv[6]);

  // The Coulomb metric is equivalent to the normal metric with am-1
  if(coulomb)
    am--;

  double tau;
  if(tol<1) {
    exps=get_exponents(am,min,max,tol,n,true);
  } else {
    // Number of functions given.
    exps=optimize_completeness(am,min,max,atoi(argv[5]),n,true,&tau);
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
  baslib.save_gaussian94("optimized.gbs");


  return 0;
}