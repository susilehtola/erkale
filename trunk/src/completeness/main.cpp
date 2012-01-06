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

#include <cstdio>
#include "optimize_completeness.h"
#include "../basislibrary.h"
#include "../global.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Completeness optimization from Hel. OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Completeness optimization from Hel. Serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2012.\n");

  print_license();

  if(argc!=6) {
    printf("Usage: %s am n min max Nf\n",argv[0]);
    printf("am:  angular momentum of shell to optimize for\n");
    printf("n:   moment to optimize for.\n");
    printf("     1 for maximal area, 2 for minimal rms deviation from unity.\n");
    printf("min: lower limit of exponent range to optimize, in log10\n");
    printf("max: upper limit of exponent range to optimize, in log10\n");
    printf("Nf:  number of functions to place on shell\n");
    return 1;
  }

  // Get parameters
  int am=atoi(argv[1]);
  int n=atoi(argv[2]);
  double min=atof(argv[3]);
  double max=atof(argv[4]);
  int Nf=atoi(argv[5]);

  // Form optimized set of primitives
  std::vector<double> exps;
  exps=optimize_completeness(am,min,max,Nf,n);
  
  // Create a basis set out of it
  ElementBasisSet el("El");
  for(size_t i=0;i<exps.size();i++) {
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
