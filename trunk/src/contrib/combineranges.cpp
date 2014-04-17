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

#include "co-opt.h"

typedef struct {
  double min;
  double max;
  double tol;
  int Nexp;
} corange_t;

int main(int argc, char **argv) {
  if(argc<3) {
    printf("Usage: %s file1 file2 (file3) (file4) (...)\n",argv[0]);
    return 1;
  }

  // Individual profiles
  std::vector< std::vector<corange_t> > cpls(argc-1);

  // Read in data
  for(int i=1;i<argc;i++) {
    FILE *in=fopen(argv[i],"r");
    if(!in) {
      std::ostringstream oss;
      oss << "Error opening input file " << argv[i] << ".\n";
      throw std::runtime_error(oss.str());
    }

    // Allocate memory
    cpls[i-1].resize(max_am+1);

    // Read in maximum am
    int maxam;
    if(fscanf(in,"%i",&maxam)!=1) {
      std::ostringstream oss;
      oss << "Error reading maximum angular momentum from input file " << argv[i] << ".\n";
      throw std::runtime_error(oss.str());
    }

    // Read in ranges
    for(int am=0;am<=maxam;am++) {
      int hlp;
      if(fscanf(in,"%i %lf %lf %lf %i",&hlp,&cpls[i-1][am].min,&cpls[i-1][am].max,&cpls[i-1][am].tol,&cpls[i-1][am].Nexp)!=5) {
	std::ostringstream oss;
	oss << "Error reading limits from input file " << argv[i] << ".\n";
	throw std::runtime_error(oss.str());
      }

      // Check am
      if(hlp!=am) {
	std::ostringstream oss;
	oss << "Error: angular momentum doesn't match in input file " << argv[i] << ".\n";
	throw std::runtime_error(oss.str());
      }
    }

    // Close input
    fclose(in);

    for(size_t am=(size_t) maxam+1;am<cpls[i-1].size();am++) {
      cpls[i-1][am].min=0.0;
      cpls[i-1][am].max=0.0;
      cpls[i-1][am].tol=0.0;
      cpls[i-1][am].Nexp=0;
    }
  }
  
  // Collect data
  std::vector<corange_t> cpl(cpls[0]);
  for(size_t i=1;i<cpls.size();i++)
    for(int am=0;am<=max_am;am++) {

      if(cpls[i][am].min<cpl[am].min)
	cpl[am].min=cpls[i][am].min;

      if(cpls[i][am].max>cpl[am].max)
	cpl[am].max=cpls[i][am].max;

      if(cpls[i][am].tol<cpl[am].tol)
	cpl[am].tol=cpls[i][am].tol;

      if(cpls[i][am].Nexp>cpl[am].Nexp)
	cpl[am].Nexp=cpls[i][am].Nexp;
    }

  // Maximum angular momentum is
  int maxam=0;
  for(int am=0;am<=max_am;am++) {
    if(cpl[am].Nexp)
      maxam=am;
  }
  
  fprintf(stderr,"Input:\n");
  for(int am=0;am<=maxam;am++)
    fprintf(stderr,"%c % .3f % .3f %e %i\n",shell_types[am],cpl[am].min,cpl[am].max,cpl[am].tol,cpl[am].Nexp);

  // Generate profile
  std::vector<coprof_t> prof(maxam+1);

  // Determine proper limits
  fprintf(stderr,"\nOutput:\n");
  for(int am=0;am<=maxam;am++) {
    prof[am].start=cpl[am].min;
    prof[am].tol=cpl[am].tol;

    // Wanted width is
    double width=cpl[am].max-cpl[am].min;

    // Determine amount of exponents necessary to achieve it
    int Nf;
    double w;
    arma::vec exps;
    for(Nf=cpl[am].Nexp;Nf<=NFMAX;Nf++) {
      exps=maxwidth_exps(am,cpl[am].tol,Nf,&w);

      if(w>=width)
	break;
    }

    // Adjust ending point
    prof[am].end=cpl[am].max + (w-width);
    prof[am].exps=move_exps(exps,prof[am].start);
    fprintf(stderr,"%c % .3f % .3f %e %i\n",shell_types[am],prof[am].start,prof[am].end,prof[am].tol,(int) prof[am].exps.size());
    fflush(stderr);
  }


  printf("%i\n",maxam);
  for(int am=0;am<=maxam;am++)
    printf("%c % .16e % .16e %e %i\n",shell_types[am],prof[am].start,prof[am].end,prof[am].tol,(int) prof[am].exps.size());
  
  return 0;
}
