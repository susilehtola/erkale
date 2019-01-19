/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "integrals.h"
#include "../linalg.h"
#include "../stringutil.h"
#include "../basis.h"
#include "../guess.h"
#include "../mathf.h"
#include "../scf.h"
#include "../timer.h"
#include "../settings.h"
#include "solvers.h"
#include <armadillo>

// Needed for libint init
#include "../eriworker.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

void testexch(const std::vector<bf_t> & basis) {
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      for(size_t k=0;k<=i;k++)
	for(size_t l=0;l<=k;l++) {
	  double eri=ERI(basis[i],basis[j],basis[k],basis[l]);
	  double geri=gaussian_ERI(basis[i],basis[j],basis[k],basis[l]);
	  if(fabs(eri-geri)<=2e-4)
	    printf("%i %i %i %i   ok, %e %e %e\n",(int) i,(int) j,(int) k,(int) l,eri,geri,eri-geri);
	  else
	    printf("%i %i %i %i fail, %e %e %e\n",(int) i,(int) j,(int) k,(int) l,eri,geri,eri-geri);
	}
    }
}

std::vector<bf_t> construct_basis(const std::string & filename) {
  // Returned array
  std::vector<bf_t> bas;

  // Open input file
  FILE *in=fopen(filename.c_str(),"r");
  if(in==NULL)
    throw std::runtime_error("Could not open basis file.\n");

  // Locate BASIS keyword
  std::string line;
  do {
    line=readline(in);
    std::vector<std::string> words=splitline(line);

    if(words.size() && words[0].size()==5)
      if(strncmp(words[0].c_str(),"BASIS",5)==0)
	break;
  } while(!feof(in));

  if(feof(in))
    throw std::runtime_error("Could not locate BASIS keyword.\n");

  // Read in exponents
  while(!feof(in)) {
    // Read line
    line=readline(in);
    //    printf("line: \"%s\"\n",line.c_str());

    // Check end of basis
    if(strncmp(splitline(line)[0].c_str(),"END",3)==0)
      break;

    // Read in type of function
    char l;
    int n;
    double z;
    sscanf(line.c_str()," %i%c %lf",&n,&l,&z);

    // Convert angular momentum to int
    int am;
    for(am=0;am<(int) (sizeof(shell_types)/sizeof(shell_types[0]));am++)
      if(shell_types[am]==l) {
	break;
      }
    if(am==sizeof(shell_types)/sizeof(shell_types[0])) {
      fprintf(stderr,"Input line was \"%s\".\n",line.c_str());
      throw std::runtime_error("Unable to parse angular momentum.\n");
    }

    // Add functions
    for(int m=-am;m<=am;m++) {
      bf_t f;
      f.n=n;
      f.zeta=z;
      f.l=am;
      f.m=m;
      bas.push_back(f);
    }
  }

  return bas;
}


void test() {
  arma::mat P(1,1);
  P(0,0)=1.0;

  bf_t F1s;
  F1s.n=1; F1s.l=0; F1s.m=0; F1s.zeta=8.7;

  bf_t F2s;
  F2s.n=2; F2s.l=0; F2s.m=0; F2s.zeta=2.6;

  bf_t F2pz;
  F2pz.n=2; F2pz.l=1; F2pz.m=0; F2pz.zeta=2.6;

  bf_t F2px;
  F2px.n=2; F2px.l=1; F2px.m=1; F2px.zeta=2.6;

  bf_t F2py;
  F2py.n=2; F2py.l=1; F2py.m=-1; F2py.zeta=2.6;

  printf("F (2pz 2pz|2pz 2pz) %.12f\n",ERI(F2pz,F2pz,F2pz,F2pz));
  printf("F (1s  2s |2s  1s ) %.12f\n",ERI(F1s,F2s,F2s,F1s));
  printf("F (2pz 2px|2pz 2px) %.12f\n",ERI(F2pz,F2px,F2pz,F2px));
  printf("F (2pz 2py|2pz 2py) %.12f\n",ERI(F2pz,F2py,F2pz,F2py));
  printf("F (2s  2pz|2s  2pz) %.12f\n",ERI(F2s,F2pz,F2s,F2pz));
}

Settings settings;

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Atoms from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Atoms from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  init_libint_base();
  //  test();

  settings.add_int("Z","Nucleus to study",0);
  settings.add_string("Basis","Basis set file in ADF format","");
  settings.add_double("LinDepThresh","Linear dependence threshold",1e-5);
  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    settings.print();
    return 1;
  }

  int Z=settings.get_int("Z");

  // Construct basis set
  std::vector<bf_t> basis=construct_basis(settings.get_string("Basis"));

  // Print out basis set
  printf("Basis set\n");
  for(size_t i=0;i<basis.size();i++)
    printf(" %3i: n=%i, zeta=%-8.5f, l=%i, m=% i\n",(int) i,basis[i].n,basis[i].zeta,basis[i].l,basis[i].m);
  printf("\n");

  double convthr(1e-6);

  if(get_ground_state(Z).mult==1) {
    rscf_t sol;
    RHF(basis,Z,sol,convthr,false);
  } else {
    uscf_t sol;
    UHF(basis,Z,sol,convthr,false,true);
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
