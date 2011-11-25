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

#include "casida.h"
#include "checkpoint.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"

#include "dftfuncs.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

void print_spectrum(const std::string & fname, const arma::mat & m) {
  FILE *out=fopen(fname.c_str(),"w");
  for(size_t it=0; it<m.n_rows; it++)
    fprintf(out,"%e %e\n",m(it,0)*HARTREEINEV, m(it,1));
  fclose(out);
}

int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - Casida from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Casida from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");

  print_license();

  if(argc!=1 && argc!=2) {
    printf("Usage: $ %s (runfile)\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_string("CasidaX","Exchange functional for Casida","lda_x");
  set.add_string("CasidaC","Correlation functional for Casida","lda_c_vwn");
  set.add_bool("CasidaPol","Perform polarized Casida calculation (when using restricted wf)",false);
  set.add_int("CasidaCoupling","Coupling mode: 0 for IPA, 1 for RPA and 2 for TDLDA",2);
  set.add_double("CasidaTol","Tolerance for Casida grid",1e-3);
  set.add_string("CasidaStates","States to include in Casida calculation, eg ""1,3-4,10,13"" ","");
  set.add_string("CasidaQval","Values of Q to compute spectrum for","");
  set.add_string("LoadChk","Checkpoint to load","erkale.chk");

  if(argc==2)
    set.parse(std::string(argv[1]));
  else
    printf("\nDefault settings used.");

  // Print settings
  set.print();

  // Get functional strings
  int xfunc=find_func(set.get_string("CasidaX"));
  int cfunc=find_func(set.get_string("CasidaC"));
  
  if(is_correlation(xfunc))
    throw std::runtime_error("Refusing to use a correlation functional as exchange.\n");
  if(is_kinetic(xfunc))
    throw std::runtime_error("Refusing to use a kinetic energy functional as exchange.\n");
  if(is_exchange(cfunc))
    throw std::runtime_error("Refusing to use an exchange functional as correlation.\n");
  if(is_kinetic(cfunc))
    throw std::runtime_error("Refusing to use a kinetic energy functional as correlation.\n");
  set.add_int("CasidaXfunc","Internal variable",xfunc);
  set.add_int("CasidaCfunc","Internal variable",cfunc);

  // Print information about used functionals
  print_info(xfunc,cfunc);

  // Get values of q to compute spectrum for
  std::vector<double> qvals=parse_range_double(set.get_string("CasidaQval"));

  // Load checkpoint
  std::string fchk=set.get_string("LoadChk");
  Checkpoint chkpt(fchk,0);

  // Check that calculation was converged
  bool conv;
  chkpt.read("Converged",conv);
  if(!conv)
    throw std::runtime_error("Refusing to run Casida calculation based on a non-converged SCF density!\n");

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  Casida cas;
  bool restr;
  chkpt.read("Restricted",restr);

  if(restr) {
    // Load energy and orbitals
    arma::mat C, P;
    arma::vec E;
    std::vector<double> occs;

    chkpt.read("P",P);
    chkpt.read("C",C);
    chkpt.read("E",E);
    chkpt.read("occs",occs);
    
    if(set.get_bool("CasidaPol")) {
      // Half occupancy (1.0 instead of 2.0)
      std::vector<double> hocc(occs);
      for(size_t i=0;i<hocc.size();i++)
	hocc[i]/=2.0;
      cas=Casida(set,basis,E,E,C,C,P/2.0,P/2.0,hocc,hocc);
    }
    else
      cas=Casida(set,basis,E,C,P,occs);

  } else {
    arma::mat Ca, Cb;
    arma::mat Pa, Pb;
    arma::vec Ea, Eb;
    std::vector<double> occa, occb;

    chkpt.read("Pa",Pa);
    chkpt.read("Pb",Pb);
    chkpt.read("Ca",Ca);
    chkpt.read("Cb",Cb);
    chkpt.read("Ea",Ea);
    chkpt.read("Eb",Eb);
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);

    cas=Casida(set,basis,Ea,Eb,Ca,Cb,Pa,Pb,occa,occb);
  }

  // Dipole transition
  print_spectrum("casida.dat",cas.dipole_transition(basis));
  // Q dependent stuff
  for(size_t iq=0;iq<qvals.size();iq++) {
    // File to save output
    char fname[80];
    sprintf(fname,"casida-%.2f.dat",qvals[iq]);

    print_spectrum(fname,cas.transition(basis,qvals[iq]));
  }

  printf("\nRunning program took %s.\n",t.elapsed().c_str());
  t.print_time();

  return 0;
}
