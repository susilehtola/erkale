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

#include "checkpoint.h"
#include "dftgrid.h"
#include "stringutil.h"
#include "dftfuncs.h"
#include "timer.h"

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - xc data from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - xc data from Hel, serial version.\n");
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
  set.add_string("Method","Functional to dump data for","mgga_x_tpss");
  set.add_double("GridTol","DFT grid tolerance to use",1e-3);
  set.add_string("DFTGrid","DFT grid to use","Auto");
  if(argc==2)
    set.parse(argv[1]);
  else printf("Using default settings.\n\n");

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);
  
  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Load density matrix
  arma::mat P;
  chkpt.read("P",P);

  // Restricted calculation?
  int restr;
  chkpt.read("Restricted",restr);
  
  // Functional id
  int x_func, c_func;
  parse_xc_func(x_func,c_func,set.get_string("Method"));

  // DFT grid, verbose operation
  DFTGrid grid(&basis,true);

  // Adaptive grid?
  bool adaptive;
  int nrad, lmax;
  double gridtol;

  // Determine grid
  std::string dftgrid=set.get_string("DFTGrid");
  if(stricmp(dftgrid,"Auto")==0) {
    adaptive=true;
    gridtol=set.get_double("GridTol");
  } else {
    adaptive=false;
    std::vector<std::string> gridsize=splitline(dftgrid);
    if(gridsize.size()!=2)
      throw std::runtime_error("Error determining grid size.\n");
    nrad=readint(gridsize[0]);
    lmax=readint(gridsize[1]);
  }

  if(restr) {
    if(adaptive)
      grid.construct(P,gridtol,x_func,c_func);
    else
      grid.construct(nrad,lmax,x_func,c_func);
    
    grid.print_density(P);
    if(x_func)
      grid.print_potential(x_func,P/2.0,P/2.0,"Vx.dat");
    if(c_func)
      grid.print_potential(c_func,P/2.0,P/2.0,"Vc.dat");
    
  } else {
    arma::mat Pa, Pb;
    chkpt.read("Pa",Pa);
    chkpt.read("Pb",Pb);

    if(adaptive)
      grid.construct(Pa,Pb,gridtol,x_func,c_func);
    else
      grid.construct(nrad,lmax,x_func,c_func);

    grid.print_density(Pa+Pb);
    if(x_func)
      grid.print_potential(x_func,Pa,Pb);
    if(c_func)
      grid.print_potential(c_func,Pa,Pb);
  }

  return 0;
}
