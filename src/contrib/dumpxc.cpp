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

#include "../checkpoint.h"
#include "../dftgrid.h"
#include "../stringutil.h"
#include "../dftfuncs.h"
#include "../timer.h"
#include "../settings.h"

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

Settings settings;

int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - xc data from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - xc data from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc!=1 && argc!=2) {
    printf("Usage: $ %s (runfile)\n",argv[0]);
    return 0;
  }

  Timer t;

  // Parse settings
  settings.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  settings.add_string("Method","Functional to dump data for","mgga_x_br89");
  settings.add_double("GridTol","DFT grid tolerance to use",1e-3);
  settings.add_string("DFTGrid","DFT grid to use","Auto");
  settings.add_string("DFTXpars","exhange functional parameters","");
  settings.add_string("DFTCpars","correlation fucntional parameters","");
  if(argc==2)
    settings.parse(argv[1]);
  else printf("Using default settings.\n\n");
  settings.print();

  // Load checkpoint
  Checkpoint chkpt(settings.get_string("LoadChk"),false);

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
  dft_t dft;
  parse_xc_func(dft.x_func,dft.c_func,settings.get_string("Method"));

  // DFT grid, verbose operation
  DFTGrid grid(&basis,true);

  // Determine grid
  std::string dftgrid(settings.get_string("DFTGrid"));
  std::string tolkw("GridTol");
  if(stricmp(dftgrid,"Auto")!=0) {
    parse_grid(dft, settings.get_string("DFTGrid"), "DFT");
  } else {
    dft.adaptive=true;
    dft.gridtol=settings.get_double(tolkw);
  }

  if(restr) {
    if(dft.adaptive) {
      // The adaptive routine does everything automatically
      grid.construct(P,dft.gridtol,dft.x_func,dft.c_func);
    } else{
      grid.construct(dft.nrad,dft.lmax,dft.x_func,dft.c_func);
      // Need to compute the xc functional to make sure all fields are populated
      arma::mat F;
      double Exc, Nel;
      grid.eval_Fxc(dft.x_func,dft.c_func,P,F,Exc,Nel);
    }

    grid.print_density(P);
    if(dft.x_func) {
      grid.print_potential(dft.x_func,P/2.0,P/2.0,"Vx.dat");
      grid.check_potential(dft.x_func,P/2.0,P/2.0,"Vx_nan.dat");
    }
    if(dft.c_func) {
      grid.print_potential(dft.c_func,P/2.0,P/2.0,"Vc.dat");
      grid.check_potential(dft.c_func,P/2.0,P/2.0,"Vc_nan.dat");
    }

  } else {
    arma::mat Pa, Pb;
    chkpt.read("Pa",Pa);
    chkpt.read("Pb",Pb);

    if(dft.adaptive) {
      grid.construct(Pa,Pb,dft.gridtol,dft.x_func,dft.c_func);
    } else {
      grid.construct(dft.nrad,dft.lmax,dft.x_func,dft.c_func);
      // Need to compute the xc functional to make sure all fields are populated
      arma::mat Fa, Fb;
      double Exc, Nel;
      grid.eval_Fxc(dft.x_func,dft.c_func,Pa,Pb,Fa,Fb,Exc,Nel);
    }

    grid.print_density(Pa,Pb);
    if(dft.x_func) {
      grid.print_potential(dft.x_func,Pa,Pb,"Vx.dat");
      grid.check_potential(dft.x_func,Pa,Pb,"Vx_nan.dat");
    }
    if(dft.c_func) {
      grid.print_potential(dft.c_func,Pa,Pb,"Vc.dat");
      grid.check_potential(dft.c_func,Pa,Pb,"Vc_nan.dat");
    }
  }

  return 0;
}
