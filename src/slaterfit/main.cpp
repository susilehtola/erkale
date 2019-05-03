/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "form_exponents.h"
#include "solve_coefficients.h"
#include "settings.h"
#include "../basis.h"

#ifdef SVNRELEASE
#include "../version.h"
#endif

Settings settings;

int main_guarded(int argc, char **argv) {
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  settings.add_double("zeta","STO exponent to fit",1.0);
  settings.add_int("l","angular momentum",0);
  settings.add_int("nfunc","number of functions to use",3);
  settings.add_int("method","method to use: 0 even-tempered, 1 well-tempered, 2 full optimization, 3 midpoint quadrature",3);

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    settings.print();
    return 1;
  }
  settings.parse(std::string(argv[1]),true);
  settings.print();

  // Read parameteres
  double zeta=settings.get_double("zeta");
  double am=settings.get_int("l");
  int Nf=settings.get_int("nfunc");
  int method=settings.get_int("method");

  // Do the optimization
  std::vector<contr_t> contr;

  if(method>=0 && method<=2)
    contr=slater_fit(zeta,am,Nf,true,method);
  else if(method==3)
    contr=slater_fit_midpoint(zeta,am,Nf);
  else throw std::runtime_error("Unknown method.\n");

  // Print them out
  printf("\nExponential contraction\nc_i\t\tz_i\t\tlg z_i\n");
  for(size_t i=0;i<contr.size();i++)
    printf("% e\t%e\t% e\n",contr[i].c,contr[i].z,log10(contr[i].z));

  // Form basis set
  ElementBasisSet elbas("El");
  FunctionShell sh(am,contr);
  elbas.add_function(sh);

  // Save the basis set
  BasisSetLibrary baslib;
  baslib.add_element(elbas);
  baslib.save_gaussian94("slater-contr.gbs");

  // also in decontracted form
  baslib.decontract();
  baslib.save_gaussian94("slater-uncontr.gbs");

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
