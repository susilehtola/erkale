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
#include "../basis.h"

int main(int argc, char **argv) {
  print_copyright();
  print_license();

  if(argc!=5) {
    printf("Usage: %s zeta l Nf method\n",argv[0]);
    printf("zeta is the STO exponent to fit\n");
    printf("l is angular momentum to use\n");
    printf("Nf is number of exponents to use\n");
    printf("method is 0 for even-tempered, 1 for well-tempered and 2 for full optimization, or 3 for midpoint quadrature.\n");
    return 1;
  }

  // Read parameteres
  double zeta=atof(argv[1]);
  double am=atoi(argv[2]);
  int Nf=atoi(argv[3]);
  int method=atoi(argv[4]);

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
