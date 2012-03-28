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

#include "../basislibrary.h"
#include "../stringutil.h"
#include "../completeness/completeness_profile.h"

std::string cmds[]={"completeness", "composition", "decontract", "dump"};


void help() {
  printf("Valid commands:\n");
  for(size_t i=0;i<sizeof(cmds)/sizeof(cmds[0]);i++)
    printf("\t%s\n",cmds[i].c_str());
}

int main(int argc, char **argv) {
  if(argc<3) {
    printf("Usage: %s input.gbs command\n\n",argv[0]);
    help();
    return 0;
  }

  // Get filename
  std::string filein(argv[1]);
  // Load input
  BasisSetLibrary bas;
  bas.load_gaussian94(filein);

  // Get command
  std::string cmd(argv[2]);
  // and determine what to do.
  if(stricmp(cmd,"completeness")==0) {
    // Print completeness profile.

    if(argc!=5) {
      printf("\nUsage: %s input.gbs completeness element output.dat\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);

    // Get wanted element from basis
    ElementBasisSet elbas=bas.get_element(el);

    // Compute completeness profile
    compprof_t prof=compute_completeness(elbas);

    // Print profile in output file
    FILE *out=fopen(fileout.c_str(),"w");
    for(size_t i=0;i<prof.lga.size();i++) {
      // Value of scanning exponent
      fprintf(out,"%13e",prof.lga[i]);
      // Print completeness of shells
      for(size_t j=0;j<prof.shells.size();j++)
	fprintf(out,"\t%13e",prof.shells[j].Y[i]);
      fprintf(out,"\n");
    }
    fclose(out);

  } else if(stricmp(cmd,"composition")==0) {
    // Determine composition of basis set.

    if(argc!=4) {
      printf("\nUsage: %s input.gbs composition El\n",argv[0]);
      return 1;
    }

    // Get element
    std::string el(argv[3]);
    // Get the basis set
    ElementBasisSet elbas=bas.get_element(el);

    // Decontracted basis
    ElementBasisSet eldec(elbas);
    eldec.decontract();

    // Get the shells
    std::vector<FunctionShell> sh=elbas.get_shells();
    std::vector<FunctionShell> decsh=eldec.get_shells();

    // Count the shells
    arma::imat Nsh(max_am,2);
    Nsh.zeros();
    for(size_t ish=0;ish<decsh.size();ish++)
      Nsh(decsh[ish].get_am(),0)++;
    for(size_t ish=0;ish<sh.size();ish++)
      Nsh(sh[ish].get_am(),1)++;

    // Print the composition
    printf("\nComposition of basis set is:\n");
    printf("\tam Nprim Ncontr\n");
    for(int am=0;am<max_am;am++)
      if(Nsh(am,0)>0)
	printf("\t%2c %5i %6i\n",shell_types[am],Nsh(am,0),Nsh(am,1));

  } else if(stricmp(cmd,"decontract")==0) {
    // Decontract basis set.

    if(argc!=4) {
      printf("\nUsage: %s input.gbs decontract output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout(argv[3]);
    bas.decontract();
    bas.save_gaussian94(fileout);
  } else if(stricmp(cmd,"dump")==0) {
    // Dump wanted element.

    if(argc!=5) {
      printf("\nUsage: %s input.gbs dump element output.gbs\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);

    // Save output
    BasisSetLibrary elbas;
    elbas.add_element(bas.get_element(el));
    elbas.save_gaussian94(fileout);
  } else {
    printf("\nInvalid command.\n");

    help();
  }

  return 0;
}
