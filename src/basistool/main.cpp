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

#include "../basislibrary.h"
#include "../stringutil.h"
#include "../completeness/completeness_profile.h"

std::string cmds[]={"completeness", "composition", "daug", "decontract", "dump", "dumpdec", "genbas", "orth", "Porth", "save", "savedalton", "sort", "taug"};


void help() {
  printf("Valid commands:\n");
  for(size_t i=0;i<sizeof(cmds)/sizeof(cmds[0]);i++)
    printf("\t%s\n",cmds[i].c_str());
}

int main(int argc, char **argv) {
  printf("ERKALE - Basis set tools from Hel.\n");
  print_copyright();
  print_license();

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
  bas.normalize();

  // Get command
  std::string cmd(argv[2]);
  // and determine what to do.
  if(stricmp(cmd,"completeness")==0) {
    // Print completeness profile.

    if(argc!=5 && argc!=6) {
      printf("\nUsage: %s input.gbs completeness element output.dat (coulomb)\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);
    bool coulomb=false;
    if(argc==6)
      coulomb=atoi(argv[5]);

    // Get wanted element from basis
    ElementBasisSet elbas=bas.get_element(el);

    // Compute completeness profile
    compprof_t prof=compute_completeness(elbas,-10.0,10.0,2001,coulomb);

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

    if(argc!=3 && argc!=4) {
      printf("\nUsage: %s input.gbs composition (El)\n",argv[0]);
      return 1;
    }

    // Elemental basis sets
    std::vector<ElementBasisSet> elbases;

    if(argc==4)
      elbases.push_back(bas.get_element(argv[3]));
    else
      elbases=bas.get_elements();

    printf("\n");
    printf("el at#  [npr|nbf] [primitive|contracted(?)]\n");
    printf("-------------------------------------------\n");

    // Loop over elements
    for(size_t iel=0;iel<elbases.size();iel++) {
      // Get the basis set
      ElementBasisSet elbas=elbases[iel];

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

      // Determine if basis set is contracted and the amount of
      // functions
      bool contr=false;
      size_t nbf=0;
      size_t nprim=0;
      for(int am=0;am<max_am;am++) {
	// Number of primitives
	nprim+=Nsh(am,0)*(2*am+1);
	// Number of contracted functions
	nbf+=Nsh(am,1)*(2*am+1);
      }
      if(nbf!=nprim)
	contr=true;

      // Print composition
      printf("%-2s %3i ",elbas.get_symbol().c_str(),(int) elbas.get_number());
      if(contr) {
	// Print amount of functions
	char cmp[20];
	sprintf(cmp,"[%i|%i]",(int) nprim,(int) nbf);
	printf("%10s [",cmp);

	// Print primitives
	for(int am=0;am<max_am;am++)
	  if(Nsh(am,0)>0)
	    printf("%i%c",Nsh(am,0),tolower(shell_types[am]));
	// Print contractions
	printf("|");
	for(int am=0;am<max_am;am++)
	  if(Nsh(am,0)!=Nsh(am,1))
	    printf("%i%c",Nsh(am,1),tolower(shell_types[am]));
	printf("]\n");
      } else {
	printf("%10i  ",(int) nbf);
	for(int am=0;am<max_am;am++)
	  if(Nsh(am,0)>0)
	    printf("%i%c",Nsh(am,0),tolower(shell_types[am]));
	printf("\n");
      }
    }
  } else if(stricmp(cmd,"daug")==0 || stricmp(cmd,"taug")==0) {
    // Augment basis set

    if(argc!=4) {
      printf("\nUsage: %s input.gbs %s output.gbs\n",argv[0],tolower(cmd).c_str());
      return 1;
    }

    int naug;
    if(stricmp(cmd,"daug")==0)
      naug=1;
    else
      naug=2;

    std::string fileout(argv[3]);
    bas.augment(naug);
    bas.save_gaussian94(fileout);
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

    if(argc!=5 && argc!=6) {
      printf("\nUsage: %s input.gbs dump element output.gbs (number)\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);

    int no=0;
    if(argc==6)
      no=atoi(argv[5]);

    // Save output
    BasisSetLibrary elbas;
    elbas.add_element(bas.get_element(el,no));
    elbas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"dumpdec")==0) {
    // Dump wanted element in decontracted form.

    if(argc!=5 && argc!=6) {
      printf("\nUsage: %s input.gbs dumpdec element output.gbs (number)\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);

    int no=0;
    if(argc==6)
      no=atoi(argv[5]);

    // Save output
    BasisSetLibrary elbas;
    bas.decontract();
    elbas.add_element(bas.get_element(el,no));
    elbas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"genbas")==0) {
    // Generate basis set for xyz file

    if(argc!=5) {
      printf("\nUsage: %s input.gbs genbas system.xyz output.gbs\n",argv[0]);
      return 1;
    }

    // Load atoms from xyz file
    std::vector<atom_t> atoms=load_xyz(argv[3]);
    // Output file
    std::string fileout(argv[4]);
    // Save output
    BasisSetLibrary elbas;

    // Collect elements
    std::vector<ElementBasisSet> els=bas.get_elements();
    // Loop over atoms in system
    for(size_t iat=0;iat<atoms.size();iat++) {
      bool found=false;

      // First, check if there is a special basis for the atom.
      for(size_t iel=0;iel<els.size();iel++)
	if(stricmp(atoms[iat].el,els[iel].get_symbol())==0 && atoms[iat].num == els[iel].get_number()) {
	  // Yes, add it.
	  elbas.add_element(els[iel]);
	  found=true;
	  break;
	}

      // Otherwise, check if a general basis is already in the basis
      if(!found) {
	std::vector<ElementBasisSet> added=elbas.get_elements();
	for(size_t j=0;j<added.size();j++)
	  if(added[j].get_number()==0 && stricmp(atoms[iat].el,added[j].get_symbol())==0)
	    found=true;
      }

      // If general basis not found, add it.
      if(!found) {
	for(size_t iel=0;iel<els.size();iel++)
	  if(stricmp(atoms[iat].el,els[iel].get_symbol())==0 && els[iel].get_number()==0) {
	    // Yes, add it.
	    elbas.add_element(els[iel]);
	    found=true;
	    break;
	  }
      }

      if(!found) {
	std::ostringstream oss;
	oss << "Basis set for element " << atoms[iat].el << " does not exist in " << filein << "!\n";
	throw std::runtime_error(oss.str());
      }
    }
    elbas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"orth")==0) {
    // Orthogonalize basis

    if(argc!=4) {
      printf("\nUsage: %s input.gbs orth output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.orthonormalize();
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"Porth")==0) {
    // P-orthogonalize basis

    if(argc!=6) {
      printf("\nUsage: %s input.gbs Porth cutoff Cortho output.gbs\n",argv[0]);
      return 1;
    }

    double cutoff=atof(argv[3]);
    double Cortho=atof(argv[4]);
    std::string fileout=argv[5];
    bas.P_orthogonalize(cutoff,Cortho);
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"save")==0) {
    // Save basis

    if(argc!=4) {
      printf("\nUsage: %s input.gbs save output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"savedalton")==0) {
    // Save basis in Dalton format

    if(argc!=4) {
      printf("\nUsage: %s input.gbs savedalton output.dal\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.save_dalton(fileout);

  } else if(stricmp(cmd,"sort")==0) {
    // Sort basis set

    if(argc!=4) {
      printf("\nUsage: %s input.gbs sort output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.sort();
    bas.save_gaussian94(fileout);
  } else {
    printf("\nInvalid command.\n");

    help();
  }

  return 0;
}
