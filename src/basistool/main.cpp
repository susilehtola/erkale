/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2019
 * Copyright (c) 2010-2019, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "../basislibrary.h"
#include "../stringutil.h"
#include "../eriworker.h"
#include "../settings.h"
#include "pivoted_cholesky_basis.h"
#include "../completeness/completeness_profile.h"
#ifdef SVNRELEASE
#include "../version.h"
#endif

std::string cmds[]={"augdiffuse", "augsteep", "cholesky", "choleskybasis", "completeness", "composition", "daug", "decontract", "densityfit", "dump", "dumpdec", "genbas", "gendecbas", "merge", "norm", "orth", "overlap", "Porth", "prodset", "save", "savecfour", "savedalton", "savemolpro", "sort", "taug"};


void help() {
  printf("Valid commands:\n");
  for(size_t i=0;i<sizeof(cmds)/sizeof(cmds[0]);i++)
    printf("\t%s\n",cmds[i].c_str());
}

Settings settings;

int main_guarded(int argc, char **argv) {
  printf("ERKALE - Basis set tools from Hel.\n");
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc<3) {
    printf("Usage: %s input.gbs command\n\n",argv[0]);
    help();
    return 0;
  }

  // Get filename
  std::string filein(argv[1]);
  // Load input
  BasisSetLibrary bas;
  bas.load_basis(filein);

  // Get command
  std::string cmd(argv[2]);
  // and determine what to do.
  if(stricmp(cmd,"augdiffuse")==0) {
    // Augment basis set

    if(argc!=5) {
      printf("\nUsage: %s input.gbs %s nexp output.gbs\n",argv[0],tolower(cmd).c_str());
      return 1;
    }

    int naug=atoi(argv[3]);
    std::string fileout(argv[4]);

    bas.augment_diffuse(naug);
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"augsteep")==0) {
    // Augment basis set

    if(argc!=5) {
      printf("\nUsage: %s input.gbs %s nexp output.gbs\n",argv[0],tolower(cmd).c_str());
      return 1;
    }

    int naug=atoi(argv[3]);
    std::string fileout(argv[4]);

    bas.augment_steep(naug);
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"cholesky")==0) {
    // Form Cholesky fitting basis set

    if(argc!=7) {
      printf("\nUsage: %s input.gbs cholesky thr maxam cholthr output.gbs\n",argv[0]);
      return 1;
    }

    double thr(atof(argv[3]));
    int maxam(atoi(argv[4]));
    double ovlthr(atof(argv[5]));
    std::string outfile(argv[6]);

    if(maxam>=LIBINT_MAX_AM) {
      printf("Setting maxam = %i because limitations in used version of LIBINT.\n",LIBINT_MAX_AM-1);
      maxam=LIBINT_MAX_AM-1;
    }

    init_libint_base();
    BasisSetLibrary ret=bas.cholesky_set(thr,maxam,ovlthr);
    ret.save_gaussian94(outfile);

  } else if(stricmp(cmd,"choleskybasis")==0) {
    if(argc!=7) {
      printf("\nUsage: %s input.gbs choleskybasis system.xyz thr uselm output.gbs\n",argv[0]);
      return 1;
    }

    std::vector<atom_t> atoms=load_xyz(argv[3],false);
    double thr(atof(argv[4]));
    int uselm(atof(argv[5]));
    std::string outfile(argv[6]);
    settings.add_scf_settings();
    settings.set_bool("UseLM",uselm);

    BasisSetLibrary ret=pivoted_cholesky_basis(atoms,bas,thr);
    ret.save_gaussian94(outfile);

  } else if(stricmp(cmd,"completeness")==0) {
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
    compprof_t prof=compute_completeness(elbas,-10.0,15.0,3001,coulomb);

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
	    printf("%i%c",(int) Nsh(am,0),tolower(shell_types[am]));
	// Print contractions
	printf("|");
	for(int am=0;am<max_am;am++)
	  if(Nsh(am,0)!=Nsh(am,1))
	    printf("%i%c",(int) Nsh(am,1),tolower(shell_types[am]));
	printf("]\n");
      } else {
	printf("%10i  ",(int) nbf);
	for(int am=0;am<max_am;am++)
	  if(Nsh(am,0)>0)
	    printf("%i%c",(int) Nsh(am,0),tolower(shell_types[am]));
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
    bas.augment_diffuse(naug);
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

  } else if(stricmp(cmd,"densityfit")==0) {
  // Generate density fitted set

    if(argc!=6) {
      printf("\nUsage: %s input.gbs densityfit lval fsam output.gbs\n",argv[0]);
      return 1;
    }

    int lval(atoi(argv[3]));
    double fsam(atof(argv[4]));
    std::string fileout(argv[5]);
    BasisSetLibrary dfit(bas.density_fitting(lval,fsam));
    dfit.save_gaussian94(fileout);

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
    std::vector<atom_t> atoms=load_xyz(argv[3],false);
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
    elbas.sort();
    elbas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"gendecbas")==0) {
    // Generate decontracted basis set for xyz file

    if(argc!=5) {
      printf("\nUsage: %s input.gbs gendecbas system.xyz output.gbs\n",argv[0]);
      return 1;
    }

    // Load atoms from xyz file
    std::vector<atom_t> atoms=load_xyz(argv[3],false);
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
    elbas.decontract();
    elbas.sort();
    elbas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"merge")==0) {
    // Merge functions with too big overlap

    if(argc!=5) {
      printf("\nUsage: %s input.gbs merge cutoff output.gbs\n",argv[0]);
      return 1;
    }

    // Cutoff value
    double cutoff=atof(argv[3]);
    bas.merge(cutoff);
    bas.save_gaussian94(argv[4]);

  } else if(stricmp(cmd,"norm")==0) {
    // Normalize basis

    if(argc!=4) {
      printf("\nUsage: %s input.gbs norm output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.normalize();
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"orth")==0) {
    // Orthogonalize basis

    if(argc!=4) {
      printf("\nUsage: %s input.gbs orth output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.orthonormalize();
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"overlap")==0) {
    // Primitive overlap

    if(argc!=4) {
      printf("\nUsage: %s input.gbs overlap element\n",argv[0]);
      return 1;
    }

    // Get element basis set
    ElementBasisSet elbas=bas.get_element(argv[3]);
    elbas.decontract();

    // Loop over angular momentum
    for(int am=0;am<=elbas.get_max_am();am++) {
      // Get primitives
      arma::vec exps;
      arma::mat contr;
      elbas.get_primitives(exps,contr,am);

      // Compute overlap matrix
      arma::mat S=overlap(exps,exps,am);

      // Print out overlap
      printf("*** %c shell ***\n",shell_types[am]);
      exps.t().print("Exponents");
      printf("\n");

      S.print("Overlap");
      printf("\n");
    }

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

  } else if(stricmp(cmd,"prodset")==0) {
    // Generate product set

    if(argc!=6) {
      printf("\nUsage: %s input.gbs prodset lval fsam output.gbs\n",argv[0]);
      return 1;
    }

    int lval(atoi(argv[3]));
    double fsam(atof(argv[4]));
    std::string fileout(argv[5]);
    BasisSetLibrary dfit(bas.product_set(lval,fsam));
    dfit.save_gaussian94(fileout);

  } else if(stricmp(cmd,"save")==0) {
    // Save basis

    if(argc!=4) {
      printf("\nUsage: %s input.gbs save output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.save_gaussian94(fileout);

  } else if(stricmp(cmd,"savecfour")==0) {
    // Save basis in CFOUR format

    if(argc!=5) {
      printf("\nUsage: %s input.gbs savecfour name basis.cfour\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    std::string name=argv[4];
    bas.save_cfour(name,fileout);

  } else if(stricmp(cmd,"savedalton")==0) {
    // Save basis in Dalton format

    if(argc!=4) {
      printf("\nUsage: %s input.gbs savedalton output.dal\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.save_dalton(fileout);

  } else if(stricmp(cmd,"savemolpro")==0) {
    // Save basis in Molpro format

    if(argc!=4) {
      printf("\nUsage: %s input.gbs savemolpro output.mol\n",argv[0]);
      return 1;
    }

    std::string fileout=argv[3];
    bas.save_molpro(fileout);

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

int main(int argc, char **argv) {
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
