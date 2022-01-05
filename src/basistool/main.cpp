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
#include "../density_fitting.h"
#include "../linalg.h"
#ifdef SVNRELEASE
#include "../version.h"
#endif

std::string cmds[]={"augdiffuse", "augsteep", "choleskyaux", "fullcholeskyaux", "choleskydens", "choleskybasis", "completeness", "composition", "contractaux", "daug", "decontract", "densityfit", "dump", "dumpdec", "fiterr", "genbas", "gendecbas", "merge", "norm", "orth", "overlap", "Porth", "prodset", "save", "savecfour", "savedalton", "savemolpro", "sort", "taug"};

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

  } else if(stricmp(cmd,"choleskyaux")==0) {
    // Form Cholesky fitting basis set

    if(argc!=5) {
      printf("\nUsage: %s input.gbs choleskyaux thr output.gbs\n",argv[0]);
      return 1;
    }

    printf("Forming reduced auxiliary basis set by pivoted Cholesky decomposition\n");
    printf("See J. Chem. Theory Comput. 17, 6886 (2021). DOI: 10.1021/acs.jctc.1c00607\n\n");

    double thr(atof(argv[3]));
    std::string outfile(argv[4]);

    init_libint_base();

    settings.add_bool("UseLM","",true);
    settings.add_bool("OptLM","",false);
    settings.add_string("Decontract","","");
    settings.add_bool("BasisRotate","",false);
    settings.add_double("BasisCutoff","",0.0);

    bool full=false;
    int metric=0;
    BasisSetLibrary ret=bas.cholesky_set(thr,full,metric);
    ret.save_gaussian94(outfile);

  } else if(stricmp(cmd,"fullcholeskyaux")==0) {
    // Form Cholesky fitting basis set

    if(argc!=5) {
      printf("\nUsage: %s input.gbs fullcholeskyaux thr output.gbs\n",argv[0]);
      return 1;
    }

    printf("Forming full auxiliary basis set by pivoted Cholesky decomposition\n");
    printf("See J. Chem. Theory Comput. 17, 6886 (2021). DOI: 10.1021/acs.jctc.1c00607\n\n");

    double thr(atof(argv[3]));
    std::string outfile(argv[4]);

    init_libint_base();

    settings.add_bool("UseLM","",true);
    settings.add_bool("OptLM","",false);
    settings.add_string("Decontract","","");
    settings.add_bool("BasisRotate","",false);
    settings.add_double("BasisCutoff","",0.0);

    bool full=true;
    int metric=0;
    BasisSetLibrary ret=bas.cholesky_set(thr,full,metric);
    ret.save_gaussian94(outfile);

  } else if(stricmp(cmd,"choleskydens")==0) {
    // Form Cholesky fitting basis set

    if(argc!=5) {
      printf("\nUsage: %s input.gbs choleskydens thr output.gbs\n",argv[0]);
      return 1;
    }

    printf("Forming density fitting auxiliary basis using pivoted Cholesky decomposition\n");
    printf("See J. Chem. Theory Comput. 17, 6886 (2021). DOI: 10.1021/acs.jctc.1c00607\n");
    printf("NOTE: using overlap metric instead of Coulomb metric!\n\n");

    double thr(atof(argv[3]));
    std::string outfile(argv[4]);

    init_libint_base();

    settings.add_bool("UseLM","",true);
    settings.add_bool("OptLM","",false);
    settings.add_string("Decontract","","");
    settings.add_bool("BasisRotate","",false);
    settings.add_double("BasisCutoff","",0.0);

    bool full=true;
    int metric=1;
    BasisSetLibrary ret=bas.cholesky_set(thr,full,metric);
    ret.save_gaussian94(outfile);

  } else if(stricmp(cmd,"choleskynuc")==0) {
    // Form Cholesky fitting basis set

    if(argc!=5) {
      printf("\nUsage: %s input.gbs choleskynuc thr output.gbs\n",argv[0]);
      return 1;
    }

    printf("Forming density fitting auxiliary basis using pivoted Cholesky decomposition\n");
    printf("See J. Chem. Theory Comput. 17, 6886 (2021). DOI: 10.1021/acs.jctc.1c00607\n");
    printf("NOTE: using nuclear attraction integrals instead of Coulomb metric!\n\n");

    double thr(atof(argv[3]));
    std::string outfile(argv[4]);

    init_libint_base();

    settings.add_bool("UseLM","",true);
    settings.add_bool("OptLM","",false);
    settings.add_string("Decontract","","");
    settings.add_bool("BasisRotate","",false);
    settings.add_double("BasisCutoff","",0.0);

    bool full=true;
    int metric=2;
    BasisSetLibrary ret=bas.cholesky_set(thr,full,metric);
    ret.save_gaussian94(outfile);

  } else if(stricmp(cmd,"choleskybasis")==0) {
    if(argc!=7) {
      printf("\nUsage: %s input.gbs choleskybasis system.xyz thr uselm output.gbs\n",argv[0]);
      return 1;
    }

    printf("Forming system-specific Cholesky orthogonalized basis\n");
    printf("See J. Chem. Phys. 151, 241102 (2019). DOI: 10.1063/1.5139948\n\n");

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

  } else if(stricmp(cmd,"contractaux")==0) {
    // Contract auxiliary basis

    if(argc!=6) {
      printf("\nUsage: %s orbbas.gbs contractaux auxbas.gbs threshold output.gbs\n",argv[0]);
      return 1;
    }

    // Load auxiliary basis
    BasisSetLibrary auxbas;
    auxbas.load_gaussian94(argv[3]);
    double threshold(atof(argv[4]));
    std::string outname(argv[5]);

    settings.add_string("Decontract","","");
    settings.add_bool("BasisRotate","",false);
    settings.add_double("BasisCutoff","",1e-10);
    settings.add_bool("UseLM","",true);
    settings.add_bool("OptLM","",false);

    init_libint_base();

    // Basis set library to write out
    BasisSetLibrary contracted;

    if(threshold<=0.0) {
      printf("Contracting auxiliary basis set with threshold specified by highest angular momentum.\n",threshold);
    } else {
      printf("Contracting auxiliary basis set with threshold %e\n",threshold);
    }

    // Loop over elements in the auxiliary basis set
    std::vector<ElementBasisSet> auxelements(auxbas.get_elements());
    for(size_t iaux=0; iaux<auxelements.size(); iaux++) {
      // Get the name of the element
      std::string element = auxelements[iaux].get_symbol();

      // Contracted auxiliary basis
      ElementBasisSet contraux(element);

      // Dummy atom
      std::vector<atom_t> atoms(1);
      atoms[0].el=element;
      atoms[0].num=0;
      atoms[0].x=0.0;
      atoms[0].y=0.0;
      atoms[0].z=0.0;
      atoms[0].Q=0;

      // Construct basis sets
      BasisSet orbbasis, auxbasis;
      construct_basis(orbbasis, atoms, bas);
      construct_basis(auxbasis, atoms, auxbas);
      auxbasis.coulomb_normalize();

      size_t Naux = auxbasis.get_Nbf();

      // Initialize density fitting code
      DensityFit dfit;
      bool direct=false;
      double erithr=1e-10;
      double linthr=1e-6;
      bool bmat=true;
      dfit.fill(orbbasis, auxbasis, direct, erithr, linthr, bmat);

      // Get the 3c integrals matrix
      arma::mat I3c;
      dfit.three_center_integrals(I3c);

      // Form the W matrix in the non-orthogonal basis
      arma::mat Wno(arma::trans(I3c)*I3c);

      // Get the shells in the auxiliary basis set
      std::vector<GaussianShell> auxshells(auxbasis.get_shells());
      // List of shells with the wanted angular momentum
      std::vector<arma::uvec> am_shells(auxbasis.get_max_am()+1);
      for(int am=0;am<=auxbasis.get_max_am();am++) {
        std::vector<size_t> shellidx;
        for(size_t is=0; is<auxshells.size(); is++) {
          // Check we have right angular momentum
          if(auxshells[is].get_am() != am) continue;
          // Check for pure angular momentum
          if(!auxshells[is].lm_in_use()) {
            throw std::logic_error("Must use spherical auxiliary basis!\n");
          }
          shellidx.push_back(is);
        }
        am_shells[am] = arma::conv_to<arma::uvec>::from(shellidx);
      }

      // Returns functions with wanted am
      auto l_functions = [am_shells, auxshells](int l) {
        std::vector<size_t> iidx;
        for(auto is: am_shells[l]) {
          for(int m=-l;m<=l;m++) {
            iidx.push_back(auxshells[is].get_first_ind() + l+m);
          }
        }
        return arma::conv_to<arma::uvec>::from(iidx);
      };
      auto lm_functions = [am_shells, auxshells](int l, int m) {
        std::vector<size_t> iidx;
        for(auto is: am_shells[l]) {
          size_t iind(auxshells[is].get_first_ind() + l+m);
          iidx.push_back(iind);
        }
        return arma::conv_to<arma::uvec>::from(iidx);
      };

#if 0
      // Compute W norm
      arma::mat Wnorm(auxbasis.get_max_am()+1,auxbasis.get_max_am()+1);
      for(size_t iam=0;iam<=auxbasis.get_max_am();iam++) {
        for(size_t jam=0;jam<=iam;jam++) {
          arma::mat Wsub(Wno(l_functions(iam),l_functions(jam)));
          Wnorm(iam,jam)=arma::norm(Wsub,"fro");
          Wnorm(jam,iam)=Wnorm(iam,jam);
        }
      }
      printf("Sum of l-diagonal norm W %e sum of off-diagonal l-l' %e\n",arma::sum(arma::abs(arma::diagvec(Wnorm))), arma::sum(arma::sum(arma::abs(Wnorm-arma::diagmat(arma::diagvec(Wnorm))))));

      // Analyze l-l blocks
      for(int am=0;am<=auxbasis.get_max_am();am++) {
        arma::mat Wmnorm(2*am+1,2*am+1,arma::fill::zeros);
        for(int im=-am;im<=am;im++) {
          for(int jm=-am;jm<=im;jm++) {
            arma::mat Wsub(Wno(lm_functions(am,im),lm_functions(am,jm)));
            Wmnorm(im+am,jm+am) = arma::norm(Wsub,"fro");
            Wmnorm(jm+am,im+am) = Wmnorm(im+am,jm+am);
          }
        }
        int m0idx = am;
        printf("%i-%i diagonal sum %e sum of other elements %e\n",am,am,arma::sum(arma::abs(arma::diagvec(Wmnorm))), arma::sum(arma::sum(arma::abs(Wmnorm-arma::diagmat(arma::diagvec(Wmnorm))))));
      }
#endif

      // Get the (a|b) integrals
      arma::mat ab = dfit.get_ab();

      // Form contractions
      std::vector<arma::vec> exps(auxbasis.get_max_am()+1);
      std::vector<arma::mat> coeffs(auxbasis.get_max_am()+1);
      std::vector<arma::vec> evals(auxbasis.get_max_am()+1);
      for(int am=0;am<=auxbasis.get_max_am();am++) {
        size_t Nprim = am_shells[am].n_elem;

        // Extract W submatrix
        auto iv = lm_functions(am,0);
        arma::mat Wnosub(Wno.submat(iv,iv));

        // Extract ab submatrix
        arma::mat absub(ab.submat(iv,iv));
        arma::vec abval;
        arma::mat abvec;
        eig_sym_ordered(abval, abvec, absub);

        // Throw out vectors with small eigenvalues
        arma::uvec indep(arma::find(abval >= 1e-7));
        abval=abval(indep);
        abvec=abvec.cols(indep);

        // Symmetric orthogonalization
        arma::mat X = abvec * arma::diagmat(arma::pow(abval, -0.5));

        // Now extract the contraction coefficients from an eigendecomposition
        arma::mat Wsub = X.t() * Wnosub * X;
        arma::vec Wval;
        arma::mat Wvec;
        eig_sym_ordered(Wval, Wvec, Wsub);
        // Convert vectors to the original non-orthogonal basis.
        Wvec = X*Wvec;

        // Collect exponents
        exps[am].zeros(Nprim);
        for(size_t ix=0;ix < Wvec.n_rows; ix++) {
          exps[am][ix] = auxshells[am_shells[am][ix]].get_contr()[0].z;
        }
        coeffs[am] = Wvec;
        evals[am] = Wval;

        std::ostringstream legend;
        legend << element << " l= " << am << " eigenvalues";
        arma::reverse(Wval).print(legend.str());
      }

      double elthresh = threshold;
      if(elthresh<0) {
        elthresh = arma::max(evals[evals.size()-1]);
        printf("Employing threshold %e for %s\n",elthresh,element.c_str());
      }

      // Number of basis functions
      size_t norig=0, ncontr=0;

      std::ostringstream ucomp, ccomp;
      for(int am=0;am<=auxbasis.get_max_am();am++) {
        // Keep the vectors above the threshold
        arma::uvec keep_idx(arma::find(evals[am] >= elthresh));
        arma::vec Wval(evals[am](keep_idx));
        arma::mat Wvec(coeffs[am].cols(keep_idx));

        // Number of basis functions
        norig += evals[am].n_elem*(2*am+1);
        ncontr += keep_idx.n_elem*(2*am+1);

        if(evals[am].n_elem)
          ucomp << evals[am].n_elem << char(tolower(shell_types[am]));
        if(keep_idx.n_elem)
          ccomp << keep_idx.n_elem << char(tolower(shell_types[am]));

        // Add functions to basis
        for(size_t ic=0;ic < Wvec.n_cols;ic++) {
          std::vector<contr_t> C;
          for(size_t ix=0;ix < Wvec.n_rows; ix++) {
            contr_t entry;
            // The exponent is just
            entry.z = exps[am][ix];
            // Our auxiliary basis functions are normalized in the
            // Coulomb metric; however, library basis sets are in the
            // overlap normalization. This means that we need to scale
            // our contraction coefficient by the square root of the
            // exponent: the one-center Coulomb overlap for angular
            // momentum l is the same as the normal overlap for
            // angular momentum l-1.
            entry.c = Wvec(ix, ic)*sqrt(entry.z);
            if(entry.c != 0) {
              C.push_back(entry);
            }
          }
          contraux.add_function(FunctionShell(am, C));
        }
      }
      contracted.add_element(contraux);

      printf("%s -> %s contraction reduces number of auxiliary functions for %s from %i to %i implying a % .1f %% reduction\n",ucomp.str().c_str(),ccomp.str().c_str(),element.c_str(),norig,ncontr,(norig-ncontr)*100.0/norig);

    }
    contracted.save_gaussian94(outname);

    if(false) {
      // Test that basis is ok
      for(size_t iaux=0; iaux<auxelements.size(); iaux++) {
        // Get the name of the element
        std::string element = auxelements[iaux].get_symbol();

        // Dummy atom
        std::vector<atom_t> atoms(1);
        atoms[0].el=element;
        atoms[0].num=0;
        atoms[0].x=0.0;
        atoms[0].y=0.0;
        atoms[0].z=0.0;
        atoms[0].Q=0;

        BasisSet orbbasis, auxbasis;
        construct_basis(orbbasis, atoms, bas);
        construct_basis(auxbasis, atoms, contracted);
        auxbasis.coulomb_normalize();

        DensityFit dfit;
        bool direct=false;
        double erithr=1e-10;
        double linthr=1e-6;
        bool bmat=false;
        dfit.fill(orbbasis, auxbasis, direct, erithr, linthr, bmat);

        // This matrix should be orthonormal, since the contracted functions are orthonormalized
        arma::mat ab(dfit.get_ab());
        ab -= arma::eye<arma::mat>(ab.n_rows,ab.n_cols);
        double dnorm(arma::norm(ab,"fro"));
        printf("%s aux basis non-orthonormality %e\n",element.c_str(),dnorm);
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

  } else if(stricmp(cmd,"fiterr")==0) {
    // Calculate fit error in auxiliary basis

    if(argc!=5) {
      printf("\nUsage: %s orbbas.gbs fiterr auxbas.gbs element\n",argv[0]);
      return 1;
    }

    // Load auxiliary basis
    BasisSetLibrary auxbas;
    auxbas.load_gaussian94(argv[3]);
    std::string element(argv[4]);

    // Dummy atoms
    std::vector<atom_t> atoms(1);
    atoms[0].el=element;
    atoms[0].num=0;
    atoms[0].x=0.0;
    atoms[0].y=0.0;
    atoms[0].z=0.0;
    atoms[0].Q=0;

    settings.add_string("Decontract","","");
    settings.add_bool("BasisRotate","",false);
    settings.add_double("BasisCutoff","",1e-10);
    settings.add_bool("UseLM","",true);
    settings.add_bool("OptLM","",true);

    init_libint_base();

    // Construct basis sets
    BasisSet orbbasis, auxbasis;
    construct_basis(orbbasis, atoms, bas);
    construct_basis(auxbasis, atoms, auxbas);

    DensityFit dfit;
    bool direct=false;
    double erithr=1e-10;
    double linthr=1e-6;
    bool bmat=true;
    dfit.fill(orbbasis, auxbasis, direct, erithr, linthr, bmat);
    dfit.fitting_error();

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
#ifdef CATCH_EXCEPTIONS
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
#else
  return main_guarded(argc, argv);
#endif
}
