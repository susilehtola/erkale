/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "../checkpoint.h"
#include "../density_fitting.h"
#include "../erichol.h"
#include "../localization.h"
#include "../timer.h"
#include "../linalg.h"
#include "../eriworker.h"
#include "../settings.h"

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

arma::mat exchange_localization(const arma::mat & Co, const arma::mat & Cv0, const arma::mat & Bph, const arma::mat & S) {
  if(Cv0.n_cols<Co.n_cols)
    throw std::runtime_error("Not enough virtuals given!\n");
  if(Co.n_rows != Cv0.n_rows)
    throw std::runtime_error("Orbital matrices not consistent!\n");
  if(Bph.n_cols != Co.n_cols*Cv0.n_cols)
    throw std::runtime_error("B matrix is not in the vo space!\n");

  // Returned orbitals
  arma::mat Cv(Cv0);

  // Loop over occupied orbitals
  for(size_t io=0;io<Co.n_cols;io++) {
    // Collect the necessary elements
    arma::mat Bp(Cv.n_cols,Bph.n_rows);
    // Loop over virtuals
    for(size_t iv=0;iv<Cv.n_cols;iv++)
      for(size_t a=0;a<Bph.n_rows;a++)
	Bp(iv,a)=Bph(a,io*Cv.n_cols+iv);

    // Work block
    arma::mat Cwrk(Cv0.cols(0,Cv0.n_cols-1));

    // Virtual-virtual exchange matrix is
    arma::mat Kvv(Bp*arma::trans(Bp));

    // Eigendecomposition
    arma::vec eval;
    arma::mat evec;
    arma::eig_sym(eval,evec,-Kvv);

    // Rotate orbitals to new basis; orbital becomes lowest-lying orbital
    Cv.col(io)=Cwrk*evec.col(0);
  }

  // Keep only active orbitals
  Cv=Cv.cols(0,Co.n_cols-1);

  // and reorthonormalize them
  arma::mat Svv(arma::trans(Cv)*S*Cv);

  // Reorthogonalize
  arma::vec sval;
  arma::mat svec;
  arma::eig_sym(sval,svec,Svv);

  arma::vec sinvh(sval);
  for(size_t i=0;i<sval.n_elem;i++)
    sinvh(i)=1.0/sqrt(sval(i));

  // Orthogonalizing matrix is
  arma::mat O(svec*arma::diagmat(sinvh)*arma::trans(svec));
  Cv=Cv*O;

  return Cv;
}

void form_F(const arma::mat & H, const arma::mat & Cl, const arma::mat & Cr, const std::string & name, arma::file_type atype) {
  arma::mat F(arma::trans(Cl)*H*Cr);
  F.save(name+".dat",atype);
}

void form_B(const arma::mat & B, const arma::mat & Cl, const arma::mat & Cr, const std::string & name, arma::file_type atype) {
  Timer t;
  printf("Forming %s ... ",name.c_str());
  fflush(stdout);

  arma::mat Bt(B_transform(B,Cl,Cr));
  Bt.save(name+".dat",atype);

  printf("done (%s)\n",t.elapsed().c_str());
  fflush(stdout);
}

void form_B(const ERIchol & chol, const arma::mat & Cl, const arma::mat & Cr, const std::string & name, arma::file_type atype) {
  Timer t;
  printf("Forming %s ... ",name.c_str());
  fflush(stdout);

  arma::mat Bt(chol.B_transform(Cl,Cr));
  Bt.save(name+".dat",atype);

  printf("done (%s)\n",t.elapsed().c_str());
  fflush(stdout);
}


int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - MO integrals from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - MO integrals from Hel, serial version.\n");
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

  // Initialize libint
  init_libint_base();

  Timer t;

  // Parse settings
  Settings set;
  set.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  set.add_bool("DensityFitting","Use density fitting instead of Cholesky?",false);
  set.add_string("FittingBasis","Fitting basis to use","");
  set.add_double("FittingThr","Linear dependency threshold for fitting basis",1e-7);
  set.add_double("CholeskyThr","Cholesky threshold to use",1e-7);
  set.add_double("CholeskyShThr","Cholesky shell threshold to use",0.01);
  set.add_double("IntegralThresh","Integral threshold",1e-10);
  set.add_int("CholeskyMode","Cholesky mode",0,true);
  set.add_int("NActive","Size of active space (0 for full transform)",0);
  set.add_bool("Localize","Localize active space orbitals?",false);
  set.add_bool("Binary","Use binary I/O?",true);

  if(argc==2)
    set.parse(argv[1]);
  else printf("Using default settings.\n\n");

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);
  bool densityfit=set.get_bool("DensityFitting");
  std::string fittingbasis=set.get_string("FittingBasis");
  double fitthr=set.get_double("FittingThr");
  double intthr=set.get_double("IntegralThresh");
  int cholmode=set.get_int("CholeskyMode");
  double cholthr=set.get_double("CholeskyThr");
  double cholshthr=set.get_double("CholeskyShThr");
  bool binary=set.get_bool("Binary");
  int Nact=set.get_int("NActive");
  bool loc=set.get_bool("Localize");

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Restricted calculation?
  int restr;
  chkpt.read("Restricted",restr);

  // Amount of electrons
  int Nela, Nelb;
  chkpt.read("Nel-a",Nela);
  chkpt.read("Nel-b",Nelb);

  // Density matrix
  arma::mat P;
  chkpt.read("P",P);

  // File type
  arma::file_type atype = binary ? arma::arma_binary : arma::raw_ascii;

  ERIchol chol;
  DensityFit dfit;

  if(densityfit) {
    // Construct fitting basis
    BasisSetLibrary fitlib;
    fitlib.load_gaussian94(fittingbasis);

    // Dummy settings
    Settings basset;
    basset.add_string("Decontract","","");
    basset.add_bool("BasisRotate","",true);
    basset.add_bool("UseLM","",true);
    basset.add_double("BasisCutoff","",1e-8);

    // Construct fitting basis
    BasisSet dfitbas;
    construct_basis(dfitbas,basis.get_nuclei(),fitlib,basset);
    printf("Auxiliary basis set has %i functions.\n",(int) dfitbas.get_Nbf());

    // Calculate fitting integrals. In-core, with RI-K
    dfit.fill(basis,dfitbas,false,intthr,fitthr,true);
  } else {

    // Calculate Cholesky vectors
    if(cholmode==-1) {
      chol.load();
      if(true) {
	printf("%i Cholesky vectors loaded from file in %s.\n",(int) chol.get_Naux(),t.elapsed().c_str());
	fflush(stdout);
      }
    }

    if(chol.get_Nbf()!=basis.get_Nbf()) {
      chol.fill(basis,cholthr,cholshthr,intthr,true);
      if(cholmode==1) {
	t.set();
	chol.save();
	printf("Cholesky vectors saved to file in %s.\n",t.elapsed().c_str());
	fflush(stdout);
      }
    }
  }

  if(restr) {
    arma::mat C;
    chkpt.read("C",C);
    arma::mat H;
    chkpt.read("H",H);

    // Occ and virt orbitals
    arma::mat Cah(C.cols(0,Nela-1));
    arma::mat Cap;
    if(C.n_cols>(size_t) Nela)
      Cap=C.cols(Nela,C.n_cols-1);

    // Size of active space
    if(Nact!=0) {
      // Sanity check
      Nact=std::min(Nact,std::min(Nela,(int) (C.n_cols-Nela)));

      // Drop inactive orbitals
      Cah=Cah.cols(Cah.n_cols-Nact,Cah.n_cols-1);

      if(loc) {
	// Localize occupied orbitals
	double measure;
	arma::cx_mat W(Cah.n_cols,Cah.n_cols);
	W.eye();
	orbital_localization(PIPEK_IAO2,basis,Cah,P,measure,W);

	// Get ph B matrix
	arma::mat Bph;
	if(densityfit)
	  Bph=B_transform(dfit.B_matrix(),Cap,Cah);
	else
	  Bph=chol.B_transform(Cap,Cah);

	arma::mat S(basis.overlap());
	Cap=exchange_localization(Cah,Cap,Bph,S);
      } else {
	// Just drop the extra virtuals
	Cap=Cap.cols(0,Nact-1);
      }

      printf("Size of active space is %i\n",(int) Nact);
      printf("%i occupied and %i virtual orbitals\n",(int) Cah.n_cols,(int) Cap.n_cols);
    }

    // Fock matrices
    form_F(H,Cah,Cah,"Fhaha",atype);
    if(Cap.n_cols) {
      form_F(H,Cah,Cap,"Fpaha",atype);
      form_F(H,Cap,Cap,"Fpapa",atype);
    }
    
    // B matrices
    if(densityfit) {
      arma::mat B(dfit.B_matrix());
      form_B(B,Cah,Cah,"Bhaha",atype);
      if(Cap.n_cols) {
	form_B(B,Cap,Cah,"Bpaha",atype);
	form_B(B,Cap,Cap,"Bpapa",atype);
      }
    } else {
      form_B(chol,Cah,Cah,"Bhaha",atype);
      if(Cap.n_cols) {
	form_B(chol,Cap,Cah,"Bpaha",atype);
	form_B(chol,Cap,Cap,"Bpapa",atype);
      }
    }

  } else {
    arma::mat Ca, Cb;
    chkpt.read("Ca",Ca);
    chkpt.read("Cb",Cb);

    arma::mat Ha, Hb;
    chkpt.read("Ha",Ha);
    chkpt.read("Hb",Hb);

    // Occ and virt orbitals
    arma::mat Cah(Ca.cols(0,Nela-1));
    arma::mat Cap;
    if(Ca.n_cols>(size_t) Nela)
      Cap=Ca.cols(Nela,Ca.n_cols-1);

    arma::mat Cbh(Cb.cols(0,Nelb-1));
    arma::mat Cbp;
    if(Cb.n_cols>(size_t) Nelb)
      Cbp=Cb.cols(Nelb,Cb.n_cols-1);

    // Size of active space
    if(Nact!=0) {
      // Sanity check
      Nact=std::min(Nact,std::min(Nela,(int) (Ca.n_cols-Nela)));
      Nact=std::min(Nact,std::min(Nelb,(int) (Cb.n_cols-Nelb)));

      // Drop orbitals
      if(Nela>Nelb) {
	// Drop highest occupied alpha
	Cah.shed_cols(Nelb,Cah.n_cols-1);
	// and highest virtual beta
	Cbp.shed_cols(Cbp.n_cols-(Nela-Nelb),Cbp.n_cols-1);
      }

      // Drop inactive orbitals
      Cah=Cah.cols(Cah.n_cols-Nact,Cah.n_cols-1);
      Cbh=Cbh.cols(Cbh.n_cols-Nact,Cbh.n_cols-1);

      if(loc) {
	// Localize occupied orbitals
	double measure;

	arma::cx_mat Wa(Cah.n_cols,Cah.n_cols);
	Wa.eye();
	orbital_localization(PIPEK_IAO2,basis,Cah,P,measure,Wa);

	arma::cx_mat Wb(Cbh.n_cols,Cbh.n_cols);
	Wb.eye();
	orbital_localization(PIPEK_IAO2,basis,Cbh,P,measure,Wb);

	// Localize virtual orbitals
	arma::mat S(basis.overlap());
	{
	  arma::mat Bph;
	  if(densityfit)
	    Bph=B_transform(dfit.B_matrix(),Cap,Cah);
	  else
	    Bph=chol.B_transform(Cap,Cah);
	  Cap=exchange_localization(Cah,Cap,Bph,S);
	}
	{
	  arma::mat Bph;
	  if(densityfit)
	    Bph=B_transform(dfit.B_matrix(),Cbp,Cbh);
	  else
	    Bph=chol.B_transform(Cbp,Cbh);
	  Cbp=exchange_localization(Cbh,Cbp,Bph,S);
	}
      } else {
	// Just drop the extra virtuals
	Cap=Cap.cols(0,Nact-1);
	Cbp=Cbp.cols(0,Nact-1);
      }
    }

    // Fock matrices
    form_F(Ha,Cah,Cah,"Fhaha",atype);
    if(Cap.n_cols) {
      form_F(Ha,Cap,Cah,"Fpaha",atype);
      form_F(Ha,Cap,Cap,"Fpapa",atype);
    }

    form_F(Hb,Cbh,Cbh,"Fhbhb",atype);
    if(Cbp.n_cols) {
      form_F(Hb,Cbp,Cbh,"Fpbhb",atype);
      form_F(Hb,Cbp,Cbp,"Fpbpb",atype);
    }

    // B matrices
    if(densityfit) {
      arma::mat B(dfit.B_matrix());
      form_B(B,Cah,Cah,"Bhaha",atype);
      if(Cap.n_cols) {
	form_B(B,Cap,Cah,"Bpaha",atype);
	form_B(B,Cap,Cap,"Bpapa",atype);
      }
      form_B(B,Cbh,Cbh,"Bhbhb",atype);
      if(Cbp.n_cols) {
	form_B(B,Cbp,Cbh,"Bpbhb",atype);
	form_B(B,Cbp,Cbp,"Bpbpb",atype);
      }
    } else {
      form_B(chol,Cah,Cah,"Bhaha",atype);
      if(Cap.n_cols) {
	form_B(chol,Cap,Cah,"Bpaha",atype);
	form_B(chol,Cap,Cap,"Bpapa",atype);
      }
      form_B(chol,Cbh,Cbh,"Bhbhb",atype);
      if(Cbp.n_cols) {
	form_B(chol,Cbp,Cbh,"Bpbhb",atype);
	form_B(chol,Cbp,Cbp,"Bpbpb",atype);
      }
    }
  }

  return 0;
}
