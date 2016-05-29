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
#include "../mathf.h"
#include "../linalg.h"
#include "../eriworker.h"
#include "../settings.h"
#include "../stringutil.h"

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

arma::mat sano_guess(const arma::mat & Co, const arma::mat & Cv, const arma::mat & Bph) {
  if(Co.n_rows != Cv.n_rows)
    throw std::runtime_error("Orbital matrices not consistent!\n");
  if(Bph.n_cols != Co.n_cols*Cv.n_cols)
    throw std::runtime_error("B matrix is not in the vo space!\n");

  // Virtual space rotation matrix
  arma::mat Rvirt(Cv.n_cols,Cv.n_cols);
  Rvirt.eye();

  // Number of orbitals to localize
  size_t Nloc=std::min(Co.n_cols,Cv.n_cols);

  // Loop over orbitals
  printf("Sano guess running\n");
  for(size_t ii=0;ii<Nloc;ii++) {
    // Reverse the order the orbitals are treated to maximize
    // variational freedom for the HOMO
    size_t io=Co.n_cols-1-ii;

    // Collect the elements of the B matrix
    arma::mat Bp(Bph.n_rows,Cv.n_cols);
    for(size_t iv=0;iv<Cv.n_cols;iv++) {
      Bp.col(iv)=Bph.col(io*Cv.n_cols+iv);
    }

    // Virtual-virtual exchange matrix is
    arma::mat Kvv(arma::trans(Bp)*Bp);
    // Left-over virtuals are
    arma::mat Rsub(Rvirt.cols(ii,Rvirt.n_cols-1));
    // Transform exchange matrix into working space
    Kvv=arma::trans(Rsub)*Kvv*Rsub;

    // Eigendecomposition
    arma::vec eval;
    arma::mat evec;
    arma::eig_sym(eval,evec,-Kvv);

    printf("Orbital %3i/%-3i: eigenvalue %e\n",(int) io+1,(int) Nloc,eval(0));

    // Rotate orbitals to new basis; column ii becomes lowest eigenvector
    Rsub=Rsub*evec;
    // Update rotations
    Rvirt.cols(ii,Rvirt.n_cols-1)=Rsub;
  }

  // Now, the localized and inactive virtuals are obtained as
  return Cv*Rvirt;
}

void form_F(const arma::mat & H, const arma::mat & Cl, const arma::mat & Cr, const std::string & name, arma::file_type atype) {
  arma::mat F(arma::trans(Cl)*H*Cr);
  F.save(name+".dat",atype);
}

void form_B(const arma::mat & B, const arma::mat & Cl, const arma::mat & Cr, const std::string & name, arma::file_type atype) {
  Timer t;
  printf("Forming %s ... ",name.c_str());
  fflush(stdout);

  arma::mat Bt;
  if(Cl.n_cols && Cr.n_cols)
    Bt=B_transform(B,Cl,Cr);
  Bt.save(name+".dat",atype);

  printf("done (%s)\n",t.elapsed().c_str());
  fflush(stdout);
}

void form_B(const ERIchol & chol, const arma::mat & Cl, const arma::mat & Cr, const std::string & name, arma::file_type atype) {
  Timer t;
  printf("Forming %s ... ",name.c_str());
  fflush(stdout);

  arma::mat Bt;
  if(Cl.n_cols && Cr.n_cols)
    Bt=chol.B_transform(Cl,Cr,true);
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
  set.add_bool("CholeskyInCore","Use more memory for Cholesky?",true);
  set.add_string("Localize","Localize given set of orbitals","");
  set.add_string("LocMethod","Localization method to use","IAO2");
  set.add_bool("Binary","Use binary I/O?",true);
  set.add_bool("MP2","MP2 mode? (Dump only ph B matrix)",true);
  set.add_bool("Sano","Run Sano guess for virtual orbitals?",true);

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
  bool cholincore=set.get_bool("CholeskyInCore");
  bool binary=set.get_bool("Binary");
  bool loc=(set.get_string("Localize").size() != 0);
  enum locmet locmethod(parse_locmet(set.get_string("LocMethod")));
  bool mp2=set.get_bool("MP2");
  bool sano=set.get_bool("Sano");

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
    fitlib.load_basis(fittingbasis);

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

    // Size of memory for screened B matrix
    size_t Nscr=chol.get_Npairs()*chol.get_Naux();
    // Size of memory for full B matrix
    size_t Nfull=chol.get_Nbf()*chol.get_Nbf()*chol.get_Naux();
    if(cholincore) {
      printf("In-core Cholesky toggled, memory use is %s vs %s\n",memory_size(Nfull*sizeof(double)).c_str(),memory_size(Nscr*sizeof(double)).c_str());
    } else {
      printf("Direct  Cholesky toggled, memory use is %s vs %s\n",memory_size(Nscr*sizeof(double)).c_str(),memory_size(Nfull*sizeof(double)).c_str());
    }
  }

  if(restr) {
    arma::mat C;
    chkpt.read("C",C);
    arma::mat H;
    chkpt.read("H",H);

    if(loc) {
      // Localize orbitals. Get the indices
      std::vector<size_t> orbs(parse_range(splitline(set.get_string("Localize"))[0]));
      arma::mat Chlp(C.n_rows,orbs.size());
      for(size_t i=0;i<orbs.size();i++)
	Chlp.col(i)=C.col(orbs[i]);

      arma::mat Chlp0(Chlp);

      // Run the localization
      double measure;
      arma::cx_mat W(real_orthogonal(Chlp.n_cols)*COMPLEX1);
      orbital_localization(locmethod,basis,Chlp,P,measure,W);

      // Rotate orbitals
      arma::mat Wr(arma::real(W));
      Chlp=Chlp*Wr;

      // Put back the orbitals
      for(size_t i=0;i<orbs.size();i++)
	C.col(orbs[i])=Chlp.col(i);
    }

    // Occ and virt orbitals
    arma::mat Cah(C.cols(0,Nela-1));
    arma::mat Cap;
    if(C.n_cols>(size_t) Nela)
      Cap=C.cols(Nela,C.n_cols-1);

    // Get ph B matrix
    if(sano) {
      arma::mat Bph;
      if(densityfit || cholincore) {
	arma::mat B;
	if(densityfit)
	  dfit.B_matrix(B);
	else
	  chol.B_matrix(B);
	Bph=B_transform(B,Cap,Cah);
      } else
	Bph=chol.B_transform(Cap,Cah);

      Cap=sano_guess(Cah,Cap,Bph);
    }

    // Fock matrices
    form_F(H,Cah,Cah,"Fhaha",atype);
    if(Cap.n_cols) {
      form_F(H,Cap,Cah,"Fpaha",atype);
      form_F(H,Cap,Cap,"Fpapa",atype);
    }

    // B matrices
    if(densityfit || cholincore) {
      arma::mat B;
      if(densityfit)
	dfit.B_matrix(B);
      else
	chol.B_matrix(B);

      if(!mp2) form_B(B,Cah,Cah,"Bhaha",atype);
      form_B(B,Cap,Cah,"Bpaha",atype);
      if(!mp2) form_B(B,Cap,Cap,"Bpapa",atype);
    } else {
      if(!mp2) form_B(chol,Cah,Cah,"Bhaha",atype);
      form_B(chol,Cap,Cah,"Bpaha",atype);
      if(!mp2) form_B(chol,Cap,Cap,"Bpapa",atype);
    }

  } else {
    arma::mat Ca, Cb;
    chkpt.read("Ca",Ca);
    chkpt.read("Cb",Cb);

    arma::mat Ha, Hb;
    chkpt.read("Ha",Ha);
    chkpt.read("Hb",Hb);

    if(loc) {
      // Localize orbitals. Get the indices
      for(size_t is=0;is<2;is++) {
	std::vector<size_t> orbs(parse_range(splitline(set.get_string("Localize"))[is]));
	arma::mat Chlp(Ca.n_rows,orbs.size());
	for(size_t i=0;i<orbs.size();i++)
	  Chlp.col(i)=is ? Cb.col(orbs[i]) : Ca.col(orbs[i]);

	// Run the localization
	double measure;
	arma::cx_mat W(real_orthogonal(Chlp.n_cols)*COMPLEX1);
	orbital_localization(locmethod,basis,Chlp,P,measure,W);

	// Rotate orbitals
	arma::mat Wr(arma::real(W));
	Chlp=Chlp*Wr;

	// Put back the orbitals
	if(is)
	  for(size_t i=0;i<orbs.size();i++)
	    Cb.col(orbs[i])=Chlp.col(i);
	else
	  for(size_t i=0;i<orbs.size();i++)
	    Ca.col(orbs[i])=Chlp.col(i);
      }
    }

    // Occ and virt orbitals
    arma::mat Cah(Ca.cols(0,Nela-1));
    arma::mat Cap;
    if(Ca.n_cols>(size_t) Nela)
      Cap=Ca.cols(Nela,Ca.n_cols-1);

    arma::mat Cbh(Cb.cols(0,Nelb-1));
    arma::mat Cbp;
    if(Cb.n_cols>(size_t) Nelb)
      Cbp=Cb.cols(Nelb,Cb.n_cols-1);

    // Get ph B matrix
    if(sano) {
      arma::mat Bpaha, Bpbhb;
      if(densityfit || cholincore) {
	arma::mat B;
	if(densityfit)
	  dfit.B_matrix(B);
	else
	  chol.B_matrix(B);
	Bpaha=B_transform(B,Cap,Cah);
	Bpbhb=B_transform(B,Cbp,Cbh);
      } else {
	Bpaha=chol.B_transform(Cap,Cah);
	Bpbhb=chol.B_transform(Cbp,Cbh);
      }
      Cap=sano_guess(Cah,Cap,Bpaha);
      Cbp=sano_guess(Cbh,Cbp,Bpbhb);
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
    if(densityfit || cholincore) {
      arma::mat B;
      if(densityfit)
	dfit.B_matrix(B);
      else
	chol.B_matrix(B);

      if(!mp2) form_B(B,Cah,Cah,"Bhaha",atype);
      form_B(B,Cap,Cah,"Bpaha",atype);
      if(!mp2) form_B(B,Cap,Cap,"Bpapa",atype);

      if(!mp2) form_B(B,Cbh,Cbh,"Bhbhb",atype);
      form_B(B,Cbp,Cbh,"Bpbhb",atype);
      if(!mp2) form_B(B,Cbp,Cbp,"Bpbpb",atype);
    } else {
      if(!mp2) form_B(chol,Cah,Cah,"Bhaha",atype);
      form_B(chol,Cap,Cah,"Bpaha",atype);
      if(!mp2) form_B(chol,Cap,Cap,"Bpapa",atype);

      if(!mp2) form_B(chol,Cbh,Cbh,"Bhbhb",atype);
      form_B(chol,Cbp,Cbh,"Bpbhb",atype);
      if(!mp2) form_B(chol,Cbp,Cbp,"Bpbpb",atype);
    }
  }

  energy_t en;
  chkpt.read(en);
  arma::vec Eref(1);
  Eref(0)=en.E;
  Eref.save("Eref.dat",atype);

  return 0;
}
