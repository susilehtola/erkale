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

arma::mat sano_guess_orig(const arma::mat & Co, const arma::mat & Cv, const arma::mat & Bph, double & eigenvalue) {
  if(Co.n_rows != Cv.n_rows)
    throw std::runtime_error("Orbital matrices not consistent!\n");
  if(Bph.n_cols != Co.n_cols*Cv.n_cols)
    throw std::runtime_error("B matrix is not in the vo space!\n");

  // Number of orbitals to localize
  size_t Nloc=std::min(Co.n_cols,Cv.n_cols);
  // Localized virtuals
  arma::mat Rloc(Cv.n_cols,Nloc);
  Rloc.zeros();

  // Loop over orbitals
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

    // Eigendecomposition
    arma::vec eval;
    arma::mat evec;
    arma::eig_sym(eval,evec,-Kvv);

    printf("Orbital %3i/%-3i: eigenvalue %e\n",(int) io+1,(int) Nloc,-eval(0));

    // Store virtual
    Rloc.col(ii)=evec.col(0);
  }

  // Compute localized virtual - localized virtual overlap matrix
  arma::mat Sloc=arma::trans(Rloc)*Rloc;

  // Eigendecompose overlap matrix and form S^{-1/2}
  arma::vec Sval;
  arma::mat Svec;
  if(!arma::eig_sym(Sval,Svec,Sloc)) {
    eigenvalue=0;
    throw std::runtime_error("Sano virtuals are linearly dependent.\n");
  }

  //Sval.t().print("Eigenvalues");

  eigenvalue=arma::min(Sval);
  if(eigenvalue<LINTHRES) {
    throw std::runtime_error("Sano virtuals are linearly dependent.\n");
  }

  // Calculate reciprocal square root
  arma::vec Sinvh(Sval);
  for(size_t i=0;i<Sinvh.n_elem;i++)
    Sinvh(i) = (Sinvh(i)>=LINTHRES) ? 1.0/sqrt(Sinvh(i)) : 0.0;

  // Orthonormalizing matrix is
  arma::mat O(Svec*arma::diagmat(Sinvh)*arma::trans(Svec));

  // Orthonormalized localized virtuals are
  arma::mat Rorth(Rloc*O);
  if(false) {
    // Check matrix is orthogonal
    arma::mat test(arma::trans(Rorth)*Rorth);
    test-=arma::eye<arma::mat>(test.n_cols,test.n_cols);
    printf("Deviation from orthogonality in loc  space is %e\n",arma::norm(test,2));
  }

  arma::mat R(Cv.n_cols,Cv.n_cols);
  R.cols(0,Nloc-1)=Rorth;
  if(R.n_cols>Nloc) {
    // Form inactive space projector: P_{ab} = delta_{ab} - sum_{i=1}^{n_p} <\psi_a|\psi_i^*> <\psi_i^*|\psi_b>
    arma::mat P(Rorth.n_rows,Rorth.n_rows);
    P.eye();
    P-=Rorth*Rorth.t();

    arma::vec Rval;
    arma::mat Rvec;
    if(!arma::eig_sym(Rval,Rvec,P))
      throw std::runtime_error("Sano virtual projector is linearly dependent.\n");

    Rvec.t().print("Projector eigenvalues");

    // Final rotation matrix
    R.cols(Nloc,Cv.n_cols-1)=Rvec.cols(Nloc,Rvec.n_cols-1);
  }

  // Check matrix is orthogonal
  if(false) {
    arma::mat test(R.t()*R);
    test-=arma::eye<arma::mat>(test.n_cols,test.n_cols);
    printf("Deviation from orthogonality in full space is %e\n",arma::norm(test,2));
  }

  // Now, the localized and inactive virtuals are obtained as
  return Cv*R;
}

arma::mat sano_guess_failsafe(const arma::mat & Co, const arma::mat & Cv, const arma::mat & Bph) {
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

    printf("Orbital %3i/%-3i: eigenvalue %e\n",(int) io+1,(int) Nloc,-eval(0));

    // Rotate orbitals to new basis; column ii becomes lowest eigenvector
    Rsub=Rsub*evec;
    // Update rotations
    Rvirt.cols(ii,Rvirt.n_cols-1)=Rsub;
  }

  // Now, the localized and inactive virtuals are obtained as
  return Cv*Rvirt;
}

arma::mat sano_guess(const arma::mat & Co, const arma::mat & Cv, const arma::mat & Bph) {
  arma::mat Cp;
  double eval;
  try {
    Cp=sano_guess_orig(Co,Cv,Bph,eval);
    printf("Default Sano guess was succesful, smallest eigenvalue of guess orbital overlap %e\n",eval);
  } catch(std::runtime_error & err) {
    printf("Default Sano guess failed due to linear dependencies (eigenvalue %e), switching to failsafe mode\n",eval);
    Cp=sano_guess_failsafe(Co,Cv,Bph);
  }

  return Cp;
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

arma::uvec get_group(int g, arma::uword Nel, bool occ) {
  // Orbital numbers
  arma::umat num;

  // Load indices
  std::ostringstream oss;
  oss << "group_" << g << ".dat";
  if(!num.load(oss.str(),arma::raw_ascii))
    throw std::runtime_error("Failed to load orbital indices from file \"" + oss.str() + "\".\n");

  // Separate into occupied and virtual orbitals
  std::vector<arma::uword> list;
  for(size_t i=0;i<num.n_elem;i++)
    if(occ && num[i]<Nel)
      list.push_back(num[i]);
    else if(!occ && num[i]>=Nel)
      list.push_back(num[i]);

  return arma::conv_to<arma::uvec>::from(list);
}

void check_groups(std::vector<size_t> grps) {
  for(size_t iig=0;iig<grps.size();iig++) {
    size_t ig=grps[iig];
    arma::uvec il(get_group(ig,0,false));

    // Check the group itself
    for(size_t i=0;i<il.n_elem;i++)
      for(size_t j=i+1;j<il.n_elem;j++)
	if(il(i)==il(j)) {
	  std::ostringstream oss;
	  oss << "Orbital " << il(i) << " occurs twice in group " << ig << "!\n";
	  throw std::logic_error(oss.str());
	}

    // Check other groups
    for(size_t jjg=0;jjg<iig;jjg++) {
      size_t jg=grps[jjg];
      arma::uvec jl(get_group(jg,0,false));

      for(size_t i=0;i<il.n_elem;i++)
	for(size_t j=0;j<jl.n_elem;j++)
	  if(il(i)==jl(j)) {
	    std::ostringstream oss;
	    oss << "Orbital " << il(i) << " occurs in groups " << ig << " and " << jg << "!\n";
	    throw std::logic_error(oss.str());
	  }
    }
  }
}

arma::uvec sort_vecs(arma::mat & C, const arma::mat & H) {
  // Orbital eigenvalues
  arma::vec e(C.n_cols);
  for(size_t i=0;i<C.n_cols;i++)
    e(i)=arma::as_scalar(arma::trans(C.col(i))*H*C.col(i));

  // Sort orbitals in ascending energy
  arma::uvec order=arma::stable_sort_index(e,"ascend");
  C=C.cols(order);

  return order;
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
  set.add_bool("Localize","Localize orbitals?",false);
  set.add_string("LocGroups","List of groups to localize","");
  set.add_string("LocMethod","Localization method to use","IAO2");
  set.add_bool("Binary","Use binary I/O?",true);
  set.add_bool("MP2","MP2 mode? (Dump only ph B matrix)",true);
  set.add_bool("Sano","Run Sano guess for virtual orbitals?",true);
  set.add_string("SanoGroups","List of groups to localize","");

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
  bool loc=set.get_bool("Localize");
  std::vector<size_t> locgrps=parse_range(set.get_string("LocGroups"));
  enum locmet locmethod(parse_locmet(set.get_string("LocMethod")));
  bool mp2=set.get_bool("MP2");
  bool sano=set.get_bool("Sano");
  std::vector<size_t> sanogrps=parse_range(set.get_string("SanoGroups"));

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
      // Localize orbitals. Localize in groups?
      if(locgrps.size()>0) {
	check_groups(locgrps);
	for(size_t igrp=0;igrp<locgrps.size();igrp++) {
	  arma::uvec list(get_group(locgrps[igrp],Nela,true));

	  arma::mat Chlp(C.n_rows,list.n_elem);
	  for(size_t i=0;i<list.size();i++)
	    Chlp.col(i)=C.col(list[i]);

	  list.t().print("Localizing occupieds");

	  // Run the localization
	  double measure;
	  arma::cx_mat W(real_orthogonal(Chlp.n_cols)*COMPLEX1);
	  orbital_localization(locmethod,basis,Chlp,P,measure,W);

	  // Rotate orbitals
	  arma::mat Wr(arma::real(W));
	  Chlp=Chlp*Wr;

	  // Put back the orbitals
	  for(size_t i=0;i<list.size();i++)
	    C.col(list[i])=Chlp.col(i);
	}
      } else {
	arma::mat Chlp(C.cols(0,Nela-1));

	// Run the localization
	double measure;
	arma::cx_mat W(real_orthogonal(Chlp.n_cols)*COMPLEX1);
	orbital_localization(locmethod,basis,Chlp,P,measure,W);

	// Rotate orbitals
	arma::mat Wr(arma::real(W));
	Chlp=Chlp*Wr;

	// Put back the orbitals
	C.cols(0,Nela-1)=Chlp;
      }
    }

    // Occ and virt orbitals
    arma::mat Cah(C.cols(0,Nela-1));
    arma::mat Cap;
    if(C.n_cols>(size_t) Nela)
      Cap=C.cols(Nela,C.n_cols-1);

    // Get ph B matrix
    if(sano) {
      if(sanogrps.size()) {
	check_groups(sanogrps);
	for(size_t igrp=0;igrp<locgrps.size();igrp++) {
	  arma::uvec hlist(get_group(sanogrps[igrp],Nela,true));
	  arma::uvec plist(get_group(sanogrps[igrp],Nela,false));

	  plist.t().print("Sano localizing virtuals");
	  hlist.t().print("that match occupieds");

	  arma::mat Chlp(C.n_rows,hlist.n_elem);
	  for(size_t i=0;i<hlist.size();i++)
	    Chlp.col(i)=C.col(hlist[i]);

	  arma::mat Cplp(C.n_rows,plist.n_elem);
	  for(size_t i=0;i<plist.size();i++)
	    Cplp.col(i)=C.col(plist[i]);

	  // Sort occupied and virtual orbitals
	  if(loc) {
	    sort_vecs(Chlp,H).t().print("Occupied orbital order");

	    arma::uvec vord(sort_vecs(Cplp,H));
	    for(arma::uword i=0;i<vord.n_elem;i++)
	      if(vord(i)!=i)
		throw std::logic_error("Indexing error in virtuals!\n");
	  }

	  arma::mat Bph;
	  if(densityfit || cholincore) {
	    arma::mat B;
	    if(densityfit)
	      dfit.B_matrix(B);
	    else
	      chol.B_matrix(B);
	    Bph=B_transform(B,Cplp,Chlp);
	  } else
	    Bph=chol.B_transform(Cplp,Chlp);

	  // Run the sano guess
	  Cplp=sano_guess(Chlp,Cplp,Bph);

	  // Restore virtuals
	  for(size_t i=0;i<plist.size();i++)
	    C.col(plist[i])=Cplp.col(i);
	}
      } else {
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
	arma::mat Chlp = (is==0) ? Ca.cols(0,Nela-1) : Cb.cols(0,Nelb-1);

	// Run the localization
	double measure;
	arma::cx_mat W(real_orthogonal(Chlp.n_cols)*COMPLEX1);
	orbital_localization(locmethod,basis,Chlp,P,measure,W);

	// Rotate orbitals
	arma::mat Wr(arma::real(W));
	Chlp=Chlp*Wr;

	// Put back the orbitals
	if(is==0)
	  Ca.cols(0,Nela-1)=Chlp;
	else
	  Cb.cols(0,Nelb-1)=Chlp;
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
