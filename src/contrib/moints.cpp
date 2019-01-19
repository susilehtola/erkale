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

#include <armadillo>
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

Settings settings;

arma::vec sano_energies(const arma::mat & Co, const arma::mat & Cv, const arma::mat & R, const arma::mat & Bph) {
  if(Co.n_rows != Cv.n_rows)
    throw std::runtime_error("Orbital matrices not consistent!\n");
  if(Bph.n_cols != Co.n_cols*Cv.n_cols)
    throw std::runtime_error("B matrix is not in the vo space!\n");

  // Number of orbitals to localize
  size_t Nloc=std::min(Co.n_cols,Cv.n_cols);
  arma::vec E(Nloc);

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

    // so the energy for the orbital is just
    E(ii)=arma::as_scalar(arma::trans(R.col(ii))*Kvv*R.col(ii));
  }

  return E;
}

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
  if(eigenvalue<settings.get_double("LinDepThresh")) {
    throw std::runtime_error("Sano virtuals are linearly dependent.\n");
  }

  // Calculate reciprocal square root
  arma::vec Sinvh(Sval);
  for(size_t i=0;i<Sinvh.n_elem;i++)
    Sinvh(i) = (Sinvh(i)>=settings.get_double("LinDepThresh")) ? 1.0/sqrt(Sinvh(i)) : 0.0;

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
  return R;
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
  return Rvirt;
}

arma::mat sano_guess(const arma::mat & Co, const arma::mat & Cv, const arma::mat & Bph) {
  arma::mat R;
  double eval;
  try {
    R=sano_guess_orig(Co,Cv,Bph,eval);
    printf("Default Sano guess was succesful, smallest eigenvalue of guess orbital overlap %e\n",eval);
  } catch(std::runtime_error & err) {
    printf("Default Sano guess failed due to linear dependencies (eigenvalue %e), switching to failsafe mode\n",eval);
    R=sano_guess_failsafe(Co,Cv,Bph);
  }

  sano_energies(Co,Cv,R,Bph).print("Sano guess exchange energies");

  return R;
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


void drop_virtuals(const std::string & dropvirt, const BasisSet & basis, arma::mat & C, int nocc) {
  std::vector<std::string> kw(splitline(dropvirt));
  if(kw.size()!=2) throw std::runtime_error("Incorrect length of DropVirt statement \"" + dropvirt + "\".\n");

  // Pi or sigma?
  bool pi;
  if(stricmp(kw[0],"pi")==0 || stricmp(kw[0],"p")==0)
    pi=true;
  else if(stricmp(kw[0],"sigma")==0 || stricmp(kw[0],"s")==0)
    pi=false;
  else throw std::runtime_error("Incorrect space in DropVirt statement \"" + dropvirt + "\".\n");

  // Axis
  int ax;
  if(stricmp(kw[1],"x")==0)
    ax=0;
  else if(stricmp(kw[1],"y")==0)
    ax=1;
  else if(stricmp(kw[1],"z")==0)
    ax=2;
  else throw std::runtime_error("Incorrect axis in DropVirt statement \"" + dropvirt + "\".\n");

  // Get coordinate matrix
  arma::mat coords=basis.get_nuclear_coords();
  // Buffer to put in each side
  double extra=2.0;
  // Minimum and maximum
  arma::rowvec minc=arma::min(coords)-extra;
  arma::rowvec maxc=arma::max(coords)+extra;

  // Number of points to sample over
  size_t Nsamp(10000);

  // Loop over height
  const double heights[]={1e-3, 1e-2, 1e-1, 1.0};

  arma::uvec slist, plist;
  for(size_t ih=0;ih<sizeof(heights)/sizeof(heights[0]);ih++) {
    // Orbitals evaluated at the top and bottom
    arma::mat schar(C.n_cols,Nsamp);
    arma::mat pchar(C.n_cols,Nsamp);

    for(size_t is=0;is<Nsamp;is++) {
      // Get uniformly distributed random numbers and convert to a
      // position in the box
      arma::vec r(arma::trans((maxc-minc)%arma::randu<arma::rowvec>(3)+minc));

      // Compute top
      r(ax)=heights[ih];
      arma::vec top(compute_orbitals(C,basis,vec_to_coords(r)));

      // Compute bottom
      r(ax)=-heights[ih];
      arma::vec bot(compute_orbitals(C,basis,vec_to_coords(r)));

      // Sigma and pi character
      schar.col(is)=arma::pow((top+bot)/2,2);
      pchar.col(is)=arma::pow((top-bot)/2,2);
    }

    // Total sigma and pi character
    arma::vec tots(arma::sum(arma::abs(schar),1));
    arma::vec totp(arma::sum(arma::abs(pchar),1));

    // Convert into percentage
    {
      arma::vec tot(tots+totp);
      tots%=100.0/tot;
      totp%=100.0/tot;
    }

    // Allowed character
    double thr(sqrt(DBL_EPSILON));
    arma::uvec pidx(arma::find(tots <= thr));
    arma::uvec sidx(arma::find(totp <= thr));
    if(pidx.n_elem + sidx.n_elem != C.n_cols) {
      std::ostringstream oss;
      oss << "Out of " << C.n_cols << " orbitals, only " << sidx.n_elem << " are sigma and " << pidx.n_elem << " are pi!\n";
      throw std::runtime_error(oss.str());
    }

    if(ih==0) {
      slist=sidx;
      plist=pidx;
    } else {
      // Check that orbitals are the same
      if(slist.n_elem != sidx.n_elem)
	throw std::runtime_error("Orbitals lists differ!\n");
      if(plist.n_elem != pidx.n_elem)
	throw std::runtime_error("Orbitals lists differ!\n");
    }
  }

  // Now, pick the orbitals
  arma::mat socc(C.cols(slist(arma::find(slist < nocc))));
  arma::mat pocc(C.cols(plist(arma::find(plist < nocc))));

  arma::mat svirt(C.cols(slist(arma::find(slist >= nocc))));
  arma::mat pvirt(C.cols(plist(arma::find(plist >= nocc))));

  // Recreate orbital coefficient matrix
  int no, nv;
  if(pi) {
    // Drop pi states
    C.zeros(C.n_rows,pocc.n_cols+socc.n_cols+svirt.n_cols);

    size_t i0=0;
    if(pocc.n_cols) {
      C.cols(i0,i0+pocc.n_cols-1)=pocc;
      i0+=pocc.n_cols;
    }
    if(socc.n_cols) {
      C.cols(i0,i0+socc.n_cols-1)=socc;
      i0+=socc.n_cols;
    }
    if(svirt.n_cols) {
      C.cols(i0,i0+svirt.n_cols-1)=svirt;
      i0+=svirt.n_cols;
    }
    no=pocc.n_cols;
    nv=pvirt.n_cols;

    pvirt.save("C_del.dat",arma::arma_binary);
  } else {
    // Drop sigma states
    C.zeros(C.n_rows,socc.n_cols+pocc.n_cols+pvirt.n_cols);

    size_t i0=0;

    if(socc.n_cols) {
      C.cols(i0,i0+socc.n_cols-1)=socc;
      i0+=socc.n_cols;
    }
    if(pocc.n_cols) {
      C.cols(i0,i0+pocc.n_cols-1)=pocc;
      i0+=pocc.n_cols;
    }
    if(pvirt.n_cols) {
      C.cols(i0,i0+pvirt.n_cols-1)=pvirt;
      i0+=pvirt.n_cols;
    }

    no=socc.n_cols;
    nv=svirt.n_cols;
    svirt.save("C_del.dat",arma::arma_binary);
  }

  const std::string types[]={"sigma","pi"};

  printf("\nMoved %i %s-type occupieds to core\n",no,types[pi].c_str());
  printf("Dropped %i %s-type virtuals\n\n",nv,types[pi].c_str());
}

void print_indices(const arma::uvec & idx, const std::string & leg) {
  printf("%s:",leg.c_str());
  for(size_t i=0;i<idx.n_elem;i++)
    printf(" %u",(unsigned int) idx(i));
  printf("\n");
}

int main_guarded(int argc, char **argv) {
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
  settings.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  settings.add_string("SaveChk","Checkpoint file to save data to","");
  settings.add_bool("DensityFitting","Use density fitting instead of Cholesky?",false);
  settings.add_string("FittingBasis","Fitting basis to use","");
  settings.add_double("FittingThr","Linear dependency threshold for fitting basis",1e-7);
  settings.add_double("CholeskyThr","Cholesky threshold to use",1e-7);
  settings.add_double("CholeskyShThr","Cholesky shell threshold to use",0.01);
  settings.add_double("IntegralThresh","Integral threshold",1e-10);
  settings.add_int("CholeskyMode","Cholesky mode",0,true);
  settings.add_bool("CholeskyInCore","Use more memory for Cholesky?",true);
  settings.add_bool("Localize","Localize orbitals?",false);
  settings.add_string("LocGroups","List of groups to localize","");
  settings.add_string("LocMethod","Localization method to use","IAO2");
  settings.add_bool("Binary","Use binary I/O?",true);
  settings.add_bool("Sano","Run Sano guess for virtual orbitals?",true);
  settings.add_int("Seed","Random seed",0);
  settings.add_string("DropVirt","Drop sigma/pi virtuals on x/y/z axis, e.g. pi z","");

  if(argc==2)
    settings.parse(argv[1]);
  else printf("Using default settings.\n\n");

  settings.print();

  // Load checkpoint
  std::string loadchk(settings.get_string("LoadChk"));
  std::string savechk(settings.get_string("SaveChk"));
  bool densityfit=settings.get_bool("DensityFitting");
  std::string fittingbasis=settings.get_string("FittingBasis");
  double fitthr=settings.get_double("FittingThr");
  double intthr=settings.get_double("IntegralThresh");
  int cholmode=settings.get_int("CholeskyMode");
  double cholthr=settings.get_double("CholeskyThr");
  double cholshthr=settings.get_double("CholeskyShThr");
  bool cholincore=settings.get_bool("CholeskyInCore");
  bool binary=settings.get_bool("Binary");
  bool loc=settings.get_bool("Localize");
  enum locmet locmethod(parse_locmet(settings.get_string("LocMethod")));
  bool sano=settings.get_bool("Sano");
  int seed=settings.get_int("Seed");
  std::string dropvirt=settings.get_string("DropVirt");

  if(savechk.size()) {
    // Copy checkpoint data
    std::ostringstream cmd;
    cmd << "cp " << loadchk << " " << savechk;
    if(system(cmd.str().c_str()))
      throw std::runtime_error("Error copying checkpoint file.\n");
  }

  Checkpoint chkpt(loadchk,false);

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
    Settings settings0(settings);
    settings=Settings();
    settings.add_string("Decontract","","");
    settings.add_bool("BasisRotate","",true);
    settings.add_bool("UseLM","",true);
    settings.add_double("BasisCutoff","",1e-8);

    // Construct fitting basis
    BasisSet dfitbas;
    construct_basis(dfitbas,basis.get_nuclei(),fitlib);
    printf("Auxiliary basis set has %i functions.\n",(int) dfitbas.get_Nbf());

    settings=settings0;

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

  // Data dump
  arma::mat Hcore;
  chkpt.read("Hcore",Hcore);
  Hcore.save("H.dat",arma::arma_binary);
  {
    arma::mat B;
    if(densityfit)
      dfit.B_matrix(B);
    else
      chol.B_matrix(B);
    B.save("B.dat",arma::arma_binary);
  }

  arma::ivec Nel(2);
  Nel(0)=Nela;
  Nel(1)=Nelb;
  Nel.save("Nel.dat",arma::arma_binary);

  arma::vec Enucr(1);
  {
    energy_t en;
    chkpt.read(en);
    Enucr(0)=en.Enucr;
    Enucr.save("Enucr.dat",arma::arma_binary);
  }

  arma::mat C;
  arma::mat H;
  if(restr) {
    chkpt.read("C",C);
    chkpt.read("H",H);
  } else {
    arma::mat Ca, Cb;
    chkpt.read("Ca",Ca);
    chkpt.read("Ha",H);
    chkpt.read("Cb",Cb);

    arma::mat SAO(basis.overlap());
    check_orth(Ca,SAO,true);

    // Form corresponding orbitals
    arma::mat S(Ca.cols(0,Nela-1).t()*SAO*Cb.cols(0,Nelb-1));
    // Decompose S^2
    arma::vec s;
    arma::mat U, V;
    arma::svd(U,s,V,S);

    s.print("Singular numbers");

    Ca.cols(0,Nela-1)=Ca.cols(0,Nela-1)*U;

    check_orth(Ca,SAO,true);

    C=Ca;
  }

  // Drop virtuals
  if(dropvirt.size()) drop_virtuals(dropvirt,basis,C,Nela);

  if(loc) {
    // Localize all Nela orbitals
    int nloc=Nela;

    arma::mat Chlp(C.cols(0,nloc-1));

    // Run the localization
    double measure;
    arma::cx_mat W(real_orthogonal(Chlp.n_cols,seed)*COMPLEX1);
    orbital_localization(locmethod,basis,Chlp,P,measure,W);

    // Rotate orbitals
    arma::mat Wr(arma::real(W));
    Chlp=Chlp*Wr;

    // and sort them in energy
    arma::vec Ehlp(arma::diagvec(Chlp.t()*H*Chlp));
    sort_eigvec(Ehlp,Chlp);

    // Put back the orbitals
    C.cols(0,nloc-1)=Chlp;
  }

  // Get ph B matrix
  if(sano) {
    // Corresponding virtuals for spin-paired orbitals
    arma::mat Cah(C.cols(0,Nelb-1));
    arma::mat Cap(C.cols(Nela,C.n_cols-1));
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

    // Rotate
    arma::mat R(sano_guess(Cah,Cap,Bph));
    // and store
    C.cols(Nela,C.n_cols-1)=Cap*R;
  }

  C.save("Ca.dat",arma::arma_binary);
  C.save("Cb.dat",arma::arma_binary);

  if(savechk.size()) {
    // Open in write mode but don't truncate
    Checkpoint save(savechk,true,false);
    save.write(basis);
    save.write("C",C);
  }

  energy_t en;
  chkpt.read(en);
  arma::vec Eref(1);
  Eref(0)=en.E;
  Eref.save("Eref.dat",atype);

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
