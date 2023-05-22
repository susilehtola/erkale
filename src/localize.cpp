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

#include "global.h"
#include "basis.h"
#include "checkpoint.h"
#include "mathf.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"
#include "linalg.h"
#include "localization.h"
// Needed for libint init
#include "eriworker.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

void size_distribution(const BasisSet & basis, arma::cx_mat & C, std::string filename, const size_t startidx, const size_t stopidx) {
  // Get the r_i^2 r_j^2 matrices
  std::vector<arma::mat> momstack=basis.moment(4);
  // Helper
  std::complex<double> one(1.0,0.0);

  // Diagonal: x^4 + y^4 + z^4
  arma::cx_mat rfour=(momstack[getind(4,0,0)] + momstack[getind(0,4,0)] + momstack[getind(0,0,4)] \
		      // Off-diagonal: 2 x^2 y^2 + 2 x^2 z^2 + 2 y^2 z^2
		      +2.0*(momstack[getind(2,2,0)]+momstack[getind(2,0,2)]+momstack[getind(0,2,2)]))*one;

  // Get R^3 matrices
  momstack=basis.moment(3);
  std::vector<arma::cx_mat> rrsq(3);
  // x^3 + xy^2 + xz^2
  rrsq[0]=(momstack[getind(3,0,0)]+momstack[getind(1,2,0)]+momstack[getind(1,0,2)])*one;
  // x^2y + y^3 + yz^2
  rrsq[1]=(momstack[getind(2,1,0)]+momstack[getind(0,3,0)]+momstack[getind(0,1,2)])*one;
  // x^2z + y^2z + z^3
  rrsq[2]=(momstack[getind(2,0,1)]+momstack[getind(0,2,1)]+momstack[getind(0,0,3)])*one;

  // Get R^2 matrix
  momstack=basis.moment(2);
  std::vector< std::vector<arma::cx_mat> > rr(3);
  for(int ic=0;ic<3;ic++)
    rr[ic].resize(3);

  // Diagonal
  rr[0][0]=momstack[getind(2,0,0)]*one;
  rr[1][1]=momstack[getind(0,2,0)]*one;
  rr[2][2]=momstack[getind(0,0,2)]*one;

  // Off-diagonal
  rr[0][1]=momstack[getind(1,1,0)]*one;
  rr[1][0]=rr[0][1];

  rr[0][2]=momstack[getind(1,0,1)]*one;
  rr[2][0]=rr[0][2];

  rr[1][2]=momstack[getind(0,1,1)]*one;
  rr[2][1]=rr[1][2];

  // and the rsq matrix
  arma::cx_mat rsq=(rr[0][0]+rr[1][1]+rr[2][2]);

  // Get r matrices
  std::vector<arma::cx_mat> rmat(3);
  momstack=basis.moment(1);
  for(int ic=0;ic<3;ic++)
    rmat[ic]=momstack[ic]*one;

  // Output file
  FILE *out=fopen(filename.c_str(),"w");
  for(size_t i=startidx;i<stopidx+1;i++) {
    
    // r^4 term
    double rfour_t=std::real(arma::as_scalar(arma::trans(C.col(i))*rfour*C.col(i)));

    // rr^2 term
    arma::vec rrsq_t(3);
    for(int ic=0;ic<3;ic++)
      rrsq_t(ic)=std::real(arma::as_scalar(arma::trans(C.col(i))*rrsq[ic]*C.col(i)));

    // rr terms
    arma::mat rr_t(3,3);
    for(int ic=0;ic<3;ic++)
      for(int jc=0;jc<=ic;jc++) {
	rr_t(ic,jc)=std::real(arma::as_scalar(arma::trans(C.col(i))*rr[ic][jc]*C.col(i)));
	rr_t(jc,ic)=rr_t(ic,jc);
      }

    // <r^2> term
    double rsq_t=std::real(arma::as_scalar(arma::trans(C.col(i))*rsq*C.col(i)));

    // <r> terms
    arma::vec r_t(3);
    for(int ic=0;ic<3;ic++)
      r_t(ic)=std::real(arma::as_scalar(arma::trans(C.col(i))*rmat[ic]*C.col(i)));

    // Second moment is
    double SM=sqrt(rsq_t - arma::dot(r_t,r_t));
    // Fourth moment is
    double FM=sqrt(sqrt(rfour_t - 4.0*arma::dot(rrsq_t,r_t) + 2.0*rsq_t*arma::dot(r_t,r_t) + 4.0 * arma::as_scalar(arma::trans(r_t)*rr_t*r_t) - 3.0*std::pow(arma::dot(r_t,r_t),2)));

    // Print
    fprintf(out,"%i %e %e (%e, %e, %e)\n",(int) i+1,SM,FM,r_t(0),r_t(1),r_t(2));
  }
  fclose(out);
}

/// Starting points for localization
enum startingpoint {
  /// Canonical orbitals - stay real
  CANORB,
  /// Atomic natural orbitals (diagonalize atomic subblock of density matrix)
  NATORB,
  /// Cholesky orbitals
  CHOLORB,
  /// Orthogonal matrix - stay real
  ORTHMAT,
  /// Unitary matrix - go complex
  UNITMAT
};

/// Find atomic natural orbitals for starting guess
arma::cx_mat atomic_orbital_guess(const BasisSet & basis, const arma::mat & P, const arma::mat & C) {
  // Returned matrix
  arma::cx_mat W(C.n_cols,C.n_cols);
  W.zeros();

  // Compute overlap matrix
  arma::mat S=basis.overlap();

  // Atomic orbitals
  std::vector<struct eigenvector<double> > orbs;

  // Loop over atoms
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    // Get functions on center inuc
    std::vector<GaussianShell> shells=basis.get_funcs(inuc);

    // and store their indices
    std::vector<size_t> idx;
    for(size_t is=0;is<shells.size();is++) {
      size_t i0=shells[is].get_first_ind();
      for(size_t fi=0;fi<shells[is].get_Nbf();fi++)
	idx.push_back(i0+fi);
    }

    // Collect atomic density and overlap matrix
    size_t N=idx.size();
    arma::mat Pat(N,N), Sat(N,N);
    for(size_t ii=0;ii<N;ii++)
      for(size_t jj=0;jj<N;jj++) {
	Pat(ii,jj)=P(idx[ii],idx[jj]);
	Sat(ii,jj)=S(idx[ii],idx[jj]);
      }

    // Get natural orbitals
    arma::mat Cat;
    arma::vec occs;
    form_NOs(Pat,Sat,Cat,occs);

    // Store the orbitals
    for(size_t iorb=0;iorb<Cat.n_cols;iorb++) {
      struct eigenvector<double> hlp;
      // Occupation
      hlp.E=occs(iorb);
      // Orbital coefficients
      hlp.c.zeros(basis.get_Nbf());
      for(size_t ii=0;ii<idx.size();ii++)
	hlp.c(idx[ii])=Cat(ii,iorb);

      orbs.push_back(hlp);
    }
  }

  // Sort orbitals by occupation number
  std::stable_sort(orbs.begin(),orbs.end());

  // Delete extra orbitals (those with smallest occupation number)
  while(orbs.size()>C.n_cols)
    // Orbitals in increasing order of occupation number
    orbs.erase(orbs.begin());

  /*
  printf("Orbital occupations\n");
  for(size_t i=0;i<orbs.size();i++)
    printf("%4i %e\n",(int) i+1,orbs[i].E);
  */

  // Collect the coefficients
  arma::mat Cat(basis.get_Nbf(),C.n_cols);
  for(size_t i=0;i<C.n_cols;i++)
    Cat.col(i)=orbs[i].c;

  // Compute orthogonalized transformation
  return orthogonalize(arma::trans(Cat)*S*C)*std::complex<double>(1.0,0.0);
}

/// Find atomic natural orbitals for starting guess
arma::cx_mat cholesky_guess(const BasisSet & basis, const arma::mat & C) {
  Timer t;
  printf("Performing Cholesky guess ... ");
  fflush(stdout);

  // Generate density matrix
  arma::mat P(C.n_rows,C.n_rows);
  P.zeros();
  for(size_t i=0;i<C.n_cols;i++)
    P+=C.col(i)*arma::trans(C.col(i));

  // Perform Cholesky factorization of density matrix
  arma::mat Cchol=incomplete_cholesky(P,C.n_cols);

  // Overlap matrix
  arma::mat S=basis.overlap();

  // Compute transformation
  arma::cx_mat ret=orthogonalize(arma::trans(Cchol)*S*C)*std::complex<double>(1.0,0.0);

  printf("done (%s)\n",t.elapsed().c_str());

  return ret;
}


void localize_wrk(const BasisSet & basis, arma::subview<double> & C, arma::subview_col<double> & E, const arma::mat & P, const arma::mat & H, enum locmet method, enum unitmethod umet, enum unitacc acc, enum startingpoint start, bool delocalize, std::string fname, double Gthr, double Fthr, int maxiter, unsigned long int seed, bool debug) {

  // assume that all orbitals in the subview of C are included in the localization.
  size_t norbs = C.n_cols;

  arma::cx_mat Cplx(C*std::complex<double>(1.0,0.0));

  arma::mat Cwrk(C);
  // and orbital energies
  arma::vec Ewrk(norbs);
  Ewrk = E;

  // Localizing matrix
  arma::cx_mat U;
  if(start==CANORB)
    // Start with canonical orbitals
    U.eye(norbs,norbs);
  else if(start==NATORB)
    U=atomic_orbital_guess(basis,P,Cwrk);
  else if(start==CHOLORB)
    U=cholesky_guess(basis,Cwrk);
  else if(start==ORTHMAT)
    // Start with orthogonal matrix
    U=std::complex<double>(1.0,0.0)*real_orthogonal(norbs,seed);
  else if(start==UNITMAT)
    // Start with unitary matrix
    U=complex_unitary(norbs,seed);
  else {
    ERROR_INFO();
    throw std::runtime_error("Starting point not implemented!\n");
    }

    // Measure
    double measure;

    // Run localization
    orbital_localization(method,basis,Cwrk,P,measure,U,true,!(start==UNITMAT),maxiter,Gthr,Fthr,umet,acc,delocalize,fname,debug);

    if(start==UNITMAT) {
      // Update orbitals, complex case
      arma::cx_mat Cloc=Cwrk*U;
      Cplx = Cloc; // same dimension, no need for submatrix views
    } else {
      // Update orbitals and energies, real case

      // Convert U to real form
      arma::mat Ur=orthogonalize(arma::real(U));
      // Transform orbitals
      arma::mat Cloc=Cwrk*Ur;
      // and energies
      arma::vec Eloc(Cloc.n_cols);
      if(H.n_rows == Cwrk.n_rows) {
	  // We have Fock operator; use it to calculate energies (in case
	  // we don't have canonical orbitals here)
	  Eloc = arma::diagvec(arma::trans(Cloc)*H*Cloc);
      } else
	  // No Fock operator given
	  Eloc=arma::diagvec(arma::trans(Ur)*arma::diagmat(Ewrk)*Ur);

      // and sort them in the new energy order
      sort_eigvec(Eloc,Cloc);

      // Update orbitals and energies
      C=Cloc;
      E=Eloc;
  }
}


void print_localize_msg(size_t start_idx, size_t stop_idx, bool delocalize){
    if(delocalize)
      printf("Delocalizing orbitals:");
    else
      printf("Localizing orbitals:");
    for(size_t io=start_idx;io<stop_idx+1;io++)
      printf(" %i",(int) io+1);
    printf("\n");
}

void localize(const BasisSet & basis, arma::mat & C, arma::vec & E, const arma::mat & P, const arma::mat & H, std::vector<double> occs, bool virt, enum locmet method, enum unitmethod umet, enum unitacc acc, enum startingpoint start, bool delocalize, std::string sizedist, bool size, std::string fname, double Gthr, double Fthr, int maxiter, unsigned long int seed, bool debug, int ncore, int nel) {
  // Run localization, occupied core space
  if(ncore) {
    arma::subview<double> C_core = C.cols(0,ncore-1);
    arma::subview_col<double> E_core = E.subvec(0,ncore-1);

    print_localize_msg(0,ncore-1,delocalize);
    localize_wrk(basis,C_core,E_core,P,H,method,umet,acc,start,delocalize,fname+".core",Gthr,Fthr,maxiter,seed,debug);

    // Compute size distribution
    if(size) {
      arma::cx_mat Chlp=C*std::complex<double>(1.0,0.0);
      size_distribution(basis,Chlp,sizedist+".core",0,ncore-1);
    }
  }

  // Run localization, occupied valence space
  arma::subview<double> C_occ = C.cols(ncore, nel-1);
  arma::subview_col<double> E_occ = E.subvec(ncore,nel-1);

  print_localize_msg(ncore,nel-1,delocalize);
  localize_wrk(basis,C_occ,E_occ,P,H,method,umet,acc,start,delocalize,fname+".val",Gthr,Fthr,maxiter,seed,debug);

  // Compute size distribution
  if(size) {
    arma::cx_mat Chlp=C*std::complex<double>(1.0,0.0);
    size_distribution(basis,Chlp,sizedist+".val",ncore,nel-1);
  }

  // Run localization, virtual space
  if(virt) {
    arma::subview<double> C_virt = C.cols(nel,C.n_cols-1);
    arma::subview_col<double> E_virt = E.subvec(nel,C.n_cols-1);

    print_localize_msg(nel,C.n_cols-1,delocalize);
    localize_wrk(basis,C_virt,E_virt,P,H,method,umet,acc,start,delocalize,fname+".virt",Gthr,Fthr,maxiter,seed,debug);

    // Compute size distribution
    if(size) {
      arma::cx_mat Chlp=C*std::complex<double>(1.0,0.0);
      size_distribution(basis,Chlp,sizedist+".virt",nel,C.n_cols-1);
    }
  }
}


void print_header() {
#ifdef _OPENMP
  printf("ERKALE - Localization from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Localization from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

Settings settings;

int main_guarded(int argc, char **argv) {
  print_header();

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  // Initialize libint
  init_libint_base();

  settings.add_string("LoadChk","Checkpoint to load","erkale.chk");
  settings.add_string("SaveChk","Checkpoint to save results to","erkale.chk");
  settings.add_string("Method","Localization method: FB, FB2, FM, FM2, MU, MU2, LO, LO2, BA, BA2, BE, BE2, HI, HI2, IHI, IHI2, IAO, IAO2, ST, ST2, VO, VO2, ER","FB");
  settings.add_bool("Virtual","Localize virtual orbitals as well?",false);
  settings.add_string("Logfile","File to store output in","");
  settings.add_string("Accelerator","Accelerator to use: SDSA, CGPR, CGFR, CGHS","CGPR");
  settings.add_string("LineSearch","Line search to use: poly_df, poly_f, poly_fdf, armijo, fourier_df","poly_df");
  settings.add_string("StartingPoint","Starting point to use: CAN, NAT, CHOL, ORTH, UNIT?","ORTH");
  settings.add_bool("Delocalize","Run delocalization instead of localization",false);
  settings.add_string("SizeDistribution","File to save orbital size distribution in","");
  settings.add_double("GThreshold","Threshold for convergence: norm of Riemannian gradient",1e-7);
  settings.add_double("FThreshold","Threshold for convergence: absolute change in function",DBL_MAX);
  settings.add_int("Maxiter","Maximum number of iterations",50000);
  settings.add_int("Seed","Random number seed",0);
  settings.add_bool("Debug","Print out line search every iteration",false);
  settings.add_int("NumCore","Number of atomic core orbitals (localized separately from valence)",0);
  settings.parse(argv[1]);
  settings.print();

  printf("Please read and cite the reference:\n%s\n%s\n%s\n\n", \
	 "S. Lehtola and H. Jónsson",         \
	 "Unitary Optimization of Localized Molecular Orbitals", \
	 "J. Chem. Theory Comput. 9 (2013), pp. 5365 - 5372.");

  std::string logfile=settings.get_string("Logfile");
  bool virt=settings.get_bool("Virtual");
  int seed=settings.get_int("Seed");

  int ncore=settings.get_int("NumCore");

  std::string loadname(settings.get_string("LoadChk"));
  std::string savename(settings.get_string("SaveChk"));
  std::string sizedist(settings.get_string("SizeDistribution"));
  bool size=stricmp(sizedist,"");
  bool debug=settings.get_bool("Debug");

  // Determine method
  enum locmet method(parse_locmet(settings.get_string("Method")));

  if(method>=PIPEK_MULLIKENH && method<=PIPEK_VORONOI4) {
    printf("Please read and cite the reference:\n%s\n%s\n%s\n\n",	\
	   "S. Lehtola and H. Jónsson",					\
	   "Pipek–Mezey Orbital Localization Using Various Partial Charge Estimates",	\
	   "J. Chem. Theory Comput. 10 (2014), pp. 642 - 649.");
  }

  // Determine accelerator
  enum unitacc acc;
  std::string accs=settings.get_string("Accelerator");
  if(stricmp(accs,"SDSA")==0)
    acc=SDSA;
  else if(stricmp(accs,"CGPR")==0)
    acc=CGPR;
  else if(stricmp(accs,"CGFR")==0)
    acc=CGFR;
  else if(stricmp(accs,"CGHS")==0)
    acc=CGHS;
  else throw std::runtime_error("Accelerator not implemented.\n");

  // Determine line search
  enum unitmethod umet;
  std::string umets=settings.get_string("LineSearch");
  if(stricmp(umets,"poly_df")==0)
    umet=POLY_DF;
  else if(stricmp(umets,"poly_f")==0)
    umet=POLY_F;
  else if(stricmp(umets,"armijo")==0)
    umet=ARMIJO;
  else if(stricmp(umets,"fourier_df")==0)
    umet=FOURIER_DF;
  else throw std::runtime_error("Accelerator not implemented.\n");

  if(stricmp(loadname,savename)!=0) {
    // Copy checkpoint
    std::ostringstream oss;
    oss << "cp " << loadname << " " << savename;
    int cp=system(oss.str().c_str());
    if(cp) {
      ERROR_INFO();
      throw std::runtime_error("Failed to copy checkpoint file.\n");
    }
  }

  // Delocalize orbitals?
  bool delocalize=settings.get_bool("Delocalize");
  // Iteration count
  int maxiter=settings.get_int("Maxiter");

  // Starting point
  std::string startp=settings.get_string("StartingPoint");
  enum startingpoint start;
  if(stricmp(startp,"CAN")==0)
    start=CANORB;
  else if(stricmp(startp,"NAT")==0)
    start=NATORB;
  else if(stricmp(startp,"CHOL")==0)
    start=CHOLORB;
  else if(stricmp(startp,"ORTH")==0)
    start=ORTHMAT;
  else if(stricmp(startp,"UNIT")==0)
    start=UNITMAT;
  else {
    ERROR_INFO();
    throw std::runtime_error("Starting point not implemented!\n");
  }

  // Convergence threshold
  double Gthr=settings.get_double("GThreshold");
  double Fthr=settings.get_double("FThreshold");

  // Open checkpoint in read-write mode, don't truncate
  Checkpoint chkpt(savename,true,false);

  // Basis set
  BasisSet basis;
  chkpt.read(basis);

  // Restricted run?
  bool restr;
  chkpt.read("Restricted",restr);

  // Total density
  arma::mat P;
  chkpt.read("P",P);

  if(restr) {
    // Orbitals
    arma::mat C;
    chkpt.read("C",C);
    // and energies
    arma::vec E;
    chkpt.read("E",E);

    // Fock matrix?
    arma::mat H;
    if(chkpt.exist("H"))
      chkpt.read("H",H);

    // Check orthogonality
    check_orth(C,basis.overlap(),false);

    // Occupation numbers
    std::vector<double> occs;
    chkpt.read("occs",occs);

    // Electron Number
    int nela;
    chkpt.read("Nel-a",nela);

    if (ncore > nela)
      throw std::runtime_error("NumCore must not exceed the number of electrons\n");

    // Run localization
    localize(basis,C,E,P,H,occs,virt,method,umet,acc,start,delocalize,sizedist,size,logfile,Gthr,Fthr,maxiter,seed,debug,ncore,nela);

    chkpt.write("C",C);
    chkpt.write("E",E);

  } else {
    // Orbitals
    arma::mat Ca, Cb;
    chkpt.read("Ca",Ca);
    chkpt.read("Cb",Cb);
    // and energies
    arma::vec Ea, Eb;
    chkpt.read("Ea",Ea);
    chkpt.read("Eb",Eb);

    // Fock matrices?
    arma::mat Ha, Hb;
    if(chkpt.exist("Ha") && chkpt.exist("Hb")) {
      chkpt.read("Ha",Ha);
      chkpt.read("Hb",Hb);
    }

    // Check orthogonality
    check_orth(Ca,basis.overlap(),false);
    check_orth(Cb,basis.overlap(),false);

    // Occupation numbers
    std::vector<double> occa, occb;
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);

    // Electron Number
    int nela,nelb;
    chkpt.read("Nel-a",nela);
    chkpt.read("Nel-b",nelb);
    
    if (ncore > nelb)
      throw std::runtime_error("NumCore must not exceed the number of electrons\n");

    // Run localization
    localize(basis,Ca,Ea,P,Ha,occa,virt,method,umet,acc,start,delocalize,sizedist+".a",size,logfile+".a",Gthr,Fthr,maxiter,seed,debug,ncore,nela);
    localize(basis,Cb,Eb,P,Hb,occb,virt,method,umet,acc,start,delocalize,sizedist+".b",size,logfile+".b",Gthr,Fthr,maxiter,seed,debug,ncore,nelb);

    chkpt.write("Ca",Ca);
    chkpt.write("Cb",Cb);
    chkpt.write("Ea",Ea);
    chkpt.write("Eb",Eb);
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
