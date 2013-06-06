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



#include <armadillo>
#include <cstdio>
#include <cfloat>

#include "adiis.h"
#include "basis.h"
#include "broyden.h"
#include "elements.h"
#include "dftfuncs.h"
#include "dftgrid.h"
#include "diis.h"
#include "global.h"
#include "guess.h"
#include "linalg.h"
#include "mathf.h"
#include "properties.h"
#include "scf.h"
#include "stringutil.h"
#include "timer.h"
#include "trdsm.h"
#include "trrh.h"

extern "C" {
#include <gsl/gsl_poly.h>
}


enum guess_t parse_guess(const std::string & val) {
  if(stricmp(val,"Core")==0)
    return COREGUESS;
  else if(stricmp(val,"Atomic")==0)
    return ATOMGUESS;
  else if(stricmp(val,"Molecular")==0)
    return MOLGUESS;
  else
    throw std::runtime_error("Guess type not supported.\n");
}

SCF::SCF(const BasisSet & basis, const Settings & set, Checkpoint & chkpt) {
  // Amount of basis functions
  Nbf=basis.get_Nbf();

  basisp=&basis;
  chkptp=&chkpt;

  // Multiplicity
  mult=set.get_int("Multiplicity");

  // Amount of electrons
  Nel=basis.Ztot()-set.get_int("Charge");

  // Parse guess
  guess=parse_guess(set.get_string("Guess"));

  usediis=set.get_bool("UseDIIS");
  diis_c1=set.get_bool("C1-DIIS");
  diisorder=set.get_int("DIISOrder");
  diisthr=set.get_double("DIISThr");
  useadiis=set.get_bool("UseADIIS");
  usebroyden=set.get_bool("UseBroyden");
  usetrrh=set.get_bool("UseTRRH");
  usetrdsm=set.get_bool("UseTRDSM");
  linesearch=set.get_bool("LineSearch");

  maxiter=set.get_int("MaxIter");
  verbose=set.get_bool("Verbose");

  direct=set.get_bool("Direct");
  strictint=set.get_bool("StrictIntegrals");

  // Perdew-Zunger SIC?
  pzcor=set.get_double("PZ-SICw");
  std::string pzs=set.get_string("PZ-SIC");
  if(stricmp(pzs,"Yes")==0)
    pz=YES;
  else if(stricmp(pzs,"Pert")==0)
    pz=PERT;
  else
    pz=NO;  

  doforce=false;

  // Check update scheme
  if(useadiis && usebroyden) {
    ERROR_INFO();
    throw std::runtime_error("ADIIS and Broyden mixing cannot be used at the same time.\n");
  }

  if(!usediis && !useadiis && !usebroyden && !usetrrh && !usetrdsm && !linesearch) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to run calculation without an update scheme.\n");
  }

  if(usetrdsm && (useadiis || usediis || usebroyden || !usetrrh)) {
    ERROR_INFO();
    throw std::runtime_error("Use of TRDSM requires use of TRRH and turning off (AE)DIIS and Broyden.\n");
  }

  // Nuclear repulsion
  Enuc=basis.Enuc();

  // Use density fitting?
  densityfit=set.get_bool("DensityFitting");
  // How much memory to allow (convert to bytes)
  fitmem=1000000*set.get_int("FittingMemory");
  // Linear dependence threshold
  fitthr=set.get_double("FittingThreshold");

  try {
    // Use Lobatto angular grid? (Lebedev is default)
    dft_lobatto=set.get_bool("DFTLobatto");
  } catch(...) {
    // Hartree-Fock doesn't have the settings
  }

  // Timer
  Timer t;
  Timer tinit;

  if(verbose) {
    basis.print();

    printf("\nForming overlap matrix ... ");
    fflush(stdout);
    t.set();
  }

  S=basis.overlap();

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());

    printf("Forming kinetic energy matrix ... ");
    fflush(stdout);
    t.set();
  }

  T=basis.kinetic();

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());

    printf("Forming nuclear attraction matrix ... ");
    fflush(stdout);
    t.set();
  }

  Vnuc=basis.nuclear();

  if(verbose)
    printf("done (%s)\n",t.elapsed().c_str());

  // Form core Hamiltonian
  Hcore=T+Vnuc;

  if(verbose) {
    printf("\n");
    t.set();
  }

  Sinvh=BasOrth(S,set);

  if(verbose) {
    printf("Basis set diagonalized in %s.\n",t.elapsed().c_str());
    t.set();

    if(Sinvh.n_cols!=Sinvh.n_rows) {
      printf("%i linear combinations of basis functions have been removed.\n",Sinvh.n_rows-Sinvh.n_cols);
    }
    printf("\n");
  }

  if(densityfit) {
    // Form density fitting basis

    // Do we need RI-K, or is RI-J sufficient?
    bool rik=false;
    if(stricmp(set.get_string("Method"),"HF")==0)
      rik=true;
    else if(stricmp(set.get_string("Method"),"ROHF")==0)
      rik=true;
    else {
      // No hartree-fock; check if functional has exact exchange part
      int xfunc, cfunc;
      parse_xc_func(xfunc,cfunc,set.get_string("Method"));
      if(exact_exchange(xfunc)!=0.0)
	rik=true;
    }

    if(stricmp(set.get_string("FittingBasis"),"Auto")==0) {
      // Check used method
      if(rik)
	throw std::runtime_error("Automatical auxiliary basis set formation not implemented for exact exchange.\nSet the FittingBasis.\n");

      // DFT, OK for now (will be checked again later on)
      dfitbas=basisp->density_fitting();
    } else {
      // Load basis library
      BasisSetLibrary fitlib;
      fitlib.load_gaussian94(set.get_string("FittingBasis"));

      // Construct fitting basis
      dfitbas=construct_basis(basisp->get_nuclei(),fitlib,set);
    }

    // Compute memory estimate
    std::string memest=memory_size(dfit.memory_estimate(*basisp,dfitbas,direct));

    if(verbose) {
      if(direct)
	printf("Initializing density fitting calculation, requiring %s memory ... ",memest.c_str());
      else
	printf("Computing density fitting integrals, requiring %s memory ... ",memest.c_str());
      fflush(stdout);
      t.set();
    }

    // Fill the basis
    dfit.fill(*basisp,dfitbas,direct,fitthr,rik);

    if(verbose) {
      printf("done (%s)\n",t.elapsed().c_str());
      printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
      fflush(stdout);
    }
  } else  {
    // Compute ERIs
    if(direct) {

      // Form decontracted basis set and get the screening matrix
      decbas=basis.decontract(decconv);

      if(verbose) {
	t.set();
	printf("Forming ERI screening matrix ... ");
	fflush(stdout);
      }
      //      scr.fill(&decbas);
      scr.fill(&basis);
    } else {
      // Compute memory requirement
      size_t N;

      if(verbose) {
	N=tab.memory_estimate(&basis);
	printf("Forming table of %lu ERIs, requiring %s of memory ... ",(long unsigned int) N,memory_size(N).c_str());
	fflush(stdout);
      }
      // Don't compute small integrals
      tab.fill(&basis,STRICTTOL);
    }

    if(verbose)
      printf("done (%s)\n",t.elapsed().c_str());
  }

  if(verbose) {
    printf("\nInitialization of computation done in %s.\n\n",tinit.elapsed().c_str());
    fflush(stdout);
  }
}

SCF::~SCF() {
}

void SCF::set_frozen(const arma::mat & C, size_t ind) {
  // Check size of array
  while(ind+1>freeze.size()) {
    arma::mat tmp;
    freeze.push_back(tmp);
  }
  // Store frozen core orbitals
  freeze[ind]=C;
}

void SCF::set_fitting(const BasisSet & fitbasv) {
  dfitbas=fitbasv;
}

void diagonalize(const arma::mat & S, const arma::mat & Sinvh, rscf_t & sol) {
  arma::mat Horth;
  arma::mat orbs;
  // Transform Hamiltonian into orthogonal basis
  Horth=arma::trans(Sinvh)*sol.H*Sinvh;
  // Update orbitals and energies
  eig_sym_ordered(sol.E,orbs,Horth);
  // Transform back to non-orthogonal basis
  sol.C=Sinvh*orbs;

  // Check orthonormality
  check_orth(sol.C,S,false);
}

void diagonalize(const arma::mat & S, const arma::mat & Sinvh, uscf_t & sol) {
  arma::mat Horth;
  arma::mat orbs;

  Horth=trans(Sinvh)*sol.Ha*Sinvh;
  eig_sym_ordered(sol.Ea,orbs,Horth);
  sol.Ca=Sinvh*orbs;
  check_orth(sol.Ca,S,false);

  Horth=trans(Sinvh)*sol.Hb*Sinvh;
  eig_sym_ordered(sol.Eb,orbs,Horth);
  sol.Cb=Sinvh*orbs;
  check_orth(sol.Cb,S,false);
}

void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::mat & NO_to_AO, arma::vec & occs) {

  // First, get eigenvectors and eigenvalues of S so that we can go to
  // an orthonormal basis.
  arma::vec Sval;
  arma::mat Svec;
  eig_sym_ordered(Sval,Svec,S);

  // Count the number of linearly independent vectors
  size_t Nind=0;
  for(size_t i=0;i<Sval.n_elem;i++)
    if(Sval[i]>LINTHRES)
      Nind++;
  // ... and get rid of the linearly dependent ones. The eigenvalues
  // and vectors are in the order of increasing eigenvalue, so we want
  // the tail.
  Sval=Sval.subvec(Sval.n_elem-Nind,Sval.n_elem-1);
  Svec=Svec.submat(0,Svec.n_cols-Nind,Svec.n_rows-1,Svec.n_cols-1);

  /* Transformation to get matrix M in orthonormal basis is
     M_ij = <i|M_AO|j> \sqrt{l_i l_j},
     where l_i and l_j are the corresponding eigenvalues of
     eigenvectors i and j.

     This can be written as
     M_ij = Sm(i,k) M(k,l) Sm(j,l)
  */

  // Form scaled vectors
  arma::mat Sm(Svec);
  arma::mat Sd(Svec);
  for(size_t i=0;i<Sval.n_elem;i++) {
    double ss=sqrt(Sval(i));
    Sm.col(i)*=ss;
    Sd.col(i)/=ss;
  }

  // P in orthonormal basis is
  arma::mat P_orth=arma::trans(Sm)*P*Sm;

  // Diagonalize P to get NOs in orthonormal basis.
  arma::mat Pvec;
  eig_sym_ordered(occs,Pvec,P_orth);

  /* Get NOs in AO basis. The natural orbital is written in the
     orthonormal basis as

     |i> = x_ai |a> = x_ai s_ja |j>
     = s_ja x_ai |j>
  */

  // The matrix that takes us from AO to NO is
  AO_to_NO=Sd*Pvec;
  // and the one that takes us from NO to AO is
  NO_to_AO=arma::trans(Sm*Pvec);
}

void ROHF_update(arma::mat & Fa_AO, arma::mat & Fb_AO, const arma::mat & P_AO, const arma::mat & S, int Nel_alpha, int Nel_beta, bool verbose, bool atomic) {
  /*
   * T. Tsuchimochi and G. E. Scuseria, "Constrained active space
   * unrestricted mean-field methods for controlling
   * spin-contamination", J. Chem. Phys. 134, 064101 (2011).
   */

  Timer t;

  arma::vec occs;
  arma::mat AO_to_NO;
  arma::mat NO_to_AO;
  form_NOs(P_AO,S,AO_to_NO,NO_to_AO,occs);

  // Construct \Delta matrix in AO basis
  arma::mat Delta_AO=(Fa_AO-Fb_AO)/2.0;

  // and take it to the NO basis.
  arma::mat Delta_NO=arma::trans(AO_to_NO)*Delta_AO*AO_to_NO;

  // Amount of independent orbitals is
  size_t Nind=AO_to_NO.n_cols;
  // Amount of core orbitals is
  size_t Nc=std::min(Nel_alpha,Nel_beta);
  // Amount of active space orbitals is
  size_t Na=std::max(Nel_alpha,Nel_beta)-Nc;
  // Amount of virtual orbitals (in NO space) is
  size_t Nv=Nind-Na-Nc;

  if(atomic) {
    // Get atomic occupations
    std::vector<double> occa=atomic_occupancy(Nel_alpha);
    std::vector<double> occb=atomic_occupancy(Nel_beta);
    // and update the values
    Nc=std::min(occa.size(),occb.size());
    Na=std::max(occa.size(),occb.size())-Nc;
    Nv=Nind-Na-Nc;
  }

  /*
    double tot=0.0;
    printf("Core orbital occupations:");
    for(size_t c=Nind-1;c>=Nind-Nc && c<Nind;c--) {
    printf(" %f",occs(c));
    tot+=occs(c);
    }
    printf("\n");

    printf("Active orbital occupations:");
    for(size_t a=Nind-Nc-1;a>=Nind-Nc-Na && a<Nind;a--) {
    printf(" %f",occs(a));
    tot+=occs(a);
    }
    printf("\n");
    printf("Total occupancy of core and active is %f.\n",tot);
  */

  // Form lambda by flipping the signs of the cv and vc blocks and
  // zeroing out everything else.
  arma::mat lambda_NO(Delta_NO);
  /*
    eig_sym_ordered puts the NOs in the order of increasing
    occupation. Thus, the lowest Nv orbitals belong to the virtual
    space, the following Na to the active space and the last Nc to the
    core orbitals.
  */
  // Zero everything
  lambda_NO.zeros();
  // and flip signs of cv and vc blocks from Delta
  for(size_t v=0;v<Nv;v++) // Loop over virtuals
    for(size_t c=Nind-Nc;c<Nind;c++) { // Loop over core orbitals
      lambda_NO(c,v)=-Delta_NO(c,v);
      lambda_NO(v,c)=-Delta_NO(v,c);
    }

  // Lambda in AO is
  arma::mat lambda_AO=arma::trans(NO_to_AO)*lambda_NO*NO_to_AO;

  // Update Fa and Fb
  Fa_AO+=lambda_AO;
  Fb_AO-=lambda_AO;

  if(verbose)
    printf("Performed CUHF update of Fock operators in %s.\n",t.elapsed().c_str());
}

void determine_occ(arma::vec & nocc, const arma::mat & C, const arma::vec & nocc_old, const arma::mat & C_old, const arma::mat & S) {
  nocc.zeros();

  // Loop over states
  for(size_t i=0;i<nocc_old.n_elem;i++)
    if(nocc_old[i]!=0.0) {

      arma::vec hlp=S*C_old.col(i);

      // Determine which state is the closest to the old one
      size_t loc=0;
      double Smax=0.0;

      for(size_t j=0;j<C.n_cols;j++) {
	double ovl=arma::dot(C.col(j),hlp);
	if(fabs(ovl)>Smax) {
	  Smax=fabs(ovl);
	  loc=j;
	}
      }

      // Copy occupancy
      if(nocc[loc]!=0.0)
	printf("Problem in determine_occ: state %i was already occupied by %g electrons!\n",(int) loc,nocc[loc]);
      nocc[loc]+=nocc_old[i];
    }
}

arma::mat form_density(const arma::mat & C, size_t nocc) {
  std::vector<double> occs(nocc,1.0);
  return form_density(C,occs);
}

arma::mat form_density(const arma::mat & C, const std::vector<double> & nocc) {
  if(nocc.size()>C.n_cols) {
    std::ostringstream oss;
    oss << "Error in function " << __FUNCTION__ << " (file " << __FILE__ << ", near line " << __LINE__ << "): there should be " << nocc.size() << " occupied orbitals but only " << C.n_cols << " orbitals exist!\n";
    throw std::runtime_error(oss.str());
  }

  // Zero matrix
  arma::mat P(C.n_rows,C.n_rows);
  P.zeros();
  // Formulate density
  for(size_t n=0;n<nocc.size();n++)
    if(nocc[n]>0.0)
      P+=nocc[n]*C.col(n)*arma::trans(C.col(n));

  return P;
}

arma::mat form_density(const arma::vec & E, const arma::mat & C, const std::vector<double> & nocc) {
  if(nocc.size()>C.n_cols) {
    std::ostringstream oss;
    oss << "Error in function " << __FUNCTION__ << " (file " << __FILE__ << ", near line " << __LINE__ << "): there should be " << nocc.size() << " occupied orbitals but only " << C.n_cols << " orbitals exist!\n";
    throw std::runtime_error(oss.str());
  }

  if(E.n_elem != C.n_cols) {
    std::ostringstream oss;
    oss << "Error in function " << __FUNCTION__ << " (file " << __FILE__ << ", near line " << __LINE__ << "): " << E.size() << " energies but " << C.n_cols << " orbitals!\n";
    throw std::runtime_error(oss.str());
  }

  // Zero matrix
  arma::mat W(C.n_rows,C.n_rows);
  W.zeros();
  // Formulate density
  for(size_t n=0;n<nocc.size();n++)
    if(nocc[n]>0.0)
      W+=nocc[n]*E(n)*C.col(n)*arma::trans(C.col(n));

  return W;
}

arma::mat purify_density(const arma::mat & P, const arma::mat & S) {
  // McWeeny purification
  arma::mat PS=P*S;
  return 3.0*PS*P - 2.0*PS*PS*P;
}

arma::mat purify_density_NO(const arma::mat & P, const arma::mat & S) {
  arma::mat C;
  return purify_density_NO(P,C,S);
}

arma::mat purify_density_NO(const arma::mat & P, arma::mat & C, const arma::mat & S) {
  // Number of electrons
  int Nel=(int) round(arma::trace(P*S));

  // Get the natural orbitals
  arma::mat NO;
  arma::mat tmp;
  arma::vec occs;
  form_NOs(P,S,NO,tmp,occs);

  // Store the NOs in inverted order (highest occupation first)
  C.zeros(NO.n_rows,NO.n_cols);
  for(size_t icol=0;icol<NO.n_cols;icol++)
    C.col(icol)=NO.col(NO.n_cols-1-icol);

  // and form the density
  return form_density(C,Nel);
}

std::vector<double> atomic_occupancy(int Nel) {
  std::vector<double> ret;

  // Atomic case. Fill 1s
  if(Nel>0) {
    ret.push_back(1.0);
    Nel--;
  }

  // Fill 2s
  if(Nel>0) {
    ret.push_back(1.0);
    Nel--;
  }

  // Fill 2p
  if(Nel>0) {
    // Amount of electrons to put in shell
    int n=std::min(3,Nel);
    // Divide equally
    for(int i=0;i<3;i++)
      ret.push_back(n/3.0);
    Nel-=n;
  }

  // Fill 3s
  if(Nel>0) {
    ret.push_back(1.0);
    Nel--;
  }

  // Fill 3p
  if(Nel>0) {
    // Amount of electrons to put in shell
    int n=std::min(3,Nel);
    // Divide equally
    for(int i=0;i<3;i++)
      ret.push_back(n/3.0);
    Nel-=n;
  }

  // Fill 4s
  if(Nel>0) {
    ret.push_back(1.0);
    Nel--;
  }

  // Fill 3d
  if(Nel>0) {
    // Amount of electrons to put in shell
    int n=std::min(5,Nel);
    // Divide equally
    for(int i=0;i<5;i++)
      ret.push_back(n/5.0);
    Nel-=n;
  }

  // Fill 4p
  if(Nel>0) {
    // Amount of electrons to put in shell
    int n=std::min(3,Nel);
    // Divide equally
    for(int i=0;i<3;i++)
      ret.push_back(n/3.0);
    Nel-=n;
  }

  // Fill 5s
  if(Nel>0) {
    ret.push_back(1.0);
    Nel--;
  }

  // Fill 4d
  if(Nel>0) {
    // Amount of electrons to put in shell
    int n=std::min(5,Nel);
    // Divide equally
    for(int i=0;i<5;i++)
      ret.push_back(n/5.0);
    Nel-=n;
  }

  // Fill 5p
  if(Nel>0) {
    // Amount of electrons to put in shell
    int n=std::min(3,Nel);
    // Divide equally
    for(int i=0;i<3;i++)
      ret.push_back(n/3.0);
    Nel-=n;
  }

  return ret;
}

std::vector<double> get_restricted_occupancy(const Settings & set, const BasisSet & basis) {
  // Returned value
  std::vector<double> ret;

  // Occupancies
  std::string occs=set.get_string("Occupancies");

  // Parse occupancies
  if(occs.size()) {
    // Split input
    std::vector<std::string> occvals=splitline(occs);
    // Resize output
    ret.resize(occvals.size());
    // Parse occupancies
    for(size_t i=0;i<occvals.size();i++)
      ret[i]=readdouble(occvals[i]);

    printf("Occupancies\n");
    for(size_t i=0;i<ret.size();i++)
      printf("%.2f ",ret[i]);
    printf("\n");
  } else {
    // Aufbau principle.
    int Nel=basis.Ztot()-set.get_int("Charge");
    if(Nel%2!=0) {
      throw std::runtime_error("Refusing to run restricted calculation on unrestricted system!\n");
    }

    if(basis.get_Nnuc()==1) {
      // Atomic case.
      ret=atomic_occupancy(Nel/2);
      // Orbitals are doubly occupied
      for(size_t i=0;i<ret.size();i++)
	ret[i]*=2.0;
    } else {
      // Resize output
      ret.resize(Nel/2);
      for(size_t i=0;i<ret.size();i++)
	ret[i]=2.0; // All orbitals doubly occupied
    }
  }

  return ret;
}

void get_unrestricted_occupancy(const Settings & set, const BasisSet & basis, std::vector<double> & occa, std::vector<double> & occb) {
  // Occupancies
  std::string occs=set.get_string("Occupancies");

  // Parse occupancies
  if(occs.size()) {
    // Split input
    std::vector<std::string> occvals=splitline(occs);
    if(occvals.size()%2!=0) {
      throw std::runtime_error("Error - specify both alpha and beta occupancies for all states!\n");
    }

    // Resize output vectors
    occa.resize(occvals.size()/2);
    occb.resize(occvals.size()/2);
    // Parse occupancies
    for(size_t i=0;i<occvals.size()/2;i++) {
      occa[i]=readdouble(occvals[2*i]);
      occb[i]=readdouble(occvals[2*i+1]);
    }

    printf("Occupancies\n");
    printf("alpha\t");
    for(size_t i=0;i<occa.size();i++)
      printf("%.2f ",occa[i]);
    printf("\nbeta\t");
    for(size_t i=0;i<occb.size();i++)
      printf("%.2f ",occb[i]);
    printf("\n");
  } else {
    // Aufbau principle. Get amount of alpha and beta electrons.

    int Nel_alpha, Nel_beta;
    get_Nel_alpha_beta(basis.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);

    if(basis.get_Nnuc()==1) {
      // Atomic case
      occa=atomic_occupancy(Nel_alpha);
      occb=atomic_occupancy(Nel_beta);
    } else {
      // Resize output
      occa.resize(Nel_alpha);
      for(size_t i=0;i<occa.size();i++)
	occa[i]=1.0;

      occb.resize(Nel_beta);
      for(size_t i=0;i<occb.size();i++)
	occb[i]=1.0;
    }
  }
}

double dip_mom(const arma::mat & P, const BasisSet & basis) {
  // Compute magnitude of dipole moment

  arma::vec dp=dipole_moment(P,basis);
  return arma::norm(dp,2);
}

arma::vec dipole_moment(const arma::mat & P, const BasisSet & basis) {
  // Get moment matrix
  std::vector<arma::mat> mommat=basis.moment(1);

  // Electronic part
  arma::vec el(3);
  // Compute dipole moments
  for(int i=0;i<3;i++) {
    // Electrons have negative charge
    el[i]=arma::trace(-P*mommat[i]);
  }

  //  printf("Electronic dipole moment is %e %e %e.\n",el(0),el(1),el(2));

  // Compute center of nuclear charge
  arma::vec nc(3);
  nc.zeros();
  for(size_t i=0;i<basis.get_Nnuc();i++) {
    // Get nucleus
    nucleus_t nuc=basis.get_nucleus(i);
    // Increment
    nc(0)+=nuc.Z*nuc.r.x;
    nc(1)+=nuc.Z*nuc.r.y;
    nc(2)+=nuc.Z*nuc.r.z;
  }
  //  printf("Nuclear dipole moment is %e %e %e.\n",nc(0),nc(1),nc(2));

  arma::vec ret=el+nc;

  return ret;
}

double electron_spread(const arma::mat & P, const BasisSet & basis) {
  // Compute <r^2> of density

  // Get number of electrons.
  std::vector<arma::mat> mom0=basis.moment(0);
  double Nel=arma::trace(P*mom0[0]);

  // Normalize P
  arma::mat Pnorm=P/Nel;

  // First, get <r>.
  std::vector<arma::mat> mom1=basis.moment(1);
  arma::vec r(3);
  r(0)=arma::trace(Pnorm*mom1[getind(1,0,0)]);
  r(1)=arma::trace(Pnorm*mom1[getind(0,1,0)]);
  r(2)=arma::trace(Pnorm*mom1[getind(0,0,1)]);

  //  printf("Center of electron cloud is at %e %e %e.\n",r(0),r(1),r(2));

  // Then, get <r^2> around r
  std::vector<arma::mat> mom2=basis.moment(2,r(0),r(1),r(2));
  double r2=arma::trace(Pnorm*(mom2[getind(2,0,0)]+mom2[getind(0,2,0)]+mom2[getind(0,0,2)]));

  double dr=sqrt(r2);

  return dr;
}

void get_Nel_alpha_beta(int Nel, int mult, int & Nel_alpha, int & Nel_beta) {
  // Check sanity of arguments
  if(mult<1)
    throw std::runtime_error("Invalid value for multiplicity, which must be >=1.\n");
  else if(Nel%2==0 && mult%2!=1) {
    std::ostringstream oss;
    oss << "Requested multiplicity " << mult << " with " << Nel << " electrons.\n";
    throw std::runtime_error(oss.str());
  } else if(Nel%2==1 && mult%2!=0) {
    std::ostringstream oss;
    oss << "Requested multiplicity " << mult << " with " << Nel << " electrons.\n";
    throw std::runtime_error(oss.str());
  }

  if(Nel%2==0)
    // Even number of electrons, the amount of spin up is
    Nel_alpha=Nel/2+(mult-1)/2;
  else
    // Odd number of electrons, the amount of spin up is
    Nel_alpha=Nel/2+mult/2;

  // The rest are spin down
  Nel_beta=Nel-Nel_alpha;
}

void calculate(const BasisSet & basis, Settings & set, bool force) {
  // Checkpoint files to load and save
  std::string loadname=set.get_string("LoadChk");
  std::string savename=set.get_string("SaveChk");

  bool verbose=set.get_bool("Verbose");

  // Print out settings
  if(verbose)
    set.print();

  // Number of electrons is
  int Nel=basis.Ztot()-set.get_int("Charge");

  // Do a plain Hartree-Fock calculation?
  bool hf= (stricmp(set.get_string("Method"),"HF")==0);
  bool rohf=(stricmp(set.get_string("Method"),"ROHF")==0);

  // Final convergence settings
  convergence_t conv;
  conv.deltaEmax=set.get_double("DeltaEmax");
  conv.deltaPmax=set.get_double("DeltaPmax");
  conv.deltaPrms=set.get_double("DeltaPrms");

  // Get exchange and correlation functionals
  dft_t dft;
  dft_t initdft;
  // Initial convergence settings
  convergence_t initconv(conv);

  if(!hf && !rohf) {
    parse_xc_func(dft.x_func,dft.c_func,set.get_string("Method"));
    dft.gridtol=0.0;
    // Use static grid?
    if(stricmp(set.get_string("DFTGrid"),"Auto")!=0) {
      std::vector<std::string> opts=splitline(set.get_string("DFTGrid"));
      if(opts.size()!=2) {
	throw std::runtime_error("Invalid DFT grid specified.\n");
      }

      dft.adaptive=false;
      dft.nrad=readint(opts[0]);
      dft.lmax=readint(opts[1]);
      if(dft.nrad<1 || dft.lmax<1) {
	throw std::runtime_error("Invalid DFT grid specified.\n");
      }
      printf("dft.nrad = %i, dft.lmax = %i\n",dft.nrad,dft.lmax);

    } else {
      dft.adaptive=true;
      dft.gridtol=set.get_double("DFTFinalTol");
    }

    initdft=dft;
    if(dft.adaptive)
      initdft.gridtol=set.get_double("DFTInitialTol");

    initconv.deltaEmax*=set.get_double("DFTDelta");
    initconv.deltaPmax*=set.get_double("DFTDelta");
    initconv.deltaPrms*=set.get_double("DFTDelta");
  }

  // Check consistency of parameters
  if(!hf && !rohf && exact_exchange(dft.x_func)!=0.0)
    if(set.get_bool("DensityFitting") && (stricmp(set.get_string("FittingBasis"),"Auto")==0)) {
      throw std::runtime_error("Automatical auxiliary basis set formation not implemented for exact exchange.\nChange the FittingBasis.\n");
    }

  // Load starting guess?
  bool doload=(stricmp(loadname,"")!=0);
  BasisSet oldbas;
  bool oldrestr;
  arma::vec Eold, Eaold, Ebold;
  arma::mat Cold, Caold, Cbold;
  arma::mat Pold;

  // Which guess to use
  enum guess_t guess=parse_guess(set.get_string("Guess"));
  // Freeze core orbitals?
  bool freezecore=set.get_bool("FreezeCore");
  if(freezecore && guess==COREGUESS)
    throw std::runtime_error("Cannot freeze core orbitals with core guess!\n");

  if(doload) {
    Checkpoint load(loadname,false);

    // Basis set
    load.read(oldbas);

    // Restricted calculation?
    load.read("Restricted",oldrestr);

    // Density matrix
    load.read("P",Pold);

    if(oldrestr) {
      // Load energies and orbitals
      load.read("C",Cold);
      load.read("E",Eold);
    } else {
      // Load energies and orbitals
      load.read("Ca",Caold);
      load.read("Ea",Eaold);
      load.read("Cb",Cbold);
      load.read("Eb",Ebold);
    }
  }

  if(set.get_int("Multiplicity")==1 && Nel%2==0 && !set.get_bool("ForcePol")) {
    // Closed shell case
    rscf_t sol;

    // Project old solution to new basis
    if(doload) {
      // Restricted calculation wanted but loaded spin-polarized one
      if(!oldrestr) {
	// Find out natural orbitals
	arma::mat hlp;
	arma::vec occs;
	form_NOs(Pold,oldbas.overlap(),Cold,hlp,occs);

	// Use alpha orbital energies
	Eold=Eaold;
      }

      basis.projectMOs(oldbas,Eold,Cold,sol.E,sol.C);
    } else if(guess == ATOMGUESS) {
      atomic_guess(basis,sol.C,sol.E,set);
    } else if(guess == MOLGUESS) {
      // Need to generate the starting guess.
      std::string name;
      molecular_guess(basis,set,name);

      // Load guess orbitals
      {
	Checkpoint guesschk(name,false);
	guesschk.read("C",sol.C);
	guesschk.read("E",sol.E);
      }
      // and remove the temporary file
      remove(name.c_str());
    }

    // Get orbital occupancies
    std::vector<double> occs=get_restricted_occupancy(set,basis);

    // Write checkpoint.
    Checkpoint chkpt(savename,true);
    chkpt.write(basis);

    // Write number of electrons
    int Nel_alpha;
    int Nel_beta;
    get_Nel_alpha_beta(basis.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);
    chkpt.write("Nel",Nel);
    chkpt.write("Nel-a",Nel_alpha);
    chkpt.write("Nel-b",Nel_beta);

    // Write method
    chkpt.write("Method",set.get_string("Method"));

    // Solver
    SCF solver(basis,set,chkpt);

    // Freeze core orbitals?
    if(freezecore) {
      // Localize the core orbitals within the occupied space
      size_t nloc=localize_core(basis,std::max(Nel_alpha,Nel_beta),sol.C,verbose);
      // and freeze them
      solver.set_frozen(sol.C.submat(0,0,sol.C.n_rows-1,nloc-1),0);
    }

    if(hf || rohf) {
      // Solve restricted Hartree-Fock
      solver.do_force(force);
      solver.RHF(sol,occs,conv);
    } else {
      // Print information about used functionals
      if(verbose)
	print_info(dft.x_func,dft.c_func);

      if(stricmp(set.get_string("DFTGrid"),"Auto")==0) {
	// Solve restricted DFT problem first on a rough grid
	enum pzsic pz=solver.do_sic();
	if(pz==PERT) solver.do_sic(NO);
	solver.RDFT(sol,occs,initconv,initdft);
	if(pz==PERT) solver.do_sic(pz);
      }
      // .. and then on the final grid
      solver.do_force(force);
      solver.RDFT(sol,occs,conv,dft);
    }

    // Do population analysis
    if(verbose)
      population_analysis(basis,sol.P);

  } else {
    uscf_t sol;

    if(doload) {
      // Running polarized calculation but given restricted guess
      if(oldrestr) {
	// Project solution to new basis
	basis.projectMOs(oldbas,Eold,Cold,sol.Ea,sol.Ca);
	sol.Eb=sol.Ea;
	sol.Cb=sol.Ca;
      } else {
	// Project to new basis.
	basis.projectMOs(oldbas,Eaold,Caold,sol.Ea,sol.Ca);
	basis.projectMOs(oldbas,Ebold,Cbold,sol.Eb,sol.Cb);
      }
    } else if(guess == ATOMGUESS) {
      atomic_guess(basis,sol.Ca,sol.Ea,set);
      sol.Cb=sol.Ca;
      sol.Eb=sol.Ea;
    } else if(guess == MOLGUESS) {
      // Need to generate the starting guess.
      std::string name;
      molecular_guess(basis,set,name);

      // Load guess orbitals
      {
	Checkpoint guesschk(name,false);
	guesschk.read("Ca",sol.Ca);
	guesschk.read("Ea",sol.Ea);
	guesschk.read("Cb",sol.Cb);
	guesschk.read("Eb",sol.Eb);
      }
      // and remove the temporary file
      remove(name.c_str());
    }

    // Get orbital occupancies
    std::vector<double> occa, occb;
    get_unrestricted_occupancy(set,basis,occa,occb);

    // Write checkpoint.
    Checkpoint chkpt(savename,true);
    chkpt.write(basis);

    // Write number of electrons
    int Nel_alpha;
    int Nel_beta;
    get_Nel_alpha_beta(basis.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);
    chkpt.write("Nel",Nel);
    chkpt.write("Nel-a",Nel_alpha);
    chkpt.write("Nel-b",Nel_beta);

    // Solver
    SCF solver(basis,set,chkpt);

    // Freeze core orbitals?
    if(freezecore) {
      // Form the density matrix
      sol.Pa=form_density(sol.Ca,occa);
      sol.Pb=form_density(sol.Cb,occb);
      sol.P=sol.Pa+sol.Pb;

      // Get the natural orbitals
      arma::mat NO;
      arma::mat tmp;
      arma::vec occs;
      form_NOs(sol.P,basis.overlap(),NO,tmp,occs);

      // Then, localize the core orbitals within the occupied space
      size_t nloc=localize_core(basis,std::max(Nel_alpha,Nel_beta),NO);
      // and freeze them
      solver.set_frozen(NO.submat(0,0,NO.n_rows-1,nloc-1),0);
      // Update the current orbitals as well
      sol.Ca=NO;
      sol.Cb=NO;
    }

    if(hf) {
      // Solve restricted Hartree-Fock
      solver.do_force(force);
      solver.UHF(sol,occa,occb,conv);
    } else if(rohf) {
      // Solve restricted open-shell Hartree-Fock

      // Solve ROHF
      solver.ROHF(sol,Nel_alpha,Nel_beta,conv);

      // Set occupancies right
      get_unrestricted_occupancy(set,basis,occa,occb);
    } else {
      // Print information about used functionals
      if(verbose)
	print_info(dft.x_func,dft.c_func);

      if(stricmp(set.get_string("DFTGrid"),"Auto")==0) {
	// Solve restricted DFT problem first on a rough grid
	enum pzsic pz=solver.do_sic();
	if(pz==PERT) solver.do_sic(NO);
	solver.UDFT(sol,occa,occb,initconv,initdft);
	if(pz==PERT) solver.do_sic(pz);
      }
      // ... and then on the more accurate grid
      solver.do_force(force);
      solver.UDFT(sol,occa,occb,conv,dft);
    }

    if(verbose)
      population_analysis(basis,sol.Pa,sol.Pb);
  }
}

bool operator<(const ovl_sort_t & lhs, const ovl_sort_t & rhs) {
  // Sort into decreasing order
  return lhs.S > rhs.S;
}

arma::mat project_orbitals(const arma::mat & Cold, const BasisSet & minbas, const BasisSet & augbas) {
  Timer ttot;
  Timer t;

  // Total number of functions in augmented set is
  const size_t Ntot=augbas.get_Nbf();
  // Amount of old orbitals is
  const size_t Nold=Cold.n_cols;

  // Identify augmentation shells.
  std::vector<size_t> augshellidx;
  std::vector<size_t> origshellidx;

  std::vector<GaussianShell> augshells=augbas.get_shells();
  std::vector<GaussianShell> origshells=minbas.get_shells();

  // Loop over shells in augmented set.
  for(size_t i=0;i<augshells.size();i++) {
    // Try to find the shell in the original set
    bool found=false;
    for(size_t j=0;j<origshells.size();j++)
      if(augshells[i]==origshells[j]) {
	found=true;
	origshellidx.push_back(i);
	break;
      }

    // If the shell was not found in the original set, it is an
    // augmentation shell.
    if(!found)
      augshellidx.push_back(i);
  }

  // Overlap matrix in augmented basis
  arma::mat S=augbas.overlap();
  arma::vec Sval;
  arma::mat Svec;
  eig_sym_ordered(Sval,Svec,S);

  printf("Condition number of overlap matrix is %e.\n",Sval(0)/Sval(Sval.n_elem-1));

  printf("Diagonalization of basis took %s.\n",t.elapsed().c_str());
  t.set();

  // Count number of independent functions
  size_t Nind=0;
  for(size_t i=0;i<Ntot;i++)
    if(Sval(i)>=LINTHRES)
      Nind++;

  printf("Augmented basis has %i linearly independent and %i dependent functions.\n",(int) Nind,(int) (Ntot-Nind));

  // Drop linearly dependent ones.
  Sval=Sval.subvec(Sval.n_elem-Nind,Sval.n_elem-1);
  Svec=Svec.submat(0,Svec.n_cols-Nind,Svec.n_rows-1,Svec.n_cols-1);

  // Form the new C matrix.
  arma::mat C(Ntot,Nind);
  C.zeros();

  // The first vectors are simply the occupied states.
  for(size_t i=0;i<Nold;i++)
    for(size_t ish=0;ish<origshellidx.size();ish++)
      C.submat(augshells[origshellidx[ish]].get_first_ind(),i,augshells[origshellidx[ish]].get_last_ind(),i)=Cold.submat(origshells[ish].get_first_ind(),i,origshells[ish].get_last_ind(),i);

  // Do a Gram-Schmidt orthogonalization to find the rest of the
  // orthonormal vectors. But first we need to drop the eigenvectors
  // of S with the largest projection to the occupied orbitals, in
  // order to avoid linear dependency problems with the Gram-Schmidt
  // method.

  // Indices to keep in the treatment
  std::vector<size_t> keepidx;
  for(size_t i=0;i<Svec.n_cols;i++)
    keepidx.push_back(i);

  // Deleted functions
  std::vector<ovl_sort_t> delidx;

  // Drop the functions with the maximum overlap
  for(size_t j=0;j<Nold;j++) {
    // Find maximum overlap
    double maxovl=0.0;
    size_t maxind=-1;

    // Helper vector
    arma::vec hlp=S*C.col(j);

    for(size_t ii=0;ii<keepidx.size();ii++) {
      // Index of eigenvector is
      size_t i=keepidx[ii];
      // Compute projection
      double ovl=fabs(arma::dot(Svec.col(i),hlp))/sqrt(Sval(i));
      // Check if it has the maximal value
      if(fabs(ovl)>maxovl) {
	maxovl=ovl;
	maxind=ii;
      }
    }

    // Add the function to the deleted functions' list
    ovl_sort_t tmp;
    tmp.S=maxovl;
    tmp.idx=keepidx[maxind];
    delidx.push_back(tmp);

    //    printf("%4i/%4i deleted function %i with overlap %e.\n",(int) j+1, (int) Nold, (int) keepidx[maxind],maxovl);

    // Delete the index
    fflush(stdout);
    keepidx.erase(keepidx.begin()+maxind);
  }

  // Print deleted functions
  std::stable_sort(delidx.begin(),delidx.end());
  for(size_t i=0;i<delidx.size();i++) {
    printf("%4i/%4i deleted function %4i with overlap %e.\n",(int) i+1, (int) Nold, (int) delidx[i].idx,delidx[i].S);
  }
  fflush(stdout);

  // Fill in the rest of the vectors
  for(size_t i=0;i<keepidx.size();i++) {
    // The index of the vector to use is
    size_t ind=keepidx[i];
    // Normalize it, too
    C.col(Nold+i)=Svec.col(ind)/sqrt(Sval(ind));
  }

  // Run the orthonormalization of the set
  for(size_t i=0;i<Nind;i++) {
    double norm=arma::as_scalar(arma::trans(C.col(i))*S*C.col(i));
    // printf("Initial norm of vector %i is %e.\n",(int) i,norm);

    // Remove projections of already orthonormalized set
    for(size_t j=0;j<i;j++) {
      double proj=arma::as_scalar(arma::trans(C.col(j))*S*C.col(i));

      //    printf("%i - %i was %e\n",(int) i, (int) j, proj);
      C.col(i)-=proj*C.col(j);
    }

    norm=arma::as_scalar(arma::trans(C.col(i))*S*C.col(i));
    // printf("Norm of vector %i is %e.\n",(int) i,norm);

    // and normalize
    C.col(i)/=sqrt(norm);
  }

  printf("Projected orbitals in %s.\n",ttot.elapsed().c_str());
  fflush(stdout);

  return C;
}

std::vector<int> symgroups(const arma::mat & C, const arma::mat & S, const std::vector<arma::mat> & freeze, bool verbose) {
  // Initialize groups.
  std::vector<int> gp(C.n_cols,0);

  // Loop over frozen core groups
  for(size_t igp=0;igp<freeze.size();igp++) {

    // Compute overlap of orbitals with frozen core orbitals
    std::vector<ovl_sort_t> ovl(C.n_cols);
    for(size_t i=0;i<C.n_cols;i++) {

      // Store index
      ovl[i].idx=i;
      // Initialize overlap
      ovl[i].S=0.0;

      // Helper vector
      arma::vec hlp=S*C.col(i);

      // Loop over frozen orbitals.
      for(size_t ifz=0;ifz<freeze[igp].n_cols;ifz++) {
	// Compute projection
	double proj=arma::dot(hlp,freeze[igp].col(ifz));
	// Increment overlap
	ovl[i].S+=proj*proj;
      }
    }

    // Sort the projections
    std::sort(ovl.begin(),ovl.end());

    // Store the symmetries
    for(size_t i=0;i<freeze[igp].n_cols;i++) {
      // The orbital with the maximum overlap is
      size_t maxind=ovl[i].idx;
      // Change symmetry of orbital with maximum overlap
      gp[maxind]=igp+1;

      if(verbose)
	printf("Set symmetry of orbital %i to %i (overlap %e).\n",(int) maxind+1,gp[maxind],ovl[i].S);
    }

  }

  return gp;
}

void freeze_orbs(const std::vector<arma::mat> & freeze, const arma::mat & C, const arma::mat & S, arma::mat & H, bool verbose) {
  // Freezes the orbitals corresponding to different symmetry groups.

  // Form H_MO
  arma::mat H_MO=arma::trans(C)*H*C;

  // Get symmetry groups
  std::vector<int> sg=symgroups(C,S,freeze,verbose);

  // Loop over H_MO and zero out elements where symmetry groups differ
  for(size_t i=0;i<H_MO.n_rows;i++)
    for(size_t j=0;j<=i;j++)
      if(sg[i]!=sg[j]) {
	H_MO(i,j)=0;
	H_MO(j,i)=0;
      }

  // Back-transform to AO
  arma::mat SC=S*C;

  H=SC*H_MO*arma::trans(SC);
}

size_t localize_core(const BasisSet & basis, int nocc, arma::mat & C, bool verbose) {
  // Check orthonormality
  arma::mat S=basis.overlap();
  check_orth(C,S,false);

  const int Nmagic=(int) (sizeof(magicno)/sizeof(magicno[0]));

  // First, figure out how many orbitals to localize on each center
  std::vector<size_t> locno(basis.get_Nnuc(),0);
  // Localize on all the atoms of the same type than the excited atom
  for(size_t i=0;i<basis.get_Nnuc();i++)
    if(!basis.get_nucleus(i).bsse) {
      // Charge of nucleus is
      int Z=basis.get_nucleus(i).Z;

      // Get the number of closed shells
      int ncl=0;
      for(int j=0;j<Nmagic-1;j++)
	if(magicno[j] <= Z && Z <= magicno[j+1]) {
	  ncl=magicno[j]/2;
	  break;
	}

      // Store number of closed shells
      locno[i]=ncl;
    } else
      locno[i]=0;

  // Amount of orbitals already localized
  size_t locd=0;
  // Amount of basis functions
  size_t Nbf=basis.get_Nbf();

  // Perform the localization.
  for(size_t inuc=0;inuc<locno.size();inuc++) {
    if(locno[inuc]==0)
      continue;

    // The nucleus is located at
    coords_t cen=basis.get_coords(inuc);

    // Compute moment integrals around the nucleus
    std::vector<arma::mat> momstack=basis.moment(2,cen.x,cen.y,cen.z);
    // Get matrix which transforms into occupied MO basis
    arma::mat transmat=C.submat(0,locd,Nbf-1,nocc-1);

    // Sum together to get x^2 + y^2 + z^2
    arma::mat rsqmat=momstack[getind(2,0,0)]+momstack[getind(0,2,0)]+momstack[getind(0,0,2)];
    // and transform into the occupied MO basis
    rsqmat=arma::trans(transmat)*rsqmat*transmat;

    // Diagonalize rsq_mo
    arma::vec reig;
    arma::mat rvec;
    eig_sym_ordered(reig,rvec,rsqmat);

    /*
      printf("\nLocalization around center %i, eigenvalues (Å):",(int) locind[i].ind+1);
      for(size_t ii=0;ii<reig.n_elem;ii++)
      printf(" %e",sqrt(reig(ii))/ANGSTROMINBOHR);
      printf("\n");
      fflush(stdout);
    */

    // Rotate yet unlocalized orbitals
    C.submat(0,locd,Nbf-1,nocc-1)=transmat*rvec;

    // Increase number of localized orbitals
    locd+=locno[inuc];

    if(verbose)
      for(size_t k=0;k<locno[inuc];k++) {
	printf("Localized orbital around nucleus %i with Rrms=%e Å.\n",(int) inuc+1,sqrt(reig(k))/ANGSTROMINBOHR);
	fflush(stdout);
      }
  }

  // Check orthonormality
  check_orth(C,S,false);

  return locd;
}

#define bracket(X,Y) (0.5*std::real(arma::trace(X*arma::trans(Y))))

arma::cx_mat SCF::localize(const arma::mat & C) const {
  Timer t;

  if(verbose) {
    printf("\nLocalizing orbitals.\n");
    fflush(stdout);
  }

  arma::cx_mat U(C.n_cols,C.n_cols);
  U.eye();

  if(C.n_cols<2)
    // No optimization is necessary.
    return U;

  // Get R^2 matrix
  arma::mat rsq;
  {
    std::vector<arma::mat> momstack=basisp->moment(2);
    rsq=momstack[getind(2,0,0)]+momstack[getind(0,2,0)]+momstack[getind(0,0,2)];
  }

  // Get r matrix
  std::vector<arma::mat> rmat=basisp->moment(1);
  int k=0;
  double B=0, Bold;

  // G matrices
  std::vector<arma::cx_mat> G;
  // H matrix
  arma::cx_mat H;
  // Imaginary unit
  std::complex<double> imagI(0,1.0);

  while(true) {
    // Increase iteration number
    k++;
    Bold=B;

    // Compute B
    B=localize_B(C,U,rmat,rsq);

    // Compute the euclidean derivative matrix, Abrudan 2009 table 3 step 2
    arma::cx_mat Gammak=localize_Bder(C,U,rmat,rsq);

    // Riemannian gradient, Abrudan 2009 table 3 step 2
    G.push_back(Gammak*arma::trans(U) - U*arma::trans(Gammak));
    // Remove old matrices from memory?
    if(G.size()>2)
      G.erase(G.begin());

    printf("\t%4i\t%e\t%e\n",(int) k,B,bracket(G[G.size()-1],G[G.size()-1]));
    fflush(stdout);

    // H matrix
    if(k==1) {
      // First iteration; initialize with gradient
      H=G[G.size()-1];
    } else {
      // Compute Polak-Ribière coefficient
      double gamma=bracket(G[G.size()-1] - G[G.size()-2], G[G.size()-1]) / bracket(G[G.size()-2],G[G.size()-2]);

      // Update H
      H=G[G.size()-1]-gamma*H;

      // Check that update is OK
      double brack=bracket(H,G[G.size()-1]);
      if(brack<=0.0)
	// Reset to gradient.
	H=G[G.size()-1];

      //      else if(B>Bold)
      (void) Bold;

      // Reset to gradient - SD performs better!!
      H=G[G.size()-1];
    }

    // Check for convergence.
    if(bracket(G[G.size()-1],G[G.size()-1])<1e-6)
      break;


    arma::vec Hval;
    arma::cx_mat Hvec;
    double wmax;
    double Tmu;

    // Diagonalize iH to find eigenvalues purely imaginary
    // eigenvalues iw_i of -H; Abrudan 2009 table 3 step 1.
    bool diagok=arma::eig_sym(Hval,Hvec,imagI*H);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("PZ-SIC: error diagonalizing H.\n");
    }

    // Find maximal eigenvalue
    wmax=0.0;
    for(size_t n=0;n<Hval.n_elem;n++)
      if(fabs(Hval(n))>wmax)
	wmax=fabs(Hval(n));
    if(wmax==0.0) {
      continue;
    }

    // Compute maximal step size.
    // Order of the cost function in the coefficients of W.
    const int q=4;
    Tmu=2.0*M_PI/(q*wmax);

    // Amount of points to use for the fit.
    const int n=4;

    // Step size
    const double deltaTmu=Tmu/(n-1);
    std::vector<arma::cx_mat> R(n);

    // Trial matrices
    R[0].eye(C.n_cols,C.n_cols);
    R[1]=Hvec*arma::diagmat(arma::exp(deltaTmu*imagI*Hval))*arma::trans(Hvec);

    for(int i=2;i<n;i++)
      R[i]=R[i-1]*R[1];

    // Evaluate the first-order derivative of the cost function at the expansion points
    std::vector<double> Jprime(n);
    for(int i=0;i<n;i++) {
      // Trial matrix is
      arma::cx_mat Utr=R[i]*U;

      // Compute derivative matrix
      arma::cx_mat der=localize_Bder(C,Utr,rmat,rsq);
      // so the derivative wrt the step size is
      Jprime[i]=-2.0*std::real(arma::trace(der*arma::trans(U)*arma::trans(R[i])*arma::trans(H)));
    }

    /*
      printf("Derivatives:");
      for(int i=0;i<n;i++)
      printf(" % e",Jprime[i]);
      printf("\n");
    */

    // Step size to use
    double step=DBL_MAX;

    // Sanity check - are all derivatives negative?
    bool allneg=true;
    for(int i=0;i<n;i++)
      if(Jprime[i]>0.0)
	allneg=false;
    if(allneg) {
      // Check for any point lower than the current one
      for(int i=n-1;i>=0;i--) {
	arma::cx_mat Utr=R[i]*U;
	double Btr=localize_B(C,Utr,rmat,rsq);
	if(Btr<B) {
	  // Use current value of i
	  step=i*deltaTmu;
	  // and proceed with the update
	  break;
	}
      }
    } else {
      // Fit derivative to polynomial of order p: J'(mu) = a0 + a1*mu + ... + ap*mu^p
      const int p=n-1;

      // Compute polynomial coefficients.
      arma::vec jvec(p);
      for(int i=0;i<p;i++) {
	jvec(i)=Jprime[i+1]-Jprime[0];
      }

      // Form mu matrix
      arma::mat mumat(p,p);
      mumat.zeros();
      for(int i=0;i<p;i++) {
	// Value of mu on the row is
	double mu=(i+1)*deltaTmu;
	// Fill entries
	for(int j=0;j<p;j++)
	  mumat(i,j)=pow(mu,j+1);
      }

      arma::vec aval;
      bool solveok=true;

      // Solve for coefficients - may not be stable numerically
      solveok=arma::solve(aval,mumat,jvec);

      if(!solveok) {
	mumat.print("Mu");
	arma::trans(jvec).print("Jvec");
	throw std::runtime_error("Error solving for coefficients a.\n");
      }

      // Find smallest positive root of a0 + a1*mu + ... + ap*mu^p = 0.
      {
	// Coefficient of highest order term must be nonzero.
	int r=p;
	while(aval(r-1)==0.0)
	  r--;

	// Coefficients
	double a[r+2];
	a[0]=Jprime[0];
	for(int i=1;i<=r;i++)
	  a[i]=aval(i-1);

	// GSL routine workspace - r:th order polynomial has r+1 coefficients
	gsl_poly_complex_workspace *w=gsl_poly_complex_workspace_alloc(r+1);

	// Return values
	double z[2*r];
	int gslok=gsl_poly_complex_solve(a,r+1,w,z);

	if(gslok!=GSL_SUCCESS) {
	  ERROR_INFO();
	  fprintf(stderr,"Solution of polynomial root failed, error: \"%s\"\n",gsl_strerror(solveok));
	  throw std::runtime_error("Error solving polynomial.\n");
	}

	// Get roots
	std::vector< std::complex<double> > roots(r);
	for(int i=0;i<r;i++) {
	  roots[i].real()=z[2*i];
	  roots[i].imag()=z[2*i+1];
	}
	// and order them into increasing absolute value
	std::stable_sort(roots.begin(),roots.end(),abscomp<double>);

	int nreal=0;
	for(size_t i=0;i<roots.size();i++)
	  if(fabs(roots[i].imag())<10*DBL_EPSILON)
	    nreal++;

	/*
	  printf("%i real roots:",nreal);
	  for(size_t i=0;i<roots.size();i++)
	  if(fabs(roots[i].imag())<10*DBL_EPSILON)
	  printf(" (% e,% e)",roots[i].real(),roots[i].imag());
	  printf("\n");
	*/

	for(size_t i=0;i<roots.size();i++)
	  if(roots[i].real()>sqrt(DBL_EPSILON) && fabs(roots[i].imag())<10*DBL_EPSILON) {
	    // Root is real and positive. Is it smaller than the current minimum?
	    if(roots[i].real()<step)
	      step=roots[i].real();
	  }

	// Free workspace
	gsl_poly_complex_workspace_free(w);
      }

      // Sanity check
      if(step==DBL_MAX)
	step=0.0;
    }

    //    printf("Step size is %e, i.e. %e dTmu.\n",step,step/deltaTmu);

    // Step size is xmin. Update U
    if(step<0.0) throw std::runtime_error("Negative step size!\n");
    if(step!=0.0) {
      arma::cx_mat Ropt=Hvec*arma::diagmat(arma::exp(step*imagI*Hval))*arma::trans(Hvec);
      U=Ropt*U;
    }
  }

  if(verbose) {
    printf("Localization done in %s.\n\n",t.elapsed().c_str());
    fflush(stdout);
  }

  return U;
}


double SCF::localize_B(const arma::mat & C, const arma::cx_mat & M, const std::vector<arma::mat> & r, const arma::mat & rsq) const {
  // Calculate rotated orbitals
  arma::cx_mat Ctilde=C*M;

  double B=0;

  // <i|r^2|i> terms
  for(size_t io=0;io<Ctilde.n_cols;io++)
    B+=std::real(arma::as_scalar(arma::trans(Ctilde.col(io))*rsq*Ctilde.col(io)));

  // <i|r|i>^2 terms
  for(size_t io=0;io<Ctilde.n_cols;io++)
    for(int ic=0;ic<3;ic++) {
      std::complex<double> t=arma::as_scalar(arma::trans(Ctilde.col(io))*r[ic]*Ctilde.col(io));
      B-=std::norm(t);
    }

  return B;
}

arma::cx_mat SCF::localize_Bder(const arma::mat & C, const arma::cx_mat & M, const std::vector<arma::mat> & r, const arma::mat & rsq) const {
  // Returned matrix
  arma::cx_mat Bder(C.n_cols,C.n_cols);

  // Rotated orbitals
  arma::cx_mat Ctilde=C*M;

  // Loop over orbitals
  for(size_t a=0;a<C.n_cols;a++)
    for(size_t b=0;b<C.n_cols;b++) {

      // r^2 term
      Bder(a,b)=arma::as_scalar(arma::trans(C.col(a))*rsq*Ctilde.col(b));

      // r terms
      for(int ic=0;ic<3;ic++) {
	std::complex<double> tl=arma::as_scalar(arma::trans(C.col(a))*r[ic]*Ctilde.col(b));
	std::complex<double> tr=arma::as_scalar(arma::trans(Ctilde.col(b))*r[ic]*Ctilde.col(b));

	Bder(a,b)-=2.0*tl*tr;
      }
    }

  return Bder;
}

void SCF::do_force(bool val) {
  doforce=val;
}

void SCF::do_sic(enum pzsic val) {
  pz=val;
}

enum pzsic SCF::do_sic() const {
  return pz;
}

arma::mat interpret_force(const arma::vec & f) {
  if(f.n_elem%3!=0) {
    ERROR_INFO();
    throw std::runtime_error("Invalid argument for interpret_force.\n");
  }

  arma::mat force(f);
  force.reshape(3,f.n_elem/3);

  // Calculate magnitude in fourth column
  arma::mat retf(f.n_elem/3,4);
  retf.submat(0,0,f.n_elem/3-1,2)=arma::trans(force);
  for(size_t i=0;i<retf.n_rows;i++)
    retf(i,3)=sqrt( pow(retf(i,0),2) + pow(retf(i,1),2) + pow(retf(i,2),2) );

  return retf;
}
