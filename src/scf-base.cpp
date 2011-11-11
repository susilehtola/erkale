/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
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
#include "linalg.h"
#include "mathf.h"
#include "scf.h"
#include "stringutil.h"
#include "timer.h"

#define ROUGHTOL 1e-8

SCF::SCF(const BasisSet & basis, const Settings & set, Checkpoint & chkpt) {
  // Amount of basis functions
  Nbf=basis.get_Nbf();
  
  basisp=&basis;
  chkptp=&chkpt;

  // Multiplicity
  mult=set.get_int("Multiplicity");

  // Amount of electrons
  Nel=basis.Ztot()-set.get_int("Charge");

  usediis=set.get_bool("UseDIIS");
  diis_c1=set.get_bool("C1-DIIS");
  diisorder=set.get_int("DIISOrder");
  diisthr=set.get_double("DIISThr");
  useadiis=set.get_bool("UseADIIS");
  usebroyden=set.get_bool("UseBroyden");

  maxiter=set.get_int("MaxIter");
  verbose=set.get_bool("Verbose");

  direct=set.get_bool("Direct");

  // Check update scheme
  if(useadiis && usebroyden) {
    ERROR_INFO();
    throw std::runtime_error("ADIIS and Broyden mixing cannot be used at the same time.\n");
  } 

  if(!usediis && !useadiis && !usebroyden) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to run calculation without an update scheme.\n");
  }

  // Nuclear repulsion
  Enuc=basis.Enuc();

  try {
    // Use density fitting?
    densityfit=set.get_bool("DFTFitting");
    // Use Lobatto angular grid? (Lebedev is default)
    dft_lobatto=set.get_bool("DFTLobatto");
    // Direct DFT calculation?
    dft_direct=set.get_bool("DFTDirect");
  } catch(...) {
    // Hartree-Fock
    densityfit=0;
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
    printf("\nDiagonalizing basis set ... ");
    fflush(stdout);
    t.set();
  }

  double Sratio;
  Sinvh=BasOrth(S,set,Sratio);
    
  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());
    
    if(Sinvh.n_cols!=Sinvh.n_rows) {
      printf("Basis set is near-degenerate, ratio of smallest eigenvalue to largest\neigenvalue of overlap matrix is %.2e.\n",Sratio);
      printf("%i linear combinations of basis functions have been removed.\n",Sinvh.n_rows-Sinvh.n_cols);
    } else {
      printf("Ratio of smallest to largest eigenvalue of overlap matrix is %.2e.\n",Sratio);
    }
    printf("\n");
  }

  if(densityfit) {
    // Density fitting.

    // Form density fitting basis
    BasisSet dfitbas=basisp->density_fitting();

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

    dfit.fill(*basisp,dfitbas,direct);
  } else  {
    // Compute ERIs
    if(direct) {
      if(verbose) {
	t.set();
	printf("Forming ERI screening matrix ... ");
	fflush(stdout);
      }
      scr.fill(&basis);
    } else {
      // Compute memory requirement
      size_t N;
      
      if(verbose) {
	N=tab.memory_estimate(&basis);
	printf("Forming table of %lu ERIs, requiring %s of memory ... ",N,memory_size(N).c_str());
	fflush(stdout);
      }
      tab.fill(&basis);
    }
  }
  if(verbose)
    printf("done (%s)\n",t.elapsed().c_str());

  if(verbose)
    printf("\nInitialization of computation done in %s.\n\n",tinit.elapsed().c_str());
}

SCF::~SCF() {
}

void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::mat & NO_to_AO) {

  // First, get eigenvectors and eigenvalues of S so that we can go to
  // an orthonormal basis.
  arma::vec Sval;
  arma::mat Svec;
  eig_sym_ordered(Sval,Svec,S);

  // Count the number of linearly independent vectors
  size_t Nind=0;
  for(size_t i=0;i<Sval.n_elem;i++)
    if(Sval[i]>1e-5)
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
  arma::vec Pval;
  arma::mat Pvec;
  eig_sym_ordered(Pval,Pvec,P_orth);

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

void ROHF_update(arma::mat & Fa_AO, arma::mat & Fb_AO, const arma::mat & P_AO, const arma::mat & S, int Nel_alpha, int Nel_beta) {
  /*
   * T. Tsuchimochi and G. E. Scuseria, "Constrained active space
   * unrestricted mean-field methods for controlling
   * spin-contamination", J. Chem. Phys. 134, 064101 (2011).
   */

  arma::mat AO_to_NO;
  arma::mat NO_to_AO;
  form_NOs(P_AO,S,AO_to_NO,NO_to_AO);

  /*
  double tot=0.0;
  printf("Core orbital occupations:");
  for(size_t c=Nind-1;c>=Nind-Nc && c<Nind;c--) {
    printf(" %f",Pval(c));
    tot+=Pval(c);
  }
  printf("\n");

  printf("Active orbital occupations:");
  for(size_t a=Nind-Nc-1;a>=Nind-Nc-Na && a<Nind;a--) {
    printf(" %f",Pval(a));
    tot+=Pval(a);
  }
  printf("\n");
  printf("Total occupancy of core and active is %f.\n",tot);
  */

  // Construct \Delta matrix in AO basis
  arma::mat Delta_AO=(Fa_AO-Fb_AO)/2.0;

  // and take it to the NO basis.
  arma::mat Delta_NO=arma::trans(AO_to_NO)*Delta_AO*AO_to_NO;

  // Amount of independent orbitals is
  const size_t Nind=AO_to_NO.n_cols;
  // Amount of core orbitals is
  const size_t Nc=std::min(Nel_alpha,Nel_beta);
  // Amount of active space orbitals is
  const size_t Na=std::max(Nel_alpha,Nel_beta)-Nc;
  // Amount of virtual orbitals (in NO space) is
  const size_t Nv=Nind-Na-Nc;

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
	double S=arma::dot(C.col(j),hlp);
	if(fabs(S)>Smax) {
	  Smax=fabs(S);
	  loc=j;
	}
      }

      // Copy occupancy
      if(nocc[loc]!=0.0)
	printf("Problem in determine_occ: state %i was already occupied by %g electrons!\n",(int) loc,nocc[loc]);
      nocc[loc]+=nocc_old[i];
    }
}
      
void form_density(arma::mat & R, const arma::mat & C, size_t nocc) {
  std::vector<double> occs(nocc,1.0);
  form_density(R,C,occs);
}

void form_density(arma::mat & R, const arma::mat & C, const std::vector<double> & nocc) {
  // Check dimensions of R
  if(R.n_rows!=C.n_rows || R.n_cols!=C.n_rows)
    R.zeros(C.n_rows,C.n_rows);

  if(nocc.size()>C.n_cols) {
    std::ostringstream oss;
    oss << "Error in function " << __FUNCTION__ << " (file " << __FILE__ << ", near line " << __LINE__ << "): there should be " << nocc.size() << " occupied orbitals but only " << C.n_cols << " orbitals exist!\n";
    throw std::runtime_error(oss.str());
  }
  
  // Zero matrix
  R.zeros();
  // Formulate density
  for(size_t n=0;n<nocc.size();n++)
    if(nocc[n]>0.0)
      R+=nocc[n]*C.col(n)*arma::trans(C.col(n));
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
    // Resize output
    ret.resize(Nel/2);
    for(size_t i=0;i<ret.size();i++)
      ret[i]=2.0; // All orbitals doubly occupied
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

    // Resize output
    occa.resize(Nel_alpha);
    for(size_t i=0;i<occa.size();i++)
      occa[i]=1.0;
    
    occb.resize(Nel_beta);
    for(size_t i=0;i<occb.size();i++)
      occb[i]=1.0;
  }
}

double dip_mom(const arma::mat & P, const BasisSet & basis) {
  // Compute magnitude of dipole moment
  
  arma::vec dp=dipole_moment(P,basis);
  return norm(dp);
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
  else if(Nel%2==0 && mult%2!=1)
    throw std::runtime_error("Incorrect multiplicity for even number of electrons.\n");
  else if(Nel%2==1 && mult%2!=0)
    throw std::runtime_error("Incorrect multiplicity for odd number of electrons.\n");

  if(Nel%2==0)
    // Even number of electrons, the amount of spin up is 
    Nel_alpha=Nel/2+(mult-1)/2;
  else
    // Odd number of electrons, the amount of spin up is
    Nel_alpha=Nel/2+mult/2;

  // The rest are spin down
  Nel_beta=Nel-Nel_alpha;
}
