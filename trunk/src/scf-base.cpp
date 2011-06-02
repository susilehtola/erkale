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

// If not using DIIS, allow an energy increase of OSCTOL without reducing mixing factor.
#define OSCTOL 1e-2
// Initial mixing factor
#define INITMIX 0.25

// Broyden mix sum and difference of spin density matrices instead of
// matrices separately?
#define MIX_SUMDIFF

SCF::SCF(const BasisSet & basis, const Settings & set) {
  // Amount of basis functions
  Nbf=basis.get_Nbf();
  
  basisp=&basis;

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

  mixdensity=set.get_bool("MixDensity");
  dynamicmix=set.get_bool("DynamicMixing");
  
  // Check update scheme
  if((usediis || useadiis) && usebroyden) {
    ERROR_INFO();
    throw std::runtime_error("(A)DIIS and Broyden mixing cannot be used at the same time.\n");
  } 

  if(!usediis && !mixdensity && !useadiis && !usebroyden) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to run calculation without an update scheme.\n");
  }

  // Nuclear repulsion
  Enuc=basis.Enuc();

  // Convergence criteria
  deltaEmax=set.get_double("DeltaEmax");
  deltaPmax=set.get_double("DeltaPmax");
  deltaPrms=set.get_double("DeltaPrms");

  if(set.dft_enabled()) {
    densityfit=set.get_bool("DensityFitting");
    
    // Initial and final tolerance of DFT grid
    dft_initialtol=set.get_double("DFTInitialTol");
    dft_finaltol=set.get_double("DFTFinalTol");
    dft_switch=set.get_double("DFTSwitch");
    dft_lobatto=set.get_bool("DFTLobatto");
    // Direct DFT calculation?
    dft_direct=set.get_bool("DFTDirect");
  } else {
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
    // Density fitting
    t.set();
    printf("Computing density fitting integrals ... ");
    fflush(stdout);
    dfit.fill(*basisp,basisp->density_fitting(),direct);
  } else {
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

int SCF::get_Nel() const {
  return Nel;
}

int SCF::get_Nel_alpha() const {
  return Nel_alpha;
}

int SCF::get_Nel_beta() const {
  return Nel_beta;
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
  // Check dimensions of R
  if(R.n_rows!=C.n_rows)
    R=arma::mat(C.n_rows,C.n_rows);
  else if(R.n_cols!=C.n_rows)
    R=arma::mat(C.n_rows,C.n_rows);

  if(nocc>C.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "There should be " << nocc << " occupied orbitals but only " << C.n_cols << " orbitals exist!\n";
    throw std::runtime_error(oss.str());
  }

  // Zero matrix
  R.zeros();
  // Formulate density
  for(size_t n=0;n<nocc;n++)
    R+=C.col(n)*trans(C.col(n));
}

void form_density(arma::mat & R, const arma::mat & C, const arma::vec & nocc) {
  // Check dimensions of R
  if(R.n_rows!=C.n_rows)
    R=arma::mat(C.n_rows,C.n_rows);
  else if(R.n_cols!=C.n_rows)
    R=arma::mat(C.n_rows,C.n_rows);

  if(nocc.n_elem!=C.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "There should be " << nocc << " occupied orbitals but only " << C.n_cols << " orbitals exist!\n";
    throw std::runtime_error(oss.str());
  }

  // Zero matrix
  R.zeros();
  // Formulate density
  for(size_t n=0;n<nocc.n_elem;n++)
    R+=nocc(n)*C.col(n)*trans(C.col(n));
}

void update_mixing(double & mix, double Ecur, double Eold, double Eold2) {
  // Determine mixing factor

  if(Ecur-Eold>OSCTOL)
    // Oscillation - reduce mixing factor
    mix/=2.0;
  else if( (Ecur-Eold<=OSCTOL) && (Eold-Eold2<=OSCTOL) && (mix<1.0)) {
    // Converging to a minimum, increase mixing factor
    mix*=cbrt(2.0);
  }
  
  // Sanity check
  if(mix>1.0)
    mix=1.0;
}
