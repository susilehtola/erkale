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
#include <climits>

#include "basis.h"
#include "broyden.h"
#include "checkpoint.h"
#include "elements.h"
#include "dftfuncs.h"
#include "dftgrid.h"
#include "global.h"
#include "guess.h"
#include "lebedev.h"
#include "linalg.h"
#include "mathf.h"
#include "properties.h"
#include "scf.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"
#include "trrh.h"
#include "localization.h"
#include "pzstability.h"

extern "C" {
#include <gsl/gsl_poly.h>
}

// Threshold for imaginary component in orbitals
#define IMAGTHR 1e-7
// Threshold for Cholesky decomposition of SAD density matrix
#define CHOLCTHR 1e-6

enum guess_t parse_guess(const std::string & val) {
  if(stricmp(val,"Core")==0)
    return CORE_GUESS;
  else if(stricmp(val,"SAD")==0 || stricmp(val,"Atomic")==0)
    return SAD_GUESS;
  else if(stricmp(val,"GSAP")==0)
    return GSAP_GUESS;
  else if(stricmp(val,"SAP")==0)
    return SAP_GUESS;
  else if(stricmp(val,"NO")==0)
    return NO_GUESS;
  else if(stricmp(val,"GWH")==0)
    return GWH_GUESS;
  else if(stricmp(val,"HUCKEL")==0)
    return HUCKEL_GUESS;
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

  // Dimer calculation?
  dimcalc=set.get_bool("DimerSymmetry");

  usediis=set.get_bool("UseDIIS");
  diisorder=set.get_int("DIISOrder");
  diiseps=set.get_double("DIISEps");
  diisthr=set.get_double("DIISThr");
  diiscomb=set.get_bool("DIISComb");
  useadiis=set.get_bool("UseADIIS");
  usebroyden=set.get_bool("UseBroyden");
  usetrrh=set.get_bool("UseTRRH");
  trrhmins=set.get_double("TRRHminS");

  maxiter=set.get_int("MaxIter");
  shift=set.get_double("Shift");
  verbose=set.get_bool("Verbose");

  direct=set.get_bool("Direct");
  decfock=set.get_bool("DecFock");
  strictint=set.get_bool("StrictIntegrals");
  // Integral screening threshold
  intthr=strictint ? DBL_EPSILON : set.get_double("IntegralThresh");
  // Sanity check
  if(intthr>1e-6) {
    fprintf(stderr,"Warning - spuriously large integral threshold %e\n",intthr);
  }

  doforce=false;

  // Check update scheme
  if(useadiis && usebroyden) {
    ERROR_INFO();
    throw std::runtime_error("ADIIS and Broyden mixing cannot be used at the same time.\n");
  }

  if(!usediis && !useadiis && !usebroyden && !usetrrh && (shift==0.0)) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to run calculation without an update scheme.\n");
  }

  // Nuclear repulsion
  Enuc=basis.Enuc();

  // Use density fitting?
  densityfit=set.get_bool("DensityFitting");
  // How much memory to allow (convert to bytes)
  fitmem=1000000*set.get_int("FittingMemory");
  // Linear dependence threshold
  fitthr=set.get_double("FittingThreshold");

  // Use Cholesky?
  cholesky=set.get_bool("Cholesky");
  cholthr=set.get_double("CholeskyThr");
  cholshthr=set.get_double("CholeskyShThr");
  cholnafthr=set.get_double("CholeskyNAFThr");
  cholmode=set.get_int("CholeskyMode");
  if(cholesky && densityfit)
    throw std::logic_error("Can't enable both Cholesky and density fitting!\n");

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
  chkptp->write("S",S);

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());

    printf("Forming kinetic energy matrix ... ");
    fflush(stdout);
    t.set();
  }

  T=basis.kinetic();
  chkptp->write("T",T);

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());

    printf("Forming nuclear attraction matrix ... ");
    fflush(stdout);
    t.set();
  }

  Vnuc=basis.nuclear();
  chkptp->write("Vnuc",Vnuc);

  if(verbose)
    printf("done (%s)\n",t.elapsed().c_str());

  // Electric field?
  arma::mat H_E(T.n_rows,T.n_cols);
  H_E.zeros();

  std::vector<std::string> Ef=splitline(set.get_string("EField"));
  if(Ef.size()!=3)
    throw std::runtime_error("EField must have 3 components!\n");
  std::vector<double> E(Ef.size());
  for(size_t i=0;i<Ef.size();i++)
    E[i]=readdouble(Ef[i]);
  if(E[0]!=0.0 || E[1] != 0.0 || E[2] != 0.0) {
    // Compute center of charge
    coords_t cen;
    cen.x=cen.y=cen.z=0.0;
    int Ztot=0;
    for(size_t i=0;i<basis.get_Nnuc();i++) {
      nucleus_t nuc=basis.get_nucleus(i);
      if(!nuc.bsse) {
	cen=cen+nuc.r*nuc.Z;
	Ztot+=nuc.Z;
      }
    }
    cen=cen/Ztot;
    fprintf(stderr,"Center of charge at % .3f % .3f % .3f\n",cen.x/ANGSTROMINBOHR,cen.y/ANGSTROMINBOHR,cen.z/ANGSTROMINBOHR);

    // Get dipole integrals
    std::vector<arma::mat> dipint(basis.moment(1,cen.x,cen.y,cen.z));
    // Accumulate
    for(size_t i=0;i<E.size();i++)
      // H_e = - E . qr
      H_E+=E[i]*dipint[i];
  }
  chkptp->write("H_E",H_E);

  // Form core Hamiltonian
  Hcore=T+Vnuc+H_E;
  chkptp->write("Hcore",Hcore);

  if(verbose) {
    printf("\n");
    t.set();
  }

  Sinvh=BasOrth(S,set);
  chkptp->write("Sinvh",Sinvh);

  if(verbose) {
    printf("Basis set diagonalized in %s.\n",t.elapsed().c_str());
    t.set();

    if(Sinvh.n_cols!=Sinvh.n_rows) {
      printf("%i linear combinations of basis functions have been removed.\n",(int) (Sinvh.n_rows-Sinvh.n_cols));
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
      fitlib.load_basis(set.get_string("FittingBasis"));

      // Construct fitting basis
      construct_basis(dfitbas,basisp->get_nuclei(),fitlib,set);
    }

    // Compute memory estimate
    std::string memest=memory_size(dfit.memory_estimate(*basisp,dfitbas,intthr,direct));

    if(verbose) {
      if(direct)
	printf("Initializing density fitting calculation, requiring %s memory ... ",memest.c_str());
      else
	printf("Computing density fitting integrals, requiring %s memory ... ",memest.c_str());
      fflush(stdout);
      t.set();
    }

    // Calculate the fitting integrals, don't run in B-matrix mode
    size_t Npairs=dfit.fill(*basisp,dfitbas,direct,intthr,fitthr,false);

    if(verbose) {
      printf("done (%s)\n",t.elapsed().c_str());
      printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
      printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
      fflush(stdout);
    }
  } else  {
    // Compute ERIs
    if(cholesky) {
      if(verbose) {
	t.set();
	printf("Computing repulsion integrals.\n");
	fflush(stdout);
      }

      if(cholmode==-1) {
	chol.load();
	if(verbose) {
	  printf("%i Cholesky vectors loaded from file in %s.\n",(int) chol.get_Naux(),t.elapsed().c_str());
	  fflush(stdout);
	}
      }

      if(chol.get_Nbf()!=basisp->get_Nbf()) {
	size_t Npairs=chol.fill(*basisp,cholthr,cholshthr,intthr,verbose);
	if(verbose) {
	  printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
	  fflush(stdout);
	}
	// Natural auxiliary function screening
	if(cholnafthr>0.0)
	  chol.naf_transform(cholnafthr,verbose);
	if(cholmode==1) {
	  t.set();
	  chol.save();
	  printf("Cholesky vectors saved to file in %s.\n",t.elapsed().c_str());
	  fflush(stdout);
	}
      }
    } else {
      if(direct) {
	size_t Npairs;
	// Form decontracted basis set and get the screening matrix
	if(verbose) {
	  t.set();
	  printf("Forming ERI screening matrix ... ");
	  fflush(stdout);
	}

	if(decfock) {
	  // Use decontracted basis
          decbas=basis.decontract(decconv);
	  Npairs=scr.fill(&decbas,intthr,verbose);
        } else {
	  // Use contracted basis
	  Npairs=scr.fill(&basis,intthr,verbose);
        }

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  if(decfock)
	    printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) decbas.get_unique_shellpairs().size());
	  else
	    printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
	}

      } else {
	// Compute memory requirement
	size_t N;

	if(verbose) {
	  N=tab.N_ints(&basis,intthr);
	  printf("Forming table of %u ERIs, requiring %s of memory ... ",(unsigned int) N,memory_size(N*sizeof(double)).c_str());
	  fflush(stdout);
	}
	// Don't compute small integrals
	size_t Npairs=tab.fill(&basis,intthr);

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
	}
      }
    }
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

void SCF::do_force(bool val) {
  doforce=val;
}

size_t SCF::get_maxiter() const {
  return maxiter;
}

void SCF::set_maxiter(size_t val) {
  maxiter=val;
}

bool SCF::get_verbose() const {
  return verbose;
}

void SCF::set_verbose(bool verb) {
  verbose=verb;
}

void SCF::fill_rs(double omega) {
  if(densityfit) {
    // Compute range separated integrals if necessary
    bool fill;
    if(!dfit_rs.get_Naux()) {
      fill=true;
    } else {
      double o, kl, ks;
      dfit_rs.get_range_separation(o,kl,ks);
      fill=(!(o==omega));
    }
    if(fill) {
      dfit_rs.set_range_separation(omega,0.0,1.0);

      // Compute memory estimate
      std::string memest=memory_size(dfit.memory_estimate(*basisp,dfitbas,intthr,direct));

      Timer t;
      if(verbose) {
	if(direct)
	  printf("Initializing short-range density fitting calculation, requiring %s memory ... ",memest.c_str());
	else
	  printf("Computing short-range density fitting integrals, requiring %s memory ... ",memest.c_str());
	fflush(stdout);
      }

      t.set();
      size_t Npairs=dfit_rs.fill(*basisp,dfitbas,direct,intthr,fitthr,dfit.Bmat_enabled());
      if(verbose) {
	printf("done (%s)\n",t.elapsed().c_str());
	printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basisp->get_unique_shellpairs().size());
	printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
	fflush(stdout);
      }
    }

  } else if(cholesky) {
    // Compute range separated integrals if necessary
    bool fill;
    if(!chol_rs.get_Naux()) {
      fill=true;
    } else {
      double o, kl, ks;
      chol_rs.get_range_separation(o,kl,ks);
      fill=(!(o==omega));
    }
    if(fill) {
      Timer t;
      if(verbose) {
	printf("Computing short-range repulsion integrals.\n");
	fflush(stdout);
      }

      chol_rs.set_range_separation(omega,0.0,1.0);
      if(cholmode==-1) {
	chol_rs.load();
	if(verbose) {
	  printf("%i Cholesky vectors loaded from file in %s.\n",(int) chol.get_Naux(),t.elapsed().c_str());
	  fflush(stdout);
	}
	fill = (chol_rs.get_Nbf() != basisp->get_Nbf());
      }
      if(fill) {
	size_t Npairs=chol_rs.fill(*basisp,cholthr,cholshthr,intthr,verbose);
	if(verbose) {
	  printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basisp->get_unique_shellpairs().size());
	  fflush(stdout);
	}
	// Natural auxiliary function screening
	if(cholnafthr>0.0)
	  chol_rs.naf_transform(cholnafthr,verbose);
	if(cholmode==1) {
	  t.set();
	  chol_rs.save();
	  printf("Cholesky vectors saved to file in %s.\n",t.elapsed().c_str());
	  fflush(stdout);
	}
      }
    }

  } else {
    if(!direct) {
      // Compute range separated integrals if necessary
      bool fill;
      if(!tab_rs.get_N()) {
	fill=true;
      } else {
	double o, kl, ks;
	tab_rs.get_range_separation(o,kl,ks);
	fill=(!(o==omega));
      }
      if(fill) {
	Timer t;
	if(verbose) {
	  printf("Computing short-range repulsion integrals ... ");
	  fflush(stdout);
	}
	tab_rs.set_range_separation(omega,0.0,1.0);
	size_t Np=tab_rs.fill(basisp,intthr);

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  printf("%i short-range shell pairs are significant.\n",(int) Np);
	  fflush(stdout);
	}
      }
    } else {
      bool fill;
      if(!scr_rs.get_N()) {
	fill=true;
      } else {
	double o, kl, ks;
	scr_rs.get_range_separation(o,kl,ks);
	fill=(!(o==omega));
      }
      if(fill) {
	Timer t;
	if(verbose) {
	  printf("Computing short-range repulsion integrals ... ");
	  fflush(stdout);
	}

	scr_rs.set_range_separation(omega,0.0,1.0);
	size_t Np=scr_rs.fill(basisp,intthr);

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  printf("%i short-range shell pairs are significant.\n",(int) Np);
	  fflush(stdout);
	}
      }
    }
  }
}

arma::mat SCF::get_S() const {
  return S;
}

arma::mat SCF::get_Sinvh() const {
  return Sinvh;
}

arma::mat SCF::get_Hcore() const {
  return Hcore;
}

Checkpoint *SCF::get_checkpoint() const {
  return chkptp;
}

bool SCF::get_strictint() const {
  return strictint;
}

void SCF::PZSIC_Fock(std::vector<arma::cx_mat> & Forb, arma::vec & Eorb, const arma::cx_mat & Ctilde, dft_t dft, DFTGrid & grid, DFTGrid & nlgrid, bool fock) {
  // Compute the orbital-dependent Fock matrices
  Eorb.resize(Ctilde.n_cols);
  if(fock)
    Forb.resize(Ctilde.n_cols);

  // Range separation constants
  double omega, kfull, kshort;
  range_separation(dft.x_func,omega,kfull,kshort);
  if(omega!=0.0)
    fill_rs(omega);

  // Orbital density matrices
  std::vector<arma::cx_mat> Pcorb(Ctilde.n_cols);
  for(size_t io=0;io<Ctilde.n_cols;io++)
    Pcorb[io]=arma::conj(Ctilde.col(io))*arma::strans(Ctilde.col(io));

  std::vector<arma::mat> Porb(Ctilde.n_cols);
  for(size_t io=0;io<Ctilde.n_cols;io++)
    Porb[io]=arma::real(Pcorb[io]);

  Timer t;

  if(densityfit) {
    if(kfull!=0.0)
      throw std::runtime_error("PZ-SIC hybrid functionals not supported with density fitting.\n");
    if(kshort!=0.0)
      throw std::runtime_error("PZ-SIC range separated functionals not supported with density fitting.\n");

    if(verbose) {
      if(fock)
	printf("Constructing orbital Coulomb matrices ... ");
      else
	printf("Computing    orbital Coulomb energies ... ");
      fflush(stdout);
      t.set();
    }

    // Coulomb matrices
    std::vector<arma::mat> Jorb;
    Jorb=dfit.calcJ(Porb);
    for(size_t io=0;io<Ctilde.n_cols;io++)
      Eorb[io]=0.5*arma::trace(Porb[io]*Jorb[io]);
    if(fock)
      for(size_t io=0;io<Ctilde.n_cols;io++)
	Forb[io]=Jorb[io]*COMPLEX1;

    if(verbose) {
      printf("done (%s)\n",t.elapsed().c_str());
      fflush(stdout);
    }

  } else {
    if(cholesky) {
      // Cholesky integrals
      if(verbose) {
	if(fock)
	  printf("Constructing orbital Coulomb matrices ... ");
	else
	  printf("Computing    orbital Coulomb energies ... ");
	fflush(stdout);
	t.set();
      }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for(size_t io=0;io<Ctilde.n_cols;io++) {
	// Calculate Coulomb term
	arma::mat Jorb=chol.calcJ(Porb[io]);
	// and Coulomb energy
	Eorb[io]=0.5*arma::trace(Porb[io]*Jorb);
	if(fock)
	  Forb[io]=Jorb*COMPLEX1;
      }

      if(verbose) {
	printf("done (%s)\n",t.elapsed().c_str());
	fflush(stdout);
      }

      // Full exchange
      if(kfull!=0.0) {
	if(verbose) {
	  if(fock)
	    printf("Constructing orbital exchange matrices ... ");
	  else
	    printf("Computing    orbital exchange energies ... ");
	  fflush(stdout);
	  t.set();
	}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for(size_t io=0;io<Ctilde.n_cols;io++) {
	  // Fock matrix
	  arma::cx_mat Korb=kfull*chol.calcK(Ctilde.col(io));
	  // and energy
	  Eorb[io]-=0.5*std::real(arma::trace(Pcorb[io]*Korb));
	  if(fock)
	    Forb[io]-=Korb;
	}

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  fflush(stdout);
	}
      }

      // Short-range part
      if(kshort!=0.0) {
	if(verbose) {
	  if(fock)
	    printf("Constructing orbital short-range exchange matrices ... ");
	  else
	    printf("Computing    orbital short-range exchange energies ... ");
	  fflush(stdout);
	  t.set();
	}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for(size_t io=0;io<Ctilde.n_cols;io++) {
	  // Potential and energy
	  arma::cx_mat Korb=kshort*chol_rs.calcK(Ctilde.col(io));
	  Eorb[io]-=0.5*std::real(arma::trace(Pcorb[io]*Korb));
	  if(fock)
	    Forb[io]-=Korb;
	}

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  fflush(stdout);
	}
      }

    } else {
      if(!direct) {
	// Tabled integrals
	if(verbose) {
	  if(fock)
	    printf("Constructing orbital Coulomb matrices ... ");
	  else
	    printf("Computing    orbital Coulomb energies ... ");
	  fflush(stdout);
	  t.set();
	}

	for(size_t io=0;io<Ctilde.n_cols;io++) {
	  // Calculate Coulomb term
	  arma::mat Jorb=tab.calcJ(Porb[io]);
	  // and Coulomb energy
	  Eorb[io]=0.5*arma::trace(Porb[io]*Jorb);
	  if(fock)
	    Forb[io]=Jorb*COMPLEX1;
	}

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  fflush(stdout);
	}

	// Full exchange
	if(kfull!=0.0) {
	  if(verbose) {
	    if(fock)
	      printf("Constructing orbital exchange matrices ... ");
	    else
	      printf("Computing    orbital exchange energies ... ");
	    fflush(stdout);
	    t.set();
	  }

	  for(size_t io=0;io<Ctilde.n_cols;io++) {
	    // Fock matrix
	    arma::cx_mat Korb=kfull*tab.calcK(Pcorb[io]);
	    // and energy
	    Eorb[io]-=0.5*std::real(arma::trace(Pcorb[io]*Korb));
	    if(fock)
	      Forb[io]-=Korb;
	  }

	  if(verbose) {
	    printf("done (%s)\n",t.elapsed().c_str());
	    fflush(stdout);
	  }
	}

	// Short-range part
	if(kshort!=0.0) {
	  if(verbose) {
	    if(fock)
	      printf("Constructing orbital short-range exchange matrices ... ");
	    else
	      printf("Computing    orbital short-range exchange energies ... ");
	    fflush(stdout);
	    t.set();
	  }

	  for(size_t io=0;io<Ctilde.n_cols;io++) {
	    // Potential and energy
	    arma::cx_mat Korb=kshort*tab_rs.calcK(Pcorb[io]);
	    Eorb[io]-=0.5*std::real(arma::trace(Pcorb[io]*Korb));
	    if(fock)
	      Forb[io]-=Korb;
	  }

	  if(verbose) {
	    printf("done (%s)\n",t.elapsed().c_str());
	    fflush(stdout);
	  }
	}

      } else {
	if(verbose) {
	  std::string leg = (kfull==0.0) ? "Coulomb" : "Coulomb and exchange";
	  if(fock)
	    printf("Constructing orbital %s matrices ... ",leg.c_str());
	  else
	    printf("Computing    orbital %s energies ... ",leg.c_str());
	  fflush(stdout);
	  t.set();
	}

	// Calculate Coulomb and exchange terms
	{
	  std::vector<arma::cx_mat> JKorb=scr.calcJK(Pcorb,1.0,kfull,intthr);
	  for(size_t io=0;io<Ctilde.n_cols;io++) {
	    // Coulomb-exchange energy is
	    Eorb[io]=0.5*std::real(arma::trace(Pcorb[io]*JKorb[io]));
	    if(fock)
	      Forb[io]=JKorb[io];
	  }
	}

	if(verbose) {
	  printf("done (%s)\n",t.elapsed().c_str());
	  fflush(stdout);
	}

	// Short-range part
	if(kshort!=0.0) {
	  if(verbose) {
	    if(fock)
	      printf("Constructing orbital short-range exchange matrices ... ");
	    else
	      printf("Computing    orbital short-range exchange energies ... ");
	    fflush(stdout);
	    t.set();
	  }

	  // Calculate exchange term
	  {
	    std::vector<arma::cx_mat> Korb=scr_rs.calcJK(Pcorb,0.0,kshort,intthr);
	    for(size_t io=0;io<Ctilde.n_cols;io++) {
	      // Short-range exchange energy is (sign already accounted for)
	      Eorb[io]+=0.5*std::real(arma::trace(Pcorb[io]*Korb[io]));
	      if(fock)
		Forb[io]+=Korb[io];
	    }
	  }

	  if(verbose) {
	    printf("done (%s)\n",t.elapsed().c_str());
	    fflush(stdout);
	  }
	}
      }
    }
  }

  // Exchange-correlation
  if(dft.x_func > 0 || dft.c_func > 0) {
    if(verbose) {
      if(fock)
	printf("Constructing orbital XC matrices ... ");
      else
	printf("Computing    orbital XC energies ... ");
      fflush(stdout);
      t.set();
    }

    std::vector<double> Nelnum; // Numerically integrated density
    std::vector<arma::mat> XC; // Exchange-correlation matrices
    std::vector<double> Exc; // Exchange-correlation energy

    grid.eval_Fxc(dft.x_func,dft.c_func,Ctilde,XC,Exc,Nelnum,fock);

    if(verbose) {
      printf("done (%s)\n",t.elapsed().c_str());
      fflush(stdout);
    }

    if(dft.nl) {
      if(verbose) {
	if(fock)
	  printf("Constructing orbital non-local correlation matrices ... ");
	else
	  printf("Computing    orbital non-local correlation energies ... ");
	fflush(stdout);
	t.set();
      }

      // Parallellization inside VV10
      for(size_t i=0;i<Ctilde.n_cols;i++) {
	double Enl=0.0;
	nlgrid.eval_VV10(nlgrid,dft.vv10_b,dft.vv10_C,Porb[i],XC[i],Enl,fock);
      }

      if(verbose) {
	printf("done (%s)\n",t.elapsed().c_str());
	fflush(stdout);
      }
    }

    // Add in the XC part to the energy
    for(size_t io=0;io<Ctilde.n_cols;io++) {
      Eorb[io]+=Exc[io];
    }

    // and the Fock matrix
    if(fock)
      for(size_t io=0;io<Ctilde.n_cols;io++)
	Forb[io]+=XC[io]*COMPLEX1;
  }
}

void SCF::core_guess(rscf_t & sol) const {
  // Get core Hamiltonian
  sol.H=Hcore;
  // and diagonalize it to get the orbitals
  diagonalize(sol);
}

void SCF::core_guess(uscf_t & sol) const {
  // Get core Hamiltonian
  sol.Ha=Hcore;
  sol.Hb=Hcore;
  // and diagonalize it to get the orbitals
  diagonalize(sol);
}

void SCF::gwh_guess(rscf_t & sol) const {
  // Initialize matrix
  sol.H=Hcore;
  for(size_t i=0;i<Hcore.n_rows;i++) {
    sol.H(i,i)=Hcore(i,i);
    for(size_t j=0;j<Hcore.n_cols;j++) {
      sol.H(i,j)=0.875*S(i,j)*(Hcore(i,i)+Hcore(j,j));
      sol.H(j,i)=sol.H(i,j);
    }
  }
  diagonalize(sol);
}

void SCF::gwh_guess(uscf_t & sol) const {
  // Initialize matrix
  sol.Ha=Hcore;
  for(size_t i=0;i<Hcore.n_rows;i++) {
    sol.Ha(i,i)=Hcore(i,i);
    for(size_t j=0;j<i;j++) {
      sol.Ha(i,j)=0.875*S(i,j)*(Hcore(i,i)+Hcore(j,j));
      sol.Ha(j,i)=sol.Ha(i,j);
    }
  }
  sol.Hb=sol.Ha;
  diagonalize(sol);
}

void SCF::sap_guess(rscf_t & sol) const {
  DFTGrid grid(basisp);

  // Use a (150,194) grid
  int nrad=150;
  int lmax=23;
  bool grad=false;
  bool tau=false;
  bool lapl=false;
  bool strict=false;
  bool nl=false;
  grid.construct(nrad,lmax,grad,tau,lapl,strict,nl);

  // Get SAP potential
  arma::mat Vsap(grid.eval_SAP());
  // Approximate Hamiltonian is
  sol.H=Hcore+Vsap;
  // and diagonalize it to get the orbitals
  diagonalize(sol);
}

void SCF::sap_guess(uscf_t & sol) const {
  DFTGrid grid(basisp);

  // Use a (150,194) grid
  int nrad=150;
  int lmax=23;
  bool grad=false;
  bool tau=false;
  bool lapl=false;
  bool strict=false;
  bool nl=false;
  grid.construct(nrad,lmax,grad,tau,lapl,strict,nl);

  // Get SAP potential
  arma::mat Vsap(grid.eval_SAP());
  // Approximate Hamiltonian is
  sol.Ha=Hcore+Vsap;
  sol.Hb=sol.Ha;
  // and diagonalize it to get the orbitals
  diagonalize(sol);
}

arma::mat SCF::exchange_localization(const arma::mat & Co, const arma::mat & Cv0) const {
  if(Cv0.n_cols<Co.n_cols)
    throw std::runtime_error("Not enough virtuals given!\n");
  if(Co.n_rows != Cv0.n_rows)
    throw std::runtime_error("Orbital matrices not consistent!\n");
  if(Co.n_rows != basisp->get_Nbf())
    throw std::runtime_error("Orbital matrix does not match basis set!\n");

  // Returned orbitals
  arma::mat Cv(Cv0);
  // Active space orbitals
  arma::mat Cvact(Cv.cols(0,Co.n_cols-1));

  // Loop over occupied orbitals
  for(size_t io=0;io<Co.n_cols;io++) {
    // Form exchange matrix
    arma::mat K;
    if(densityfit) {
      // Because this is done an orbital at a time, it doesn't really
      // matter how much fitting memory is given so we just allow
      // however much
      std::vector<double> occs(1,1.0);
      K=dfit.calcK(Co.col(io),occs,INT_MAX);
    } else {
      if(cholesky) {
	K=chol.calcK(Co.col(io));
      } else {
	arma::mat P(Co.col(io)*arma::trans(Co.col(io)));
	if(direct)
	  K=scr.calcK(P,intthr);
	else
	  K=tab.calcK(P);
      }
    }

    // Virtual-virtual exchange matrix is
    arma::mat Kvv(arma::trans(Cv)*K*Cv);

    // Eigendecomposition
    arma::vec eval;
    arma::mat evec;
    arma::eig_sym(eval,evec,-Kvv);

    // Rotate orbitals to new basis; orbital becomes lowest-lying orbital
    Cvact.col(io)=Cv*evec.col(0);
  }

  // Reorthogonalize active orbitals
  arma::vec sval;
  arma::mat svec;
  arma::mat Svv(arma::trans(Cvact)*S*Cvact);
  arma::eig_sym(sval,svec,Svv);

  arma::vec sinvh(sval);
  for(size_t i=0;i<sval.n_elem;i++)
    sinvh(i)=1.0/sqrt(sval(i));

  // Orthogonalizing matrix is
  arma::mat O(svec*arma::diagmat(sinvh)*arma::trans(svec));
  Cvact=Cvact*O;

  // Build projection matrix
  arma::mat Sv(arma::trans(Cvact)*S*Cv);
  arma::mat Pv(Cv.n_cols,Cv.n_cols);
  Pv.eye();
  Pv-=arma::trans(Sv)*Sv;

  // Run eigendecomposition
  arma::vec Pval;
  arma::mat Pvec;
  arma::eig_sym(Pval,Pvec,Pv);

  // Now, the inactive virtuals have eigenvalues 1.0, whereas the
  // active orbitals have eigenvalues ~ 0.0. The localized virtual
  // space is then given by
  Cv=Cv*Pvec;

  return Cv;
}

void imag_lost(const rscf_t & sol, const arma::mat & S, double & d) {
  // Compute amount of electrons
  int Nel=(int) round(arma::trace(S*sol.P))/2;

  // MO overlap matrix
  if(sol.cC.n_cols == sol.C.n_cols) {
    arma::cx_mat MOovl=arma::trans(sol.C.cols(0,Nel-1))*S*sol.cC.cols(0,Nel-1);

    // Amount of electrons lost in the approximation
    d=2.0*(Nel-std::real(arma::trace(MOovl*arma::trans(MOovl))));
  } else
    d=0.0;
}

void imag_lost(const uscf_t & sol, const arma::mat & S, double & da, double & db) {
  // Compute amount of electrons
  int Nela=(int) round(arma::trace(S*sol.Pa));
  int Nelb=(int) round(arma::trace(S*sol.Pb));

  // MO overlap matrix
  if(sol.cCa.n_cols == sol.Ca.n_cols) {
    arma::cx_mat MOovla=arma::trans(sol.Ca.cols(0,Nela-1))*S*sol.cCa.cols(0,Nela-1);
    da=Nela-std::real(arma::trace(MOovla*arma::trans(MOovla)));
  } else
    da=0.0;

  if(sol.cCb.n_cols == sol.Cb.n_cols) {
    arma::cx_mat MOovlb=arma::trans(sol.Cb.cols(0,Nelb-1))*S*sol.cCb.cols(0,Nelb-1);
    db=Nelb-std::real(arma::trace(MOovlb*arma::trans(MOovlb)));
  } else
    db=0.0;
}

template<typename T> void diagonalize_wrk(const arma::mat & S, const arma::mat & Sinvh, const arma::mat & P, const arma::Mat<T> & H, double shift, arma::Mat<T> & C, arma::vec & E) {
  // Transform Hamiltonian into orthogonal basis
  arma::Mat<T> Horth;
  if(shift==0.0)
    Horth=arma::trans(Sinvh)*H*Sinvh;
  else
    Horth=arma::trans(Sinvh)*(H-shift*S*P/2.0*S)*Sinvh;
  // Update orbitals and energies
  arma::Mat<T> orbs;
  eig_sym_ordered_wrk(E,orbs,Horth);
  // Transform back to non-orthogonal basis
  C=Sinvh*orbs;

  if(shift!=0.0) {
    // Orbital energies occupied by shift, so recompute these
    E=arma::real(arma::diagvec(arma::trans(C)*H*C));
  }
}

void SCF::diagonalize(rscf_t & sol, double sval) const {
  ::diagonalize(S,Sinvh,sol,sval);
}

void SCF::diagonalize(uscf_t & sol, double sval) const {
  ::diagonalize(S,Sinvh,sol,sval);
}

void diagonalize(const arma::mat & S, const arma::mat & Sinvh, rscf_t & sol, double shift) {
  diagonalize_wrk<double>(S,Sinvh,sol.P/2.0,sol.H,shift,sol.C,sol.E);
  check_orth(sol.C,S,false);
}

void diagonalize(const arma::mat & S, const arma::mat & Sinvh, uscf_t & sol, double shift) {
  diagonalize_wrk<double>(S,Sinvh,sol.Pa,sol.Ha,shift,sol.Ca,sol.Ea);
  check_orth(sol.Ca,S,false);

  diagonalize_wrk<double>(S,Sinvh,sol.Pb,sol.Hb,shift,sol.Cb,sol.Eb);
  check_orth(sol.Cb,S,false);
}


void ROHF_update(arma::mat & Fa_AO, arma::mat & Fb_AO, const arma::mat & P_AO, const arma::mat & S, std::vector<double> occa, std::vector<double> occb, bool verbose) {
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

  // Get rid of trailing zeros
  while(occa[occa.size()-1]==0.0)
    occa.erase(occa.begin()+occa.size()-1);
  while(occb[occb.size()-1]==0.0)
    occb.erase(occb.begin()+occb.size()-1);

  // Amount of independent orbitals is
  size_t Nind=AO_to_NO.n_cols;
  // Amount of core orbitals is
  size_t Nc=std::min(occa.size(),occb.size());
  // Amount of active space orbitals is
  size_t Na=std::max(occa.size(),occb.size())-Nc;
  // Amount of virtual orbitals (in NO space) is
  size_t Nv=Nind-Na-Nc;

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
  for(size_t c=0;c<Nc;c++) // Loop over core orbitals
    for(size_t v=Nind-Nv;v<Nind;v++) { // Loop over virtuals
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

arma::mat form_density(const arma::mat & C, int nocc) {
  arma::vec occs(C.n_cols);
  if(nocc)
    occs.subvec(0,nocc-1).ones();
  return form_density(C,occs);
}

arma::mat form_density(const arma::mat & C, const arma::vec & occs0) {
  arma::vec occs(C.n_cols);
  occs.zeros();
  if(occs0.n_elem) {
    size_t nel=std::min(occs.n_elem,occs0.n_elem);
    if(nel)
      occs.subvec(0,nel-1)=occs0.subvec(0,nel-1);
  }
  return C*arma::diagmat(occs)*arma::trans(C);
}

void form_density(rscf_t & sol, size_t nocc) {
  arma::vec occs(sol.C.n_cols);
  occs.zeros();
  if(nocc)
    occs.subvec(0,nocc-1)=2.0*arma::ones(nocc);
  form_density(sol,occs);
}

void form_density(rscf_t & sol, const arma::vec & occs0) {
  arma::vec occs(sol.C.n_cols);
  occs.zeros();
  if(occs0.n_elem) {
    size_t nel=std::min(occs.n_elem,occs0.n_elem);
    occs.subvec(0,nel-1)=occs0.subvec(0,nel-1);
  }

  sol.P=sol.C*arma::diagmat(occs)*arma::trans(sol.C);
}

void form_density(uscf_t & sol, const arma::vec & occa0, const arma::vec & occb0) {
  arma::vec occa(sol.Ca.n_cols);
  occa.zeros();
  if(occa0.n_elem) {
    size_t nel=std::min(occa.n_elem,occa0.n_elem);
    occa.subvec(0,nel-1)=occa0.subvec(0,nel-1);
  }
  arma::vec occb(sol.Cb.n_cols);
  occb.zeros();
  if(occb0.n_elem) {
    size_t nel=std::min(occb.n_elem,occb0.n_elem);
    occb.subvec(0,nel-1)=occb0.subvec(0,nel-1);
  }
  sol.Pa=sol.Ca*arma::diagmat(occa)*arma::trans(sol.Ca);
  sol.Pb=sol.Cb*arma::diagmat(occb)*arma::trans(sol.Cb);
  sol.P=sol.Pa+sol.Pb;
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
  arma::vec occs;
  form_NOs(P,S,C,occs);

  // and form the density
  return form_density(C,Nel);
}

std::vector<double> atomic_occupancy(double Nel, int Nbf) {
  // The density is spread over the whole of the possible valence shell.
  // Only the noble gas core is doubly occupied
  static const int magic_values[] = {0, 2, 10, 18, 36, 54, 86, 118};
  static const size_t Nmagic = sizeof(magic_values)/sizeof(magic_values[0]);
  // Shell index
  size_t imagic = 0;
  while(Nel>magic_values[imagic+1]) {
    imagic++;
    if(imagic+1>=Nmagic)
      throw std::logic_error("Noble elements beyond Oganesson not known\n");
  }

  // Number of fully occupied orbitals
  int nfull = magic_values[imagic]/2;
  // Number of partially occupied orbitals
  int norbpart = magic_values[imagic+1]/2 - nfull;
  // Sanity check for minimal basis
  if(nfull + norbpart > Nbf)
    norbpart = Nbf-nfull;

  // Number of partially occupied electrons
  double nelpart = Nel - nfull;

  // Form occupations
  std::vector<double> ret(nfull+norbpart,1.0);
  for(int i=nfull;i<nfull+norbpart;i++)
    ret[i]=nelpart*1.0/norbpart;

  return ret;
}

std::vector<double> get_restricted_occupancy(const Settings & set, const BasisSet & basis, bool atomic) {
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

    /*
      printf("Occupancies\n");
      for(size_t i=0;i<ret.size();i++)
      printf("%.2f ",ret[i]);
      printf("\n");
    */
  } else {
    // Aufbau principle.
    int Nel=basis.Ztot()-set.get_int("Charge");
    if(Nel%2!=0) {
      throw std::runtime_error("Refusing to run restricted calculation on unrestricted system!\n");
    }

    if(atomic && basis.get_Nnuc()==1) {
      // Atomic case.
      ret=atomic_occupancy(Nel/2,basis.get_Nbf());
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

void get_unrestricted_occupancy(const Settings & set, const BasisSet & basis, std::vector<double> & occa, std::vector<double> & occb, bool atomic) {
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

    /*
      printf("Occupancies\n");
      printf("alpha\t");
      for(size_t i=0;i<occa.size();i++)
      printf("%.2f ",occa[i]);
      printf("\nbeta\t");
      for(size_t i=0;i<occb.size();i++)
      printf("%.2f ",occb[i]);
      printf("\n");
    */
  } else {
    // Aufbau principle. Get amount of alpha and beta electrons.

    int Nel_alpha, Nel_beta;
    get_Nel_alpha_beta(basis.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);

    if(atomic && basis.get_Nnuc()==1) {
      // Atomic case
      occa=atomic_occupancy(Nel_alpha,basis.get_Nbf());
      occb=atomic_occupancy(Nel_beta,basis.get_Nbf());
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

  if(Nel_alpha<0) {
    std::ostringstream oss;
    oss << "A multiplicity of " << mult << " would mean " << Nel_alpha << " alpha electrons!\n";
    throw std::runtime_error(oss.str());
  } else if(Nel_beta<0) {
    std::ostringstream oss;
    oss << "A multiplicity of " << mult << " would mean " << Nel_beta << " beta electrons!\n";
    throw std::runtime_error(oss.str());
  }
}

dft_t parse_pzmet(const std::string & pzmod, const dft_t & method) {
  bool X=false;
  bool C=false;
  bool D=false;

  for(size_t i=0;i<pzmod.size();i++)
    switch(pzmod[i]) {
    case('x'):
    case('X'):
      X=true;
    break;

    case('c'):
    case('C'):
      C=true;
    break;

    case('d'):
    case('D'):
      D=true;
    break;

    case(' '):
    case('\0'):
      break;

    default:
      throw std::runtime_error("Invalid PZmode\n");
    }

  dft_t oomethod(method);
  if(!X)
    oomethod.x_func=0;
  if(!C)
    oomethod.c_func=0;
  if(!D)
    oomethod.nl=false;

  return oomethod;
}

int parse_pzimag(const std::string & str) {
  int ret;
  if(stricmp(str,"true")==0 || stricmp(str,"yes")==0)
    ret=1;
  else if(stricmp(str,"auto")==0)
    ret=-1;
  else if(stricmp(str,"false")==0 || stricmp(str,"no")==0)
    ret=0;
  else
    throw std::logic_error("Invalid value for PZimag\n");
  return ret;
}

pz_scaling_t parse_pzscale(const std::string & scale) {
  if(stricmp(scale,"Const")==0 || stricmp(scale,"Constant")==0)
    return PZ_SCALE_CONSTANT;
  if(stricmp(scale,"Dens")==0 || stricmp(scale,"Density")==0)
    return PZ_SCALE_DENSITY;
  if(stricmp(scale,"Kin")==0 || stricmp(scale,"Kinetic")==0)
    return PZ_SCALE_KINETIC;

  throw std::runtime_error("Setting \"" + scale + "\" not recognized!\n");
}

dft_t parse_dft(const Settings & set, bool init) {
  dft_t dft;
  dft.gridtol=0.0;
  dft.nl=false;
  dft.vv10_b=0.0;
  dft.vv10_C=0.0;

  // Use Lobatto quadrature?
  dft.lobatto=set.get_bool("DFTLobatto");

  // Tolerance
  std::string tolkw = init ? "DFTInitialTol" : "DFTFinalTol";

  // Use static grid?
  if(stricmp(set.get_string("DFTGrid"),"Auto")!=0) {
    std::vector<std::string> opts=splitline(set.get_string("DFTGrid"));
    if(opts.size()!=2) {
      throw std::runtime_error("Invalid DFT grid specified.\n");
    }

    dft.adaptive=false;
    dft.nrad=readint(opts[0]);
    dft.lmax=readint(opts[1]);
    if(dft.nrad<1 || dft.lmax==0) {
      throw std::runtime_error("Invalid DFT radial grid specified.\n");
    }

    // Check if l was given in number of points
    if(dft.lmax<0) {
      // Try to find corresponding Lebedev grid
      for(size_t i=0;i<sizeof(lebedev_degrees)/sizeof(lebedev_degrees[0]);i++)
	if(lebedev_degrees[i]==-dft.lmax) {
	  dft.lmax=lebedev_orders[i];
	  break;
	}
      if(dft.lmax<0) {
	std::ostringstream oss;
	oss << "Invalid DFT angular grid specified. Supported Lebedev grids:";
	for(size_t i=0;i<sizeof(lebedev_degrees)/sizeof(lebedev_degrees[0]);i++) {
	  if(i)
	    oss << ",";
	  oss << " " << lebedev_degrees[i];
	}
	oss << ".\n";
	throw std::runtime_error(oss.str());
      }
    }

  } else {
    dft.adaptive=true;
    dft.gridtol=set.get_double(tolkw);
  }

  // Parse functionals
  parse_xc_func(dft.x_func,dft.c_func,set.get_string("Method"));

  // Parse VV10
  std::string vv10s(set.get_string("VV10"));
  if(stricmp(vv10s,"Auto")==0) {
    // Determine automatically if VV10 is necessary
    if(dft.x_func>0)
      dft.nl=needs_VV10(dft.x_func,dft.vv10_b,dft.vv10_C);
    if(!dft.nl && dft.c_func>0)
      dft.nl=needs_VV10(dft.c_func,dft.vv10_b,dft.vv10_C);

  } else if(stricmp(vv10s,"True")==0 || stricmp(vv10s,"Yes")==0) {
    dft.nl=true;

    std::vector<std::string> vvopts=splitline(set.get_string("VV10Pars"));
    if(vvopts.size()!=2)
      throw std::runtime_error("Invalid VV10Pars!\n");

    dft.vv10_b=readdouble(vvopts[0]);
    dft.vv10_C=readdouble(vvopts[1]);

  } else if(stricmp(vv10s,"False")==0 || stricmp(vv10s,"No")==0) {
    // Do nothing

  } else if(vv10s.size()) {

    throw std::runtime_error("Error parsing VV10 setting.\n");
  }

  if(dft.nl) {
    if(dft.vv10_b <= 0.0 || dft.vv10_C <= 0.0) {
      std::ostringstream oss;
      oss << "VV10 parameters given b = " << dft.vv10_b << ", C = " << dft.vv10_C << " are not valid.\n";
      throw std::runtime_error(oss.str());
    }

    if(dft.adaptive)
      throw std::runtime_error("Adaptive DFT grids not supported with VV10.\n");

    std::vector<std::string> opts=splitline(set.get_string("NLGrid"));
    dft.nlnrad=readint(opts[0]);
    dft.nllmax=readint(opts[1]);
    if(dft.nlnrad<1 || dft.nllmax==0) {
      throw std::runtime_error("Invalid DFT radial grid specified.\n");
    }

    // Check if l was given in number of points
    if(dft.nllmax<0) {
      // Try to find corresponding Lebedev grid
      for(size_t i=0;i<sizeof(lebedev_degrees)/sizeof(lebedev_degrees[0]);i++)
	if(lebedev_degrees[i]==-dft.nllmax) {
	  dft.nllmax=lebedev_orders[i];
	  break;
	}
      if(dft.nllmax<0)
	throw std::runtime_error("Invalid DFT angular grid specified.\n");
    }

    // Check that xc grid is larger than nl grid
    if(dft.nrad < dft.nlnrad || dft.lmax < dft.nllmax)
      throw std::runtime_error("xc grid should be bigger than nl grid!\n");
  }

  return dft;
}

void calculate(const BasisSet & basis, const Settings & set, bool force) {
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
  double convthr(set.get_double("ConvThr"));

  // Get exchange and correlation functionals
  dft_t dft;
  dft_t initdft;
  // Initial convergence settings
  double initconvthr(convthr);

  if(!hf && !rohf) {
    parse_xc_func(dft.x_func,dft.c_func,set.get_string("Method"));

    initdft=parse_dft(set,true);
    dft=parse_dft(set,false);

    initconvthr*=set.get_double("DFTDelta");
  }

  // Check consistency of parameters
  if(!hf && !rohf && (exact_exchange(dft.x_func)!=0.0 || is_range_separated(dft.x_func)))
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

  arma::cx_mat CW, CWa, CWb;
  bool doCW=false;

  // Which guess to use
  enum guess_t guess=parse_guess(set.get_string("Guess"));

  // Amount of electrons
  int Nel_alpha=-1;
  int Nel_beta=-1;
  if(!set.get_string("Occupancies").size())
    get_Nel_alpha_beta(basis.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);

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
      if(load.exist("CW.re")) {
	arma::cx_mat CWold;
	load.cread("CW",CWold);
	basis.projectOMOs(oldbas,CWold,CW,Nel_alpha);
	doCW=true;
      }
    } else {
      // Load energies and orbitals
      load.read("Ca",Caold);
      load.read("Ea",Eaold);
      load.read("Cb",Cbold);
      load.read("Eb",Ebold);
      if(load.exist("CWa.re") && load.exist("CWb.re")) {
	arma::cx_mat CWaold, CWbold;
	load.cread("CWa",CWaold);
	load.cread("CWb",CWbold);

	basis.projectOMOs(oldbas,CWaold,CWa,Nel_alpha);
	basis.projectOMOs(oldbas,CWbold,CWb,Nel_beta);
	doCW=true;
      }
    }
  }

  if(set.get_int("Multiplicity")==1 && Nel%2==0 && !set.get_bool("ForcePol")) {
    // Closed shell case
    rscf_t sol;
    // Initialize energy
    memset(&sol.en, 0, sizeof(energy_t));

    // Project old solution to new basis
    if(doload) {
      // Restricted calculation wanted but loaded spin-polarized one
      if(!oldrestr) {
	// Find out natural orbitals
	arma::vec occs;
	form_NOs(Pold,oldbas.overlap(),Cold,occs);

	// Use alpha orbital energies
	Eold=Eaold;
      }

      // Orbitals
      basis.projectMOs(oldbas,Eold,Cold,sol.E,sol.C,Nel_alpha);
    }

    // Get orbital occupancies
    std::vector<double> occs=get_restricted_occupancy(set,basis);

    // Write checkpoint.
    Checkpoint chkpt(savename,true);
    chkpt.write(basis);

    // Write number of electrons
    chkpt.write("Nel",Nel);
    chkpt.write("Nel-a",Nel_alpha);
    chkpt.write("Nel-b",Nel_beta);

    // Write OMOs
    if(doCW) {
      if(!oldrestr)
	chkpt.cwrite("CW",CWa);
      else
	chkpt.cwrite("CW",CW);
    }

    // Write method
    chkpt.write("Method",set.get_string("Method"));

    // Solver
    SCF solver(basis,set,chkpt);

    // Handle guesses
    if(!doload) {
      if(guess==CORE_GUESS)
	solver.core_guess(sol);
      else if(guess==SAP_GUESS)
	solver.sap_guess(sol);
      else if(guess==GWH_GUESS)
	solver.gwh_guess(sol);
      else if(guess == GSAP_GUESS) {
        sol.H=solver.get_Hcore()+sap_guess(basis,set);
        solver.diagonalize(sol);
      } else if(guess == HUCKEL_GUESS) {
        sol.H=huckel_guess(basis,set);
        solver.diagonalize(sol);
      } else if((guess == SAD_GUESS) || (guess == NO_GUESS)) {
        sol.P=sad_guess(basis,set);

        // Build Fock operator from SAD density matrix. Get natural orbitals
	arma::vec occ;
	form_NOs(sol.P,solver.get_S(),sol.C,occ);

	if(guess==SAD_GUESS) {
	  // Avoid computing exact exchange with all NOs by truncating
	  // the occupations to the nonzero ones
	  {
	    double othr(sqrt(DBL_EPSILON));
	    arma::uword pos;
	    for(pos=0;pos<occ.n_elem;pos++)
	      if(occ(pos)<=othr)
		break;
            if(pos<occ.n_elem)
              occ.subvec(pos,occ.n_elem-1).zeros();
	  }

	  std::vector<double> goccs(arma::conv_to< std::vector<double> >::from(occ));
	  size_t maxiter(solver.get_maxiter());
	  solver.set_maxiter(0);

	  if(hf || rohf)
	    solver.RHF(sol,goccs,convthr);
	  else {
	    dft_t nonl(dft);
	    nonl.nl=false;
	    solver.RDFT(sol,goccs,convthr,nonl);
	  }
	  solver.set_maxiter(maxiter);

	  // Do the diagonalization
	  solver.diagonalize(sol,0.0);
	}
      } else {
        throw std::logic_error("Guess not handled!\n");
      }
    }
    sol.P=form_density(sol.C,occs);

    if(hf || rohf) {
      // Solve restricted Hartree-Fock
      solver.do_force(force);
      solver.RHF(sol,occs,convthr);
    } else {
      // Print information about used functionals
      if(verbose) {
	print_info(dft.x_func,dft.c_func);
	if(dft.nl)
	  printf("Using VV10 non-local correlation with b = % f, C = %f\n",dft.vv10_b,dft.vv10_C);
      }

      // Perdew-Zunger?
      bool pz=set.get_bool("PZ");
      int pzprec=set.get_int("PZprec");
      bool pzov=set.get_bool("PZov");
      bool pzoo=set.get_bool("PZoo");
      int pzloc=parse_pzimag(set.get_string("PZloc"));
      enum locmet pzlocmet=parse_locmet(set.get_string("PZlocmet"));
      double pzw=set.get_double("PZw");
      int pzmax=set.get_int("PZiter");
      double pzIthr=set.get_double("PZIthr");
      double pzOOthr=set.get_double("PZOOthr");
      double pzOVthr=set.get_double("PZOVthr");
      double pzNRthr=set.get_double("PZNRthr");
      double pzEthr=set.get_double("PZEthr");
      int pzimag=parse_pzimag(set.get_string("PZimag"));
      int pzstab=set.get_int("PZstab");
      double pzstabthr=-set.get_double("PZstabThr"); // Minus sign!
      int seed=set.get_int("PZseed");
      dft_t oodft(parse_pzmet(set.get_string("PZmode"),dft));
      pz_scaling_t pzscale(parse_pzscale(set.get_string("PZscale")));
      double pzscaleexp(set.get_double("PZscaleExp"));

      if(!pz) {
	if(dft.adaptive && (initdft.x_func>0 || initdft.c_func>0)) {
	  // Solve restricted DFT problem first on a rough grid
	  solver.RDFT(sol,occs,initconvthr,initdft);

	  if(verbose) {
	    fprintf(stderr,"\n");
	    fflush(stderr);
	  }
	}

	// ... and then on the more accurate grid
	solver.do_force(force);
	solver.RDFT(sol,occs,convthr,dft);

	// Write out the results
	chkpt.write("C",sol.C);
	chkpt.write("E",sol.E);
	chkpt.write("P",sol.P);
	chkpt.write("Converged",1);
	chkpt.write(sol.en);

      } else {
	// Calculation has not yet converged
	chkpt.write("Converged",0);

	// The localizing matrix
	arma::cx_mat W;
	if(chkpt.exist("CW.re")) {
	  if(verbose) printf("Read localization matrix from checkpoint.\n");

	  // Get old localized orbitals
	  chkpt.cread("CW",CW);
	  // Save the orbitals
	  sol.cC=CW;

	  // Imaginary treatment?
	  if(rms_norm(arma::imag(sol.cC))>IMAGTHR) {
	    if(pzimag==-1) {
	      if(verbose) printf("Norm of imaginary part %e, turning on imaginary dof.\n",rms_norm(arma::imag(sol.cC)));
	      pzimag=1;
	    } else if(pzimag==0) {
	      throw std::runtime_error("Requesting calculation without imaginary dofs, but given orbitals have imaginary component!\n");
	    }
	    // Can't localize
	    pzloc=0;
	  } else {
	    // Store real part in orbital matrix
	    sol.C=arma::real(sol.cC);
	    if(pzloc==-1)
	      // Assume orbitals are already localized
	      pzloc=0;
	    W.eye(Nel_alpha,Nel_alpha);
	  }
	} else { // No entry in checkpoint
	  sol.cC=sol.C*COMPLEX1;
	  if(!pzoo) {
	    W.eye(Nel_alpha,Nel_alpha);
	  } else {
	    if(pzimag!=1)
	      W=real_orthogonal(Nel_alpha,seed)*std::complex<double>(1.0,0.0);
	    else
	      W=complex_unitary(Nel_alpha,seed);
	  }
	  if(pzloc==-1) pzloc=1;
	  sol.cC.zeros(sol.C.n_rows,sol.C.n_cols);
	  sol.cC.cols(0,Nel_alpha-1)=sol.C.cols(0,Nel_alpha-1)*W;
	  if(sol.C.n_cols>(size_t) Nel_alpha)
	    sol.cC.cols(Nel_alpha,sol.C.n_cols-1)=sol.C.cols(Nel_alpha,sol.C.n_cols-1)*COMPLEX1;
	}

	if(pzloc==1) {
	  // Localize starting guess
	  if(verbose) printf("\nInitial localization.\n");
	  Timer tloc;
	  double measure;
	  // Max 1e5 iterations, gradient norm <= pzIthr
	  orbital_localization(pzlocmet,basis,sol.C.cols(0,Nel_alpha-1),sol.P,measure,W,verbose,pzimag!=1,1e5,pzIthr,DBL_MAX);
	  if(verbose) {
	    printf("\n");

	    fprintf(stderr,"%-64s %10.3f\n","    Initial localization",tloc.get());
	    fflush(stderr);
	  }

	  // Save the complex orbitals
	  sol.cC.zeros(sol.C.n_rows,sol.C.n_cols);
	  sol.cC.cols(0,Nel_alpha-1)=sol.C.cols(0,Nel_alpha-1)*W;
	  if(sol.C.n_cols>(size_t) Nel_alpha)
	    sol.cC.cols(Nel_alpha,sol.C.n_cols-1)=sol.C.cols(Nel_alpha,sol.C.n_cols-1)*COMPLEX1;
	}

	// Save the orbitals
	chkpt.cwrite("CW",sol.cC);

	PZStability stab(&solver,verbose);
	stab.set_method(dft,oodft,pzw,pzscale,pzscaleexp);
	stab.set(sol);

	while(true) {
	  double dEo=0.0, dEv=0.0;
	  if(pzoo) {
	    // First, optimize the OO transformations
	    stab.set_params(true,pzimag==1,false,true);
	    dEo=stab.optimize(pzmax,pzOOthr,pzNRthr,pzEthr,pzprec);
	    sol=stab.get_rsol();
	  }
	  if(pzov) {
	    // Then the OV transformations
	    stab.set_params(true,pzimag==1,true,false);
	    dEv=stab.optimize(pzmax,pzOVthr,0.0,pzEthr,pzprec);
	    sol=stab.get_rsol();
	  }
	  if(dEo!=0.0 || dEv!=0.0)
	    continue;

	  // Check stability of OO rotations
	  if(pzoo && pzstab==-1) {
	    stab.set_params(true,pzimag==1,false,true);
	    bool instab=stab.check(true,pzstabthr);
	    if(instab) {
	      sol=stab.get_rsol();
	      continue;
	    }

	    // Imaginary instability?
	    if(pzimag==-1) {
	      stab.set_params(false,true,false,true);
	      instab=stab.check(true,pzstabthr);
	      if(instab) {
		pzimag=1;
		stab.optimize(pzmax,pzOOthr,pzNRthr,pzEthr,pzprec);
		if(pzov) {
		  stab.set_params(false,true,true,false);
		  stab.optimize(pzmax,pzOVthr,pzNRthr,0.0,pzprec);
		}
		sol=stab.get_rsol();
		continue;
	      }
	    }
	  } else if(pzstab==-2) {
	    // Check stability of OO+OV rotations
	    stab.set_params(true,pzimag==1,pzov,pzoo);
	    bool instab=stab.check(true,pzstabthr);
	    if(instab) {
	      sol=stab.get_rsol();
	      continue;
	    }

	    // Imaginary instability?
	    if(pzimag==-1) {
	      stab.set_params(false,true,pzov,pzoo);
	      instab=stab.check(true,pzstabthr);
	      if(instab) {
		pzimag=1;
		stab.optimize(pzmax,pzOOthr,pzNRthr,pzEthr,pzprec);
		sol=stab.get_rsol();
		continue;
	      }
	    }
	  }
	  // No instabilities
	  break;
	}

	// Stability analysis?
	if(pzoo && pzstab==1) {
	  stab.set_params(true,pzimag!=0,false,true);
	  stab.check();
	}
	if(pzstab==2) {
	  // Check stability of OO+OV rotations
	  stab.set_params(true,pzimag!=0,pzov,pzoo);
	  stab.check();
	}
      }

      // Calculation has converged
      chkpt.write("Converged",1);
    }

    // Do population analysis
    if(verbose) {
      population_analysis(basis,sol.P);
    }

  } else {
    uscf_t sol;
    // Initialize energy
    memset(&sol.en, 0, sizeof(energy_t));

    if(doload) {
      // Running polarized calculation but given restricted guess
      if(oldrestr) {
	// Project solution to new basis
	basis.projectMOs(oldbas,Eold,Cold,sol.Ea,sol.Ca,Nel_alpha);
	sol.Eb=sol.Ea;
	sol.Cb=sol.Ca;
      } else {
	// Project to new basis.
	basis.projectMOs(oldbas,Eaold,Caold,sol.Ea,sol.Ca,Nel_alpha);
	basis.projectMOs(oldbas,Ebold,Cbold,sol.Eb,sol.Cb,Nel_beta);
      }
    }

    // Get orbital occupancies
    std::vector<double> occa, occb;
    get_unrestricted_occupancy(set,basis,occa,occb);

    // Write checkpoint.
    Checkpoint chkpt(savename,true);
    chkpt.write(basis);

    // Write number of electrons
    chkpt.write("Nel",Nel);
    chkpt.write("Nel-a",Nel_alpha);
    chkpt.write("Nel-b",Nel_beta);

    // Write OMOs
    if(doCW) {
      if(oldrestr) {
	chkpt.cwrite("CWa",CW);
	chkpt.cwrite("CWb",CW);
      } else {
	chkpt.cwrite("CWa",CWa);
	chkpt.cwrite("CWb",CWb);
      }
    }

    // Solver
    SCF solver(basis,set,chkpt);

    // Handle guesses
    if(!doload) {
      if(guess==CORE_GUESS)
	solver.core_guess(sol);
      else if(guess==SAP_GUESS)
	solver.sap_guess(sol);
      else if(guess==GWH_GUESS)
	solver.gwh_guess(sol);
      else if(guess == GSAP_GUESS) {
        sol.Ha=solver.get_Hcore()+sap_guess(basis,set);
        sol.Hb=sol.Ha;
        solver.diagonalize(sol);
      } else if(guess == HUCKEL_GUESS) {
        sol.Ha=huckel_guess(basis,set);
        sol.Hb=sol.Ha;
        solver.diagonalize(sol);
      } else if((guess == SAD_GUESS) || (guess == NO_GUESS)) {
        // Form SAD density matrix
        sol.P=sad_guess(basis,set);

	// Build Fock operator from SAD density matrix. Get natural orbitals
	arma::vec occ;
	form_NOs(sol.P,solver.get_S(),sol.Ca,occ);
	sol.Cb=sol.Ca;

        if(guess==SAD_GUESS) {
	  // Avoid computing exact exchange with all NOs by truncating
	  // the occupations to the nonzero ones
	  {
	    double othr(sqrt(DBL_EPSILON));
	    arma::uword pos;
	    for(pos=0;pos<occ.n_elem;pos++)
	      if(occ(pos)<=othr)
		break;
            if(pos<occ.n_elem)
              occ.subvec(pos,occ.n_elem-1).zeros();
	  }

	  std::vector<double> goccs(arma::conv_to< std::vector<double> >::from(occ));
	  size_t maxiter(solver.get_maxiter());
	  solver.set_maxiter(0);
          rscf_t rsol;
          rsol.P=sol.P;
          rsol.C=sol.Ca;

	  if(hf || rohf)
	    solver.RHF(rsol,goccs,convthr);
	  else {
	    dft_t nonl(dft);
	    nonl.nl=false;
	    solver.RDFT(rsol,goccs,convthr,nonl);
	  }
	  solver.set_maxiter(maxiter);

	  // Do the diagonalization
	  solver.diagonalize(rsol,0.0);
	  sol.Ca=sol.Cb=rsol.C;
	}
      } else {
        throw std::logic_error("Guess not handled!\n");
      }
    }
    // Form density matrix
    sol.Pa=form_density(sol.Ca,occa);
    sol.Pb=form_density(sol.Cb,occb);
    sol.P=sol.Pa+sol.Pb;

    if(hf) {
      // Solve restricted Hartree-Fock
      solver.do_force(force);
      solver.UHF(sol,occa,occb,convthr);
    } else if(rohf) {
      // Solve restricted open-shell Hartree-Fock

      // Solve ROHF
      solver.ROHF(sol,occa,occb,convthr);

      // Set occupancies right
      get_unrestricted_occupancy(set,basis,occa,occb);
    } else {
      // Print information about used functionals
      if(verbose) {
	print_info(dft.x_func,dft.c_func);
	if(dft.nl)
	  printf("Using VV10 non-local correlation with b = % f, C = %f\n",dft.vv10_b,dft.vv10_C);
      }

      // Perdew-Zunger?
      bool pz=set.get_bool("PZ");
      int pzprec=set.get_int("PZprec");
      bool pzov=set.get_bool("PZov");
      bool pzoo=set.get_bool("PZoo");
      int pzloc=parse_pzimag(set.get_string("PZloc"));
      enum locmet pzlocmet=parse_locmet(set.get_string("PZlocmet"));
      double pzw=set.get_double("PZw");
      int pzmax=set.get_int("PZiter");
      double pzIthr=set.get_double("PZIthr");
      double pzOOthr=set.get_double("PZOOthr");
      double pzOVthr=set.get_double("PZOVthr");
      double pzEthr=set.get_double("PZEthr");
      double pzNRthr=set.get_double("PZNRthr");
      int pzimag=parse_pzimag(set.get_string("PZimag"));
      int pzstab=set.get_int("PZstab");
      double pzstabthr=-set.get_double("PZstabThr"); // Minus sign!
      int seed=set.get_int("PZseed");
      dft_t oodft(parse_pzmet(set.get_string("PZmode"),dft));
      pz_scaling_t pzscale(parse_pzscale(set.get_string("PZscale")));
      double pzscaleexp(set.get_double("PZscaleExp"));

      if(!pz) {
	if(dft.adaptive && (initdft.x_func>0 || initdft.c_func>0)) {
	  // Solve unrestricted DFT problem first on a rough grid
	  solver.UDFT(sol,occa,occb,initconvthr,initdft);

	  if(verbose) {
	    fprintf(stderr,"\n");
	    fflush(stderr);
	  }
	}
	// ... and then on the more accurate grid
	solver.do_force(force);
	solver.UDFT(sol,occa,occb,convthr,dft);

	// and update checkpoint file entries
	chkpt.write("Ca",sol.Ca);
	chkpt.write("Cb",sol.Cb);
	chkpt.write("Ea",sol.Ea);
	chkpt.write("Eb",sol.Eb);
	chkpt.write("Pa",sol.Pa);
	chkpt.write("Pb",sol.Pb);
	chkpt.write("P",sol.P);
	chkpt.write(sol.en);
	chkpt.write("Converged",1);


      } else {
	chkpt.write("Converged",0);

	// The localizing matrix
	arma::cx_mat Wa;
	if(chkpt.exist("CWa.re")) {
	  if(verbose) printf("Read localization matrix from checkpoint.\n");

	  // Get old localized orbitals
	  chkpt.cread("CWa",CWa);
	  // Save the orbitals
	  sol.cCa=CWa;

	  // Imaginary treatment?
	  if(rms_norm(arma::imag(sol.cCa))>IMAGTHR) {
	    if(pzimag==-1) {
	      if(verbose) printf("Norm of imaginary part %e, turning on imaginary dof.\n",rms_norm(arma::imag(sol.cCa)));
	      pzimag=1;
	    } else if(pzimag==0) {
	      throw std::runtime_error("Requesting calculation without imaginary dofs, but given orbitals have imaginary component!\n");
	    }
	    // Can't localize
	    pzloc=0;
	  } else {
	    // Store real part in orbital matrix
	    sol.Ca=arma::real(sol.cCa);
	    if(pzloc==-1)
	      // Assume orbitals are already localized
	      pzloc=0;
	    Wa.eye(Nel_alpha,Nel_alpha);
	  }
	} else { // No entry in checkpoint
	  sol.cCa=sol.Ca*COMPLEX1;
	  if(!pzoo) {
	    Wa.eye(Nel_alpha,Nel_alpha);
	  } else {
	    if(pzimag!=1)
	      Wa=real_orthogonal(Nel_alpha,seed)*std::complex<double>(1.0,0.0);
	    else
	      Wa=complex_unitary(Nel_alpha,seed);
	  }
	  if(pzloc==-1) pzloc=1;
	  sol.cCa.zeros(sol.Ca.n_rows,sol.Ca.n_cols);
	  sol.cCa.cols(0,Nel_alpha-1)=sol.Ca.cols(0,Nel_alpha-1)*Wa;
	  if(sol.Ca.n_cols>(size_t) Nel_alpha)
	    sol.cCa.cols(Nel_alpha,sol.Ca.n_cols-1)=sol.Ca.cols(Nel_alpha,sol.Ca.n_cols-1)*COMPLEX1;
	}

	if(pzloc==1) {
	  // Localize starting guess
	  if(verbose) printf("\nInitial localization.\n");
	  Timer tloc;
	  double measure;
	  // Max 1e5 iterations, gradient norm <= pzIthr
	  orbital_localization(pzlocmet,basis,sol.Ca.cols(0,Nel_alpha-1),sol.P,measure,Wa,verbose,pzimag!=1,1e5,pzIthr,DBL_MAX);
	  if(verbose) {
	    printf("\n");

	    fprintf(stderr,"%-64s %10.3f\n","    Initial localization",tloc.get());
	    fflush(stderr);
	  }

	  // Save the complex orbitals
	  sol.cCa.zeros(sol.Ca.n_rows,sol.Ca.n_cols);
	  sol.cCa.cols(0,Nel_alpha-1)=sol.Ca.cols(0,Nel_alpha-1)*Wa;
	  if(sol.Ca.n_cols>(size_t) Nel_alpha)
	    sol.cCa.cols(Nel_alpha,sol.Ca.n_cols-1)=sol.Ca.cols(Nel_alpha,sol.Ca.n_cols-1)*COMPLEX1;
	}

	// The localizing matrix
	arma::cx_mat Wb;
	if(chkpt.exist("CWb.re")) {
	  if(verbose) printf("Read localization matrix from checkpoint.\n");

	  // Get old localized orbitals
	  chkpt.cread("CWb",CWb);
	  // Save the orbitals
	  sol.cCb=CWb;

	  // Imaginary treatment?
	  if(rms_norm(arma::imag(sol.cCb))>IMAGTHR) {
	    if(pzimag==-1) {
	      if(verbose) printf("Norm of imaginary part %e, turning on imaginary dof.\n",rms_norm(arma::imag(sol.cCb)));
	      pzimag=1;
	    } else if(pzimag==0) {
	      throw std::runtime_error("Requesting calculation without imaginary dofs, but given orbitals have imaginary component!\n");
	    }
	    // Can't localize
	    pzloc=0;
	  } else {
	    // Store real part in orbital matrix
	    sol.Cb=arma::real(sol.cCb);
	    if(pzloc==-1)
	      // Assume orbitals are already localized
	      pzloc=0;
	    Wb.eye(Nel_beta,Nel_beta);
	  }
	} else { // No entry in checkpoint
	  sol.cCb=sol.Cb*COMPLEX1;

	  if(!pzoo) {
	    Wb.eye(Nel_beta,Nel_beta);
	  } else {
	    if(pzimag!=1)
	      Wb=real_orthogonal(Nel_beta,seed)*std::complex<double>(1.0,0.0);
	    else
	      Wb=complex_unitary(Nel_beta,seed);
	  }
	  if(pzloc==-1) pzloc=1;
	  sol.cCb.zeros(sol.Cb.n_rows,sol.Cb.n_cols);
	  if(Nel_beta>0)
	    sol.cCb.cols(0,Nel_beta-1)=sol.Cb.cols(0,Nel_beta-1)*Wb;
	  if(sol.Cb.n_cols>(size_t) Nel_beta)
	    sol.cCb.cols(Nel_beta,sol.Cb.n_cols-1)=sol.Cb.cols(Nel_beta,sol.Cb.n_cols-1)*COMPLEX1;
	}

	if(pzloc==1) {
	  // Localize starting guess
	  if(verbose) printf("\nInitial localization.\n");
	  Timer tloc;
	  double measure;
	  // Max 1e5 iterations, gradient norm <= pzIthr
	  orbital_localization(pzlocmet,basis,sol.Cb.cols(0,Nel_beta-1),sol.P,measure,Wb,verbose,pzimag!=1,1e5,pzIthr,DBL_MAX);
	  if(verbose) {
	    printf("\n");

	    fprintf(stderr,"%-64s %10.3f\n","    Initial localization",tloc.get());
	    fflush(stderr);
	  }

	  // Save the complex orbitals
	  sol.cCb.zeros(sol.Cb.n_rows,sol.Cb.n_cols);
	  sol.cCb.cols(0,Nel_beta-1)=sol.Cb.cols(0,Nel_beta-1)*Wb;
	  if(sol.Cb.n_cols>(size_t) Nel_beta)
	    sol.cCb.cols(Nel_beta,sol.Cb.n_cols-1)=sol.Cb.cols(Nel_beta,sol.Cb.n_cols-1)*COMPLEX1;
	}

	PZStability stab(&solver,verbose);
	stab.set_method(dft,oodft,pzw,pzscale,pzscaleexp);
	stab.set(sol);

	while(true) {
	  double dEo=0.0, dEv=0.0;
	  if(pzoo) {
	    // First, optimize the OO transformations
	    stab.set_params(true,pzimag==1,false,true);
	    dEo=stab.optimize(pzmax,pzOOthr,pzNRthr,pzEthr,pzprec);
	    sol=stab.get_usol();
	  }
	  if(pzov) {
	    // Then the OV transformations
	    stab.set_params(true,pzimag==1,true,false);
	    dEv=stab.optimize(pzmax,pzOVthr,0.0,pzEthr,pzprec);
	    sol=stab.get_usol();
	  }
	  if(dEo!=0.0 || dEv!=0.0)
	    continue;

	  // Check stability of OO rotations
	  if(pzoo && pzstab==-1) {
	    stab.set_params(true,pzimag==1,false,true);
	    bool instab=stab.check(true,pzstabthr);
	    if(instab) {
	      sol=stab.get_usol();
	      continue;
	    }

	    // Imaginary instability?
	    if(pzimag==-1) {
	      stab.set_params(false,true,false,true);
	      instab=stab.check(true,pzstabthr);
	      if(instab) {
		pzimag=1;
		stab.optimize(pzmax,pzOOthr,pzNRthr,pzEthr,pzprec);
		if(pzov) {
		  stab.set_params(false,true,true,false);
		  stab.optimize(pzmax,pzOVthr,pzNRthr,0.0,pzprec);
		}
		sol=stab.get_usol();
		continue;
	      }
	    }
	  } else if(pzstab==-2) {
	    // Check stability of OO+OV rotations
	    stab.set_params(true,pzimag==1,pzov,pzoo);
	    bool instab=stab.check(true,pzstabthr);
	    if(instab) {
	      sol=stab.get_usol();
	      continue;
	    }

	    // Imaginary instability?
	    if(pzimag==-1) {
	      stab.set_params(false,true,pzov,pzoo);
	      instab=stab.check(true,pzstabthr);
	      if(instab) {
		pzimag=1;
		stab.optimize(pzmax,pzOOthr,0.0,pzEthr,pzprec);
		sol=stab.get_usol();
		continue;
	      }
	    }
	  }
	  // No instabilities
	  break;
	}

	// Stability analysis?
	if(pzoo && pzstab==1) {
	  stab.set_params(true,pzimag!=0,false,true);
	  stab.check();
	}
	else if(pzstab==2) {
	  // Check stability of OO+OV rotations
	  stab.set_params(true,pzimag!=0,pzov,pzoo);
	  stab.check();
	}
      }

      // Calculation has converged.
      chkpt.write("Converged",1);
    }

    if(verbose) {
      population_analysis(basis,sol.Pa,sol.Pb);

      if(!set.get_string("Occupancies").size()) {
	arma::mat Ca(sol.Ca.cols(0,Nel_alpha-1));
	arma::mat Cb;
	if(Nel_beta)
	  Cb=sol.Cb.cols(0,Nel_beta-1);
	double S2(spin_S2(basis,Ca,Cb));
	printf("\nThe spin expectation value <S^2> is %f\n",S2);
      } else {
	printf("\nNot calculating spin expectation value since occupancies is set.\n");
      }
    }
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
  Svec=Svec.cols(Svec.n_cols-Nind,Svec.n_cols-1);

  // Form Sinvh
  arma::mat Sinvh(Ntot,Nind);
  for(size_t i=0;i<Nind;i++)
    Sinvh.col(i)=Svec.col(i)/sqrt(Sval(i));

  // Form the new C matrix.
  arma::mat C(Ntot,Nind);
  C.zeros();

  // The first vectors are simply the occupied states.
  for(size_t i=0;i<Nold;i++)
    for(size_t ish=0;ish<origshellidx.size();ish++)
      C.submat(augshells[origshellidx[ish]].get_first_ind(),i,augshells[origshellidx[ish]].get_last_ind(),i)=Cold.submat(origshells[ish].get_first_ind(),i,origshells[ish].get_last_ind(),i);

  // Determine the rest. Compute the overlap of the functions
  arma::mat X=arma::trans(Sinvh)*S*C.cols(0,Nold-1);
  // and perform SVD
  arma::mat U, V;
  arma::vec s;
  bool svdok=arma::svd(U,s,V,X);
  if(!svdok)
    throw std::runtime_error("SVD decomposition failed!\n");

  // Rotate eigenvectors.
  Sinvh=Sinvh*U;

  // Now, the subspace of the small basis set is found in the first
  // Nmo eigenvectors.
  C.cols(Nold,Nind-1)=Sinvh.cols(Nold,Nind-1);

  try {
    // Check orthogonality of orbitals
    check_orth(C,S,false);
  } catch(std::runtime_error & err) {
    std::ostringstream oss;
    oss << "Projected orbitals are not orthonormal. Please report this bug.";
    throw std::runtime_error(oss.str());
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
  // Perform the localization.
  for(size_t inuc=0;inuc<locno.size();inuc++) {
    if(locno[inuc]==0)
      continue;

    // The nucleus is located at
    coords_t cen=basis.get_nuclear_coords(inuc);

    // Compute moment integrals around the nucleus
    std::vector<arma::mat> momstack=basis.moment(2,cen.x,cen.y,cen.z);
    // Get matrix which transforms into occupied MO basis
    arma::mat transmat=C.cols(locd,nocc-1);

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
    C.cols(locd,nocc-1)=transmat*rvec;

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

arma::mat interpret_force(const arma::vec & f) {
  if(f.n_elem%3!=0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Error in intepret_force: expecting a vector containing forces, but given vector has " << f.n_elem << " elements!\n";
    throw std::runtime_error(oss.str());
  }

  arma::mat force(f);
  force.reshape(3,f.n_elem/3);
  return force;
}
