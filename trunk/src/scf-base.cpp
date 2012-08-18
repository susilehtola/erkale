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
#include "global.h"
#include "guess.h"
#include "linalg.h"
#include "mathf.h"
#include "properties.h"
#include "scf.h"
#include "stringutil.h"
#include "timer.h"
#include "trrh.h"

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

  maxiter=set.get_int("MaxIter");
  verbose=set.get_bool("Verbose");

  direct=set.get_bool("Direct");
  strictint=set.get_bool("StrictIntegrals");

  // Check update scheme
  if(useadiis && usebroyden) {
    ERROR_INFO();
    throw std::runtime_error("ADIIS and Broyden mixing cannot be used at the same time.\n");
  } 

  if(!usediis && !useadiis && !usebroyden && !usetrrh) {
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
    printf("\n");
    t.set();
  }

  Sinvh=BasOrth(S,set);
  
  if(verbose) {
    printf("Basis set diagonalized in %s.\n",t.elapsed().c_str());
    
    if(Sinvh.n_cols!=Sinvh.n_rows) {
      printf("%i linear combinations of basis functions have been removed.\n",Sinvh.n_rows-Sinvh.n_cols);
    }
    printf("\n");
  }

  if(densityfit) {
    // Density fitting.

    // Form density fitting basis
    BasisSet dfitbas;

    if(stricmp(set.get_string("DFTFittingBasis"),"Auto")==0)
      dfitbas=basisp->density_fitting();
    else {
      // Load basis library
      BasisSetLibrary fitlib;
      fitlib.load_gaussian94(set.get_string("DFTFittingBasis"));

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
	printf("Forming table of %lu ERIs, requiring %s of memory ... ",(long unsigned int) N,memory_size(N).c_str());
	fflush(stdout);
      }
      if(strictint)
	tab.fill(&basis);
      else
	// Don't compute small integrals
	tab.fill(STRICTTOL,&basis);
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

void ROHF_update(arma::mat & Fa_AO, arma::mat & Fb_AO, const arma::mat & P_AO, const arma::mat & S, int Nel_alpha, int Nel_beta, bool verbose) {
  /*
   * T. Tsuchimochi and G. E. Scuseria, "Constrained active space
   * unrestricted mean-field methods for controlling
   * spin-contamination", J. Chem. Phys. 134, 064101 (2011).
   */

  Timer t;

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

void calculate(const BasisSet & basis, Settings & set) {
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
    dft.gridtol=set.get_double("DFTFinalTol");

    initdft=dft;
    initdft.gridtol=set.get_double("DFTInitialTol");

    initconv.deltaEmax*=set.get_double("DFTDelta");
    initconv.deltaPmax*=set.get_double("DFTDelta");
    initconv.deltaPrms*=set.get_double("DFTDelta");
  }

  // Check consistency of parameters
  if(!hf && !rohf && exact_exchange(dft.x_func)!=0.0)
    if(set.get_bool("DFTFitting")) {
      printf("A hybrid functional is used, turning off density fitting.\n");
      set.set_bool("DFTFitting",false);
    }

  // Load starting guess?
  bool doload=(stricmp(loadname,"")!=0);
  BasisSet oldbas;
  bool oldrestr;
  arma::vec Eold, Eaold, Ebold;
  arma::mat Cold, Caold, Cbold;
  arma::mat Pold;
  
  // Use core guess?
  enum guess_t guess=parse_guess(set.get_string("Guess"));

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
	form_NOs(Pold,oldbas.overlap(),Cold,hlp);

	// Use alpha orbital energies
	Eold=Eaold;
      }
      
      basis.projectMOs(oldbas,Eold,Cold,sol.E,sol.C);
    } else if(guess == ATOMGUESS) {
      atomic_guess(basis,sol.C,sol.E,verbose);
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

    
    // Solver
    SCF solver(basis,set,chkpt);

    if(hf || rohf) {
      // Solve restricted Hartree-Fock
      solver.RHF(sol,occs,conv);
    } else {
      // Print information about used functionals
      if(verbose)
	print_info(dft.x_func,dft.c_func);
      // Solve restricted DFT problem first on a rough grid
      solver.RDFT(sol,occs,initconv,initdft);
      // .. and then on the final grid
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
      atomic_guess(basis,sol.Ca,sol.Ea,verbose);
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

    if(hf) {
      // Solve restricted Hartree-Fock
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
      // Solve restricted DFT problem first on a rough grid
      solver.UDFT(sol,occa,occb,initconv,initdft);
      // ... and then on the more accurate grid
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
    if(Sval(i)>=1e-5)
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
