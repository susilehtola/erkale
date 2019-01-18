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

/**
 * NOTE - THIS FILE HAS BEEN DEPRECATED IN FAVOR OF THE MORE MODERN,
 * GENERAL ROUTINES IN THE FILE co-opt.h AND WILL PROBABLY NOT EVEN
 * COMPILE, BUT IS INCLUDED FOR HISTORIC REFERENCE.
 */

/**
 * This file contains routines for the automatical formation of
 * application-specific basis sets.
 *
 * The algorithm performs completeness-optimization that was introduced in
 *
 * P. Manninen and J. Vaara, "Systematic Gaussian Basis-Set Limit
 * Using Completeness-Optimized Primitive Sets. A Case for Magnetic
 * Properties", J. Comp. Chem. 27, 434 (2006).
 *
 *
 * The automatic algorithm for generating primitive basis sets was
 * introduced in
 *
 * J. Lehtola, P. Manninen, M. Hakala, and K. Hämäläinen,
 * "Completeness-optimized basis sets: Application to ground-state
 * electron momentum densities", J. Chem. Phys. 137, 104105 (2012).
 *
 * and it was generalized to contracted basis sets in
 *
 * S. Lehtola, P. Manninen, M. Hakala, and K. Hämäläinen, "Contraction
 * of completeness-optimized basis sets: Application to ground-state
 * electron momentum densities", J. Chem. Phys. 138, 044109 (2013).
 */

#include <cstdio>
#include <iostream>
#include "completeness/optimize_completeness.h"
#include "completeness/completeness_profile.h"
#include "external/fchkpt_tools.h"
#include "emd/emd_gto.h"
#include "emd/emd.h"
#include "basislibrary.h"
#include "elements.h"
#include "global.h"
#include "guess.h"
#include "mathf.h"
#include "scf.h"
#include "timer.h"
#include "linalg.h"
#include "stringutil.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

extern "C" {
#include <gsl/gsl_multimin.h>
}

/// Debug polarization routine for angular momentum
//#define DEBUGPOL

/// Tolerance for EMD integrals
#define EMDTOL 1e-8

/// Tolerance for trace of density matrix
#define DMTRACETHR 1e-6

/// Dimer bond lengths in bohr and multiplicities
double dimR[]={0.0, 1.401, 5.612, 5.051, 4.000, 3.005, 2.348, 2.074, 2.282, 2.668, 5.858, 5.818, 7.351, 4.660, 4.244, 3.578, 3.570, 3.755, 7.102};
int dimM[]={0, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1};

/// Number of threads to use
int nthreads;
/// Amount of memory to use
std::string memory;
/// Moments to optimize
std::vector<size_t> moms;
/// Program to use
std::string prog;
/// Do the dimer calculations in the profile extension
bool calcdimer;
/// Do the middle trial?
bool domiddle;
/// Default co tolerance
double cotol;
/// Energy tolerance
double Etol;
/// Initial tolerance
double inittol;
/// Maximum angular momentum
int am_max;
/// Minimum value of polarization exponent
double minpol;

/// Completeness-optimization structure
typedef struct {
  /// Start of completeness profile
  double start;
  /// End of completeness profile
  double end;
  /// Tolerance
  double tol;

  /// The completeness-optimized exponents
  std::vector<double> exps;
} coprof_t;

// Maximum number of functions
#define NFMAX 40

/// Moment to optimize: area
#define OPTMOMIND 1

bool isscf(std::string method) {
  if(stricmp(prog,"gaussian")==0) {
    if(stricmp(method.substr(0,2),"RO")==0)
      method=method.substr(2);
    else if(stricmp(method.substr(0,1),"R")==0)
      method=method.substr(1);
    else if(stricmp(method.substr(0,1),"U")==0)
      method=method.substr(1);

    return ((stricmp(method,"hf")==0) || (stricmp(method,"rohf")==0));
  } else if(stricmp(prog,"psi4")==0)
    return (stricmp(method,"scf")==0);
  else if(stricmp(prog,"erkale")==0)
    return ((stricmp(method,"hf")==0) || (stricmp(method,"rohf")==0));
  else throw std::runtime_error("Unknown program!\n");
}

std::string scfmet() {
  if(stricmp(prog,"gaussian")==0)
    return "HF";
  else if(stricmp(prog,"psi4")==0)
    return "scf";
  else if(stricmp(prog,"erkale")==0)
    return "HF";
  else throw std::runtime_error("Unknown program!\n");
}

int maxam(const std::vector<coprof_t> & cpl) {
  for(size_t i=cpl.size()-1;i<cpl.size();i--)
    if(cpl[i].exps.size()>0)
      return (int) i;

  // Dummy statement
  return -1;
}

/// Move starting point of exponents
std::vector<double> move_exps(const std::vector<double> & exps, double start) {
  std::vector<double> ret(exps);
  double c=pow(10.0,start);
  for(size_t i=0;i<ret.size();i++)
    ret[i]*=c;

  return ret;
}

/// Check if element is a noble gas
bool isnoble(int Z) {
  for(size_t i=0;i<sizeof(magicno)/sizeof(magicno[0]);i++)
    if(Z==magicno[i])
      return true;

  return false;
}

/// Compare ease of calculating element: is Z1 easier than Z2?
bool compare_easiness(int Z1, int Z2) {
  // First - is either one a noble gas?
  if(isnoble(Z1) && !isnoble(Z2))
    return true;
  if(!isnoble(Z1) && isnoble(Z2))
    return false;
  if(isnoble(Z1) && isnoble(Z2))
    return Z1<Z2;

  // Is the dimer ground state a singlet?
  if(std::max(Z1,Z2)<(int) (sizeof(dimR)/sizeof(dimR[0]))) {
    if(dimM[Z1]==1 && dimM[Z2]!=1)
      return true;
    if(dimM[Z1]!=1 && dimM[Z2]==1)
      return false;
  }

  // Is the atomic ground state a singlet?
  gs_conf_t gs1=get_ground_state(Z1);
  gs_conf_t gs2=get_ground_state(Z2);

  if(gs1.mult==1 && gs2.mult!=1)
    return true;
  if(gs1.mult!=1 && gs2.mult==1)
    return false;

  // Sort by charge.
  return Z1<Z2;
}

/// l shells occupied up to atomam in a gas-phase atom
int atom_am(const std::vector<int> & els) {
  // Determine largest Z
  std::vector<int> elssort(els);
  std::sort(elssort.begin(),elssort.end());

  int atomam;
  if(elssort[elssort.size()-1]<5)
    atomam=0;
  else if(elssort[elssort.size()-1]<21)
    atomam=1;
  else if(elssort[elssort.size()-1]<57)
    atomam=2;
  else
    atomam=3;

  return atomam;
}

ElementBasisSet get_element_library(const std::string & el, const std::vector<coprof_t> & prof) {
  ElementBasisSet elbas(el);
  for(size_t am=0;am<prof.size();am++)
    for(size_t ish=0;ish<prof[am].exps.size();ish++) {
      FunctionShell sh(am);
      sh.add_exponent(1.0,prof[am].exps[ish]);
      elbas.add_function(sh);
    }
  elbas.sort();

  return elbas;
}

BasisSetLibrary get_library(const std::string & el, const std::vector<coprof_t> & prof) {
  // Construct the basis set library
  BasisSetLibrary baslib;
  baslib.add_element(get_element_library(el,prof));

  return baslib;
}

BasisSetLibrary get_library(const std::vector<int> & els, const std::vector<coprof_t> & prof) {
  BasisSetLibrary ret;
  for(size_t iel=0;iel<els.size();iel++)
    ret.add_element(get_element_library(element_symbols[els[iel]],prof));
  return ret;
}

BasisSetLibrary get_library(const std::string & el, const std::vector<coprof_t> & prof, const std::vector<arma::mat> & C, const std::vector<arma::vec> & exps, const std::vector<int> & contract) {
  // Construct the basis set library
  ElementBasisSet elbas(el);

  // Contraction of the same amount of functions that there are NOs
  // doesn't affect anything.
  std::vector<int> ncontr(contract);
  for(size_t l=0;l<ncontr.size();l++)
    if(C[l].n_cols==0 || ncontr[l]<=(int) C[l].n_cols)
      ncontr[l]=0;

  // Contractions
  for(size_t am=0;am<ncontr.size();am++) {
    int nfree=(int) prof[am].exps.size()-ncontr[am];

    // Add contracted functions
    if(nfree<(int) prof[am].exps.size()) {
      for(size_t ic=0;ic<C[am].n_cols;ic++) {
	FunctionShell sh(am);
	for(int iexp=0;iexp<(int) ncontr[am];iexp++) {
	  double cv=C[am](iexp,ic);
	  double zv=exps[am](iexp);
	  sh.add_exponent(cv,zv);
	}
	elbas.add_function(sh);
      }
    }
    // and free primitives
    for(int iexp=ncontr[am];iexp<(int) exps[am].n_elem;iexp++) {
      FunctionShell sh(am);
      sh.add_exponent(1.0,exps[am](iexp));
      elbas.add_function(sh);
    }
  }

  // Higher functions
  for(size_t am=ncontr.size();am<prof.size();am++)
    for(size_t ish=0;ish<prof[am].exps.size();ish++) {
      FunctionShell sh(am);
      sh.add_exponent(1.0,prof[am].exps[ish]);
      elbas.add_function(sh);
    }
  elbas.sort();
  elbas.normalize();
  //  elbas.print();

  BasisSetLibrary baslib;
  baslib.add_element(elbas);

  return baslib;
}

arma::vec compute_value_psi(int Z, bool dimer, std::vector<coprof_t> & cpl, const std::string & method) {
  // Construct the basis set
  BasisSetLibrary baslib=get_library(element_symbols[Z],cpl);

  // Save the basis set. First initialize the file
  const std::string basfile="basis.gbs";
  FILE *out=fopen(basfile.c_str(),"w");
  fprintf(out,"[cobas]\n");
  fprintf(out,"spherical\n");
  fprintf(out,"****\n");
  fclose(out);
  // Then write out the basis set, appending the file
  baslib.save_gaussian94(basfile,true);

  /*
  // Do higher level calculations on hydrogen always with the dimer.
  if(Z==1 && stricmp(method,scfmet())!=0)
  dimer=true;
  */

  // If solving single atom, determine multiplicity
  int mult;
  if(!dimer) {
    // Determine ground state
    gs_conf_t gs=get_ground_state(Z);
    // Set multiplicity
    mult=gs.mult;
  } else {
    // Solve dimer, set multiplicity
    mult=dimM[Z];
  }

  // Write PSI4 input file.
  out=fopen("input.dat","w");

  // Assign memory
  fprintf(out,"memory %s\n",memory.c_str());
  // Assign molecules
  fprintf(out,"molecule mol {\n");
  fprintf(out,"\t0 %i\n",mult);
  fprintf(out,"\t%-2s 0.0 0.0 0.0\n",element_symbols[Z].c_str());
  if(dimer)
    fprintf(out,"\t%-2s 0.0 0.0 %f\n",element_symbols[Z].c_str(),dimR[Z]/ANGSTROMINBOHR);
  fprintf(out,"}\n\n");

  // Set reference
  if(mult!=1) {
    fprintf(out,"set scf {\n");
    // UHF reference
    //    fprintf(out,"reference uhf\n");
    // ROHF reference
    fprintf(out,"reference rohf\n");
    fprintf(out,"}\n\n");
  }


  // Set basis to use
  fprintf(out,"basis file %s\n",basfile.c_str());
  fprintf(out,"set basis cobas\n\n");

  // Compute the energy
  fprintf(out,"set maxiter %i\n",300);
  fprintf(out,"en = energy('%s')\n\n",method.c_str());

  // State output file
  fprintf(out,"f = open('result.dat','w')\n");
  fprintf(out,"f.write('%%.16e\\n' %% en);\n");
  fprintf(out,"f.close()\n");

  // Close the input file
  fclose(out);

  // Run Psi4
  char cmd[80];
  sprintf(cmd,"psi4 -n %i",nthreads);
  int psival=system(cmd);
  if(psival)
    throw std::runtime_error("Error running psi4.\n");

  // Get the energy
  arma::vec ret(1);
  int readval;
  FILE *in=fopen("result.dat","r");
  readval=fscanf(in,"%le",&ret(0));
  fclose(in);

  if(readval!=1)
    throw std::runtime_error("Error reading result of calculation.\n");

  /*
    printf("Composition:\n");
    for(int am=0;am<=maxam(cpl);am++)
    printf("%c % .2f % .2f %2i\n",shell_types[am],cpl[am].start,cpl[am].end,(int) cpl[am].exps.size());
    printf("=====> %.10e\n",ret);
  */

  return ret;
}


void get_NOs(const arma::mat & P, const arma::mat & S, arma::mat & Pvec, arma::vec & Pval) {
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
  eig_sym_ordered(Pval,Pvec,P_orth);

  // Transform NOs to AO basis.
  Pvec=Sd*Pvec;
}

void compute_density_gaussian(int Z, bool dimer, const BasisSetLibrary & baslib, std::string method, BasisSet & basis, arma::mat & P, int & Nel, double & E) {
  // Save the basis set. First initialize the file
  const std::string basfile="basis.gbs";
  baslib.save_gaussian94(basfile);

  // If solving single atom, determine multiplicity
  int mult;
  if(!dimer) {
    // Determine ground state
    gs_conf_t gs=get_ground_state(Z);
    // Set multiplicity
    mult=gs.mult;
  } else {
    // Solve dimer, set multiplicity
    mult=dimM[Z];
  }

  bool rohf=false;
  enum POLMETHOD {
    RHF,
    UHF,
    ROHF
  } polmethod;

  // Check if polarization method has been given
  if(stricmp(method.substr(0,2),"RO")==0)
    polmethod=ROHF;
  else if(stricmp(method.substr(0,1),"R")==0)
    polmethod=RHF;
  else if(stricmp(method.substr(0,1),"U")==0)
    polmethod=UHF;
  else {
    // Check multiplicity to define the method
    if(mult==1)
      polmethod=RHF;
    else
      polmethod=UHF;
  }

  // Check if we use ROHF
  if(polmethod==ROHF) {
    method="HF";
    // Do we need to actually run ROHF?
    if(mult==1)
      polmethod=RHF;
  }

  // Do MP2 always in unrestricted mode
  if(stricmp(method.substr(0,3),"MP2")==0) {
    polmethod=UHF;
  }

  // Write G09 input file.
  FILE *out;

  out=fopen("input.com","w");

  // Write the headers
  fprintf(out,"%%RWF=temp\n%%Int=temp\n%%D2E=temp\n%%NoSave\n%%Chk=chkpt\n");

  // Memory option.
  fprintf(out,"%%Mem=%s\n",memory.c_str());
  fprintf(out,"%%NProcShared=%i\n",nthreads);
  fprintf(out,"#P GFINPUT Density=Current SCF=(XQC,IntRep,NoIncFock,DSymm,NoVarAcc) Integral=(Acc2E=12,Grid=450974)\n");

  // Redo calculation until a real minimum is reached. Only look at
  // internal instabilities (keep the same constraints, no RHF -> UHF)
  //  fprintf(out,"#SP, %s/Gen\n",method.c_str());
  if(polmethod == RHF)
    fprintf(out,"#Stable=(Opt,RRHF,XQC), %s/Gen\n",method.c_str());
  else
    fprintf(out,"#Stable=(Opt,RUHF,XQC), %s/Gen\n",method.c_str());

  // Comment
  fprintf(out,"\ncompleteness optimization\n\n");

  // Charge and spin
  fprintf(out,"%i %i\n",0,mult);
  // Write the atoms
  fprintf(out,"%-2s 0.0 0.0 0.0\n",element_symbols[Z].c_str());
  if(dimer)
    fprintf(out,"%-2s 0.0 0.0 %f\n",element_symbols[Z].c_str(),dimR[Z]/ANGSTROMINBOHR);

  // Set basis to use
  fprintf(out,"\n@%s\n\n",basfile.c_str());

  // Close the input file
  fclose(out);

  // Run Gaussian
  char cmd[80];
  sprintf(cmd,"g09 input.com");
  int g09val=system(cmd);

  if(g09val) {
    printf("Error running Gaussian '09. Trying again.\n");
    g09val=system(cmd);
  }

  if(g09val)
    throw std::runtime_error("Error running Gaussian '09.\n");

  // Do we need to do ROHF?
  if(polmethod==ROHF) {
    out=fopen("input.com","w");

    // Write the headers
    fprintf(out,"%%RWF=temp\n%%Int=temp\n%%D2E=temp\n%%NoSave\n%%Chk=chkpt\n");

    // Memory option.
    fprintf(out,"%%Mem=%s\n",memory.c_str());
    fprintf(out,"%%NProcShared=%i\n",nthreads);

    fprintf(out,"#P GFINPUT Density=Current Guess=Read SCF=(DSymm,IntRep,NoIncFock,NoVarAcc,MaxCycle=300) Integral=(Acc2E=12)\n");
    fprintf(out,"#SP, ROHF/Gen\n");

    // Comment
    fprintf(out,"\ncompleteness optimization\n\n");

    // Charge and spin
    fprintf(out,"%i %i\n",0,mult);
    // Write the atoms
    fprintf(out,"%-2s 0.0 0.0 0.0\n",element_symbols[Z].c_str());
    if(dimer)
      fprintf(out,"%-2s 0.0 0.0 %f\n",element_symbols[Z].c_str(),dimR[Z]/ANGSTROMINBOHR);

    // Set basis to use
    fprintf(out,"\n@%s\n\n",basfile.c_str());

    // Close the input file
    fclose(out);

    // Run Gaussian
    sprintf(cmd,"g09 input.com");
    g09val=system(cmd);
  }
  if(g09val)
    throw std::runtime_error("Error running ROHF.\n");

  // Do we need to do a post-HF calculation?
  if(!isscf(method)) {
    out=fopen("input.com","w");

    // Write the headers
    fprintf(out,"%%RWF=temp\n%%Int=temp\n%%D2E=temp\n%%NoSave\n%%Chk=chkpt\n");

    // Memory option.
    fprintf(out,"%%Mem=%s\n",memory.c_str());
    fprintf(out,"%%NProcShared=%i\n",nthreads);

    fprintf(out,"#P GFINPUT Density=Current Guess=Read SCF=(XQC,DSymm,IntRep,NoIncFock,NoVarAcc,MaxCycle=300) Integral=(Acc2E=12)\n");
    fprintf(out,"#SP, %s/Gen\n",method.c_str());

    // Comment
    fprintf(out,"\ncompleteness optimization\n\n");

    // Charge and spin
    fprintf(out,"%i %i\n",0,mult);
    // Write the atoms
    fprintf(out,"%-2s 0.0 0.0 0.0\n",element_symbols[Z].c_str());
    if(dimer)
      fprintf(out,"%-2s 0.0 0.0 %f\n",element_symbols[Z].c_str(),dimR[Z]/ANGSTROMINBOHR);

    // Set basis to use
    fprintf(out,"\n@%s\n\n",basfile.c_str());

    // Close the input file
    fclose(out);

    // Run Gaussian
    sprintf(cmd,"g09 input.com");
    g09val=system(cmd);
  }

  if(g09val)
    throw std::runtime_error("Error running post-HF Gaussian '09.\n");

  // Convert the checkpoint file into formatted form
  sprintf(cmd,"formchk chkpt.chk chkpt.fchk &> formchk.log");
  g09val=system(cmd);
  if(g09val)
    throw std::runtime_error("Error converting checkpoint file.\n");

  // Load the converted checkpoint file
  Storage stor=parse_fchk("chkpt.fchk");

  // Construct basis set
  basis=form_basis(stor);

  // Construct density matrix
  P=form_density(stor);

  arma::mat S=basis.overlap();
  // Work around bug in Gaussian
  if(rohf || stricmp(method,"ROHF")==0) {
    arma::mat Ca=form_orbital(stor,"Alpha MO coefficients");

    int Nela=stor.get_int("Number of alpha electrons");
    int Nelb=stor.get_int("Number of beta electrons");

    P.zeros();
    for(int i=0;i<Nela;i++)
      P+=Ca.col(i)*arma::trans(Ca.col(i));
    for(int i=0;i<Nelb;i++)
      P+=Ca.col(i)*arma::trans(Ca.col(i));
  }

  // Number of electrons is
  Nel=stor.get_int("Number of electrons");

  // Check that density matrix is sane
  double NelP=arma::trace(P*basis.overlap());
  if(fabs(NelP-Nel)>=DMTRACETHR) {
    printf("Density matrix contains %.6f electrons but system should have %i! Renormalizing.\n",NelP,Nel);
    //    throw std::runtime_error("Density matrix error.\n");
    // Scale density matrix
    P*=Nel/NelP;
  }

  // And energy is
  E=stor.get_double("Total Energy");

  // Remove work files
  remove("input.com");
  remove("input.log");
  remove("chkpt.chk");
  remove("chkpt.fchk");
  remove("formchk.log");
  int rmval=system("rm -f Gau-*.scr");
  if(rmval)
    throw std::runtime_error("Error removing files.\n");
}

void compute_density_erkale(int Z, bool dimer, const BasisSetLibrary & baslib, const std::string & method, BasisSet & basis, arma::mat & P, int & Nel, double & E) {
  // Save the basis set. First initialize the file
  const std::string basfile="basis.gbs";
  baslib.save_gaussian94(basfile);

  // Write the atoms
  FILE *out=fopen("atoms.xyz","w");
  if(dimer)
    fprintf(out,"2\n");
  else
    fprintf(out,"1\n");
  fprintf(out,"Completeness optimization atoms\n");
  fprintf(out,"%-2s 0.0 0.0 0.0\n",element_symbols[Z].c_str());
  if(dimer)
    fprintf(out,"%-2s 0.0 0.0 %f\n",element_symbols[Z].c_str(),dimR[Z]/ANGSTROMINBOHR);
  fclose(out);

  // If solving single atom, determine multiplicity
  int mult;
  if(!dimer) {
    // Determine ground state
    gs_conf_t gs=get_ground_state(Z);
    // Set multiplicity
    mult=gs.mult;
  } else {
    // Solve dimer, set multiplicity
    mult=dimM[Z];
  }

  // Write erkale input file. First broyden
  out=fopen("input.erk","w");
  fprintf(out,"System atoms.xyz\n");
  fprintf(out,"Method %s\n",method.c_str());
  fprintf(out,"Multiplicity %i\n",mult);
  fprintf(out,"Basis basis.gbs\n");
  fprintf(out,"SaveChk erkale.chk\n");
  fprintf(out,"Guess Core\n");
  fprintf(out,"Maxiter 300\n");
  fprintf(out,"Direct true\n");
  //  fprintf(out,"StrictIntegrals true\n");
  fprintf(out,"UseADIIS false\n");
  fprintf(out,"UseTRRH false\n");
  fprintf(out,"UseDIIS false\n");
  fprintf(out,"UseBroyden true\n");
  fclose(out);

  char cmd[80];
#ifdef _OPENMP
  sprintf(cmd,"erkale_omp input.erk &> erkale.stdout");
#else
  sprintf(cmd,"erkale input.erk &> erkale.stdout");
#endif

  int erkval=system(cmd);


  if(erkval) {
    fprintf(stderr,"Error running ERKALE. Trying again with TRRH.\n");
    out=fopen("input.erk","w");
    fprintf(out,"System atoms.xyz\n");
    fprintf(out,"Method %s\n",method.c_str());
    fprintf(out,"Multiplicity %i\n",mult);
    fprintf(out,"Basis basis.gbs\n");
    fprintf(out,"SaveChk erkale.chk\n");
    fprintf(out,"Guess Core\n");
    fprintf(out,"UseDIIS false\n");
    fprintf(out,"UseADIIS false\n");
    fprintf(out,"UseBroyden false\n");
    fprintf(out,"UseTRRH true\n");
    fprintf(out,"Maxiter 300\n");
    fprintf(out,"Verbose true\n");
    fprintf(out,"Direct true\n");
    // fprintf(out,"StrictIntegrals true\n");
    fclose(out);

    erkval=system(cmd);
  }

  if(erkval) {
    fprintf(stderr,"Error running ERKALE. Trying again with DIIS.\n");
    out=fopen("input.erk","w");
    fprintf(out,"System atoms.xyz\n");
    fprintf(out,"Method %s\n",method.c_str());
    fprintf(out,"Multiplicity %i\n",mult);
    fprintf(out,"Basis basis.gbs\n");
    fprintf(out,"SaveChk erkale.chk\n");
    fprintf(out,"Guess Core\n");
    fprintf(out,"UseDIIS true\n");
    fprintf(out,"UseADIIS true\n");
    fprintf(out,"UseBroyden false\n");
    fprintf(out,"UseTRRH false\n");
    fprintf(out,"Maxiter 300\n");
    fprintf(out,"Direct true\n");
    //    fprintf(out,"StrictIntegrals true\n");
    fprintf(out,"Verbose true\n");
    fclose(out);

    erkval=system(cmd);
  }

  if(erkval) {
    fprintf(stderr,"Error running ERKALE. Trying again with DIIS and line search.\n");
    out=fopen("input.erk","w");
    fprintf(out,"System atoms.xyz\n");
    fprintf(out,"Method %s\n",method.c_str());
    fprintf(out,"Multiplicity %i\n",mult);
    fprintf(out,"Basis basis.gbs\n");
    fprintf(out,"SaveChk erkale.chk\n");
    fprintf(out,"Guess Core\n");
    fprintf(out,"UseDIIS true\n");
    fprintf(out,"UseADIIS false\n");
    fprintf(out,"UseBroyden false\n");
    fprintf(out,"UseTRRH false\n");
    fprintf(out,"LineSearch true\n");
    fprintf(out,"Maxiter 300\n");
    fprintf(out,"Direct true\n");
    fprintf(out,"StrictIntegrals true\n");
    fprintf(out,"Verbose true\n");
    fclose(out);

    erkval=system(cmd);
  }

  if(erkval) throw std::runtime_error("Error running ERKALE.\n");

  // Load the checkpoint file
  Checkpoint chkpt("erkale.chk",false);

  // Construct basis set
  chkpt.read(basis);

  // Construct density matrix
  chkpt.read("P",P);

  // Number of electrons is
  chkpt.read("Nel",Nel);

  // And energy is
  energy_t en;
  chkpt.read(en);
  E=en.E;

  // Remove work files
  remove("basis.gbs");
  remove("input.erk");
  remove("erkale.log");
  remove("erkale.stdout");
  remove("erkale.chk");
}

void compute_density(int Z, bool dimer, const BasisSetLibrary & baslib, const std::string & method, BasisSet & basis, arma::mat & P, int & Nel, double & E) {
  // Don't try to calculate nonexisting dimers
  if(dimer && (Z>(int) (sizeof(dimR)/sizeof(dimR[0])))) {
    P.zeros(basis.get_Nbf(),basis.get_Nbf());
    Nel=0;
    E=0.0;
    return;
  }

  if(stricmp(prog,"gaussian")==0)
    compute_density_gaussian(Z,dimer,baslib,method,basis,P,Nel,E);
  else if(stricmp(prog,"erkale")==0)
    compute_density_erkale(Z,dimer,baslib,method,basis,P,Nel,E);
  else throw std::runtime_error("Unknown program.");
}

void compute_density(int Z, bool dimer, const std::vector<coprof_t> & cpl, const std::string & method, BasisSet & bas, arma::mat & P, int & Nel, double & E) {
  // Construct the basis set
  BasisSetLibrary baslib=get_library(element_symbols[Z],cpl);
  compute_density(Z,dimer,baslib,method,bas,P,Nel,E);
}

// Get the contraction matrix
void get_contraction(int Z, const std::vector<coprof_t> & cpl, const std::string & method, std::vector<arma::mat> & contr, std::vector<arma::vec> & exps) {
  /* First, run the calculation. */

  BasisSet basis;
  arma::mat P;
  int Nel;
  double E;

  // Get the density matrix
  compute_density(Z,false,cpl,method,basis,P,Nel,E);

  // Check that density matrix is OK
  arma::mat S=basis.overlap();
  if(fabs(arma::trace(P*S)-Nel)>DMTRACETHR) {
    printf("Trace of density matrix differs from Nel by %e.\n",arma::trace(P*S)-Nel);
    throw std::runtime_error("Error with density matrix.\n");
  }

  // Then, compute r_{rms}=\sqrt{<r^2>} around center
  std::vector<arma::mat> mom2=basis.moment(2,0.0,0.0,0.0);
  arma::mat rsqop=mom2[getind(2,0,0)]+mom2[getind(0,2,0)]+mom2[getind(0,0,2)];

  // Collect s and p type orbitals. First, get the shells in the basis set.
  std::vector<GaussianShell> shells=basis.get_shells();

  // Determine amount of s, p and d electrons
  std::vector<int> nel=shell_count(Z);

  // Returned contractions
  contr.resize(nel.size());
  exps.resize(nel.size());

  for(size_t l=0;l<nel.size();l++) {
    if(nel[l]==0)
      continue;

    // Get shells with given l
    std::vector<GaussianShell> lsh;
    for(size_t i=0;i<shells.size();i++)
      if(shells[i].get_am()==(int) l)
	lsh.push_back(shells[i]);

    // Get density and overlap in this subspace
    int nf=2*l+1;
    std::vector<arma::mat> Psub(nf);
    std::vector<arma::mat> Ssub(nf);
    std::vector<arma::mat> rsqsub(nf);
    for(int f=0;f<nf;f++) {
      Psub[f].zeros(lsh.size(),lsh.size());
      Ssub[f].zeros(lsh.size(),lsh.size());
      rsqsub[f].zeros(lsh.size(),lsh.size());
    }

    // Collect elements
    for(int ii=0;ii<nf;ii++)
      for(size_t i=0;i<lsh.size();i++)
	for(size_t j=0;j<lsh.size();j++) {
	  Psub[ii](i,j)=P(lsh[i].get_first_ind()+ii,lsh[j].get_first_ind()+ii);
	  Ssub[ii](i,j)=S(lsh[i].get_first_ind()+ii,lsh[j].get_first_ind()+ii);
	  rsqsub[ii](i,j)=rsqop(lsh[i].get_first_ind()+ii,lsh[j].get_first_ind()+ii);
	}

    // Get exponents
    exps[l].zeros(lsh.size());
    for(size_t i=0;i<lsh.size();i++)
      exps[l](i)=lsh[i].get_contr()[0].z;

    // Diagonalize density matrix in these subspaces
    std::vector<arma::mat> Pvec(nf);
    std::vector<arma::vec> Pval(nf);
    for(int ii=0;ii<nf;ii++) {
      get_NOs(Psub[ii],Ssub[ii],Pvec[ii],Pval[ii]);

      // Get rid of extra entries
      Pvec[ii]=Pvec[ii].submat(0,Pvec[ii].n_cols-nel[l],Pvec[ii].n_rows-1,Pvec[ii].n_cols-1);
      Pval[ii]=Pval[ii].subvec(Pval[ii].n_elem-nel[l],Pval[ii].n_elem-1);
    }

    // Determine RMS radius of NOs
    std::vector<arma::vec> rmsrad(nf);
    for(int i=0;i<nf;i++) {
      rmsrad[i].zeros(Pvec[i].n_cols);
      for(size_t j=0;j<Pvec[i].n_cols;j++)
	rmsrad[i](j)=sqrt(arma::as_scalar(arma::trans(Pvec[i].col(j))*rsqsub[i]*Pvec[i].col(j)));
    }

    // Average the coefficients: c(nx,nc)
    arma::mat c(Pvec[0].n_rows,nel[l]);
    c.zeros();

    // Loop over orbitals
    for(int io=0;io<nel[l];io++) {
      // Add up alternate contractions.
      for(int j=0;j<nf;j++) {
	// Determine element with maximum absolute value
	double maxabs=0.0;
	for(size_t k=0;k<Pvec[j].n_rows;k++)
	  if(fabs(Pvec[j](k,io))>fabs(maxabs))
	    maxabs=Pvec[j](k,io);

	// Only the ratios between the coefficients are important, so
	// perform the averaging as.
	c.col(io)+=Pval[j](io)*Pvec[j].col(io)/maxabs;
      }
      // Normalize the coefficients
      c.col(io)/=sqrt(arma::as_scalar(arma::trans(c.col(io))*Ssub[0]*c.col(io)));
    }

    // Store the contractions.
    contr[l]=c;

    printf("l = %i eigenvalues:\n",(int) l);
    for(int i=0;i<nf;i++) {
      printf("\t%i",i+1);
      for(size_t j=Pval[i].n_elem-1;j<Pval[i].n_elem;j--)
	printf("  %.3e (%1.4f)",rmsrad[i](j),Pval[i](j));
      printf("\n");
    }
    printf("Average rms radii:");
    for(size_t i=c.n_cols-1;i<c.n_cols;i--)
      printf(" %.3e",sqrt(arma::as_scalar(arma::trans(c.col(i))*rsqsub[0]*c.col(i))));
    printf("\n");
  }
}


arma::vec compute_value(int Z, bool dimer, const BasisSetLibrary & baslib, const std::string & method) {
  // Don't try to calculate nonexisting dimers. Return dummy value
  if(dimer && (Z>(int) (sizeof(dimR)/sizeof(dimR[0])))) {
    arma::vec ret(8);
    ret.zeros();
    return ret;
  }

  // Basis set and density
  BasisSet basis;
  arma::mat P;
  int Nel;
  double E;

  // Compute the density and other stuff
  compute_density(Z,dimer,baslib,method,basis,P,Nel,E);

  // EMD evaluator
  GaussianEMDEvaluator eval(basis,P);
  EMD emd(&eval, Nel);
  emd.initial_fill(false);
  emd.find_electrons(false);
  emd.optimize_moments(false,EMDTOL);

  // Get moments
  arma::mat mom=emd.moments();
  // Return value: moments
  arma::vec ret(mom.n_rows+1);
  // Plug in energy as well
  ret.subvec(0,mom.n_rows-1)=mom.col(1);
  ret(mom.n_rows)=E;

  return ret;
}

arma::vec compute_value(int Z, bool dimer, const std::vector<coprof_t> & cpl, const std::string & method) {
  // Construct the basis set
  BasisSetLibrary baslib=get_library(element_symbols[Z],cpl);
  return compute_value(Z,dimer,baslib,method);
}

double compute_mog(const arma::vec & val, const arma::vec & ref) {
  // Find maximum error in moments
  double maxerr=0.0;
  for(size_t i=0;i<moms.size();i++) {
    double err=fabs((val(moms[i])-ref(moms[i]))/ref(moms[i]));
    if(!std::isnormal(err) && err!=0) {
      //      printf("moms[%i]=%i gives unnormal mog %e: val=%e, ref=%e.\n",(int) i, (int) moms[i], err, val(moms[i]), ref(moms[i]));
      throw std::runtime_error("MOG error.\n");
    }
    if(err>maxerr)
      maxerr=err;
  }

  // Do energy as well?
  if(Etol>0) {
    // Index where to find energy
    size_t idx=val.n_elem-1;
    // Change in energy
    double Eerr=fabs(val(idx)-ref(idx));
    // Convert to same scale as moments
    double err=Eerr/Etol*inittol;

    if(err>maxerr)
      maxerr=err;
  }

  return maxerr;
}

double compute_mog(const std::vector<arma::vec> & val, const std::vector<arma::vec> & ref) {
  // Find maximum error in moments
  double maxerr=0.0;
  for(size_t isys=0;isys<val.size();isys++) {
    double err=compute_mog(val[isys],ref[isys]);
    if(err>maxerr)
      maxerr=err;
  }

  return maxerr;
}

void update_values(std::vector< std::vector<arma::vec> > & curval, std::vector<coprof_t> & cpl, const std::vector<int> & els, const std::string & method, bool dodimer=true, bool verbose=true) {
  std::vector<std::string> systems(2);
  systems[0]="monomer";
  systems[1]="  dimer";

  size_t max=1;
  if(dodimer)
    max=2;

  if(curval.size()!=max)
    curval.resize(max);

  for(size_t isys=0;isys<max;isys++) {
    if(curval[isys].size()!=els.size())
      curval[isys].resize(els.size());

    for(size_t i=0;i<els.size();i++) {
      Timer tc;
      if(verbose) {
	printf("Updating value for %-2s %s %s ... ",element_symbols[els[i]].c_str(),tolower(method).c_str(),systems[isys].c_str());
	fflush(stdout);
      }
      curval[isys][i]=compute_value(els[i],(bool) isys,cpl,method);
      if(verbose) {
	printf("done (%s)\n",tc.elapsed().c_str());
	fflush(stdout);
      }
    }
  }
}

void print_values(const std::vector< std::vector<arma::vec> > & curval, const std::vector<int> & els, std::vector< std::vector<FILE *> > & out) {
  // Print values
  for(size_t isys=0;isys<curval.size();isys++)
    for(size_t iel=0;iel<curval[isys].size();iel++) {
      for(size_t n=0;n<curval[isys][iel].n_elem;n++)
	fprintf(out[isys][iel],"%e ",curval[isys][iel](n));
      fprintf(out[isys][iel],"\n");
      fflush(out[isys][iel]);
    }
}

std::vector<coprof_t> generate_limits_old(const std::vector<int> & els, const std::string & basislib, double tol) {
  // Make initial completeness profile.

  // Limit for completeness
  const double cplthr=0.99;

  // Returned array
  std::vector<coprof_t> cpl(max_am+1);
  for(size_t i=0;i<cpl.size();i++) {
    cpl[i].start=0.0;
    cpl[i].end=0.0;
    cpl[i].tol=tol;
  }

  double clows[max_am+1];
  double chighs[max_am+1];
  double dlows[max_am+1];
  double dhighs[max_am+1];

  for(int i=0;i<=max_am;i++) {
    clows[i]=DBL_MAX;
    dlows[i]=DBL_MAX;
    chighs[i]=-DBL_MAX;
    dhighs[i]=-DBL_MAX;
  }

  // Load basis set library
  BasisSetLibrary bas;
  bas.load_basis(basislib);

  // Loop over elements.
  for(size_t iel=0;iel<els.size();iel++) {
    // Get elemental basis
    ElementBasisSet elbas=bas.get_element(element_symbols[els[iel]]);
    // Decontract basis
    ElementBasisSet decbas(elbas);
    decbas.decontract();

    // Compute completeness profile
    compprof_t prof=compute_completeness(elbas);
    compprof_t dprof=compute_completeness(decbas);

    // Loop over angular momentum
    for(size_t am=0;am<prof.shells.size();am++) {
      // Determine lower limit
      double clow=DBL_MAX;
      for(size_t ia=0;ia<prof.lga.size();ia++)
	if(prof.shells[am].Y[ia]>=cplthr) {
	  clow=prof.lga[ia];
	  break;
	}

      double dlow=DBL_MAX;
      for(size_t ia=0;ia<dprof.lga.size();ia++)
	if(dprof.shells[am].Y[ia]>=cplthr) {
	  dlow=dprof.lga[ia];
	  break;
	}

      // Determine upper limit
      double chigh=-DBL_MAX;
      for(size_t ia=prof.lga.size()-1;ia<prof.lga.size();ia--)
	if(prof.shells[am].Y[ia]>=cplthr) {
	  chigh=prof.lga[ia];
	  break;
	}

      double dhigh=-DBL_MAX;
      for(size_t ia=dprof.lga.size()-1;ia<dprof.lga.size();ia--)
	if(dprof.shells[am].Y[ia]>=cplthr) {
	  dhigh=dprof.lga[ia];
	  break;
	}

      if(clows[am]>clow)
	clows[am]=clow;
      if(chighs[am]<chigh)
	chighs[am]=chigh;

      if(dlows[am]>dlow)
	dlows[am]=dlow;
      if(dhighs[am]<dhigh)
	dhighs[am]=dhigh;
    }
  }

  // Renormalize limits
  for(int i=0;i<=max_am;i++) {
    if(clows[i]==DBL_MAX)
      clows[i]=0.0;
    if(dlows[i]==DBL_MAX)
      dlows[i]=0.0;
    if(chighs[i]==-DBL_MAX)
      chighs[i]=0.0;
    if(dhighs[i]==-DBL_MAX)
      dhighs[i]=0.0;
  }

  // Set limits
  for(int am=0;am<=max_am;am++) {
    /*
    // Average limits
    double low=(clow+dlow)/2.0;
    double high=(chigh+dhigh)/2.0;
    */

    // Use decontracted limit
    double low=dlows[am];
    double high=dhighs[am];

    // Update lower and upper limits.
    /*
      if(low<0)
      low=-step*ceil(-low/step);
      else
      low=step*ceil(low/step);
    */
    cpl[am].start=low;

    /*
      if(high<0)
      high=-step*ceil(-high/step);
      else
      high=step*ceil(high/step);
    */
    cpl[am].end=high;
  }

  // Clear out polarization shells
  for(int am=atom_am(els)+1;am<max_am;am++) {
    cpl[am].start=0.0;
    cpl[am].end=0.0;
  }


  printf("Initial composition:\n");


  // Form exponents.
  for(int am=0;am<max_am;am++) {
    if(cpl[am].start == 0.0 && cpl[am].end == 0.0)
      continue;

    // Compute width
    double width=cpl[am].end-cpl[am].start;

    // Determine amount of exponents necessary to obtain width
    int nf;
    for(nf=1;nf<=NFMAX;nf++) {
      double w;
      std::vector<double> exps=maxwidth_exps(am,cpl[am].tol,nf,&w,OPTMOMIND);

      // Have we reached the necessary width?
      if(w>=width) {
        // Yes, we have. Adjust starting and ending points
        cpl[am].start-=(w-width)/2.0;
        cpl[am].end+=(w-width)/2.0;
        // and store exponents, moving starting point to the correct
        // location
        cpl[am].exps=move_exps(exps,cpl[am].start);
        // and stop the function loop
        break;
      }
    }

    printf("%c % .2f % .2f %2i (% .2f % .2f | % .2f % .2f)\n",shell_types[am],cpl[am].start,cpl[am].end,(int) cpl[am].exps.size(),dlows[am],dhighs[am],clows[am],chighs[am]);
  }

  printf("\n");

  return cpl;
}

double get_pol_x(double x) {
  // Safeguard against going to too diffuse functions. Minimal value is
  /*
    // Old version - use exponential scaling
    const double a=1.0/5.0;

    // Convert to safe scale
    return minpol+exp(a*x);
  */


  /*
  // New version - make function go smoothly to d0 at x=d0, and switch to line at x=-d0

  double d0=minpol;

  if(x<d0)
    return d0;
  else if(x<-d0) {
    // Third degree polynomial fit: y=ax^3 + bx^2 + cx + d.
    // Monotonic in the interval

    double a=-1/(4.0*d0*d0);
    double b=-1.0/(4.0*d0);
    double c=5.0/4.0;
    double d=d0/4.0;

    return a*x*x*x + b*x*x + c*x + d;
  } else
    // We are in the linear part
    return x;
  */

  // Just do the cutoff
  if(x<minpol)
    return minpol;
  else
    return x;
}

double zero_pol_x() {
  // Find where pol_x is zero
  double lx=minpol;
  double rx=-minpol;
  double mx=(lx+rx)/2.0;

  if(rx<lx)
    std::swap(lx,rx);

  while(rx-lx>=10*DBL_EPSILON) {
    mx=(lx+rx)/2.0;
    if(get_pol_x(mx)<0.0)
      lx=mx;
    else if(get_pol_x(mx)>0.0)
      rx=mx;
    else
      break;
  }

  return mx;
}

double check_polarization(double tol, const std::vector<int> & els, std::vector<coprof_t> & cpl, std::vector<arma::vec> & curval, bool dimer, const std::string & method) {
  // Check necessity of additional polarization shell.
  Timer t;
  const int max=maxam(cpl);
  if(max>=am_max) {
    printf("Won't add another polarization shell, MaxAM=%i reached.\n",am_max);
    fflush(stdout);
    return 0.0;
  }

  Timer tpol;

  // Width of completeness
  double pw;

  // Exponent to add: l=max+1, tolerance is cotol, 1 function.
  const int addam=max+1;
  std::vector<double> pexp=maxwidth_exps(addam,cotol,1,&pw,OPTMOMIND);

#ifdef DEBUGPOL
  printf("DEBUG: computing mogs for polarization shell.\n");

  static int iter=0;

  for(size_t iel=0;iel<els.size();iel++) {

    // Just compile a list of mogs for the added angular momentum

    char fname[80];
    sprintf(fname,"mog-%s-%c-%i.dat",element_symbols[els[iel]].c_str(),tolower(shell_types[addam]),iter);
    FILE *out=fopen(fname,"w");
    for(int id=-30;id<=20;id++) {
      // Do trial completeness profile
      double d=id*0.1;

      std::vector<coprof_t> trcpl(cpl);
      trcpl[addam].start=d-pw/2.0;
      trcpl[addam].end=d+pw/2.0;
      trcpl[addam].exps=move_exps(pexp,trcpl[addam].start);
      // Compute the value
      arma::vec trval=compute_value(els[iel],dimer,trcpl,method);
      // and the mog
      double trmog=compute_mog(trval,curval[iel]);
      printf("%e %e\n",d,trmog);
      fprintf(out,"%e %e\n",d,trmog);
      fflush(out);
    }
    fclose(out);
  }

  iter++;
  printf("END DEBUG SECTION\n");

#endif

  if(dimer && isscf(method))
    printf("\n****** %s pol ********\n",tolower(method).c_str());
  else if(dimer && !isscf(method))
    printf("\n****** %s pol/corr ********\n",tolower(method).c_str());
  else if(!dimer && !isscf(method))
    printf("\n****** %s corr ********\n",tolower(method).c_str());
  //  else
    // Not trying to add polarization shell for non-correlated atom calculation
    //    return 0.0;

  fflush(stdout);

  // Initial number of trials
  int ntrials=5;
  // Where the maximum should be
  int thmax=2;

  // Initial offset.
  double offset=zero_pol_x();
  // Initial step size.
  double step=0.15;

  // Maximum mog
  double maxmog=0.0;
  double oldmog=0.0;
  int maxel=-1;

  // Trial profiles
  std::vector< std::vector<coprof_t> > ptrials(ntrials);

  // Values obtained with the trial profiles
  std::vector< std::vector<arma::vec> > trvals(ntrials);

  for(int i=0;i<ntrials;i++)
    trvals[i]=curval;

  // Do we need to compute the value for trial i?
  std::vector<bool> compute(ntrials);
  for(int i=0;i<ntrials;i++)
    compute[i]=true;
  // The mogs for the trials
  std::vector<double> mogs(ntrials);
  for(int i=0;i<ntrials;i++)
    mogs[i]=-1.0;
  // The element with the maximum mog
  std::vector<int> maxels(ntrials);

  // Bracket the maximum
  do {
    // Save old mog
    oldmog=maxmog;

    // Form the trials
    for(int i=0;i<ntrials;i++) {
      ptrials[i]=cpl;
      ptrials[i][addam].start=get_pol_x(offset+(i-thmax)*step)-pw/2.0;
      ptrials[i][addam].end=ptrials[i][addam].start+pw;
      ptrials[i][addam].exps=move_exps(pexp,ptrials[i][addam].start);
    }
    // and compute the mogs
    for(int i=0;i<ntrials;i++) {
      if(compute[i]) {
	for(size_t iel=0;iel<els.size();iel++) {
	  trvals[i][iel]=compute_value(els[iel],dimer,ptrials[i],method);
	  double imog=compute_mog(trvals[i][iel],curval[iel]);

	  if(imog>mogs[i]) {
	    mogs[i]=imog;
	    maxels[i]=els[iel];
	  }
	}
	//	    printf("Trial %i computed.\n",i);
      }
      //	  printf("Trial %i: % .2e .. % .2e, mog %e\n",i+1,ptrials[i][addam].start,ptrials[i][addam].end,mogs[i]);
    }

    // Reset computation flags
    for(int i=0;i<ntrials;i++)
      compute[i]=true;

    maxel=els[0];
    maxmog=mogs[0];
    for(int i=1;i<ntrials;i++)
      if(mogs[i]>maxmog) {
	maxmog=mogs[i];
	maxel=maxels[i];
      }

    if(maxmog<10*EMDTOL)
      // Mog is too small for shell to be important, don't try
      // to optimize the shell
      break;

    if(maxmog<tol/100) {
      // Mog too small to be important, stop trying.
      break;
    }

    // Check if maximum has been bracketed, account for
    // inaccuracy in the integral
    if(fabs(mogs[thmax]-maxmog)>10*EMDTOL) {
      bool moved=false;

      // Maximum is to the right
      for(int i=thmax+1;i<ntrials;i++)
	if(mogs[i]==maxmog) {
	  // What is the displacement
	  int d=i-thmax;

	  // Move offset
	  offset+=d*step;
	  // and the solutions
	  for(int isol=thmax-d;isol<=thmax;isol++) {
	    trvals[isol]=trvals[isol+d];
	    mogs[isol]=mogs[isol+d];
	    compute[isol]=false;
	  }

	  printf("%c % .2f % .2f %e %2i %e %e %-2s %s.\n",shell_types[addam],ptrials[i][addam].start,ptrials[i][addam].end,ptrials[i][addam].tol,(int) ptrials[i][addam].exps.size(),maxmog,maxmog-oldmog,element_symbols[maxel].c_str(),t.elapsed().c_str());
	  fflush(stdout);
	  t.set();
	  moved=true;

	  break;
	}

      if(!moved)
	// Maximum is to the left.
	for(int i=0;i<thmax;i++)
	  if(mogs[i]==maxmog) {
	    // Displacement is
	    int d=thmax-i;
	    // Move offset
	    offset-=d*step;

	    // and the solutions
	    for(int isol=thmax+d;isol>=thmax;isol--) {
	      trvals[isol]=trvals[isol-d];
	      mogs[isol]=mogs[isol-d];
	      compute[isol]=false;
	    }

	    printf("%c % .2f % .2f %e %2i %e %e %-2s %s.\n",shell_types[addam],ptrials[i][addam].start,ptrials[i][addam].end,ptrials[i][addam].tol,(int) ptrials[i][addam].exps.size(),maxmog,maxmog-oldmog,element_symbols[maxel].c_str(),t.elapsed().c_str());
	    fflush(stdout);
	    t.set();
	    break;
	  }
    } else { // We have bracketed the maximum - change step size

      if(maxmog<tol/10) {
	// Mog too small to be important, stop trying.
	break;
      }

      // Find min mog
      double minmog=DBL_MAX;
      for(int i=0;i<ntrials;i++)
	if(mogs[i]<minmog)
	  minmog=mogs[i];

      if(maxmog-minmog<tol/10)
	// Refined to enough accuracy
	break;

      step/=2.0;
      printf("Reducing step size to %e, pw is %e, mog uncertainty is %e (%s).\n",step,pw,maxmog-minmog,t.elapsed().c_str());
      fflush(stdout);
      t.set();
    }
  } while(step>pw/10.0);

  // Do we need to add the shell?
  if(maxmog<tol) {
    printf("Not adding %c shell, mog = %e for %s. (%s)\n\n",shell_types[addam],maxmog,element_symbols[maxel].c_str(),tpol.elapsed().c_str());
    fflush(stdout);
    t.set();
  } else {
    // Yes, we do.
    printf("Added %c shell: % .2f % .2f (%i funcs) with tolerance %e, mog = %e for %s. (%s)\n\n",shell_types[addam],ptrials[thmax][addam].start,ptrials[thmax][addam].end,(int) ptrials[thmax][addam].exps.size(),ptrials[thmax][addam].tol,maxmog,element_symbols[maxel].c_str(),tpol.elapsed().c_str());
    fflush(stdout);
    t.set();

    // Update values
    cpl=ptrials[thmax];
    curval=trvals[thmax];
  }

  return maxmog;
}


double extend_profile(double tol, const std::vector<int> & els, std::vector<coprof_t> & cpl, std::vector<arma::vec> & curval, int ammin, int ammax, bool dimer, const std::string & method) {
  // Maximum mog
  double extmog=0.0;
  bool added;

  // Starting from which AM has the element been converged?
  std::vector<int> converged(els.size());
  for(size_t i=0;i<converged.size();i++)
    converged[i]=INT_MAX;

  // Check existing shells. Loop over elements
  while(true) {
    added=false;

    // Reset extension mog
    extmog=0.0;

    for(size_t iel=0;iel<els.size();iel++) {

      // Element has been already converged
      if(converged[iel]==ammin)
	continue;

      // Converged above
      int convl=0;

      // Trial values
      arma::vec lval(curval[iel]), mval(curval[iel]);
      arma::vec rval(curval[iel]), aval(curval[iel]);

      if(dimer) {
	if(maxam(cpl)>=ammin)
	  printf("\n****** %-2s dim %s ********\n",element_symbols[els[iel]].c_str(),tolower(method).c_str());
      } else
	printf("\n****** %-2s mono %s ********\n",element_symbols[els[iel]].c_str(),tolower(method).c_str());

      for(int am=ammin;am<=std::min(maxam(cpl),ammax);am++) {

	// Check if am has been already converged
	if(converged[iel]<=am)
	  continue;

	while(true) {
	  Timer t;

	  // Form trials: extension of completeness range
	  std::vector<coprof_t> left(cpl);
	  std::vector<coprof_t> middle(cpl);
	  std::vector<coprof_t> right(cpl);

	  // Get exponents
	  printf("Determining exponents for %c shell ... ",shell_types[am]);
	  fflush(stdout);
	  double width;
	  std::vector<double> exps=maxwidth_exps(am,cpl[am].tol,cpl[am].exps.size()+1,&width,OPTMOMIND);

	  double step=width-(cpl[am].end-cpl[am].start);

	  left[am].start=cpl[am].start-step;
	  left[am].exps=move_exps(exps,left[am].start);

	  if(domiddle) {
	    middle[am].start=cpl[am].start-step/2.0;
	    middle[am].end=cpl[am].end+step/2.0;
	    middle[am].exps=move_exps(exps,middle[am].start);
	  }

	  right[am].end=cpl[am].end+step;
	  right[am].exps=move_exps(exps,right[am].start);


	  // .. or addition of an exponent
	  bool expadd=false;
	  std::vector<coprof_t> add(cpl);
	  if(add[am].tol>MINTAU) {
	    add[am].exps=optimize_completeness(am,add[am].start,add[am].end,add[am].exps.size()+1,OPTMOMIND,false,&add[am].tol);
	    if(add[am].tol>MINTAU)
	      expadd=true;
	  }
	  printf("(%s), step size is %e.\n",t.elapsed().c_str(),step);
	  fflush(stdout);
	  t.set();

	  // Update current value
	  curval[iel]=compute_value(els[iel],dimer,cpl,method);

	  // and compute the trials
	  lval=compute_value(els[iel],dimer,left, method);
	  if(domiddle) mval=compute_value(els[iel],dimer,middle,method);
	  rval=compute_value(els[iel],dimer,right,method);
	  if(expadd)
	    aval=compute_value(els[iel],dimer,add,  method);
	  else
	    aval=curval[iel];

	  // and mogs
	  double lmog=compute_mog(lval,curval[iel]);
	  double mmog=-10.0;
	  if(domiddle) compute_mog(mval,curval[iel]);
	  double rmog=compute_mog(rval,curval[iel]);
	  double amog=compute_mog(aval,curval[iel]);

	  // Interpolated value and mog
	  std::vector<coprof_t> trcpl(cpl);
	  arma::vec trval;
	  double trmog;

	  if(domiddle) {
	    // Use inverse parabolic interpolation to find the maximum
	    double xa=left[am].start;
	    double xb=middle[am].start;
	    double xc=right[am].start;

	    double ya=lmog;
	    double yb=mmog;
	    double yc=rmog;

	    // Numerical recipes 10.2.1
	    double nom=(xb-xa)*(xb-xa)*(yb-yc) - (xb-xc)*(xb-xc)*(yb-ya);
	    double den=(xb-xa)*(yb-yc) - (xb-xc)*(yb-ya);

	    // Compute new point
	    double x=DBL_MAX;
	    if(fabs(den)>DBL_EPSILON) {
	      x=xb-0.5*nom/den;
	      //	      printf("Interpolation gives % .2f.\n",x);
	    }

	    // Check that x is in the trusted range
	    if(fabs(x-xb)<step/2.0) {
	      // and the value in the point
	      trcpl[am].start=x;
	      trcpl[am].end=x+width;
	      trcpl[am].exps=move_exps(exps,trcpl[am].start);

	      trval=compute_value(els[iel],dimer,trcpl,method);
	      trmog=compute_mog(trval,curval[iel]);

	      // Check that the trial mog isn't unbelievably large
	      if(trmog>=10.0*std::max(std::max(lmog,mmog),rmog))
		// Don't use the interpolated trial.
		trmog=-10.0;

	      //	      printf("Trial mog is %e.\n",trmog);
	    } else {
	      // Interpolation didn't work, use dummy value
	      //	      printf("Interpolation refused.\n");
	      trmog=-10.0;
	    }
	  } else trmog=-10.0;

	  double maxmog=std::max(std::max(std::max(lmog,rmog),std::max(mmog,amog)),trmog);

	  if(fabs(maxmog)<tol) {
	    // Existing shell is converged.

	    // Set convergence flag
	    if(am<convl)
	      convl=am;

	    if(fabs(maxmog)>extmog)
	      extmog=fabs(maxmog);

	    printf("%c % .2f % .2f %e %2i %e %s, converged.\n\n",shell_types[am],cpl[am].start,cpl[am].end,cpl[am].tol,(int) cpl[am].exps.size(),maxmog,t.elapsed().c_str());

	    fflush(stdout);
	    break;
	  }

	  added=true;
	  int dir=INT_MAX;

	  // Reset convergence flags
	  for(size_t i=0;i<converged.size();i++)
	    converged[i]=INT_MAX;
	  convl=INT_MAX;

	  if(trmog == maxmog) {
	    // Interpolation worked.
	    dir=0;
	    cpl=trcpl;
	    curval[iel]=trval;
	  } else if(amog == maxmog) {
	    // Prefer to add exponent
	    dir=1;
	    cpl=add;
	    curval[iel]=aval[iel];
	  } else if(rmog == maxmog) {
	    // Accept right move
	    dir=2;
	    cpl=right;
	    curval[iel]=rval[iel];
	  } else if(mmog == maxmog) {
	    // Accept middle move
	    dir=3;
	    cpl=middle;
	    curval[iel]=mval[iel];
	  } else {
	    // Accept left move.
	    dir=4;
	    cpl=left;
	    curval[iel]=lval[iel];
	  }
	  const std::string dirs[]={"interp","add","right","middle","left"};

	  printf("%c % .2f % .2f %e %2i %e move=%-6s %s.\n",shell_types[am],cpl[am].start,cpl[am].end,cpl[am].tol,(int) cpl[am].exps.size(),maxmog,dirs[dir].c_str(),t.elapsed().c_str());
	  fflush(stdout);
	}
      }
      converged[iel]=convl;
    }

    // No functions added this iteration.
    if(!added) {
      break;
    }
  }

  return extmog;
}

std::vector<coprof_t> generate_limits(const std::vector<int> & els, double tol) {
  // Plug in nf functions per shell.

  // Determine largest Z
  std::vector<int> elssort(els);
  std::sort(elssort.begin(),elssort.end());

  std::vector<coprof_t> cpl(am_max+1);
  for(size_t i=0;i<cpl.size();i++) {
    cpl[i].start=0.0;
    cpl[i].end=0.0;
    cpl[i].tol=cotol;
  }

  printf("Generating initial profile.\n");
  fflush(stdout);

  // Start with a single s function
  //  cpl[0].exps=maxwidth_exps(0,cotol,1,&cpl[0].end,OPTMOMIND);
  cpl[0].exps=maxwidth_exps(0,cotol,2,&cpl[0].end,OPTMOMIND);

  // Loop over magic numbers
  size_t Nmagic=sizeof(magicno)/sizeof(magicno[0]);
  for(size_t i=0;i<Nmagic;i++)
    if(magicno[i]>0 && magicno[i]<=elssort[elssort.size()-1]) {
      // Extend the profile using this noble gas.

      // Get the corresponding amount of electrons.
      std::vector<int> nel=shell_count(magicno[i]);

      // Plug in the functions
      for(size_t l=0;l<nel.size();l++)
	if(!cpl[l].exps.size())
	  //	  cpl[l].exps=maxwidth_exps(l,cotol,1,&cpl[l].end,OPTMOMIND);
	  cpl[l].exps=maxwidth_exps(l,cotol,2,&cpl[l].end,OPTMOMIND);

      // Helper
      std::vector<int> hlp;
      hlp.push_back(magicno[i]);

      // Do the expansion
      std::vector< std::vector<arma::vec> > trval;
      update_values(trval,cpl,hlp,scfmet(),false,false);
      extend_profile(tol,hlp,cpl,trval[0],0,am_max,false,scfmet());
    }

  // Get the configuration of the heaviest element
  std::vector<int> nel=shell_count(elssort[elssort.size()-1]);
  // and plug in the functions
  for(size_t l=0;l<nel.size();l++)
    if(!cpl[l].exps.size())
      //      cpl[l].exps=maxwidth_exps(l,cotol,nel[l]*nf,&cpl[l].end,OPTMOMIND);
      //      cpl[l].exps=maxwidth_exps(l,cotol,1,&cpl[l].end,OPTMOMIND);
      cpl[l].exps=maxwidth_exps(l,cotol,2,&cpl[l].end,OPTMOMIND);

  printf("Initial composition:\n");
  for(int am=0;am<=maxam(cpl);am++)
    printf("%c % .2f % .2f %2i\n",shell_types[am],cpl[am].start,cpl[am].end,(int) cpl[am].exps.size());
  printf("\n");
  fflush(stdout);

  return cpl;
}


double reduce_profile(double tol, const std::vector<int> & els, std::vector<coprof_t> & cpl, std::vector< std::vector<arma::vec> > & curval, const std::vector< std::vector<arma::vec> > & refval, const std::string & method) {
  printf("\n\n***************************** PROFILE REDUCTION *********************************\n\n");

  // Trials, consisting of either the move of the starting or the ending point or the removal of an exponent
  std::vector< std::vector<coprof_t> > trials(maxam(cpl)+1);

  // Trial values
  std::vector< std::vector< std::vector<arma::vec> > > trvals(maxam(cpl)+1);

  // Mogs
  double trmog[maxam(cpl)+1];
  int dir[maxam(cpl)+1];

  // Methods to use for monomer and dimer
  std::vector<std::string> methods(2);
  methods[0]=method;
  // Just scf for dimer
  //  methods[1]=scfmet();
  // Use post-HF for dimer, too
  methods[1]=method;

  // Update values

  // Do the reduction.
  double minmog=0.0;
  while(minmog<=tol) {
    Timer ttot;

    minmog=0.0;
    for(int am=0;am<=maxam(cpl);am++) {
      Timer t;
      // Check that all higher shells have a smaller amount of functions
      bool ok=true;
      for(int j=am+1;j<=am_max;j++)
	if(cpl[am].exps.size() <= cpl[j].exps.size()) {
	  printf("%c shell limited due to %c shell.\n",shell_types[am],shell_types[j]);
	  trials[am]=cpl;
	  trmog[am]=1.0;
	  ok=false;
	}
      if(!ok)
	continue;

      // Form trials
      std::vector<coprof_t> left(cpl);
      std::vector<coprof_t> middle(cpl);
      std::vector<coprof_t> right(cpl);
      std::vector<coprof_t> del(cpl);

      if(cpl[am].tol<MINTAU)
	printf("cpl[%i]=%e < %e!\n",am,cpl[am].tol,MINTAU);

      // Get exponents
      printf("Determining exponents for %c shell ... ",shell_types[am]);
      fflush(stdout);

      double step;

      if(cpl[am].exps.size()>1) {
	double width;
	std::vector<double> exps=maxwidth_exps(am,cpl[am].tol,cpl[am].exps.size()-1,&width,OPTMOMIND);

	step=cpl[am].end-cpl[am].start-width;
	left[am].start=cpl[am].start+step;
	left[am].exps=move_exps(exps,left[am].start);

	middle[am].start=cpl[am].start+step/2.0;
	middle[am].end=cpl[am].end-step/2.0;
	middle[am].exps=move_exps(exps,middle[am].start);

	right[am].end=cpl[am].end-step;
	right[am].exps=move_exps(exps,right[am].start);

	del[am].exps=optimize_completeness(am,del[am].start,del[am].end,del[am].exps.size()-1,OPTMOMIND,false,&del[am].tol);
      } else {
	step=left[am].end-left[am].start;

	// Dummy tries
	left[am].start=0.0;
	left[am].end=0.0;
	left[am].exps.clear();

	middle[am].start=0.0;
	middle[am].end=0.0;
	middle[am].exps.clear();

	right[am].start=0.0;
	right[am].end=0.0;
	right[am].exps.clear();

	del[am].exps.clear();
	del[am].start=0.0;
	del[am].end=0.0;
      }
      printf("(%s), step size is %e.\n",t.elapsed().c_str(),step);
      t.set();

      printf("Computing drop of function from %c shell (%2i funcs)",shell_types[am],(int) cpl[am].exps.size());
      fflush(stdout);

      // Compute values
      std::vector< std::vector<arma::vec> > lval(curval);
      std::vector< std::vector<arma::vec> > mval(curval);
      std::vector< std::vector<arma::vec> > rval(curval);
      std::vector< std::vector<arma::vec> > dval(curval);

      bool lvalfail=false, mvalfail=false, rvalfail=false, dvalfail=false;

      for(size_t isys=0;isys<curval.size();isys++) {
	for(size_t iel=0;iel<els.size();iel++) {

	  if(cpl[am].exps.size()>1) {
	    try {
	      lval[isys][iel]=compute_value(els[iel],(bool) isys,left,methods[isys]);
	    } catch(std::runtime_error) {
	      lvalfail=true;
	    }

	    try {
	      mval[isys][iel]=compute_value(els[iel],(bool) isys,middle,methods[isys]);
	    } catch(std::runtime_error) {
	      mvalfail=true;
	    }

	    try {
	      rval[isys][iel]=compute_value(els[iel],(bool) isys,right,methods[isys]);
	    } catch(std::runtime_error) {
	      rvalfail=true;
	    }

	  } else {
	    // Trying to drop last exponent, make these dummy tries
	    // since they will be equal to the one below
	    lvalfail=true;
	    mvalfail=true;
	    rvalfail=true;
	  }

	  try {
	    dval[isys][iel]=compute_value(els[iel],(bool) isys,del,methods[isys]);
	  } catch(std::runtime_error) {
	    dvalfail=true;
	  }
	}
      }

      // and mogs
      double lmonomog, mmonomog, rmonomog, dmonomog;
      double ldimmog, mdimmog, rdimmog, ddimmog;

      if(lvalfail) {
	lmonomog=DBL_MAX;
	ldimmog=DBL_MAX;
      } else {
	lmonomog=compute_mog(lval[0],refval[0]);
	ldimmog=compute_mog(lval[1],refval[1]);
      }

      if(mvalfail) {
	mmonomog=DBL_MAX;
	mdimmog=DBL_MAX;
      } else {
	mmonomog=compute_mog(mval[0],refval[0]);
	mdimmog=compute_mog(mval[1],refval[1]);
      }

      if(rvalfail) {
	rmonomog=DBL_MAX;
	rdimmog=DBL_MAX;
      } else {
	rmonomog=compute_mog(rval[0],refval[0]);
	rdimmog=compute_mog(rval[1],refval[1]);
      }

      if(dvalfail) {
	dmonomog=DBL_MAX;
	ddimmog=DBL_MAX;
      } else {
	dmonomog=compute_mog(dval[0],refval[0]);
	ddimmog=compute_mog(dval[1],refval[1]);
      }

      double lmog=std::max(lmonomog,ldimmog);
      double mmog=std::max(mmonomog,mdimmog);
      double rmog=std::max(rmonomog,rdimmog);
      double dmog=std::max(dmonomog,ddimmog);

      minmog=std::min(std::min(lmog,rmog),std::min(mmog,dmog));

      // Check which trial to try.
      if(lmog==minmog) {
	// Move left end
	trmog[am]=lmog;
	trvals[am]=lval;
	trials[am]=left;
	dir[am]=1;
      } else if(mmog==minmog) {
	// Move symmerically
	trmog[am]=mmog;
	trvals[am]=mval;
	trials[am]=middle;
	dir[am]=2;
      } else if(dmog==minmog) {
	// Drop a function.
	trmog[am]=dmog;
	trvals[am]=dval;
	trials[am]=del;
	dir[am]=0;
      } else {
	// Move right end
	trmog[am]=rmog;
	trvals[am]=rval;
	trials[am]=right;
	dir[am]=-1;
      }

      printf(", mog is %e (%s)\n",trmog[am],t.elapsed().c_str());
    }

    // Figure out minimal mog
    size_t minind=0;
    minmog=trmog[0];
    for(int am=0;am<=maxam(cpl);am++)
      if(trmog[am]<minmog) {
	minmog=trmog[am];
	minind=am;
      }

    if(minmog<=tol) {
      // Update profile
      cpl=trials[minind];
      // Update current value
      curval=trvals[minind];

      if(dir[minind]==-1)
	printf("Moved   ending point of %c shell, range is now % .2f ... % .2f (%2i funcs), tol = %e, mog %e (%s).\n\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());
      else if(dir[minind]==1)
	printf("Moved starting point of %c shell, range is now % .2f ... % .2f (%2i funcs), tol = %e, mog %e (%s).\n\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());
      else if(dir[minind]==2)
	printf("Moved    both limits of %c shell, range is now % .2f ... % .2f (%2i funcs), tol = %e, mog %e (%s).\n\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());
      else if(dir[minind]==0)
	printf("Dropped exponent   from %c shell, range is now % .2f ... % .2f (%2i funcs), tol = %e, mog %e (%s).\n\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());
    } else {
      printf("Converged, minimal mog is %e.\n\n",minmog);
      break;
    }
  }

  return minmog;
}

void print_scheme(const BasisSetLibrary & baslib, const std::vector<int> & els, int len=0) {
  // Get contraction scheme
  ElementBasisSet el=baslib.get_element(element_symbols[els[0]]);

  // Number of exponents and contractions
  std::vector<int> nc, nx;

  for(int am=0;am<=el.get_max_am();am++) {
    std::vector<double> exps;
    arma::mat coeffs;
    el.get_primitives(exps,coeffs,am);
    nx.push_back(coeffs.n_rows);
    nc.push_back(coeffs.n_cols);
  }

  // Is the set contracted?
  bool contr=false;
  for(size_t i=0;i<nx.size();i++)
    if(nx[i]!=nc[i])
      contr=true;

  std::string out;
  char tmp[80];

  if(contr) {
    out="[";
    for(int l=0;l<=el.get_max_am();l++) {
      sprintf(tmp,"%i%c",nx[l],tolower(shell_types[l]));
      out+=std::string(tmp);
    }
    out+="|";
    for(int l=0;l<=el.get_max_am();l++)
      if(nc[l]!=nx[l]) {
	sprintf(tmp,"%i%c",nc[l],tolower(shell_types[l]));
	out+=std::string(tmp);
      }
    out+="]";
  } else {
    out="(";
    for(int l=0;l<=el.get_max_am();l++) {
      sprintf(tmp,"%i%c",nx[l],tolower(shell_types[l]));
      out+=std::string(tmp);
    }
    out+=")";
  }

  if(len==0)
    printf("%s",out.c_str());
  else {
    // Format specifier
    sprintf(tmp,"%%-%is",len);
    printf(tmp,out.c_str());
  }
}

double contraction_mog(int Z, const std::vector<int> & trial, const std::vector<coprof_t> & cpl, std::vector<arma::mat> Cmat, std::vector<arma::vec> exps, const std::string & method, const arma::vec & monoref, const arma::vec & dimref) {
  Timer t;

  // Get the basis set
  BasisSetLibrary baslib=get_library(element_symbols[Z],cpl,Cmat,exps,trial);

  // Compute new values
  double monomog, dimmog;
  try {
    arma::vec monoval=compute_value(Z,false,baslib,method);
    monomog=compute_mog(monoval,monoref);
  } catch(std::runtime_error) {
    monomog=DBL_MAX;
  }

  try {
    arma::vec dimval=compute_value(Z,true,baslib,method);
    dimmog=compute_mog(dimval,dimref);
  } catch(std::runtime_error) {
    dimmog=DBL_MAX;
  }

  // Print the scheme
  printf("%-2s ",element_symbols[Z].c_str());
  {
    std::vector<int> dummy;
    dummy.push_back(Z);
    print_scheme(baslib,dummy,25);
  }
  printf(" %e %e (%s)\n",monomog,dimmog,t.elapsed().c_str());
  fflush(stdout);

  return std::max(monomog,dimmog);
}

BasisSetLibrary contract_basis(double tol, const std::vector<int> & els, std::vector<coprof_t> & cpl, const std::vector< std::vector<arma::vec> > & refval, const std::string & method, const std::string & contrmet) {

  printf("\n\n***************************** BASIS CONTRACTION *********************************\n\n");

  // Amount of functions
  const int Nf[2]={(int) cpl[0].exps.size(), (int) cpl[1].exps.size()};

  // Methods to use for monomer and dimer
  std::vector<std::string> methods(2);
  methods[0]=method;
  methods[1]=method;

  // Get the contraction matrices
  Timer ttot;
  printf("Determining contraction coefficients.\n");
  std::vector< std::vector<arma::mat> > Cmat(els.size());
  std::vector< std::vector<arma::vec> > exps(els.size());
  for(size_t i=0;i<els.size();i++) {
    printf("%s:\n",element_symbols[els[i]].c_str());
    fflush(stdout);
    get_contraction(els[i],cpl,contrmet,Cmat[i],exps[i]);
  }
  printf("done (%s).\n\n",ttot.elapsed().c_str());
  fflush(stdout);

  ttot.set();

  // Maximum angular momentum to contract
  int cmaxam=atom_am(els);

  // How many functions to contract.
  std::vector<int> contract(cmaxam+1);
  for(size_t l=0;l<contract.size();l++)
    contract[l]=INT_MAX;

  for(size_t iel=0;iel<els.size();iel++)
    for(size_t l=0;l<Cmat[iel].size();l++) {
      // At minimum contract the same amount of functions as there are
      // natural orbitals, this doesn't affect the degrees of freedom
      // in any way.
      if((int) Cmat[iel][l].n_cols < contract[l])
        contract[l]=(int) Cmat[iel][l].n_cols;
    }

  // Do the contraction.
  double minmog=0.0;

  while(true) {
    Timer tc;

    // Compute mogs of contracting one more function on each shell
    std::vector<double> mogs(cmaxam+1);
    for(int am=0;am<=cmaxam;am++) {

      // We still have free functions left, calculate mog
      if(contract[am]<Nf[am]) {
	mogs[am]=0.0;
	for(size_t iel=0;iel<els.size();iel++) {
	  // Trial contraction
	  std::vector<int> trial(contract);
	  trial[am]++;

	  double mog=contraction_mog(els[iel],trial,cpl,Cmat[iel],exps[iel],method,refval[0][iel],refval[1][iel]);

	  if(mog>mogs[am])
	    mogs[am]=mog;
	}

	if(cmaxam>0)
	  printf("%c mog is %e.\n",shell_types[am],mogs[am]);
      } else
	// No free primitives left.
	mogs[am]=DBL_MAX;
    }

    // Determine minimal mog
    minmog=DBL_MAX;
    for(int l=0;l<=cmaxam;l++)
      minmog=std::min(minmog,mogs[l]);

    // Accept move?
    if(minmog<=tol) {
      for(int l=cmaxam;l>=0;l--)
	if(mogs[l]==minmog) {
	  printf("Contracting a %c function, mog is %e (%s).\n\n",shell_types[l],minmog,tc.elapsed().c_str());
	  fflush(stdout);
	  contract[l]++;
	  break;
	}
    } else {
      // Converged
      if(minmog==DBL_MAX)
	printf("No functions left to contract. Scheme is ");
      else
	printf("Minimal mog is %e, converged (%s). Scheme is ",minmog,tc.elapsed().c_str());
      break;
    }
  }

  // Compile library
  BasisSetLibrary ret;
  for(size_t iel=0;iel<els.size();iel++) {
    BasisSetLibrary baslib=get_library(element_symbols[els[iel]],cpl,Cmat[iel],exps[iel],contract);
    ret.add_element(baslib.get_element(element_symbols[els[iel]]));
  }

  print_scheme(ret,els);
  printf(".\nContraction took %s.\n\n",ttot.elapsed().c_str());
  fflush(stdout);

  return ret;
}

int get_nfuncs(const std::vector<coprof_t> & cpl) {
  int n=0;
  for(int am=0;am<=maxam(cpl);am++)
    n+=(2*am+1)*cpl[am].exps.size();
  return n;
}

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Completeness optimization from Hel. OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Completeness optimization from Hel. Serial version.\n");
#endif
  print_header();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  Timer ttot;

  Settings set;
  set.add_string("OptEl","Elements to optimize basis for","");
  set.add_string("OptMom","Moments to optimize (0 means <-2>)","1,3:5");
  set.add_double("COTol","Initial tolerance for completeness profile",1e-3);
  set.add_double("InitTol","Initial tolerance",1e-4);
  set.add_double("Etol","Energy tolerance in a.u. (0 for don't optimize)",0.0);
  set.add_string("Theory","Level of theory to use","HF");
  set.add_string("Memory","Memory allowed for calculation","8GB");
  set.add_string("Program","Program to use","gaussian");
  set.add_bool("DoDimer","Do the incrementation also using the dimers?",true);
  set.add_bool("DoMiddle","Also do the middle try in the incrementation?",true);
  set.add_bool("DimerPol","Restrict dimer calculation to polarization shells only?",true);
  set.add_bool("FirstPol","Check first need for polarization shells?",false);
  set.add_bool("SCFContr","Use SCF coefficients for contraction (instead of NOs)",false);
  set.add_int("MaxAM","Maximum angular momentum",13);
  set.add_double("MinPol","Minimal allowed value for first polarization exponent",-1.5);

  set.parse(argv[1]);
  set.print();

  // Set parameters
  memory=set.get_string("Memory");
  prog=set.get_string("Program");
  calcdimer=set.get_bool("DoDimer");
  domiddle=set.get_bool("DoMiddle");
  am_max=std::min(set.get_int("MaxAM"),max_am);
  minpol=set.get_double("MinPol");

#ifdef _OPENMP
  nthreads=omp_get_max_threads();
#else
  nthreads=1;
#endif

  // Get parameters
  inittol=set.get_double("InitTol");

  // Allowed deviance from completeness, adjusted later.
  cotol=set.get_double("COTol");

  // Method to use
  std::string method=set.get_string("Theory");

  // Energy tolerance
  Etol=set.get_double("Etol");

  // Moments to optimize
  try {
    moms=parse_range(set.get_string("OptMom"));
  } catch(std::runtime_error) {
    if(Etol<=0.0)
      throw std::runtime_error("Neither OptMom nor Etol specified!\n");
    else
      printf("OptMom not specified.\n");
  }

  if(moms[moms.size()-1]>6 || moms[0]<0)
    throw std::runtime_error("Error - OptMom must be in the range 0 ... 6!");

  // Elements to optimize
  std::vector<int> els;
  {
    std::vector<std::string> help=splitline(set.get_string("OptEl"));
    for(size_t i=0;i<help.size();i++)
      els.push_back(get_Z(help[i]));
  }

  if(els.size()==0)
    throw std::runtime_error("Need element to optimize basis for!\n");

  // Method to use for contraction
  std::string contrmet=method;
  if(set.get_bool("SCFContr"))
    contrmet="ROHF";

  // Do dimer only for polarization shells?
  bool dimerpol=set.get_bool("DimerPol");
  bool firstpol=set.get_bool("FirstPol");

  printf("Optimizing %s values of ",method.c_str());
  for(size_t i=0;i<moms.size();i++) {
    if(moms[i] != 2 && moms[i] != 7)
      printf(" <p^%+i>",(int) moms[i]-2);
    else if(moms[i]==7)
      printf(" E");
  }
  printf("\nfor elements with geometries and electronic states:\n\t%-3s %5s %s\n","el","  R  ","M");
  for(size_t i=0;i<els.size();i++) {
    // Monomer ground state
    gs_conf_t gs=get_ground_state(els[i]);
    printf("\t%-3s %5s %i\n",element_symbols[els[i]].c_str()," *** ",gs.mult);
    if(els[i]<(int) (sizeof(dimR)/sizeof(dimR[0])))
      printf("\t%-3s %1.3f %i\n",(element_symbols[els[i]]+"2").c_str(),dimR[els[i]],dimM[els[i]]);

  }
  std::stable_sort(els.begin(),els.end(),compare_easiness);
  printf("Elements optimized in the order:");
  for(size_t i=0;i<els.size();i++)
    printf(" %s",element_symbols[els[i]].c_str());
  printf("\n");

  // Output files
  std::vector< std::vector<FILE *> > out(2);
  for(size_t isys=0;isys<2;isys++)
    out[isys].resize(els.size());
  for(size_t iel=0;iel<els.size();iel++) {
    char fname[80];

    // Monomer
    sprintf(fname,"%s-monomer-%s.dat",element_symbols[els[iel]].c_str(),method.c_str());
    out[0][iel]=fopen(fname,"w");
    // Dimer
    sprintf(fname,"%s-dimer-%s.dat",element_symbols[els[iel]].c_str(),method.c_str());
    out[1][iel]=fopen(fname,"w");
  }


  // Get initial completeness profile.
  const std::vector<coprof_t> initprof=generate_limits(els,1e-1);
  // Completeness profile to optimize
  std::vector<coprof_t> cpl(initprof);

  // Get initial values
  std::vector< std::vector<arma::vec> > curval(2);
  curval[0].resize(els.size());
  curval[1].resize(els.size());

  // Tolerance
  double tau=1.0;

  // Extend completeness profile.
  do {
    tau/=10.0;
    if(tau<inittol)
      tau=inittol;

    printf("\n\n\n***************************** tau = %e ********************************\n",tau);

    double polerr=0.0;

    do {
      if(firstpol) {
	do {
	  // Update values
	  update_values(curval,cpl,els,method,true);

	  // Check if polarization shells are necessary.
	  if(calcdimer)
	    polerr=check_polarization(tau,els,cpl,curval[1],true,method);
	  else
	    polerr=check_polarization(tau,els,cpl,curval[0],false,method);
	} while(polerr>=tau);
      }

      // Update values once again (monomer value might have changed)
      update_values(curval,cpl,els,method,true);

      // Then extend the monomer profile
      extend_profile(tau,els,cpl,curval[0],0,am_max,false,method);
      // Update once again (at least for contraction)
      update_values(curval,cpl,els,method,true);

      // and then the dimer if wanted
      if(calcdimer) {

	if(dimerpol) {
	  if(maxam(cpl)>=atom_am(els)+1) {
	    // Extend only polarization shells
	    extend_profile(tau,els,cpl,curval[1],atom_am(els)+1,am_max,true,method);
	    update_values(curval,cpl,els,method,true);
	  }
	} else
	  // Full extension
	  extend_profile(tau,els,cpl,curval[1],0,am_max,true,method);
      }

      // Check once again if polarization shells are
      // necessary. Include shells that would be relevant at the next
      // iteration.
      double poltau=std::max(tau/10.0,inittol);
      if(calcdimer)
	polerr=check_polarization(poltau,els,cpl,curval[1],true,method);
      else
	polerr=check_polarization(poltau,els,cpl,curval[0],false,method);

    } while(polerr>=tau);

    // Save values
    print_values(curval,els,out);

    printf("\nComposition:\n");
    for(int am=0;am<=maxam(cpl);am++)
      printf("%c % .2f % .2f %2i\n",shell_types[am],cpl[am].start,cpl[am].end,(int) cpl[am].exps.size());

    // Save basis set
    {
      BasisSetLibrary baslib=get_library(els,cpl);
      char fname[80];
      sprintf(fname,"%s-%e-full.gbs",tolower(method).c_str(),tau);
      baslib.save_gaussian94(fname);
    }

    // Contract basis set and save it, too
    {
      BasisSetLibrary contrbas=contract_basis(tau,els,cpl,curval,method,contrmet);
      char fname[80];
      sprintf(fname,"%s-%e-full-contr.gbs",tolower(method).c_str(),tau);
      contrbas.save_gaussian94(fname);
    }
  } while(tau>inittol);

  // Close output files
  for(size_t isys=0;isys<out.size();isys++)
    for(size_t iel=0;iel<els.size();iel++)
      fclose(out[isys][iel]);

  // Reference value
  const std::vector< std::vector<arma::vec> > refval(curval);

  // Output files
  std::vector< std::vector<FILE *> > redout(2);
  for(size_t isys=0;isys<2;isys++)
    redout[isys].resize(els.size());
  for(size_t iel=0;iel<els.size();iel++) {
    char fname[80];

    // Monomer
    sprintf(fname,"%s-monomer-red.dat",element_symbols[els[iel]].c_str());
    redout[0][iel]=fopen(fname,"w");
    // Dimer
    sprintf(fname,"%s-dimer-red.dat",element_symbols[els[iel]].c_str());
    redout[1][iel]=fopen(fname,"w");

  }

  // Reduce completeness profile
  tau=1.0;
  while(tau>inittol)
    tau/=10.0;

  while((1-sqrt(DBL_EPSILON))*tau<=1e-2) {
    printf("\n\n\n***************************** tau = %e ********************************\n",tau);

    // Reduce the profile
    reduce_profile(tau,els,cpl,curval,refval,method);

    // Update errors
    double monoerr=compute_mog(curval[0],refval[0]);
    double dimerr=compute_mog(curval[1],refval[1]);

    printf("\nErrors after reduction: %e for monomer and %e for dimer.\n",monoerr,dimerr);

    printf("Final composition:\n");
    for(int am=0;am<=maxam(cpl);am++)
      printf("%c % .2f % .2f %2i\n",shell_types[am],cpl[am].start,cpl[am].end,(int) cpl[am].exps.size());
    printf("\n");

    // Save basis set
    {
      BasisSetLibrary baslib=get_library(els,cpl);
      char fname[80];
      sprintf(fname,"%s-%e.gbs",tolower(method).c_str(),tau);
      baslib.save_gaussian94(fname);
    }

    // Contract the basis
    {
      BasisSetLibrary contrbas=contract_basis(tau,els,cpl,refval,method,contrmet);
      char fname[80];
      sprintf(fname,"%s-%e-contr.gbs",tolower(method).c_str(),tau);
      contrbas.save_gaussian94(fname);
    }

    // Increase tolerance
    tau*=10.0;
  }

  printf("\n************************** FINISHED *********************************\n");

  // Close output files
  for(size_t isys=0;isys<redout.size();isys++)
    for(size_t iel=0;iel<els.size();iel++)
      fclose(redout[isys][iel]);

  printf("\nRunning program took %s.\n",ttot.elapsed().c_str());

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
