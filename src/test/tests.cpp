/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "checkpoint.h"
#include "basislibrary.h"
#include "global.h"
#include "mathf.h"
#include "scf.h"
#include "solidharmonics.h"
#include "timer.h"
#include "xyzutils.h"

#include <cfloat>
#include <cmath>
#include <cstdio>

/// Relative tolerance in total energy
const double tol=1e-6;
/// Absolute tolerance in orbital energies
const double otol=1e-5;
/// Absolute tolerance for dipole moment
const double dtol=1e-5;

/// Absolute tolerance for normalization of basis functions
const double normtol=1e-10;

/// Check orthogonality of spherical harmonics up to
const int Lmax=10;
/// Tolerance for orthonormality
const double orthtol=500*DBL_EPSILON;

/// Initial DFT grid tolerance
const double dft_initialtol=1e-3;
/// Final DFT grid tolerance
const double dft_finaltol=1e-6;

/// Initial convergence settings (for DFT)
const convergence_t init_conv={1e-4, 1e-4, 1e-6};
/// Final convergence settings
const convergence_t final_conv={1e-6, 1e-6, 1e-8};

/// To compute references instead of running tests
//#define COMPUTE_REFERENCE

/// Compute relative difference \f$ (x-y)/y \f$
double rel_diff(double x, double y) {
  return (x-y)/y;
}

/// Check if \f$ | (x - y)/y | < \tau \f$
bool rel_compare(double x, double y, double tau) {
  // Compute relative difference
  double d=rel_diff(x,y);

  if(fabs(d)<tau) {
    //    printf("%e vs %e, difference %e, ok\n",x,y,d);
    return 1;
  } else {
    //    printf("%e vs %e, difference %e, fail\n",x,y,d);
    return 0;
  }
}

/// Check if \f$ | (x - y) | < \tau \f$
bool abs_compare(double x, double y, double tau) {
  // Compute relative difference
  double d=fabs(x-y);

  if(fabs(d)<tau) {
    //    printf("%e vs %e, difference %e, ok\n",x,y,d);
    return 1;
  } else {
    //    printf("%e vs %e, difference %e, fail\n",x,y,d);
    return 0;
  }
}

/// Check that x == y within precision tau.
bool compare(const arma::vec & x, const arma::vec & y, double tau, size_t & nsucc, size_t & nfail) {
  if(x.n_elem!=y.n_elem)
    throw std::runtime_error("Error - differing amount of computed and reference orbital energies!\n");

  size_t N=std::min(x.n_elem,y.n_elem);

  nsucc=0;
  nfail=0;

  bool ok=1;
  for(size_t i=0;i<N;i++) {
    double d=x(i)-y(i);

    if(fabs(d)>tau) {
      //      printf("%e vs %e, difference %e, fail\n",x(i),y(i),d);
      ok=0;
      nfail++;
    } else {
      //      printf("%e vs %e, difference %e, ok\n",x(i),y(i),d);
      nsucc++;
    }
  }

  return ok;
}

/// Compute maximum difference of elements of x and y
double max_diff(const arma::vec & x, const arma::vec & y) {
  if(x.n_elem!=y.n_elem)
    throw std::runtime_error("Error - differing amount of computed and reference orbital energies!\n");

  double m=0;
  for(size_t i=0;i<x.n_elem;i++) {
    double d=fabs(x(i)-y(i));
    if(d>m)
      m=d;
  }

  return m;
}

/// Convert units from Ångström to a.u.
atom_t convert_to_bohr(const atom_t & in) {
  atom_t ret=in;

  ret.x*=ANGSTROMINBOHR;
  ret.y*=ANGSTROMINBOHR;
  ret.z*=ANGSTROMINBOHR;

  return ret;
}

// Possible statuses
const char * stat[]={"fail","ok"};

// Check proper normalization of basis
void check_norm(const BasisSet & bas) {
  size_t Nbf=bas.get_Nbf();
  arma::mat S=bas.overlap();
  
  for(size_t i=0;i<Nbf;i++)
    if(fabs(S(i,i)-1.0)>=normtol) {
      std::ostringstream oss;
      ERROR_INFO();
      fflush(stdout);
      oss << "Function " << i+1 << " is not normalized: norm is " << S(i,i) << "!.\n";
      throw std::runtime_error(oss.str());
    }
}


  // Check normalization of spherical harmonics
double cartint(int l, int m, int n) {
  // J. Comput. Chem. 27, 1009-1019 (2006)
  // \int x^l y^m z^n d\Omega =
  // 4 \pi (l-1)!! (m-1)!! (n-1)!! / (l+m+n+1)!! if l,m,n even,
  // 0 otherwise

  if(l%2==1 || m%2==1 || n%2==1)
    return 0.0;

  return 4.0*M_PI*doublefact(l-1)*doublefact(m-1)*doublefact(n-1)/doublefact(l+m+n+1);
}

// Check norm of Y_{l,m}.
void check_sph_orthonorm(int lmax) {
  
  // Left hand value of l
  for(int ll=0;ll<=lmax;ll++)
    // Right hand value of l
    for(int lr=ll;lr<=lmax;lr++) {

      // Loop over m values
      for(int ml=-ll;ml<=ll;ml++) {
	// Get the coefficients
	std::vector<double> cl=calcYlm_coeff(ll,ml);
	
	// Form the list of cartesian functions
	std::vector<shellf_t> cartl(((ll+1)*(ll+2))/2);
	size_t n=0;
	for(int i=0; i<=ll; i++) {
	  int nx = ll - i;
	  for(int j=0; j<=i; j++) {
	    int ny = i-j;
	    int nz = j;
	    
	    cartl[n].l=nx;
	    cartl[n].m=ny;
	    cartl[n].n=nz;
	    cartl[n].relnorm=cl[n];
	    n++;
	  }
	}
	
	for(int mr=-lr;mr<=lr;mr++) {
	  // Get the coefficients
	  std::vector<double> cr=calcYlm_coeff(lr,mr);
	  
	  // Form the list of cartesian functions
	  std::vector<shellf_t> cartr(((lr+1)*(lr+2))/2);
	  size_t n=0;
	  for(int i=0; i<=lr; i++) {
	    int nx = lr - i;
	    for(int j=0; j<=i; j++) {
	      int ny = i-j;
	      int nz = j;
	      
	      cartr[n].l=nx;
	      cartr[n].m=ny;
	      cartr[n].n=nz;
	      cartr[n].relnorm=cr[n];
	      n++;
	    }
	  }
	  
	  // Compute dot product
	  double norm=0.0;
	  for(size_t i=0;i<cartl.size();i++)
	    for(size_t j=0;j<cartr.size();j++)
	      norm+=cartl[i].relnorm*cartr[j].relnorm*cartint(cartl[i].l+cartr[j].l,cartl[i].m+cartr[j].m,cartl[i].n+cartr[j].n);

	  if( (ll==lr) && (ml==mr) ) {
	    if(fabs(norm-1.0)>orthtol) {
	      fprintf(stderr,"Square norm of (%i,%i) is %e, deviation %e from unity!\n",ll,ml,norm,norm-1.0);
	      throw std::runtime_error("Wrong norm.\n");
	    }
	  } else {
	    if(fabs(norm)>orthtol) {
	      fprintf(stderr,"Inner product of (%i,%i) and (%i,%i) is %e!\n",ll,ml,lr,mr,norm);
	      throw std::runtime_error("Functions not orthogonal.\n");
	    }
	  }
	}
      }
    }
}


/// Test RHF solution
#ifdef COMPUTE_REFERENCE
#define rhf_test(at,baslib,set,Etot,Eorb,label,dipmom) rhf_test_run(at,baslib,set,Etot,Eorb,label,dipmom); printf("rhf_test(" #at "," #baslib "," #set "," #Etot "," #Eorb "," #label "," #dipmom ");\n\n");
#else
#define rhf_test(at,baslib,set,Etot,Eorb,label,dipmom) rhf_test_run(at,baslib,set,Etot,Eorb,label,dipmom);
#endif
void rhf_test_run(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorb, const std::string & label, double dipmom) {
  Timer t;

#ifndef COMPUTE_REFERENCE
  printf("%s, ",label.c_str());
  fflush(stdout);
#endif

  arma::vec E;
  arma::mat C;

  // Construct basis set
  BasisSet bas=construct_basis(at,baslib,set);
  // Get orbital occupancies
  std::vector<double> occs=get_restricted_occupancy(set,bas);
  // Solve SCF equations
  rscf_t sol;
  Checkpoint chkpt("test.chk",1);
  SCF solver=SCF(bas,set,chkpt);
  solver.RHF(sol,occs,final_conv);
  // Compute dipole moment
  double dip=dip_mom(sol.P,bas);

  // Check normalization of basis
  check_norm(bas);

#ifdef COMPUTE_REFERENCE
  printf("Etot=%.16e;\n",sol.en.E);
  printf("dip=%.16e;\n",dip);
  printf("Eorb=\"");
  for(size_t i=0;i<sol.E.n_elem;i++)
    printf("%.16e ",sol.E(i));
  printf("\";\n");
#else
  // Compare results
  bool Eok=1, Dok=1, ok=1;
  size_t nsucc=0, nfail=0;
  compare(sol.E,Eorb,otol,nsucc,nfail); // Compare orbital energies
  Eok=rel_compare(sol.en.E,Etot,tol); // Compare total energies
  Dok=abs_compare(dip,dipmom,dtol); // Compare dipole moments
  ok=(Eok && Dok);
  printf("E=%f %s, dp=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],dip,stat[Dok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, difference in dipole moment is %e.\nMaximum difference of orbital energy is %e.\n",rel_diff(sol.en.E,Etot),dip-dipmom,max_diff(sol.E,Eorb));

  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    fflush(stdout);
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }
#endif
}

/// Test UHF solution
#ifdef COMPUTE_REFERENCE
#define uhf_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom) uhf_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom); printf("uhf_test(" #at "," #baslib "," #set "," #Etot "," #Eorba "," #Eorbb "," #label "," #dipmom ");\n\n");
#else
#define uhf_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom) uhf_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom);
#endif
void uhf_test_run(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorba, const arma::vec & Eorbb, const std::string & label, double dipmom) {
  Timer t;

#ifndef COMPUTE_REFERENCE
  printf("%s, ",label.c_str());
  fflush(stdout);
#endif

  arma::vec Ea, Eb;
  arma::mat Ca, Cb;

  // Construct basis set
  BasisSet bas=construct_basis(at,baslib,set);
  // Get orbital occupancies
  std::vector<double> occa, occb;
  get_unrestricted_occupancy(set,bas,occa,occb);
  // Solve SCF equations
  uscf_t sol;
  Checkpoint chkpt("test.chk",1);
  SCF solver=SCF(bas,set,chkpt);
  solver.UHF(sol,occa,occb,final_conv);
  // Compute dipole moment
  double dip=dip_mom(sol.P,bas);

#ifdef COMPUTE_REFERENCE
  printf("Etot=%.16e;\n",sol.en.E);
  printf("dip=%.16e;\n",dip);
  printf("Eorba=\"");
  for(size_t i=0;i<sol.Ea.n_elem;i++)
    printf("%.16e ",sol.Ea(i));
  printf("\";\n");
  printf("Eorbb=\"");
  for(size_t i=0;i<sol.Eb.n_elem;i++)
    printf("%.16e ",sol.Eb(i));
  printf("\";\n");
#else
  // Compare results
  bool Eok=1, Dok=1, ok=1;
  size_t nsucca=0, nfaila=0;
  size_t nsuccb=0, nfailb=0;
  compare(sol.Ea,Eorba,otol,nsucca,nfaila); // Compare orbital energies
  compare(sol.Eb,Eorbb,otol,nsuccb,nfailb); // Compare orbital energies
  size_t nsucc=nsucca+nsuccb;
  size_t nfail=nfaila+nfailb;

  Eok=rel_compare(sol.en.E,Etot,tol); // Compare total energies
  Dok=abs_compare(dip,dipmom,dtol); // Compare dipole moments
  ok=(Eok && Dok);
  printf("E=%f %s, dp=%f %s, orbital energies %i ok, %i failed (%s)\n",sol.en.E,stat[Eok],dip,stat[Dok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, difference in dipole moment is %e.\nMaximum differences of orbital energies are %e and %e.\n",rel_diff(sol.en.E,Etot),dip-dipmom,max_diff(sol.Ea,Eorba),max_diff(sol.Eb,Eorbb));

  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }
#endif
}

/// Test ROHF solution
#ifdef COMPUTE_REFERENCE
#define rohf_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom) rohf_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom); printf("rohf_test(" #at "," #baslib "," #set "," #Etot "," #Eorba "," #Eorbb "," #label "," #dipmom ");\n\n");
#else
#define rohf_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom) rohf_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom);
#endif
void rohf_test_run(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorba, const arma::vec & Eorbb, const std::string & label, double dipmom) {
  Timer t;

#ifndef COMPUTE_REFERENCE
  printf("%s, ",label.c_str());
  fflush(stdout);
#endif

  arma::vec Ea, Eb;
  arma::mat Ca, Cb;

  // Construct basis set
  BasisSet bas=construct_basis(at,baslib,set);
  int Nel_alpha;
  int Nel_beta;
  get_Nel_alpha_beta(bas.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);
  // Solve SCF equations
  uscf_t sol;
  Checkpoint chkpt("test.chk",1);
  SCF solver=SCF(bas,set,chkpt);
  solver.ROHF(sol,Nel_alpha,Nel_beta,final_conv);
  // Compute dipole moment
  double dip=dip_mom(sol.P,bas);

#ifdef COMPUTE_REFERENCE
  printf("Etot=%.16e;\n",sol.en.E);
  printf("dip=%.16e;\n",dip);
  printf("Eorba=\"");
  for(size_t i=0;i<sol.Ea.n_elem;i++)
    printf("%.16e ",sol.Ea(i));
  printf("\";\n");
  printf("Eorbb=\"");
  for(size_t i=0;i<sol.Eb.n_elem;i++)
    printf("%.16e ",sol.Eb(i));
  printf("\";\n");
#else
  // Compare results
  bool Eok=1, Dok=1, ok=1;
  size_t nsucca=0, nfaila=0;
  size_t nsuccb=0, nfailb=0;
  compare(sol.Ea,Eorba,otol,nsucca,nfaila); // Compare orbital energies
  compare(sol.Eb,Eorbb,otol,nsuccb,nfailb); // Compare orbital energies
  size_t nsucc=nsucca+nsuccb;
  size_t nfail=nfaila+nfailb;

  Eok=rel_compare(sol.en.E,Etot,tol); // Compare total energies
  Dok=abs_compare(dip,dipmom,dtol); // Compare dipole moments
  ok=(Eok && Dok);
  printf("E=%f %s, dp=%f %s, orbital energies %i ok, %i failed (%s)\n",sol.en.E,stat[Eok],dip,stat[Dok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, difference in dipole moment is %e.\nMaximum differences of orbital energies are %e and %e.\n",rel_diff(sol.en.E,Etot),dip-dipmom,max_diff(sol.Ea,Eorba),max_diff(sol.Eb,Eorbb));

  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }
#endif
}

/// Test RDFT solution
#ifdef COMPUTE_REFERENCE
#define rdft_test(at,baslib,set,Etot,Eorb,label,dipmom,xfunc,cfunc) rdft_test_run(at,baslib,set,Etot,Eorb,label,dipmom,xfunc,cfunc); printf("rdft_test(" #at "," #baslib "," #set "," #Etot "," #Eorb "," #label "," #dipmom "," #xfunc "," #cfunc ");\n\n");
#else
#define rdft_test(at,baslib,set,Etot,Eorb,label,dipmom,xfunc,cfunc) rdft_test_run(at,baslib,set,Etot,Eorb,label,dipmom,xfunc,cfunc);
#endif
void rdft_test_run(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorb, const std::string & label, double dipmom, int xfunc, int cfunc) {
  Timer t;

#ifndef COMPUTE_REFERENCE
  printf("%s, ",label.c_str());
  fflush(stdout);
#endif

  arma::vec E;
  arma::mat C;

  // Construct basis set
  BasisSet bas=construct_basis(at,baslib,set);
  // Get orbital occupancies
  std::vector<double> occs=get_restricted_occupancy(set,bas);
  // Solve SCF equations
  rscf_t sol;
  Checkpoint chkpt("test.chk",1);
  SCF solver=SCF(bas,set,chkpt);

  // Check normalization of basis
  check_norm(bas);

  // Final dft settings
  dft_t dft_f;
  dft_f.x_func=xfunc;
  dft_f.c_func=cfunc;
  dft_f.gridtol=dft_finaltol;

  // Initial dft settings
  dft_t dft_i(dft_f);
  dft_i.gridtol=dft_initialtol;

  solver.RDFT(sol,occs,init_conv,dft_i);
  solver.RDFT(sol,occs,final_conv,dft_f);
  // Compute dipole moment
  double dip=dip_mom(sol.P,bas);

#ifdef COMPUTE_REFERENCE
  printf("Etot=%.16e;\n",sol.en.E);
  printf("dip=%.16e;\n",dip);
  printf("Eorb=\"");
  for(size_t i=0;i<sol.E.n_elem;i++)
    printf("%.16e ",sol.E(i));
  printf("\";\n");
#else
  // Compare results
  bool Eok=1, Dok=1, ok=1;
  size_t nsucc=0, nfail=0;
  compare(sol.E,Eorb,otol,nsucc,nfail); // Compare orbital energies

  Eok=rel_compare(sol.en.E,Etot,tol); // Compare total energies
  Dok=abs_compare(dip,dipmom,dtol); // Compare dipole moments
  ok=(Eok && Dok);
  printf("E=%f %s, dp=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],dip,stat[Dok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, difference in dipole moment is %e.\nMaximum difference of orbital energy is %e.\n",rel_diff(sol.en.E,Etot),dip-dipmom,max_diff(sol.E,Eorb));

  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }
#endif
}

/// Test UDFT solution
#ifdef COMPUTE_REFERENCE
#define udft_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,xfunc,cfunc) udft_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,xfunc,cfunc); printf("udft_test(" #at "," #baslib "," #set "," #Etot "," #Eorba "," #Eorbb "," #label "," #dipmom "," #xfunc "," #cfunc ");\n\n");
#else
#define udft_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,xfunc,cfunc) udft_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,xfunc,cfunc);
#endif
void udft_test_run(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorba, const arma::vec & Eorbb, const std::string & label, double dipmom, int xfunc, int cfunc) {
  Timer t;

#ifndef COMPUTE_REFERENCE
  printf("%s, ",label.c_str());
  fflush(stdout);
#endif

  arma::vec Ea, Eb;
  arma::mat Ca, Cb;

  // Final dft settings
  dft_t dft_f;
  dft_f.x_func=xfunc;
  dft_f.c_func=cfunc;
  dft_f.gridtol=dft_finaltol;

  // Initial dft settings
  dft_t dft_i(dft_f);
  dft_i.gridtol=dft_initialtol;

  // Construct basis set
  BasisSet bas=construct_basis(at,baslib,set);
  // Get orbital occupancies
  std::vector<double> occa, occb;
  get_unrestricted_occupancy(set,bas,occa,occb);
  // Solve SCF equations
  uscf_t sol;
  Checkpoint chkpt("test.chk",1);
  SCF solver=SCF(bas,set,chkpt);
  solver.UDFT(sol,occa,occb,init_conv,dft_i);
  solver.UDFT(sol,occa,occb,final_conv,dft_f);
  // Compute dipole moment
  double dip=dip_mom(sol.P,bas);

#ifdef COMPUTE_REFERENCE
  printf("Etot=%.16e;\n",sol.en.E);
  printf("dip=%.16e;\n",dip);
  printf("Eorba=\"");
  for(size_t i=0;i<sol.Ea.n_elem;i++)
    printf("%.16e ",sol.Ea(i));
  printf("\";\n");
  printf("Eorbb=\"");
  for(size_t i=0;i<sol.Eb.n_elem;i++)
    printf("%.16e ",sol.Eb(i));
  printf("\";\n");
#else
  // Compare results
  bool Eok=1, Dok=1, ok=1;
  size_t nsucca=0, nfaila=0;
  size_t nsuccb=0, nfailb=0;
  compare(sol.Ea,Eorba,otol,nsucca,nfaila); // Compare orbital energies
  compare(sol.Eb,Eorbb,otol,nsuccb,nfailb); // Compare orbital energies
  size_t nsucc=nsucca+nsuccb;
  size_t nfail=nfaila+nfailb;

  Eok=rel_compare(sol.en.E,Etot,tol); // Compare total energies
  Dok=abs_compare(dip,dipmom,dtol); // Compare dipole moments
  ok=(Eok && Dok);
  printf("E=%f %s, dp=%f %s, orbital energies %i ok, %i failed (%s)\n",sol.en.E,stat[Eok],dip,stat[Dok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, difference in dipole moment is %e.\nMaximum differences of orbital energies are %e and %e.\n",rel_diff(sol.en.E,Etot),dip-dipmom,max_diff(sol.Ea,Eorba),max_diff(sol.Eb,Eorbb));

  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }
#endif
}

/// Run unit tests by comparing calculations to ones that should be OK
int main(void) {

  // Initialize libint
  init_libint_base();

  // First, check norms of spherical harmonics.
  check_sph_orthonorm(Lmax);
  printf("Solid harmonics OK.\n");

  // Load basis sets

  // Redirect stderr to file, since scf routines print out info there.
#ifdef COMPUTE_REFERENCE
  FILE *errors;
  errors=freopen("errors.log","w",stderr);
  // Avoid unused variable warning
  (void) errors;
#endif

  printf("****** Loading basis sets *******\n");

  BasisSetLibrary b3_21G;
  b3_21G.load_gaussian94("3-21G");

  BasisSetLibrary b6_31Gpp;
  b6_31Gpp.load_gaussian94("6-31G**");

  BasisSetLibrary cc_pVDZ;
  cc_pVDZ.load_gaussian94("cc-pVDZ");

  BasisSetLibrary cc_pVTZ;
  cc_pVTZ.load_gaussian94("cc-pVTZ");

  BasisSetLibrary cc_pVQZ;
  cc_pVQZ.load_gaussian94("cc-pVQZ");

  BasisSetLibrary aug_cc_pVDZ;
  aug_cc_pVDZ.load_gaussian94("aug-cc-pVDZ");

  /*
  BasisSetLibrary cc_pV5Z;
  cc_pV5Z.load_gaussian94("cc-pV5Z");

  BasisSetLibrary cc_pV6Z;
  cc_pV6Z.load_gaussian94("cc-pV6Z");

  BasisSetLibrary aug_cc_pVTZ;
  aug_cc_pVTZ.load_gaussian94("aug-cc-pVTZ");

  BasisSetLibrary aug_cc_pVQZ;
  aug_cc_pVQZ.load_gaussian94("aug-cc-pVQZ");
  */

  // Helper structure
  atom_t at;

  // Neon atom
  std::vector<atom_t> Ne;
  at.el="Ne"; at.x=0.0; at.y=0.0; at.z=0.0; Ne.push_back(convert_to_bohr(at));

  // Chlorine
  std::vector<atom_t> Cl;
  at.el="Cl"; at.x=0.0; at.y=0.0; at.z=0.0; Cl.push_back(convert_to_bohr(at));

  // Hydrogen molecule
  std::vector<atom_t> H2;
  at.el="H"; at.x=0.0; at.y=0.0; at.z=-0.37; H2.push_back(convert_to_bohr(at));
  at.el="H"; at.x=0.0; at.y=0.0; at.z= 0.37; H2.push_back(convert_to_bohr(at));

  // Water monomer optimized at B3LYP/aug-cc-pVTZ level
  std::vector<atom_t> h2o;
  at.el="O"; at.x= 0.000000; at.y= 0.117030; at.z=0.000000; h2o.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 0.763404; at.y=-0.468123; at.z=0.000000; h2o.push_back(convert_to_bohr(at));
  at.el="H"; at.x=-0.763404; at.y=-0.468123; at.z=0.000000; h2o.push_back(convert_to_bohr(at));

  // Cadmium complex
  std::vector<atom_t> cdcplx;
  at.el="Cd"; at.x= 0.000000; at.y= 0.000000; at.z= 0.000000; cdcplx.push_back(convert_to_bohr(at));
  at.el="N";  at.x= 0.000000; at.y= 0.000000; at.z=-2.260001; cdcplx.push_back(convert_to_bohr(at));
  at.el="N";  at.x=-0.685444; at.y= 0.000000; at.z=-4.348035; cdcplx.push_back(convert_to_bohr(at));
  at.el="C";  at.x= 0.676053; at.y= 0.000000; at.z=-4.385069; cdcplx.push_back(convert_to_bohr(at));
  at.el="C";  at.x= 1.085240; at.y= 0.000000; at.z=-3.091231; cdcplx.push_back(convert_to_bohr(at));
  at.el="C";  at.x=-1.044752; at.y= 0.000000; at.z=-3.060220; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x= 1.231530; at.y= 0.000000; at.z=-5.300759; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x= 2.088641; at.y= 0.000000; at.z=-2.711077; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x=-2.068750; at.y= 0.000000; at.z=-2.726515; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x=-1.313170; at.y= 0.000000; at.z=-5.174718; cdcplx.push_back(convert_to_bohr(at));

  // 1-decanol
  std::vector<atom_t> decanol;
  at.el="C"; at.x= 3.951300; at.y= 3.953900; at.z= 3.422300; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 5.374200; at.y= 3.650100; at.z= 2.991900; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 5.462800; at.y= 3.376900; at.z= 1.498000; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 3.868100; at.y= 4.261800; at.z= 4.909900; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 6.848500; at.y= 2.886000; at.z= 1.103900; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 2.476900; at.y= 4.734000; at.z= 5.303900; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 7.148200; at.y= 1.524300; at.z= 1.712000; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 2.137100; at.y= 6.069500; at.z= 4.659400; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 8.591100; at.y= 1.087300; at.z= 1.479900; decanol.push_back(convert_to_bohr(at));
  at.el="C"; at.x= 0.698900; at.y= 6.461200; at.z= 4.906200; decanol.push_back(convert_to_bohr(at));
  at.el="O"; at.x= 9.420700; at.y= 1.797400; at.z= 2.371700; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 3.287600; at.y= 3.101000; at.z= 3.176500; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 3.544800; at.y= 4.820400; at.z= 2.851400; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 5.764700; at.y= 2.762000; at.z= 3.541600; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 6.048100; at.y= 4.485600; at.z= 3.267600; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 5.215200; at.y= 4.295200; at.z= 0.930600; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 4.700900; at.y= 2.627700; at.z= 1.202600; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 4.616500; at.y= 5.034600; at.z= 5.179100; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 4.142500; at.y= 3.363300; at.z= 5.496700; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 6.926800; at.y= 2.833000; at.z= 0.000000; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 7.613500; at.y= 3.620500; at.z= 1.424900; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 2.408100; at.y= 4.818800; at.z= 6.406100; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 1.724700; at.y= 3.972500; at.z= 5.015100; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 6.912400; at.y= 1.558800; at.z= 2.801200; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 6.470000; at.y= 0.761200; at.z= 1.282600; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 2.344300; at.y= 6.002200; at.z= 3.566900; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 2.817300; at.y= 6.856200; at.z= 5.040700; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 8.698800; at.y= 0.000000; at.z= 1.663000; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 8.897200; at.y= 1.276900; at.z= 0.431600; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 0.461600; at.y= 7.423200; at.z= 4.435000; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 0.000000; at.y= 5.716800; at.z= 4.502200; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 0.486000; at.y= 6.559000; at.z= 5.979000; decanol.push_back(convert_to_bohr(at));
  at.el="H"; at.x=10.311700; at.y= 1.558300; at.z= 2.157100; decanol.push_back(convert_to_bohr(at));

  // Construct settings
  Settings sph;
  sph.add_scf_settings();
  sph.set_bool("Verbose",false);
  // Use core guess for tests.
  sph.set_bool("CoreGuess",true);

  // No spherical harmonics
  Settings cart=sph;
  cart.set_bool("UseLM",false);

  // Direct calculation
  Settings direct=sph;
  direct.set_bool("Direct",true);

  // Polarized calculation
  Settings pol=sph;
  pol.set_int("Multiplicity",2);
  pol.set_double("DFTInitialTol",1e-4);

  // DFT tests

  // Settings for DFT
  Settings dftsph=sph; // Normal settings
  dftsph.add_dft_settings();

  Settings dftcart=cart; // Cartesian basis
  dftcart.add_dft_settings();

  Settings dftnofit=dftsph; // No density fitting
  dftnofit.set_bool("DFTFitting",false);

  Settings dftcart_nofit=dftcart;
  dftcart_nofit.set_bool("DFTFitting",false);

  Settings dftdirect=dftsph; // Direct calculation
  dftdirect.set_bool("Direct",true);
  dftdirect.set_bool("DFTDirect",true);

  Settings dftpol=pol; // Polarized calculation
  dftpol.add_dft_settings();

  Settings dftpol_nofit=dftpol; // Polarized calculation, no density fitting
  dftpol_nofit.set_bool("DFTFitting",false);

  printf("****** Running calculations *******\n");
  Timer t;

  // Reference total energy
  double Etot;
  // Reference dipole moment
  double dip;
  // Reference orbital energies
  arma::vec Eorb;
  arma::vec Eorba;
  arma::vec Eorbb;


Etot=-1.2848877555174064e+02;
dip=1.0935477903750759e-15;
Eorb="-3.2765635418561985e+01 -1.9187982340182725e+00 -8.3209725199383955e-01 -8.3209725199383811e-01 -8.3209725199383322e-01 1.6945577282677746e+00 1.6945577282677786e+00 1.6945577282677813e+00 2.1594249508221424e+00 5.1967114014293987e+00 5.1967114014293996e+00 5.1967114014294022e+00 5.1967114014294058e+00 5.1967114014294067e+00 ";
rhf_test(Ne,cc_pVDZ,sph,Etot,Eorb,"Neon, HF/cc-pVDZ",dip);

Etot=-1.2848886617203743e+02;
dip=5.2006754423565525e-16;
Eorb="-3.2765400794020664e+01 -1.9190111542734112e+00 -8.3228219763730160e-01 -8.3228219763729983e-01 -8.3228219763729228e-01 1.6944246985986735e+00 1.6944246985986762e+00 1.6944246985986795e+00 1.9905987677722516e+00 5.1964245988049127e+00 5.1964245988049154e+00 5.1964245988049216e+00 5.1964245988049251e+00 5.1964245988049269e+00 1.0383358434342473e+01 ";
rhf_test(Ne,cc_pVDZ,cart,Etot,Eorb,"Neon, HF/cc-pVDZ cart",dip);

Etot=-1.2853186163632134e+02;
dip=1.0110825798481719e-15;
Eorb="-3.2769110714442625e+01 -1.9270833039331179e+00 -8.4541551017125605e-01 -8.4541551017125316e-01 -8.4541551017124839e-01 1.0988680367309738e+00 1.0988680367309787e+00 1.0988680367309851e+00 1.4176388079666651e+00 2.8142175659544875e+00 2.8142175659544901e+00 2.8142175659544915e+00 2.8142175659544928e+00 2.8142175659544986e+00 6.1558667266260443e+00 6.1558667266260478e+00 6.1558667266260612e+00 9.6473695825589161e+00 9.6473695825589232e+00 9.6473695825589285e+00 9.6473695825589338e+00 9.6473695825589356e+00 9.6473695825589392e+00 9.6473695825589427e+00 1.1227312685739026e+01 1.1227312685739037e+01 1.1227312685739051e+01 1.1227312685739053e+01 1.1227312685739058e+01 1.1744558070672406e+01 ";
rhf_test(Ne,cc_pVTZ,sph,Etot,Eorb,"Neon, HF/cc-pVTZ",dip);

Etot=-1.2853200998517832e+02;
dip=1.9212574334045013e-15;
Eorb="-3.2769827641505223e+01 -1.9274545147299600e+00 -8.4572301645475589e-01 -8.4572301645473824e-01 -8.4572301645473424e-01 8.8038911931132613e-01 1.0282198417367483e+00 1.0282198417367601e+00 1.0282198417367665e+00 2.8138968475434680e+00 2.8138968475434760e+00 2.8138968475434809e+00 2.8138968475434849e+00 2.8138968475434898e+00 4.1362240335656937e+00 4.6398467078786210e+00 4.6398467078786929e+00 4.6398467078787986e+00 9.6470056647257536e+00 9.6470056647257696e+00 9.6470056647257820e+00 9.6470056647257909e+00 9.6470056647257945e+00 9.6470056647258033e+00 9.6470056647258247e+00 1.1226914499430476e+01 1.1226914499430499e+01 1.1226914499430517e+01 1.1226914499430523e+01 1.1226914499430537e+01 1.1317534802704253e+01 1.1317534802704481e+01 1.1317534802704563e+01 1.6394442679721312e+01 2.8816114661027299e+01 ";
rhf_test(Ne,cc_pVTZ,cart,Etot,Eorb,"Neon, HF/cc-pVTZ cart",dip);

Etot=-1.2854346965912140e+02;
dip=1.2488981360131299e-15;
Eorb="-3.2771496241762378e+01 -1.9293376415940271e+00 -8.4895896339474886e-01 -8.4895896339474375e-01 -8.4895896339473420e-01 8.0890413629067626e-01 8.0890413629068003e-01 8.0890413629068270e-01 9.3559988696303109e-01 1.9978112778522696e+00 1.9978112778522776e+00 1.9978112778522827e+00 1.9978112778522850e+00 1.9978112778522936e+00 3.9328189024751059e+00 3.9328189024751325e+00 3.9328189024751552e+00 5.8106845385601034e+00 5.9042211353633700e+00 5.9042211353633718e+00 5.9042211353633762e+00 5.9042211353633798e+00 5.9042211353633816e+00 5.9042211353633833e+00 5.9042211353633993e+00 6.7616951502847469e+00 6.7616951502847487e+00 6.7616951502847664e+00 6.7616951502847815e+00 6.7616951502847993e+00 1.4903626158026730e+01 1.4903626158026732e+01 1.4903626158026739e+01 1.4903626158026762e+01 1.4903626158026762e+01 1.4903626158026766e+01 1.4903626158026771e+01 1.4903626158026778e+01 1.4903626158026814e+01 1.5804420579007040e+01 1.5804420579007056e+01 1.5804420579007093e+01 1.9794585637786362e+01 1.9794585637786366e+01 1.9794585637786387e+01 1.9794585637786405e+01 1.9794585637786426e+01 1.9794585637786430e+01 1.9794585637786454e+01 2.0954549899743192e+01 2.0954549899743263e+01 2.0954549899743281e+01 2.0954549899743366e+01 2.0954549899743370e+01 6.6550956467622356e+01 ";
rhf_test(Ne,cc_pVQZ,sph,Etot,Eorb,"Neon, HF/cc-pVQZ",dip);

Etot=-1.2854353449722353e+02;
dip=5.1777815354287719e-15;
Eorb="-3.2771625128940265e+01 -1.9294942839210425e+00 -8.4906688454351353e-01 -8.4906688454347634e-01 -8.4906688454341839e-01 5.8690441478497923e-01 7.1271797763397526e-01 7.1271797763412525e-01 7.1271797763419475e-01 1.9879845922667230e+00 1.9879845922667607e+00 1.9879845922667736e+00 1.9879845922668016e+00 1.9879845922668242e+00 2.5105148503731662e+00 2.7214792302635815e+00 2.7214792302641504e+00 2.7214792302643742e+00 5.9040962890184785e+00 5.9040962890184812e+00 5.9040962890185167e+00 5.9040962890185185e+00 5.9040962890185229e+00 5.9040962890185327e+00 5.9040962890185673e+00 6.4115733392332599e+00 6.5684069305207888e+00 6.5684069305207977e+00 6.5684069305208075e+00 6.5684069305208759e+00 6.5684069305208901e+00 6.7659166003445694e+00 6.7659166003453359e+00 6.7659166003458777e+00 1.4004805313821324e+01 1.4903514354965671e+01 1.4903514354965688e+01 1.4903514354965713e+01 1.4903514354965743e+01 1.4903514354965752e+01 1.4903514354965797e+01 1.4903514354965818e+01 1.4903514354965854e+01 1.4903514354965926e+01 1.8145155385580495e+01 1.8145155385582893e+01 1.8145155385584918e+01 1.8145155385585433e+01 1.8145155385586435e+01 1.8540067452850035e+01 1.8540067452850085e+01 1.8540067452850781e+01 1.9794449045610229e+01 1.9794449045610243e+01 1.9794449045610264e+01 1.9794449045610278e+01 1.9794449045610317e+01 1.9794449045610357e+01 1.9794449045610381e+01 2.9727979557286790e+01 3.9089870736544519e+01 3.9089870736600275e+01 3.9089870736626906e+01 3.9089870736635611e+01 3.9089870736666825e+01 3.9551871550329658e+01 3.9551871550337090e+01 3.9551871550341211e+01 5.8376821812408984e+01 2.0568373998146996e+02 ";
rhf_test(Ne,cc_pVQZ,cart,Etot,Eorb,"Neon, HF/cc-pVQZ cart",dip);

Etot=-4.5944650895295206e+02;
dip=4.0917538212801623e-15;
Eorba="-1.0487192686721927e+02 -1.0607731071925958e+01 -8.0945432699289519e+00 -8.0689327613839126e+00 -8.0689327613838984e+00 -1.1333588836797468e+00 -5.7594497897942076e-01 -5.0142013017816112e-01 -5.0142013017816067e-01 5.0038038865765921e-01 5.6758183301189058e-01 6.0901889938650189e-01 6.0901889938650589e-01 1.0467142869001447e+00 1.0627768645650262e+00 1.0627768645650284e+00 1.1177572002437157e+00 1.1177572002437193e+00 ";
Eorbb="-1.0486051303418067e+02 -1.0596350263578435e+01 -8.0629604795547163e+00 -8.0629604795546985e+00 -8.0461996891861585e+00 -1.0112315256196238e+00 -4.7613909673129096e-01 -4.7613909673127908e-01 -3.7645538720355740e-02 5.2464103240676652e-01 6.1785782465953509e-01 6.1785782465954531e-01 6.6235454616556888e-01 1.1278253592469247e+00 1.1278253592469270e+00 1.1604789862920335e+00 1.1604789862920357e+00 1.1734240348555958e+00 ";
uhf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, HF/6-31G** polarized",dip);

Etot=-4.5944278096474716e+02;
dip=2.4186661747145860e-15;
Eorba="-1.0487302754877020e+02 -1.0608495044088627e+01 -8.0951284766627083e+00 -8.0696303359015200e+00 -8.0696303359015111e+00 -1.1298654784171371e+00 -5.6816383676262838e-01 -5.0428443835496573e-01 -5.0428443835495917e-01 5.0036401839302014e-01 5.6967762505975528e-01 6.0847101029745887e-01 6.0847101029746986e-01 1.0477284905464330e+00 1.0645039353681489e+00 1.0645039353681538e+00 1.1152518467697132e+00 1.1152518467697159e+00 ";
Eorbb="-1.0486247998616315e+02 -1.0598189445957557e+01 -8.0647978369114135e+00 -8.0647978369114046e+00 -8.0481963318778575e+00 -1.0119816546887981e+00 -4.7452860520919704e-01 -4.7452860520919460e-01 -4.4311866777204216e-02 5.2403723359506971e-01 6.1759764962106267e-01 6.1759764962108399e-01 6.5850376781959752e-01 1.1291203810033386e+00 1.1291203810033508e+00 1.1573546833101565e+00 1.1573546833101760e+00 1.1669379491891998e+00 ";
rohf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, ROHF/6-31G**",dip);

Etot=-4.6013022813814212e+02;
dip=9.1092287694315848e-15;
Eorba="-1.0170613249846950e+02 -9.6270392241155047e+00 -7.4214820723514583e+00 -7.3996431182435218e+00 -7.3996431177632580e+00 -8.8861424907121045e-01 -4.5634985498061748e-01 -4.0418732489213177e-01 -4.0418732457474904e-01 2.5058371382327643e-01 3.1404058767211673e-01 3.4409216794344610e-01 3.4409216811999244e-01 7.4272102849222588e-01 7.5561779226967118e-01 7.5561779263651685e-01 7.9852485964713238e-01 7.9852485965416953e-01 ";
Eorbb="-1.0170188025690520e+02 -9.6221026078303353e+00 -7.4057879873886643e+00 -7.3958706291835865e+00 -7.3958706291296474e+00 -8.4901906883125078e-01 -3.8947165375645643e-01 -3.8947164056288541e-01 -3.3071643856543731e-01 2.5640708896978415e-01 3.2914291803171625e-01 3.4938759470325514e-01 3.4938761720103434e-01 7.8954077235568143e-01 7.9792008643072021e-01 7.9792013930810379e-01 8.1008383174932208e-01 8.1008383296140296e-01 ";
udft_test(Cl,b6_31Gpp,dftpol_nofit,Etot,Eorba,Eorbb,"Chlorine, B3LYP/6-31G** polarized",dip,402,0);

Etot=-1.1287000934442006e+00;
dip=9.0679309369372219e-16;
Eorb="-5.9241098675988113e-01 1.9744005678171092e-01 4.7932104279452348e-01 9.3732369034517815e-01 1.2929037077598993e+00 1.2929037077599004e+00 1.9570226068896537e+00 2.0435200521330916e+00 2.0435200521330934e+00 3.6104742320374132e+00 ";
rhf_test(H2,cc_pVDZ,sph,Etot,Eorb,"Hydrogen molecule, HF/cc-pVDZ",dip);

Etot=-1.1676141182477320e+00;
dip=5.2414907308513145e-14;
Eorb="-3.9201515930900610e-01 3.6518736496390099e-02 2.9071356473378096e-01 6.5832910702777681e-01 9.7502281428651238e-01 9.7502281428651671e-01 1.6066119799002065e+00 1.7001805817601925e+00 1.7001805817601963e+00 3.1926513611695513e+00 ";
rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, SVWN(RPA)/cc-pVDZ",dip,1,8);

Etot=-1.1603966181999295e+00;
dip=5.6414866683589043e-14;
Eorb="-3.7849174586616247e-01 5.3520825347398271e-02 3.0277169431072920e-01 6.6374632575268444e-01 9.9246453656851230e-01 9.9246453656851807e-01 1.6235424258671627e+00 1.7198877535389772e+00 1.7198877535389816e+00 3.2019321531425082e+00 ";
rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, PBEPBE/cc-pVDZ",dip,101,130);

Etot=-7.6056825377225863e+01;
dip=7.9472744117421690e-01;
Eorb="-2.0555281284980545e+01 -1.3428635570454313e+00 -7.0828437085196583e-01 -5.7575384621878445e-01 -5.0391498187500083e-01 1.4187951535871338e-01 2.0351537907851339e-01 5.4324870338348141e-01 5.9753586713945417e-01 6.6949546706731300e-01 7.8747678840851099e-01 8.0274150139238387e-01 8.0481260678513500e-01 8.5898803317571559e-01 9.5702121697681419e-01 1.1344778938628284e+00 1.1928203497604379e+00 1.5241753059557281e+00 1.5579529863767783e+00 2.0324408680812991e+00 2.0594682913995244e+00 2.0654407645533914e+00 2.1686553949150058e+00 2.2363161850313631e+00 2.5909431266905605e+00 2.9581971180394886e+00 3.3610002617098331e+00 3.4914002741128964e+00 3.5741938457711209e+00 3.6463660396926354e+00 3.7977214203032847e+00 3.8739670187009656e+00 3.8824466769763886e+00 3.9569498237220886e+00 4.0199059042116447e+00 4.0760332599918581e+00 4.1862021901759450e+00 4.3092789369299460e+00 4.3875716374145686e+00 4.5640073747294405e+00 4.6817931162277757e+00 4.8550947794811998e+00 5.1380848595366455e+00 5.2500191164845154e+00 5.5275547744780091e+00 6.0402478774596586e+00 6.5453259374506931e+00 6.9113516604359448e+00 6.9366142639201964e+00 7.0003720370423421e+00 7.0078239226349899e+00 7.0609382554020810e+00 7.1598075600263478e+00 7.2256524647845213e+00 7.4561719737155148e+00 7.7799625470240032e+00 8.2653639951677800e+00 1.2804358854760078e+01 ";
rhf_test(h2o,cc_pVTZ,sph,Etot,Eorb,"Water, HF/cc-pVTZ",dip);

Etot=-7.6056825376389909e+01;
dip=7.9472743842895532e-01;
Eorb="-2.0555281283650558e+01 -1.3428635575838916e+00 -7.0828437205736605e-01 -5.7575384686897313e-01 -5.0391498233663234e-01 1.4187951525480322e-01 2.0351537894455032e-01 5.4324870308990436e-01 5.9753586665549085e-01 6.6949546697884510e-01 7.8747678844627400e-01 8.0274150108003028e-01 8.0481260592562653e-01 8.5898803276164748e-01 9.5702121659972506e-01 1.1344778966675380e+00 1.1928203495149399e+00 1.5241753049423556e+00 1.5579529865688551e+00 2.0324408676067249e+00 2.0594682883502147e+00 2.0654407639964516e+00 2.1686553934379891e+00 2.2363161865612931e+00 2.5909431294010727e+00 2.9581971145530672e+00 3.3610002617425945e+00 3.4914002758300762e+00 3.5741938446661701e+00 3.6463660350903866e+00 3.7977214194813653e+00 3.8739670155065702e+00 3.8824466821182901e+00 3.9569498202078681e+00 4.0199059065227676e+00 4.0760332557263981e+00 4.1862021978054553e+00 4.3092789332373700e+00 4.3875716441377506e+00 4.5640073605187048e+00 4.6817931156722228e+00 4.8550947724011744e+00 5.1380848486225981e+00 5.2500191167131272e+00 5.5275547692519194e+00 6.0402478774126438e+00 6.5453259352285107e+00 6.9113516608327474e+00 6.9366142634848158e+00 7.0003720368018305e+00 7.0078239233690063e+00 7.0609382515161574e+00 7.1598075591295842e+00 7.2256524623351481e+00 7.4561719743169093e+00 7.7799625492131845e+00 8.2653640006675584e+00 1.2804358799394166e+01 ";
rhf_test(h2o,cc_pVTZ,direct,Etot,Eorb,"Water, HF/cc-pVTZ direct",dip);

Etot=-7.6064480528902067e+01;
dip=7.8765851934599063e-01;
Eorb="-2.0560341500359886e+01 -1.3467109726751991e+00 -7.1286865369593333e-01 -5.7999183567935142e-01 -5.0759009672422406e-01 1.1677961755798599e-01 1.7061237371511614e-01 4.4878631163263444e-01 4.6240974135182406e-01 4.9860081955072005e-01 5.8389461617055827e-01 6.0602249078779791e-01 6.1386901242935699e-01 6.5509670352031113e-01 7.1940581515916280e-01 8.5193265900818493e-01 9.1760642849329954e-01 1.1091391721934816e+00 1.1559117533805503e+00 1.3479064849403748e+00 1.4144381964398940e+00 1.4776186115666217e+00 1.4856774095022904e+00 1.5814608160486687e+00 1.6854835725164254e+00 1.9096187745183197e+00 2.0727777034852388e+00 2.1976502978066605e+00 2.2888869979994788e+00 2.3588905650000118e+00 2.4246094540338001e+00 2.4837778802332124e+00 2.5224544361244328e+00 2.5800657964329661e+00 2.5803867667647573e+00 2.6507304508857579e+00 2.6683130248373033e+00 2.8407379959161658e+00 2.8643130328433752e+00 3.0412098305491284e+00 3.1190680437556675e+00 3.2889763527301565e+00 3.3518967317892936e+00 3.4467312463445317e+00 3.6214003286951653e+00 3.8285931971768759e+00 3.9968185327771377e+00 4.1278766612708884e+00 4.1879463035004907e+00 4.2176486589022399e+00 4.4343620886273429e+00 4.4925098798599246e+00 4.6832772416010791e+00 4.7403725803114991e+00 4.8079058224308460e+00 4.9140701220222489e+00 5.3503959761452657e+00 5.4039303380563846e+00 5.9860940846909232e+00 6.1030498809766778e+00 6.2449376421651115e+00 6.3029981824418924e+00 6.7000543808304238e+00 6.7926548567357568e+00 7.0589633249054868e+00 7.2683601254266508e+00 7.3171930274066055e+00 7.3671378833761239e+00 7.4371265016060377e+00 7.5184752972461508e+00 7.5458434103189598e+00 7.5694204764327626e+00 8.0046360219757080e+00 8.0708295764575322e+00 8.0987711897068202e+00 8.1338237926744306e+00 8.1523664214459206e+00 8.2695443440422487e+00 8.3150962725086845e+00 8.3485048876528580e+00 8.4164827932157920e+00 8.6181288313280007e+00 8.8336406106007690e+00 8.9048326534998790e+00 8.9437734454537541e+00 9.2166367028856548e+00 9.3761387931300231e+00 9.3791690876804878e+00 9.9423093863498746e+00 1.0035594104197978e+01 1.0257561211989241e+01 1.0425629822857662e+01 1.0646599818002985e+01 1.0757780268307986e+01 1.0806846319402116e+01 1.1272406010308352e+01 1.1390414020121082e+01 1.1595907192698439e+01 1.1644666358571957e+01 1.1693629518556691e+01 1.1844883874208410e+01 1.2158546422045477e+01 1.2320144573670445e+01 1.2398213789889716e+01 1.2413264232259918e+01 1.2465013636301427e+01 1.3602019536639743e+01 1.3763660839109356e+01 1.4247885731484562e+01 1.4614348061642533e+01 1.4639079461745965e+01 1.4826337835534442e+01 1.6435472094802478e+01 1.6799107183274430e+01 4.4322817448607211e+01 ";
rhf_test(h2o,cc_pVQZ,sph,Etot,Eorb,"Water, HF/cc-pVQZ",dip);

Etot=-7.6064480532894464e+01;
dip=7.8765852829945349e-01;
Eorb="-2.0560341501794230e+01 -1.3467109748254695e+00 -7.1286865515228426e-01 -5.7999183602599524e-01 -5.0759009554548806e-01 1.1677962159723451e-01 1.7061237606545807e-01 4.4878630937417852e-01 4.6240974615361402e-01 4.9860082020167235e-01 5.8389461556542421e-01 6.0602248981630780e-01 6.1386901217772671e-01 6.5509670133561348e-01 7.1940581423805738e-01 8.5193266449959648e-01 9.1760642553949823e-01 1.1091391807989668e+00 1.1559117485474897e+00 1.3479064723053105e+00 1.4144382155564030e+00 1.4776186086330960e+00 1.4856774027636654e+00 1.5814608134232970e+00 1.6854835724328114e+00 1.9096187683538797e+00 2.0727777001708061e+00 2.1976503024062231e+00 2.2888869945502428e+00 2.3588905518313186e+00 2.4246094540405227e+00 2.4837778805044048e+00 2.5224544207033963e+00 2.5800657871946231e+00 2.5803867667686946e+00 2.6507304360788289e+00 2.6683130293882940e+00 2.8407379668430539e+00 2.8643130433007635e+00 3.0412098376988399e+00 3.1190680040851775e+00 3.2889763027601280e+00 3.3518967752450362e+00 3.4467312991112848e+00 3.6214003080743233e+00 3.8285932279982715e+00 3.9968185312538500e+00 4.1278766232158626e+00 4.1879462864453094e+00 4.2176486438523382e+00 4.4343620561357113e+00 4.4925098621327129e+00 4.6832772455365816e+00 4.7403725630343745e+00 4.8079057047151448e+00 4.9140700692426815e+00 5.3503959102955498e+00 5.4039304326344020e+00 5.9860940866273298e+00 6.1030499066482644e+00 6.2449376318635519e+00 6.3029981723377899e+00 6.7000543612490944e+00 6.7926548344996522e+00 7.0589632914617617e+00 7.2683600651074176e+00 7.3171930263347047e+00 7.3671378845671294e+00 7.4371264607869323e+00 7.5184751961603737e+00 7.5458434379129891e+00 7.5694205376576953e+00 8.0046360305260738e+00 8.0708295815174562e+00 8.0987711931032393e+00 8.1338237872618251e+00 8.1523664037125307e+00 8.2695443527152559e+00 8.3150962646753896e+00 8.3485048801239774e+00 8.4164827140603506e+00 8.6181287897516139e+00 8.8336405773029547e+00 8.9048326988694217e+00 8.9437733342770329e+00 9.2166365374307500e+00 9.3761387758083554e+00 9.3791690965753194e+00 9.9423092291189263e+00 1.0035594175842718e+01 1.0257561180183288e+01 1.0425629800408315e+01 1.0646599823453228e+01 1.0757780240235824e+01 1.0806846286169890e+01 1.1272406087716673e+01 1.1390413922159285e+01 1.1595907186469310e+01 1.1644666306168887e+01 1.1693629565397693e+01 1.1844883851614796e+01 1.2158546313586571e+01 1.2320144543167183e+01 1.2398213735795641e+01 1.2413264191223064e+01 1.2465013573622400e+01 1.3602019454747721e+01 1.3763660418687198e+01 1.4247885692685630e+01 1.4614347843375278e+01 1.4639079540371952e+01 1.4826337737337534e+01 1.6435472086640079e+01 1.6799106950111515e+01 4.4322817048512618e+01 ";
rhf_test(h2o,cc_pVQZ,direct,Etot,Eorb,"Water, HF/cc-pVQZ direct",dip);

Etot=-7.6372964544447399e+01;
dip=7.3249256921470296e-01;
Eorb="-1.8738949549145836e+01 -9.1586673382631767e-01 -4.7160108832110181e-01 -3.2507321775160658e-01 -2.4816690605937122e-01 8.4538723156205153e-03 8.1125678908667218e-02 3.3806728074863740e-01 3.7949162463553593e-01 4.6548870132121983e-01 5.4539128007423410e-01 5.9072331145094548e-01 5.9687821317390055e-01 6.4398052716222776e-01 7.4418109823671841e-01 8.8306700126951176e-01 9.7381866155329189e-01 1.2412764626995456e+00 1.2611345911143275e+00 1.6999126349434914e+00 1.7261711603940662e+00 1.7619585691963691e+00 1.8256262273164632e+00 1.9002763475172595e+00 2.1846823179760531e+00 2.5326904001215511e+00 2.9875043893565625e+00 3.1260226754020741e+00 3.1892009008024806e+00 3.2799584790163632e+00 3.2859635538378220e+00 3.4399388771525312e+00 3.5114854123964836e+00 3.5697880969115974e+00 3.6175829706170464e+00 3.6363349668162739e+00 3.6947670981909346e+00 3.9240130786522212e+00 3.9512246317678907e+00 4.1557715298182396e+00 4.1932309574676445e+00 4.4292388789520922e+00 4.6459623879757954e+00 4.7303851486906909e+00 4.9898260881115295e+00 5.4868949562042308e+00 5.9838980541677556e+00 6.2843644400809326e+00 6.2901682529190799e+00 6.3781138797764658e+00 6.4202764164813013e+00 6.4811644096661292e+00 6.5329741910821495e+00 6.6594419595104810e+00 6.8404617999847099e+00 7.1724952073261763e+00 7.6259347169227922e+00 1.1962232604401258e+01 ";
rdft_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",dip,101,130);

Etot=-7.6373028398391725e+01;
dip=7.3223548298072993e-01;
Eorb="-1.8739063760825058e+01 -9.1594743466157047e-01 -4.7168660758274233e-01 -3.2515056402070491e-01 -2.4824169526451695e-01 7.6873411241415767e-03 8.0201426411730767e-02 3.3767223371286204e-01 3.7911651247252559e-01 4.6520447242891372e-01 5.4523058190301465e-01 5.9029243977089063e-01 5.9663421782202009e-01 6.4384853023195199e-01 7.4405572522367280e-01 8.8293653007464212e-01 9.7364965937347936e-01 1.2410684186897598e+00 1.2610084014182299e+00 1.6997995096811467e+00 1.7260543164573097e+00 1.7615704572712694e+00 1.8254646132924679e+00 1.8999197545209527e+00 2.1845241297311540e+00 2.5325347756885832e+00 2.9874617928605605e+00 3.1259289557228440e+00 3.1891621641170507e+00 3.2797686726709814e+00 3.2858649042759702e+00 3.4399088410529974e+00 3.5114590908132373e+00 3.5697301750970749e+00 3.6174581123105520e+00 3.6361186612218654e+00 3.6945898256483463e+00 3.9240165141737116e+00 3.9511489223544021e+00 4.1557157432344312e+00 4.1932261637832422e+00 4.4291612156909492e+00 4.6459395537264587e+00 4.7302925501878184e+00 4.9896637941531301e+00 5.4868666659990639e+00 5.9839041251945195e+00 6.2842950465418888e+00 6.2900639240948761e+00 6.3780044814161547e+00 6.4202857505702680e+00 6.4811636299171340e+00 6.5328571016732209e+00 6.6594825470246768e+00 6.8404437462589405e+00 7.1725303079377296e+00 7.6257023228844529e+00 1.1962113808673925e+01 ";
rdft_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",dip,101,130);

Etot=-7.6373028398391696e+01;
dip=7.3223548298069896e-01;
Eorb="-1.8739063760825111e+01 -9.1594743466157680e-01 -4.7168660758275238e-01 -3.2515056402070225e-01 -2.4824169526451945e-01 7.6873411240566160e-03 8.0201426411664001e-02 3.3767223371282867e-01 3.7911651247251477e-01 4.6520447242886553e-01 5.4523058190295037e-01 5.9029243977081669e-01 5.9663421782197978e-01 6.4384853023191679e-01 7.4405572522365537e-01 8.8293653007485562e-01 9.7364965937342418e-01 1.2410684186896885e+00 1.2610084014181988e+00 1.6997995096811147e+00 1.7260543164572546e+00 1.7615704572711426e+00 1.8254646132924095e+00 1.8999197545210462e+00 2.1845241297311189e+00 2.5325347756884882e+00 2.9874617928605782e+00 3.1259289557228800e+00 3.1891621641170649e+00 3.2797686726709672e+00 3.2858649042758175e+00 3.4399088410529215e+00 3.5114590908132715e+00 3.5697301750970851e+00 3.6174581123105307e+00 3.6361186612218219e+00 3.6945898256482166e+00 3.9240165141737817e+00 3.9511489223544154e+00 4.1557157432344400e+00 4.1932261637831907e+00 4.4291612156909261e+00 4.6459395537263832e+00 4.7302925501877953e+00 4.9896637941530493e+00 5.4868666659989742e+00 5.9839041251944947e+00 6.2842950465420699e+00 6.2900639240950929e+00 6.3780044814162578e+00 6.4202857505702609e+00 6.4811636299170639e+00 6.5328571016733168e+00 6.6594825470246430e+00 6.8404437462590009e+00 7.1725303079376719e+00 7.6257023228845950e+00 1.1962113808673710e+01 ";
rdft_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",dip,101,130);

Etot=-7.6374648621893527e+01;
dip=7.3271967268560279e-01;
Eorb="-1.8741542847544597e+01 -9.1787961906827698e-01 -4.7348532451257996e-01 -3.2745859712427700e-01 -2.5054924514018539e-01 2.7896940056802572e-03 7.8041129926720451e-02 3.2383342989948993e-01 3.5923978172960874e-01 4.5242193720681556e-01 5.1413414939963886e-01 5.7767373621190254e-01 5.8423657180437250e-01 6.4253363323625290e-01 6.5976605060292803e-01 7.4242006874709465e-01 9.7186950169862651e-01 1.1822082345491207e+00 1.2023227703860808e+00 1.5759176865951394e+00 1.6360777580996682e+00 1.6982205464151749e+00 1.7245170595450861e+00 1.8628731055911469e+00 1.9081340495078896e+00 2.1641867801286589e+00 2.3473129536954840e+00 2.7892739263199071e+00 3.0049314501080300e+00 3.0831687857212877e+00 3.1876658158769393e+00 3.2328093477759046e+00 3.4168943866503345e+00 3.4579170211996533e+00 3.5049980584137987e+00 3.5282495884100715e+00 3.5683150461279856e+00 3.5772259477532660e+00 3.8379017961531408e+00 3.9226527147071510e+00 4.0867004669491340e+00 4.0926885678294020e+00 4.3310830826520315e+00 4.4154606605952251e+00 4.4322653300584935e+00 4.6027456223328196e+00 5.1266098215654923e+00 5.2200924223705121e+00 5.4840615205310428e+00 6.1494589150345318e+00 6.2799836140358121e+00 6.2885828169544462e+00 6.3502582273154049e+00 6.4059399691202481e+00 6.4358009686079534e+00 6.6570180573971278e+00 6.7153053387707820e+00 6.7372429461243568e+00 6.9398798725532149e+00 7.3406793997692548e+00 8.2789339188697184e+00 8.3551996761671106e+00 9.3390720947480563e+00 1.4480083115494288e+01 1.5822311920507591e+01 ";
rdft_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",dip,101,130);

Etot=-5.6637319431552560e+03;
dip=4.1660087891427908e+00;
Eorb="-9.4985623400049417e+02 -1.4146635095155003e+02 -1.3119294651375273e+02 -1.3119287266570007e+02 -1.3119259492741716e+02 -2.7676547682168476e+01 -2.3232980782912730e+01 -2.3232722797346284e+01 -2.3231117082849920e+01 -1.6049049275837675e+01 -1.6049045517137976e+01 -1.6047827619001310e+01 -1.6047723988359522e+01 -1.6047713419879468e+01 -1.5604531125108002e+01 -1.5532249524609277e+01 -1.1296661985298261e+01 -1.1249243916538013e+01 -1.1232665101731335e+01 -4.3970789264055998e+00 -2.8992751978270852e+00 -2.8986598112526063e+00 -2.8951186644345275e+00 -1.4177741084736841e+00 -1.2312596935594773e+00 -1.0610694399839127e+00 -8.7645300677375038e-01 -8.5303384276651240e-01 -8.1305177477854351e-01 -7.2468343347283282e-01 -7.1752257368810857e-01 -7.1751283841380864e-01 -7.1453923747155434e-01 -7.1369020418219975e-01 -6.5649509281848528e-01 -6.5484509911155342e-01 -6.4819563018025550e-01 -6.1951447647646629e-01 -5.1149278123004638e-01 -4.5694083366489086e-01 -3.6925757412432292e-01 -1.8059222598167177e-01 6.9314383909734961e-02 7.4011367232244774e-02 1.1409015187798319e-01 1.4993230378890474e-01 1.8266979171349690e-01 1.9355783565499271e-01 2.1197839620810291e-01 2.5237136177704711e-01 2.7656210758970678e-01 2.8532362353373403e-01 3.0336607382877451e-01 3.3343210564863524e-01 3.3688909001286121e-01 3.9652955212860025e-01 4.2174259419945365e-01 5.4893794753148939e-01 5.6113635053643651e-01 6.8232567993706694e-01 8.8548529727750169e-01 9.2615819814387723e-01 9.2670939423432452e-01 9.6328467851688881e-01 9.8346700887442373e-01 9.9887403107330464e-01 1.0364505367172789e+00 1.0834412222590375e+00 1.0936564326233837e+00 1.1989337347766149e+00 1.2617669937885077e+00 1.2818433228653869e+00 1.3193949625540493e+00 1.3895935208613626e+00 1.4308893014509934e+00 1.4702798346869674e+00 1.4945329053711267e+00 1.5683750136090946e+00 1.5822512398578519e+00 1.6271532071253343e+00 1.6323133210824046e+00 1.6700777070393382e+00 1.7294530550325080e+00 1.8374560547866039e+00 1.9460155940573529e+00 1.9779608107756965e+00 2.0568938244954711e+00 2.2440133859232150e+00 2.9829355441942838e+00 3.0788481811188206e+00 5.2757403753205185e+00 2.1121787322900565e+02 ";
rhf_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G",dip);

Etot=-5.6676693771690543e+03;
dip=3.3439668910029381e+00;
Eorb="-9.3989978310422589e+02 -1.3767101240558807e+02 -1.2795640135473509e+02 -1.2795629725703834e+02 -1.2795609267949722e+02 -2.5984496633230478e+01 -2.1817716331692765e+01 -2.1817474469981111e+01 -2.1816391377327985e+01 -1.5092625921296944e+01 -1.5092613658980492e+01 -1.5091968035901234e+01 -1.5091822542157910e+01 -1.5091777798587739e+01 -1.4452050933898059e+01 -1.4376826258937689e+01 -1.0301664407258851e+01 -1.0271613222707554e+01 -1.0254342769278230e+01 -3.7636291439810079e+00 -2.4165879922586218e+00 -2.4159471478544394e+00 -2.4131569841300462e+00 -1.1211260980569357e+00 -9.6404302921374463e-01 -8.2563594553555386e-01 -7.0330466799492386e-01 -6.8267077885409466e-01 -6.4806972675219643e-01 -5.4367439326040490e-01 -5.3721118995061490e-01 -5.2553329911152580e-01 -5.2468731232714583e-01 -5.2443006939043313e-01 -5.1974384823890007e-01 -5.1950908127495266e-01 -5.1476943352400240e-01 -4.9177448294460369e-01 -3.8454479540706382e-01 -3.8028082221215775e-01 -3.2171062768392833e-01 -1.8694110925469179e-01 -5.9086341958679289e-02 -5.6817116765926254e-02 -4.6514237405702326e-02 -4.3790553348049890e-02 -1.6269944627621647e-02 -1.5551555433461688e-02 3.4348180885209634e-02 5.9149658015173620e-02 6.8898580836830078e-02 9.1044720388410369e-02 1.0204094307352377e-01 1.2425776925316440e-01 1.3808997802104853e-01 1.5836841344132721e-01 1.8698885008431404e-01 3.0351682833518773e-01 3.0990118068795830e-01 3.9883824466261542e-01 5.8813342764864429e-01 5.9391038240466221e-01 6.1271411219707417e-01 6.5246465008463206e-01 6.6253193226235874e-01 6.7302838024085065e-01 6.9271307310766983e-01 7.7761508280091762e-01 7.9855541070956659e-01 8.6965249780623011e-01 8.9908840594167549e-01 9.3737296267616799e-01 9.6540579004121352e-01 9.9998469459009942e-01 1.0476202523368865e+00 1.1414384991919895e+00 1.1555410303749585e+00 1.2222188088854584e+00 1.2400230173448969e+00 1.2657698974145131e+00 1.2729543517543687e+00 1.3149215018292866e+00 1.3519148273148294e+00 1.4869557983498567e+00 1.6023139385263123e+00 1.6213994254891471e+00 1.7041935573883860e+00 1.8438995276071974e+00 2.6031625593463859e+00 2.7024556838744882e+00 4.7145848672821673e+00 2.0804451024935128e+02 ";
rdft_test(cdcplx,b3_21G,dftcart_nofit,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",dip,402,0);

Etot=-4.6701321134007890e+02;
dip=5.6558459094798919e-01;
Eorb="-1.8627155389956641e+01 -9.8468884036289328e+00 -9.8000403183325915e+00 -9.7985030239421000e+00 -9.7964382107193533e+00 -9.7961807306956956e+00 -9.7961550319734165e+00 -9.7957612801034006e+00 -9.7955844204322027e+00 -9.7955253352607361e+00 -9.7902010912754385e+00 -9.4290462441771250e-01 -7.5219755438729585e-01 -7.3523264406259770e-01 -7.0339379685311509e-01 -6.7054043807218844e-01 -6.3196338440824906e-01 -5.8307141582132771e-01 -5.5339064274312699e-01 -5.4222274091723888e-01 -5.1586083035246411e-01 -5.0988718149352885e-01 -4.8254315882395044e-01 -4.3597789464549674e-01 -4.3184704381827332e-01 -4.1614324149951470e-01 -4.0900731464843165e-01 -4.0310301041724783e-01 -3.8729721919360577e-01 -3.7610092569918391e-01 -3.7081200767009531e-01 -3.4983203920735023e-01 -3.4595253032774387e-01 -3.3597079201316105e-01 -3.2455694688320652e-01 -3.2107059207405636e-01 -3.1361176774227767e-01 -2.9876327774399969e-01 -2.9369154547910942e-01 -2.9084626563448235e-01 -2.8883510489184305e-01 -2.8097010189430266e-01 -2.7633843786564949e-01 -2.6485318863237001e-01 -2.2609495944832309e-01 2.9806191456843583e-02 4.3502577895044706e-02 4.7425902186554750e-02 5.3710247758175275e-02 6.2798357351571590e-02 6.8714799799356616e-02 7.6075437473691415e-02 8.0543340409434311e-02 9.0741951284708050e-02 1.0571675307933805e-01 1.1202229715299251e-01 1.1622893552140466e-01 1.2071515528285720e-01 1.3140988221374467e-01 1.3642600335065927e-01 1.3882306098024610e-01 1.4089032603965915e-01 1.4347204306483785e-01 1.4890078757327760e-01 1.5392861589658322e-01 1.6458475875856796e-01 1.6999257970608128e-01 1.7439288052838625e-01 1.8134945580987860e-01 1.9191609853944125e-01 1.9907633932141436e-01 2.0636216826160989e-01 2.1896327644889174e-01 2.2823888360232603e-01 2.4034506366055219e-01 2.4854950627880562e-01 2.5464806636529158e-01 4.2493271044151887e-01 4.2850886073862765e-01 4.4064410534204784e-01 4.5736194528973911e-01 4.6331708602314836e-01 4.6892492596778929e-01 4.7902752042459440e-01 4.8942009397372216e-01 5.0243422073199073e-01 5.1194298157998142e-01 5.1904520698252732e-01 5.3632788782776186e-01 5.4651472592214301e-01 5.7350453364919896e-01 5.9413007574084065e-01 6.0273922324307028e-01 6.0614929459325118e-01 6.1191796541258225e-01 6.1545221769977243e-01 6.3140381040641558e-01 6.4954779272287810e-01 6.7503495400628777e-01 6.8567421261794637e-01 6.9676129688112709e-01 7.1482468328092674e-01 7.2341293409227281e-01 7.4573470724059432e-01 7.5077683294795761e-01 7.6086260324018573e-01 7.7072141404698669e-01 7.7609700261663261e-01 7.8186561515689623e-01 7.9605246629383031e-01 8.0732894780411457e-01 8.1699563566935385e-01 8.2988574520481284e-01 8.3747020125982097e-01 8.3872073202726849e-01 8.4332465977476812e-01 8.4899137143872250e-01 8.5771909611058572e-01 8.6351982183487208e-01 8.6889346748477103e-01 8.7768101606081617e-01 8.8613576089534785e-01 8.9531622458650617e-01 9.0580268370584971e-01 9.1529394540575792e-01 9.2211729513997231e-01 9.4122869076800897e-01 9.6566335830235261e-01 9.7153227984595358e-01 9.8203184740601335e-01 1.0177357272962468e+00 1.0490679045339768e+00 1.0974974182921275e+00 1.1473859290652444e+00 1.1642856507066606e+00 1.2116505169577916e+00 1.2321588896721969e+00 1.2665602300829735e+00 1.2725658258616079e+00 1.3173851672246615e+00 1.3344567204637887e+00 1.3696815749860647e+00 1.4032274262079663e+00 1.4066934558639566e+00 1.4522678806920895e+00 1.4859434685669852e+00 1.4994164796124645e+00 1.5182768892919072e+00 1.5407584241816283e+00 1.5551907008891375e+00 1.5718427975992550e+00 1.5854147634529392e+00 1.6035526835099196e+00 1.6248642993530151e+00 1.6295953898522324e+00 1.6386264156108672e+00 1.6518537217365696e+00 1.6708603659755863e+00 1.7082441379610895e+00 1.7240858682661551e+00 1.7309955634625909e+00 1.7768180305784205e+00 1.7799398226936811e+00 1.7966929365250275e+00 1.7986813203641532e+00 1.8208975438917168e+00 1.8371991719322283e+00 1.8486132033959948e+00 1.8627496348970551e+00 1.8684090728741751e+00 1.8910556463264823e+00 1.9068268277653682e+00 1.9273963883594998e+00 1.9366440438004862e+00 1.9517986860745971e+00 1.9711843615650670e+00 1.9748131897645251e+00 1.9784538526556863e+00 2.0029272575482855e+00 2.0163942521807128e+00 2.0242113317484032e+00 2.0282111337115007e+00 2.0446483730346796e+00 2.0506332741200421e+00 2.0622352426892405e+00 2.0764523140000706e+00 2.0982714696873601e+00 2.1124504075234838e+00 2.1473840449053241e+00 2.1546265279888122e+00 2.1669072493778110e+00 2.1723423249113507e+00 2.1811756005192593e+00 2.1987631393033960e+00 2.2110770698356723e+00 2.2189960023915742e+00 2.2523875261024284e+00 2.2601866817420673e+00 2.2680782163885174e+00 2.2959486228647346e+00 2.3105472765039443e+00 2.3159943413151662e+00 2.3268618438079276e+00 2.3486729947066025e+00 2.3828964190698834e+00 2.3876850293583241e+00 2.4069231530879409e+00 2.4220203027433453e+00 2.4322426930311609e+00 2.4627677967067503e+00 2.4929212135225667e+00 2.5133876324328321e+00 2.5312878448982086e+00 2.5380551592109062e+00 2.5674710847598372e+00 2.5816439126604860e+00 2.5894765242031692e+00 2.6092017889490000e+00 2.6302178344561984e+00 2.6355319058422935e+00 2.6434882762097254e+00 2.6604218986513328e+00 2.6727747136990150e+00 2.6917618214807217e+00 2.6952996954001951e+00 2.7073942127192017e+00 2.7113426270878866e+00 2.7285878278452618e+00 2.7487932280876075e+00 2.7749378214700524e+00 2.7823800350127690e+00 2.7848198275791560e+00 2.7958607806693405e+00 2.8014816855309346e+00 2.8080213289140437e+00 2.8118708845357414e+00 2.8150036081648042e+00 2.8202128434306060e+00 2.8419229221642643e+00 2.8601398015926369e+00 2.8723562366688467e+00 2.9059069930677750e+00 3.0865699410455285e+00 3.1217730417504077e+00 3.1464491108056118e+00 3.1676576393132527e+00 3.1811710461883429e+00 3.1906302973437772e+00 3.1946229008887066e+00 3.2130861547170784e+00 3.2280261737681117e+00 3.2455312694842706e+00 3.2648008252385079e+00 3.2857850473139654e+00 3.3101638865456935e+00 3.3528868365854283e+00 3.3823782117903152e+00 3.3896321401912526e+00 3.3912089324907728e+00 3.4239218193009688e+00 3.4639873567768280e+00 3.4735422037541248e+00 3.4830952414619971e+00 3.4844489531120000e+00 3.8198817187203673e+00 4.1566323093442898e+00 4.2134192340979251e+00 4.2755955525229608e+00 4.3733051124823650e+00 4.4240197355181117e+00 4.4487034795528446e+00 4.5000880330249782e+00 4.5662886423430624e+00 4.6275751493314843e+00 4.7059770509688974e+00 ";
rdft_test(decanol,b6_31Gpp,dftcart,Etot,Eorb,"1-decanol, SVWN(RPA)/6-31G**",dip,1,8);


#ifndef COMPUTE_REFERENCE
  printf("****** Tests completed in %s *******\n",t.elapsed().c_str());
#endif

  return 0;
}
