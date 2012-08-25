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
const double dft_initialtol=1e-4;
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
	  size_t N=0;
	  for(int i=0; i<=lr; i++) {
	    int nx = lr - i;
	    for(int j=0; j<=i; j++) {
	      int ny = i-j;
	      int nz = j;
	      
	      cartr[N].l=nx;
	      cartr[N].m=ny;
	      cartr[N].n=nz;
	      cartr[N].relnorm=cr[N];
	      N++;
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
    fflush(stdout);
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
    fflush(stdout);
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
    fflush(stdout);
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
    fflush(stdout);
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
  // Use core guess and no density fitting for tests.
  sph.set_string("Guess","Core");
  sph.set_bool("DensityFitting",false);

  // No spherical harmonics
  Settings cart=sph;
  cart.set_bool("UseLM",false);

  // Direct calculation
  Settings direct=sph;
  direct.set_bool("Direct",true);

  // Polarized calculation
  Settings pol=sph;
  pol.set_int("Multiplicity",2);

  // DFT tests

  // Settings for DFT
  Settings dftsph=sph; // Normal settings
  dftsph.add_dft_settings();
  dftsph.set_bool("DensityFitting",true);

  Settings dftcart=cart; // Cartesian basis
  dftcart.add_dft_settings();
  dftcart.set_bool("DensityFitting",true);

  Settings dftnofit=dftsph; // No density fitting
  dftnofit.set_bool("DensityFitting",false);

  Settings dftcart_nofit=dftcart;
  dftcart_nofit.set_bool("DensityFitting",false);

  Settings dftdirect=dftsph; // Direct calculation
  dftdirect.set_bool("Direct",true);
  dftdirect.set_bool("DFTDirect",true);

  Settings dftpol=pol; // Polarized calculation
  dftpol.add_dft_settings();
  dftpol.set_double("DFTInitialTol",1e-4);

  Settings dftpol_nofit=dftpol; // Polarized calculation, no density fitting
  dftpol_nofit.set_bool("DensityFitting",false);

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

  Etot=-1.2848877555174082e+02;
  dip=4.3460353067759622e-16;
  Eorb="-3.2765635418561338e+01 -1.9187982340179033e+00 -8.3209725199350892e-01 -8.3209725199350559e-01 -8.3209725199349815e-01 1.6945577282675361e+00 1.6945577282675413e+00 1.6945577282675452e+00 2.1594249508218963e+00 5.1967114014294067e+00 5.1967114014294076e+00 5.1967114014294085e+00 5.1967114014294129e+00 5.1967114014294173e+00 ";
  rhf_test(Ne,cc_pVDZ,sph,Etot,Eorb,"Neon, HF/cc-pVDZ",dip);

  Etot=-1.2848886617203755e+02;
  dip=3.6746372416365226e-16;
  Eorb="-3.2765400811469348e+01 -1.9190111585186405e+00 -8.3228220464890301e-01 -8.3228220464890135e-01 -8.3228220464890001e-01 1.6944246989475953e+00 1.6944246989475973e+00 1.6944246989476028e+00 1.9905987652939690e+00 5.1964245950191925e+00 5.1964245950192005e+00 5.1964245950192023e+00 5.1964245950192058e+00 5.1964245950192129e+00 1.0383358428328878e+01 ";
  rhf_test(Ne,cc_pVDZ,cart,Etot,Eorb,"Neon, HF/cc-pVDZ cart",dip);

  Etot=-1.2853186163632139e+02;
  dip=8.3502519156009142e-16;
  Eorb="-3.2769110713076643e+01 -1.9270833030665337e+00 -8.4541550982383029e-01 -8.4541550982382441e-01 -8.4541550982382108e-01 1.0988680373861486e+00 1.0988680373861495e+00 1.0988680373861568e+00 1.4176388084675071e+00 2.8142175665198019e+00 2.8142175665198086e+00 2.8142175665198099e+00 2.8142175665198144e+00 2.8142175665198192e+00 6.1558667275843053e+00 6.1558667275843169e+00 6.1558667275843231e+00 9.6473695834491160e+00 9.6473695834491249e+00 9.6473695834491284e+00 9.6473695834491373e+00 9.6473695834491462e+00 9.6473695834491497e+00 9.6473695834491746e+00 1.1227312686815079e+01 1.1227312686815083e+01 1.1227312686815091e+01 1.1227312686815099e+01 1.1227312686815116e+01 1.1744558071686216e+01 ";
  rhf_test(Ne,cc_pVTZ,sph,Etot,Eorb,"Neon, HF/cc-pVTZ",dip);

  Etot=-1.2853200998517838e+02;
  dip=6.6380079248466308e-16;
  Eorb="-3.2769827645370881e+01 -1.9274545165098154e+00 -8.4572301714341613e-01 -8.4572301714341047e-01 -8.4572301714340359e-01 8.8038911834282818e-01 1.0282198401387626e+00 1.0282198401387714e+00 1.0282198401387843e+00 2.8138968464505494e+00 2.8138968464505556e+00 2.8138968464505623e+00 2.8138968464505649e+00 2.8138968464505756e+00 4.1362240323294408e+00 4.6398467065847120e+00 4.6398467065847404e+00 4.6398467065848195e+00 9.6470056629945820e+00 9.6470056629946264e+00 9.6470056629946406e+00 9.6470056629946441e+00 9.6470056629946583e+00 9.6470056629946850e+00 9.6470056629946921e+00 1.1226914497219928e+01 1.1226914497219932e+01 1.1226914497219971e+01 1.1226914497219974e+01 1.1226914497219980e+01 1.1317534800892307e+01 1.1317534800892322e+01 1.1317534800892561e+01 1.6394442678210712e+01 2.8816114658169489e+01 ";
  rhf_test(Ne,cc_pVTZ,cart,Etot,Eorb,"Neon, HF/cc-pVTZ cart",dip);

  Etot=-1.2854346965912143e+02;
  dip=3.2095576000267208e-15;
  Eorb="-3.2771496233517290e+01 -1.9293376374525115e+00 -8.4895896099165180e-01 -8.4895896099161694e-01 -8.4895896099160384e-01 8.0890413889981250e-01 8.0890413889981760e-01 8.0890413889983614e-01 9.3559988915417558e-01 1.9978112798088274e+00 1.9978112798088357e+00 1.9978112798088368e+00 1.9978112798088374e+00 1.9978112798088499e+00 3.9328189059163301e+00 3.9328189059163385e+00 3.9328189059163883e+00 5.8106845428882137e+00 5.9042211384000538e+00 5.9042211384000645e+00 5.9042211384000653e+00 5.9042211384000725e+00 5.9042211384000760e+00 5.9042211384000778e+00 5.9042211384000902e+00 6.7616951546319441e+00 6.7616951546319601e+00 6.7616951546319699e+00 6.7616951546319752e+00 6.7616951546319877e+00 1.4903626162246129e+01 1.4903626162246139e+01 1.4903626162246150e+01 1.4903626162246159e+01 1.4903626162246189e+01 1.4903626162246193e+01 1.4903626162246201e+01 1.4903626162246209e+01 1.4903626162246239e+01 1.5804420585275807e+01 1.5804420585275855e+01 1.5804420585276043e+01 1.9794585643251352e+01 1.9794585643251366e+01 1.9794585643251420e+01 1.9794585643251434e+01 1.9794585643251434e+01 1.9794585643251452e+01 1.9794585643251470e+01 2.0954549905858485e+01 2.0954549905858521e+01 2.0954549905858595e+01 2.0954549905858624e+01 2.0954549905858777e+01 6.6550956475278127e+01 ";
  rhf_test(Ne,cc_pVQZ,sph,Etot,Eorb,"Neon, HF/cc-pVQZ",dip);

  Etot=-1.2854353449722535e+02;
  dip=4.9111891471703474e-15;
  Eorb="-3.2771625129442207e+01 -1.9294942841597944e+00 -8.4906688462322333e-01 -8.4906688462317337e-01 -8.4906688462316682e-01 5.8690441469416910e-01 7.1271797748979482e-01 7.1271797748989851e-01 7.1271797748994248e-01 1.9879845921823593e+00 1.9879845921823780e+00 1.9879845921823940e+00 1.9879845921823951e+00 1.9879845921824026e+00 2.5105148502433825e+00 2.7214792301496118e+00 2.7214792301497104e+00 2.7214792301504458e+00 5.9040962888723980e+00 5.9040962888724051e+00 5.9040962888724069e+00 5.9040962888724140e+00 5.9040962888724264e+00 5.9040962888724309e+00 5.9040962888724549e+00 6.4115733390523166e+00 6.5684069303044801e+00 6.5684069303046506e+00 6.5684069303046630e+00 6.5684069303047075e+00 6.5684069303047696e+00 6.7659166001584987e+00 6.7659166001586906e+00 6.7659166001598354e+00 1.4004805313565495e+01 1.4903514354744658e+01 1.4903514354744772e+01 1.4903514354744789e+01 1.4903514354744805e+01 1.4903514354744823e+01 1.4903514354744855e+01 1.4903514354744882e+01 1.4903514354744903e+01 1.4903514354744928e+01 1.8145155385275633e+01 1.8145155385276635e+01 1.8145155385279438e+01 1.8145155385279814e+01 1.8145155385280798e+01 1.8540067452508058e+01 1.8540067452508215e+01 1.8540067452508730e+01 1.9794449045302592e+01 1.9794449045302603e+01 1.9794449045302631e+01 1.9794449045302674e+01 1.9794449045302706e+01 1.9794449045302727e+01 1.9794449045302855e+01 2.9727979556957887e+01 3.9089870736201604e+01 3.9089870736219702e+01 3.9089870736252436e+01 3.9089870736264608e+01 3.9089870736270100e+01 3.9551871550109936e+01 3.9551871550112317e+01 3.9551871550113653e+01 5.8376821811945391e+01 2.0568373998233861e+02 ";
  rhf_test(Ne,cc_pVQZ,cart,Etot,Eorb,"Neon, HF/cc-pVQZ cart",dip);

  Etot=-4.5929863365640989e+02;
  dip=1.2244011504065293e-14;
  Eorba="-1.0490206724478891e+02 -1.0633455727500298e+01 -8.1040440220502550e+00 -8.1040440220502514e+00 -8.1040440220502425e+00 -1.1496923108323660e+00 -5.4051812023291523e-01 -5.4051812023290502e-01 -5.4051812023290224e-01 4.9527159118920455e-01 5.8723233571039113e-01 5.8723233571040545e-01 5.8723233571041689e-01 1.0645439012887277e+00 1.0645439012887321e+00 1.0645439012887401e+00 1.0645439012887412e+00 1.0645439012887694e+00 ";
  Eorbb="-1.0488804088980702e+02 -1.0619379134957526e+01 -8.0797501762810668e+00 -8.0797501762810651e+00 -8.0797501762810544e+00 -1.0135907512454072e+00 -3.2998928467463101e-01 -3.2998928467462951e-01 -3.2998928467462812e-01 5.1691166703002978e-01 6.2702739937235863e-01 6.2702739937235941e-01 6.2702739937236307e-01 1.1416804426056471e+00 1.1416804426056508e+00 1.1416804426056530e+00 1.1416804426056557e+00 1.1416804426056602e+00 ";
  uhf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, HF/6-31G** polarized",dip);

  Etot=-4.5929468403476301e+02;
  dip=5.1034125596712059e-16;
  Eorba="-1.0490985539356139e+02 -1.0639223023705611e+01 -8.1086672205943149e+00 -8.1086672205943149e+00 -8.1086672205942900e+00 -1.1427051922058722e+00 -5.3421034965985981e-01 -5.3421034965985792e-01 -5.3421034965985670e-01 4.9202839886734012e-01 5.8916626654184756e-01 5.8916626654185067e-01 5.8916626654185478e-01 1.0673063872335229e+00 1.0673063872335360e+00 1.0673063872335369e+00 1.0673063872335384e+00 1.0673063872335387e+00 ";
  Eorbb="-1.0490015299355959e+02 -1.0629710473260911e+01 -8.0913085795158217e+00 -8.0913085795158128e+00 -8.0913085795158057e+00 -1.0275446617675257e+00 -3.3913561848669821e-01 -3.3913561848669760e-01 -3.3913561848669638e-01 5.1611864609989144e-01 6.1978236033055112e-01 6.1978236033055256e-01 6.1978236033055489e-01 1.1316262091582592e+00 1.1316262091582701e+00 1.1316262091582729e+00 1.1316262091582754e+00 1.1316262091582880e+00 ";
  rohf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, ROHF/6-31G**",dip);

  Etot=-4.6010297760915580e+02;
  dip=8.9928885936030105e-15;
  Eorba="-1.0158419200170279e+02 -9.5119813449869532e+00 -7.2706311059273210e+00 -7.2706311059273165e+00 -7.2706311059273103e+00 -8.4711265424867288e-01 -3.7353924722515414e-01 -3.7353924722515330e-01 -3.7353924722515236e-01 3.2665107498161333e-01 4.1369807671988751e-01 4.1369807671989112e-01 4.1369807671989234e-01 8.0595462145567209e-01 8.0595462145567476e-01 8.0595462145567720e-01 8.0595462145567864e-01 8.0595462145568908e-01 ";
  Eorbb="-1.0157705223195735e+02 -9.5053419675798896e+00 -7.2605067044516813e+00 -7.2605067044516733e+00 -7.2605067044516698e+00 -7.9606595272714353e-01 -3.0983181416419081e-01 -3.0983181416418981e-01 -3.0983181416418687e-01 3.4216414980023580e-01 4.3147951937548029e-01 4.3147951937548762e-01 4.3147951937549045e-01 8.4918624484331440e-01 8.4918624484331673e-01 8.4918624484332250e-01 8.4918624484333338e-01 8.4918624484333505e-01 ";
  udft_test(Cl,b6_31Gpp,dftpol_nofit,Etot,Eorba,Eorbb,"Chlorine, B3LYP/6-31G** polarized",dip,402,0);

  Etot=-1.1287000934441980e+00;
  dip=2.1343411060601321e-15;
  Eorb="-5.9241098911311940e-01 1.9744005746521254e-01 4.7932104724330460e-01 9.3732369227359724e-01 1.2929037097066669e+00 1.2929037097066682e+00 1.9570226089315619e+00 2.0435200542705290e+00 2.0435200542705307e+00 3.6104742345380396e+00 ";
  rhf_test(H2,cc_pVDZ,sph,Etot,Eorb,"Hydrogen molecule, HF/cc-pVDZ",dip);

  Etot=-1.1676141182306354e+00;
  dip=1.1656887264340757e-11;
  Eorb="-3.9201515929287012e-01 3.6518736507866585e-02 2.9071356474513632e-01 6.5832910704230607e-01 9.7502281430276683e-01 9.7502281430276727e-01 1.6066119799166587e+00 1.7001805817762130e+00 1.7001805817762154e+00 3.1926513611865524e+00 ";
  rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, SVWN(RPA)/cc-pVDZ",dip,1,8);

  Etot=-1.1603962551524631e+00;
  dip=1.3500735269314753e-11;
  Eorb="-3.7849160709600571e-01 5.3520939144858150e-02 3.0277187275095591e-01 6.6374649499006155e-01 9.9246457875921301e-01 9.9246457875921390e-01 1.6235425150397582e+00 1.7198878039951977e+00 1.7198878039952001e+00 3.2019324159988081e+00 ";
  rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, PBEPBE/cc-pVDZ",dip,101,130);

  Etot=-7.6056825377225564e+01;
  dip=7.9472744343929869e-01;
  Eorb="-2.0555281278776970e+01 -1.3428635537281559e+00 -7.0828436772836423e-01 -5.7575384365951965e-01 -5.0391497913463179e-01 1.4187951599425103e-01 2.0351537963994643e-01 5.4324870449138551e-01 5.9753586796449110e-01 6.6949546818534122e-01 7.8747678943406219e-01 8.0274150242571207e-01 8.0481260841383739e-01 8.5898803403194812e-01 9.5702121794526473e-01 1.1344778956407338e+00 1.1928203504066246e+00 1.5241753072412736e+00 1.5579529876615632e+00 2.0324408702192183e+00 2.0594682931278281e+00 2.0654407666615451e+00 2.1686553969423561e+00 2.2363161872847197e+00 2.5909431281582012e+00 2.9581971198961607e+00 3.3610002630919418e+00 3.4914002753883344e+00 3.5741938470236509e+00 3.6463660407477341e+00 3.7977214227148788e+00 3.8739670203702525e+00 3.8824466778069948e+00 3.9569498248557631e+00 4.0199059055182360e+00 4.0760332616599868e+00 4.1862021921761468e+00 4.3092789383370826e+00 4.3875716395997424e+00 4.5640073761554314e+00 4.6817931187719548e+00 4.8550947813824870e+00 5.1380848619208335e+00 5.2500191192691386e+00 5.5275547774048421e+00 6.0402478806294075e+00 6.5453259404069435e+00 6.9113516638993611e+00 6.9366142677525842e+00 7.0003720404165710e+00 7.0078239262479061e+00 7.0609382582980622e+00 7.1598075638127874e+00 7.2256524677488203e+00 7.4561719771478145e+00 7.7799625502786478e+00 8.2653639985152694e+00 1.2804358858241361e+01 ";
  rhf_test(h2o,cc_pVTZ,sph,Etot,Eorb,"Water, HF/cc-pVTZ",dip);

  Etot=-7.6056825376389511e+01;
  dip=7.9472744046754185e-01;
  Eorb="-2.0555281275360336e+01 -1.3428635535334648e+00 -7.0828436837923359e-01 -5.7575384375947969e-01 -5.0391497891294001e-01 1.4187951603664034e-01 2.0351537963632929e-01 5.4324870442454332e-01 5.9753586766334532e-01 6.6949546836456608e-01 7.8747678978968727e-01 8.0274150230500829e-01 8.0481260791849729e-01 8.5898803375094956e-01 9.5702121773383775e-01 1.1344778988379318e+00 1.1928203502522781e+00 1.5241753065163055e+00 1.5579529881509557e+00 2.0324408701864676e+00 2.0594682903994426e+00 2.0654407665446972e+00 2.1686553958713022e+00 2.2363161892813523e+00 2.5909431311271849e+00 2.9581971167697469e+00 3.3610002633540312e+00 3.4914002773054067e+00 3.5741938461112692e+00 3.6463660362999084e+00 3.7977214224559028e+00 3.8739670174671939e+00 3.8824466830528692e+00 3.9569498215092440e+00 4.0199059080863551e+00 4.0760332577487652e+00 4.1862022002507100e+00 4.3092789348970291e+00 4.3875716467711392e+00 4.5640073622167296e+00 4.6817931187886703e+00 4.8550947747008646e+00 5.1380848515215884e+00 5.2500191201087727e+00 5.5275547728274663e+00 6.0402478812502682e+00 6.5453259388178946e+00 6.9113516651018045e+00 6.9366142682447771e+00 7.0003720409657948e+00 7.0078239278055570e+00 7.0609382550465707e+00 7.1598075638151570e+00 7.2256524659473742e+00 7.4561719785674576e+00 7.7799625531987076e+00 8.2653640047782950e+00 1.2804358803848945e+01 ";
  rhf_test(h2o,cc_pVTZ,direct,Etot,Eorb,"Water, HF/cc-pVTZ direct",dip);

  Etot=-7.6064480528902479e+01;
  dip=7.8765851962769917e-01;
  Eorb="-2.0560341499270589e+01 -1.3467109720217496e+00 -7.1286865313183456e-01 -5.7999183516266961e-01 -5.0759009626244245e-01 1.1677961767350450e-01 1.7061237380347088e-01 4.4878631182631712e-01 4.6240974154111197e-01 4.9860081971591441e-01 5.8389461634349749e-01 6.0602249093736016e-01 6.1386901266918004e-01 6.5509670363871331e-01 7.1940581530280601e-01 8.5193265933746987e-01 9.1760642858290364e-01 1.1091391723647659e+00 1.1559117535700829e+00 1.3479064851673335e+00 1.4144381966570361e+00 1.4776186118957291e+00 1.4856774098325205e+00 1.5814608164171593e+00 1.6854835728292772e+00 1.9096187748790210e+00 2.0727777036945647e+00 2.1976502980203221e+00 2.2888869981559608e+00 2.3588905651392751e+00 2.4246094542541567e+00 2.4837778803533217e+00 2.5224544363405776e+00 2.5800657965810885e+00 2.5803867670953373e+00 2.6507304510360146e+00 2.6683130249784979e+00 2.8407379961544641e+00 2.8643130332654625e+00 3.0412098309390854e+00 3.1190680440555587e+00 3.2889763530299758e+00 3.3518967322469329e+00 3.4467312467144260e+00 3.6214003290947696e+00 3.8285931976970762e+00 3.9968185332513020e+00 4.1278766618243612e+00 4.1879463040549041e+00 4.2176486594224594e+00 4.4343620890827413e+00 4.4925098804020385e+00 4.6832772420633066e+00 4.7403725807774757e+00 4.8079058229063900e+00 4.9140701224929595e+00 5.3503959766022877e+00 5.4039303384743329e+00 5.9860940849439928e+00 6.1030498812572054e+00 6.2449376423994831e+00 6.3029981826679533e+00 6.7000543811473969e+00 6.7926548570324394e+00 7.0589633252358643e+00 7.2683601257540333e+00 7.3171930276894628e+00 7.3671378836459960e+00 7.4371265018414272e+00 7.5184752974575586e+00 7.5458434106222390e+00 7.5694204766462425e+00 8.0046360222396746e+00 8.0708295767684763e+00 8.0987711899968140e+00 8.1338237929497641e+00 8.1523664217240253e+00 8.2695443442848600e+00 8.3150962727273559e+00 8.3485048879745349e+00 8.4164827935449011e+00 8.6181288316032774e+00 8.8336406109731485e+00 8.9048326538891960e+00 8.9437734458433162e+00 9.2166367032190610e+00 9.3761387936289697e+00 9.3791690880304639e+00 9.9423093869840340e+00 1.0035594104802046e+01 1.0257561212651373e+01 1.0425629823483774e+01 1.0646599818841560e+01 1.0757780268942588e+01 1.0806846320036330e+01 1.1272406010921758e+01 1.1390414020786878e+01 1.1595907193510442e+01 1.1644666359296814e+01 1.1693629519324093e+01 1.1844883875029035e+01 1.2158546422725857e+01 1.2320144574477068e+01 1.2398213790657289e+01 1.2413264233065656e+01 1.2465013637020194e+01 1.3602019537401793e+01 1.3763660839734902e+01 1.4247885732248553e+01 1.4614348062391949e+01 1.4639079462510823e+01 1.4826337836293808e+01 1.6435472095510654e+01 1.6799107184001958e+01 4.4322817449503709e+01 ";
  rhf_test(h2o,cc_pVQZ,sph,Etot,Eorb,"Water, HF/cc-pVQZ",dip);

  Etot=-7.6064480532894763e+01;
  dip=7.8765852675648074e-01;
  Eorb="-2.0560341504671111e+01 -1.3467109757507181e+00 -7.1286865563075530e-01 -5.7999183589483350e-01 -5.0759009668720623e-01 1.1677962171577008e-01 1.7061237603481758e-01 4.4878630925126956e-01 4.6240974591688105e-01 4.9860082036279973e-01 5.8389461476515081e-01 6.0602248987570051e-01 6.1386901174331088e-01 6.5509670163548817e-01 7.1940581433417861e-01 8.5193266391892086e-01 9.1760642581506369e-01 1.1091391809183024e+00 1.1559117484781947e+00 1.3479064722576932e+00 1.4144382156042175e+00 1.4776186083046539e+00 1.4856774023796986e+00 1.5814608129744541e+00 1.6854835723337092e+00 1.9096187679633918e+00 2.0727777003752732e+00 2.1976503026051137e+00 2.2888869950421835e+00 2.3588905523108017e+00 2.4246094541612417e+00 2.4837778810534403e+00 2.5224544209261999e+00 2.5800657875536688e+00 2.5803867662859652e+00 2.6507304364669344e+00 2.6683130297587465e+00 2.8407379668963406e+00 2.8643130428121371e+00 3.0412098372698391e+00 3.1190680039924290e+00 3.2889763027922752e+00 3.3518967746208896e+00 3.4467312988143552e+00 3.6214003077281300e+00 3.8285932272098062e+00 3.9968185307232784e+00 4.1278766223170456e+00 4.1879462855107743e+00 4.2176486431199365e+00 4.4343620556219392e+00 4.4925098612802161e+00 4.6832772450729827e+00 4.7403725625255992e+00 4.8079057041372835e+00 4.9140700687681171e+00 5.3503959099367000e+00 5.4039304323793917e+00 5.9860940870058110e+00 6.1030499069182031e+00 6.2449376323384298e+00 6.3029981728563911e+00 6.7000543614402650e+00 6.7926548347071840e+00 7.0589632915692739e+00 7.2683600652839777e+00 7.3171930266086651e+00 7.3671378848872831e+00 7.4371264613003074e+00 7.5184751967190806e+00 7.5458434381210004e+00 7.5694205382422943e+00 8.0046360309735203e+00 8.0708295818380762e+00 8.0987711934928779e+00 8.1338237877456869e+00 8.1523664040661359e+00 8.2695443532763910e+00 8.3150962653283802e+00 8.3485048803820785e+00 8.4164827142540890e+00 8.6181287900931896e+00 8.8336405773768938e+00 8.9048326988154933e+00 8.9437733343310697e+00 9.2166365377317092e+00 9.3761387754285384e+00 9.3791690967303225e+00 9.9423092280896004e+00 1.0035594174979247e+01 1.0257561179108400e+01 1.0425629799440141e+01 1.0646599821700168e+01 1.0757780239406292e+01 1.0806846285326071e+01 1.1272406086883407e+01 1.1390413921053730e+01 1.1595907184841902e+01 1.1644666304884353e+01 1.1693629564129328e+01 1.1844883849843320e+01 1.2158546312309563e+01 1.2320144541632427e+01 1.2398213734145687e+01 1.2413264189578783e+01 1.2465013572156144e+01 1.3602019453238052e+01 1.3763660417689284e+01 1.4247885691243058e+01 1.4614347842100035e+01 1.4639079538703688e+01 1.4826337735735549e+01 1.6435472085349989e+01 1.6799106948715078e+01 4.4322817045905964e+01 ";
  rhf_test(h2o,cc_pVQZ,direct,Etot,Eorb,"Water, HF/cc-pVQZ direct",dip);

  Etot=-7.6372960091487556e+01;
  dip=7.3249263898782657e-01;
  Eorb="-1.8738948429978532e+01 -9.1586662298410071e-01 -4.7160098682673152e-01 -3.2507312110059816e-01 -2.4816681489651393e-01 8.4539946277438389e-03 8.1125782160355098e-02 3.3806739529329138e-01 3.7949173608938658e-01 4.6548883661112594e-01 5.4539141643148170e-01 5.9072343390210746e-01 5.9687830460036961e-01 6.4398060383516931e-01 7.4418118818247858e-01 8.8306708991538774e-01 9.7381877470435652e-01 1.2412765965980648e+00 1.2611347124682801e+00 1.6999126761271168e+00 1.7261712004009595e+00 1.7619586778167080e+00 1.8256262936927541e+00 1.9002764144047402e+00 2.1846825266685794e+00 2.5326905668905142e+00 2.9875044051036106e+00 3.1260227084714272e+00 3.1892009386445483e+00 3.2799585244377845e+00 3.2859636719728709e+00 3.4399389490815730e+00 3.5114854920858143e+00 3.5697881509618243e+00 3.6175830606363348e+00 3.6363350934114207e+00 3.6947672374317717e+00 3.9240131299608207e+00 3.9512247156884928e+00 4.1557716012586656e+00 4.1932310311822105e+00 4.4292389574081064e+00 4.6459624866239952e+00 4.7303852627990999e+00 4.9898261806822495e+00 5.4868949994525735e+00 5.9838981563620957e+00 6.2843645529589951e+00 6.2901683610854455e+00 6.3781139884101314e+00 6.4202764891352935e+00 6.4811645041167862e+00 6.5329743115822172e+00 6.6594420275154587e+00 6.8404619006548373e+00 7.1724953354064915e+00 7.6259348749423985e+00 1.1962233213355514e+01 ";
  rdft_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",dip,101,130);

  Etot=-7.6373023947425040e+01;
  dip=7.3223555302009014e-01;
  Eorb="-1.8739062641973323e+01 -9.1594732399710610e-01 -4.7168650625328040e-01 -3.2515046754205257e-01 -2.4824160427950492e-01 7.6874633463247495e-03 8.0201529568266511e-02 3.3767234820596947e-01 3.7911662387975020e-01 4.6520460763114130e-01 5.4523071812090040e-01 5.9029256228709415e-01 5.9663430911121229e-01 6.4384860676791689e-01 7.4405581502588214e-01 8.8293661865882422e-01 9.7364977239492123e-01 1.2410685524610152e+00 1.2610085226334991e+00 1.6997995507077175e+00 1.7260543563063422e+00 1.7615705657444682e+00 1.8254646795208649e+00 1.8999198212203057e+00 2.1845243382626780e+00 2.5325349423053249e+00 2.9874618084840625e+00 3.1259289886697132e+00 3.1891622018426791e+00 3.2797687179831438e+00 3.2858650222306589e+00 3.4399089128186455e+00 3.5114591703872060e+00 3.5697302290254358e+00 3.6174582021917470e+00 3.6361187876760708e+00 3.6945899647270330e+00 3.9240165653554344e+00 3.9511490061627290e+00 4.1557158145447257e+00 4.1932262373379645e+00 4.4291612940744534e+00 4.6459396523080150e+00 4.7302926640597143e+00 4.9896638865593923e+00 5.4868667090939711e+00 5.9839042272429515e+00 6.2842951592350520e+00 6.2900640320565300e+00 6.3780045898347613e+00 6.4202858230534527e+00 6.4811637242196500e+00 6.5328572219835088e+00 6.6594826148814796e+00 6.8404438467596167e+00 7.1725304358357498e+00 7.6257024807295721e+00 1.1962114417445628e+01 ";
  rdft_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",dip,101,130);

  Etot=-7.6373023947237542e+01;
  dip=7.3223555296985721e-01;
  Eorb="-1.8739062641940894e+01 -9.1594732398961454e-01 -4.7168650624657926e-01 -3.2515046753241883e-01 -2.4824160427050576e-01 7.6874633489942652e-03 8.0201529571312768e-02 3.3767234820880399e-01 3.7911662388173839e-01 4.6520460763705918e-01 5.4523071812767065e-01 5.9029256229324611e-01 5.9663430911408344e-01 6.4384860676950928e-01 7.4405581502858975e-01 8.8293661866590967e-01 9.7364977239646056e-01 1.2410685524638019e+00 1.2610085226364030e+00 1.6997995507127623e+00 1.7260543563118627e+00 1.7615705657489422e+00 1.8254646795254164e+00 1.8999198212265578e+00 2.1845243382648856e+00 2.5325349423093897e+00 2.9874618084859690e+00 3.1259289886709323e+00 3.1891622018428305e+00 3.2797687179835120e+00 3.2858650222397285e+00 3.4399089128204303e+00 3.5114591703870066e+00 3.5697302290266140e+00 3.6174582021935353e+00 3.6361187876810703e+00 3.6945899647352292e+00 3.9240165653569758e+00 3.9511490061664887e+00 4.1557158145471824e+00 4.1932262373430058e+00 4.4291612940772254e+00 4.6459396523129906e+00 4.7302926640664307e+00 4.9896638865661203e+00 5.4868667091002266e+00 5.9839042272485790e+00 6.2842951592458034e+00 6.2900640320682610e+00 6.3780045898455189e+00 6.4202858230609348e+00 6.4811637242273736e+00 6.5328572219941261e+00 6.6594826148881872e+00 6.8404438467693396e+00 7.1725304358442248e+00 7.6257024807381519e+00 1.1962114417458340e+01 ";
  rdft_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",dip,101,130);

  Etot=-7.6374644167572512e+01;
  dip=7.3271974251554295e-01;
  Eorb="-1.8741541731759810e+01 -9.1787950636738691e-01 -4.7348522116670827e-01 -3.2745849791223164e-01 -2.5054915145591655e-01 2.7898124205604407e-03 7.8041232374228900e-02 3.2383354376787288e-01 3.5923989385176153e-01 4.5242208383861915e-01 5.1413429025246249e-01 5.7767382354498076e-01 5.8423669317291782e-01 6.4253371029187423e-01 6.5976615322687293e-01 7.4242015765422575e-01 9.7186961456465293e-01 1.1822083570359594e+00 1.2023229166544720e+00 1.5759178758447987e+00 1.6360778896597516e+00 1.6982205899567406e+00 1.7245171020954757e+00 1.8628731621330781e+00 1.9081341118202926e+00 2.1641868811391021e+00 2.3473130498617900e+00 2.7892740579566060e+00 3.0049314683630102e+00 3.0831688108093949e+00 3.1876658550395254e+00 3.2328093871926491e+00 3.4168944845610532e+00 3.4579171565755153e+00 3.5049981823137650e+00 3.5282496943570854e+00 3.5683151012784271e+00 3.5772260315055342e+00 3.8379019141525523e+00 3.9226527662625346e+00 4.0867005425148371e+00 4.0926886256468027e+00 4.3310831878584350e+00 4.4154607873264089e+00 4.4322654831829471e+00 4.6027456854642175e+00 5.1266099094690158e+00 5.2200925451478959e+00 5.4840615662050087e+00 6.1494589937174000e+00 6.2799837231732276e+00 6.2885829261169617e+00 6.3502583089139018e+00 6.4059400739329408e+00 6.4358010457409289e+00 6.6570181281617504e+00 6.7153054057773502e+00 6.7372430419576466e+00 6.9398799604533767e+00 7.3406795354123613e+00 8.2789340425838596e+00 8.3551997731992582e+00 9.3390722337102439e+00 1.4480083705886720e+01 1.5822312179923705e+01 ";
  rdft_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",dip,101,130);

  Etot=-5.6637319431552269e+03;
  dip=4.1660088061504972e+00;
  Eorb="-9.4985623400254997e+02 -1.4146635095337089e+02 -1.3119294651562322e+02 -1.3119287266756504e+02 -1.3119259492920537e+02 -2.7676547683110730e+01 -2.3232980783872318e+01 -2.3232722798290002e+01 -2.3231117083556573e+01 -1.6049049276775264e+01 -1.6049045518071299e+01 -1.6047827619839651e+01 -1.6047723989166524e+01 -1.6047713420772880e+01 -1.5604531119814240e+01 -1.5532249519996229e+01 -1.1296661984748688e+01 -1.1249243907588900e+01 -1.1232665096274223e+01 -4.3970789262363299e+00 -2.8992751970566046e+00 -2.8986598103930916e+00 -2.8951186636211221e+00 -1.4177741029625148e+00 -1.2312596877270838e+00 -1.0610694350762055e+00 -8.7645300192619890e-01 -8.5303383785325215e-01 -8.1305177132523232e-01 -7.2468343175269601e-01 -7.1752257242798956e-01 -7.1751283714677005e-01 -7.1453923616897674e-01 -7.1369020299665686e-01 -6.5649508814902890e-01 -6.5484509382498435e-01 -6.4819562624000604e-01 -6.1951447194970333e-01 -5.1149277317008124e-01 -4.5694083009866726e-01 -3.6925757013033722e-01 -1.8059222293960120e-01 6.9314384834826825e-02 7.4011367977397380e-02 1.1409015231636327e-01 1.4993230658575632e-01 1.8266979086609392e-01 1.9355783963427042e-01 2.1197839865704907e-01 2.5237136262091242e-01 2.7656210807034870e-01 2.8532362563875135e-01 3.0336607555728407e-01 3.3343210664029688e-01 3.3688909262207323e-01 3.9652955682820679e-01 4.2174259742956793e-01 5.4893795120060540e-01 5.6113635473889356e-01 6.8232568571181806e-01 8.8548530011995874e-01 9.2615820162378037e-01 9.2670939793457818e-01 9.6328468187080274e-01 9.8346701281301152e-01 9.9887403513510875e-01 1.0364505421301553e+00 1.0834412260808501e+00 1.0936564357019709e+00 1.1989337382265584e+00 1.2617669988538658e+00 1.2818433266079519e+00 1.3193949660362958e+00 1.3895935254617837e+00 1.4308893060532106e+00 1.4702798388478833e+00 1.4945329082993719e+00 1.5683750176059559e+00 1.5822512425610162e+00 1.6271532085814402e+00 1.6323133226289412e+00 1.6700777107312283e+00 1.7294530576881546e+00 1.8374560586959574e+00 1.9460155992022372e+00 1.9779608147927583e+00 2.0568938289856304e+00 2.2440133897945587e+00 2.9829355493840426e+00 3.0788481878696974e+00 5.2757403757962811e+00 2.1121787322744808e+02 ";
  rhf_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G",dip);

  Etot=-5.6676815335032607e+03;
  dip=3.7002317036606138e+00;
  Eorb="-9.3984200681110383e+02 -1.3753207293563835e+02 -1.2775492453787326e+02 -1.2775482517535237e+02 -1.2775460308684549e+02 -2.5877636330403867e+01 -2.1698213476069157e+01 -2.1697968938458910e+01 -2.1696704899798089e+01 -1.4963823118300505e+01 -1.4963812691959960e+01 -1.4963019524679636e+01 -1.4962873318553701e+01 -1.4962836925079902e+01 -1.4360307511109218e+01 -1.4284298056059704e+01 -1.0222099616264661e+01 -1.0189091421505740e+01 -1.0172541512781828e+01 -3.7205449766201131e+00 -2.3708080819084811e+00 -2.3701431324751594e+00 -2.3667600322783473e+00 -1.0858774692634061e+00 -9.3089363740127318e-01 -7.9260907485469556e-01 -6.6325307733299343e-01 -6.4256580137649910e-01 -6.0923607777315714e-01 -4.9709272067536603e-01 -4.8503230449713164e-01 -4.8171579230720679e-01 -4.7251894025038121e-01 -4.6879535217913376e-01 -4.6872828013403978e-01 -4.6436539131386556e-01 -4.6398131112891083e-01 -4.5266960416614471e-01 -3.3969102485982305e-01 -3.2987575018863585e-01 -2.7190968854195924e-01 -1.3325502825837374e-01 -4.5377277860056877e-03 -2.5452570196682722e-03 7.9053281087351629e-03 3.3668099680141332e-02 3.7586253534334563e-02 6.3640326215540569e-02 9.5522623346438865e-02 1.3206140374708533e-01 1.3440146861026245e-01 1.5776397320745142e-01 1.6224463455020047e-01 1.8802863453441063e-01 2.0173060797921427e-01 2.1971894218827176e-01 2.4645592541853792e-01 3.6713719728969840e-01 3.7513181221055220e-01 4.5357912845512000e-01 6.5944751684980774e-01 6.7596288788574099e-01 6.7892129221674324e-01 7.2415347958123577e-01 7.3375434123718342e-01 7.3584026308764583e-01 7.7452085726585829e-01 8.4047864874115386e-01 8.5273435876673531e-01 9.2809759824914695e-01 9.7420818823252608e-01 1.0043108029773538e+00 1.0256257211834729e+00 1.0797444957816169e+00 1.1228262131272593e+00 1.1979813647609487e+00 1.2156268320382597e+00 1.2858833671162966e+00 1.3032976137586048e+00 1.3297849523380059e+00 1.3368291430037851e+00 1.3822192363817614e+00 1.4204708968483097e+00 1.5533763536484373e+00 1.6676765552942929e+00 1.6794890664289810e+00 1.7665915290741188e+00 1.9160564376107643e+00 2.6597655363661397e+00 2.7556227138007365e+00 4.7845696038276699e+00 2.0810566662890562e+02 ";
  rdft_test(cdcplx,b3_21G,dftcart_nofit,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",dip,402,0);

  Etot=-4.6701321137062700e+02;
  dip=5.6558460353209283e-01;
  Eorb="-1.8627155391457261e+01 -9.8468883956112965e+00 -9.8000403160528027e+00 -9.7985030192799236e+00 -9.7964382098128002e+00 -9.7961807271748640e+00 -9.7961550277323042e+00 -9.7957612753748933e+00 -9.7955844233619533e+00 -9.7955253329214909e+00 -9.7902010909561508e+00 -9.4290462267276642e-01 -7.5219755377223751e-01 -7.3523264270799571e-01 -7.0339379628197807e-01 -6.7054043702887633e-01 -6.3196338350669290e-01 -5.8307141502875803e-01 -5.5339064212592681e-01 -5.4222274090365918e-01 -5.1586083053230347e-01 -5.0988718092502650e-01 -4.8254315754359156e-01 -4.3597789502935247e-01 -4.3184704285525777e-01 -4.1614324000167124e-01 -4.0900731415217656e-01 -4.0310300960657697e-01 -3.8729721846206161e-01 -3.7610092536615514e-01 -3.7081200669031733e-01 -3.4983203938405183e-01 -3.4595252981279484e-01 -3.3597079142586528e-01 -3.2455694687846559e-01 -3.2107059196600329e-01 -3.1361176718987510e-01 -2.9876327694288268e-01 -2.9369154471655551e-01 -2.9084626582970297e-01 -2.8883510414644881e-01 -2.8097010147869828e-01 -2.7633843713493111e-01 -2.6485318849747869e-01 -2.2609495855006834e-01 2.9806192290392833e-02 4.3502577780544914e-02 4.7425901926521130e-02 5.3710247716224804e-02 6.2798357087970699e-02 6.8714799751040279e-02 7.6075437604299065e-02 8.0543340638187397e-02 9.0741951398397497e-02 1.0571675341097821e-01 1.1202229719361251e-01 1.1622893576896096e-01 1.2071515524393225e-01 1.3140988288967598e-01 1.3642600353429132e-01 1.3882306098981226e-01 1.4089032619225283e-01 1.4347204313778514e-01 1.4890078770125176e-01 1.5392861551102316e-01 1.6458475911018383e-01 1.6999258000171452e-01 1.7439288072758835e-01 1.8134945654707318e-01 1.9191609914768332e-01 1.9907634033921279e-01 2.0636216880861247e-01 2.1896327684407113e-01 2.2823888375286169e-01 2.4034506371558256e-01 2.4854950671650858e-01 2.5464806654737326e-01 4.2493271094059948e-01 4.2850886099440066e-01 4.4064410594136144e-01 4.5736194587928819e-01 4.6331708670399840e-01 4.6892492669186708e-01 4.7902752167095475e-01 4.8942009419242338e-01 5.0243422108922409e-01 5.1194298161808560e-01 5.1904520746964766e-01 5.3632788859734704e-01 5.4651472618126795e-01 5.7350453440593574e-01 5.9413007609607349e-01 6.0273922361404864e-01 6.0614929508181969e-01 6.1191796712487623e-01 6.1545221879901502e-01 6.3140381129038481e-01 6.4954779318123312e-01 6.7503495422841886e-01 6.8567421319586230e-01 6.9676129735933068e-01 7.1482468370743091e-01 7.2341293384041783e-01 7.4573470780706397e-01 7.5077683309306320e-01 7.6086260402574901e-01 7.7072141477946510e-01 7.7609700261734316e-01 7.8186561550178701e-01 7.9605246671624019e-01 8.0732894774107800e-01 8.1699563598529312e-01 8.2988574556376626e-01 8.3747020149497586e-01 8.3872073228803778e-01 8.4332465984670413e-01 8.4899137153227822e-01 8.5771909574060401e-01 8.6351982184206333e-01 8.6889346749276086e-01 8.7768101589643166e-01 8.8613576101322833e-01 8.9531622471268535e-01 9.0580268361310012e-01 9.1529394562154442e-01 9.2211729533358244e-01 9.4122869085435856e-01 9.6566335852449037e-01 9.7153227992218738e-01 9.8203184774613683e-01 1.0177357276275714e+00 1.0490679048734122e+00 1.0974974184064521e+00 1.1473859294693729e+00 1.1642856510399122e+00 1.2116505175370667e+00 1.2321588904611254e+00 1.2665602310176585e+00 1.2725658261601944e+00 1.3173851683637137e+00 1.3344567208730451e+00 1.3696815752456353e+00 1.4032274262112134e+00 1.4066934560309814e+00 1.4522678810954939e+00 1.4859434689889990e+00 1.4994164804759276e+00 1.5182768893462120e+00 1.5407584246424093e+00 1.5551907016617632e+00 1.5718427978962703e+00 1.5854147636285247e+00 1.6035526836999050e+00 1.6248642993513742e+00 1.6295953902867923e+00 1.6386264160230943e+00 1.6518537213958844e+00 1.6708603664138071e+00 1.7082441386483631e+00 1.7240858691623080e+00 1.7309955647299635e+00 1.7768180308453236e+00 1.7799398232295720e+00 1.7966929376543859e+00 1.7986813205279357e+00 1.8208975445205899e+00 1.8371991726444703e+00 1.8486132039339520e+00 1.8627496354288031e+00 1.8684090738511838e+00 1.8910556476881752e+00 1.9068268283867666e+00 1.9273963897389339e+00 1.9366440447967901e+00 1.9517986870735198e+00 1.9711843610472877e+00 1.9748131899605390e+00 1.9784538518373438e+00 2.0029272578917205e+00 2.0163942525243508e+00 2.0242113320427984e+00 2.0282111342993994e+00 2.0446483729968845e+00 2.0506332749704606e+00 2.0622352424720773e+00 2.0764523142715570e+00 2.0982714696496236e+00 2.1124504076196486e+00 2.1473840442978203e+00 2.1546265286447737e+00 2.1669072492661070e+00 2.1723423244757405e+00 2.1811756006923217e+00 2.1987631391869709e+00 2.2110770700414504e+00 2.2189960020656758e+00 2.2523875264355104e+00 2.2601866815199085e+00 2.2680782162603097e+00 2.2959486227306249e+00 2.3105472764824362e+00 2.3159943418256055e+00 2.3268618442761357e+00 2.3486729953518770e+00 2.3828964186611823e+00 2.3876850296449788e+00 2.4069231529996555e+00 2.4220203031663976e+00 2.4322426933442869e+00 2.4627677968116677e+00 2.4929212136477763e+00 2.5133876327032851e+00 2.5312878454967578e+00 2.5380551594235872e+00 2.5674710849937470e+00 2.5816439124311494e+00 2.5894765248548350e+00 2.6092017888053913e+00 2.6302178349441636e+00 2.6355319067044722e+00 2.6434882763824556e+00 2.6604218992813014e+00 2.6727747139288258e+00 2.6917618215642252e+00 2.6952996954797985e+00 2.7073942131418396e+00 2.7113426273160157e+00 2.7285878283272287e+00 2.7487932280257419e+00 2.7749378211934173e+00 2.7823800344883702e+00 2.7848198282276502e+00 2.7958607811597824e+00 2.8014816856811935e+00 2.8080213294383789e+00 2.8118708849500358e+00 2.8150036084560806e+00 2.8202128434503928e+00 2.8419229221037101e+00 2.8601398015936041e+00 2.8723562375220726e+00 2.9059069938909214e+00 3.0865699410575247e+00 3.1217730414893872e+00 3.1464491109780144e+00 3.1676576393550682e+00 3.1811710464410945e+00 3.1906302978570693e+00 3.1946229015124032e+00 3.2130861550093734e+00 3.2280261748428960e+00 3.2455312692546228e+00 3.2648008255959167e+00 3.2857850472807395e+00 3.3101638864144691e+00 3.3528868367836555e+00 3.3823782119507948e+00 3.3896321409392702e+00 3.3912089335559443e+00 3.4239218196540513e+00 3.4639873569895583e+00 3.4735422040319475e+00 3.4830952411426064e+00 3.4844489535811141e+00 3.8198817193801875e+00 4.1566323101697904e+00 4.2134192360005800e+00 4.2755955533574852e+00 4.3733051136501313e+00 4.4240197375597763e+00 4.4487034811745669e+00 4.5000880340018314e+00 4.5662886434854633e+00 4.6275751512384815e+00 4.7059770514396533e+00 ";
  rdft_test(decanol,b6_31Gpp,dftcart,Etot,Eorb,"1-decanol, SVWN(RPA)/6-31G**",dip,1,8);


#ifndef COMPUTE_REFERENCE
  printf("****** Tests completed in %s *******\n",t.elapsed().c_str());
#endif

  return 0;
}
