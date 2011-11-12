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
#include "timer.h"
#include "xyzutils.h"

#include <cmath>
#include <cstdio>

/// Relative tolerance in total energy
const double tol=1e-6;
/// Absolute tolerance in orbital energies
const double otol=1e-5;
/// Absolute tolerance for dipole moment
const double dtol=1e-5;

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
  sph.set_bool("Verbose",0);

  // No spherical harmonics
  Settings cart=sph;
  cart.set_bool("UseLM",0);

  // Direct calculation
  Settings direct=sph;
  direct.set_bool("Direct",1);

  // Polarized calculation
  Settings pol=sph;
  pol.set_int("Multiplicity",2);

  // DFT tests

  // Settings for DFT
  Settings dftsph=sph; // Normal settings
  dftsph.add_dft_settings();

  Settings dftcart=cart; // Cartesian basis
  dftcart.add_dft_settings();

  Settings dftnofit=dftsph; // No density fitting
  dftnofit.set_bool("DFTFitting",0);

  Settings dftcart_nofit=dftcart;
  dftcart_nofit.set_bool("DFTFitting",0);

  Settings dftdirect=dftsph; // Direct calculation
  dftdirect.set_bool("Direct",1);
  dftdirect.set_bool("DFTDirect",1);

  Settings dftpol=pol; // Polarized calculation
  dftpol.add_dft_settings();

  Settings dftpol_nofit=dftpol; // Polarized calculation, no density fitting
  dftpol_nofit.set_bool("DFTFitting",0);

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

Etot=-1.2848877555174070e+02;
dip=4.2577768034742185e-16;
Eorb="-3.2765635418395981e+01 -1.9187982338958312e+00 -8.3209725195411433e-01 -8.3209725195410722e-01 -8.3209725195410555e-01 1.6945577283440862e+00 1.6945577283440931e+00 1.6945577283440949e+00 2.1594249508455423e+00 5.1967114015094733e+00 5.1967114015094769e+00 5.1967114015094786e+00 5.1967114015094822e+00 5.1967114015094831e+00 ";
rhf_test(Ne,cc_pVDZ,sph,Etot,Eorb,"Neon, HF/cc-pVDZ",dip);

Etot=-1.2848886617203752e+02;
dip=5.9579004504530739e-16;
Eorb="-3.2765400753051729e+01 -1.9190111439475050e+00 -8.3228218139541099e-01 -8.3228218139540755e-01 -8.3228218139540533e-01 1.6944246985486151e+00 1.6944246985486193e+00 1.6944246985486207e+00 1.9905987739661712e+00 5.1964246081625154e+00 5.1964246081625181e+00 5.1964246081625189e+00 5.1964246081625216e+00 5.1964246081625234e+00 1.0383358448969771e+01 ";
rhf_test(Ne,cc_pVDZ,cart,Etot,Eorb,"Neon, HF/cc-pVDZ cart",dip);

Etot=-1.2853186274066951e+02;
dip=1.2253057810216141e-15;
Eorb="-3.2769120924928458e+01 -1.9270902371391023e+00 -8.4542165337299313e-01 -8.4542165337298514e-01 -8.4542165337297792e-01 1.0976353396416005e+00 1.0976353396416030e+00 1.0976353396416092e+00 1.4176319713976184e+00 2.8141914490294671e+00 2.8141914490294728e+00 2.8141914490294768e+00 2.8142410601974572e+00 2.8142410601974621e+00 6.1093599283716760e+00 6.1093599283717239e+00 6.1093599283717381e+00 9.6473575709792083e+00 9.6473615307789622e+00 9.6473615307789728e+00 9.6473615307789888e+00 9.6968456863136723e+00 9.6968456863136812e+00 9.6968456863136936e+00 1.1227294609864268e+01 1.1227294609864279e+01 1.1227294609864284e+01 1.1227320203475491e+01 1.1227320203475500e+01 1.1744550906659153e+01 ";
rhf_test(Ne,cc_pVTZ,sph,Etot,Eorb,"Neon, HF/cc-pVTZ",dip);

Etot=-1.2853200998517835e+02;
dip=1.7772084115894966e-15;
Eorb="-3.2769827664058582e+01 -1.9274545261953908e+00 -8.4572302044211711e-01 -8.4572302044211023e-01 -8.4572302044210779e-01 8.8038911281179932e-01 1.0282198312978226e+00 1.0282198312978308e+00 1.0282198312978332e+00 2.8138968402503348e+00 2.8138968402503446e+00 2.8138968402503504e+00 2.8138968402503708e+00 2.8138968402503770e+00 4.1362240254778415e+00 4.6398466995992358e+00 4.6398466995992855e+00 4.6398466995993397e+00 9.6470056533527000e+00 9.6470056533527089e+00 9.6470056533527195e+00 9.6470056533527231e+00 9.6470056533527337e+00 9.6470056533527444e+00 9.6470056533527639e+00 1.1226914485524096e+01 1.1226914485524109e+01 1.1226914485524132e+01 1.1226914485524173e+01 1.1226914485524192e+01 1.1317534790996881e+01 1.1317534790996989e+01 1.1317534790997131e+01 1.6394442669789104e+01 2.8816114643496686e+01 ";
rhf_test(Ne,cc_pVTZ,cart,Etot,Eorb,"Neon, HF/cc-pVTZ cart",dip);

Etot=-1.2854347022282593e+02;
dip=5.4503317526329511e-15;
Eorb="-3.2771505026319055e+01 -1.9293376872570711e+00 -8.4896009185628896e-01 -8.4896009185628118e-01 -8.4896009185626209e-01 8.0808837024085012e-01 8.0808837024085312e-01 8.0808837024086855e-01 9.2960643755536165e-01 1.9977855000048264e+00 1.9977855000048366e+00 1.9977897544018810e+00 1.9977897544018872e+00 1.9977897544018941e+00 3.9129701500840688e+00 3.9129701500841469e+00 3.9129701500841825e+00 5.6881660757793728e+00 5.9042206757002971e+00 5.9042206757003060e+00 5.9042206757003211e+00 5.9042214194923881e+00 5.9212854400253763e+00 5.9212854400253851e+00 5.9212854400254020e+00 6.7608899318212625e+00 6.7608899318212945e+00 6.7612175572270070e+00 6.7612175572270319e+00 6.7612175572270496e+00 1.4561015877916223e+01 1.4903624044543927e+01 1.4903624044543957e+01 1.4903624044543980e+01 1.4913822169025655e+01 1.4913822169025673e+01 1.4913822169025680e+01 1.4921113455810399e+01 1.4921113455810454e+01 1.5734128922772415e+01 1.5734128922772742e+01 1.5734128922772756e+01 1.9794586190805710e+01 1.9794586190805742e+01 1.9794586190805749e+01 1.9794589079827531e+01 1.9857009295631080e+01 1.9857009295631148e+01 1.9857009295631183e+01 2.0961123055107794e+01 2.0961123055107951e+01 2.0961123055107969e+01 2.0965848737833618e+01 2.0965848737834190e+01 6.7460902813221438e+01 ";
rhf_test(Ne,cc_pVQZ,sph,Etot,Eorb,"Neon, HF/cc-pVQZ",dip);

Etot=-1.2854353449722339e+02;
dip=4.8024440044317859e-15;
Eorb="-3.2771625133531764e+01 -1.9294942860573698e+00 -8.4906688514309958e-01 -8.4906688514309492e-01 -8.4906688514305306e-01 5.8690441396935189e-01 7.1271797631749889e-01 7.1271797631750311e-01 7.1271797631762124e-01 1.9879845915547711e+00 1.9879845915548096e+00 1.9879845915548107e+00 1.9879845915548666e+00 1.9879845915549341e+00 2.5105148491858986e+00 2.7214792292153480e+00 2.7214792292154550e+00 2.7214792292160803e+00 5.9040962877416092e+00 5.9040962877416518e+00 5.9040962877416829e+00 5.9040962877416883e+00 5.9040962877416892e+00 5.9040962877416998e+00 5.9040962877417167e+00 6.4115733377599318e+00 6.5684069285856070e+00 6.5684069285856292e+00 6.5684069285856568e+00 6.5684069285856710e+00 6.5684069285857607e+00 6.7659165986397669e+00 6.7659165986404384e+00 6.7659165986412946e+00 1.4004805311711563e+01 1.4903514352972373e+01 1.4903514352972387e+01 1.4903514352972408e+01 1.4903514352972440e+01 1.4903514352972483e+01 1.4903514352972506e+01 1.4903514352972518e+01 1.4903514352972552e+01 1.4903514352972566e+01 1.8145155382781539e+01 1.8145155382784200e+01 1.8145155382784981e+01 1.8145155382786175e+01 1.8145155382788055e+01 1.8540067449710449e+01 1.8540067449711103e+01 1.8540067449711440e+01 1.9794449042785452e+01 1.9794449042785491e+01 1.9794449042785502e+01 1.9794449042785548e+01 1.9794449042785580e+01 1.9794449042785654e+01 1.9794449042785697e+01 2.9727979554450570e+01 3.9089870733234612e+01 3.9089870733281558e+01 3.9089870733295953e+01 3.9089870733317959e+01 3.9089870733362140e+01 3.9551871548229094e+01 3.9551871548237962e+01 3.9551871548243902e+01 5.8376821808720806e+01 2.0568373997745439e+02 ";
rhf_test(Ne,cc_pVQZ,cart,Etot,Eorb,"Neon, HF/cc-pVQZ cart",dip);


Etot=-4.5944650895291505e+02;
dip=1.6279610443996865e-15;
Eorba="-1.0487193636164037e+02 -1.0607738780246452e+01 -8.0945509705693688e+00 -8.0689409087202364e+00 -8.0689409087202009e+00 -1.1333638261126586e+00 -5.7594850068639247e-01 -5.0142413714901823e-01 -5.0142413714901701e-01 5.0037888527708141e-01 5.6757983809027079e-01 6.0901660173608185e-01 6.0901660173609418e-01 1.0467100945496615e+00 1.0627725583396097e+00 1.0627725583396108e+00 1.1177523880811377e+00 1.1177523880811429e+00 ";
Eorbb="-1.0486051371469196e+02 -1.0596350819637072e+01 -8.0629610577828146e+00 -8.0629610577828021e+00 -8.0462002537036170e+00 -1.0112318701805689e+00 -4.7613937358732594e-01 -4.7613937358732400e-01 -3.7645713587398447e-02 5.2464091242015976e-01 6.1785765544032534e-01 6.1785765544033100e-01 6.6235436561625982e-01 1.1278250304107702e+00 1.1278250304107853e+00 1.1604786776790408e+00 1.1604786776790805e+00 1.1734237347660605e+00 ";
uhf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, HF/6-31G** polarized",dip);

Etot=-4.5944278096727612e+02;
dip=7.0090932824453482e-16;
Eorba="-1.0487302831574709e+02 -1.0608495662325550e+01 -8.0951291015495777e+00 -8.0696309800107802e+00 -8.0696309800107731e+00 -1.1298658044427727e+00 -5.6816410791988337e-01 -5.0428469389427422e-01 -5.0428469389427233e-01 5.0036388892020800e-01 5.6967746969977351e-01 6.0847085717999871e-01 6.0847085718001703e-01 1.0477281890196122e+00 1.0645036184690844e+00 1.0645036184691088e+00 1.1152515027775189e+00 1.1152515027775249e+00 ";
Eorbb="-1.0486248004895859e+02 -1.0598189498391704e+01 -8.0647978854622870e+00 -8.0647978854622764e+00 -8.0481964071440277e+00 -1.0119816525363763e+00 -4.7452860595709051e-01 -4.7452860595708729e-01 -4.4311873785400233e-02 5.2403725517101241e-01 6.1759764253060523e-01 6.1759764253061722e-01 6.5850375934815786e-01 1.1291203963033460e+00 1.1291203963033514e+00 1.1573546698285953e+00 1.1573546698286017e+00 1.1669379293461675e+00 ";
rohf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, ROHF/6-31G**",dip);

Etot=-4.6013022751535834e+02;
dip=3.4940743940151340e-15;
Eorba="-1.0170616372367826e+02 -9.6270613481585947e+00 -7.4215007210384627e+00 -7.3996629851727951e+00 -7.3996629827147915e+00 -8.8862294034893807e-01 -4.5635683138477418e-01 -4.0419451985748289e-01 -4.0419451765045877e-01 2.5057969404128683e-01 3.1403582128529006e-01 3.4408761620909517e-01 3.4408761750637101e-01 7.4271225985955980e-01 7.5560893571135435e-01 7.5560893829162468e-01 7.9851537085357815e-01 7.9851537094200875e-01 ";
Eorbb="-1.0170188614300524e+02 -9.6221049691543996e+00 -7.4057859324844806e+00 -7.3958695740026066e+00 -7.3958695727808887e+00 -8.4901671029495263e-01 -3.8946955454424120e-01 -3.8946952834186305e-01 -3.3071553199144799e-01 2.5640881950134414e-01 3.2914364303653620e-01 3.4938978863489872e-01 3.4938987699352775e-01 7.8954186444513408e-01 7.9792128890956149e-01 7.9792139915451998e-01 8.1008595668805228e-01 8.1008598458136449e-01 ";
udft_test(Cl,b6_31Gpp,dftpol_nofit,Etot,Eorba,Eorbb,"Chlorine, B3LYP/6-31G** polarized",dip,402,0);


Etot=-1.1287000934442002e+00;
dip=1.3035568079666237e-15;
Eorb="-5.9241098912997037e-01 1.9744005747008161e-01 4.7932104727503055e-01 9.3732369228734513e-01 1.2929037097205383e+00 1.2929037097205385e+00 1.9570226089461711e+00 2.0435200542857599e+00 2.0435200542857648e+00 3.6104742345559311e+00 ";
rhf_test(H2,cc_pVDZ,sph,Etot,Eorb,"Hydrogen molecule, HF/cc-pVDZ",dip);

Etot=-1.1676143092623921e+00;
dip=5.7230394244387611e-14;
Eorb="-3.9201513545790040e-01 3.6521693428308653e-02 2.9071357867631598e-01 6.5833191106402889e-01 9.7502316701959901e-01 9.7502316701960345e-01 1.6066125614720521e+00 1.7001805903516158e+00 1.7001805903516216e+00 3.1926504577916557e+00 ";
rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, SVWN(RPA)/cc-pVDZ",dip,1,8);

Etot=-1.1603968183189468e+00;
dip=5.7829569965225553e-14;
Eorb="-3.7849179642446074e-01 5.3523955584588896e-02 3.0277109480161463e-01 6.6374922440171613e-01 9.9246481785307061e-01 9.9246481785307472e-01 1.6235428847664179e+00 1.7198876745459353e+00 1.7198876745459397e+00 3.2019315228044354e+00 ";
rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, PBEPBE/cc-pVDZ",dip,101,130);


Etot=-7.6056846941492239e+01;
dip=7.9468052620252772e-01;
Eorb="-2.0555336494290437e+01 -1.3429167316882891e+00 -7.0831770005881056e-01 -5.7581634276390836e-01 -5.0395807492503641e-01 1.4187676846717273e-01 2.0351837529069691e-01 5.4325209915269290e-01 5.9743073210030484e-01 6.6902610737716672e-01 7.8618191066547838e-01 8.0264561250954125e-01 8.0477457822149878e-01 8.5895459635771587e-01 9.5696003496595528e-01 1.1369012691880582e+00 1.1927526960039001e+00 1.5238457560921839e+00 1.5568958809015019e+00 2.0323642923778209e+00 2.0594264917537419e+00 2.0653811350682667e+00 2.1684100423644539e+00 2.2356007678042022e+00 2.5930811522366031e+00 2.9554689196892463e+00 3.3611156372923165e+00 3.4906525506065513e+00 3.5741552492651958e+00 3.6466195206482825e+00 3.8098890783242800e+00 3.8721537693802599e+00 3.8807784202249862e+00 3.9569185356527172e+00 3.9987665717213092e+00 4.0736543414563497e+00 4.1801785785414598e+00 4.3092425549595230e+00 4.3760461783329285e+00 4.5715947031945925e+00 4.6348116501414145e+00 4.8596765239600774e+00 5.1402288830676008e+00 5.2874806355104695e+00 5.5463372077378423e+00 6.0424050972607155e+00 6.5330036909229863e+00 6.9016693572441303e+00 6.9365453918719293e+00 6.9802319185887889e+00 7.0182977432260367e+00 7.1338792443059091e+00 7.2154670869360693e+00 7.2255943859438876e+00 7.4235576550768911e+00 7.7211430733016044e+00 8.2697507670958679e+00 1.2755307290415695e+01 ";
rhf_test(h2o,cc_pVTZ,sph,Etot,Eorb,"Water, HF/cc-pVTZ",dip);
rhf_test(h2o,cc_pVTZ,direct,Etot,Eorb,"Water, HF/cc-pVTZ direct",dip);

Etot=-7.6064486129654597e+01;
dip=7.8763344210832442e-01;
Eorb="-2.0560367084761562e+01 -1.3467279558537981e+00 -7.1288511365578910e-01 -5.8000887698669745e-01 -5.0760164857204293e-01 1.1671779557461866e-01 1.7059110198268104e-01 4.4885066079262448e-01 4.6188036046385150e-01 4.9872307734485455e-01 5.8265741576508212e-01 6.0652132003561854e-01 6.1372824572608198e-01 6.5415548135142254e-01 7.1859272806445429e-01 8.5096442850783116e-01 9.1889568229617158e-01 1.1089440956478569e+00 1.1547897327499324e+00 1.3489353408510727e+00 1.4145430582017147e+00 1.4772771456199136e+00 1.4849254796210460e+00 1.5824421698859339e+00 1.6814168738257553e+00 1.9064203213929718e+00 2.0719426292294285e+00 2.1991315389088379e+00 2.2841432964349546e+00 2.3586502549157262e+00 2.4266159681151964e+00 2.4848059348237355e+00 2.5282276233822349e+00 2.5586983026946624e+00 2.5784363646038262e+00 2.6480496044424147e+00 2.6695678012793276e+00 2.8427253096601075e+00 2.8775605697524358e+00 3.0425617289224696e+00 3.1259879432215660e+00 3.2920350652404640e+00 3.3395021121538457e+00 3.4423186471801346e+00 3.6243324540558279e+00 3.7991421486269816e+00 3.9963385448786375e+00 4.1262788567926183e+00 4.1837260596952639e+00 4.2120469576544748e+00 4.4530091232399052e+00 4.4836444650904701e+00 4.6832751526737209e+00 4.7273379550557051e+00 4.7606790955662168e+00 4.9144446781241440e+00 5.3663102740568807e+00 5.4263411162175172e+00 5.9901140316210633e+00 6.0936990075105619e+00 6.2358276609635093e+00 6.3038740273133271e+00 6.6880102715796150e+00 6.7937504101472523e+00 7.0711800481109606e+00 7.2310189200719979e+00 7.2927776942141183e+00 7.3308237091490938e+00 7.3646486762815861e+00 7.5296481601020409e+00 7.6177264386491963e+00 7.7259762399426961e+00 8.0066864817845680e+00 8.0888021493254438e+00 8.1005754452224981e+00 8.1132910697830418e+00 8.1864355212313189e+00 8.2671470057373018e+00 8.3145007752866427e+00 8.3183957290779666e+00 8.4098910004378382e+00 8.5910748709171969e+00 8.9095636037501400e+00 8.9339142540469023e+00 8.9932612632776348e+00 9.1554548077369322e+00 9.2899792406820936e+00 9.3532959323842597e+00 9.9081699805293972e+00 1.0058748517604206e+01 1.0262787786344534e+01 1.0435357071922109e+01 1.0575253930423669e+01 1.0631852880602649e+01 1.0758111735942432e+01 1.1259613030137521e+01 1.1410287471913387e+01 1.1573640507475204e+01 1.1667889984853142e+01 1.1716060749837725e+01 1.1846399335792830e+01 1.2192549637744344e+01 1.2296711079032852e+01 1.2419184467652103e+01 1.2441159735387341e+01 1.2467586335048107e+01 1.3576020917998552e+01 1.3762676882543792e+01 1.4186349628182191e+01 1.4557092138645370e+01 1.4720169032263019e+01 1.4865456868711838e+01 1.6429257204119970e+01 1.6870043607765183e+01 4.4590591578286137e+01 ";
rhf_test(h2o,cc_pVQZ,sph,Etot,Eorb,"Water, HF/cc-pVQZ",dip);
rhf_test(h2o,cc_pVQZ,direct,Etot,Eorb,"Water, HF/cc-pVQZ direct",dip);

Etot=-7.6373015774859411e+01;
dip=7.3245474364228935e-01;
Eorb="-1.8739027176571941e+01 -9.1593177064181119e-01 -4.7162507547539667e-01 -3.2513040969496232e-01 -2.4828781845582931e-01 8.4298285711835799e-03 8.1123589306273933e-02 3.3805332085243173e-01 3.7944807500365380e-01 4.6504201700220477e-01 5.4461928263523196e-01 5.9069081829763959e-01 5.9681408700463023e-01 6.4396732302180526e-01 7.4405630720304683e-01 8.8495843281167141e-01 9.7375998141061126e-01 1.2410362424058821e+00 1.2604588452773935e+00 1.6998263280155383e+00 1.7261026983319157e+00 1.7619295038171823e+00 1.8253896498807232e+00 1.8994352971627309e+00 2.1868863094069377e+00 2.5299481007944427e+00 2.9877820415920651e+00 3.1249844792711858e+00 3.1891654232721134e+00 3.2804945219315624e+00 3.2904149797601363e+00 3.4371339908943814e+00 3.5077600889126619e+00 3.5697576660929839e+00 3.5997876942957805e+00 3.6362361651934596e+00 3.6869817024523552e+00 3.9239622873547373e+00 3.9460269658998630e+00 4.1576805398109009e+00 4.1640324681760239e+00 4.4333534324481052e+00 4.6494714940360309e+00 4.7623037799261541e+00 5.0045778092246023e+00 5.4886956516499614e+00 5.9727904802182268e+00 6.2826502227467484e+00 6.2900867157407516e+00 6.3851747094888314e+00 6.3900722804133245e+00 6.5556459769915927e+00 6.5919087332916355e+00 6.6592987100708063e+00 6.8086279859775063e+00 7.1135701621100145e+00 7.6305321399837940e+00 1.1915611445461266e+01 ";
rdft_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",dip,101,130);

Etot=-7.6373079893452314e+01;
dip=7.3219773633470164e-01;
Eorb="-1.8739142088578276e+01 -9.1601297181475128e-01 -4.7171104265230407e-01 -3.2520825199624509e-01 -2.4836336973204318e-01 7.6601589987456333e-03 8.0195388358481098e-02 3.3765682940311470e-01 3.7907177054242541e-01 4.6475499424206823e-01 5.4445502863035999e-01 5.9025778748566593e-01 5.9656863868960774e-01 6.4383439260957753e-01 7.4392939449220774e-01 8.8482749913670489e-01 9.7358976950521436e-01 1.2408268481793427e+00 1.2603313810633168e+00 1.6997122179426580e+00 1.7259855256831620e+00 1.7615396401910233e+00 1.8252276135531587e+00 1.8990778039474190e+00 2.1867270514033432e+00 2.5297913375491707e+00 2.9877392287672504e+00 3.1248908049107742e+00 3.1891266247883276e+00 3.2803033083574289e+00 3.2903102584899977e+00 3.4371038345394052e+00 3.5077381951785696e+00 3.5696997160723174e+00 3.5996632722374025e+00 3.6360207952183141e+00 3.6867991136017402e+00 3.9239658191044953e+00 3.9459457670962790e+00 4.1576759350343560e+00 4.1639729212759935e+00 4.4332751826456063e+00 4.6494297776071623e+00 4.7622366649318177e+00 5.0044243213517241e+00 5.4886667224783006e+00 5.9727890440068050e+00 6.2825939356229581e+00 6.2899820449931196e+00 6.3850574061020442e+00 6.3900558348651337e+00 6.5556360599838710e+00 6.5918049163586003e+00 6.6593396925040373e+00 6.8086180722950429e+00 7.1136038363065603e+00 7.6303042381744977e+00 1.1915497058543648e+01 ";
rdft_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",dip,101,130);
rdft_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",dip,101,130);

Etot=-7.6374652625361463e+01;
dip=7.3272138191672287e-01;
Eorb="-1.8741555993853424e+01 -9.1788332691442143e-01 -4.7348563345394695e-01 -3.2745767998187358e-01 -2.5054811788598164e-01 2.7902909073756798e-03 7.8041353971598948e-02 3.2383354481960280e-01 3.5924081020054355e-01 4.5242281654942418e-01 5.1413448271123252e-01 5.7767337511395889e-01 5.8423714124606063e-01 6.4253338491613743e-01 6.5976427792134440e-01 7.4241899941266865e-01 9.7186945683983506e-01 1.1822066966049074e+00 1.2023219758637302e+00 1.5759177346289304e+00 1.6360770227879295e+00 1.6982195959162769e+00 1.7245160457949971e+00 1.8628724225116338e+00 1.9081328543298910e+00 2.1641869109523628e+00 2.3473123833624467e+00 2.7892817994941663e+00 3.0049313589875335e+00 3.0831687504528213e+00 3.1876655118720070e+00 3.2328096309250971e+00 3.4168965972815806e+00 3.4579229704907020e+00 3.5050043398539628e+00 3.5282519119376041e+00 3.5683147286756953e+00 3.5772256884568354e+00 3.8379025117513628e+00 3.9226522050236485e+00 4.0867030409744229e+00 4.0926883755162908e+00 4.3310822057998717e+00 4.4154607766877820e+00 4.4322657362960483e+00 4.6027454144647058e+00 5.1266091749009668e+00 5.2200926905619944e+00 5.4840599775811771e+00 6.1494605841411385e+00 6.2799797314090071e+00 6.2885759840151003e+00 6.3502579561974661e+00 6.4059333089989945e+00 6.4357988498319223e+00 6.6570159394443067e+00 6.7153035962087770e+00 6.7372358719708849e+00 6.9398737783096118e+00 7.3406789124075740e+00 8.2789334416020530e+00 8.3551946539830162e+00 9.3390665069469492e+00 1.4480084218173177e+01 1.5822389089335877e+01 ";
rdft_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",dip,101,130);


Etot=-5.6637319431552505e+03;
dip=4.1660090782244827e+00;
Eorb="-9.4985623404324394e+02 -1.4146635099122020e+02 -1.3119294655421876e+02 -1.3119287270611147e+02 -1.3119259496786984e+02 -2.7676547720402841e+01 -2.3232980821838634e+01 -2.3232722836101512e+01 -2.3231117121670440e+01 -1.6049049315200051e+01 -1.6049045556498967e+01 -1.6047827658242213e+01 -1.6047724027585080e+01 -1.6047713458707900e+01 -1.5604531097548763e+01 -1.5532249496359425e+01 -1.1296661973756294e+01 -1.1249243895093182e+01 -1.1232665103771085e+01 -4.3970789601403419e+00 -2.8992752335524243e+00 -2.8986598470609359e+00 -2.8951186992178997e+00 -1.4177740924272384e+00 -1.2312596778565208e+00 -1.0610694274888925e+00 -8.7645299195720050e-01 -8.5303383017589762e-01 -8.1305176260999268e-01 -7.2468345937522116e-01 -7.1752260893690478e-01 -7.1751287362360971e-01 -7.1453927096582881e-01 -7.1369023729064263e-01 -6.5649508087306740e-01 -6.5484508971221644e-01 -6.4819561364580236e-01 -6.1951446971103685e-01 -5.1149277612520905e-01 -4.5694083034690181e-01 -3.6925756183658126e-01 -1.8059223079898942e-01 6.9314373055344658e-02 7.4011358154239038e-02 1.1409014945265671e-01 1.4993230227407597e-01 1.8266978402172024e-01 1.9355783398487214e-01 2.1197841085286168e-01 2.5237135015405704e-01 2.7656209472406329e-01 2.8532363502078562e-01 3.0336608170787632e-01 3.3343210193517742e-01 3.3688908923283800e-01 3.9652956651816207e-01 4.2174259977195599e-01 5.4893795783299704e-01 5.6113636018611412e-01 6.8232569239580376e-01 8.8548528960108186e-01 9.2615820683911998e-01 9.2670940578916072e-01 9.6328468558478197e-01 9.8346702018784671e-01 9.9887404064082908e-01 1.0364505443043599e+00 1.0834412264707201e+00 1.0936564417373447e+00 1.1989337429099993e+00 1.2617670077947041e+00 1.2818433277043921e+00 1.3193949782406433e+00 1.3895935370179857e+00 1.4308892989337034e+00 1.4702798437804183e+00 1.4945329130005354e+00 1.5683750074167071e+00 1.5822512320218465e+00 1.6271531769229546e+00 1.6323132920882668e+00 1.6700777141869541e+00 1.7294530360141209e+00 1.8374560574555538e+00 1.9460156088860219e+00 1.9779608207087003e+00 2.0568938351850083e+00 2.2440133809086360e+00 2.9829355616526563e+00 3.0788481978082940e+00 5.2757403445528555e+00 2.1121787319104163e+02 ";
rhf_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G",dip);

Etot=-5.6676694487080886e+03;
dip=3.3439626626281620e+00;
Eorb="-9.3989970802324854e+02 -1.3767102651461383e+02 -1.2795643245359548e+02 -1.2795632835312686e+02 -1.2795612375298256e+02 -2.5984469851694044e+01 -2.1817700276382919e+01 -2.1817458415910192e+01 -2.1816375309883387e+01 -1.5092628193760929e+01 -1.5092615930476127e+01 -1.5091970302460082e+01 -1.5091824810205463e+01 -1.5091780067270058e+01 -1.4452049631540724e+01 -1.4376825215348255e+01 -1.0301666235038896e+01 -1.0271614405801493e+01 -1.0254343790679556e+01 -3.7636257246221980e+00 -2.4165865858654190e+00 -2.4159457387903300e+00 -2.4131554866896527e+00 -1.1211260790742024e+00 -9.6404171299762298e-01 -8.2563489340376783e-01 -7.0330471988316878e-01 -6.8267059748353254e-01 -6.4806997339508654e-01 -5.4367494773968295e-01 -5.3721128038117849e-01 -5.2553390638845543e-01 -5.2468812784878238e-01 -5.2443092901283650e-01 -5.1974411480468552e-01 -5.1950960977335880e-01 -5.1476981752248874e-01 -4.9177475505953366e-01 -3.8454542132115443e-01 -3.8028108105913538e-01 -3.2171095463210797e-01 -1.8694100513849543e-01 -5.9084801281267314e-02 -5.6818001344707249e-02 -4.6515536576166534e-02 -4.3790677046845111e-02 -1.6270531120991014e-02 -1.5552733133593670e-02 3.4348606028108186e-02 5.9150203685535807e-02 6.8898918402361967e-02 9.1045793496376301e-02 1.0203724929194383e-01 1.2425960398372865e-01 1.3808906163208742e-01 1.5836786694185412e-01 1.8698825679956552e-01 3.0351634099703206e-01 3.0990079062358727e-01 3.9883842872769976e-01 5.8813206014783670e-01 5.9391015063791075e-01 6.1271397990948606e-01 6.5246284448473701e-01 6.6253039084265597e-01 6.7302721925981657e-01 6.9271084717653797e-01 7.7761267809071322e-01 7.9855444858793856e-01 8.6965079545012625e-01 8.9908636257207963e-01 9.3736998360266499e-01 9.6540438833404174e-01 9.9998313715702380e-01 1.0476183373289532e+00 1.1414378586673011e+00 1.1555399725280446e+00 1.2222181518451256e+00 1.2400223661873069e+00 1.2657682922383289e+00 1.2729528024019972e+00 1.3149204480085610e+00 1.3519132868290871e+00 1.4869553360999517e+00 1.6023132137417548e+00 1.6214003549954670e+00 1.7041933595649283e+00 1.8438991871808690e+00 2.6031647528278725e+00 2.7024602421548751e+00 4.7145901000799508e+00 2.0804459430073808e+02 ";
rdft_test(cdcplx,b3_21G,dftcart_nofit,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",dip,402,0);


Etot=-4.6701319818693014e+02;
dip=5.6559855225611855e-01;
Eorb="-1.8627155301581880e+01 -9.8468874173455170e+00 -9.8000384489074204e+00 -9.7985016745498328e+00 -9.7964371507226442e+00 -9.7961805571382232e+00 -9.7961545994074140e+00 -9.7957608900872284e+00 -9.7955848267094403e+00 -9.7955236806244539e+00 -9.7902008615455447e+00 -9.4290502202757454e-01 -7.5219671105579522e-01 -7.3523093011033325e-01 -7.0339270417721422e-01 -6.7053952764389169e-01 -6.3196276493270021e-01 -5.8307080622444185e-01 -5.5339026493589416e-01 -5.4222267239430488e-01 -5.1586102756443741e-01 -5.0988669304581613e-01 -4.8254342844467740e-01 -4.3597802916277439e-01 -4.3184684300699944e-01 -4.1614248866447223e-01 -4.0900699654416223e-01 -4.0310234166130748e-01 -3.8729602580381511e-01 -3.7610094325442883e-01 -3.7081059352513135e-01 -3.4983156380159530e-01 -3.4595188008050054e-01 -3.3597038783102806e-01 -3.2455651687506426e-01 -3.2107021440036948e-01 -3.1361114588627592e-01 -2.9876285072317005e-01 -2.9369126910038756e-01 -2.9084593296399330e-01 -2.8883502959380813e-01 -2.8096967503901837e-01 -2.7633800431594591e-01 -2.6485286810151637e-01 -2.2609508600063494e-01 2.9806358607672157e-02 4.3501515813973948e-02 4.7426331622794665e-02 5.3710398057438324e-02 6.2797761719347547e-02 6.8713925369995751e-02 7.6076273276068279e-02 8.0543672895948618e-02 9.0742671049196325e-02 1.0571719421231489e-01 1.1202254190262607e-01 1.1622956768994037e-01 1.2071476251097721e-01 1.3140890768329697e-01 1.3642536092606194e-01 1.3882279701866651e-01 1.4089000099155002e-01 1.4347194630350155e-01 1.4889817838486480e-01 1.5392813952404985e-01 1.6458426273602861e-01 1.6998895848019457e-01 1.7439311117942460e-01 1.8135049706306686e-01 1.9191440460943743e-01 1.9907617129475319e-01 2.0636234908676498e-01 2.1896311327548065e-01 2.2823818703083906e-01 2.4034428615542849e-01 2.4854804819717405e-01 2.5464843499588924e-01 4.2493065121595069e-01 4.2850802263754134e-01 4.4063971489779896e-01 4.5736035357203575e-01 4.6331474724548838e-01 4.6892402398939359e-01 4.7902709797137721e-01 4.8941869116927655e-01 5.0243309050792850e-01 5.1194329755084056e-01 5.1904635631233065e-01 5.3632493037260753e-01 5.4651307214580269e-01 5.7350188963405135e-01 5.9412743333427176e-01 6.0273739926020975e-01 6.0614700962877033e-01 6.1191668141724365e-01 6.1545345380989636e-01 6.3140321778531838e-01 6.4954414131314764e-01 6.7503142267427862e-01 6.8567452681798835e-01 6.9675953359883369e-01 7.1482446783774434e-01 7.2341278269656761e-01 7.4573344307292699e-01 7.5077708319055614e-01 7.6086205134131346e-01 7.7072161138511530e-01 7.7609470848177187e-01 7.8186544898739263e-01 7.9605245162639104e-01 8.0732940619050897e-01 8.1699472358119385e-01 8.2988590893425895e-01 8.3746931589010853e-01 8.3872088289890157e-01 8.4332455768264025e-01 8.4899318188638107e-01 8.5771951340422214e-01 8.6351984811909899e-01 8.6889316590960441e-01 8.7768169100443805e-01 8.8613617548042278e-01 8.9531622261748256e-01 9.0580427466154889e-01 9.1529337171024450e-01 9.2211808511806970e-01 9.4122945248052103e-01 9.6566452470084518e-01 9.7153254467470884e-01 9.8203275895322961e-01 1.0177363041860121e+00 1.0490687796367200e+00 1.0974974191477564e+00 1.1473858808804045e+00 1.1642857044232651e+00 1.2116509713413928e+00 1.2321591725898140e+00 1.2665609852532511e+00 1.2725685255996300e+00 1.3173852768004477e+00 1.3344569630125862e+00 1.3696828991364194e+00 1.4032271656744788e+00 1.4066944040415530e+00 1.4522673664731967e+00 1.4859420175944789e+00 1.4994159138103147e+00 1.5182761687121662e+00 1.5407576497029549e+00 1.5551897836697612e+00 1.5718412836407119e+00 1.5854144158184076e+00 1.6035526853838804e+00 1.6248642126214461e+00 1.6295951420084813e+00 1.6386248588264674e+00 1.6518533116703891e+00 1.6708607907925055e+00 1.7082437982613836e+00 1.7240827783293706e+00 1.7309942350107630e+00 1.7768183389365839e+00 1.7799398660297212e+00 1.7966935556128929e+00 1.7986810453057829e+00 1.8208994360926991e+00 1.8371998609345161e+00 1.8486108500063434e+00 1.8627498022903046e+00 1.8684091264479474e+00 1.8910559128748419e+00 1.9068258026476459e+00 1.9273957858030992e+00 1.9366436757129160e+00 1.9517985648677585e+00 1.9711842068945538e+00 1.9748130241014061e+00 1.9784540785465190e+00 2.0029274792623188e+00 2.0163942909747652e+00 2.0242110419693762e+00 2.0282103953521955e+00 2.0446485101895266e+00 2.0506329498041191e+00 2.0622358768252460e+00 2.0764526144849600e+00 2.0982718449097013e+00 2.1124502161243970e+00 2.1473839706375863e+00 2.1546259208188099e+00 2.1669067770308290e+00 2.1723421942782410e+00 2.1811754703162349e+00 2.1987624020101171e+00 2.2110761554917797e+00 2.2189941107844779e+00 2.2523870667925605e+00 2.2601853996398482e+00 2.2680784527354190e+00 2.2959483712284485e+00 2.3105464713927222e+00 2.3159938576842114e+00 2.3268620142028262e+00 2.3486724221497908e+00 2.3828960538448967e+00 2.3876849961438911e+00 2.4069227384647367e+00 2.4220205548502212e+00 2.4322425353593831e+00 2.4627680071926359e+00 2.4929214356964908e+00 2.5133881938478617e+00 2.5312882302810977e+00 2.5380554930215360e+00 2.5674715217525552e+00 2.5816434852592032e+00 2.5894759729766359e+00 2.6092016674939962e+00 2.6302175742339333e+00 2.6355318820675575e+00 2.6434877962204877e+00 2.6604217859474315e+00 2.6727747259057910e+00 2.6917619686532532e+00 2.6952993948960380e+00 2.7073943399018838e+00 2.7113429426762989e+00 2.7285876900225849e+00 2.7487932419390999e+00 2.7749375514895620e+00 2.7823800012439523e+00 2.7848200351928587e+00 2.7958610646980211e+00 2.8014819209540867e+00 2.8080216681274797e+00 2.8118710988508266e+00 2.8150035493770225e+00 2.8202133592891041e+00 2.8419232392016966e+00 2.8601395937970748e+00 2.8723560777640125e+00 2.9059065585981982e+00 3.0865699678960490e+00 3.1217730680453579e+00 3.1464489621489218e+00 3.1676574300975533e+00 3.1811710242337097e+00 3.1906303056768661e+00 3.1946230264740647e+00 3.2130866408916745e+00 3.2280265519129876e+00 3.2455317269176942e+00 3.2648011366348704e+00 3.2857852337910640e+00 3.3101637942455899e+00 3.3528866631410152e+00 3.3823782792983450e+00 3.3896317331023371e+00 3.3912085870745052e+00 3.4239212642844334e+00 3.4639871645179832e+00 3.4735415295990375e+00 3.4830950700637198e+00 3.4844482482024386e+00 3.8198817583626821e+00 4.1566308478299998e+00 4.2134200285956664e+00 4.2755953750368123e+00 4.3733047198138166e+00 4.4240191496274504e+00 4.4487028257935384e+00 4.5000873476792060e+00 4.5662880204349827e+00 4.6275732208147646e+00 4.7059750715076651e+00 ";
rdft_test(decanol,b6_31Gpp,dftcart,Etot,Eorb,"1-decanol, SVWN(RPA)/6-31G**",dip,1,8);




#ifndef COMPUTE_REFERENCE
  printf("****** Tests completed in %s *******\n",t.elapsed().c_str());
#endif

  return 0;
}
