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

Etot=-4.6013022808792300e+02;
dip=5.0379795830803330e-15;
Eorba="-1.0170613118760916e+02 -9.6270381887329926e+00 -7.4214810327865512e+00 -7.3996420320561160e+00 -7.3996420317234861e+00 -8.8861368852771316e-01 -4.5634941954800934e-01 -4.0418685582401531e-01 -4.0418685545226163e-01 2.5058397968354051e-01 3.1404088101582717e-01 3.4409248377731305e-01 3.4409248398016196e-01 7.4272156291347680e-01 7.5561833848938897e-01 7.5561833899845998e-01 7.9852545219887183e-01 7.9852545221700011e-01 ";
Eorbb="-1.0170188054910723e+02 -9.6221028417899142e+00 -7.4057882276731002e+00 -7.3958708707388308e+00 -7.3958708707060365e+00 -8.4901920057402891e-01 -3.8947175091527730e-01 -3.8947174942595331e-01 -3.3071654059575673e-01 2.5640702756042633e-01 3.2914286244594759e-01 3.4938752793185901e-01 3.4938754226916802e-01 7.8954056288262076e-01 7.9792000652055828e-01 7.9792001832911741e-01 8.1008369919664791e-01 8.1008370084890580e-01 ";
udft_test(Cl,b6_31Gpp,dftpol_nofit,Etot,Eorba,Eorbb,"Chlorine, B3LYP/6-31G** polarized",dip,402,0);

Etot=-1.1287000934442002e+00;
dip=1.3035568079666237e-15;
Eorb="-5.9241098912997037e-01 1.9744005747008161e-01 4.7932104727503055e-01 9.3732369228734513e-01 1.2929037097205383e+00 1.2929037097205385e+00 1.9570226089461711e+00 2.0435200542857599e+00 2.0435200542857648e+00 3.6104742345559311e+00 ";
rhf_test(H2,cc_pVDZ,sph,Etot,Eorb,"Hydrogen molecule, HF/cc-pVDZ",dip);

Etot=-1.1676141182477346e+00;
dip=5.2498096767036032e-14;
Eorb="-3.9201515930900521e-01 3.6518736496371641e-02 2.9071356473378135e-01 6.5832910702777703e-01 9.7502281428651250e-01 9.7502281428651727e-01 1.6066119799002103e+00 1.7001805817601914e+00 1.7001805817601963e+00 3.1926513611695286e+00 ";
rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, SVWN(RPA)/cc-pVDZ",dip,1,8);

Etot=-1.1603966181999281e+00;
dip=6.4431445107792183e-14;
Eorb="-3.7849174586615986e-01 5.3520825347401782e-02 3.0277169431072132e-01 6.6374632575271353e-01 9.9246453656852118e-01 9.9246453656852618e-01 1.6235424258671591e+00 1.7198877535389889e+00 1.7198877535389929e+00 3.2019321531425726e+00 ";
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

Etot=-7.6373011395128387e+01;
dip=7.3245302812334034e-01;
Eorb="-1.8739011902300884e+01 -9.1592775834747797e-01 -4.7162463042003333e-01 -3.2513123126533056e-01 -2.4828873364932166e-01 8.4304564548632263e-03 8.1123760224361199e-02 3.3805356793844493e-01 3.7944835337946387e-01 4.6504216039907492e-01 5.4461891097867110e-01 5.9069013555461791e-01 5.9681434212284179e-01 6.4396751609853731e-01 7.4405734251168720e-01 8.8496139402637697e-01 9.7375991609568979e-01 1.2410365187444512e+00 1.2604610688863378e+00 1.6998276471463050e+00 1.7261039070391588e+00 1.7619299971633904e+00 1.8253903650754808e+00 1.8994378826619920e+00 2.1868861722042801e+00 2.5299477923118046e+00 2.9877821706086540e+00 3.1249847100117241e+00 3.1891658307902686e+00 3.2804949224123749e+00 3.2904059756920119e+00 3.4371344754162920e+00 3.5077605537795709e+00 3.5697580747369999e+00 3.5997868264515072e+00 3.6362294681506615e+00 3.6869723882727103e+00 3.9239629196171744e+00 3.9460259590277142e+00 4.1576791814188834e+00 4.1640297740808085e+00 4.4333536008958623e+00 4.6494718592258035e+00 4.7622993632604977e+00 5.0045752857308434e+00 5.4886975306467383e+00 5.9727893780298311e+00 6.2826548131202875e+00 6.2900941742848033e+00 6.3851819576109810e+00 6.3900737887882144e+00 6.5556449291984826e+00 6.5919113145882662e+00 6.6593011743616506e+00 6.8086339893727201e+00 7.1135743887350200e+00 7.6305345806109468e+00 1.1915583960831951e+01 ";
rdft_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",dip,101,130);

Etot=-7.6373075513150312e+01;
dip=7.3219601972570314e-01;
Eorb="-1.8739126814643665e+01 -9.1600895865126808e-01 -4.7171059736526111e-01 -3.2520907358864859e-01 -2.4836428498766563e-01 7.6607870493656102e-03 8.0195561633317369e-02 3.3765707626325314e-01 3.7907205523275889e-01 4.6475514065445261e-01 5.4445465752367617e-01 5.9025710414018195e-01 5.9656889527833112e-01 6.4383458585354980e-01 7.4393042954989663e-01 8.8483045977478747e-01 9.7358970482260643e-01 1.2408271248304577e+00 1.2603336061284516e+00 1.6997135381477306e+00 1.7259867353942753e+00 1.7615401362822705e+00 1.8252283301120753e+00 1.8990803900546718e+00 2.1867269152086282e+00 2.5297910294693131e+00 2.9877393581567029e+00 3.1248910361673179e+00 3.1891270330365700e+00 3.2803037095595928e+00 3.2903012521084953e+00 3.4371043190980570e+00 3.5077386593645050e+00 3.5697001246967064e+00 3.5996624039699077e+00 3.6360140974303419e+00 3.6867898005951383e+00 3.9239664512658510e+00 3.9459447580933706e+00 4.1576745778165902e+00 4.1639702277768009e+00 4.4332753515828385e+00 4.6494301433248770e+00 4.7622322483851978e+00 5.0044217953725134e+00 5.4886686012985315e+00 5.9727879408942544e+00 6.2825985317478983e+00 6.2899895055011852e+00 6.3850646577478862e+00 6.3900573380084680e+00 6.5556350102648899e+00 6.5918074997358316e+00 6.6593421555316761e+00 6.8086240730657490e+00 7.1136080641996413e+00 7.6303066787149190e+00 1.1915469569441468e+01 ";
rdft_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",dip,101,130);
rdft_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",dip,101,130);

Etot=-7.6374648621893485e+01;
dip=7.3271967473888966e-01;
Eorb="-1.8741542806358083e+01 -9.1787960148698466e-01 -4.7348530740531686e-01 -3.2745858328284672e-01 -2.5054922870352703e-01 2.7897009947078625e-03 7.8041137459767540e-02 3.2383343840970996e-01 3.5923979174723303e-01 4.5242194391541346e-01 5.1413415993981970e-01 5.7767374349981426e-01 5.8423657911074067e-01 6.4253364284879100e-01 6.5976605831627355e-01 7.4242007624458317e-01 9.7186950908432224e-01 1.1822082442881903e+00 1.2023227809890384e+00 1.5759176971920341e+00 1.6360777675466989e+00 1.6982205565947992e+00 1.7245170715262932e+00 1.8628731154152631e+00 1.9081340610465236e+00 2.1641867907063816e+00 2.3473129641924420e+00 2.7892739420522865e+00 3.0049314612838676e+00 3.0831687968618846e+00 3.1876658277062067e+00 3.2328093601073160e+00 3.4168944008787094e+00 3.4579170361801981e+00 3.5049980735866009e+00 3.5282496020511034e+00 3.5683150582362981e+00 3.5772259600583016e+00 3.8379018102668216e+00 3.9226527270189493e+00 4.0867004810572922e+00 4.0926885816278302e+00 4.3310830948896184e+00 4.4154606730799202e+00 4.4322653444552245e+00 4.6027456360423260e+00 5.1266098375268410e+00 5.2200924385789325e+00 5.4840615352446624e+00 6.1494589306481098e+00 6.2799836318742246e+00 6.2885828365962038e+00 6.3502582428175725e+00 6.4059399842873272e+00 6.4358009854224072e+00 6.6570180726329351e+00 6.7153053536281506e+00 6.7372429636854809e+00 6.9398798868408278e+00 7.3406794159374966e+00 8.2789339365448491e+00 8.3551996899385994e+00 9.3390721097988969e+00 1.4480083135670908e+01 1.5822311942407310e+01 ";
rdft_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",dip,101,130);

Etot=-5.6637319431552505e+03;
dip=4.1660090782244827e+00;
Eorb="-9.4985623404324394e+02 -1.4146635099122020e+02 -1.3119294655421876e+02 -1.3119287270611147e+02 -1.3119259496786984e+02 -2.7676547720402841e+01 -2.3232980821838634e+01 -2.3232722836101512e+01 -2.3231117121670440e+01 -1.6049049315200051e+01 -1.6049045556498967e+01 -1.6047827658242213e+01 -1.6047724027585080e+01 -1.6047713458707900e+01 -1.5604531097548763e+01 -1.5532249496359425e+01 -1.1296661973756294e+01 -1.1249243895093182e+01 -1.1232665103771085e+01 -4.3970789601403419e+00 -2.8992752335524243e+00 -2.8986598470609359e+00 -2.8951186992178997e+00 -1.4177740924272384e+00 -1.2312596778565208e+00 -1.0610694274888925e+00 -8.7645299195720050e-01 -8.5303383017589762e-01 -8.1305176260999268e-01 -7.2468345937522116e-01 -7.1752260893690478e-01 -7.1751287362360971e-01 -7.1453927096582881e-01 -7.1369023729064263e-01 -6.5649508087306740e-01 -6.5484508971221644e-01 -6.4819561364580236e-01 -6.1951446971103685e-01 -5.1149277612520905e-01 -4.5694083034690181e-01 -3.6925756183658126e-01 -1.8059223079898942e-01 6.9314373055344658e-02 7.4011358154239038e-02 1.1409014945265671e-01 1.4993230227407597e-01 1.8266978402172024e-01 1.9355783398487214e-01 2.1197841085286168e-01 2.5237135015405704e-01 2.7656209472406329e-01 2.8532363502078562e-01 3.0336608170787632e-01 3.3343210193517742e-01 3.3688908923283800e-01 3.9652956651816207e-01 4.2174259977195599e-01 5.4893795783299704e-01 5.6113636018611412e-01 6.8232569239580376e-01 8.8548528960108186e-01 9.2615820683911998e-01 9.2670940578916072e-01 9.6328468558478197e-01 9.8346702018784671e-01 9.9887404064082908e-01 1.0364505443043599e+00 1.0834412264707201e+00 1.0936564417373447e+00 1.1989337429099993e+00 1.2617670077947041e+00 1.2818433277043921e+00 1.3193949782406433e+00 1.3895935370179857e+00 1.4308892989337034e+00 1.4702798437804183e+00 1.4945329130005354e+00 1.5683750074167071e+00 1.5822512320218465e+00 1.6271531769229546e+00 1.6323132920882668e+00 1.6700777141869541e+00 1.7294530360141209e+00 1.8374560574555538e+00 1.9460156088860219e+00 1.9779608207087003e+00 2.0568938351850083e+00 2.2440133809086360e+00 2.9829355616526563e+00 3.0788481978082940e+00 5.2757403445528555e+00 2.1121787319104163e+02 ";
rhf_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G",dip);

Etot=-5.6676693771661421e+03;
dip=3.3439670027302162e+00;
Eorb="-9.3989978305709883e+02 -1.3767101235880884e+02 -1.2795640130744825e+02 -1.2795629720977534e+02 -1.2795609263231353e+02 -2.5984496595112077e+01 -2.1817716293166651e+01 -2.1817474431498340e+01 -2.1816391339016832e+01 -1.5092625881653708e+01 -1.5092613619339582e+01 -1.5091967996373837e+01 -1.5091822502695790e+01 -1.5091777759089554e+01 -1.4452050934482607e+01 -1.4376826253895450e+01 -1.0301664388325793e+01 -1.0271613224807075e+01 -1.0254342778343020e+01 -3.7636291234642720e+00 -2.4165879739805085e+00 -2.4159471295089396e+00 -2.4131569660895460e+00 -1.1211260957173994e+00 -9.6404302606318970e-01 -8.2563594417899366e-01 -7.0330466686439452e-01 -6.8267077638262985e-01 -6.4806972386762352e-01 -5.4367438569452486e-01 -5.3721118529917955e-01 -5.2553328911172814e-01 -5.2468729995429808e-01 -5.2443005642535812e-01 -5.1974384350902481e-01 -5.1950907039454519e-01 -5.1476942837690842e-01 -4.9177447949513681e-01 -3.8454478630564265e-01 -3.8028081816984233e-01 -3.2171062621790597e-01 -1.8694110980628684e-01 -5.9086339998533466e-02 -5.6817112547789442e-02 -4.6514235269079524e-02 -4.3790552586238062e-02 -1.6269943044053027e-02 -1.5551554017854316e-02 3.4348180130330398e-02 5.9149658629029066e-02 6.8898585584800554e-02 9.1044724397948390e-02 1.0204094474703973e-01 1.2425777224911136e-01 1.3808997980681514e-01 1.5836841539326971e-01 1.8698885250401062e-01 3.0351683271371693e-01 3.0990118427403840e-01 3.9883824618130337e-01 5.8813342675214242e-01 5.9391038681768871e-01 6.1271411858861025e-01 6.5246465393870745e-01 6.6253193136985844e-01 6.7302838334438098e-01 6.9271307200088572e-01 7.7761508604676055e-01 7.9855541502640415e-01 8.6965250456534360e-01 8.9908840536659707e-01 9.3737296392559966e-01 9.6540579064643850e-01 9.9998469482637009e-01 1.0476202604520013e+00 1.1414385022101405e+00 1.1555410364331966e+00 1.2222188133571497e+00 1.2400230254109847e+00 1.2657699052180140e+00 1.2729543595299466e+00 1.3149215034973667e+00 1.3519148347062178e+00 1.4869558019479454e+00 1.6023139406098947e+00 1.6213994266475538e+00 1.7041935595537991e+00 1.8438995330423285e+00 2.6031625605277999e+00 2.7024556903076737e+00 4.7145848804773802e+00 2.0804451029094577e+02 ";
rdft_test(cdcplx,b3_21G,dftcart_nofit,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",dip,402,0);

Etot=-4.6701321134007662e+02;
dip=5.6558463052292229e-01;
Eorb="-1.8627155384081266e+01 -9.8468883979559330e+00 -9.8000403126040876e+00 -9.7985030129442947e+00 -9.7964382288224154e+00 -9.7961807286093254e+00 -9.7961550293408717e+00 -9.7957612760979682e+00 -9.7955844316870841e+00 -9.7955253378228342e+00 -9.7902010868624156e+00 -9.4290461935754100e-01 -7.5219755434919366e-01 -7.3523264259267429e-01 -7.0339379495003351e-01 -6.7054043595509405e-01 -6.3196338273616259e-01 -5.8307141514691152e-01 -5.5339064139282768e-01 -5.4222274015653960e-01 -5.1586083086253109e-01 -5.0988718158866941e-01 -4.8254315577542595e-01 -4.3597789495357370e-01 -4.3184704205578883e-01 -4.1614323888114252e-01 -4.0900731412096486e-01 -4.0310300909258251e-01 -3.8729721699978392e-01 -3.7610092492012015e-01 -3.7081200544692589e-01 -3.4983203853285250e-01 -3.4595252879660598e-01 -3.3597079170016803e-01 -3.2455694631551935e-01 -3.2107059114127173e-01 -3.1361176616237624e-01 -2.9876327594203755e-01 -2.9369154379021178e-01 -2.9084626712752276e-01 -2.8883510373524718e-01 -2.8097010167244757e-01 -2.7633843643912653e-01 -2.6485318866918695e-01 -2.2609495559127085e-01 2.9806193504776469e-02 4.3502578359478614e-02 4.7425903525503334e-02 5.3710247675323092e-02 6.2798356970489064e-02 6.8714800248653465e-02 7.6075439156983754e-02 8.0543340835444163e-02 9.0741950976236274e-02 1.0571675380212146e-01 1.1202229749875897e-01 1.1622893690551395e-01 1.2071515604916412e-01 1.3140988318499075e-01 1.3642600386008621e-01 1.3882306112856418e-01 1.4089032713112823e-01 1.4347204416652132e-01 1.4890078813557270e-01 1.5392861576712463e-01 1.6458475993418770e-01 1.6999258068765480e-01 1.7439288002526973e-01 1.8134945723108287e-01 1.9191610015998725e-01 1.9907634157487733e-01 2.0636216937355031e-01 2.1896327825756370e-01 2.2823888332171538e-01 2.4034506412358131e-01 2.4854950616822252e-01 2.5464806693901576e-01 4.2493271058579213e-01 4.2850886036985592e-01 4.4064410673972737e-01 4.5736194585786288e-01 4.6331708732589549e-01 4.6892492720818701e-01 4.7902752246481706e-01 4.8942009562723943e-01 5.0243422219506706e-01 5.1194298072644451e-01 5.1904520795686671e-01 5.3632788891284289e-01 5.4651472740219431e-01 5.7350453564290660e-01 5.9413007676845753e-01 6.0273922261531476e-01 6.0614929539337614e-01 6.1191796798582365e-01 6.1545221847709430e-01 6.3140381182749539e-01 6.4954779333860402e-01 6.7503495455919926e-01 6.8567421382145455e-01 6.9676129772653228e-01 7.1482468473032812e-01 7.2341293410448804e-01 7.4573470761597604e-01 7.5077683343567214e-01 7.6086260505263703e-01 7.7072141477158906e-01 7.7609700188666852e-01 7.8186561602926674e-01 7.9605246749726832e-01 8.0732894918493814e-01 8.1699563675079001e-01 8.2988574673292514e-01 8.3747020143152529e-01 8.3872073323423835e-01 8.4332466024242370e-01 8.4899137192798368e-01 8.5771909639379840e-01 8.6351982172063124e-01 8.6889346786697286e-01 8.7768101581839919e-01 8.8613576168752539e-01 8.9531622505229758e-01 9.0580268396883268e-01 9.1529394696026845e-01 9.2211729675857201e-01 9.4122869150406863e-01 9.6566335916909496e-01 9.7153228146920134e-01 9.8203184838341617e-01 1.0177357287046964e+00 1.0490679055687577e+00 1.0974974185440014e+00 1.1473859301827838e+00 1.1642856513068853e+00 1.2116505176540435e+00 1.2321588920350728e+00 1.2665602319410143e+00 1.2725658267906883e+00 1.3173851695157819e+00 1.3344567223701835e+00 1.3696815759719549e+00 1.4032274261173063e+00 1.4066934565555864e+00 1.4522678805877283e+00 1.4859434698094978e+00 1.4994164825263736e+00 1.5182768894457344e+00 1.5407584259063927e+00 1.5551907031565255e+00 1.5718427982244367e+00 1.5854147648810688e+00 1.6035526841992898e+00 1.6248643005588146e+00 1.6295953897725399e+00 1.6386264172140363e+00 1.6518537220841929e+00 1.6708603668113917e+00 1.7082441377175162e+00 1.7240858694253498e+00 1.7309955654811142e+00 1.7768180303879373e+00 1.7799398232982888e+00 1.7966929382751382e+00 1.7986813212398562e+00 1.8208975450700491e+00 1.8371991727279466e+00 1.8486132041326688e+00 1.8627496353199442e+00 1.8684090743011355e+00 1.8910556492541404e+00 1.9068268272806574e+00 1.9273963918054939e+00 1.9366440451386604e+00 1.9517986893450212e+00 1.9711843625392234e+00 1.9748131916718210e+00 1.9784538521553483e+00 2.0029272570769772e+00 2.0163942535614345e+00 2.0242113328828850e+00 2.0282111350807734e+00 2.0446483741293959e+00 2.0506332742719486e+00 2.0622352410335418e+00 2.0764523144145715e+00 2.0982714698906451e+00 2.1124504077066808e+00 2.1473840444179420e+00 2.1546265289422069e+00 2.1669072488053254e+00 2.1723423263883141e+00 2.1811756007732037e+00 2.1987631388348792e+00 2.2110770707786491e+00 2.2189960019821653e+00 2.2523875271341267e+00 2.2601866815485465e+00 2.2680782162689370e+00 2.2959486229381589e+00 2.3105472768134789e+00 2.3159943431577830e+00 2.3268618460567545e+00 2.3486729967275268e+00 2.3828964184055712e+00 2.3876850312431994e+00 2.4069231530607604e+00 2.4220203044487669e+00 2.4322426941379622e+00 2.4627677971980599e+00 2.4929212126317530e+00 2.5133876324148887e+00 2.5312878463291733e+00 2.5380551602408938e+00 2.5674710852611566e+00 2.5816439131100832e+00 2.5894765270472435e+00 2.6092017876142028e+00 2.6302178354678420e+00 2.6355319088825766e+00 2.6434882768074273e+00 2.6604219000078926e+00 2.6727747140860716e+00 2.6917618219416641e+00 2.6952996959532638e+00 2.7073942136304412e+00 2.7113426268735568e+00 2.7285878295968842e+00 2.7487932280965532e+00 2.7749378221916428e+00 2.7823800337461022e+00 2.7848198295151345e+00 2.7958607818729657e+00 2.8014816873707455e+00 2.8080213294769791e+00 2.8118708854908481e+00 2.8150036089415145e+00 2.8202128424248634e+00 2.8419229229885672e+00 2.8601397999967055e+00 2.8723562382583192e+00 2.9059069953942451e+00 3.0865699426818507e+00 3.1217730407734394e+00 3.1464491121701319e+00 3.1676576391695188e+00 3.1811710454162441e+00 3.1906302981159396e+00 3.1946229031233759e+00 3.2130861533891086e+00 3.2280261756725905e+00 3.2455312684423872e+00 3.2648008262514363e+00 3.2857850481463777e+00 3.3101638867056451e+00 3.3528868359917379e+00 3.3823782135658513e+00 3.3896321435388437e+00 3.3912089354674242e+00 3.4239218201494217e+00 3.4639873580775853e+00 3.4735422031987890e+00 3.4830952418428467e+00 3.4844489523185636e+00 3.8198817222848227e+00 4.1566323092946940e+00 4.2134192370906813e+00 4.2755955542131270e+00 4.3733051126841529e+00 4.4240197377789912e+00 4.4487034809326520e+00 4.5000880332566506e+00 4.5662886441039880e+00 4.6275751514939598e+00 4.7059770497940701e+00 ";
rdft_test(decanol,b6_31Gpp,dftcart,Etot,Eorb,"1-decanol, SVWN(RPA)/6-31G**",dip,1,8);


#ifndef COMPUTE_REFERENCE
  printf("****** Tests completed in %s *******\n",t.elapsed().c_str());
#endif

  return 0;
}
