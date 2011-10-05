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

/// DFT grid tolerance
const double dft_initialtol=1e-3;
const double dft_finaltol=5e-5;

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
  SCF solver=SCF(bas,set);
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
  SCF solver=SCF(bas,set);
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
  SCF solver=SCF(bas,set);
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
  SCF solver=SCF(bas,set);

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
  SCF solver=SCF(bas,set);
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
  // Load basis sets

  // Redirect stderr to file, since scf routines print out info there.
#ifdef COMPUTE_REFERENCE
  FILE *outstream=freopen("errors.log","w",stderr);
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

  // Oxygen atom
  std::vector<atom_t> O;
  at.el="O"; at.x=0.0; at.y=0.0; at.z=0.0; O.push_back(convert_to_bohr(at));

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
  pol.set_bool("UseADIIS",0);
  //pol.set_bool("UseDIIS",0);
  //  pol.set_bool("UseBroyden",1);

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

  // Oxygen, HF, cc-pVDZ
  Etot=-7.4665278726428056e+01;
  dip=1.5485884796707439e-15;
  Eorb="-2.0699668429305746e+01 -1.2524344055704906e+00 -5.7227047496319350e-01 -5.7227047496318983e-01 9.5299818088128430e-03 1.1361646554023428e+00 1.1361646554023428e+00 1.1867354030039940e+00 1.3655979286585465e+00 2.8659324494165808e+00 2.8750733910794426e+00 2.8750733910794448e+00 2.9026327547131290e+00 2.9026327547131339e+00 ";
  rhf_test(O,cc_pVDZ,sph,Etot,Eorb,"Oxygen, HF/cc-pVDZ",dip);
  
  // Same thing, but in cartesian basis
  Etot=-7.4665341652249595e+01;
  dip=6.5398992291784997e-16;
  Eorb="-2.0700150573071568e+01 -1.2525283341059998e+00 -5.7239952104172087e-01 -5.7239952104171865e-01 9.4002487921881692e-03 1.1360412695063438e+00 1.1360412695063511e+00 1.1866110121148032e+00 1.2042501265596652e+00 2.8654339936256386e+00 2.8749240213220477e+00 2.8749240213220530e+00 2.9024847901001474e+00 2.9024847901001509e+00 5.4458845250115502e+00 ";
  rhf_test(O,cc_pVDZ,cart,Etot,Eorb,"Oxygen, HF/cc-pVDZ cart",dip);
  
  // Oxygen, HF, cc-pVTZ
  Etot=-7.4684268595655197e+01;
  dip=1.6336193207140594e-15;
  Eorb="-2.0701080241863945e+01 -1.2592536718365432e+00 -5.8238193028096241e-01 -5.8237976944939795e-01 -6.0716129946690508e-03 7.4747716854730561e-01 7.4750984841719470e-01 7.8549089599699462e-01 9.1573641975757558e-01 1.6768284861807219e+00 1.6827254426294571e+00 1.6827274046147616e+00 1.7063059550513668e+00 1.7063418698574662e+00 3.9740741994128257e+00 4.0001706032281570e+00 4.0005472213160225e+00 5.4497687068984009e+00 5.4515097606359886e+00 5.4729085084936759e+00 5.4843566232934222e+00 5.4858452885125439e+00 5.5615099921020086e+00 5.5620783634976521e+00 6.5734653124073992e+00 6.5921325403296338e+00 6.5921354155241705e+00 6.6507732252354943e+00 6.6507900055106202e+00 7.5068008274033309e+00 ";
  rhf_test(O,cc_pVTZ,sph,Etot,Eorb,"Oxygen, HF/cc-pVTZ",dip);

  Etot=-7.4684356892528882e+01;
  dip=2.4626402583529788e-15;
  Eorb="-2.0701536775058809e+01 -1.2595137436165238e+00 -5.8257302456041660e-01 -5.8257302456040994e-01 -7.0450704776825902e-03 5.4358665530308359e-01 7.0312654470985325e-01 7.0312654470987179e-01 7.4062039970009297e-01 1.6753602293710117e+00 1.6825186080030616e+00 1.6825186080030672e+00 1.7061332545246279e+00 1.7061332545246337e+00 2.4722680700878610e+00 3.2663392552319257e+00 3.2743962771753798e+00 3.2743962771756436e+00 5.4346154442115315e+00 5.4470028292994535e+00 5.4470028292994597e+00 5.4840798363016248e+00 5.4840798363016408e+00 5.5459896157797566e+00 5.5459896157797726e+00 6.5734771192420238e+00 6.5918390642532643e+00 6.5918390642532687e+00 6.6504852351698602e+00 6.6504852351698727e+00 6.9515361997236553e+00 7.0434585291730478e+00 7.0434585291737521e+00 1.0235626663273584e+01 1.6315304721718302e+01 ";
  rhf_test(O,cc_pVTZ,cart,Etot,Eorb,"Oxygen, HF/cc-pVTZ cart",dip);

  // Oxygen, HF, cc-pVQZ
  Etot=-7.4689672843416247e+01;
  dip=3.2649956478791751e-15;
  Eorb="-2.0703820850652416e+01 -1.2613757860610233e+00 -5.8515820401525842e-01 -5.8515626168638679e-01 -1.2773114139443339e-02 5.5661622961439550e-01 5.5663058127836662e-01 5.8788521374436087e-01 5.9191007928788542e-01 1.1933085501962044e+00 1.1986398251507469e+00 1.1986451506247371e+00 1.2201229067780728e+00 1.2201232718269319e+00 2.5890955863581229e+00 2.6097064684889455e+00 2.6102977629201303e+00 3.3389558405156956e+00 3.3417713785367633e+00 3.3458769614753376e+00 3.3704177935649602e+00 3.3709100750466483e+00 3.4202391774223204e+00 3.4204631723940282e+00 3.5414323467402253e+00 4.0175914867490059e+00 4.0312297366452894e+00 4.0312788499335932e+00 4.0787434762931287e+00 4.0787659226596311e+00 9.0267558311359579e+00 9.2013564861642596e+00 9.2041799223646059e+00 9.2333845757742203e+00 9.2345715194220883e+00 9.2746991382936539e+00 9.2778221862902619e+00 9.3070095349844983e+00 9.3371211629771622e+00 1.0169404116113526e+01 1.0238681349281997e+01 1.0239379903715387e+01 1.1336991523005540e+01 1.1341555839089402e+01 1.1366111978452995e+01 1.1379197954245200e+01 1.1383715767166285e+01 1.1478055521749118e+01 1.1479846090152098e+01 1.2352317702769591e+01 1.2377654559953479e+01 1.2377672308733279e+01 1.2455029147334740e+01 1.2455354377885225e+01 4.0522642408279978e+01 ";
  rhf_test(O,cc_pVQZ,sph,Etot,Eorb,"Oxygen, HF/cc-pVQZ",dip);

  Etot=-7.4689715828296229e+01;
  dip=7.0639676788223686e-15;
  Eorb="-2.0703952880794414e+01 -1.2614956234962647e+00 -5.8522312023719447e-01 -5.8522312023718270e-01 -1.3897555557295202e-02 3.6164281566372314e-01 4.8359788068299114e-01 4.8359788068349535e-01 5.1482865923517329e-01 1.1906629384523224e+00 1.1978440993058239e+00 1.1978440993058435e+00 1.2193416487839022e+00 1.2193416487839137e+00 1.5341864690374116e+00 1.8147913232440758e+00 1.8169278302116920e+00 1.8169278302149063e+00 3.3312137446012406e+00 3.3411101457889334e+00 3.3411101457889525e+00 3.3703143933466282e+00 3.3703143933466584e+00 3.4181364641976621e+00 3.4181364641976772e+00 3.9072895327631274e+00 3.9099510692021449e+00 3.9536552618026781e+00 3.9536552618060137e+00 4.0155445818244342e+00 4.0294090489955527e+00 4.0294090489955785e+00 4.0769950895750586e+00 4.0769950895750791e+00 8.6106719250285515e+00 9.1940185016134421e+00 9.2029174766099349e+00 9.2029174766099686e+00 9.2296300625493544e+00 9.2296300625493721e+00 9.2742052898610456e+00 9.2742052898610652e+00 9.3367360511824504e+00 9.3367360511824860e+00 1.0716113434992034e+01 1.0791985584452901e+01 1.0791985584453323e+01 1.1321009934360756e+01 1.1335435691746344e+01 1.1335435691746392e+01 1.1378849397013628e+01 1.1378849397013676e+01 1.1455413425303213e+01 1.1455413425303238e+01 1.1989400103296067e+01 1.2016038688125050e+01 1.2016038688125763e+01 1.2098734998645313e+01 1.2098734998646725e+01 1.9042245039421747e+01 2.1033406519935131e+01 2.1058049747096206e+01 2.1058049747130351e+01 2.1141460827702613e+01 2.1141460827753402e+01 2.9093638233881713e+01 2.9208807542291503e+01 2.9208807542304289e+01 3.5633212270346213e+01 1.1545321426691261e+02 ";
  rhf_test(O,cc_pVQZ,cart,Etot,Eorb,"Oxygen, HF/cc-pVQZ cart",dip);

  // Chlorine, UHF, cc-pVTZ
  printf("\n");
  Etot=-4.5944650895295348e+02;
  dip=3.1928333236769229e-15;
  Eorba="-1.0487192732495824e+02 -1.0607731456916534e+01 -8.0945436505354653e+00 -8.0689331675307958e+00 -8.0689331675307781e+00 -1.1333591389956215e+00 -5.7594515800480972e-01 -5.0142033910584072e-01 -5.0142033910583150e-01 5.0038031102824110e-01 5.6758173166975023e-01 6.0901878014412858e-01 6.0901878014416266e-01 1.0467140711056362e+00 1.0627766420431100e+00 1.0627766420431282e+00 1.1177569478757050e+00 1.1177569478757194e+00 ";
  Eorbb="-1.0486051335060348e+02 -1.0596350522410871e+01 -8.0629607486960353e+00 -8.0629607486960229e+00 -8.0461999518581084e+00 -1.0112316861757684e+00 -4.7613922577244949e-01 -4.7613922577244805e-01 -3.7645620135944476e-02 5.2464097650703956e-01 6.1785774580383201e-01 6.1785774580384689e-01 6.6235446209432691e-01 1.1278252059342930e+00 1.1278252059342986e+00 1.1604788425159802e+00 1.1604788425159864e+00 1.1734238950977001e+00 ";
  uhf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, HF/6-31G** polarized",dip);

  Etot=-4.5944278096727612e+02;
  dip=7.0090932824453482e-16;
  Eorba="-1.0487302831574709e+02 -1.0608495662325550e+01 -8.0951291015495777e+00 -8.0696309800107802e+00 -8.0696309800107731e+00 -1.1298658044427727e+00 -5.6816410791988337e-01 -5.0428469389427422e-01 -5.0428469389427233e-01 5.0036388892020800e-01 5.6967746969977351e-01 6.0847085717999871e-01 6.0847085718001703e-01 1.0477281890196122e+00 1.0645036184690844e+00 1.0645036184691088e+00 1.1152515027775189e+00 1.1152515027775249e+00 ";
  Eorbb="-1.0486248004895859e+02 -1.0598189498391704e+01 -8.0647978854622870e+00 -8.0647978854622764e+00 -8.0481964071440277e+00 -1.0119816525363763e+00 -4.7452860595709051e-01 -4.7452860595708729e-01 -4.4311873785400233e-02 5.2403725517101241e-01 6.1759764253060523e-01 6.1759764253061722e-01 6.5850375934815786e-01 1.1291203963033460e+00 1.1291203963033514e+00 1.1573546698285953e+00 1.1573546698286017e+00 1.1669379293461675e+00 ";
  rohf_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, ROHF/6-31G**",dip);


  // Polarized calculation
  Etot=-4.6013019223191941e+02;
  dip=2.3962741402637484e-15;
  Eorba="-1.0170623791272868e+02 -9.6270662826494711e+00 -7.4215076416483248e+00 -7.3996667087414103e+00 -7.3996666727794542e+00 -8.8859992510415953e-01 -4.5634331389943239e-01 -4.0418355532833150e-01 -4.0418351168250466e-01 2.5059971789131175e-01 3.1405361695712558e-01 3.4410439171939605e-01 3.4410448009785694e-01 7.4272144749599700e-01 7.5561747993506567e-01 7.5561751983895209e-01 7.9852268986681385e-01 7.9852269056944303e-01 ";
  Eorbb="-1.0170198876430835e+02 -9.6221172071317103e+00 -7.4057928719978250e+00 -7.3958859923334241e+00 -7.3958859823844865e+00 -8.4902814482539635e-01 -3.8947786681371932e-01 -3.8947722187956629e-01 -3.3073282781126112e-01 2.5640444527682960e-01 3.2911710473175754e-01 3.4938941034189586e-01 3.4939124508323255e-01 7.8952672672295898e-01 7.9789820787481780e-01 7.9790126784876736e-01 8.1007372883211459e-01 8.1007392066009898e-01 ";
  udft_test(Cl,b6_31Gpp,dftpol_nofit,Etot,Eorba,Eorbb,"Chlorine, B3LYP/6-31G** polarized",dip,402,0);

  printf("\n");

  // Hydrogen molecule
  Etot=-1.1287000934442002e+00;
  dip=1.3589652627408088e-15;
  Eorb="-5.9241098912997037e-01 1.9744005747008161e-01 4.7932104727503055e-01 9.3732369228734513e-01 1.2929037097205383e+00 1.2929037097205385e+00 1.9570226089461711e+00 2.0435200542857599e+00 2.0435200542857648e+00 3.6104742345559311e+00 ";
  rhf_test(H2,cc_pVDZ,sph,Etot,Eorb,"Hydrogen molecule, HF/cc-pVDZ",dip);

  Etot=-1.1676136201569758e+00;
  dip=5.3341051865077310e-14;
  Eorb="-3.9201366065784427e-01 3.6521249130926178e-02 2.9071950323739204e-01 6.5833748123070401e-01 9.7502260955131670e-01 9.7502260955132081e-01 1.6066114003824421e+00 1.7001805168332509e+00 1.7001805168332547e+00 3.1926407112280497e+00 ";
  rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, SVWN(RPA)/cc-pVDZ",dip,1,8);

  Etot=-1.1603963207432755e+00;
  dip=4.7454134530222485e-14;
  Eorb="-3.7849076504350060e-01 5.3524954821963011e-02 3.0277496332444026e-01 6.6375297922053600e-01 9.9246470651630048e-01 9.9246470651630514e-01 1.6235395220308566e+00 1.7198888927811349e+00 1.7198888927811393e+00 3.2019339351714096e+00 ";
  rdft_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, PBEPBE/cc-pVDZ",dip,101,130);

  printf("\n");

  // Water
  Etot=-7.6056846941492140e+01;
  dip=7.9468053450274523e-01;
  Eorb="-2.0555336512446136e+01 -1.3429167382449509e+00 -7.0831770497817836e-01 -5.7581634784361757e-01 -5.0395808010501919e-01 1.4187676681805728e-01 2.0351837389142008e-01 5.4325209722579593e-01 5.9743073024741433e-01 6.6902610468331047e-01 7.8618190724898895e-01 8.0264561074431795e-01 8.0477457545814735e-01 8.5895459484528003e-01 9.5696003337862734e-01 1.1369012661203641e+00 1.1927526949086305e+00 1.5238457534432934e+00 1.5568958780903333e+00 2.0323642887776714e+00 2.0594264891496969e+00 2.0653811314862094e+00 2.1684100388512082e+00 2.2356007641932392e+00 2.5930811495622179e+00 2.9554689164925163e+00 3.3611156348999587e+00 3.4906525484143147e+00 3.5741552469566278e+00 3.6466195186854216e+00 3.8098890733441348e+00 3.8721537664289571e+00 3.8807784184569889e+00 3.9569185336285515e+00 3.9987665691432910e+00 4.0736543382360759e+00 4.1801785743668392e+00 4.3092425523822966e+00 4.3760461746849284e+00 4.5715947004936837e+00 4.6348116455461890e+00 4.8596765202504564e+00 5.1402288784775960e+00 5.2874806303149482e+00 5.5463372022724071e+00 6.0424050918362262e+00 6.5330036855353182e+00 6.9016693509017975e+00 6.9365453839677347e+00 6.9802319116329183e+00 7.0182977359694316e+00 7.1338792390135071e+00 7.2154670800292067e+00 7.2255943806766592e+00 7.4235576483656631e+00 7.7211430674544292e+00 8.2697507603910339e+00 1.2755307282278480e+01 ";
  rhf_test(h2o,cc_pVTZ,sph,Etot,Eorb,"Water, HF/cc-pVTZ",dip);
  // Direct calculation
  rhf_test(h2o,cc_pVTZ,direct,Etot,Eorb,"Water, HF/cc-pVTZ direct",dip);

  Etot=-7.6064486129654838e+01;
  dip=7.8763344405962654e-01;
  Eorb="-2.0560367083092903e+01 -1.3467279536137002e+00 -7.1288511158692414e-01 -5.8000887468733786e-01 -5.0760164793373264e-01 1.1671779600428910e-01 1.7059110220397972e-01 4.4885066155701464e-01 4.6188036162704088e-01 4.9872307793883419e-01 5.8265741625145184e-01 6.0652132051897722e-01 6.1372824648728630e-01 6.5415548179807159e-01 7.1859272859498169e-01 8.5096442991343868e-01 9.1889568268979871e-01 1.1089440961277794e+00 1.1547897331683232e+00 1.3489353418272723e+00 1.4145430590140939e+00 1.4772771468759238e+00 1.4849254809415808e+00 1.5824421714321399e+00 1.6814168751177621e+00 1.9064203228266428e+00 2.0719426300783361e+00 2.1991315397094970e+00 2.2841432969724553e+00 2.3586502555174431e+00 2.4266159687902111e+00 2.4848059352820600e+00 2.5282276242594972e+00 2.5586983036486903e+00 2.5784363651669544e+00 2.6480496048814701e+00 2.6695678016983715e+00 2.8427253105515375e+00 2.8775605714523635e+00 3.0425617303083397e+00 3.1259879443802183e+00 3.2920350663505715e+00 3.3395021140770988e+00 3.4423186484868329e+00 3.6243324555565524e+00 3.7991421505641907e+00 3.9963385467594508e+00 4.1262788588488650e+00 4.1837260617254266e+00 4.2120469596144465e+00 4.4530091250453978e+00 4.4836444670704969e+00 4.6832751543114606e+00 4.7273379567989053e+00 4.7606790975031990e+00 4.9144446799323287e+00 5.3663102757703021e+00 5.4263411178004466e+00 5.9901140326394104e+00 6.0936990086036786e+00 6.2358276619589414e+00 6.3038740282453940e+00 6.6880102727757196e+00 6.7937504113140408e+00 7.0711800492880750e+00 7.2310189210479345e+00 7.2927776955341477e+00 7.3308237099446893e+00 7.3646486773308748e+00 7.5296481610317709e+00 7.6177264392664865e+00 7.7259762409076949e+00 8.0066864828459483e+00 8.0888021505559919e+00 8.1005754462990360e+00 8.1132910708336787e+00 8.1864355223336620e+00 8.2671470066514239e+00 8.3145007761742509e+00 8.3183957301830258e+00 8.4098910016661357e+00 8.5910748720605064e+00 8.9095636051860403e+00 8.9339142553758482e+00 8.9932612646958088e+00 9.1554548089163639e+00 9.2899792425610173e+00 9.3532959336521859e+00 9.9081699825475109e+00 1.0058748519553284e+01 1.0262787788640669e+01 1.0435357074135831e+01 1.0575253932755590e+01 1.0631852883011510e+01 1.0758111738428751e+01 1.1259613032249154e+01 1.1410287474206084e+01 1.1573640510087980e+01 1.1667889987332583e+01 1.1716060752594750e+01 1.1846399338484458e+01 1.2192549639801085e+01 1.2296711081761133e+01 1.2419184470419435e+01 1.2441159737784739e+01 1.2467586337468990e+01 1.3576020920514928e+01 1.3762676884748108e+01 1.4186349630642237e+01 1.4557092141149957e+01 1.4720169034630636e+01 1.4865456870908915e+01 1.6429257206289083e+01 1.6870043609924409e+01 4.4590591579992576e+01 ";
  rhf_test(h2o,cc_pVQZ,sph,Etot,Eorb,"Water, HF/cc-pVQZ",dip);
  // Direct calculation should yield same energies
  rhf_test(h2o,cc_pVQZ,direct,Etot,Eorb,"Water, HF/cc-pVQZ direct",dip);


  Etot=-7.6373040009500770e+01;
  dip=7.3243773848080218e-01;
  Eorb="-1.8739146525693975e+01 -9.1595065597589542e-01 -4.7162284405333393e-01 -3.2513104405574061e-01 -2.4828714200439703e-01 8.4276128445114010e-03 8.1122977366280144e-02 3.3805164054406173e-01 3.7944648406266185e-01 4.6504842295281890e-01 5.4462136743131939e-01 5.9069528371916213e-01 5.9681283986707245e-01 6.4396655512979972e-01 7.4405281784716026e-01 8.8494888818218331e-01 9.7376011862068412e-01 1.2410363479392439e+00 1.2604540810485065e+00 1.6998172345465494e+00 1.7260935783940679e+00 1.7619284021055013e+00 1.8253710949927187e+00 1.8994331383844969e+00 2.1868858210209599e+00 2.5299527223858291e+00 2.9877813026023081e+00 3.1249811502112261e+00 3.1891644380140449e+00 3.2804949383455866e+00 3.2904628564588512e+00 3.4371365948765060e+00 3.5077601951324304e+00 3.5697562870657911e+00 3.5997943202057714e+00 3.6362619221314998e+00 3.6870227572294509e+00 3.9239594681597749e+00 3.9460276797462828e+00 4.1576872075676645e+00 4.1640517413974703e+00 4.4333440047457131e+00 4.6494746194233709e+00 4.7623256364669606e+00 5.0045959007370726e+00 5.4886730319968384e+00 5.9727737547176449e+00 6.2826198041846535e+00 6.2900360499917349e+00 6.3851200642384311e+00 6.3900587200860288e+00 6.5556519291461282e+00 6.5918883465913147e+00 6.6592868900688256e+00 6.8085957269625634e+00 7.1135678783311809e+00 7.6305909909586340e+00 1.1915705878437169e+01 ";
  rdft_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",dip,101,130);

  Etot=-7.6373104131216692e+01;
  dip=7.3218073203806877e-01;
  Eorb="-1.8739261438776342e+01 -9.1603186237185552e-01 -4.7170881420058908e-01 -3.2520888758987687e-01 -2.4836269417359266e-01 7.6579312387476534e-03 8.0194729434885689e-02 3.3765515273050628e-01 3.7907016814873334e-01 4.6476139458514681e-01 5.4445711009545872e-01 5.9026224707203057e-01 5.9656738414842325e-01 6.4383362376583542e-01 7.4392590690340554e-01 8.8481795863434809e-01 9.7358990617001895e-01 1.2408269503776459e+00 1.2603266134855655e+00 1.6997031190770033e+00 1.7259764000784652e+00 1.7615385245936064e+00 1.8252090584530640e+00 1.8990756276149874e+00 2.1867265602539963e+00 2.5297959530748182e+00 2.9877384881165576e+00 3.1248874741192081e+00 3.1891256411864704e+00 3.2803037255061289e+00 3.2903581461037668e+00 3.4371064320396600e+00 3.5077382970071054e+00 3.5696983368929436e+00 3.5996698911137774e+00 3.6360465563065278e+00 3.6868401653792708e+00 3.9239629919172554e+00 3.9459464906949639e+00 4.1576825914162576e+00 4.1639921885859357e+00 4.4332657478223476e+00 4.6494329060292099e+00 4.7622585152630101e+00 5.0044424268431271e+00 5.4886441055534370e+00 5.9727723223625810e+00 6.2825634654563229e+00 6.2899313617765005e+00 6.3850027373829494e+00 6.3900423211101423e+00 6.5556420187762425e+00 6.5917845379824929e+00 6.6593278813246917e+00 6.8085858043463823e+00 7.1136015570031867e+00 7.6303630674229925e+00 1.1915591504391699e+01 ";
  rdft_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",dip,101,130);
  // This should also give the same energies
  rdft_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",dip,101,130);  

  Etot=-7.6374677307848643e+01;
  dip=7.3270504143624482e-01;
  Eorb="-1.8741666893928429e+01 -9.1790290316294820e-01 -4.7348415912142705e-01 -3.2745957242391310e-01 -2.5054880361686738e-01 2.7876125466591473e-03 7.8040086378478729e-02 3.2383112174569850e-01 3.5923673950966140e-01 4.5242306017652040e-01 5.1412624401441476e-01 5.7766754520628372e-01 5.8424025024115245e-01 6.4253359834079782e-01 6.5974120849671625e-01 7.4241304776397665e-01 9.7186888943375793e-01 1.1822037411877988e+00 1.2023200108379661e+00 1.5759088395858551e+00 1.6360767195364074e+00 1.6982101163244860e+00 1.7245074761683494e+00 1.8628528872030010e+00 1.9081245211343227e+00 2.1641861462034160e+00 2.3473067502656701e+00 2.7893216684377840e+00 3.0049300409812623e+00 3.0831639213612134e+00 3.1876646754386697e+00 3.2328104451780968e+00 3.4169088729957089e+00 3.4579477186008849e+00 3.5050274089952889e+00 3.5282602598814883e+00 3.5683135416562348e+00 3.5772243290908379e+00 3.8378984399703153e+00 3.9226495504935941e+00 4.0867147723272881e+00 4.0926848574676189e+00 4.3310798407195445e+00 4.4154577012445317e+00 4.4322762577386063e+00 4.6027416845369054e+00 5.1266039075906997e+00 5.2200769943536338e+00 5.4840374147540762e+00 6.1494454235288689e+00 6.2799558526560837e+00 6.2885259462743823e+00 6.3502454303291795e+00 6.4058759390252540e+00 6.4358349324376514e+00 6.6570016411612789e+00 6.7152835832869533e+00 6.7372127758238163e+00 6.9398690736349584e+00 7.3406543149374350e+00 8.2789554883406264e+00 8.3551813327367945e+00 9.3390509783959867e+00 1.4480073175258985e+01 1.5822735165759308e+01 ";
  rdft_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",dip,101,130);

  printf("\n");
  Etot=-5.6637319431552542e+03;
  dip=4.1660090619805059e+00;
  Eorb="-9.4985623404724640e+02 -1.4146635099481537e+02 -1.3119294655790947e+02 -1.3119287270980476e+02 -1.3119259497155753e+02 -2.7676547723307586e+01 -2.3232980824791177e+01 -2.3232722839059900e+01 -2.3231117124621701e+01 -1.6049049318195593e+01 -1.6049045559494157e+01 -1.6047827661237889e+01 -1.6047724030585371e+01 -1.6047713461698791e+01 -1.5604531096175771e+01 -1.5532249495339656e+01 -1.1296661975779381e+01 -1.1249243893110558e+01 -1.1232665103576455e+01 -4.3970789623268498e+00 -2.8992752356968392e+00 -2.8986598491756888e+00 -2.8951187013127626e+00 -1.4177740920679858e+00 -1.2312596773475406e+00 -1.0610694272332577e+00 -8.7645299171191615e-01 -8.5303382986580056e-01 -8.1305176302358773e-01 -7.2468346087412472e-01 -7.1752261085064628e-01 -7.1751287553314735e-01 -7.1453927281242291e-01 -7.1369023915374641e-01 -6.5649508080456487e-01 -6.5484508950431364e-01 -6.4819561332649445e-01 -6.1951446998987547e-01 -5.1149277555337280e-01 -4.5694083032695021e-01 -3.6925756150072292e-01 -1.8059223164171517e-01 6.9314372645446695e-02 7.4011357792932359e-02 1.1409014903119051e-01 1.4993230304771982e-01 1.8266978390123689e-01 1.9355783526532955e-01 2.1197841084614730e-01 2.5237134980523557e-01 2.7656209454278885e-01 2.8532363500255348e-01 3.0336608143599647e-01 3.3343210153511638e-01 3.3688908925909111e-01 3.9652956711664084e-01 4.2174259963986660e-01 5.4893795783883570e-01 5.6113636035921555e-01 6.8232569332609239e-01 8.8548528956161354e-01 9.2615820687312378e-01 9.2670940567220006e-01 9.6328468546665924e-01 9.8346702031833988e-01 9.9887404085382081e-01 1.0364505452793376e+00 1.0834412263203919e+00 1.0936564413194840e+00 1.1989337425113284e+00 1.2617670086600381e+00 1.2818433273954613e+00 1.3193949782396717e+00 1.3895935373116759e+00 1.4308892993946829e+00 1.4702798441260581e+00 1.4945329125555009e+00 1.5683750074308811e+00 1.5822512314622954e+00 1.6271531755736810e+00 1.6323132908024358e+00 1.6700777143042629e+00 1.7294530352661555e+00 1.8374560573356020e+00 1.9460156096804855e+00 1.9779608208018149e+00 2.0568938354858832e+00 2.2440133806995215e+00 2.9829355623500020e+00 3.0788481987182283e+00 5.2757403427864542e+00 2.1121787318755844e+02 ";
  rhf_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G",dip);

  Etot=-5.6676693040108521e+03;
  dip=3.3440241655134439e+00;
  Eorb="-9.3989949630777653e+02 -1.3767085566884194e+02 -1.2795628115848666e+02 -1.2795617703962785e+02 -1.2795597246351552e+02 -2.5984513230392327e+01 -2.1817821867966586e+01 -2.1817579989651602e+01 -2.1816497141585881e+01 -1.5092697750898502e+01 -1.5092685475195312e+01 -1.5092039961894534e+01 -1.5091894480680356e+01 -1.5091849706585810e+01 -1.4451997411414936e+01 -1.4376772096004634e+01 -1.0301646276571871e+01 -1.0271591424733591e+01 -1.0254324778758805e+01 -3.7636150617162669e+00 -2.4165974821986707e+00 -2.4159566370877319e+00 -2.4131649118380043e+00 -1.1211327312539410e+00 -9.6404423700905140e-01 -8.2564383266872376e-01 -7.0330635825666288e-01 -6.8267276098233920e-01 -6.4807232963941908e-01 -5.4367335440912368e-01 -5.3720703164543648e-01 -5.2553076265084553e-01 -5.2468512038316606e-01 -5.2442769409718004e-01 -5.1974327080402816e-01 -5.1950507258617451e-01 -5.1476736328884964e-01 -4.9177264492963846e-01 -3.8454257590911212e-01 -3.8028019254527695e-01 -3.2170840064505718e-01 -1.8693874986382264e-01 -5.9081491633645879e-02 -5.6813400532874973e-02 -4.6509529933826808e-02 -4.3796957420039834e-02 -1.6268007322999919e-02 -1.5559973979324449e-02 3.4351142983196609e-02 5.9147172501492623e-02 6.8897615577252588e-02 9.1042104072634578e-02 1.0205943465032943e-01 1.2425801181611794e-01 1.3807721356573649e-01 1.5836554692570415e-01 1.8698799354523893e-01 3.0351791252041793e-01 3.0990400106102722e-01 3.9883575135214178e-01 5.8814212059025350e-01 5.9392870388159724e-01 6.1272091412542262e-01 6.5247360340961869e-01 6.6254328091868164e-01 6.7303431267175606e-01 6.9271694904580372e-01 7.7762481970709441e-01 7.9855435035795252e-01 8.6965866139509196e-01 8.9909697373911457e-01 9.3737136134920651e-01 9.6540094972240853e-01 9.9998288887657705e-01 1.0476134865257727e+00 1.1414373172392103e+00 1.1555321083025583e+00 1.2222114665422974e+00 1.2400112735313158e+00 1.2657747893799614e+00 1.2729558315267144e+00 1.3149191529077382e+00 1.3519184036423351e+00 1.4869465150991110e+00 1.6022996827170741e+00 1.6213851499170890e+00 1.7041812321525001e+00 1.8439074177811228e+00 2.6031529393680573e+00 2.7024442958321506e+00 4.7145830216863231e+00 2.0804442116265670e+02 ";
  rdft_test(cdcplx,b3_21G,dftcart_nofit,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",dip,402,0);

  printf("\n");
  Etot=-4.6701334076811759e+02;
  dip=5.6554045414292597e-01;
  Eorb="-1.8627160618579857e+01 -9.8468913225497907e+00 -9.8000444995770337e+00 -9.7984994023896483e+00 -9.7964421810294624e+00 -9.7961832486110776e+00 -9.7961575410898032e+00 -9.7957638797653406e+00 -9.7955871058619355e+00 -9.7955290707145419e+00 -9.7901930268443422e+00 -9.4290382125178995e-01 -7.5220423770661227e-01 -7.3523835454190090e-01 -7.0339560800953027e-01 -6.7054475348342391e-01 -6.3196604708554860e-01 -5.8307347307891277e-01 -5.5339073142974393e-01 -5.4222347900014767e-01 -5.1586286549118954e-01 -5.0988816284001692e-01 -4.8254124104179963e-01 -4.3597527860546864e-01 -4.3185246270427824e-01 -4.1614586353246885e-01 -4.0901313649111221e-01 -4.0310405023892848e-01 -3.8729796952892548e-01 -3.7610221514194236e-01 -3.7081714137483074e-01 -3.4983450170469360e-01 -3.4595424881009995e-01 -3.3597616048471413e-01 -3.2455716592894673e-01 -3.2107118208099167e-01 -3.1361284918795801e-01 -2.9876392741220825e-01 -2.9369246085466189e-01 -2.9084628480198921e-01 -2.8883667205108543e-01 -2.8097182575457730e-01 -2.7633927025036792e-01 -2.6485460106769293e-01 -2.2609632806373114e-01 2.9806269344380406e-02 4.3513650367088277e-02 4.7432025536249402e-02 5.3708161390239643e-02 6.2799573739704184e-02 6.8711854740801978e-02 7.6072842314436784e-02 8.0543588749803940e-02 9.0740587137554216e-02 1.0571249207625363e-01 1.1202749158537995e-01 1.1623555619972997e-01 1.2071977551948453e-01 1.3140537232939617e-01 1.3642960468215548e-01 1.3881991651168135e-01 1.4088835007125391e-01 1.4347481890774244e-01 1.4890175933193187e-01 1.5393183364427507e-01 1.6458410849968069e-01 1.7000640041268841e-01 1.7439066765879821e-01 1.8134332134479716e-01 1.9192207586801288e-01 1.9908172104566643e-01 2.0636476817206503e-01 2.1897290489887597e-01 2.2824324944455437e-01 2.4035131863817782e-01 2.4855206090316906e-01 2.5464930872902708e-01 4.2494632417760148e-01 4.2851670083892629e-01 4.4064005578504556e-01 4.5736515611467921e-01 4.6333624857924055e-01 4.6892488548559680e-01 4.7903433807396811e-01 4.8942994454684546e-01 5.0244608425564374e-01 5.1195186633798184e-01 5.1905123144654219e-01 5.3633582914298894e-01 5.4652898950651974e-01 5.7351232724728851e-01 5.9413717744822991e-01 6.0274026584641294e-01 6.0615698945095930e-01 6.1191819679498149e-01 6.1544675107058211e-01 6.3139801598747536e-01 6.4954256480018857e-01 6.7503754855642573e-01 6.8566986752039805e-01 6.9676173259717111e-01 7.1482706700159848e-01 7.2342081643846723e-01 7.4572907415223366e-01 7.5078024860734816e-01 7.6086405353374087e-01 7.7072417055495224e-01 7.7610480309271412e-01 7.8186593431584583e-01 7.9605209548835421e-01 8.0733181828422329e-01 8.1699733928969942e-01 8.2988488342651967e-01 8.3746954857779565e-01 8.3872015210598294e-01 8.4332343588174530e-01 8.4898952610275391e-01 8.5771801108715706e-01 8.6351539870014893e-01 8.6889062799420891e-01 8.7767937557317321e-01 8.8612146901482136e-01 8.9531850394028745e-01 9.0580188753675406e-01 9.1529622197973204e-01 9.2211552016569021e-01 9.4122671982881567e-01 9.6566442732217261e-01 9.7153233956427132e-01 9.8202947574376120e-01 1.0177340869517977e+00 1.0490647393333790e+00 1.0974968663904356e+00 1.1473850765003775e+00 1.1642819843872845e+00 1.2116493287417176e+00 1.2321557709028614e+00 1.2665573825001251e+00 1.2725642082198882e+00 1.3173837452423711e+00 1.3344538361184934e+00 1.3696810404173876e+00 1.4032291855855110e+00 1.4066970512336281e+00 1.4522692408851694e+00 1.4859490103184807e+00 1.4994197352291139e+00 1.5182876491384976e+00 1.5407638830930812e+00 1.5551926765748403e+00 1.5718394985642383e+00 1.5854184608164086e+00 1.6035576901085817e+00 1.6248651982588473e+00 1.6295984046635825e+00 1.6386335427162928e+00 1.6518587809906651e+00 1.6708671503663652e+00 1.7082473974920513e+00 1.7241037716180572e+00 1.7310039954049297e+00 1.7768204602086948e+00 1.7799438398895646e+00 1.7966958429956399e+00 1.7986848911927458e+00 1.8208933535857927e+00 1.8372040482834400e+00 1.8486200322542545e+00 1.8627436485765863e+00 1.8684089652163440e+00 1.8910581892605929e+00 1.9068277499061539e+00 1.9273999918177549e+00 1.9366460591377772e+00 1.9518012795194517e+00 1.9711854952519661e+00 1.9748121118739372e+00 1.9784542106357592e+00 2.0029267669498636e+00 2.0163952627079693e+00 2.0242130832360439e+00 2.0282114395015274e+00 2.0446481718356710e+00 2.0506328943157168e+00 2.0622362579787166e+00 2.0764536620016285e+00 2.0982720962997643e+00 2.1124504928226564e+00 2.1473887707654349e+00 2.1546302415211351e+00 2.1669122198878550e+00 2.1723465360679022e+00 2.1811765086415345e+00 2.1987684165944898e+00 2.2110817616585630e+00 2.2190091205617395e+00 2.2523935845290075e+00 2.2601863732449230e+00 2.2680783135507490e+00 2.2959485112011824e+00 2.3105491811003045e+00 2.3159969825042284e+00 2.3268617106622353e+00 2.3486712830980894e+00 2.3828964663910881e+00 2.3876851123818672e+00 2.4069237877026950e+00 2.4220219813004755e+00 2.4322430956049477e+00 2.4627687685107809e+00 2.4929227058747516e+00 2.5133868433523898e+00 2.5312881599903041e+00 2.5380548228491984e+00 2.5674717233448248e+00 2.5816431572859160e+00 2.5894774396793143e+00 2.6092032218949588e+00 2.6302194965401116e+00 2.6355319264408990e+00 2.6434918130177629e+00 2.6604238249178209e+00 2.6727736109178113e+00 2.6917624189988612e+00 2.6953016329145489e+00 2.7073953832892088e+00 2.7113431228161624e+00 2.7285906263504867e+00 2.7487933889129117e+00 2.7749411869195271e+00 2.7823829336140706e+00 2.7848202772994401e+00 2.7958613463120439e+00 2.8014837067103251e+00 2.8080227107125597e+00 2.8118739930958982e+00 2.8150044326037338e+00 2.8202144957458373e+00 2.8419189785367012e+00 2.8601404074896992e+00 2.8723583002379467e+00 2.9059083834614627e+00 3.0865723015587911e+00 3.1217752947387161e+00 3.1464509345805283e+00 3.1676601274444569e+00 3.1811730776939235e+00 3.1906320275858437e+00 3.1946232055315660e+00 3.2130852013420688e+00 3.2280245974215838e+00 3.2455276291118258e+00 3.2647992521600089e+00 3.2857800041065297e+00 3.3101565347945590e+00 3.3528837952236104e+00 3.3823732023908675e+00 3.3896336504841638e+00 3.3912070172848545e+00 3.4239186078881620e+00 3.4639853231504261e+00 3.4735407254663522e+00 3.4830943212562526e+00 3.4844479190687689e+00 3.8198758587280714e+00 4.1566357404952532e+00 4.2134282511937204e+00 4.2756045924079435e+00 4.3733092979332211e+00 4.4240297038448109e+00 4.4487183777125736e+00 4.5001091249726777e+00 4.5663108131926871e+00 4.6276072342650059e+00 4.7060107846949073e+00 ";
 rdft_test(decanol,b6_31Gpp,dftcart,Etot,Eorb,"1-decanol, SVWN(RPA)/6-31G**",dip,1,8);


#ifndef COMPUTE_REFERENCE
  printf("****** Tests completed in %s *******\n",t.elapsed().c_str());
#endif

  return 0;
}
