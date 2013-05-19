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

#include "checkpoint.h"
#include "basislibrary.h"
#include "global.h"
#include "mathf.h"
#include "scf.h"
#include "solidharmonics.h"
#include "timer.h"
#include "xyzutils.h"
#include "emd/emd.h"
#include "emd/emd_gto.h"

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

/// Test indices
void testind() {
  for(int am=0;am<max_am;am++) {
    int idx=0;
    for(int ii=0;ii<=am;ii++)
      for(int jj=0;jj<=ii;jj++) {
	int l=am-ii;
	int m=ii-jj;
	int n=jj;

	int ind=getind(l,m,n);
	if(ind!=idx) {
	  ERROR_INFO();
	  printf("l=%i, m=%i, n=%i, ind=%i, idx=%i.\n",l,m,n,ind,idx);
	  throw std::runtime_error("Indexing error.\n");
	}

	idx++;
      }
  }

  printf("Indices OK.\n");
}

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


/// Test restricted solution
#ifdef COMPUTE_REFERENCE
#define restr_test(at,baslib,set,Etot,Eorb,label,dipmom,emd) restr_test_run(at,baslib,set,Etot,Eorb,label,dipmom,emd); printf(#set ".set_string(\"Method\",\"%s\");\n",set.get_string("Method").c_str()); printf("restr_test(" #at "," #baslib "," #set "," #Etot "," #Eorb "," #label "," #dipmom "," #emd ");\n\n");
#else
#define restr_test(at,baslib,set,Etot,Eorb,label,dipmom,emd) restr_test_run(at,baslib,set,Etot,Eorb,label,dipmom,emd);
#endif
void restr_test_run(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, Settings set, double Etot, const arma::vec & Eorb, const std::string & label, double dipmom, bool doemd) {
  Timer t;

#ifndef COMPUTE_REFERENCE
  printf("%s, ",label.c_str());
  fflush(stdout);
#endif

  // Construct the basis set
  BasisSet bas=construct_basis(at,baslib,set);
  // Check normalization of basis
  check_norm(bas);

  // Temporary file name
  char *tmpfile=tempnam("./",".chk");
  set.set_string("SaveChk",tmpfile);

  // Run the calculation
  calculate(bas,set);

  // Density matrix
  arma::mat P;
  // The orbital energies
  arma::vec Eo;
  // and the total energy
  energy_t en;
  // and the amount of electrons
  int Nel;
  {
    // Open the checkpoint
    Checkpoint chkpt(tmpfile,false);

    // Load everything necessary
    chkpt.read("P",P);
    chkpt.read("E",Eo);
    chkpt.read(en);
    chkpt.read("Nel",Nel);
  }

  // Get rid of the temporary file
  remove(tmpfile);
  free(tmpfile);

  // Compute norm of density matrix
  arma::mat S=bas.overlap();
  double Nelnum=arma::trace(P*S);
  if(fabs(Nelnum-Nel)>1e-7) {
    bas.print(true);
    P.print("P");
    S.print("S");
    fprintf(stderr,"Nelnum=%e, Nel=%i, Nelnum-Nel=%e\n",Nelnum,Nel,Nelnum-Nel);
    throw std::runtime_error("Problem with density matrix.\n");
  }

  // Compute dipole moment
  double dip=dip_mom(P,bas);

  // Check EMD
  bool emdok=true;
  if(doemd) {
    GaussianEMDEvaluator eval(bas,P);
    EMD emd(&eval, Nel);
    emd.initial_fill(false);
    emd.find_electrons(false);
    emd.optimize_moments(false);

    // Get moments
    arma::mat mom=emd.moments();

    // Compare <p^2> with T and <p^0> with tr(P*S)
    emdok=((fabs(mom(4,1)-2.0*en.Ekin)<=10*mom(4,2)) && (fabs(mom(2,1)-arma::trace(P*S))<=10*mom(2,2)));

    if(!emdok) {
      printf("<p^2> = %e, 2Ekin=%e, diff %e vs %e\n",mom(4,1),2.0*en.Ekin,mom(4,1)-2.0*en.Ekin,mom(4,2));
      printf("<p^0> = %e, Tr(PS)=%e, diff %e vs %e\n",mom(2,1),arma::trace(P*S),mom(2,1)-arma::trace(P*S),mom(2,2));
    }
  }

#ifdef COMPUTE_REFERENCE
  if(!set.get_bool("Direct")) {
    printf("Etot=%.16e;\n",en.E);
    printf("dip=%.16e;\n",dip);
    printf("Eorb=\"");
    for(size_t i=0;i<Eo.n_elem;i++)
      printf("%.16e ",Eo(i));
    printf("\";\n");
  }

  // Dummy statements
  (void) emdok;
  (void) Etot;
  (void) Eorb;
  (void) label;
  (void) dipmom;

#else
  // Compare results
  bool Eok=1, Dok=1, ok=1;
  size_t nsucc=0, nfail=0;
  compare(Eo,Eorb,otol,nsucc,nfail); // Compare orbital energies
  Eok=rel_compare(en.E,Etot,tol); // Compare total energies
  Dok=abs_compare(dip,dipmom,dtol); // Compare dipole moments
  ok=(Eok && Dok && emdok);
  printf("E=%f %s, dp=%f %s, orbital energies %i ok, %i failed (%s)\n",en.E,stat[Eok],dip,stat[Dok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, difference in dipole moment is %e.\nMaximum difference of orbital energy is %e.",rel_diff(en.E,Etot),dip-dipmom,max_diff(Eo,Eorb));
  if(doemd) {
    if(emdok)
      printf(" EMD ok.\n");
    else
      printf(" EMD failed.\n");
  } else
    printf("\n");

  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    fflush(stdout);
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }
#endif

  fflush(stdout);
}

/// Test unrestricted solution
#ifdef COMPUTE_REFERENCE
#define unrestr_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,emd) unrestr_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,emd);  printf(#set ".set_string(\"Method\",\"%s\");\n",set.get_string("Method").c_str()); printf("unrestr_test(" #at "," #baslib "," #set "," #Etot "," #Eorba "," #Eorbb "," #label "," #dipmom "," #emd ");\n\n");
#else
#define unrestr_test(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,emd) unrestr_test_run(at,baslib,set,Etot,Eorba,Eorbb,label,dipmom,emd);
#endif
void unrestr_test_run(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, Settings set, double Etot, const arma::vec & Eorba, const arma::vec & Eorbb, const std::string & label, double dipmom, bool doemd) {
  Timer t;

#ifndef COMPUTE_REFERENCE
  printf("%s, ",label.c_str());
  fflush(stdout);
#endif

  // Construct basis set
  BasisSet bas=construct_basis(at,baslib,set);
  // Check normalization of basis
  check_norm(bas);

  // Temporary file name
  char *tmpfile=tempnam("./",".chk");
  set.set_string("SaveChk",tmpfile);

  // Run the calculation
  calculate(bas,set);
  // Density matrix
  arma::mat P;
  // The orbital energies
  arma::vec Eao, Ebo;
  // and the total energy
  energy_t en;
  // and the amount of electrons
  int Nel;
  {
    // Open the checkpoint
    Checkpoint chkpt(tmpfile,false);

    // Load everything necessary
    chkpt.read("P",P);
    chkpt.read("Ea",Eao);
    chkpt.read("Eb",Ebo);
    chkpt.read(en);
    chkpt.read("Nel",Nel);
  }

  // Get rid of the temporary file
  remove(tmpfile);
  free(tmpfile);

  // Compute dipole moment
  double dip=dip_mom(P,bas);

  // Check EMD
  bool emdok=true;
  if(doemd) {
    arma::mat S=bas.overlap();
    GaussianEMDEvaluator eval(bas,P);
    EMD emd(&eval, Nel);
    emd.initial_fill(false);
    emd.find_electrons(false);
    emd.optimize_moments(false);

    // Get moments
    arma::mat mom=emd.moments();

    // Compare <p^2> with T and <p^0> with tr(P*S)
    emdok=((fabs(mom(4,1)-2.0*en.Ekin)<=10*mom(4,2)) && (fabs(mom(2,1)-arma::trace(P*S))<=10*mom(2,2)));
  }

#ifdef COMPUTE_REFERENCE
  if(!set.get_bool("Direct")) {
    printf("Etot=%.16e;\n",en.E);
    printf("dip=%.16e;\n",dip);
    printf("Eorba=\"");
    for(size_t i=0;i<Eao.n_elem;i++)
      printf("%.16e ",Eao(i));
    printf("\";\n");
    printf("Eorbb=\"");
    for(size_t i=0;i<Ebo.n_elem;i++)
      printf("%.16e ",Ebo(i));
    printf("\";\n");
  }

  // Dummy statements
  (void) emdok;
  (void) Etot;
  (void) Eorba;
  (void) Eorbb;
  (void) label;
  (void) dipmom;

#else
  // Compare results
  bool Eok=1, Dok=1, ok=1;
  size_t nsucca=0, nfaila=0;
  size_t nsuccb=0, nfailb=0;
  compare(Eao,Eorba,otol,nsucca,nfaila); // Compare orbital energies
  compare(Ebo,Eorbb,otol,nsuccb,nfailb); // Compare orbital energies
  size_t nsucc=nsucca+nsuccb;
  size_t nfail=nfaila+nfailb;

  Eok=rel_compare(en.E,Etot,tol); // Compare total energies
  Dok=abs_compare(dip,dipmom,dtol); // Compare dipole moments
  ok=(Eok && Dok && emdok);
  printf("E=%f %s, dp=%f %s, orbital energies %i ok, %i failed (%s)\n",en.E,stat[Eok],dip,stat[Dok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, difference in dipole moment is %e.\nMaximum differences of orbital energies are %e and %e.",rel_diff(en.E,Etot),dip-dipmom,max_diff(Eao,Eorba),max_diff(Ebo,Eorbb));
  if(doemd) {
    if(emdok)
      printf(" EMD ok.\n");
    else
      printf(" EMD failed.\n");
  } else
    printf("\n");

  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    fflush(stdout);
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }
#endif

  fflush(stdout);
}


/// Run unit tests by comparing calculations to ones that should be OK
int main(void) {
  testind();

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
  sph.add_string("SaveChk","Save checkpoint to","erkale.chk");
  sph.add_string("LoadChk","Load checkpoint","");
  sph.add_bool("FreezeCore","Freeze the cores of the atoms",false);
  sph.add_bool("ForcePol","Force polarized calculation?",false);
  sph.set_bool("Verbose",false);
  // Use core guess and no density fitting for tests.
  sph.set_string("Guess","Core");
  sph.set_bool("DensityFitting",false);

  // No spherical harmonics
  Settings cart(sph);
  cart.set_bool("UseLM",false);

  // Direct calculation
  Settings direct(sph);
  direct.set_bool("Direct",true);

  // Polarized calculation
  Settings pol(sph);
  pol.set_int("Multiplicity",2);

  // DFT tests

  // Settings for DFT
  Settings dftsph(sph); // Normal settings
  dftsph.add_dft_settings();
  dftsph.set_bool("DensityFitting",true);

  Settings dftcart(cart); // Cartesian basis
  dftcart.add_dft_settings();
  dftcart.set_bool("DensityFitting",true);

  Settings dftnofit(dftsph); // No density fitting
  dftnofit.set_bool("DensityFitting",false);

  Settings dftcart_nofit(dftcart);
  dftcart_nofit.set_bool("DensityFitting",false);

  Settings dftdirect(dftsph); // Direct calculation
  dftdirect.set_bool("Direct",true);

  Settings dftpol(pol); // Polarized calculation
  dftpol.add_dft_settings();
  dftpol.set_double("DFTInitialTol",1e-4);

  Settings dftpol_nofit(dftpol); // Polarized calculation, no density fitting
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

  Etot=-1.2848877555174084e+02;
  dip=6.3191957996789011e-16;
  Eorb="-3.2765635418543845e+01 -1.9187982340080592e+00 -8.3209725198901119e-01 -8.3209725198900986e-01 -8.3209725198900442e-01 1.6945577282756135e+00 1.6945577282756150e+00 1.6945577282756175e+00 2.1594249508272716e+00 5.1967114014381295e+00 5.1967114014381321e+00 5.1967114014381330e+00 5.1967114014381348e+00 5.1967114014381348e+00 ";
  sph.set_string("Method","HF");
  restr_test(Ne,cc_pVDZ,sph,Etot,Eorb,"Neon, HF/cc-pVDZ",dip,true);

  Etot=-1.2848886617203718e+02;
  dip=1.1957740115954232e-16;
  Eorb="-3.2765398101180708e+01 -1.9190105008574396e+00 -8.3228111468368138e-01 -8.3228111468367993e-01 -8.3228111468367849e-01 1.6944246415940818e+00 1.6944246415940882e+00 1.6944246415940916e+00 1.9905991489168089e+00 5.1964251810756945e+00 5.1964251810757078e+00 5.1964251810757087e+00 5.1964251810757087e+00 5.1964251810757141e+00 1.0383359360177629e+01 ";
  cart.set_string("Method","HF");
  restr_test(Ne,cc_pVDZ,cart,Etot,Eorb,"Neon, HF/cc-pVDZ cart",dip,false);

  Etot=-1.2853186163632120e+02;
  dip=4.0450442702803107e-15;
  Eorb="-3.2769110714725052e+01 -1.9270833040458379e+00 -8.4541551022178718e-01 -8.4541551022178441e-01 -8.4541551022178174e-01 1.0988680366258867e+00 1.0988680366258894e+00 1.0988680366258903e+00 1.4176388078878395e+00 2.8142175658874948e+00 2.8142175658874984e+00 2.8142175658875006e+00 2.8142175658875113e+00 2.8142175658875179e+00 6.1558667264988864e+00 6.1558667264988980e+00 6.1558667264988980e+00 9.6473695824492296e+00 9.6473695824492331e+00 9.6473695824492385e+00 9.6473695824492527e+00 9.6473695824492616e+00 9.6473695824492669e+00 9.6473695824492918e+00 1.1227312685587517e+01 1.1227312685587522e+01 1.1227312685587531e+01 1.1227312685587545e+01 1.1227312685587558e+01 1.1744558070520076e+01 ";
  sph.set_string("Method","HF");
  restr_test(Ne,cc_pVTZ,sph,Etot,Eorb,"Neon, HF/cc-pVTZ",dip,false);

  Etot=-1.2853200998517826e+02;
  dip=1.8580962204673266e-15;
  Eorb="-3.2769827644860229e+01 -1.9274545161945864e+00 -8.4572301688930818e-01 -8.4572301688930751e-01 -8.4572301688930385e-01 8.8038911847859358e-01 1.0282198402493341e+00 1.0282198402493665e+00 1.0282198402493723e+00 2.8138968467097238e+00 2.8138968467097318e+00 2.8138968467097691e+00 2.8138968467097811e+00 2.8138968467098033e+00 4.1362240326160489e+00 4.6398467069077487e+00 4.6398467069078695e+00 4.6398467069079103e+00 9.6470056633944754e+00 9.6470056633945145e+00 9.6470056633945234e+00 9.6470056633945305e+00 9.6470056633945376e+00 9.6470056633945411e+00 9.6470056633945536e+00 1.1226914497700383e+01 1.1226914497700392e+01 1.1226914497700401e+01 1.1226914497700408e+01 1.1226914497700415e+01 1.1317534801293887e+01 1.1317534801294105e+01 1.1317534801294258e+01 1.6394442678499036e+01 2.8816114658691834e+01 ";
  cart.set_string("Method","HF");
  restr_test(Ne,cc_pVTZ,cart,Etot,Eorb,"Neon, HF/cc-pVTZ cart",dip,false);

  Etot=-1.2854346965912177e+02;
  dip=8.2442858206770346e-16;
  Eorb="-3.2771496234564395e+01 -1.9293376379563942e+00 -8.4895896127267301e-01 -8.4895896127266124e-01 -8.4895896127265902e-01 8.0890413856904120e-01 8.0890413856904864e-01 8.0890413856905918e-01 9.3559988890065537e-01 1.9978112795881973e+00 1.9978112795881977e+00 1.9978112795882028e+00 1.9978112795882090e+00 1.9978112795882159e+00 3.9328189055243050e+00 3.9328189055243339e+00 3.9328189055243672e+00 5.8106845423858555e+00 5.9042211380533054e+00 5.9042211380533169e+00 5.9042211380533232e+00 5.9042211380533240e+00 5.9042211380533329e+00 5.9042211380533374e+00 5.9042211380533516e+00 6.7616951541199066e+00 6.7616951541199075e+00 6.7616951541199244e+00 6.7616951541199279e+00 6.7616951541199617e+00 1.4903626161762311e+01 1.4903626161762318e+01 1.4903626161762332e+01 1.4903626161762347e+01 1.4903626161762373e+01 1.4903626161762377e+01 1.4903626161762391e+01 1.4903626161762396e+01 1.4903626161762405e+01 1.5804420584518153e+01 1.5804420584518185e+01 1.5804420584518324e+01 1.9794585642608951e+01 1.9794585642608983e+01 1.9794585642608997e+01 1.9794585642609025e+01 1.9794585642609036e+01 1.9794585642609047e+01 1.9794585642609047e+01 2.0954549905127163e+01 2.0954549905127202e+01 2.0954549905127255e+01 2.0954549905127298e+01 2.0954549905127351e+01 6.6550956474323570e+01 ";
  sph.set_string("Method","HF");
  restr_test(Ne,cc_pVQZ,sph,Etot,Eorb,"Neon, HF/cc-pVQZ",dip,false);

  Etot=-1.2854353449722353e+02;
  dip=2.7081001936224192e-15;
  Eorb="-3.2771625129430866e+01 -1.9294942841516589e+00 -8.4906688460229296e-01 -8.4906688460227964e-01 -8.4906688460226887e-01 5.8690441469805377e-01 7.1271797749432564e-01 7.1271797749447208e-01 7.1271797749450816e-01 1.9879845921931765e+00 1.9879845921932036e+00 1.9879845921932195e+00 1.9879845921932215e+00 1.9879845921932324e+00 2.5105148502387213e+00 2.7214792301541828e+00 2.7214792301553241e+00 2.7214792301554565e+00 5.9040962888846851e+00 5.9040962888846957e+00 5.9040962888847011e+00 5.9040962888847144e+00 5.9040962888847242e+00 5.9040962888847330e+00 5.9040962888847499e+00 6.4115733390466758e+00 6.5684069303162635e+00 6.5684069303163728e+00 6.5684069303163843e+00 6.5684069303164749e+00 6.5684069303165051e+00 6.7659166001663928e+00 6.7659166001671966e+00 6.7659166001676834e+00 1.4004805313564866e+01 1.4903514354754087e+01 1.4903514354754181e+01 1.4903514354754192e+01 1.4903514354754217e+01 1.4903514354754266e+01 1.4903514354754275e+01 1.4903514354754286e+01 1.4903514354754304e+01 1.4903514354754350e+01 1.8145155385283601e+01 1.8145155385285364e+01 1.8145155385286433e+01 1.8145155385287133e+01 1.8145155385287520e+01 1.8540067452516094e+01 1.8540067452516379e+01 1.8540067452516784e+01 1.9794449045308635e+01 1.9794449045308689e+01 1.9794449045308699e+01 1.9794449045308706e+01 1.9794449045308721e+01 1.9794449045308752e+01 1.9794449045308799e+01 2.9727979556984007e+01 3.9089870736220654e+01 3.9089870736246866e+01 3.9089870736272644e+01 3.9089870736278165e+01 3.9089870736282180e+01 3.9551871550112850e+01 3.9551871550120545e+01 3.9551871550133903e+01 5.8376821811885058e+01 2.0568373997987064e+02 ";
  cart.set_string("Method","HF");
  restr_test(Ne,cc_pVQZ,cart,Etot,Eorb,"Neon, HF/cc-pVQZ cart",dip,true);

  Etot=-4.5932237447598095e+02;
  dip=2.9019424150427053e-15;
  Eorba="-1.0491252709199125e+02 -1.0631421029531081e+01 -8.1008786074406274e+00 -8.1008786074406185e+00 -8.1008786074406007e+00 -1.1461899103772701e+00 -5.3682606675849953e-01 -5.3682606675849698e-01 -5.3682606675847655e-01 7.0964827767708949e-01 7.0964827767709326e-01 7.0964827767709571e-01 7.7555571150375613e-01 9.1004757868316954e-01 9.1004757868317065e-01 9.1004757868318187e-01 9.1004757868318331e-01 9.1004757868318398e-01 ";
  Eorbb="-1.0489888357495002e+02 -1.0617440915882856e+01 -8.0767815326021726e+00 -8.0767815326021619e+00 -8.0767815326021513e+00 -1.0127979324413448e+00 -3.2803124521225097e-01 -3.2803124521224991e-01 -3.2803124521224897e-01 7.5256307097699338e-01 7.5256307097699859e-01 7.5256307097699882e-01 8.0348088556010211e-01 9.8202244061927013e-01 9.8202244061927801e-01 9.8202244061927935e-01 9.8202244061929256e-01 9.8202244061930000e-01 ";
  pol.set_string("Method","HF");
  unrestr_test(Cl,cc_pVDZ,pol,Etot,Eorba,Eorbb,"Chlorine, HF/cc-pVDZ polarized",dip,false);

  Etot=-4.5931897730353910e+02;
  dip=2.2894258458955431e-15;
  Eorba="-1.0492065248418804e+02 -1.0637484285900859e+01 -8.1058298895541796e+00 -8.1058298895541672e+00 -8.1058298895541530e+00 -1.1407130437783808e+00 -5.3178636944085045e-01 -5.3178636944084767e-01 -5.3178636944084634e-01 7.1164168190545574e-01 7.1164168190545862e-01 7.1164168190547172e-01 7.7272015171539910e-01 9.1065022945520735e-01 9.1065022945521235e-01 9.1065022945521479e-01 9.1065022945521601e-01 9.1065022945521856e-01 ";
  Eorbb="-1.0491112044133085e+02 -1.0627874488847446e+01 -8.0884126659043716e+00 -8.0884126659043591e+00 -8.0884126659043449e+00 -1.0253430496871596e+00 -3.3631614823633815e-01 -3.3631614823633554e-01 -3.3631614823633266e-01 7.4500082502070675e-01 7.4500082502071396e-01 7.4500082502071607e-01 8.0113058873578968e-01 9.7501562951396692e-01 9.7501562951397425e-01 9.7501562951397991e-01 9.7501562951398490e-01 9.7501562951398513e-01 ";
  pol.set_string("Method","ROHF");
  unrestr_test(Cl,cc_pVDZ,pol,Etot,Eorba,Eorbb,"Chlorine, ROHF/cc-pVDZ",dip,false);

  Etot=-4.6012860120990240e+02;
  dip=6.5727686679658322e-15;
  Eorba="-1.0159643069825238e+02 -9.5096541021047205e+00 -7.2678516281229051e+00 -7.2678516281228855e+00 -7.2678516281228811e+00 -8.4429448190652678e-01 -3.7028595312913104e-01 -3.7028595312912532e-01 -3.7028595312911183e-01 5.1686917235121377e-01 5.1686917235121643e-01 5.1686917235122987e-01 5.7513032232954231e-01 6.7958161582382526e-01 6.7958161582383547e-01 6.7958161582384380e-01 6.7958161582384613e-01 6.7958161582384702e-01 ";
  Eorbb="-1.0158948387043372e+02 -9.5029412874731225e+00 -7.2576908430225737e+00 -7.2576908430225515e+00 -7.2576908430225480e+00 -7.9358972241536263e-01 -3.0681784500526793e-01 -3.0681784500526721e-01 -3.0681784500526332e-01 5.3672596280617002e-01 5.3672596280617324e-01 5.3672596280617513e-01 5.9497680070872505e-01 7.2029670129687506e-01 7.2029670129688050e-01 7.2029670129688350e-01 7.2029670129688750e-01 7.2029670129688916e-01 ";
  dftpol_nofit.set_string("Method","hyb_gga_xc_b3lyp");
  unrestr_test(Cl,cc_pVDZ,dftpol_nofit,Etot,Eorba,Eorbb,"Chlorine, B3LYP/cc-pVDZ polarized",dip,false);

  Etot=-1.1287000934441989e+00;
  dip=9.2162168441805856e-16;
  Eorb="-5.9241098912717283e-01 1.9744005746927498e-01 4.7932104726975810e-01 9.3732369228505696e-01 1.2929037097182312e+00 1.2929037097182312e+00 1.9570226089437421e+00 2.0435200542832299e+00 2.0435200542832308e+00 3.6104742345529286e+00 ";
  sph.set_string("Method","HF");
  restr_test(H2,cc_pVDZ,sph,Etot,Eorb,"Hydrogen molecule, HF/cc-pVDZ",dip,false);

  Etot=-1.1676141294604032e+00;
  dip=5.6859192362448887e-14;
  Eorb="-3.9201515131319209e-01 3.6521080362270163e-02 2.9071322496817015e-01 6.5833098538585200e-01 9.7502316975325032e-01 9.7502316975325554e-01 1.6066122928283000e+00 1.7001809420471337e+00 1.7001809420471374e+00 3.1926496355617036e+00 ";
  dftsph.set_string("Method","lda_x-lda_c_vwn_rpa");
  restr_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, SVWN(RPA)/cc-pVDZ",dip,false);

  Etot=-1.1603962667280401e+00;
  dip=4.8274110164559418e-14;
  Eorb="-3.7849167468785799e-01 5.3523492272332530e-02 3.0277089209790559e-01 6.6374851097937648e-01 9.9246487779861337e-01 9.9246487779861836e-01 1.6235426412156309e+00 1.7198880924572555e+00 1.7198880924572595e+00 3.2019311002044435e+00 ";
  dftsph.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, PBEPBE/cc-pVDZ",dip,false);

  Etot=-7.6056825377225252e+01;
  dip=7.9472744664025285e-01;
  Eorb="-2.0555281282042621e+01 -1.3428635537362767e+00 -7.0828436707370579e-01 -5.7575384366640980e-01 -5.0391497946518526e-01 1.4187951595897991e-01 2.0351537962048122e-01 5.4324870457422358e-01 5.9753586795272495e-01 6.6949546804733273e-01 7.8747678901491536e-01 8.0274150259376886e-01 8.0481260838927693e-01 8.5898803427653603e-01 9.5702121816221897e-01 1.1344778956268646e+00 1.1928203506308623e+00 1.5241753072248003e+00 1.5579529875899483e+00 2.0324408703481227e+00 2.0594682933949504e+00 2.0654407667654406e+00 2.1686553971251299e+00 2.2363161874117430e+00 2.5909431284301432e+00 2.9581971201263584e+00 3.3610002634259324e+00 3.4914002757500708e+00 3.5741938473998349e+00 3.6463660411025263e+00 3.7977214225715725e+00 3.8739670207200199e+00 3.8824466781478200e+00 3.9569498252361384e+00 4.0199059056521573e+00 4.0760332617326842e+00 4.1862021921032850e+00 4.3092789386061208e+00 4.3875716397017808e+00 4.5640073763724383e+00 4.6817931187181339e+00 4.8550947814965264e+00 5.1380848619434252e+00 5.2500191192331043e+00 5.5275547773795655e+00 6.0402478807550120e+00 6.5453259404803603e+00 6.9113516636805281e+00 6.9366142673484505e+00 7.0003720401843355e+00 7.0078239260656270e+00 7.0609382582616016e+00 7.1598075634980702e+00 7.2256524677469338e+00 7.4561719768599959e+00 7.7799625501996488e+00 8.2653639983397404e+00 1.2804358857417395e+01 ";
  sph.set_string("Method","HF");
  restr_test(h2o,cc_pVTZ,sph,Etot,Eorb,"Water, HF/cc-pVTZ",dip,true);

  direct.set_string("Method","HF");
  restr_test(h2o,cc_pVTZ,direct,Etot,Eorb,"Water, HF/cc-pVTZ direct",dip,false);

  Etot=-7.6064480528902266e+01;
  dip=7.8765852143275583e-01;
  Eorb="-2.0560341502778531e+01 -1.3467109753410504e+00 -7.1286865725285009e-01 -5.7999183680365585e-01 -5.0759009717622539e-01 1.1677961566566232e-01 1.7061237209598409e-01 4.4878630937111402e-01 4.6240973962339532e-01 4.9860081760452651e-01 5.8389461420286259e-01 6.0602248882036258e-01 6.1386901002319849e-01 6.5509670177640444e-01 7.1940581316402608e-01 8.5193265612198898e-01 9.1760642694819017e-01 1.1091391701662416e+00 1.1559117510947330e+00 1.3479064825873261e+00 1.4144381939813520e+00 1.4776186083002847e+00 1.4856774062766740e+00 1.5814608126828418e+00 1.6854835694915613e+00 1.9096187711773898e+00 2.0727777006363111e+00 2.1976502949619636e+00 2.2888869953848694e+00 2.3588905626077934e+00 2.4246094512903618e+00 2.4837778778670776e+00 2.5224544333415424e+00 2.5800657938296654e+00 2.5803867641373381e+00 2.6507304485542562e+00 2.6683130227292824e+00 2.8407379930653396e+00 2.8643130295805381e+00 3.0412098271013805e+00 3.1190680411063720e+00 3.2889763498471307e+00 3.3518967282438314e+00 3.4467312433255999e+00 3.6214003251247471e+00 3.8285931938920621e+00 3.9968185290732627e+00 4.1278766572046930e+00 4.1879462991018732e+00 4.2176486548067551e+00 4.4343620851337029e+00 4.4925098762834974e+00 4.6832772384484675e+00 4.7403725766624696e+00 4.8079058183533103e+00 4.9140701184738873e+00 5.3503959728177888e+00 5.4039303345343193e+00 5.9860940812609691e+00 6.1030498775684796e+00 6.2449376388022086e+00 6.3029981791479699e+00 6.7000543774896446e+00 6.7926548533146960e+00 7.0589633214720635e+00 7.2683601221408791e+00 7.3171930239210070e+00 7.3671378799490981e+00 7.4371264981024199e+00 7.5184752942734736e+00 7.5458434070978866e+00 7.5694204731575150e+00 8.0046360183899452e+00 8.0708295728154233e+00 8.0987711859956431e+00 8.1338237889722613e+00 8.1523664179263715e+00 8.2695443402901105e+00 8.3150962688266077e+00 8.3485048840797411e+00 8.4164827896957242e+00 8.6181288279177295e+00 8.8336406067752780e+00 8.9048326501038453e+00 8.9437734416378714e+00 9.2166366992705626e+00 9.3761387892174692e+00 9.3791690841206563e+00 9.9423093828417972e+00 1.0035594100863809e+01 1.0257561208013197e+01 1.0425629819244005e+01 1.0646599814804917e+01 1.0757780264732423e+01 1.0806846315627814e+01 1.1272406006226191e+01 1.1390414016257084e+01 1.1595907188990871e+01 1.1644666354801172e+01 1.1693629515528702e+01 1.1844883870829925e+01 1.2158546419136957e+01 1.2320144570935765e+01 1.2398213786608304e+01 1.2413264229603016e+01 1.2465013632991969e+01 1.3602019533785365e+01 1.3763660836495754e+01 1.4247885728482006e+01 1.4614348058894169e+01 1.4639079459431741e+01 1.4826337833139203e+01 1.6435472092016276e+01 1.6799107180558408e+01 4.4322817445366631e+01 ";
  sph.set_string("Method","HF");
  restr_test(h2o,cc_pVQZ,sph,Etot,Eorb,"Water, HF/cc-pVQZ",dip,false);

  direct.set_string("Method","HF");
  restr_test(h2o,cc_pVQZ,direct,Etot,Eorb,"Water, HF/cc-pVQZ direct",dip,false);

  Etot=-7.6372961222395105e+01;
  dip=7.3249328280729165e-01;
  Eorb="-1.8738953501512679e+01 -9.1586785162753037e-01 -4.7160106589899620e-01 -3.2507287613094971e-01 -2.4816655857445680e-01 8.4534036267782161e-03 8.1125737520961611e-02 3.3806729297807764e-01 3.7949122775514627e-01 4.6548871040173651e-01 5.4539135465823385e-01 5.9072372577235854e-01 5.9687803067217471e-01 6.4398044404236865e-01 7.4418029766589633e-01 8.8306547474459707e-01 9.7381887521619070e-01 1.2412763516764236e+00 1.2611338609415526e+00 1.6999121835648221e+00 1.7261707906469292e+00 1.7619584545393498e+00 1.8256258288422624e+00 1.9002743849716250e+00 2.1846823312316639e+00 2.5326906159399063e+00 2.9875043516088855e+00 3.1260225297410185e+00 3.1892007455508677e+00 3.2799583723193377e+00 3.2859664655621859e+00 3.4399388077632351e+00 3.5114854019505684e+00 3.5697880009196870e+00 3.6175833251931930e+00 3.6363368817606583e+00 3.6947695819154647e+00 3.9240129133178043e+00 3.9512251176804631e+00 4.1557724959073044e+00 4.1932317708017361e+00 4.4292385921947766e+00 4.6459620146407561e+00 4.7303864407694407e+00 4.9898268736839073e+00 5.4868943652626241e+00 5.9838983452501582e+00 6.2843629405005936e+00 6.2901656603459815e+00 6.3781111568912996e+00 6.4202760176873284e+00 6.4811650993450813e+00 6.5329728518003405e+00 6.6594412404199472e+00 6.8404602024852181e+00 7.1724946503564206e+00 7.6259352319102387e+00 1.1962240167936494e+01 ";
  dftnofit.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",dip,false);

  Etot=-7.6373025076488872e+01;
  dip=7.3223619687875485e-01;
  Eorb="-1.8739067712785552e+01 -9.1594855273558817e-01 -4.7168658517962037e-01 -3.2515022234011071e-01 -2.4824134771120071e-01 7.6868694316258868e-03 8.0201485083603849e-02 3.3767224652041705e-01 3.7911611253651400e-01 4.6520448054228208e-01 5.4523065610005383e-01 5.9029285519174779e-01 5.9663403525992331e-01 6.4384844712932743e-01 7.4405492511078031e-01 8.8293500311146300e-01 9.7364987322558194e-01 1.2410683077857867e+00 1.2610076706292146e+00 1.6997990579830355e+00 1.7260539463664428e+00 1.7615703419248678e+00 1.8254642144178477e+00 1.8999177919378853e+00 2.1845241425542001e+00 2.5325349915415720e+00 2.9874617550487961e+00 3.1259288098730806e+00 3.1891620087522075e+00 3.2797685659271805e+00 3.2858678167537163e+00 3.4399087715591770e+00 3.5114590806630823e+00 3.5697300791389450e+00 3.6174584672066952e+00 3.6361205767157769e+00 3.6945923089440953e+00 3.9240163487240993e+00 3.9511494091589179e+00 4.1557167093090994e+00 4.1932269766325643e+00 4.4291609287368683e+00 4.6459391796764296e+00 4.7302938427362680e+00 4.9896645806172479e+00 5.4868660751963372e+00 5.9839044167325186e+00 6.2842935455016953e+00 6.2900613307461279e+00 6.3780017572022940e+00 6.4202853530423711e+00 6.4811643201759521e+00 6.5328557623566619e+00 6.6594818284423987e+00 6.8404421490481653e+00 7.1725297508538493e+00 7.6257028377015139e+00 1.1962121373955231e+01 ";
  dftsph.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",dip,false);

  dftdirect.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",dip,false);

  Etot=-7.6374645230136593e+01;
  dip=7.3272054999646929e-01;
  Eorb="-1.8741546114801416e+01 -9.1788065208198932e-01 -4.7348527540156676e-01 -3.2745824709354521e-01 -2.5054888296150196e-01 2.7901830544596663e-03 7.8041748636134778e-02 3.2383381111101078e-01 3.5924016484360816e-01 4.5242229923105004e-01 5.1413396046000970e-01 5.7767374228851276e-01 5.8423722555787660e-01 6.4253357514569631e-01 6.5976489676493655e-01 7.4241931809352668e-01 9.7186975918755969e-01 1.1822077426645410e+00 1.2023227295161900e+00 1.5759172313861667e+00 1.6360776272982627e+00 1.6982201437026592e+00 1.7245167623651108e+00 1.8628724330059621e+00 1.9081332377655740e+00 2.1641869722694231e+00 2.3473125251763642e+00 2.7892764468584805e+00 3.0049313704663860e+00 3.0831686635910072e+00 3.1876656925770970e+00 3.2328093756756293e+00 3.4168951292173810e+00 3.4579185918059512e+00 3.5049998737850245e+00 3.5282503323286494e+00 3.5683149815125246e+00 3.5772258007782538e+00 3.8379020258068985e+00 3.9226525885207644e+00 4.0867013276911681e+00 4.0926885575200016e+00 4.3310827329017627e+00 4.4154604410280074e+00 4.4322656550566544e+00 4.6027454835197803e+00 5.1266097225209011e+00 5.2200923676067168e+00 5.4840610597673303e+00 6.1494592260592933e+00 6.2799826595305204e+00 6.2885803768666371e+00 6.3502580499093790e+00 6.4059374497214510e+00 6.4358015216727491e+00 6.6570174354990499e+00 6.7153040590271287e+00 6.7372414808558796e+00 6.9398791763068193e+00 7.3406783795840296e+00 8.2789338893508830e+00 8.3551996988136867e+00 9.3390714650735482e+00 1.4480083318497456e+01 1.5822331894565211e+01 ";
  dftcart.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",dip,false);

  Etot=-5.6637319431552378e+03;
  dip=4.1660087967675050e+00;
  Eorb="-9.4985623399812744e+02 -1.4146635094929266e+02 -1.3119294651147109e+02 -1.3119287266341721e+02 -1.3119259492511748e+02 -2.7676547679805573e+01 -2.3232980780528159e+01 -2.3232722794961312e+01 -2.3231117080400512e+01 -1.6049049273427904e+01 -1.6049045514726380e+01 -1.6047827616560472e+01 -1.6047723985913308e+01 -1.6047713417459679e+01 -1.5604531124999619e+01 -1.5532249524598729e+01 -1.1296661986534080e+01 -1.1249243916440941e+01 -1.1232665102166420e+01 -4.3970789240547319e+00 -2.8992751952960827e+00 -2.8986598087306978e+00 -2.8951186618387261e+00 -1.4177741087342834e+00 -1.2312596936885631e+00 -1.0610694402548646e+00 -8.7645300687122252e-01 -8.5303384294488516e-01 -8.1305177509832338e-01 -7.2468343132486879e-01 -7.1752257114991291e-01 -7.1751283587848846e-01 -7.1453923493376237e-01 -7.1369020162895591e-01 -6.5649509297561537e-01 -6.5484509896979237e-01 -6.4819563037340588e-01 -6.1951447667181392e-01 -5.1149278097037365e-01 -4.5694083392342755e-01 -3.6925757458615477e-01 -1.8059222541578152e-01 6.9314384691905540e-02 7.4011368021968921e-02 1.1409015243786030e-01 1.4993230366869620e-01 1.8266979212308798e-01 1.9355783547294034e-01 2.1197839620144157e-01 2.5237136250916398e-01 2.7656210829344802e-01 2.8532362340693024e-01 3.0336607367287599e-01 3.3343210641995469e-01 3.3688909021539371e-01 3.9652955211547919e-01 4.2174259412621845e-01 5.4893794739746460e-01 5.6113635035532594e-01 6.8232567993302073e-01 8.8548529804693454e-01 9.2615819819803746e-01 9.2670939376619932e-01 9.6328467853269295e-01 9.8346700852382907e-01 9.9887403095928062e-01 1.0364505366104815e+00 1.0834412221767782e+00 1.0936564323948290e+00 1.1989337343801332e+00 1.2617669938536031e+00 1.2818433227209698e+00 1.3193949625250834e+00 1.3895935209003545e+00 1.4308893018588948e+00 1.4702798348151693e+00 1.4945329052436476e+00 1.5683750145557185e+00 1.5822512407754710e+00 1.6271532093503522e+00 1.6323133232152085e+00 1.6700777074669464e+00 1.7294530566974886e+00 1.8374560550516141e+00 1.9460155941151682e+00 1.9779608109076197e+00 2.0568938245396389e+00 2.2440133871466421e+00 2.9829355441190257e+00 3.0788481810437531e+00 5.2757403776476863e+00 2.1121787323240511e+02 ";
  cart.set_string("Method","HF");
  restr_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G",dip,true);

  Etot=-5.6676815794767272e+03;
  dip=3.7002300030873547e+00;
  Eorb="-9.3984197878959537e+02 -1.3753207522053614e+02 -1.2775494107412986e+02 -1.2775484171232620e+02 -1.2775461960894467e+02 -2.5877644442527373e+01 -2.1698212815100565e+01 -2.1697968271954661e+01 -2.1696704158528675e+01 -1.4963813791342242e+01 -1.4963803365239274e+01 -1.4963010152159761e+01 -1.4962863939931490e+01 -1.4962827551383636e+01 -1.4360306731928931e+01 -1.4284297994991533e+01 -1.0222101845510821e+01 -1.0189092847676903e+01 -1.0172543507953884e+01 -3.7205486431921089e+00 -2.3708088893299499e+00 -2.3701439379808553e+00 -2.3667607303778544e+00 -1.0858782838739469e+00 -9.3089329284247424e-01 -7.9260853129183317e-01 -6.6325335527214524e-01 -6.4256578410632736e-01 -6.0923656532522175e-01 -4.9709259456247573e-01 -4.8503224683848578e-01 -4.8171575506566477e-01 -4.7251855262767217e-01 -4.6879509575058409e-01 -4.6872800456219099e-01 -4.6436499521899027e-01 -4.6398093875520052e-01 -4.5266952059657528e-01 -3.3969175958949394e-01 -3.2987594723582159e-01 -2.7191017549392671e-01 -1.3325525844030733e-01 -4.5383616398526503e-03 -2.5448036053601006e-03 7.9044973041197241e-03 3.3666773904073487e-02 3.7585799420959326e-02 6.3639043549022917e-02 9.5522169861084263e-02 1.3206159977476106e-01 1.3440150130190801e-01 1.5776398028046859e-01 1.6224243525732554e-01 1.8802877350429267e-01 2.0172854652596459e-01 2.1971847445260764e-01 2.4645550592121568e-01 3.6713699334040706e-01 3.7513123098594359e-01 4.5357894586887376e-01 6.5944642837493050e-01 6.7596336906936427e-01 6.7892117828141851e-01 7.2415269433583329e-01 7.3375297001409212e-01 7.3583957260118482e-01 7.7451977484223711e-01 8.4047815076828869e-01 8.5273298296583389e-01 9.2809648967891500e-01 9.7420657389023768e-01 1.0043092051504543e+00 1.0256246087198848e+00 1.0797443972179583e+00 1.1228254103219337e+00 1.1979813921682569e+00 1.2156262812760956e+00 1.2858830965237353e+00 1.3032976682435771e+00 1.3297849877503438e+00 1.3368291650049697e+00 1.3822188324659626e+00 1.4204706180459332e+00 1.5533759863685421e+00 1.6676753248298970e+00 1.6794892260334329e+00 1.7665908000704755e+00 1.9160562753852401e+00 2.6597651277283858e+00 2.7556240575700519e+00 4.7845674021944999e+00 2.0810560203522525e+02 ";
  dftcart_nofit.set_string("Method","hyb_gga_xc_b3lyp");
  restr_test(cdcplx,b3_21G,dftcart_nofit,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",dip,false);

  Etot=-4.6701319106482447e+02;
  dip=5.6560553488190835e-01;
  Eorb="-1.8627156137417327e+01 -9.8468881482334449e+00 -9.8000391517456240e+00 -9.7985011419395285e+00 -9.7964364575666831e+00 -9.7961796182371188e+00 -9.7961538181943340e+00 -9.7957599877770818e+00 -9.7955835271464142e+00 -9.7955232617844441e+00 -9.7902003723421469e+00 -9.4290504921995910e-01 -7.5219613177750255e-01 -7.3523070964906101e-01 -7.0339284896587462e-01 -6.7053970652455031e-01 -6.3196272183137003e-01 -5.8307096954742310e-01 -5.5339028077667529e-01 -5.4222232574055551e-01 -5.1586037470143331e-01 -5.0988656850395486e-01 -4.8254334762174023e-01 -4.3597690315274562e-01 -4.3184612756158730e-01 -4.1614320411970401e-01 -4.0900622004227322e-01 -4.0310188711240980e-01 -3.8729582524577333e-01 -3.7610044215073379e-01 -3.7081071380025027e-01 -3.4983175698086622e-01 -3.4595200750183713e-01 -3.3597040211515034e-01 -3.2455675534576583e-01 -3.2107022335785029e-01 -3.1361134265713031e-01 -2.9876292893618772e-01 -2.9369136716423205e-01 -2.9084571652734992e-01 -2.8883498873149438e-01 -2.8096971437779500e-01 -2.7633805939887562e-01 -2.6485285973526329e-01 -2.2609541072794082e-01 2.9805855890125205e-02 4.3502677336301421e-02 4.7426725358816237e-02 5.3710757373764281e-02 6.2797617355154711e-02 6.8714814049659734e-02 7.6075981752471336e-02 8.0543504055024862e-02 9.0742583170960109e-02 1.0571678684073378e-01 1.1202417779875126e-01 1.1622977423363913e-01 1.2071505887759834e-01 1.3141047000973574e-01 1.3642706211039182e-01 1.3882287416267391e-01 1.4089055726261568e-01 1.4347231788566947e-01 1.4890062747822827e-01 1.5392887231669791e-01 1.6458457312108646e-01 1.6999277437151880e-01 1.7439450527523565e-01 1.8135077373176176e-01 1.9191613434592550e-01 1.9907690618583784e-01 2.0636287814779239e-01 2.1896427853323663e-01 2.2823888475242760e-01 2.4034566912078567e-01 2.4854964939139448e-01 2.5464927713572211e-01 4.2493471204915423e-01 4.2851185152207577e-01 4.4064832814501587e-01 4.5736258205561031e-01 4.6331892590474794e-01 4.6892714839122762e-01 4.7902895761183256e-01 4.8942038635110141e-01 5.0243462398614447e-01 5.1194661341214032e-01 5.1904667789129089e-01 5.3632921696117519e-01 5.4651593943983845e-01 5.7350572278151102e-01 5.9413190619695777e-01 6.0274011684506235e-01 6.0615107216739139e-01 6.1192054488302916e-01 6.1545345280250008e-01 6.3140456749211349e-01 6.4954861771030892e-01 6.7503539962620718e-01 6.8567621358039499e-01 6.9676237746204350e-01 7.1482451925204060e-01 7.2341450054095380e-01 7.4573454757339441e-01 7.5077764004480818e-01 7.6086360294916189e-01 7.7072201661858797e-01 7.7609737860640371e-01 7.8186598771249005e-01 7.9605299507692362e-01 8.0733034509648116e-01 8.1699593686094529e-01 8.2988605370022717e-01 8.3747034317358782e-01 8.3872144621008993e-01 8.4332432024295800e-01 8.4899282517817176e-01 8.5771956081311873e-01 8.6352094748228103e-01 8.6889377491099895e-01 8.7768136666174590e-01 8.8613716944134924e-01 8.9531601913176428e-01 9.0580566191783574e-01 9.1529333166776872e-01 9.2211841552923623e-01 9.4122939281064977e-01 9.6566486778200644e-01 9.7153246956211392e-01 9.8203219682794429e-01 1.0177360042227663e+00 1.0490684286401382e+00 1.0974976528006326e+00 1.1473874617741593e+00 1.1642859026476202e+00 1.2116509187234092e+00 1.2321590391973998e+00 1.2665604261644305e+00 1.2725684625955243e+00 1.3173850202461097e+00 1.3344566273045220e+00 1.3696820715020588e+00 1.4032275848633804e+00 1.4066934922700578e+00 1.4522683232196363e+00 1.4859434634818798e+00 1.4994160378392594e+00 1.5182767451661230e+00 1.5407584835309405e+00 1.5551904002112866e+00 1.5718411887145096e+00 1.5854149676144540e+00 1.6035523896270985e+00 1.6248643113168528e+00 1.6295959660148072e+00 1.6386262242225920e+00 1.6518538728066250e+00 1.6708612449332727e+00 1.7082446818965493e+00 1.7240850952351370e+00 1.7309952793402690e+00 1.7768187846159593e+00 1.7799403646898644e+00 1.7966946016865031e+00 1.7986817518737850e+00 1.8208995570983242e+00 1.8372013232100144e+00 1.8486146070580709e+00 1.8627514421022584e+00 1.8684098765923618e+00 1.8910561096102061e+00 1.9068270583354132e+00 1.9273968288151746e+00 1.9366445871493319e+00 1.9517986017360678e+00 1.9711844780258170e+00 1.9748133079850092e+00 1.9784543231904848e+00 2.0029278940052051e+00 2.0163943715147652e+00 2.0242114756621645e+00 2.0282110266350348e+00 2.0446485567156190e+00 2.0506337590002648e+00 2.0622360718396298e+00 2.0764528328399323e+00 2.0982719010444155e+00 2.1124509195388050e+00 2.1473844095824055e+00 2.1546267201978755e+00 2.1669075167056113e+00 2.1723427463360641e+00 2.1811757011080273e+00 2.1987630017790925e+00 2.2110769871866456e+00 2.2189959413640294e+00 2.2523882270903721e+00 2.2601867182510005e+00 2.2680781760091131e+00 2.2959489188609550e+00 2.3105475471526731e+00 2.3159945969373190e+00 2.3268620433846765e+00 2.3486729796965666e+00 2.3828969309036196e+00 2.3876853759829482e+00 2.4069231194196723e+00 2.4220207900319255e+00 2.4322427568698468e+00 2.4627682578958106e+00 2.4929223015084152e+00 2.5133883875321357e+00 2.5312880356329388e+00 2.5380554046584898e+00 2.5674716654750469e+00 2.5816435389132626e+00 2.5894762032825001e+00 2.6092020143582131e+00 2.6302177174816777e+00 2.6355319705963547e+00 2.6434882100175709e+00 2.6604218865296438e+00 2.6727749979160729e+00 2.6917620788793659e+00 2.6952997280632234e+00 2.7073943192411920e+00 2.7113430298809296e+00 2.7285879153786974e+00 2.7487932105994859e+00 2.7749377073289558e+00 2.7823799816904224e+00 2.7848199307394479e+00 2.7958608219995158e+00 2.8014818217458046e+00 2.8080216831761553e+00 2.8118711270925507e+00 2.8150035616401028e+00 2.8202130279814552e+00 2.8419227985129667e+00 2.8601401507153428e+00 2.8723563155085690e+00 2.9059065752003774e+00 3.0865702800657693e+00 3.1217735371526696e+00 3.1464493193518370e+00 3.1676580405152510e+00 3.1811714301760512e+00 3.1906306283708816e+00 3.1946230916616387e+00 3.2130867125830638e+00 3.2280265646555804e+00 3.2455319088060648e+00 3.2648013869386534e+00 3.2857855810474614e+00 3.3101643538846028e+00 3.3528874734104566e+00 3.3823787880156475e+00 3.3896325761672110e+00 3.3912089878962131e+00 3.4239218421893685e+00 3.4639876952564412e+00 3.4735427201688931e+00 3.4830955915506934e+00 3.4844494961941388e+00 3.8198802524283990e+00 4.1566349269328278e+00 4.2134221437934123e+00 4.2755969057774372e+00 4.3733062553903288e+00 4.4240204790626088e+00 4.4487042219307691e+00 4.5000884822840517e+00 4.5662887855788883e+00 4.6275754521984185e+00 4.7059780253374575e+00 ";
  dftcart.set_string("Method","lda_x-lda_c_vwn_rpa");
  restr_test(decanol,b6_31Gpp,dftcart,Etot,Eorb,"1-decanol, SVWN(RPA)/6-31G**",dip,true);


#ifndef COMPUTE_REFERENCE
  printf("****** Tests completed in %s *******\n",t.elapsed().c_str());
#endif

  return 0;
}
