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
    emdok=((fabs(mom(4,1)-2.0*en.Ekin)<=mom(4,2)) && (fabs(mom(2,1)-arma::trace(P*S))<=mom(2,2)));
  }


#ifdef COMPUTE_REFERENCE
  printf("Etot=%.16e;\n",en.E);
  printf("dip=%.16e;\n",dip);
  printf("Eorb=\"");
  for(size_t i=0;i<Eo.n_elem;i++)
    printf("%.16e ",Eo(i));
  printf("\";\n");
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

    // Compare <p^2> with T
    emdok=(fabs(mom(4,1)-2.0*en.Ekin)<mom(4,2));
  }


#ifdef COMPUTE_REFERENCE
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

  Etot=-1.2848877555174082e+02;
  dip=5.7206424166128525e-16;
  Eorb="-3.2765635418578533e+01 -1.9187982340008261e+00 -8.3209725200484452e-01 -8.3209725200484153e-01 -8.3209725200484086e-01 1.6945577282762601e+00 1.6945577282762661e+00 1.6945577282762763e+00 2.1594249508065495e+00 5.1967114014289297e+00 5.1967114014289351e+00 5.1967114014289448e+00 5.1967114014289502e+00 5.1967114014289617e+00 ";
  sph.set_string("Method","HF");
  restr_test(Ne,cc_pVDZ,sph,Etot,Eorb,"Neon, HF/cc-pVDZ",dip,true);

  Etot=-1.2848886617203752e+02;
  dip=4.4557482940797417e-16;
  Eorb="-3.2765400792453278e+01 -1.9190111534959520e+00 -8.3228219724415120e-01 -8.3228219724414843e-01 -8.3228219724414521e-01 1.6944246993916028e+00 1.6944246993916092e+00 1.6944246993916154e+00 1.9905987683859308e+00 5.1964245996467664e+00 5.1964245996467779e+00 5.1964245996467842e+00 5.1964245996467886e+00 5.1964245996467975e+00 1.0383358435419975e+01 ";
  cart.set_string("Method","HF");
  restr_test(Ne,cc_pVDZ,cart,Etot,Eorb,"Neon, HF/cc-pVDZ cart",dip,false);

  Etot=-1.2853186163632137e+02;
  dip=1.4214203598379343e-15;
  Eorb="-3.2769110714528658e+01 -1.9270833039703708e+00 -8.4541551017319616e-01 -8.4541551017319438e-01 -8.4541551017318473e-01 1.0988680366807098e+00 1.0988680366807102e+00 1.0988680366807186e+00 1.4176388079436251e+00 2.8142175659355386e+00 2.8142175659355466e+00 2.8142175659355471e+00 2.8142175659355488e+00 2.8142175659355524e+00 6.1558667265949829e+00 6.1558667265949838e+00 6.1558667265950016e+00 9.6473695825273680e+00 9.6473695825273733e+00 9.6473695825273804e+00 9.6473695825273822e+00 9.6473695825273875e+00 9.6473695825274035e+00 9.6473695825274071e+00 1.1227312685696281e+01 1.1227312685696283e+01 1.1227312685696287e+01 1.1227312685696305e+01 1.1227312685696312e+01 1.1744558070628615e+01 ";
  sph.set_string("Method","HF");
  restr_test(Ne,cc_pVTZ,sph,Etot,Eorb,"Neon, HF/cc-pVTZ",dip,false);

  Etot=-1.2853200998517838e+02;
  dip=1.1051055106472576e-15;
  Eorb="-3.2769827645786037e+01 -1.9274545167728918e+00 -8.4572301732492183e-01 -8.4572301732491673e-01 -8.4572301732490052e-01 8.8038911821659349e-01 1.0282198400142224e+00 1.0282198400142359e+00 1.0282198400142488e+00 2.8138968462395724e+00 2.8138968462395728e+00 2.8138968462396035e+00 2.8138968462396123e+00 2.8138968462396292e+00 4.1362240320956083e+00 4.6398467063294477e+00 4.6398467063295730e+00 4.6398467063296058e+00 9.6470056626706793e+00 9.6470056626706864e+00 9.6470056626706899e+00 9.6470056626707041e+00 9.6470056626707361e+00 9.6470056626707557e+00 9.6470056626707823e+00 1.1226914496837626e+01 1.1226914496837649e+01 1.1226914496837665e+01 1.1226914496837681e+01 1.1226914496837709e+01 1.1317534800567527e+01 1.1317534800567607e+01 1.1317534800567788e+01 1.6394442677951606e+01 2.8816114657744336e+01 ";
  cart.set_string("Method","HF");
  restr_test(Ne,cc_pVTZ,cart,Etot,Eorb,"Neon, HF/cc-pVTZ cart",dip,false);

  Etot=-1.2854346965912185e+02;
  dip=3.2021748118834346e-15;
  Eorb="-3.2771496235374315e+01 -1.9293376383772340e+00 -8.4895896153025774e-01 -8.4895896153025752e-01 -8.4895896153024974e-01 8.0890413831431107e-01 8.0890413831431496e-01 8.0890413831433072e-01 9.3559988865755328e-01 1.9978112793680989e+00 1.9978112793681106e+00 1.9978112793681158e+00 1.9978112793681326e+00 1.9978112793681404e+00 3.9328189051347304e+00 3.9328189051347402e+00 3.9328189051347495e+00 5.8106845419074720e+00 5.9042211377158509e+00 5.9042211377158651e+00 5.9042211377158731e+00 5.9042211377158758e+00 5.9042211377158837e+00 5.9042211377158873e+00 5.9042211377158909e+00 6.7616951536519663e+00 6.7616951536519707e+00 6.7616951536519734e+00 6.7616951536519858e+00 6.7616951536519903e+00 1.4903626161293799e+01 1.4903626161293806e+01 1.4903626161293818e+01 1.4903626161293834e+01 1.4903626161293856e+01 1.4903626161293863e+01 1.4903626161293888e+01 1.4903626161293904e+01 1.4903626161293904e+01 1.5804420583863358e+01 1.5804420583863383e+01 1.5804420583863422e+01 1.9794585642017378e+01 1.9794585642017385e+01 1.9794585642017399e+01 1.9794585642017413e+01 1.9794585642017434e+01 1.9794585642017466e+01 1.9794585642017505e+01 2.0954549904477808e+01 2.0954549904477833e+01 2.0954549904477854e+01 2.0954549904477872e+01 2.0954549904477894e+01 6.6550956473550229e+01 ";
  sph.set_string("Method","HF");
  restr_test(Ne,cc_pVQZ,sph,Etot,Eorb,"Neon, HF/cc-pVQZ",dip,false);

  Etot=-1.2854353449722555e+02;
  dip=9.5502946808503636e-16;
  Eorb="-3.2771625129304006e+01 -1.9294942840840930e+00 -8.4906688457015289e-01 -8.4906688457013879e-01 -8.4906688457011326e-01 5.8690441472046939e-01 7.1271797752943955e-01 7.1271797752952104e-01 7.1271797752966903e-01 1.9879845922197152e+00 1.9879845922197266e+00 1.9879845922197301e+00 1.9879845922197374e+00 1.9879845922197594e+00 2.5105148502691832e+00 2.7214792301836028e+00 2.7214792301844364e+00 2.7214792301849156e+00 5.9040962889273514e+00 5.9040962889273638e+00 5.9040962889273887e+00 5.9040962889274056e+00 5.9040962889274082e+00 5.9040962889274189e+00 5.9040962889274340e+00 6.4115733390970142e+00 6.5684069303762236e+00 6.5684069303762520e+00 6.5684069303762831e+00 6.5684069303762946e+00 6.5684069303763151e+00 6.7659166002162312e+00 6.7659166002172855e+00 6.7659166002178068e+00 1.4004805313638407e+01 1.4903514354815833e+01 1.4903514354815854e+01 1.4903514354815885e+01 1.4903514354815902e+01 1.4903514354815918e+01 1.4903514354815931e+01 1.4903514354815984e+01 1.4903514354816046e+01 1.4903514354816060e+01 1.8145155385363683e+01 1.8145155385363875e+01 1.8145155385364870e+01 1.8145155385367524e+01 1.8145155385370863e+01 1.8540067452605978e+01 1.8540067452606280e+01 1.8540067452606625e+01 1.9794449045390628e+01 1.9794449045390706e+01 1.9794449045390767e+01 1.9794449045390770e+01 1.9794449045390810e+01 1.9794449045390834e+01 1.9794449045390873e+01 2.9727979557050126e+01 3.9089870736327491e+01 3.9089870736330781e+01 3.9089870736339627e+01 3.9089870736357135e+01 3.9089870736407022e+01 3.9551871550183897e+01 3.9551871550189333e+01 3.9551871550197419e+01 5.8376821811951636e+01 2.0568373997926801e+02 ";
  cart.set_string("Method","HF");
  restr_test(Ne,cc_pVQZ,cart,Etot,Eorb,"Neon, HF/cc-pVQZ cart",dip,true);

  Etot=-4.5929863365655353e+02;
  dip=7.0056679727986178e-15;
  Eorba="-1.0490206922615540e+02 -1.0633457343120563e+01 -8.1040456970866757e+00 -8.1040456970866721e+00 -8.1040456970866472e+00 -1.1496933695315230e+00 -5.4051895138115480e-01 -5.4051895138115258e-01 -5.4051895138114769e-01 4.9527126688407580e-01 5.8723188307406682e-01 5.8723188307406982e-01 5.8723188307407737e-01 1.0645429475081860e+00 1.0645429475082009e+00 1.0645429475082102e+00 1.0645429475082151e+00 1.0645429475082309e+00 ";
  Eorbb="-1.0488803471646810e+02 -1.0619374104092802e+01 -8.0797449831608752e+00 -8.0797449831608592e+00 -8.0797449831608485e+00 -1.0135876578143137e+00 -3.2998710690358241e-01 -3.2998710690356170e-01 -3.2998710690354610e-01 5.1691272642155983e-01 6.2702896022137533e-01 6.2702896022138654e-01 6.2702896022138777e-01 1.1416832737677816e+00 1.1416832737677824e+00 1.1416832737677851e+00 1.1416832737677969e+00 1.1416832737678109e+00 ";
  pol.set_string("Method","HF");
  unrestr_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, HF/6-31G** polarized",dip,false);

  Etot=-4.5929468403432463e+02;
  dip=4.3533650946783002e-16;
  Eorba="-1.0490985520662321e+02 -1.0639222875879000e+01 -8.1086670674509218e+00 -8.1086670674509094e+00 -8.1086670674509058e+00 -1.1427051245339961e+00 -5.3421029373398843e-01 -5.3421029373398787e-01 -5.3421029373398676e-01 4.9202842554983406e-01 5.8916630212254073e-01 5.8916630212254428e-01 5.8916630212255061e-01 1.0673064590407824e+00 1.0673064590407901e+00 1.0673064590407919e+00 1.0673064590407964e+00 1.0673064590407966e+00 ";
  Eorbb="-1.0490015301985268e+02 -1.0629710492175453e+01 -8.0913085983052326e+00 -8.0913085983052238e+00 -8.0913085983052184e+00 -1.0275446925912901e+00 -3.3913564185933154e-01 -3.3913564185933087e-01 -3.3913564185933026e-01 5.1611862854758883e-01 6.1978235357213896e-01 6.1978235357214184e-01 6.1978235357214606e-01 1.1316261852044098e+00 1.1316261852044363e+00 1.1316261852044429e+00 1.1316261852044434e+00 1.1316261852044465e+00 ";
  pol.set_string("Method","ROHF");
  unrestr_test(Cl,b6_31Gpp,pol,Etot,Eorba,Eorbb,"Chlorine, ROHF/6-31G**",dip,false);

  Etot=-4.6010297757405141e+02;
  dip=1.2022435269365581e-14;
  Eorba="-1.0158417621491525e+02 -9.5119736881692951e+00 -7.2706233268179323e+00 -7.2706233268179297e+00 -7.2706233268179199e+00 -8.4710663930193486e-01 -3.7353479972172710e-01 -3.7353479972172604e-01 -3.7353479972172493e-01 3.2665389851463505e-01 4.1370113075889564e-01 4.1370113075890524e-01 4.1370113075891690e-01 8.0595993079137007e-01 8.0595993079137507e-01 8.0595993079137707e-01 8.0595993079138251e-01 8.0595993079138306e-01 ";
  Eorbb="-1.0157708646536911e+02 -9.5053733435112431e+00 -7.2605395772294559e+00 -7.2605395772294496e+00 -7.2605395772294479e+00 -7.9608357397051188e-01 -3.0984544157977617e-01 -3.0984544157976435e-01 -3.0984544157974092e-01 3.4215684959920273e-01 4.3147074805148838e-01 4.3147074805148866e-01 4.3147074805149388e-01 8.4916878410204688e-01 8.4916878410206209e-01 8.4916878410206209e-01 8.4916878410206476e-01 8.4916878410209151e-01 ";
  dftpol_nofit.set_string("Method","hyb_gga_xc_b3lyp");
  unrestr_test(Cl,b6_31Gpp,dftpol_nofit,Etot,Eorba,Eorbb,"Chlorine, B3LYP/6-31G** polarized",dip,false);

  Etot=-1.1287000934442020e+00;
  dip=4.9196533417830540e-15;
  Eorb="-5.9241098912717105e-01 1.9744005746927679e-01 4.7932104726976116e-01 9.3732369228506429e-01 1.2929037097182370e+00 1.2929037097182376e+00 1.9570226089437432e+00 2.0435200542832326e+00 2.0435200542832348e+00 3.6104742345529406e+00 ";
  sph.set_string("Method","HF");
  restr_test(H2,cc_pVDZ,sph,Etot,Eorb,"Hydrogen molecule, HF/cc-pVDZ",dip,false);

  Etot=-1.1676141294603308e+00;
  dip=1.2920120371587861e-14;
  Eorb="-3.9201515131303832e-01 3.6521080362697057e-02 2.9071322496894020e-01 6.5833098538632140e-01 9.7502316975335390e-01 9.7502316975335457e-01 1.6066122928284690e+00 1.7001809420472254e+00 1.7001809420472280e+00 3.1926496355618328e+00 ";
  dftsph.set_string("Method","lda_x-lda_c_vwn_rpa");
  restr_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, SVWN(RPA)/cc-pVDZ",dip,false);

  Etot=-1.1603962667280063e+00;
  dip=1.7633027220705997e-15;
  Eorb="-3.7849167468778783e-01 5.3523492272485353e-02 3.0277089209824404e-01 6.6374851097954213e-01 9.9246487779866777e-01 9.9246487779866832e-01 1.6235426412156961e+00 1.7198880924572890e+00 1.7198880924572910e+00 3.2019311002043445e+00 ";
  dftsph.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(H2,cc_pVDZ,dftsph,Etot,Eorb,"Hydrogen molecule, PBEPBE/cc-pVDZ",dip,false);

  Etot=-7.6056825377225422e+01;
  dip=7.9472745072200657e-01;
  Eorb="-2.0555281285934726e+01 -1.3428635537909901e+00 -7.0828436625348579e-01 -5.7575384380924743e-01 -5.0391497984008182e-01 1.4187951589759165e-01 2.0351537962038532e-01 5.4324870468950459e-01 5.9753586791489099e-01 6.6949546783762515e-01 7.8747678855204362e-01 8.0274150280423950e-01 8.0481260840460700e-01 8.5898803455371819e-01 9.5702121842132137e-01 1.1344778955823469e+00 1.1928203508899533e+00 1.5241753072179296e+00 1.5579529874909843e+00 2.0324408704982821e+00 2.0594682937122948e+00 2.0654407668810433e+00 2.1686553973436147e+00 2.2363161875432400e+00 2.5909431287455948e+00 2.9581971204027107e+00 3.3610002638181835e+00 3.4914002761854577e+00 3.5741938478471766e+00 3.6463660415151087e+00 3.7977214224126157e+00 3.8739670211315360e+00 3.8824466785427605e+00 3.9569498256901841e+00 4.0199059058121316e+00 4.0760332618203368e+00 4.1862021920006081e+00 4.3092789389183386e+00 4.3875716398195577e+00 4.5640073766296867e+00 4.6817931186516955e+00 4.8550947816336150e+00 5.1380848619670463e+00 5.2500191191868639e+00 5.5275547773335951e+00 6.0402478809183231e+00 6.5453259405837096e+00 6.9113516634168608e+00 6.9366142668500848e+00 7.0003720398995481e+00 7.0078239258509960e+00 7.0609382581883109e+00 7.1598075631432812e+00 7.2256524677377953e+00 7.4561719765011816e+00 7.7799625501198211e+00 8.2653639981405256e+00 1.2804358856428037e+01 ";
  sph.set_string("Method","HF");
  restr_test(h2o,cc_pVTZ,sph,Etot,Eorb,"Water, HF/cc-pVTZ",dip,true);
  direct.set_string("Method","HF");
  restr_test(h2o,cc_pVTZ,direct,Etot,Eorb,"Water, HF/cc-pVTZ direct",dip,false);

  Etot=-7.6064480528901996e+01;
  dip=7.8765852091213884e-01;
  Eorb="-2.0560341503070870e+01 -1.3467109753317892e+00 -7.1286865704812985e-01 -5.7999183696220857e-01 -5.0759009739011363e-01 1.1677961589096358e-01 1.7061237229712237e-01 4.4878630960720378e-01 4.6240973978032041e-01 4.9860081780452231e-01 5.8389461441260848e-01 6.0602248903580302e-01 6.1386901025638518e-01 6.5509670197439718e-01 7.1940581338715970e-01 8.5193265637445159e-01 9.1760642713404861e-01 1.1091391703789830e+00 1.1559117513387467e+00 1.3479064828239411e+00 1.4144381942402131e+00 1.4776186086096790e+00 1.4856774065797351e+00 1.5814608129854377e+00 1.6854835697724755e+00 1.9096187714790473e+00 2.0727777009518751e+00 2.1976502952743173e+00 2.2888869956948557e+00 2.3588905628937868e+00 2.4246094515834740e+00 2.4837778781609683e+00 2.5224544336426380e+00 2.5800657941447209e+00 2.5803867643471823e+00 2.6507304488272769e+00 2.6683130229705831e+00 2.8407379933648804e+00 2.8643130298355373e+00 3.0412098274030632e+00 3.1190680413397298e+00 3.2889763501206364e+00 3.3518967285195242e+00 3.4467312435745998e+00 3.6214003254381333e+00 3.8285931940886737e+00 3.9968185293596479e+00 4.1278766575022532e+00 4.1879462994524301e+00 4.2176486551288326e+00 4.4343620853999530e+00 4.4925098765109226e+00 4.6832772386553669e+00 4.7403725769474701e+00 4.8079058187004913e+00 4.9140701187379188e+00 5.3503959730578057e+00 5.4039303348339010e+00 5.9860940816386616e+00 6.1030498779280915e+00 6.2449376391805291e+00 6.3029981795203440e+00 6.7000543778133013e+00 6.7926548536651277e+00 7.0589633218031382e+00 7.2683601224501091e+00 7.3171930242900993e+00 7.3671378803166627e+00 7.4371264985030203e+00 7.5184752946049738e+00 7.5458434074120815e+00 7.5694204735356037e+00 8.0046360187860319e+00 8.0708295731913307e+00 8.0987711863948846e+00 8.1338237893781606e+00 8.1523664183036892e+00 8.2695443407258793e+00 8.3150962692646164e+00 8.3485048844392153e+00 8.4164827900428314e+00 8.6181288282799464e+00 8.8336406071438933e+00 8.9048326503928354e+00 8.9437734419928479e+00 9.2166366996281148e+00 9.3761387895206560e+00 9.3791690844601199e+00 9.9423093830121729e+00 1.0035594101020509e+01 1.0257561208226431e+01 1.0425629819423381e+01 1.0646599814788384e+01 1.0757780264898562e+01 1.0806846315825933e+01 1.1272406006487746e+01 1.1390414016451706e+01 1.1595907189072937e+01 1.1644666354945205e+01 1.1693629515526762e+01 1.1844883870853446e+01 1.2158546419172534e+01 1.2320144570863606e+01 1.2398213786649526e+01 1.2413264229519632e+01 1.2465013633068635e+01 1.3602019533761757e+01 1.3763660836516737e+01 1.4247885728479471e+01 1.4614348058859646e+01 1.4639079459321810e+01 1.4826337833045631e+01 1.6435472092033965e+01 1.6799107180542922e+01 4.4322817445320311e+01 ";
  sph.set_string("Method","HF");
  restr_test(h2o,cc_pVQZ,sph,Etot,Eorb,"Water, HF/cc-pVQZ",dip,false);
  direct.set_string("Method","HF");
  restr_test(h2o,cc_pVQZ,direct,Etot,Eorb,"Water, HF/cc-pVQZ direct",dip,false);

  Etot=-7.6372961222395176e+01;
  dip=7.3249328281426695e-01;
  Eorb="-1.8738953501512668e+01 -9.1586785162758488e-01 -4.7160106589892448e-01 -3.2507287613067931e-01 -2.4816655857437953e-01 8.4534036387175372e-03 8.1125737524763347e-02 3.3806729297976212e-01 3.7949122776263966e-01 4.6548871040459011e-01 5.4539135465825450e-01 5.9072372577610277e-01 5.9687803067281364e-01 6.4398044404233656e-01 7.4418029766585625e-01 8.8306547475122232e-01 9.7381887521623300e-01 1.2412763516779126e+00 1.2611338609439329e+00 1.6999121835648665e+00 1.7261707906470463e+00 1.7619584545398468e+00 1.8256258288425098e+00 1.9002743849740662e+00 2.1846823312343671e+00 2.5326906159406088e+00 2.9875043516088651e+00 3.1260225297411024e+00 3.1892007455508291e+00 3.2799583723193648e+00 3.2859664655622591e+00 3.4399388077633155e+00 3.5114854019505550e+00 3.5697880009196892e+00 3.6175833251931109e+00 3.6363368817608812e+00 3.6947695819172353e+00 3.9240129133177546e+00 3.9512251176804907e+00 4.1557724959072777e+00 4.1932317708017095e+00 4.4292385921948565e+00 4.6459620146423370e+00 4.7303864407719258e+00 4.9898268736842226e+00 5.4868943652627520e+00 5.9838983452499930e+00 6.2843629405007277e+00 6.2901656603461262e+00 6.3781111568916353e+00 6.4202760176874385e+00 6.4811650993453380e+00 6.5329728518002055e+00 6.6594412404200245e+00 6.8404602024857608e+00 7.1724946503567466e+00 7.6259352319106544e+00 1.1962240167932499e+01 ";
  dftnofit.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",dip,false);

  Etot=-7.6373025076488673e+01;
  dip=7.3223619689403330e-01;
  Eorb="-1.8739067712785722e+01 -9.1594855273587239e-01 -4.7168658517960171e-01 -3.2515022233953389e-01 -2.4824134771110840e-01 7.6868694473288309e-03 8.0201485087255525e-02 3.3767224652239702e-01 3.7911611254247746e-01 4.6520448054549846e-01 5.4523065610039867e-01 5.9029285519597252e-01 5.9663403526027015e-01 6.4384844712900602e-01 7.4405492511090687e-01 8.8293500311302198e-01 9.7364987322544572e-01 1.2410683077869380e+00 1.2610076706322275e+00 1.6997990579831839e+00 1.7260539463664721e+00 1.7615703419251318e+00 1.8254642144178257e+00 1.8999177919380381e+00 2.1845241425571515e+00 2.5325349915420712e+00 2.9874617550488836e+00 3.1259288098732232e+00 3.1891620087522048e+00 3.2797685659270934e+00 3.2858678167539623e+00 3.4399087715591086e+00 3.5114590806631694e+00 3.5697300791389694e+00 3.6174584672067120e+00 3.6361205767158391e+00 3.6945923089450781e+00 3.9240163487242503e+00 3.9511494091589037e+00 4.1557167093092815e+00 4.1932269766328245e+00 4.4291609287371365e+00 4.6459391796805267e+00 4.7302938427415056e+00 4.9896645806173474e+00 5.4868660751967662e+00 5.9839044167317903e+00 6.2842935455017033e+00 6.2900613307459370e+00 6.3780017572026706e+00 6.4202853530426838e+00 6.4811643201762257e+00 6.5328557623554646e+00 6.6594818284426678e+00 6.8404421490490348e+00 7.1725297508540526e+00 7.6257028377016018e+00 1.1962121373941059e+01 ";
  dftsph.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",dip,false);
  dftdirect.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",dip,false);

  Etot=-7.6374645230136579e+01;
  dip=7.3272055001056791e-01;
  Eorb="-1.8741546114801132e+01 -9.1788065208209446e-01 -4.7348527540133711e-01 -3.2745824709260957e-01 -2.5054888296114508e-01 2.7901830605709353e-03 7.8041748636364830e-02 3.2383381111154008e-01 3.5924016484003485e-01 4.5242229923291338e-01 5.1413396046066395e-01 5.7767374228604651e-01 5.8423722555860824e-01 6.4253357514563425e-01 6.5976489675954308e-01 7.4241931809380313e-01 9.7186975918768803e-01 1.1822077426651889e+00 1.2023227295161933e+00 1.5759172313869179e+00 1.6360776272990976e+00 1.6982201437031095e+00 1.7245167623654827e+00 1.8628724330066746e+00 1.9081332377655202e+00 2.1641869722699374e+00 2.3473125251804889e+00 2.7892764468587328e+00 3.0049313704665610e+00 3.0831686635910702e+00 3.1876656925771618e+00 3.2328093756756040e+00 3.4168951292175458e+00 3.4579185918048125e+00 3.5049998737852537e+00 3.5282503323274854e+00 3.5683149815126400e+00 3.5772258007783102e+00 3.8379020258070216e+00 3.9226525885208940e+00 4.0867013276915669e+00 4.0926885575202059e+00 4.3310827329021171e+00 4.4154604410275207e+00 4.4322656550567450e+00 4.6027454835204953e+00 5.1266097225215299e+00 5.2200923676093147e+00 5.4840610597680879e+00 6.1494592260597187e+00 6.2799826595305293e+00 6.2885803768662605e+00 6.3502580499100194e+00 6.4059374497217982e+00 6.4358015216736266e+00 6.6570174354996077e+00 6.7153040590271038e+00 6.7372414808564365e+00 6.9398791763075751e+00 7.3406783795788462e+00 8.2789338893516629e+00 8.3551996988201971e+00 9.3390714650755893e+00 1.4480083318411110e+01 1.5822331894566005e+01 ";
  dftcart.set_string("Method","gga_x_pbe-gga_c_pbe");
  restr_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",dip,false);

  Etot=-5.6637319431552451e+03;
  dip=4.1660088834246976e+00;
  Eorb="-9.4985623401084922e+02 -1.4146635096151900e+02 -1.3119294652385088e+02 -1.3119287267577752e+02 -1.3119259493746125e+02 -2.7676547691792301e+01 -2.3232980792733077e+01 -2.3232722807094294e+01 -2.3231117092468306e+01 -1.6049049285751671e+01 -1.6049045527048754e+01 -1.6047827628809060e+01 -1.6047723998141574e+01 -1.6047713429571882e+01 -1.5604531114638549e+01 -1.5532249508743689e+01 -1.1296661981993843e+01 -1.1249243905448862e+01 -1.1232665100177721e+01 -4.3970789345813728e+00 -2.8992752064223999e+00 -2.8986598198201792e+00 -2.8951186726513405e+00 -1.4177741000934041e+00 -1.2312596845201322e+00 -1.0610694333369168e+00 -8.7645299904748530e-01 -8.5303383590368376e-01 -8.1305176899424181e-01 -7.2468343875797003e-01 -7.1752258214429510e-01 -7.1751284685512984e-01 -7.1453924533189261e-01 -7.1369021199259119e-01 -6.5649508637232634e-01 -6.5484509295061721e-01 -6.4819562287760424e-01 -6.1951447150925953e-01 -5.1149277324130693e-01 -4.5694083043993949e-01 -3.6925756806739601e-01 -1.8059222432334721e-01 6.9314381254108728e-02 7.4011365129729501e-02 1.1409015162784526e-01 1.4993230420133646e-01 1.8266978920779786e-01 1.9355783709120195e-01 2.1197840197699441e-01 2.5237135912475550e-01 2.7656210436371020e-01 2.8532362813164641e-01 3.0336607729429366e-01 3.3343210573533683e-01 3.3688909108465009e-01 3.9652955932324596e-01 4.2174259761515015e-01 5.4893795286654889e-01 5.6113635599052614e-01 6.8232568727284570e-01 8.8548529678096843e-01 9.2615820257226422e-01 9.2670939968291288e-01 9.6328468236703935e-01 9.8346701432100769e-01 9.9887403607853043e-01 1.0364505418087133e+00 1.0834412255935677e+00 1.0936564370961901e+00 1.1989337393459161e+00 1.2617670006488484e+00 1.2818433261317506e+00 1.3193949689649473e+00 1.3895935277553249e+00 1.4308893047157156e+00 1.4702798396743568e+00 1.4945329094972504e+00 1.5683750147507491e+00 1.5822512400322790e+00 1.6271531996210089e+00 1.6323133139909889e+00 1.6700777111800558e+00 1.7294530519032278e+00 1.8374560579013428e+00 1.9460156012210297e+00 1.9779608167721001e+00 2.0568938304150897e+00 2.2440133879579096e+00 2.9829355522082452e+00 3.0788481916082322e+00 5.2757403673940324e+00 2.1121787321920766e+02 ";
  cart.set_string("Method","HF");
  restr_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G",dip,true);

  Etot=-5.6676815794766926e+03;
  dip=3.7002300085044890e+00;
  Eorb="-9.3984197878952841e+02 -1.3753207522046588e+02 -1.2775494107405805e+02 -1.2775484171225446e+02 -1.2775461960887357e+02 -2.5877644442456802e+01 -2.1698212815028793e+01 -2.1697968271882321e+01 -2.1696704158457113e+01 -1.4963813791270914e+01 -1.4963803365167337e+01 -1.4963010152088080e+01 -1.4962863939859645e+01 -1.4962827551311833e+01 -1.4360306731961844e+01 -1.4284297995016706e+01 -1.0222101845542946e+01 -1.0189092847713537e+01 -1.0172543507985537e+01 -3.7205486431198804e+00 -2.3708088892569479e+00 -2.3701439379079918e+00 -2.3667607303053226e+00 -1.0858782839065826e+00 -9.3089329287251510e-01 -7.9260853132579101e-01 -6.6325335530145113e-01 -6.4256578413793441e-01 -6.0923656535837034e-01 -4.9709259456890159e-01 -4.8503224686270330e-01 -4.8171575509472009e-01 -4.7251855258258801e-01 -4.6879509568055117e-01 -4.6872800449357094e-01 -4.6436499517171681e-01 -4.6398093869607449e-01 -4.5266952061687249e-01 -3.3969175959210790e-01 -3.2987594726594271e-01 -2.7191017553038022e-01 -1.3325525837862687e-01 -4.5383611666855311e-03 -2.5448034298397684e-03 7.9044987764334759e-03 3.3666775538252508e-02 3.7585799687484309e-02 6.3639044594268168e-02 9.5522170012342783e-02 1.3206159995685737e-01 1.3440150146033925e-01 1.5776398038903308e-01 1.6224244042871572e-01 1.8802877366446027e-01 2.0172854854460281e-01 2.1971847445246126e-01 2.4645550608891145e-01 3.6713699341960859e-01 3.7513123097133150e-01 4.5357894584789393e-01 6.5944642835063838e-01 6.7596336913781108e-01 6.7892117829366350e-01 7.2415269434745366e-01 7.3375296998000461e-01 7.3583957280832468e-01 7.7451977483344381e-01 8.4047815074913601e-01 8.5273298305566703e-01 9.2809648988551208e-01 9.7420657391615106e-01 1.0043092051244433e+00 1.0256246087480674e+00 1.0797443971853455e+00 1.1228254103218724e+00 1.1979813921467568e+00 1.2156262812631009e+00 1.2858830965429917e+00 1.3032976683000577e+00 1.3297849878102790e+00 1.3368291650747819e+00 1.3822188324525937e+00 1.4204706180888911e+00 1.5533759863601031e+00 1.6676753248022878e+00 1.6794892260077781e+00 1.7665908000423525e+00 1.9160562755412367e+00 2.6597651277331500e+00 2.7556240575712274e+00 4.7845674022918327e+00 2.0810560203555852e+02 ";
  dftcart_nofit.set_string("Method","hyb_gga_xc_b3lyp");
  restr_test(cdcplx,b3_21G,dftcart_nofit,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",dip,false);

  Etot=-4.6701319106466121e+02;
  dip=5.6560553480429165e-01;
  Eorb="-1.8627156137414740e+01 -9.8468881482340400e+00 -9.8000391517469598e+00 -9.7985011419330252e+00 -9.7964364575688929e+00 -9.7961796182329461e+00 -9.7961538181902839e+00 -9.7957599877667558e+00 -9.7955835271397795e+00 -9.7955232617832486e+00 -9.7902003723404949e+00 -9.4290504921817542e-01 -7.5219613177441991e-01 -7.3523070964674742e-01 -7.0339284896349685e-01 -6.7053970652227235e-01 -6.3196272182918134e-01 -5.8307096954512283e-01 -5.5339028077466890e-01 -5.4222232573820928e-01 -5.1586037469875401e-01 -5.0988656850143577e-01 -4.8254334761967033e-01 -4.3597690315016052e-01 -4.3184612755778290e-01 -4.1614320411869582e-01 -4.0900622003916576e-01 -4.0310188710970768e-01 -3.8729582524320422e-01 -3.7610044214693328e-01 -3.7081071379694314e-01 -3.4983175697788466e-01 -3.4595200749954713e-01 -3.3597040211245149e-01 -3.2455675534227874e-01 -3.2107022335441965e-01 -3.1361134265387458e-01 -2.9876292893223810e-01 -2.9369136716103117e-01 -2.9084571652371832e-01 -2.8883498872629759e-01 -2.8096971437661467e-01 -2.7633805939563316e-01 -2.6485285973129780e-01 -2.2609541072740300e-01 2.9805855894624440e-02 4.3502677346194965e-02 4.7426725365729533e-02 5.3710757378925465e-02 6.2797617369385411e-02 6.8714814059148324e-02 7.6075981759246930e-02 8.0543504060975143e-02 9.0742583177356201e-02 1.0571678684068528e-01 1.1202417780225085e-01 1.1622977423060403e-01 1.2071505888504254e-01 1.3141047001852843e-01 1.3642706211485034e-01 1.3882287416996306e-01 1.4089055727029007e-01 1.4347231788511447e-01 1.4890062748763838e-01 1.5392887232504460e-01 1.6458457312368394e-01 1.6999277437489285e-01 1.7439450528282893e-01 1.8135077373572689e-01 1.9191613434721402e-01 1.9907690618972165e-01 2.0636287815046414e-01 2.1896427853694983e-01 2.2823888475458817e-01 2.4034566912244662e-01 2.4854964939856927e-01 2.5464927714167190e-01 4.2493471205429295e-01 4.2851185152694626e-01 4.4064832814845228e-01 4.5736258205995822e-01 4.6331892590647267e-01 4.6892714839174315e-01 4.7902895761514092e-01 4.8942038635596535e-01 5.0243462398944239e-01 5.1194661341493730e-01 5.1904667789537817e-01 5.3632921696527924e-01 5.4651593944248456e-01 5.7350572278483780e-01 5.9413190620158196e-01 6.0274011685043871e-01 6.0615107217002095e-01 6.1192054489011105e-01 6.1545345280523878e-01 6.3140456749582996e-01 6.4954861771530648e-01 6.7503539963164005e-01 6.8567621358330944e-01 6.9676237746694314e-01 7.1482451925833412e-01 7.2341450054639544e-01 7.4573454757777824e-01 7.5077764004667824e-01 7.6086360295405930e-01 7.7072201661701478e-01 7.7609737861476846e-01 7.8186598771562921e-01 7.9605299507983973e-01 8.0733034510044455e-01 8.1699593686656935e-01 8.2988605370214474e-01 8.3747034317472602e-01 8.3872144621050893e-01 8.4332432024613835e-01 8.4899282518215702e-01 8.5771956081874789e-01 8.6352094748052399e-01 8.6889377491992381e-01 8.7768136666514496e-01 8.8613716944752563e-01 8.9531601913513925e-01 9.0580566192370371e-01 9.1529333167764970e-01 9.2211841553416385e-01 9.4122939281480900e-01 9.6566486778488980e-01 9.7153246956680528e-01 9.8203219683326715e-01 1.0177360042264518e+00 1.0490684286457130e+00 1.0974976528007792e+00 1.1473874617809137e+00 1.1642859026502761e+00 1.2116509187269016e+00 1.2321590391999906e+00 1.2665604261674661e+00 1.2725684625970817e+00 1.3173850202476802e+00 1.3344566273077512e+00 1.3696820715100828e+00 1.4032275848680187e+00 1.4066934922770138e+00 1.4522683232239595e+00 1.4859434634868329e+00 1.4994160378384340e+00 1.5182767451716950e+00 1.5407584835347834e+00 1.5551904002181023e+00 1.5718411887172332e+00 1.5854149676169729e+00 1.6035523896312616e+00 1.6248643113187842e+00 1.6295959660185155e+00 1.6386262242244998e+00 1.6518538728083023e+00 1.6708612449382079e+00 1.7082446818989399e+00 1.7240850952393305e+00 1.7309952793447123e+00 1.7768187846207442e+00 1.7799403646935756e+00 1.7966946016904104e+00 1.7986817518766915e+00 1.8208995571027895e+00 1.8372013232145044e+00 1.8486146070636016e+00 1.8627514421049500e+00 1.8684098765938302e+00 1.8910561096108354e+00 1.9068270583406370e+00 1.9273968288188883e+00 1.9366445871516782e+00 1.9517986017352753e+00 1.9711844780304517e+00 1.9748133079906700e+00 1.9784543231955742e+00 2.0029278940081015e+00 2.0163943715198167e+00 2.0242114756656524e+00 2.0282110266384845e+00 2.0446485567204040e+00 2.0506337590043184e+00 2.0622360718407018e+00 2.0764528328463112e+00 2.0982719010479896e+00 2.1124509195415229e+00 2.1473844095863899e+00 2.1546267202019820e+00 2.1669075167091925e+00 2.1723427463382512e+00 2.1811757011113042e+00 2.1987630017831785e+00 2.2110769871906704e+00 2.2189959413675835e+00 2.2523882270946554e+00 2.2601867182536926e+00 2.2680781760112536e+00 2.2959489188668827e+00 2.3105475471568120e+00 2.3159945969413989e+00 2.3268620433861074e+00 2.3486729796987103e+00 2.3828969309073971e+00 2.3876853759860190e+00 2.4069231194231686e+00 2.4220207900350226e+00 2.4322427568743290e+00 2.4627682578986847e+00 2.4929223015120900e+00 2.5133883875349854e+00 2.5312880356365914e+00 2.5380554046631825e+00 2.5674716654780778e+00 2.5816435389163557e+00 2.5894762032807774e+00 2.6092020143611041e+00 2.6302177174833363e+00 2.6355319705984823e+00 2.6434882100204957e+00 2.6604218865324785e+00 2.6727749979186739e+00 2.6917620788808696e+00 2.6952997280663236e+00 2.7073943192439862e+00 2.7113430298838037e+00 2.7285879153821773e+00 2.7487932106016753e+00 2.7749377073323349e+00 2.7823799816919839e+00 2.7848199307417638e+00 2.7958608220036849e+00 2.8014818217484256e+00 2.8080216831783305e+00 2.8118711270951047e+00 2.8150035616433464e+00 2.8202130279837041e+00 2.8419227985166580e+00 2.8601401507184305e+00 2.8723563155111278e+00 2.9059065752014646e+00 3.0865702800717769e+00 3.1217735371578956e+00 3.1464493193581116e+00 3.1676580405212769e+00 3.1811714301819114e+00 3.1906306283765531e+00 3.1946230916664633e+00 3.2130867125886793e+00 3.2280265646609783e+00 3.2455319088108858e+00 3.2648013869433909e+00 3.2857855810527092e+00 3.3101643538895371e+00 3.3528874734145844e+00 3.3823787880194054e+00 3.3896325761751194e+00 3.3912089879022735e+00 3.4239218421938657e+00 3.4639876952600530e+00 3.4735427201728499e+00 3.4830955915546586e+00 3.4844494961986330e+00 3.8198802524409041e+00 4.1566349269402991e+00 4.2134221438014716e+00 4.2755969057884062e+00 4.3733062554011015e+00 4.4240204790734863e+00 4.4487042219417159e+00 4.5000884822941307e+00 4.5662887855876724e+00 4.6275754522083838e+00 4.7059780253464272e+00 ";
  dftcart.set_string("Method","lda_x-lda_c_vwn_rpa");
  restr_test(decanol,b6_31Gpp,dftcart,Etot,Eorb,"1-decanol, SVWN(RPA)/6-31G**",dip,true);


#ifndef COMPUTE_REFERENCE
  printf("****** Tests completed in %s *******\n",t.elapsed().c_str());
#endif

  return 0;
}
