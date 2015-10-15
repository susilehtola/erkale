/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "../chebyshev.h"
#include "../timer.h"
#include <armadillo>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

arma::mat e_os_MP2(const arma::mat & Bph, const arma::mat & delta) {
  // Calculate \f$ \sum_{ai} L_{ai}^P L_{ai}^Q \f$
  arma::mat e(Bph.n_rows,Bph.n_rows);
  for(size_t P=0;P<Bph.n_rows;P++)
    for(size_t Q=0;Q<Bph.n_rows;Q++) {
      double el=0.0;
      for(size_t a=0;a<delta.n_cols;a++)
	for(size_t i=0;i<delta.n_rows;i++) {
	  //printf("% e % e % e\n",Bph(P,i*delta.n_cols+a),Bph(P,a*delta.n_rows+i),Bph(P,i*delta.n_cols+a)-Bph(P,a*delta.n_rows+i));
	  el+=Bph(P,i*delta.n_cols+a)*Bph(Q,i*delta.n_cols+a)*std::pow(delta(i,a),2);
	}
      e(P,Q)=el;
    }

  return e;
}

double E_os_MP2_Laplace(const arma::mat & Bpaha, const arma::mat & Bpbhb, const arma::vec & Eoa, const arma::vec & Eob, const arma::vec & Eva, const arma::vec & Evb) {
  // Get integral nodes and weights
  static const size_t N=15;
  std::vector<double> xt, wt;
  radial_chebyshev(N,xt,wt);

  // Energy
  double E=0.0;
#ifdef _OPENMP
#pragma omp parallel reduction(+:E)
#endif
  {
    // Energy difference matrix
    arma::mat delta;

#ifdef _OPENMP
#pragma omp for ordered
#endif
    for(size_t it=0;it<N;it++) {
      // Difference matrix
      delta.zeros(Eoa.n_elem,Eva.n_elem);
      for(size_t i=0;i<Eoa.n_elem;i++)
	for(size_t a=0;a<Eva.n_elem;a++)
	  delta(i,a)=exp((Eoa(i)-Eva(a))*xt[it]);
      arma::mat aa(e_os_MP2(Bpaha,delta));

      delta.zeros(Eob.n_elem,Evb.n_elem);
      for(size_t i=0;i<Eob.n_elem;i++)
	for(size_t a=0;a<Evb.n_elem;a++)
	  delta(i,a)=exp((Eob(i)-Evb(a))*xt[it]);
      arma::mat bb(e_os_MP2(Bpbhb,delta));

      // Correlation energy is (why do I need a factor 2?)
      double Ep(-2.0*wt[it]*arma::dot(aa,bb));
#ifdef _OPENMP
#pragma omp ordered
#endif
      printf("Laplace point %2i: correlation energy % .10f\n",(int) it+1,Ep);

      // Increment energy
      E+=Ep;
    }
  }

  //printf("Numeric  OS energy %e\n",E);

  return E;
}

double E_os_MP2(const arma::mat & Bpaha, const arma::mat & Bpbhb, const arma::vec & Eoa, const arma::vec & Eob, const arma::vec & Eva, const arma::vec & Evb) {
  // Energy
  double E=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:E) collapse(4)
#endif
  for(size_t i=0;i<Eoa.n_elem;i++)
    for(size_t j=0;j<Eob.n_elem;j++)
      for(size_t a=0;a<Eva.n_elem;a++)
	for(size_t b=0;b<Evb.n_elem;b++) {
	  // Get two-electron integral
	  double tei=arma::dot(Bpaha.col(i*Eva.n_elem+a),Bpbhb.col(j*Evb.n_elem+b));
	  // Increment result
	  E-=tei*tei/(Eva(a)+Evb(b)-Eoa(i)-Eob(j));
	}

  //  printf("Analytic OS energy %e\n",E);

  return E;
}

double E_ss_MP2(const arma::mat & Bph, const arma::vec & Eo, const arma::vec & Ev) {
  // Energy
  double E=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:E)
#endif
  for(size_t i=0;i<Eo.n_elem;i++)
    for(size_t j=0;j<i;j++)
      // Antisymmetric wrt i and j
      for(size_t a=0;a<Ev.n_elem;a++)
	for(size_t b=0;b<a;b++) {
	  // Antisymmetric wrt a and b

	  // Get two-electron integral
	  double tei=arma::dot(Bph.col(i*Ev.n_elem+a),Bph.col(j*Ev.n_elem+b)) - arma::dot(Bph.col(j*Ev.n_elem+a),Bph.col(i*Ev.n_elem+b));
	  // Increment result
	  E-=tei*tei/(Ev(a)+Ev(b)-Eo(i)-Eo(j));

	  //printf("%2i %2i %2i %2i %e\n",(int) i, (int) j, (int) a, (int) b, -tei*tei/(Ev(a)+Ev(b)-Eo(i)-Eo(j)));
	}

  //printf("Analytic SS energy %e\n",E);

  return E;
}

double E_singles_MP2(const arma::mat & Fph, const arma::vec & Eo, const arma::vec & Ev) {
  double E=0.0;
  for(size_t i=0;i<Eo.n_elem;i++)
    for(size_t a=0;a<Ev.n_elem;a++)
      E+=std::pow(Fph(a,i),2)/(Eo(i)-Ev(a));

  return E;
}

int main(void) {
#ifdef _OPENMP
  printf("ERKALE - RIMP2 from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - RIMP2 from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  arma::file_type atype = arma::arma_binary;

  Timer t, ttot;

  // Fock and B matrices; alpha spin
  arma::mat Fhaha, Fpaha, Fpapa;
  arma::mat Bpaha;
  arma::vec Eref;
  
  Fhaha.load("Fhaha.dat",atype);
  Fpaha.load("Fpaha.dat",atype);
  Fpapa.load("Fpapa.dat",atype);
  Bpaha.load("Bpaha.dat",atype);
  Eref.load("Eref.dat",atype);

  // Sanity check
  if(Fhaha.n_rows != Fhaha.n_cols) throw std::runtime_error("Fhaha is not square!\n");
  if(Fpapa.n_rows != Fpapa.n_cols) throw std::runtime_error("Fpapa is not square!\n");
  if(Fpaha.n_rows != Fpapa.n_cols) throw std::runtime_error("Fpaha has wrong amount of rows!\n");
  if(Fhaha.n_rows != Fpaha.n_cols) throw std::runtime_error("Fpaha has wrong amount of columns!\n");
  if(Bpaha.n_cols != Fpapa.n_cols*Fhaha.n_cols) throw std::runtime_error("Bpaha does not correspond to F!\n");
    
  // Beta spin
  bool pol;
  arma::mat Fhbhb, Fpbhb, Fpbpb;
  arma::mat Bpbhb;
  try {
    if(!Fhbhb.quiet_load("Fhbhb.dat",atype)) throw std::runtime_error("Fhbhb does not exist!\n");
    if(!Fpbhb.quiet_load("Fpbhb.dat",atype)) throw std::runtime_error("Fpbhb does not exist!\n");
    if(!Fpbpb.quiet_load("Fpbpb.dat",atype)) throw std::runtime_error("Fpbpb does not exist!\n");
    if(!Bpbhb.quiet_load("Bpbhb.dat",atype)) throw std::runtime_error("Bpbhb does not exist!\n");
    pol=true;
  } catch(std::runtime_error) {
    pol=false;
  }

  // Sanity check
  if(pol) {
    if(Fhbhb.n_rows != Fhbhb.n_cols) throw std::runtime_error("Fhbhb is not square!\n");
    if(Fpbpb.n_rows != Fpbpb.n_cols) throw std::runtime_error("Fpbpb is not square!\n");
    if(Fpbhb.n_rows != Fpbpb.n_cols) throw std::runtime_error("Fpbhb has wrong amount of rows!\n");
    if(Fhbhb.n_rows != Fpbhb.n_cols) throw std::runtime_error("Fpbhb has wrong amount of columns!\n");
    if(Bpbhb.n_cols != Fpbpb.n_cols*Fhbhb.n_cols) throw std::runtime_error("Bpbhb does not correspond to F!\n");
  }
  
  printf("Matrices loaded in %s.\n\n",t.elapsed().c_str());
  if(pol) {
    printf("alpha: %i occupied, %i virtual orbitals\n",Fhaha.n_rows,Fpapa.n_rows);
    printf("beta : %i occupied, %i virtual orbitals\n",Fhbhb.n_rows,Fpbpb.n_rows);
  } else {
    printf("%i occupied and %i virtual orbitals.\n",Fhaha.n_rows,Fpapa.n_rows);
  }
  printf("\n");

  fflush(stdout);
  t.set();

  // Singles energies
  double Es;
  if(pol)
    Es=E_singles_MP2(Fpaha,arma::diagvec(Fhaha),arma::diagvec(Fpapa)) + \
       E_singles_MP2(Fpbhb,arma::diagvec(Fhbhb),arma::diagvec(Fpbpb));
  else
    Es=2.0*E_singles_MP2(Fpaha,arma::diagvec(Fhaha),arma::diagvec(Fpapa));
  
  // Correlation energies
  double Eaa=0.0, Eab=0.0, Ebb=0.0;
  bool osonly=false;
  if(osonly) {
    if(pol)
      Eab=E_os_MP2_Laplace(Bpaha,Bpbhb,arma::diagvec(Fhaha),arma::diagvec(Fhbhb),arma::diagvec(Fpapa),arma::diagvec(Fpbpb));
    else
      Eab=E_os_MP2_Laplace(Bpaha,Bpaha,arma::diagvec(Fhaha),arma::diagvec(Fhaha),arma::diagvec(Fpapa),arma::diagvec(Fpapa));

      printf("Laplace OS MP2 done in %s.\n",t.elapsed().c_str());
      fflush(stdout);
      t.set();
  } else {
    if(!pol) {
      Ebb=Eaa=E_ss_MP2(Bpaha,arma::diagvec(Fhaha),arma::diagvec(Fpapa));
      printf("Same-spin     MP2 done in %s.\n",t.elapsed().c_str());
      fflush(stdout);
      t.set();

      Eab=E_os_MP2(Bpaha,Bpaha,arma::diagvec(Fhaha),arma::diagvec(Fhaha),arma::diagvec(Fpapa),arma::diagvec(Fpapa));
      printf("Opposite-spin MP2 done in %s.\n",t.elapsed().c_str());
      fflush(stdout);
      t.set();
    } else {
      Eaa=E_ss_MP2(Bpaha,arma::diagvec(Fhaha),arma::diagvec(Fpapa));
      printf("Same-spin alpha MP2 done in %s.\n",t.elapsed().c_str());
      fflush(stdout);
      t.set();

      Ebb=E_ss_MP2(Bpbhb,arma::diagvec(Fhbhb),arma::diagvec(Fpbpb));
      printf("Same-spin  beta MP2 done in %s.\n",t.elapsed().c_str());
      fflush(stdout);
      t.set();

      Eab=E_os_MP2(Bpaha,Bpbhb,arma::diagvec(Fhaha),arma::diagvec(Fhbhb),arma::diagvec(Fpapa),arma::diagvec(Fpbpb));
      printf("Opposite-spin   MP2 done in %s.\n",t.elapsed().c_str());
      fflush(stdout);
      t.set();
    }
  }

  printf("\n");
  if(!osonly)
    printf(" alpha-alpha correlation energy is % 16.10f\n",Eaa);
  printf("  alpha-beta correlation energy is % 16.10f\n",Eab);
  if(!osonly)
    printf("   beta-beta correlation energy is % 16.10f\n",Ebb);
  printf("   non-Brillouin singles energy is % 16.10f\n",Es);
  printf(" -----------------------------------------------\n");
  printf(" Total       correlation energy is % 16.10f\n",Eaa+Eab+Ebb+Es);

  printf("\n Total energy is % .10f\n",Eref(0)+Eaa+Eab+Ebb+Es);

  printf("\nCalculation finished in %s.\n",ttot.elapsed().c_str());

  return 0;
}
