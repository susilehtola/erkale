/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "atomtable.h"
#include "solvers.h"
#include "../linalg.h"
#include "../timer.h"
#include "../diis.h"
#include "../guess.h"
#include "../mathf.h"

void form_density(const arma::mat & Ca, const arma::mat & Cb, arma::mat & Pa, arma::mat & Pb, int Z) {
  // Get ground state
  gs_conf_t gs=get_ground_state(Z);

  // Calculate amount of alpha and beta electrons
  int Nel_alpha;
  int Nel_beta;
  get_Nel_alpha_beta(Z,gs.mult,Nel_alpha,Nel_beta);

  // Get occupancies
  std::vector<double> occa=atomic_occupancy(Nel_alpha,Ca.n_rows);
  std::vector<double> occb=atomic_occupancy(Nel_beta,Cb.n_rows);

  // Form density matrix
  size_t Nbf=Ca.n_rows;
  Pa.zeros(Nbf,Nbf);
  Pb.zeros(Nbf,Nbf);

  for(size_t i=0;i<occa.size();i++)
    Pa+=occa[i]*Ca.col(i)*arma::trans(Ca.col(i));
  for(size_t i=0;i<occb.size();i++)
    Pb+=occb[i]*Cb.col(i)*arma::trans(Cb.col(i));
}

void form_density(const arma::mat & C, arma::mat & P, int Z) {
  // Get ground state
  gs_conf_t gs=get_ground_state(Z);
  if(gs.mult!=1)
    throw std::runtime_error("Not creating restricted density for unrestricted system!\n");

  // Form density matrix
  size_t Nbf=C.n_rows;
  P.zeros(Nbf,Nbf);

  std::vector<double> occs=atomic_occupancy(Z/2,C.n_rows);
  for(size_t i=0;i<occs.size();i++)
    P+=2*occs[i]*C.col(i)*arma::trans(C.col(i));
}

void print_E(const arma::vec & Ea, const arma::vec & Eb, int Z) {
  gs_conf_t gs=get_ground_state(Z);

  // Calculate amount of alpha and beta electrons
  int Nel_alpha;
  int Nel_beta;
  get_Nel_alpha_beta(Z,gs.mult,Nel_alpha,Nel_beta);

  // Get occupancies
  std::vector<double> occa=atomic_occupancy(Nel_alpha,Ea.n_elem);
  std::vector<double> occb=atomic_occupancy(Nel_beta,Eb.n_elem);

  printf("\nAlpha orbital energies\n");
  for(size_t i=0;i<occa.size();i++)
    if(occa[i]==1.0)
      printf("% .6f* ",Ea[i]);
    else if(occa[i]>0)
      printf("% .6f+ ",Ea[i]);
    else
      printf("% .6f  ",Ea[i]);
  for(size_t i=occa.size();i<Ea.size();i++)
    printf("% .6f  ",Ea[i]);
  printf("\n");

  printf("Beta orbital energies:\n");
  for(size_t i=0;i<occb.size();i++)
    if(occb[i]==1.0)
      printf("% .6f* ",Eb[i]);
    else if(occb[i]>0)
      printf("% .6f+ ",Eb[i]);
    else
      printf("% .6f  ",Eb[i]);
  for(size_t i=occb.size();i<Eb.size();i++)
    printf("% .6f  ",Eb[i]);
  printf("\n");

  fflush(stdout);
}

void print_E(const arma::vec & E, int Z) {
  gs_conf_t gs=get_ground_state(Z);

  // Calculate amount of alpha and beta electrons
  int Nel_alpha;
  int Nel_beta;
  get_Nel_alpha_beta(Z,gs.mult,Nel_alpha,Nel_beta);

  // Get occupancies
  std::vector<double> occs=atomic_occupancy(Nel_alpha,E.n_elem);

  printf("\nOrbital energies\n");
  for(size_t i=0;i<occs.size();i++)
    if(occs[i]==1.0)
      printf("% .6f* ",E[i]);
    else if(occs[i]>0)
      printf("% .6f+ ",E[i]);
    else
      printf("% .6f  ",E[i]);

  for(size_t i=occs.size();i<E.size();i++)
    printf("% .6f  ",E[i]);
  printf("\n");

  fflush(stdout);
}


void UHF(const std::vector<bf_t> & basis, int Z, uscf_t & sol, double convthr, bool direct, bool ROHF, bool verbose) {
  // Amount of basis function
  size_t Nbf=basis.size();

  // Construct matrices
  arma::mat S=overlap(basis);
  arma::mat T=kinetic(basis);
  arma::mat V=nuclear(basis,Z);
  arma::mat Hcore=T+V;

  // Inverse overlap
  arma::mat Sinvh=BasOrth(S,verbose);
  if(Sinvh.n_cols!=Sinvh.n_rows && verbose) {
    printf("%i nondegenerate basis functions.\n",(int) Sinvh.n_cols);
    fflush(stdout);
  }

  // Integrals
  AtomTable tab;
  if(!direct) {
    tab.fill(basis,verbose);
  }

  // Get core guess
  sol.Ha=Hcore;
  sol.Hb=Hcore;
  diagonalize(S,Sinvh,sol);

  arma::mat Paold, Pbold, Pold;
  form_density(sol.Ca,sol.Cb,sol.Pa,sol.Pb,Z);
  sol.P=sol.Pa+sol.Pb;

  size_t iiter=0;
  double Pmax=0;
  double Prms;
  double oldE;

  if(verbose) {
    printf("Entering SCF loop, basis contains %i functions.\n",(int) Nbf);
    fflush(stdout);
  }

  bool usediis=true;
  const double diiseps=0.1;
  const double diisthr=0.01;
  bool useadiis=true;
  bool diiscomb=false;
  const int diisorder=10;

  uDIIS diis(S,Sinvh,diiscomb,usediis,diiseps,diisthr,useadiis,diisorder,verbose);
  double diiserr;

  arma::mat oldHa, oldHb;
  sol.Ha.zeros();
  sol.Hb.zeros();

  do {
    Timer t;

    iiter++;

    // Form Coulomb and exchange operators
    if(!direct) {
      sol.J=tab.calcJ(sol.P);
      sol.Ka=tab.calcK(sol.Pa);
      sol.Kb=tab.calcK(sol.Pb);
    } else {
      sol.J=coulomb(basis,sol.P);
      sol.Ka=::exchange(basis,sol.Pa);
      sol.Kb=::exchange(basis,sol.Pb);
    }

    oldHa=sol.Ha;
    oldHb=sol.Hb;
    sol.Ha=Hcore+sol.J-sol.Ka;
    sol.Hb=Hcore+sol.J-sol.Kb;

    if(ROHF) {
      // Get ground state
      gs_conf_t gs=get_ground_state(Z);

      // Calculate amount of alpha and beta electrons
      int Nel_alpha;
      int Nel_beta;
      get_Nel_alpha_beta(Z,gs.mult,Nel_alpha,Nel_beta);
      std::vector<double> occa=atomic_occupancy(Nel_alpha,sol.P.n_rows);
      std::vector<double> occb=atomic_occupancy(Nel_beta,sol.P.n_rows);
      ROHF_update(sol.Ha,sol.Hb,sol.P,S,occa,occb,verbose);
    }

    // Calculate energy
    sol.en.Exc=-0.5*(arma::trace(sol.Pa*sol.Ka)+arma::trace(sol.Pb*sol.Kb));
    sol.en.Ekin=arma::trace(sol.P*T);
    sol.en.Enuca=arma::trace(sol.P*V);
    sol.en.Eone=arma::trace(sol.P*Hcore);
    sol.en.Ecoul=0.5*arma::trace(sol.P*sol.J);

    oldE=sol.en.E;
    sol.en.Eel=sol.en.Eone+sol.en.Ecoul+sol.en.Exc;
    sol.en.Enucr=0.0;
    sol.en.E=sol.en.Eel;

    //    printf("Exc=%e, Ekin=%e, Enuc=%e, Ecoul=%e\n",Exc,Ekin,Enuc,Ecoul);

    // Update DIIS stacks
    diis.update(sol.Ha,sol.Hb,sol.Pa,sol.Pb,sol.en.E,diiserr);
    // and solve the updated matrices
    diis.solve_F(sol.Ha,sol.Hb);

    // Solve new orbitals
    diagonalize(S,Sinvh,sol);

    // Form density matrix
    Paold=sol.Pa;
    Pbold=sol.Pb;
    Pold=sol.P;
    form_density(sol.Ca,sol.Cb,sol.Pa,sol.Pb,Z);
    sol.P=sol.Pa+sol.Pb;

    // Check for convergence
    Pmax=max_abs(sol.P-Pold);
    Prms=rms_norm(sol.P-Pold);

    if(verbose) {
      printf("%3i\t%.12f\t%e\t%e\t%e (%s)\n",(int) iiter,sol.en.E,sol.en.E-oldE,Pmax,Prms,t.elapsed().c_str());
      fflush(stdout);
    }
  } while(diiserr>=convthr);

  if(verbose) print_E(sol.Ea,sol.Eb,Z);
}

void RHF(const std::vector<bf_t> & basis, int Z, rscf_t & sol, double convthr, bool direct, bool verbose) {
  // Amount of basis functions
  size_t Nbf=basis.size();

  // Construct matrices
  arma::mat S=overlap(basis);
  arma::mat T=kinetic(basis);
  arma::mat V=nuclear(basis,Z);
  arma::mat Hcore=T+V;

  // Inverse overlap
  arma::mat Sinvh=BasOrth(S,verbose);
  if(Sinvh.n_cols!=Sinvh.n_rows && verbose) {
    printf("%i nondegenerate basis functions.\n",(int) Sinvh.n_cols);
    fflush(stdout);
  }

  // Get core guess
  sol.H=Hcore;
  diagonalize(S,Sinvh,sol);

  // Integrals
  AtomTable tab;
  if(!direct) {
    tab.fill(basis,verbose);
  }

  arma::mat Pold;
  form_density(sol.C,sol.P,Z);

  size_t iiter=0;
  double Pmax=0;
  double Prms;
  double oldE;

  if(verbose) {
    printf("Entering SCF loop, basis contains %i functions.\n",(int) Nbf);
    fflush(stdout);
  }

  bool usediis=true;
  const double diiseps=0.01;
  const double diisthr=0.01;
  bool useadiis=true;
  const int diisorder=10;
  rDIIS diis(S,Sinvh,usediis,diiseps,diisthr,useadiis,diisorder,verbose);
  double diiserr;

  arma::mat oldH;
  sol.H.zeros(Nbf,Nbf);

  do {
    Timer t;

    iiter++;

    // Form Coulomb and exchange operators
    if(!direct) {
      sol.J=tab.calcJ(sol.P);
      sol.K=tab.calcK(sol.P);

      /*
      arma::mat Jr=coulomb(basis,sol.P);
      arma::mat Kr=exchange(basis,sol.P);
      (sol.J-Jr).print("J diff");
      (sol.K-Kr).print("K diff");
      */
    } else {
      sol.J=coulomb(basis,sol.P);
      sol.K=::exchange(basis,sol.P);
    }

    oldH=sol.H;
    sol.H=Hcore+sol.J-0.5*sol.K;

    // Calculate energy
    sol.en.Ekin=arma::trace(sol.P*T);
    sol.en.Enuca=arma::trace(sol.P*V);
    sol.en.Eone=arma::trace(sol.P*Hcore);

    sol.en.Ecoul=0.5*arma::trace(sol.P*sol.J);
    sol.en.Exc=-0.25*arma::trace(sol.P*sol.K);

    oldE=sol.en.E;
    sol.en.Enucr=0.0;
    sol.en.E=sol.en.Eone+sol.en.Ecoul+sol.en.Exc;

    //    printf("Exc=%e, Ekin=%e, Enuc=%e, Ecoul=%e\n",Exc,Ekin,Enuc,Ecoul);

    // Update DIIS stacks
    diis.update(sol.H,sol.P,sol.en.E,diiserr);
    // and solve for the new matrices
    diis.solve_F(sol.H);

    // Solve new orbitals
    diagonalize(S,Sinvh,sol);

    // Form density matrix
    Pold=sol.P;
    form_density(sol.C,sol.P,Z);

    // Check for convergence
    Pmax=max_abs(sol.P-Pold);
    Prms=rms_norm(sol.P-Pold);

    if(verbose) {
      printf("%3i\t%.12f\t%e\t%e\t%e (%s)\n",(int) iiter,sol.en.E,sol.en.E-oldE,Pmax,Prms,t.elapsed().c_str());
      fflush(stdout);
    }
  } while(diiserr>=convthr);

  if(verbose) print_E(sol.E,Z);
}

