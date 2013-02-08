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

#include "solvers.h"
#include "linalg.h"
#include "timer.h"
#include "diis.h"
#include "adiis.h"
#include "guess.h"
#include "mathf.h"

void diagonalize(const arma::mat & H, const arma::mat & Sinvh, arma::mat & C, arma::vec & E) {
  arma::mat Horth;
  arma::mat orbs;
  // Transform Hamiltonian into orthogonal basis
  Horth=arma::trans(Sinvh)*H*Sinvh;
  // Update orbitals and energies
  eig_sym_ordered(E,orbs,Horth);
  // Transform back to non-orthogonal basis
  C=Sinvh*orbs;
}

void form_density(const arma::mat & Ca, const arma::mat & Cb, arma::mat & Pa, arma::mat & Pb, int Z) {
  // Get ground state
  gs_conf_t gs=get_ground_state(Z);

  // Calculate amount of alpha and beta electrons
  int Nel_alpha;
  int Nel_beta;
  get_Nel_alpha_beta(Z,gs.mult,Nel_alpha,Nel_beta);

  // Get occupancies
  std::vector<double> occa=atomic_occupancy(Nel_alpha);
  std::vector<double> occb=atomic_occupancy(Nel_beta);

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

  std::vector<double> occs=atomic_occupancy(Z/2);
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
  std::vector<double> occa=atomic_occupancy(Nel_alpha);
  std::vector<double> occb=atomic_occupancy(Nel_beta);

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
}

void print_E(const arma::vec & E, int Z) {
  gs_conf_t gs=get_ground_state(Z);

  // Calculate amount of alpha and beta electrons
  int Nel_alpha;
  int Nel_beta;
  get_Nel_alpha_beta(Z,gs.mult,Nel_alpha,Nel_beta);

  // Get occupancies
  std::vector<double> occs=atomic_occupancy(Nel_alpha);

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
}


void UHF(const std::vector<bf_t> & basis, int Z, uscf_t & sol, const convergence_t conv, bool ROHF, bool verbose) {
  // Construct matrices
  arma::mat S=overlap(basis);
  arma::mat T=kinetic(basis);
  arma::mat V=nuclear(basis,Z);
  arma::mat Hcore=T+V;

  // Inverse overlap
  arma::mat Sinvh=BasOrth(S,true);
  if(Sinvh.n_cols!=Sinvh.n_rows && verbose)
    printf("%i nondegenerate basis functions.\n",Sinvh.n_cols);

  // Get core guess
  diagonalize(Hcore,Sinvh,sol.Ca,sol.Ea);
  sol.Cb=sol.Ca;
  sol.Eb=sol.Ea;

  size_t Nbf=basis.size();
  arma::mat Paold, Pbold, Pold;
  form_density(sol.Ca,sol.Cb,sol.Pa,sol.Pb,Z);
  sol.P=sol.Pa+sol.Pb;

  size_t iiter=0;
  double Pmax=0;
  double Prms;
  double oldE;

  if(verbose) printf("Entering SCF loop, basis contains %i functions.\n",(int) Nbf);

  DIIS diisa(S), diisb(S);
  const double diisthr=0.05;
  ADIIS adiisa, adiisb;

  arma::mat oldHa, oldHb;
  sol.Ha.zeros();
  sol.Hb.zeros();

  do {
    Timer t;

    iiter++;

    // Form Coulomb and exchange operators
    sol.J=coulomb(basis,sol.P);
    sol.Ka=exchange(basis,sol.Pa);
    sol.Kb=exchange(basis,sol.Pb);

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
      ROHF_update(sol.Ha,sol.Hb,sol.P,S,Nel_alpha,Nel_beta,verbose,true);
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

    // Update ADIIS stacks
    adiisa.push(sol.en.E,sol.Pa,sol.Ha);
    adiisb.push(sol.en.E,sol.Pb,sol.Hb);

    // Update DIIS stacks
    double diiserra, diiserrb, diiserr;
    diisa.update(sol.Ha,sol.Pa,diiserra);
    diisb.update(sol.Hb,sol.Pb,diiserrb);
    diiserr=std::max(diiserra,diiserrb);

    if(diiserr<diisthr) {
      diisa.solve(sol.Ha);
      diisb.solve(sol.Hb);
    } else {
      sol.Ha=adiisa.get_H();
      sol.Hb=adiisb.get_H();
    }

    // Solve new orbitals
    diagonalize(sol.Ha,Sinvh,sol.Ca,sol.Ea);
    diagonalize(sol.Hb,Sinvh,sol.Cb,sol.Eb);

    // Form density matrix
    Paold=sol.Pa;
    Pbold=sol.Pb;
    Pold=sol.P;
    form_density(sol.Ca,sol.Cb,sol.Pa,sol.Pb,Z);
    sol.P=sol.Pa+sol.Pb;

    // Check for convergence
    Pmax=max_abs(sol.P-Pold);
    Prms=rms_norm(sol.P-Pold);

    char econv='*';
    char pmax='*';
    char prms='*';
    if(fabs(sol.en.E-oldE)>conv.deltaEmax)
      econv=' ';
    if(Pmax>conv.deltaPmax)
      pmax=' ';
    if(Prms>conv.deltaPrms)
      prms=' ';

    if(verbose) printf("%3i\t%.12f\t%e%c\t%e%c\t%e%c (%s)\n",(int) iiter,sol.en.E,sol.en.E-oldE,econv,Pmax,pmax,Prms,prms,t.elapsed().c_str());
  } while(fabs(sol.en.E-oldE)>conv.deltaEmax || Pmax>conv.deltaPmax || Prms>conv.deltaPrms);

  if(verbose) print_E(sol.Ea,sol.Eb,Z);
}

void RHF(const std::vector<bf_t> & basis, int Z, rscf_t & sol, const convergence_t conv, bool verbose) {
  // Construct matrices
  arma::mat S=overlap(basis);
  arma::mat T=kinetic(basis);
  arma::mat V=nuclear(basis,Z);
  arma::mat Hcore=T+V;

  // Inverse overlap
  arma::mat Sinvh=BasOrth(S,true);
  if(Sinvh.n_cols!=Sinvh.n_rows && verbose)
    printf("%i nondegenerate basis functions.\n",Sinvh.n_cols);

  // Get core guess
  diagonalize(Hcore,Sinvh,sol.C,sol.E);

  size_t Nbf=basis.size();
  arma::mat Pold;
  form_density(sol.C,sol.P,Z);

  size_t iiter=0;
  double Pmax=0;
  double Prms;
  double oldE;

  if(verbose) printf("Entering SCF loop, basis contains %i functions.\n",(int) Nbf);

  ADIIS adiis;
  DIIS diis(S);
  const double diisthr=0.05;

  arma::mat oldH;
  sol.H.zeros(Nbf,Nbf);

  do {
    Timer t;

    iiter++;

    // Form Coulomb and exchange operators
    sol.J=coulomb(basis,sol.P);
    sol.K=exchange(basis,sol.P);

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

    // Update ADIIS stacks
    adiis.push(sol.en.E,sol.P,sol.H);

    // Update DIIS stacks
    double diiserr;
    diis.update(sol.H,sol.P,diiserr);

    if(diiserr<diisthr) {
      diis.solve(sol.H);
    } else {
      sol.H=adiis.get_H();
    }

    // Solve new orbitals
    diagonalize(sol.H,Sinvh,sol.C,sol.E);

    // Form density matrix
    Pold=sol.P;
    form_density(sol.C,sol.P,Z);

    // Check for convergence
    Pmax=max_abs(sol.P-Pold);
    Prms=rms_norm(sol.P-Pold);

    char econv='*';
    char pmax='*';
    char prms='*';
    if(fabs(sol.en.E-oldE)>conv.deltaEmax)
      econv=' ';
    if(Pmax>conv.deltaPmax)
      pmax=' ';
    if(Prms>conv.deltaPrms)
      prms=' ';

    if(verbose) printf("%3i\t%.12f\t%e%c\t%e%c\t%e%c (%s)\n",(int) iiter,sol.en.E,sol.en.E-oldE,econv,Pmax,pmax,Prms,prms,t.elapsed().c_str());
  } while(fabs(sol.en.E-oldE)>conv.deltaEmax || Pmax>conv.deltaPmax || Prms>conv.deltaPrms);

  if(verbose) print_E(sol.E,Z);
}

