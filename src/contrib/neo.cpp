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

#include "basislibrary.h"
#include "basis.h"
#include "checkpoint.h"
#include "dftgrid.h"
#include "elements.h"
#include "find_molecules.h"
#include "guess.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
#include "sap.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"

// Needed for libint init
#include "eriworker.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

//#include "openorbital/scfsolver.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

void print_header() {
#ifdef _OPENMP
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

Settings settings;

std::pair<arma::vec,arma::mat> diagonalize(const arma::mat & F) {
  arma::vec E;
  arma::mat C;
  arma::eig_sym(E,C,F);
  return std::make_pair(E,C);
}

std::pair<arma::vec,arma::mat> diagonalize(const arma::mat & F, const arma::mat & X) {
  arma::mat Fo(X.t() * F * X);
  auto [E, C] = diagonalize(Fo);
  C = X*C;
  return std::make_pair(E,C);
}

int main_guarded(int argc, char **argv) {
  print_header();

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  Timer t;
  t.print_time();

  settings.add_scf_settings();
  settings.add_string("ProtonBasis", "Protonic basis set", "");
  settings.add_int("StepwiseSCFIter", "Number of stepwise SCF macroiterations, 0 to skip to simultaneous solution", 0);
  settings.add_string("QuantumProtons", "Indices of protons to make quantum", "");
  settings.add_string("ErrorNorm", "Error norm to use in the SCF code", "rms");
  settings.add_double("ProtonMass", "Protonic mass", 1836.15267389);
  settings.add_double("InitConvThr", "Initialization convergence threshold", 1e-5);
  settings.add_int("Verbosity", "Verboseness level", 5);
  settings.add_int("MaxInitIter", "Maximum number of iterations in the stepwise solutions", 50);
  settings.add_string("SaveChk", "Checkpoint file to save to", "neo.chk");
  settings.add_string("LoadChk", "Checkpoint file to load from", "");

  // Parse settings
  settings.parse(std::string(argv[1]),true);
  settings.print();
  int Q = settings.get_int("Charge");
  int M = settings.get_int("Multiplicity");
  int verbosity = settings.get_int("Verbosity");
  int maxiter = settings.get_int("MaxIter");
  int maxinititer = settings.get_int("MaxInitIter");
  int diisorder = settings.get_int("DIISOrder");
  double proton_mass = settings.get_double("ProtonMass");
  double intthr = settings.get_double("IntegralThresh");
  double init_convergence_threshold = settings.get_double("InitConvThr");
  double convergence_threshold = settings.get_double("ConvThr");
  bool verbose = settings.get_bool("Verbose");
  int nstepwise = settings.get_int("StepwiseSCFIter");
  size_t fitmem = 1000000*settings.get_int("FittingMemory");
  std::string error_norm = settings.get_string("ErrorNorm");
  std::string loadchk = settings.get_string("LoadChk");
  std::string savechk = settings.get_string("SaveChk");

  Checkpoint chkpt(savechk,true);

  // Read in basis set
  BasisSetLibrary baslib;
  baslib.load_basis(settings.get_string("Basis"));
  BasisSetLibrary pbaslib;
  if(settings.get_string("ProtonBasis").size())
    pbaslib.load_basis(settings.get_string("ProtonBasis"));

  // Read in SAP potential
  BasisSetLibrary potlib;
  potlib.load_basis(settings.get_string("SAPBasis"));

  auto atoms=load_xyz(settings.get_string("System"),!settings.get_bool("InputBohr"));
  std::vector<size_t> proton_indices;
  if(stricmp(settings.get_string("QuantumProtons"),"")!=0) {
    // Check for '*'
    std::string str=settings.get_string("QuantumProtons");
    if(str.size()==1 && str[0]=='*') {
      for(size_t i=0;i<atoms.size();i++)
        if(atoms[i].el.compare("H")==0)
          proton_indices.push_back(i);
    } else {
      // Parse and convert to C++ indexing
      proton_indices = parse_range(settings.get_string("QuantumProtons"),true);
    }
  }

  // Collect quantum protons
  std::vector<atom_t> quantum_protons;
  for(auto idx: proton_indices) {
    quantum_protons.push_back(atoms[idx]);
  }
  for(size_t i=0; i<quantum_protons.size(); i++)
    quantum_protons[i].num=i;

  // Collect classical nuclei
  std::vector<std::tuple<int,double,double,double>> classical_nuclei;
  for(size_t i=0;i<atoms.size();i++) {
    // Ghost nucleus
    if(atoms[i].el.size()>3 && atoms[i].el.substr(atoms[i].el.size()-3,3)=="-Bq")
      continue;
    // Skip over quantum nuclei
    bool quantum=false;
    for(auto proton_idx: proton_indices)
      if(proton_idx==i)
        quantum=true;
    if(quantum)
      continue;
    // Add to list
    classical_nuclei.push_back(std::make_tuple(get_Z(atoms[i].el), atoms[i].x, atoms[i].y, atoms[i].z));
  }

  // Construct the basis set
  BasisSet basis;
  construct_basis(basis,atoms,baslib);
  chkpt.write(basis);

  BasisSet pbasis;
  construct_basis(pbasis,quantum_protons,pbaslib);
  chkpt.write(pbasis,"pbasis");

  // Classical nucleus repulsion energy
  double Ecnucr=0.0;
  for(size_t i=0;i<classical_nuclei.size();i++) {
    auto [Qi, xi, yi, zi] = classical_nuclei[i];
    for(size_t j=0;j<i;j++) {
      auto [Qj, xj, yj, zj] = classical_nuclei[j];
      Ecnucr+=Qi*Qj/sqrt(std::pow(xi-xj,2)+std::pow(yi-yj,2)+std::pow(zi-zj,2));
    }
  }

  // Construct density fitting basis set
  BasisSetLibrary fitlib;
  fitlib.load_basis(settings.get_string("FittingBasis"));
  BasisSet dfitbas;
  {
    // Construct fitting basis
    bool uselm=settings.get_bool("UseLM");
    settings.set_bool("UseLM",true);
    construct_basis(dfitbas,basis.get_nuclei(),fitlib);
    dfitbas.coulomb_normalize();
    settings.set_bool("UseLM",uselm);
  }

  // Construct linearly independent basis
  arma::mat S(basis.overlap());
  arma::mat Sp(pbasis.overlap());

  arma::mat X(BasOrth(S));
  arma::mat Xp;
  if(Sp.n_elem)
    Xp = BasOrth(Sp);

  // Calculate matrices
  arma::mat T(basis.kinetic());
  arma::mat Vfull(basis.nuclear());
  arma::mat Vc(basis.nuclear(classical_nuclei));
  arma::mat Vsap(basis.sap_potential(potlib));
  arma::mat Vpc, Tp, Vpsap;
  std::vector<arma::mat> pr;
  if(Sp.n_elem) {
    Vpc=-pbasis.nuclear(classical_nuclei);
    Tp=pbasis.kinetic()/proton_mass;
    Vpsap=-pbasis.sap_potential(potlib);
    pr=pbasis.moment(1);
  }

  std::function<arma::mat(const arma::mat &)> extract_atomic_block_diagonal = [&pbasis, &Xp, &Sp] (const arma::mat & F) {
    arma::mat Fblock(F.n_rows,F.n_cols,arma::fill::zeros);
    for(size_t inuc=0;inuc<pbasis.get_Nnuc();inuc++) {
      // Get shells on nucleus
      auto shells = pbasis.get_shell_inds(inuc);
      // Accumulate list of functions on atom
      std::vector<size_t> idx;
      for(auto shell_idx: shells)
        for(size_t ibf=pbasis.get_first_ind(shell_idx); ibf<=pbasis.get_last_ind(shell_idx); ibf++)
          idx.push_back(ibf);
      arma::uvec idxv(arma::conv_to<arma::uvec>::from(idx));
      Fblock(idxv,idxv) = F(idxv,idxv);
    }
    return Fblock;
  };

  // Guess Fock
  arma::mat Fguess(X.t()*(T+Vfull+Vsap)*X);

  // Compute density fitting integrals
  bool direct=settings.get_bool("Direct");
  double fitthr=settings.get_double("FittingThreshold");
  double cholfitthr=settings.get_double("FittingCholeskyThreshold");

  DensityFit dfit;
  size_t Npairs_e=dfit.fill(basis,dfitbas,direct,intthr,fitthr,cholfitthr);
  DensityFit pfit;
  size_t Npairs_p=0;
  if(Sp.n_elem)
    Npairs_p=pfit.fill(pbasis,dfitbas,direct,intthr,fitthr,cholfitthr);

  printf("%i electronic shell pairs out of %i are significant.\n",(int) Npairs_e, (int) basis.get_unique_shellpairs().size());
  printf("%i protonic shell pairs out of %i are significant.\n",(int) Npairs_p, (int) pbasis.get_unique_shellpairs().size());
  printf("Electronic basis contains %i functions out of which %i are linearly dependent.\n",(int) X.n_rows, (int) (X.n_rows-X.n_cols));
  printf("Protonic basis contains %i functions out of which %i are linearly dependent.\n",(int) Xp.n_rows, (int) (Xp.n_rows-Xp.n_cols));
  printf("Auxiliary basis contains %i functions out of which %i are linearly dependent.\n",(int) dfit.get_Naux(),(int) (dfit.get_Naux()-dfit.get_Naux_indep()));
  fflush(stdout);
  if(Sp.n_elem>0 and dfit.get_Naux() != pfit.get_Naux())
    throw std::logic_error("Electronic and protonic density fitting basis sets don't have the same number of functions!\n");

  if(nstepwise==0 and quantum_protons.size()==0) {
    nstepwise=1;
  }

  arma::mat frozen_Jpe(T.n_rows, T.n_cols, arma::fill::zeros);
  arma::mat frozen_Jep(Tp.n_rows, Tp.n_cols, arma::fill::zeros);
  double frozen_Ee=0.0, frozen_Ep=0.0;

  std::function<std::pair<arma::mat,arma::mat>(const arma::mat & P, const arma::vec & occs)> electronic_terms = [&dfit, &fitmem](const arma::mat & C, const arma::vec & occs) {
    arma::mat P(C*arma::diagmat(occs)*C.t());
    arma::mat J(dfit.calcJ(P));
    arma::mat K(-dfit.calcK(C,arma::conv_to<std::vector<double>>::from(occs), fitmem));
    return std::make_pair(J,K);
  };
  std::function<std::pair<arma::mat,arma::mat>(const arma::mat & P, const arma::vec & occs)> protonic_terms = [&pfit, &fitmem](const arma::mat & C, const arma::vec & occs) {
    arma::mat P(C*arma::diagmat(occs)*C.t());
    arma::mat J(pfit.calcJ(P));
    arma::mat K(-pfit.calcK(C,arma::conv_to<std::vector<double>>::from(occs), fitmem));
    return std::make_pair(J,K);
  };
  std::function<arma::mat(const arma::mat & P)> electron_proton_coulomb = [&dfit, &pfit](const arma::mat & Pe) {
    arma::vec c(dfit.compute_expansion(Pe));
    arma::mat J=-pfit.calcJ_vector(c);
    return J;
  };
  std::function<arma::mat(const arma::mat & P)> proton_electron_coulomb = [&dfit, &pfit](const arma::mat & Pp) {
    arma::vec c(pfit.compute_expansion(Pp));
    arma::mat J=-dfit.calcJ_vector(c);
    return J;
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> restricted_builder = [&X, &T, &Vc, &dfit, electronic_terms, &Ecnucr, &verbosity, &frozen_Jpe, &frozen_Ep](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Get the electronic and protonic orbital coefficients
    arma::mat Ce = X*orbitals[0];

    // and occupations
    arma::vec occe = occupations[0];

    // Density matrices
    arma::mat Pe = Ce * arma::diagmat(occe) * Ce.t();

    // Compute the terms in the Fock matrices
    arma::mat J, K;
    std::tie(J, K) = electronic_terms(Ce, occe);
    // Form the Fock matrices
    arma::mat Fe = X.t() * (T + Vc + J + .5*K + frozen_Jpe) * X;
    std::vector<arma::mat> fock({Fe});

    // Compute energy terms
    double Ekin = arma::trace(Pe*T);
    double Enuc = arma::trace(Pe*Vc);
    double Ecoul = 0.5*arma::trace(J*Pe);
    double Eexch = 0.25*arma::trace(K*Pe);
    double Epe = arma::trace(Pe*frozen_Jpe);
    double Etot = Ekin+Enuc+Ecoul+Eexch+Ecnucr+Epe+frozen_Ep;

    if(verbosity>=10) {
      printf("e kinetic energy         % .10f\n",Ekin);
      printf("e nuclear attraction     % .10f\n",Enuc);
      printf("e-p frozen attraction    % .10f\n",Epe);
      printf("e-e Coulomb energy       % .10f\n",Ecoul);
      printf("e-e exchange energy      % .10f\n",Eexch);
      printf("nuclear repulsion energy % .10f\n",Ecnucr);
      printf("frozen proton energy     % .10f\n",frozen_Ep);
      printf("Total energy             % .10f\n",Etot);
    }

    return std::make_pair(Etot,fock);
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> unrestricted_builder = [&X, &T, &Vc, &dfit, electronic_terms, &Ecnucr, &verbosity, &frozen_Jpe, &frozen_Ep](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Get the electronic and protonic orbital coefficients
    arma::mat Ca = X*orbitals[0];
    arma::mat Cb = X*orbitals[1];

    // and occupations
    arma::vec occa = occupations[0];
    arma::vec occb = occupations[1];

    // Density matrices
    arma::mat Pa, Pb, Pp;
    Pa = Ca * arma::diagmat(occa) * Ca.t();
    Pb = Cb * arma::diagmat(occb) * Cb.t();
    arma::mat Pe = Pa+Pb;

    // Compute the terms in the Fock matrices
    arma::mat Ja, Jb, Ka, Kb, Jp, Kp, Jep, Jpe;
    std::tie(Ja, Ka) = electronic_terms(Ca, occa);
    std::tie(Jb, Kb) = electronic_terms(Cb, occb);
    // Form the Fock matrices
    arma::mat Fa = X.t() * (T + Vc + Ja + Jb + Ka + frozen_Jpe) * X;
    arma::mat Fb = X.t() * (T + Vc + Ja + Jb + Kb + frozen_Jpe) * X;
    std::vector<arma::mat> fock({Fa,Fb});

    // Compute energy terms
    double Ekin = arma::trace(Pe*T);
    double Enuc = arma::trace(Pe*Vc);
    double Ecoul = 0.5*arma::trace((Ja+Jb)*Pe);
    double Eexch = 0.5*(arma::trace(Ka*Pa)+arma::trace(Kb*Pb));
    double Epe = arma::trace(Pe*frozen_Jpe);
    double Etot = Ekin+Enuc+Ecoul+Eexch+Ecnucr+Epe+frozen_Ep;

    if(verbosity>=10) {
      printf("e kinetic energy         % .10f\n",Ekin);
      printf("e nuclear attraction     % .10f\n",Enuc);
      printf("e-p frozen attraction    % .10f\n",Epe);
      printf("e-e Coulomb energy       % .10f\n",Ecoul);
      printf("e-e exchange energy      % .10f\n",Eexch);
      printf("nuclear repulsion energy % .10f\n",Ecnucr);
      printf("frozen proton energy     % .10f\n",frozen_Ep);
      printf("Total energy             % .10f\n",Etot);
    }

    return std::make_pair(Etot,fock);
  };

  // Compute expectation values for protonic coordinates
  std::function<void(const arma::vec &, const arma::mat &)> print_protonic_coordinates = [&pr](const arma::vec & occp, const arma::mat & Cp) {
    for(size_t ip=0;ip<occp.n_elem;ip++)
      if(occp(ip)>0.0) {
        double r[3];
        for(int ic=0;ic<3;ic++)
          r[ic]=arma::as_scalar(Cp.col(ip).t()*pr[ic]*Cp.col(ip));
        printf("Expected position for proton %i: % .6f % .6f % .6f angstrom\n",ip,r[0]/ANGSTROMINBOHR,r[1]/ANGSTROMINBOHR,r[2]/ANGSTROMINBOHR);
      }
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> restricted_neo_builder = [&X, &Xp, &T, &Tp, &Vc, &Vpc, &dfit, &pfit, &pr, electronic_terms, protonic_terms, electron_proton_coulomb, proton_electron_coulomb, &Ecnucr, &verbosity, print_protonic_coordinates](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Get the electronic and protonic orbital coefficients
    arma::mat Ce = X*orbitals[0];
    arma::mat Cp = Xp*orbitals[1];

    // and occupations
    arma::vec occe = occupations[0];
    arma::vec occp = occupations[1];

    // Density matrices
    arma::mat Pe = Ce * arma::diagmat(occe) * Ce.t();
    arma::mat Pp = Cp * arma::diagmat(occp) * Cp.t();

    // Print out locations of protons
    //print_protonic_coordinates(occp, Cp);

    // Compute the terms in the Fock matrices
    arma::mat J, K, Jp, Kp, Jep, Jpe;
    std::tie(J, K) = electronic_terms(Ce, occe);
    std::tie(Jp, Kp) = protonic_terms(Cp, occp);
    Jep = electron_proton_coulomb(Pe);
    Jpe = proton_electron_coulomb(Pp);

    // Form the Fock matrices
    arma::mat Fe = X.t() * (T + Vc + J + .5*K + Jpe) * X;
    arma::mat Fp = Xp.t() * (Tp + Vpc + Jp + Kp + Jep) * Xp;
    std::vector<arma::mat> fock({Fe,Fp});

    // Compute energy terms
    double Ekin = arma::trace(Pe*T);
    double Epkin = arma::trace(Pp*Tp);
    double Enuc = arma::trace(Pe*Vc);
    double Epnuc = arma::trace(Pp*Vpc);
    double Ecoul = 0.5*arma::trace(J*Pe);
    double Epcoul = 0.5*arma::trace(Jp*Pp);
    double Eexch = 0.25*arma::trace(K*Pe);
    double Epexch = 0.5*arma::trace(Kp*Pp);
    double Eepcoul = arma::trace(Jep*Pp);
    double Etot = Ekin+Epkin+Enuc+Epnuc+Ecoul+Epcoul+Eexch+Epexch+Eepcoul+Ecnucr;

    if(verbosity>=10) {
      printf("e kinetic energy         % .10f\n",Ekin);
      printf("p kinetic energy         % .10f\n",Epkin);
      printf("e nuclear attraction     % .10f\n",Enuc);
      printf("p nuclear repulsion      % .10f\n",Epnuc);
      printf("e-e Coulomb energy       % .10f\n",Ecoul);
      printf("p-p Coulomb energy       % .10f\n",Epcoul);
      printf("e-p Coulomb energy       % .10f\n",Eepcoul);
      printf("e-e exchange energy      % .10f\n",Eexch);
      printf("p-p exchange energy      % .10f\n",Epexch);
      printf("nuclear repulsion energy % .10f\n",Ecnucr);
      printf("Total energy             % .10f\n",Etot);
    }

    return std::make_pair(Etot,fock);
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> unrestricted_neo_builder = [&X, &Xp, &T, &Tp, &Vc, &Vpc, &dfit, &pfit, &pr, electronic_terms, protonic_terms, electron_proton_coulomb, proton_electron_coulomb, &Ecnucr, &verbosity, &print_protonic_coordinates](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Get the electronic and protonic orbital coefficients
    arma::mat Ca = X*orbitals[0];
    arma::mat Cb = X*orbitals[1];
    arma::mat Cp = Xp*orbitals[2];

    // and occupations
    arma::vec occa = occupations[0];
    arma::vec occb = occupations[1];
    arma::vec occp = occupations[2];

    // Density matrices
    arma::mat Pa = Ca * arma::diagmat(occa) * Ca.t();
    arma::mat Pb = Cb * arma::diagmat(occb) * Cb.t();
    arma::mat Pe = Pa+Pb;
    arma::mat Pp = Cp * arma::diagmat(occp) * Cp.t();

    // Print out locations of protons
    //print_protonic_coordinates(occp, Cp);

    // Compute the terms in the Fock matrices
    arma::mat Ja, Jb, Ka, Kb, Jp, Kp, Jep, Jpe;
    std::tie(Ja, Ka) = electronic_terms(Ca, occa);
    std::tie(Jb, Kb) = electronic_terms(Cb, occb);
    std::tie(Jp, Kp) = protonic_terms(Cp, occp);
    Jep = electron_proton_coulomb(Pa+Pb);
    Jpe = proton_electron_coulomb(Pp);

    // Form the Fock matrices
    arma::mat Fa = X.t() * (T + Vc + Ja + Jb + Ka + Jpe) * X;
    arma::mat Fb = X.t() * (T + Vc + Ja + Jb + Kb + Jpe) * X;
    arma::mat Fp = Xp.t() * (Tp + Vpc + Jp + Kp + Jep) * Xp;
    std::vector<arma::mat> fock({Fa,Fb,Fp});

    // Compute energy terms
    double Ekin = arma::trace(Pe*T);
    double Epkin = arma::trace(Pp*Tp);
    double Enuc = arma::trace(Pe*Vc);
    double Epnuc = arma::trace(Pp*Vpc);
    double Ecoul = 0.5*arma::trace((Ja+Jb)*Pe);
    double Epcoul = 0.5*arma::trace(Jp*Pp);
    double Eexch = 0.5*(arma::trace(Ka*Pa)+arma::trace(Kb*Pb));
    double Epexch = 0.5*arma::trace(Kp*Pp);
    double Eepcoul = arma::trace(Jep*Pp);
    double Etot = Ekin+Epkin+Enuc+Epnuc+Ecoul+Epcoul+Eexch+Epexch+Eepcoul+Ecnucr;

    if(verbosity>=10) {
      printf("e kinetic energy         % .10f\n",Ekin);
      printf("p kinetic energy         % .10f\n",Epkin);
      printf("e nuclear attraction     % .10f\n",Enuc);
      printf("p nuclear repulsion      % .10f\n",Epnuc);
      printf("e-e Coulomb energy       % .10f\n",Ecoul);
      printf("p-p Coulomb energy       % .10f\n",Epcoul);
      printf("e-p Coulomb energy       % .10f\n",Eepcoul);
      printf("e-e exchange energy      % .10f\n",Eexch);
      printf("p-p exchange energy      % .10f\n",Epexch);
      printf("nuclear repulsion energy % .10f\n",Ecnucr);
      printf("Total energy             % .10f\n",Etot);
    }

    return std::make_pair(Etot,fock);
  };

  // Data for SCF solver
  arma::uvec number_of_blocks_per_particle_type;
  arma::vec maximum_occupation;
  arma::vec number_of_particles;
  std::vector<std::string> block_descriptions;
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder;

  int Nel = basis.Ztot()-Q;
  int Nela = (Nel+M-1)/2;
  int Nelb = Nel-Nela;
  int Np = (int) proton_indices.size();
  printf("Nela = %i Nelb = %i\n",Nela,Nelb);
  if(Nela<0 or Nelb<0)
    throw std::logic_error("Negative number of electrons!\n");
  if(Nelb>Nela)
    throw std::logic_error("Nelb > Nela, check your charge and multiplicity!\n");

  // We will need the density matrices in the calculation
  OpenOrbitalOptimizer::DensityMatrix<double, double> electronic_dm;
  OpenOrbitalOptimizer::DensityMatrix<double, double> protonic_dm;

  // Set up protonic guess
  if(quantum_protons.size()) {
    if(loadchk != "") {
      Checkpoint load(loadchk,false);

      BasisSet oldpbasis;
      load.read(oldpbasis,"pbasis");

      // Protonic solution
      arma::vec Ep;
      arma::mat oldCp, Cp;
      load.read("Ep",Ep);
      load.read("Cp",oldCp);
      Cp = arma::trans(Xp) * pbasis.overlap(oldpbasis) * oldCp;
      if(Cp.n_rows < Cp.n_cols) {
        Ep=Ep.subvec(0,Np-1);
        Cp=Cp.cols(0,Np-1);
      }
      arma::vec occs(Cp.n_cols,arma::fill::zeros);
      occs.subvec(0,Np-1).ones();

      protonic_dm = std::make_pair(std::vector<arma::mat>({Cp}),std::vector<arma::vec>({occs}));
    } else {
      // We need to use a nuclear guess to get the electronic wave function right
      BasisSetLibrary spbaslib;
      spbaslib.load_basis("pb-1s.gbs");

      BasisSet spbasis;
      construct_basis(spbasis,quantum_protons,spbaslib);

      size_t Np=spbasis.get_Nbf();
      arma::mat oldCp(Np,Np);
      oldCp.eye();

      arma::mat Cp;
      Cp = arma::trans(Xp) * pbasis.overlap(spbasis) * oldCp;
      arma::vec occs(Np,arma::fill::ones);

      protonic_dm = std::make_pair(std::vector<arma::mat>({Cp}),std::vector<arma::vec>({occs}));
    }
  }

  // Save the matrices to disk
  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<double,double> &,const OpenOrbitalOptimizer::FockMatrix<double> &)> save_proton_matrices = [&chkpt, &Xp](const OpenOrbitalOptimizer::DensityMatrix<double,double> & pdm, const OpenOrbitalOptimizer::FockMatrix<double> & pfock) {
    const OpenOrbitalOptimizer::Orbitals<double> & porbitals = pdm.first;
    const OpenOrbitalOptimizer::OrbitalOccupations<double> & poccupations = pdm.second;
    arma::mat Cp = Xp*porbitals[0];
    arma::vec Ep = arma::diagvec(porbitals[0].t()*pfock[0]*porbitals[0]);
    chkpt.write("Cp",Cp);
    chkpt.write("Ep",Ep);
  };

  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<double,double> &,const OpenOrbitalOptimizer::FockMatrix<double> &)> save_electron_matrices = [&chkpt, &M, &X](const OpenOrbitalOptimizer::DensityMatrix<double,double> & eldm, const OpenOrbitalOptimizer::FockMatrix<double> & elfock) {
    const OpenOrbitalOptimizer::Orbitals<double> & eorbitals = eldm.first;
    const OpenOrbitalOptimizer::OrbitalOccupations<double> & eoccupations = eldm.second;
    if(M==1) {
      arma::mat Ce = X*eorbitals[0];
      arma::vec Ee = arma::diagvec(eorbitals[0].t()*elfock[0]*eorbitals[0]);
      chkpt.write("C",Ce);
      chkpt.write("E",Ee);
    } else {
      arma::mat Ca = X*eorbitals[0];
      arma::mat Cb = X*eorbitals[1];
      arma::vec Ea = arma::diagvec(eorbitals[0].t()*elfock[0]*eorbitals[0]);
      arma::vec Eb = arma::diagvec(eorbitals[1].t()*elfock[1]*eorbitals[1]);
      chkpt.write("Ca",Ca);
      chkpt.write("Cb",Cb);
      chkpt.write("Ea",Ea);
      chkpt.write("Eb",Eb);
    }
  };

  // Set up electronic guess
  if(loadchk != "") {
    Checkpoint load(loadchk,false);

    BasisSet oldbasis;
    load.read(oldbasis);

    if(M==1) {
      arma::mat oldC, C;
      load.read("C",oldC);
      C = arma::trans(X) * basis.overlap(oldbasis) * oldC;
      if(C.n_rows < C.n_cols) {
        C=C.cols(0,Nela-1);
      }
      std::vector<arma::mat> old_orbs({C});

      // Occupations
      arma::vec occ(C.n_cols,arma::fill::zeros);
      occ.subvec(0,Nela-1).ones();
      occ *= 2.0;
      std::vector<arma::vec> old_occs({occ});

      electronic_dm = std::make_pair(old_orbs, old_occs);

    } else {
      arma::mat oldCa, oldCb, Ca, Cb;
      load.read("Ca",oldCa);
      load.read("Cb",oldCb);

      Ca = arma::trans(X) * basis.overlap(oldbasis) * oldCa;
      Cb = arma::trans(X) * basis.overlap(oldbasis) * oldCb;
      if(Ca.n_rows < Ca.n_cols) {
        Ca = Ca.cols(0,Nela-1);
        Cb = Cb.cols(0,Nela-1);
      }
      std::vector<arma::mat> old_orbs({Ca, Cb});

      // Occupations
      arma::vec occa(Ca.n_cols,arma::fill::zeros);
      arma::vec occb(Cb.n_cols,arma::fill::zeros);
      occa.subvec(0,Nela-1).ones();
      if(Nelb)
        occb.subvec(0,Nelb-1).ones();
      std::vector<arma::vec> old_occs({occa, occb});

      electronic_dm = std::make_pair(old_orbs, old_occs);
    }
  } else {
    // Guess from SAP
    arma::vec E;
    arma::mat C;
    arma::eig_sym(E,C,Fguess);

    if(M==1) {
      std::vector<arma::mat> orbs({C});
      std::vector<arma::vec> occs({arma::vec(C.n_cols,arma::fill::zeros)});
      occs[0].subvec(0,Nela-1).ones();
      occs[0]*=2;
      electronic_dm=std::make_pair(orbs,occs);
    } else {
      std::vector<arma::mat> orbs({C, C});
      std::vector<arma::vec> occs({arma::vec(C.n_cols,arma::fill::zeros), arma::vec(C.n_cols,arma::fill::zeros)});
      occs[0].subvec(0,Nela-1).ones();
      if(Nelb)
        occs[1].subvec(0,Nelb-1).ones();
      electronic_dm=std::make_pair(orbs,occs);
    }
  }

  double Eold=0.0;
  for(size_t istep=0;istep<nstepwise;istep++) {
    // Run electronic calculation
    if(M==1) {
      number_of_blocks_per_particle_type = {1};
      maximum_occupation = {2.0};
      number_of_particles = {(double) (Nel)};
      block_descriptions = {"electronic"};
      fock_builder = restricted_builder;
    } else {
      number_of_blocks_per_particle_type = {1,1};
      maximum_occupation = {1.0,1.0};
      number_of_particles = {(double) (Nela), (double) Nelb};
      block_descriptions = {"electronic alpha", "electronic beta"};
      fock_builder = unrestricted_builder;
    }

    // Update protonic terms for electronic solution
    arma::mat Pp(Xp.n_rows,Xp.n_rows,arma::fill::zeros);
    {
      const OpenOrbitalOptimizer::Orbitals<double> & orbitals = protonic_dm.first;
      const OpenOrbitalOptimizer::OrbitalOccupations<double>  & occupations = protonic_dm.second;
      for(size_t i=0;i<orbitals.size();i++) {
        arma::mat Cp = Xp*orbitals[i];
        arma::vec occp = occupations[i];
        Pp += Cp * arma::diagmat(occp) * Cp.t();
      }
    }
    if(quantum_protons.size())
      frozen_Jpe = proton_electron_coulomb(Pp);

    // Update the frozen proton energy
    if(quantum_protons.size()) {
      const OpenOrbitalOptimizer::Orbitals<double> & orbitals = protonic_dm.first;
      const OpenOrbitalOptimizer::OrbitalOccupations<double>  & occupations = protonic_dm.second;

      // Get the electronic and protonic orbital coefficients
      arma::mat Cp = Xp*orbitals[0];
      // and occupations
      arma::vec occp = occupations[0];
      // Density matrices
      arma::mat Pp = Cp * arma::diagmat(occp) * Cp.t();

      // Compute the terms in the Fock matrices
      arma::mat Jp, Kp;
      std::tie(Jp, Kp) = protonic_terms(Cp, occp);

      // Compute energy terms
      double Epkin = arma::trace(Pp*Tp);
      double Epnuc = arma::trace(Pp*Vpc);
      double Epcoul = 0.5*arma::trace(Jp*Pp);
      double Epexch = 0.5*arma::trace(Kp*Pp);
      frozen_Ep = Epkin+Epnuc+Epcoul+Epexch;
    }

    printf("\n\n\nIteration %i: solving electronic SCF\n", istep);
    OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
    scfsolver.initialize_with_orbitals(electronic_dm.first, electronic_dm.second);
    scfsolver.error_norm(error_norm);
    scfsolver.convergence_threshold(convergence_threshold); // use full threshold here since the electronic part is easy
    scfsolver.verbosity(verbosity);
    scfsolver.maximum_iterations(maxinititer);
    scfsolver.maximum_history_length(diisorder);
    scfsolver.run();

    // Grab the new density matrix
    electronic_dm = scfsolver.get_solution();
    save_electron_matrices(electronic_dm, scfsolver.get_fock_matrix());

    if(quantum_protons.size()) {
      // Run protonic calculation.
      arma::mat Pe(X.n_rows,X.n_rows,arma::fill::zeros);
      {
        const OpenOrbitalOptimizer::Orbitals<double> & orbitals = electronic_dm.first;
        const OpenOrbitalOptimizer::OrbitalOccupations<double>  & occupations = electronic_dm.second;
        for(size_t i=0;i<orbitals.size();i++) {
          arma::mat Ce = X*orbitals[i];
          arma::vec occe = occupations[i];
          Pe += Ce * arma::diagmat(occe) * Ce.t();
        }
      }
      frozen_Jep = electron_proton_coulomb(Pe);

      // Update the frozen electron energy
      {
        const auto & orbitals = electronic_dm.first;
        const auto & occupations = electronic_dm.second;

        if(M==1) {
          // Get the electronic and protonic orbital coefficients
          arma::mat Ce = X*orbitals[0];

          // and occupations
          arma::vec occe = occupations[0];

          // Density matrices
          arma::mat Pe = Ce * arma::diagmat(occe) * Ce.t();

          // Compute the terms in the Fock matrices
          arma::mat J, K;
          std::tie(J, K) = electronic_terms(Ce, occe);
          // Form the Fock matrices
          arma::mat Fe = X.t() * (T + Vc + J + .5*K + frozen_Jpe) * X;
          std::vector<arma::mat> fock({Fe});

          // Compute energy terms
          double Ekin = arma::trace(Pe*T);
          double Enuc = arma::trace(Pe*Vc);
          double Ecoul = 0.5*arma::trace(J*Pe);
          double Eexch = 0.25*arma::trace(K*Pe);
          frozen_Ee = Ekin+Enuc+Ecoul+Eexch;

        } else {
          // Get the electronic and protonic orbital coefficients
          arma::mat Ca = X*orbitals[0];
          arma::mat Cb = X*orbitals[1];
          // and occupations
          arma::vec occa = occupations[0];
          arma::vec occb = occupations[1];
          // Density matrices
          arma::mat Pa = Ca * arma::diagmat(occa) * Ca.t();
          arma::mat Pb = Cb * arma::diagmat(occb) * Cb.t();
          arma::mat Pe = Pa+Pb;

          // Compute the terms in the Fock matrices
          arma::mat Ja, Jb, Ka, Kb, Jp, Kp, Jep, Jpe;
          std::tie(Ja, Ka) = electronic_terms(Ca, occa);
          std::tie(Jb, Kb) = electronic_terms(Cb, occb);

          // Compute energy terms
          double Ekin = arma::trace(Pe*T);
          double Enuc = arma::trace(Pe*Vc);
          double Ecoul = 0.5*arma::trace((Ja+Jb)*Pe);
          double Eexch = 0.5*(arma::trace(Ka*Pa)+arma::trace(Kb*Pb));
          frozen_Ee = Ekin+Enuc+Ecoul+Eexch;
        }
      }

      OpenOrbitalOptimizer::FockBuilder<double, double> nuclear_builder = [&Xp, &Tp, &Vpc, &pfit, &pr, &protonic_terms, &frozen_Jep, &frozen_Ee, &verbosity, print_protonic_coordinates, &Ecnucr](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
        const auto & orbitals = dm.first;
        const auto & occupations = dm.second;

        // Get the electronic and protonic orbital coefficients
        arma::mat Cp = Xp*orbitals[0];
        // and occupations
        arma::vec occp = occupations[0];

        // Density matrices
        arma::mat Pp = Cp * arma::diagmat(occp) * Cp.t();
        //print_protonic_coordinates(occp, Cp);

        // Compute the terms in the Fock matrices
        arma::mat Jp, Kp;
        std::tie(Jp, Kp) = protonic_terms(Cp, occp);

        // Form the Fock matrices
        arma::mat Fp = Xp.t() * (Tp + Vpc + Jp + Kp + frozen_Jep) * Xp;
        std::vector<arma::mat> fock({Fp});

        // Compute energy terms
        double Epkin = arma::trace(Pp*Tp);
        double Epnuc = arma::trace(Pp*Vpc);
        double Epcoul = 0.5*arma::trace(Jp*Pp);
        double Epexch = 0.5*arma::trace(Kp*Pp);
        double Eepcoul = arma::trace(frozen_Jep*Pp);
        double Etot = Epkin+Epnuc+Epcoul+Epexch+Eepcoul+frozen_Ee+Ecnucr;

        if(verbosity>=10) {
          printf("p kinetic energy         % .10f\n",Epkin);
          printf("p nuclear repulsion      % .10f\n",Epnuc);
          printf("p-p Coulomb energy       % .10f\n",Epcoul);
          printf("e-p Coulomb energy       % .10f\n",Eepcoul);
          printf("p-p exchange energy      % .10f\n",Epexch);
          printf("frozen electron energy   % .10f\n",frozen_Ee);
          printf("nuclear repulsion energy % .10f\n",Ecnucr);
          printf("Total energy             % .10f\n",Etot);
        }

        return std::make_pair(Etot,fock);
      };

      number_of_blocks_per_particle_type = {1};
      maximum_occupation = {1.0};
      number_of_particles = {(double) quantum_protons.size()};
      block_descriptions = {"protonic"};
      fock_builder = nuclear_builder;

      // Run protonic SCF
      printf("\n\nContinuing with protonic SCF\n");
      OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
      scfsolver.initialize_with_orbitals(protonic_dm.first, protonic_dm.second);
      scfsolver.error_norm(error_norm);
      scfsolver.convergence_threshold(init_convergence_threshold);
      scfsolver.verbosity(verbosity);
      scfsolver.maximum_iterations(maxinititer);
      scfsolver.maximum_history_length(diisorder);
      scfsolver.run();

      // Grab the proton density matrix
      protonic_dm = scfsolver.get_solution();
      save_proton_matrices(protonic_dm, scfsolver.get_fock_matrix());

      double Enew = scfsolver.get_energy();
      if(std::abs(Enew-Eold) < init_convergence_threshold) {
        printf("Stepwise SCF converged: energy changed by %e from last macroiteration!\n", Enew-Eold);
        break;
      }
      Eold = Enew;
    } else {
      break;
    }
  }
  if(quantum_protons.size()) {
    if(nstepwise)
      printf("\n\nProceeding with simultaneous electron-proton SCF\n");

    // Proceed with nuclear-electronic calculation
    OpenOrbitalOptimizer::Orbitals<double> guess_orbitals;
    OpenOrbitalOptimizer::OrbitalOccupations<double> guess_occupations;
    if(M==1) {
      guess_orbitals={electronic_dm.first[0], protonic_dm.first[0]};
      guess_occupations={electronic_dm.second[0], protonic_dm.second[0]};
      number_of_blocks_per_particle_type = {1,1};
      maximum_occupation = {2.0,1.0};
      number_of_particles = {(double) (Nel),(double) proton_indices.size()};
      block_descriptions = {"electronic", "protonic"};
      fock_builder = restricted_neo_builder;
    } else {
      guess_orbitals={electronic_dm.first[0], electronic_dm.first[1], protonic_dm.first[0]};
      guess_occupations={electronic_dm.second[0], electronic_dm.second[1], protonic_dm.second[0]};
      number_of_blocks_per_particle_type = {1,1,1};
      maximum_occupation = {1.0,1.0,1.0};
      number_of_particles = {(double) (Nela), (double) Nelb,(double) proton_indices.size()};
      block_descriptions = {"electronic alpha", "electronic beta", "protonic"};
      fock_builder = unrestricted_neo_builder;
    }
    OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
    scfsolver.initialize_with_orbitals(guess_orbitals, guess_occupations);
    scfsolver.error_norm(error_norm);
    scfsolver.convergence_threshold(convergence_threshold);
    scfsolver.verbosity(verbosity);
    scfsolver.maximum_iterations(maxiter);
    scfsolver.maximum_history_length(diisorder);
    scfsolver.run();

    auto dm = scfsolver.get_solution();
    auto fock = scfsolver.get_fock_matrix();
    size_t Nmat = fock.size();
    save_proton_matrices(std::make_pair(std::vector<arma::mat>({dm.first[Nmat-1]}),std::vector<arma::vec>({dm.second[Nmat-1]})), std::vector<arma::mat>({fock[Nmat-1]}));
    if(M==1) {
      save_electron_matrices(std::make_pair(std::vector<arma::mat>({dm.first[0]}),std::vector<arma::vec>({dm.second[0]})), std::vector<arma::mat>({fock[0]}));
    } else {
      save_electron_matrices(std::make_pair(std::vector<arma::mat>({dm.first[0],dm.first[1]}),std::vector<arma::vec>({dm.second[0],dm.second[1]})), std::vector<arma::mat>({fock[0],fock[1]}));
    }
  }

  printf("\nRunning program took %s.\n",t.elapsed().c_str());
  return 0;
}

int main(int argc, char **argv) {
#ifdef CATCH_EXCEPTIONS
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
#else
  return main_guarded(argc, argv);
#endif
}
