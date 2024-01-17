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

#include "scfsolver.hpp"

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
  settings.add_string("QuantumProtons", "Indices of protons to make quantum", "");
  settings.add_double("ProtonMass", "Protonic mass", 1836.15267389);
  settings.add_int("Verbosity", "Verboseness level", 5);

  // Parse settings
  settings.parse(std::string(argv[1]),true);
  settings.print();
  int Q = settings.get_int("Charge");
  int M = settings.get_int("Multiplicity");
  int verbosity = settings.get_int("Verbosity");
  double proton_mass = settings.get_double("ProtonMass");
  double intthr = settings.get_double("IntegralThresh");
  double convergence_threshold = settings.get_double("ConvThr");
  bool verbose = settings.get_bool("Verbose");
  size_t fitmem = 1000000*settings.get_int("FittingMemory");

  // Read in basis set
  BasisSetLibrary baslib;
  baslib.load_basis(settings.get_string("Basis"));
  BasisSetLibrary pbaslib;
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
    printf("adding atoms[%i] to list of quantum protons\n",idx);
  }
  for(size_t i=0; i<quantum_protons.size(); i++)
    quantum_protons[i].num=i;
  //if(quantum_protons.size()==0)
  //  throw std::runtime_error("No quantum protons!\n");

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

  // Classical nucleus repulsion energy
  double Enucr=0.0;
  for(size_t i=0;i<classical_nuclei.size();i++) {
    auto [Qi, xi, yi, zi] = classical_nuclei[i];
    for(size_t j=0;j<i;j++) {
      auto [Qj, xj, yj, zj] = classical_nuclei[j];
      Enucr+=Qi*Qj/sqrt(std::pow(xi-xj,2)+std::pow(yi-yj,2)+std::pow(zi-zj,2));
    }
  }

  // Construct the basis set
  BasisSet basis;
  construct_basis(basis,atoms,baslib);
  BasisSet pbasis;
  construct_basis(pbasis,quantum_protons,pbaslib);

  printf("Quantum proton basis\n");
  pbasis.print();

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
  arma::mat V(basis.nuclear());
  arma::mat Vc(basis.nuclear(classical_nuclei));
  arma::mat Vsap(basis.sap_potential(potlib));
  arma::mat Vpc, Tp, Vpsap;
  if(Sp.n_elem) {
    Vpc=-pbasis.nuclear(classical_nuclei);
    Tp=pbasis.kinetic()/proton_mass;
    Vpsap=pbasis.sap_potential(potlib);
  }

  // Guess Fock
  //arma::mat Fguess(X.t()*(T+V+Vsap)*X);
  arma::mat Fguess(X.t()*(T+V)*X); //debug
  printf("*** USING CORE GUESS FOR DEBUG PURPOSES ****\n");
  arma::mat Fpguess;
  if(Sp.n_elem)
    Fpguess = Xp.t()*(Tp-Vpsap)*Xp;
  std::vector<arma::mat> fock_guess({Fguess, Fpguess});

  // Compute density fitting integrals
  // Calculate the fitting integrals, running in B-matrix mode
  bool direct=settings.get_bool("Direct");
  double fitthr=settings.get_double("FittingThreshold");
  double cholfitthr=settings.get_double("FittingCholeskyThreshold");
  bool bmat=true;

  DensityFit dfit;
  size_t Npairs_e=dfit.fill(basis,dfitbas,direct,intthr,fitthr,cholfitthr,bmat);
  DensityFit pfit;
  size_t Npairs_p=0;
  if(Sp.n_elem)
    Npairs_p=pfit.fill(pbasis,dfitbas,direct,intthr,fitthr,cholfitthr,bmat);

  printf("%i electronic shell pairs out of %i are significant.\n",(int) Npairs_e, (int) basis.get_unique_shellpairs().size());
  printf("%i protonic shell pairs out of %i are significant.\n",(int) Npairs_p, (int) pbasis.get_unique_shellpairs().size());
  printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
  fflush(stdout);
  if(Sp.n_elem>0 and dfit.get_Naux() != pfit.get_Naux())
    throw std::logic_error("Electronic and protonic density fitting basis sets don't have the same number of functions!\n");

  // Number of blocks per particle type
  arma::uvec number_of_blocks_per_particle_type({1,1});
  arma::vec maximum_occupation({2.0,1.0});

  int Nel=basis.Ztot()-settings.get_int("Charge");
  arma::vec number_of_particles({(double) (Nel),(double) proton_indices.size()});

  std::vector<std::string> block_descriptions({"electronic", "protonic"});

  std::function<std::pair<arma::mat,arma::mat>(const arma::mat & P, const arma::vec & occs)> electronic_terms = [dfit, fitmem](const arma::mat & C, const arma::vec & occs) {
    arma::mat P(C*arma::diagmat(occs)*C.t());
    arma::mat J(dfit.calcJ(P));
    arma::mat K(-dfit.calcK(C,arma::conv_to<std::vector<double>>::from(occs), fitmem));
    return std::make_pair(J,K);
  };
  std::function<std::pair<arma::mat,arma::mat>(const arma::mat & P, const arma::vec & occs)> protonic_terms = [pfit, fitmem](const arma::mat & C, const arma::vec & occs) {
    arma::mat P(C*arma::diagmat(occs)*C.t());
    arma::mat J(pfit.calcJ(P));
    arma::mat K(-pfit.calcK(C,arma::conv_to<std::vector<double>>::from(occs), fitmem));
    return std::make_pair(J,K);
  };
  std::function<arma::mat(const arma::mat & P, const arma::vec & occs)> electron_proton_coulomb = [dfit, pfit](const arma::mat & Ce, const arma::vec & eoccs) {
    arma::mat Pe(Ce*arma::diagmat(eoccs)*Ce.t());
    arma::vec c(dfit.compute_expansion(Pe));
    arma::mat J=-pfit.calcJ_vector(c);
    return J;
  };
  std::function<arma::mat(const arma::mat & P, const arma::vec & occs)> proton_electron_coulomb = [dfit, pfit](const arma::mat & Cp, const arma::vec & poccs) {
    arma::mat Pp(Cp*arma::diagmat(poccs)*Cp.t());
    arma::vec c(pfit.compute_expansion(Pp));
    arma::mat J=-dfit.calcJ_vector(c);
    return J;
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [X, Xp, T, Tp, Vc, Vpc, dfit, pfit, electronic_terms, protonic_terms, electron_proton_coulomb, proton_electron_coulomb, Enucr, verbosity](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Get the electronic and protonic orbital coefficients
    arma::mat Ce = X*orbitals[0];
    arma::mat Cp;
    if(Xp.n_elem)
      Cp = Xp*orbitals[1];

    // and occupations
    arma::vec occe = occupations[0];
    arma::vec occp = occupations[1];

    // Compute the terms in the Fock matrices
    arma::mat J, K, Jp, Kp, Jep, Jpe;
    std::tie(J, K) = electronic_terms(Ce, occe);
    if(Xp.n_elem) {
      std::tie(Jp, Kp) = protonic_terms(Cp, occp);
      Jep = electron_proton_coulomb(Ce, occe);
      Jpe = proton_electron_coulomb(Cp, occp);
    }
    // Form the Fock matrices
    arma::mat Fe = T + Vc + J + .5*K;
    if(Xp.n_elem) {
      Fe += Jpe;
    }
    Fe = X.t() * Fe * X;

    arma::mat Fp;
    if(Xp.n_elem)
      Fp = Xp.t() * (Tp + Vpc + Jp + Kp + Jep) * Xp;
    std::vector<arma::mat> fock({Fe,Fp});

    // Density matrices
    arma::mat Pe, Pp;
    Pe = Ce * arma::diagmat(occe) * Ce.t();
    if(Xp.n_elem)
      Pp = Cp * arma::diagmat(occp) * Cp.t();

    // Compute energy terms
    double Ekin = arma::trace(Pe*T);
    double Epkin = arma::trace(Pp*Tp);
    double Enuc = arma::trace(Pe*Vc);
    double Epnuc = arma::trace(Pp*Vpc);
    double Ecoul = 0.5*arma::trace(J*Pe);
    double Epcoul = 0.5*arma::trace(Jp*Pp);
    double Eexch = 0.25*arma::trace(K*Pe);
    double Epexch = 0.5*arma::trace(Kp*Pp);
    double Eepcoul = Xp.n_elem>0 ? arma::trace(Jep*Pp) : 0.0;
    double Etot = Ekin+Epkin+Enuc+Epnuc+Ecoul+Epcoul+Eexch+Epexch+Eepcoul+Enucr;

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
      printf("nuclear repulsion energy % .10f\n",Epexch);
      printf("Total energy             % .10f\n",Etot);
    }

    return std::make_pair(Etot,fock);
  };

  // Initialize SCF solver
  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  scfsolver.set_convergence_threshold(convergence_threshold);
  scfsolver.set_verbosity(verbosity);
  scfsolver.initialize_with_fock(fock_guess);
  scfsolver.run();

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
