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
  settings.add_string("ErrorNorm", "Error norm to use in the SCF code", "rms");
  settings.add_double("InitConvThr", "Initialization convergence threshold", 1e-5);
  settings.add_int("Verbosity", "Verboseness level", 5);
  settings.add_int("MaxInitIter", "Maximum number of iterations in the stepwise solutions", 50);
  settings.add_string("SaveChk", "Checkpoint file to save to", "complex_basis.chk");
  settings.add_string("LoadChk", "Checkpoint file to load from", "");

  // Parse settings
  settings.parse(std::string(argv[1]),true);
  settings.print();
  int Q = settings.get_int("Charge");
  int M = settings.get_int("Multiplicity");
  int verbosity = settings.get_int("Verbosity");
  int maxinititer = settings.get_int("MaxInitIter");
  int diisorder = settings.get_int("DIISOrder");
  double intthr = settings.get_double("IntegralThresh");
  double convergence_threshold = settings.get_double("ConvThr");
  size_t fitmem = 1000000*settings.get_int("FittingMemory");
  std::string error_norm = settings.get_string("ErrorNorm");
  std::string loadchk = settings.get_string("LoadChk");
  std::string savechk = settings.get_string("SaveChk");

  Checkpoint chkpt(savechk,true);

  // Read in basis set
  BasisSetLibrary baslib;
  baslib.load_basis(settings.get_string("Basis"));

  // Read in SAP potential
  BasisSetLibrary potlib;
  potlib.load_basis(settings.get_string("SAPBasis"));

  auto atoms=load_xyz(settings.get_string("System"),!settings.get_bool("InputBohr"));
  
  // Nucleus repulsion energy
  double Enucr = 0.0;
  for(size_t i = 0; i < atoms.size(); i++) {
    // Ghost nucleus
    if(atoms[i].el.size()>3 && atoms[i].el.substr(atoms[i].el.size()-3, 3) == "-Bq")
      continue;
    auto Qi(get_Z(atoms[i].el));
    auto xi(atoms[i].x), yi(atoms[i].y), zi(atoms[i].z);
    for(size_t j = 0; j < i; j++) {
      auto Qj(get_Z(atoms[j].el));
      auto xj(atoms[j].x), yj(atoms[j].y), zj(atoms[j].z);
      Enucr += Qi * Qj / sqrt(std::pow(xi - xj, 2) + std::pow(yi - yj, 2) + std::pow(zi - zj, 2));
    }
  }

  // Construct the basis set
  BasisSet basis;
  construct_basis(basis, atoms, baslib);
  chkpt.write(basis);

  int maxam=basis.get_max_am();
  std::vector<arma::uvec> m_indices(2*maxam+1);
  for (int m = -maxam; m <= maxam; m++) {
    m_indices[m+maxam]=basis.m_indices(m);
  }

  // Construct density fitting basis set
  BasisSetLibrary fitlib;
  fitlib.load_basis(settings.get_string("FittingBasis"));
  BasisSet dfitbas;
  {
    // Construct fitting basis
    bool uselm = settings.get_bool("UseLM");
    settings.set_bool("UseLM", true);
    construct_basis(dfitbas,basis.get_nuclei(), fitlib);
    dfitbas.coulomb_normalize();
    settings.set_bool("UseLM", uselm);
  }

  // Calculate matrices
  arma::mat S(basis.overlap());
  arma::mat T(basis.kinetic());
  arma::mat V(basis.nuclear());
  arma::mat Vsap(basis.sap_potential(potlib));
  arma::mat fock_terms = T + V + Vsap; //Helper

  // Blocked matrices
  size_t Nmo=0;
  std::vector<arma::mat> X(2*maxam+1);
  for (size_t m=0; m<X.size(); m++) {
    X[m] = BasOrth(S(m_indices[m], m_indices[m]));
    Nmo += X[m].n_cols;
  }
  size_t Nbf = basis.get_Nbf();

  // Guess Fock
  bool unrestricted = M - 1;
  std::vector<arma::mat> Fguess((1 + unrestricted) * (2 * maxam + 1));
  for (size_t m=0; m<X.size(); m++) {
    Fguess[m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
    if (unrestricted)
      Fguess[X.size() + m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
  }

  // Calculate density fitting integrals
  bool direct = settings.get_bool("Direct");
  double fitthr = settings.get_double("FittingThreshold");
  double cholfitthr = settings.get_double("FittingCholeskyThreshold");

  DensityFit dfit;
  size_t Npairs = dfit.fill(basis, dfitbas, direct, intthr, fitthr, cholfitthr);

  std::function<std::pair<arma::mat,arma::vec>(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> collect_orbitals = [&](const auto & orbitals, const auto & occupations) {
    arma::vec occs(Nmo, arma::fill::zeros);
    arma::mat C(Nbf, Nmo, arma::fill::zeros);
    size_t imo=0;
    for (size_t m=0; m<X.size(); m++) {
      arma::mat Csub = X[m]*orbitals[m];
      arma::mat Cpad(Nbf,X[m].n_cols,arma::fill::zeros);
      Cpad.rows(m_indices[m]) = Csub;
      
      occs.subvec(imo,imo+X[m].n_cols-1) = occupations[m];
      C.cols(imo,imo+X[m].n_cols-1) = Cpad;
      imo += X[m].n_cols;
    }
    if(imo != Nmo)
      throw std::logic_error("Indexing problem\n");
    return std::make_pair(C,occs);
  };
  
  std::function<std::vector<arma::mat>(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> electronic_terms = [&](const auto & orbitals, const auto & occupations) {
    arma::mat C;
    arma::vec occs;
    std::tie(C,occs) = collect_orbitals(orbitals,occupations);

    /*for (size_t m=0; m<orbitals.size(); m++)
      std::cout << orbitals[m] << std::endl;
    printf("!\n");
    for (size_t m=0; m<occupations.size(); m++)
    std::cout << occupations[m] << std::endl;*/
    arma::mat P = C*arma::diagmat(occs)*C.t();
    arma::mat J(dfit.calcJ(P));
    arma::mat K(-dfit.calcK(C, arma::conv_to<std::vector<double>>::from(occs), fitmem));
    std::vector terms = {P, J, K};
    return terms;
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> restricted_fock_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    std::vector<arma::mat> fock(2 * maxam + 1);
    std::vector<arma::mat> el_terms(electronic_terms(orbitals, occupations));
    arma::mat P(el_terms[0]);
    arma::mat J(el_terms[1]);
    arma::mat K(el_terms[2]);

    // Form the Fock matrices
    for (size_t m=0; m<X.size(); m++) {
      fock[m] = X[m].t() * (T(m_indices[m], m_indices[m]) + V(m_indices[m], m_indices[m]) + J(m_indices[m], m_indices[m]) + .5 * K(m_indices[m], m_indices[m])) * X[m];
    }

    // Compute energy terms
    double Ekin = arma::trace(P * T);
    double Enuc = arma::trace(P * V);
    double Ecoul = 0.5 * arma::trace(P * J);
    double Eexch = 0.25 * arma::trace(P * K);
    double Etot = Ekin + Enuc + Ecoul + Eexch + Enucr;

    if(verbosity >= 10) {
      printf("e kinetic energy         % .10f\n", Ekin);
      printf("e nuclear attraction     % .10f\n", Enuc);
      printf("e-e Coulomb energy       % .10f\n", Ecoul);
      printf("e-e exchange energy      % .10f\n", Eexch);
      printf("nuclear repulsion energy % .10f\n", Enucr);
      printf("Total energy             % .10f\n", Etot);
    }

    return std::make_pair(Etot, fock);
  }; //restrcted Fock builder

  
  OpenOrbitalOptimizer::FockBuilder<double, double> unrestricted_fock_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {

    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;
    std::vector<arma::mat> fock(4 * maxam + 2);

    // Alpha electrons
    std::vector<arma::mat> a_orbs;
    std::vector<arma::vec> a_occs;
    for (size_t i = 0; i < occupations.size() / 2; i++) {
      a_orbs.push_back(orbitals[i]);
      a_occs.push_back(occupations[i]);
    }
    std::vector<arma::mat> a_el_terms(electronic_terms(a_orbs, a_occs));
    arma::mat Pa(a_el_terms[0]);
    arma::mat Ja(a_el_terms[1]);
    arma::mat Ka(a_el_terms[2]);

    // Beta electrons
    std::vector<arma::mat> b_orbs;
    std::vector<arma::vec> b_occs;
    for (size_t i = occupations.size() / 2; i < occupations.size(); i++) {
      b_orbs.push_back(orbitals[i]);
      b_occs.push_back(occupations[i]);
    }
    std::vector<arma::mat> b_el_terms(electronic_terms(b_orbs, b_occs));
    arma::mat Pb(b_el_terms[0]);
    arma::mat Jb(b_el_terms[1]);
    arma::mat Kb(b_el_terms[2]);
    arma::mat P = Pa + Pb;

    // Form the Fock matrices
    for (size_t m=0; m<X.size(); m++) {
      fock[m] = X[m].t() * (T(m_indices[m], m_indices[m]) + V(m_indices[m], m_indices[m]) + Ja(m_indices[m], m_indices[m]) + Jb(m_indices[m], m_indices[m]) + Ka(m_indices[m], m_indices[m])) * X[m];
      fock[X.size() + m] = X[m].t() * (T(m_indices[m], m_indices[m]) + V(m_indices[m], m_indices[m]) + Ja(m_indices[m], m_indices[m]) + Jb(m_indices[m], m_indices[m]) + Kb(m_indices[m], m_indices[m])) * X[m];
    }
    
    // Compute energy terms
    double Ekin = arma::trace(P * T);
    double Enuc = arma::trace(P * V);
    double Ecoul = 0.5 * arma::trace(P * (Ja + Jb));
    double Eexch = 0.5 * (arma::trace(Pa * Ka) + arma::trace(Pb * Kb));
    double Etot = Ekin + Enuc + Ecoul + Eexch + Enucr;

    if(verbosity >= 10) {
      printf("e kinetic energy         % .10f\n", Ekin);
      printf("e nuclear attraction     % .10f\n", Enuc);
      printf("e-e Coulomb energy       % .10f\n", Ecoul);
      printf("e-e exchange energy      % .10f\n", Eexch);
      printf("nuclear repulsion energy % .10f\n", Enucr);
      printf("Total energy             % .10f\n", Etot);
    }

    return std::make_pair(Etot, fock);
  }; //unrestricted Fock builder

  int Nel = basis.Ztot() - Q;
  int Nela = (Nel + M - 1) / 2;
  int Nelb = Nel - Nela;
  printf("Nela = %i Nelb = %i\n", Nela, Nelb);
  if (Nela < 0 or Nelb < 0)
    throw std::logic_error("Negative number of electrons!\n");
  if (Nelb > Nela)
    throw std::logic_error("Nelb > Nela, check your charge and multiplicity!\n");

  // Density matrix
  OpenOrbitalOptimizer::DensityMatrix<double, double> dm;

  // Save matrices to disk
  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<double, double> &, const OpenOrbitalOptimizer::FockMatrix<double> &)> save_matrices = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm, const OpenOrbitalOptimizer::FockMatrix<double> fock) {
    const OpenOrbitalOptimizer::Orbitals<double> & orbitals = dm.first;
    if (M == 1) {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat C = X[m] * orbitals[m];
	arma::vec E = arma::diagvec(orbitals[m].t() * fock[m] * orbitals[m]);
	chkpt.write("C", C);
	chkpt.write("E", E);
      }
    } else {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat Ca = X[m] * orbitals[m];
	arma::mat Cb = X[m] * orbitals[m];
	arma::vec Ea = arma::diagvec(orbitals[m].t() * fock[0][m] * orbitals[0]);
	arma::vec Eb = arma::diagvec(orbitals[m].t() * fock[1][m] * orbitals[1]);
	chkpt.write("Ca", Ca);
	chkpt.write("Cb", Cb);
	chkpt.write("Ea", Ea);
	chkpt.write("Eb", Eb);
      }
    }
  }; // Save density matrix to disk

  // Data for SCF solver
  int nblocks = Fguess.size();
  arma::uvec number_of_blocks_per_particle_type;
  arma::vec maximum_occupation;
  arma::vec number_of_particles;
  std::vector<std::string> block_descriptions;
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder;
  
  // Run SCF
  if (M == 1) {
    number_of_blocks_per_particle_type = {nblocks};
    maximum_occupation.set_size(nblocks).fill(2.0);
    number_of_particles = {(double) (Nel)};
    for (int k=0; k<nblocks; k++) {
      std::string str = "m=" + std::to_string(k - maxam);
      block_descriptions.push_back(str);
    }
    fock_builder = restricted_fock_builder;
  } else {
    number_of_blocks_per_particle_type = {nblocks / 2, nblocks / 2};
    maximum_occupation.set_size(nblocks).fill(1.0);
    number_of_particles = {(double) (Nela), (double) (Nelb)};
    for (int l=0; l<2; l++) {
      for (int k=0; k<nblocks / 2; k++) {
	std::string str = "m=" + std::to_string(k - maxam);
	block_descriptions.push_back(str);
      }
    }
    fock_builder = unrestricted_fock_builder;
  }
  
  printf("\n\n\nSolving SCF\n");
  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(Fguess);
  scfsolver.error_norm(error_norm);
  scfsolver.convergence_threshold(convergence_threshold);
  scfsolver.verbosity(verbosity);
  scfsolver.maximum_iterations(maxinititer);
  scfsolver.maximum_history_length(diisorder);
  scfsolver.run();

  //dm = scfsolver.get_solution();
  //save_matrices(dm, scfsolver.get_fock_matrix());
  
  double E = scfsolver.get_energy();
  printf("SCF converged: energy is %e.\n", E);

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
    
