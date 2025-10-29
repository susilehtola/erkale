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
  settings.add_int("StepwiseSCFIter", "Number of stepwise SCF macroiterations, 0 to skip to simultaneous solution", 0);
  settings.add_string("ErrorNorm", "Error norm to use in the SCF code", "rms");
  settings.add_double("ProtonMass", "Protonic mass", 1836.15267389);
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
  std::vector<arma::mat> Fguess(2*maxam+1);
  for (size_t m=0; m<X.size(); m++)
    Fguess[m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];

  // Calculate density fitting integrals
  bool direct = settings.get_bool("Direct");
  double fitthr = settings.get_double("FittingThreshold");
  double cholfitthr = settings.get_double("FittingCholeskyThreshold");

  DensityFit dfit;
  size_t Npairs = dfit.fill(basis, dfitbas, direct, intthr, fitthr, cholfitthr);

  std::function<std::pair<arma::mat, arma::mat>(const arma::mat & P, const arma::vec & occs)> electronic_terms = [&dfit, &fitmem](const arma::mat & C, const arma::vec & occs) {
	
	arma::mat P(C * arma::diagmat(occs) * C.t());
	arma::mat J(dfit.calcJ(P));
	arma::mat K(-dfit.calcK(C, arma::conv_to<std::vector<double>>::from(occs), fitmem));

	return std::make_pair(J, K);
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> restricted_fock_builder = [&X, &T, &V, &dfit, electronic_terms, &Enucr, &verbosity, &maxam, &Nbf, &Nmo, &m_indices](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    arma::mat C(Nbf, Nmo, arma::fill::zeros);
    arma::vec occs(Nmo, arma::fill::zeros);    
    arma::mat P(Nbf, Nbf, arma::fill::zeros);
    std::vector<arma::mat> fock(2 * maxam + 1);
    arma::mat J, K;
    
    size_t imo=0;
    for (size_t m=0; m<X.size(); m++) {
      arma::mat Csub = X[m]*orbitals[m];
      arma::vec occs = occupations[m];
      P(m_indices[m], m_indices[m]) = Csub * arma::diagmat(occs) * Csub;
      std::tie(J, K) = electronic_terms(P(m_indices[m], m_indices[m]), occs);

      // Form the Fock matrices
      arma::mat F = X[m].t() * (T(m_indices[m], m_indices[m]) + V(m_indices[m], m_indices[m]) + J + K) * X[m];
      
      fock[m] = F;
    }

    /* Get the electronic orbital coefficients
    arma::mat C = X * orbitals[0];

    // and occupations
    arma::vec occ = occupations[0];

    // Density matrices
    arma::mat P;
    P = C * arma::diagmat(occ) * C.t();
    */

    // Compute the terms in the Fock matrices
    

    // Compute energy terms
    double Ekin = arma::trace(P * T);
    double Enuc = arma::trace(P * V);
    double Ecoul = 0.5 * arma::trace(P * J);
    double Eexch = 0.5 * arma::trace(P * K);
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

  
  OpenOrbitalOptimizer::FockBuilder<double, double> unrestricted_fock_builder = [&X, &T, &V, &dfit, electronic_terms, &Enucr, &verbosity, &maxam, &Nbf, &Nmo, &m_indices](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    arma::mat Ca(Nbf, Nmo, arma::fill::zeros);
    arma::mat Cb(Nbf, Nmo, arma::fill::zeros);
    arma::vec occa(Nmo, arma::fill::zeros);
    arma::vec occb(Nmo, arma::fill::zeros);
    arma::mat Pa(Nbf, Nbf, arma::fill::zeros);
    arma::mat Pb(Nbf, Nbf, arma::fill::zeros);
    std::vector<arma::mat> fock(4 * maxam + 2);
    arma::mat Ja, Ka, Jb, Kb;
    
    size_t imo=0;
    for (size_t m=0; m<X.size(); m++) {
      arma::mat Casub = X[m]*orbitals[0][m];
      arma::mat Cbsub = X[m]*orbitals[1][m];
      arma::vec occa = occupations[m];
      arma::vec occb = occupations[m];
      Pa(m_indices[m], m_indices[m]) = Casub * arma::diagmat(occa) * Casub;
      Pb(m_indices[m], m_indices[m]) = Cbsub * arma::diagmat(occb) * Cbsub;

      std::tie(Ja, Ka) = electronic_terms(Pa(m_indices[m], m_indices[m]), occa);
      std::tie(Jb, Kb) = electronic_terms(Pb(m_indices[m], m_indices[m]), occb);

      // Form the Fock matrices
      arma::mat Fa = X[m].t() * (T(m_indices[m], m_indices[m]) + V(m_indices[m], m_indices[m])+ Ja + Ka) * X[m];
      arma::mat Fb = X[m].t() * (T(m_indices[m], m_indices[m]) + V(m_indices[m], m_indices[m])+ Jb + Kb) * X[m];

      fock[m] = Fa;
      fock[maxam + m] = Fb;
    }

    /* Get the electronic orbital coefficients
    arma::mat Ca = X * orbitals[0];
    arma::mat Cb = X * orbitals[1];

    // and occupations
    arma::vec occa = occupations[0];
    arma::vec occb = occupations[1];

    // Density matrices
    arma::mat Pa, Pb;
    Pa = Ca * arma::diagmat(occa) * Ca.t();
    Pb = Cb * arma::diagmat(occb) * Cb.t();
    arma::mat P = Pa + Pb;

    // Compute the terms in the Fock matrices
    arma::mat Ja, Jb, Ka, Kb;

    // Form the Fock matrices
    arma::mat Fa = X.t() * (T + V + Ja + Jb + Ka) * X;
    arma::mat Fb = X.t() * (T + V + Ja + Jb + Kb) * X;

    std::vector<arma::mat> fock({Fa, Fb});
    */

    arma::mat P = Pa + Pb;
    
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
  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<double, double> &, const OpenOrbitalOptimizer::FockMatrix<double> &)> save_matrices = [&chkpt, &M, &X, &m_indices](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm, const OpenOrbitalOptimizer::FockMatrix<double> fock) {
    const OpenOrbitalOptimizer::Orbitals<double> & orbitals = dm.first;
    const OpenOrbitalOptimizer::OrbitalOccupations<double> & occupations = dm.second;
    if (M == 1) {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat C = X[m] * orbitals[0];
	arma::vec E = arma::diagvec(orbitals[0].t() * fock[0][m] * orbitals[0]);
	chkpt.write("C", C);
	chkpt.write("E", E);
      }
    } else {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat Ca = X[m] * orbitals[0];
	arma::mat Cb = X[m] * orbitals[1];
	arma::vec Ea = arma::diagvec(orbitals[0].t() * fock[0][m] * orbitals[0]);
	arma::vec Eb = arma::diagvec(orbitals[1].t() * fock[1][m] * orbitals[1]);
	chkpt.write("Ca", Ca);
	chkpt.write("Cb", Cb);
	chkpt.write("Ea", Ea);
	chkpt.write("Eb", Eb);
      }
    }
  }; // Save density matrix to disk

  // Set up electronic guess
  if (loadchk != "") {
    Checkpoint load(loadchk, false);

    BasisSet oldbasis;
    load.read(oldbasis);

    std::vector<arma::mat> old_orbs;
    
    if (M == 1) {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat oldC, C;
	load.read("C", oldC);
	C = arma::trans(X[m]) * basis.overlap(oldbasis) * oldC;
	if (C.n_rows < C.n_cols) {
	  C = C.cols(0, Nela - 1);
	}
	old_orbs.push_back(C);

	// Occupations
	arma::vec occ(C.n_cols, arma::fill::zeros);
	occ.subvec(0, Nela - 1).ones();
	occ *= 2.0;
	std::vector<arma::vec> old_occs({occ});

	dm = std::make_pair(old_orbs, old_occs);
      }
      
    } else {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat oldCa, oldCb, Ca, Cb;
	load.read("Ca", oldCa);
	load.read("Cb", oldCb);

	Ca = arma::trans(X[m]) * basis.overlap(oldbasis) * oldCa;
	Cb = arma::trans(X[m]) * basis.overlap(oldbasis) * oldCb;
	if (Ca.n_rows < Ca.n_cols) {
	  Ca = Ca.cols(0, Nela - 1);
	  if(Nelb)
	    Cb = Cb.cols(0, Nelb - 1);
	}
	old_orbs.push_back(Ca);
	old_orbs.push_back(Cb);

      // Occupations
      arma::vec occa(Ca.n_cols, arma::fill::zeros);
      arma::vec occb(Cb.n_cols, arma::fill::zeros);
      occa.subvec(0, Nela - 1).ones();
      if(Nelb)
	occb.subvec(0, Nelb - 1).ones();
      std::vector<arma::vec> old_occs({occa, occb});

      dm = std::make_pair(old_orbs, old_occs);
      }
    }
  } else {//if loadchk
    // Guess from SAP
    // Core guess ?
    for (size_t m=0; m<X.size(); m++) {
      arma::vec E;
      arma::mat C;
      arma::eig_sym(E, C, Fguess[m]);

      if (M == 1) {
	std::vector<arma::mat> orbs({C});
	std::vector<arma::vec> occs({arma::vec(C.n_cols, arma::fill::zeros)});
	occs[0].subvec(0, Nela - 1).ones();
	occs[0] *= 2;
	dm = std::make_pair(orbs, occs);
      } else {
	std::vector<arma::mat> orbs({C, C});
	std::vector<arma::vec> occs({arma::vec(C.n_cols, arma::fill::zeros), arma::vec(C.n_cols, arma::fill::zeros)});
	occs[0].subvec(0, Nela - 1).ones();
	if(Nelb)
	  occs[1].subvec(0, Nelb - 1).ones();
	dm = std::make_pair(orbs, occs);
      }
    }
  }

  double Eold = 0.0;

  // Data for SCF solver
  int nblocks = Fguess.size();
  arma::uvec number_of_blocks_per_particle_type;
  arma::vec maximum_occupation;
  arma::vec number_of_particles;
  std::vector<std::string> block_descriptions;
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder;
  
  for (size_t istep = 0; istep < nstepwise; istep++) {
    // Run SCF
    if (M == 1) {
      number_of_blocks_per_particle_type = {nblocks};
      maximum_occupation.set_size(nblocks).fill(2.0);
      number_of_particles = {(double) (Nel)};
      for (int k=0; k<nblocks; k++) {
	std::string str = std::to_string(k - maxam);
	block_descriptions.push_back(str);
      }
      fock_builder = restricted_fock_builder;
    } else {
      number_of_blocks_per_particle_type = {nblocks, nblocks};
      maximum_occupation.set_size(2 * nblocks).fill(1.0);
      number_of_particles = {(double) (Nela), (double) (Nelb)};
      for (int l=0; l<2; l++) {
	for (int k=0; k<nblocks; k++) {
	  std::string str = std::to_string(k - maxam);
	  block_descriptions.push_back(str);
	}
      }
      fock_builder = unrestricted_fock_builder;
    }

    printf("\n\n\nIteration %i: solving electronic SCF\n", istep);
    OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
    scfsolver.initialize_with_fock(Fguess);
    scfsolver.error_norm(error_norm);
    scfsolver.convergence_threshold(convergence_threshold);
    scfsolver.verbosity(verbosity);
    scfsolver.maximum_iterations(maxinititer);
    scfsolver.maximum_history_length(diisorder);
    scfsolver.run();

    // Grad the new density matrix
    dm = scfsolver.get_solution();
    save_matrices(dm, scfsolver.get_fock_matrix());

    double Enew = scfsolver.get_energy();
    if (std::abs(Enew - Eold) < init_convergence_threshold) {
      printf("SCF converged: energy changed by %e from last iterations.\n", Enew - Eold);
      break;
    }
    Eold = Enew;
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
    
