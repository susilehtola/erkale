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

#include "openorbitaloptimizer/scfcolver.hpp"

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
  //BasisSetLibrary pbaslib;
  //if(settings.get_string("ProtonBasis").size())
  //pbaslib.load_basis(settings.get_string("ProtonBasis"));

  // Read in SAP potential
  BasisSetLibrary potlib;
  potlib.load_basis(settings.get_string("SAPBasis"));

  auto atoms=load_xyz(settings.get_string("System"),!settings.get_bool("InputBohr"));
  /*std::vector<size_t> proton_indices;
  //if(stricmp(settings.get_string("QuantumProtons"),"")!=0) {
    // Check for '*'
    //std::string str=settings.get_string("QuantumProtons");
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
  */
  
  // Collect classical nuclei
  std::vector<std::tuple<int,double,double,double>> classical_nuclei;
  for(size_t i=0;i<atoms.size();i++) {
    // Ghost nucleus
    if(atoms[i].el.size()>3 && atoms[i].el.substr(atoms[i].el.size()-3,3)=="-Bq")
      continue;
    // Skip over quantum nuclei
    /*bool quantum=false;
    for(auto proton_idx: proton_indices)
      if(proton_idx==i)
        quantum=true;
    if(quantum)
      continue;
    */
    // Add to list
    classical_nuclei.push_back(std::make_tuple(get_Z(atoms[i].el), atoms[i].x, atoms[i].y, atoms[i].z));
  }

  // Construct the basis set
  BasisSet basis;
  construct_basis(basis,atoms,baslib);
  chkpt.write(basis);

  /*
  BasisSet pbasis;
  construct_basis(pbasis,quantum_protons,pbaslib);
  chkpt.write(pbasis,"pbasis");
  */

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
  arma::mat X(BasOrth(S));

  // Guess Fock
  arma::mat Fguess(X.t() * (T) * X);

  OpenOrbitalOptimizer::FockBuilder<double, double> unrestricted_fock_builder = [&X, &T](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Get the electronic orbital coefficients
    arma::mat Ca = X * orbitals[0];
    arma::mat Cb = X * orbitals[1];

    // and occupations
    arma::vec occa = occupations[0];
    arma::vec occb = occupations[1];

    // Densirt matrices
    arma::mat Pa, Pb;
    Pa = Ca * arma::diagmat(occa) * Ca.t();
    Pb = Cb * arma::diagmat(occb) * Cb.t();
    arma::mat P = Pa + Pb;

    // Compute the terms in the Fock matrices
    arma::mat Ja, Jb, Ka, Kb;

    // Form the Fock matrices
    arma::mat Fa = X.t() * (T + Vc + Ja + Jb + Ka) * X;
    arma::mat Fb = X.t() * (T + Vc + Ja + Jb + Kb) * X;

    // Block Fock matrix by m values ??
    std::vector<arma::mat> fock({Fa, Fb});

    // Compute energy terms
    double Ekin = arma::trave(P * T);
    double Enuc = arma::trace(P * Vc);
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



  // Data for SCF solver



  int Nel = basis.Ztot() - Q;
  int Nela = (Nel + M - 1) / 2;
  int Nelb = Nel - Nela;
  printf("Nela = %i Nelb = %i\n", Nela, Nelb);
  if (Nela < 0 or Nelb < 0)
    throw std::logic_error("Negative number of electrons!\n");
  if (Nelb < Nela)
    throw std::logic_error("Nelb > Nela, check your charge and multiplicity!\n");


  // Density matrix
  OpenOrbitalOptimizer::DensityMatrix<double, double> dm;


  // Save density matrix to disk
  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<double, double> &, const OpenOrbitalOptimizer::FockMatrix<double> &)> save_density_matrix = [&chkpt, &M, &X](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm, const OpenOrbitalOptimizer::FockMatrix<double> fock) {
    const OpenOrbitalOptimizer::Orbitals<double> & orbitals = dm.first;
    const OpenOrbitalOptimizer::OrbitalOccupations<double> & occupations = dm.second;
    if (M == 1) {
      arma::mat C = X * orbitals[0];
      arma::vec E = arma::diagvec(orbitals[0].t() * fock[0] * orbitals[0]);
      chpt.write("C", C);
      chpt.write("E", E);
    } else {
      arma::mat Ca = X * orbitals[0];
      arma::mat Cb = X * orbitals[1];
      arma::vec Ea = arma::diagvec(orbitals[0].t() * fock[0] * orbitals[0]);
      arma::vec Eb = arma::diagvec(orbitals[1].t() * fock[1] * orbitals[1]);
      chpt.write("Ca", Ca);
      chpt.write("Cb", Cb);
      chpt.write("Ea", Ea);
      chpt.write("Eb", Eb);
    }
  }; // Save density matrix to disk

  // Set up electronic guess

  // Occuppations

  // Core guess

  // Run SCF

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
    
