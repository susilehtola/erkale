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

// Construct matrices to transform orbitals from real to complex
arma::cx_mat basis_transform_mat(int l) {
  arma::cx_mat Lmat(2 * l + 1, 2 * l + 1, arma::fill::zeros);
  for (int m=-l; m<=l; m++) {
    if (m < 0) {
      Lmat(m+l, m+l) -= std::complex<double>(0.0, std::sqrt(0.5));
      //Lmat(m+l, 2*l-(m+l)) += std::sqrt(0.5);
      Lmat(2*l-(m+l), m+l) += std::sqrt(0.5);
    } else if (m == 0)
      Lmat(l, l) += 1.0;
    else {
      //Lmat(m+l, 2*l-(m+l)) += std::complex<double>(0.0, std::pow(-1.0, m) * std::sqrt(0.5));
      Lmat(2*l-(m+l), m+l) += std::complex<double>(0.0, std::pow(-1.0, m) * std::sqrt(0.5));
      Lmat(m+l, m+l) += std::pow(-1.0, m) * std::sqrt(0.5);
    }
  }
  return Lmat;
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
  settings.add_bool("Complexbas", "Use complex basis?", false);
  settings.add_bool("Restricted", "Spin restricted?", false);

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
  bool verbose = settings.get_bool("Verbose");
  size_t fitmem = 1000000*settings.get_int("FittingMemory");
  std::string error_norm = settings.get_string("ErrorNorm");
  std::string loadchk = settings.get_string("LoadChk");
  std::string savechk = settings.get_string("SaveChk");
  bool complexbas = settings.get_bool("Complexbas");
  int readlinocc = settings.get_int("LinearOccupations");
  std::string linoccfname = settings.get_string("LinearOccupationFile");
  double linB = settings.get_double("LinearB");
  bool unrestricted = !(settings.get_bool("Restricted"));
  bool density_fitting = settings.get_bool("DensityFitting");
  std::string guess = settings.get_string("Guess");

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

  int maxam = basis.get_max_am();
  auto mvals = basis.get_m_values();
  std::vector<arma::uvec> m_indices(2*maxam+1);
  for (int m = -maxam; m <= maxam; m++) {
    m_indices[m+maxam]=basis.m_indices(m);
  }

  // Construct density fitting basis set
  BasisSetLibrary fitlib;
  BasisSet dfitbas;
  ERIchol chol;
  DensityFit dfit;
  bool direct = settings.get_bool("Direct");
  double fitthr = settings.get_double("FittingThreshold");
  double cholfitthr = settings.get_double("FittingCholeskyThreshold");
  size_t Npairs = 0;

  if (density_fitting) {
    fitlib.load_basis(settings.get_string("FittingBasis"));
    // Construct fitting basis
    bool uselm = settings.get_bool("UseLM");
    settings.set_bool("UseLM", true);
    construct_basis(dfitbas,basis.get_nuclei(), fitlib);
    dfitbas.coulomb_normalize();
    settings.set_bool("UseLM", uselm);
    Npairs = dfit.fill(basis, dfitbas, direct, intthr, fitthr, cholfitthr);
  } else {
    double cholthr = settings.get_double("CholeskyThr");
    double cholshthr = settings.get_double("CholeskyShThr");
    double shtol = settings.get_double("IntegralThresh");
    Npairs = chol.fill(basis, cholthr, cholshthr, shtol, verbose);
  }
  size_t Nbf = basis.get_Nbf();

  // Calculate matrices
  arma::mat S(basis.overlap());
  arma::mat T(basis.kinetic());
  arma::mat V(basis.nuclear());
  arma::mat Vsap(basis.sap_potential(potlib));
  if(guess=="core")
    Vsap.zeros();
  arma::mat fock_terms = T + V + Vsap; // Helper

  arma::cx_mat D(Nbf, Nbf, arma::fill::zeros);
  if (complexbas) {
    const auto & shells = basis.get_shells();
    for (size_t i=0; i<shells.size(); i++) {
      D.submat(shells[i].get_first_ind(), shells[i].get_first_ind(), shells[i].get_last_ind(), shells[i].get_last_ind()) = basis_transform_mat(shells[i].get_am());
    }
  } else
    D.eye();

  // Blocked matrices
  arma::mat S_c = arma::real(D * S * D.t());
  size_t Nmo=0;
  std::vector<arma::mat> X(2*maxam+1);
  for (size_t m=0; m<X.size(); m++) {
    const auto & Smat = complexbas ? S_c : S;
    X[m] = BasOrth(Smat(m_indices[m], m_indices[m]));
    Nmo += X[m].n_cols;
  }

  int Nel = basis.Ztot() - Q;
  int Nela;
  int Nelb;

  // Force occupations?
  arma::imat linoccs;
  linoccs.load(linoccfname,arma::raw_ascii);
  arma::vec occnuma(X.size(), arma::fill::zeros);
  arma::vec occnumb(X.size(), arma::fill::zeros);
  std::vector<arma::uvec> occsym(linoccs.n_rows);
  if (readlinocc < 0) {
    for (size_t i=0; i<linoccs.n_rows; i++) {
      occnuma(linoccs(i, 2) + maxam) += linoccs(i, 0);
      occnumb(linoccs(i, 2) + maxam) += linoccs(i, 1);
    }
    Nela = arma::accu(occnuma);
    Nelb = arma::accu(occnumb);
    if ((Nela - Nelb) + 1 != M)
      throw std::logic_error("Multiplicity does not match occupations!");
  } else {
    Nela = (Nel + M - 1) / 2;
    Nelb = Nel - Nela;
  }
  printf("Nela = %i Nelb = %i\n", Nela, Nelb);
  if (Nela < 0 or Nelb < 0)
    throw std::logic_error("Negative number of electrons!\n");
  if (Nelb > Nela)
    throw std::logic_error("Nelb > Nela, check your charge and multiplicity!\n");

  if (!unrestricted) {
    for (size_t occ = 0; occ < occnuma.size(); occ++) {
      if (occnuma(occ) != occnumb(occ))
	throw std::logic_error("Alpha and beta occupations do not match even though calculation is spin retricted!");
    }
  }

  // Guess Fock
  std::vector<arma::mat> Fguess((1 + unrestricted) * (2 * maxam + 1));
  for (size_t m=0; m<X.size(); m++) {
    Fguess[m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
    if (unrestricted)
      Fguess[X.size() + m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
  }


  std::function<arma::vec(const int & nocc, const int & m)> set_occupations = [&](const int & nocc, const int & m) {

    arma::vec occsub(X[m].n_cols, arma::fill::zeros);
    if (nocc > occsub.size())
      throw std::logic_error("Not enough basis functions to satisfy symmetry restrictions!\n");

    double nleft = nocc;
    size_t io = 0;
    while (nleft > 0.0) {
      if (io == occsub.size())
	throw std::logic_error("Index overflow!\n");
      double occ;
      if (unrestricted)
	occ = std::min(1.0, nleft);
      else
	occ = std::min(2.0, nleft);
      occsub(io) = occ;
      nleft -= occ;
      io++;
    }
    return occsub;
  };

  std::function<std::pair<arma::mat,arma::vec>(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> collect_orbitals = [&](const auto & orbitals, const auto & occupations) {
    arma::vec occs(Nmo, arma::fill::zeros);
    arma::mat C(Nbf, Nmo, arma::fill::zeros);
    size_t imo=0;
    for (size_t m=0; m<X.size(); m++) {
      arma::mat Csub = X[m] * orbitals[m];
      arma::mat Cpad(Nbf,X[m].n_cols,arma::fill::zeros);
      Cpad.rows(m_indices[m]) = Csub;
      occs.subvec(imo, imo + X[m].n_cols - 1) = occupations[m];
      C.cols(imo,imo+X[m].n_cols-1) = Cpad;
      imo += X[m].n_cols;
    }
    if(imo != Nmo)
      throw std::logic_error("Indexing problem\n");
    return std::make_pair(C, occs);
  };

  arma::mat Bterms(Nbf, Nbf, arma::fill::zeros);
  if (linB) {
    double cenx = 0.0, ceny = 0.0, cenz = 0.0;
    std::vector<arma::mat> momstack = basis.moment(2, cenx, ceny, cenz);
    arma::mat xymat = momstack[getind(2, 0, 0)] + momstack[getind(0, 2, 0)];

    if(complexbas) {
      xymat = arma::real(D.t()*xymat*D);
      for (size_t j = 0; j < Nbf; j++)
	Bterms.col(j) = 0.5 * linB * mvals(j) * S_c.col(j) + 0.125 * linB * linB * xymat.col(j);

    } else {
      for (size_t j = 0; j < Nbf; j++)
	Bterms.col(j) = 0.5 * linB * mvals(j) * S.col(j) + 0.125 * linB * linB * xymat.col(j);

    }
  }

  std::function<std::tuple<arma::mat, arma::mat, arma::cx_mat>(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> electronic_terms = [&](const auto & orbitals, const auto & occupations) {
    arma::mat C;
    arma::vec occs;
    std::tie(C,occs) = collect_orbitals(orbitals, occupations);

    arma::cx_mat C_c = D * C;
    arma::mat P = arma::real(C_c * arma::diagmat(occs) * C_c.t());
    arma::mat J;
    arma::cx_mat K;
    if (density_fitting) {
      J = dfit.calcJ(P);
      K = -dfit.calcK(C_c, arma::conv_to<std::vector<double>>::from(occs), fitmem);
    } else {
      J = chol.calcJ(P);
      K = -chol.calcK(C_c, arma::conv_to<std::vector<double>>::from(occs));
    }

    // The code in ERKALE has a different convention for complex integrals; this modification makes it compatible with this code
    K = arma::conj(K);

    return std::make_tuple(P, J, K);
  };

  std::function<arma::cx_mat(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> complex_density = [&](const auto & orbitals, const auto & occupations) {
    arma::mat C;
    arma::vec occs;
    std::tie(C,occs) = collect_orbitals(orbitals, occupations);

    arma::cx_mat C_c = D * C;
    arma::cx_mat Pc = C_c * arma::diagmat(occs) * C_c.t();
    return Pc;
  };

  std::function<arma::mat(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> complex_basis_density = [&](const auto & orbitals, const auto & occupations) {
    arma::mat C;
    arma::vec occs;
    std::tie(C,occs) = collect_orbitals(orbitals, occupations);

    arma::mat P = C * arma::diagmat(occs) * C.t();
    return P;
  };

  OpenOrbitalOptimizer::FockBuilder<double, double> restricted_fock_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    std::vector<arma::mat> fock(2 * maxam + 1);
    arma::mat P, J;
    arma::cx_mat K;
    std::tie(P, J, K) = electronic_terms(orbitals, occupations);
    arma::cx_mat cP = complex_density(orbitals, occupations);
    arma::mat cbP = complex_basis_density(orbitals, occupations);

    // Form the Fock matrices
    arma::cx_mat F = T + V + J + 0.5*K;
    arma::cx_mat DFD;
    if(complexbas)
      DFD = D.t()*F*D + Bterms;
    else
      DFD = (F + Bterms)*std::complex<double>(1.0,0.0);
    for (size_t m=0; m<X.size(); m++)
      fock[m] = arma::real(DFD(m_indices[m], m_indices[m]));
    for (size_t m=0; m<X.size(); m++)
      fock[m] = X[m].t() * fock[m] * X[m];

    // Compute energy terms
    double Ekin = arma::trace(P * T);
    double Enuc = arma::trace(P * V);
    double Ecoul = 0.5 * arma::trace(P * J);
    double Eexch = 0.25 * std::real(arma::trace(cP * K));
    double Emag = arma::trace(cbP * Bterms);
    double Etot = Ekin + Enuc + Ecoul + Eexch + Enucr + Emag;

    if(verbosity >= 10) {
      printf("e kinetic energy            % .10f\n", Ekin);
      printf("e nuclear attraction        % .10f\n", Enuc);
      printf("e-e Coulomb energy          % .10f\n", Ecoul);
      printf("e-e exchange energy         % .10f\n", Eexch);
      printf("nuclear repulsion energy    % .10f\n", Enucr);
      printf("magnetic interaction energy % .10f\n", Emag);
      printf("Total energy                % .10f\n", Etot);
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
    // Beta electrons
    std::vector<arma::mat> b_orbs;
    std::vector<arma::vec> b_occs;
    for (size_t i = occupations.size() / 2; i < occupations.size(); i++) {
      b_orbs.push_back(orbitals[i]);
      b_occs.push_back(occupations[i]);
    }

    arma::mat Pa, Ja;
    arma::cx_mat Ka;
    std::tie(Pa, Ja, Ka) = electronic_terms(a_orbs, a_occs);
    arma::cx_mat cPa = complex_density(a_orbs, a_occs);
    arma::mat cbPa = complex_basis_density(a_orbs, a_occs);

    arma::mat Pb, Jb;
    arma::cx_mat Kb;
    std::tie(Pb, Jb, Kb) = electronic_terms(b_orbs, b_occs);
    arma::cx_mat cPb = complex_density(b_orbs, b_occs);
    arma::mat cbPb = complex_basis_density(b_orbs, b_occs);

    arma::mat P = Pa + Pb;

    arma::mat Ba = - 0.5 * linB * S;
    arma::mat Bb = + 0.5 * linB * S;

    // Form the Fock matrices
    arma::cx_mat Fa = T + V + Ja + Jb + Ka + Ba;
    arma::cx_mat Fb = T + V + Ja + Jb + Kb + Bb;
    arma::cx_mat DFDa, DFDb;
    if(complexbas) {
      DFDa = D.t() * Fa * D + Bterms;
      DFDb = D.t() * Fb * D + Bterms;
    } else {
      DFDa = (Fa + Bterms)*std::complex<double>(1.0,0.0);
      DFDb = (Fb + Bterms)*std::complex<double>(1.0,0.0);
    }
    for (size_t m=0; m<X.size(); m++) {
      fock[m] = arma::real(DFDa(m_indices[m], m_indices[m]));
      fock[X.size() + m] = arma::real(DFDb(m_indices[m], m_indices[m]));
    }
    for (size_t m=0; m<X.size(); m++) {
      fock[m] = X[m].t() * fock[m] * X[m];
      fock[X.size() + m] = X[m].t() * fock[X.size() + m] * X[m];
    }

    // Compute energy terms
    double Ekin = arma::trace(P * T);
    double Enuc = arma::trace(P * V);
    double Ecoul = 0.5 * arma::trace(P * (Ja + Jb));
    double Eexch = 0.5 * std::real(arma::trace(cPa * Ka) + arma::trace(cPb * Kb));
    double Emag = arma::trace((cbPa + cbPb) * Bterms) - linB * 0.5 * (Nela - Nelb);
    double Etot = Ekin + Enuc + Ecoul + Eexch + Enucr + Emag;

    if(verbosity >= 10) {
      printf("e kinetic energy            % .10f\n", Ekin);
      printf("e nuclear attraction        % .10f\n", Enuc);
      printf("e-e Coulomb energy          % .10f\n", Ecoul);
      printf("e-e exchange energy         % .10f\n", Eexch);
      printf("nuclear repulsion energy    % .10f\n", Enucr);
      printf("magnetic interaction energy % .10f\n", Emag);
      printf("Total energy                % .10f\n", Etot);
    }

    return std::make_pair(Etot, fock);
  }; //unrestricted Fock builder

  // Save matrices to disk
  std::function<void(const OpenOrbitalOptimizer::DensityMatrix<double, double> &, const OpenOrbitalOptimizer::FockMatrix<double> &)> save_matrices = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm, const OpenOrbitalOptimizer::FockMatrix<double> fock) {
    const OpenOrbitalOptimizer::Orbitals<double> & orbitals = dm.first;
    if (!unrestricted) {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat C = X[m] * orbitals[m];
	arma::vec E = arma::diagvec(orbitals[m].t() * fock[m] * orbitals[m]);
	chkpt.write("C", C);
	chkpt.write("E", E);
      }
    } else {
      for (size_t m=0; m<X.size(); m++) {
	arma::mat Ca = X[m] * orbitals[m];
	arma::mat Cb = X[m] * orbitals[X.size()+m];
	arma::vec Ea = arma::diagvec(orbitals[m].t() * fock[m] * orbitals[m]);
	arma::vec Eb = arma::diagvec(orbitals[X.size()+m].t() * fock[X.size()+m] * orbitals[X.size()+m]);
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
  arma::vec number_of_particles_per_block;
  std::vector<std::string> block_descriptions;
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder;

  // Run SCF
  if (!unrestricted) {
    number_of_blocks_per_particle_type = {nblocks};
    maximum_occupation.set_size(nblocks).fill(2.0);
    number_of_particles = {(double) (Nel)};
    if (readlinocc)
      number_of_particles_per_block = occnuma + occnumb;
    for (int k=0; k<nblocks; k++) {
      std::string str = "m=" + std::to_string(k - maxam);
      block_descriptions.push_back(str);
    }
    fock_builder = restricted_fock_builder;
  } else {
    number_of_blocks_per_particle_type = {nblocks / 2, nblocks / 2};
    maximum_occupation.set_size(nblocks).fill(1.0);
    number_of_particles = {(double) (Nela), (double) (Nelb)};
    if (readlinocc)
      number_of_particles_per_block = arma::join_cols(occnuma, occnumb);
    for (int s=0; s<2; s++) {
      std::string spin = s ? "beta" : "alpha";
      for (int k=0; k<nblocks / 2; k++) {
	std::string str = spin + " m=" + std::to_string(k - maxam);
	block_descriptions.push_back(str);
      }
    }
    fock_builder = unrestricted_fock_builder;
  }

  printf("\n\n\nSolving SCF\n");
  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  if (readlinocc)
    scfsolver.fixed_number_of_particles_per_block(number_of_particles_per_block);
  scfsolver.initialize_with_fock(Fguess);
  if (readlinocc)
    scfsolver.frozen_occupations(true);
  scfsolver.error_norm(error_norm);
  scfsolver.convergence_threshold(convergence_threshold);
  scfsolver.verbosity(verbosity);
  scfsolver.maximum_iterations(maxinititer);
  scfsolver.maximum_history_length(diisorder);
  scfsolver.run();

  //dm = scfsolver.get_solution();
  //save_matrices(dm, scfsolver.get_fock_matrix());

  double E = scfsolver.get_energy();
  //printf("SCF converged: energy is %e.\n", E);

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
