#include "basislibrary.h"
#include "basis.h"
#include "checkpoint.h"
#include "density_fitting.h"
#include "jkbuilder.h"
#include "dftgrid.h"
#include "electronic_xc.h"
#ifdef ERKALE_TRUST_REGION
#include "trust_region.h"
#endif
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

#include "eriworker.h"

#include "openorbitaloptimizer/scfsolver.hpp"
// The Armadillo compatibility shim (OpenOrbitalOptimizer::Armadillo::) is a
// separate header that scfsolver.hpp no longer pulls in automatically.
#include "openorbitaloptimizer/armadillo_compat.hpp"

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

// Construct matrices to transform orbitals from real to complex
arma::cx_mat basis_transform_mat(int l) {
  arma::cx_mat Lmat(2 * l + 1, 2 * l + 1, arma::fill::zeros);
  for (int m=-l; m<=l; m++) {
    if (m < 0) {
      Lmat(l+m, l+m) = -std::complex<double>(0.0, std::sqrt(0.5));
      Lmat(l-m, l+m) = std::sqrt(0.5);
    } else if (m == 0)
      Lmat(l, l) = 1.0;
    else {
      Lmat(l-m, l+m) = std::complex<double>(0.0, std::pow(-1.0, m) * std::sqrt(0.5));
      Lmat(l+m, l+m) = std::pow(-1.0, m) * std::sqrt(0.5);
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


  Timer t;
  t.print_time();

  settings.add_scf_settings();
  settings.add_string("ErrorNorm", "Error norm to use in the SCF code", "rms");
  settings.add_int("Verbosity", "Verboseness level", 5);
  settings.add_string("SaveChk", "Checkpoint file to save to", "complex_basis.chk");
  settings.add_string("LoadChk", "Checkpoint file to load from", "");
  settings.add_bool("ComplexBasis", "Use complex basis?", false);
  settings.add_bool("Restricted", "Spin restricted?", false);
  settings.add_string("SCFMethods", "SCF convergence methods to use", "DIIS + LBFGS");
  settings.add_bool("TrustRegion", "Finish the SCF with second-order trust-region optimization (OpenTrustRegion)?", false);
  settings.add_bool("StabilityAnalysis", "Check the stability of the solution within its symmetry, following instabilities if TrustRegion is used?", false);

  // Parse settings
  settings.parse(std::string(argv[1]),true);
  settings.print();
  int Q = settings.get_int("Charge");
  int M = settings.get_int("Multiplicity");
  int verbosity = settings.get_int("Verbosity");
  int maxiter = settings.get_int("MaxIter");
  int diisorder = settings.get_int("DIISOrder");
  double intthr = settings.get_double("IntegralThresh");
  double convergence_threshold = settings.get_double("ConvThr");
  bool verbose = settings.get_bool("Verbose");
  std::string error_norm = settings.get_string("ErrorNorm");
  std::string loadchk = settings.get_string("LoadChk");
  std::string savechk = settings.get_string("SaveChk");
  bool complexbas = settings.get_bool("ComplexBasis");
  int readlinocc = settings.get_int("LinearOccupations");
  std::string linoccfname = settings.get_string("LinearOccupationFile");
  double linB = settings.get_double("LinearB");
  double linE = settings.get_double("LinearE");
  double confinement = settings.get_double("HarmonicConfinement");
  bool unrestricted = !(settings.get_bool("Restricted"));
  std::string guess = settings.get_string("Guess");
  std::string scfmethods = settings.get_string("SCFMethods");
  bool trustregion = settings.get_bool("TrustRegion");
  bool stability = settings.get_bool("StabilityAnalysis");
#ifndef ERKALE_TRUST_REGION
  if(trustregion || stability)
    throw std::runtime_error("TrustRegion and StabilityAnalysis need ERKALE built with OpenTrustRegion.\n");
#endif

  Checkpoint chkpt(savechk,true);

  // Read in basis set
  BasisSetLibrary baslib;
  baslib.load_basis(settings.get_string("Basis"));

  // Read in SAP potential
  BasisSetLibrary potlib;
  potlib.load_basis(settings.get_string("SAPBasis"));

  auto atoms=load_system(settings.get_string("System"),!settings.get_bool("InputBohr"));

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

  int maxam = basis.max_am();
  auto mvals = basis.m_values();
  std::vector<arma::uvec> m_indices(2*maxam+1);
  for (int m = -maxam; m <= maxam; m++) {
    m_indices[m+maxam]=basis.m_indices(m);
  }

  // Coulomb/exchange via the unified driver: it resolves JKMethod
  // (RI / Cholesky / CDFit / four-index), owns the integral engine and
  // exposes the same calcJ/calcK routines used below.
  JKBuilder jk;
  jk.configure(settings);
  if(!jk.is_densityfit())
    throw std::runtime_error("erkale_complex_orbs needs a density-fitting JKMethod (RI, Cholesky or CDFit).\n");
  jk.init(basis, verbose);
  size_t Nbf = basis.Nbf();

  // Exchange-correlation (Hartree-Fock when Method is HF). The basis
  // functions are real, so the density, its gradient and the kinetic
  // energy density depend only on the real part of the density matrix,
  // which the Fock builders below pass on.
  ElectronicXC exc(basis, verbose);
  exc.setup(settings.get_string("Method"), settings.get_string("DFTGrid"));
  // Exact exchange: kfull K + kshort K_sr(omega); this also builds the
  // short-range integrals of a range-separated functional
  jk.set_range_separation(exc.kfull(), exc.kshort(), exc.omega());
  // Without the current density, the kinetic energy density is not gauge
  // invariant in a magnetic field.
  if(linB != 0.0 && exc.is_meta_gga())
    throw std::runtime_error("Meta-GGA functionals are not supported with a magnetic field (LinearB).\n");

  // Calculate matrices
  arma::mat S(basis.overlap());
  arma::mat T(basis.kinetic());
  arma::mat V(basis.nuclear());
  arma::mat Vsap(basis.sap_potential(potlib));
  if(guess=="core")
    Vsap.zeros();
  arma::mat fock_terms = T + V + Vsap; // Helper
  std::vector<arma::mat> Fguess((1 + unrestricted) * (2 * maxam + 1));

  arma::cx_mat D(Nbf, Nbf, arma::fill::zeros);
  if (complexbas) {
    const auto & shells = basis.shells();
    for (size_t i=0; i<shells.size(); i++) {
      // The transform acts on one set of 2l+1 real functions in the order
      // m = -l, ..., l. The functions are not stored in that order for p
      // shells (x, y, z), so place each function by its m label. A
      // generally contracted shell carries nctr such sets stacked
      // contraction-slowest, so place the transform block-diagonally,
      // once per contraction.
      const int l = shells[i].am();
      const arma::cx_mat Tm = basis_transform_mat(l);
      const size_t Nlm = Tm.n_rows;
      for(size_t ic=0; ic<shells[i].Nctr(); ic++) {
        const size_t i0 = shells[i].first_ind() + ic*Nlm;
        // Index of the function with m = k-l
        arma::uvec idx(Nlm);
        for(size_t p=0; p<Nlm; p++)
          idx(mvals(i0+p)+l) = i0+p;
        D.submat(idx, idx) = Tm;
      }
    }
  } else
    D.eye();

  // Blocked matrices
  arma::mat S_c = arma::real(D.t() * S * D);
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

  // Force occupations? LinearOccupations < 0 reads per-symmetry
  // alpha/beta occupations from LinearOccupationFile; otherwise fill by
  // aufbau from the total electron count. (Only read the file when
  // actually used, so normal runs don't depend on linoccs.dat.)
  arma::vec occnuma(X.size(), arma::fill::zeros);
  arma::vec occnumb(X.size(), arma::fill::zeros);
  if (readlinocc < 0) {
    arma::imat linoccs;
    linoccs.load(linoccfname,arma::raw_ascii);
    for (size_t i=0; i<linoccs.n_rows; i++) {
      int occa = linoccs(i, 0);
      int occb = linoccs(i, 1);
      int m = linoccs(i, 2);
      if (std::abs(m) > maxam)
	throw std::logic_error("Not enough basis functions to satisfy symmetry restrictions!\n");
      occnuma(m + maxam) += occa;
      occnumb(m + maxam) += occb;
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
  fflush(stdout);
  if (Nela < 0 or Nelb < 0)
    throw std::logic_error("Negative number of electrons!\n");
  if (Nelb > Nela)
    throw std::logic_error("Nelb > Nela, check your charge and multiplicity!\n");

  if (!unrestricted) {
    for (size_t occ = 0; occ < occnuma.size(); occ++) {
      if (occnuma(occ) != occnumb(occ))
	throw std::logic_error("Alpha and beta occupations do not match even though calculation is spin restricted!");
    }
  }

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

  // One-electron field operator added to the Fock matrix. Collects the
  // magnetic terms (orbital Zeeman -1/2 B L_z, diamagnetic 1/8 B^2 (x^2+y^2)),
  // the electric dipole (E z along the bond axis), and an optional harmonic
  // confinement (1/2 k r^2). z preserves L_z, so the m-block structure is kept;
  // it mixes l within a block, which is what polarizes the density. The
  // confinement is an artificial regulariser (a static field has no bound
  // ground state) -- leave it off (k=0) for weak fields with a non-diffuse
  // basis; turn it on to prevent variational collapse toward the continuum.
  arma::mat Bterms(Nbf, Nbf, arma::fill::zeros);
  if (linB || linE || confinement) {
    double cenx = 0.0, ceny = 0.0, cenz = 0.0;
    std::vector<arma::mat> dip = basis.moment(1, cenx, ceny, cenz);
    std::vector<arma::mat> momstack = basis.moment(2, cenx, ceny, cenz);
    arma::mat xymat = momstack[getind(2, 0, 0)] + momstack[getind(0, 2, 0)];
    arma::mat zmat = dip[getind(0, 0, 1)];
    arma::mat r2mat = xymat + momstack[getind(0, 0, 2)];
    const auto & Smat = complexbas ? S_c : S;
    if(complexbas) {
      xymat = arma::real(D.t()*xymat*D);
      zmat = arma::real(D.t()*zmat*D);
      r2mat = arma::real(D.t()*r2mat*D);
    }
    for (size_t j = 0; j < Nbf; j++)
      Bterms.col(j) = -0.5 * linB * mvals(j) * Smat.col(j)   // orbital Zeeman
        + 0.125 * linB * linB * xymat.col(j)                 // diamagnetic
        + linE * zmat.col(j)                                 // electric dipole
        + 0.5 * confinement * r2mat.col(j);                  // harmonic confinement
  }

  std::function<std::tuple<arma::mat, arma::mat, arma::cx_mat>(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> electronic_terms = [&](const auto & orbitals, const auto & occupations) {
    arma::mat C;
    arma::vec occs;
    std::tie(C,occs) = collect_orbitals(orbitals, occupations);

    arma::cx_mat C_c = D * C;
    arma::mat P = arma::real(C_c * arma::diagmat(occs) * C_c.t());
    arma::mat J = jk.calcJ(P);

    // Exact exchange, including its admixture. The code in ERKALE has a
    // different convention for complex integrals; the complex conjugate
    // makes it compatible with this code.
    const std::vector<double> occv = arma::conv_to<std::vector<double>>::from(occs);
    arma::cx_mat K(Nbf, Nbf, arma::fill::zeros);
    if(exc.kfull() != 0.0)
      K -= exc.kfull() * arma::conj(jk.calcK(C_c, occv, S));
    if(exc.kshort() != 0.0)
      K -= exc.kshort() * arma::conj(jk.calcK_short(C_c, occv, S));

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

  OpenOrbitalOptimizer::Armadillo::FockBuilder<double, double> restricted_fock_builder = [&](const OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    std::vector<arma::mat> fock(2 * maxam + 1);
    arma::mat P, J;
    arma::cx_mat K;
    std::tie(P, J, K) = electronic_terms(orbitals, occupations);
    arma::cx_mat cP = complex_density(orbitals, occupations);
    arma::mat cbP = complex_basis_density(orbitals, occupations);

    arma::mat Vxc;
    double Exc = exc.eval(P, Vxc);

    // Form the Fock matrices
    arma::cx_mat F = T + V + J + 0.5*K + Vxc;
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
    double Etot = Ekin + Enuc + Ecoul + Eexch + Exc + Enucr + Emag;

    if(verbosity >= 10) {
      printf("e kinetic energy            % .10f\n", Ekin);
      printf("e nuclear attraction        % .10f\n", Enuc);
      printf("e-e Coulomb energy          % .10f\n", Ecoul);
      printf("e-e exchange energy         % .10f\n", Eexch);
      if(exc.active())
        printf("e exchange-correlation      % .10f\n", Exc);
      printf("nuclear repulsion energy    % .10f\n", Enucr);
      printf("field interaction energy    % .10f\n", Emag);
      printf("Total energy                % .10f\n", Etot);
      fflush(stdout);
    }

    return std::make_pair(Etot, fock);
  }; //restricted Fock builder


  OpenOrbitalOptimizer::Armadillo::FockBuilder<double, double> unrestricted_fock_builder = [&](const OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double> & dm) {

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

    arma::mat Vxca, Vxcb;
    double Exc = exc.eval(Pa, Pb, Vxca, Vxcb);

    arma::mat Ba = - 0.5 * linB * S;
    arma::mat Bb = + 0.5 * linB * S;

    // Form the Fock matrices
    arma::cx_mat Fa = T + V + Ja + Jb + Ka + Vxca + Ba;
    arma::cx_mat Fb = T + V + Ja + Jb + Kb + Vxcb + Bb;
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
    double Etot = Ekin + Enuc + Ecoul + Eexch + Exc + Enucr + Emag;

    if(verbosity >= 10) {
      printf("e kinetic energy            % .10f\n", Ekin);
      printf("e nuclear attraction        % .10f\n", Enuc);
      printf("e-e Coulomb energy          % .10f\n", Ecoul);
      printf("e-e exchange energy         % .10f\n", Eexch);
      if(exc.active())
        printf("e exchange-correlation      % .10f\n", Exc);
      printf("nuclear repulsion energy    % .10f\n", Enucr);
      printf("field interaction energy    % .10f\n", Emag);
      printf("Total energy                % .10f\n", Etot);
      fflush(stdout);
    }

    return std::make_pair(Etot, fock);
  }; //unrestricted Fock builder

  // Save matrices to disk
  std::function<void(const OpenOrbitalOptimizer::Armadillo::FockMatrix<double> &)> save_matrices = [&](const OpenOrbitalOptimizer::Armadillo::FockMatrix<double> fock) {

    if(!unrestricted) {
      for (size_t m=0; m<X.size(); m++) {
	std::string fm = "F" + std::to_string(m);
	chkpt.write(fm,fock[m]);
      }
    } else {
      for (size_t m=0; m<X.size(); m++) {
	std::string fam = "Fa" + std::to_string(m);
	chkpt.write(fam, fock[m]);
	std::string fbm = "Fb" + std::to_string(m);
	chkpt.write(fbm, fock[X.size() + m]);
      }
    }
  }; // Save matrices to disk


  if(loadchk != "") {
    Checkpoint load(loadchk,false);

    if(!unrestricted) {
      for (size_t m=0; m<X.size(); m++) {
	std::string fm = "F" + std::to_string(m);
	load.read(fm,Fguess[m]);
      }
    } else {
      for (size_t m=0; m<X.size(); m++) {
	std::string fam = "Fa" + std::to_string(m);
	load.read(fam,Fguess[m]);
	std::string fbm = "Fb" + std::to_string(m);
	load.read(fbm,Fguess[X.size() + m]);
      }
    }
  } else { // Else guess Fock from SAP/core
    for (size_t m=0; m<X.size(); m++) {
      Fguess[m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
      if (unrestricted)
	Fguess[X.size() + m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
    }
  }// if(loadchk != "")
    

  // Data for SCF solver
  int nblocks = Fguess.size();
  arma::uvec number_of_blocks_per_particle_type;
  arma::vec maximum_occupation;
  arma::vec number_of_particles;
  arma::vec number_of_particles_per_block;
  std::vector<std::string> block_descriptions;
  OpenOrbitalOptimizer::Armadillo::FockBuilder<double, double> fock_builder;

  // Run SCF
  if (!unrestricted) {
    number_of_blocks_per_particle_type = {(arma::uword) nblocks};
    maximum_occupation.set_size(nblocks).fill(2.0);
    number_of_particles = {(double) (Nel)};
    if (readlinocc < 0)
      number_of_particles_per_block = occnuma + occnumb;
    for (int k=0; k<nblocks; k++) {
      std::string str = "m=" + std::to_string(k - maxam);
      block_descriptions.push_back(str);
    }
    fock_builder = restricted_fock_builder;
  } else {
    number_of_blocks_per_particle_type = {(arma::uword) nblocks / 2, (arma::uword) nblocks / 2};
    maximum_occupation.set_size(nblocks).fill(1.0);
    number_of_particles = {(double) (Nela), (double) (Nelb)};
    if (readlinocc < 0)
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
  fflush(stdout);
  OpenOrbitalOptimizer::Armadillo::SCFSolver<double,double> scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  if (readlinocc < 0)
    scfsolver.fixed_number_of_particles_per_block(number_of_particles_per_block);
  scfsolver.initialize_with_fock(Fguess);
  if (readlinocc < 0)
    scfsolver.frozen_occupations(true);
  scfsolver.error_norm(error_norm);
  scfsolver.convergence_threshold(convergence_threshold);
  scfsolver.verbosity(verbosity);
  scfsolver.maximum_iterations(maxiter);
  scfsolver.maximum_history_length(diisorder);
  scfsolver.run(scfmethods);

  auto fock = scfsolver.get_fock_matrix();
  double E = scfsolver.get_energy();

#ifdef ERKALE_TRUST_REGION
  // Second-order trust-region optimization and stability analysis. The
  // rotations stay within the symmetry blocks.
  if(trustregion || stability) {
    const int otr_verbose = verbosity >= 10 ? 4 : (verbosity >= 5 ? 3 : 2);
    TrustRegionSCF tr(fock_builder, scfsolver.get_solution());
    if(trustregion) {
      printf("\nTrust-region optimization with OpenTrustRegion\n");
      fflush(stdout);
      tr.optimize(convergence_threshold, stability, otr_verbose);
    } else {
      printf("\nStability analysis with OpenTrustRegion\n");
      fflush(stdout);
      if(!tr.is_stable(convergence_threshold, otr_verbose))
        printf("Warning: the solution is unstable within its symmetry; use TrustRegion to follow the instability.\n");
      else
        printf("The solution is stable within its symmetry.\n");
      fflush(stdout);
    }
    auto ret = fock_builder(tr.density_matrix());
    E = ret.first;
    fock = ret.second;
  }
#endif
  save_matrices(fock);

  printf("SCF converged. Total energy is % .10f\n", E);
  fflush(stdout);

  printf("\nRunning program took %s.\n",t.elapsed().c_str());
  fflush(stdout);
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
