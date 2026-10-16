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

/**
 * Transform from the real to the complex basis functions; the identity
 * in the real basis.
 */
arma::cx_mat complex_transform(const BasisSet & basis, bool complexbas) {
  const size_t Nbf = basis.Nbf();
  arma::cx_mat D(Nbf, Nbf, arma::fill::zeros);
  if (!complexbas) {
    D.eye();
    return D;
  }
  const arma::ivec mvals = basis.m_values();
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
  return D;
}

/// The functions of each m = -maxam, ..., maxam
std::vector<arma::uvec> m_blocks(const BasisSet & basis) {
  const int maxam = basis.max_am();
  std::vector<arma::uvec> idx(2*maxam+1);
  for (int m = -maxam; m <= maxam; m++)
    idx[m+maxam] = basis.m_indices(m);
  return idx;
}

/**
 * Orthonormal bases of the m blocks from the overlap S of the (real or
 * complex) basis functions. With CylindricalSymmetry, the functions of
 * m and -m correspond shell by shell, so the -m block reuses the
 * orthonormal basis of m, which would otherwise differ by the signs of
 * its vectors.
 */
std::vector<arma::mat> block_bases(const arma::mat & S, const std::vector<arma::uvec> & m_indices, bool cylindrical) {
  const int maxam = (m_indices.size()-1)/2;
  std::vector<arma::mat> X(m_indices.size());
  for (size_t m=0; m<X.size(); m++)
    X[m] = BasOrth(S(m_indices[m], m_indices[m]));
  if (cylindrical)
    for (int m = 1; m <= maxam; m++) {
      const arma::uvec & ip = m_indices[maxam+m];
      const arma::uvec & im = m_indices[maxam-m];
      if (arma::abs(S(ip, ip) - S(im, im)).max() > 1e-10)
        throw std::logic_error("The functions of m=" + std::to_string(m) + " and m=" + std::to_string(-m) + " do not correspond.\n");
      X[maxam-m] = X[maxam+m];
    }
  return X;
}

/**
 * The Fock matrices of the m blocks of an old calculation, given in the
 * orthonormal bases Xold of the functions oldidx of the old basis, in the
 * orthonormal bases X of the functions idx of the new basis, where Sno is
 * the overlap of the new and old (real or complex) functions. The old Fock
 * operator is carried over, with its eigenvalues on the old orbitals that
 * the new basis represents; the directions the old basis does not
 * represent get the eigenvalue lambda, which lies above those of the old
 * orbitals so that the aufbau occupies the same orbitals as before. In the
 * same basis, this is the old Fock matrix.
 */
std::vector<arma::mat> project_fock(const std::vector<arma::mat> & Fold, const std::vector<arma::mat> & Xold, const std::vector<arma::uvec> & oldidx, const std::vector<arma::mat> & X, const std::vector<arma::uvec> & idx, const arma::mat & Sno, double lambda) {
  const int maxam = (idx.size()-1)/2;
  const int oldmaxam = (oldidx.size()-1)/2;
  std::vector<arma::mat> F(X.size());
  for (int m = -maxam; m <= maxam; m++) {
    const arma::mat & Xm = X[m+maxam];
    F[m+maxam] = lambda*arma::eye(Xm.n_cols, Xm.n_cols);
    if (std::abs(m) > oldmaxam)
      continue;
    const size_t mo = m+oldmaxam;
    const arma::mat O(Xm.t() * Sno(idx[m+maxam], oldidx[mo]) * Xold[mo]);
    F[m+maxam] += O * (Fold[mo] - lambda*arma::eye(arma::size(Fold[mo]))) * O.t();
  }
  return F;
}

/**
 * Matrix of the reflection y -> -y through the molecular axis in the
 * real basis: the functions of m < 0 go as sin(|m| phi) and change sign.
 */
arma::mat reflection_matrix(const arma::ivec & mvals) {
  arma::vec sign(mvals.n_elem, arma::fill::ones);
  sign(arma::find(mvals < 0)).fill(-1.0);
  return arma::diagmat(sign);
}

/**
 * Matrix of the inversion through the centroid of the nuclei in the real
 * basis, which maps a function of angular momentum l on one nucleus to
 * (-1)^l times the same function on the image nucleus. Empty if the
 * molecule, basis included, has no inversion center; the overlap and
 * nuclear attraction matrices S and V must be invariant.
 */
arma::mat inversion_matrix(const BasisSet & basis, const arma::mat & S, const arma::mat & V) {
  const std::vector<nucleus_t> nuclei(basis.nuclei());
  coords_t c = {0.0, 0.0, 0.0};
  for (const auto & nuc : nuclei)
    c = c + nuc.r;
  c = c / (double) nuclei.size();

  arma::mat I(basis.Nbf(), basis.Nbf(), arma::fill::zeros);
  for (const auto & a : nuclei) {
    const coords_t image = c*2.0 - a.r;
    bool found = false;
    for (const auto & b : nuclei) {
      if (norm(b.r - image) > 1e-6 || b.Z != a.Z || b.bsse != a.bsse || b.shells.size() != a.shells.size())
        continue;
      for (size_t is=0; is<a.shells.size(); is++) {
        const GaussianShell & sa = *a.shells[is];
        const GaussianShell & sb = *b.shells[is];
        if (sa.am() != sb.am() || sa.Nbf() != sb.Nbf())
          return arma::mat();
        const double sign = (sa.am() % 2) ? -1.0 : 1.0;
        for (size_t k=0; k<sa.Nbf(); k++)
          I(sb.first_ind()+k, sa.first_ind()+k) = sign;
      }
      found = true;
      break;
    }
    if (!found)
      return arma::mat();
  }
  // Same nuclei and shells, but are the exponents and contractions the same?
  const double tol = 1e-8;
  if (arma::abs(I*S*I.t() - S).max() > tol || arma::abs(I*V*I.t() - V).max() > tol)
    return arma::mat();
  return I;
}

/// Name of the term with angular momentum Lambda about the axis
std::string lambda_label(int Lambda) {
  static const char * names[] = {"Sigma", "Pi", "Delta", "Phi", "Gamma", "H", "I", "K", "L", "M"};
  return Lambda < 10 ? names[Lambda] : "Lambda=" + std::to_string(Lambda);
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
  // The orbitals are blocked by m, which is exact only for a linear
  // molecule along the z axis
  settings.set_bool("LinearSymmetry", true);
  settings.add_string("ErrorNorm", "Error norm to use in the SCF code", "rms");
  settings.add_int("Verbosity", "Verboseness level", 5);
  settings.add_string("SaveChk", "Checkpoint file to save to", "complex_basis.chk");
  settings.add_string("LoadChk", "Checkpoint file to load from", "");
  settings.add_bool("ComplexBasis", "Use complex basis?", false);
  settings.add_bool("Restricted", "Spin restricted?", false);
  settings.add_bool("CylindricalSymmetry", "Give the m and -m orbitals the same radial-axial part and equal occupations, so that the density is cylindrically symmetric?", false);
  settings.add_string("SCFMethods", "SCF convergence methods to use", "DIIS + LBFGS");
  settings.add_bool("TrustRegion", "Finish the SCF with second-order trust-region optimization (OpenTrustRegion)?", false);
  settings.add_bool("StabilityAnalysis", "Check the stability of the solution within its symmetry, following instabilities if TrustRegion is used?", false);
  settings.add_double("TrustRegionHandover", "Convergence threshold at which the first-order SCF hands over to the trust-region solver (0: ConvThr)", 0.0);
  settings.add_double("HundBias", "Hund-like aufbau: lower the orbital energies of the m block by HundBias*m during the first-order SCF, so that +m fills before -m and open shells maximize M_L; the energy of an m/-m pair is unchanged (0: off)", 0.0);
  settings.add_bool("FractionalOccupations", "Let the trust-region solver optimize the occupations of the m blocks, allowing fractional occupations at a common Fermi level? Otherwise it keeps the occupations of the first-order SCF", false);
  settings.add_bool("TrustRegionExactPrecond", "Precondition the trust-region occupation directions with the exact occupation Hessian? (Off only to measure what it buys)", true);
  settings.add_double("TrustRegionCheck", "Check the trust-region gradient and Hessian against finite differences of the energy with this step at the handover (0: no check)", 0.0);

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
  bool cylindrical = settings.get_bool("CylindricalSymmetry");
  if(!settings.get_bool("LinearSymmetry"))
    throw std::runtime_error("erkale_complex_orbs blocks the orbitals by m, so it only runs with LinearSymmetry.\n");
  std::string guess = settings.get_string("Guess");
  std::string scfmethods = settings.get_string("SCFMethods");
  bool trustregion = settings.get_bool("TrustRegion");
  bool stability = settings.get_bool("StabilityAnalysis");
  double handover = settings.get_double("TrustRegionHandover");
  double trcheck = settings.get_double("TrustRegionCheck");
  bool trexact = settings.get_bool("TrustRegionExactPrecond");
  bool fractional = settings.get_bool("FractionalOccupations");
  double hundbias = settings.get_double("HundBias");
  // The orbital Zeeman term splits m from -m
  if(cylindrical && linB != 0.0)
    throw std::runtime_error("CylindricalSymmetry is incompatible with a magnetic field (LinearB).\n");
  if(fractional && readlinocc < 0)
    throw std::runtime_error("FractionalOccupations optimizes the occupations, so it cannot be combined with occupations read in with LinearOccupations.\n");
  if(hundbias != 0.0 && (readlinocc < 0 || cylindrical))
    throw std::runtime_error("HundBias steers the aufbau between m and -m, so it needs neither LinearOccupations nor CylindricalSymmetry.\n");
#ifndef ERKALE_TRUST_REGION
  if(trustregion || stability)
    throw std::runtime_error("TrustRegion and StabilityAnalysis need ERKALE built with OpenTrustRegion.\n");
#endif
  // The guess: the Fock matrices of the calculation in LoadChk projected onto
  // the basis, the core Hamiltonian, or the superposition of atomic
  // potentials
  const bool loadguess = load_guess(guess, loadchk, "erkale_complex_orbs");

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
  basis.check_linear();
  chkpt.write(basis);
  // The blocks of the Fock matrices in the checkpoint depend on these
  chkpt.write("ComplexBasis", complexbas);
  chkpt.write("CylindricalSymmetry", cylindrical);

  int maxam = basis.max_am();
  auto mvals = basis.m_values();
  const std::vector<arma::uvec> m_indices(m_blocks(basis));

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
  // Meta-GGAs of tau are current corrected, which makes them gauge
  // invariant in a magnetic field; there is no current correction for
  // the laplacian.
  if(linB != 0.0 && exc.needs_laplacian())
    throw std::runtime_error("Meta-GGA functionals of the laplacian are not supported with a magnetic field (LinearB).\n");

  // Calculate matrices
  arma::mat S(basis.overlap());
  arma::mat T(basis.kinetic());
  arma::mat V(basis.nuclear());
  arma::mat Vsap(basis.sap_potential(potlib));
  if(stricmp(guess,"Core")==0)
    Vsap.zeros();
  arma::mat fock_terms = T + V + Vsap; // Helper
  std::vector<arma::mat> Fguess((1 + unrestricted) * (2 * maxam + 1));

  const arma::cx_mat D(complex_transform(basis, complexbas));

  // Blocked matrices
  arma::mat S_c = arma::real(D.t() * S * D);
  const std::vector<arma::mat> X(block_bases(complexbas ? S_c : S, m_indices, cylindrical));
  size_t Nmo=0;
  for (const auto & Xm : X)
    Nmo += Xm.n_cols;

  // Symmetry blocks seen by the SCF solver. By default, these are the m
  // blocks. With CylindricalSymmetry, m and -m share one |m| block, whose
  // orbitals are used for both, half of each orbital's occupation going to
  // each, and whose Fock matrix is the average of theirs, which gives the
  // energy derivatives with respect to the shared orbitals and occupations
  // (the derivative with respect to an occupation is half the sum); the
  // two share their orthonormal basis.
  std::vector<size_t> symblock(X.size()), symmult;
  for (int m = -maxam; m <= maxam; m++)
    symblock[m+maxam] = cylindrical ? std::abs(m) : m+maxam;
  symmult.assign(symblock.back()+1, 0);
  for (size_t m=0; m<X.size(); m++)
    symmult[symblock[m]]++;
  const size_t nsym = symmult.size();

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

  // Exact exchange of the (complex) orbitals C with occupations occ,
  // including its admixture. The code in ERKALE has a different
  // convention for complex integrals; the complex conjugate makes it
  // compatible with this code.
  std::function<arma::cx_mat(const arma::cx_mat &, const std::vector<double> &)> exchange = [&](const arma::cx_mat & C_c, const std::vector<double> & occv) {
    arma::cx_mat K(Nbf, Nbf, arma::fill::zeros);
    if(exc.kfull() != 0.0)
      K -= exc.kfull() * arma::conj(jk.calcK(C_c, occv, S));
    if(exc.kshort() != 0.0)
      K -= exc.kshort() * arma::conj(jk.calcK_short(C_c, occv, S));
    return K;
  };

  std::function<std::tuple<arma::mat, arma::mat, arma::cx_mat>(const std::vector<arma::mat> orbitals, const std::vector<arma::vec> & occupations)> electronic_terms = [&](const auto & orbitals, const auto & occupations) {
    arma::mat C;
    arma::vec occs;
    std::tie(C,occs) = collect_orbitals(orbitals, occupations);

    arma::cx_mat C_c = D * C;
    arma::mat P = arma::real(C_c * arma::diagmat(occs) * C_c.t());
    arma::mat J = jk.calcJ(P);

    arma::cx_mat K = exchange(C_c, arma::conv_to<std::vector<double>>::from(occs));

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

  // An AO Fock matrix (plus the field terms in the basis of the blocks, if
  // given) in the orthonormal bases of the symmetry blocks
  std::function<std::vector<arma::mat>(const arma::cx_mat &, const arma::mat &)> to_blocks = [&](const arma::cx_mat & F, const arma::mat & field) {
    arma::cx_mat DFD;
    if(complexbas)
      DFD = D.t()*F*D + field;
    else
      DFD = (F + field)*std::complex<double>(1.0,0.0);
    std::vector<arma::mat> blocks(X.size());
    for (size_t m=0; m<X.size(); m++)
      blocks[m] = X[m].t() * arma::real(DFD(m_indices[m], m_indices[m])) * X[m];
    return blocks;
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

    arma::cx_mat Vxc;
    double Exc = exc.eval(cP, Vxc);

    // Form the Fock matrices
    arma::cx_mat F = T + V + J + 0.5*K + Vxc;
    fock = to_blocks(F, Bterms);

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

    arma::cx_mat Vxca, Vxcb;
    double Exc = exc.eval(cPa, cPb, Vxca, Vxcb);

    arma::mat Ba = - 0.5 * linB * S;
    arma::mat Bb = + 0.5 * linB * S;

    // Form the Fock matrices
    arma::cx_mat Fa = T + V + Ja + Jb + Ka + Vxca + Ba;
    arma::cx_mat Fb = T + V + Ja + Jb + Kb + Vxcb + Bb;
    const std::vector<arma::mat> focka = to_blocks(Fa, Bterms), fockb = to_blocks(Fb, Bterms);
    for (size_t m=0; m<X.size(); m++) {
      fock[m] = focka[m];
      fock[X.size() + m] = fockb[m];
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


  if(loadguess) {
    Checkpoint load(loadchk,false);
    BasisSet oldbas;
    load.read(oldbas);
    oldbas.check_linear();
    bool oldcomplex(complexbas), oldcylindrical(cylindrical);
    if(load.exist("ComplexBasis"))
      load.read("ComplexBasis", oldcomplex);
    if(load.exist("CylindricalSymmetry"))
      load.read("CylindricalSymmetry", oldcylindrical);
    if(oldcomplex != complexbas)
      throw std::runtime_error("LoadChk was not calculated with the same ComplexBasis setting.\n");

    // The m blocks of the old calculation
    const std::vector<arma::uvec> oldidx(m_blocks(oldbas));
    const arma::cx_mat oldD(complex_transform(oldbas, complexbas));
    const std::vector<arma::mat> oldX(block_bases(arma::real(oldD.t()*oldbas.overlap()*oldD), oldidx, oldcylindrical));
    const arma::mat Sno(arma::real(D.t()*basis.overlap(oldbas)*oldD));

    // The old Fock matrices of the m blocks of the spin s ("" restricted),
    // from the AO Fock matrix of an erkale calculation
    std::function<std::vector<arma::mat>(const std::string &)> old_fock = [&](const std::string & s) {
      std::vector<arma::mat> F(oldX.size());
      if (load.exist("F" + s + "0")) {
        for (size_t m=0; m<F.size(); m++)
          load.read("F" + s + std::to_string(m), F[m]);
      } else {
        arma::mat H;
        load.read("H" + s, H);
        const arma::mat Hc(arma::real(oldD.t()*H*oldD));
        for (size_t m=0; m<F.size(); m++)
          F[m] = oldX[m].t() * Hc(oldidx[m], oldidx[m]) * oldX[m];
      }
      return F;
    };
    // A restricted calculation has the same Fock matrix for both spins,
    // and one started from an unrestricted calculation the average
    const bool oldrestr = load.exist("F0") || load.exist("H");
    std::vector<arma::mat> Fa(old_fock(oldrestr ? "" : "a"));
    const std::vector<arma::mat> Fb(oldrestr ? Fa : old_fock("b"));
    if (!unrestricted)
      for (size_t m=0; m<Fa.size(); m++)
        Fa[m] = 0.5*(Fa[m] + Fb[m]);
    double lambda = 0.0;
    for (size_t m=0; m<oldX.size(); m++)
      if (Fa[m].n_elem)
        lambda = std::max(lambda, std::max(arma::eig_sym(Fa[m]).max(), arma::eig_sym(Fb[m]).max()));

    Fguess = project_fock(Fa, oldX, oldidx, X, m_indices, Sno, lambda);
    if (unrestricted) {
      const std::vector<arma::mat> Fgb(project_fock(Fb, oldX, oldidx, X, m_indices, Sno, lambda));
      Fguess.insert(Fguess.end(), Fgb.begin(), Fgb.end());
    }
  } else { // Else guess Fock from SAP/core
    for (size_t m=0; m<X.size(); m++) {
      Fguess[m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
      if (unrestricted)
	Fguess[X.size() + m] = X[m].t() * fock_terms(m_indices[m], m_indices[m]) * X[m];
    }
  }// if(loadguess)
    

  // The solver's density matrix for the m blocks, and the solver's Fock
  // matrices from those of the m blocks and back. A particle type has
  // X.size() m blocks and nsym solver blocks.
  const size_t nparticle = unrestricted ? 2 : 1;
  std::function<OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double>(const OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double> &)> unfold_density = [&](const auto & dm) {
    OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double> full;
    for (size_t p=0; p<nparticle; p++)
      for (size_t m=0; m<X.size(); m++) {
        const size_t b = p*nsym + symblock[m];
        full.first.push_back(dm.first[b]);
        full.second.push_back(dm.second[b] / symmult[symblock[m]]);
      }
    return full;
  };
  std::function<std::vector<arma::mat>(const std::vector<arma::mat> &)> fold_fock = [&](const auto & full) {
    std::vector<arma::mat> folded(nparticle*nsym);
    for (size_t p=0; p<nparticle; p++)
      for (size_t m=0; m<X.size(); m++) {
        arma::mat & F = folded[p*nsym + symblock[m]];
        if (F.is_empty())
          F = full[p*X.size() + m];
        else
          F += full[p*X.size() + m];
      }
    for (size_t b=0; b<folded.size(); b++)
      folded[b] /= symmult[b % nsym];
    return folded;
  };
  std::function<std::vector<arma::mat>(const std::vector<arma::mat> &)> unfold_blocks = [&](const auto & folded) {
    std::vector<arma::mat> full;
    for (size_t p=0; p<nparticle; p++)
      for (size_t m=0; m<X.size(); m++)
        full.push_back(folded[p*nsym + symblock[m]]);
    return full;
  };
  // Occupation numbers of the m blocks for the solver blocks; m and -m
  // must carry the same
  std::function<arma::vec(const arma::vec &)> fold_occupations = [&](const arma::vec & occ) {
    arma::vec folded(nsym, arma::fill::zeros);
    for (size_t m=0; m<X.size(); m++) {
      if (occ(m) != occ(X.size()-1-m))
        throw std::logic_error("CylindricalSymmetry needs equal occupations for m and -m in LinearOccupationFile.\n");
      folded(symblock[m]) += occ(m);
    }
    return folded;
  };

  // Data for SCF solver
  int nblocks = nparticle*nsym;
  arma::uvec number_of_blocks_per_particle_type;
  arma::vec maximum_occupation;
  arma::vec number_of_particles;
  arma::vec number_of_particles_per_block;
  std::vector<std::string> block_descriptions;
  OpenOrbitalOptimizer::Armadillo::FockBuilder<double, double> fock_builder;

  // Run SCF
  if (!unrestricted) {
    number_of_blocks_per_particle_type = {(arma::uword) nblocks};
    maximum_occupation = 2.0*arma::conv_to<arma::vec>::from(symmult);
    number_of_particles = {(double) (Nel)};
    if (readlinocc < 0)
      number_of_particles_per_block = cylindrical ? fold_occupations(occnuma + occnumb) : arma::vec(occnuma + occnumb);
    for (int k=0; k<nblocks; k++) {
      std::string str = cylindrical ? "|m|=" + std::to_string(k) : "m=" + std::to_string(k - maxam);
      block_descriptions.push_back(str);
    }
    fock_builder = restricted_fock_builder;
  } else {
    number_of_blocks_per_particle_type = {(arma::uword) nblocks / 2, (arma::uword) nblocks / 2};
    maximum_occupation = arma::join_cols(arma::conv_to<arma::vec>::from(symmult), arma::conv_to<arma::vec>::from(symmult));
    number_of_particles = {(double) (Nela), (double) (Nelb)};
    if (readlinocc < 0)
      number_of_particles_per_block = cylindrical ? arma::join_cols(fold_occupations(occnuma), fold_occupations(occnumb)) : arma::join_cols(occnuma, occnumb);
    for (int s=0; s<2; s++) {
      std::string spin = s ? "beta" : "alpha";
      for (int k=0; k<nblocks / 2; k++) {
	std::string str = spin + (cylindrical ? " |m|=" + std::to_string(k) : " m=" + std::to_string(k - maxam));
	block_descriptions.push_back(str);
      }
    }
    fock_builder = unrestricted_fock_builder;
  }
  if (cylindrical) {
    const auto full_builder = fock_builder;
    fock_builder = [&, full_builder](const OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double> & dm) {
      auto ret = full_builder(unfold_density(dm));
      ret.second = fold_fock(ret.second);
      return ret;
    };
    Fguess = fold_fock(Fguess);
  }

  // Hund-like aufbau: the first-order SCF sees the orbital energies of
  // block m lowered by HundBias*m. A constant shift of a block leaves its
  // orbitals and their convergence unchanged, and changes only which
  // blocks the aufbau fills.
  std::function<void(std::vector<arma::mat> &, double)> shift_m = [&](std::vector<arma::mat> & F, double s) {
    for (size_t b=0; b<F.size(); b++)
      F[b].diag() -= s*((int) (b % nsym) - maxam);
  };
  OpenOrbitalOptimizer::Armadillo::FockBuilder<double, double> scf_fock_builder = fock_builder;
  if (hundbias != 0.0) {
    scf_fock_builder = [&](const OpenOrbitalOptimizer::Armadillo::DensityMatrix<double, double> & dm) {
      auto ret = fock_builder(dm);
      shift_m(ret.second, hundbias);
      return ret;
    };
    shift_m(Fguess, hundbias);
  }

  printf("\n\n\nSolving SCF\n");
  fflush(stdout);
  OpenOrbitalOptimizer::Armadillo::SCFSolver<double,double> scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, scf_fock_builder, block_descriptions);
  if (readlinocc < 0)
    scfsolver.fixed_number_of_particles_per_block(number_of_particles_per_block);
  scfsolver.initialize_with_fock(Fguess);
  if (readlinocc < 0)
    scfsolver.frozen_occupations(true);
  scfsolver.error_norm(error_norm);
  // The trust-region solver cannot discover an occupation pattern, so it
  // takes over from a well-converged first-order solution
  scfsolver.convergence_threshold((trustregion && handover > 0.0) ? handover : convergence_threshold);
  scfsolver.verbosity(verbosity);
  scfsolver.maximum_iterations(maxiter);
  scfsolver.maximum_history_length(diisorder);
  scfsolver.run(scfmethods);
  // Like erkale, do not carry on from an SCF that did not converge; the
  // trust-region solver, however, finishes it
  if(!scfsolver.converged() && !trustregion)
    throw std::runtime_error("SCF did not converge in " + std::to_string(maxiter) + " iterations!\n");

  auto fock = scfsolver.get_fock_matrix();
  shift_m(fock, -hundbias);
  double E = scfsolver.get_energy();
  auto dm = scfsolver.get_solution();

#ifdef ERKALE_TRUST_REGION
  // Second-order trust-region optimization and stability analysis. The
  // rotations stay within the symmetry blocks.
  if(trustregion || stability) {
    const int otr_verbose = verbosity >= 10 ? 4 : (verbosity >= 5 ? 3 : 2);

    // Fock response to the transition densities D_b = L_b R_b^T + R_b L_b^T
    // in the symmetry blocks, for analytic Hessian-vector products. The
    // Coulomb and exact-exchange parts are linear in the density; the
    // exchange of D follows from that of L+R and L-R. The XC part is the
    // libxckernel second-derivative contraction.
    // AO coefficients (in the complex basis, if used) of columns given in
    // the blocks b0, ..., b0+X.size()-1
    std::function<arma::cx_mat(const std::vector<arma::mat> &, size_t)> ao_columns = [&](const std::vector<arma::mat> & B, size_t b0) {
      size_t ncol=0;
      for (size_t m=0; m<X.size(); m++)
        ncol += B[b0+m].n_cols;
      arma::mat C(Nbf, ncol, arma::fill::zeros);
      size_t ic=0;
      for (size_t m=0; m<X.size(); m++) {
        const size_t n=B[b0+m].n_cols;
        if(n)
          C.submat(m_indices[m], arma::regspace<arma::uvec>(ic, ic+n-1)) = X[m]*B[b0+m];
        ic += n;
      }
      return arma::cx_mat(D*C);
    };
    // Hermitian transition density and its exact exchange. The Coulomb
    // and most XC terms see its real part; the current of a
    // current-corrected meta-GGA sees its imaginary part.
    std::function<void(const arma::cx_mat &, const arma::cx_mat &, arma::cx_mat &, arma::cx_mat &)> transition = [&](const arma::cx_mat & Lc, const arma::cx_mat & Rc, arma::cx_mat & Dc, arma::cx_mat & dK) {
      Dc = Lc*Rc.t() + Rc*Lc.t();
      const std::vector<double> ones(Lc.n_cols, 1.0);
      dK = 0.5*(exchange(Lc+Rc, ones) - exchange(Lc-Rc, ones));
    };
    // Density of the blocks b0, ..., b0+X.size()-1
    std::function<arma::cx_mat(const TrustRegionSCF::DensityMatrix &, size_t)> block_density = [&](const TrustRegionSCF::DensityMatrix & dm, size_t b0) {
      std::vector<arma::mat> orbs(dm.first.begin()+b0, dm.first.begin()+b0+X.size());
      std::vector<arma::vec> occs(dm.second.begin()+b0, dm.second.begin()+b0+X.size());
      return complex_density(orbs, occs);
    };
    const arma::mat nofield(Nbf, Nbf, arma::fill::zeros);
    TrustRegionSCF::ResponseBuilder response;
    if(!unrestricted)
      response = [&](const TrustRegionSCF::DensityMatrix & dm, const std::vector<arma::mat> & L, const std::vector<arma::mat> & R) {
        arma::cx_mat Dc, dK;
        transition(ao_columns(L, 0), ao_columns(R, 0), Dc, dK);
        arma::cx_mat dF = jk.calcJ(arma::real(Dc)) + 0.5*dK;
        if(exc.active())
          dF += exc.response(block_density(dm, 0), Dc);
        return to_blocks(dF, nofield);
      };
    else
      response = [&](const TrustRegionSCF::DensityMatrix & dm, const std::vector<arma::mat> & L, const std::vector<arma::mat> & R) {
        arma::cx_mat Dca, Dcb, dKa, dKb;
        transition(ao_columns(L, 0), ao_columns(R, 0), Dca, dKa);
        transition(ao_columns(L, X.size()), ao_columns(R, X.size()), Dcb, dKb);
        const arma::mat dJ = jk.calcJ(arma::real(Dca + Dcb));
        arma::cx_mat dFa = dJ + dKa, dFb = dJ + dKb;
        if(exc.active()) {
          arma::cx_mat dVa, dVb;
          exc.response(block_density(dm, 0), block_density(dm, X.size()), Dca, Dcb, dVa, dVb);
          dFa += dVa;
          dFb += dVb;
        }
        std::vector<arma::mat> blocks = to_blocks(dFa, nofield);
        const std::vector<arma::mat> blocksb = to_blocks(dFb, nofield);
        blocks.insert(blocks.end(), blocksb.begin(), blocksb.end());
        return blocks;
      };
    // occ-RI-K assumes the exchanged orbitals are the occupied ones, which
    // L+R and L-R are not: use finite differences then
    if(jk.is_occ_rik())
      response = TrustRegionSCF::ResponseBuilder();
    // A transition density of an |m| block goes half to m and half to -m,
    // like its occupations
    else if(cylindrical) {
      const auto full_response = response;
      response = [&, full_response](const TrustRegionSCF::DensityMatrix & dm, const std::vector<arma::mat> & L, const std::vector<arma::mat> & R) {
        std::vector<arma::mat> Rhalf(R);
        for (size_t b=0; b<Rhalf.size(); b++)
          Rhalf[b] /= symmult[b % nsym];
        return fold_fock(full_response(unfold_density(dm), unfold_blocks(L), unfold_blocks(Rhalf)));
      };
    }

    printf("\nFirst-order SCF energy % .10f\n", E);
    fflush(stdout);
    // With FractionalOccupations, the trust-region solver optimizes the
    // distribution of the electrons of each particle type over the m
    // blocks along with the orbitals
    TrustRegionSCF::FreeOccupations free_occupations;
    free_occupations.blocks_per_particle = arma::conv_to<std::vector<size_t>>::from(number_of_blocks_per_particle_type);
    free_occupations.maxocc = arma::conv_to<std::vector<double>>::from(maximum_occupation);
    free_occupations.descriptions = block_descriptions;
    TrustRegionSCF tr = !fractional ? TrustRegionSCF(fock_builder, scfsolver.get_solution(), response) : TrustRegionSCF(fock_builder, scfsolver.get_solution(), free_occupations, response);
    tr.exact_preconditioner(trexact);
    if(trcheck > 0.0)
      tr.check_derivatives(trcheck, 1e-4);
    if(trustregion) {
      printf("\nTrust-region optimization with OpenTrustRegion\n");
      fflush(stdout);
      if(!tr.optimize(convergence_threshold, stability, otr_verbose))
        printf("Warning: the trust-region optimization stopped above the convergence threshold.\n");
      if(trcheck > 0.0)
        tr.check_derivatives(trcheck, 1e-4);
    } else {
      printf("\nStability analysis with OpenTrustRegion\n");
      fflush(stdout);
      if(!tr.is_stable(convergence_threshold, otr_verbose))
        printf("Warning: the solution is unstable within its symmetry; use TrustRegion to follow the instability.\n");
      else
        printf("The solution is stable within its symmetry.\n");
      fflush(stdout);
    }
    dm = tr.density_matrix();
    auto ret = fock_builder(dm);
    E = ret.first;
    fock = ret.second;
  }
#endif
  // The checkpoint holds the Fock matrices of the m blocks
  save_matrices(cylindrical ? unfold_blocks(fock) : fock);

  // The solution: the occupations of each block over its canonical
  // orbitals, the eigenvectors of the Fock matrix, which at convergence
  // span the occupied space
  printf("\nCanonical orbital energies (occupations) of the occupied blocks, up to the lowest empty orbital\n");
  for (size_t b=0; b<fock.size(); b++) {
    const double nb = arma::sum(dm.second[b]);
    if (nb == 0.0)
      continue;
    arma::vec eps;
    arma::mat V;
    arma::eig_sym(eps, V, fock[b]);
    const arma::vec n(arma::square(V.t()*dm.first[b]) * dm.second[b]);
    const arma::uvec occupied(arma::find(n > 1e-6));
    const size_t nprint = std::min<size_t>(occupied.max()+2, n.n_elem);
    printf("%s: %.6f electrons\n", block_descriptions[b].c_str(), nb);
    for (size_t k=0; k<nprint; k++) {
      printf(" % 12.6f (%.6f)", eps(k), n(k));
      if (k%4==3 || k+1==nprint)
        printf("\n");
    }
    fflush(stdout);
  }
  // <S^2> = Sz(Sz+1) + Nb - sum_ij na_i nb_j |<a_i|b_j>|^2, within each m
  // block; a folded |m| block carries the electrons of m and -m
  const double Sz = unrestricted ? 0.5*(Nela-Nelb) : 0.0;
  double S2 = 0.0;
  if (unrestricted) {
    S2 = Sz*(Sz+1) + Nelb;
    for (size_t b=0; b<nsym; b++) {
      const arma::mat O(dm.first[b].t()*dm.first[nsym+b]);
      S2 -= arma::as_scalar(dm.second[b].t()*arma::square(O)*dm.second[nsym+b]) / symmult[b];
    }
    printf("<S^2> = %.6f\n", S2);
    fflush(stdout);
  }

  // Term symbol of the determinant. The orbitals have definite m, so M_L
  // is exact; S is taken as |Sz|. The inversion parity and, for
  // a Sigma term, the character of the reflection through the axis are
  // <D|O D>, the product over the particle types of det(C^T S O C) over
  // the occupied orbitals; the character of a determinant that is not an
  // eigenfunction of O lies strictly between -1 and 1.
  {
    const auto mdm = cylindrical ? unfold_density(dm) : dm;
    const double perorb = unrestricted ? 1.0 : 2.0;
    bool integer = true;
    for (const auto & o : mdm.second)
      for (double n : o)
        if (std::min(std::abs(n), std::abs(n-perorb)) > 1e-6)
          integer = false;
    if (!integer) {
      printf("The occupations are fractional: the solution is an ensemble of states, with no term symbol.\n");
    } else {
      std::vector<arma::cx_mat> Cocc(nparticle);
      double ML = 0.0;
      for (size_t p=0; p<nparticle; p++) {
        std::vector<arma::cx_mat> cols;
        for (size_t k=0; k<X.size(); k++) {
          const size_t b = p*X.size() + k;
          const arma::uvec occ(arma::find(mdm.second[b] > 0.5*perorb));
          arma::mat C(Nbf, occ.n_elem, arma::fill::zeros);
          C.rows(m_indices[k]) = X[k]*mdm.first[b].cols(occ);
          cols.push_back(arma::cx_mat(C, arma::mat(C.n_rows, C.n_cols, arma::fill::zeros)));
          ML += perorb*occ.n_elem*((int) k - maxam);
        }
        Cocc[p] = cols[0];
        for (size_t k=1; k<cols.size(); k++)
          Cocc[p] = arma::join_rows(Cocc[p], cols[k]);
      }
      const arma::cx_mat Sw(D.t()*S*D);
      std::function<double(const arma::mat &)> character = [&](const arma::mat & O) {
        const arma::cx_mat Ow(arma::solve(D, arma::cx_mat(O*D)));
        std::complex<double> chi = 1.0;
        for (size_t p=0; p<nparticle; p++) {
          if (!Cocc[p].n_cols)
            continue;
          const std::complex<double> d = arma::det(arma::cx_mat(Cocc[p].t()*Sw*Ow*Cocc[p]));
          chi *= unrestricted ? d : d*d;
        }
        return chi.real();
      };
      const double tol = 1e-4;
      const int Lambda = (int) std::lround(std::abs(ML));
      std::string term = lambda_label(Lambda);
      std::string notes;
      const arma::mat Inv(inversion_matrix(basis, S, V));
      if (!Inv.is_empty()) {
        const double chi = character(Inv);
        if (std::abs(chi-1.0) < tol)
          term += "_g";
        else if (std::abs(chi+1.0) < tol)
          term += "_u";
        else
          notes += "; no definite parity (<D|i D> = " + std::to_string(chi) + ")";
      }
      if (Lambda == 0) {
        const double chi = character(reflection_matrix(mvals));
        if (std::abs(chi-1.0) < tol)
          term += "+";
        else if (std::abs(chi+1.0) < tol)
          term += "-";
        else
          notes += "; no definite reflection symmetry (<D|sigma_v D> = " + std::to_string(chi) + ")";
      }
      // <S^2> of Kohn-Sham orbitals is only a diagnostic, so note
      // contamination above a small threshold
      term = std::to_string((int) std::lround(2*std::abs(Sz)+1)) + term;
      if (S2 - Sz*(Sz+1) > 1e-2)
        notes += "; spin contaminated, <S^2> exceeds Sz(Sz+1) by " + std::to_string(S2 - Sz*(Sz+1));
      printf("Term symbol %s (M_L = %i%s)\n", term.c_str(), (int) std::lround(ML), notes.c_str());
    }
    fflush(stdout);
  }
  printf("\n");
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
