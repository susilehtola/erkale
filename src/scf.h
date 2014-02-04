/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#ifndef ERKALE_SCF
#define ERKALE_SCF

#include "global.h"

#include <armadillo>
#include <vector>
#include "dftgrid.h"
#include "badergrid.h"
#include "xcfit.h"
#include "eritable.h"
#include "eriscreen.h"
#include "density_fitting.h"
#include "settings.h"

class Checkpoint;

/**
 * \class SCF
 *
 * \brief Self-consistent field solver routines
 *
 * This class contains the driver routines for performing restricted
 * and unrestricted Hartree-Fock and density-functional theory
 * calculations.
 *
 * \author Susi Lehtola
 * \date 2011/05/12 00:54
 */

/// Convergence criteria
typedef struct {
  /// Convergence criterion for change of energy
  double deltaEmax;
  /// Convergence criterion for maximum change of an element of the density matrix
  double deltaPmax;
  /// Convergence criterion for the RMS change of the density matrix
  double deltaPrms;
} convergence_t;

/// DFT settings
typedef struct {
  /// Used exchange functional
  int x_func;
  /// Used correlation functional
  int c_func;

  /// Adaptive grid?
  bool adaptive;
  /// Integration grid tolerance
  double gridtol;
  /// Amount of radial shells (if not adaptive)
  int nrad;
  /// Maximum angular quantum number to integrate exactly (if not adaptive)
  int lmax;
} dft_t;

/// Energy info
typedef struct {
  /// Coulombic energy
  double Ecoul;
  /// Kinetic energy
  double Ekin;
  /// Nuclear attraction energy
  double Enuca;
  /// Exchange(-correlation) energy
  double Exc;
  /// One-electron contribution
  double Eone;

  /// Total electronic energy
  double Eel;
  /// Nuclear repulsion energy
  double Enucr;
  /// Effective energy
  double Eeff;

  /// Total energy
  double E;
} energy_t;


/// Restricted solver info
typedef struct {
  /// Orbitals
  arma::mat C;
  /// Orbital energies
  arma::vec E;
  /// Fock operator
  arma::mat H;
  /// Density matrix
  arma::mat P;

  // Coulomb operator
  arma::mat J;
  // Exchange operator
  arma::mat K;
  // KS-XC matrix
  arma::mat XC;

  // Effective potential
  arma::mat Heff;

  /// Energy information
  energy_t en;
} rscf_t;

/// Unrestricted solver info
typedef struct {
  /// Orbitals
  arma::mat Ca, Cb;
  /// Orbital energies
  arma::vec Ea, Eb;
  /// Fock operators
  arma::mat Ha, Hb;
  /// Density matrices
  arma::mat P, Pa, Pb;

  // Coulomb operator
  arma::mat J;
  // Exchange operators
  arma::mat Ka, Kb;
  // KS-XC matrix
  arma::mat XCa, XCb;

  // Effective potentials
  arma::mat Heffa, Heffb;
  
  /// Energy information
  energy_t en;
} uscf_t;


/// Possible guess types
enum guess_t {
  COREGUESS,
  ATOMGUESS,
  MOLGUESS
};

/// Perdew-Zunger SIC run mode
enum pzrun {
  /// No correction.
  NO,
  /// Full correction.
  FULL,
  /// Perturbative correction, after SCF convergence
  PERT,
  /// Full correction with real orbitals
  REAL,
  /// Perturbative correction with real orbitals
  REALPERT,
  /// Full correction using canonical orbitals (no optimization of SIC energy)
  CAN,
  /// Perturbative correction using canonical orbitals
  CANPERT
};

// Parse run mode
enum pzrun parse_pzrun(const std::string & str);

/// Perdew-Zunger SIC mode
enum pzmet {
  /// Coulomb
  COUL,
  /// Coulomb + exchange
  COULX,
  /// Coulomb + correlation
  COULC,
  /// Coulomb + exchange + correlation
  COULXC
};

// Parse PZ method
enum pzmet parse_pzmet(const std::string & str);

class SCF {
 protected:
 /// Overlap matrix
  arma::mat S;

  /// Kinetic energy matrix
  arma::mat T;
  /// Nuclear attraction matrix
  arma::mat Vnuc;
  /// Core Hamiltonian
  arma::mat Hcore;

  /// Basis set to use (needed for DFT grid operation)
  const BasisSet * basisp;
  /// Density fitting basis
  BasisSet dfitbas;
  /// Checkpoint file
  Checkpoint * chkptp;

  /// Basis orthogonalizing matrix
  arma::mat Sinvh;

  /// Amount of basis functions
  size_t Nbf;

  /// Total number of electrons
  int Nel;

  /// Multiplicity
  int mult;

  /// Which guess to use
  enum guess_t guess;

  /// Use DIIS?
  bool usediis;
  /// Use C1-DIIS instead of C2-DIIS?
  bool diis_c1;
  /// Number of DIIS matrices to use
  int diisorder;
  /// Threshold of enabling use of DIIS
  double diisthr;

  /// Use ADIIS?
  bool useadiis;
  /// Use Broyden accelerator?
  bool usebroyden;
  /// Use Trust-Region Roothaan-Hall?
  bool usetrrh;
  /// Use trust-region DSM?
  bool usetrdsm;
  /// Do line search in level shift?
  bool linesearch;

  /// Maximum number of iterations
  int maxiter;
  /// Level shift
  double shift;
  /// Verbose calculation?
  bool verbose;

  /// Direct calculation?
  bool direct;
  /// Use decontracted basis to construct Fock matrix? (Direct formation)
  bool decfock;
  /// Strict integrals?
  bool strictint;
  /// Density fitting calculation?
  bool densityfit;
  /// Memory allocation for density fitting
  size_t fitmem;
  /// Threshold for density fitting
  double fitthr;

  /// Use Lobatto angular grid instead of Lebedev grid (DFT)
  bool dft_lobatto;

  /// Calculate forces?
  bool doforce;

  /// Nuclear repulsion energy
  double Enuc;

  /// Electron repulsion table
  ERItable tab;
  /// Electron repulsion screening table (for direct calculations)
  ERIscreen scr;
  /// Density fitting table
  DensityFit dfit;

  /// Decontracted basis set
  BasisSet decbas;
  /// Conversion matrix
  arma::mat decconv;

  /// List of frozen orbitals by symmetry group. index+1 is symmetry group, group 0 contains all non-frozen orbitals
  std::vector<arma::mat> freeze;

  /// Helper routine for PZ-SIC
  void PZSIC_calculate(rscf_t & sol, arma::cx_mat & W, dft_t dft, double pzcor, DFTGrid & grid, double Etol, bool canonical, bool real);

 public:
  /// Constructor
  SCF(const BasisSet & basis, const Settings & set, Checkpoint & chkpt);
  ~SCF();

  /// Calculate restricted Hartree-Fock solution
  void RHF(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv);
  /// Calculate restricted open-shell Hartree-Fock solution
  void ROHF(uscf_t & sol, int Nel_alpha, int Nel_beta, const convergence_t conv);
  /// Calculate unrestricted Hartree-Fock solution
  void UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv);

  /// Calculate restricted density-functional theory solution
  void RDFT(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv, const dft_t dft);
  /// Calculate unrestricted density-functional theory solution
  void UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv, const dft_t dft);

  /// Calculate restricted Hartree-Fock solution using line search (slow!)
  void RHF_ls(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv);
  /// Calculate restricted open-shell Hartree-Fock solution using line search (slow!)
  void ROHF_ls(uscf_t & sol, int Nel_alpha, int Nel_beta, const convergence_t conv);
  /// Calculate unrestricted Hartree-Fock solution using line search (slow!)
  void UHF_ls(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv);

  /// Calculate restricted density-functional theory solution using line search (slow!)
  void RDFT_ls(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv, const dft_t dft);
  /// Calculate unrestricted density-functional theory solution using line search (slow!)
  void UDFT_ls(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv, const dft_t dft);

  /// Calculate restricted Hartree-Fock operator
  void Fock_RHF(rscf_t & sol, const std::vector<double> & occs, const rscf_t & oldsol, double tol) const;
  /// Calculate restricted open-shell Hartree-Fock operator
  void Fock_ROHF(uscf_t & sol, int Nel_alpha, int Nel_beta, const uscf_t & oldsol, double tol) const;
  /// Calculate unrestricted Hartree-Fock operator
  void Fock_UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const uscf_t & oldsol, double tol) const;

  /// Calculate restricted density-functional theory KS-Fock operator
  void Fock_RDFT(rscf_t & sol, const std::vector<double> & occs, const dft_t dft, const rscf_t & oldsol, DFTGrid & grid, double tol) const;
  /// Calculate unrestricted density-functional theory KS-Fock operator
  void Fock_UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const dft_t dft, const uscf_t & oldsol, DFTGrid & grid, double tol) const;

  /// Helper for PZ-SIC: compute orbital-dependent Fock matrices
  void PZSIC_Fock(std::vector<arma::mat> & Forb, arma::vec & Eorb, const arma::cx_mat & Ctilde, dft_t dft, DFTGrid & grid);

  /// Calculate force in restricted Hartree-Fock
  arma::vec force_RHF(rscf_t & sol, const std::vector<double> & occs, double tol);
  /// Calculate force in restricted open-shell Hartree-Fock
  arma::vec force_ROHF(uscf_t & sol, int Nel_alpha, int Nel_beta, double tol);
  /// Calculate force in unrestricted Hartree-Fock
  arma::vec force_UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, double tol);
  /// Calculate force in restricted density-functional theory
  arma::vec force_RDFT(rscf_t & sol, const std::vector<double> & occs, const dft_t dft, DFTGrid & grid, double tol);
  /// Calculate force in unrestricted density-functional theory
  arma::vec force_UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const dft_t dft, DFTGrid & grid, double tol);

  /// Perform Perdew-Zunger self-interaction correction
  void PZSIC_RDFT(rscf_t & sol, const std::vector<double> & occs, dft_t dft, enum pzmet pzmet, double pzcor, const DFTGrid & grid, bool reconstruct, double Etol, bool canonical=false, bool localize=true, bool real=false);
  /// Perform Perdew-Zunger self-interaction correction
  void PZSIC_UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, dft_t dft, enum pzmet pzmet, double pzcor, const DFTGrid & grid, bool reconstruct, double Etol, bool canonical=false, bool localize=true, bool real=false);

  /// Set frozen orbitals in ind:th symmetry group. ind+1 is the resulting symmetry group, group 0 contains all non-frozen orbitals
  void set_frozen(const arma::mat & C, size_t ind);

  /// Set the density-fitting basis set
  void set_fitting(const BasisSet & fitbas);

  /// Toggle calculation of forces
  void do_force(bool val);

  /// Get overlap matrix
  arma::mat get_S() const;
  /// Get half-inverse overlap matrix
  arma::mat get_Sinvh() const;
};

/// Diagonalize Fock matrix
void diagonalize(const arma::mat & S, const arma::mat & Sinvh, rscf_t & sol, double shift=0.0);
/// Diagonalize Fock matrix
void diagonalize(const arma::mat & S, const arma::mat & Sinvh, uscf_t & sol, double shift=0.0);

/**
 * Find natural orbitals from P. Orbitals returned in decreasing occupation number.
 */
void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::mat & NO_to_AO, arma::vec & occs);

/**
 * Find natural orbitals from P. Orbitals returned in decreasing occupation number.
 */
void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::vec & occs);


/**
 * Make ROHF / CUHF update to (Hartree-)Fock operators Fa and Fb,
 * using total density matrix P and overlap matrix S.
 *
 * T. Tsuchimochi and G. E. Scuseria, "Constrained active space
 * unrestricted mean-field methods for controlling
 * spin-contamination", J. Chem. Phys. 134, 064101 (2011).
 */
void ROHF_update(arma::mat & Fa, arma::mat & Fb, const arma::mat & P, const arma::mat & S, int Nel_alpha, int Nel_beta, bool verbose=true, bool atomic=false);


/// Update occupations by occupying states with maximum overlap
void determine_occ(arma::vec & nocc, const arma::mat & C, const arma::vec & nocc_old, const arma::mat & C_old, const arma::mat & S);

/// Form density matrix by occupying nocc lowest lying states
arma::mat form_density(const arma::mat & C, size_t nocc);
/// Form density matrix with occupations nocc
arma::mat form_density(const arma::mat & C, const std::vector<double> & nocc);
/// Form energy weighted density matrix
arma::mat form_density(const arma::vec & E, const arma::mat & C, const std::vector<double> & nocc);
/// Purify the density matrix (N.B. requires occupations to be 0<=n<=1 !)
arma::mat purify_density(const arma::mat & P, const arma::mat & S);
/// Purify the density matrix with natural orbitals
arma::mat purify_density_NO(const arma::mat & P, const arma::mat & S);
/// Purify the density matrix with natural orbitals (stored in C)
arma::mat purify_density_NO(const arma::mat & P, arma::mat & C, const arma::mat & S);

/// Get atomic occupancy (spherical average)
std::vector<double> atomic_occupancy(int Nel);
/// Generate orbital occupancies
std::vector<double> get_restricted_occupancy(const Settings & set, const BasisSet & basis, bool atomic=false);
/// Generate orbital occupancies
void get_unrestricted_occupancy(const Settings & set, const BasisSet & basis, std::vector<double> & occa, std::vector<double> & occb, bool atomic=false);

/// Compute magnitude of dipole moment
double dip_mom(const arma::mat & P, const BasisSet & basis);
/// Compute dipole moment
arma::vec dipole_moment(const arma::mat & P, const BasisSet & basis);
/// Compute spread of electrons: \f$ r = \sqrt{ \left\langle \hat{\bf r}^2 \right\rangle - \left\langle \hat{\bf r} \right\rangle^2 } \f$
double electron_spread(const arma::mat & P, const BasisSet & basis);

/// Determine amount of alpha and beta electrons based on multiplicity
void get_Nel_alpha_beta(int Nel, int mult, int & Nel_alpha, int & Nel_beta);

/// Run the calculation
void calculate(const BasisSet & basis, Settings & set, bool doforce=false);

/// Helper for sorting orbitals into maximum overlap
typedef struct {
  /// Overlap
  double S;
  /// Index
  size_t idx;
} ovl_sort_t;

/// Order into decreasing overlap
bool operator<(const ovl_sort_t & lhs, const ovl_sort_t & rhs);

/**
 * Project orbitals from a minimal basis to an augmented
 * basis. Existing functions stay the same (just padded with zeros),
 * extra functions are determined from eigenvectors of the overlap
 * matrix.
 */
arma::mat project_orbitals(const arma::mat & Cold, const BasisSet & minbas, const BasisSet & augbas);

/// Get symmetry groups of orbitals
std::vector<int> symgroups(const arma::mat & C, const arma::mat & S, const std::vector<arma::mat> & freeze, bool verbose=false);

/// Freeze orbitals
void freeze_orbs(const std::vector<arma::mat> & freeze, const arma::mat & C, const arma::mat & S, arma::mat & H, bool verbose=false);

/// Localize core orbitals, returns number of localized orbitals.
size_t localize_core(const BasisSet & basis, int nocc, arma::mat & C, bool verbose=false);

/// Convert force vector to matrix
arma::mat interpret_force(const arma::vec & f);

#include "unitary.h"

/// Localization methods
enum locmet {
  /// Boys
  BOYS,
  /// Boys, penalty 2
  BOYS_2,
  /// Boys, penalty 3
  BOYS_3,
  /// Boys, penalty 4
  BOYS_4,
  /// Fourth moment
  FM_1,
  /// Fourth moment, penalty 2
  FM_2,
  /// Fourth moment, penalty 3
  FM_3,
  /// Fourth moment, penalty 4
  FM_4,
  /// Pipek-Mezey, Mulliken charge, p=1.5
  PIPEK_MULLIKENH,
  /// Pipek-Mezey, Mulliken charge, p=2
  PIPEK_MULLIKEN2,
  /// Pipek-Mezey, Mulliken charge, p=4
  PIPEK_MULLIKEN4,
  /// Pipek-Mezey, Löwdin charge, p=1.5
  PIPEK_LOWDINH,
  /// Pipek-Mezey, Löwdin charge, p=2
  PIPEK_LOWDIN2,
  /// Pipek-Mezey, Löwdin charge, p=4
  PIPEK_LOWDIN4,
  /// Pipek-Mezey, Bader charge, p=1.5
  PIPEK_BADERH,
  /// Pipek-Mezey, Bader charge, p=2
  PIPEK_BADER2,
  /// Pipek-Mezey, Bader charge, p=4
  PIPEK_BADER4,
  /// Pipek-Mezey, Becke charge, p=1.5
  PIPEK_BECKEH,
  /// Pipek-Mezey, Becke charge, p=2
  PIPEK_BECKE2,
  /// Pipek-Mezey, Becke charge, p=4
  PIPEK_BECKE4,
  /// Pipek-Mezey, Hirshfeld charge, p=1.5
  PIPEK_HIRSHFELDH,
  /// Pipek-Mezey, Hirshfeld charge, p=2
  PIPEK_HIRSHFELD2,
  /// Pipek-Mezey, Hirshfeld charge, p=4
  PIPEK_HIRSHFELD4,
  /// Pipek-Mezey, intrinsic atomic orbital charge, p=1.5
  PIPEK_IAOH,
  /// Pipek-Mezey, intrinsic atomic orbital charge, p=2
  PIPEK_IAO2,
  /// Pipek-Mezey, intrinsic atomic orbital charge, p=4
  PIPEK_IAO4,
  /// Pipek-Mezey, Stockholder charge, p=1.5
  PIPEK_STOCKHOLDERH,
  /// Pipek-Mezey, Stockholder charge, p=2
  PIPEK_STOCKHOLDER2,
  /// Pipek-Mezey, Stockholder charge, p=4
  PIPEK_STOCKHOLDER4,
  /// Pipek-Mezey, Voronoi charge, p=1.5
  PIPEK_VORONOIH,
  /// Pipek-Mezey, Voronoi charge, p=2
  PIPEK_VORONOI2,
  /// Pipek-Mezey, Voronoi charge, p=4
  PIPEK_VORONOI4,
  /// Edmiston-Ruedenberg
  EDMISTON
};

/// Charge method methods
enum chgmet {
  /// Mulliken charge
  MULLIKEN,
  /// Löwdin charge
  LOWDIN,
  /// Bader charge
  BADER,
  /// Becke charge
  BECKE,
  /// Hirshfeld charge
  HIRSHFELD,
  /// intrinsic atomic orbital charge
  IAO,
  /// Stockholder charge
  STOCKHOLDER,
  /// Voronoi charge
  VORONOI
};

/// Boys localization
class Boys : public Unitary {
  /// Penalty
  int n;

  /// R^2 matrix
  arma::mat rsq;
  /// r_x matrix
  arma::mat rx;
  /// r_y matrix
  arma::mat ry;
  /// r_z matrix
  arma::mat rz;

 public:
  /// Constructor. n gives the penalty power to use
  Boys(const BasisSet & basis, const arma::mat & C, int n, double Gthr=1e-5, double Fthr=1e-6,  bool verbose=true, bool delocalize=false);
  /// Destructor
  ~Boys();

  /// Reset penalty
  void set_n(int n);

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};

/// Fourth moment localization
class FMLoc : public Unitary {
  /// Penalty
  int n;

  /// r^4 contributions
  arma::mat rfour;
  /// rr^2 matrices
  std::vector<arma::mat> rrsq;
  /// rr matrices
  std::vector< std::vector<arma::mat> > rr;
  /// and the r^2 matrix
  arma::mat rsq;
  /// r matrices
  std::vector<arma::mat> rmat;

 public:
  /// Constructor. n gives the penalty power to use
  FMLoc(const BasisSet & basis, const arma::mat & C, int n, double Gthr=1e-5, double Fthr=1e-6, bool verbose=true, bool delocalize=false);
  /// Destructor
  ~FMLoc();

  /// Reset penalty
  void set_n(int n);

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};


/// Pipek-Mezey localization
class Pipek : public Unitary {
  // The localization is based on partial charches on atoms, so the
  // memory requirement would be Nat * Nocc^2, limiting the routine to
  // small systems. Instead, we calculate everything on-the-fly, so
  // the memory requirement is only Nocc^2

  /// Method
  enum chgmet chg;
  /// Amount of charges
  size_t N;

  /// Penalty exponent, p=2 for conventional Pipek-Mezey
  double p;

  /// Orbitals
  arma::mat C;
  /// Overlap matrix for Mulliken
  arma::mat S;
  /// Half-overlap matrix for Löwdin
  arma::mat Sh;
  /// Shell list for Löwdin and Mulliken
  std::vector< std::vector<GaussianShell> > shells;

  /// Integration grid for Becke, Hirshfeld or Stockholder localization
  DFTGrid grid;
  /// Hirshfeld / Stockholder density
  Hirshfeld hirsh;
  /// Bader localization grid
  BaderGrid bader;

  /// Free-atom AOs for IAO localization
  arma::mat C_iao;
  /// Indices of centers for IAO localization
  std::vector< std::vector<size_t> > idx_iao;

  /// Get the charge matrix for the i:th region
  arma::mat get_charge(size_t i);

 public:
  Pipek(enum chgmet chg, const BasisSet & basis, const arma::mat & C, const arma::mat & P, double p=2.0, double Gthr=1e-5, double Fthr=1e-6, bool verbose=true, bool delocalize=false);
  ~Pipek();

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};

/// Edmiston-Ruedenberg localization
class Edmiston : public Unitary {
  /// Density fitting object
  DensityFit dfit;
  /// Orbitals
  arma::mat C;

 public:
  Edmiston(const BasisSet & basis, const arma::mat & C, double Gthr=1e-5, double Fthr=1e-6, bool verbose=true, bool delocalize=false);
  ~Edmiston();

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};


/// Perdew-Zunger self-interaction correction
class PZSIC : public Unitary {
  /// SCF object for constructing Fock matrix
  SCF * solver;
  /// Settings for DFT calculation
  dft_t dft;
  /// XC grid
  DFTGrid * grid;

  /// Solution
  rscf_t sol;
  /// Occupation number
  double occnum;
  /// Coefficient for PZ-SIC
  double pzcor;

  /// Convergence criterion: rms kappa
  double rmstol;
  /// Convergence criterion: max kappa
  double maxtol;
  /// Convergence criterion: change of energy
  double Etol;

  /// Orbital Fock matrices
  std::vector<arma::mat> Forb;
  /// Orbital SIC energies
  arma::vec Eorb;
  /// Kappa matrix
  arma::cx_mat kappa;

  /// SIC Fock operator
  arma::mat HSIC;

  /// Calculate R and K
  void get_rk(double & R, double & Krms, double & Kmax) const;

  /// Print legend
  void print_legend() const;
  /// Print progress
  void print_progress(size_t k) const;
  /// Print progress
  void print_time(const Timer & t) const;

  /// Initialize convergence criterion
  void initialize(const arma::cx_mat & W0);
  /// Check convergence
  bool converged(const arma::cx_mat & W);

 public:
  /// Constructor
  PZSIC(SCF *solver, dft_t dft, DFTGrid * grid, double Etol, double maxtol, double rmstol, bool verbose=true);
  /// Destructor
  ~PZSIC();

  /// Set orbitals
  void set(const rscf_t & ref, double pzcor);

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);

  /// Get SIC energy
  double get_ESIC() const;
  /// Get orbital-by-orbital SIC
  arma::vec get_Eorb() const;
  /// Get SIC Hamiltonian
  arma::mat get_HSIC() const;
};

/// Orbital localization. Density matrix is only used for construction of Bader grid (if applicable)
void orbital_localization(enum locmet method, const BasisSet & basis, const arma::mat & C, const arma::mat & P, double & measure, arma::cx_mat & U, bool verbose=true, bool real=true, int maxiter=50000, double Gthr=1e-6, double Fthr=1e-7, enum unitmethod met=POLY_DF, enum unitacc acc=CGPR, bool delocalize=false, std::string logfile="", bool debug=false);

#include "checkpoint.h"


#endif
