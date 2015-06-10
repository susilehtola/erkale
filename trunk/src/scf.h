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
class DFTGrid;
#include "eritable.h"
#include "eriscreen.h"
#include "density_fitting.h"
class Settings;

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
  /// Use Lobatto quadrature?
  bool lobatto;

  // Non-local part?
  bool nl;
  double vv10_b, vv10_C;
  int nlnrad, nllmax;
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
  /// Non-local energy
  double Enl;

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

  /// Coulomb operator
  arma::mat J;
  /// Exchange operator
  arma::mat K;
  /// KS-XC matrix
  arma::mat XC;

  /// Complex orbitals (for SIC)
  arma::cx_mat cC;
  /// Imaginary part of complex-CMO density matrix (for complex exchange contribution)
  arma::mat P_im;
  /// Imaginary exchange
  arma::mat K_im;
  
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

  /// Coulomb operator
  arma::mat J;
  /// Exchange operators
  arma::mat Ka, Kb;
  /// KS-XC matrix
  arma::mat XCa, XCb;

  /// Imaginary exchange
  arma::mat Ka_im, Kb_im;
  
  /// Complex orbitals (for SIC)
  arma::cx_mat cCa, cCb;
  /// Imaginary part of complex-CMO density matrix (for complex exchange contribution)
  arma::mat Pa_im, Pb_im;
  
  /// Energy information
  energy_t en;
} uscf_t;


/// Possible guess types
enum guess_t {
  /// Core guess
  COREGUESS,
  /// Atomic guess
  ATOMGUESS,
  /// Molecular guess
  MOLGUESS,
  /// Generalized Wolfsberg--Helmholz
  GWHGUESS
};

/// Perdew-Zunger SIC mode
typedef struct {
  /// Exchange?
  bool X;
  /// Correlation?
  bool C;
  /// Non-local correlation?
  bool D;
} pzmet_t;

// Parse PZ method
pzmet_t parse_pzmet(const std::string & str);

/// P-Z Hamiltonian
enum pzham {
  /// Symmetrize
  PZSYMM,
  /// United Hamiltonian
  PZUNITED
};

// Parse PZ hamiltonian
enum pzham parse_pzham(const std::string & str);

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
  double diiseps;
  /// Threshold of enabling full use of DIIS
  double diisthr;

  /// Use ADIIS?
  bool useadiis;
  /// Use Broyden accelerator?
  bool usebroyden;
  /// Use Trust-Region Roothaan-Hall?
  bool usetrrh;
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
  /// Integral screening threshold
  double intthr;
  
  /// Density fitting calculation?
  bool densityfit;
  /// Memory allocation for density fitting
  size_t fitmem;
  /// Threshold for density fitting
  double fitthr;

  /// Calculate forces?
  bool doforce;

  /// Nuclear repulsion energy
  double Enuc;

  /// Electron repulsion table
  ERItable tab;
  /// Electron repulsion table, range separation
  ERItable tab_rs;
  /// Electron repulsion screening table (for direct calculations)
  ERIscreen scr;
  /// Electron repulsion screening table, range separation
  ERIscreen scr_rs;
  /// Density fitting table
  DensityFit dfit;

  /// Decontracted basis set
  BasisSet decbas;
  /// Conversion matrix
  arma::mat decconv;

  /// List of frozen orbitals by symmetry group. index+1 is symmetry group, group 0 contains all non-frozen orbitals
  std::vector<arma::mat> freeze;

 public:
  /// Constructor
  SCF(const BasisSet & basis, const Settings & set, Checkpoint & chkpt);
  ~SCF();

  /// Calculate restricted Hartree-Fock solution
  void RHF(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv);
  /// Calculate restricted open-shell Hartree-Fock solution
  void ROHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv);
  /// Calculate unrestricted Hartree-Fock solution
  void UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv);

  /// Calculate restricted density-functional theory solution
  void RDFT(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv, const dft_t dft);
  /// Calculate unrestricted density-functional theory solution
  void UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv, const dft_t dft);

  /// Calculate restricted Hartree-Fock operator
  void Fock_RHF(rscf_t & sol, const std::vector<double> & occs) const;
  /// Calculate restricted open-shell Hartree-Fock operator
  void Fock_ROHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb) const;
  /// Calculate unrestricted Hartree-Fock operator
  void Fock_UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb) const;

  /// Calculate restricted density-functional theory KS-Fock operator
  void Fock_RDFT(rscf_t & sol, const std::vector<double> & occs, const dft_t dft, DFTGrid & grid, DFTGrid & nlgrid) const;
  /// Calculate unrestricted density-functional theory KS-Fock operator
  void Fock_UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const dft_t dft, DFTGrid & grid, DFTGrid & nlgrid) const;

  /// Helper for PZ-SIC: compute orbital-dependent Fock matrices
  void PZSIC_Fock(std::vector<arma::cx_mat> & Forb, arma::vec & Eorb, const arma::cx_mat & C, dft_t dft, DFTGrid & grid, DFTGrid & nlgrid, bool fock);

  /// Calculate force in restricted Hartree-Fock
  arma::vec force_RHF(rscf_t & sol, const std::vector<double> & occs, double tol);
  /// Calculate force in restricted open-shell Hartree-Fock
  arma::vec force_ROHF(uscf_t & sol, int Nel_alpha, int Nel_beta, double tol);
  /// Calculate force in unrestricted Hartree-Fock
  arma::vec force_UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, double tol);
  /// Calculate force in restricted density-functional theory
  arma::vec force_RDFT(rscf_t & sol, const std::vector<double> & occs, const dft_t dft, DFTGrid & grid, DFTGrid & nlgrid, double tol);
  /// Calculate force in unrestricted density-functional theory
  arma::vec force_UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const dft_t dft, DFTGrid & grid, DFTGrid & nlgrid, double tol);

  /// Set frozen orbitals in ind:th symmetry group. ind+1 is the resulting symmetry group, group 0 contains all non-frozen orbitals
  void set_frozen(const arma::mat & C, size_t ind);

  /// Set the density-fitting basis set
  void set_fitting(const BasisSet & fitbas);

  /// Set verbose setting
  void set_verbose(bool verb);
  /// Set verbose setting
  bool get_verbose() const;

  /// Toggle calculation of forces
  void do_force(bool val);

  /// Get overlap matrix
  arma::mat get_S() const;
  /// Get half-inverse overlap matrix
  arma::mat get_Sinvh() const;
  /// Get core Hamiltonian matrix
  arma::mat get_Hcore() const;
  /// Get checkpoint file
  Checkpoint *get_checkpoint() const;
  /// Using strict integrals?
  bool get_strictint() const;
  
  /// Do core guess
  void core_guess(rscf_t & sol) const;
  /// Do core guess
  void core_guess(uscf_t & sol) const;

  /// Do GWH guess
  void gwh_guess(rscf_t & sol) const;
  /// Do GWH guess
  void gwh_guess(uscf_t & sol) const;
};

/// Determine effect of imaginary part of Fock operator on eigenvectors
void imag_lost(const rscf_t & sol, const arma::mat & S, double & d);
/// Determine effect of imaginary part of Fock operator on eigenvectors
void imag_lost(const uscf_t & sol, const arma::mat & S, double & da, double & db);

/// Diagonalize Fock matrix
void diagonalize(const arma::mat & S, const arma::mat & Sinvh, rscf_t & sol, double shift=0.0);
/// Diagonalize Fock matrix
void diagonalize(const arma::mat & S, const arma::mat & Sinvh, uscf_t & sol, double shift=0.0);


/**
 * Make ROHF / CUHF update to (Hartree-)Fock operators Fa and Fb,
 * using total density matrix P and overlap matrix S.
 *
 * T. Tsuchimochi and G. E. Scuseria, "Constrained active space
 * unrestricted mean-field methods for controlling
 * spin-contamination", J. Chem. Phys. 134, 064101 (2011).
 */
void ROHF_update(arma::mat & Fa, arma::mat & Fb, const arma::mat & P, const arma::mat & S, std::vector<double> occa, std::vector<double> occb, bool verbose=true);


/// Update occupations by occupying states with maximum overlap
void determine_occ(arma::vec & nocc, const arma::mat & C, const arma::vec & nocc_old, const arma::mat & C_old, const arma::mat & S);

/// Form density matrix by occupying nocc lowest lying states
arma::mat form_density(const arma::mat & C, size_t nocc);
/// Form density matrix by occupying nocc lowest lying states
arma::mat form_density(const arma::mat & C, const arma::vec & occs);
/// Form density matrix by occupying nocc lowest lying states
void form_density(rscf_t & sol, size_t nocc);
/// Form density matrix with occupations nocc
void form_density(rscf_t & sol, const arma::vec & occa);
/// Form density matrix with occupations nocc
void form_density(uscf_t & sol, const arma::vec & occa, const arma::vec & occb);
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

/// Parse DFT grid settings
dft_t parse_dft(const Settings & set, bool init);

/// Run the calculation
void calculate(const BasisSet & basis, const Settings & set, bool doforce=false);

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

/// Needed to get solvers to compilex
#include "checkpoint.h"

#endif
