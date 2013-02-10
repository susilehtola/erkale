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



#include "global.h"

#ifndef ERKALE_SCF
#define ERKALE_SCF

#include <armadillo>
#include <vector>
#include "dftgrid.h"
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
  /// Integration grid tolerance
  double gridtol;
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

  /// Energy information
  energy_t en;
} uscf_t;

/// Possible guess types
enum guess_t {
  COREGUESS,
  ATOMGUESS,
  MOLGUESS
};

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
  BasisSet xcfitbas;
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
  /// Verbose calculation?
  bool verbose;

  /// Direct calculation?
  bool direct;
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
  /// Use XC fitting?
  bool dft_xcfit;

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

 public:
  /// Constructor
  SCF(const BasisSet & basis, const Settings & set, Checkpoint & chkpt);
  ~SCF();

  /// Calculate restricted Hartree-Fock solution
  void RHF(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv) const;
  /// Calculate restricted open-shell Hartree-Fock solution
  void ROHF(uscf_t & sol, int Nel_alpha, int Nel_beta, const convergence_t conv) const;
  /// Calculate unrestricted Hartree-Fock solution
  void UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv) const;

  /// Calculate restricted density-functional theory solution
  void RDFT(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv, const dft_t dft) const;
  /// Calculate unrestricted density-functional theory solution
  void UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv, const dft_t dft) const;

  /// Calculate restricted Hartree-Fock solution using line search (slow!)
  void RHF_ls(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv) const;
  /// Calculate restricted open-shell Hartree-Fock solution using line search (slow!)
  void ROHF_ls(uscf_t & sol, int Nel_alpha, int Nel_beta, const convergence_t conv) const;
  /// Calculate unrestricted Hartree-Fock solution using line search (slow!)
  void UHF_ls(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv) const;

  /// Calculate restricted density-functional theory solution using line search (slow!)
  void RDFT_ls(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv, const dft_t dft) const;
  /// Calculate unrestricted density-functional theory solution using line search (slow!)
  void UDFT_ls(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv, const dft_t dft) const;

  /// Calculate restricted Hartree-Fock operator
  void Fock_RHF(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv, const rscf_t & oldsol, double tol) const;
  /// Calculate restricted open-shell Hartree-Fock operator
  void Fock_ROHF(uscf_t & sol, int Nel_alpha, int Nel_beta, const convergence_t conv, const uscf_t & oldsol, double tol) const;
  /// Calculate unrestricted Hartree-Fock operator
  void Fock_UHF(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv, const uscf_t & oldsol, double tol) const;

  /// Calculate restricted density-functional theory KS-Fock operator
  void Fock_RDFT(rscf_t & sol, const std::vector<double> & occs, const convergence_t conv, const dft_t dft, const rscf_t & oldsol, DFTGrid & grid, XCGrid & fitgrid, double tol) const;
  /// Calculate unrestricted density-functional theory KS-Fock operator
  void Fock_UDFT(uscf_t & sol, const std::vector<double> & occa, const std::vector<double> & occb, const convergence_t conv, const dft_t dft, const uscf_t & oldsol, DFTGrid & grid, XCGrid & fitgrid, double tol) const;

  /// Set frozen orbitals in ind:th symmetry group. ind+1 is the resulting symmetry group, group 0 contains all non-frozen orbitals
  void set_frozen(const arma::mat & C, size_t ind);
};

/// Diagonalize Fock matrix
void diagonalize(const arma::mat & S, const arma::mat & Sinvh, rscf_t & sol);
/// Diagonalize Fock matrix
void diagonalize(const arma::mat & S, const arma::mat & Sinvh, uscf_t & sol);


/**
 * Find natural orbitals from P.
 */
void form_NOs(const arma::mat & P, const arma::mat & S, arma::mat & AO_to_NO, arma::mat & NO_to_AO, arma::vec & occs);

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
/// Purify the density matrix (N.B. requires occupations to be 0<=n<=1 !)
arma::mat purify_density(const arma::mat & P, const arma::mat & S);
/// Purify the density matrix with natural orbitals
arma::mat purify_density_NO(const arma::mat & P, const arma::mat & S);
/// Purify the density matrix with natural orbitals (stored in C)
arma::mat purify_density_NO(const arma::mat & P, arma::mat & C, const arma::mat & S);

/// Get atomic occupancy (spherical average)
std::vector<double> atomic_occupancy(int Nel);
/// Generate orbital occupancies
std::vector<double> get_restricted_occupancy(const Settings & set, const BasisSet & basis);
/// Generate orbital occupancies
void get_unrestricted_occupancy(const Settings & set, const BasisSet & basis, std::vector<double> & occa, std::vector<double> & occb);

/// Compute magnitude of dipole moment
double dip_mom(const arma::mat & P, const BasisSet & basis);
/// Compute dipole moment
arma::vec dipole_moment(const arma::mat & P, const BasisSet & basis);
/// Compute spread of electrons: \f$ r = \sqrt{ \left\langle \hat{\bf r}^2 \right\rangle - \left\langle \hat{\bf r} \right\rangle^2 } \f$
double electron_spread(const arma::mat & P, const BasisSet & basis);

/// Determine amount of alpha and beta electrons based on multiplicity
void get_Nel_alpha_beta(int Nel, int mult, int & Nel_alpha, int & Nel_beta);

/// Run the calculation
void calculate(const BasisSet & basis, Settings & set);

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


#include "checkpoint.h"


#endif
