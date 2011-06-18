/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
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
#include "eritable.h"
#include "eriscreen.h"
#include "density_fitting.h"
#include "settings.h"

/**
 * \class SCF
 *
 * \brief Self-consistent field solver routines
 *
 * This class contains the driver routines for performing restricted
 * and unrestricted Hartree-Fock and density-functional theory
 * calculations.
 *
 * \author Jussi Lehtola
 * \date 2011/05/12 00:54
 */

class SCF {
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

  /// Basis orthogonalizing matrix
  arma::mat Sinvh;

  /// Amount of basis functions
  size_t Nbf;

  /// Total number of electrons
  int Nel;
  /// Number of spin-up electrons
  int Nel_alpha;
  /// Number of spin-down electrons
  int Nel_beta;

  /// Multiplicity
  int mult;

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

  /// Maximum number of iterations
  int maxiter;
  /// Verbose calculation?
  bool verbose;

  /// Direct calculation?
  bool direct;
#if DFT_ENABLED
  /// Density fitting calculation? (Pure DFT XC functionals)
  bool densityfit;
#endif

  /// Mix density matrices?
  bool mixdensity;
  /// Dynamically change mixing factor?
  bool dynamicmix;

  /// Convergence criterion for change of energy
  double deltaEmax;
  /// Convergence criterion for maximum change of an element of the density matrix
  double deltaPmax;
  /// Convergence criterion for the RMS change of the density matrix
  double deltaPrms;

#if DFT_ENABLED
  /// Initial tolerance for the DFT grid
  double dft_initialtol;
  /// Final tolerance for the DFT grid
  double dft_finaltol;
  /// When to switch to final DFT grid? Indicates fraction of deltaE and deltaP of above criteria.
  double dft_switch;

  /// Use Lobatto angular grid instead of Lebedev grid (DFT)
  bool dft_lobatto;
  /// Save memory by reforming DFT grid on every iteration?
  bool dft_direct;
#endif

  /// Nuclear repulsion energy
  double Enuc;

  /// Electron repulsion table
  ERItable tab;
  /// Electron repulsion screening table (for direct calculations)
  ERIscreen scr;
#if DFT_ENABLED
  /// Density fitting table
  DensityFit dfit;
#endif

 public:
  /// Constructor
  SCF(const BasisSet & basis, const Settings & set);
  ~SCF();

  /// Calculate restricted Hartree-Fock solution
  double RHF(arma::mat & C, arma::vec & E);
  /// Calculate unrestricted Hartree-Fock solution
  double UHF(arma::mat & Ca, arma::mat & Cb, arma::vec & Ea, arma::vec & Eb);

#if DFT_ENABLED
  /// Calculate restricted density-functional theory solution
  double RDFT(arma::mat & C, arma::vec & E, int x_func, int c_func);
  /// Calculate unrestricted density-functional theory solution
  double UDFT(arma::mat & Ca, arma::mat & Cb, arma::vec & Ea, arma::vec & Eb, int x_func, int c_func);
#endif

  /// Get number of electrons
  int get_Nel() const;
  /// Get number of spin up electrons
  int get_Nel_alpha() const;  
  /// Get number of spin down electrons
  int get_Nel_beta() const;
};

/// Update occupations by occupying states with maximum overlap
void determine_occ(arma::vec & nocc, const arma::mat & C, const arma::vec & nocc_old, const arma::mat & C_old, const arma::mat & S);

/// Form density matrix by occupying nocc lowest lying states
void form_density(arma::mat & R, const arma::mat & C, size_t nocc);
/// Form density matrix with occupations nocc
void form_density(arma::mat & R, const arma::mat & C, const arma::vec & nocc);

/// Dynamical update of mixing factor
void update_mixing(double & mix, double Ecur, double Eold, double Eold2);

/// Compute magnitude of dipole moment
double dip_mom(const arma::mat & P, const BasisSet & basis);
/// Compute dipole moment
arma::vec dipole_moment(const arma::mat & P, const BasisSet & basis);
/// Compute spread of electrons: \f$ r = \sqrt{ \langle \hat{\bf r}^2 \rangle - \langle \hat{\bf r} \rangle^2 \f$
double electron_spread(const arma::mat & P, const BasisSet & basis);


#endif
