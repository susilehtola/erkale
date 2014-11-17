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



#ifndef ERKALE_LOCALIZATION
#define ERKALE_LOCALIZATION

#include "basis.h"
#include "scf.h"
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
  /// Pipek-Mezey, iterative Hirshfeld charge, p=1.5
  PIPEK_ITERHIRSHH,
  /// Pipek-Mezey, iterative Hirshfeld charge, p=2
  PIPEK_ITERHIRSH2,
  /// Pipek-Mezey, iterative Hirshfeld charge, p=4
  PIPEK_ITERHIRSH4,
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
  /// iterative Hirshfeld charge
  ITERHIRSH,
  /// intrinsic atomic orbital charge
  IAO,
  /// Stockholder charge
  STOCKHOLDER,
  /// Voronoi charge
  VORONOI
};

/// Boys localization
class Boys : public UnitaryFunction {
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
  Boys(const BasisSet & basis, const arma::mat & C, int n,  bool verbose=true, bool delocalize=false);
  /// Destructor
  ~Boys();
  /// Copy
  Boys * copy() const;

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
class FMLoc : public UnitaryFunction {
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
  FMLoc(const BasisSet & basis, const arma::mat & C, int n, bool verbose=true, bool delocalize=false);
  /// Destructor
  ~FMLoc();
  /// Copy
  FMLoc * copy() const;

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
class Pipek : public UnitaryFunction {
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
  /// Constructor
  Pipek(enum chgmet chg, const BasisSet & basis, const arma::mat & C, const arma::mat & P, double p=2.0, bool verbose=true, bool delocalize=false);
  /// Destructor
  ~Pipek();
  /// Copy
  Pipek * copy() const;

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};

/// Edmiston-Ruedenberg localization
class Edmiston : public UnitaryFunction {
  /// Density fitting object
  DensityFit dfit;
  /// Orbitals
  arma::mat C;

 public:
  /// Constructor
  Edmiston(const BasisSet & basis, const arma::mat & C, bool delocalize=false);
  /// Destructor
  ~Edmiston();
  /// Copy
  Edmiston * copy() const;

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};

/// Perdew-Zunger self-interaction correction
class PZSIC : public UnitaryFunction {
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

  /// Hamiltonian method
  enum pzham ham;

  /// Convergence criterion: rms kappa
  double rmstol;
  /// Convergence criterion: max kappa
  double maxtol;

  /// Orbital Fock matrices
  std::vector<arma::mat> Forb;
  /// Orbital SIC energies
  arma::vec Eorb;
  /// Kappa matrix
  arma::cx_mat kappa;

  /// SIC Fock operator
  arma::cx_mat HSIC;

  /// Calculate kappa rms and kappa max
  void get_k_rms_max(double & Krms, double & Kmax) const;

  /// Print legend
  std::string legend() const;
  /// Print progress
  std::string status(bool lfmt) const;
  /// Print progress
  void print_time(const Timer & t) const;

  /// Check convergence
  bool converged() const;

 public:
  /// Constructor
  PZSIC(SCF *solver, dft_t dft, DFTGrid * grid, double maxtol, double rmstol, enum pzham ham);
  /// Destructor
  ~PZSIC();
  /// Copy
  PZSIC * copy() const;

  /// Set orbitals
  void set(const rscf_t & ref, double pzcor);
  /// Set transformation matrix
  void setW(const arma::cx_mat & W);

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
  arma::cx_mat get_HSIC() const;
};

/// Analyze orbitals if they really are complex
void analyze_orbitals(const BasisSet & basis, const arma::cx_mat & C);

/// Analyze orbital if it really is complex
double analyze_orbital(const arma::mat & S, const arma::cx_vec & C);

/// Orbital localization. Density matrix is only used for construction of Bader grid (if applicable)
void orbital_localization(enum locmet method, const BasisSet & basis, const arma::mat & C, const arma::mat & P, double & measure, arma::cx_mat & U, bool verbose=true, bool real=true, int maxiter=50000, double Gthr=1e-6, double Fthr=1e-7, enum unitmethod met=POLY_DF, enum unitacc acc=CGPR, bool delocalize=false, std::string logfile="", bool debug=false);

#endif
