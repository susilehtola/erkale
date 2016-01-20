/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_PZSTAB
#define ERKALE_PZSTAB

#include "orbital_rotation.h"
#include "scf.h"
#include "dftgrid.h"

/// Classify parameters
typedef struct {
  /// Name of the block
  std::string name;
  /// Degrees of freedom in block
  arma::uvec idx;
} pz_rot_par_t;

/// PZ optimizer and stability analysis
class PZStability: public OrbitalRotation {
 protected:
  /// SCF solver, used for energy calculations
  SCF * solverp;
  /// Basis set
  BasisSet basis;
  /// DFT grid
  DFTGrid grid;
  /// NL grid
  DFTGrid nlgrid;

  /// OV method
  dft_t ovmethod;
  /// OO method
  dft_t oomethod;
  /// Weight for PZ correction
  double pzw;

  /// Reference solution. Spin-restricted
  rscf_t rsol;
  /// or unrestricted
  uscf_t usol;

  /// Reference self-interaction energies
  arma::vec ref_Eorba, ref_Eorbb;
  /// Reference orbital Fock matrices
  std::vector<arma::cx_mat> ref_Forba, ref_Forbb;

  /// Spin-restricted?
  int restr;

  /// Maximum step size
  double Tmu;

  /// Classify parameters
  std::vector<pz_rot_par_t> classify() const;

  /// Construct unified Hamiltonian
  arma::cx_mat unified_H(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0) const;

  /// Evaluate analytic gradient
  arma::vec gradient();
  /// Evaluate analytic gradient at point x
  arma::vec gradient(const arma::vec & x, bool ref);
  /// Evaluate semi-analytic Hessian
  arma::mat hessian();

  /// Parallel transport
  void parallel_transport(arma::vec & gold, const arma::vec & sd, double step) const;

  /// Update step size
  void update_step(const arma::vec & g);
  /// Perform quasicanonical diagonalisation
  void diagonalize();

  /// Evaluate function at x, and possibly orbital Fock matrices
  double eval(const arma::vec & x, rscf_t & sol, std::vector<arma::cx_mat> & Forb, arma::vec & Eorb, bool ks, bool fock, double pzweight, bool useref);
  /// Evaluate function at x, and possibly orbital Fock matrices
  double eval(const arma::vec & x, uscf_t & sol, std::vector<arma::cx_mat> & Forba, arma::vec & Eorba, std::vector<arma::cx_mat> & Forbb, arma::vec & Eorbb, bool ks, bool fock, double pzweight, bool useref);
  /// Evaluate function at x
  double eval(const arma::vec & x);

  /// Update solution
  void update(const arma::vec & x);
  /// Update reference
  void update_reference(bool sort);
  /// Update (adaptive) integration grid. If init=true, initialization is done for a static grid
  void update_grid(bool init);

  /// Print status of optimization
  void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;

  /// Print information on solution
  void print_info(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0, const arma::vec & Eorb);

  /// Get the full Fock matrix
  arma::cx_mat get_H(const rscf_t & sol) const;
  /// Get the full Fock matrix
  arma::cx_mat get_H(const uscf_t & sol, bool spin) const;
  
  /// Precondition gradient vector with unified Hamiltonian
  arma::vec precondition_unified(const arma::vec & g) const;
  /// Precondition gradient vector with orbital Hamiltonian
  arma::vec precondition_orbital(const arma::vec & g) const;

 public:
  /// Constructor
  PZStability(SCF *solver);
  /// Destructor
  ~PZStability();

  /// Set method and weight
  void set_method(const dft_t & ovmethod, const dft_t & oomethod, double pzw);
  /// Set parameters. real: real rotations? imag: imaginary rotations? ov: ov rotations? oo: oo rotations?
  void set_params(bool real, bool imag, bool ov, bool oo);
  
  /// Set reference
  void set(const rscf_t & sol);
  /// Set reference
  void set(const uscf_t & sol);

  /// Evaluate energy
  double get_E();
  
  /// Get updated solution
  rscf_t get_rsol() const;
  /// Get updated solution
  uscf_t get_usol() const;

  /// Check stability of solution.
  bool check(bool stability=false, double cutoff=-1e-3);
  /// Print out a line search
  void linesearch(const std::string & fname="pz_ls.dat", int prec=1, int Np=100);

  /// Print information
  void print_info();

  /// Run optimization
  virtual double optimize(size_t maxiter=1000, double gthr=1e-4, double nrthr=1e-4, double dEthr=1e-9, int preconditioning=1);
};

#endif
