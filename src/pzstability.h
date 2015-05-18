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

#include "scf.h"
#include "dftgrid.h"
class Timer;


class FDHessian {
 protected:
  /// Finite difference derivative step size
  double ss_fd;
  /// Line search step size
  double ss_ls;
  
  /// Print optimization status
  virtual void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;
  
 public:
  /// Constructor
  FDHessian();
  /// Destructor
  virtual ~FDHessian();

  /// Get amount of parameters
  virtual size_t count_params() const=0;
  /// Evaluate function
  virtual double eval(const arma::vec & x)=0;
  /// Update solution
  virtual void update(const arma::vec & x, bool ref=true);

  /// Evaluate finite difference gradient
  virtual arma::vec gradient();
  /// Evaluate finite difference Hessian
  virtual arma::mat hessian();

  /// Run optimization
  virtual double optimize(size_t maxiter=1000, double gthr=1e-4, bool max=false);
};

  
class PZStability: public FDHessian {
 protected:
  /// SCF solver, used for energy calculations
  SCF * solverp;
  /// Basis set
  BasisSet basis;
  /// DFT grid
  DFTGrid grid;
  /// NL grid
  DFTGrid nlgrid;
  /// Method
  dft_t method;

  /// Reference solution. Spin-restricted
  rscf_t rsol;
  /// or unrestricted
  uscf_t usol;

  /// Real part of transformations?
  bool real;
  /// Imaginary part of transformations?
  bool imag;
  /// Check stability of canonical orbitals?
  bool cancheck;
  /// Check stability of oo block
  bool oocheck;

  /// Spin-restricted?
  bool restr;
  /// Amount of occupied orbitals
  size_t oa, ob;
  /// Amount of virtual orbitals
  size_t va, vb;

  /// Maximum step size
  double Tmu;
  
  /// Count amount of parameters for rotations
  size_t count_ov_params(size_t o, size_t v) const;
  /// Count amount of parameters for rotations
  size_t count_oo_params(size_t o) const;
  /// Count amount of parameters for rotations
  size_t count_params(size_t o, size_t v) const;
  /// Count amount of parameters
  size_t count_params() const;
  
  /// Get indices of real and imaginary parameters
  void real_imag_idx(arma::uvec & idxr, arma::uvec & idxi) const;

  /// Calculate rotation matrix
  arma::cx_mat rotation(const arma::vec & x, bool spin=false) const;
  /// Form rotation parameter matrix
  arma::cx_mat rotation_pars(const arma::vec & x, bool spin=false) const;
  /// Calculate matrix exponential
  arma::cx_mat matexp(const arma::cx_mat & X) const;

  /// Evaluate analytic gradient
  arma::vec gradient();
  /// Evaluate semi-analytic Hessian
  arma::mat hessian();
  
  /// Update step size
  void update_step(const arma::vec & g);
  /// Perform quasicanonical diagonalisation
  void diagonalize();
  
  /// Evaluate function
  double eval(const arma::vec & x);
  /// Update solution
  void update(const arma::vec & x, bool ref=true);

  /// Print status of optimization
  void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;

  /// Print information on solution
  void print_info(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0, const arma::vec & Eorb);
  
 public:
  /// Constructor
  PZStability(SCF *solver, dft_t method);
  /// Destructor
  ~PZStability();

  /// Set parameters. drop: drop orbitals from calculation. cplx: complex rotations? ov: ov rotations? oo: oo rotations?
  void set(const rscf_t & sol, const arma::uvec & drop, bool real, bool imag, bool ov, bool oo);
  /// Set parameters. dropa, dropb: drop orbitals from calculation. cplx: complex rotations? ov: ov rotations? oo: oo rotations?
  void set(const uscf_t & sol, const arma::uvec & dropa, const arma::uvec & dropb, bool real, bool imag, bool ov, bool oo);

  /// Get updated solution
  rscf_t get_rsol() const;
  /// Get updated solution
  uscf_t get_usol() const;
  
  /// Check stability of solution.
  void check();
  /// Print out a line search
  void linesearch();

  /// Print information
  void print_info();

  /// Run optimization
  virtual double optimize(size_t maxiter=1000, double gthr=1e-4, bool max=false);
};

#endif
