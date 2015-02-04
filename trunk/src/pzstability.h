/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_PZSTAB
#define ERKALE_PZSTAB

#include "scf.h"
#include "timer.h"

class FDHessian {
  /// Step size
  double ss;

 protected:
  /// Print optimization status
  virtual void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;
  
 public:
  /// Constructor
  FDHessian();
  /// Destructor
  ~FDHessian();

  /// Get amount of parameters
  virtual size_t count_params() const=0;
  /// Evaluate function
  virtual double eval(const arma::vec & x)=0;
  /// Evaluate function
  virtual double eval(const arma::vec & x, int mode)=0;
  /// Update solution
  virtual void update(const arma::vec & x);

  /// Evaluate finite difference gradient
  arma::vec gradient();
  /// Evaluate finite difference Hessian
  arma::mat hessian();

  /// Run optimization
  void optimize(size_t maxiter=1000, double gthr=1e-5, bool max=false);
};

  
class PZStability: public FDHessian {
 protected:
  /// SCF solver, used for energy calculations
  SCF * solverp;
  /// Basis set
  BasisSet basis;
  /// DFT grid
  DFTGrid grid;
  /// Method
  dft_t method;

  /// Reference solution. Spin-restricted
  rscf_t rsol;
  /// or unrestricted
  uscf_t usol;

  /// Use reference to do evaluations?
  bool useref;
  /// Reference SCF energy
  double ref_E0;
  /// Reference orbital energies
  arma::vec ref_Eo;
  /// Reference orbital energies
  arma::vec ref_Eoa, ref_Eob;
  
  /// Complex transformations?
  bool cplx;
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

  /// Get oo rotation matrix
  arma::cx_mat oo_rotation(const arma::vec & x, bool spin=false) const;
  /// Get ov rotation matrix
  arma::cx_mat ov_rotation(const arma::vec & x, bool spin=false) const;

  /// Evaluate function
  double eval(const arma::vec & x);
    /// Evaluate function. mode: -1 for reference update, 0 for full
    /// evaluation, 1 for evaluation wrt a reference
  double eval(const arma::vec & x, int mode);
  /// Update solution
  void update(const arma::vec & x);

  /// Print status of optimization
  void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;

 public:
  /// Constructor
  PZStability(SCF *solver, dft_t method);
  /// Destructor
  ~PZStability();

  /// Set parameters. drop: drop orbitals from calculation. cplx: complex rotations? ov: ov rotations? oo: oo rotations?
  void set(const rscf_t & sol, const arma::uvec & drop, bool cplx, bool ov, bool oo=true);
  /// Set parameters. dropa, dropb: drop orbitals from calculation. cplx: complex rotations? ov: ov rotations? oo: oo rotations?
  void set(const uscf_t & sol, const arma::uvec & dropa, const arma::uvec & dropb, bool cplx, bool ov, bool oo=true);
  
  /// Check stability of solution.
  void check();
};

#endif
