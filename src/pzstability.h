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

class FDHessian {
  /// Step size
  double ss;

 public:
  /// Constructor
  FDHessian();
  /// Destructor
  ~FDHessian();

  /// Get amount of parameters
  virtual size_t count_params() const=0;
  /// Evaluate function
  virtual double eval(const arma::vec & x)=0;

  /// Evaluate finite difference gradient
  arma::vec gradient();
  /// Evaluate finite difference Hessian
  arma::mat hessian();
};

  
class PZStability: public FDHessian {
  /// SCF solver, used for energy calculations
  SCF * solverp;
  /// DFT grid
  DFTGrid * grid;
  /// Method
  dft_t method;

  /// Reference solution. Spin-restricted
  rscf_t rsol;
  /// or unrestricted
  uscf_t usol;
  
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
  /// Total amount of orbitals
  size_t N;

  /// Count amount of parameters for rotations
  size_t count_ov_params(size_t o, size_t v) const;
  /// Count amount of parameters for rotations
  size_t count_oo_params(size_t o) const;
  /// Count amount of parameters for rotations
  size_t count_params(size_t o, size_t v) const;
    /// Count amount of parameters
  size_t count_params() const;

  /// Get oo rotation matrix
  arma::cx_mat oo_rotation(const arma::vec & x, bool spin=false) const;
  /// Get ov rotation matrix
  arma::cx_mat ov_rotation(const arma::vec & x, bool spin=false) const;

  /// Evaluate function
  double eval(const arma::vec & x);

 public:
  /// Constructor
  PZStability(SCF *solver, dft_t method);
  /// Destructor
  ~PZStability();
  
  /// Check stability of solution. cplx: complex rotations? can: check stability of canonical orbitals
  void check(const rscf_t & sol, bool cplx, bool can);
    /// Check stability of solution. cplx: complex rotations? can: check stability of canonical orbitals
  void check(const uscf_t & sol, bool cplx, bool can);
};

#endif
