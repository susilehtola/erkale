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

#ifndef ERKALE_ORBROT
#define ERKALE_ORBROT

#include "global.h"
#include "fdhessian.h"
#include <armadillo>
#include <vector>

/// Classification of orbital rotation parameters
typedef struct {
  /// Name of the block
  std::string name;
  /// Degrees of freedom in block
  arma::uvec idx;
} orb_rot_par_t;

class OrbitalRotation: public FDHessian {
 private:
  /// Count parameters in rotation class
  size_t count_params(size_t i, size_t j) const;

 protected:
  /// Matrix of orbital coefficients
  arma::cx_mat C;
  
  /// Orbital groups
  std::vector<arma::uvec> orbgroups;
  /// Legends for orbital groups
  std::vector<std::string> orblegend;
  
  /// Enabled rotations matrix 
  arma::umat enabled;

  /// Real rotations enabled?
  bool real;
  /// Imaginary rotations enabled?
  bool imag;

  /// Collect rotation parameters from gradient matrices
  arma::vec collect(const std::vector<arma::cx_mat> & G) const;
  /// Spread rotation parameters to gradient matrices
  std::vector<arma::cx_mat> spread(const arma::vec & p) const;
  /// Get full rotation parameter matrix
  arma::cx_mat rotation_pars(const arma::vec & p) const;

  /// Calculate gradient
  virtual std::vector<arma::cx_mat> block_gradient();
  /// Calculate gradient at point x
  virtual std::vector<arma::cx_mat> block_gradient(const arma::vec & x);
  /// Evaluate step size
  double evaluate_step(const arma::vec & g, int n) const;

  /// Get classification of orbital rotation parameters
  std::vector<orb_rot_par_t> classify() const;

 public:
  /// Constructor
  OrbitalRotation();
  /// Destructor
  virtual ~OrbitalRotation();

  /// Get amount of parameters
  size_t count_params() const;
  /// Evaluate function
  virtual double eval(const arma::vec & x)=0;
  /// Update solution
  virtual void update(const arma::vec & x);

  /// Evaluate step size, default value n=4 (override if necessary)
  virtual double evaluate_step(const arma::vec & g) const;
  
  /// Evaluate rotated orbital coefficients
  arma::cx_mat rotate(const arma::vec & d) const;

  /// Evaluate analytic gradient
  virtual arma::vec gradient();
  /// Evaluate analytic gradient at point x
  virtual arma::vec gradient(const arma::vec & x);
  /// Evaluate finite difference Hessian
  virtual arma::mat hessian();
};

#endif
