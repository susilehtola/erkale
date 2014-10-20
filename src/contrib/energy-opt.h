/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/**
 * This file contains routines for energy-optimization of basis sets.
 *
 * The algorithms are based on gradient descent methods, and use
 * Legendre polynomials to expand the exponents in a 10-base logarithm
 * basis; see G. Petersson, S. Zhong, J. A. Montgomery and
 * M. J. Frish, "On the optimization of Gaussian basis sets",
 * J. Chem. Phys. 118, 1101 (2003).
 */
#ifndef ERKALE_ENOPT
#define ERKALE_ENOPT

#include "../global.h"
#include "../basislibrary.h"
#include "../tempered.h"
#include <string>
#include <cstdlib>
#include <armadillo>
#include <vector>

class EnergyOptimizer {
 private:
  /// Shell angular momentum
  arma::ivec sham;
  /// Amount of functions on the shells
  arma::uvec nf;
  /// Amount of parameters
  arma::uvec npar;
  /// Optimize shell?
  arma::uvec optsh;

  /// Step size in finite difference gradient
  double fd_h;
  /// Step size in line search
  double ls_h;
  /// Amount of consecutive trials
  size_t ntr;

  /// Get exponents
  std::vector<arma::vec> get_exps(const arma::vec & x) const;
  /// Calculate gradient
  arma::vec calcG(const arma::vec & x);
  /// Pad vector to fit into x
  arma::vec pad_vec(const arma::vec & sd) const;

 protected:
  /// Element to optimize
  std::string el;
  
 public:
  /// Constructor
  EnergyOptimizer(const std::string & el, const arma::ivec & am, const arma::uvec & nf, const arma::uvec & npar, const arma::uvec & optsh);
  /// Destructor
  ~EnergyOptimizer();

  /// Generate basis set
  virtual BasisSetLibrary form_basis(const arma::vec & x) const;
  /// Generate basis set
  virtual BasisSetLibrary form_basis(const std::vector<arma::vec> & exps) const;
  /// Calculate energy for basis set
  virtual std::vector<double> calcE(const std::vector<BasisSetLibrary> & baslib)=0;

  /// Run optimization. Returns energy, and optimized parameters in x
  double optimize(arma::vec & x, size_t maxiter);

  /// Look for optimal polarization exponent
  double scan(arma::vec & x, double xmin, double xmax, double dx);

  /// Get parameter index vector
  arma::uvec idx_vec() const;
  /// Print parameter info
  void print_info(const arma::vec & x, const std::string & msg) const;
};

#endif
