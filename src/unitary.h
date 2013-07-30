/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_UNITARY
#define ERKALE_UNITARY

#include "global.h"
#include "timer.h"
#include <armadillo>


/**
 * This file contains algorithms for unitary optimization of matrices.
 * The algorithms are based on the following references
 *
 * T. E. Abrudan, J. Eriksson, and V. Koivunen, "Steepest Descent
 * Algorithms for Optimization Under Unitary Matrix Constraint",
 * IEEE Transactions on Signal Processing 56 (2008), 1134.
 *
 * T. Abrudan, J. Eriksson, and V. Koivunen, "Conjugate gradient
 * algorithm for optimization under unitary matrix constraint",
 * Signal Processing 89 (2009), 1704.
 */

enum unitmethod {
  /// Polynomial search, fit function
  POLY_F,
  /// Polynomial search, fit derivative <--- the default
  POLY_DF,
  /// Polynomial search, fit function and derivative
  POLY_FDF,
  /// Fourier transform
  FOURIER_DF,
  /// Armijo method
  ARMIJO
};

enum unitacc {
  /// Steepest descent / steepest ascent
  SDSA,
  /// Polak-Ribière conjugate gradients
  CGPR,
  /// Fletcher-Reeves conjugate gradients
  CGFR,
  /// Hestenes-Stiefel conjugate gradients
  CGHS
};

/// Unitary optimization worker
class Unitary {
 protected:
  /// Order of cost function in the unitary matrix W
  int q;
  /// Verbose operation?
  bool verbose;
  /// Maximization or minimization?
  int sign;
  /// Operate with real or complex matrices?
  bool real;

  /// Convergence threshold wrt norm of Riemannian derivative
  double Gthr;
  /// Convergence threshold wrt relative change in function
  double Fthr;

  /// Degree of polynomial used for fit: a_0 + a_1*mu + ... + a_(d-1)*mu^(d-1)
  int polynomial_degree;

  /// Amount of quasi-periods for Fourier method (N_T = 1, 2, ...)
  int fourier_periods;
  /// Amount of samples per one period (K = 3, 4, or 5)
  int fourier_samples;

  /// Debugging mode - print out line search every iteration
  bool debug;

  /// Value of cost function
  double J;
  /// Old value
  double oldJ;
  /// G matrix
  arma::cx_mat G;
  /// H matrix
  arma::cx_mat H;

  /// Eigendecomposition of -iH
  arma::vec Hval;
  /// Eigendecomposition of -iH
  arma::cx_mat Hvec;
  /// Maximum step size
  double Tmu;

  /// Log file
  FILE *log;

  /// Initialize possible convergence criteria
  virtual void initialize(const arma::cx_mat & W0);
  /// Check convergence
  virtual bool converged(const arma::cx_mat & W);

  /// Print legend
  virtual void print_legend() const;
  /// Print progress
  virtual void print_progress(size_t k) const;
  /// Print time
  virtual void print_time(const Timer & t) const;
  /// Print chosen step length
  virtual void print_step(enum unitmethod & met, double step) const;

  /// Check that the matrix is unitary
  void check_unitary(const arma::cx_mat & W) const;
  /// Check that the programmed cost function and its derivative are OK
  void check_derivative(const arma::cx_mat & W0);
  /// Classify matrix
  void classify(const arma::cx_mat & W) const;

  /// Update cost function value and spherical gradient vector
  void update_gradient(const arma::cx_mat & W);
  /// Compute new search direction (diagonalize H)
  void update_search_direction();

  /// Get rotation matrix with wanted step size
  arma::cx_mat get_rotation(double step) const;
  /// Get derivative wrt step length
  double step_der(const arma::cx_mat & W, const arma::cx_mat & der) const;

  /// Set degree
  void set_q(int q);

  /// Armijo step, return step length
  double armijo_step(const arma::cx_mat & W);
  /// Polynomial step (fit function), return step length
  double polynomial_step_f(const arma::cx_mat & W);
  /// Polynomial step (fit only derivative), return step length
  double polynomial_step_df(const arma::cx_mat & W);
  /// Polynomial step (fit function and derivative), return step length
  double polynomial_step_fdf(const arma::cx_mat & W);
  /// Fourier step
  double fourier_step_df(const arma::cx_mat & W);

  /// Optimizer routine
  double optimizer(arma::cx_mat & W, enum unitmethod met, enum unitacc acc, size_t maxiter);

 public:
  /// Constructor
  Unitary(int q, double Gthr, double Fthr, bool maximize, bool verbose=true, bool real=false);
  /// Destructor
  ~Unitary();

  /// Evaluate cost function
  virtual double cost_func(const arma::cx_mat & W)=0;
  /// Evaluate derivative of cost function
  virtual arma::cx_mat cost_der(const arma::cx_mat & W)=0;
  /// Evaluate cost function and its derivative
  virtual void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der)=0;

  /// Open log file
  void open_log(const std::string & fname);
  /// Set debug mode
  void set_debug(bool dbg);

  /// Set polynomial search options
  void set_poly(int deg);
  /// Set Fourier search options
  void set_fourier(int Nsamples, int Nperiods);
  /// Set convergence threshold
  void set_thr(double Gtol, double Ftol);

  /// Unitary optimization
  double optimize(arma::cx_mat & W, enum unitmethod met=POLY_DF, enum unitacc acc=CGPR, size_t maxiter=50000);
  /// Orthogonal optimization
  double optimize(arma::mat & W, enum unitmethod met=POLY_DF, enum unitacc acc=CGPR, size_t maxiter=50000);
};


/// Compute the bracket product
double bracket(const arma::cx_mat & X, const arma::cx_mat & Y);

/// Shift Fourier coefficients
arma::cx_vec fourier_shift(const arma::cx_vec & v);

/// Fit polynomial of wanted degree to function
arma::vec fit_polynomial(const arma::vec & x, const arma::vec & y, int deg=-1);
/// Fit polynomial of wanted degree to function and derivative, return coefficients of function
arma::vec fit_polynomial_fdf(const arma::vec & x, const arma::vec & y, const arma::vec & dy, int deg=-1);

/// Convert coefficients of y(x) = c0 + c1*x + ... + c^N x^N to those of y'(x)
arma::vec derivative_coefficients(const arma::vec & c);

/// Solve roots of v0 + v1*x + v2*x^2 + ... + v^(N-1)*x^N
arma::cx_vec solve_roots_cplx(const arma::cx_vec & v);
/// Solve roots of v0 + v1*x + v2*x^2 + ... + v^(N-1)*x^N
arma::cx_vec solve_roots_cplx(const arma::vec & v);
/// Solve roots of v0 + v1*x + v2*x^2 + ... + v^(N-1)*x^N
arma::vec solve_roots(const arma::vec & v);
/// Get smallest positive root
double smallest_positive(const arma::vec & v);


/// Brockett
class Brockett : public Unitary {
  /// Sigma matrix
  arma::cx_mat sigma;
  /// N matrix
  arma::mat Nmat;

  /// Unitarity and diagonality criteria
  double unit, diag;

  /// Log file
  FILE *log;

  /// Print legend
  void print_legend() const;
  /// Print progress
  void print_progress(size_t k) const;
  /// Don't print step length
  void print_step(enum unitmethod & met, double step) const;

  /// Check convergence
  bool converged(const arma::cx_mat & W);
  /// Compute diagonality criterion
  double diagonality(const arma::cx_mat & W) const;
  /// Compute unitarity criterion
  double unitarity(const arma::cx_mat & W) const;

 public:
  Brockett(size_t N, unsigned long int seed=0);
  ~Brockett();

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};


#endif
