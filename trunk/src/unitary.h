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


/// Unitary function optimizer, used to hold values during the optimization
class UnitaryFunction {
 protected:
  /// Present matrix
  arma::cx_mat W;
  /// Present value
  double f;
  /// Order in W
  int q;
  /// Maximization or minimization?
  int sign;
  
 public:
  /// Constructor
  UnitaryFunction(int q, bool max);
  /// Destructor
  virtual ~UnitaryFunction();

  /// Set matrix
  virtual void setW(const arma::cx_mat & W);
  /// Get matrix
  arma::cx_mat getW() const;

  /// Get q
  int getq() const;
  /// Get function value
  double getf() const;
  /// Get sign
  int getsign() const;

  /// Copy constructor
  virtual UnitaryFunction *copy() const=0;
  /// Evaluate cost function
  virtual double cost_func(const arma::cx_mat & W)=0;
  /// Evaluate derivative of cost function
  virtual arma::cx_mat cost_der(const arma::cx_mat & W)=0;
  /// Evaluate cost function and its derivative
  virtual void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der)=0;

  /// Get status legend
  virtual std::string legend() const;
  /// Print status information, possibly in a longer format
  virtual std::string status(bool lfmt=false) const;
  /// Check convergence
  virtual bool converged();
};

/// Unitary optimization worker
class UnitaryOptimizer {
 private:
  /// Gradient
  arma::cx_mat G;
  /// Search direction
  arma::cx_mat H;
  /// Eigenvectors of search direction
  arma::cx_mat Hvec;
  /// Eigenvalues of search direction
  arma::vec Hval;
  /// Maximum step size
  double Tmu;

 protected:
  /// Verbose operation?
  bool verbose;
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

  /// Log file
  FILE *log;

  /// Print legend
  virtual void print_legend(const UnitaryFunction *f) const;
  /// Print progress
  virtual void print_progress(size_t k, const UnitaryFunction *f, const UnitaryFunction *fold) const;
  /// Print time
  virtual void print_time(const Timer & t) const;
  /// Print chosen step length
  virtual void print_step(enum unitmethod & met, double step) const;
  
  /// Check that the matrix is unitary
  void check_unitary(const arma::cx_mat & W) const;
  /// Check that the programmed cost function and its derivative are OK
  void check_derivative(const UnitaryFunction *f);
  /// Classify matrix
  void classify(const arma::cx_mat & W) const;

  /// Get new gradient direction
  void update_gradient(const arma::cx_mat & W, UnitaryFunction *f);
  /// Compute new search direction (diagonalize H) and max step length
  void update_search_direction(int q);

  /// Get rotation matrix with wanted step size
  arma::cx_mat get_rotation(double step) const;
  /// Get derivative wrt step length
  double step_der(const arma::cx_mat & W, const arma::cx_mat & der) const;

  /// Set degree
  void set_q(int q);

  /// Armijo step
  void armijo_step(UnitaryFunction* & f);
  /// Polynomial step (fit function)
  void polynomial_step_f(UnitaryFunction* & f);
  /// Polynomial step (fit only derivative)
  void polynomial_step_df(UnitaryFunction* & f);
  /// Fourier step
  void fourier_step_df(UnitaryFunction* & f);

 public:
  /// Constructor
  UnitaryOptimizer(double Gthr, double Fthr, bool verbose=true, bool real=false);
  /// Destructor
  ~UnitaryOptimizer();

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
  double optimize(UnitaryFunction* & f, enum unitmethod met, enum unitacc acc, size_t maxiter);
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
/// Get smallest positive element
double smallest_positive(const arma::vec & v);


/// Brockett
class Brockett : public UnitaryFunction {
  /// Sigma matrix
  arma::cx_mat sigma;
  /// N matrix
  arma::mat Nmat;

  /// Print legend
  std::string legend() const;
  /// Print progress
  std::string status(bool lfmt=false) const;

  /// Compute diagonality criterion
  double diagonality() const;
  /// Compute unitarity criterion
  double unitarity() const;

 public:
  Brockett(size_t N, unsigned long int seed=0);
  ~Brockett();

  /// Copy constructor
  Brockett *copy() const;
  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};


#endif
