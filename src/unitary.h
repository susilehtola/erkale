#ifndef ERKALE_UNITARY
#define ERKALE_UNITARY

#include "global.h"
#include <armadillo>

enum unitmethod {
  /// Polynomial search, fit derivative
  POLY_DF,
  /// Polynomial search, fit function and derivative
  POLY_FDF,
  /// Armijo method
  ARMIJO
};

enum unitacc {
  /// Steepest descent / steepest ascent
  SDSA,
  /// Polak-Ribière conjugate gradients
  CGPR,
  /// Fletcher-Reeves conjugate gradients
  CGFR
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

  /// Convergence threshold
  double eps;

  /// Amount of points to use in df polynomial search
  int npoly_df;
  /// Amount of points to use in fdf polynomial search
  int npoly_fdf;

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

  /// Check convergence
  virtual bool converged(const arma::cx_mat & W);
  /// Print progress
  virtual void print_progress(size_t k) const;

  /// Check that the matrix is unitary
  void check_unitary(const arma::cx_mat & W) const;
  /// Classify matrix
  void classify(const arma::cx_mat & W) const;

  /// Get rotation matrix with wanted step size
  arma::cx_mat get_rotation(double step) const;

  /// Armijo step, return step length
  double armijo_step(const arma::cx_mat & W);
  /// Polynomial step (fit only derivative), return step length
  double polynomial_step_df(const arma::cx_mat & W);
  /// Polynomial step (fit function and derivative), return step length
  double polynomial_step_fdf(const arma::cx_mat & W);

  /// Compute the bracket product
  double bracket(const arma::cx_mat & X, const arma::cx_mat & Y) const;
  /// Solve roots of v0 + v1*x + v2*x^2 + ... + v^(N-1)*x^N
  arma::vec solve_roots(const arma::vec & v) const;
  /// Solve roots of v0 + v1*x + v2*x^2 + ... + v^(N-1)*x^N
  arma::cx_vec solve_roots_cplx(const arma::vec & v) const;
  /// Get smallest positive root
  double smallest_positive(const arma::vec & v) const;

 public:
  /// Constructor
  Unitary(int q, double thr, bool maximize, bool verbose=true);
  /// Destructor
  ~Unitary();


  /// Evaluate cost function
  virtual double cost_func(const arma::cx_mat & W)=0;
  /// Evaluate derivative of cost function
  virtual arma::cx_mat cost_der(const arma::cx_mat & W)=0;
  /// Evaluate cost function and its derivative
  virtual void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der)=0;

  /// Set polynomial search options
  void set_poly(int n_df, int n_fdf);

  /// Unitary optimization
  double optimize(arma::cx_mat & W, enum unitmethod met=POLY_DF, enum unitacc acc=CGPR, size_t maxiter=10000);
};

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
  /// Print progress
  void print_progress(size_t k) const;

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
