#ifndef ERKALE_UNITARY
#define ERKALE_UNITARY

#include "global.h"
#include "basis.h"
#include <armadillo>

enum unitmethod {
  /// Polynomial search
  POLYNOMIAL,
  /// Armijo method
  ARMIJO
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

  /* Polynomial search */
  /// Amount of points to use
  int npoly;
  /// Fit function and derivative instead of just derivative?
  bool fdf;

  /// Value of cost function
  double J;
  /// G matrices
  std::vector<arma::cx_mat> G;
  /// H matrix
  arma::cx_mat H;

  /// Eigendecomposition of -iH
  arma::vec Hval;
  /// Eigendecomposition of -iH
  arma::cx_mat Hvec;
  /// Maximum step size
  double Tmu;

  /// Check convergence
  virtual bool converged() const;
  /// Evaluate cost function
  virtual double cost_func(const arma::cx_mat & W)=0;
  /// Evaluate derivative of cost function
  virtual arma::cx_mat cost_der(const arma::cx_mat & W)=0;
  /// Evaluate cost function and its derivative
  virtual void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der)=0;

  /// Check that the matrix is unitary
  void check_unitary(const arma::cx_mat & W) const;

  /// Get rotation matrix with wanted step size
  arma::cx_mat get_rotation(double step) const;

  /// Armijo step, return step length
  double armijo_step(arma::cx_mat & W);
  /// Polynomial step, return step length
  double polynomial_step(arma::cx_mat & W);

 public:
  /// Constructor
  Unitary(int q, double thr, bool maximize=true, bool verbose=true);
  /// Destructor
  ~Unitary();

  /// Set polynomial search options
  void set_poly(int n, bool fdf);

  /// Unitary optimization
  double optimize(arma::cx_mat & W, enum unitmethod met=POLYNOMIAL);
};

/// Boys localization
class Boys : public Unitary {
  /// R^2 matrix
  arma::mat rsq;
  /// r_x matrix
  arma::mat rx;
  /// r_y matrix
  arma::mat ry;
  /// r_z matrix
  arma::mat rz;

 public:
  Boys(const BasisSet & basis, const arma::mat & C, double thr, bool verbose=true);
  ~Boys();

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};

#endif
