#ifndef ERKALE_UNITARY
#define ERKALE_UNITARY

#include "global.h"
#include "basis.h"
#include <armadillo>

/// Unitary optimization worker
class Unitary {
 protected:
  /// Order of cost function in the unitary matrix W
  int q;
  /// Verbose operation?
  bool verbose;
  /// Maximization or minimization?
  int sign;

  /// Evaluate cost function
  virtual double cost_func(const arma::cx_mat & W)=0;
  /// Evaluate derivative of cost function
  virtual arma::cx_mat cost_der(const arma::cx_mat & W)=0;
  /// Evaluate cost function and its derivative
  virtual void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der)=0;

  void check_unitary(const arma::cx_mat & W) const;
  
 public:
  /// Constructor
  Unitary(int q, bool maximize=true, bool verbose=true);
  /// Destructor
  ~Unitary();

  /// Unitary optimization, polynomial method. thr gives threshold for
  /// convergence. fdf: fit both function and its derivative?
  double optimize_poly(arma::cx_mat & W, int n, double thr, bool fdf=false);
  /// Unitary optimization, armijo method
  double optimize_armijo(arma::cx_mat & W, double thr);
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
  Boys(const BasisSet & basis, const arma::mat & C, bool verbose);
  ~Boys();

  /// Evaluate cost function                       
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};

#endif
