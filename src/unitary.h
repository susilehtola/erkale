#ifndef ERKALE_UNITARY
#define ERKALE_UNITARY

#include "global.h"
#include "basis.h"
#include "scf.h"
#include <armadillo>

enum unitmethod {
  /// Polynomial search, fit derivative
  POLY_DF,
  /// Polynomial search, fit function and derivative
  POLY_FDF,
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

  /// Amount of points to use in polynomial search
  int npoly;

  /// Value of cost function
  double J;
  /// Old value
  double oldJ;
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
  void set_poly(int n);

  /// Unitary optimization
  double optimize(arma::cx_mat & W, enum unitmethod met=POLY_DF);
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


/// Perdew-Zunger self-interaction correction
class PZSIC : public Unitary {
  /// SCF object for constructing Fock matrix
  SCF * solver;
  /// Settings for DFT calculation
  dft_t dft;
  /// XC grid
  DFTGrid * grid;

  /// Solution
  rscf_t sol;
  /// Occupation number
  double occnum;
  /// Coefficient for PZ-SIC
  double pzcor;

  /// Orbital Fock matrices
  std::vector<arma::mat> Forb;
  /// Orbital SIC energies
  arma::vec Eorb;
  /// Kappa matrix
  arma::cx_mat kappa;

  /// SIC Fock operator
  arma::mat HSIC;

  /// Check convergence
  bool converged();

  /// Calculate R and K
  void get_rk(double & R, double & K) const;

  /// Print progress
  void print_progress(size_t k) const;
  /// Check convergence
  bool converged() const;

 public:
  /// Constructor
  PZSIC(SCF *solver, dft_t dft, DFTGrid * grid, bool verbose);
  /// Destructor
  ~PZSIC();

  /// Set orbitals
  void set(const rscf_t & ref, double occ, double pzcor);

  /// Evaluate cost function
  double cost_func(const arma::cx_mat & W);
  /// Evaluate derivative of cost function
  arma::cx_mat cost_der(const arma::cx_mat & W);
  /// Evaluate cost function and its derivative
  void cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der);
};



#endif
