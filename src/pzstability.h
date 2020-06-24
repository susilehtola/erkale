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

#ifndef ERKALE_PZSTAB
#define ERKALE_PZSTAB

#include "scf.h"
#include "dftgrid.h"
class Timer;

class FDHessian {
 protected:
  /// Verbose operation?
  bool verbose;
  /// Finite difference derivative step size
  double ss_fd;
  /// Line search step size
  double ss_ls;

  /// Print optimization status
  virtual void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;

 public:
  /// Constructor
  FDHessian(bool verbose=true);
  /// Destructor
  virtual ~FDHessian();

  /// Get amount of parameters
  virtual size_t count_params() const=0;
  /// Evaluate function
  virtual double eval(const arma::vec & x)=0;
  /// Update solution
  virtual void update(const arma::vec & x);

  /// Evaluate finite difference gradient
  virtual arma::vec gradient();
  /// Evaluate finite difference gradient at point x
  virtual arma::vec gradient(const arma::vec & x);
  /// Evaluate finite difference Hessian
  virtual arma::mat hessian();

  /// Run optimization
  virtual double optimize(size_t maxiter=1000, double gthr=1e-4, bool max=false);
};

/// Classify parameters
typedef struct {
  /// Name of the block
  std::string name;
  /// Degrees of freedom in block
  arma::uvec idx;
} pz_rot_par_t;

/// Orbital depedent scaling
typedef enum {
  /// Constant scaling
  PZ_SCALE_CONSTANT,
  /// Density based scaling
  PZ_SCALE_DENSITY,
  /// Kinetic energy based scaling
  PZ_SCALE_KINETIC
} pz_scaling_t;

/// PZ optimizer and stability analysis
class PZStability: public FDHessian {
 protected:
  /// SCF solver, used for energy calculations
  SCF * solverp;
  /// Basis set
  BasisSet basis;
  /// DFT grid
  DFTGrid grid;
  /// NL grid
  DFTGrid nlgrid;

  /// OV method
  dft_t ovmethod;
  /// OO method
  dft_t oomethod;
  /// Weight for PZ correction
  double pzw;
  /// or scaling method
  pz_scaling_t scale;
  /// and scaling exponent
  double scaleexp;

  /// Reference solution. Spin-restricted
  rscf_t rsol;
  /// or unrestricted
  uscf_t usol;
  /// Reference self-interaction energies
  arma::vec ref_Eorb, ref_Eorba, ref_Eorbb;
  /// Reference orbital Fock matrices
  std::vector<arma::cx_mat> ref_Forb, ref_Forba, ref_Forbb;
  /// Reference weighting factors
  arma::vec ref_worb, ref_worba, ref_worbb;

  /// Real part of transformations?
  bool real;
  /// Imaginary part of transformations?
  bool imag;
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

  /// Maximum step size
  double Tmu;

  /// Count amount of parameters for rotations
  size_t count_ov_params(size_t o, size_t v) const;
  /// Count amount of parameters for rotations
  size_t count_oo_params(size_t o) const;
  /// Count amount of parameters for rotations
  size_t count_params(size_t o, size_t v) const;
  /// Count amount of parameters
  size_t count_params() const;

  /// Classify parameters
  std::vector<pz_rot_par_t> classify() const;

  /// Calculate rotation matrix
  arma::cx_mat rotation(const arma::vec & x, bool spin=false) const;
  /// Form rotation parameter matrix
  arma::cx_mat rotation_pars(const arma::vec & x, bool spin=false) const;
  /// Calculate matrix exponential
  arma::cx_mat matexp(const arma::cx_mat & X) const;

  /// Construct unified Hamiltonian
  arma::cx_mat unified_H(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::vec & worb, const arma::cx_mat & H0) const;

  /// Evaluate analytic gradient
  arma::vec gradient();
  /// Evaluate analytic gradient at point x
  arma::vec gradient(const arma::vec & x, bool ref);
  /// Evaluate semi-analytic Hessian
  arma::mat hessian();

  /// Parallel transport
  void parallel_transport(arma::vec & gold, const arma::vec & sd, double step) const;

  /// Update step size
  void update_step(const arma::vec & g);
  /// Perform quasicanonical diagonalisation
  void diagonalize();

  /// Get orbital weights
  arma::vec compute_worb(const arma::cx_mat & C);
  /// Put in the scaling part of the OO gradient
  void scaling_gradient_oo(arma::cx_mat & gOO, const arma::cx_mat & CO, const arma::vec & Eorb);
  /// Put in the scaling part of the OV gradient
  void scaling_gradient_ov(arma::cx_mat & gOV, const arma::cx_mat & CO, const arma::vec & Eorb, const arma::cx_mat & CV);

  /// Evaluate function at x, and possibly orbital Fock matrices
  double eval(const arma::vec & x, rscf_t & sol, std::vector<arma::cx_mat> & Forb, arma::vec & Eorb, arma::vec & worb, bool ks, bool fock, bool useref);
  /// Evaluate function at x, and possibly orbital Fock matrices
  double eval(const arma::vec & x, uscf_t & sol, std::vector<arma::cx_mat> & Forba, arma::vec & Eorba, arma::vec & worba, std::vector<arma::cx_mat> & Forbb, arma::vec & Eorbb, arma::vec & worbb, bool ks, bool fock, bool useref);
  /// Evaluate function at x
  double eval(const arma::vec & x);

  /// Update solution
  void update(const arma::vec & x);
  /// Update reference
  void update_reference(bool sort);
  /// Update (adaptive) integration grid. If init=true, initialization is done for a static grid
  void update_grid(bool init);

  /// Calculate orbital centroids
  arma::mat centroids(const arma::cx_mat & CO) const;

  /// Print status of optimization
  void print_status(size_t iiter, const arma::vec & g, const Timer & t) const;

  /// Print information on solution
  void print_info(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0, const arma::vec & Eorb, const arma::vec & worb);

  /// Get the full Fock matrix
  arma::cx_mat get_H(const rscf_t & sol) const;
  /// Get the full Fock matrix
  arma::cx_mat get_H(const uscf_t & sol, bool spin) const;

  /// Precondition gradient vector with unified Hamiltonian
  arma::vec precondition_unified(const arma::vec & g) const;
  /// Precondition gradient vector with orbital Hamiltonian
  arma::vec precondition_orbital(const arma::vec & g) const;

  /// Get occupied orbitals (restricted)
  arma::cx_mat get_CO(const rscf_t & sol) const;
  arma::cx_mat get_CO() const;
  /// Get occupied orbitals (unrestricted)
  arma::cx_mat get_CO(bool spin, const uscf_t & sol) const;
  arma::cx_mat get_CO(bool spin) const;
  /// Get virtual orbitals (restricted)
  arma::cx_mat get_CV(const rscf_t & sol) const;
  arma::cx_mat get_CV() const;
    /// Get virtual orbitals (unrestricted)
  arma::cx_mat get_CV(bool spin, const uscf_t & sol) const;
  arma::cx_mat get_CV(bool spin) const;

 public:
  /// Constructor
  PZStability(SCF *solver, bool verbose=true);
  /// Destructor
  ~PZStability();

  /// Set method and weight
  void set_method(const dft_t & ovmethod, const dft_t & oomethod, double pzw, pz_scaling_t scale, double scaleexp);
  /// Set parameters. real: real rotations? imag: imaginary rotations? ov: ov rotations? oo: oo rotations?
  void set_params(bool real, bool imag, bool ov, bool oo);

  /// Set reference
  void set(const rscf_t & sol);
  /// Set reference
  void set(const uscf_t & sol);

  /// Evaluate energy
  double get_E();

  /// Get updated solution
  rscf_t get_rsol() const;
  /// Get updated solution
  uscf_t get_usol() const;

  /// Check stability of solution.
  bool check(bool stability=false, double cutoff=-1e-3, double dEthr=-1e-7);
  /// Print out a line search
  void linesearch(const std::string & fname="pz_ls.dat", int prec=1, int Np=100);

  /// Print information
  void print_info();

  /// Add in a small random perturbation to the solution
  void perturb(double h=1e-6);

  /// Run optimization
  virtual double optimize(size_t maxiter=1000, double gthr=1e-4, double nrthr=1e-4, double dEthr=1e-9, int preconditioning=1);
};

#endif
