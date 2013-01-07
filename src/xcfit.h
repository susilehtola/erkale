/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2011
 * Copyright (c) 2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_XCFIT
#define ERKALE_XCFIT

#include "global.h"
#include "basis.h"
#include "dftgrid.h"
#include "density_fitting.h"

/**
 * \class XCAtomGrid
 *
 * \brief Integration grid for an atom
 *
 * This file contains routines for computing the matrix elements of
 * the used exchange-correlation functional for density-functional
 * theory calculations using exchange-correlation fitting.
 *
 * \author Susi Lehtola
 * \date 2013/01/06 12:30
 */

class XCAtomGrid {
 protected:
  /// Integration points
  std::vector<gridpoint_t> grid;

  /// Values of functions in grid points
  std::vector<bf_f_t> flist;
  /// Gradients, length 3*flist
  std::vector<double> glist;

  /// Is gradient needed?
  bool do_grad;

  /// Spin-polarized calculation?
  bool polarized;

  /// GGA functional used? (Set in compute_xc, only affects eval_Fxc)
  bool do_gga;

  // LDA stuff:

  /// Density
  std::vector<double> rho;
  /// Energy density
  std::vector<double> exc;
  /// Functional derivative of energy wrt electron density
  std::vector<double> vxc;

  // GGA stuff

  /// Gradient of electron density
  std::vector<double> grho;
  /// Dot products of gradient of electron density
  std::vector<double> sigma;
  /// Functional derivative of energy wrt gradient of electron density
  std::vector<double> vsigma;

  /// Grid tolerance (for pruning grid)
  double tol;

  /// Use Lobatto quadrature? (Default is Lebedev)
  bool use_lobatto;

 public:
  /// Constructor. Need to set tolerance as well before using constructor!
  XCAtomGrid(bool lobatto=false, double tol=1e-4);
  /// Destructor
  ~XCAtomGrid();
  
  /// Set tolerance
  void set_tolerance(double toler);
  /// Check necessity of computing gradient and laplacians, necessary
  /// for compute_bf!
  void check_grad(int x_func, int c_func);

  /// Construct adaptively a grid centered on the cenind:th center,
  /// restricted calculation
  atomgrid_t construct(const BasisSet & bas, const arma::vec & gamma, size_t cenind, int x_func, int c_func, bool verbose, const DensityFit & dfit);
  /// Construct adaptively a grid centered on the cenind:th center, unrestricted calculation
  atomgrid_t construct(const BasisSet & bas, const arma::vec & gammaa, const arma::vec & gammab, size_t cenind, int x_func, int c_func, bool verbose, const DensityFit & dfit);

  /// Form shells on an atom, as according to list of radial shells
  void form_grid(const BasisSet & bas, atomgrid_t & g);
  /// Compute values of basis functions in all grid points
  void compute_bf(const BasisSet & bas, const atomgrid_t & g);

  /// Compute Becke weight for grid points on shell irad
  void becke_weights(const BasisSet & bas, const atomgrid_t & g, size_t ir);
  /// Prune points with small weight
  void prune_points(double tol, const radshell_t & rg);

  /// Add radial shell in Lobatto angular scheme, w/o Becke partitioning or pruning
  void add_lobatto_shell(atomgrid_t & g, size_t ir);
  /// Add radial shell in Lebedev scheme, w/o Becke partitioning or pruning
  void add_lebedev_shell(atomgrid_t & g, size_t ir);

  /// Compute basis functions on grid points on shell irad
  void compute_bf(const BasisSet & bas, const atomgrid_t & g, size_t irad);

  /// Free memory
  void free();

  /// Update values of density, restricted calculation
  void update_density(const arma::vec & gamma);
  /// Update values of density, unrestricted calculation
  void update_density(const arma::vec & gammaa, const arma::vec & gammab);

  /// Evaluate density at a point
  double eval_dens(const arma::vec & gamma, size_t ip) const;
  /// Evaluate gradient at a point
  void eval_grad(const arma::vec & gamma, size_t ip, double *g) const;

  /// Compute number of electrons
  double compute_Nel() const;

  /// Compute memory requirements for grid points
  size_t memory_req_grid() const;
  /// Compute memory requirements for storing values of basis functions at grid points
  size_t memory_req_bf() const;
  /// Compute total memory requirements
  size_t memory_req() const;

  /// Initialize XC arrays
  void init_xc();
  /// Compute XC functional from density and add to total XC array
  void compute_xc(int func_id);
  /// Evaluate exchange/correlation energy
  double eval_Exc() const;

  /// Evaluate Fock vector, restricted calculation
  void eval_Fxc(arma::vec & H) const;
  /// Evaluate Fock vector, unrestricted calculation
  void eval_Fxc(arma::vec & Ha, arma::vec & Hb) const;
};


/**
 * \class XCGrid
 *
 * \brief DFT quadrature grid
 *
 * This class contains routines for computing the matrix elements of
 * the used exchange-correlation functional for density-functional
 * theory calculations.
 *
 * The space integral is decomposed into atom-centered volume
 * integrals, as was proposed in
 *
 * A. D. Becke, "A multicenter numerical integration scheme for
 * polyatomic molecules", J. Chem. Phys. 88, p. 2547 - 2553 (1988).
 *
 * The actual work is done in the XCAtomGrid class.
 *
 * \author Susi Lehtola
 * \date 2013/01/06 12:30
 */

class XCGrid {
  /// Work grids
  std::vector<XCAtomGrid> wrk;
  /// Atomic grid
  std::vector<atomgrid_t> grids;

  /// Fitting basis set
  const BasisSet * fitbasp;
  /// Density fitting routine
  const DensityFit * dfitp;
  /// Verbose operation?
  bool verbose;
  /// Use Lobatto quadrature?
  bool use_lobatto;

  /// Compute expansion of density
  arma::vec expand(const arma::mat & P) const;
  /// Compute inversion of expansion
  arma::mat invert(const arma::vec & gamma) const;

 public:
  /// Constructor
  XCGrid(const BasisSet * fitbas, const DensityFit * dfit, bool verbose=true, bool lobatto=false);
  /// Destructor
  ~XCGrid();

  /// Create grid for restricted calculation
  void construct(const arma::mat & P, double tol, int x_func, int c_func);
  /// Create grid for unrestricted calculation
  void construct(const arma::mat & Pa, const arma::mat & Pb, double tol, int x_func, int c_func);

  /// Get amount of points
  size_t get_Npoints() const;
  /// Get amount of functions
  size_t get_Nfuncs() const;

  /// Get memory requirement for grid points
  size_t memory_req_grid() const;
  /// Get memory requirement for storing values of basis functions
  size_t memory_req_bf() const;
  /// Get total memory requirements
  size_t memory_req() const;
  /// Print memory requirements
  void print_memory_req() const;

  /// Compute Fock matrix, exchange-correlation energy and integrated electron density, restricted case
  void eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Exc, double & Nel);
  /// Compute Fock matrix, exchange-correlation energy and integrated electron density, unrestricted case
  void eval_Fxc(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ha, arma::mat & Hb, double & Exc, double & Nel);
};

#endif
