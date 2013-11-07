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


#ifndef ERKALE_DFTGRID
#define ERKALE_DFTGRID

#include "global.h"
#include "basis.h"
#include "hirshfeld.h"

/**
 * Structure for value of basis function in a point (LDA). The
 * gradients are stored in an array which uses the same indexing. */
typedef struct {
  /// Global index of function
  size_t ind;
  /// Value of function
  double f;
} bf_f_t;

/**
 * Structure for a grid point (density and its gradient are indexed in
 * a similar manner) */
typedef struct {
  /// Coordinates of the point
  coords_t r;
  /// Integration weight (both spherical jacobian and Becke weight)
  double w;

  /// Index of first basis function on grid point
  size_t f0;
  /// Number of functions on grid point
  size_t nf;
} gridpoint_t;

/// Info for radial shell
typedef struct {
  /// Radius of shell
  double r;
  /// Radial weight
  double w;
  /// Order of quadrature rule
  int l;

  /// First point in grid array of this shell
  size_t ind0;
  /// Number of points on shell
  size_t np;
} radshell_t;

/// Info for atomic grid
typedef struct {
  /// Atomic inedx
  size_t atind;
  /// Coordinates
  coords_t cen;

  /// Number of grid points
  size_t ngrid;
  /// Number of function values
  size_t nfunc;
  /// Radial shells
  std::vector<radshell_t> sh;
} atomgrid_t;


/**
 * \class AtomGrid
 *
 * \brief Integration grid for an atom
 *
 * This file contains routines for computing the matrix elements of
 * the used exchange-correlation functional for density-functional
 * theory calculations.
 *
 * The actual values of exchange-correlation functionals are computed
 * by libxc. The Fock matrix is formed as described in
 * J. A. Pople, P. M. W. Gill and B. G. Johnson, "Kohn-Sham
 * density-functional theory within a finite basis set",
 * Chem. Phys. Lett. 199 (1992), pp. 557 - 560.
 *
 * For speed and to guarantee the accuracy of the results, the
 * integration grid is formed adaptively by default, as described in
 * A. M. Köster, R. Flores-Morano and J. U. Reveles, "Efficient and
 * reliable numerical integration of exchange-correlation energies and
 * potentials", J. Chem. Phys. 121, pp. 681 - 690 (2004). Fixed grids
 * can also be used.
 *
 * In addition to the above adaptive procedure, there is also an older
 * variant for adaptive grid generation that is mostly useful for
 * computing partial charges, i.e. M. Krack and A. M. Köster, "An
 * adaptive numerical integrator for molecular integrals",
 * J. Chem. Phys. 108, 3226 (2008).
 *
 * The radial integral is done with Gauss-Chebyshev quadrature, with
 * the change of variables used being the parameter free version
 * \f$ \displaystyle{ r_A = \frac 1 {\ln 2} \ln \left( \frac 2 {1-x_A} \right) } \f$
 * as used in the aforementioned article.
 *
 * The angular integrals are performed either by Lebedev quadrature
 * or by Lobatto quadrature, as described in
 *
 * C. W. Murray, N. C. Handy and G. J. Laming, "Quadrature schemes for
 * integrals of density functional theory", Mol. Phys. 78, pp. 997 -
 * 1014 (1993).
 *
 * O. Treutler and R. Ahlrichs, "Efficient molecular integration
 * schemes", J. Chem. Phys. 102, pp. 346 - 354 (1994).
 *
 * \author Susi Lehtola
 * \date 2011/05/11 22:35
 */

class AtomGrid {
 protected:
  /// Integration points
  std::vector<gridpoint_t> grid;

  /// Values of functions in grid points
  std::vector<bf_f_t> flist;
  /// Gradients, length 3*flist
  std::vector<double> glist;
  /// Laplacians, length flist
  std::vector<double> llist;

  /// Is gradient needed?
  bool do_grad;
  /// Is laplacian needed?
  bool do_lapl;

  /// Spin-polarized calculation?
  bool polarized;

  /// GGA functional used? (Set in compute_xc, only affects eval_Fxc)
  bool do_gga;
  /// Meta-GGA used? (Set in compute_xc, only affects eval_Fxc)
  bool do_mgga;

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

  // Meta-GGA stuff

  /// Laplacian of electron density
  std::vector<double> lapl_rho;
  /// Kinetic energy density
  std::vector<double> tau;

  /// Functional derivative of energy wrt laplacian of electron density
  std::vector<double> vlapl;
  /// Functional derivative of energy wrt kinetic energy density
  std::vector<double> vtau;

  /// Grid tolerance (for pruning grid)
  double tol;

  /// Use Lobatto quadrature? (Default is Lebedev)
  bool use_lobatto;

 public:
  /// Constructor. Need to set tolerance as well before using constructor!
  AtomGrid(bool lobatto=false, double tol=1e-4);
  /// Destructor
  ~AtomGrid();

  /// Set tolerance
  void set_tolerance(double toler);
  /// Check necessity of computing gradient and laplacians, necessary for compute_bf!
  void check_grad_lapl(int x_func, int c_func);

  /// Construct a fixed size grid
  atomgrid_t construct(const BasisSet & bas, size_t cenind, int nrad, int lmax, bool verbose);
  /// Construct adaptively a grid centered on the cenind:th center, restricted calculation
  atomgrid_t construct(const BasisSet & bas, const arma::mat & P, size_t cenind, int x_func, int c_func, bool verbose);
  /// Construct adaptively a grid centered on the cenind:th center, unrestricted calculation
  atomgrid_t construct(const BasisSet & bas, const arma::mat & Pa, const arma::mat & Pb, size_t cenind, int x_func, int c_func, bool verbose);

  /// Construct a dummy grid that is only meant for the overlap matrix (Becke charges)
  atomgrid_t construct_becke(const BasisSet & bas, size_t cenind, bool verbose);
  /// Construct a dummy grid that is only meant for the overlap matrix (Hirshfeld charges)
  atomgrid_t construct_hirshfeld(const BasisSet & bas, size_t cenind, const Hirshfeld & hirsh, bool verbose);

  /// Construct adaptively a grid centered on the cenind:th center, SIC calculation
  atomgrid_t construct(const BasisSet & bas, const std::vector<arma::mat> & Pa, size_t cenind, int x_func, int c_func, bool restr, bool verbose);

  /// Form shells on an atom, as according to list of radial shells
  void form_grid(const BasisSet & bas, atomgrid_t & g);
  /// Form shells on an atom, but using Hirshfeld weight instead of Becke weight
  void form_hirshfeld_grid(const Hirshfeld & hirsh, atomgrid_t & g);

  /// Compute values of basis functions in all grid points
  void compute_bf(const BasisSet & bas, const atomgrid_t & g);

  /**
   * Compute Becke weight for grid points on shell irad.
   *
   * The weighting scheme is from the article R. E. Stratmann,
   * G. E. Scuseria, M. J. Frisch, "Achieving linear scaling in
   * exchange-correlation density functional quadratures",
   * Chem. Phys. Lett. 257, 213 (1996).
   *
   * The default value for the constant a is 0.7, as in the Köster et
   * al. (2004) paper.
   */
  void becke_weights(const BasisSet & bas, const atomgrid_t & g, size_t ir, double a=0.7);
  /// Compute Hirshfeld weight for grid points on shell irad
  void hirshfeld_weights(const Hirshfeld & hirsh, const atomgrid_t & g, size_t ir);

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
  void update_density(const arma::mat & P);
  /// Update values of density, unrestricted calculation
  void update_density(const arma::mat & Pa, const arma::mat & Pb);

  /// Evaluate density at a point
  double eval_dens(const arma::mat & P, size_t ip) const;
  /// Evaluate gradient at a point
  void eval_grad(const arma::mat & P, size_t ip, double *g) const;
  /// Evaluate laplacian of density and kinetic density at a point
  void eval_lapl_kin_dens(const arma::mat & P, size_t ip, double & lapl, double & kin) const;

  /// Evaluate integral of density with given cutoff
  double eval_dens_cutoff(const arma::mat & P, double cutoff) const;

  /// Compute number of electrons
  double compute_Nel() const;

  /// Print atomic grid
  void print_grid() const;

  /// Initialize XC arrays
  void init_xc();
  /// Compute XC functional from density and add to total XC array
  void compute_xc(int func_id);
  /// Evaluate exchange/correlation energy
  double eval_Exc() const;

  /// Evaluate atomic contribution to overlap matrix
  void eval_overlap(arma::mat & S) const;
  /// Evaluate diagonal elements of overlap matrix
  void eval_diag_overlap(arma::vec & S) const;

  /// Evaluate Fock matrix, restricted calculation
  void eval_Fxc(arma::mat & H) const;
  /// Evaluate Fock matrix, unrestricted calculation
  void eval_Fxc(arma::mat & Ha, arma::mat & Hb) const;

  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), restricted calculation
  void eval_diag_Fxc(arma::vec & H) const;
  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), unrestricted calculation
  void eval_diag_Fxc(arma::vec & Ha, arma::vec & Hb) const;

  /// Evaluate force
  arma::vec eval_force(const BasisSet & bas, const arma::mat & P) const;
  /// Evaluate force
  arma::vec eval_force(const BasisSet & bas, const arma::mat & Pa, const arma::mat & Pb) const;
};


/**
 * \class DFTGrid
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
 * The actual work is done in the AtomGrid class.
 *
 * \author Susi Lehtola
 * \date 2011/05/11 22:35
 */

class DFTGrid {
  /// Work grids
  std::vector<AtomGrid> wrk;
  /// Atomic grid
  std::vector<atomgrid_t> grids;

  /// Basis set
  const BasisSet * basp;
  /// Verbose operation?
  bool verbose;
  /// Use Lobatto quadrature?
  bool use_lobatto;

 public:
  /// Dummy constructor
  DFTGrid();
  /// Constructor
  DFTGrid(const BasisSet * bas, bool verbose=true, bool lobatto=false);
  /// Destructor
  ~DFTGrid();

  /// Create fixed size grid
  void construct(int nrad, int lmax, int x_func, int c_func);
  /// Create grid for restricted calculation
  void construct(const arma::mat & P, double tol, int x_func, int c_func);
  /// Create grid for unrestricted calculation
  void construct(const arma::mat & Pa, const arma::mat & Pb, double tol, int x_func, int c_func);

  /// Create dummy grid for Becke charges (only overlap matrix)
  void construct_becke(double tol);
  /// Create dummy grid for Hirshfeld charges (only overlap matrix)
  void construct_hirshfeld(const Hirshfeld & hirsh, double tol);

  /// Create grid for SIC calculation
  void construct(const std::vector<arma::mat> & Pa, double tol, int x_func, int c_func, bool restr);

  /// Get amount of points
  size_t get_Npoints() const;
  /// Get amount of functions
  size_t get_Nfuncs() const;

  /// Evaluate amount of electrons
  double compute_Nel(const arma::mat & P);
  /// Evaluate amount of electrons
  double compute_Nel(const arma::mat & Pa, const arma::mat & Pb);

  /// Compute Fock matrix, exchange-correlation energy and integrated electron density, restricted case
  void eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Exc, double & Nel);
  /// Compute Fock matrix, exchange-correlation energy and integrated electron density, unrestricted case
  void eval_Fxc(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ha, arma::mat & Hb, double & Exc, double & Nel);

  /// Compute Fock matrix, exchange-correlation energy and integrated electron density, SIC calculation
  void eval_Fxc(int x_func, int c_func, const std::vector<arma::mat> & Pa, std::vector<arma::mat> & Ha, std::vector<double> & Exc, std::vector<double> & Nel);

  /// Evaluate overlap matrix numerically
  arma::mat eval_overlap();
  /// Evaluate overlap matrix numerically in the inuc:th region
  arma::mat eval_overlap(size_t inuc);
  /// Evaluate overlap matrices numerically
  std::vector<arma::mat> eval_overlaps();
  /// Evaluate overlap matrices numerically
  arma::mat eval_hirshfeld_overlap(const Hirshfeld & hirsh, size_t inuc);
  /// Evaluate overlap matrices numerically
  std::vector<arma::mat> eval_hirshfeld_overlaps(const Hirshfeld & hirsh);

  /// Evaluate density with given cutoff
  double eval_dens_cutoff(const arma::mat & P, double cutoff);

  /// Evaluate force
  arma::vec eval_force(int x_func, int c_func, const arma::mat & P);
  /// Evaluate force
  arma::vec eval_force(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb);
};


/* Partitioning functions */
inline double f_p(double mu) {
  return 1.5*mu-0.5*mu*mu*mu;
}

inline double f_q(double mu, double a) {
  if(mu<-a)
    return -1.0;
  else if(mu<a)
    return f_p(mu/a);
  else
    return 1.0;
}

inline double f_s(double mu, double a) {
  return 0.5*(1.0-f_p(f_p(f_q(mu,a))));
}

#endif
