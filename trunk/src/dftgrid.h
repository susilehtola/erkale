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

/// Helper for determining density cutoffs for plots
typedef struct {
  /// Density at point
  double d;
  /// Integration weight
  double w;
} dens_list_t;

/// Helper for sort
bool operator<(const dens_list_t & lhs, const dens_list_t & rhs);

/// Helper for debugging output: density input
typedef struct {
  /// Alpha and beta density
  double rhoa, rhob;
  /// Sigma variables
  double sigmaaa, sigmaab, sigmabb;
  /// Laplacians
  double lapla, laplb;
  /// Kinetic energy density
  double taua, taub;
} libxc_dens_t;

/// Helper for debugging output: potential output
typedef struct {
  /// Alpha and beta potential
  double vrhoa, vrhob;
  /// Sigma potential
  double vsigmaaa, vsigmaab, vsigmabb;
  /// Laplacian potential
  double vlapla, vlaplb;
  /// Kinetic energy potential
  double vtaua, vtaub;
} libxc_pot_t;

typedef struct {
  libxc_dens_t dens;
  libxc_pot_t pot;
} libxc_debug_t;

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

  // VV10 stuff
  /// \omega_0
  arma::vec omega0;
  /// \kappa
  arma::vec kappa;
  /// Density threshold
  double VV10_thr;
  /// VV10 helper array
  arma::mat VV10_arr;

  /// Grid tolerance (for pruning grid)
  double tol;

  /// Use Lobatto quadrature? (Default is Lebedev)
  bool use_lobatto;

  /// Extract weights
  arma::rowvec get_weights() const;
  /// Extract function values: N x Ngrid
  arma::mat get_fval(size_t N) const;
  /// Extract gradient values
  arma::cube get_gval(size_t N) const;
  /// Extract laplacian values: N x Ngrid
  arma::mat get_lval(size_t N) const;

  /// Extract LDA potential
  arma::rowvec get_vxc(bool spin) const;
  /// Extract density gradient
  arma::mat get_grho(bool spin) const;
  /// Extract GGA potential
  arma::rowvec get_vsigma(int c) const;
  /// Extract MGGA potential
  arma::rowvec get_vtau(bool spin) const;
  /// Extract MGGA potential
  arma::rowvec get_vlapl(bool spin) const;

  /// Get density data for wanted point
  libxc_dens_t get_dens(size_t idx) const;
  /// Get potential data for wanted point
  libxc_pot_t get_pot(size_t idx) const;
  /// Get density and potential data for wanted point
  libxc_debug_t get_data(size_t idx) const;

 public:
  /// Constructor. Need to set tolerance as well before using constructor!
  AtomGrid(bool lobatto=false, double tol=1e-4);
  /// Destructor
  ~AtomGrid();

  /// Set tolerance
  void set_tolerance(double toler);
  /// Check necessity of computing gradient and laplacians, necessary for compute_bf!
  void check_grad_lapl(int x_func, int c_func);
  /// Get necessity of computing gradient and laplacians
  void get_grad_lapl(bool & grad, bool & lapl) const;
  /// Set necessity of computing gradient and laplacians, necessary for compute_bf!
  void set_grad_lapl(bool grad, bool lapl);

  /// Construct a fixed size grid
  atomgrid_t construct(const BasisSet & bas, size_t cenind, int nrad, int lmax, bool verbose);
  /// Construct adaptively a grid centered on the cenind:th center, restricted calculation
  atomgrid_t construct(const BasisSet & bas, const arma::mat & P, size_t cenind, int x_func, int c_func, bool verbose);
  /// Construct adaptively a grid centered on the cenind:th center, unrestricted calculation
  atomgrid_t construct(const BasisSet & bas, const arma::mat & Pa, const arma::mat & Pb, size_t cenind, int x_func, int c_func, bool verbose);
  /// Construct adaptively a grid centered on the cenind:th center, SIC calculation
  atomgrid_t construct(const BasisSet & bas, const arma::cx_vec & C, size_t cenind, int x_func, int c_func, bool verbose);

  /// Construct a dummy grid that is only meant for the overlap matrix (Becke charges)
  atomgrid_t construct_becke(const BasisSet & bas, size_t cenind, bool verbose);
  /// Construct a dummy grid that is only meant for the overlap matrix (Hirshfeld charges)
  atomgrid_t construct_hirshfeld(const BasisSet & bas, size_t cenind, const Hirshfeld & hirsh, bool verbose);

  /// Form shells on an atom, as according to list of radial shells
  void form_grid(const BasisSet & bas, atomgrid_t & g);
  /// Form shells on an atom, but using Hirshfeld weight instead of Becke weight
  void form_hirshfeld_grid(const Hirshfeld & hirsh, atomgrid_t & g);

  /// Compute values of basis functions in all grid points
  void compute_bf(const BasisSet & bas, atomgrid_t & g);

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
  /// Update values of density using BLAS routines, restricted calculation. This is slower than the normal routine...
  void update_density_blas(const arma::mat & P);
  /// Update values of density, unrestricted calculation
  void update_density(const arma::mat & Pa, const arma::mat & Pb);
  /// Update values of density, self-interaction correction
  void update_density(const arma::cx_vec & C);

  /// Evaluate density at a point
  double eval_dens(const arma::mat & P, size_t ip) const;
  /// Evaluate density at a point (for PZ-SIC)
  double eval_dens(const arma::cx_vec & C, size_t ip) const;
  /// Evaluate gradient at a point
  void eval_grad(const arma::mat & P, size_t ip, double *g) const;
  /// Evaluate gradient at a point (for PZ-SIC)
  void eval_grad(const arma::cx_vec & C, size_t ip, double *g) const;
  /// Evaluate laplacian of density and kinetic density at a point
  void eval_lapl_kin_dens(const arma::mat & P, size_t ip, double & lapl, double & kin) const;
  /// Evaluate laplacian of density and kinetic density at a point (for PZ-SIC)
  void eval_lapl_kin_dens(const arma::cx_vec & C, size_t ip, double & lapl, double & kin) const;

  /// Evaluate density
  void eval_dens(const arma::mat & P, std::vector<dens_list_t> & d) const;

  /// Compute number of electrons
  double compute_Nel() const;

  /// Print atomic grid
  void print_grid() const;

  /// Print density information
  void print_density(FILE *f) const;
  /// Print potential information
  void print_potential(int func_id, FILE *f) const;

  /// Initialize XC arrays
  void init_xc();
  /// Compute XC functional from density and add to total XC
  /// array. Pot toggles evaluation of potential
  void compute_xc(int func_id, bool pot);
  /// Evaluate exchange/correlation energy
  double eval_Exc() const;

  /// Initialize VV10 calculation
  void init_VV10(double b, double C, bool pot);
  /// Collect VV10 data
  void collect_VV10(arma::mat & data, std::vector<size_t> & idx, bool nl) const;

  /**
   * Evaluates VV10 energy and potential and add to total array
   *
   * Implementation is described in O. Vydrov and T. Van Voorhis,
   * "Nonlocal van der Waals density functional: The simpler the
   * better", J. Chem. Phys. 133, 244103 (2010).
   */
  void compute_VV10(const std::vector<arma::mat> & nldata, double C);
  /// Same thing, but also evaluate the grid contribution to the force
  arma::vec compute_VV10_F(const std::vector<arma::mat> & nldata, double C, size_t iat);

  /// Evaluate atomic contribution to overlap matrix
  void eval_overlap(arma::mat & S) const;
  /// Evaluate diagonal elements of overlap matrix
  void eval_diag_overlap(arma::vec & S) const;

  /// Evaluate Fock matrix, restricted calculation
  void eval_Fxc(arma::mat & H) const;
  /// Evaluate Fock matrix using BLAS routines, restricted calculation
  void eval_Fxc_blas(arma::mat & H) const;
  /// Evaluate Fock matrix, unrestricted calculation
  void eval_Fxc(arma::mat & Ha, arma::mat & Hb) const;
  /// Evaluate Fock matrix using BLAS routines, unrestricted calculation
  void eval_Fxc_blas(arma::mat & Ha, arma::mat & Hb, bool beta=true) const;
  /// Evaluate Fock matrix, SIC calculation
  void eval_Fxc(const arma::cx_mat & C, size_t nocc, arma::cx_mat & H) const;

  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), restricted calculation
  void eval_diag_Fxc(arma::vec & H) const;
  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), unrestricted calculation
  void eval_diag_Fxc(arma::vec & Ha, arma::vec & Hb) const;
  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), unrestricted calculation
  void eval_diag_Fxc_SIC(arma::vec & H) const;

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
  /// Create fixed size grid
  void construct(int nrad, int lmax, bool gga, bool mgga, bool nl);
  /// Create grid for restricted calculation
  void construct(const arma::mat & P, double tol, int x_func, int c_func);
  /// Create grid for unrestricted calculation
  void construct(const arma::mat & Pa, const arma::mat & Pb, double tol, int x_func, int c_func);

  /// Create dummy grid for Becke charges (only overlap matrix)
  void construct_becke(double tol);
  /// Create dummy grid for Hirshfeld charges (only overlap matrix)
  void construct_hirshfeld(const Hirshfeld & hirsh, double tol);

  /// Create grid for SIC calculation
  void construct(const arma::cx_mat & C, double tol, int x_func, int c_func);

  /// Get amount of points
  size_t get_Npoints() const;
  /// Get amount of functions
  size_t get_Nfuncs() const;

  /// Evaluate amount of electrons
  double compute_Nel(const arma::mat & P);
  /// Evaluate amount of electrons
  double compute_Nel(const arma::mat & Pa, const arma::mat & Pb);

  /// Evaluate amount of electrons in each atomic region
  arma::vec compute_atomic_Nel(const arma::mat & P);
  /// Evaluate amount of electrons in each atomic region
  arma::vec compute_atomic_Nel(const Hirshfeld & hirsh, const arma::mat & P);

  /// Compute Fock matrix, exchange-correlation energy and integrated electron density, restricted case
  void eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Exc, double & Nel);
  /// Compute Fock matrix, exchange-correlation energy and integrated electron density, unrestricted case
  void eval_Fxc(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ha, arma::mat & Hb, double & Exc, double & Nel);

  /**
   * Compute Fock matrix, exchange-correlation energy and integrated
   * electron density, SIC calculation. Note that all orbitals are
   * necessary here.
   */
  void eval_Fxc(int x_func, int c_func, const arma::cx_mat & C, const arma::cx_mat & W, std::vector<arma::mat> & H, std::vector<double> & Exc, std::vector<double> & Nel, bool fock);

  /// Compute VV10
  void eval_VV10(DFTGrid & nlgrid, double b, double C, const arma::mat & P, arma::mat & H, double & Exc);

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

  /// Evaluate density
  std::vector<dens_list_t> eval_dens_list(const arma::mat & P);

  /// Evaluate force
  arma::vec eval_force(int x_func, int c_func, const arma::mat & P);
  /// Evaluate force
  arma::vec eval_force(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb);
  /// Evaluate NL force
  arma::vec eval_VV10_force(DFTGrid & nlgrid, double b, double C, const arma::mat & P);

  /// Print out density data
  void print_density(const arma::mat & P, std::string densname="density.dat");
  /// Print out potential data
  void print_potential(int func_id, const arma::mat & Pa, const arma::mat & Pb, std::string potname="potential.dat");
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
