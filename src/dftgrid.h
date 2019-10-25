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
#include "sap.h"
class Hirshfeld;

/// Screen out points with Becke weights smaller than 1e-8 * tol
/// (Köster et al 2004). Used in DFTGrid as well as Bader and Casida
#define PRUNETHR 1e-8

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
  /// Atomic index
  size_t atind;
  /// Coordinates of center
  coords_t cen;

  /// Radius of shell
  double R;
  /// Radial weight
  double w;
  /// Order of quadrature rule
  int l;
  /// Tolerance threshold
  double tol;

  /// Number of points on shell
  size_t np;
  /// Number of functions on shell
  size_t nfunc;
} angshell_t;

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

/// Helper for getting data
typedef struct {
  /// Density
  libxc_dens_t dens;
  /// Potential
  libxc_pot_t pot;
  /// Energy density
  double e;
} libxc_debug_t;

/**
 * \class AngularGrid
 *
 * \brief Angular integration grid on a radial shell of an atom
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

class AngularGrid {
 protected:
  /// Shell info
  angshell_t info;
  /// Basis set pointer
  const BasisSet *basp;
  /// Use Lobatto quadrature? (Default is Lebedev)
  bool use_lobatto;

  /// Integration points
  std::vector<gridpoint_t> grid;

  /// List of potentially important shells
  std::vector<size_t> pot_shells;
  /// List of potentially important functions
  arma::uvec pot_bf_ind;

  /// List of important shells
  std::vector<size_t> shells;
  /// Indices of first functions on shell
  arma::uvec bf_i0;
  /// Amount of functions on shell
  arma::uvec bf_N;

  /// List of important functions
  arma::uvec bf_ind;
  /// List of important functions in potentials' list
  arma::uvec bf_potind;

  /// Duplicate values of weights here
  arma::rowvec w;
  /// Values of important functions in grid points, Nbf * Ngrid
  arma::mat bf;
  /// x gradient
  arma::mat bf_x;
  /// y gradient
  arma::mat bf_y;
  /// z gradient
  arma::mat bf_z;
  /// Values of laplacians in grid points, (3*Nbf) * Ngrid
  arma::mat bf_lapl;

  /// Values of Hessians in grid points, (9*Nbf) * Ngrid; used for GGA force
  arma::mat bf_hess;
  /// Values of x gradient of laplacian; used for MGGA force
  arma::mat bf_lx;
  /// Values of y gradient of laplacian; used for MGGA force
  arma::mat bf_ly;
  /// Values of z gradient of laplacian; used for MGGA force
  arma::mat bf_lz;


  /// Density helper matrices: P_{uv} chi_v, and P_{uv} nabla(chi_v)
  arma::mat Pv, Pv_x, Pv_y, Pv_z;
  /// Same for spin-polarized
  arma::mat Pav, Pav_x, Pav_y, Pav_z;
  arma::mat Pbv, Pbv_x, Pbv_y, Pbv_z;

  /// Is gradient needed?
  bool do_grad;
  /// Is kinetic energy density needed?
  bool do_tau;
  /// Is laplacian needed?
  bool do_lapl;
  /// Is Hessian needed? (For GGA force)
  bool do_hess;
  /// Is gradient of laplacian needed? (For MGGA force)
  bool do_lgrad;

  /// Spin-polarized calculation?
  bool polarized;

  /// GGA functional used? (Set in compute_xc, only affects eval_Fxc)
  bool do_gga;
  /// Meta-GGA tau used? (Set in compute_xc, only affects eval_Fxc)
  bool do_mgga_t;
  /// Meta-GGA lapl used? (Set in compute_xc, only affects eval_Fxc)
  bool do_mgga_l;

  // LDA stuff:

  /// Density, Nrho x Npts
  arma::mat rho;
  /// Energy density, Npts
  arma::vec exc;
  /// Functional derivative of energy wrt electron density, Nrho x Npts
  arma::mat vxc;

  // GGA stuff

  /// Gradient of electron density, (3 x Nrho) x Npts
  arma::mat grho;
  /// Dot products of gradient of electron density, N x Npts; N=1 for closed-shell and 3 for open-shell
  arma::mat sigma;
  /// Functional derivative of energy wrt gradient of electron density
  arma::mat vsigma;

  // Meta-GGA stuff

  /// Laplacian of electron density
  arma::mat lapl;
  /// Kinetic energy density
  arma::mat tau;

  /// Functional derivative of energy wrt laplacian of electron density
  arma::mat vlapl;
  /// Functional derivative of energy wrt kinetic energy density
  arma::mat vtau;

  // VV10 stuff
  /// Density threshold
  double VV10_thr;
  /// Helper array used in kernel computation to avoid memory thrashing
  arma::mat VV10_arr;

  /// Get density data for wanted point
  libxc_dens_t get_dens(size_t idx) const;
  /// Get potential data for wanted point
  libxc_pot_t get_pot(size_t idx) const;
  /// Get density and potential data for wanted point
  libxc_debug_t get_data(size_t idx) const;

  /// Add radial shell in Lobatto angular scheme, w/o Becke partitioning or pruning
  void lobatto_shell();
  /// Add radial shell in Lebedev scheme, w/o Becke partitioning or pruning
  void lebedev_shell();
  /// Update list of important basis functions
  void update_shell_list();
  /// Collect weights from grid into w array
  void get_weights();

  /// Next angular grid
  void next_grid();

 public:
  /**
   * Constructor. Need to set shell and basis set before using the
   * construct() functions */
  AngularGrid(bool lobatto=false);
  /// Destructor
  ~AngularGrid();

  /// Set basis set
  void set_basis(const BasisSet & basis);
  /// Set radial shell
  void set_grid(const angshell_t & shell);

  /// Get the quadrature grid
  std::vector<gridpoint_t> get_grid() const;

  /// Check necessity of computing gradient and laplacians, necessary for compute_bf!
  void check_grad_tau_lapl(int x_func, int c_func);
  /// Get necessity of computing gradient and laplacians
  void get_grad_tau_lapl(bool & grad, bool & tau, bool & lapl) const;
  /// Set necessity of computing gradient and laplacians, necessary for compute_bf!
  void set_grad_tau_lapl(bool grad, bool tau, bool lapl);
  /// Set necessity of computing Hessian and gradient of Laplacian
  void set_hess_lgrad(bool hess, bool lgrad);

  /// Construct a fixed size grid
  angshell_t construct();
  /// Construct adaptively a grid centered on the cenind:th center, restricted calculation
  angshell_t construct(const arma::mat & P, double ftol, int x_func, int c_func);
  /// Construct adaptively a grid centered on the cenind:th center, unrestricted calculation
  angshell_t construct(const arma::mat & Pa, const arma::mat & Pb, double ftol, int x_func, int c_func);
  /// Construct adaptively a grid centered on the cenind:th center, SIC calculation
  angshell_t construct(const arma::cx_vec & C, double ftol, int x_func, int c_func);

  /// Construct a dummy grid that is only meant for the overlap matrix (Becke charges)
  angshell_t construct_becke(double otol);
  /// Construct a dummy grid that is only meant for the overlap matrix (Hirshfeld charges)
  angshell_t construct_hirshfeld(const Hirshfeld & Hirsh, double otol);

  /// Form radial shell and compute basis functions
  void form_grid();
  /// Form radial shell using Hirshfeld weights and compute basis functions
  void form_hirshfeld_grid(const Hirshfeld & hirsh);

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
  void becke_weights(double a=0.7);
  /// Compute Hirshfeld weight for grid points
  void hirshfeld_weights(const Hirshfeld & hirsh);

  /// Prune points with small weight
  void prune_points();
  /// Compute basis functions on grid points
  void compute_bf();
  /// Free memory
  void free();

  /// Screen wrt small density, returns list of points with nonnegligible values
  arma::uvec screen_density(double thr=1e-10) const;

  /// Update values of density, restricted calculation
  void update_density(const arma::mat & P);
  /// Update values of density, unrestricted calculation
  void update_density(const arma::mat & Pa, const arma::mat & Pb);
  /// Update values of density, self-interaction correction
  void update_density(const arma::cx_vec & C);

  /// Get density list; used to determine isosurface values for orbital plots
  void get_density(std::vector<dens_list_t> & list) const;

  /// Compute number of electrons
  double compute_Nel() const;

  /// Print out grid information
  void print_grid() const;
  /// Print density information
  void print_density(FILE *f) const;
  /// Print potential information
  void print_potential(int func_id, FILE *f) const;
  /// Check potential data for NaNs
  void check_potential(FILE *f) const;

  /// Initialize XC arrays
  void init_xc();
  /// Compute XC functional from density and add to total XC
  /// array. Pot toggles evaluation of potential
  void compute_xc(int func_id, bool pot);
  /// Evaluate exchange/correlation energy
  double eval_Exc() const;
  /// Zero out energy
  void zero_Exc();
  /// Numerical clean up of xc
  void check_xc();

  /// Initialize VV10 calculation
  void init_VV10(double b, double C, bool pot);
  /// Collect VV10 data
  void collect_VV10(arma::mat & data, std::vector<size_t> & idx, double b, double C, bool nl) const;

  /**
   * Evaluates VV10 energy and potential and add to total array
   *
   * Implementation is described in O. Vydrov and T. Van Voorhis,
   * "Nonlocal van der Waals density functional: The simpler the
   * better", J. Chem. Phys. 133, 244103 (2010).
   */
  void compute_VV10(const std::vector<arma::mat> & nldata, double b, double C);
  /// Same thing, but also evaluate the grid contribution to the force
  arma::vec compute_VV10_F(const std::vector<arma::mat> & nldata, const std::vector<angshell_t> & nlgrids, double b, double C);

  /// Evaluate atomic contribution to overlap matrix
  void eval_overlap(arma::mat & S) const;
  /// Evaluate diagonal elements of overlap matrix
  void eval_diag_overlap(arma::vec & S) const;

  /**
   *
   * Evaluate atomic contribution to weighted overlap matrix given by
   *
   * \f$ \int \frac
   * {\rho_{i\sigma}^{k}(\mathbf{r})\chi_{\mu}(\mathbf{r})\chi_{\nu}(\mathbf{r})}
   * {\rho_{\sigma}^{k}(\mathbf{r})} {\rm d}^{3}\mathbf{r} \f$
   */
  void eval_overlap(const arma::cx_mat & Cocc, size_t io, double k, arma::mat & S, double thr) const;

  /// Same thing, but do contraction over SI energies for derivatives
  void eval_overlap(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, arma::mat & S, double thr) const;

  /**
   *
   * Evaluate atomic contribution to weighted overlap matrix given by
   *
   * \f$ \int \left(
   * \frac{\tau_{\sigma}^{\text{W}}(\mathbf{r})}{\tau_{\sigma}(\mathbf{r})}
   * \right)^{k} \chi_{\mu} (\mathbf{r}) \chi_{\nu} (\mathbf{r})
   * \mathrm{d}^{3} \mathbf{r} \f$
   */
  void eval_tau_overlap(const arma::cx_mat & Cocc, double k, arma::mat & S, double thr) const;

  /// Calculate the GGA and meta-GGA type terms for the derivative
  void eval_tau_overlap_deriv(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, arma::mat & S, double thr) const;

  /// Evaluate Fock matrix, restricted calculation
  void eval_Fxc(arma::mat & H) const;
  /// Evaluate Fock matrix, unrestricted calculation
  void eval_Fxc(arma::mat & Ha, arma::mat & Hb, bool beta=true) const;

  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), restricted calculation
  void eval_diag_Fxc(arma::vec & H) const;
  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), unrestricted calculation
  void eval_diag_Fxc(arma::vec & Ha, arma::vec & Hb) const;
  /// Evaluate diagonal elements of Fock matrix (for adaptive grid formation), unrestricted calculation
  void eval_diag_Fxc_SIC(arma::vec & H) const;

  /// Evaluate force, restricted
  arma::vec eval_force_r() const;
  /// Evaluate force, unrestricted
  arma::vec eval_force_u() const;

  /// Evaluate SAP
  void eval_SAP(const SAP & sap, arma::mat & Vo) const;
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
  std::vector<AngularGrid> wrk;
  /// Radial grids
  std::vector<angshell_t> grids;
  /// Basis set
  const BasisSet * basp;
  /// Verbose operation?
  bool verbose;

  /// Prune shells with no points
  void prune_shells();

 public:
  /// Dummy constructor
  DFTGrid();
  /// Constructor
  DFTGrid(const BasisSet * bas, bool verbose=true, bool lobatto=false);
  /// Destructor
  ~DFTGrid();

  /// Set verbose operation
  void set_verbose(bool ver);

  /// Create fixed size grid
  void construct(int nrad, int lmax, int x_func, int c_func, bool strict);
  /// Create fixed size grid
  void construct(int nrad, int lmax, bool grad, bool tau, bool lapl, bool strict, bool nl);
  /// Create grid for restricted calculation
  void construct(const arma::mat & P, double ftol, int x_func, int c_func);
  /// Create grid for unrestricted calculation
  void construct(const arma::mat & Pa, const arma::mat & Pb, double ftol, int x_func, int c_func);
  /// Create grid for SIC calculation
  void construct(const arma::cx_mat & C, double ftol, int x_func, int c_func);

  /// Create dummy grid for Becke charges (only overlap matrix)
  void construct_becke(double stol);
  /// Create dummy grid for Hirshfeld charges (only overlap matrix)
  void construct_hirshfeld(const Hirshfeld & hirsh, double stol);

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
   * electron density, SIC calculation.
   */
  void eval_Fxc(int x_func, int c_func, const arma::cx_mat & C, std::vector<arma::mat> & H, std::vector<double> & Exc, std::vector<double> & Nel, bool fock);

  /// Compute VV10
  void eval_VV10(DFTGrid & nlgrid, double b, double C, const arma::mat & P, arma::mat & H, double & Exc, bool fock=true);

  /// Evaluate overlap matrix numerically
  arma::mat eval_overlap();
  /// Evaluate overlap matrix numerically in the inuc:th region
  arma::mat eval_overlap(size_t inuc);
  /// Evaluate overlap matrices numerically
  std::vector<arma::mat> eval_overlaps();

  /// Evaluate weighted overlap (for PZ-SIC)
  arma::mat eval_overlap(const arma::cx_mat & Cocc, size_t io, double k, double thr=1e-10);
  /// Evaluate weighted overlap derivative terms (for PZ-SIC)
  arma::mat eval_overlap(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, double thr=1e-10);
  /// Evaluate weighted overlap (for PZ-SIC)
  arma::mat eval_tau_overlap(const arma::cx_mat & Cocc, double k, double thr=1e-10);
  /// Evaluate weighted overlap derivative terms (for PZ-SIC)
  arma::mat eval_tau_overlap_deriv(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, double thr=1e-10);

  /// Evaluate overlap matrices numerically
  arma::mat eval_hirshfeld_overlap(const Hirshfeld & hirsh, size_t inuc);
  /// Evaluate overlap matrices numerically
  std::vector<arma::mat> eval_hirshfeld_overlaps(const Hirshfeld & hirsh);

  /// Evaluate SAP
  arma::mat eval_SAP();

  /// Evaluate density
  std::vector<dens_list_t> eval_dens_list(const arma::mat & P);

  /// Evaluate force
  arma::vec eval_force(int x_func, int c_func, const arma::mat & P);
  /// Evaluate force
  arma::vec eval_force(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb);
  /// Evaluate NL force
  arma::vec eval_VV10_force(DFTGrid & nlgrid, double b, double C, const arma::mat & P);

  /// Compute density cutoff threshold.
  double density_threshold(const arma::mat & P, double thr);

  /// Print out grid information
  void print_grid(std::string met="XC") const;

  /// Print out grid information for a Krack grid
  void krack_grid_info(double otol) const;
  /// Print out grid information for a Koster grid
  void koster_grid_info(double ftol) const;

  /// Print out density data
  void print_density(const arma::mat & P, std::string densname="density.dat");
  /// Print out density data
  void print_density(const arma::mat & Pa, const arma::mat & Pb, std::string densname="density.dat");
  /// Print out potential data
  void print_potential(int func_id, const arma::mat & Pa, const arma::mat & Pb, std::string potname="potential.dat");
  /// Check potential data
  void check_potential(int func_id, const arma::mat & Pa, const arma::mat & Pb, std::string potname="potential_nan.dat");
};

/// BLAS routine for LDA-type quadrature
template<typename T> void increment_lda(arma::Mat<T> & H, const arma::rowvec & vxc, const arma::Mat<T> & f) {
  if(f.n_cols != vxc.n_elem) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Number of functions " << f.n_cols << " and potential values " << vxc.n_elem << " do not match!\n";
    throw std::runtime_error(oss.str());
  }
  if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Size of basis function (" << f.n_rows << "," << f.n_cols << ") and Fock matrix (" << H.n_rows << "," << H.n_cols << ") doesn't match!\n";
    throw std::runtime_error(oss.str());
  }

  // Form helper matrix
  arma::Mat<T> fhlp(f);
  for(size_t i=0;i<fhlp.n_rows;i++)
    for(size_t j=0;j<fhlp.n_cols;j++)
      fhlp(i,j)*=vxc(j);
  H+=fhlp*arma::trans(f);
}

/// Same but with density-based screening
template<typename T> void increment_lda(arma::Mat<T> & H, const arma::rowvec & vxc, const arma::Mat<T> & f, const arma::uvec & screen) {
  if(f.n_cols != vxc.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of matrices doesn't match!\n");
  }
  if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");
  }

  // Form helper matrix
  arma::Mat<T> fhlp(f);
  for(size_t i=0;i<fhlp.n_rows;i++)
    for(size_t j=0;j<fhlp.n_cols;j++)
      fhlp(i,j)*=vxc(j);
  H+=fhlp.cols(screen)*arma::trans(f.cols(screen));
}

/// BLAS routine for GGA-type quadrature
template<typename T> void increment_gga(arma::Mat<T> & H, const arma::mat & gn, const arma::Mat<T> & f, arma::Mat<T> f_x, arma::Mat<T> f_y, arma::Mat<T> f_z) {
  if(gn.n_cols!=3) {
    ERROR_INFO();
    throw std::runtime_error("Grad rho must have three columns!\n");
  }
  if(f.n_rows != f_x.n_rows || f.n_cols != f_x.n_cols || f.n_rows != f_y.n_rows || f.n_cols != f_y.n_cols || f.n_rows != f_z.n_rows || f.n_cols != f_z.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function and derivative matrices doesn't match!\n");
  }
  if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");
  }

  // Compute helper: gamma_{ip} = \sum_c \chi_{ip;c} gr_{p;c}
  //                 (N, Np)    =        (N Np; c)    (Np, 3)
  arma::Mat<T> gamma(f.n_rows,f.n_cols);
  gamma.zeros();
  {
    // Helper
    arma::rowvec gc;

    // x gradient
    gc=arma::trans(gn.col(0));
    for(size_t j=0;j<f_x.n_cols;j++)
      for(size_t i=0;i<f_x.n_rows;i++)
	f_x(i,j)*=gc(j);
    gamma+=f_x;

    // x gradient
    gc=arma::trans(gn.col(1));
    for(size_t j=0;j<f_y.n_cols;j++)
      for(size_t i=0;i<f_y.n_rows;i++)
	f_y(i,j)*=gc(j);
    gamma+=f_y;

    // z gradient
    gc=arma::trans(gn.col(2));
    for(size_t j=0;j<f_z.n_cols;j++)
      for(size_t i=0;i<f_z.n_rows;i++)
	f_z(i,j)*=gc(j);
    gamma+=f_z;
  }

  // Form Fock matrix
  H+=gamma*arma::trans(f) + f*arma::trans(gamma);
}

/// BLAS routine for GGA-type quadrature
template<typename T> void increment_gga(arma::Mat<T> & H, const arma::mat & gn, const arma::Mat<T> & f, arma::Mat<T> f_x, arma::Mat<T> f_y, arma::Mat<T> f_z, const arma::uvec & screen) {
  if(gn.n_cols!=3) {
    ERROR_INFO();
    throw std::runtime_error("Grad rho must have three columns!\n");
  }
  if(f.n_rows != f_x.n_rows || f.n_cols != f_x.n_cols || f.n_rows != f_y.n_rows || f.n_cols != f_y.n_cols || f.n_rows != f_z.n_rows || f.n_cols != f_z.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function and derivative matrices doesn't match!\n");
  }
  if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");
  }

  // Compute helper: gamma_{ip} = \sum_c \chi_{ip;c} gr_{p;c}
  //                 (N, Np)    =        (N Np; c)    (Np, 3)
  arma::Mat<T> gamma(f.n_rows,f.n_cols);
  gamma.zeros();
  {
    // Helper
    arma::rowvec gc;

    // x gradient
    gc=arma::trans(gn.col(0));
    for(size_t j=0;j<f_x.n_cols;j++)
      for(size_t i=0;i<f_x.n_rows;i++)
	f_x(i,j)*=gc(j);
    gamma+=f_x;

    // x gradient
    gc=arma::trans(gn.col(1));
    for(size_t j=0;j<f_y.n_cols;j++)
      for(size_t i=0;i<f_y.n_rows;i++)
	f_y(i,j)*=gc(j);
    gamma+=f_y;

    // z gradient
    gc=arma::trans(gn.col(2));
    for(size_t j=0;j<f_z.n_cols;j++)
      for(size_t i=0;i<f_z.n_rows;i++)
	f_z(i,j)*=gc(j);
    gamma+=f_z;
  }

  // Form Fock matrix
  H+=gamma.cols(screen)*arma::trans(f.cols(screen)) + f.cols(screen)*arma::trans(gamma.cols(screen));
}

/// BLAS routine for meta-GGA kinetic energy type quadrature
template<typename T> void increment_mgga_kin(arma::Mat<T> & H, const arma::rowvec & vtaul, const arma::Mat<T> & f_x, const arma::Mat<T> & f_y, const arma::Mat<T> & f_z) {
  // This is equivalent to LDA incrementation on the three components!
  increment_lda<T>(H,vtaul,f_x);
  increment_lda<T>(H,vtaul,f_y);
  increment_lda<T>(H,vtaul,f_z);
}

template<typename T> void increment_mgga_kin(arma::Mat<T> & H, const arma::rowvec & vtaul, const arma::Mat<T> & f_x, const arma::Mat<T> & f_y, const arma::Mat<T> & f_z, const arma::uvec & screen) {
  // This is equivalent to LDA incrementation on the three components!
  increment_lda<T>(H,vtaul,f_x,screen);
  increment_lda<T>(H,vtaul,f_y,screen);
  increment_lda<T>(H,vtaul,f_z,screen);
}

/// BLAS routine for meta-GGA laplacian type quadrature
template<typename T> void increment_mgga_lapl(arma::Mat<T> & H, const arma::rowvec & vl, const arma::Mat<T> & f, const arma::Mat<T> & f_lapl, const arma::uvec & screen) {
  if(f.n_rows != f_lapl.n_rows || f.n_cols != f_lapl.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function and laplacian matrices doesn't match!\n");
  }
  if(f.n_cols != vl.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function matrix and potential doesn't match!\n");
  }
  if(H.n_rows != f.n_rows || H.n_cols != f.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of basis function and Fock matrices doesn't match!\n");
  }

  // Absorb the potential into the function values
  arma::Mat<T> fhlp(f);
  for(size_t i=0;i<fhlp.n_rows;i++)
    for(size_t j=0;j<fhlp.n_cols;j++)
      fhlp(i,j)*=vl(j);
  // Fock matrix contribution is
  H+=f_lapl.cols(screen)*arma::trans(fhlp.cols(screen)) + fhlp.cols(screen)*arma::trans(f_lapl.cols(screen));
}

#endif
