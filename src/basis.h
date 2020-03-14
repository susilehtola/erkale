/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_BASIS
#define ERKALE_BASIS

#include "global.h"

#include <armadillo>
#include <vector>
#include <string>
#include <cfloat>

/// Forward declaration
class Settings;

#include "xyzutils.h"

/// Angular momentum notation for shells
const char shell_types[]={'S','P','D','F','G','H','I','J','K','L','M','N','O','Q','R'};
/// Maximum angular momentum supported in current version of ERKALE
const int max_am=sizeof(shell_types)/sizeof(shell_types[0])-1;

/// Structure for defining shells of functions
typedef struct {
  /// Angular momentum in x direction
  int l;
  /// Angular momentum in y direction
  int m;
  /// Angular momentum in z direction
  int n;
  /// Relative normalization coefficient
  double relnorm;
} shellf_t;

/// Coordinates structure
typedef struct {
  /// x coordinate
  double x;
  /// y coordinate
  double y;
  /// z coordinate
  double z;
} coords_t;

arma::vec coords_to_vec(const coords_t & c);
coords_t vec_to_coords(const arma::vec & v);

// Forward declaration
class GaussianShell;

/// Nucleus structure
typedef struct {
  /// Index of nucleus
  size_t ind;
  /// Location of nucleus
  coords_t r;
  /// Counterpoise nucleus..?
  bool bsse;

  /// Type of nucleus
  std::string symbol;
  /// Nuclear charge
  int Z;
  /// Net charge in system (used for atomic guess)
  int Q;

  /// List of shells located on nucleus
  std::vector<const GaussianShell *> shells;
} nucleus_t;

/// Comparison operator
bool operator==(const nucleus_t & lhs, const nucleus_t & rhs);

/// Structure for unique shell pairs
typedef struct {
  /// Index of first shell
  size_t is;
  /// Index of first functions on first shell
  size_t i0;
  /// Angular momentum of first shell
  int li;

  /// Index of second shell
  size_t js;
  /// Index of first function on second shell
  size_t j0;
  /// Angular momentum of second shell
  int lj;
} shellpair_t;

/// Comparison operator for shellpairs for ordering into libint order
bool operator<(const shellpair_t & lhs, const shellpair_t & rhs);

/// Helper for integral sorts
typedef struct {
  /// First shell
  size_t is;
  /// First function on shell
  size_t i0;
  /// Amount of functions on shell
  size_t Ni;

  /// Second shell
  size_t js;
    /// First function on shell
  size_t j0;
  /// Amount of functions on shell
  size_t Nj;

  /// Maximum (uv|uv)^1/2 on shell
  double eri;
} eripair_t;

/// Comparison operator
bool operator<(const eripair_t & lhs, const eripair_t & rhs);

// Forward declaration
class BasisSetLibrary;
class ElementBasisSet;

/// Order shells solely on merit of exponents (for forming density fitting basis)
bool exponent_compare(const GaussianShell & lhs, const GaussianShell & rhs);

/// Comparison operator
bool operator==(const coords_t & lhs, const coords_t & rhs);
/// Compute displacement
coords_t operator-(const coords_t & lhs, const coords_t & rhs);
/// Compute sum
coords_t operator+(const coords_t & lhs, const coords_t & rhs);
/// Compute scaling by division
coords_t operator/(const coords_t & lhs, double fac);
/// Compute scaling by multiplication
coords_t operator*(const coords_t & lhs, double fac);

/// Compute squared norm
inline double normsq(const coords_t & r) {
  return r.x*r.x + r.y*r.y + r.z*r.z;
}

/// Compute norm
inline double norm(const coords_t & r) {
  return sqrt(normsq(r));
}

/// Structure for contractions
typedef struct {
  /// Coefficient
  double c;
  /// Exponent
  double z;
} contr_t;
/// Comparison for contractions
bool operator<(const contr_t & lhs, const contr_t & rhs);
/// Identity for contractions
bool operator==(const contr_t & lhs, const contr_t & rhs);

#include "basislibrary.h"


/**
 * \class BasisSet
 *
 * \brief Class for a Gaussian basis set
 *
 * This class contains the data structures necessary for Gaussian
 * basis sets, and functions for evaluating integrals over them.
 *
 * \author Susi Lehtola
 * \date 2011/05/05 20:17
 */

/// Basis set
class BasisSet {
  /// Nuclei
  std::vector<nucleus_t> nuclei;
  /// Basis functions
  std::vector<GaussianShell> shells;

  /// Use spherical harmonics by default as basis?
  bool uselm;
  /// Use cartesian s and p functions if spherical harmonics are used?
  bool optlm;

  /// Internuclear distances
  arma::mat nucleardist;
  /// List of unique shell pairs
  std::vector<shellpair_t> shellpairs;

  /// Ranges of shells
  std::vector<double> shell_ranges;

  /// Check for same geometry
  bool same_geometry(const BasisSet & rhs) const;
  /// Check for same shells
  bool same_shells(const BasisSet & rhs) const;

 public:
  /// Dummy constructor
  BasisSet();
  /// Construct basis set with Nat atoms, using given settings
  BasisSet(size_t Nat);
  /// Destructor
  ~BasisSet();

  /**
   * Generate density fitting basis
   *
   * The procedure has been documented in the article
   *
   * R. Yang, A. P. Rendell and M. J. Frisch, "Automatically generated
   * Coulomb fitting basis sets: Design and accuracy for systems
   * containing H to Kr", J. Chem. Phys. 127 (2007), 074102.
   */
  BasisSet density_fitting(double fsam=1.5, int lmaxinc=1) const;

  /**
   * Generate Coulomb and exchange fitting basis
   *
   * The procedure has been documented in the article
   *
   * O. Vahtras, J. Almlöf and M. W. Feyereisen, "Integral approximations
   * for LCAO-SCF calculations", Chem. Phys. Lett. 213, p. 514 - 518 (1993).
   */
  BasisSet exchange_fitting() const;

  /// Decontract basis set, m gives mapping from old functions to new ones
  BasisSet decontract(arma::mat & m) const;

  /// Add nucleus
  void add_nucleus(const nucleus_t & nuc);
  /// Add a shell to a nucleus and sort functions if wanted
  void add_shell(size_t nucind, const GaussianShell & sh, bool sort=true);
  /// Add a shell to a nucleus and sort functions if wanted
  void add_shell(size_t nucind, int am, bool uselm, const std::vector<contr_t> & C, bool sort=true);
  /// Add all shells to a nucleus and sort functions if wanted
  void add_shells(size_t nucind, ElementBasisSet el, bool sort=true);

  /// Sort shells in nuclear order, then by angular momentum, then by exponents
  void sort();
  /// Check numbering of basis functions
  void check_numbering();
  /// Update the nuclear list of shells
  void update_nuclear_shell_list();

  /* Finalization routines */

  /// Compute nuclear distance table
  void compute_nuclear_distances();

  /// Form list of unique shell pairs
  void form_unique_shellpairs();
  /// Get list of unique shell pairs
  std::vector<shellpair_t> get_unique_shellpairs() const;

  /// Get list of ERI pairs
  std::vector<eripair_t> get_eripairs(arma::mat & Q, arma::mat & M, double thr, double omega=0.0, double alpha=1.0, double beta=0.0, bool verbose=false) const;

  /// Convert contractions from normalized primitives to unnormalized primitives
  void convert_contractions();
  /// Convert contraction on given shell
  void convert_contraction(size_t ish);

  /// Normalize contractions. If !coeffs, only cartesian factors are calculated.
  void normalize(bool coeffs=true);
  /// Normalize contractions in Coulomb norm (for density fitting)
  void coulomb_normalize();

  /// Do all of the above
  void finalize(bool convert=false, bool normalize=true);

  /// Get distance of nuclei
  double nuclear_distance(size_t i, size_t j) const;
  /// Get nuclear distances
  arma::mat nuclear_distances() const;

  /// Get angular momentum of shell
  int get_am(size_t shind) const;
  /// Get maximum angular momentum in basis set
  int get_max_am() const;
  /// Get maximum number of contractions
  size_t get_max_Ncontr() const;

  /// Get index of last function, throws an exception if no functions exist
  size_t get_last_ind() const;
  /// Get index of first function on shell
  size_t get_first_ind(size_t shind) const;
  /// Get index of last function on shell
  size_t get_last_ind(size_t shind) const;

  /// Get R^2 expectation value (measure of basis function extent)
  arma::vec get_bf_Rsquared() const;
  /// Get shell indices of basis functions
  arma::uvec shell_indices() const;
  /// Find shell index of basis function
  size_t find_shell_ind(size_t find) const;

  /// Get shells in basis set
  std::vector<GaussianShell> get_shells() const;
  /// Get ind:th shell
  GaussianShell get_shell(size_t shind) const;
  /// Get index of the center of the ind'th shell
  size_t get_shell_center_ind(size_t shind) const;
  /// Get coordinates of center of ind'th shell
  coords_t get_shell_center(size_t shind) const;

  /// Get exponential contraction of the ind:th shell
  std::vector<contr_t> get_contr(size_t ind) const;
  /// Get normalized exponential contraction of the ind:th shell
  std::vector<contr_t> get_contr_normalized(size_t ind) const;
  /// Get the cartesian functions on the ind:th shell
  std::vector<shellf_t> get_cart(size_t ind) const;

  /// Are spherical harmonics the default for new shells?
  bool is_lm_default() const;
  /// Is shell ind using spherical harmonics?
  bool lm_in_use(size_t ind) const;
  /// Toggle the use of spherical harmonics on shell ind
  void set_lm(size_t ind, bool lm);

  /// Get m values of basis functions
  arma::ivec get_m_values() const;
  /// Unique m values in basis set
  arma::ivec unique_m_values() const;
  /// Mapping from m value to unique m value
  std::map<int, arma::uword> unique_m_map() const;
  /// Count occupied orbitals
  arma::imat count_m_occupied(const arma::mat & C) const;
  /// Count occupied orbitals
  arma::imat count_m_occupied(const arma::mat & Ca, const arma::mat & Cb) const;

  /// Get indices of basis functions with wanted m value
  arma::uvec m_indices(int m) const;

  /// Get transformation matrix
  arma::mat get_trans(size_t ind) const;

  /// Fill spherical harmonics transformation table
  void fill_sph_transmat();

  /// Get size of basis set
  size_t get_Nbf() const;
  /// Get amount of cartesian functions on shell
  size_t get_Ncart() const;
  /// Get amount of spherical harmonics on shell
  size_t get_Nlm() const;

  /// Get number of shells
  size_t get_Nshells() const;
  /// Get number of basis functions on shell
  size_t get_Nbf(size_t ind) const;
  /// Get number of cartesians on shell
  size_t get_Ncart(size_t ind) const;

  /**
   * Get range of shell (distance at which functions have dropped below epsilon)
   *
   * The default parameter is from the reference R. E. Stratmann,
   * G. E. Scuseria, and M. J. Frisch, "Achieving linear scaling in
   * exchage-correlation density functional quadratures",
   * Chem. Phys. Lett. 257 (1993), pp. 213-223.
   */
  void compute_shell_ranges(double eps=1e-10);
  /// Get precomputed ranges of shells
  std::vector<double> get_shell_ranges() const;
  /// Get range of shells with given value of epsilon
  std::vector<double> get_shell_ranges(double eps) const;

  /// Get distances to other nuclei
  std::vector<double> get_nuclear_distances(size_t inuc) const;

  /// Get number of nuclei
  size_t get_Nnuc() const;
  /// Get nucleus
  nucleus_t get_nucleus(size_t inuc) const;
  /// Get nuclei
  std::vector<nucleus_t> get_nuclei() const;

  /// Get coordinates of all nuclei
  arma::mat get_nuclear_coords() const;
  /// Set coordinates of all nuclei
  void set_nuclear_coords(const arma::mat & coords);

  /// Get coordinates of nucleus
  coords_t get_nuclear_coords(size_t inuc) const;
  /// Get charge of nucleus
  int get_Z(size_t inuc) const;
  /// Get symbol of nucleus
  std::string get_symbol(size_t inuc) const;
  /// Get human readable symbol of nucleus (-Bq)
  std::string get_symbol_hr(size_t inuc) const;

  /// Get basis functions centered on a given atom
  std::vector<GaussianShell> get_funcs(size_t inuc) const;
  /// Get indices of shells centered on a given atom
  std::vector<size_t> get_shell_inds(size_t inuc) const;

  /// Evaluate functions at (x,y,z)
  arma::vec eval_func(double x, double y, double z) const;
  /// Evaluate gradient at (x,y,z)
  arma::mat eval_grad(double x, double y, double z) const;
  /// Evaluate Hessian at (x,y,z)
  arma::mat eval_hess(double x, double y, double z) const;

  /// Evaluate functions of shell ish at (x,y,z)
  arma::vec eval_func(size_t ish, double x, double y, double z) const;
  /// Evaluate gradients of shell ish at (x,y,z)
  arma::mat eval_grad(size_t ish, double x, double y, double z) const;
  /// Evaluate laplacian of shell ish at (x,y,z)
  arma::vec eval_lapl(size_t ish, double x, double y, double z) const;
  /// Evaluate Hessian of shell ish at (x,y,z)
  arma::mat eval_hess(size_t ish, double x, double y, double z) const;
  /// Evaluate gradient of laplacian of shell ish at (x,y,z)
  arma::mat eval_laplgrad(size_t ish, double x, double y, double z) const;

  /// Print out basis set
  void print(bool verbose=false) const;

  /// Calculate transformation matrix from cartesians to spherical harmonics
  arma::mat cart_to_sph_trans() const;
  /// Calculate transfomration matrix from spherical harmonics to cartesians
  arma::mat sph_to_cart_trans() const;

  /// Calculate overlap matrix
  arma::mat overlap() const;
  /// Calculate overlap matrix in Coulomb metric
  arma::mat coulomb_overlap() const;
  /// Calculate overlap with another basis set
  arma::mat overlap(const BasisSet & rhs) const;
  /// Calculate overlap in Coulomb metric with another basis set
  arma::mat coulomb_overlap(const BasisSet & rhs) const;
  /// Calculate kinetic energy matrix
  arma::mat kinetic() const;
  /// Calculate nuclear repulsion matrix
  arma::mat nuclear() const;
  /// Calculate electric potential matrix
  arma::mat potential(coords_t r) const;

  /**
     Calculates the ERI screening matrices
       \f$ Q_{\mu \nu} = (\mu \nu | \mu \nu)^{1/2} \f$
     and
       \f$ M_{\mu \nu} = (\mu \mu | \nu \nu)^{1/2} \f$
     as described in J. Chem. Phys. 147, 144101 (2017).
  */
  void eri_screening(arma::mat & Q, arma::mat & M, double omega=0.0, double alpha=1.0, double beta=0.0) const;

  /// Calculate nuclear Pulay forces
  arma::vec nuclear_pulay(const arma::mat & P) const;
  /// Calculate nuclear Hellman-Feynman force
  arma::vec nuclear_der(const arma::mat & P) const;
  /// Calculate kinetic Pulay forces
  arma::vec kinetic_pulay(const arma::mat & P) const;
  /// Calculate overlap derivative force
  arma::vec overlap_der(const arma::mat & W) const;
  /// Calculate nuclear repulsion force
  arma::vec nuclear_force() const;

  /// Compute moment integral around (x,y,z)
  std::vector<arma::mat> moment(int mom, double x=0.0, double y=0.0, double z=0.0) const;

  /// Compute integrals of basis functions (used in xc-fitting)
  arma::vec integral() const;

  /// Calculate nuclear charge
  int Ztot() const;

  /// Calculate nuclear repulsion energy
  double Enuc() const;

  /// Project MOs to new basis set
  void projectMOs(const BasisSet & old, const arma::colvec & oldE, const arma::mat & oldMOs, arma::colvec & E, arma::mat & MOs, size_t nocc) const;
  /// Translate OMOs to new geometry, assuming same basis set. Virtuals are discarded and regenerated.
  void projectOMOs(const BasisSet & old, const arma::cx_mat & oldOMOs, arma::cx_mat & OMOs, size_t nocc) const;

  /// Are the basis sets the same?
  bool operator==(const BasisSet & rhs) const;

  /// Find "identical" shells in basis set.
  std::vector< std::vector<size_t> > find_identical_shells() const;
};

/// Compute three-center overlap integral
arma::cube three_overlap(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks);

/**
 * \class GaussianShell
 *
 * \brief A shell of Gaussian basis functions of a given angular
 * momentum, sharing the same exponential contraction.
 *
 * \author Susi Lehtola
 * \date 2011/05/05 20:17
 */
class GaussianShell {
  /// Number of first function on shell
  size_t indstart;

  /// Coordinates of center
  coords_t cen;
  /// Index of center
  size_t cenind;

  /// Use spherical harmonics?
  bool uselm;
  /// Transformation matrix to spherical basis
  arma::mat transmat;

  /**
   * Contraction of unnormalized primitives.
   * N.B. Normalization is wrt first function of shell.
   */
  std::vector<contr_t> c;

  /// Angular momentum of shell
  int am;

  /**
   * Table of cartesians, containing am indices
   * and relative normalization factors.
   */
  std::vector<shellf_t> cart;

 public:
  /// Dummy constructor
  GaussianShell();
  /// Constructor, need also to set index of first function and nucleus (see below)
  GaussianShell(int am, bool lm, const std::vector<contr_t> & C);
  /// Destructor
  ~GaussianShell();

  /// Set index of first basis function
  void set_first_ind(size_t ind);
  /// Set center
  void set_center(const coords_t & cenv, size_t cenindv);

  /// Sort exponents in decreasing order
  void sort();

  /**
   * Convert contraction from coefficients of normalized primitives to
   * those of unnormalized primitives.
   */
  void convert_contraction();

  /**
   * Convert contraction from coefficients of normalized density
   * primitives to those of unnormalized primitives.
   */
  void convert_sap_contraction();

  /// Normalize contractions
  void normalize(bool coeffs=true);
  /// Normalize contractions in Coulomb norm (for density fitting)
  void coulomb_normalize();

  /// Get the exponential contraction
  std::vector<contr_t> get_contr() const;
  /// Get cartesians
  std::vector<shellf_t> get_cart() const;

  /**
   * Get contraction coefficients of normalized primitives. For some
   * reason these are at variance with what is put in, e.g. the
   * cc-pVXZ basis set data from the ESML basis set exchange.  Maybe
   * the input isn't really normalized..?
   */
  std::vector<contr_t> get_contr_normalized() const;

  /// Number of functions on shell
  size_t get_Nbf() const;
  /// Number of cartesians on shell
  size_t get_Ncart() const;
  /// Number of spherical harmonics on shell
  size_t get_Nlm() const;

  /**
   * Compute range of shell - how far must one go for absolute value
   * of functions to drop below eps?
   */
  double range(double eps) const;

  /// Are spherical harmonics in use?
  bool lm_in_use() const;
  /// Toggle use of spherical harmonics
  void set_lm(bool lm);
  /// Get transformation matrix to spherical harmonics basis
  arma::mat get_trans() const;

  /// Get number of contractions
  size_t get_Ncontr() const;
  /// Get angular momentum
  int get_am() const;
  /// Get nucleus index
  size_t get_center_ind() const;
  /// Get coordinates
  coords_t get_center() const;

  /// Comparison operator for angular momentum ordering
  bool operator<(const GaussianShell & rhs) const;
  /// Are the two the same?
  bool operator==(const GaussianShell & rhs) const;

  /// Get index of first function on shell
  size_t get_first_ind() const;
  /// Get index of last function on shell
  size_t get_last_ind() const;

  /// Print out information about shell
  void print() const;

  /// Evaluate functions at (x,y,z)
  arma::vec eval_func(double x, double y, double z) const;
  /// Evaluate gradients at (x,y,z)
  arma::mat eval_grad(double x, double y, double z) const;
  /// Evaluate laplacian at (x,y,z)
  arma::vec eval_lapl(double x, double y, double z) const;
  /// Evaluate Hessian at (x,y,z)
  arma::mat eval_hess(double x, double y, double z) const;
  /// Evaluate gradient of laplacian at (x,y,z)
  arma::mat eval_laplgrad(double x, double y, double z) const;

  /// Calculate block overlap matrix between shells
  arma::mat overlap(const GaussianShell & rhs) const;
  /// Calculate block Coulomb overlap matrix between shells
  arma::mat coulomb_overlap(const GaussianShell & rhs) const;
  /// Calculate kinetic energy matrix between shells
  arma::mat kinetic(const GaussianShell & rhs) const;
  /// Calculate nuclear repulsion matrix between shells
  arma::mat nuclear(double cx, double cy, double cz, const GaussianShell & rhs) const;

  /// Calculate nuclear Pulay forces
  arma::vec nuclear_pulay(double cx, double cy, double cz, const arma::mat & P, const GaussianShell & rhs) const;
  /// Calculate nuclear Hellman-Feynman force
  arma::vec nuclear_der(double cx, double cy, double cz, const arma::mat & P, const GaussianShell & rhs) const;
  /// Calculate kinetic Pulay forces
  arma::vec kinetic_pulay(const arma::mat & P, const GaussianShell & rhs) const;
  /// Calculate overlap derivative
  arma::vec overlap_der(const arma::mat & W, const GaussianShell & rhs) const;

  /// Calculate integral over function (used in xc-fitting)
  arma::vec integral() const;

  /// Calculate moment integrals around (x,y,z) between shells
  std::vector<arma::mat> moment(int mom, double x, double y, double z, const GaussianShell & rhs) const;
};

/// Get dummy shell
GaussianShell dummyshell();

/// Form index helper table: i*(i+1)/2
std::vector<size_t> i_idx(size_t N);

/// Construct basis set from input
void construct_basis(BasisSet & basis, const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib);
/// Constuct basis set from input
void construct_basis(BasisSet & basis, const std::vector<nucleus_t> & nuclei, const BasisSetLibrary & baslib);

/// Compute values of orbitals at given point
arma::vec compute_orbitals(const arma::mat & C, const BasisSet & bas, const coords_t & r);
/// Compute Fermi-Löwdin orbitals
arma::mat fermi_lowdin_orbitals(const arma::mat & C, const BasisSet & bas, const arma::mat & r);

/// Compute density at given point
double compute_density(const arma::mat & P, const BasisSet & bas, const coords_t & r);
/// Compute density and gradient at a given point
void compute_density_gradient(const arma::mat & P, const BasisSet & bas, const coords_t & r, double & d, arma::vec & g);
/// Compute density, gradient and hessian at a given point
void compute_density_gradient_hessian(const arma::mat & P, const BasisSet & bas, const coords_t & r, double & d, arma::vec & g, arma::mat & h);

/// Compute electrostatic potential at given point
double compute_potential(const arma::mat & P, const BasisSet & bas, const coords_t & r);

/**
 * Compute electron localization function
 *
 * A. D. Becke and K. E. Edgecombe, "A simple measure of electron
 * localization in atomic and molecular systems", J. Chem. Phys. 92,
 * 5397 (1990).
 */
double compute_elf(const arma::mat & P, const BasisSet & bas, const coords_t & r);

/// Calculate difference from orthogonality
double orth_diff(const arma::mat & C, const arma::mat & S);
/// Check orthonormality of complex molecular orbitals
double orth_diff(const arma::cx_mat & C, const arma::mat & S);

/// Check orthonormality of real molecular orbitals
void check_orth(const arma::mat & C, const arma::mat & S, bool verbose, double thr=sqrt(DBL_EPSILON));
/// Check orthonormality of complex molecular orbitals
void check_orth(const arma::cx_mat & C, const arma::mat & S, bool verbose, double thr=sqrt(DBL_EPSILON));

/**
 * Construct intrinsic atomic orbitals.
 *
 * G. Knizia, "Intrinsic Atomic Orbitals: An Unbiased Bridge between
 * Quantum Theory and Chemical Concepts", J. Chem. Theory
 * Comput. 9, 4834 (2013). doi: 10.1021/ct400687b.
 *
 * The algorithm returns the IAO matrix, and stores the atomic indices
 * in idx.
 */
arma::mat construct_IAO(const BasisSet & basis, const arma::mat & C, std::vector< std::vector<size_t> > & idx, bool verbose=true, std::string minbas="MINAO.gbs");
/// Same, but using complex coefficients
arma::cx_mat construct_IAO(const BasisSet & basis, const arma::cx_mat & C, std::vector< std::vector<size_t> > & idx, bool verbose=true, std::string minbas="MINAO.gbs");

/// Block matrix by m value
arma::mat block_m(const arma::mat & F, const arma::ivec & mv);
/// Get norm by m value block
arma::mat m_norm(const arma::mat & C, const arma::ivec & mv);
/// Classify by m value
arma::ivec m_classify(const arma::mat & C, const arma::ivec & mv);

#endif
