/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_BASIS
#define ERKALE_BASIS

#include "global.h"

/// Use Obara-Saika for computing 1-electron integrals?
#define OBARASAIKA

/// Use libint for computing two-electron integrals? (Set by CMake)
//#define LIBINT

#include <armadillo>
#include <vector>
#include <string>

#include "basislibrary.h"
#include "settings.h"
#include "xyzutils.h"

#ifdef LIBINT
#include <libint/libint.h>
#endif

/// Angular momentum notation for shells
const char shell_types[]={'S','P','D','F','G','H','I','K','L','M'};
/// Maximum angular momentum supported in current version of ERKALE
const int maxam=9;

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

/// Nucleus structure
typedef struct {
  /// x coordinate of nucleus
  double x;
  /// y coordinate of nucleus
  double y;
  /// z coordinate of nucleus
  double z;
  /// Index of atom in system
  int atind;
  /// Charge
  int Z;
  /// Counterpoise nucleus..?
  bool bsse;
  /// Type of nucleus
  std::string symbol;
} nucleus_t;

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

// Forward declarations
class GaussianShell;
class ElementBasisSet;
class BasisSetLibrary;

/// Order shells solely on merit of exponents (for forming density fitting basis)
bool exponent_compare(const GaussianShell & lhs, const GaussianShell & rhs);

/// Compute displacement
coords_t operator-(const coords_t & lhs, const coords_t & rhs);
/// Compute sum
coords_t operator+(const coords_t & lhs, const coords_t & rhs);
/// Compute scaling by division
coords_t operator/(const coords_t & lhs, double fac);
/// Compute scaling by multiplication
coords_t operator*(const coords_t & lhs, double fac);


/// Compute squared norm
double normsq(const coords_t & r);
/// Compute norm
double norm(const coords_t & r);

/**
 * \class BasisSet
 *
 * \brief Class for a Gaussian basis set
 *
 * This class contains the data structures necessary for Gaussian
 * basis sets, and functions for evaluating integrals over them.
 *
 * \author Jussi Lehtola
 * \date 2011/05/05 20:17
 */

/// Basis set
class BasisSet {
  /// Shells of basis functions
  std::vector<GaussianShell> shells;
  /// Nuclei
  std::vector<nucleus_t> nuclei;

  /// Use spherical harmonics by default as basis?
  bool uselm;

  /// Internuclear distances
  arma::mat nucleardist;
  /// List of unique shell pairs
  std::vector<shellpair_t> shellpairs;
  
  /// Ranges of shells
  std::vector<double> shell_ranges;

#ifdef LIBINT
  /// Libint initialized?
  bool libintok;
#else
  // This variable only exists if LIBINT support has been enabled
  //  bool libintok;
#endif

 public:
  /// Construct basis set with Nat atoms, using given settings
  BasisSet(size_t Nat, const Settings & set);
  /// Destructor
  ~BasisSet();

#ifdef DFT_ENABLED
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
#else
  // This function only exists when DFT support has been enabled
  //  BasisSet density_fitting(double fsam=1.5, int lmaxinc=1) const;
#endif
  
  /// Add functions for element el at cen
  void add_functions(int atind, coords_t cen, ElementBasisSet el);
  /// Add functions at cen
  void add_functions(int atind, coords_t cen, int am, std::vector<double> C, std::vector<double> zeta);
  /// Add nucleus
  void add_nucleus(nucleus_t nuc);
  /// Add nucleus
  void add_nucleus(int atind, coords_t cen, int Z, std::string symbol, bool bsse=0);
  /// Add a shell
  void add_shell(GaussianShell sh);

  /// Check numbering
  void check_numbering();
  /// Sort shells in increasing order of angular momentum
  void sort_am();

  /* Finalization routines */
  /// Compute nuclear distance table
  void compute_nuclear_distances();   

  /// Form list of unique shell pairs
  void form_unique_shellpairs();
  /// Find shell pair
  size_t find_pair(size_t is, size_t js) const;

  /// Get list of unique shell pairs
  std::vector<shellpair_t> get_unique_shellpairs() const;

  /// Convert contractions from normalized primitives to unnormalized primitives
  void convert_contractions();

  /// Normalize contractions
  void normalize();
  /// Normalize contractions in Coulomb norm (for density fitting)
  void coulomb_normalize();

#ifdef LIBINT
  /// Initialize libint
  void libint_init();
  /// Libint has already been initialized elsewhere
  void set_libint_ok();
#else
  // These functions only exist when LIBINT support has been enabled
  //  void libint_init();
  //  void set_libint_ok();
#endif

  /// Do all of the above
#ifdef LIBINT
  void finalize(bool convert=0, bool libintok=0);
#else
  void finalize(bool convert=0);
#endif

  /// Get distance of nuclei
  double nuclear_distance(size_t i, size_t j) const;

  /// Get angular momentum of shell
  int get_am(size_t ind) const;  
  /// Get maximum angular momentum in basis set
  int get_max_am() const;
  /// Get maximum number of contractions
  size_t get_max_Ncontr() const;

  /// Get index of last function, throws an exception if no functions exist
  size_t get_last_ind() const;
  /// Get index of first function on shell
  size_t get_first_ind(size_t ind) const;
  /// Get index of last function on shell
  size_t get_last_ind(size_t ind) const;
  /// Get center of ind'th shell
  size_t get_center_ind(size_t ind) const;

  /// Get shells in basis set
  std::vector<GaussianShell> get_shells() const;
  /// Get ind:th shell
  GaussianShell get_shell(size_t ind) const;
  /// Get coordinates of center of ind'th shell
  coords_t get_shell_coords(size_t ind) const;

  /// Get exponents on the ind:th shell
  std::vector<double> get_zetas(size_t ind) const;
  /// Get contraction coefficients of the ind:th shell
  std::vector<double> get_contr(size_t ind) const;
  /// Get the cartesian functions on the ind:th shell
  std::vector<shellf_t> get_cart(size_t ind) const;

  /// Are spherical harmonics the default for new shells?
  bool is_lm_default() const;
  /// Is shell ind using spherical harmonics?
  bool lm_in_use(size_t ind) const;
  /// Toggle the use of spherical harmonics on shell ind
  void set_lm(size_t ind, bool lm);
  
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

  /// Get range of shell (distance at which functions have dropped below epsilon)
  void compute_shell_ranges(double eps=1e-10);
  /// Get range of shells (distance at which functions have dropped below epsilon)
  std::vector<double> get_shell_ranges() const;

  /// Get distances to other nuclei
  std::vector<double> get_nuclear_distances(size_t inuc) const;

  /// Get number of nuclei
  size_t get_Nnuc() const;
  /// Get nucleus
  nucleus_t get_nuc(size_t inuc) const;
  /// Get coordinates of nucleus
  coords_t get_nuclear_coords(size_t inuc) const;
  /// Get charge of nucleus
  int get_Z(size_t inuc) const;
  /// Get symbol of nucleus
  std::string get_symbol(size_t inuc) const;
  /// Get basis functions centered on a given atom
  std::vector<GaussianShell> get_funcs(size_t inuc) const;
  /// Get indices of shells centered on a given atom
  std::vector<size_t> get_shell_inds(size_t inuc) const;

  /// Evaluate functions at (x,y,z)
  arma::vec eval_func(size_t ish, double x, double y, double z) const;
  /// Evaluate gradients at (x,y,z)
  arma::mat eval_grad(size_t ish, double x, double y, double z) const;
  /// Evaluate laplacian at (x,y,z)
  arma::vec eval_lapl(size_t ish, double x, double y, double z) const;

  /// Print out basis set
  void print() const;

  /// Calculate transformation matrix from cartesians to spherical harmonics
  arma::mat cart_to_sph_trans() const;
  /// Calculate transfomration matrix from spherical harmonics to cartesians
  arma::mat sph_to_cart_trans() const;

  /// Calculate overlap matrix
  arma::mat overlap() const;
  /// Calculate overlap with another basis set
  arma::mat overlap(const BasisSet & rhs) const;
  /// Calculate kinetic energy matrix
  arma::mat kinetic() const;
  /// Calculate nuclear repulsion matrix
  arma::mat nuclear() const;

  /// Compute moment integral around (x,y,z)
  std::vector<arma::mat> moment(int mom, double x=0.0, double y=0.0, double z=0.0) const;

  /// Compute a shell of ERIs, transformed into spherical basis if necessary
  std::vector<double> ERI(size_t is, size_t js, size_t ks, size_t ls) const;

  /// Helper for ERI: compute a shell of cartesian ERIs
  std::vector<double> ERI_cart(size_t is, size_t js, size_t ks, size_t ls) const;

  /// Helper for ERI: calculate cartesian ERI using Huzinaga (mostly for debugging)
  double ERI_cart(size_t is, size_t ii, size_t js, size_t jj, size_t ks, size_t kk, size_t ls, size_t ll) const;

  /// Calculate nuclear charge
  int Ztot() const;
  
  /// Calculate nuclear repulsion energy
  double Enuc() const;

  /// Project MOs from other basis set
  void projectMOs(const BasisSet & old, const arma::colvec & oldE, const arma::mat & oldMOs, arma::colvec & E, arma::mat & MOs) const;
};

/// Compute a shell of ERIs, transformed into spherical basis if necessary
std::vector<double> ERI(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);

#ifdef LIBINT
/// Compute data necessary for libint
void compute_libint_data(Libint_t & libint, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
#endif

/// Compute ERI over cartesian Gaussians
std::vector<double> ERI_cart(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);

/// Collect the integrals from ints and place them in ret, undoing any swaps that may have taken place
void libint_collect(std::vector<double> & ret, const double * ints, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls, bool swap_ij, bool swap_kl, bool swap_ijkl);

/// Compatibility function (G++ 4.5.1 refuses to use the above functions) "invalid conversion from ‘const GaussianShell*’ to ‘size_t’"
std::vector<double> ERI_wrap(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
/// Compatibility function (G++ 4.5.1 refuses to use the above functions) "invalid conversion from ‘const GaussianShell*’ to ‘size_t’"
std::vector<double> ERI_cart_wrap(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);


/**
 * \class GaussianShell
 *
 * \brief A shell of Gaussian basis functions of a given angular
 * momentum, sharing the same exponential contraction.
 *
 * \author Jussi Lehtola
 * \date 2011/05/05 20:17
 */

class GaussianShell {

  /// Number of first function on shell
  size_t indstart;

  /// Coordinates of center
  coords_t cen;
  /// Index of atom
  size_t atind;

  /// Use spherical harmonics?
  bool uselm;
  /// Transformation matrix to spherical basis
  arma::mat transmat;

  /**
   * Contraction coefficients of unnormalized primitives. 
   * N.B. Normalization is wrt first function of shell.*/
  std::vector<double> c;
  /// Exponents of primitives
  std::vector<double> zeta;

  /// Angular momentum functions
  int am;

  /**
   * Table of cartesians, containing am indices
   * and relative normalization factors.
   */
  std::vector<shellf_t> cart; 

 public:
  /// Construct shell, use spherical harmonics by default
  GaussianShell(bool lm=1);
  /// Construct a shell
  GaussianShell(size_t indstart, int am, bool lm, int atind, coords_t cen, std::vector<double> C, std::vector<double> zeta);
  /// Destructor
  ~GaussianShell();

  /**
   * Convert contraction from coefficients of normalized primitives to
   * those of unnormalized primitives.
   */
  void convert_contraction();

  /// Normalize contractions
  void normalize();
  /// Normalize contractions in Coulomb norm (for density fitting)
  void coulomb_normalize();

  /// Get exponents
  std::vector<double> get_zetas() const;
  /// Get contraction coefficients
  std::vector<double> get_contr() const;
  /// Get cartesians
  std::vector<shellf_t> get_cart() const;

  /**
   * Get contraction coefficients of normalized primitives. For some
   * reason these are at variance with what is put in, e.g. the
   * cc-pVXZ basis set data from the ESML basis set exchange.  Maybe
   * the input isn't really normalized..?
   */
  std::vector<double> get_contr_normalized() const;

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
  size_t get_inuc() const;
  /// Get coordinates
  coords_t get_coords() const;

  /// Comparison operator for angular momentum ordering
  bool operator<(const GaussianShell & rhs) const;

  /// Get index of first function on shell
  size_t get_first_ind() const;
  /// Get index of last function on shell
  size_t get_last_ind() const;

  /// Set index of first basis function
  void set_first_ind(size_t ind);
  /// Set index of center
  void set_center_ind(size_t inuc);
  
  /// Print out information about shell
  void print() const;

  /// Evaluate functions at (x,y,z)
  arma::vec eval_func(double x, double y, double z) const;
  /// Evaluate gradients at (x,y,z)
  arma::mat eval_grad(double x, double y, double z) const;
  /// Evaluate laplacian at (x,y,z)
  arma::vec eval_lapl(double x, double y, double z) const;

  /// Calculate block overlap matrix between shells
  arma::mat overlap(const GaussianShell & rhs) const;
  /// Calculate kinetic energy matrix between shells
  arma::mat kinetic(const GaussianShell & rhs) const;
  /// Calculate nuclear repulsion matrix between shells
  arma::mat nuclear(double cx, double cy, double cz, const GaussianShell & rhs) const;

  /// Calculate moment integrals around (x,y,z) between shells
  std::vector<arma::mat> moment(int mom, double x, double y, double z, const GaussianShell & rhs) const;

  /// Compute ERI over cartesian functions
  friend std::vector<double> ERI_cart(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
  /// Compute ERI over cartesian functions using the Huzinaga algorithm (mostly for debugging)
  friend double BasisSet::ERI_cart(size_t is, size_t ii, size_t js, size_t jj, size_t ks, size_t kk, size_t ls, size_t ll) const;
  /// Compute ERI over cartesian functions
  friend std::vector<double> BasisSet::ERI_cart(size_t is, size_t js, size_t ks, size_t ls) const;
#ifdef LIBINT
  /// Normalize libint integrals
  friend void libint_collect(std::vector<double> & ret, const double * ints, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls, bool swap_ij, bool swap_kl, bool swap_ijkl);
  /// Compute data for LIBINT
  friend void compute_libint_data(Libint_t & libint, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
#else
  // These functions only exist when LIBINT support has been enabled
  //  friend void libint_collect(std::vector<double> & ret, const double * ints, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls, bool swap_ij, bool swap_kl, bool swap_ijkl);
  //  friend void compute_libint_data(Libint_t & libint, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls);
#endif
};

/// Compute index of swapped integral
size_t get_swapped_ind(size_t i, size_t Ni, size_t j, size_t Nj, size_t k, size_t Nk, size_t l, size_t Nl, bool swap_ij, bool swap_kl, bool swap_ijkl);

/// Form index helper table: i*(i+1)/2
std::vector<size_t> i_idx(size_t N);

#ifdef LIBINT
/// Construct basis set from input
BasisSet construct_basis(const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib, const Settings & set, bool libintok=0);
#else
/// Construct basis set from input
BasisSet construct_basis(const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib, const Settings & set);
#endif



#endif
