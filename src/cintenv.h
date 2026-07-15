/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_CINTENV
#define ERKALE_CINTENV

#include "basis.h"

#include <memory>
#include <vector>

/// The libcint integral kernels ERKALE uses. The optimizer (precomputed
/// primitive pair data) is per kernel, so they are enumerated here.
typedef enum {
  /// (ij|kl)
  CINT_ERI,
  /// d(ij|kl) / dR_i
  CINT_ERI_IP1,
  /// d(ij|kl) / dR_k
  CINT_ERI_IP2,
  /// (ij|k), three-center two-electron
  CINT_3C2E,
  /// d(ij|k) / dR_i
  CINT_3C2E_IP1,
  /// d(ij|k) / dR_k
  CINT_3C2E_IP2,
  /// (i|j), two-center two-electron
  CINT_2C2E,
  /// d(i|j) / dR_i
  CINT_2C2E_IP1,
  /// Number of kernels
  CINT_NKERNEL
} cint_kernel_t;

/// The libcint one-electron kernels ERKALE uses. The operator
/// derivatives ip and their ket-side counterparts differentiate the
/// first resp. the second shell of the pair.
typedef enum {
  CINT1E_OVLP,    ///< <i|j>
  CINT1E_KIN,     ///< <i|-1/2 nabla^2|j>
  CINT1E_RINV,    ///< <i|1/|r-C||j>
  CINT1E_OVLPIP,  ///< <i|nabla j>
  CINT1E_IPOVLP,  ///< <nabla i|j>
  CINT1E_IPKIN,   ///< bra derivative of the kinetic energy
  CINT1E_KINIP,   ///< ket derivative of the kinetic energy
  CINT1E_IPRINV,  ///< bra derivative of 1/|r-C|
  CINT1E_R,       ///< first moments around the common origin
  CINT1E_RR,      ///< second moments
  CINT1E_RRR,     ///< third moments
  CINT1E_RRRR,    ///< fourth moments
  CINT1E_NKERNEL
} cint_1e_kernel_t;

/// Number of operator components of a one-electron kernel
int cint_1e_ncomp(cint_1e_kernel_t kernel);

/**
 * \class CintEnv
 *
 * \brief libcint description of a set of shells
 *
 * Holds the atm / bas / env tables that describe a basis set (or the
 * concatenation of an orbital basis and an auxiliary basis) to libcint,
 * together with the integral optimizers, which cache the primitive pair
 * data. Building the tables and the optimizers once per basis instead of
 * once per shell quartet is what makes the integral evaluation fast: the
 * environment is read-only during integral evaluation, so a single
 * instance is shared by all threads.
 *
 * Shells are numbered as in the source basis set; in the two-basis
 * constructor the auxiliary shells follow the orbital shells, so
 * auxiliary shell i is shell get_Nsh_orb() + i.
 *
 * The contraction coefficients are stored in libcint's convention
 * (normalized primitives), so the integrals come out over normalized
 * basis functions: in spherical mode they are ERKALE's basis functions
 * as-is, and in cartesian mode they need the per-function scaling given
 * by get_cartnorm().
 *
 * The environment data array is copied by each integral worker, which
 * needs to write the range separation constant into it; the shell tables
 * and the optimizers are read-only and shared.
 */
class CintEnv {
  /// libcint atom table
  std::vector<int> cint_atm;
  /// libcint shell table
  std::vector<int> cint_bas;
  /// libcint data array: coordinates, exponents, contraction coefficients
  std::vector<double> cint_env;

  /// Integral optimizers, one per kernel. They cache the primitive pair
  /// data and are read-only during integral evaluation, so copies of an
  /// environment share them. Opaque here: libcint declares CINTOpt as an
  /// anonymous struct typedef, which cannot be forward declared, and
  /// cint.h is not fit to be included everywhere.
  struct OptSet {
    std::vector<void *> opts;
    ~OptSet();
  };
  std::shared_ptr<OptSet> opts;

  /// Number of shells in the orbital basis
  size_t Nsh_orb;
  /// Number of functions in each shell
  std::vector<size_t> shell_Nbf;
  /// Index of the first function of each shell
  std::vector<size_t> shell_first;
  /// Maximum number of functions in a shell
  size_t max_Nbf;

  /// The shells themselves
  std::vector<GaussianShell> shells;

  /// Are the integrals evaluated in the spherical harmonics basis?
  bool lm;
  /// Normalization of each function of each shell, relative to
  /// libcint's convention: ERKALE scales its basis functions with the
  /// per-function relnorm factors, which the Coulomb normalization used
  /// for auxiliary basis sets modifies after the fact, whereas libcint
  /// normalizes the shell internally. The factors are measured against
  /// ERKALE's own overlap integrals when the environment is built, so
  /// the environment must be built from a finalized basis set.
  std::vector<std::vector<double>> fnorm;
  /// Are all the normalization factors unity?
  bool unit_norm;

  /// Fill the tables from a list of shells
  void build(const std::vector<GaussianShell> & shells, size_t Nsh_orbital, bool build_opts);

 public:
  /// Dummy constructor
  CintEnv();
  /// Construct for a basis set
  explicit CintEnv(const BasisSet & basis, bool build_opts=true);
  /// Construct for an orbital basis followed by an auxiliary basis
  CintEnv(const BasisSet & basis, const BasisSet & aux, bool build_opts=true);
  /// Construct for an explicit list of shells
  explicit CintEnv(const std::vector<GaussianShell> & shells, bool build_opts=true);
  /// Construct for an explicit list of shells, of which the first
  /// Nsh_orbital are the orbital shells and the rest auxiliary: the
  /// auxiliary shells are then addressed as get_Nsh_orb() + i
  CintEnv(const std::vector<GaussianShell> & shells, size_t Nsh_orbital, bool build_opts=true);

  /// Is the environment initialized?
  bool is_filled() const;

  /// Number of shells
  size_t get_Nsh() const;
  /// The ish'th shell
  const GaussianShell & get_shell(size_t ish) const;
  /// Number of shells in the orbital basis (the rest are auxiliary)
  size_t get_Nsh_orb() const;
  /// Number of functions in shell ish
  size_t get_Nbf(size_t ish) const;
  /// Index of the first function of shell ish
  size_t get_first_ind(size_t ish) const;
  /// Maximum number of functions in a shell
  size_t get_max_Nbf() const;
  /// Are the integrals in the spherical harmonics basis?
  bool lm_in_use() const;

  /// Normalization factors of the functions of shell ish, relative to
  /// libcint's convention
  const std::vector<double> & get_fnorm(size_t ish) const;
  /// Are all the normalization factors unity, i.e. can the scaling be skipped?
  bool has_unit_norm() const;

  /// libcint tables. The tables are logically const during integral
  /// evaluation, but libcint's interface takes non-const pointers.
  int * get_atm() const;
  int get_natm() const;
  int * get_bas() const;
  int get_nbas() const;
  /// The data array, to be copied by the worker that evaluates integrals
  const std::vector<double> & get_env() const;

  /// Integral optimizer for the given kernel (a libcint CINTOpt *)
  void * get_opt(cint_kernel_t kernel) const;

};

#endif
