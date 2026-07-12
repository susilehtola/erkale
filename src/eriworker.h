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


#ifndef ERIWORKER_H
#define ERIWORKER_H

#include <memory>
#include <vector>

#include "basis.h"
#include "cintenv.h"

/**
 * Workers for electron repulsion integrals and their derivatives.
 *
 * The shells are addressed by their index in a CintEnv, which holds the
 * libcint description of the basis (and, where the caller built it that
 * way, of an auxiliary basis appended after it) together with the
 * integral optimizers. The environment is shared by all threads; the
 * workers are per thread.
 *
 * The integrals come out directly in ERKALE's basis and index order:
 * libcint runs its first shell fastest, so the workers evaluate the
 * reversed shell tuple -- which by the permutational symmetry of the
 * integrals is the same integral -- to get ERKALE's last-index-fastest
 * ordering, and they call the spherical kernels when the basis is
 * spherical, so no transformation step is needed.
 */
class IntegralWorker {
 protected:
  /// The libcint description of the shells (not owned)
  const CintEnv * envp;

  /// Private copy of the environment data array. The range separation
  /// constant is written into it, so it cannot be shared between threads.
  std::vector<double> env;
  /// libcint scratch memory
  std::vector<double> cache;
  /// Buffer for the second range separation component
  std::vector<double> srbuf;
  /// Buffer for the raw libcint output, before the layout remap
  std::vector<double> tmp;

  /// Integrals in ERKALE order
  std::vector<double> ints;

  /// Range separation: the integrals are
  /// rs_alpha * full Coulomb + rs_beta * erfc(rs_omega r12) / r12
  double rs_omega, rs_alpha, rs_beta;

  /// Integral optimizers for the attenuated kernels, one per kernel,
  /// built on demand and owned by the worker. The optimizers of the
  /// environment are only valid for the full-range kernels: libcint
  /// bakes the range separation constant into the primitive pair data,
  /// so the attenuated kernels need their own, and since the workers
  /// are per thread these can be built without a lock.
  std::vector<void *> sr_opts;

  /// The optimizer to use for the given kernel and range separation
  void * get_opt(cint_kernel_t kernel, double omega);

  /// Evaluate a kernel over the given shells, combining the range
  /// separation components. nsh is the number of shells the kernel takes
  /// (2, 3 or 4), ncomp the number of operator components and N the
  /// number of integrals per component; the results are left in out.
  void evaluate(cint_kernel_t kernel, int nsh, const int * shls, int ncomp, size_t N,
                std::vector<double> & out);

  /// Scale the integrals to ERKALE's normalization of the basis
  /// functions (a no-op when the environment has unit factors). The
  /// shells are listed in ERKALE index order, the slowest index first.
  void normalize(const size_t * shls, int nsh, int ncomp, std::vector<double> & out) const;

  /// Remap a three-center block from libcint's layout to ERKALE's
  void remap_3c(const std::vector<double> & in, int ncomp,
                size_t Ni, size_t Nj, size_t Nk, std::vector<double> & out) const;

 public:
  /// Constructor
  IntegralWorker(const CintEnv & env, double omega=0.0, double alpha=1.0, double beta=0.0);
  /// Destructor
  virtual ~IntegralWorker();
};

/// Worker for computing electron repulsion integrals
class ERIWorker: public IntegralWorker {
 public:
  /// Constructor
  ERIWorker(const CintEnv & env, double omega=0.0, double alpha=1.0, double beta=0.0);
  /// Destructor
  virtual ~ERIWorker();

  /// Compute the four-center integrals (ij|kl)
  void compute(size_t is, size_t js, size_t ks, size_t ls);
  /// Compute the three-center integrals (ij|k)
  void compute_3c(size_t is, size_t js, size_t ks);
  /// Compute the two-center integrals (i|j)
  void compute_2c(size_t is, size_t js);

  /// Compute the four-center integrals with the in-house Huzinaga
  /// routines: the independent reference used by integraltest
  void compute_debug(size_t is, size_t js, size_t ks, size_t ls);

  /// Get the integrals
  std::vector<double> get() const;
  /// Get a reference to the integrals
  std::vector<double> & rget();
  /// Get a pointer to the integrals
  const std::vector<double> * getp() const;
};

/// Worker for computing electron repulsion integral derivatives
class dERIWorker: public IntegralWorker {
  /// Derivatives with respect to the shell centers, in ERKALE order:
  /// 3*nsh components of N integrals each
  std::vector<double> dR;
  /// Scratch buffer for the layout remaps
  std::vector<double> scr;

  /// Number of integrals in the current shell tuple
  size_t N;
  /// Number of shells in the current tuple
  int nsh;

 public:
  /// Constructor
  dERIWorker(const CintEnv & env, double omega=0.0, double alpha=1.0, double beta=0.0);
  /// Destructor
  virtual ~dERIWorker();

  /// Compute the derivatives of the four-center integrals (ij|kl)
  void compute(size_t is, size_t js, size_t ks, size_t ls);
  /// Compute the derivatives of the three-center integrals (ij|k)
  void compute_3c(size_t is, size_t js, size_t ks);
  /// Compute the derivatives of the two-center integrals (i|j)
  void compute_2c(size_t is, size_t js);

  /// Get the derivatives with respect to index idx = 3*ish + ic, where
  /// ish numbers the shells in the order they were given to compute and
  /// ic is the cartesian component
  std::vector<double> get(int idx);
  /// Get a pointer to the derivatives with respect to index idx
  const std::vector<double> * getp(int idx);
};

/// Worker for computing one-electron integrals
class Int1eWorker: public IntegralWorker {
 public:
  /// Constructor
  explicit Int1eWorker(const CintEnv & env);
  /// Destructor
  virtual ~Int1eWorker();

  /// Compute the block of one-electron integrals over the shell pair.
  /// rinv_orig gives the origin of a 1/|r-C| operator and common_orig
  /// that of a multipole operator; both are ignored by the kernels that
  /// do not need them. The result has cint_1e_ncomp(kernel) components
  /// of Ni x Nj integrals, the component running slowest and the second
  /// shell fastest.
  void compute(cint_1e_kernel_t kernel, size_t is, size_t js,
               const double * rinv_orig=NULL, const double * common_orig=NULL);

  /// Get a pointer to the integrals
  const std::vector<double> * getp() const;
  /// Get the ic'th component as a matrix
  arma::mat get_mat(int ic, size_t is, size_t js) const;
};

/// Worker for computing short- and long-range electron repulsion integrals
class ERIWorker_srlr: public ERIWorker {
 public:
  /// Constructor
  ERIWorker_srlr(const CintEnv & env, double omega, double alpha, double beta);
  /// Destructor
  ~ERIWorker_srlr();
};

/// Worker for computing short- and long-range electron repulsion integral derivatives
class dERIWorker_srlr: public dERIWorker {
 public:
  /// Constructor
  dERIWorker_srlr(const CintEnv & env, double omega, double alpha, double beta);
  /// Destructor
  ~dERIWorker_srlr();
};

/// Allocate an ERIWorker matching the given range-separation parameters.
/// (omega, alpha, beta) == (0, 1, 0) is the plain-Coulomb default and
/// gets a vanilla ERIWorker; everything else gets ERIWorker_srlr.
inline std::unique_ptr<ERIWorker>
make_eri_worker(const CintEnv & env, double omega, double alpha, double beta) {
  if(omega == 0.0 && alpha == 1.0 && beta == 0.0)
    return std::unique_ptr<ERIWorker>(new ERIWorker(env));
  return std::unique_ptr<ERIWorker>(new ERIWorker_srlr(env, omega, alpha, beta));
}

inline std::unique_ptr<dERIWorker>
make_deri_worker(const CintEnv & env, double omega, double alpha, double beta) {
  if(omega == 0.0 && alpha == 1.0 && beta == 0.0)
    return std::unique_ptr<dERIWorker>(new dERIWorker(env));
  return std::unique_ptr<dERIWorker>(new dERIWorker_srlr(env, omega, alpha, beta));
}

#endif
