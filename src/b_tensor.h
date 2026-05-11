/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_B_TENSOR
#define ERKALE_B_TENSOR

#include "global.h"
#include "basis.h"
#include "eriworker.h"

#include <functional>
#include <memory>

/**
 * Block-indexed view of a three-index tensor (mu nu | J), where mu nu
 * range over orbital pairs grouped by orbital shellpair, and J ranges
 * over an auxiliary dimension (DF aux basis or CD pivot orbital
 * pairs). Each block has shape (Naux, Nmu * Nnu).
 *
 * The interface is small and polymorphic so the same J/K kernels can
 * consume cached, direct-DF, or direct-CD-derived blocks. Block
 * access is by value but returning either an owning matrix (Direct*)
 * or a non-owning aux_mem view (Cached) -- the move into the
 * caller's `arma::mat` is a pointer swap, so the worst-case (Naux x 1)
 * ss block carries no copy overhead.
 *
 * The abstraction is the in-tree precursor to libcholesky's
 * lc_b_tensor_ops_t (cf. CLAUDE.md memory project_libcholesky_consumer).
 */
class BTensorBlocks {
public:
  virtual ~BTensorBlocks() = default;

  /// Number of orbital-shellpair blocks
  virtual size_t n_blocks() const = 0;
  /// Auxiliary dimension (Naux for DF, Naux_pivot for CD)
  virtual size_t naux() const = 0;
  /// Total orbital basis size
  virtual size_t nbf() const = 0;

  /// Shell pair (mu_shell, nu_shell) for the ip-th block
  virtual std::pair<size_t, size_t> shellpair(size_t ip) const = 0;
  /// First basis-function index on (mu_shell, nu_shell)
  virtual std::pair<size_t, size_t> shellpair_first(size_t ip) const = 0;
  /// Number of basis functions on (mu_shell, nu_shell)
  virtual std::pair<size_t, size_t> shellpair_size(size_t ip) const = 0;

  /// Get the ip-th block as (naux x nmu*nnu).
  ///
  /// In cached mode this returns a non-owning aux_mem view backed by
  /// the underlying flat storage; the view is valid only while *this
  /// is alive and unmodified. In direct mode this constructs a fresh
  /// owning matrix from libint+metric on each call. Either way the
  /// caller binds the result to a local arma::mat and uses it within
  /// the call site.
  virtual arma::mat get_block(size_t ip) const = 0;
};

/**
 * Cached, in-memory block storage. Owns a single flat backing
 * std::vector<double> plus per-block (offset, nmu, nnu) lookup;
 * get_block(ip) hands back an aux_mem view into that buffer.
 *
 * Built by the consumer (DensityFit::fill or the CD two-step path)
 * by reserving the total flat size up front and writing each block's
 * integrals directly into the slice it owns -- avoiding the
 * vector<arma::mat> allocation churn that the previous DF storage
 * had before the block-shellpair refactor.
 */
class CachedBlocks : public BTensorBlocks {
  /// Total orbital basis size
  size_t Nbf_;
  /// Auxiliary dimension
  size_t Naux_;
  /// Per-block shell indices (mu_shell, nu_shell)
  std::vector<std::pair<size_t, size_t>> shellpairs_;
  /// Per-block first-function indices (mu0, nu0)
  std::vector<std::pair<size_t, size_t>> firsts_;
  /// Per-block sizes (Nmu, Nnu)
  std::vector<std::pair<size_t, size_t>> sizes_;
  /// Per-block offset into storage_
  std::vector<size_t> offsets_;
  /// Flat backing storage, naux*sum(Nmu*Nnu) doubles
  std::vector<double> storage_;

public:
  CachedBlocks();
  CachedBlocks(size_t Nbf, size_t Naux,
               std::vector<std::pair<size_t, size_t>> shellpairs,
               std::vector<std::pair<size_t, size_t>> firsts,
               std::vector<std::pair<size_t, size_t>> sizes);
  ~CachedBlocks() override = default;

  size_t n_blocks() const override { return shellpairs_.size(); }
  size_t naux() const override { return Naux_; }
  size_t nbf() const override { return Nbf_; }
  std::pair<size_t, size_t> shellpair(size_t ip) const override { return shellpairs_[ip]; }
  std::pair<size_t, size_t> shellpair_first(size_t ip) const override { return firsts_[ip]; }
  std::pair<size_t, size_t> shellpair_size(size_t ip) const override { return sizes_[ip]; }
  arma::mat get_block(size_t ip) const override;

  /// Mutable access used by fill paths to write the ip-th block in
  /// place (same aux_mem view but with copy_aux_mem=false &
  /// strict=true so assignment goes to the backing store).
  arma::mat block_mut(size_t ip);

  /// Total backing-store size in doubles
  size_t storage_size() const { return storage_.size(); }
};

/**
 * Direct-mode block source for density fitting: each get_block(ip)
 * call computes the (alpha | mu nu) three-center integrals for the
 * ip-th orbital shellpair on demand via libint, returning an owning
 * (Naux x Nmu*Nnu) matrix. No precomputed (alpha | mu nu) storage --
 * the whole point of direct mode.
 *
 * Holds an internal per-thread ERIWorker cache so concurrent
 * get_block() calls from an outer `#pragma omp parallel` region
 * don't need to thread through their own worker. The cache is
 * lazily populated by the first call from each thread; cleanup is
 * automatic at destruction.
 */
class DirectDFBlocks : public BTensorBlocks {
  size_t Nbf_;
  size_t Naux_;
  std::vector<std::pair<size_t, size_t>> shellpairs_;
  std::vector<std::pair<size_t, size_t>> firsts_;
  std::vector<std::pair<size_t, size_t>> sizes_;

  /// Orbital and auxiliary shells, owned copies (cheap).
  std::vector<GaussianShell> orb_shells_;
  std::vector<GaussianShell> aux_shells_;
  GaussianShell dummy_;
  /// Range separation
  double omega_, alpha_, beta_;
  /// libint worker dimensions
  int max_am_;
  int max_contr_;

  /// Per-thread ERIWorker cache (lazy). Mutable because get_block
  /// is logically const but materialises a thread-local worker on
  /// first use.
  mutable std::vector<std::unique_ptr<ERIWorker>> eri_cache_;
  /// Per-thread scratch buffer of shape (Naux, max_NmuNnu) where
  /// max_NmuNnu is the worst-case across shellpairs. get_block(ip)
  /// fills the first Nmu*Nnu columns and returns a non-owning view
  /// of that slice. Avoids a fresh per-call arma::mat allocation in
  /// the SCF hot path.
  mutable std::vector<arma::mat> scratch_;
  /// Worst-case Nmu*Nnu across shellpairs (column dim of scratch_).
  size_t max_NmuNnu_;

public:
  DirectDFBlocks(size_t Nbf, size_t Naux,
                 std::vector<std::pair<size_t, size_t>> shellpairs,
                 std::vector<std::pair<size_t, size_t>> firsts,
                 std::vector<std::pair<size_t, size_t>> sizes,
                 std::vector<GaussianShell> orb_shells,
                 std::vector<GaussianShell> aux_shells,
                 GaussianShell dummy,
                 double omega, double alpha, double beta,
                 int max_am, int max_contr);
  ~DirectDFBlocks() override = default;

  size_t n_blocks() const override { return shellpairs_.size(); }
  size_t naux() const override { return Naux_; }
  size_t nbf() const override { return Nbf_; }
  std::pair<size_t, size_t> shellpair(size_t ip) const override { return shellpairs_[ip]; }
  std::pair<size_t, size_t> shellpair_first(size_t ip) const override { return firsts_[ip]; }
  std::pair<size_t, size_t> shellpair_size(size_t ip) const override { return sizes_[ip]; }
  arma::mat get_block(size_t ip) const override;

 private:
  ERIWorker * thread_eri() const;
};

// ===========================================================================
// Perturbation / derivative-integral abstraction
//
// Layout chosen to match libcholesky's lc_perturbation_set (cf. memory
// project_libcholesky_perturbation_compat): each perturbation carries a
// `kind` tag and two integer parameters, and a derivative request is an
// ordered list of such perturbations. v1 only implements first-order
// LC_PERT_NUCLEAR_CARTESIAN; the API accepts arbitrary multi-indices and
// other kinds, but unsupported requests throw.
// ===========================================================================

struct Perturbation {
  /// Match libcholesky lc_perturbation_kind values bit-for-bit so the
  /// eventual bridge to libcholesky is a memcpy on equivalent structs.
  enum Kind : int {
    NuclearCartesian = 0, // p1 = atom index, p2 = component (0..2)
    ElectricField    = 1, // p1 unused,      p2 = component (0..2)
    MagneticUniform  = 2, // reserved v1.x
    NuclearMagnetic  = 3, // reserved v1.x
    BasisParameter   = 4  // p1 = basis selector, p2 = parameter index
  };
  int kind;
  int p1;
  int p2;

  static Perturbation nuclear(int atom, int xyz) {
    return Perturbation{NuclearCartesian, atom, xyz};
  }
};

/// Ordered list of perturbations of length n; derivative order = n.
using PerturbationSet = std::vector<Perturbation>;

/**
 * Streaming view of derivative integrals (mu nu | aux) per orbital
 * shellpair. The interface mirrors libcholesky's perturbed B-tensor
 * model: each call to `for_each_pert(ip, fn)` invokes `fn` once for
 * each (perturbation_set, aux_first, sub_block) tuple that has a
 * non-zero contribution for shellpair `ip`. The consumer accumulates
 * forces (or higher-order properties) by contracting `sub_block`
 * with density / coefficient slices.
 *
 * For first-order nuclear-Cartesian: `fn` fires once per
 * (touching_atom, xyz, contributing_aux_shell) -- that's typically
 * O(3 * N_atoms_with_aux) callbacks per shellpair, each with a
 * (Na_aux_shell x Nmu*Nnu) sub_block view of the derivative
 * integrals. Lifetime contract on `sub_block` is identical to
 * BTensorBlocks::get_block: valid until the next call on the same
 * thread on the same object.
 *
 * Derivative-order / kind support is implementation-defined; v1
 * implementations support first-order NuclearCartesian only and
 * throw on other requests.
 */
class PerturbedBTensorBlocks {
public:
  virtual ~PerturbedBTensorBlocks() = default;

  virtual size_t n_blocks() const = 0;
  virtual size_t naux() const = 0;
  virtual size_t nbf() const = 0;
  virtual size_t nnuc() const = 0;

  virtual std::pair<size_t, size_t> shellpair(size_t ip) const = 0;
  virtual std::pair<size_t, size_t> shellpair_first(size_t ip) const = 0;
  virtual std::pair<size_t, size_t> shellpair_size(size_t ip) const = 0;

  /// Stream non-zero derivative contributions for shellpair `ip`. The
  /// `fn` callback is invoked once per (Perturbation, aux_first,
  /// sub_block) tuple. Sub-block shape: (aux_count, Nmu * Nnu) where
  /// aux_count is the number of contributing aux functions in this
  /// callback's slice (e.g. one aux shell's worth).
  ///
  /// Currently only order-1 NuclearCartesian is supported.
  virtual void for_each_pert(
      size_t ip,
      const std::function<void(const Perturbation& pert,
                              size_t aux_first,
                              const arma::mat& sub_block)>& fn) const = 0;
};

/**
 * Direct-mode subclass for density-fitting derivative integrals.
 * Computes d(mu nu | aux_shell)/dR on demand via dERIWorker for each
 * orbital shellpair and aux shell, and streams the resulting
 * sub-blocks to the consumer through `for_each_pert`.
 *
 * Holds a per-thread dERIWorker cache + per-thread scratch buffer
 * the same way DirectDFBlocks does for value integrals.
 */
class DirectDFPerturbedBlocks : public PerturbedBTensorBlocks {
  size_t Nbf_;
  size_t Naux_;
  size_t Nnuc_;
  std::vector<std::pair<size_t, size_t>> shellpairs_;
  std::vector<std::pair<size_t, size_t>> firsts_;
  std::vector<std::pair<size_t, size_t>> sizes_;

  std::vector<GaussianShell> orb_shells_;
  std::vector<GaussianShell> aux_shells_;
  GaussianShell dummy_;
  double omega_, alpha_, beta_;
  int max_am_;
  int max_contr_;

  /// Per-thread dERIWorker cache (lazy).
  mutable std::vector<std::unique_ptr<dERIWorker>> deri_cache_;
  /// Per-thread scratch for a single (Na_aux x Nmu*Nnu) sub-block.
  /// Sized to (max_Na_aux x max_NmuNnu) at construction.
  mutable std::vector<arma::mat> scratch_;
  size_t max_Na_;
  size_t max_NmuNnu_;

 public:
  DirectDFPerturbedBlocks(size_t Nbf, size_t Naux, size_t Nnuc,
                          std::vector<std::pair<size_t, size_t>> shellpairs,
                          std::vector<std::pair<size_t, size_t>> firsts,
                          std::vector<std::pair<size_t, size_t>> sizes,
                          std::vector<GaussianShell> orb_shells,
                          std::vector<GaussianShell> aux_shells,
                          GaussianShell dummy,
                          double omega, double alpha, double beta,
                          int max_am, int max_contr);
  ~DirectDFPerturbedBlocks() override = default;

  size_t n_blocks() const override { return shellpairs_.size(); }
  size_t naux() const override { return Naux_; }
  size_t nbf() const override { return Nbf_; }
  size_t nnuc() const override { return Nnuc_; }
  std::pair<size_t, size_t> shellpair(size_t ip) const override { return shellpairs_[ip]; }
  std::pair<size_t, size_t> shellpair_first(size_t ip) const override { return firsts_[ip]; }
  std::pair<size_t, size_t> shellpair_size(size_t ip) const override { return sizes_[ip]; }

  void for_each_pert(
      size_t ip,
      const std::function<void(const Perturbation& pert,
                              size_t aux_first,
                              const arma::mat& sub_block)>& fn) const override;

 private:
  dERIWorker * thread_deri() const;
};

#endif
