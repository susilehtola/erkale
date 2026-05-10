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

#endif
