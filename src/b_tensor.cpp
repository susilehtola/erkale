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

#include "b_tensor.h"

#include <stdexcept>

CachedBlocks::CachedBlocks() : Nbf_(0), Naux_(0) {}

CachedBlocks::CachedBlocks(size_t Nbf, size_t Naux,
                           std::vector<std::pair<size_t, size_t>> shellpairs,
                           std::vector<std::pair<size_t, size_t>> firsts,
                           std::vector<std::pair<size_t, size_t>> sizes)
    : Nbf_(Nbf), Naux_(Naux),
      shellpairs_(std::move(shellpairs)),
      firsts_(std::move(firsts)),
      sizes_(std::move(sizes)) {
  if(shellpairs_.size() != firsts_.size() || shellpairs_.size() != sizes_.size())
    throw std::logic_error("CachedBlocks: shellpair / first / size vectors disagree in length");

  // Lay out per-block offsets into a single flat buffer.
  offsets_.resize(shellpairs_.size());
  size_t off = 0;
  for(size_t ip = 0; ip < shellpairs_.size(); ip++) {
    offsets_[ip] = off;
    off += Naux_ * sizes_[ip].first * sizes_[ip].second;
  }
  storage_.assign(off, 0.0);
}

arma::mat CachedBlocks::get_block(size_t ip) const {
  const size_t Nmu = sizes_[ip].first;
  const size_t Nnu = sizes_[ip].second;
  // arma::Mat constructor takes a non-const eT*; we cast away const
  // because the view is consumed read-only by the caller -- the
  // const correctness guarantee is on `*this`, not on the returned
  // view's modifiability (which the caller is trusted not to abuse).
  double * mem = const_cast<double *>(storage_.data() + offsets_[ip]);
  return arma::mat(mem, Naux_, Nmu * Nnu, /*copy_aux_mem*/false, /*strict*/true);
}

arma::mat CachedBlocks::block_mut(size_t ip) {
  const size_t Nmu = sizes_[ip].first;
  const size_t Nnu = sizes_[ip].second;
  return arma::mat(storage_.data() + offsets_[ip], Naux_, Nmu * Nnu, /*copy_aux_mem*/false, /*strict*/true);
}
