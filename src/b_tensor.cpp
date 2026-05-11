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

#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

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

DirectDFBlocks::DirectDFBlocks(size_t Nbf, size_t Naux,
                               std::vector<std::pair<size_t, size_t>> shellpairs,
                               std::vector<std::pair<size_t, size_t>> firsts,
                               std::vector<std::pair<size_t, size_t>> sizes,
                               std::vector<GaussianShell> orb_shells,
                               std::vector<GaussianShell> aux_shells,
                               GaussianShell dummy,
                               double omega, double alpha, double beta,
                               int max_am, int max_contr)
    : Nbf_(Nbf), Naux_(Naux),
      shellpairs_(std::move(shellpairs)),
      firsts_(std::move(firsts)),
      sizes_(std::move(sizes)),
      orb_shells_(std::move(orb_shells)),
      aux_shells_(std::move(aux_shells)),
      dummy_(std::move(dummy)),
      omega_(omega), alpha_(alpha), beta_(beta),
      max_am_(max_am), max_contr_(max_contr) {
  if(shellpairs_.size() != firsts_.size() || shellpairs_.size() != sizes_.size())
    throw std::logic_error("DirectDFBlocks: shellpair / first / size vectors disagree in length");

  // Worst-case Nmu*Nnu across shellpairs; per-thread scratch is sized
  // to this at construction so get_block(ip) never reallocs.
  max_NmuNnu_ = 0;
  for(size_t ip=0; ip<sizes_.size(); ip++)
    max_NmuNnu_ = std::max(max_NmuNnu_, sizes_[ip].first * sizes_[ip].second);

  // Allocate the per-thread ERIWorker cache + scratch up front.
  // omp_get_max_threads() outside a parallel region returns the
  // maximum the runtime would launch.
#ifdef _OPENMP
  const int nthr = omp_get_max_threads();
#else
  const int nthr = 1;
#endif
  eri_cache_.resize(nthr);
  scratch_.resize(nthr);
  for(int t=0; t<nthr; t++)
    scratch_[t].set_size(Naux_, max_NmuNnu_);
}

ERIWorker * DirectDFBlocks::thread_eri() const {
#ifdef _OPENMP
  const int tid = omp_get_thread_num();
#else
  const int tid = 0;
#endif
  // The cache slot is owned by this thread for the lifetime of the
  // outer parallel region; lazily fill it the first time we hit a
  // get_block on this thread.
  if(!eri_cache_[tid]) {
    if(omega_==0.0 && alpha_==1.0 && beta_==0.0)
      eri_cache_[tid].reset(new ERIWorker(max_am_, max_contr_));
    else
      eri_cache_[tid].reset(new ERIWorker_srlr(max_am_, max_contr_, omega_, alpha_, beta_));
  }
  return eri_cache_[tid].get();
}

DirectDFPerturbedBlocks::DirectDFPerturbedBlocks(
    size_t Nbf, size_t Naux, size_t Nnuc,
    std::vector<std::pair<size_t, size_t>> shellpairs,
    std::vector<std::pair<size_t, size_t>> firsts,
    std::vector<std::pair<size_t, size_t>> sizes,
    std::vector<GaussianShell> orb_shells,
    std::vector<GaussianShell> aux_shells,
    GaussianShell dummy,
    double omega, double alpha, double beta,
    int max_am, int max_contr)
    : Nbf_(Nbf), Naux_(Naux), Nnuc_(Nnuc),
      shellpairs_(std::move(shellpairs)),
      firsts_(std::move(firsts)),
      sizes_(std::move(sizes)),
      orb_shells_(std::move(orb_shells)),
      aux_shells_(std::move(aux_shells)),
      dummy_(std::move(dummy)),
      omega_(omega), alpha_(alpha), beta_(beta),
      max_am_(max_am), max_contr_(max_contr) {
  if(shellpairs_.size() != firsts_.size() || shellpairs_.size() != sizes_.size())
    throw std::logic_error("DirectDFPerturbedBlocks: descriptor vectors disagree");

  // Largest aux-shell function count and largest orbital-shellpair
  // function count; the per-thread scratch is sized to (max_Na_ x
  // max_NmuNnu_) so for_each_pert never reallocs.
  max_Na_ = 0;
  for(size_t ia=0; ia<aux_shells_.size(); ia++)
    max_Na_ = std::max(max_Na_, aux_shells_[ia].get_Nbf());
  max_NmuNnu_ = 0;
  for(size_t ip=0; ip<sizes_.size(); ip++)
    max_NmuNnu_ = std::max(max_NmuNnu_, sizes_[ip].first * sizes_[ip].second);

#ifdef _OPENMP
  const int nthr = omp_get_max_threads();
#else
  const int nthr = 1;
#endif
  deri_cache_.resize(nthr);
  scratch_.resize(nthr);
  for(int t=0; t<nthr; t++)
    scratch_[t].set_size(max_Na_, max_NmuNnu_);
}

dERIWorker * DirectDFPerturbedBlocks::thread_deri() const {
#ifdef _OPENMP
  const int tid = omp_get_thread_num();
#else
  const int tid = 0;
#endif
  if(!deri_cache_[tid]) {
    if(omega_==0.0 && alpha_==1.0 && beta_==0.0)
      deri_cache_[tid].reset(new dERIWorker(max_am_, max_contr_));
    else
      deri_cache_[tid].reset(new dERIWorker_srlr(max_am_, max_contr_, omega_, alpha_, beta_));
  }
  return deri_cache_[tid].get();
}

void DirectDFPerturbedBlocks::for_each_pert(
    size_t ip,
    const std::function<void(const Perturbation& pert,
                            size_t aux_first,
                            const arma::mat& sub_block)>& fn) const {
  const size_t imus = shellpairs_[ip].first;
  const size_t inus = shellpairs_[ip].second;
  const size_t Nmu = sizes_[ip].first;
  const size_t Nnu = sizes_[ip].second;
  const size_t inuc = orb_shells_[imus].get_center_ind();
  const size_t jnuc = orb_shells_[inus].get_center_ind();

  dERIWorker * deri = thread_deri();
#ifdef _OPENMP
  const int tid = omp_get_thread_num();
#else
  const int tid = 0;
#endif
  arma::mat & buf = scratch_[tid];

  // Iterate aux shells. For each aux shell, dERIWorker yields
  // derivatives w.r.t. all four centers (aux, dummy, mu, nu); we
  // skip the dummy's three components since they're identically
  // zero, leaving 9 non-trivial perturbation slots per aux shell
  // (anuc * xyz, inuc * xyz, jnuc * xyz). For each, hand the
  // consumer a view of the (Na_aux x Nmu*Nnu) sub-block.
  static const int comp_index[]  = {0, 1, 2, 6, 7, 8, 9, 10, 11};
  static const int comp_atom[]   = {0, 0, 0, 2, 2, 2, 3, 3, 3}; // 0=anuc, 2=inuc, 3=jnuc
  static const int comp_xyz[]    = {0, 1, 2, 0, 1, 2, 0, 1, 2};

  for(size_t ia=0; ia<aux_shells_.size(); ia++) {
    const size_t Na = aux_shells_[ia].get_Nbf();
    const size_t a0 = aux_shells_[ia].get_first_ind();
    const size_t anuc = aux_shells_[ia].get_center_ind();

    // If all three centres coincide the integral can't depend on
    // any nuclear coordinate (translational invariance leaves no
    // independent slot).
    if(inuc==jnuc && jnuc==anuc)
      continue;

    deri->compute(&aux_shells_[ia], &dummy_, &orb_shells_[imus], &orb_shells_[inus]);

    for(size_t k=0; k<sizeof(comp_index)/sizeof(comp_index[0]); k++) {
      const int ic = comp_index[k];
      // Map slot k to (atom, xyz).
      const size_t atom_slot = comp_atom[k];
      size_t atom_idx;
      if(atom_slot==0) atom_idx = anuc;
      else if(atom_slot==2) atom_idx = inuc;
      else atom_idx = jnuc;
      const int xyz = comp_xyz[k];

      const std::vector<double> * erip = deri->getp(ic);
      // Pack the (Na x Nmu*Nnu) slice for this perturbation
      // component into the first Na*Nmu*Nnu doubles of the
      // thread scratch, column-major with row-stride = Na (not
      // max_Na_!) so the arma::mat aux_mem view below sees a
      // contiguous Na-row matrix. dERIWorker indexing for one
      // component: (a * Nmu + imu) * Nnu + inu.
      double * buf_ptr = buf.memptr();
      for(size_t a=0; a<Na; a++)
        for(size_t imu=0; imu<Nmu; imu++)
          for(size_t inu=0; inu<Nnu; inu++) {
            const size_t j = inu*Nmu + imu;
            buf_ptr[j * Na + a] = (*erip)[(a*Nmu+imu)*Nnu+inu];
          }

      arma::mat sub(buf_ptr, Na, Nmu*Nnu, /*copy_aux_mem*/false, /*strict*/true);
      Perturbation pert = Perturbation::nuclear((int) atom_idx, xyz);
      fn(pert, a0, sub);
    }
  }
}

arma::mat DirectDFBlocks::get_block(size_t ip) const {
  const size_t imus = shellpairs_[ip].first;
  const size_t inus = shellpairs_[ip].second;
  const size_t Nmu = sizes_[ip].first;
  const size_t Nnu = sizes_[ip].second;

  ERIWorker * eri = thread_eri();
#ifdef _OPENMP
  const int tid = omp_get_thread_num();
#else
  const int tid = 0;
#endif

  // Write directly into the first Nmu*Nnu columns of the thread's
  // pre-sized scratch buffer; we'll hand the caller back a view of
  // exactly that slice. No allocation in the hot path.
  arma::mat & buf = scratch_[tid];
  // Zero only the slice we'll actually use; full buffer was zeroed
  // at construction.
  std::memset(buf.memptr(), 0, sizeof(double) * Naux_ * Nmu * Nnu);

  // Iterate over auxiliary shells: one libint call per (aux_shell)
  // gives Na * Nnu * Nmu entries straight into our slice in
  // (a, nu, mu) ordering. Matches the cached compute_a_munu layout
  // exactly so the J/K kernels see identical block contents either
  // way.
  double * buf_ptr = buf.memptr();
  for(size_t ia=0; ia<aux_shells_.size(); ia++) {
    const size_t Na = aux_shells_[ia].get_Nbf();
    const size_t a0 = aux_shells_[ia].get_first_ind();
    eri->compute(&aux_shells_[ia], &dummy_, &orb_shells_[inus], &orb_shells_[imus]);
    const std::vector<double> * erip = eri->getp();
    for(size_t a=0; a<Na; a++)
      for(size_t imunu=0; imunu<Nmu*Nnu; imunu++)
        // Column-major layout: buf(a0+a, imunu) = buf_ptr[imunu*Naux_ + a0+a]
        buf_ptr[imunu * Naux_ + a0 + a] = (*erip)[a*Nnu*Nmu + imunu];
  }
  // Return a non-owning view of the populated slice. Lifetime
  // contract: valid until the next get_block call on this thread.
  return arma::mat(buf_ptr, Naux_, Nmu * Nnu, /*copy_aux_mem*/false, /*strict*/true);
}
