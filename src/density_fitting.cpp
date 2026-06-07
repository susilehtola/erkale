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

#include "density_fitting.h"
#include "checkpoint.h"
#include "eriworker.h"
#include "linalg.h"
#include "mathf.h"
#include "scf.h"
#include "stringutil.h"
#include "timer.h"
#include <cstdio>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// \delta parameter in Eichkorn et al
#define DELTA 1e-9


DensityFit::DensityFit() {
  omega=0.0;
  alpha=1.0;
  beta=0.0;
  cholesky_mode=false;
  cd_pivot_sentinel=0;
}

DensityFit::~DensityFit() {
}

void DensityFit::set_range_separation(double w, double a, double b) {
  omega=w;
  alpha=a;
  beta=b;
}

void DensityFit::get_range_separation(double & w, double & a, double & b) const {
  w=omega;
  a=alpha;
  b=beta;
}

void DensityFit::init_orbital_state(const BasisSet & orbbas, bool dir) {
  Nbf  = orbbas.get_Nbf();
  Nnuc = orbbas.get_Nnuc();
  direct = dir;
  orbshells   = orbbas.get_shells();
  dummy       = dummyshell();
  maxorbam    = orbbas.get_max_am();
  maxorbcontr = orbbas.get_max_Ncontr();
}

void DensityFit::build_shellpair_descriptor(
    std::vector<std::pair<size_t, size_t>> & sp_pairs,
    std::vector<std::pair<size_t, size_t>> & sp_firsts,
    std::vector<std::pair<size_t, size_t>> & sp_sizes) const {
  sp_pairs.resize(orbpairs.size());
  sp_firsts.resize(orbpairs.size());
  sp_sizes.resize(orbpairs.size());
  for(size_t ip=0; ip<orbpairs.size(); ip++) {
    const size_t imus = orbpairs[ip].is;
    const size_t inus = orbpairs[ip].js;
    sp_pairs[ip]  = std::make_pair(imus, inus);
    sp_firsts[ip] = std::make_pair(orbshells[imus].get_first_ind(), orbshells[inus].get_first_ind());
    sp_sizes[ip]  = std::make_pair(orbshells[imus].get_Nbf(), orbshells[inus].get_Nbf());
  }
}

size_t DensityFit::fill(const BasisSet & orbbas, const BasisSet & auxbas, bool dir, double erithr, double linthr, double cholthr) {
  cholesky_mode=false;
  cd_pivot_index.reset();
  cd_pivot_shellpairs_vec.clear();
  pivot_shellpairs.clear();

  init_orbital_state(orbbas, dir);
  Naux = auxbas.get_Nbf();
  orbpairs = orbbas.compute_screening(erithr).shpairs;
  auxshells = auxbas.get_shells();
  maxauxam = auxbas.get_max_am();
  maxauxcontr = auxbas.get_max_Ncontr();
  maxam = std::max(maxorbam, maxauxam);
  maxcontr = (int) std::max(maxorbcontr, maxauxcontr);

  // First, compute the two-center integrals
  ab.zeros(Naux,Naux);

  // Get list of unique auxiliary shell pairs
  std::vector<shellpair_t> auxpairs=auxbas.get_unique_shellpairs();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri = make_eri_worker(maxam, maxcontr, omega, alpha, beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<auxpairs.size();ip++) {
      // The shells in question are
      size_t is=auxpairs[ip].is;
      size_t js=auxpairs[ip].js;

      // Compute (a|b)
      eri->compute(&auxshells[is],&dummy,&auxshells[js],&dummy);
      erip=eri->getp();

      // Store integrals
      size_t Ni=auxshells[is].get_Nbf();
      size_t Nj=auxshells[js].get_Nbf();
      for(size_t ii=0;ii<Ni;ii++) {
	size_t ai=auxshells[is].get_first_ind()+ii;
	for(size_t jj=0;jj<Nj;jj++) {
	  size_t aj=auxshells[js].get_first_ind()+jj;

	  ab(ai,aj)=(*erip)[ii*Nj+jj];
	  ab(aj,ai)=(*erip)[ii*Nj+jj];
	}
      }
    }
  }

  ab_invh = PartialCholeskyOrth(ab, cholthr, linthr);
  ab_inv = ab_invh * ab_invh.t();

  // Build the per-shellpair block descriptor; the same descriptor
  // feeds either the cached or direct BTensorBlocks subclass below.
  std::vector<std::pair<size_t, size_t>> sp_pairs, sp_firsts, sp_sizes;
  build_shellpair_descriptor(sp_pairs, sp_firsts, sp_sizes);

  if(!direct) {
    // Compute and store the (alpha | mu nu) integrals in a flat
    // CachedBlocks backing store. Each block stores raw integrals;
    // metric application happens in the J/K kernels via ab_invh /
    // ab_inv as before.
    auto cached = std::make_shared<CachedBlocks>(Nbf, Naux, sp_pairs, sp_firsts, sp_sizes);
    printf("(A|uv) integrals require %.3f GB\n", cached->storage_size()*8*1e-9);
    fflush(stdout);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      auto eri = make_eri_worker(maxam, maxcontr, omega, alpha, beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	// Write straight into the CachedBlocks-owned slot for this ip.
	arma::mat slot = cached->block_mut(ip);
	(void) compute_a_munu(eri.get(), ip, slot.memptr());
      }
    }
    blocks = cached;
  } else {
    // Direct mode: build a DirectDFBlocks that computes (alpha|mu nu)
    // on demand. The J/K kernels see the same blocks->get_block(ip)
    // interface as the cached path.
    blocks = std::make_shared<DirectDFBlocks>(
        Nbf, Naux, std::move(sp_pairs), std::move(sp_firsts), std::move(sp_sizes),
        orbshells, auxshells, dummy,
        omega, alpha, beta, maxam, maxcontr);
  }

  return orbpairs.size();
}

// Two-step CD pivot selection (phases A-C: diagonal, pair
// enumeration, pivoted selection). Pivots-only: it returns the pivot
// list, the product map and the pivot shellpairs. The (mu nu | piv)
// integrals are not returned -- both the cached and the direct block
// builders recompute them per block, so nothing of size Nprod x Nsel
// is held here. Returns the number of pivots selected.
//
// This is the engine that powers both fill_cholesky and
// find_cholesky_pivots; the only difference between CD and DF in
// the rest of DensityFit is which aux selection drives the metric +
// three-center storage.
size_t DensityFit::select_two_step_pivots(const BasisSet & basis,
                                          double cholesky_tol,
                                          double shell_reuse_thr,
                                          double shell_screen_tol,
                                          bool verbose,
                                          arma::uvec & pi,
                                          arma::umat & invmap,
                                          std::set<std::pair<size_t, size_t>> & piv_shellpairs) const {
  // prodmap ((mu,nu) -> product index) is internal scratch for the
  // pivoted selection; the callers never need it.
  arma::umat prodmap;
  if(cholesky_tol < shell_screen_tol) {
    fprintf(stderr,"Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
    printf("Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
    fflush(stdout);
  }

  ScreeningData scr=basis.compute_screening(shell_screen_tol,omega,alpha,beta,verbose);
  const arma::mat & Q = scr.Q;
  const arma::mat & M_screen = scr.M;
  const std::vector<eripair_t> & shpairs = scr.shpairs;

  const size_t Nbf_local=basis.get_Nbf();
  const std::vector<GaussianShell> & shells=basis.get_shells_ref();

  Timer t, ttot;
  double t_int=0.0, t_chol=0.0;

  // Phase A: diagonal (mu nu | mu nu)
  arma::vec d(Nbf_local*Nbf_local, arma::fill::zeros);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri = make_eri_worker(basis.get_max_am(), basis.get_max_Ncontr(), omega, alpha, beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t ip=0;ip<shpairs.size();ip++) {
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;
      double QQ=Q(is,js)*Q(is,js);
      if(QQ<shell_screen_tol) continue;
      eri->compute(&shells[is],&shells[js],&shells[is],&shells[js]);
      erip=eri->getp();
      size_t Ni(shells[is].get_Nbf());
      size_t Nj(shells[js].get_Nbf());
      size_t i0(shells[is].get_first_ind());
      size_t j0(shells[js].get_first_ind());
      for(size_t ii=0;ii<Ni;ii++)
        for(size_t jj=0;jj<Nj;jj++) {
          size_t i=i0+ii;
          size_t j=j0+jj;
          d(i*Nbf_local+j)=(*erip)[((ii*Nj+jj)*Ni+ii)*Nj+jj];
          d(j*Nbf_local+i)=d(i*Nbf_local+j);
        }
    }
  }
  t_int+=t.get();

  // Phase B: enumerate the significant orbital pairs into invmap
  // (compact product index -> (i,j)) and prodmap ((i,j) -> compact
  // index). The flat index i*Nbf+j is recovered from invmap when the
  // diagonal is gathered below, so no separate prodidx is kept.
  {
    prodmap.ones(Nbf_local,Nbf_local);
    prodmap*=-1;
    size_t iprod=0;
    invmap.zeros(2,Nbf_local*Nbf_local);
    for(size_t ip=0;ip<shpairs.size();ip++) {
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;
      size_t Ni(shells[is].get_Nbf());
      size_t Nj(shells[js].get_Nbf());
      size_t i0(shells[is].get_first_ind());
      size_t j0(shells[js].get_first_ind());
      if(is==js) {
        for(size_t i=i0;i<i0+Ni;i++) {
          for(size_t j=j0;j<i;j++) {
            invmap(0,iprod)=i;
            invmap(1,iprod)=j;
            prodmap(i,j)=iprod;
            prodmap(j,i)=iprod;
            iprod++;
          }
          invmap(0,iprod)=i;
          invmap(1,iprod)=i;
          prodmap(i,i)=iprod;
          iprod++;
        }
      } else {
        for(size_t i=i0;i<i0+Ni;i++)
          for(size_t j=j0;j<j0+Nj;j++) {
            invmap(0,iprod)=i;
            invmap(1,iprod)=j;
            prodmap(i,j)=iprod;
            prodmap(j,i)=iprod;
            iprod++;
          }
      }
    }
    // Trim invmap to the significant products. (Shed the whole unused
    // tail, including the n_cols-1 column the old < n_cols-1 guard
    // skipped, which would otherwise leak into the cached B_raw
    // scatter that iterates over invmap.n_cols.)
    if(iprod < invmap.n_cols)
      invmap.shed_cols(iprod, invmap.n_cols-1);
  }
  const size_t Nprod = invmap.n_cols;

  if(verbose) {
    printf("Two-step CD: screening reduced dofs by factor %.2f.\n",d.n_elem*1.0/Nprod);
    fflush(stdout);
  }

  // Restrict the diagonal to the significant products: the flat index
  // of compact product k is invmap(0,k)*Nbf + invmap(1,k).
  d = d(arma::uvec(invmap.row(0).t()*Nbf_local + invmap.row(1).t()));
  // Phase C: pivoted selection of the Cholesky basis (paper step I).
  // Only the pivots are needed, so a product whose residual diagonal
  // (mu nu|mu nu) drops below tau can never be selected -- it is removed
  // from the working set. The Cholesky-vector scratch is held compactly,
  // following the active set rather than the full product list:
  //   Lact(r, c) = vector r's component at the product of active col c
  //   actprod[c] = product index of active column c
  //   colof[p]   = active column of product p, or SENT
  // Lact gains a row per accepted pivot and loses a column whenever a
  // product is selected or shed (swap-with-last, with a periodic shrink
  // of the allocation), so its footprint rises and then falls with the
  // shrinking active set -- the memory curve of Folkestad/Kjonstad/Koch
  // (JCP 150, 194112) Fig. 3.
  const arma::uword SENT = std::numeric_limits<arma::uword>::max();
  const size_t pivot_grow_chunk = 100;
  arma::uvec colof(Nprod);
  colof.fill(SENT);
  std::vector<arma::uword> actprod;
  actprod.reserve(Nprod);
  for(arma::uword p=0; p<Nprod; p++)
    if(d(p) >= cholesky_tol) {
      colof(p) = actprod.size();
      actprod.push_back(p);
    }

  arma::mat Lact(pivot_grow_chunk, std::max<size_t>(actprod.size(),1), arma::fill::zeros);
  size_t nvec = 0;
  std::vector<arma::uword> sel;          // selected pivot products, in order
  sel.reserve(actprod.size());

  // Remove active column c: move the last active column into slot c
  // (preserving its nvec built components) and drop the active count.
  auto drop_col = [&](arma::uword c) {
    const arma::uword last = actprod.size()-1;
    const arma::uword removed = actprod[c];
    if(c != last) {
      if(nvec>0)
        Lact.submat(0,c,nvec-1,c) = Lact.submat(0,last,nvec-1,last);
      actprod[c] = actprod[last];
      colof(actprod[c]) = c;
    }
    colof(removed) = SENT;
    actprod.pop_back();
  };

  while(!actprod.empty()) {
    // Paper step 4: largest residual diagonal among the active
    // candidates defines the next pivot's shell pair.
    arma::uword bestc=0;
    double bestval=d(actprod[0]);
    for(arma::uword c=1;c<actprod.size();c++) {
      const double v=d(actprod[c]);
      if(v>bestval) { bestval=v; bestc=c; }
    }
    if(bestval<=cholesky_tol) break;
    arma::uword pim=actprod[bestc];

    size_t max_k=invmap(0,pim);
    size_t max_l=invmap(1,pim);
    size_t max_ks=basis.find_shell_ind(max_k);
    size_t max_ls=basis.find_shell_ind(max_l);
    size_t max_Nk=basis.get_Nbf(max_ks);
    size_t max_Nl=basis.get_Nbf(max_ls);
    size_t max_k0=basis.get_first_ind(max_ks);
    size_t max_l0=basis.get_first_ind(max_ls);

    arma::mat A(d.n_elem,max_Nk*max_Nl, arma::fill::zeros);
    t.set();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      auto eri = make_eri_worker(basis.get_max_am(), basis.get_max_Ncontr(), omega, alpha, beta);
      const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
      for(size_t ipair=0;ipair<shpairs.size();ipair++) {
        size_t is=shpairs[ipair].is;
        size_t js=shpairs[ipair].js;
        double QQ=Q(is,js)*Q(max_ks,max_ls);
        if(QQ<shell_screen_tol) continue;
        double MM1=M_screen(is,max_ks)*M_screen(js,max_ls);
        if(MM1<shell_screen_tol) continue;
        double MM2=M_screen(is,max_ls)*M_screen(js,max_ks);
        if(MM2<shell_screen_tol) continue;

        eri->compute(&shells[is],&shells[js],&shells[max_ks],&shells[max_ls]);
        erip=eri->getp();
        size_t Ni(shells[is].get_Nbf());
        size_t Nj(shells[js].get_Nbf());
        size_t i0(shells[is].get_first_ind());
        size_t j0(shells[js].get_first_ind());
        for(size_t ii=0;ii<Ni;ii++)
          for(size_t jj=0;jj<Nj;jj++) {
            size_t i=i0+ii;
            size_t j=j0+jj;
            if(prodmap(i,j)>Nbf_local*Nbf_local) continue;
            for(size_t kk=0;kk<max_Nk;kk++)
              for(size_t ll=0;ll<max_Nl;ll++) {
                A(prodmap(i,j),kk*max_Nl+ll)=(*erip)[((ii*Nj+jj)*max_Nk+kk)*max_Nl+ll];
              }
          }
      }
    }
    t_int+=t.get();
    t.set();

    // Inner sweep (paper step 6): qualify the significant diagonals in
    // this shell block -- the integrals are already in A -- build their
    // Cholesky vectors over the (contiguous) active columns, and shed
    // products that drop below tau. The shell pair is reused while its
    // best residual stays within shell_reuse_thr (the span factor sigma)
    // of the active maximum.
    while(!actprod.empty()) {
      double errmax=0.0;
      for(arma::uword c=0;c<actprod.size();c++)
        errmax=std::max(errmax, d(actprod[c]));
      double blockerr=0;
      arma::uword blockc=SENT;
      size_t Aind=0;
      for(size_t kk=0;kk<max_Nk;kk++)
        for(size_t ll=0;ll<max_Nl;ll++) {
          const size_t ind=prodmap(kk+max_k0,ll+max_l0);
          if(ind>Nbf_local*Nbf_local) continue;     // not a significant product
          const arma::uword c=colof(ind);
          if(c==SENT) continue;                     // already a pivot, or shed
          if(d(ind)>blockerr) {
            Aind=kk*max_Nl+ll;
            blockc=c;
            blockerr=d(ind);
          }
        }
      if(blockerr==0.0 || blockerr<shell_reuse_thr*errmax)
        break;
      const arma::uword piv=actprod[blockc];

      if(nvec>=Lact.n_rows)
        Lact.resize(Lact.n_rows+pivot_grow_chunk, Lact.n_cols);

      const double inv=1.0/sqrt(d(piv));
      const arma::uword nact=actprod.size();
      // New Cholesky vector nvec, over every active column c:
      //   L_nvec(c) = ( (c|piv) - sum_{r<nvec} L_r(c) L_r(piv) ) / sqrt(d_piv)
      // then the residual diagonal d(c) -= L_nvec(c)^2. The pivot's own
      // column gets L_nvec(piv) = sqrt(d_piv), d(piv) -> 0; it is dropped
      // immediately below. The loop is OpenMP-parallel and the dot runs
      // on the contiguous first nvec entries of each (col-major) column.
      if(nvec==0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(arma::uword c=0;c<nact;c++) {
          const arma::uword p=actprod[c];
          Lact(0,c)=A(p,Aind)*inv;
          d(p)-=Lact(0,c)*Lact(0,c);
        }
      } else {
        const arma::vec bcol(Lact.submat(0,blockc,nvec-1,blockc));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(arma::uword c=0;c<nact;c++) {
          const arma::uword p=actprod[c];
          const double tdot=arma::dot(Lact.submat(0,c,nvec-1,c), bcol);
          Lact(nvec,c)=(A(p,Aind)-tdot)*inv;
          d(p)-=Lact(nvec,c)*Lact(nvec,c);
        }
      }
      nvec++;
      sel.push_back(piv);

      // The pivot's column is a finished vector (never read again); drop
      // it, then shed any product whose residual fell below tau (their
      // diagonals decrease monotonically, so they can never be pivots).
      drop_col(blockc);
      for(arma::uword c=0; c<actprod.size(); ) {
        if(d(actprod[c])<cholesky_tol) drop_col(c);
        else c++;
      }

      // Reclaim allocation once the stored width runs well ahead of the
      // active set (active columns are compacted into [0, nact)).
      if(Lact.n_cols > actprod.size() + actprod.size()/4 + pivot_grow_chunk)
        Lact.resize(Lact.n_rows, std::max<size_t>(actprod.size(),1));
    }
    t_chol+=t.get();
  }

  const size_t Nselected=nvec;
  pi = arma::uvec(sel);
  Lact.reset();

  // Form pivot shellpairs (small inline body -- no separate helper).
  piv_shellpairs.clear();
  for(size_t i=0;i<pi.n_elem;i++) {
    size_t is=basis.find_shell_ind(invmap(0,pi(i)));
    size_t js=basis.find_shell_ind(invmap(1,pi(i)));
    if(js<is) std::swap(is,js);
    piv_shellpairs.insert({is,js});
  }

  if(verbose) {
    printf("Two-step CD selected %i pivot orbital pairs after pivoting (%s).\n",
           (int) Nselected, ttot.elapsed().c_str());
    if(t_int+t_chol > 0.0)
      printf("Pivot selection time use: integrals %3.1f %%, linear algebra %3.1f %%.\n",
             100*t_int/(t_int+t_chol),100*t_chol/(t_int+t_chol));
    fflush(stdout);
  }

  return Nselected;
}

std::set<std::pair<size_t, size_t>> DensityFit::find_cholesky_pivots(const BasisSet & basis,
                                                                     double cholesky_tol,
                                                                     double shell_reuse_thr,
                                                                     double shell_screen_tol,
                                                                     bool verbose) const {
  // Pivots-only: run phases A-C with no (mu nu | piv) column save
  // and no metric / B construction. Used by basislibrary for atom-CD
  // aux-basis construction. Honors this object's range separation.
  arma::uvec pi;
  arma::umat invmap;
  std::set<std::pair<size_t, size_t>> piv_shellpairs;
  select_two_step_pivots(basis, cholesky_tol, shell_reuse_thr, shell_screen_tol,
                         verbose, pi, invmap, piv_shellpairs);
  return piv_shellpairs;
}

size_t DensityFit::fill_cholesky(const BasisSet & basis,
                                 bool dir,
                                 double cholesky_tol,
                                 double shell_reuse_thr,
                                 double shell_screen_tol,
                                 double fit_cholesky_thr,
                                 bool verbose) {
  // Two-step CD = density fitting with an aux basis chosen by
  // pivoted Cholesky on orbital products. DF and CD share the
  // downstream J/K/forceJ kernels; only the aux selection differs.
  cholesky_mode = true;
  init_orbital_state(basis, dir);
  auxshells.clear();
  maxauxam    = 0;
  maxauxcontr = 0;
  maxam       = maxorbam;
  maxcontr    = maxorbcontr;

  Timer ttot;

  // Phases A-C: pivot selection only. The (mu nu | piv) integrals are
  // not materialised here -- both the cached and direct block builders
  // recompute them per block (DirectCDBlocks::get_block), so no
  // Nprod x Nselected tensor is held during fill.
  arma::uvec pi;
  arma::umat invmap;
  const size_t Nselected = select_two_step_pivots(basis, cholesky_tol, shell_reuse_thr, shell_screen_tol,
                                                  verbose, pi, invmap, pivot_shellpairs);
  Naux = Nselected;
  cd_pivot_shellpairs_vec.assign(pivot_shellpairs.begin(), pivot_shellpairs.end());

  // (mu, nu) -> pivot rank lookup for forceJ_cholesky.
  cd_pivot_sentinel = Nselected;
  cd_pivot_index.set_size(Nbf, Nbf);
  cd_pivot_index.fill(cd_pivot_sentinel);
  for(arma::uword p=0; p<Nselected; p++) {
    const arma::uword pii = pi(p);
    cd_pivot_index(invmap(0,pii), invmap(1,pii)) = p;
    cd_pivot_index(invmap(1,pii), invmap(0,pii)) = p;
  }

  // Phase D: build the two-center metric (piv | piv). This is the
  // CD analog of the (alpha | beta) two-center metric DensityFit::fill
  // builds for a Gaussian aux basis.
  const std::vector<GaussianShell> & shells = basis.get_shells_ref();
  Timer t;
  arma::mat M_metric(Nselected, Nselected, arma::fill::zeros);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri = make_eri_worker(basis.get_max_am(), basis.get_max_Ncontr(), omega, alpha, beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t ip=0; ip<cd_pivot_shellpairs_vec.size(); ip++) {
      const size_t is = cd_pivot_shellpairs_vec[ip].first;
      const size_t js = cd_pivot_shellpairs_vec[ip].second;
      const size_t Ni = shells[is].get_Nbf();
      const size_t Nj = shells[js].get_Nbf();
      const size_t i0 = shells[is].get_first_ind();
      const size_t j0 = shells[js].get_first_ind();
      for(size_t jp=0; jp<=ip; jp++) {
        const size_t ks = cd_pivot_shellpairs_vec[jp].first;
        const size_t ls = cd_pivot_shellpairs_vec[jp].second;
        const size_t Nk = shells[ks].get_Nbf();
        const size_t Nl = shells[ls].get_Nbf();
        const size_t k0 = shells[ks].get_first_ind();
        const size_t l0 = shells[ls].get_first_ind();

        eri->compute(&shells[is], &shells[js], &shells[ks], &shells[ls]);
        erip = eri->getp();
        for(size_t ii=0; ii<Ni; ii++)
          for(size_t jj=0; jj<Nj; jj++) {
            const arma::uword pidx = cd_pivot_index(i0+ii, j0+jj);
            if(pidx == cd_pivot_sentinel) continue;
            for(size_t kk=0; kk<Nk; kk++)
              for(size_t ll=0; ll<Nl; ll++) {
                const arma::uword qidx = cd_pivot_index(k0+kk, l0+ll);
                if(qidx == cd_pivot_sentinel) continue;
                const double val = (*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
                // Each (pidx, qidx) pair maps to a unique unordered
                // pivot-shellpair pair, which the outer for(ip)
                // for(jp <= ip) loop visits exactly once -- so each
                // metric element is written by one thread, once.
                M_metric(pidx, qidx) = val;
                M_metric(qidx, pidx) = val;
              }
          }
      }
    }
  }
  double t_int_de = t.get();

  // Phase E: orthogonalise the pivot metric to get X with X^T M X = I
  // over the lindep-cleaned subspace.
  //
  // The pivot orbital products have widely varying self-overlap
  // (piv|piv), so orthogonalising the raw metric lets the threshold
  // discard functions by sheer magnitude rather than by genuine
  // linear dependence -- a small-norm but independent product gets
  // thrown out. Normalise to unit diagonal first ((A|A)=1 for every
  // fitting function), canonical-orthogonalise the correlation matrix
  // (eigenvalues of a unit-diagonal matrix directly measure linear
  // dependence), then fold the normalisation back in:
  //   M~ = D^-1 M D^-1   (D = diag(sqrt(M_pp))),   X~^T M~ X~ = I,
  // so X = D^-1 X~ satisfies X^T M X = I on the true (unnormalised) M.
  t.set();
  ab = std::move(M_metric);
  arma::vec dinv(ab.n_rows);
  for(arma::uword p=0; p<ab.n_rows; p++) {
    const double dpp = ab(p,p);
    // Pivots are selected from significant (mu nu|mu nu), so the
    // diagonal is strictly positive; guard defensively anyway.
    dinv(p) = (dpp > 0.0) ? 1.0/std::sqrt(dpp) : 0.0;
  }
  arma::mat Mtilde(ab);
  Mtilde.each_col() %= dinv;     // M~_pq = M_pq / d_p ...
  Mtilde.each_row() %= dinv.t(); //                  ... / d_q
  ab_invh = CanonicalOrth(Mtilde, fit_cholesky_thr);
  ab_invh.each_col() %= dinv;    // X = D^-1 X~
  double t_chol_de = t.get();

  if(verbose) {
    printf("Two-step CD: pivot metric orthogonalisation reduced %i -> %i functions (%s).\n",
           (int) Nselected, (int) ab_invh.n_cols, t.elapsed().c_str());
    fflush(stdout);
  }

  // Build the (piv_p | mu nu) block storage. Per-shellpair block
  // shape and layout match the DF path so the J/K kernels are
  // insensitive to direct vs cached.
  orbpairs = basis.compute_screening(shell_screen_tol).shpairs;

  std::vector<std::pair<size_t, size_t>> sp_pairs, sp_firsts, sp_sizes;
  build_shellpair_descriptor(sp_pairs, sp_firsts, sp_sizes);

  // Bake the metric into the block storage so the J/K kernels see
  // L = B_raw * X with identity metric. The on-the-fly form
  // (raw integrals + X X^T applied per call) loses precision because
  // X X^T = M^-1 has eigenvalues 1/lambda up to ~1e8-1e11, so forming
  // it explicitly rounds badly; baking X into L sidesteps that.
  // forceJ_cholesky / forceK still need X to map the indep-space
  // expansion d back to a pivot-space coefficient (c = X d) and to
  // build the per-orbital Z, so keep it in cd_X; the metric M itself
  // is recomputed on the fly for derivatives, not stored. ab / ab_inv
  // / ab_invh become identity over the cleaned subspace. Applies to
  // both cached and direct paths so they share the L-baked convention.
  const size_t Naux_indep = ab_invh.n_cols;
  Naux = Naux_indep;
  cd_X = std::move(ab_invh);
  // The metric is fully baked into the L blocks now, so the J/K
  // kernels need no (a|b) matrices in CD mode -- they branch on
  // cholesky_mode and skip the multiply. Free ab / ab_inv / ab_invh
  // rather than carry Naux_indep^2 identity matrices around.
  ab.reset();
  ab_inv.reset();
  ab_invh.reset();

  // Both modes build the same L blocks: DirectCDBlocks recomputes
  // (piv | mu nu) from libint per block and bakes cd_X so the J/K
  // kernels see L = X^T (piv|mu nu) with identity metric. In direct mode
  // the builder *is* the block store (recomputed on each get_block); in
  // cached mode we materialise it once into a CachedBlocks and keep the
  // result. Either way nothing of size Nprod x Nselected is held -- the
  // cached fill peak is the store itself, not store + B_raw -- at the
  // cost of recomputing the (mu nu | piv) integrals once here (they are
  // no longer saved during pivoting).
  auto builder = std::make_shared<DirectCDBlocks>(
      Nbf, Naux, sp_pairs, sp_firsts, sp_sizes,
      orbshells, cd_pivot_shellpairs_vec, cd_pivot_index, cd_pivot_sentinel,
      cd_X, omega, alpha, beta, maxam, maxcontr);
  if(direct) {
    blocks = builder;
  } else {
    auto cached = std::make_shared<CachedBlocks>(
        Nbf, Naux, std::move(sp_pairs), std::move(sp_firsts), std::move(sp_sizes));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(size_t ip=0; ip<orbpairs.size(); ip++) {
      arma::mat slot = cached->block_mut(ip);
      slot = builder->get_block(ip);
    }
    blocks = cached;
  }

  if(verbose) {
    printf("Two-step CD finished in %s.\n", ttot.elapsed().c_str());
    if(t_int_de+t_chol_de > 0.0)
      printf("Metric build / orthogonalisation time use: integrals %3.1f %%, linear algebra %3.1f %%.\n",
             100*t_int_de/(t_int_de+t_chol_de),100*t_chol_de/(t_int_de+t_chol_de));
    fflush(stdout);
  }

  return orbpairs.size();
}

// Helper for the two CD force loops. Both compute a 4-shell dERIWorker
// derivative and accumulate into a per-thread fwrk; the only thing
// that varies is the outer shellpair list and the per-quartet
// contraction body. The helper handles dERIWorker construction +
// per-thread fwrk + omp critical reduction; the caller writes only
// the inner-loop math.
//
// `body(ip, deri, fout)` is invoked once per outer index ip with a
// thread-local dERIWorker and an arma::vec reference to write into
// (the per-thread accumulator under OMP, the shared f otherwise).
namespace {
template<typename Body>
void run_force_loop(size_t Nouter,
                    arma::vec & f,
                    int max_am, int max_ncon,
                    double omega, double alpha, double beta,
                    Body && body) {
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto deri = make_deri_worker(max_am, max_ncon, omega, alpha, beta);

#ifdef _OPENMP
    arma::vec fwrk(f); fwrk.zeros();
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0; ip<Nouter; ip++) {
#ifdef _OPENMP
      body(ip, deri.get(), fwrk);
#else
      body(ip, deri.get(), f);
#endif
    }
#ifdef _OPENMP
#pragma omp critical
    f += fwrk;
#endif
  }
}
}

template<typename M_lookup>
void DensityFit::accumulate_2c_metric_force(arma::vec & f, M_lookup && M, double sign) const {
  if(!cholesky_mode) {
    // DF dispatch: aux shellpair pairs (ias, jas <= ias). dERIWorker
    // (aux, dummy, aux, dummy) gives 12 components, of which the
    // physical ones are index[] = {0,1,2,6,7,8} (the dummy shells
    // have no nuclear position). The same-atom (anuc == bnuc) case
    // gives a vanishing (a|b)/dR and is skipped. run_force_loop
    // owns the per-thread dERIWorker + fwrk reduction; we iterate
    // over the outer aux shell here and run the inner jas <= ias
    // loop in the body.
    run_force_loop(auxshells.size(), f, maxam, maxcontr, omega, alpha, beta,
                   [&](size_t ias, dERIWorker * deri, arma::vec & fout) {
      for(size_t jas=0; jas<=ias; jas++) {
        // Off-diagonal aux-shellpair pair contributes both (ias, jas)
        // and (jas, ias) via integral symmetry; the (jas <= ias)
        // iteration visits the pair once, so double the factor there.
        double fac = (ias != jas) ? 1.0 : 0.5;
        const size_t Na   = auxshells[ias].get_Nbf();
        const size_t anuc = auxshells[ias].get_center_ind();
        const size_t Nb   = auxshells[jas].get_Nbf();
        const size_t bnuc = auxshells[jas].get_center_ind();
        if(anuc == bnuc) continue;

        deri->compute(&auxshells[ias], &dummy, &auxshells[jas], &dummy);
        const static int index[]={0, 1, 2, 6, 7, 8};
        double ders[6] = {0,0,0,0,0,0};
        for(size_t iid=0; iid<6; iid++) {
          const int ic = index[iid];
          const std::vector<double> * erip = deri->getp(ic);
          for(size_t iia=0; iia<Na; iia++) {
            const size_t ia = auxshells[ias].get_first_ind() + iia;
            for(size_t iib=0; iib<Nb; iib++) {
              const size_t ib = auxshells[jas].get_first_ind() + iib;
              ders[iid] += (*erip)[iia*Nb+iib] * M(ia, ib);
            }
          }
          ders[iid] *= fac * sign;
        }
        for(int ic=0; ic<3; ic++) {
          fout(3*anuc + ic) += ders[ic];
          fout(3*bnuc + ic) += ders[ic+3];
        }
      }
    });
  } else {
    // CD dispatch: pivot shellpair pairs (ip, jp <= ip). 4-shell
    // dERIWorker gives 12 derivative components mapped to 4 centers.
    // M is looked up via cd_pivot_index(orb_idx_1, orb_idx_2).
    const std::vector<GaussianShell> & shells = orbshells;
    run_force_loop(cd_pivot_shellpairs_vec.size(), f, maxam, maxcontr, omega, alpha, beta,
                   [&](size_t ip, dERIWorker * deri, arma::vec & fout) {
      const size_t is = cd_pivot_shellpairs_vec[ip].first;
      const size_t js = cd_pivot_shellpairs_vec[ip].second;
      const size_t Ni = shells[is].get_Nbf();
      const size_t Nj = shells[js].get_Nbf();
      const size_t i0 = shells[is].get_first_ind();
      const size_t j0 = shells[js].get_first_ind();
      const size_t i_at = shells[is].get_center_ind();
      const size_t j_at = shells[js].get_center_ind();

      for(size_t jp=0; jp<=ip; jp++) {
        const size_t ks = cd_pivot_shellpairs_vec[jp].first;
        const size_t ls = cd_pivot_shellpairs_vec[jp].second;
        const size_t Nk = shells[ks].get_Nbf();
        const size_t Nl = shells[ls].get_Nbf();
        const size_t k0 = shells[ks].get_first_ind();
        const size_t l0 = shells[ls].get_first_ind();
        const size_t k_at = shells[ks].get_center_ind();
        const size_t l_at = shells[ls].get_center_ind();

        const double fac_sp = (ip == jp) ? 0.5 : 1.0;
        deri->compute(&shells[is], &shells[js], &shells[ks], &shells[ls]);

        const size_t atoms[4] = {i_at, j_at, k_at, l_at};
        for(int ic=0; ic<12; ic++) {
          const size_t aA = atoms[ic / 3];
          const std::vector<double> * erip = deri->getp(ic);
          double accum = 0.0;
          for(size_t ii=0; ii<Ni; ii++)
            for(size_t jj=0; jj<Nj; jj++) {
              const arma::uword pidx = cd_pivot_index(i0+ii, j0+jj);
              if(pidx == cd_pivot_sentinel) continue;
              for(size_t kk=0; kk<Nk; kk++)
                for(size_t ll=0; ll<Nl; ll++) {
                  const arma::uword qidx = cd_pivot_index(k0+kk, l0+ll);
                  if(qidx == cd_pivot_sentinel) continue;
                  accum += (*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll] * M(pidx, qidx);
                }
            }
          fout(3*aA + ic%3) += fac_sp * sign * accum;
        }
      }
    });
  }
}

template<typename BuildQ>
void DensityFit::accumulate_3c_force_DF(arma::vec & f, double sign, BuildQ && build_q) const {
  // Build the per-shellpair descriptor once, hand it to a
  // DirectDFPerturbedBlocks instance, then iterate. for_each_pert
  // streams (perturbation, aux_first, sub_block) tuples; each
  // contributes sign * <sub_block, Q_ip.rows(a0, a0+Na_sh-1)>.
  std::vector<std::pair<size_t, size_t>> sp_pairs, sp_firsts, sp_sizes;
  build_shellpair_descriptor(sp_pairs, sp_firsts, sp_sizes);
  DirectDFPerturbedBlocks pblocks(Nbf, Naux, Nnuc,
                                  std::move(sp_pairs), std::move(sp_firsts), std::move(sp_sizes),
                                  orbshells, auxshells, dummy,
                                  omega, alpha, beta, maxam, maxcontr);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    arma::vec fwrk(f); fwrk.zeros();
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0; ip<orbpairs.size(); ip++) {
      const arma::mat Q_ip = build_q(ip);  // (Naux x Nmu*Nnu)
      pblocks.for_each_pert(ip,
          [&](const Perturbation & pert, size_t a0, const arma::mat & sub_block) {
            const size_t Na_sh = sub_block.n_rows;
            const arma::mat Qslice = Q_ip.rows(a0, a0 + Na_sh - 1);
            const double ders = arma::dot(arma::vectorise(sub_block), arma::vectorise(Qslice));
#ifdef _OPENMP
            fwrk(3 * pert.p1 + pert.p2) += sign * ders;
#else
            f(3 * pert.p1 + pert.p2)    += sign * ders;
#endif
          });
    }
#ifdef _OPENMP
#pragma omp critical
    f += fwrk;
#endif
  }
}

template<typename BuildQ>
void DensityFit::accumulate_3c_force_CD(const BasisSet & basis, arma::vec & f, double sign, BuildQ && build_q) const {
  // Iterate (orbital_shellpair, pivot_shellpair) quartets. For each
  // quartet, 4-shell dERIWorker gives 12 derivative components; the
  // inner contraction with build_q(...)(qidx, ii*Nj+jj) is summed
  // over (ii, jj, kk, ll) with cd_pivot_index deciding which (kk, ll)
  // entries land on selected pivots.
  //
  // build_q signature: (size_t ipair, size_t is, size_t js, size_t Ni,
  //                     size_t Nj, size_t i0, size_t j0) -> arma::mat
  // returning a (Naux x Ni*Nj) matrix with column index = ii*Nj + jj.
  const std::vector<eripair_t> orb_shps =
    basis.compute_screening(/*tol*/0.0, omega, alpha, beta, false).shpairs;
  const std::vector<GaussianShell> & shells = basis.get_shells_ref();

  run_force_loop(orb_shps.size(), f, maxam, maxcontr, omega, alpha, beta,
                 [&](size_t ipair, dERIWorker * deri, arma::vec & fout) {
    const size_t is = orb_shps[ipair].is;
    const size_t js = orb_shps[ipair].js;
    const size_t Ni = shells[is].get_Nbf();
    const size_t Nj = shells[js].get_Nbf();
    const size_t i0 = shells[is].get_first_ind();
    const size_t j0 = shells[js].get_first_ind();
    const size_t i_at = shells[is].get_center_ind();
    const size_t j_at = shells[js].get_center_ind();

    const arma::mat Q_ip = build_q(ipair, is, js, Ni, Nj, i0, j0);  // (Naux x Ni*Nj), col = ii*Nj+jj

    for(size_t jp=0; jp<cd_pivot_shellpairs_vec.size(); jp++) {
      const size_t ks = cd_pivot_shellpairs_vec[jp].first;
      const size_t ls = cd_pivot_shellpairs_vec[jp].second;
      const size_t Nk = shells[ks].get_Nbf();
      const size_t Nl = shells[ls].get_Nbf();
      const size_t k0 = shells[ks].get_first_ind();
      const size_t l0 = shells[ls].get_first_ind();
      const size_t k_at = shells[ks].get_center_ind();
      const size_t l_at = shells[ls].get_center_ind();

      deri->compute(&shells[is], &shells[js], &shells[ks], &shells[ls]);

      const size_t atoms[4] = {i_at, j_at, k_at, l_at};
      for(int ic=0; ic<12; ic++) {
        const size_t aA = atoms[ic / 3];
        const std::vector<double> * erip = deri->getp(ic);
        double accum = 0.0;
        for(size_t ii=0; ii<Ni; ii++)
          for(size_t jj=0; jj<Nj; jj++) {
            const size_t col = ii*Nj + jj;
            for(size_t kk=0; kk<Nk; kk++)
              for(size_t ll=0; ll<Nl; ll++) {
                const arma::uword qidx = cd_pivot_index(k0+kk, l0+ll);
                if(qidx == cd_pivot_sentinel) continue;
                accum += (*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll] * Q_ip(qidx, col);
              }
          }
        fout(3*aA + ic%3) += sign * accum;
      }
    }
  });
}

arma::vec DensityFit::forceJ_cholesky(const BasisSet & basis, const arma::mat & P) const {
  if(!cholesky_mode)
    throw std::runtime_error("DensityFit::forceJ_cholesky requires fill_cholesky to have been called.\n");
  if(P.n_rows != Nbf || P.n_cols != Nbf)
    throw std::runtime_error("DensityFit::forceJ_cholesky: density matrix dimension mismatch.\n");

  // The blocks store L = B_raw * X (indep-space orthonormal vectors),
  // so compute_expansion returns d = L^T Pv (with doubling),
  // which is the indep-space expansion. The force algebra is in
  // pivot space: c_raw = X * d gives the pivot-space coefficient
  // that goes against M, dM/dR and the 3-center derivatives, the
  // same c that the force kernels used before the L-baked storage.
  const arma::vec d = compute_expansion(P);
  const arma::vec c = cd_X * d;
  const size_t Naux_pivot = cd_X.n_rows;  // == Nselected

  arma::vec f(3 * Nnuc, arma::fill::zeros);

  // Part 1: f += (1/2) c^T (dM/dR) c. accumulate_2c_metric_force
  // handles the CD pivot-shellpair-pair scaffolding.
  accumulate_2c_metric_force(f,
      [&c](arma::uword a, arma::uword b) { return c(a) * c(b); },
      +1.0);

  // Part 2: f -= sum_munu P (d(mu nu | piv)/dR) c. accumulate_3c_force_CD
  // handles the (orb_shellpair, pivot_shellpair) iteration; per
  // orbital shellpair we hand it the rank-1 Q(qidx, ii*Nj+jj) =
  // fac_sp * P(i0+ii, j0+jj) * c(qidx) tensor.
  accumulate_3c_force_CD(basis, f, -1.0,
      [&](size_t /*ipair*/, size_t is, size_t js,
          size_t Ni, size_t Nj, size_t i0, size_t j0) {
        const double fac_sp = (is == js) ? 1.0 : 2.0;
        arma::mat Q(Naux_pivot, Ni*Nj);
        for(size_t ii=0; ii<Ni; ii++)
          for(size_t jj=0; jj<Nj; jj++) {
            const double Pval = P(i0+ii, j0+jj);
            Q.col(ii*Nj + jj) = fac_sp * Pval * c;
          }
        return Q;
      });

  return f;
}

arma::vec DensityFit::forceK(const BasisSet & basis, const arma::mat & Corig, const std::vector<double> & occo, double kfrac) const {
  if(Corig.n_rows != Nbf)
    throw std::runtime_error("DensityFit::forceK: orbital matrix doesn't match basis set.\n");

  // Filter to occupied orbitals (drop columns with zero occupation).
  arma::mat C;
  arma::vec occs;
  filter_occupied(Corig, occo, C, occs);
  const size_t Nmo = C.n_cols;
  // Closed-shell density.
  const arma::mat P = C * arma::diagmat(occs) * C.t();

  // Per-orbital half-transform aui[io](a, mu) = sum_nu (a|mu nu) C(nu, io).
  // In DF mode Z[io] = ab_inv * aui[io] (M^{-1} aui in aux space),
  // shape (Naux_DF x Nbf). In CD mode the blocks store L (indep-space
  // orthonormal), aui = L^T C is already in indep space, and the
  // dM/dR / 3-center derivative kernels iterate pivot space, so
  // Z = cd_X * aui projects up to (Nselected x Nbf).
  const size_t Naux_force = cholesky_mode ? cd_X.n_rows : Naux;
  arma::cube Z(Naux_force, Nbf, Nmo, arma::fill::zeros);

  size_t Nmax = 0;
  for(size_t s=0; s<orbshells.size(); s++)
    Nmax = std::max(Nmax, orbshells[s].get_Nbf());

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::mat aui;
    arma::mat ui_scratch(Naux*Nmax, 1);
    arma::mat vi_scratch(Naux*Nmax, 1);
    arma::mat anumu_scratch(Naux, Nmax*Nmax);

#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t io=0; io<Nmo; io++) {
      aui.zeros(Naux, Nbf);
      for(size_t ip=0; ip<orbpairs.size(); ip++) {
        const size_t imus = orbpairs[ip].is;
        const size_t inus = orbpairs[ip].js;
        const size_t mu0  = orbshells[imus].get_first_ind();
        const size_t nu0  = orbshells[inus].get_first_ind();
        const size_t Nmu  = orbshells[imus].get_Nbf();
        const size_t Nnu  = orbshells[inus].get_Nbf();
        arma::mat amunu = blocks->get_block(ip);

        {
          arma::mat ui(ui_scratch.memptr(), Naux*Nmu, 1, false, true);
          ui = arma::reshape(amunu, Naux*Nmu, Nnu) * C.submat(nu0, io, nu0+Nnu-1, io);
          ui.reshape(Naux, Nmu);
          aui.cols(mu0, mu0+Nmu-1) += ui;
        }
        if(imus != inus) {
          arma::mat anumu(anumu_scratch.memptr(), Naux, Nmu*Nnu, false, true);
          for(size_t mu=0; mu<Nmu; mu++)
            for(size_t nu=0; nu<Nnu; nu++)
              anumu.col(mu*Nnu+nu) = amunu.col(nu*Nmu+mu);
          arma::mat vi(vi_scratch.memptr(), Naux*Nnu, 1, false, true);
          vi = arma::reshape(anumu, Naux*Nnu, Nmu) * C.submat(mu0, io, mu0+Nmu-1, io);
          vi.reshape(Naux, Nnu);
          aui.cols(nu0, nu0+Nnu-1) += vi;
        }
      }
      // DF: aui is in aux space (Naux DF aux fns); Z = M^{-1} aui.
      // CD: blocks store L, aui = X^T aui_raw is indep-space; the
      //     dM/dR / 3-center derivative kernels iterate pivot space,
      //     so Z = X * aui = X X^T aui_raw = M^{-1} aui_raw lives
      //     in pivot space.
      if(cholesky_mode)
        Z.slice(io) = cd_X * aui;
      else
        Z.slice(io) = ab_inv * aui;
    }
  }
  const size_t Naux_pivot = Naux_force;

  // V[io] = P Z[io]^T (Nbf x Nselected). Used for both G and the
  // 3-center contraction.
  arma::cube V(Nbf, Naux_pivot, Nmo);
  for(size_t io=0; io<Nmo; io++)
    V.slice(io) = P * Z.slice(io).t();

  // G(a, b) = sum_io n_io (Z[io] P Z[io]^T)(a, b) over pivot space.
  arma::mat G(Naux_pivot, Naux_pivot, arma::fill::zeros);
  for(size_t io=0; io<Nmo; io++)
    G += occs(io) * Z.slice(io) * V.slice(io);

  // Geometric force accumulator. kfrac applied at return.
  // dE_K/dR = - 3c_term + (1/2) 2c_term, so f_K = -dE_K/dR
  //         = + 3c_term - (1/2) 2c_term, scaled by kfrac.
  arma::vec f_geom(3*Nnuc, arma::fill::zeros);

  // ========================================================================
  // 2-center derivative: f_geom -= (1/2) sum_ab (d_R M_ab) G(a, b)
  // ========================================================================
  // accumulate_2c_metric_force handles both DF and CD dispatch.
  accumulate_2c_metric_force(f_geom,
      [&G](arma::uword a, arma::uword b) { return G(a, b); },
      -1.0);

  // ========================================================================
  // 3-center derivative: f_geom += sum_i n_i sum_aνλ d_R(a|νλ) C(λ,i) V_i(ν,a)
  // ========================================================================
  // Per orbital shellpair (s1, s2), build a Q_combined matrix that
  // pre-mixes occupied orbitals into a single contraction tensor.
  // The (s1 != s2) branch absorbs the (mu <-> nu) swap term that
  // orbpairs doesn't double-count. accumulate_3c_force_{DF,CD}
  // handles the integral dispatch.
  auto build_Qcomb_DF = [&](size_t ip) -> arma::mat {
    const size_t imus = orbpairs[ip].is;
    const size_t inus = orbpairs[ip].js;
    const size_t mu0  = orbshells[imus].get_first_ind();
    const size_t nu0  = orbshells[inus].get_first_ind();
    const size_t Nmu  = orbshells[imus].get_Nbf();
    const size_t Nnu  = orbshells[inus].get_Nbf();
    const bool   off_diag = (imus != inus);

    // Q(a, inu*Nmu + imu) = sum_io n_io [
    //   C(nu0+inu, io) * V_io(mu0+imu, a)
    //   + (off_diag ? C(mu0+imu, io) * V_io(nu0+inu, a) : 0)
    // ]
    // V.slice(io) lives in the force aux dimension (DF aux for !cholesky_mode,
    // pivot space Nselected for cholesky_mode), which matches Naux_force.
    arma::mat Qcomb(Naux_force, Nmu*Nnu, arma::fill::zeros);
    for(size_t io=0; io<Nmo; io++) {
      const double n_io = occs(io);
      for(size_t inu=0; inu<Nnu; inu++)
        for(size_t imu=0; imu<Nmu; imu++) {
          const size_t col = inu*Nmu + imu;
          Qcomb.col(col) += n_io * C(nu0+inu, io) * V.slice(io).row(mu0+imu).t();
          if(off_diag)
            Qcomb.col(col) += n_io * C(mu0+imu, io) * V.slice(io).row(nu0+inu).t();
        }
    }
    return Qcomb;
  };

  if(!cholesky_mode) {
    accumulate_3c_force_DF(f_geom, +1.0, build_Qcomb_DF);
  } else {
    // CD column layout uses ii*Nj+jj rather than DF's inu*Nmu+imu;
    // the build function adapts accordingly.
    accumulate_3c_force_CD(basis, f_geom, +1.0,
        [&](size_t /*ipair*/, size_t is, size_t js,
            size_t Ni, size_t Nj, size_t i0, size_t j0) {
          const bool off_diag = (is != js);
          arma::mat Qcomb(Naux_force, Ni*Nj, arma::fill::zeros);
          for(size_t io=0; io<Nmo; io++) {
            const double n_io = occs(io);
            for(size_t ii=0; ii<Ni; ii++)
              for(size_t jj=0; jj<Nj; jj++) {
                const size_t col = ii*Nj + jj;
                Qcomb.col(col) += n_io * C(j0+jj, io) * V.slice(io).row(i0+ii).t();
                if(off_diag)
                  Qcomb.col(col) += n_io * C(i0+ii, io) * V.slice(io).row(j0+jj).t();
              }
          }
          return Qcomb;
        });
  }

  // ERKALE's K matrix (from calcK) carries the closed-shell doubling
  // explicitly (K = sum_i occs[i] aui_i^T M^-1 aui_i with occs[i]=2),
  // and the Fock build correspondingly uses F = h + J - 0.5*K (see
  // scf-fock.cpp.in:683). The exchange energy contribution to the
  // total is therefore E_K = -(1/4) tr(P K), and the gradient
  // f_geom assembled above corresponds to -(1/2) tr(P dK/dR), which
  // is twice the actual gradient. Halve before returning.
  return 0.5 * kfrac * f_geom;
}

double DensityFit::fitting_error() const {
  arma::mat error_matrix(maxorbam+1, maxorbam+1, arma::fill::zeros);

  // Loop over pairs
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::mat wrk_error(error_matrix);

    auto eri = make_eri_worker(maxam, maxcontr, omega, alpha, beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      // Shells in question are
      size_t imus=orbpairs[ip].is;
      size_t inus=orbpairs[ip].js;
      // Amount of functions
      size_t Nmu=orbshells[imus].get_Nbf();
      size_t Nnu=orbshells[inus].get_Nbf();

      // Compute the (A|uv) integrals
      arma::mat auv(compute_a_munu(eri.get(), ip));

      // This gives the density fitted (uv|uv) integrals as
      arma::mat dfit_uvuv(arma::trans(auv) * ab_inv * auv);

      // The correct integrals are, however
      eri->compute(&orbshells[inus],&orbshells[imus],&orbshells[inus],&orbshells[imus]);
      const std::vector<double> * erip(eri->getp());

      double shell_error=0.0;
      for(size_t mu=0;mu<Nmu;mu++)
        for(size_t nu=0;nu<Nnu;nu++) {
          size_t imunu = nu*Nmu+mu;
          size_t ieri = imunu*(Nmu*Nnu) + imunu;
          double delta= (*erip)[ieri] - dfit_uvuv(imunu, imunu);
          //printf("(%c%c|%c%c): (%i %i|%i %i) = %e (fit) vs %e (exact), error %e\n", shell_types[orbshells[inus].get_am()], shell_types[orbshells[imus].get_am()], shell_types[orbshells[inus].get_am()], shell_types[orbshells[imus].get_am()], (int) (nu0+nu),(int) (mu0+mu),(int) (nu0+nu),(int) (mu0+mu),dfit_uvuv(imunu, imunu),(*erip)[ieri],delta);
          shell_error += delta;
        }

      wrk_error(orbshells[imus].get_am(), orbshells[inus].get_am()) += shell_error;
      if(imus != inus)
        wrk_error(orbshells[inus].get_am(), orbshells[imus].get_am()) += shell_error;
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    error_matrix += wrk_error;
  }

  printf("\n");
  for(int iam=0;iam<=maxorbam;iam++)
    for(int jam=0;jam<=iam;jam++)
      printf("Total (%c%c|%c%c) error %e\n",shell_types[jam],shell_types[iam],shell_types[jam],shell_types[iam],error_matrix(jam,iam));

  double total_error = arma::sum(arma::sum(error_matrix));
  printf("Total error is %.15e\n",total_error);

  return total_error;
}

arma::mat DensityFit::compute_a_munu(ERIWorker *eri, size_t ip, double *memptr) const {
  // Shells in question are
  size_t imus=orbpairs[ip].is;
  size_t inus=orbpairs[ip].js;
  // Amount of functions
  size_t Nmu=orbshells[imus].get_Nbf();
  size_t Nnu=orbshells[inus].get_Nbf();

  // Allocate storage. If the caller supplied a backing buffer
  // (memptr), wrap it as an advisory mat (no copy / no resize); else
  // own the allocation.
  arma::mat amunu;
  if(memptr != nullptr)
    amunu=arma::mat(memptr, Naux, Nmu*Nnu, false, true);
  else
    amunu.zeros(Naux,Nmu*Nnu);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ia=0;ia<auxshells.size();ia++) {
    // Number of functions on shell
    size_t Na=auxshells[ia].get_Nbf();
    size_t a0=auxshells[ia].get_first_ind();

    // Compute (a|vu)
    eri->compute(&auxshells[ia],&dummy,&orbshells[inus],&orbshells[imus]);
    const std::vector<double> * erip(eri->getp());

    // Store integrals
    for(size_t a=0;a<Na;a++)
      for(size_t imunu=0;imunu<Nmu*Nnu;imunu++)
	// Use Fortran ordering so it's compatible with Armadillo
	//amunu(a0+a,nu*Nmu+mu)=(*erip)[(a*Nnu+nu)*Nmu+mu];
	amunu(a0+a,imunu)=(*erip)[a*Nnu*Nmu+imunu];
  }

  return amunu;
}

void DensityFit::check_density_dims(const arma::mat & P) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }
}

void DensityFit::project_density_to_aux(const arma::mat & P, size_t ip, const arma::mat & amunu, arma::vec & gamma) const {
  check_density_dims(P);

  // Shells in question are
  size_t imus=orbpairs[ip].is;
  size_t inus=orbpairs[ip].js;
  // First function on shell
  size_t mubeg=orbshells[imus].get_first_ind();
  size_t nubeg=orbshells[inus].get_first_ind();
  // Amount of functions
  size_t muend=orbshells[imus].get_last_ind();
  size_t nuend=orbshells[inus].get_last_ind();

  // Density submatrix
  arma::vec Psub;
  if(imus != inus)
    // On the off-diagonal we're twice degenerate
    Psub=2.0*arma::vectorise(P.submat(mubeg,nubeg,muend,nuend));
  else
    Psub=arma::vectorise(P.submat(mubeg,nubeg,muend,nuend));

  // Contract over indices
  gamma+=amunu*Psub;
}

void DensityFit::contract_aux_to_J(const arma::vec & gamma, size_t ip, const arma::mat & amunu, arma::mat & J) const {
  // Shells in question are
  size_t imus=orbpairs[ip].is;
  size_t inus=orbpairs[ip].js;
  // First function on shell
  size_t mu0=orbshells[imus].get_first_ind();
  size_t nu0=orbshells[inus].get_first_ind();
  // Amount of functions
  size_t Nmu=orbshells[imus].get_Nbf();
  size_t Nnu=orbshells[inus].get_Nbf();

  // vec(uv) = c_a (a,uv)
  arma::mat Jsub(arma::trans(gamma)*amunu);
  // Reshape into matrix
  Jsub.reshape(Nmu,Nnu);

  J.submat(mu0,nu0,mu0+Nmu-1,nu0+Nnu-1)=Jsub;
  J.submat(nu0,mu0,nu0+Nnu-1,mu0+Nmu-1)=arma::trans(Jsub);
}

template<typename T>
void DensityFit::accumulate_K_from_blocks(const arma::Mat<T> & C, const arma::vec & occs, arma::Mat<T> & K) const {
  if(C.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", C.n_rows = " << C.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // K_uv = sum_i n_i (ui|vi) = sum_i n_i (a|ui) (a|b)^-1 (b|vi)
  // Parallelise over orbitals; per-thread scratch holds the
  // half-transformed aui + the (a|nu mu) swap workspace for the
  // off-diagonal-shellpair branch. For complex orbitals the
  // conjugation enters implicitly via arma::trans (which is the
  // Hermitian transpose in arma::Mat<complex>); on real orbitals
  // arma::trans is the plain transpose and the same expression
  // works.
  size_t Nmax=0;
  for(size_t is=0;is<orbshells.size();is++)
    Nmax=std::max(Nmax, orbshells[is].get_Nbf());

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::Mat<T> aui;
    arma::Mat<T> ui_scratch(Naux*Nmax, 1);
    arma::Mat<T> vi_scratch(Naux*Nmax, 1);
    arma::mat    anumu_scratch(Naux, Nmax*Nmax);  // real -- amunu is always real

#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t io=0;io<C.n_cols;io++) {
      aui.zeros(Naux,Nbf);
      for(size_t ip=0;ip<orbpairs.size();ip++) {
        const size_t imus = orbpairs[ip].is;
        const size_t inus = orbpairs[ip].js;
        const size_t mu0  = orbshells[imus].get_first_ind();
        const size_t nu0  = orbshells[inus].get_first_ind();
        const size_t Nmu  = orbshells[imus].get_Nbf();
        const size_t Nnu  = orbshells[inus].get_Nbf();
        // (Naux x Nmu*Nnu) block, always real (integrals).
        arma::mat amunu = blocks->get_block(ip);

        // Half-transform (a | u; i): reshape amunu to (Naux*Nmu, Nnu)
        // and contract over nu with C(nu, io). Advisory view into
        // ui_scratch avoids per-quartet heap alloc.
        {
          arma::Mat<T> ui(ui_scratch.memptr(), Naux*Nmu, 1, false, true);
          ui = arma::reshape(amunu, Naux*Nmu, Nnu) * C.submat(nu0,io,nu0+Nnu-1,io);
          ui.reshape(Naux, Nmu);
          aui.cols(mu0, mu0+Nmu-1) += ui;
        }

        if(imus != inus) {
          // Off-diagonal shellpair: swap mu/nu axes to compute (a | v; i).
          arma::mat anumu(anumu_scratch.memptr(), Naux, Nmu*Nnu, false, true);
          for(size_t mu=0;mu<Nmu;mu++)
            for(size_t nu=0;nu<Nnu;nu++)
              anumu.col(mu*Nnu+nu) = amunu.col(nu*Nmu+mu);

          arma::Mat<T> vi(vi_scratch.memptr(), Naux*Nnu, 1, false, true);
          vi = arma::reshape(anumu, Naux*Nnu, Nmu) * C.submat(mu0,io,mu0+Nmu-1,io);
          vi.reshape(Naux, Nnu);
          aui.cols(nu0, nu0+Nnu-1) += vi;
        }
      }
      // K_uv = (a|ui) (a|b)^-1 (b|vi); ab_invh is the canonical-orth
      // half-inverse X with X^T (a|b) X = I, so (a|b)^{-1} ≈ X X^T
      // and the half-transform is X^T aui. In CD mode the blocks are
      // already L = X^T B_raw, so aui = sum_munu L C is the
      // half-transform directly -- no X^T multiply (ab_invh is empty).
      if(!cholesky_mode)
        aui = ab_invh.t() * aui;
      arma::Mat<T> K_io = occs[io] * arma::trans(aui) * aui;
#ifdef _OPENMP
#pragma omp critical
#endif
      K += K_io;
    }
  }
}

template void DensityFit::accumulate_K_from_blocks<double>(const arma::mat &, const arma::vec &, arma::mat &) const;
template void DensityFit::accumulate_K_from_blocks<std::complex<double>>(const arma::cx_mat &, const arma::vec &, arma::cx_mat &) const;

size_t DensityFit::memory_estimate(const BasisSet & orbbas, const BasisSet & auxbas, double thr, bool dir) const {
  // Amount of auxiliary functions (for representing the electron density)
  size_t Na=auxbas.get_Nbf();
  // Amount of memory required for calculation
  size_t Nmem=0;

  // Memory taken up by  ( \alpha | \mu \nu)
  if(!dir) {
    // Form screening matrix
    std::vector<eripair_t> opairs=orbbas.compute_screening(thr).shpairs;

    // Count number of function pairs
    size_t np=0;
    for(size_t ip=0;ip<opairs.size();ip++)
      np+=orbbas.get_Nbf(opairs[ip].is)*orbbas.get_Nbf(opairs[ip].js);
    Nmem+=Na*np*sizeof(double);
  }

  // Memory taken by (\alpha | \beta) and its inverse
  Nmem+=2*Na*Na*sizeof(double);
  // We also have (a|b)^(-1/2)
  Nmem+=Na*Na*sizeof(double);

  // Memory taken by gamma and expansion coefficients
  Nmem+=2*Na*sizeof(double);

  return Nmem;
}

arma::vec DensityFit::compute_expansion(const arma::mat & P) const {
  check_density_dims(P);

  arma::vec gamma(Naux);
  gamma.zeros();

  // Compute gamma; blocks->get_block(ip) routes to cached storage or
  // recomputes on the fly depending on the BTensorBlocks subclass.
#ifdef _OPENMP
#pragma omp parallel
  {
    arma::vec gv(Naux);
    gv.zeros();
#pragma omp for schedule(dynamic)
    for(size_t ip=0;ip<orbpairs.size();ip++)
      project_density_to_aux(P,ip,blocks->get_block(ip),gv);
#pragma omp critical
    gamma+=gv;
  }
#else
  for(size_t ip=0;ip<orbpairs.size();ip++)
    project_density_to_aux(P,ip,blocks->get_block(ip),gamma);
#endif

  // CD mode: the L blocks already carry the metric (L = X^T B_raw),
  // so the projection gamma_j = sum_munu L_{j,munu} P_munu is the
  // indep-space expansion directly -- no (a|b)^-1 multiply.
  if(cholesky_mode)
    return gamma;
  return ab_inv*gamma;
}

std::vector<arma::vec> DensityFit::compute_expansion(const std::vector<arma::mat> & P) const {
  for(size_t i=0;i<P.size();i++) {
    if(P[i].n_rows != Nbf || P[i].n_cols != Nbf) {
      std::ostringstream oss;
      oss << "Error in DensityFit: Nbf = " << Nbf << ", P[" << i << "].n_rows = " << P[i].n_rows << ", P[" << i << "].n_cols = " << P[i].n_cols << "!\n";
      throw std::logic_error(oss.str());
    }
  }

  std::vector<arma::vec> gamma(P.size());
  for(size_t i=0;i<P.size();i++)
    gamma[i].zeros(Naux);

  // Compute gamma; blocks->get_block(ip) routes to cached storage or
  // recomputes on the fly depending on the BTensorBlocks subclass.
  for(size_t iden=0;iden<P.size();iden++) {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      arma::vec gv(Naux);
      gv.zeros();
#pragma omp for schedule(dynamic)
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++) {
#ifdef _OPENMP
	project_density_to_aux(P[iden],ip,blocks->get_block(ip),gv);
#else
	project_density_to_aux(P[iden],ip,blocks->get_block(ip),gamma[iden]);
#endif
      }
#ifdef _OPENMP
#pragma omp critical
      gamma[iden]+=gv;
#endif
    }
  }

  // CD mode: L blocks carry the metric, gamma is already the
  // indep-space expansion (see single-density overload).
  if(!cholesky_mode)
    for(size_t ig=0;ig<P.size();ig++)
      gamma[ig]=ab_inv*gamma[ig];

  return gamma;
}

arma::mat DensityFit::calcJ(const arma::mat & P) const {
  check_density_dims(P);

  // Get the expansion coefficients
  arma::vec c=compute_expansion(P);
  return calcJ_vector(c);
}

arma::mat DensityFit::calcJ_vector(const arma::vec & c) const {
  if(c.n_elem != Naux) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Naux = " << Naux << ", c.n_elem = " << c.n_elem << "!\n";
    throw std::logic_error(oss.str());
  }

  arma::mat J(Nbf,Nbf);
  J.zeros();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<orbpairs.size();ip++)
    contract_aux_to_J(c,ip,blocks->get_block(ip),J);

  return J;
}

std::vector<arma::mat> DensityFit::calcJ(const std::vector<arma::mat> & P) const {
  // Get the expansion coefficients
  std::vector<arma::vec> c=compute_expansion(P);

  std::vector<arma::mat> J(P.size());
  for(size_t iden=0;iden<P.size();iden++)
    J[iden].zeros(Nbf,Nbf);

  // One libint sweep over orbital shellpairs covers all densities;
  // blocks->get_block(ip) is uniform across cached/direct backends.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<orbpairs.size();ip++) {
    arma::mat amunu = blocks->get_block(ip);
    for(size_t iden=0;iden<P.size();iden++)
      contract_aux_to_J(c[iden],ip,amunu,J[iden]);
  }

  return J;
}

arma::vec DensityFit::forceJ(const arma::mat & P) const {
  // First, compute the expansion
  arma::vec c=compute_expansion(P);

  // The force
  arma::vec f(3*Nnuc, arma::fill::zeros);

  // First part: f += (1/2) c^T (dM/dR) c. accumulate_2c_metric_force
  // handles the aux-shellpair-pair scaffolding (same helper as
  // forceJ_cholesky and forceK use).
  accumulate_2c_metric_force(f,
      [&c](arma::uword a, arma::uword b) { return c(a) * c(b); },
      +1.0);

  // Second part: f -= gamma_a' c_a (three-center derivative term).
  // accumulate_3c_force_DF runs the orbpair iteration and
  // DirectDFPerturbedBlocks streaming; the build_q callable
  // materialises the (Naux x Nmu*Nnu) per-shellpair Q matrix as
  // the rank-1 outer product fac * c x Psub^T. The for_each_pert
  // contraction then collapses to fac * (sub_block * Psub) . c on
  // each pert, equivalent to the original kernel.
  accumulate_3c_force_DF(f, -1.0,
      [&](size_t ip) {
        const size_t imus = orbpairs[ip].is;
        const size_t inus = orbpairs[ip].js;
        const size_t mu0  = orbshells[imus].get_first_ind();
        const size_t nu0  = orbshells[inus].get_first_ind();
        const size_t Nmu  = orbshells[imus].get_Nbf();
        const size_t Nnu  = orbshells[inus].get_Nbf();
        const double fac  = (imus == inus) ? 1.0 : 2.0;

        // P submatrix vectorised with mu fastest, matching the
        // value-side sub_block(a, inu*Nmu+imu) column layout.
        arma::rowvec Psub(Nmu * Nnu);
        for(size_t inu=0; inu<Nnu; inu++)
          for(size_t imu=0; imu<Nmu; imu++)
            Psub(inu*Nmu + imu) = P(mu0+imu, nu0+inu);

        // Rank-1 outer product fac * c * Psub.
        return arma::mat(fac * c * Psub);
      });

  return f;
}

template<typename T>
void DensityFit::filter_occupied(const arma::Mat<T> & Corig, const std::vector<double> & occo,
                                 arma::Mat<T> & C_out, arma::vec & occs_out) const {
  size_t Nmo = 0;
  for(size_t i=0; i<occo.size(); i++)
    if(occo[i] > 0) Nmo++;
  C_out.set_size(Corig.n_rows, Nmo);
  occs_out.set_size(Nmo);
  size_t io = 0;
  for(size_t i=0; i<occo.size(); i++)
    if(occo[i] > 0) {
      C_out.col(io) = Corig.col(i);
      occs_out(io) = occo[i];
      io++;
    }
}

template<typename T>
arma::Mat<T> DensityFit::calcK_impl(const arma::Mat<T> & Corig, const std::vector<double> & occo) const {
  if(Corig.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", Corig.n_rows = " << Corig.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  arma::Mat<T> C;
  arma::vec occs;
  filter_occupied(Corig, occo, C, occs);

  arma::Mat<T> K(Nbf, Nbf, arma::fill::zeros);
  // accumulate_K_from_blocks consumes blocks->get_block(ip) uniformly
  // across cached/direct backends; peak memory is one (Naux x Nbf)
  // aui per thread.
  accumulate_K_from_blocks(C, occs, K);
  return K;
}

arma::mat DensityFit::calcK(const arma::mat & Corig, const std::vector<double> & occo) const {
  return calcK_impl(Corig, occo);
}

arma::cx_mat DensityFit::calcK(const arma::cx_mat & Corig, const std::vector<double> & occo) const {
  return calcK_impl(Corig, occo);
}

size_t DensityFit::get_Naux() const {
  return Naux;
}

size_t DensityFit::get_Naux_indep() const {
  // DF: ab_invh's column count is the aux-space rank after the lindep
  // cleanup. CD: the cleanup already happened in fill_cholesky and
  // Naux was set to the cleaned-subspace size (ab_invh is empty), so
  // Naux is the independent count.
  return cholesky_mode ? Naux : ab_invh.n_cols;
}

const arma::mat & DensityFit::get_ab() const {
  return ab;
}

void DensityFit::three_center_integrals(arma::mat & ints) const {
  // blocks->get_block(ip) works in either cached or direct mode;
  // no need to forbid direct here. In CD mode auxshells is empty
  // but the block still has (Naux x Nmu*Nnu) entries, so iterate
  // over flat aux indices rather than per-shell — the same loop
  // shape works for both DF (Naux = aux basis size) and CD
  // (Naux = number of orthonormal L vectors).
  ints.zeros(Nbf*Nbf,Naux);
  for(size_t ip=0;ip<orbpairs.size();ip++) {
    const size_t imus=orbpairs[ip].is;
    const size_t inus=orbpairs[ip].js;
    const size_t Nmu=orbshells[imus].get_Nbf();
    const size_t Nnu=orbshells[inus].get_Nbf();
    const size_t mu0=orbshells[imus].get_first_ind();
    const size_t nu0=orbshells[inus].get_first_ind();

    arma::mat amunu = blocks->get_block(ip);  // (Naux x Nmu*Nnu)

    for(size_t imu=0;imu<Nmu;imu++) {
      const size_t mu=imu+mu0;
      for(size_t inu=0;inu<Nnu;inu++) {
        const size_t nu=inu+nu0;
        for(size_t a=0;a<Naux;a++) {
          const double el = amunu(a, inu*Nmu + imu);
          ints(mu*Nbf+nu, a) = el;
          ints(nu*Nbf+mu, a) = el;
        }
      }
    }
  }
}

void DensityFit::B_matrix(arma::mat & B) const {
  // three_center_integrals + the metric multiply work in either
  // cached or direct mode; the latter recomputes the shellpair
  // blocks on demand via libint inside the iteration. In DF mode
  // ab_invh is the metric half-inverse; in CD mode the blocks are
  // already L = X^T B_raw (metric baked in, ab_invh empty), so the
  // three-center integrals are the final B with no extra multiply.
  three_center_integrals(B);
  if(!cholesky_mode)
    B*=ab_invh;
}

arma::mat DensityFit::B_transform(const arma::mat & Cl, const arma::mat & Cr, bool verbose) const {
  if(Cl.n_rows != Nbf || Cr.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "DensityFit::B_transform: orbital matrices don't match basis set! Nbf = " << Nbf << ", Cl.n_rows = " << Cl.n_rows << ", Cr.n_rows = " << Cr.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  Timer t;

  // Build the dense B matrix once (Nbf*Nbf x Naux). Each column is
  // a (Nbf, Nbf) block transformed by two GEMMs.
  arma::mat Bdense;
  B_matrix(Bdense);

  if(verbose) {
    printf("Built dense B in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  const size_t Nl = Cl.n_cols;
  const size_t Nr = Cr.n_cols;
  arma::mat Br(Naux, Nl*Nr);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t P=0; P<Naux; P++) {
    const arma::mat block_P(Bdense.colptr(P), Nbf, Nbf, false, true);
    const arma::mat Tmo(Cl.t() * block_P.t() * Cr);
    Br.row(P) = arma::trans(arma::vectorise(Tmo));
  }

  if(verbose) {
    printf("MO transform done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  return Br;
}

// ===========================================================================
// HDF5 save / load. Used by SCF (and moints) to skip the fill on a
// repeated run when the orbital + auxiliary basis match a previously
// cached state. Plain DF and range-separated DF coexist in the same
// file by keying every entry with a (omega, alpha, beta)-derived
// prefix. CD entries use the same convention; DF and CD never share a
// key (the cholesky_mode flag is the first thing checked on load).
// ===========================================================================
namespace {
std::string df_cache_prefix(double omega, double alpha, double beta) {
  if(omega == 0.0 && alpha == 1.0 && beta == 0.0)
    return "";
  std::ostringstream oss;
  oss << "rs_o" << omega << "_a" << alpha << "_b" << beta << "_";
  return oss.str();
}

// Convert between size_t-style storage in arma::umat / std::vector<size_t>
// and the hsize_t-vector form Checkpoint understands.
std::vector<hsize_t> umat_to_hsize(const arma::umat & m) {
  return arma::conv_to<std::vector<hsize_t>>::from(arma::vectorise(m));
}
arma::umat hsize_to_umat(const std::vector<hsize_t> & v, size_t nrows, size_t ncols) {
  return arma::reshape(arma::conv_to<arma::umat>::from(v), nrows, ncols);
}
}

void DensityFit::save(const std::string & fname) const {
  if(direct)
    throw std::runtime_error("DensityFit::save: nothing to cache in direct mode.\n");
  if(Nbf == 0 || Naux == 0)
    throw std::runtime_error("DensityFit::save: object is uninitialised.\n");
  auto * cached = dynamic_cast<CachedBlocks *>(blocks.get());
  if(!cached)
    throw std::runtime_error("DensityFit::save: blocks backing store is not a CachedBlocks.\n");

  // Open the file: truncate if it doesn't exist, append otherwise.
  const bool trunc = !file_exists(fname);
  Checkpoint chkpt(fname, true, trunc);
  const std::string P = df_cache_prefix(omega, alpha, beta);

  chkpt.write(P+"cholesky_mode", (int) (cholesky_mode ? 1 : 0));
  chkpt.write(P+"Nbf",  (hsize_t) Nbf);
  chkpt.write(P+"Naux", (hsize_t) Naux);
  chkpt.write(P+"Nnuc", (hsize_t) Nnuc);
  chkpt.write(P+"omega", omega);
  chkpt.write(P+"alpha", alpha);
  chkpt.write(P+"beta",  beta);
  chkpt.write(P+"maxam",        maxam);
  chkpt.write(P+"maxcontr",     maxcontr);
  chkpt.write(P+"maxorbam",     maxorbam);
  chkpt.write(P+"maxorbcontr",  (hsize_t) maxorbcontr);
  chkpt.write(P+"maxauxam",     maxauxam);
  chkpt.write(P+"maxauxcontr",  (hsize_t) maxauxcontr);

  // DF keeps the (a|b) metric and its (half-)inverse; CD bakes the
  // metric into the L blocks and keeps only X for the force kernels.
  if(cholesky_mode) {
    chkpt.write(P+"cd_X", cd_X);
  } else {
    chkpt.write(P+"ab",      ab);
    chkpt.write(P+"ab_inv",  ab_inv);
    chkpt.write(P+"ab_invh", ab_invh);
  }

  // orbpair (is, js) -- the rest is recomputed from the basis on load.
  std::vector<hsize_t> orb_is(orbpairs.size()), orb_js(orbpairs.size());
  for(size_t i=0; i<orbpairs.size(); i++) {
    orb_is[i] = orbpairs[i].is;
    orb_js[i] = orbpairs[i].js;
  }
  chkpt.write(P+"orb_is", orb_is);
  chkpt.write(P+"orb_js", orb_js);

  // CachedBlocks flat backing storage.
  chkpt.write(P+"storage", cached->storage());

  if(cholesky_mode) {
    chkpt.write(P+"cd_pivot_index", umat_to_hsize(cd_pivot_index));
    chkpt.write(P+"cd_pivot_sentinel", (hsize_t) cd_pivot_sentinel);
    std::vector<hsize_t> piv_is(cd_pivot_shellpairs_vec.size());
    std::vector<hsize_t> piv_js(cd_pivot_shellpairs_vec.size());
    for(size_t i=0; i<cd_pivot_shellpairs_vec.size(); i++) {
      piv_is[i] = cd_pivot_shellpairs_vec[i].first;
      piv_js[i] = cd_pivot_shellpairs_vec[i].second;
    }
    chkpt.write(P+"pivot_sp_is", piv_is);
    chkpt.write(P+"pivot_sp_js", piv_js);
  }
}

bool DensityFit::load(const BasisSet & basis, const BasisSet * auxbas, const std::string & fname) {
  if(!file_exists(fname)) return false;
  Checkpoint chkpt(fname, false);
  const std::string P = df_cache_prefix(omega, alpha, beta);
  if(!chkpt.exist(P+"Nbf")) return false;

  // Read header and validate match against the caller-supplied basis.
  hsize_t Nbf_in, Naux_in, Nnuc_in;
  chkpt.read(P+"Nbf",  Nbf_in);
  chkpt.read(P+"Naux", Naux_in);
  chkpt.read(P+"Nnuc", Nnuc_in);
  if(Nbf_in != basis.get_Nbf() || Nnuc_in != basis.get_Nnuc())
    return false;

  int chol_mode_in;
  chkpt.read(P+"cholesky_mode", chol_mode_in);
  const bool want_cd = (auxbas == nullptr);
  if(want_cd != (chol_mode_in != 0))
    return false;
  if(auxbas && Naux_in != auxbas->get_Nbf())
    return false;

  // Commit to populating *this -- past this point we mutate state.
  cholesky_mode = (chol_mode_in != 0);
  Nbf  = Nbf_in;
  Naux = Naux_in;
  Nnuc = Nnuc_in;
  direct = false;
  chkpt.read(P+"omega", omega);
  chkpt.read(P+"alpha", alpha);
  chkpt.read(P+"beta",  beta);
  chkpt.read(P+"maxam",       maxam);
  chkpt.read(P+"maxcontr",    maxcontr);
  chkpt.read(P+"maxorbam",    maxorbam);
  hsize_t maxorbcontr_in, maxauxcontr_in;
  chkpt.read(P+"maxorbcontr", maxorbcontr_in);
  chkpt.read(P+"maxauxam",    maxauxam);
  chkpt.read(P+"maxauxcontr", maxauxcontr_in);
  maxorbcontr = maxorbcontr_in;
  maxauxcontr = maxauxcontr_in;

  if(cholesky_mode) {
    chkpt.read(P+"cd_X", cd_X);
    ab.reset(); ab_inv.reset(); ab_invh.reset();
  } else {
    chkpt.read(P+"ab",      ab);
    chkpt.read(P+"ab_inv",  ab_inv);
    chkpt.read(P+"ab_invh", ab_invh);
    cd_X.reset();
  }

  orbshells = basis.get_shells();
  auxshells = auxbas ? auxbas->get_shells() : std::vector<GaussianShell>();
  dummy = dummyshell();

  std::vector<hsize_t> orb_is, orb_js;
  chkpt.read(P+"orb_is", orb_is);
  chkpt.read(P+"orb_js", orb_js);
  orbpairs.assign(orb_is.size(), eripair_t{});
  for(size_t i=0; i<orb_is.size(); i++) {
    orbpairs[i].is = orb_is[i];
    orbpairs[i].js = orb_js[i];
    orbpairs[i].i0 = orbshells[orb_is[i]].get_first_ind();
    orbpairs[i].j0 = orbshells[orb_js[i]].get_first_ind();
    orbpairs[i].Ni = orbshells[orb_is[i]].get_Nbf();
    orbpairs[i].Nj = orbshells[orb_js[i]].get_Nbf();
    orbpairs[i].eri = 0.0;  // screening field, unused after fill
  }

  // Rebuild the CachedBlocks descriptor and slot the saved storage_
  // back in. build_shellpair_descriptor needs orbpairs and orbshells
  // populated -- both set above.
  std::vector<std::pair<size_t, size_t>> sp_pairs, sp_firsts, sp_sizes;
  build_shellpair_descriptor(sp_pairs, sp_firsts, sp_sizes);
  auto cached = std::make_shared<CachedBlocks>(Nbf, Naux, sp_pairs, sp_firsts, sp_sizes);
  std::vector<double> storage;
  chkpt.read(P+"storage", storage);
  if(storage.size() != cached->storage_size()) {
    // Shouldn't happen if Nbf and orbpairs matched, but bail out
    // rather than corrupt *this.
    return false;
  }
  cached->storage() = std::move(storage);
  blocks = cached;

  if(cholesky_mode) {
    hsize_t piv_sentinel_in;
    std::vector<hsize_t> piv_index, piv_is, piv_js;
    chkpt.read(P+"cd_pivot_index",    piv_index);
    chkpt.read(P+"cd_pivot_sentinel", piv_sentinel_in);
    chkpt.read(P+"pivot_sp_is",       piv_is);
    chkpt.read(P+"pivot_sp_js",       piv_js);
    cd_pivot_index     = hsize_to_umat(piv_index, Nbf, Nbf);
    cd_pivot_sentinel  = piv_sentinel_in;
    cd_pivot_shellpairs_vec.resize(piv_is.size());
    pivot_shellpairs.clear();
    for(size_t i=0; i<piv_is.size(); i++) {
      cd_pivot_shellpairs_vec[i] = std::make_pair((size_t) piv_is[i], (size_t) piv_js[i]);
      pivot_shellpairs.insert(cd_pivot_shellpairs_vec[i]);
    }
  } else {
    cd_pivot_index.reset();
    cd_pivot_shellpairs_vec.clear();
    pivot_shellpairs.clear();
  }

  return true;
}
