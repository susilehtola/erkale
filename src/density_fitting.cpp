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
#include "eriworker.h"
#include "linalg.h"
#include "mathf.h"
#include "scf.h"
#include "stringutil.h"
#include "timer.h"
#include <cstdio>
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
  cholesky_mode_=false;
  cd_pivot_sentinel_=0;
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

size_t DensityFit::fill(const BasisSet & orbbas, const BasisSet & auxbas, bool dir, double erithr, double linthr, double cholthr) {
  // Construct density fitting basis
  cholesky_mode_=false;
  cd_pivot_index_.reset();
  cd_pivot_shellpairs_vec_.clear();
  pivot_shellpairs_.clear();

  // Store amount of functions
  Nbf=orbbas.get_Nbf();
  Naux=auxbas.get_Nbf();
  Nnuc=orbbas.get_Nnuc();
  direct=dir;

  // Fill list of shell pairs
  orbpairs=orbbas.compute_screening(erithr).shpairs;

  // Get orbital shells, auxiliary shells and dummy shell
  orbshells=orbbas.get_shells();
  auxshells=auxbas.get_shells();
  dummy=dummyshell();

  maxorbam=orbbas.get_max_am();
  maxauxam=auxbas.get_max_am();
  maxorbcontr=orbbas.get_max_Ncontr();
  maxauxcontr=auxbas.get_max_Ncontr();
  maxam=std::max(orbbas.get_max_am(),auxbas.get_max_am());
  maxcontr=std::max(orbbas.get_max_Ncontr(),auxbas.get_max_Ncontr());

  // First, compute the two-center integrals
  ab.zeros(Naux,Naux);

  // Get list of unique auxiliary shell pairs
  std::vector<shellpair_t> auxpairs=auxbas.get_unique_shellpairs();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker *eri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(maxam,maxcontr);
    else
      eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);
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

    delete eri;
  }

  ab_invh = PartialCholeskyOrth(ab, cholthr, linthr);
  ab_inv = ab_invh * ab_invh.t();

  // Build the per-shellpair block descriptor; the same descriptor
  // feeds either the cached or direct BTensorBlocks subclass below.
  std::vector<std::pair<size_t, size_t>> sp_pairs(orbpairs.size());
  std::vector<std::pair<size_t, size_t>> sp_firsts(orbpairs.size());
  std::vector<std::pair<size_t, size_t>> sp_sizes(orbpairs.size());
  for(size_t ip=0;ip<orbpairs.size();ip++) {
    const size_t imus=orbpairs[ip].is;
    const size_t inus=orbpairs[ip].js;
    sp_pairs[ip] = std::make_pair(imus, inus);
    sp_firsts[ip] = std::make_pair(orbshells[imus].get_first_ind(), orbshells[inus].get_first_ind());
    sp_sizes[ip] = std::make_pair(orbshells[imus].get_Nbf(), orbshells[inus].get_Nbf());
  }

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
      ERIWorker *eri;
      if(omega==0.0 && alpha==1.0 && beta==0.0)
	eri=new ERIWorker(maxam,maxcontr);
      else
	eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	// Write straight into the CachedBlocks-owned slot for this ip.
	arma::mat slot = cached->block_mut(ip);
	(void) compute_a_munu(eri, ip, slot.memptr());
      }

      delete eri;
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
// enumeration, pivoted selection). Populates the by-ref output
// parameters with the pivoting machinery. b_raw_out non-null ->
// save each pivot's (mu nu | piv) column; nullptr -> compute and
// discard (used by find_cholesky_pivots when the caller only needs
// the pivot set). Returns the number of pivots selected.
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
                                          arma::umat & prodmap,
                                          arma::uvec & prodidx,
                                          arma::uvec & odiagidx,
                                          std::set<std::pair<size_t, size_t>> & piv_shellpairs,
                                          arma::mat * b_raw_out) const {
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
    ERIWorker *eri;
    const std::vector<double> * erip;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
    else
      eri=new ERIWorker_srlr(basis.get_max_am(),basis.get_max_Ncontr(),omega,alpha,beta);

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
    delete eri;
  }
  t_int+=t.get();

  // Phase B: enumerate orbital pairs into prodidx / invmap / prodmap / odiagidx.
  {
    prodmap.ones(Nbf_local,Nbf_local);
    prodmap*=-1;
    size_t iprod=0;
    size_t iodiag=0;
    prodidx.resize(Nbf_local*Nbf_local);
    odiagidx.resize(Nbf_local*Nbf_local);
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
            size_t idx=i*Nbf_local+j;
            prodidx(iprod)=idx;
            invmap(0,iprod)=i;
            invmap(1,iprod)=j;
            odiagidx(iodiag)=iprod;
            prodmap(i,j)=iprod;
            prodmap(j,i)=iprod;
            iprod++;
            iodiag++;
          }
          size_t idx=i*Nbf_local+i;
          prodidx(iprod)=idx;
          invmap(0,iprod)=i;
          invmap(1,iprod)=i;
          prodmap(i,i)=iprod;
          iprod++;
        }
      } else {
        for(size_t i=i0;i<i0+Ni;i++)
          for(size_t j=j0;j<j0+Nj;j++) {
            size_t idx=i*Nbf_local+j;
            prodidx(iprod)=idx;
            invmap(0,iprod)=i;
            invmap(1,iprod)=j;
            odiagidx(iodiag)=iprod;
            prodmap(i,j)=iprod;
            prodmap(j,i)=iprod;
            iprod++;
            iodiag++;
          }
      }
    }
    prodidx.resize(iprod);
    odiagidx.resize(iodiag);
    if(iprod<invmap.n_cols-1)
      invmap.shed_cols(iprod,invmap.n_cols-1);
  }

  if(verbose) {
    printf("Two-step CD: screening reduced dofs by factor %.2f.\n",d.n_elem*1.0/prodidx.n_elem);
    fflush(stdout);
  }

  d=d(prodidx);
  double error(arma::max(d));

  // Phase C: pivot selection.
  pi=arma::linspace<arma::uvec>(0,d.n_elem-1,d.n_elem);
  arma::mat B_temp;
  B_temp.zeros(100,prodidx.n_elem);
  if(b_raw_out) {
    b_raw_out->set_size(prodidx.n_elem, 100);
    b_raw_out->zeros();
  }
  size_t m=0;

  while(error>cholesky_tol && m<d.n_elem) {
    {
      arma::uword best=m;
      double bestval=d(pi(m));
      for(arma::uword i=m+1;i<d.n_elem;i++) {
        const double v=d(pi(i));
        if(v>bestval) { bestval=v; best=i; }
      }
      if(best!=m)
        std::swap(pi(m),pi(best));
    }
    size_t pim=pi(m);

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
      ERIWorker *eri;
      const std::vector<double> * erip;
      if(omega==0.0 && alpha==1.0 && beta==0.0)
        eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
      else
        eri=new ERIWorker_srlr(basis.get_max_am(),basis.get_max_Ncontr(),omega,alpha,beta);

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
      delete eri;
    }
    t_int+=t.get();
    t.set();

    while(true) {
      double errmax=d(pi(m));
      for(size_t i=m+1;i<d.n_elem;i++) {
        const double v=d(pi(i));
        if(v>errmax) errmax=v;
      }
      double blockerr=0;
      size_t blockind=0;
      size_t Aind=0;
      for(size_t kk=0;kk<max_Nk;kk++)
        for(size_t ll=0;ll<max_Nl;ll++) {
          size_t k=kk+max_k0;
          size_t l=ll+max_l0;
          size_t ind=prodmap(k,l);
          if(ind>Nbf_local*Nbf_local) continue;
          if(d(ind)>blockerr) {
            bool found=false;
            for(size_t i=0;i<m;i++)
              if(pi(i)==ind) { found=true; break; }
            if(!found) {
              Aind=kk*max_Nl+ll;
              blockind=ind;
              blockerr=d(ind);
            }
          }
        }
      if(blockerr==0.0 || blockerr<shell_reuse_thr*errmax)
        break;

      if(pi(m)!=blockind) {
        bool found=false;
        for(size_t i=m+1;i<pi.n_elem;i++)
          if(pi(i)==blockind) {
            found=true;
            std::swap(pi(i),pi(m));
            break;
          }
        if(!found) {
          std::ostringstream oss;
          oss << "Pivot index " << blockind << " not found, m = " << m << " !\n";
          throw std::logic_error(oss.str());
        }
      }
      pim=pi(m);

      if(m>=B_temp.n_rows)
        B_temp.resize(B_temp.n_rows+100, B_temp.n_cols);
      if(b_raw_out && m>=b_raw_out->n_cols)
        b_raw_out->resize(b_raw_out->n_rows, b_raw_out->n_cols+100);

      if(b_raw_out)
        b_raw_out->col(m) = A.col(Aind);

      B_temp(m,pim)=sqrt(d(pim));
      if(m==0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i=m+1;i<d.n_elem;i++) {
          size_t pii=pi(i);
          B_temp(m,pii)=A(pii,Aind)/B_temp(m,pim);
          d(pii)-=B_temp(m,pii)*B_temp(m,pii);
        }
      } else {
        const arma::vec tdot(arma::trans(B_temp.submat(0,0,m-1,B_temp.n_cols-1))*B_temp.submat(0,pim,m-1,pim));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i=m+1;i<d.n_elem;i++) {
          size_t pii=pi(i);
          B_temp(m,pii)=(A(pii,Aind)-tdot(pii))/B_temp(m,pim);
          d(pii)-=B_temp(m,pii)*B_temp(m,pii);
        }
      }

      m++;
    }
    error=0.0;
    for(size_t i=m;i<pi.n_elem;i++) {
      const double v=d(pi(i));
      if(v>error) error=v;
    }
    t_chol+=t.get();
  }

  const size_t Nselected=m;
  pi=pi.subvec(0,Nselected-1);
  if(b_raw_out && Nselected<b_raw_out->n_cols)
    b_raw_out->shed_cols(Nselected,b_raw_out->n_cols-1);
  B_temp.reset();

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
                                                                      bool verbose) {
  // Pivots-only: run phases A-C with no (mu nu | piv) column save
  // and no metric / B construction. Used by basislibrary for atom-CD
  // aux-basis construction.
  DensityFit cd;
  arma::uvec pi;
  arma::umat invmap, prodmap;
  arma::uvec prodidx, odiagidx;
  std::set<std::pair<size_t, size_t>> piv_shellpairs;
  cd.select_two_step_pivots(basis, cholesky_tol, shell_reuse_thr, shell_screen_tol,
                            verbose, pi, invmap, prodmap, prodidx, odiagidx,
                            piv_shellpairs, nullptr);
  return piv_shellpairs;
}

size_t DensityFit::fill_cholesky(const BasisSet & basis,
                                 double cholesky_tol,
                                 double shell_reuse_thr,
                                 double shell_screen_tol,
                                 double fit_cholesky_thr,
                                 bool verbose) {
  // Two-step CD = density fitting with an aux basis chosen by
  // pivoted Cholesky on orbital products. DF and CD share the
  // downstream J/K/forceJ kernels; only the aux selection differs.
  cholesky_mode_ = true;
  Nbf  = basis.get_Nbf();
  Nnuc = basis.get_Nnuc();
  direct = false;

  orbshells   = basis.get_shells();
  auxshells.clear();
  dummy       = dummyshell();
  maxorbam    = basis.get_max_am();
  maxorbcontr = basis.get_max_Ncontr();
  maxauxam    = 0;
  maxauxcontr = 0;
  maxam       = maxorbam;
  maxcontr    = maxorbcontr;

  Timer ttot;

  // Phases A-C: pivot selection. Also gives us B_raw = (mu nu | piv).
  arma::uvec pi;
  arma::umat invmap, prodmap;
  arma::uvec prodidx, odiagidx;
  arma::mat B_raw;
  const size_t Nselected = select_two_step_pivots(basis, cholesky_tol, shell_reuse_thr, shell_screen_tol,
                                                  verbose, pi, invmap, prodmap, prodidx, odiagidx,
                                                  pivot_shellpairs_, &B_raw);
  Naux = Nselected;
  cd_pivot_shellpairs_vec_.assign(pivot_shellpairs_.begin(), pivot_shellpairs_.end());

  // (mu, nu) -> pivot rank lookup for forceJ_cholesky.
  cd_pivot_sentinel_ = Nselected;
  cd_pivot_index_.set_size(Nbf, Nbf);
  cd_pivot_index_.fill(cd_pivot_sentinel_);
  for(arma::uword p=0; p<Nselected; p++) {
    const arma::uword pii = pi(p);
    cd_pivot_index_(invmap(0,pii), invmap(1,pii)) = p;
    cd_pivot_index_(invmap(1,pii), invmap(0,pii)) = p;
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
    ERIWorker *eri;
    const std::vector<double> * erip;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());
    else
      eri=new ERIWorker_srlr(basis.get_max_am(),basis.get_max_Ncontr(),omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t ip=0; ip<cd_pivot_shellpairs_vec_.size(); ip++) {
      const size_t is = cd_pivot_shellpairs_vec_[ip].first;
      const size_t js = cd_pivot_shellpairs_vec_[ip].second;
      const size_t Ni = shells[is].get_Nbf();
      const size_t Nj = shells[js].get_Nbf();
      const size_t i0 = shells[is].get_first_ind();
      const size_t j0 = shells[js].get_first_ind();
      for(size_t jp=0; jp<=ip; jp++) {
        const size_t ks = cd_pivot_shellpairs_vec_[jp].first;
        const size_t ls = cd_pivot_shellpairs_vec_[jp].second;
        const size_t Nk = shells[ks].get_Nbf();
        const size_t Nl = shells[ls].get_Nbf();
        const size_t k0 = shells[ks].get_first_ind();
        const size_t l0 = shells[ls].get_first_ind();

        eri->compute(&shells[is], &shells[js], &shells[ks], &shells[ls]);
        erip = eri->getp();
        for(size_t ii=0; ii<Ni; ii++)
          for(size_t jj=0; jj<Nj; jj++) {
            const arma::uword pidx = cd_pivot_index_(i0+ii, j0+jj);
            if(pidx == cd_pivot_sentinel_) continue;
            for(size_t kk=0; kk<Nk; kk++)
              for(size_t ll=0; ll<Nl; ll++) {
                const arma::uword qidx = cd_pivot_index_(k0+kk, l0+ll);
                if(qidx == cd_pivot_sentinel_) continue;
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
    delete eri;
  }
  double t_int_de = t.get();

  // Phase E: orthogonalise the metric. PartialCholeskyOrth gives X
  // with X^T M X = I over the lindep-cleaned subspace, so
  // (a|b)^{-1} ≈ X X^T. Standard DF formulation.
  t.set();
  ab = std::move(M_metric);
  ab_invh = PartialCholeskyOrth(ab, fit_cholesky_thr, 0.0);
  ab_inv = ab_invh * ab_invh.t();
  double t_chol_de = t.get();

  if(verbose) {
    printf("Two-step CD: pivot metric orthogonalisation reduced %i -> %i functions (%s).\n",
           (int) Nselected, (int) ab_invh.n_cols, t.elapsed().c_str());
    fflush(stdout);
  }

  // Build the CachedBlocks containing (piv_p | mu nu) per orbital
  // shellpair. The B_raw rows are indexed by orbital products
  // (prodidx layout); we reshape to the same per-shellpair block
  // layout DensityFit::fill uses for DF. Pairs ERIchol's pivoting
  // dropped become zero-filled rows -- that's the right answer
  // (their (mu nu | piv) integrals are below threshold).
  orbpairs = basis.compute_screening(shell_screen_tol).shpairs;
  const arma::uword sentinel = invmap.n_cols;
  arma::umat prodlookup(Nbf, Nbf);
  prodlookup.fill(sentinel);
  for(arma::uword i=0; i<invmap.n_cols; i++) {
    const arma::uword mu = invmap(0, i);
    const arma::uword nu = invmap(1, i);
    prodlookup(mu, nu) = i;
    prodlookup(nu, mu) = i;
  }

  std::vector<std::pair<size_t, size_t>> sp_pairs(orbpairs.size());
  std::vector<std::pair<size_t, size_t>> sp_firsts(orbpairs.size());
  std::vector<std::pair<size_t, size_t>> sp_sizes(orbpairs.size());
  for(size_t ip=0; ip<orbpairs.size(); ip++) {
    const size_t imus = orbpairs[ip].is;
    const size_t inus = orbpairs[ip].js;
    sp_pairs[ip]  = std::make_pair(imus, inus);
    sp_firsts[ip] = std::make_pair(orbshells[imus].get_first_ind(), orbshells[inus].get_first_ind());
    sp_sizes[ip]  = std::make_pair(orbshells[imus].get_Nbf(), orbshells[inus].get_Nbf());
  }
  auto cached = std::make_shared<CachedBlocks>(Nbf, Naux, sp_pairs, sp_firsts, sp_sizes);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0; ip<orbpairs.size(); ip++) {
    arma::mat slot = cached->block_mut(ip);    // (Naux x Nmu*Nnu), zero-init
    const size_t imus = orbpairs[ip].is;
    const size_t inus = orbpairs[ip].js;
    const size_t mu0  = orbshells[imus].get_first_ind();
    const size_t nu0  = orbshells[inus].get_first_ind();
    const size_t Nmu  = orbshells[imus].get_Nbf();
    const size_t Nnu  = orbshells[inus].get_Nbf();

    for(size_t inu=0; inu<Nnu; inu++) {
      for(size_t imu=0; imu<Nmu; imu++) {
        const arma::uword prow = prodlookup(mu0+imu, nu0+inu);
        if(prow == sentinel) continue;  // pair below CD significance
        slot.col(inu*Nmu + imu) = B_raw.row(prow).t();
      }
    }
  }

  blocks = cached;

  if(verbose) {
    printf("Two-step CD finished in %s.\n", ttot.elapsed().c_str());
    if(t_int_de+t_chol_de > 0.0)
      printf("Metric build / orthogonalisation time use: integrals %3.1f %%, linear algebra %3.1f %%.\n",
             100*t_int_de/(t_int_de+t_chol_de),100*t_chol_de/(t_int_de+t_chol_de));
    fflush(stdout);
  }

  return orbpairs.size();
}

arma::vec DensityFit::forceJ_cholesky(const BasisSet & basis, const arma::mat & P) const {
  if(!cholesky_mode_)
    throw std::runtime_error("DensityFit::forceJ_cholesky requires fill_cholesky to have been called.\n");
  if(P.n_rows != Nbf || P.n_cols != Nbf)
    throw std::runtime_error("DensityFit::forceJ_cholesky: density matrix dimension mismatch.\n");

  // c is the standard DF expansion coefficient in the pivot-orbital
  // basis: c = ab_inv * gamma_pivot = (X X^T) * gamma_pivot, with
  // gamma_pivot_p = (piv_p | density). compute_expansion handles
  // both the contraction and the metric-inverse multiplication.
  const arma::vec c = compute_expansion(P);

  const std::vector<GaussianShell> & shells = basis.get_shells_ref();
  const int max_am   = basis.get_max_am();
  const int max_ncon = basis.get_max_Ncontr();

  arma::vec f(3 * Nnuc);
  f.zeros();

  // Part 1: f += 0.5 c (dM/dR) c on pivot shellpairs.
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    dERIWorker * deri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      deri = new dERIWorker(max_am, max_ncon);
    else
      deri = new dERIWorker_srlr(max_am, max_ncon, omega, alpha, beta);

#ifdef _OPENMP
    arma::vec fwrk(f); fwrk.zeros();
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0; ip<cd_pivot_shellpairs_vec_.size(); ip++) {
      const size_t is = cd_pivot_shellpairs_vec_[ip].first;
      const size_t js = cd_pivot_shellpairs_vec_[ip].second;
      const size_t Ni = shells[is].get_Nbf();
      const size_t Nj = shells[js].get_Nbf();
      const size_t i0 = shells[is].get_first_ind();
      const size_t j0 = shells[js].get_first_ind();
      const size_t i_at = shells[is].get_center_ind();
      const size_t j_at = shells[js].get_center_ind();

      for(size_t jp=0; jp<=ip; jp++) {
        const size_t ks = cd_pivot_shellpairs_vec_[jp].first;
        const size_t ls = cd_pivot_shellpairs_vec_[jp].second;
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
          const int center = ic / 3;
          const int xyz    = ic % 3;
          const size_t aA  = atoms[center];

          const std::vector<double> * erip = deri->getp(ic);
          double accum = 0.0;
          for(size_t ii=0; ii<Ni; ii++)
            for(size_t jj=0; jj<Nj; jj++) {
              const arma::uword pidx = cd_pivot_index_(i0+ii, j0+jj);
              if(pidx == cd_pivot_sentinel_) continue;
              for(size_t kk=0; kk<Nk; kk++)
                for(size_t ll=0; ll<Nl; ll++) {
                  const arma::uword qidx = cd_pivot_index_(k0+kk, l0+ll);
                  if(qidx == cd_pivot_sentinel_) continue;
                  const double val = (*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
                  accum += val * c(pidx) * c(qidx);
                }
            }
#ifdef _OPENMP
          fwrk(3*aA + xyz) += fac_sp * accum;
#else
          f(3*aA + xyz)    += fac_sp * accum;
#endif
        }
      }
    }
#ifdef _OPENMP
#pragma omp critical
    f += fwrk;
#endif
    delete deri;
  }

  // Part 2: f -= sum_munu P (d(mu nu | piv)/dR) c.
  const std::vector<eripair_t> orb_shps =
    basis.compute_screening(/*tol*/0.0, omega, alpha, beta, false).shpairs;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    dERIWorker * deri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      deri = new dERIWorker(max_am, max_ncon);
    else
      deri = new dERIWorker_srlr(max_am, max_ncon, omega, alpha, beta);

#ifdef _OPENMP
    arma::vec fwrk(f); fwrk.zeros();
#pragma omp for schedule(dynamic)
#endif
    for(size_t ipair=0; ipair<orb_shps.size(); ipair++) {
      const size_t is = orb_shps[ipair].is;
      const size_t js = orb_shps[ipair].js;
      const size_t Ni = shells[is].get_Nbf();
      const size_t Nj = shells[js].get_Nbf();
      const size_t i0 = shells[is].get_first_ind();
      const size_t j0 = shells[js].get_first_ind();
      const size_t i_at = shells[is].get_center_ind();
      const size_t j_at = shells[js].get_center_ind();

      const double fac_sp = (is == js) ? 1.0 : 2.0;

      for(size_t jp=0; jp<cd_pivot_shellpairs_vec_.size(); jp++) {
        const size_t ks = cd_pivot_shellpairs_vec_[jp].first;
        const size_t ls = cd_pivot_shellpairs_vec_[jp].second;
        const size_t Nk = shells[ks].get_Nbf();
        const size_t Nl = shells[ls].get_Nbf();
        const size_t k0 = shells[ks].get_first_ind();
        const size_t l0 = shells[ls].get_first_ind();
        const size_t k_at = shells[ks].get_center_ind();
        const size_t l_at = shells[ls].get_center_ind();

        deri->compute(&shells[is], &shells[js], &shells[ks], &shells[ls]);

        const size_t atoms[4] = {i_at, j_at, k_at, l_at};
        for(int ic=0; ic<12; ic++) {
          const int center = ic / 3;
          const int xyz    = ic % 3;
          const size_t aA  = atoms[center];

          const std::vector<double> * erip = deri->getp(ic);
          double accum = 0.0;
          for(size_t ii=0; ii<Ni; ii++)
            for(size_t jj=0; jj<Nj; jj++) {
              const double Pval = P(i0+ii, j0+jj);
              for(size_t kk=0; kk<Nk; kk++)
                for(size_t ll=0; ll<Nl; ll++) {
                  const arma::uword qidx = cd_pivot_index_(k0+kk, l0+ll);
                  if(qidx == cd_pivot_sentinel_) continue;
                  const double val = (*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
                  accum += val * Pval * c(qidx);
                }
            }
#ifdef _OPENMP
          fwrk(3*aA + xyz) -= fac_sp * accum;
#else
          f(3*aA + xyz)    -= fac_sp * accum;
#endif
        }
      }
    }
#ifdef _OPENMP
#pragma omp critical
    f += fwrk;
#endif
    delete deri;
  }

  return f;
}

double DensityFit::fitting_error() const {
  arma::mat error_matrix(maxorbam+1, maxorbam+1, arma::fill::zeros);

  // Loop over pairs
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::mat wrk_error(error_matrix);

    ERIWorker *eri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(maxam,maxcontr);
    else
      eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

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
      arma::mat auv(compute_a_munu(eri,ip));

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

void DensityFit::project_density_to_aux(const arma::mat & P, size_t ip, const arma::mat & amunu, arma::vec & gamma) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

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
      // and the half-transform is X^T aui.
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
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

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

  for(size_t ig=0;ig<P.size();ig++)
    gamma[ig]=ab_inv*gamma[ig];

  return gamma;
}

arma::mat DensityFit::calcJ(const arma::mat & P) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

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

arma::vec DensityFit::forceJ(const arma::mat & P) {
  // First, compute the expansion
  arma::vec c=compute_expansion(P);

  // The force
  arma::vec f(3*Nnuc);
  f.zeros();

  // First part: f = *#* 1/2 c_a (a|b)' c_b *#* - gamma_a' c_a
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    dERIWorker *deri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      deri=new dERIWorker(maxam,maxcontr);
    else
      deri=new dERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
    // Worker stack for each matrix
    arma::vec fwrk(f);
    fwrk.zeros();

#pragma omp for schedule(dynamic)
#endif
    for(size_t ias=0;ias<auxshells.size();ias++)
      for(size_t jas=0;jas<=ias;jas++) {

	// Symmetry factor
	double fac=0.5;
	if(ias!=jas)
	  fac*=2.0;

	size_t Na=auxshells[ias].get_Nbf();
	size_t anuc=auxshells[ias].get_center_ind();
	size_t Nb=auxshells[jas].get_Nbf();
	size_t bnuc=auxshells[jas].get_center_ind();

	if(anuc==bnuc)
	  // Contributions vanish
	  continue;

	// Compute (a|b)
	deri->compute(&auxshells[ias],&dummy,&auxshells[jas],&dummy);

	// Compute forces
	const static int index[]={0, 1, 2, 6, 7, 8};
	const size_t Nidx=sizeof(index)/sizeof(index[0]);
	double ders[Nidx];

	for(size_t iid=0;iid<Nidx;iid++) {
	  ders[iid]=0.0;
	  // Index is
	  int ic=index[iid];

	  // Increment force, anuc
	  erip=deri->getp(ic);
	  for(size_t iia=0;iia<Na;iia++) {
	    size_t ia=auxshells[ias].get_first_ind()+iia;

	    for(size_t iib=0;iib<Nb;iib++) {
	      size_t ib=auxshells[jas].get_first_ind()+iib;

	      // The integral is
	      double res=(*erip)[iia*Nb+iib];

	      ders[iid]+= res*c(ia)*c(ib);
	    }
	  }
	  ders[iid]*=fac;
	}

	// Increment forces
	for(int ic=0;ic<3;ic++) {
#ifdef _OPENMP
	  fwrk(3*anuc+ic)+=ders[ic];
	  fwrk(3*bnuc+ic)+=ders[ic+3];
#else
	  f(3*anuc+ic)+=ders[ic];
	  f(3*bnuc+ic)+=ders[ic+3];
#endif
	}
      }

#ifdef _OPENMP
#pragma omp critical
    // Sum results together
    f+=fwrk;
#endif
    delete deri;
  } // end parallel section


  // Second part: f -= gamma_a' c_a (three-center derivative term).
  // Routed through DirectDFPerturbedBlocks: for_each_pert streams
  // (perturbation, aux_first, derivative_sub_block) tuples per
  // orbital shellpair, identically to how the value-side
  // DirectDFBlocks streams (alpha | mu nu) blocks. The libcholesky
  // perturbation API uses the same shape; once we depend on
  // libcholesky this construction becomes a thin adapter.
  {
    std::vector<std::pair<size_t,size_t>> sp_pairs(orbpairs.size());
    std::vector<std::pair<size_t,size_t>> sp_firsts(orbpairs.size());
    std::vector<std::pair<size_t,size_t>> sp_sizes(orbpairs.size());
    for(size_t ip=0; ip<orbpairs.size(); ip++) {
      const size_t imus = orbpairs[ip].is;
      const size_t inus = orbpairs[ip].js;
      sp_pairs[ip]  = std::make_pair(imus, inus);
      sp_firsts[ip] = std::make_pair(orbshells[imus].get_first_ind(),
                                     orbshells[inus].get_first_ind());
      sp_sizes[ip]  = std::make_pair(orbshells[imus].get_Nbf(),
                                     orbshells[inus].get_Nbf());
    }
    DirectDFPerturbedBlocks pblocks(Nbf, Naux, Nnuc,
                                    std::move(sp_pairs),
                                    std::move(sp_firsts),
                                    std::move(sp_sizes),
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
        const size_t imus = orbpairs[ip].is;
        const size_t inus = orbpairs[ip].js;
        const size_t mu0  = orbshells[imus].get_first_ind();
        const size_t nu0  = orbshells[inus].get_first_ind();
        const size_t Nmu  = orbshells[imus].get_Nbf();
        const size_t Nnu  = orbshells[inus].get_Nbf();
        // Off-diagonal pairs are counted only once in orbpairs;
        // double-count the contribution.
        const double fac  = (imus == inus) ? 1.0 : 2.0;

        // P submatrix laid out as (mu fastest, nu slowest) so it
        // pairs with the sub_block's column index inu*Nmu+imu.
        arma::vec Psub(Nmu * Nnu);
        for(size_t inu=0; inu<Nnu; inu++)
          for(size_t imu=0; imu<Nmu; imu++)
            Psub(inu*Nmu + imu) = P(mu0+imu, nu0+inu);

        pblocks.for_each_pert(ip,
            [&](const Perturbation & pert, size_t a0, const arma::mat & sub_block) {
              // sub_block: (Na_aux x Nmu*Nnu). Contract with P over
              // (mu, nu) and with c over a; the scalar derivative is
              // accumulated into f at (atom = pert.p1, xyz = pert.p2).
              const arma::vec hlp = sub_block * Psub;
              const arma::vec ca  = c.subvec(a0, a0 + sub_block.n_rows - 1);
              const double ders   = fac * arma::dot(hlp, ca);
#ifdef _OPENMP
              fwrk(3 * pert.p1 + pert.p2) -= ders;
#else
              f(3 * pert.p1 + pert.p2)    -= ders;
#endif
            });
      }
#ifdef _OPENMP
#pragma omp critical
      f += fwrk;
#endif
    } // end parallel
  }

  return f;
}

arma::mat DensityFit::calcK(const arma::mat & Corig, const std::vector<double> & occo) const {
  if(Corig.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", Corig.n_rows = " << Corig.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // Count number of orbitals
  size_t Nmo=0;
  for(size_t i=0;i<occo.size();i++)
    if(occo[i]>0)
      Nmo++;

  // Collect orbitals to use
  arma::mat C(Nbf,Nmo);
  arma::vec occs(Nmo);
  {
    size_t io=0;
    for(size_t i=0;i<occo.size();i++)
      if(occo[i]>0) {
	// Store orbital and occupation number
	C.col(io)=Corig.col(i);
	occs[io]=occo[i];
	io++;
      }
  }

  // Returned matrix
  arma::mat K(Nbf,Nbf);
  K.zeros();

  // accumulate_K_from_blocks works against any BTensorBlocks
  // subclass; in direct mode the blocks recompute (alpha | mu nu) on
  // the fly per call. Peak memory is one (Naux x Nbf) aui per
  // thread, bounded.
  accumulate_K_from_blocks(C,occs,K);
  return K;
}

arma::cx_mat DensityFit::calcK(const arma::cx_mat & Corig, const std::vector<double> & occo) const {
  if(Corig.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", Corig.n_rows = " << Corig.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // Count number of orbitals
  size_t Nmo=0;
  for(size_t i=0;i<occo.size();i++)
    if(occo[i]>0)
      Nmo++;

  // Collect orbitals to use
  arma::cx_mat C(Nbf,Nmo);
  arma::vec occs(Nmo);
  {
    size_t io=0;
    for(size_t i=0;i<occo.size();i++)
      if(occo[i]>0) {
	// Store orbital and occupation number
	C.col(io)=Corig.col(i);
	occs[io]=occo[i];
	io++;
      }
  }

  // Returned matrix
  arma::cx_mat K(Nbf,Nbf);
  K.zeros();

  // accumulate_K_from_blocks consumes blocks->get_block(ip) uniformly; the
  // direct-mode path now works for the complex case as well.
  accumulate_K_from_blocks(C,occs,K);
  return K;
}

size_t DensityFit::get_Norb() const {
  return Nbf;
}

size_t DensityFit::get_Naux() const {
  return Naux;
}

size_t DensityFit::get_Naux_indep() const {
  // ab_invh's column count is the aux-space rank after the
  // PartialCholeskyOrth lindep cleanup of the two-center metric.
  // Same logic in DF and CD; the only difference between the two
  // is how the aux functions were chosen.
  return ab_invh.n_cols;
}

arma::mat DensityFit::get_ab() const {
  return ab;
}

arma::mat DensityFit::get_ab_inv() const {
  return ab_inv;
}

arma::mat DensityFit::get_ab_invh() const {
  return ab_invh;
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
  // blocks on demand via libint inside the iteration. CD and DF
  // share the same logic: ab_invh is the metric half-inverse, just
  // built from different aux selections.
  three_center_integrals(B);
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
    const arma::mat block_P((double*) Bdense.colptr(P), Nbf, Nbf, false, true);
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
