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
#include "linalg.h"
#include "scf.h"
#include "stringutil.h"
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

// \delta parameter in Eichkorn et al
#define DELTA 1e-9

DensityFit::DensityFit() {
  omega=0.0;
  alpha=1.0;
  beta=0.0;
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

void DensityFit::accumulate_K_from_blocks(const arma::mat & C, const arma::vec & occs, arma::mat & K) const {
  if(C.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", C.n_rows = " << C.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // Find maximum number of functions per shell
  size_t Nmax=0;
  for(size_t is=0;is<orbshells.size();is++)
    Nmax=std::max(Nmax, orbshells[is].get_Nbf());

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Helper array (allocated once per thread, resized+zeroed per
    // orbital below: aui = ab_invh.t()*aui at the bottom of the loop
    // shrinks aui from (Naux, Nbf) to (Naux_eff, Nbf), and the next
    // iteration must start at (Naux, Nbf) again or the per-shellpair
    // (Naux, Nmu) accumulation hits a shape mismatch).
    arma::mat aui;

    // Helper memory
    std::vector<double> scratch(Naux*Nmax*Nmax,0);
    std::vector<double> scratch2(Naux*Nmax*Nmax,0);

    // a_munu contains (a,uv). We want to build the exchange
    // K_uv = \sum_i n_i (ui|vi) = \sum_i n_i (a|ui) (a|b)^-1 (b|vi)
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t io=0;io<C.n_cols;io++) {
      aui.zeros(Naux,Nbf);
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	// Shells in question are
	size_t imus=orbpairs[ip].is;
	size_t inus=orbpairs[ip].js;
	// First function on shell
	size_t mu0=orbshells[imus].get_first_ind();
	size_t nu0=orbshells[inus].get_first_ind();
	// Amount of functions
	size_t Nmu=orbshells[imus].get_Nbf();
	size_t Nnu=orbshells[inus].get_Nbf();
	// Block view (Naux x Nmu*Nnu); aux_mem view, cheap to bind once.
	arma::mat amunu = blocks->get_block(ip);

	// amunu is stored in the form amunu(a,nu*Nmu+mu) =
	// amunu[(nu*Nmu+mu)*Naux+a], so we can reshape it to a
	// (Naux*Nmu,Nnu) matrix. Half-transformed (a|u;i) is then
        {
          arma::mat ui(&(scratch[0]),Naux*Nmu,1,false,true);
          ui=arma::reshape(amunu,Naux*Nmu,Nnu)*C.submat(nu0,io,nu0+Nnu-1,io);
          ui.reshape(Naux,Nmu);
          aui.cols(mu0,mu0+Nmu-1)+=ui;
        }

	if(imus != inus) {
	  // Get (a|vu)
	  arma::mat anumu(&(scratch[0]),Naux,Nmu*Nnu,false,true);
	  anumu.zeros();
	  for(size_t mu=0;mu<Nmu;mu++)
	    for(size_t nu=0;nu<Nnu;nu++)
	      anumu.col(mu*Nnu+nu)=amunu.col(nu*Nmu+mu);

	  // Half-transformed (a|v;i) is
	  arma::mat vi(&(scratch2[0]),Naux*Nnu,1,false,true);
          vi=arma::reshape(anumu,Naux*Nnu,Nmu)*C.submat(mu0,io,mu0+Nmu-1,io);
	  vi.reshape(Naux,Nnu);
	  aui.cols(nu0,nu0+Nnu-1)+=vi;
        }
      }
      // K_uv = (ui|vi) = (a|ui) (a|b)^-1 (b|vi)
      // ab_invh is the canonical-orth half-inverse X = U Λ^{-1/2};
      // X^T (a|b) X = I, so (a|b)^{-1} ≈ X X^T and the half-transform
      // is X^T aui (not X aui — X is not symmetric in general).
      aui = ab_invh.t()*aui;
      arma::mat K_io = occs[io]*arma::trans(aui)*aui;
#ifdef _OPENMP
#pragma omp critical
#endif
      K += K_io;
    }
  }
}

void DensityFit::accumulate_K_from_blocks(const arma::cx_mat & C, const arma::vec & occs, arma::cx_mat & K) const {
  if(C.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", C.n_rows = " << C.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // a_munu contains (a,uv). We want to build the exchange
  // K_uv = \sum_i n_i (ui|vi) = \sum_i n_i (a|ui) (a|b)^-1 (b|vi)
  for(size_t io=0;io<C.n_cols;io++) {
    // Helper array
    arma::cx_mat aui(Naux,Nbf);
    aui.zeros();

    // Fill integrals
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      arma::cx_mat hlp(aui);
#pragma omp for schedule(dynamic)
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	// Shells in question are
	size_t imus=orbpairs[ip].is;
	size_t inus=orbpairs[ip].js;
	// First function on shell
	size_t mu0=orbshells[imus].get_first_ind();
	size_t nu0=orbshells[inus].get_first_ind();
	// Amount of functions
	size_t Nmu=orbshells[imus].get_Nbf();
	size_t Nnu=orbshells[inus].get_Nbf();
	// Block view (Naux x Nmu*Nnu); aux_mem view, cheap to bind once.
	arma::mat amunu = blocks->get_block(ip);

	// amunu is stored in the form amunu(a,nu*Nmu+mu) =
	// amunu[(nu*Nmu+mu)*Naux+a], so we can reshape it to a
	// (Naux*Nmu,Nnu) matrix. Half-transformed (a|u;i) is then
	arma::cx_mat ui(arma::reshape(amunu,Naux*Nmu,Nnu)*C.submat(nu0,io,nu0+Nnu-1,io));
	ui.reshape(Naux,Nmu);

#ifdef _OPENMP
	hlp.cols(mu0,mu0+Nmu-1)+=ui;
#else
	aui.cols(mu0,mu0+Nmu-1)+=ui;
#endif

	if(imus != inus) {
	  // Get (a|vu)
	  arma::mat anumu(Naux,Nmu*Nnu);
	  anumu.zeros();
	  for(size_t mu=0;mu<Nmu;mu++)
	    for(size_t nu=0;nu<Nnu;nu++)
	      anumu.col(mu*Nnu+nu)=amunu.col(nu*Nmu+mu);

	  // Half-transformed (a|v;i) is
	  arma::cx_mat vi(arma::reshape(anumu,Naux*Nnu,Nmu)*C.submat(mu0,io,mu0+Nmu-1,io));
	  vi.reshape(Naux,Nnu);

#ifdef _OPENMP
	  hlp.cols(nu0,nu0+Nnu-1)+=vi;
#else
	  aui.cols(nu0,nu0+Nnu-1)+=vi;
#endif
	}
      }
#ifdef _OPENMP
#pragma omp critical
      aui+=hlp;
#endif
    }

    // K_uv = (ui|vi) = (a|ui) (a|b)^-1 (b|vi)
    aui = ab_invh.t()*aui;
    K += occs[io]*arma::trans(aui)*aui;
  }
}

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

  // Compute and return c
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

  // Compute and return c
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
  // no need to forbid direct here.

  // Collect AO integrals
  ints.zeros(Nbf*Nbf,Naux);
  for(size_t ip=0;ip<orbpairs.size();ip++) {
    size_t imus=orbpairs[ip].is;
    size_t inus=orbpairs[ip].js;
    size_t Nmu=orbshells[imus].get_Nbf();
    size_t Nnu=orbshells[inus].get_Nbf();
    size_t mu0=orbshells[imus].get_first_ind();
    size_t nu0=orbshells[inus].get_first_ind();

    arma::mat amunu = blocks->get_block(ip);

    for(size_t ias=0;ias<auxshells.size();ias++) {
      size_t Na=auxshells[ias].get_Nbf();
      size_t a0=auxshells[ias].get_first_ind();

      for(size_t imu=0;imu<Nmu;imu++)
	for(size_t inu=0;inu<Nnu;inu++)
	  for(size_t ia=0;ia<Na;ia++) {
	    size_t mu=imu+mu0;
	    size_t nu=inu+nu0;
	    size_t a=ia+a0;

            double el(amunu(ia+a0,inu*Nmu+imu));
	    ints(mu*Nbf+nu,a)=el;
	    ints(nu*Nbf+mu,a)=el;
	  }
    }
  }
}

void DensityFit::B_matrix(arma::mat & B) const {
  // three_center_integrals + the metric multiply now works in
  // either cached or direct mode; the latter recomputes the
  // shellpair blocks on demand via libint inside the iteration.
  three_center_integrals(B);
  B*=ab_invh;
}
