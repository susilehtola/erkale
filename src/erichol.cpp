/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "checkpoint.h"
#include "erichol.h"
#include "linalg.h"
#include "eriworker.h"
#include "mathf.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"

extern Settings settings;

#include <cstdio>
// For exceptions
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CHOLFILE "cholesky.chk"

ERIchol::ERIchol() {
  Nbf=0;
  omega=0.0;
  alpha=1.0;
  beta=0.0;
  two_step_=false;
}

ERIchol::~ERIchol() {
}

void ERIchol::set_range_separation(double w, double a, double b) {
  omega=w;
  alpha=a;
  beta=b;
}

void ERIchol::get_range_separation(double & w, double & a, double & b) const {
  w=omega;
  a=alpha;
  b=beta;
}

void ERIchol::load() {
  Checkpoint chkpt(CHOLFILE,false);

  // Suffix in checkpoint
  std::string suffix;
  if(omega!=1.0) {
    std::ostringstream oss;
    oss << "_" << omega;
    suffix=oss.str();
  }

  // Read B matrix
  chkpt.read("B"+suffix,B);

  // Read amount of basis functions
  {
    hsize_t Nbft;
    chkpt.read("Nbf",Nbft);
    Nbf=Nbft;
  }

  // Read product index
  {
    std::vector<hsize_t> prodidxv;
    chkpt.read("prodidx"+suffix,prodidxv);
    prodidx=arma::conv_to<arma::uvec>::from(prodidxv);
  }
  // and the off-diagonal index
  {
    std::vector<hsize_t> odiagidxv;
    chkpt.read("odiagidx"+suffix,odiagidxv);
    odiagidx=arma::conv_to<arma::uvec>::from(odiagidxv);
  }
  // the product map
  {
    std::vector<hsize_t> prodmapv;
    chkpt.read("prodmap"+suffix,prodmapv);
    prodmap=arma::reshape(arma::conv_to<arma::umat>::from(prodmapv),Nbf,Nbf);
  }
  // and the inverse map
  {
    std::vector<hsize_t> invmapv;
    chkpt.read("invmap"+suffix,invmapv);
    invmap=arma::reshape(arma::conv_to<arma::umat>::from(invmapv),2,prodidx.n_elem);
  }
}

void ERIchol::save() const {
  // Check consistency. Default is to not truncate
  bool trunc=false;
  if(file_exists(CHOLFILE)) {
    // Open in read-only mode and try to get Nbf
    Checkpoint chkpt(CHOLFILE,false);
    hsize_t Nbft;
    if(chkpt.exist("Nbf")) {
      chkpt.read("Nbf",Nbft);
      if(Nbf!=Nbft)
	trunc=true;
    } else
      trunc=true;
  }
  // Open in write mode
  Checkpoint chkpt(CHOLFILE,true,trunc);

  // Suffix in checkpoint
  std::string suffix;
  if(omega!=1.0) {
    std::ostringstream oss;
    oss << "_" << omega;
    suffix=oss.str();
  }

  // Write B matrix
  chkpt.write("B"+suffix,B);

  // Save amount of basis functions
  {
    hsize_t Nbft(Nbf);
    chkpt.write("Nbf",Nbft);
  }

  // Save product index
  {
    std::vector<hsize_t> prodidxv(arma::conv_to< std::vector<hsize_t> >::from(prodidx));
    chkpt.write("prodidx"+suffix,prodidxv);
  }
  // and the off-diagonal index
  {
    std::vector<hsize_t> odiagidxv(arma::conv_to< std::vector<hsize_t> >::from(odiagidx));
    chkpt.write("odiagidx"+suffix,odiagidxv);
  }
  // the product map
  {
    std::vector<hsize_t> prodmapv(arma::conv_to< std::vector<hsize_t> >::from(arma::vectorise(prodmap)));
    chkpt.write("prodmap"+suffix,prodmapv);
  }
  // and the inverse map
  {
    std::vector<hsize_t> invmapv(arma::conv_to< std::vector<hsize_t> >::from(arma::vectorise(invmap)));
    chkpt.write("invmap"+suffix,invmapv);
  }
}

size_t ERIchol::fill(const BasisSet & basis, double cholesky_tol, double shell_reuse_thr, double shell_screen_tol, bool verbose) {
  // Mark this object as one-step state; clear any stale two-step
  // metric data from a previous fill_two_step call.
  two_step_ = false;
  two_step_metric_.reset();
  two_step_metric_invh_.reset();
  if(cholesky_tol < shell_screen_tol) {
    fprintf(stderr,"Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
    printf("Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
  }

  // Screening matrix and pairs
  ScreeningData scr=basis.compute_screening(shell_screen_tol,omega,alpha,beta,verbose);
  const arma::mat & Q = scr.Q;
  const arma::mat & M = scr.M;
  const std::vector<eripair_t> & shpairs = scr.shpairs;

  // Amount of basis functions
  Nbf=basis.get_Nbf();
  // Shells
  std::vector<GaussianShell> shells=basis.get_shells();

  Timer t;
  Timer ttot;

  // Integral time
  double t_int=0.0;
  // Cholesky time
  double t_chol=0.0;

  // Calculate diagonal element vector
  arma::vec d(Nbf*Nbf);
  d.zeros();
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

      // We have already computed the Schwarz screening
      double QQ=Q(is,js)*Q(is,js);
      if(QQ<shell_screen_tol) {
        continue;
      }

      // Compute integrals
      eri->compute(&shells[is],&shells[js],&shells[is],&shells[js]);
      erip=eri->getp();

      // and store them
      size_t Ni(shells[is].get_Nbf());
      size_t Nj(shells[js].get_Nbf());
      size_t i0(shells[is].get_first_ind());
      size_t j0(shells[js].get_first_ind());

      for(size_t ii=0;ii<Ni;ii++)
	for(size_t jj=0;jj<Nj;jj++) {
	  size_t i=i0+ii;
	  size_t j=j0+jj;
	  d(i*Nbf+j)=(*erip)[((ii*Nj+jj)*Ni+ii)*Nj+jj];
	  d(j*Nbf+i)=d(i*Nbf+j);
	}
    }

    delete eri;
  }
  t_int+=t.get();
  t.set();

  // Amount of pairs surviving shell pair screening
  size_t Nshp=0;
  {
    prodmap.ones(Nbf,Nbf);
    prodmap*=-1; // Go to UINT_MAX

    size_t iprod=0;
    size_t iodiag=0;
    prodidx.resize(Nbf*Nbf);
    odiagidx.resize(Nbf*Nbf);
    invmap.zeros(2,Nbf*Nbf);
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
	    Nshp++;
	    // Global product index is
	    size_t idx=i*Nbf+j;
            prodidx(iprod)=idx;
            // Function indices are
            invmap(0,iprod)=i;
            invmap(1,iprod)=j;
            // Off-diagonal product
            odiagidx(iodiag)=iprod;
            // Product index mapping is
            prodmap(i,j)=iprod;
            prodmap(j,i)=iprod;
            // Increment indices
            iprod++;
            iodiag++;
	  }

	  Nshp++;
	  // Global product index is
          size_t idx=i*Nbf+i;
	  if(true || d(idx)>=shell_screen_tol) {
            prodidx(iprod)=idx;
            // Function indices are
            invmap(0,iprod)=i;
            invmap(1,iprod)=i;
	    // Product index mapping is
	    prodmap(i,i)=iprod;
            // Increment index
	    iprod++;
	  }
	}
      } else {
      	for(size_t i=i0;i<i0+Ni;i++)
	  for(size_t j=j0;j<j0+Nj;j++) {
	    Nshp++;
            size_t idx=i*Nbf+j;
            prodidx(iprod)=idx;
            // Function indices are
            invmap(0,iprod)=i;
            invmap(1,iprod)=j;
            // Product index mapping is
            prodmap(i,j)=iprod;
            // Off-diagonal product
            odiagidx(iodiag)=iprod;
              // Increment indices
            iprod++;
            iodiag++;
	  }
      }
    }
    // Resize prodidx
    prodidx.resize(iprod);
    odiagidx.resize(iodiag);
    if(iprod<invmap.n_cols-1)
      invmap.shed_cols(iprod,invmap.n_cols-1);
  }

  if(verbose) {
    printf("Screening by shell pairs and symmetry reduced dofs by factor %.2f.\n",d.n_elem*1.0/Nshp);
    printf("Individual screening reduced dofs by a total factor %.2f.\n",d.n_elem*1.0/prodidx.n_elem);
    printf("Computing Cholesky vectors. Estimated memory size is %s - %s.\n",memory_size(3*Nbf*prodidx.n_elem*sizeof(double),true).c_str(),memory_size(10*Nbf*prodidx.n_elem*sizeof(double),true).c_str());
  }

  // Drop unnecessary vectors
  d=d(prodidx);

  // Error is
  double error(arma::max(d));

  // Pivot index
  pi = arma::linspace<arma::uvec>(0,d.n_elem-1,d.n_elem);
  // Allocate memory
  B.zeros(100,prodidx.n_elem);
  // Loop index
  size_t m(0);

  while(error>cholesky_tol && m<d.n_elem) {
    // Update the pivot index. Only the maximum-error pivot at position
    // m matters for the next step; sorting the entire tail every outer
    // iteration was O(N log N) per step (O(M N log N) overall) for no
    // benefit beyond identifying the max. Find the max with index_max
    // and swap it into position m.
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

    // Pivot index to use is
    size_t pim=pi(m);
    //printf("Pivot index is %4i, corresponding to product %i, with error %e, error is %e\n",(int) pim, (int) prodidx(pim), d(pim), error);

    // Off-diagonal elements: find out which shells the pivot index
    // belongs to. The relevant function indices are
    size_t max_k, max_l;
    // and they belong to the shells
    size_t max_ks, max_ls;
    // that have N functions
    size_t max_Nk, max_Nl;
    // where the first functions are
    size_t max_k0, max_l0;
    {
      // The corresponding functions are
      max_k=invmap(0,pim);
      max_l=invmap(1,pim);
      // which are on the shells
      max_ks=basis.find_shell_ind(max_k);
      max_ls=basis.find_shell_ind(max_l);
      // that have N functions
      max_Nk=basis.get_Nbf(max_ks);
      max_Nl=basis.get_Nbf(max_ls);
      // and the function indices are
      max_k0=basis.get_first_ind(max_ks);
      max_l0=basis.get_first_ind(max_ls);
    }
    //    printf("Pivot corresponds to functions %i and %i on shells %i and %i.\n",(int) max_k, (int) max_l, (int) max_ks, (int) max_ls);

    // Compute integrals on the rows
    arma::mat A(d.n_elem,max_Nk*max_Nl);
    A.zeros();
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

        // Schwarz screening estimates
        double QQ=Q(is,js)*Q(max_ks,max_ls);
        if(QQ<shell_screen_tol) {
          continue;
        }
        double MM1=M(is,max_ks)*M(js,max_ls);
        if(MM1<shell_screen_tol) {
          continue;
        }
        double MM2=M(is,max_ls)*M(js,max_ks);
        if(MM2<shell_screen_tol) {
          continue;
        }

	// Compute integrals
	eri->compute(&shells[is],&shells[js],&shells[max_ks],&shells[max_ls]);
	erip=eri->getp();

	// and store them
	size_t Ni(shells[is].get_Nbf());
	size_t Nj(shells[js].get_Nbf());
	size_t i0(shells[is].get_first_ind());
	size_t j0(shells[js].get_first_ind());

	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t jj=0;jj<Nj;jj++) {
	    size_t i=i0+ii;
	    size_t j=j0+jj;

	    // Check if function pair is significant
	    if(prodmap(i,j)>Nbf*Nbf)
	      continue;

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

    size_t nb=0;
    size_t b0=m;
    while(true) {
      // Did we already treat everything in the block?
      if(nb==A.n_cols)
	break;
      // Remaining pivot is
      arma::uvec pileft(pi.subvec(m,d.n_elem-1));
      // Remaining errors in pivoted order
      arma::vec errs(d(pileft));
      // Find global largest error
      double errmax=arma::max(errs);
      // and the largest error within the current block
      double blockerr=0;
      size_t blockind=0;
      size_t Aind=0;
      for(size_t kk=0;kk<max_Nk;kk++)
	for(size_t ll=0;ll<max_Nl;ll++) {
	  // Function indices are
	  size_t k=kk+max_k0;
	  size_t l=ll+max_l0;
	  // Corresponding index in the array is
	  size_t ind = prodmap(k,l);
	  if(ind > Nbf*Nbf)
	    continue;

	  if(d(ind)>blockerr) {
	    // Check that the index is not in the old pivots
	    bool found=false;
	    for(size_t i=0;i<m;i++)
	      if(pi(i)==ind)
		found=true;
	    if(!found) {
	      Aind=kk*max_Nl+ll;
	      blockind=ind;
	      blockerr=d(ind);
	    }
	  }
	}
      // Move to next block.
      if(blockerr==0.0 || blockerr<shell_reuse_thr*errmax) {
	//printf("Block error is %e compared to global error %e, stopping\n",blockerr,errmax);
	break;
      }

      // Increment amount of vectors in the block
      nb++;

      // Switch the pivot
      if(pi(m)!=blockind) {
	bool found=false;
	for(size_t i=m+1;i<pi.n_elem;i++)
	  if(pi(i)==blockind) {
	    found=true;
	    std::swap(pi(i),pi(m));
	    break;
	  }
	if(!found) {
	  pi.t().print("Pivot");
	  fflush(stdout);
	  std::ostringstream oss;
	  oss << "Pivot index " << blockind << " not found, m = " << m << " !\n";
	  throw std::logic_error(oss.str());
	}
      }

      pim=pi(m);

      // Insert new rows if necessary. resize zero-extends and replaces
      // the deprecated 3-arg insert_rows(row_num, N, set_to_zero).
      if(m>=B.n_rows)
	B.resize(B.n_rows + 100, B.n_cols);

      // Compute diagonal element
      B(m,pim)=sqrt(d(pim));

      // Off-diagonal elements
      if(m==0) {
	// No B contribution here; avoid if clause in for loop
#ifdef _OPENMP
#pragma omp parallel for
#endif
     	for(size_t i=m+1;i<d.n_elem;i++) {
	  size_t pii=pi(i);
	  // Compute element
	  B(m,pii)=A(pii,Aind)/B(m,pim);
	  // Update d
	  d(pii)-=B(m,pii)*B(m,pii);
	}
      } else {
	// One GEMV computes the inner-product correction t(j) =
	// trans(B(0..m-1, pim)) * B(0..m-1, j) for every j in one go;
	// the previous implementation did m * (n_elem) scalar dots
	// inside the omp loop via arma::as_scalar.
	const arma::vec t(arma::trans(B.submat(0,0,m-1,B.n_cols-1))*B.submat(0,pim,m-1,pim));
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(size_t i=m+1;i<d.n_elem;i++) {
	  size_t pii=pi(i);
	  // Compute element
	  B(m,pii)=(A(pii,Aind) - t(pii))/B(m,pim);
	  // Update d
	  d(pii)-=B(m,pii)*B(m,pii);
	}
      }

      // Update error
      error=(m+1<=pi.n_elem-1) ? arma::max(d(pi.subvec(m+1,pi.n_elem-1))) : 0.0;
      // Increase m
      m++;
    }
    t_chol+=t.get();

    if(verbose) {
      printf("Cholesky vectors no %5i - %5i computed, error is %e (%s).\n",(int) b0, (int) (b0+nb-1),error,t.elapsed().c_str());
      fflush(stdout);
      t.set();
    }
  }

  if(verbose) {
    printf("Cholesky decomposition finished in %s. Realized memory size is %s.\n",ttot.elapsed().c_str(),memory_size(B.n_elem*sizeof(double)).c_str());
    printf("Time use: integrals %3.1f %%, linear algebra %3.1f %%.\n",100*t_int/(t_int+t_chol),100*t_chol/(t_int+t_chol));
  }

  // Transpose to get Cholesky vectors as columns
  arma::inplace_trans(B);

  // Drop any unnecessary columns
  if(m<B.n_cols)
    B.shed_cols(m,B.n_cols-1);

  // and pivot vectors
  pi=pi.subvec(0,m-1);
  // Form the pivot shellpairs
  form_pivot_shellpairs(basis);

  return shpairs.size();
}

size_t ERIchol::fill_two_step(const BasisSet & basis,
                              double cholesky_tol,
                              double shell_reuse_thr,
                              double shell_screen_tol,
                              double fit_cholesky_thr,
                              bool verbose) {
  if(cholesky_tol < shell_screen_tol) {
    fprintf(stderr,"Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
    printf("Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
    fflush(stdout);
  }

  // Screening matrix and pairs
  ScreeningData scr=basis.compute_screening(shell_screen_tol,omega,alpha,beta,verbose);
  const arma::mat & Q = scr.Q;
  const arma::mat & M_screen = scr.M;
  const std::vector<eripair_t> & shpairs = scr.shpairs;

  // Amount of basis functions
  Nbf=basis.get_Nbf();
  // Shells
  std::vector<GaussianShell> shells=basis.get_shells();

  Timer t;
  Timer ttot;

  // Integral / linear-algebra timers
  double t_int=0.0;
  double t_chol=0.0;

  // ===========================================================
  // Phase A: compute the (mu nu | mu nu) diagonal
  // ===========================================================
  arma::vec d(Nbf*Nbf);
  d.zeros();
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
      if(QQ<shell_screen_tol)
        continue;

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
          d(i*Nbf+j)=(*erip)[((ii*Nj+jj)*Ni+ii)*Nj+jj];
          d(j*Nbf+i)=d(i*Nbf+j);
        }
    }
    delete eri;
  }
  t_int+=t.get();

  // ===========================================================
  // Phase B: enumerate orbital pairs (mu <= nu) that pass
  // shellpair screening and build prodidx / invmap / prodmap /
  // odiagidx in this object. Identical layout to fill().
  // ===========================================================
  size_t Nshp=0;
  {
    prodmap.ones(Nbf,Nbf);
    prodmap*=-1; // UINT_MAX

    size_t iprod=0;
    size_t iodiag=0;
    prodidx.resize(Nbf*Nbf);
    odiagidx.resize(Nbf*Nbf);
    invmap.zeros(2,Nbf*Nbf);
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
            Nshp++;
            size_t idx=i*Nbf+j;
            prodidx(iprod)=idx;
            invmap(0,iprod)=i;
            invmap(1,iprod)=j;
            odiagidx(iodiag)=iprod;
            prodmap(i,j)=iprod;
            prodmap(j,i)=iprod;
            iprod++;
            iodiag++;
          }
          Nshp++;
          size_t idx=i*Nbf+i;
          prodidx(iprod)=idx;
          invmap(0,iprod)=i;
          invmap(1,iprod)=i;
          prodmap(i,i)=iprod;
          iprod++;
        }
      } else {
        for(size_t i=i0;i<i0+Ni;i++)
          for(size_t j=j0;j<j0+Nj;j++) {
            Nshp++;
            size_t idx=i*Nbf+j;
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

  // Restrict the diagonal to significant pairs
  d=d(prodidx);
  double error(arma::max(d));

  // ===========================================================
  // Phase C: pivot selection. Identical to fill()'s loop:
  // shellpair-batched libint call per max pivot, in-block addition
  // gated by shell_reuse_thr, Schur update on d. We also save the
  // selected pivot's column of A = (mu nu | shp_pivot) into B_raw
  // so Phase F doesn't recompute integrals.
  // ===========================================================
  pi=arma::linspace<arma::uvec>(0,d.n_elem-1,d.n_elem);
  // Transient B-rows used for the Schur update only.
  arma::mat B_temp;
  B_temp.zeros(100,prodidx.n_elem);
  // Saved columns of (mu nu | piv) for each retained pivot, in the
  // pivot order pi(0..m-1).
  arma::mat B_raw;
  B_raw.zeros(prodidx.n_elem, 100);
  size_t m=0;

  while(error>cholesky_tol && m<d.n_elem) {
    // Find max pivot, swap into position m
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

    // Identify the pivot's shellpair
    size_t max_k=invmap(0,pim);
    size_t max_l=invmap(1,pim);
    size_t max_ks=basis.find_shell_ind(max_k);
    size_t max_ls=basis.find_shell_ind(max_l);
    size_t max_Nk=basis.get_Nbf(max_ks);
    size_t max_Nl=basis.get_Nbf(max_ls);
    size_t max_k0=basis.get_first_ind(max_ks);
    size_t max_l0=basis.get_first_ind(max_ls);

    // Compute integrals A = (mu nu | shp_pivot) for all (mu nu)
    arma::mat A(d.n_elem,max_Nk*max_Nl);
    A.zeros();
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
            if(prodmap(i,j)>Nbf*Nbf) continue;
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

    // Block iteration: greedily pick pivots from this shellpair
    // gated by shell_reuse_thr; same pattern as fill().
    while(true) {
      arma::uvec pileft(pi.subvec(m,d.n_elem-1));
      arma::vec errs(d(pileft));
      double errmax=arma::max(errs);
      double blockerr=0;
      size_t blockind=0;
      size_t Aind=0;
      for(size_t kk=0;kk<max_Nk;kk++)
        for(size_t ll=0;ll<max_Nl;ll++) {
          size_t k=kk+max_k0;
          size_t l=ll+max_l0;
          size_t ind=prodmap(k,l);
          if(ind>Nbf*Nbf) continue;
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

      // Swap blockind into pi(m)
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

      // Grow B_temp / B_raw if needed
      if(m>=B_temp.n_rows)
        B_temp.resize(B_temp.n_rows+100, B_temp.n_cols);
      if(m>=B_raw.n_cols)
        B_raw.resize(B_raw.n_rows, B_raw.n_cols+100);

      // Save the (mu nu | piv) column for this newly-selected pivot.
      B_raw.col(m) = A.col(Aind);

      // Schur update on the diagonal d via a transient B row.
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
    error=(m<=pi.n_elem-1) ? arma::max(d(pi.subvec(m,pi.n_elem-1))) : 0.0;
    t_chol+=t.get();
  }

  const size_t Nselected=m;
  pi=pi.subvec(0,Nselected-1);
  if(Nselected<B_raw.n_cols)
    B_raw.shed_cols(Nselected,B_raw.n_cols-1);
  // Free the transient Schur-update workspace.
  B_temp.reset();

  if(verbose) {
    printf("Two-step CD selected %i pivot orbital pairs after pivoting (%s).\n",
           (int) Nselected, ttot.elapsed().c_str());
    fflush(stdout);
  }

  // ===========================================================
  // Phase D: build the metric M[p, q] = (piv_p | piv_q) over
  // selected pivot orbital pairs via libint over pivot shellpair
  // quadruples. Map (mu, nu) -> pivot index for the dispatch.
  // ===========================================================
  arma::umat pivot_to_index(Nbf,Nbf);
  // Sentinel UINT64_MAX marks "not a pivot"; underflow of size_t.
  pivot_to_index.fill(static_cast<arma::uword>(-1));
  for(size_t i=0;i<Nselected;i++) {
    size_t pii=pi(i);
    pivot_to_index(invmap(0,pii),invmap(1,pii))=i;
    pivot_to_index(invmap(1,pii),invmap(0,pii))=i;
  }
  form_pivot_shellpairs(basis);
  // Vector of pivot shellpairs, each as (is, js) with is <= js.
  std::vector<std::pair<size_t,size_t>> piv_shps(pivot_shellpairs.begin(), pivot_shellpairs.end());

  arma::mat M_metric(Nselected,Nselected);
  M_metric.zeros();
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
    for(size_t ip=0;ip<piv_shps.size();ip++) {
      size_t is=piv_shps[ip].first;
      size_t js=piv_shps[ip].second;
      size_t Ni=shells[is].get_Nbf();
      size_t Nj=shells[js].get_Nbf();
      size_t i0=shells[is].get_first_ind();
      size_t j0=shells[js].get_first_ind();
      for(size_t jp=0;jp<=ip;jp++) {
        size_t ks=piv_shps[jp].first;
        size_t ls=piv_shps[jp].second;
        size_t Nk=shells[ks].get_Nbf();
        size_t Nl=shells[ls].get_Nbf();
        size_t k0=shells[ks].get_first_ind();
        size_t l0=shells[ls].get_first_ind();

        eri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
        erip=eri->getp();
        for(size_t ii=0;ii<Ni;ii++)
          for(size_t jj=0;jj<Nj;jj++) {
            size_t mu_p=ii+i0;
            size_t nu_p=jj+j0;
            arma::uword pidx=pivot_to_index(mu_p,nu_p);
            if(pidx==static_cast<arma::uword>(-1)) continue;
            for(size_t kk=0;kk<Nk;kk++)
              for(size_t ll=0;ll<Nl;ll++) {
                size_t mu_q=kk+k0;
                size_t nu_q=ll+l0;
                arma::uword qidx=pivot_to_index(mu_q,nu_q);
                if(qidx==static_cast<arma::uword>(-1)) continue;
                double val=(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
                // Write both triangles explicitly: pidx and qidx are
                // unordered with respect to each other (they reflect
                // pivot-selection order, not shellpair order), so a
                // post-hoc symmatu/symmatl can't recover the missing
                // half. Atomic writes are cheap relative to the libint
                // call that produced `val`.
#ifdef _OPENMP
#pragma omp atomic write
#endif
                M_metric(pidx,qidx)=val;
#ifdef _OPENMP
#pragma omp atomic write
#endif
                M_metric(qidx,pidx)=val;
              }
          }
      }
    }
    delete eri;
  }
  t_int+=t.get();

  // ===========================================================
  // Phase E: orthogonalise the metric. PartialCholeskyOrth(M) -> X
  // such that X^T M X = I_eff over the lindep-cleaned subspace.
  // ===========================================================
  t.set();
  arma::mat M_invh = PartialCholeskyOrth(M_metric, fit_cholesky_thr, 0.0);
  t_chol+=t.get();

  if(verbose) {
    printf("Two-step CD: pivot metric orthogonalisation reduced %i -> %i Cholesky vectors (%s).\n",
           (int) Nselected, (int) M_invh.n_cols, t.elapsed().c_str());
    fflush(stdout);
  }

  // ===========================================================
  // Phase F: build the final L vectors. B_raw[(mu nu), p] is
  // (mu nu | piv_p); the standard ERI factorisation says
  //   (mu nu | la si) = sum_{p,q} (mu nu | piv_p) (piv|piv)^{-1}_{p,q} (piv_q | la si)
  //                   = sum_J L_J(mu nu) L_J(la si)
  // with L = B_raw * X (where X = M_invh and X X^T = (piv|piv)^{-1}).
  // ===========================================================
  t.set();
  B = B_raw * M_invh;
  t_chol+=t.get();

  // Stash the pivot metric and its half-inverse for forceJ. Without
  // these, the gradient pipeline can't reconstruct (dM/dR) /
  // (d(mu nu | piv)/dR) contributions.
  two_step_metric_ = std::move(M_metric);
  two_step_metric_invh_ = std::move(M_invh);
  two_step_ = true;

  if(verbose) {
    printf("Two-step CD finished in %s. Realised memory size is %s.\n",
           ttot.elapsed().c_str(), memory_size(B.n_elem*sizeof(double)).c_str());
    printf("Time use: integrals %3.1f %%, linear algebra %3.1f %%.\n",
           100*t_int/(t_int+t_chol),100*t_chol/(t_int+t_chol));
    fflush(stdout);
  }

  return shpairs.size();
}

arma::uvec ERIchol::get_pivot() const {
  return pi;
}

void ERIchol::form_pivot_shellpairs(const BasisSet & basis) {
  pivot_shellpairs.clear();
  for(size_t i=0;i<pi.n_elem;i++) {
    // The corresponding shell indices are
    size_t is=basis.find_shell_ind(invmap(0,pi(i)));
    size_t js=basis.find_shell_ind(invmap(1,pi(i)));
    if(js<is) std::swap(is,js);
    pivot_shellpairs.insert(std::pair<size_t,size_t>(is,js));
  }
}

std::set< std::pair<size_t, size_t> > ERIchol::get_pivot_shellpairs() const {
  return pivot_shellpairs;
}


size_t ERIchol::naf_transform(double thr, bool verbose) {
  /**
   * Mihály Kállay, "A systematic way for the cost reduction of density
   * fitting methods", J. Chem. Phys. 141, 244113 (2014).
   */

  // Helper matrix
  arma::mat W(arma::trans(B)*B);

  // which is then diagonalized
  arma::vec Wval;
  arma::mat Wvec;
  eig_sym_ordered(Wval,Wvec,W);

  // Then, the eigenvectors that are smaller than the threshold are
  // dropped. Remember that smallest eigenvalues come first
  arma::uword p;
  for(p=0;p<Wval.n_elem;p++)
    if(Wval(p)>=thr)
      break;

  // Original and dropped number of functions
  size_t norig(B.n_cols);
  size_t ndrop(p-1);

  // and the eigenvectors are rotated
  B*=Wvec.cols(p,Wvec.n_cols-1);

  if(verbose)
    printf("%i out of %i natural auxiliary functions dropped.\n",(int) ndrop,(int) norig);

  return ndrop;
}

size_t ERIchol::get_Naux() const {
  return B.n_cols;
}

size_t ERIchol::get_Nbf() const {
  return Nbf;
}

size_t ERIchol::get_Npairs() const {
  return B.n_rows;
}

arma::mat ERIchol::get() const {
  return B;
}

arma::umat ERIchol::get_invmap() const {
  return invmap;
}

arma::mat ERIchol::calcJ(const arma::mat & P) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Density matrix doesn't match basis set! N = " << Nbf << ", Nrows = " << P.n_rows << ", Ncols = " << P.n_cols << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Vectorize P
  arma::rowvec Pv(arma::trans(P(prodidx)));
  // Twice the off-diagonal contribution
  Pv(odiagidx)*=2.0;
  // Calculate expansion coefficients
  arma::rowvec g(Pv*B);
  // Form Coulomb matrix
  arma::vec Jv(B*arma::trans(g));
  // and restore it
  arma::mat J(P.n_rows,P.n_cols);
  J.zeros();
  for(size_t i=0;i<prodidx.size();i++)
    J(invmap(0,i),invmap(1,i))=Jv(i);
  for(size_t i=0;i<odiagidx.size();i++)
    J(invmap(1,odiagidx(i)),invmap(0,odiagidx(i)))=Jv(odiagidx(i));

  return J;
}

arma::mat ERIchol::calcK(const arma::vec & C) const {
  if(C.n_elem != Nbf) {
    std::ostringstream oss;
    oss << "Orbital vector doesn't match basis set! N = " << Nbf << ", N(C) = " << C.n_elem << "!\n";
    throw std::runtime_error(oss.str());
  }

  // K_uv = C_r C_s (ur|vs) = (L^P_ur C_r) (L^P_vs Cs)
  arma::mat v(C.n_elem,B.n_cols);
  v.zeros();

  // First part: diagonal and above diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t i=0;i<prodidx.size();i++)
      v(invmap(0,i),P)+=B(i,P)*C(invmap(1,i));
  // Below diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t ii=0;ii<odiagidx.n_elem;ii++) {
      size_t i=odiagidx(ii);
      v(invmap(1,i),P)+=B(i,P)*C(invmap(0,i));
    }

  return v*arma::trans(v);
}

arma::cx_mat ERIchol::calcK(const arma::cx_vec & C0) const {
  if(C0.n_elem != Nbf) {
    std::ostringstream oss;
    oss << "Orbital vector doesn't match basis set! N = " << Nbf << ", N(C) = " << C0.n_elem << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Need to complex conjugate C
  arma::cx_vec C(arma::conj(C0));

  // K_uv = C_r C_s (ur|vs) = (L^P_ur C_r) (L^P_vs Cs)
  arma::cx_mat v(C.n_elem,B.n_cols);
  v.zeros();

  // First part: diagonal and above diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t i=0;i<prodidx.size();i++)
      v(invmap(0,i),P)+=B(i,P)*C(invmap(1,i));
  // Below diagonal
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t ii=0;ii<odiagidx.n_elem;ii++) {
      size_t i=odiagidx(ii);
      v(invmap(1,i),P)+=B(i,P)*C(invmap(0,i));
    }

  return v*arma::trans(v);

}

arma::mat ERIchol::calcK(const arma::mat & C, const std::vector<double> & occs) const {
  if(C.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Orbital matrix doesn't match basis set! N = " << Nbf << ", N(C) = " << C.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Build the dense B once (Nbf*Nbf x Naux) and view it as
  // Nbf x (Nbf*Naux); each per-orbital contribution is then a single
  // GEMV + DSYRK rather than the indexed-scatter loops the
  // single-orbital calcK(vec) used. Per-thread K accumulation
  // replaces the per-orbital omp critical to remove serialisation
  // for high thread counts.
  arma::mat Bdense;
  B_matrix(Bdense);
  const size_t Naux = B.n_cols;
  const arma::mat Breshape((double *) Bdense.memptr(), Nbf, Nbf*Naux, false, true);

  arma::mat K(C.n_rows,C.n_rows,arma::fill::zeros);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::mat Kloc(Nbf, Nbf, arma::fill::zeros);
    arma::mat Awork(Nbf, Naux);
#ifdef _OPENMP
#pragma omp for schedule(dynamic) nowait
#endif
    for(size_t i=0;i<occs.size();i++)
      if(occs[i]!=0.0) {
	Awork = arma::reshape(arma::trans(Breshape)*C.col(i), Nbf, Naux);
	Kloc += occs[i] * Awork * arma::trans(Awork);
      }
#ifdef _OPENMP
#pragma omp critical
#endif
    K += Kloc;
  }
  return K;
}

arma::mat ERIchol::calcK(const arma::mat & C, const arma::vec & occs) const {
  return calcK(C, arma::conv_to<std::vector<double>>::from(occs));
}

arma::cx_mat ERIchol::calcK(const arma::cx_mat & C, const std::vector<double> & occs) const {
  if(C.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Orbital matrix doesn't match basis set! N = " << Nbf << ", N(C) = " << C.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Mirror the real calcK reformulation: dense B once, GEMV+DSYRK
  // per orbital, per-thread K accumulation. The complex orbital
  // requires a conj() on C as in the per-orbital calcK(cx_vec).
  arma::mat Bdense;
  B_matrix(Bdense);
  const size_t Naux = B.n_cols;
  const arma::mat Breshape((double *) Bdense.memptr(), Nbf, Nbf*Naux, false, true);

  arma::cx_mat K(C.n_rows,C.n_rows,arma::fill::zeros);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::cx_mat Kloc(Nbf, Nbf, arma::fill::zeros);
    arma::cx_mat Awork(Nbf, Naux);
#ifdef _OPENMP
#pragma omp for schedule(dynamic) nowait
#endif
    for(size_t i=0;i<occs.size();i++)
      if(occs[i]!=0.0) {
	const arma::cx_vec Cconj(arma::conj(C.col(i)));
	Awork = arma::reshape(arma::trans(Breshape)*Cconj, Nbf, Naux);
	Kloc += occs[i] * Awork * arma::trans(Awork);
      }
#ifdef _OPENMP
#pragma omp critical
#endif
    K += Kloc;
  }
  return K;
}

arma::cx_mat ERIchol::calcK(const arma::cx_mat & C, const arma::vec & occs) const {
  return calcK(C, arma::conv_to<std::vector<double>>::from(occs));
}

void ERIchol::B_matrix(arma::mat & Br) const {
  Br.zeros(Nbf*Nbf,B.n_cols);
  for(size_t P=0;P<B.n_cols;P++)
    for(size_t i=0;i<prodidx.size();i++) {
      size_t u=invmap(0,i);
      size_t v=invmap(1,i);
      Br(u*Nbf+v,P)=B(i,P);
      Br(v*Nbf+u,P)=B(i,P);
    }
}

void ERIchol::B_matrix(arma::mat & Br, arma::uword first, arma::uword last) const {
  Br.zeros(Nbf*Nbf,last-first+1);
  for(size_t P=first;P<=last;P++)
    for(size_t i=0;i<prodidx.size();i++) {
      size_t u=invmap(0,i);
      size_t v=invmap(1,i);
      Br(u*Nbf+v,P-first)=B(i,P);
      Br(v*Nbf+u,P-first)=B(i,P);
    }
}

arma::mat ERIchol::B_transform(const arma::mat & Cl, const arma::mat & Cr, bool verbose) const {
  // Amount of basis and auxiliary functions
  if(Cl.n_rows != Nbf || Cr.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Orbital matrices don't match basis set! N = " << Nbf << ", N(Cl) = " << Cl.n_rows << ", N(Cr) = " << Cr.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  Timer t;

  // Build the dense B matrix once (Nbf*Nbf x Naux). Each auxiliary
  // column is then a small Nbf x Nbf block that we transform with two
  // GEMMs per P, replacing the prodidx scatter loops + index-shuffle
  // dance the previous implementation used.
  arma::mat Bdense;
  B_matrix(Bdense);

  if(verbose) {
    printf("Built dense B in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  const size_t Nl = Cl.n_cols;
  const size_t Nr = Cr.n_cols;
  arma::mat Br(B.n_cols, Nl*Nr);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t P=0; P<B.n_cols; P++) {
    // View Bdense column P as a (Nbf, Nbf) matrix block_P(v, u) where
    // block_P(v, u) = Bdense(u*Nbf+v, P): column-major view, with v
    // varying fastest in memory.
    const arma::mat block_P((double*) Bdense.colptr(P), Nbf, Nbf, false, true);

    // Two-sided transform: Tmo(l, r) = sum_{u,v} Cl(u,l) Cr(v,r) Bdense(u*Nbf+v, P)
    //                                = (Cl.t() * block_P.t() * Cr)(l, r)
    const arma::mat Tmo(Cl.t() * block_P.t() * Cr);

    // Output layout (preserving the original convention):
    //   Br(P, r*Nl + l) = Tmo(l, r)
    // arma::vectorise of Tmo (column-major) gives flat[l + r*Nl] = Tmo(l, r).
    Br.row(P) = arma::trans(arma::vectorise(Tmo));
  }

  if(verbose) {
    printf("MO transform done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  return Br;
}

arma::vec ERIchol::forceJ(const BasisSet & basis, const arma::mat & P) const {
  if(!two_step_)
    throw std::runtime_error("ERIchol::forceJ requires the two-step pivot metric (fill_two_step). Forces are not implemented for one-step CD on this branch.\n");
  if(P.n_rows != Nbf || P.n_cols != Nbf)
    throw std::runtime_error("ERIchol::forceJ: density matrix dimension mismatch.\n");

  const size_t Naux = pi.n_elem;
  const size_t Nnuc = basis.get_Nnuc();

  // === Set up pivot-pair / pivot-shellpair lookup ===========================
  // pivot_to_index(mu, nu) -> pivot rank p in 0..Naux-1, or sentinel
  // UINT64_MAX. Mirrors the same table fill_two_step builds during
  // metric construction.
  arma::umat pivot_to_index(Nbf, Nbf);
  pivot_to_index.fill(static_cast<arma::uword>(-1));
  for(size_t p=0; p<Naux; p++) {
    const size_t pii = pi(p);
    pivot_to_index(invmap(0,pii), invmap(1,pii)) = p;
    pivot_to_index(invmap(1,pii), invmap(0,pii)) = p;
  }
  // Canonical vector of pivot shellpairs (lexicographic).
  std::vector<std::pair<size_t,size_t>> piv_shps(pivot_shellpairs.begin(), pivot_shellpairs.end());

  // === Compute gamma_J = (Pv * B) and c = M^{-1/2} gamma_J ==================
  // c lives in pivot-orbital-pair space (Naux entries); the
  // force contractions in Parts 1 and 2 are both c-on-c / P-on-c.
  arma::rowvec Pv(arma::trans(P(prodidx)));
  Pv(odiagidx) *= 2.0;
  const arma::vec gamma_J = arma::trans(Pv * B);
  const arma::vec c       = two_step_metric_invh_ * gamma_J;

  // libint dispatcher constants
  const std::vector<GaussianShell> shells = basis.get_shells();
  const int max_am   = basis.get_max_am();
  const int max_ncon = basis.get_max_Ncontr();

  arma::vec f(3 * Nnuc);
  f.zeros();

  // === Part 1: f += 0.5 c (dM/dR) c =========================================
  // d/dR_A (piv_p | piv_q) for pivot orbital pairs on pivot shellpairs.
  // Iterate over the lower-triangular (jp <= ip) of pivot shellpair pairs;
  // the ip == jp diagonal is included (off-diagonal contributions are
  // symmetric, both pidx,qidx and qidx,pidx accumulate).
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
    for(size_t ip=0; ip<piv_shps.size(); ip++) {
      const size_t is = piv_shps[ip].first;
      const size_t js = piv_shps[ip].second;
      const size_t Ni = shells[is].get_Nbf();
      const size_t Nj = shells[js].get_Nbf();
      const size_t i0 = shells[is].get_first_ind();
      const size_t j0 = shells[js].get_first_ind();
      const size_t i_at = shells[is].get_center_ind();
      const size_t j_at = shells[js].get_center_ind();

      for(size_t jp=0; jp<=ip; jp++) {
        const size_t ks = piv_shps[jp].first;
        const size_t ls = piv_shps[jp].second;
        const size_t Nk = shells[ks].get_Nbf();
        const size_t Nl = shells[ls].get_Nbf();
        const size_t k0 = shells[ks].get_first_ind();
        const size_t l0 = shells[ls].get_first_ind();
        const size_t k_at = shells[ks].get_center_ind();
        const size_t l_at = shells[ls].get_center_ind();

        // Off-diagonal pivot-shellpair contributions count twice.
        const double fac_sp = (ip == jp) ? 0.5 : 1.0;

        deri->compute(&shells[is], &shells[js], &shells[ks], &shells[ls]);

        // dERIWorker gives 12 derivative components; map each to
        // (atom_idx, xyz). Order: ic = 4*center + xyz, where
        // center = 0..3 maps to (i, j, k, l).
        const size_t atoms[4] = {i_at, j_at, k_at, l_at};
        for(int ic=0; ic<12; ic++) {
          const int center = ic / 3;
          const int xyz    = ic % 3;
          const size_t aA  = atoms[center];

          const std::vector<double> * erip = deri->getp(ic);
          double accum = 0.0;
          for(size_t ii=0; ii<Ni; ii++)
            for(size_t jj=0; jj<Nj; jj++) {
              const arma::uword pidx = pivot_to_index(i0+ii, j0+jj);
              if(pidx == static_cast<arma::uword>(-1)) continue;
              for(size_t kk=0; kk<Nk; kk++)
                for(size_t ll=0; ll<Nl; ll++) {
                  const arma::uword qidx = pivot_to_index(k0+kk, l0+ll);
                  if(qidx == static_cast<arma::uword>(-1)) continue;
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

  // === Part 2: f -= sum_munu P (d(mu nu | piv)/dR) c ========================
  // 4-center derivatives over (orbital_shellpair, pivot_shellpair).
  // Same shape as DensityFit::forceJ Part 2 (which uses 3-center
  // derivatives), but routed via dERIWorker.compute on 4 real shells.
  // dERIWorker yields 12 components mapped to the 4 centers.
  std::vector<eripair_t> orb_shps_ =
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
    for(size_t ipair=0; ipair<orb_shps_.size(); ipair++) {
      const size_t is = orb_shps_[ipair].is;
      const size_t js = orb_shps_[ipair].js;
      const size_t Ni = shells[is].get_Nbf();
      const size_t Nj = shells[js].get_Nbf();
      const size_t i0 = shells[is].get_first_ind();
      const size_t j0 = shells[js].get_first_ind();
      const size_t i_at = shells[is].get_center_ind();
      const size_t j_at = shells[js].get_center_ind();

      const double fac_sp = (is == js) ? 1.0 : 2.0;

      for(size_t jp=0; jp<piv_shps.size(); jp++) {
        const size_t ks = piv_shps[jp].first;
        const size_t ls = piv_shps[jp].second;
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
                  const arma::uword qidx = pivot_to_index(k0+kk, l0+ll);
                  if(qidx == static_cast<arma::uword>(-1)) continue;
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

