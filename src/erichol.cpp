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
  if(cholesky_tol < shell_screen_tol) {
    fprintf(stderr,"Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
    printf("Warning - used Cholesky threshold is smaller than the integral screening threshold. Results may be inaccurate!\n");
  }

  // Screening matrix and pairs
  arma::mat Q, M;
  std::vector<eripair_t> shpairs=basis.get_eripairs(Q,M,shell_screen_tol,omega,alpha,beta,verbose);

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

// ===========================================================================
// ERIfit namespace -- merged in from src/erifit.cpp.
// ===========================================================================
namespace ERIfit {

  bool operator<(const bf_pair_t & lhs, const bf_pair_t & rhs) {
    return lhs.idx < rhs.idx;
  }

  void get_basis(BasisSet & basis, const BasisSetLibrary & blib, const ElementBasisSet & orbel) {
    // Settings needed to form basis set
    Settings settings0(settings);
    settings.add_scf_settings();
    settings.set_bool("BasisRotate", false);
    settings.set_string("Decontract", "");
    settings.set_bool("UseLM", true);

    // Atoms
    std::vector<atom_t> atoms(1);
    atoms[0].el=orbel.get_symbol();
    atoms[0].num=0;
    atoms[0].x=atoms[0].y=atoms[0].z=0.0;
    atoms[0].Q=0;

    // Form basis set
    construct_basis(basis,atoms,blib);
  }

  void orthonormal_ERI_trans(const ElementBasisSet & orbel, double linthr, arma::mat & trans) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);

    // Get orthonormal orbitals
    arma::mat S(basis.overlap());
    arma::mat Sinvh(CanonicalOrth(S,linthr));

    // Sizes
    size_t Nbf(Sinvh.n_rows);
    size_t Nmo(Sinvh.n_cols);

    // Fill matrix
    trans.zeros(Nbf*Nbf,Nmo*Nmo);
    printf("Size of orthogonal transformation matrix is %i x %i\n",(int) trans.n_rows,(int) trans.n_cols);

    for(size_t iao=0;iao<Nbf;iao++)
      for(size_t jao=0;jao<Nbf;jao++)
	for(size_t imo=0;imo<Nmo;imo++)
	  for(size_t jmo=0;jmo<Nmo;jmo++)
	    trans(iao*Nbf+jao,imo*Nmo+jmo)=Sinvh(iao,imo)*Sinvh(jao,jmo);
  }

  void compute_ERIs(const BasisSet & basis, arma::mat & eris) {
    // Amount of functions
    size_t Nbf(basis.get_Nbf());

    // Get shells in basis set
    std::vector<GaussianShell> shells(basis.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> shpairs(basis.get_unique_shellpairs());

    // Print basis
    //    basis.print(true);

    // Allocate memory for the integrals
    eris.zeros(Nbf*Nbf,Nbf*Nbf);
    printf("Size of integral matrix is %i x %i\n",(int) eris.n_rows,(int) eris.n_cols);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      ERIWorker *eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
      for(size_t ip=0;ip<shpairs.size();ip++)
	for(size_t jp=0;jp<=ip;jp++) {
	  // Shells are
	  size_t is=shpairs[ip].is;
	  size_t js=shpairs[ip].js;
	  size_t ks=shpairs[jp].is;
	  size_t ls=shpairs[jp].js;

	  // First functions on shells
	  size_t i0=shells[is].get_first_ind();
	  size_t j0=shells[js].get_first_ind();
	  size_t k0=shells[ks].get_first_ind();
	  size_t l0=shells[ls].get_first_ind();

	  // Amount of functions
	  size_t Ni=shells[is].get_Nbf();
	  size_t Nj=shells[js].get_Nbf();
	  size_t Nk=shells[ks].get_Nbf();
	  size_t Nl=shells[ls].get_Nbf();

	  // Compute integral block
	  eri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	  // Get array
	  const std::vector<double> *erip=eri->getp();

	  // Store integrals
	  for(size_t ii=0;ii<Ni;ii++) {
	    size_t i=i0+ii;
	    for(size_t jj=0;jj<Nj;jj++) {
	      size_t j=j0+jj;
	      for(size_t kk=0;kk<Nk;kk++) {
		size_t k=k0+kk;
		for(size_t ll=0;ll<Nl;ll++) {
		  size_t l=l0+ll;

		  // Go through the 8 permutation symmetries
		  double mel=(*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
		  eris(i*Nbf+j,k*Nbf+l)=mel;
		  if(js!=is)
		    eris(j*Nbf+i,k*Nbf+l)=mel;
		  if(ks!=ls)
		    eris(i*Nbf+j,l*Nbf+k)=mel;
		  if(is!=js && ks!=ls)
		    eris(j*Nbf+i,l*Nbf+k)=mel;

		  if(ip!=jp) {
		    eris(k*Nbf+l,i*Nbf+j)=mel;
		    if(js!=is)
		      eris(k*Nbf+l,j*Nbf+i)=mel;
		    if(ks!=ls)
		      eris(l*Nbf+k,i*Nbf+j)=mel;
		    if(is!=js && ks!=ls)
		      eris(l*Nbf+k,j*Nbf+i)=mel;
		  }
		}
	      }
	    }
	  }
	}

      // Free memory
      delete eri;
    }
  }

  void compute_ERIs(const ElementBasisSet & orbel, arma::mat & eris) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);

    // Calculate
    compute_ERIs(basis,eris);
  }

  void compute_diag_ERIs(const ElementBasisSet & orbel, arma::mat & eris) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);

    size_t Nbf(basis.get_Nbf());

    // Get shells in basis set
    std::vector<GaussianShell> shells(basis.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> shpairs(basis.get_unique_shellpairs());

    // Print basis
    //    basis.print(true);

    // Allocate memory for the integrals
    eris.zeros(Nbf,Nbf);
    printf("Size of integral matrix is %i x %i\n",(int) eris.n_rows,(int) eris.n_cols);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      ERIWorker *eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
      for(size_t ip=0;ip<shpairs.size();ip++) {
	// Shells are
	size_t is=shpairs[ip].is;
	size_t js=shpairs[ip].js;

	// First functions on shells
	size_t i0=shells[is].get_first_ind();
	size_t j0=shells[js].get_first_ind();

	// Amount of functions
	size_t Ni=shells[is].get_Nbf();
	size_t Nj=shells[js].get_Nbf();

	// Compute integral block
	eri->compute(&shells[is],&shells[js],&shells[is],&shells[js]);
	// Get array
	const std::vector<double> *erip=eri->getp();

	// Store integrals
	for(size_t ii=0;ii<Ni;ii++) {
	  size_t i=i0+ii;
	  for(size_t jj=0;jj<Nj;jj++) {
	    size_t j=j0+jj;
	    eris(i,j)=(*erip)[((ii*Nj+jj)*Ni+ii)*Nj+jj];
	  }
	}
      }

      // Free memory
      delete eri;
    }
  }

  void unique_exponent_pairs(const ElementBasisSet & orbel, int am1, int am2, std::vector< std::vector<shellpair_t> > & pairs, std::vector<double> & exps) {
    // Form orbital basis set library
    BasisSetLibrary orblib;
    orblib.add_element(orbel);

    // Form orbital basis set
    BasisSet basis;
    get_basis(basis,orblib,orbel);

    // Get shells
    std::vector<GaussianShell> shells(basis.get_shells());
    // and list of unique shell pairs
    std::vector<shellpair_t> shpairs(basis.get_unique_shellpairs());

    // Create the exponent list
    exps.clear();
    for(size_t ip=0;ip<shpairs.size();ip++) {
      // Shells are
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;

      // Check am
      if(!( (shells[is].get_am()==am1 && shells[js].get_am()==am2) || (shells[is].get_am()==am2 && shells[js].get_am()==am1)))
	continue;

      // Check that shells aren't contracted
      if(shells[is].get_Ncontr()!=1 || shells[js].get_Ncontr()!=1) {
	ERROR_INFO();
	throw std::runtime_error("Must use primitive basis set!\n");
      }

      // Exponent value is
      double zeta=shells[is].get_contr()[0].z + shells[js].get_contr()[0].z;
      sorted_insertion<double>(exps,zeta);
    }

    // Create the pair list
    pairs.resize(exps.size());
    for(size_t ip=0;ip<shpairs.size();ip++) {
      // Shells are
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;

      // Check am
      if(!( (shells[is].get_am()==am1 && shells[js].get_am()==am2) || (shells[is].get_am()==am2 && shells[js].get_am()==am1)))
	continue;

      // Pair is
      double zeta=shells[is].get_contr()[0].z + shells[js].get_contr()[0].z;
      size_t pos=sorted_insertion<double>(exps,zeta);

      // Insert pair
      pairs[pos].push_back(shpairs[ip]);
    }
  }

  void compute_cholesky_T(const ElementBasisSet & orbel, int am1, int am2, arma::mat & eris, arma::vec & exps_) {
    // Form basis set library
    BasisSetLibrary blib;
    blib.add_element(orbel);
    // Decontract the basis set
    blib.decontract();

    // Form basis set
    BasisSet basis;
    get_basis(basis,blib,orbel);

    // Get shells in basis sets
    std::vector<GaussianShell> shells(basis.get_shells());

    // Get list of unique exponent pairs
    std::vector< std::vector<shellpair_t> > upairs;
    std::vector<double> exps;
    unique_exponent_pairs(orbel,am1,am2,upairs,exps);

    // Store exponents
    exps_=arma::conv_to<arma::vec>::from(exps);

    // Allocate memory for the integrals
    eris.zeros(exps.size(),exps.size());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      ERIWorker *eri=new ERIWorker(basis.get_max_am(),basis.get_max_Ncontr());

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
      // Loop over unique exponent pairs
      for(size_t iip=0;iip<upairs.size();iip++)
	for(size_t jjp=0;jjp<=iip;jjp++) {

	  // Loop over individual shell pairs in the group
	  for(size_t ip=0;ip<upairs[iip].size();ip++)
	    for(size_t jp=0;jp<upairs[jjp].size();jp++) {
	      // Shells are
	      size_t is=upairs[iip][ip].is;
	      size_t js=upairs[iip][ip].js;
	      size_t ks=upairs[jjp][jp].is;
	      size_t ls=upairs[jjp][jp].js;

	      // Amount of functions
	      size_t Ni=shells[is].get_Nbf();
	      size_t Nj=shells[js].get_Nbf();
	      size_t Nk=shells[ks].get_Nbf();
	      size_t Nl=shells[ls].get_Nbf();

	      // Compute integral block
	      eri->compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	      // Get array
	      const std::vector<double> *erip=eri->getp();

	      // Store integrals
	      for(size_t ii=0;ii<Ni;ii++)
		for(size_t jj=0;jj<Nj;jj++)
		  for(size_t kk=0;kk<Nk;kk++)
		    for(size_t ll=0;ll<Nl;ll++) {
		      double mel=std::abs((*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll]);
		      // mel*=mel;

		      eris(iip,jjp)+=mel;
		      if(iip!=jjp)
			eris(jjp,iip)+=mel;
		    }
	    }
	}

      // Free memory
      delete eri;
    }
  }

  void compute_fitint(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, arma::mat & fitint) {
    // Form orbital basis set library
    BasisSetLibrary orblib;
    orblib.add_element(orbel);

    // Form orbital basis set
    BasisSet orbbas;
    get_basis(orbbas,orblib,orbel);

    // and fitting basis set
    BasisSet fitbas;
    get_basis(fitbas,fitlib,orbel);

    // Coulomb normalize the fitting set
    fitbas.coulomb_normalize();

    // Get shells in basis sets
    std::vector<GaussianShell> orbsh(orbbas.get_shells());
    std::vector<GaussianShell> fitsh(fitbas.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> orbpairs(orbbas.get_unique_shellpairs());

    // Dummy shell
    GaussianShell dummy(dummyshell());

    // Problem sizes
    const size_t Norb(orbbas.get_Nbf());
    const size_t Nfit(fitbas.get_Nbf());

    // Allocate memory
    fitint.zeros(Norb*Norb,Nfit);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      int maxam=std::max(orbbas.get_max_am(),fitbas.get_max_am());
      ERIWorker *eri=new ERIWorker(maxam,orbbas.get_max_Ncontr());

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++)
	for(size_t as=0;as<fitsh.size();as++) {
	  // Orbital shells are
	  size_t is=orbpairs[ip].is;
	  size_t js=orbpairs[ip].js;

	  // First function is
	  size_t i0=orbsh[is].get_first_ind();
	  size_t j0=orbsh[js].get_first_ind();
	  size_t a0=fitsh[as].get_first_ind();

	  // Amount of functions
	  size_t Ni=orbsh[is].get_Nbf();
	  size_t Nj=orbsh[js].get_Nbf();
	  size_t Na=fitsh[as].get_Nbf();

	  // Compute integral block
	  eri->compute(&orbsh[is],&orbsh[js],&dummy,&fitsh[as]);
	  // Get array
	  const std::vector<double> *erip=eri->getp();

	  // Store integrals
	  for(size_t ii=0;ii<Ni;ii++) {
	    size_t i=i0+ii;
	    for(size_t jj=0;jj<Nj;jj++) {
	      size_t j=j0+jj;
	      for(size_t aa=0;aa<Na;aa++) {
		size_t a=a0+aa;

		// Go through the two permutation symmetries
		double mel=(*erip)[(ii*Nj+jj)*Na+aa];
		fitint(i*Norb+j,a)=mel;
		fitint(j*Norb+i,a)=mel;
	      }
	    }
	  }
	}

      // Free memory
      delete eri;
    }
  }

  void compute_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri) {
    // Form orbital basis set library
    BasisSetLibrary orblib;
    orblib.add_element(orbel);

    // Form orbital basis set
    BasisSet orbbas;
    get_basis(orbbas,orblib,orbel);

    // and fitting basis set
    BasisSet fitbas;
    get_basis(fitbas,fitlib,orbel);
    // Coulomb normalize the fitting set
    fitbas.coulomb_normalize();

    if(fitint.n_cols != fitbas.get_Nbf())
      throw std::runtime_error("Need to supply fitting integrals for ERIfit!\n");

    // Get shells in basis sets
    std::vector<GaussianShell> orbsh(orbbas.get_shells());
    std::vector<GaussianShell> fitsh(fitbas.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> orbpairs(orbbas.get_unique_shellpairs());

    // Dummy shell
    GaussianShell dummy(dummyshell());

    // Problem size
    size_t Nfit(fitbas.get_Nbf());
    // Overlap matrix
    arma::mat S(Nfit,Nfit);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      int maxam=std::max(orbbas.get_max_am(),fitbas.get_max_am());
      ERIWorker *eri=new ERIWorker(maxam,orbbas.get_max_Ncontr());

      // Compute the fitting basis overlap
#ifdef _OPENMP
#pragma omp for
#endif
      for(size_t i=0;i<fitsh.size();i++)
	for(size_t j=0;j<=i;j++) {
	  // Compute integral block
	  eri->compute(&fitsh[i],&dummy,&fitsh[j],&dummy);
	  // Get array
	  const std::vector<double> *erip=eri->getp();
	  // Store integrals
	  size_t i0=fitsh[i].get_first_ind();
	  size_t j0=fitsh[j].get_first_ind();
	  size_t Ni=fitsh[i].get_Nbf();
	  size_t Nj=fitsh[j].get_Nbf();
	  for(size_t ii=0;ii<Ni;ii++)
	    for(size_t jj=0;jj<Nj;jj++) {
	      double mel=(*erip)[ii*Nj+jj];
	      S(i0+ii,j0+jj)=mel;
	      S(j0+jj,i0+ii)=mel;
	    }
	}

      // Free memory
      delete eri;
    }

    // Do the eigendecomposition
    arma::vec Sval;
    arma::mat Svec;
    eig_sym_ordered(Sval,Svec,S);

    // Count linearly independent vectors
    size_t Nind=0;
    for(size_t i=0;i<Sval.n_elem;i++)
      if(Sval(i)>=linthr)
	Nind++;
    // and drop the linearly dependent ones
    Sval=Sval.subvec(Sval.n_elem-Nind,Sval.n_elem-1);
    Svec=Svec.cols(Svec.n_cols-Nind,Svec.n_cols-1);

    // Form inverse overlap matrix
    arma::mat S_inv;
    S_inv.zeros(Svec.n_rows,Svec.n_rows);
    for(size_t i=0;i<Sval.n_elem;i++)
      S_inv+=Svec.col(i)*arma::trans(Svec.col(i))/Sval(i);

    // Fitted ERIs are
    fiteri=fitint*S_inv*arma::trans(fitint);
  }

  void compute_diag_ERIfit(const BasisSetLibrary & fitlib, const ElementBasisSet & orbel, double linthr, const arma::mat & fitint, arma::mat & fiteri) {
    // Form orbital basis set library
    BasisSetLibrary orblib;
    orblib.add_element(orbel);

    // Form orbital basis set
    BasisSet orbbas;
    get_basis(orbbas,orblib,orbel);

    // and fitting basis set
    BasisSet fitbas;
    get_basis(fitbas,fitlib,orbel);
    // Coulomb normalize the fitting set
    fitbas.coulomb_normalize();

    if(fitint.n_rows != orbbas.get_Nbf()*orbbas.get_Nbf())
      throw std::runtime_error("Need to supply fitting integrals for ERIfit!\n");
    if(fitint.n_cols != fitbas.get_Nbf())
      throw std::runtime_error("Need to supply fitting integrals for ERIfit!\n");

    // Get shells in basis sets
    std::vector<GaussianShell> orbsh(orbbas.get_shells());
    std::vector<GaussianShell> fitsh(fitbas.get_shells());
    // Get list of shell pairs
    std::vector<shellpair_t> orbpairs(orbbas.get_unique_shellpairs());

    // Dummy shell
    GaussianShell dummy(dummyshell());

    // Problem size
    size_t Nfit(fitbas.get_Nbf());
    // Overlap matrix
    arma::mat S(Nfit,Nfit);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Integral worker
      int maxam=std::max(orbbas.get_max_am(),fitbas.get_max_am());
      ERIWorker *eri=new ERIWorker(maxam,orbbas.get_max_Ncontr());

      // Compute the fitting basis overlap
#ifdef _OPENMP
#pragma omp for
#endif
      for(size_t i=0;i<fitsh.size();i++)
	for(size_t j=0;j<=i;j++) {
	  // Compute integral block
	  eri->compute(&fitsh[i],&dummy,&fitsh[j],&dummy);
	  // Get array
	  const std::vector<double> *erip=eri->getp();
	  // Store integrals
	  size_t i0=fitsh[i].get_first_ind();
	  size_t j0=fitsh[j].get_first_ind();
	  size_t Ni=fitsh[i].get_Nbf();
	  size_t Nj=fitsh[j].get_Nbf();
	  for(size_t ii=0;ii<Ni;ii++)
	    for(size_t jj=0;jj<Nj;jj++) {
	      double mel=(*erip)[ii*Nj+jj];
	      S(i0+ii,j0+jj)=mel;
	      S(j0+jj,i0+ii)=mel;
	    }
	}

      // Free memory
      delete eri;
    }

    // Do the eigendecomposition
    arma::vec Sval;
    arma::mat Svec;
    eig_sym_ordered(Sval,Svec,S);

    // Count linearly independent vectors
    size_t Nind=0;
    for(size_t i=0;i<Sval.n_elem;i++)
      if(Sval(i)>=linthr)
	Nind++;
    // and drop the linearly dependent ones
    Sval=Sval.subvec(Sval.n_elem-Nind,Sval.n_elem-1);
    Svec=Svec.cols(Svec.n_cols-Nind,Svec.n_cols-1);

    // Form inverse overlap matrix
    arma::mat S_inv;
    S_inv.zeros(Svec.n_rows,Svec.n_rows);
    for(size_t i=0;i<Sval.n_elem;i++)
      S_inv+=Svec.col(i)*arma::trans(Svec.col(i))/Sval(i);

    // Fitted ERIs are
    size_t Nbf(orbbas.get_Nbf());
    fiteri.zeros(Nbf,Nbf);
    for(size_t i=0;i<Nbf;i++)
      for(size_t j=0;j<=i;j++) {
	double el=arma::as_scalar(fitint.row(i*Nbf+j)*S_inv*arma::trans(fitint.row(i*Nbf+j)));
	fiteri(i,j)=el;
	fiteri(j,i)=el;
      }
  }
}
