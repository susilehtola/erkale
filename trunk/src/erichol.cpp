#include "erichol.h"
#include "eriworker.h"
#include "mathf.h"
#include "stringutil.h"
#include "timer.h"

#include <cstdio>
// For exceptions
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

ERIchol::ERIchol() {
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

size_t ERIchol::fill(const BasisSet & basis, double tol, double shthr, double shtol, bool verbose) {
  // Screening matrix and pairs
  arma::mat screen;
  std::vector<eripair_t> shpairs=basis.get_eripairs(screen,shtol,omega,alpha,beta,verbose);

  // Amount of basis functions
  const size_t Nbf(basis.get_Nbf());
  // Shells
  std::vector<GaussianShell> shells=basis.get_shells();

  Timer t;

  if(verbose)
    printf("Computing Cholesky vectors. Estimated memory size is %s - %s.\n",memory_size(3*Nbf*Nbf*Nbf*sizeof(double),true).c_str(),memory_size(10*Nbf*Nbf*Nbf*sizeof(double),true).c_str());

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
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shpairs.size();ip++) {
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;
      
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
  // Error is
  double error(arma::sum(d));

  // Pivot index
  arma::uvec pi(arma::linspace<arma::uvec>(0,d.n_elem-1,d.n_elem));
  // Clear Cholesky vectors
  B.clear();
  // Loop index
  size_t m(0);
    
  while(error>tol && m<d.n_elem) {
    // Errors in pivoted order
    arma::vec errs(d(pi));
    // Sort the upcoming errors so that largest one is first
    arma::uvec idx=arma::stable_sort_index(errs.subvec(m,d.n_elem-1),"descend");

    // Update the pivot index
    arma::uvec pisub(pi.subvec(m,d.n_elem-1));
    pisub=pisub(idx);
    pi.subvec(m,d.n_elem-1)=pisub;

    // Pivot index
    size_t pim=pi(m);
    //printf("Pivot index is %4i with error %e, error is %e\n",(int) pim, d(pim), error);
    
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
      max_k=pim/Nbf;
      max_l=pim%Nbf;
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

    // Compute integrals on the rows
    arma::mat A(Nbf*Nbf,max_Nk*max_Nl);
    A.zeros();
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
#pragma omp for schedule(dynamic)
#endif
      for(size_t ipair=0;ipair<shpairs.size();ipair++) {
	size_t is=shpairs[ipair].is;
	size_t js=shpairs[ipair].js;

	// Do we need to compute the shell?
	if(screen(is,js)*screen(max_ks,max_ls)<shtol)
	  continue;
	
	// Compute integrals
	eri->compute(&shells[is],&shells[js],&shells[max_ks],&shells[max_ls]);
	erip=eri->getp();
	
	// and store them
	size_t Ni(shells[is].get_Nbf());
	size_t Nj(shells[js].get_Nbf());
	size_t i0(shells[is].get_first_ind());
	size_t j0(shells[js].get_first_ind());

	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t jj=0;jj<Nj;jj++)
	    for(size_t kk=0;kk<max_Nk;kk++)
	      for(size_t ll=0;ll<max_Nl;ll++) {
		size_t i=i0+ii;
		size_t j=j0+jj;
		
		A(i*Nbf+j,kk*max_Nl+ll)=(*erip)[((ii*Nj+jj)*max_Nk+kk)*max_Nl+ll];
		A(j*Nbf+i,kk*max_Nl+ll)=A(i*Nbf+j,kk*max_Nl+ll);
	      }
      }
      
      delete eri;
    }

    size_t nb=0;
    while(true) {
      // Find global largest error
      errs=d(pi);
      double errmax=arma::max(errs.subvec(m,d.n_elem-1));
      // and the largest error within the current block
      double blockerr=0;
      size_t blockind=0;
      size_t Aind=0;
      for(size_t kk=0;kk<max_Nk;kk++)
	for(size_t ll=0;ll<max_Nl;ll++) {
	  size_t ind=(kk+max_k0)*Nbf+ll+max_l0;
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
      if(blockerr<shthr*errmax) {
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
	  std::ostringstream oss;
	  oss << "Pivot index " << blockind << " not found, m = " << m << " !\n";
	  throw std::logic_error(oss.str());
	}
      }
      
      pim=pi(m);
      
      // Insert new rows
      if(m==0)
	B.zeros(1,Nbf*Nbf);
      else
	B.insert_rows(B.n_rows,1,true);
      
      // Compute diagonal element
      B(m,pim)=sqrt(d(pim));
      
      // Off-diagonal elements
      for(size_t i=m+1;i<d.n_elem;i++) {
	size_t pii=pi(i);
	// Compute element
	B(m,pii)= (m>0) ? (A(pii,Aind) - arma::dot(B.col(pim).subvec(0,m-1),B.col(pii).subvec(0,m-1)))/B(m,pim) : (A(pii,Aind))/B(m,pim);
	// Update d
	d(pii)-=B(m,pii)*B(m,pii);
      }
      
      // Update error
      error=arma::sum(d(pi.subvec(m+1,pi.n_elem-1)));
      // Increase m
      m++;
    }
    
    if(verbose) {
      printf("Cholesky vectors no %5i - %5i computed, error is %e (%s).\n",(int) (B.n_rows-nb+1),(int) B.n_rows,error,t.elapsed().c_str());
      fflush(stdout);
      t.set();
    }
  }

  if(verbose)
    printf("Cholesky decomposition finished, realized memory size is %s.\n",memory_size(B.n_elem*sizeof(double)).c_str());
  
  // Transpose to get Cholesky vectors as columns
  arma::inplace_trans(B);
  
  return shpairs.size();
}

size_t ERIchol::get_N() const {
  return B.n_cols;
}

arma::mat ERIchol::get() const {
  return B;
}

arma::mat ERIchol::calcJ(const arma::mat & P) const {
  if(P.n_elem != B.n_rows)
    throw std::runtime_error("Density matrix does not match basis set!\n");

  // Vectorize P
  arma::rowvec Pv(arma::trans(arma::vectorise(P)));
  // Calculate expansion coefficients
  arma::rowvec g(Pv*B);
  // Form Coulomb matrix
  arma::vec Jv(B*arma::trans(g));
  // and restore it
  return arma::mat(Jv.memptr(),P.n_rows,P.n_cols);
}
  
arma::mat ERIchol::calcK(const arma::vec & C) const {
  if(B.n_rows != C.n_elem*C.n_elem)
    throw std::runtime_error("Orbital not match basis set!\n");
  size_t Nbf(C.n_elem);
  size_t Naux(B.n_cols);

  // Reshape B
  arma::mat Br(arma::reshape(B,Nbf,Nbf*Naux));

  // Do transform to (i,uP)
  Br.reshape(Nbf,Nbf*Naux);
  Br=arma::trans(C)*Br;

  // Reshape to Nbf x Naux
  Br.reshape(Nbf,Naux);
  // so now we can get the exchange matrix as
  return Br*arma::trans(Br);
}

arma::cx_mat ERIchol::calcK(const arma::cx_vec & C) const {
  if(B.n_rows != C.n_elem*C.n_elem)
    throw std::runtime_error("Orbital not match basis set!\n");
  size_t Nbf(C.n_elem);
  size_t Naux(B.n_cols);

  // Reshape B
  arma::mat Br(arma::reshape(B,Nbf,Nbf*Naux));

  // Do transform to (i,uP)
  Br.reshape(Nbf,Nbf*Naux);
  arma::cx_mat Bc(arma::trans(C)*Br);

  // Reshape to Nbf x Naux
  Bc.reshape(Nbf,Naux);
  // so now we can get the exchange matrix as
  return Bc*arma::trans(Bc);
}

arma::mat ERIchol::calcK(const arma::mat & C, const std::vector<double> & occs) const {
  arma::mat K(C.n_rows,C.n_rows);
  K.zeros();
  for(size_t i=0;i<occs.size();i++)
    if(occs[i]!=0.0)
      K+=occs[i]*calcK(C.col(i));
  return K;
}

arma::cx_mat ERIchol::calcK(const arma::cx_mat & C, const std::vector<double> & occs) const {
  arma::cx_mat K(C.n_rows,C.n_rows);
  K.zeros();
  for(size_t i=0;i<occs.size();i++)
    if(occs[i]!=0.0)
      K+=occs[i]*calcK(C.col(i));
  return K;
}
