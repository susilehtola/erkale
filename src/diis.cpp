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



#include <cfloat>
#include "diis.h"
#include "linalg.h"
#include "mathf.h"
#include "stringutil.h"

// Maximum allowed absolute weight for a Fockian.
#define MAXWEIGHT 10.0
// Trigger cooloff if energy rises more than
#define COOLTHR 0.1

bool operator<(const diis_pol_entry_t & lhs, const diis_pol_entry_t & rhs) {
  return lhs.E < rhs.E;
}

bool operator<(const diis_unpol_entry_t & lhs, const diis_unpol_entry_t & rhs) {
  return lhs.E < rhs.E;
}

DIIS::DIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool usediis_, bool c1diis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) {
  S=S_;
  Sinvh=Sinvh_;
  usediis=usediis_;
  c1diis=c1diis_;
  useadiis=useadiis_;
  verbose=verbose_;
  imax=imax_;

  // Start mixing in DIIS weight when error is
  diiseps=diiseps_;
  // and switch to full DIIS when error is
  diisthr=diisthr_;

  // No cooloff
  cooloff=0;
}

rDIIS::rDIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool usediis_, bool c1diis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) : DIIS(S_,Sinvh_,usediis_,c1diis_,diiseps_,diisthr_,useadiis_,verbose_,imax_) {
}

uDIIS::uDIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool usediis_, bool c1diis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) : DIIS(S_,Sinvh_,usediis_,c1diis_,diiseps_,diisthr_,useadiis_,verbose_,imax_) {
}

DIIS::~DIIS() {
}

rDIIS::~rDIIS() {
}

uDIIS::~uDIIS() {
}

void rDIIS::clear() {
  stack.clear();
}

void uDIIS::clear() {
  stack.clear();
}

void rDIIS::erase_last() {
  stack.erase(stack.begin());
}

void uDIIS::erase_last() {
  stack.erase(stack.begin());
}

void rDIIS::update(const arma::mat & F, const arma::mat & P, double E, double & error) {
  // New entry
  diis_unpol_entry_t hlp;
  hlp.F=F;
  hlp.P=P;
  hlp.E=E;

  // Compute error matrix
  arma::mat errmat=F*P*S-S*P*F;
  // and transform it to the orthonormal basis (1982 paper, page 557)
  errmat=arma::trans(Sinvh)*errmat*Sinvh;
  // and store it
  hlp.err=MatToVec(errmat);

  // DIIS error is
  error=max_abs(errmat);

  // Is stack full?
  if(stack.size()==imax) {
    erase_last();
  }
  // Add to stack
  stack.push_back(hlp);

  // Update ADIIS helpers
  PiF_update();
}

void rDIIS::PiF_update() {
  arma::mat Fn=stack[stack.size()-1].F;
  arma::mat Pn=stack[stack.size()-1].P;
  
  // Update matrices
  PiF.zeros(stack.size());
  for(size_t i=0;i<stack.size();i++)
    PiF(i)=arma::trace((stack[i].P-Pn)*Fn);
  
  PiFj.zeros(stack.size(),stack.size());
  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<stack.size();j++)
      PiFj(i,j)=arma::trace((stack[i].P-Pn)*(stack[j].F-Fn));
}

void uDIIS::update(const arma::mat & Fa, const arma::mat & Fb, const arma::mat & Pa, const arma::mat & Pb, double E, double & error) {
  // New entry
  diis_pol_entry_t hlp;
  hlp.Fa=Fa;
  hlp.Fb=Fb;
  hlp.Pa=Pa;
  hlp.Pb=Pb;
  hlp.E=E;

  // Compute error matrices
  arma::mat errmata=Fa*Pa*S-S*Pa*Fa;
  arma::mat errmatb=Fb*Pb*S-S*Pb*Fb;
  // and transform them to the orthonormal basis (1982 paper, page 557)
  arma::mat errmat=arma::trans(Sinvh)*(errmata+errmatb)*Sinvh;
  // and store it
  hlp.err=MatToVec(errmat);

  // DIIS error is
  error=max_abs(errmat);

  // Is stack full?
  if(stack.size()==imax) {
    erase_last();
  }
  // Add to stack
  stack.push_back(hlp);

  // Update ADIIS helpers
  PiF_update();
}

void uDIIS::PiF_update() {
  arma::mat Fan=stack[stack.size()-1].Fa;
  arma::mat Fbn=stack[stack.size()-1].Fb;
  arma::mat Pan=stack[stack.size()-1].Pa;
  arma::mat Pbn=stack[stack.size()-1].Pb;

  // Update matrices
  PiF.zeros(stack.size());
  for(size_t i=0;i<stack.size();i++)
    PiF(i)=arma::trace((stack[i].Pa-Pan)*Fan) + arma::trace((stack[i].Pb-Pbn)*Fbn);

  PiFj.zeros(stack.size(),stack.size());
  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<stack.size();j++)
      PiFj(i,j)=arma::trace((stack[i].Pa-Pan)*(stack[j].Fa-Fan))+arma::trace((stack[i].Pb-Pbn)*(stack[j].Fb-Fbn));
}

arma::vec rDIIS::get_energies() const {
  arma::vec E(stack.size());
  for(size_t i=0;i<stack.size();i++)
    E(i)=stack[i].E;
  return E;
}
  
arma::mat rDIIS::get_diis_error() const {
  arma::mat err(stack[0].err.n_elem,stack.size());
  for(size_t i=0;i<stack.size();i++)
    err.col(i)=stack[i].err;
  return err;
}

arma::vec uDIIS::get_energies() const {
  arma::vec E(stack.size());
  for(size_t i=0;i<stack.size();i++)
    E(i)=stack[i].E;
  return E;
}
arma::mat uDIIS::get_diis_error() const {
  arma::mat err(stack[0].err.n_elem,stack.size());
  for(size_t i=0;i<stack.size();i++)
    err.col(i)=stack[i].err;
  return err;
}

arma::vec DIIS::get_w() {
  // DIIS error
  arma::mat de=get_diis_error();
  double err=arma::max(arma::abs(de.col(de.n_cols-1)));

  // Weight
  arma::vec w;
  
  if(useadiis && !usediis) {
    w=get_w_adiis();
    if(verbose) {
      printf("ADIIS weights\n");
      print_mat(w.t(),"% .2e ");
    }    
  } else if(!useadiis && usediis) {
    // Use DIIS only if error is smaller than threshold
    if(err>diisthr)
      throw std::runtime_error("DIIS error too large.\n");

    w=get_w_diis();

    if(verbose) {
      printf("DIIS weights\n");
      print_mat(w.t(),"% .2e ");
    }    
  } else if(useadiis && usediis) {
    // Sliding scale
    double diisw=std::max(std::min(1.0 - (err-diisthr)/(diiseps-diisthr), 1.0), 0.0);

    // Determine cooloff
    if(cooloff>0) {
      diisw=0.0;
      cooloff--;
    } else {
      // Check if energy has increased
      arma::vec E=get_energies();
      if(E.n_elem>1 &&  E(E.n_elem-1)-E(E.n_elem-2) > COOLTHR) {
	cooloff=2;
	diisw=0.0;
      }
    }

    arma::vec wa=get_w_adiis();
    arma::vec wd=get_w_diis();
    w=diisw*wd + (1.0-diisw)*wa;
    
    if(verbose) {
      printf("ADIIS weights\n");
      print_mat(wa.t(),"% .2e ");
      printf("CDIIS weights\n");
      print_mat(wd.t(),"% .2e ");
      printf(" DIIS weights\n");
      print_mat(w.t(),"% .2e ");
    }    
    
  } else
    throw std::runtime_error("Nor DIIS or ADIIS has been turned on.\n");
    
  return w;
}

arma::vec DIIS::get_w_diis() const {
  arma::mat errs=get_diis_error();
  return get_w_diis_wrk(errs);
}
  
arma::vec DIIS::get_w_diis_wrk(const arma::mat & errs) const {
  // Size of LA problem
  int N=(int) errs.n_cols;
  
  // DIIS weights
  arma::vec sol(N);
  sol.zeros();
  
  if(c1diis) {
    // Original, Pulay's DIIS (C1-DIIS)
    
    // Array holding the errors
    arma::mat B(N+1,N+1);
    B.zeros();
    // Compute errors
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++) {
	B(i,j)=arma::dot(errs.col(i),errs.col(j));
      }
    // Fill in the rest of B
    for(int i=0;i<N;i++) {
      B(i,N)=-1.0;
      B(N,i)=-1.0;
    }
    
    // RHS vector
    arma::vec A(N+1);
    A.zeros();
    A(N)=-1.0;
    
    // Solve B*X = A
    arma::vec X;
    bool succ;
    
    succ=arma::solve(X,B,A);
    
    if(succ) {
      // Solution is (last element of X is DIIS error)
      sol=X.subvec(0,N-1);
      
      // Check that weights are within tolerance
      if(arma::max(arma::abs(sol))>=MAXWEIGHT) {
	printf("Large coefficient produced by DIIS. Reducing to %i matrices.\n",N-1);
	arma::vec w0=get_w_diis_wrk(errs.submat(1,1,errs.n_rows-1,errs.n_cols-1));
	
	// Helper vector
	arma::vec w(N);
	w.zeros();
	w.subvec(w.n_elem-w0.n_elem,w.n_elem-1)=w0;
	return w;
      }
    }
    
    if(!succ) {
      // Failed to invert matrix. Use the two last iterations instead.
      printf("C1-DIIS was not succesful, mixing matrices instead.\n");
      sol.zeros();
      sol(0)=0.5;
      sol(1)=0.5;
    }
  } else {
    // C2-DIIS

    // Array holding the errors
    arma::mat B(N,N);
    B.zeros();
    // Compute errors
    for(int i=0;i<N;i++)
      for(int j=0;j<N;j++) {
	B(i,j)=arma::dot(errs.col(i),errs.col(j));
      }

    // Solve eigenvectors of B
    arma::mat Q;
    arma::vec lambda;
    eig_sym_ordered(lambda,Q,B);

    // Normalize weights
    for(int i=0;i<N;i++) {
      Q.col(i)/=arma::sum(Q.col(i));
    }

    // Choose solution by picking out solution with smallest error
    arma::vec errors(N);
    arma::mat eQ=errs*Q;
    // The weighted error is
    for(int i=0;i<N;i++) {
      errors(i)=arma::norm(eQ.col(i),2);
    }

    // Find minimal error
    double mine=DBL_MAX;
    int minloc=-1;
    for(int i=0;i<N;i++) {
      if(errors[i]<mine) {
	// Check weights
	bool ok=arma::max(arma::abs(Q.col(i)))<MAXWEIGHT;
	if(ok) {
	  mine=errors(i);
	  minloc=i;
	}
      }
    }

    if(minloc!=-1) {
      // Solution is
      sol=Q.col(minloc);
    } else {
      printf("C2-DIIS did not find a suitable solution. Mixing matrices instead.\n");
      
      sol.zeros();
      sol(0)=0.5;
      sol(1)=0.5;
    }
  }
  
  return sol;
}

void rDIIS::solve_F(arma::mat & F) {
  arma::vec sol=get_w();
 
  // Form weighted Fock matrix
  F.zeros();
  for(size_t i=0;i<stack.size();i++)
    F+=sol(i)*stack[i].F;
}

void uDIIS::solve_F(arma::mat & Fa, arma::mat & Fb) {
  arma::vec sol=get_w();
 
  // Form weighted Fock matrix
  Fa.zeros();
  Fb.zeros();
  for(size_t i=0;i<stack.size();i++) {
    Fa+=sol(i)*stack[i].Fa;
    Fb+=sol(i)*stack[i].Fb;
  }
}

void rDIIS::solve_P(arma::mat & P) {
  arma::vec sol=get_w();
 
  // Form weighted density matrix
  P.zeros();
  for(size_t i=0;i<stack.size();i++)
    P+=sol(i)*stack[i].P;
}

void uDIIS::solve_P(arma::mat & Pa, arma::mat & Pb) {
  arma::vec sol=get_w();
 
  // Form weighted density matrix
  Pa.zeros();
  Pb.zeros();
  for(size_t i=0;i<stack.size();i++) {
    Pa+=sol(i)*stack[i].Pa;
    Pb+=sol(i)*stack[i].Pb;
  }
}

arma::vec DIIS::get_w_adiis() const {
  // Number of parameters
  size_t N=PiF.n_elem;

  if(N==1) {
    // Trivial case.
    arma::vec ret(1);
    ret.ones();
    return ret;
  }

  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;

  gsl_vector *x;
  gsl_multimin_function_fdf minfunc;
  minfunc.f = adiis::min_f;
  minfunc.df = adiis::min_df;
  minfunc.fdf = adiis::min_fdf;
  minfunc.n = N;
  minfunc.params = (void *) this;

  T=gsl_multimin_fdfminimizer_vector_bfgs2;
  s=gsl_multimin_fdfminimizer_alloc(T,N);

  // Starting point: equal weights on all matrices
  x=gsl_vector_alloc(N);
  gsl_vector_set_all(x,1.0/N);

  // Initial energy estimate
  // double E_initial=get_E(x);

  // Initialize the optimizer. Use initial step size 0.02, and an
  // orthogonality tolerance of 0.1 in the line searches (recommended
  // by GSL manual for bfgs).
  gsl_multimin_fdfminimizer_set(s, &minfunc, x, 0.02, 0.1);

  size_t iter=0;
  int status;
  do {
    iter++;
    //    printf("iteration %lu\n",iter);
    status = gsl_multimin_fdfminimizer_iterate (s);

    if (status) {
      //      printf("Error %i in minimization\n",status);
      break;
    }

    status = gsl_multimin_test_gradient (s->gradient, 1e-7);

    /*
    if (status == GSL_SUCCESS)
      printf ("Minimum found at:\n");

    printf("%5lu ", iter);
    for(size_t i=0;i<N;i++)
      printf("%.5g ",gsl_vector_get(s->x,i));
    printf("%10.5g\n",s->f);
    */
  }
  while (status == GSL_CONTINUE && iter < 1000);

  // Final estimate
  // double E_final=get_E(s->x);

  // Form minimum
  arma::vec c=adiis::compute_c(s->x);

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  //  printf("Minimized estimate of %lu matrices by %e from %e to %e in %lu iterations.\n",D.size(),E_final-E_initial,E_initial,E_final,iter);

  return c;
}

double DIIS::get_E_adiis(const gsl_vector * x) const {
  // Consistency check
  if(x->size != PiF.n_elem) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }

  arma::vec c=adiis::compute_c(x);

  // Compute energy
  double Eval=0.0;
  Eval+=2.0*arma::dot(c,PiF);
  Eval+=arma::as_scalar(arma::trans(c)*PiFj*c);

  return Eval;
}

void DIIS::get_dEdx_adiis(const gsl_vector * x, gsl_vector * dEdx) const {
  // Compute contraction coefficients
  arma::vec c=adiis::compute_c(x);

  // Compute derivative of energy
  arma::vec dEdc=2.0*PiF + PiFj*c + arma::trans(PiFj)*c;

  // Compute jacobian of transformation: jac(i,j) = dc_i / dx_j
  arma::mat jac=adiis::compute_jac(x);

  // Finally, compute dEdx by plugging in Jacobian of transformation
  // dE/dx_i = dc_j/dx_i dE/dc_j
  arma::vec dEdxv=arma::trans(jac)*dEdc;
  for(size_t i=0;i<PiF.n_elem;i++)
    gsl_vector_set(dEdx,i,dEdxv(i));
}

void DIIS::get_E_dEdx_adiis(const gsl_vector * x, double * Eval, gsl_vector * dEdx) const {
  // Consistency check
  if(x->size != PiF.n_elem) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }
  if(x->size != dEdx->size) {
    ERROR_INFO();
    throw std::domain_error("x and dEdx have different sizes!\n");
  }

  // Compute energy
  *Eval=get_E_adiis(x);
  // and its derivative
  get_dEdx_adiis(x,dEdx);
}

double adiis::min_f(const gsl_vector * x, void * params) {
  DIIS * a=(DIIS *) params;
  return a->get_E_adiis(x);
}

void adiis::min_df(const gsl_vector * x, void * params, gsl_vector * g) {
  DIIS * a=(DIIS *) params;
  a->get_dEdx_adiis(x,g);
}

void adiis::min_fdf(const gsl_vector *x, void * params, double * f, gsl_vector * g) {
  DIIS * a=(DIIS *) params;
  a->get_E_dEdx_adiis(x,f,g);
}

arma::vec adiis::compute_c(const gsl_vector * x) {
  // Compute contraction coefficients
  arma::vec c(x->size);

  double xnorm=0.0;
  for(size_t i=0;i<x->size;i++) {
    c[i]=gsl_vector_get(x,i);
    c[i]=c[i]*c[i]; // c_i = x_i^2
    xnorm+=c[i];
  }
  for(size_t i=0;i<x->size;i++)
    c[i]/=xnorm; // c_i = x_i^2 / \sum_j x_j^2

  return c;
}

arma::mat adiis::compute_jac(const gsl_vector * x) {
  // Compute jacobian of transformation: jac(i,j) = dc_i / dx_j

  // Compute coefficients
  std::vector<double> c(x->size);

  double xnorm=0.0;
  for(size_t i=0;i<x->size;i++) {
    c[i]=gsl_vector_get(x,i);
    c[i]=c[i]*c[i]; // c_i = x_i^2
    xnorm+=c[i];
  }
  for(size_t i=0;i<x->size;i++)
    c[i]/=xnorm; // c_i = x_i^2 / \sum_j x_j^2

  arma::mat jac(c.size(),c.size());
  for(size_t i=0;i<c.size();i++) {
    double xi=gsl_vector_get(x,i);

    for(size_t j=0;j<c.size();j++) {
      double xj=gsl_vector_get(x,j);

      jac(i,j)=-c[i]*2.0*xj/xnorm;
    }

    // Extra term on diagonal
    jac(i,i)+=2.0*xi/xnorm;
  }

  return jac;
}
