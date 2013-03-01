/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#include "ediis.h"
#include <gsl/gsl_rng.h>

bool operator<(const ediis_t & lhs, const ediis_t & rhs) {
  return lhs.E<rhs.E;
}

EDIIS::EDIIS(size_t m) {
  max=m;
}

EDIIS::~EDIIS() {
}

void EDIIS::push(double Es, const arma::mat & Ps, const arma::mat & Fs) {
  // Add to stack
  ediis_t hlp;
  hlp.E=Es;
  hlp.P=Ps;
  hlp.F=Fs;
  stack.push_back(hlp);

  // Sort stack
  std::stable_sort(stack.begin(),stack.end());

  // Delete entry with highest energy
  if(stack.size()>max) {
    stack.erase(stack.end());
  }

  // Update trace matrix
  FDtr.zeros(stack.size(),stack.size());
  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<=i;j++) {
      FDtr(i,j)=arma::trace((stack[i].F-stack[j].F)*(stack[i].P-stack[j].P));
      FDtr(j,i)=FDtr(i,j);
    }     
}

void EDIIS::clear() {
  stack.clear();
}  

arma::mat EDIIS::get_P() const {
  // Get coefficients
  std::vector<double> c=get_c();

  /*
  printf("EDIIS weights are");
  for(size_t i=0;i<c.size();i++)
    printf(" % e",c[i]);
  printf("\n");
  */

  arma::mat ret=c[0]*stack[0].P;
  for(size_t i=1;i<stack.size();i++)
    ret+=c[i]*stack[i].P;

  return ret;
}

arma::mat EDIIS::get_H() const {
  // Get coefficients
  std::vector<double> c=get_c();

  /*
  printf("EDIIS weights are");
  for(size_t i=0;i<c.size();i++)
    printf(" % e",c[i]);
  printf("\n");
  */

  arma::mat H=c[0]*stack[0].F;
  for(size_t i=1;i<stack.size();i++)
    H+=c[i]*stack[i].F;

  return H;
}

std::vector<double> EDIIS::get_c() const {
  // Number of parameters
  size_t N=stack.size();

  if(N==1) {
    // Trivial case.
    std::vector<double> ret;
    ret.push_back(1.0);
    return ret;
  }    

  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;

  gsl_vector *x;
  gsl_multimin_function_fdf minfunc;
  minfunc.f = ediis::min_f;
  minfunc.df = ediis::min_df;
  minfunc.fdf = ediis::min_fdf;
  minfunc.n = N;
  minfunc.params = (void *) this;
  
  T=gsl_multimin_fdfminimizer_vector_bfgs2;
  //  T=gsl_multimin_fdfminimizer_conjugate_fr;
  s=gsl_multimin_fdfminimizer_alloc(T,N);

  // Starting point: equal weights on all matrices
  x=gsl_vector_alloc(N);
  gsl_vector_set_all(x,1.0/N);

  // Initial energy estimate
  // double E_initial=get_E(x);

  // Initialize the optimizer. Use initial step size 0.02, and an
  // orthogonality tolerance of 0.01 in the line searches.
  gsl_multimin_fdfminimizer_set(s, &minfunc, x, 0.02, 0.01);

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
  std::vector<double> c=ediis::compute_c(s->x);
     
  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  //  printf("Minimized estimate of %lu matrices by %e from %e to %e in %lu iterations.\n",D.size(),E_final-E_initial,E_initial,E_final,iter);

  return c;
}

double EDIIS::get_E(const gsl_vector * x) const {
  // Consistency check
  if(x->size != stack.size()) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }

  std::vector<double> c=ediis::compute_c(x);

  // Compute energy
  double Eval=0.0;
  for(size_t i=0;i<stack.size();i++)
    Eval+=c[i]*stack[i].E;

  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<i;j++)
      Eval-=0.5*c[i]*c[j]*FDtr(i,j);
  
  return Eval;
}
    

void EDIIS::get_dEdx(const gsl_vector * x, gsl_vector * dEdx) const {
  // Consistency check
  if(x->size != stack.size()) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }
  if(x->size != dEdx->size) {
    ERROR_INFO();
    throw std::domain_error("x and dEdx have different sizes!\n");
  }

  // Compute contraction coefficients
  std::vector<double> c=ediis::compute_c(x);

  // Compute derivative of energy
  std::vector<double> dEdc;
  for(size_t i=0;i<stack.size();i++) {
    // Constant term
    double el=stack[i].E;
    
    // FD contribution
    for(size_t j=0;j<stack.size();j++)
      el+=c[j]*FDtr(j,i);

    dEdc.push_back(el);
  }

  // Compute jacobian of transformation: jac(i,j) = dc_i / dx_j
  arma::mat jac=ediis::compute_jac(x);

  // Finally, compute dEdx by plugging in Jacobian of transformation
  // dE/dx_i = dc_j/dx_i dE/dc_j
  
  for(size_t i=0;i<stack.size();i++) {
    double d=0.0;
    for(size_t j=0;j<stack.size();j++)
      d+=jac(j,i)*dEdc[j];
    
    gsl_vector_set(dEdx,i,d);
  }

  /*
  printf("get_dEdx finished\n");
  printf("dEdc=(");
  for(size_t i=0;i<D.size();i++)
    printf(" %e,",dEdc[i]);
  printf(")\n");

  printf("dEdx=(");
  for(size_t i=0;i<D.size();i++)
    printf(" %e,",gsl_vector_get(dEdx,i));
  printf(")\n");
  */
}
    

void EDIIS::get_E_dEdx(const gsl_vector * x, double * Eval, gsl_vector * dEdx) const {
  // Consistency check
  if(x->size != stack.size()) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }
  if(x->size != dEdx->size) {
    ERROR_INFO();
    throw std::domain_error("x and dEdx have different sizes!\n");
  }

  // Compute energy
  *Eval=get_E(x);

  // Compute derivative of energy
  get_dEdx(x,dEdx);
}
    
double ediis::min_f(const gsl_vector * x, void * params) {
  EDIIS * a=(EDIIS *) params;
  return a->get_E(x);
}

void ediis::min_df(const gsl_vector * x, void *params, gsl_vector * g) {
  EDIIS * a=(EDIIS *) params;
  a->get_dEdx(x,g);
}

void ediis::min_fdf(const gsl_vector *x, void* params, double * f, gsl_vector * g) {
  EDIIS * a=(EDIIS *) params;
  a->get_E_dEdx(x,f,g);
}

std::vector<double> ediis::compute_c(const gsl_vector * x) {
  // Compute contraction coefficients
  std::vector<double> c(x->size);

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

arma::mat ediis::compute_jac(const gsl_vector * x) {
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
