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



#include "adiis.h"
#include <gsl/gsl_rng.h>

ADIIS::ADIIS(size_t m) {
  max=m;
}

rADIIS::rADIIS(size_t m): ADIIS(m) {
}

uADIIS::uADIIS(size_t m): ADIIS(m) {
}

ADIIS::~ADIIS() {
}

rADIIS::~rADIIS() {
}

uADIIS::~uADIIS() {
}

void rADIIS::update(const arma::mat & Pn, const arma::mat & Fn, double En) {
  radiis_entry_t tmp;
  tmp.E=En;
  tmp.P=Pn;
  tmp.F=Fn;
  stack.push_back(tmp);

  if(stack.size()>max) {
    stack.erase(stack.begin());
  }

  // Update matrices
  PiF.zeros(stack.size());
  for(size_t i=0;i<stack.size();i++)
    PiF(i)=arma::trace((stack[i].P-Pn)*Fn);

  PiFj.zeros(stack.size(),stack.size());
  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<stack.size();j++)
      PiFj(i,j)=arma::trace((stack[i].P-Pn)*(stack[j].F-Fn));
}

void uADIIS::update(const arma::mat & Pan, const arma::mat & Pbn, const arma::mat & Fan, const arma::mat & Fbn, double En) {
  uadiis_entry_t tmp;
  tmp.E=En;
  tmp.Pa=Pan;
  tmp.Pb=Pbn;
  tmp.Fa=Fan;
  tmp.Fb=Fbn;
  stack.push_back(tmp);

  if(stack.size()>max) {
    stack.erase(stack.begin());
  }

  // Update matrices
  PiF.zeros(stack.size());
  for(size_t i=0;i<stack.size();i++)
    PiF(i)=arma::trace((stack[i].Pa-Pan)*Fan) + arma::trace((stack[i].Pb-Pbn)*Fbn);

  PiFj.zeros(stack.size(),stack.size());
  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<stack.size();j++)
      PiFj(i,j)=arma::trace((stack[i].Pa-Pan)*(stack[j].Fa-Fan))+arma::trace((stack[i].Pb-Pbn)*(stack[j].Fb-Fbn));
}

void rADIIS::clear() {
  stack.clear();
}

void uADIIS::clear() {
  stack.clear();
}

void rADIIS::get_P(arma::mat & P, bool verbose) const {
  // Get coefficients
  arma::vec c=get_c(verbose);

  P.zeros(stack[0].P.n_rows,stack[0].P.n_cols);
  for(size_t i=0;i<stack.size();i++)
    P+=c[i]*stack[i].P;
}

void uADIIS::get_P(arma::mat & Pa, arma::mat & Pb, bool verbose) const {
  // Get coefficients
  arma::vec c=get_c(verbose);
  
  Pa.zeros(stack[0].Pa.n_rows,stack[0].Pa.n_cols);
  Pb.zeros(stack[0].Pb.n_rows,stack[0].Pb.n_cols);
  for(size_t i=0;i<stack.size();i++) {
    Pa+=c[i]*stack[i].Pa;
    Pb+=c[i]*stack[i].Pb;
  }
}

void rADIIS::get_F(arma::mat & F, bool verbose) const {
  // Get coefficients
  arma::vec c=get_c(verbose);

  F.zeros(stack[0].F.n_rows,stack[0].F.n_cols);
  for(size_t i=0;i<stack.size();i++)
    F+=c[i]*stack[i].F;
}

void uADIIS::get_F(arma::mat & Fa, arma::mat & Fb, bool verbose) const {
  // Get coefficients
  arma::vec c=get_c(verbose);

  Fa.zeros(stack[0].Fa.n_rows,stack[0].Fa.n_cols);
  Fb.zeros(stack[0].Fb.n_rows,stack[0].Fb.n_cols);
  for(size_t i=0;i<stack.size();i++) {
    Fa+=c[i]*stack[i].Fa;
    Fb+=c[i]*stack[i].Fb;
  }
}

arma::vec ADIIS::get_c(bool verbose) const {
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
  if(verbose)
    arma::trans(c).print("ADIIS weights");


  return c;
}


double ADIIS::get_E(const gsl_vector * x) const {
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

void ADIIS::get_dEdx(const gsl_vector * x, gsl_vector * dEdx) const {
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

void ADIIS::get_E_dEdx(const gsl_vector * x, double * Eval, gsl_vector * dEdx) const {
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
  *Eval=get_E(x);
  // and its derivative
  get_dEdx(x,dEdx);
}

double adiis::min_f(const gsl_vector * x, void * params) {
  ADIIS * a=(ADIIS *) params;
  return a->get_E(x);
}

void adiis::min_df(const gsl_vector * x, void *params, gsl_vector * g) {
  ADIIS * a=(ADIIS *) params;
  a->get_dEdx(x,g);
}

void adiis::min_fdf(const gsl_vector *x, void* params, double * f, gsl_vector * g) {
  ADIIS * a=(ADIIS *) params;
  a->get_E_dEdx(x,f,g);
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
