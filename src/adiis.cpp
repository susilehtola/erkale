/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
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

ADIIS::~ADIIS() {
}

void ADIIS::push(double Es, const arma::mat & Ds, const arma::mat & Fs) {
  E.push_back(Es);
  D.push_back(Ds);
  F.push_back(Fs);

  if(D.size()>max) {
    E.erase(E.begin());
    D.erase(D.begin());
    F.erase(F.begin());
  }
}

void ADIIS::clear() {
  E.clear();
  D.clear();
  F.clear();
}  

arma::mat ADIIS::get_D() const {
  // Get coefficients
  std::vector<double> c=get_c();

  /*
  printf("ADIIS weights are");
  for(size_t i=0;i<c.size();i++)
    printf(" % e",c[i]);
  printf("\n");
  */

  arma::mat ret=c[0]*D[0];
  for(size_t i=1;i<D.size();i++)
    ret+=c[i]*D[i];

  return ret;
}

arma::mat ADIIS::get_H() const {
  // Get coefficients
  std::vector<double> c=get_c();

  /*
  printf("ADIIS weights are");
  for(size_t i=0;i<c.size();i++)
    printf(" % e",c[i]);
  printf("\n");
  */

  arma::mat H=c[0]*F[0];
  for(size_t i=1;i<D.size();i++)
    H+=c[i]*F[i];

  return H;
}

std::vector<double> ADIIS::get_c() const {
  // Number of parameters
  size_t N=D.size();

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
  minfunc.f = min_f;
  minfunc.df = min_df;
  minfunc.fdf = min_fdf;
  minfunc.n = N;
  minfunc.params = (void *) this;
  
  T=gsl_multimin_fdfminimizer_vector_bfgs2;
  //  T=gsl_multimin_fdfminimizer_conjugate_fr;
  s=gsl_multimin_fdfminimizer_alloc(T,N);

  // Starting point: equal weights on all matrices
  x=gsl_vector_alloc(N);
  gsl_vector_set_all(x,1.0/N);

  // Initial estimate
  double E_initial=get_E(x);

  // Step size 0.01, stop optimization with tolerance 1e-4
  gsl_multimin_fdfminimizer_set(s, &minfunc, x, 0.01, 1e-4);

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
  double E_final=get_E(s->x);

  // Form minimum
  std::vector<double> c=compute_c(s->x);
     
  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  //  printf("Minimized estimate of %lu matrices by %e from %e to %e in %lu iterations.\n",D.size(),E_final-E_initial,E_initial,E_final,iter);

  return c;
}

double ADIIS::get_E(const gsl_vector * x) const {
  // Consistency check
  if(x->size != D.size()) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }

  std::vector<double> c=compute_c(x);

  // Compute averaged D and F
  arma::mat cF=c[0]*F[0]; // c_j F_j
  for(size_t i=1;i<F.size();i++)
    cF+=c[i]*F[i];

  arma::mat cD=c[0]*D[0]; // c_j D_j
  for(size_t i=1;i<D.size();i++)
    cD+=c[i]*D[i];

  arma::mat Dn=D[D.size()-1];
  arma::mat Fn=F[F.size()-1];

  // Compute energy
  double Eval=E[E.size()-1];
  Eval+=2.0*arma::trace(arma::trans(cD-Dn)*Fn);
  Eval+=arma::trace(arma::trans(cD-Dn)*(cF-Fn));
  
  return Eval;
}
    

void ADIIS::get_dEdx(const gsl_vector * x, gsl_vector * dEdx) const {
  // Consistency check
  if(x->size != D.size()) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }
  if(x->size != dEdx->size) {
    ERROR_INFO();
    throw std::domain_error("x and dEdx have different sizes!\n");
  }

  // Compute contraction coefficients
  std::vector<double> c=compute_c(x);

  // Compute averaged D and F
  arma::mat cF=c[0]*F[0]; // c_j F_j
  for(size_t i=1;i<F.size();i++)
    cF+=c[i]*F[i];

  arma::mat cD=c[0]*D[0]; // c_j D_j
  for(size_t i=1;i<D.size();i++)
    cD+=c[i]*D[i];

  arma::mat Dn=D[D.size()-1];
  arma::mat Fn=F[F.size()-1];
 
  // Compute derivative of energy
  std::vector<double> dEdc;
  for(size_t i=0;i<D.size();i++) {
    // Constant term
    double d=2.0*arma::trace(arma::trans(D[i]-Dn)*Fn);
    d+=arma::trace(arma::trans(D[i]-Dn)*(cF-Fn))+arma::trace(arma::trans(cD-Dn)*(F[i]-Fn));

    dEdc.push_back(d);
  }

  // Compute jacobian of transformation: jac(i,j) = dc_i / dx_j
  arma::mat jac=compute_jac(x);

  // Finally, compute dEdx by plugging in Jacobian of transformation
  // dE/dx_i = dc_j/dx_i dE/dc_j
  
  for(size_t i=0;i<D.size();i++) {
    double d=0.0;
    for(size_t j=0;j<D.size();j++)
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
    

void ADIIS::get_E_dEdx(const gsl_vector * x, double * Eval, gsl_vector * dEdx) const {
  // Consistency check
  if(x->size != D.size()) {
    ERROR_INFO();
    throw std::domain_error("Incorrect number of parameters.\n");
  }
  if(x->size != dEdx->size) {
    ERROR_INFO();
    throw std::domain_error("x and dEdx have different sizes!\n");
  }

  // Compute contraction coefficients
  std::vector<double> c=compute_c(x);

  // Compute averaged D and F
  arma::mat cF=c[0]*F[0]; // c_j F_j
  for(size_t i=1;i<F.size();i++)
    cF+=c[i]*F[i];

  arma::mat cD=c[0]*D[0]; // c_j D_j
  for(size_t i=1;i<D.size();i++)
    cD+=c[i]*D[i];

  arma::mat Dn=D[D.size()-1];
  arma::mat Fn=F[F.size()-1];

  // Compute energy
  *Eval=E[E.size()-1];
  *Eval+=2.0*arma::trace(arma::trans(cD-Dn)*Fn);
  *Eval+=arma::trace(arma::trans(cD-Dn)*(cF-Fn));

  // Compute derivative of energy
  std::vector<double> dEdc;
  for(size_t i=0;i<D.size();i++) {
    // Constant term
    double d=2.0*arma::trace(arma::trans(D[i]-Dn)*Fn);
    d+=arma::trace(arma::trans(D[i]-Dn)*(cF-Fn))+arma::trace(arma::trans(cD-Dn)*(F[i]-Fn));

    dEdc.push_back(d);
  }

  // Compute jacobian of transformation: jac(i,j) = dc_i / dx_j
  arma::mat jac=compute_jac(x);

  //  printf("Jacobian is\n");
  //  jac.print();

  // Finally, compute dEdx by plugging in Jacobian of transformation
  // dE/dx_i = dc_j/dx_i dE/dc_j
  
  for(size_t i=0;i<D.size();i++) {
    double d=0.0;
    for(size_t j=0;j<D.size();j++)
      d+=jac(j,i)*dEdc[j];
    gsl_vector_set(dEdx,i,d);
  }

  /*
  printf("E=%e\n",*Eval);

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
    
double min_f(const gsl_vector * x, void * params) {
  ADIIS * a=(ADIIS *) params;
  return a->get_E(x);
}

void min_df(const gsl_vector * x, void *params, gsl_vector * g) {
  ADIIS * a=(ADIIS *) params;
  a->get_dEdx(x,g);
}

void min_fdf(const gsl_vector *x, void* params, double * f, gsl_vector * g) {
  ADIIS * a=(ADIIS *) params;
  a->get_E_dEdx(x,f,g);
}

std::vector<double> compute_c(const gsl_vector * x) {
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

arma::mat compute_jac(const gsl_vector * x) {
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
