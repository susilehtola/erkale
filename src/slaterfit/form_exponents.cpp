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



#include <algorithm>
#include "solve_coefficients.h"
#include "../basis.h"
#include "../tempered.h"

// Minimization routines
extern "C" {
#include <gsl/gsl_multimin.h>
}

/// Parameters of Slater function to fit
typedef struct {
  /// Exponent
  double zeta;
  /// Angular momentum
  int l;
  /// Method to use: 0 for even-tempered, 1 for well-tempered, 2 for full optimization
  int method;
  /// Number of functions to use for fit
  int Nf;
} sto_params_t;

std::vector<double> get_exps_full(const gsl_vector *v, size_t Nf) {
  // Helper array
  std::vector<double> exps;
  for(size_t i=0;i<Nf;i++)
    exps.push_back(exp(gsl_vector_get(v,i)));

  // Sort exponents
  stable_sort(exps.begin(),exps.end());

  return exps;
}

std::vector<double> get_exps_welltempered(const gsl_vector *v, int Nf) {

  std::vector<double> exps=welltempered_set(exp(gsl_vector_get(v,0)),exp(gsl_vector_get(v,1)),exp(gsl_vector_get(v,2)),exp(gsl_vector_get(v,3)),Nf);

  // Sort exponents
  stable_sort(exps.begin(),exps.end());

  return exps;
}

std::vector<double> get_exps_eventempered(const gsl_vector *v, int Nf) {

  std::vector<double> exps=eventempered_set(exp(gsl_vector_get(v,0)),exp(gsl_vector_get(v,1)),Nf);

  // Sort exponents
  stable_sort(exps.begin(),exps.end());

  return exps;
}

std::vector<double> get_exps(const gsl_vector *v, const sto_params_t *p) {
  std::vector<double> exps;
  if(p->method==0)
    exps=get_exps_eventempered(v,p->Nf);
  else if(p->method==1)
    exps=get_exps_welltempered(v,p->Nf);
  else
    exps=get_exps_full(v,p->Nf);

  return exps;
}

// Evaluate difference from identity
double eval_difference(const gsl_vector *v, void *params) {
  // Parameters
  sto_params_t *p=(sto_params_t *)params;

  // Form vector of exponents
  std::vector<double> exps=get_exps(v,p);

  // Compute difference
  return compute_difference(exps,p->zeta,p->l);
}


std::vector<contr_t> slater_fit(double zeta, int am, int nf, bool verbose, int method) {
  sto_params_t par;
  par.zeta=zeta;
  par.l=am;
  par.Nf=nf;
  par.method=method;

  int maxiter=1000;

  // Degrees of freedom
  int dof;
  if(par.method==0)
    dof=2;
  else if(par.method==1)
    dof=4;
  else
    dof=par.Nf;

  gsl_multimin_function minfunc;
  minfunc.n=dof;
  minfunc.f=eval_difference;
  minfunc.params=(void *) &par;

  gsl_multimin_fminimizer *min;
  // Allocate minimizer
  min=gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2,dof);

  gsl_vector *x, *ss;
  x=gsl_vector_alloc(dof);
  ss=gsl_vector_alloc(dof);
  
  // Set starting point
  switch(par.method) {

  case(1):
    // Well tempered - set gamma and delta to 0
    gsl_vector_set_all(x,0.0);
  case(0):
    // Even tempered, set alpha=1.0 and beta=2.0
    gsl_vector_set(x,0,1.0);
    gsl_vector_set(x,1,2.0);
    break;

  case(2):
    // Free minimization, set exponents to i
    for(int i=0;i<nf;i++)
      gsl_vector_set(x,i,i);
    break;

  default:
    ERROR_INFO();
    throw std::runtime_error("Unknown Slater fitting method.\n");
  }

  // Set initial step sizes
  gsl_vector_set_all(ss,0.1);

  // Set minimizer
  gsl_multimin_fminimizer_set(min, &minfunc, x, ss);

  // Iterate
  int iter=0;
  int iterdelta=0;
  int status;
  double size;
  double cost=0;

  if(verbose) printf("Iteration\tDelta\n");
  do {
    iter++;
    iterdelta++;
    
    // Simplex
    status = gsl_multimin_fminimizer_iterate(min);
    if (status) 
      break;

    // Get convergence estimate.
    size = gsl_multimin_fminimizer_size (min);
    
    // Are we converged?
    status = gsl_multimin_test_size (size, DBL_EPSILON);
    if (verbose && status == GSL_SUCCESS)
      {
        printf ("converged to minimum at\n");
      }

    if(min->fval!=cost) {
      if(verbose) printf("%i\t%e\t%e\n",iter,min->fval,min->fval-cost);
      cost=min->fval;
      iterdelta=0;
    }

  } while (status == GSL_CONTINUE && iterdelta < maxiter);
  
  // Get best exponents and coefficients
  std::vector<double> optexp=get_exps(min->x,&par);
  arma::vec optc=solve_coefficients(optexp,par.zeta,par.l);

  // Free memory
  gsl_vector_free(x);
  gsl_vector_free(ss);  
  gsl_multimin_fminimizer_free(min);

  // Return
  std::vector<contr_t> ret(nf);
  for(int i=0;i<nf;i++) {
    ret[i].z=optexp[i];
    ret[i].c=optc[i];
  }

  return ret;
}
