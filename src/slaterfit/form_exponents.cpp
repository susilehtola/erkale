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


int main(int argc, char **argv) {
  
  if(argc!=5) {
    printf("Usage: %s zeta l Nf method\n",argv[0]);
    printf("zeta is the STO exponent to fit\n");
    printf("l is angular momentum to use\n");
    printf("Nf is number of exponents to use\n");
    printf("method is 0 for even-tempered, 1 for well-tempered and 2 for full optimization.\n");
    return 1;
  }

  sto_params_t par;
  par.zeta=atof(argv[1]);
  par.l=atoi(argv[2]);
  par.Nf=atoi(argv[3]);
  par.method=atoi(argv[4]);

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
  for(int i=0;i<dof;i++)
    gsl_vector_set(x,i,i);
  // Set initial step sizes
  gsl_vector_set_all(ss,1.0);

  // Set minimizer
  gsl_multimin_fminimizer_set(min, &minfunc, x, ss);
  printf("Minimizer set.\n\n");

  // Iterate
  int iter=0;
  int iterdelta=0;
  int status;
  double size;
  double cost=0;

  printf("Iteration\tDelta\n");
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
    if (status == GSL_SUCCESS)
      {
        printf ("converged to minimum at\n");
      }

    if(min->fval!=cost) {
      printf("%i\t%e\t%e\n",iter,min->fval,min->fval-cost);
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

  // Print them out
  printf("\nExponential contraction:\n");
  for(size_t i=0;i<optexp.size();i++)
    printf("%e\t%e\n",optc(i),optexp[i]);

  FILE *out;
  out=fopen("slater-contr.gbs","w");
  // Write out name of element
  fprintf(out,"H\t0\n");
  // Print out type and length of shell
  fprintf(out,"%c   %i   1.00\n",shell_types[par.l],(int) optexp.size());
  // Print out contraction
  for(size_t iexp=0;iexp<optexp.size();iexp++)
    fprintf(out,"\t%.16e\t\t%.16e\n",optexp[iexp],optc(iexp));
  // Close entry
  fprintf(out,"****\n");
  fclose(out);

  out=fopen("slater-uncontr.gbs","w");
  // Write out name of element
  fprintf(out,"H\t0\n");
  for(size_t iexp=0;iexp<optexp.size();iexp++) {
    // Print out type and length of shell
    fprintf(out,"%c   %i   1.00\n",shell_types[par.l],1);
    // Print out exponent
    fprintf(out,"\t%.16e\t\t%.16e\n",optexp[iexp],1.0);
  }
  // Close entry
  fprintf(out,"****\n");
  fclose(out);

  return 0;
}
