/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "completeness_profile.h"
#include "optimize_completeness.h"
#include "../linalg.h"
#include "../timer.h"

extern "C" {
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf_legendre.h>
}

#ifdef _OPENMP
#include <omp.h>
#endif

// Maximum number of functions allowed in completeness optimization
#define NFMAX 50

arma::vec get_exponents(const gsl_vector *xv, const completeness_scan_t * p) {
  // Collect elements of x
  arma::vec x(xv->size);
  for(size_t i=0;i<x.n_elem;i++)
    x(i)=gsl_vector_get(xv,i);

  // Take absolute value and sort into ascending order
  x=sort(arma::abs(x),0);

  // Get exponent values
  arma::vec A;

  // Is the amount of parameters even or odd?
  if(p->odd) {
    A.zeros(2*x.n_elem-1);
    
    // Smallest exponent is at the center
    A(0)=exp(x(0));
    // followed by the other exponents, symmetrically copied
    for(size_t i=1;i<x.n_elem;i++) {
      A(2*i-1)=exp(x(i));
      A(2*i)=exp(-x(i));
    }
    
  } else {
    A.zeros(2*x.n_elem);
    for(size_t i=0;i<x.n_elem;i++) {
      // Exponents are placed symmetrically around the origin.
      A(2*i)=exp(x(i));
      A(2*i+1)=exp(-x(i));
    }
  }

  return sort(A);
}

arma::mat self_overlap(const arma::vec & z, int am) {
  // Compute self-overlap
  size_t N=z.n_elem;
  arma::mat Suv(N,N);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<=i;j++) {
      Suv(i,j)=pow(4.0*z(i)*z(j)/pow(z(i)+z(j),2),am/2.0+0.75);
      Suv(j,i)=Suv(i,j);
    }

  return Suv;
}

arma::vec completeness_profile(const gsl_vector * x, void * params) {
  // Get parameters
  completeness_scan_t *par=(completeness_scan_t *) params;

  // Get exponents
  arma::vec z=get_exponents(x,par);

  // Get self-overlap
  arma::mat Suv=self_overlap(z,par->am);

  // and its half inverse matrix
  arma::mat Sinvh=BasOrth(Suv,false);
 
  // Get overlap of primitives with scanning terms
  arma::mat amu=overlap(par->scanexp,z,par->am);

  // Compute intermediary result, Np x N
  arma::mat J=amu*Sinvh;

  // The completeness profile
  size_t N=par->scanexp.size();
  arma::vec Y(N);
  for(size_t i=0;i<N;i++)
    Y[i]=arma::dot(J.row(i),J.row(i));

  return Y;
}

double compl_mog(const gsl_vector * x, void * params) {
  // Get parameters
  completeness_scan_t *p=(completeness_scan_t *) params;

  // Get completeness profile
  arma::vec Y=completeness_profile(x,params);

  // Compute MOG.
  double phi=0.0;

  size_t nint=0;

  switch(p->n) {

  case(1):
    for(size_t i=1;i<Y.n_elem-1;i+=2) {
      // Compute differences from unity
      double ld=1.0-Y[i-1];
      double md=1.0-Y[i  ];
      double rd=1.0-Y[i+1];
      // Increment profile measure
      phi+=ld+4.0*md+rd;
      nint++;
    }
    break;

  case(2):
    for(size_t i=1;i<Y.n_elem-1;i+=2) {
      // Compute differences from unity
      double ld=1.0-Y[i-1];
      double md=1.0-Y[i  ];
      double rd=1.0-Y[i+1];
      // Increment profile measure
      phi+=ld*ld+4.0*md*md+rd*rd;
      nint++;
    }
    break;

  default:
    ERROR_INFO();
    throw std::runtime_error("Value of n not supported!\n");
  }
  // Plug in normalization factors
  phi/=6.0*nint;

  //  printf("MOG is %e.\n",phi);

  return phi;
}

arma::vec optimize_completeness(int am, double min, double max, int Nf, int n, bool verbose, double *mog) {
  // Time minimization
  Timer tmin;

  // Optimized profile will be always symmetric around the midpoint,
  // so we can use this to reduce the amount of degrees of freedom in
  // the optimization.
  int Ndof=Nf/2;
  if(Nf%2)
    Ndof++;

  // Length of interval is
  double len=max-min;

  // Parameters for the optimization.
  completeness_scan_t pars;
  // Angular momentum
  pars.am=am;
  // Moment to optimize
  pars.n=n;
  // Scanning exponents
  pars.scanexp=get_scanning_exponents(-len/2.0,len/2.0,50*Nf+1);
  // Odd amount of exponents?
  pars.odd=Nf%2;

  // Maximum number of iterations
  size_t maxiter = 10000;

  // GSL stuff
  const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer *s = NULL;
  gsl_multimin_function minfunc;

  size_t iter = 0;
  int status;
  double size;

  /* Starting point */
  gsl_vector *x = gsl_vector_alloc (Ndof);
  for(int i=0;i<Ndof;i++)
    gsl_vector_set(x,i,log(10.0)*((i+0.5)*len/(2.0*Ndof)));

  /* Set initial step sizes to unity */
  gsl_vector *ss = gsl_vector_alloc (Ndof);
  gsl_vector_set_all (ss, 1.0);

  /* Initialize method and iterate */
  minfunc.n = Ndof;
  minfunc.f = compl_mog;
  minfunc.params = (void *) &pars;

  s = gsl_multimin_fminimizer_alloc (T, Ndof);
  gsl_multimin_fminimizer_set (s, &minfunc, x, ss);

  // Progress timer
  Timer t;

  // Legend
  if(verbose) {
    printf("Optimizing tau_%i for a=[%.3f ... %.3f] of %c shell with %i exponents.\n\n",n,min,max,shell_types[am],Nf);

    printf("iter ");
    char num[80];
    for(int i=0;i<Nf;i++) {
      sprintf(num,"lg exp%i",i+1);
      printf("%9s ",num);
    }
    printf(" %12s  %12s\n","tau","size");
  }

  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);

      if (status)
	break;

      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-3);

      if (status == GSL_SUCCESS && verbose)
	printf ("converged to minimum at\n");

      if(verbose) {
	t.set();
	printf("%4u ",(unsigned int) iter);
	for(int i=0;i<Ndof;i++)
	  // Convert to 10-base logarithm
	  printf("% 9.5f ",log10(M_E)*gsl_vector_get(s->x,i));
	printf(" %e  %e\n",pow(s->fval,1.0/n),size);

	// print_gradient(s->x,(void *) &pars);
      }
    }
  while (status == GSL_CONTINUE && iter < maxiter);

  // Save mog
  if(mog!=NULL)
    *mog=pow(s->fval,1.0/n);

  // The optimized exponents in descending order
  arma::vec exps=arma::sort(get_exponents(s->x,&pars),1);
  // Move starting point to start
  exps*=pow(10.0,min+len/2.0);

  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);

  if(verbose)
    printf("\nMinimization completed in %s.\n",tmin.elapsed().c_str());

  return exps;
}

double maxwidth(int am, double tol, int nexp, int nval) {
  // Dummy value
  double width=-1.0;
  maxwidth_exps(am,tol,nexp,&width,nval);
  return width;
}

arma::vec maxwidth_exps(int am, double tol, int nexp, double *width, int nval) {
  // Error check
  if(nexp<=0) {
    arma::vec exps;
    return exps;
  }

  if(tol<MINTAU) {
    //    printf("Renormalized CO tolerance to 1e-5.\n");
    tol=MINTAU;
  }

  // Left value - vanishing plateau, i.e. too large completeness value
  double left=0.0;

  // Right value - pretty big plateau, i.e. too small completeness value
  double right=0.5*nexp;
  double rval;

  // Check that right value is OK
  arma::vec rexps=optimize_completeness(am,0.0,right,nexp,nval,false,&rval);
  while(rval<tol) {
    left=right;
    right*=2.0;
    rexps=optimize_completeness(am,0.0,right,nexp,nval,false,&rval);
  }

  // Left value
  if(left==0.0) {
    double lval=rval;
    arma::vec lexps;
    while(lval>tol) {
      left/=2.0;
      lexps=optimize_completeness(am,0.0,left,nexp,nval,false,&lval);
    }
  }

#ifndef _OPENMP
  int nth=1;
#else
  int nth=omp_get_max_threads();
#endif
  
  // Worker stack  
  std::vector<arma::vec> expstack(nth);
  // Widths
  std::vector<double> wstack(nth);
  // Tolerances
  std::vector<double> tstack(nth);
  
  do {
    // Set widths
    for(int i=0;i<nth;i++)
      wstack[i]=left + (i+1)*(right-left)/(nth+1);

    // Get exponents
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Get thread number
#ifdef _OPENMP
      int ith=omp_get_thread_num();
#else
      int ith=0;
#endif
      // Get exponents
      expstack[ith]=optimize_completeness(am,0.0,wstack[ith],nexp,nval,false,&tstack[ith]);
    }
    
    // Figure out how to move ends
    for(int i=0;i<nth;i++) {
      if(tstack[i]>tol && right>wstack[i])
	right=wstack[i];
      else if(tstack[i]<tol && left<wstack[i])
	left=wstack[i];
    }
  } while(right-left>1e-7);

  // Find value closest to tol
  double min=DBL_MAX;
  int mini=-1;
  for(int i=0;i<nth;i++)
    if(fabs(tstack[i]-tol)<min) {
      min=fabs(tstack[i]-tol);
      mini=i;
    }
  
  // Set width
  *width=wstack[mini];
  
  return expstack[mini];
}


/// Perform completeness-optimization of exponents
arma::vec get_exponents(int am, double start, double end, double tol, int nval, bool verbose) {
  // Exponents
  arma::vec exps;
  bool succ=false;

  // Work array
  std::vector<arma::vec> expwrk;
  std::vector<double> mog;

  // Sanity check
  if(tol<MINTAU) {
    if(verbose)
      printf("Renormalized CO tolerance to %e.\n",MINTAU);
    tol=MINTAU;
  }

  // Allocate work memory
#ifdef _OPENMP
  int nth=omp_get_max_threads();
#else
  int nth=1;
#endif

  expwrk.resize(nth);
  mog.resize(nth);

  // Do completeness optimization
  int nf=1;
  if(verbose)
    printf("\tNf  tau_%i\n",nval);

  while(nf<=NFMAX) {

#ifdef _OPENMP
#pragma omp parallel
#endif
    {

#ifdef _OPENMP
      int ith=omp_get_thread_num();
#else
      int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for ordered
#endif
      for(int mf=nf;mf<nf+nth;mf++) {
	// Get exponents.
	mog[ith]=-1.0;
	expwrk[ith]=optimize_completeness(am,start,end,mf,nval,false,&(mog[ith]));
#ifdef _OPENMP
#pragma omp ordered
#endif
	if(verbose) {
	  if(mog[ith]<(1+sqrt(DBL_EPSILON))*tol)
	    printf("\t%2i *%e\n",mf,mog[ith]);
	  else
	    printf("\t%2i  %e\n",mf,mog[ith]);
	}
      }
    }

    // Did we achieve the wanted mog?
    for(int i=0;i<nth;i++) {
      if(mog[i]<(1+sqrt(DBL_EPSILON))*tol) {
	// Tolerance achieved. Save exponents.
	exps=expwrk[i];
	succ=true;
	break;
      }
    }

    // Need another break clause here.
    if(succ)
      break;

    // Increase nf
    nf+=nth;
  }

  if(!succ) {
    fprintf(stderr,"Could not get exponents for %c shell with tol=%e.\n",shell_types[am],tol);
    throw std::runtime_error("Unable to achieve wanted tolerance.\n");
  } else if(verbose)
    printf("Wanted tolerance achieved with %i exponents.\n",(int) exps.size());

  return exps;
}
