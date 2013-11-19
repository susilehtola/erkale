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
}

#ifdef _OPENMP
#include <omp.h>
#endif

// Maximum number of functions allowed in completeness optimization
#define NFMAX 50

std::vector<double> get_exponents(const gsl_vector *x) {
  // Get exponents
  size_t N=x->size;
  std::vector<double> z(N);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<N;i++) {
    z[i]=exp(gsl_vector_get(x,i));
  }
  return z;
}

arma::mat self_overlap(const std::vector<double> & z, int am) {
  // Compute self-overlap
  size_t N=z.size();
  arma::mat Suv(N,N);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<=i;j++) {
      Suv(i,j)=pow(4.0*z[i]*z[j]/pow(z[i]+z[j],2),am/2.0+0.75);
      Suv(j,i)=Suv(i,j);
    }

  return Suv;
}

std::vector<double> completeness_profile(const gsl_vector * x, void * params) {
  // Get parameters
  completeness_scan_t *par=(completeness_scan_t *) params;

  // Get exponents
  std::vector<double> z=get_exponents(x);

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
  std::vector<double> Y(N);

  for(size_t i=0;i<N;i++)
    Y[i]=arma::dot(J.row(i),J.row(i));

  /*
    FILE *out=fopen("Y.dat","w");
    for(size_t ia=0;ia<par->scanexp.size();ia++) {
    fprintf(out,"% e %e\n",log10(par->scanexp[ia]),Y[ia]);
    }
    fclose(out);
  */

  return Y;
}

double compl_mog(const gsl_vector * x, void * params) {
  // Get parameters
  completeness_scan_t *p=(completeness_scan_t *) params;

  // Get completeness profile
  std::vector<double> Y=completeness_profile(x,params);

  // Compute MOG.
  double phi=0.0;

  size_t nint=0;

  switch(p->n) {

  case(1):
    for(size_t i=1;i<Y.size()-1;i+=2) {
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
    for(size_t i=1;i<Y.size()-1;i+=2) {
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

std::vector<double> optimize_completeness(int am, double min, double max, int Nf, int n, bool verbose, double *mog) {
  // Time minimization
  Timer tmin;

  // Parameters for the optimization.
  completeness_scan_t pars;
  // Angular momentum
  pars.am=am;
  // Moment to optimize
  pars.n=n;
  // Scanning exponents
  pars.scanexp=get_scanning_exponents(min,max,50*Nf+1);

  // Maximum number of iterations
  size_t maxiter = 10000;

  // GSL stuff
  const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer *s = NULL;
  gsl_multimin_function minfunc;

  size_t iter = 0;
  int status;
  double size;

  /* Starting point: even tempered set */
  gsl_vector *x = gsl_vector_alloc (Nf);
  for(int i=0;i<Nf;i++)
    // Need to convert to natural logarithm
    gsl_vector_set(x,i,log(10.0)*(min + (i+0.5)*(max-min)/Nf));

  /* Set initial step sizes to 0.45 times the spacing */
  gsl_vector *ss = gsl_vector_alloc (Nf);
  //  gsl_vector_set_all (ss, 0.45*log(10.0)*(max-min)/Nf);
  gsl_vector_set_all (ss, 1.0);

  /* Initialize method and iterate */
  minfunc.n = Nf;
  minfunc.f = compl_mog;
  minfunc.params = (void *) &pars;

  s = gsl_multimin_fminimizer_alloc (T, Nf);
  gsl_multimin_fminimizer_set (s, &minfunc, x, ss);

  // Progress timer
  Timer t;

  // Legend
  if(verbose) {
    printf("iter ");
    for(int i=0;i<Nf;i++)
      printf(" e%-3i ",i+1);
    printf("mog\n");
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
	for(int i=0;i<Nf;i++)
	  // Convert to 10-base logarithm
	  printf("% 8.3e ",log10(M_E)*gsl_vector_get(s->x,i));
	printf("%e %e\n",pow(s->fval,1.0/n),size);

	// print_gradient(s->x,(void *) &pars);
      }
    }
  while (status == GSL_CONTINUE && iter < maxiter);

  // Save mog
  if(mog!=NULL)
    *mog=pow(s->fval,1.0/n);

  // The returned exponents
  std::vector<double> ret=get_exponents(s->x);
  // Sort into ascending order
  std::sort(ret.begin(),ret.end());
  // and reverse the order
  std::reverse(ret.begin(),ret.end());

  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);

  if(verbose)
    printf("\nMinimization completed in %s.\n",tmin.elapsed().c_str());

  return ret;
}

double maxwidth(int am, double tol, int nexp, int nval) {
  // Dummy value
  double width=-1.0;
  std::vector<double> exps=maxwidth_exps(am,tol,nexp,&width,nval);
  return width;
}

std::vector<double> maxwidth_exps(int am, double tol, int nexp, double *width, int nval) {
  // Error check
  if(nexp<=0) {
    std::vector<double> exps;
    return exps;
  }

  if(tol<MINTAU) {
    //    printf("Renormalized CO tolerance to 1e-5.\n");
    tol=MINTAU;
  }

  // Sanity check
  if(*width<0.0)
    *width=nexp/2.0;

  // Right value
  double left=0.0;
  double right=*width;
  double rval;
  std::vector<double> rexps=optimize_completeness(am,0.0,right,nexp,nval,false,&rval);
  while(rval<tol) {
    left=right;
    right*=2.0;
    rexps=optimize_completeness(am,0.0,right,nexp,nval,false,&rval);
  }

  // Left value
  if(left==0.0) {
    double lval=rval;
    std::vector<double> lexps;
    while(lval>=tol) {
      left/=2.0;
      lexps=optimize_completeness(am,0.0,left,nexp,nval,false,&lval);
    }
  }

  std::vector<double> mexps;

  double middle;
  do {
    middle=(left+right)/2.0;

    // Get exponents
    double mval;
    mexps=optimize_completeness(am,0.0,middle,nexp,nval,false,&mval);
    
    // Figure out which end to move
    if(mval>tol) {
      right=middle;
    }
    else {
      left=middle;
    }
  } while(right-left>1e-6);

  // Set width
  *width=middle;

  return mexps;
}


/// Perform completeness-optimization of exponents
std::vector<double> get_exponents(int am, double start, double end, double tol, int nval, bool verbose) {
  // Exponents
  std::vector<double> exps;
  bool succ=false;

  // Work array
  std::vector< std::vector<double> > expwrk;
  std::vector<double> mog;

  // Sanity check
  if(tol<MINTAU) {
    printf("Renormalized CO tolerance to 1e-5.\n");
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
