/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Jussi Lehtola
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

std::vector<arma::mat> self_inv_overlap_logder(const arma::mat & Sinv, const arma::mat & D) {
  size_t N=Sinv.n_rows;

  // Returned stack of matrices
  std::vector<arma::mat> Ik(N);
  for(size_t k=0;k<N;k++)
    Ik[k].zeros(N,N);

  arma::mat DS=D*Sinv;

  // Form matrices
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t k=0;k<N;k++)
    for(size_t mu=0;mu<N;mu++)
      for(size_t nu=0;nu<N;nu++)
	Ik[k](mu,nu)=(-Sinv(k,mu)*DS(k,nu)+DS(k,mu)*Sinv(k,nu));

  return Ik;
}

arma::mat overlap_logder(const std::vector<double> & z, const std::vector<double> & zp, const arma::mat & S, int am) {
  // Computes self-overlap derivative matrix
  size_t N=z.size();
  size_t Np=zp.size();

  double f=am/2.0+0.75;

  arma::mat Sd(N,Np);
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<Np;j++)
      Sd(i,j)=f*S(i,j)*(zp[j]-z[i])/(zp[j]+z[i]);
    
  return Sd;
}

std::vector<double> completeness_profile(const gsl_vector * x, void * params) {
  // Get parameters
  completeness_scan_t *par=(completeness_scan_t *) params;

  // Get exponents
  std::vector<double> z=get_exponents(x);

  // Get self-overlap
  arma::mat Suv=self_overlap(z,par->am);
  // and its half inverse matrix
  arma::mat Sinvh=CanonicalOrth(Suv);

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

std::vector< std::vector<double> > completeness_profile_logder_num(const gsl_vector * x, void * params) {
  // Get parameters
  completeness_scan_t *par=(completeness_scan_t *) params;
  size_t Nf=x->size;

  // Helper vectors
  gsl_vector *left=gsl_vector_alloc(Nf);
  gsl_vector *right=gsl_vector_alloc(Nf);
  
  // Output
  std::vector< std::vector<double> > ret(par->scanexp.size());
  for(size_t i=0;i<par->scanexp.size();i++)
    ret[i].resize(Nf);
  
  // Compute derivatives
  for(size_t ig=0;ig<Nf;ig++) {
    // Form test values
    gsl_vector_memcpy(left,x);
    gsl_vector_memcpy(right,x);
    
    // Step size
    double step=sqrt(DBL_EPSILON);
    gsl_vector_set(left,ig,gsl_vector_get(x,ig)-step);
    gsl_vector_set(right,ig,gsl_vector_get(x,ig)+step);
    
    // Compute completeness profiles
    std::vector<double> Yl=completeness_profile(left,params);
    std::vector<double> Yr=completeness_profile(right,params);

    // Save derivative
    for(size_t ia=0;ia<par->scanexp.size();ia++)
      ret[ia][ig]=(Yr[ia]-Yl[ia])/(2*step);
  }
  
  gsl_vector_free(left);
  gsl_vector_free(right);

  /*
  FILE *out=fopen("Ynumder.dat","w");
  for(size_t ia=0;ia<par->scanexp.size();ia++) {
    fprintf(out,"% e ",log10(par->scanexp[ia]));

    // Print derivatives
    for(size_t fi=0;fi<Nf;fi++)
      fprintf(out,"% e ",ret[ia][fi]);
    fprintf(out,"\n");
  }
  fclose(out);
  */

  return ret;
}

std::vector< std::vector<double> > completeness_profile_logder(const gsl_vector * x, void * params) {
  // Get parameters
  completeness_scan_t *par=(completeness_scan_t *) params;

  // Get exponents
  std::vector<double> z=get_exponents(x);

  // Get self-overlap
  arma::mat S=self_overlap(z,par->am);
  // and its inverse matrix
  arma::mat Sinvh=CanonicalOrth(S);
  arma::mat Sinv=Sinvh*Sinvh;

  // Get overlap of primitives with scanning terms
  arma::mat Ss=overlap(z,par->scanexp,par->am);
  // Compute derivative of scanning overlap
  arma::mat Ds=overlap_logder(z,par->scanexp,Ss,par->am);

  // Compute M matrix
  arma::mat M=Sinv*Ss;

  // Compute derivative of overlap matrix
  arma::mat D=overlap_logder(z,z,S,par->am);

  // Compute stack of derivatives of inverse overlap
  std::vector<arma::mat> Sk=self_inv_overlap_logder(Sinv,D);

  // Output
  std::vector< std::vector<double> > ret(par->scanexp.size());
  for(size_t i=0;i<par->scanexp.size();i++)
    ret[i].resize(z.size());

  // Loop over k
  for(size_t ik=0;ik<z.size();ik++) {
    arma::mat SSS=arma::trans(Ss)*Sk[ik]*Ss;

    // Loop over scanning exponents
    for(size_t ia=0;ia<par->scanexp.size();ia++)
      ret[ia][ik]=2*Ds(ik,ia)*M(ik,ia)+SSS(ia,ia);
  }

  /*
  FILE *out=fopen("Yder.dat","w");
  for(size_t ia=0;ia<par->scanexp.size();ia++) {
    fprintf(out,"% e ",log10(par->scanexp[ia]));

    // Print derivatives
    for(size_t fi=0;fi<z.size();fi++)
      fprintf(out,"% e ",ret[ia][fi]);
    fprintf(out,"\n");
  }
  fclose(out);
  */

  return ret;
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

void compl_mog_num_df(const gsl_vector * x, void * params, gsl_vector *gv) {
  size_t Nf=x->size;

  // Helper vectors
  gsl_vector *left=gsl_vector_alloc(Nf);
  gsl_vector *right=gsl_vector_alloc(Nf);

  // Loop over components
  for(size_t ig=0;ig<Nf;ig++) {
    // Form test values
    gsl_vector_memcpy(left,x);
    gsl_vector_memcpy(right,x);
    
    // Step size
    double step=sqrt(DBL_EPSILON);
    gsl_vector_set(left,ig,gsl_vector_get(x,ig)-step);
    gsl_vector_set(right,ig,gsl_vector_get(x,ig)+step);

    // Compute lhs and rhs values
    double lval=compl_mog(left,params);
    double rval=compl_mog(right,params);

    gsl_vector_set(gv,ig,(rval-lval)/(2.0*step));
  }

  gsl_vector_free(left);
  gsl_vector_free(right);
}  

void compl_mog_df(const gsl_vector * x, void * params, gsl_vector *gv) {
  // Get parameters
  completeness_scan_t *p=(completeness_scan_t *) params;
  size_t Nf=x->size;

  // Get completeness profile
  std::vector<double> Y=completeness_profile(x,params);
  // and its derivative
  std::vector< std::vector<double> > Yd=completeness_profile_logder(x,params);

  /*
  // Do numerical derivative as well
  std::vector< std::vector<double> > Yd_num=completeness_profile_logder_num(x,params);
  */  

  // Values of the gradients
  std::vector<double> g(Nf,0.0);

  size_t nint=0;

  // Compute gradients
  switch(p->n) {

  case(1):
    // Loop over points
    for(size_t i=1;i<Y.size()-1;i+=2) {
      // Loop over functions'
      for(size_t fi=0;fi<Nf;fi++) {
	// Compute derivatives in the points
	double ld=Yd[i-1][fi];
	double md=Yd[i  ][fi];
	double rd=Yd[i+1][fi];
	// Increment total derivative
	g[fi]+=ld+4.0*md+rd;
      }
      nint++;
    }
    break;

  case(2):
    // Loop over points
    for(size_t i=1;i<Y.size()-1;i+=2) {
      // Loop over functions'
      for(size_t fi=0;fi<Nf;fi++) {
	// Compute derivatives in the points
	double ld=2.0*(1.0-Y[i-1])*Yd[i-1][fi];
	double md=2.0*(1.0-Y[i  ])*Yd[i  ][fi];
	double rd=2.0*(1.0-Y[i+1])*Yd[i+1][fi];
	// Increment total derivative
	g[fi]+=ld+4.0*md+rd;
      }
      nint++;
    }
    break;
    
  default:
    ERROR_INFO();
    throw std::runtime_error("Value of n not supported!\n");
  }

  // Plug in normalization factors and the minus sign
  for(size_t fi=0;fi<Nf;fi++)
    gsl_vector_set(gv,fi,-g[fi]/(6.0*nint));

  /*
  printf("Gradient vector is ");
  for(size_t fi=0;fi<Nf;fi++)
    printf("% e ",gsl_vector_get(gv,fi));
  printf("\n");
  */
}

void compl_mog_fdf(const gsl_vector * x, void * params, double *f, gsl_vector *g) {
  *f=compl_mog(x,params);
  compl_mog_df(x,params,g);
}

void compl_mog_num_fdf(const gsl_vector * x, void * params, double *f, gsl_vector *g) {
  *f=compl_mog(x,params);
  compl_mog_num_df(x,params,g);
}

void print_gradient(const gsl_vector *x, void * pars) {
  size_t Nf=x->size;

  // Compute gradient
  gsl_vector *gv=gsl_vector_alloc(Nf);
  compl_mog_df(x, pars, gv);
  
  printf("Gradient           %e:",gsl_blas_dnrm2(gv));
  for(size_t i=0;i<Nf;i++)
    printf(" % e",gsl_vector_get(gv,i));
  printf("\n");
  
  compl_mog_num_df(x,(void *) &pars, gv);
  printf("Numerical gradient %e:",gsl_blas_dnrm2(gv));
  for(size_t i=0;i<Nf;i++)
    printf(" % e",gsl_vector_get(gv,i));
  printf("\n");
  
  gsl_vector_free(gv);
}

std::vector<double> optimize_completeness(int am, double min, double max, int Nf, int n) {
  // Time minimization
  Timer tmin;

  // Parameters for the optimization.
  completeness_scan_t pars;
  // Angular momentum
  pars.am=am;
  // Moment to optimize
  pars.n=n;
  // Scanning exponents
  pars.scanexp=get_scanning_exponents(min,max,50*Nf);

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
  gsl_vector_set_all (ss, 0.45*log(10.0)*(max-min)/Nf);
  
  /* Initialize method and iterate */
  minfunc.n = Nf;
  minfunc.f = compl_mog;
  minfunc.params = (void *) &pars;
  
  s = gsl_multimin_fminimizer_alloc (T, Nf);
  gsl_multimin_fminimizer_set (s, &minfunc, x, ss);

  // Progress timer
  Timer t;

  // Legend
  printf("iter ");
  for(int i=0;i<Nf;i++)
    printf(" e%-3i ",i+1);
  printf("mog\n");
  
  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);
      
      if (status)
	break;
      
      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-2);
      
      if (status == GSL_SUCCESS)
	{
	  printf ("converged to minimum at\n");
	}

      if(iter==1 || status == GSL_SUCCESS || t.get()>=1.0) {
	t.set();
	printf("%4u ",(unsigned int) iter);
	for(int i=0;i<Nf;i++)
	  // Convert to 10-base logarithm
	  printf("% 2.2f ",log10(M_E)*gsl_vector_get(s->x,i));
	printf("%e\n",pow(s->fval,1.0/n));

	// print_gradient(s->x,(void *) &pars);
      }
    }
  while (status == GSL_CONTINUE && iter < maxiter);

  // The returned exponents
  std::vector<double> ret=get_exponents(s->x);
  
  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);

  printf("\nMinimization completed in %s.\n",tmin.elapsed().c_str());

  return ret;
}

std::vector<double> optimize_completeness_df(int am, double min, double max, int Nf, int n) {
  // Time minimization
  Timer tmin;


  // Parameters for the optimization.
  completeness_scan_t pars;
  // Angular momentum
  pars.am=am;
  // Moment to optimize
  pars.n=n;
  // Scanning exponents
  pars.scanexp=get_scanning_exponents(min,max,50*Nf);

  // Maximum number of iterations
  size_t maxiter = 10000;

  // GSL stuff
  size_t iter = 0;
  int status;

  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;

  gsl_multimin_function_fdf minfunc;

  minfunc.n = Nf;
  minfunc.f = &compl_mog;

  // Analytical derivatives
  minfunc.df = &compl_mog_df;
  minfunc.fdf = &compl_mog_fdf;

  // Numerical derivatives
  //  minfunc.df = &compl_mog_num_df;
  // minfunc.fdf = &compl_mog_num_fdf;

  minfunc.params = (void *) &pars;

  /* Starting point: even tempered set */
  gsl_vector *x = gsl_vector_alloc (Nf);
  for(int i=0;i<Nf;i++)
    // Need to convert to natural logarithm
    gsl_vector_set(x,i,log(10.0)*(min + (i+0.5)*(max-min)/Nf));

  // Use conjugate gradient minimizer
  //  T = gsl_multimin_fdfminimizer_conjugate_fr;

  // Use steepest descent
  // T = gsl_multimin_fdfminimizer_steepest_descent;

  // Use BFGS
  T = gsl_multimin_fdfminimizer_vector_bfgs2;

  s = gsl_multimin_fdfminimizer_alloc (T, Nf);

  // Set minimizer
  gsl_multimin_fdfminimizer_set (s, &minfunc, x, 0.01, 1e-4);

  // Progress timer
  Timer t;

  // Legend
  printf("iter ");
  for(int i=0;i<Nf;i++)
    printf(" e%-3i ",i+1);
  printf("mog\n");
  
  do
    {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate (s);

      if (status)
	break;

      status = gsl_multimin_test_gradient (s->gradient, 1e-10);

      if (status == GSL_SUCCESS)
	printf ("Minimum found at:\n");

      if(iter==1 || status == GSL_SUCCESS || t.get()>=1.0) {
	t.set();
	printf("%4u ",(unsigned int) iter);
	for(int i=0;i<Nf;i++)
	  // Convert to 10-base logarithm
	  printf("% 2.2f ",log10(M_E)*gsl_vector_get(s->x,i));
	printf("%e\n",pow(s->f,1.0/n));
      }
    }
  while (status == GSL_CONTINUE && iter < maxiter);

  if(status!=GSL_CONTINUE && status!= GSL_SUCCESS) {
    printf("Error encountered in minimization.\n");
    print_gradient(s->x,(void *) &pars);
  }

  if(iter==maxiter)
    printf("Maximum number of iterations reached.\n");

  // The returned exponents
  std::vector<double> ret=get_exponents(s->x);

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  printf("\nMinimization completed in %s.\n",tmin.elapsed().c_str());

  return ret;
}
