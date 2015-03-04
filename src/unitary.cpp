/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "unitary.h"
#include "timer.h"
#include "mathf.h"
#include <cfloat>

UnitaryFunction::UnitaryFunction(int qv, bool max): W(arma::cx_mat()), f(0.0), q(qv) {
  /// Maximize or minimize?
  sign = max ? 1 : -1;
}

UnitaryFunction::~UnitaryFunction() {
}

void UnitaryFunction::setW(const arma::cx_mat & Wv) {
  W=Wv;
}

arma::cx_mat UnitaryFunction::getW() const {
  return W;
}

int UnitaryFunction::getq() const {
  return q;
}

double UnitaryFunction::getf() const {
  return f;
}

int UnitaryFunction::getsign() const {
  return sign;
}

bool UnitaryFunction::converged() {
  /// Dummy default function
  return false;
}

std::string UnitaryFunction::legend() const {
  /// Dummy default function
  return "";
}

std::string UnitaryFunction::status(bool lfmt) {
  /// Dummy default function
  (void) lfmt;
  return "";
}

UnitaryOptimizer::UnitaryOptimizer(double Gthrv, double Fthrv, bool ver, bool realv) : G(arma::cx_mat()), H(arma::cx_mat()), Hvec(arma::cx_mat()), Hval(arma::vec()), Tmu(0.0), verbose(ver), real(realv), Gthr(Gthrv), Fthr(Fthrv) {

  // Defaults
  // use 3rd degree polynomial to fit derivative
  polynomial_degree=4;
  // and in the fourier transform use
  fourier_periods=5; // five quasi-periods
  fourier_samples=3; // three points per period

  debug=false;

  // Logfile is closed
  log=NULL;
}

UnitaryOptimizer::~UnitaryOptimizer() {
  if(log!=NULL)
    fclose(log);
}

void UnitaryOptimizer::set_debug(bool d) {
  debug=d;
}

void UnitaryOptimizer::set_thr(double Geps, double Feps) {
  Gthr=Geps;
  Fthr=Feps;
}

void UnitaryOptimizer::open_log(const std::string & fname) {
  if(log!=NULL)
    fclose(log);
  
  if(fname.length())
    log=fopen(fname.c_str(),"w");
}

void UnitaryOptimizer::check_unitary(const arma::cx_mat & W) const {
  arma::cx_mat prod=arma::trans(W)*W-arma::eye(W.n_cols,W.n_cols);
  double norm=rms_cnorm(prod);
  
  if(norm>=sqrt(DBL_EPSILON)) {
    std::ostringstream oss;
    oss << "Matrix is not unitary: || W W^H -1 || = " << norm << "!\n";
    throw std::runtime_error(oss.str());
  }
}

arma::cx_mat UnitaryOptimizer::get_rotation(double step) const {
  // Rotation matrix is
  arma::cx_mat rot=Hvec*arma::diagmat(arma::exp(step*COMPLEXI*Hval))*arma::trans(Hvec);
  if(real)
    // Zero out possible imaginary part
    rot=arma::real(rot)*COMPLEX1;
  
  return rot;
}

void UnitaryOptimizer::set_poly(int deg) {
  polynomial_degree=deg;
}

void UnitaryOptimizer::set_fourier(int samples, int pers) {
  fourier_periods=pers;
  fourier_samples=samples;
}

void UnitaryOptimizer::update_gradient(const arma::cx_mat & W, UnitaryFunction *f) {
  // Euclidean gradient
  arma::cx_mat Gammak;
  double J;
  f->cost_func_der(W,J,Gammak);
  // Fix sign
  //  Gammak*=f->getsign();

  // Riemannian gradient, Abrudan 2009 table 3 step 2
  G=Gammak*arma::trans(W) - W*arma::trans(Gammak);
}

void UnitaryOptimizer::update_search_direction(int q) {
  // Diagonalize -iH to find eigenvalues purely imaginary
  // eigenvalues iw_i of H; Abrudan 2009 table 3 step 1.
  bool diagok=arma::eig_sym(Hval,Hvec,-COMPLEXI*H);
  if(!diagok) {
    ERROR_INFO();
    throw std::runtime_error("Unitary optimization: error diagonalizing H.\n");
  }

  // Max step length
  double wmax=arma::max(arma::abs(Hval));
  Tmu=2.0*M_PI/(q*wmax);
}

double UnitaryOptimizer::optimize(UnitaryFunction* & f, enum unitmethod met, enum unitacc acc, size_t maxiter) {
  // Get the matrix
  arma::cx_mat W=f->getW();
  if(real)
    W=arma::real(W)*COMPLEX1;

  // Current and old gradient
  arma::cx_mat oldG;
  G.zeros(W.n_cols,W.n_cols);
  // Current and old search direction
  arma::cx_mat oldH;
  H.zeros(W.n_cols,W.n_cols);
  
  if(W.n_cols<2) {
    // No optimization is necessary.
    W.eye();
    return f->cost_func(W);
  }

  // Check matrix
  check_unitary(W);
  // Check derivative
  check_derivative(f);

  // Info from previous iteration
  UnitaryFunction *oldf=NULL;

  // Iteration number
  size_t k=0;

  // Print out the legend
  if(verbose)
    print_legend(f);

  if(maxiter>0)
    while(true) {
      Timer t;

      // Store old gradient and search direction
      oldG=G;
      oldH=H;

      // Compute the cost function and the euclidean derivative, Abrudan 2009 table 3 step 2
      update_gradient(W,f);

      // Compute update coefficient
      double gamma=0.0;
      if(acc==SDSA || (k-1)%W.n_cols==0) {
	// Reset step in CG, or steepest descent / steepest ascent
	gamma=0.0;
      } else if(acc==CGPR) {
	// Compute Polak-Ribière coefficient
	gamma=bracket(G - oldG, G) / bracket(oldG, oldG);
      } else if(acc==CGFR) {
	// Fletcher-Reeves
	gamma=bracket(G, G) / bracket(oldG, oldG);
      } else if(acc==CGHS) {
	// Hestenes-Stiefel
	gamma=bracket(G - oldG, G) / bracket(G - oldG, oldH);
      } else
	throw std::runtime_error("Unsupported update.\n");
    
      // Perform update
      if(gamma==0.0) {
	// Use gradient direction
	H=G;
      } else {
	// Compute update
	H=G+gamma*H;
	// Make sure H stays skew symmetric
	H=0.5*(H-arma::trans(H));
      
	// Check that update is OK
	if(bracket(G,H)<0.0) {
	  H=G;
	  printf("CG search direction reset.\n");
	}
      }
      // Update search direction
      update_search_direction(f->getq());

      // Save old iteration data
      if(oldf)
	delete oldf;
      oldf=f->copy();

      // Take a step
      if(met==POLY_DF) {
	polynomial_step_df(f);
      } else if(met==POLY_F) {
	polynomial_step_f(f);
      } else if(met==FOURIER_DF) {
	fourier_step_df(f);
      } else if(met==ARMIJO) {
	armijo_step(f);
      } else {
	ERROR_INFO();
	throw std::runtime_error("Method not implemented.\n");
      }
    
      // Increase iteration number
      k++;
      // Update matrix
      W=f->getW();

      // Print progress
      if(verbose) {
	print_progress(k,f,oldf);
	print_time(t);
      }
    
      // Check for convergence. Don't do the check in the first
      // iteration, because for canonical orbitals it can be a really
      // bad saddle point
      double J=f->getf();
    
      double oldJ = (oldf==NULL) ? 0.0 : oldf->getf();
      if( k>1 && (f->converged() || (bracket(G,G)<Gthr && fabs(J-oldJ)<Fthr))) {
	if(verbose) {
	  printf("Converged.\n");
	  fflush(stdout);
	
	  // Print classification
	  classify(W);
	}

	break;
      } else if(k==maxiter) {
	if(verbose) {
	  printf(" %s\nNot converged.\n",t.elapsed().c_str());
	  fflush(stdout);
	}
      
	break;
      }
    
      if(debug) {
	char fname[80];
	sprintf(fname,"unitary_%04i.dat",(int) k);
	FILE *p=fopen(fname,"w");
	UnitaryFunction *uf=f->copy();      
	for(int i=-80;i<=80;i++) {
	  double x=i*0.05;
	  double xT=x*Tmu;

	  double y=uf->cost_func(get_rotation(xT)*W);
	  fprintf(p,"% e % e % e\n",x,xT,y);
	}
	fclose(p);
	delete uf;
      }
    }
  
  if(oldf)
    delete oldf;
  
  return f->getf();
}

void UnitaryOptimizer::print_legend(const UnitaryFunction *f) const {
  printf("  %4s  %13s  %13s  %12s  %s\n","iter","J","delta J","<G,G>",f->legend().c_str());
}

void UnitaryOptimizer::print_progress(size_t k, UnitaryFunction *f, const UnitaryFunction *fold) const {
  double J=f->getf();
  if(fold==NULL)
    // No info on delta J
    printf("  %4i  % e  %13s %e  %s",(int) k,J,"",bracket(G,G),f->status().c_str());
  else {
    double oldJ=fold->getf();
    printf("  %4i  % e  % e  %e  %s",(int) k,J,J-oldJ,bracket(G,G),f->status().c_str());
  }
  fflush(stdout);

  if(log!=NULL) {
    fprintf(log,"%4i % .16e %.16e %s",(int) k,J,bracket(G,G),f->status(true).c_str());
    fflush(log);
  }
}

void UnitaryOptimizer::print_time(const Timer & t) const {
  printf(" %s\n",t.elapsed().c_str());
  fflush(stdout);

  if(log!=NULL) {
    fprintf(log,"%e\n",t.get());
    fflush(log);
  }
}

void UnitaryOptimizer::print_step(enum unitmethod & met, double step) const {
  /*
  if(met==POLY_DF)
    printf("Polynomial_df  step %e (%e of Tmu)\n",step,step/Tmu);
  else if(met==POLY_FDF)
    printf("Polynomial_fdf step %e (%e of Tmu)\n",step,step/Tmu);
  else if(met==FOURIER_DF)
    printf("Fourier_df     step %e (%e of Tmu)\n",step,step/Tmu);
  else if(met==ARMIJO)
    printf("Armijo         step %e (%e of Tmu)\n",step,step/Tmu);
  else {
    ERROR_INFO();
    throw std::runtime_error("Method not implemented.\n");
  }
  */

  (void) met;
  (void) step;
  if(log!=NULL)
    fprintf(log,"%e\n",step);
}

void UnitaryOptimizer::classify(const arma::cx_mat & W) const {
  if(real)
    return;
  
  // Classify matrix
  double realpart=rms_norm(arma::real(W));
  double imagpart=rms_norm(arma::imag(W));

  printf("Transformation matrix is");
  if(imagpart<sqrt(DBL_EPSILON)*realpart)
    printf(" real");
  else if(realpart<sqrt(DBL_EPSILON)*imagpart)
    printf(" imaginary");
  else
    printf(" complex");

  printf(", re norm %e, im norm %e\n",realpart,imagpart);
}

void UnitaryOptimizer::check_derivative(const UnitaryFunction *fp0) {
  UnitaryFunction *fp=fp0->copy();
  
  arma::cx_mat W0=fp0->getW();

  // Compute gradient
  update_gradient(W0,fp);
  // and the search direction
  arma::cx_mat Hs=G;
  update_search_direction(fp->getq());

  // Get cost function and derivative matrix
  arma::cx_mat der;
  double Jo;
  fp->cost_func_der(W0,Jo,der);
  // The derivative is
  double dfdmu=step_der(W0,der);

  // Compute trial value.
  double trstep=Tmu*sqrt(DBL_EPSILON);
  arma::cx_mat Wtr=get_rotation(trstep*fp0->getsign())*W0;
  double Jtr=fp->cost_func(Wtr);
  
  // Estimated change in function is
  double dfest=trstep*dfdmu;
  // Real change in function is
  double dfreal=Jtr-Jo;

  // Is the difference ok? Check absolute or relative magnitude
  if(fabs(dfest)>sqrt(DBL_EPSILON)*std::max(1.0,fabs(fp->getf())) && fabs(dfest-dfreal)>1e-2*fabs(dfest)) {
    fprintf(stderr,"\nDerivative mismatch error!\n");
    fprintf(stderr,"Used step size %e, value of function % e.\n",trstep,Jo);
    fprintf(stderr,"Estimated change of function % e\n",dfest);
    fprintf(stderr,"Realized  change of function % e\n",dfreal);
    fprintf(stderr,"Difference in changes        % e\n",dfest-dfreal);
    fprintf(stderr,"Relative error in changes    % e\n",(dfest-dfreal)/dfest);
    throw std::runtime_error("Derivative mismatch! Check your cost function and its derivative.\n");
  }

  delete fp;
}

void UnitaryOptimizer::polynomial_step_f(UnitaryFunction* & fp) {
  // Amount of points to use is
  int npoints=polynomial_degree;
  // Spacing
  double deltaTmu=Tmu/(npoints-1);
  // Step size
  double step=0.0;
  arma::cx_mat W=fp->getW();
    
  UnitaryFunction* fline[npoints];
  for(int i=0;i<npoints;i++)
    fline[i]=fp->copy();

  while(step==0.0) {
    // Evaluate the cost function at the expansion points
    arma::vec mu(npoints);
    arma::vec f(npoints);

    // 0th point is current point!
    for(int i=0;i<npoints;i++) {
      // Mu in the point is
      mu(i)=i*deltaTmu;
      
      // Trial matrix is
      arma::cx_mat Wtr=get_rotation(mu(i)*fp->getsign())*W;
      // and the function is
      f(i)=fline[i]->cost_func(Wtr);
    }
    
    // Fit to polynomial of order p
    arma::vec coeff=fit_polynomial(mu,f);
    
    // Find out zeros of the derivative polynomial
    arma::vec roots=solve_roots(derivative_coefficients(coeff));
    // get the smallest positive one
    step=smallest_positive(roots);
    if(step==0.0) {
      deltaTmu/=2.0;
      printf("No root found, halving step size to %e.\n",deltaTmu);
      continue;
    }

    // Is the step length in the allowed region?
    if(step>0.0 && step <=Tmu) {
      // Yes. Calculate the new value
      arma::cx_mat Wtr=get_rotation(step*fp->getsign())*W;
      UnitaryFunction *newf=fp->copy();

      double J=fp->getf();
      double Jtr=newf->cost_func(Wtr);

      // Accept the step?
      if( fp->getsign()*(Jtr-J) > 0.0) {
	// Yes.
	delete fp;
	fp=newf;
	// Free memory
	for(int i=0;i<npoints;i++)
	  delete fline[i];

	return;
      } else
	delete newf;
    }
    
    // If we are still here, then just get the minimum value
    if(step==0.0 || step > Tmu) {
      fprintf(stderr,"Line search interpolation failed.\n");
      fflush(stderr);
      double minval=arma::max(fp->getsign()*f);
      for(size_t i=0;i<mu.n_elem;i++)
	if(minval==f(i)) {
	  delete fp;
	  fp=fline[i];
	  break;
	}
    }
  }
  
  for(int i=0;i<npoints;i++)
    delete fline[i];

  return;
}

void UnitaryOptimizer::polynomial_step_df(UnitaryFunction* & fp) {
  // Matrix
  arma::cx_mat W=fp->getW();
  // Amount of points to use is
  int npoints=polynomial_degree;
  // Spacing
  double deltaTmu=Tmu/(npoints-1);
  int halved=0;

  UnitaryFunction* fline[npoints];
  for(int i=0;i<npoints;i++)
    fline[i]=fp->copy();

  while(true) {
    // Evaluate the cost function at the expansion points
    arma::vec mu(npoints);
    arma::vec fd(npoints);
    arma::vec fv(npoints);

    for(int i=0;i<npoints;i++) {
      // Mu in the point is
      mu(i)=i*deltaTmu;
      
      // Trial matrix is
      arma::cx_mat Wtr=get_rotation(mu(i)*fp->getsign())*W;
      // and the function is
      arma::cx_mat der;
      //der=fline[i]->cost_der(Wtr);
      fline[i]->cost_func_der(Wtr,fv(i),der);
      // and the derivative is
      fd(i)=step_der(Wtr,der);
    }

    // Sanity check - is derivative of the right sign?
    if(fd(0)<0.0) {
      printf("Derivative is of the wrong sign!\n");
      arma::trans(mu).print("mu");
      arma::trans(fd).print("J'(mu)");
      //      throw std::runtime_error("Derivative consistency error.\n");
      fprintf(stderr,"Warning - inconsistent sign of derivative.\n");
    }
    
    // Fit to polynomial of order p
    arma::vec coeff=fit_polynomial(mu,fd);
    
    // Find out zeros of the polynomial
    arma::vec roots=solve_roots(coeff);
    // get the smallest positive one
    double step=smallest_positive(roots);

    /*
    printf("Trial step size is %e.\n",step);
    mu.t().print("mu");
    fd.t().print("f'");
    fv.t().print("f");
    */
    
    // Is the step length in the allowed region?
    if(step>0.0 && step <=Tmu) {
      // Yes. Calculate the new value
      arma::cx_mat Wtr=get_rotation(step*fp->getsign())*W;
      UnitaryFunction *newf=fp->copy();

      double J=fp->getf();
      double Jtr=newf->cost_func(Wtr);

      // Accept the step?
      if( fp->getsign()*(Jtr-J) > 0.0) {
	// Yes.
	//	printf("Function value changed by %e, accept.\n",Jtr-J);
	delete fp;
	fp=newf;
	break;
      } else {
	fprintf(stderr,"Line search interpolation failed.\n");
	fflush(stderr);
	if(halved<1) {
	  delete newf;
	  halved++;
	  deltaTmu/=2.0;
	  continue;
	} else {
	  // Try an Armijo step
	  return armijo_step(fp);
	  
	  /*
	  ERROR_INFO();
	  throw std::runtime_error("Problem in polynomial line search - could not find suitable extremum!\n");
	  */
	}
      }

      delete newf;
    } else {
      fprintf(stderr,"Line search interpolation failed.\n");
      fflush(stderr);
      if(halved<4) {
	halved++;
	deltaTmu/=2.0;
	continue;
      } else {
	ERROR_INFO();
	  throw std::runtime_error("Problem in polynomial line search - could not find suitable extremum!\n");
      }
    }
  }

  for(int i=0;i<npoints;i++)
    delete fline[i];
}

double UnitaryOptimizer::step_der(const arma::cx_mat & W, const arma::cx_mat & der) const {
  return 2.0*std::real(arma::trace(der*arma::trans(W)*arma::trans(H)));
}

void UnitaryOptimizer::armijo_step(UnitaryFunction* & fp) {
  // Start with half of maximum.
  double step=Tmu/2.0;

  // Initial rotation matrix
  arma::cx_mat R=get_rotation(step*fp->getsign());

  // Helper
  UnitaryFunction *hlp=fp->copy();

  // Original rotation
  arma::cx_mat W(fp->getW());

  // Current value
  double J=fp->getf();

  // Evaluate function at R2
  double J2=hlp->cost_func(R*R*W);

  if(fp->getsign()==-1) {
    // Minimization.

    // First condition: f(W) - f(R^2 W) >= mu*<G,H>
    while(J-J2 >= step*bracket(G,H)) {
      // Increase step size.
      step*=2.0;
      R=get_rotation(step*fp->getsign());

      // and re-evaluate J2
      J2=hlp->cost_func(R*R*W);
    }

    // Evaluate function at R
    double J1=hlp->cost_func(R*W);

    // Second condition: f(W) - f(R W) <= mu/2*<G,H>
    while(J-J1 < step/2.0*bracket(G,H)) {
      // Decrease step size.
      step/=2.0;
      R=get_rotation(step*fp->getsign());

      // and re-evaluate J1
      J1=hlp->cost_func(R*W);
    }

  } else if(fp->getsign()==1) {
    // Maximization

    // First condition: f(W) - f(R^2 W) >= mu*<G,H>
    while(J-J2 <= -step*bracket(G,H)) {
      // Increase step size.
      step*=2.0;
      R=get_rotation(step*fp->getsign());

      // and re-evaluate J2
      J2=hlp->cost_func(R*R*W);
    }

    // Evaluate function at R
    double J1=hlp->cost_func(R*W);

    // Second condition: f(W) - f(R W) <= mu/2*<G,H>
    while(J-J1 > -step/2.0*bracket(G,H)) {
      // Decrease step size.
      step/=2.0;
      R=get_rotation(step*fp->getsign());

      // and re-evaluate J1
      J1=hlp->cost_func(R*W);
    }
  } else
    throw std::runtime_error("Invalid optimization direction!\n");

  // Update solution
  delete fp;
  fp=hlp;
}

arma::cx_vec fourier_shift(const arma::cx_vec & c) {
  // Amount of elements
  size_t N=c.n_elem;

  // Midpoint is at at
  size_t m=N/2;
  if(N%2==1)
    m++;

  // Returned vector
  arma::cx_vec ret(N);
  ret.zeros();

  // Low frequencies
  ret.subvec(0,N-1-m)=c.subvec(m,N-1);
  // High frequencies
  ret.subvec(N-m,N-1)=c.subvec(0,m-1);

  return ret;
}

void UnitaryOptimizer::fourier_step_df(UnitaryFunction* & f) {
  // Length of DFT interval
  double fourier_interval=fourier_periods*Tmu;
  // and of the transform. We want integer division here!
  int fourier_length=2*((fourier_samples*fourier_periods)/2)+1;
    
  // Step length is
  double deltaTmu=fourier_interval/fourier_length;

  // Helpers
  UnitaryFunction * fs[fourier_length];
  for(int i=0;i<fourier_length;i++)
    fs[i]=f->copy();
  arma::cx_mat W=f->getW();
  
  // Values of mu, J(mu) and J'(mu)
  arma::vec mu(fourier_length);
  arma::vec fv(fourier_length);
  arma::vec fp(fourier_length);
  for(int i=0;i<fourier_length;i++) {
    // Value of mu is
    mu(i)=i*deltaTmu;
    
    // Trial matrix is
    arma::cx_mat Wtr=get_rotation(mu(i)*f->getsign())*W;
    arma::cx_mat der;
    fs[i]->cost_func_der(Wtr,fv(i),der);
    
    // Compute the derivative
    fp(i)=step_der(Wtr,der);
  }
  
  // Compute Hann window
  arma::vec hannw(fourier_length);
  for(int i=0;i<fourier_length;i++)
    hannw(i)=0.5*(1-cos((i+1)*2.0*M_PI/(fourier_length+1.0)));
  
  // Windowed derivative is
  arma::vec windowed(fourier_length);
  for(int i=0;i<fourier_length;i++)
    windowed(i)=fp(i)*hannw(i);
    
  // Fourier coefficients
  arma::cx_vec coeffs=arma::fft(windowed)/fourier_length;
  
  // Reorder coefficients
  arma::cx_vec shiftc=fourier_shift(coeffs);
  
  // Find roots of polynomial
  arma::cx_vec croots=solve_roots_cplx(shiftc);
  
    // Figure out roots on the unit circle
  double circletol=1e-2;
  std::vector<double> muval;
  for(size_t i=0;i<croots.n_elem;i++)
    if(fabs(std::abs(croots(i))-1)<circletol) {
      // Root is on the unit circle. Angle is
      double phi=std::imag(std::log(croots(i)));
      
      // Convert to the real length scale
      phi*=fourier_interval/(2*M_PI);
      
      // Avoid aliases
      phi=fmod(phi,fourier_interval);
      // and check for negative values (fmod can return negative values)
      if(phi<0.0)
	phi+=fourier_interval;
      
      // Add to roots
      muval.push_back(phi);
    }
  
  // Sort the roots
  std::sort(muval.begin(),muval.end());
  
  // Sanity check
  if(!muval.size()) {
    // Failed. Use polynomial step instead.
    printf("No root found, falling back to polynomial step.\n");
    polynomial_step_df(f);
    return;
  }

  if( fabs(windowed(0))<sqrt(DBL_EPSILON)) {
    // Failed. Use polynomial step instead.
    printf("Derivative value %e %e too small, falling back to polynomial step.\n",fp(0),windowed(0));
    polynomial_step_df(f);
    return;
  }
  
  // Figure out where the function goes to the wanted direction
  double findJ;
  findJ=arma::max(f->getsign()*fv);
  
  // and the corresponding value of mu is
  double findmu=mu(0);
  for(int i=0;i<fourier_length;i++)
    if(f->getsign()*fv(i)==findJ) {
      findmu=mu(i);
      // Stop at closest extremum
      break;
    }
  
  // Find closest value of mu
  size_t rootind=0;
  double diffmu=fabs(muval[0]-findmu);
  for(size_t i=1;i<muval.size();i++)
    if(fabs(muval[i]-findmu)<diffmu) {
      rootind=i;
      diffmu=fabs(muval[i]-findmu);
    }
  
  // Optimized step size is
  double step=muval[rootind];

  // Is the step length in the allowed region?
  if(step>0.0 && step <=fourier_interval) {
    // Yes. Calculate the new value
    arma::cx_mat Wtr=get_rotation(step*f->getsign())*W;
    UnitaryFunction *newf=f->copy();
    
    double J=f->getf();
    double Jtr=newf->cost_func(Wtr);
    
    // Accept the step?
    if( f->getsign()*(Jtr-J) > 0.0) {
      // Yes.
      delete f;
      f=newf;
      // Free memory
      for(int i=0;i<fourier_length;i++)
	delete fs[i];
      
      return;
    } else
      delete newf;
  }
  
  // If we are still here, then just get the minimum value
  if(step==0.0 || step > Tmu) {
    double minval=arma::max(f->getsign()*fv);
    for(size_t i=0;i<mu.n_elem;i++)
      if(minval==fv(i)) {
	delete f;
	f=fs[i];
	break;
      }
  }
  
  for(int i=0;i<fourier_length;i++)
    delete fs[i];
}

double bracket(const arma::cx_mat & X, const arma::cx_mat & Y) {
  return 0.5*std::real(arma::trace(arma::trans(X)*Y));
}

arma::cx_mat companion_matrix(const arma::cx_vec & c) {
  if(c.size()<=1) {
    // Dummy return
    arma::cx_mat dum;
    return dum;
  }

  // Form companion matrix
  size_t N=c.size()-1;
  if(c(N)==0.0) {
    ERROR_INFO();
    throw std::runtime_error("Coefficient of highest term vanishes!\n");
  }

  arma::cx_mat companion(N,N);
  companion.zeros();

  // First row - coefficients normalized to that of highest term.
  for(size_t j=0;j<N;j++)
    companion(0,j)=-c(N-(j+1))/c(N);
  // Fill out the unit matrix part
  for(size_t j=1;j<N;j++)
    companion(j,j-1)=1.0;
  
  return companion;
}

arma::cx_vec solve_roots_cplx(const arma::vec & a) {
  return solve_roots_cplx(a*COMPLEX1);
}

arma::cx_vec solve_roots_cplx(const arma::cx_vec & a) {
  // Find roots of a_0 + a_1*mu + ... + a_(p-1)*mu^(p-1) = 0.

  // Coefficient of highest order term must be nonzero.
  size_t r=a.size();
  while(a(r-1)==0.0 && r>=1)
    r--;

  if(r==1) {
    // Zeroth degree - no zeros!
    arma::cx_vec dummy;
    dummy.zeros(0);
    return dummy;
  }

  // Form companion matrix
  arma::cx_mat comp=companion_matrix(a.subvec(0,r-1));

  // and diagonalize it
  arma::cx_vec eigval;
  arma::cx_mat eigvec;
  arma::eig_gen(eigval, eigvec, comp);

  // Return the sorted roots
  return arma::sort(eigval);
}

arma::vec solve_roots(const arma::vec & a) {
  // Solve the roots
  arma::cx_vec croots=solve_roots_cplx(a);

  // Collect real roots
  size_t nreal=0;
  for(size_t i=0;i<croots.n_elem;i++)
    if(fabs(std::imag(croots[i]))<10*DBL_EPSILON)
      nreal++;

  // Real roots
  arma::vec roots(nreal);
  size_t ir=0;
  for(size_t i=0;i<croots.n_elem;i++)
    if(fabs(std::imag(croots(i)))<10*DBL_EPSILON)
      roots(ir++)=std::real(croots(i));

  // Sort roots
  roots=arma::sort(roots);

  return roots;
}

double smallest_positive(const arma::vec & a) {
  double step=0.0;
  for(size_t i=0;i<a.size();i++) {
    // Omit extremely small steps because they might get you stuck.
    if(a(i)>sqrt(DBL_EPSILON)) {
      step=a(i);
      break;
    }
  }

  return step;
}

arma::vec derivative_coefficients(const arma::vec & c) {
  // Coefficients for polynomial expansion of y'
  arma::vec cder(c.n_elem-1);
  for(size_t i=1;i<c.n_elem;i++)
    cder(i-1)=i*c(i);

  return cder;
}

arma::vec fit_polynomial(const arma::vec & x, const arma::vec & y, int deg) {
  // Fit function to polynomial of order p: y(x) = a_0 + a_1*x + ... + a_(p-1)*x^(p-1)

  if(x.n_elem!=y.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("x and y have different dimensions!\n");
  }
  size_t N=x.n_elem;

  // Check degree
  if(deg<0)
    deg=(int) x.size();
  if(deg>(int) N) {
    ERROR_INFO();
    throw std::runtime_error("Underdetermined polynomial!\n");
  }

  // Form mu matrix
  arma::mat mumat(N,deg);
  mumat.zeros();
  for(size_t i=0;i<N;i++)
    for(int j=0;j<deg;j++)
      mumat(i,j)=pow(x(i),j);

  // Solve for coefficients: mumat * c = y
  arma::vec c;
  bool solveok=arma::solve(c,mumat,y);
  if(!solveok) {
    arma::trans(x).print("x");
    arma::trans(y).print("y");
    mumat.print("Mu");
    throw std::runtime_error("Error solving for coefficients a.\n");
  }

  return c;
}

arma::vec fit_polynomial_fdf(const arma::vec & x, const arma::vec & y, const arma::vec & dy, int deg) {
  // Fit function and its derivative to polynomial of order p:
  // y(x)  = a_0 + a_1*x + ... +       a_(p-1)*x^(p-1)
  // y'(x) =       a_1   + ... + (p-1)*a_(p-1)*x^(p-2)
  // return coefficients of y'

  if(x.n_elem!=y.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("x and y have different dimensions!\n");
  }
  if(y.n_elem!=dy.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("y and dy have different dimensions!\n");
  }

  // Length of vectors is
  size_t N=x.n_elem;
  if(deg<0) {
    // Default degree of polynomial is
    deg=(int) 2*N;
  } else {
    // We need one more degree so that the derivative is of order deg
    deg++;
  }

  if(deg>(int) (2*N)) {
    ERROR_INFO();
    throw std::runtime_error("Underdetermined polynomial!\n");
  }

  // Form mu matrix.
  arma::mat mumat(2*N,deg);
  mumat.zeros();
  // First y(x)
  for(size_t i=0;i<N;i++)
    for(int j=0;j<deg;j++)
      mumat(i,j)=pow(x(i),j);
  // Then y'(x)
  for(size_t i=0;i<N;i++)
    for(int j=1;j<deg;j++)
      mumat(i+N,j)=j*pow(x(i),j-1);

  // Form rhs vector
  arma::vec data(2*N);
  data.subvec(0,N-1)=y;
  data.subvec(N,2*N-1)=dy;

  // Solve for coefficients: mumat * c = data
  arma::vec c;
  bool solveok=arma::solve(c,mumat,data);
  if(!solveok) {
    arma::trans(x).print("x");
    arma::trans(y).print("y");
    arma::trans(dy).print("dy");
    mumat.print("Mu");
    throw std::runtime_error("Error solving for coefficients a.\n");
  }

  return c;
}


Brockett::Brockett(size_t N, unsigned long int seed) : UnitaryFunction(2, true) {
  // Get random complex matrix
  sigma=randn_mat(N,N,seed)+COMPLEXI*randn_mat(N,N,seed+1);
  // Hermitize it
  sigma=sigma+arma::trans(sigma);
  // Get N matrix
  Nmat.zeros(N,N);
  for(size_t i=0;i<N;i++)
    Nmat(i,i)=i+1;
}

Brockett::~Brockett() {
}

Brockett* Brockett::copy() const {
  return new Brockett(*this);
}

double Brockett::cost_func(const arma::cx_mat & Wv) {
  W=Wv;
  f=std::real(arma::trace(arma::trans(W)*sigma*W*Nmat));
  return f;
}

arma::cx_mat Brockett::cost_der(const arma::cx_mat & Wv) {
  W=Wv;
  return sigma*W*Nmat;
}

void Brockett::cost_func_der(const arma::cx_mat & Wv, double & fv, arma::cx_mat & der) {
  fv=cost_func(Wv);
  der=cost_der(Wv);
}

std::string Brockett::legend() const {
  char stat[1024];
  sprintf(stat,"%13s  %13s", "diag", "unit");
  return std::string(stat);
}

std::string Brockett::status(bool lfmt) {
  char stat[1024];
  if(lfmt)
    sprintf(stat,"% .16e  % .16e", diagonality(), unitarity());
  else
    sprintf(stat,"% e  % e", diagonality(), unitarity());
  return std::string(stat);
}

double Brockett::diagonality() const {
  arma::cx_mat WSW=arma::trans(W)*sigma*W;

  double off=0.0;
  double dg=0.0;

  for(size_t i=0;i<WSW.n_cols;i++)
    dg+=std::norm(WSW(i,i));

  for(size_t i=0;i<WSW.n_cols;i++) {
    for(size_t j=0;j<i;j++)
      off+=std::norm(WSW(i,j));
    for(size_t j=i+1;j<WSW.n_cols;j++)
      off+=std::norm(WSW(i,j));
  }

  return 10*log10(off/dg);
}

double Brockett::unitarity() const {
  arma::cx_mat U=W*arma::trans(W);
  arma::cx_mat eye(W);
  eye.eye();

  double norm=pow(arma::norm(U-eye,"fro"),2);
  return 10*log10(norm);
}
