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

Unitary::Unitary(int qv, double thr, bool max, bool ver, bool realv) {
  q=qv;
  eps=thr;
  verbose=ver;
  real=realv;

  /// Maximize or minimize?
  if(max)
    sign=1;
  else
    sign=-1;

  // Defaults
  // use 3rd degree polynomial to fit derivative
  polynomial_degree=4;
  // and in the fourier transform use
  fourier_samples=3; // three points per period
  fourier_periods=5; // five quasio-periods

  // Logfile is closed
  log=NULL;
}

Unitary::~Unitary() {
  if(log!=NULL)
    fclose(log);
}

void Unitary::set_q(int qv) {
  q=qv;
}

void Unitary::set_thr(double epsv) {
  eps=epsv;
}

void Unitary::open_log(const std::string & fname) {
  if(log!=NULL)
    fclose(log);
  
  if(fname.length())
    log=fopen(fname.c_str(),"w");
}

void Unitary::check_unitary(const arma::cx_mat & W) const {
  arma::cx_mat prod=arma::trans(W)*W;
  for(size_t i=0;i<prod.n_cols;i++)
    prod(i,i)-=1.0;
  double norm=rms_cnorm(prod);
  if(norm>=sqrt(DBL_EPSILON))
    throw std::runtime_error("Matrix is not unitary!\n");
}

arma::cx_mat Unitary::get_rotation(double step) const {
  // Imaginary unit
  std::complex<double> imagI(0,1.0);

  // Rotation matrix is
  arma::cx_mat rot=Hvec*arma::diagmat(arma::exp(sign*step*imagI*Hval))*arma::trans(Hvec);
  if(real)
    // Zero out possible imaginary part
    rot=arma::real(rot)*std::complex<double>(1.0,0.0);
  
  return rot;
}

void Unitary::set_poly(int deg) {
  polynomial_degree=deg;
}

void Unitary::set_fourier(int samples, int pers) {
  fourier_periods=pers;
  fourier_samples=samples;
}

void Unitary::initialize(const arma::cx_mat & W0) {
  (void) W0;
}

bool Unitary::converged(const arma::cx_mat & W) {
  /// Dummy default function, just check norm of gradient
  (void) W;
  return false;
}

void Unitary::update_gradient(const arma::cx_mat & W) {
  // Euclidean gradient
  arma::cx_mat Gammak;
  cost_func_der(W,J,Gammak);

  // Riemannian gradient, Abrudan 2009 table 3 step 2
  G=Gammak*arma::trans(W) - W*arma::trans(Gammak);
}

void Unitary::update_search_direction() {
  // Imaginary unit
  std::complex<double> imagI(0,1.0);
  
  // Diagonalize -iH to find eigenvalues purely imaginary
  // eigenvalues iw_i of H; Abrudan 2009 table 3 step 1.
  bool diagok=arma::eig_sym(Hval,Hvec,-imagI*H);
  if(!diagok) {
    ERROR_INFO();
    throw std::runtime_error("Unitary optimization: error diagonalizing H.\n");
  }
}

double Unitary::optimize(arma::cx_mat & W, enum unitmethod met, enum unitacc acc, size_t maxiter) {
  // Operating on complex matrix.
  real=false;
  return optimizer(W,met,acc,maxiter);
}

double Unitary::optimize(arma::mat & W, enum unitmethod met, enum unitacc acc, size_t maxiter) {
  // Operating on real matrix.
  real=true;

  arma::cx_mat Whlp=W*std::complex<double>(1.0,0.0);
  double o=optimizer(Whlp,met,acc,maxiter);
  W=arma::real(Whlp);

  return o;
}

double Unitary::optimizer(arma::cx_mat & W, enum unitmethod met, enum unitacc acc, size_t maxiter) {
  // Old gradient
  arma::cx_mat oldG;
  G.zeros(W.n_cols,W.n_cols);
  // Old search direction
  arma::cx_mat oldH;
  H.zeros(W.n_cols,W.n_cols);

  if(W.n_cols<2) {
    // No optimization is necessary.
    W.eye();
    J=cost_func(W);
    return 0.0;
  }

  // Check matrix
  check_unitary(W);
  // Check derivative
  check_derivative(W);

  // Perform any necessary initialization
  initialize(W);

  // Iteration number
  size_t k=0;
  J=0;

  // Print out the legend
  if(verbose)
    print_legend();

  while(true) {
    // Increase iteration number
    k++;

    Timer t;

    // Store old values
    oldJ=J;
    oldG=G;
    oldH=H;

    // Compute the cost function and the euclidean derivative, Abrudan 2009 table 3 step 2
    update_gradient(W);

    // Print progress
    if(verbose)
      print_progress(k);

    // Compute update coefficient
    double gamma=0.0;
    if(acc==SDSA || k==1) {
      // First step in CG, or steepest descent / steepest ascent
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
    
    // Check for convergence. Don't do the check in the first
    // iteration, because for canonical orbitals it can be a really
    // bad saddle point
    if(k>1 && (bracket(G,G)<eps || converged(W))) {

      if(verbose) {
	print_time(t);
	printf("Converged.\n");
	fflush(stdout);

	// Print classification
	classify(W);
      }

      break;
    } else if(k==maxiter) {
      if(verbose) {
	print_time(t);

	printf(" %s\nNot converged.\n",t.elapsed().c_str());
	fflush(stdout);
      }

      break;
    }

    // Compute new search direction
    update_search_direction();

    // Find maximal eigenvalue
    double wmax=arma::max(arma::abs(Hval));
    if(wmax==0.0) {
      continue;
    }

    // Compute maximal step size.
    // Order of the cost function in the coefficients of W.
    Tmu=2.0*M_PI/(q*wmax);

    // Find optimal step size
    double step;
    if(met==POLY_DF) {
      step=polynomial_step_df(W);
    } else if(met==POLY_F) {
      step=polynomial_step_f(W);
    } else if(met==POLY_FDF) {
      step=polynomial_step_fdf(W);
    } else if(met==FOURIER_DF) {
      step=fourier_step_df(W);
    } else if(met==ARMIJO) {
      step=armijo_step(W);
    } else {
      ERROR_INFO();
      throw std::runtime_error("Method not implemented.\n");
    }

    // Check step size
    if(step<0.0) throw std::runtime_error("Negative step size!\n");
    if(step==DBL_MAX) throw std::runtime_error("Could not find step size!\n");

    if(verbose)
      print_step(met,step);

    // Take step
    if(step!=0.0) {
      W=get_rotation(step)*W;
    }

    if(verbose) {
      print_time(t);
    }
  }

  return J;
}

void Unitary::print_legend() const {
  printf("\t%4s\t%13s\t%13s\t%12s\n","iter","J","delta J","<G,G>");
}

void Unitary::print_progress(size_t k) const {
  if(k==1)
    // No info on delta J
    printf("\t%4i\t% e\t%13s\t%e ",(int) k,J,"",bracket(G,G));
  else
    printf("\t%4i\t% e\t% e\t%e ",(int) k,J,J-oldJ,bracket(G,G));
  fflush(stdout);

  if(log!=NULL) {
    fprintf(log,"%4i % .16e %.16e ",(int) k,J,bracket(G,G));
    fflush(log);
  }
}

void Unitary::print_time(const Timer & t) const {
  printf(" %s\n",t.elapsed().c_str());
  fflush(stdout);

  if(log!=NULL) {
    fprintf(log,"%e\n",t.get());
    fflush(log);
  }
}

void Unitary::print_step(enum unitmethod & met, double step) const {
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
  if(log!=NULL) {
    fprintf(log,"%e %e ",step,Tmu);
    fflush(log);
  }
}

void Unitary::classify(const arma::cx_mat & W) const {
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

void Unitary::check_derivative(const arma::cx_mat & W0) {
  // Compute gradient
  update_gradient(W0);
  // and the search direction
  H=G;
  update_search_direction();

  // Determine step size
  double wmax=arma::max(arma::abs(Hval));
  if(wmax==0.0) {
    // Derivative vanishes - don't do anything
    return;
  }
  
  // Compute maximal step size.
  // Order of the cost function in the coefficients of W.
  Tmu=2.0*M_PI/(q*wmax);

  // Get derivative matrix
  arma::cx_mat der=cost_der(W0);
  // The derivative is
  double dfdmu=step_der(W0,der);

  // Compute trial value.
  double trstep=Tmu*sqrt(DBL_EPSILON);
  arma::cx_mat Wtr=get_rotation(trstep)*W0;
  double Jtr=cost_func(Wtr);
  
  // Estimated change in function is
  double dfest=trstep*dfdmu;
  // Real change in function is
  double dfreal=Jtr-J;

  // Is the difference ok? Check absolute or relative magnitude
  if(fabs(dfest)>sqrt(DBL_EPSILON)*fabs(J) && fabs(dfest-dfreal)>1e-2*fabs(dfest)) {
    fprintf(stderr,"\nDerivative mismatch error!\n");
    fprintf(stderr,"Used step size %e, value of function % e.\n",trstep,J);
    fprintf(stderr,"Estimated change of function % e\n",dfest);
    fprintf(stderr,"Realized  change of function % e\n",dfreal);
    fprintf(stderr,"Difference in changes        % e\n",dfest-dfreal);
    fprintf(stderr,"Relative error in changes    % e\n",(dfest-dfreal)/dfest);
    throw std::runtime_error("Derivative mismatch! Check your cost function and its derivative.\n");
  }
}

double Unitary::polynomial_step_f(const arma::cx_mat & W) {
  // Amount of points to use is
  int npoints=polynomial_degree;

  // Spacing
  double deltaTmu=Tmu/(npoints-1);
  // Step size
  double step=0.0;
  
  while(step==0.0) {
    // Evaluate the cost function at the expansion points
    arma::vec mu(npoints);
    arma::vec f(npoints);
    for(int i=0;i<npoints;i++) {
      // Mu in the point is
      mu(i)=i*deltaTmu;
      
      // Trial matrix is
      arma::cx_mat Wtr=get_rotation(mu(i))*W;
      // and the function is
      f(i)=cost_func(Wtr);
    }
    
    // Fit to polynomial of order p
    arma::vec coeff=fit_polynomial(mu,f);
    
    // Find out zeros of the derivative polynomial
    arma::vec roots=solve_roots(derivative_coefficients(coeff));
    // and return the smallest positive one
    step=smallest_positive(roots);
    
    // If root vanishes, go to minimum value
    if(step==0.0 || step > Tmu) {
      double minval=arma::min(f);
      for(size_t i=0;i<mu.n_elem;i++)
	if(minval==f(i)) {
	  step=mu(i);
	  break;
	}
    }

    if(step==0.0) {
      deltaTmu/=2.0;
      printf("No root found, halving step size to %e.\n",deltaTmu);
    }
  }

  return step;
}

double Unitary::step_der(const arma::cx_mat & W, const arma::cx_mat & der) const {
  return sign*2.0*std::real(arma::trace(der*arma::trans(W)*arma::trans(H)));
}

double Unitary::polynomial_step_df(const arma::cx_mat & W) {
  // Amount of points to use is
  int npoints=polynomial_degree;

  // Step size
  double step=0.0;
  // Spacing
  double deltaTmu=Tmu/(npoints-1);

  while(step==0.0) {
    // Evaluate the first-order derivative of the cost function at the expansion points
    arma::vec mu(npoints);
    arma::vec fp(npoints);
    for(int i=0;i<npoints;i++) {
      // Mu in the point is
      mu(i)=i*deltaTmu;
      
      // Trial matrix is
      arma::cx_mat Wtr=get_rotation(mu(i))*W;
      // Compute derivative matrix
      arma::cx_mat der=cost_der(Wtr);
      // so the derivative wrt the step is
      fp(i)=step_der(Wtr,der);
    }
    
    // Sanity check - is derivative of the right sign?
    if(sign*fp[0]<0.0) {
      printf("Derivative is of the wrong sign!\n");
      arma::trans(mu).print("mu");
      arma::trans(fp).print("J'(mu)");
      throw std::runtime_error("Derivative consistency error.\n");
    }
    
    // Fit derivative to polynomial of order p: J'(mu) = a0 + a1*mu + ... + ap*mu^p
    arma::vec coeff=fit_polynomial(mu,fp);
    
    // Find out zeros of the polynomial
    arma::vec roots=solve_roots(coeff);
    // and return the smallest positive one
    step=smallest_positive(roots);

    if(step==0.0) {
      deltaTmu/=2.0;
      printf("No root found, halving step size to %e.\n",deltaTmu);
    }
  }

  return step;
}

double Unitary::polynomial_step_fdf(const arma::cx_mat & W) {
  // Amount of points to use is
  int npoints=(int) ceil((polynomial_degree+1)/2.0);

  // Step size
  double step=0.0;
  // Spacing
  double deltaTmu=Tmu/(npoints-1);

  while(true) {
    // Evaluate the first-order derivative of the cost function at the expansion points
    arma::vec mu(npoints);
    arma::vec f(npoints);
    arma::vec fp(npoints);
    for(int i=0;i<npoints;i++) {
      // Value of mu is
      mu(i)=i*deltaTmu;
      
      // Trial matrix is
      arma::cx_mat Wtr=get_rotation(mu(i))*W;
      arma::cx_mat der;
      cost_func_der(Wtr,f(i),der);
      
      // Compute the derivative
      fp(i)=step_der(Wtr,der);
    }
    
    // Sanity check - is derivative of the right sign?
    if(sign*fp[0]<0.0) {
      printf("Derivative is of the wrong sign!\n");
      arma::trans(mu).print("mu");
      arma::trans(fp).print("J'(mu)");
      throw std::runtime_error("Derivative consistency error.\n");
    }
    
    // Fit function to polynomial of order p
    //  J(mu)  = a_0 + a_1*mu + ... + a_(p-1)*mu^(p-1)
    // and its derivative to the function
    //  J'(mu) = a_1 + 2*a_2*mu + ... + (p-1)*a_(p-1)*mu^(p-2).
    // Pull out coefficients of derivative
    arma::vec ader=derivative_coefficients(fit_polynomial_fdf(mu,f,fp,polynomial_degree));
    
    // Find out zeros of the polynomial
    arma::vec roots=solve_roots(ader);
    // and get the smallest positive one
    step=smallest_positive(roots);
    
    // If root vanishes, go to minimum value
    if(step==0.0 || step > Tmu) {
      double minval=arma::min(f);
      for(size_t i=0;i<mu.n_elem;i++)
	if(minval==f(i)) {
	  step=mu(i);
	  break;
	}
    }

    // Check step
    if(step==0.0) {
      deltaTmu/=2.0;
      printf("No root found, halving step size to %e.\n",deltaTmu);
    }
  }

  return step;
}

double Unitary::armijo_step(const arma::cx_mat & W) {
  // Start with half of maximum.
  double step=Tmu/2.0;

  // Initial rotation matrix
  arma::cx_mat R=get_rotation(step);

  // Evaluate function at R2
  double J2=cost_func(R*R*W);

  if(sign==-1) {
    // Minimization.

    // First condition: f(W) - f(R^2 W) >= mu*<G,H>
    while(J-J2 >= step*bracket(G,H)) {
      // Increase step size.
      step*=2.0;
      R=get_rotation(step);

      // and re-evaluate J2
      J2=cost_func(R*R*W);
    }

    // Evaluate function at R
    double J1=cost_func(R*W);

    // Second condition: f(W) - f(R W) <= mu/2*<G,H>
    while(J-J1 < step/2.0*bracket(G,H)) {
      // Decrease step size.
      step/=2.0;
      R=get_rotation(step);

      // and re-evaluate J1
      J1=cost_func(R*W);
    }

  } else if(sign==1) {
    // Maximization

    // First condition: f(W) - f(R^2 W) >= mu*<G,H>
    while(J-J2 <= -step*bracket(G,H)) {
      // Increase step size.
      step*=2.0;
      R=get_rotation(step);

      // and re-evaluate J2
      J2=cost_func(R*R*W);
    }

    // Evaluate function at R
    double J1=cost_func(R*W);

    // Second condition: f(W) - f(R W) <= mu/2*<G,H>
    while(J-J1 > -step/2.0*bracket(G,H)) {
      // Decrease step size.
      step/=2.0;
      R=get_rotation(step);

      // and re-evaluate J1
      J1=cost_func(R*W);
    }
  } else
    throw std::runtime_error("Invalid optimization direction!\n");

  return step;
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

double Unitary::fourier_step_df(const arma::cx_mat & W) {
  // Length of DFT interval
  double fourier_interval=fourier_periods*Tmu;
  // and of the transform. We want integer division here!
  int fourier_length=2*((fourier_samples*fourier_periods)/2)+1;
    
  // Step length is
  double deltaTmu=fourier_interval/fourier_length;
  
  // Values of mu, J(mu) and J'(mu)
  arma::vec mu(fourier_length);
  arma::vec f(fourier_length);
  arma::vec fp(fourier_length);
  for(int i=0;i<fourier_length;i++) {
    // Value of mu is
    mu(i)=i*deltaTmu;
    
    // Trial matrix is
    arma::cx_mat Wtr=get_rotation(mu(i))*W;
    arma::cx_mat der;
    cost_func_der(Wtr,f(i),der);
    
    // Compute the derivative
    fp[i]=step_der(Wtr,der);
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
  if(!muval.size() || fabs(windowed(0))<sqrt(DBL_EPSILON)) {
    // Failed. Reduce Tmu and try again
    printf("No root found, falling back to polynomial step.\n");
    return polynomial_step_df(W);
  }
  
  // Figure out where the function goes to the wanted direction
  double findJ;
  if(sign==1)
    findJ=arma::max(f);
  else
    findJ=arma::min(f);
  
  // and the corresponding value of mu is
  double findmu=mu(0);
  for(int i=1;i<fourier_length;i++)
    if(f(i)==findJ) {
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
  return muval[rootind];
}

double bracket(const arma::cx_mat & X, const arma::cx_mat & Y) {
  return 0.5*std::real(arma::trace(arma::trans(X)*Y));
}

arma::cx_mat companion_matrix(const arma::cx_vec & c) {
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
  companion.submat(1,0,N-1,N-2).eye();

  return companion;
}

arma::cx_vec solve_roots_cplx(const arma::vec & a) {
  return solve_roots_cplx(std::complex<double>(1.0,0.0)*a);
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


Brockett::Brockett(size_t N, unsigned long int seed) : Unitary(2, sqrt(DBL_EPSILON), true, true) {
  // Get random complex matrix
  sigma=randn_mat(N,N,seed)+std::complex<double>(0.0,1.0)*randn_mat(N,N,seed+1);
  // Hermitize it
  sigma=sigma+arma::trans(sigma);
  // Get N matrix
  Nmat.zeros(N,N);
  for(size_t i=0;i<N;i++)
    Nmat(i,i)=i+1;

  log=fopen("brockett.dat","w");
}

Brockett::~Brockett() {
  fclose(log);
}

bool Brockett::converged(const arma::cx_mat & W) {
  // Update diagonality and unitarity criteria
  unit=unitarity(W);
  diag=diagonality(W);
  // Dummy return
  return false;
}

double Brockett::cost_func(const arma::cx_mat & W) {
  return std::real(arma::trace(arma::trans(W)*sigma*W*Nmat));
}

arma::cx_mat Brockett::cost_der(const arma::cx_mat & W) {
  return sigma*W*Nmat;
}

void Brockett::cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der) {
  f=cost_func(W);
  der=cost_der(W);
}

void Brockett::print_legend() const {
  printf("%4s %13s %13s %13s %13s\n","iter","J","<G,G>","diag","unit");
}

void Brockett::print_progress(size_t k) const {
  printf("%4i % e % e % e % e",(int) k, J, bracket(G,G), diag, unit);

  fprintf(log,"%4i % e % e % e % e\n",(int) k, J, 10*log10(bracket(G,G)), diag, unit);
  fflush(log);
}

void Brockett::print_step(enum unitmethod & met, double step) const {
  (void) met;
  (void) step;
}

double Brockett::diagonality(const arma::cx_mat & W) const {
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

double Brockett::unitarity(const arma::cx_mat & W) const {
  arma::cx_mat U=W*arma::trans(W);
  arma::cx_mat eye(W);
  eye.eye();

  double norm=pow(arma::norm(U-eye,"fro"),2);
  return 10*log10(norm);
}
