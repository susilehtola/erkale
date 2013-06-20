#include "unitary.h"
#include "timer.h"
#include "mathf.h"
#include <cfloat>

extern "C" {
#include <gsl/gsl_poly.h>
#include <gsl/gsl_errno.h>
}

Unitary::Unitary(int qv, double thr, bool max, bool ver) {
  q=qv;
  eps=thr;
  verbose=ver;

  /// Maximize or minimize?
  if(max)
    sign=1;
  else
    sign=-1;

  // Default options
  npoly=4; // 4 points for polynomial
}

Unitary::~Unitary() {
}

double Unitary::bracket(const arma::cx_mat & X, const arma::cx_mat & Y) const {
  return 0.5*std::real(arma::trace(arma::trans(X)*Y));
}

void Unitary::check_unitary(const arma::cx_mat & W) const {
  arma::cx_mat prod=arma::trans(W)*W;
  for(size_t i=0;i<prod.n_cols;i++)
    prod(i,i)-=1.0;
  double norm=rms_cnorm(prod);
  if(norm>=1e-10)
    throw std::runtime_error("Matrix is not unitary!\n");
}

arma::cx_mat Unitary::get_rotation(double step) const {
  // Imaginary unit
  std::complex<double> imagI(0,1.0);

  return Hvec*arma::diagmat(arma::exp(sign*step*imagI*Hval))*arma::trans(Hvec);
}

void Unitary::set_poly(int n) {
  npoly=n;
}

bool Unitary::converged(const arma::cx_mat & W) {
  /// Dummy default function, just check norm of gradient
  (void) W;
  return false;
}

double Unitary::optimize(arma::cx_mat & W, enum unitmethod met, enum unitacc acc, size_t maxiter) {
  // Old gradient
  arma::cx_mat oldG;
  G.zeros(W.n_cols,W.n_cols);

  if(W.n_cols<2) {
    // No optimization is necessary.
    W.eye();
    J=cost_func(W);
    return 0.0;
  }

  // Check matrix
  check_unitary(W);

  // Iteration number
  size_t k=0;
  J=0;

  while(true) {
    // Increase iteration number
    k++;
    
    Timer t;
  
    // Store old value
    oldJ=J;

    // Compute the cost function and the euclidean derivative, Abrudan 2009 table 3 step 2
    arma::cx_mat Gammak;
    cost_func_der(W,J,Gammak);

    // Riemannian gradient, Abrudan 2009 table 3 step 2
    oldG=G;
    G=Gammak*arma::trans(W) - W*arma::trans(Gammak);
    
    // Print progress
    if(verbose)
      print_progress(k);

    // H matrix
    if(k==1) {
      // First iteration; initialize with gradient
      H=G;

    } else {

      double gamma=0.0;
      if(acc==SD) {
	// Steepest descent
	gamma=0.0;
      } else if(acc==CGPR) {
	// Compute Polak-Ribière coefficient
	gamma=bracket(G - oldG, G) / bracket(oldG, oldG);
      } else if(acc==CGFR) {
	// Fletcher-Reeves
	gamma=bracket(G, G) / bracket(oldG, oldG);
      } else
	throw std::runtime_error("Unsupported update.\n");
      
      H=G+gamma*H;
      
      // Check that update is OK
      if(bracket(G,H)<0.0) {
	H=G;
	printf("CG search direction reset.\n");
      }
    }
    
    // Check for convergence.
    if(bracket(G,G)<eps || converged(W)) {
      
      if(verbose) {
	fprintf(stderr," %10.3f\n",t.get());
	fflush(stderr);
	
	printf(" %s\nConverged.\n",t.elapsed().c_str());
	fflush(stdout);
	
	// Print classification
	classify(W);
      }
      
      break;
    } else if(k==maxiter) {
      if(verbose) {
	fprintf(stderr," %10.3f\n",t.get());
	fflush(stderr);

	printf(" %s\nNot converged.\n",t.elapsed().c_str());
	fflush(stdout);
      }

      break;
    }

    // Imaginary unit
    std::complex<double> imagI(0,1.0);

    // Diagonalize -iH to find eigenvalues purely imaginary
    // eigenvalues iw_i of H; Abrudan 2009 table 3 step 1.
    bool diagok=arma::eig_sym(Hval,Hvec,-imagI*H);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Unitary optimization: error diagonalizing H.\n");
    }

    // Find maximal eigenvalue
    double wmax=max(abs(Hval));
    if(wmax==0.0) {
      continue;
    }

    // Compute maximal step size.
    // Order of the cost function in the coefficients of W.
    Tmu=2.0*M_PI/(q*wmax);

    // Find optimal step size
    double step;
    if(met==POLY_DF) {
      //      printf("Polynomial df step\n");
      step=polynomial_step_df(W);
    } else if(met==POLY_FDF) {
      //      printf("Polynomial fdf step\n");
      step=polynomial_step_fdf(W);
    } else if(met==ARMIJO) {
      //      printf("Armijo step\n");
      step=armijo_step(W);
    } else throw std::runtime_error("Method not implemented.\n");

    // Check step size
    if(step<0.0) throw std::runtime_error("Negative step size!\n");
    if(step==DBL_MAX) throw std::runtime_error("Could not find step size!\n");

    // Take step
    if(step!=0.0) {
      W=get_rotation(step)*W;
    }

    if(verbose) {
      fprintf(stderr," %10.3f\n",t.get());
      fflush(stderr);

      printf(" %s\n",t.elapsed().c_str());
      fflush(stdout);
    }
  }

  return J;
}

void Unitary::print_progress(size_t k) const {
  printf("\t%4i\t% e\t% e\t%e ",(int) k,J,J-oldJ,bracket(G,G));
  fflush(stdout);
  
  fprintf(stderr,"\t%4i\t% e\t% e\t%e ",(int) k,J,J-oldJ,bracket(G,G));
  fflush(stderr);
}

void Unitary::classify(const arma::cx_mat & W) const {
  // Classify matrix
  double real=rms_norm(arma::real(W));
  double imag=rms_norm(arma::imag(W));
  
  printf("Transformation matrix is");
  if(imag<sqrt(DBL_EPSILON)*real)
    printf(" real");
  else if(real<sqrt(DBL_EPSILON)*imag)
    printf(" imaginary");
  else
    printf(" complex");
  
  printf(", re norm %e, im norm %e\n",real,imag);
}

double Unitary::polynomial_step_df(const arma::cx_mat & W) {
  // Step size
  const double deltaTmu=Tmu/(npoly-1);
  std::vector<arma::cx_mat> R(npoly);

  // Trial matrices
  R[0].eye(W.n_cols,W.n_cols);
  R[1]=get_rotation(deltaTmu);
  for(int i=2;i<npoly;i++)
    R[i]=R[i-1]*R[1];
  
  // Step size to use
  double step=DBL_MAX;

  // Evaluate the first-order derivative of the cost function at the expansion points
  std::vector<double> Jprime(npoly);
  for(int i=0;i<npoly;i++) {
    // Trial matrix is
    arma::cx_mat Wtr=R[i]*W;

    // Compute derivative matrix
    arma::cx_mat der=cost_der(Wtr);
    // so the derivative wrt the step is
    Jprime[i]=sign*2.0*std::real(arma::trace(der*arma::trans(Wtr)*arma::trans(H)));
  }

  // Sanity check - is derivative of the right sign?
  if(sign*Jprime[0]<0.0) {
    ERROR_INFO();
    throw std::runtime_error("Derivative is of the wrong sign!\n");
  }

  // Fit derivative to polynomial of order p: J'(mu) = a0 + a1*mu + ... + ap*mu^p
  const int p=npoly-1;

  // Compute polynomial coefficients.
  arma::vec jvec(p);
  for(int i=0;i<p;i++) {
    jvec(i)=Jprime[i+1]-Jprime[0];
  }

  // Form mu matrix
  arma::mat mumat(p,p);
  mumat.zeros();
  for(int i=0;i<p;i++) {
    // Value of mu on the row is
    double mu=(i+1)*deltaTmu;
    // Fill entries
    for(int j=0;j<p;j++)
      mumat(i,j)=pow(mu,j+1);
  }

  arma::vec aval;
  bool solveok=true;

  // Solve for coefficients - may not be stable numerically
  solveok=arma::solve(aval,mumat,jvec);

  if(!solveok) {
    mumat.print("Mu");
    arma::trans(jvec).print("Jvec");
    throw std::runtime_error("Error solving for coefficients a.\n");
  }

  // Find smallest positive root of a0 + a1*mu + ... + ap*mu^p = 0.
  {
    // Coefficient of highest order term must be nonzero.
    int r=p;
    while(aval(r-1)==0.0)
      r--;

    // Coefficients
    double a[r+2];
    a[0]=Jprime[0];
    for(int i=1;i<=r;i++)
      a[i]=aval(i-1);

    // GSL routine workspace - r:th order polynomial has r+1 coefficients
    gsl_poly_complex_workspace *w=gsl_poly_complex_workspace_alloc(r+1);

    // Return values
    double z[2*r];
    int gslok=gsl_poly_complex_solve(a,r+1,w,z);

    if(gslok!=GSL_SUCCESS) {
      ERROR_INFO();
      fprintf(stderr,"Solution of polynomial root failed, error: \"%s\"\n",gsl_strerror(solveok));
      throw std::runtime_error("Error solving polynomial.\n");
    }

    // Get roots
    std::vector< std::complex<double> > roots(r);
    for(int i=0;i<r;i++) {
      roots[i].real()=z[2*i];
      roots[i].imag()=z[2*i+1];
    }
    // and order them into increasing absolute value
    std::stable_sort(roots.begin(),roots.end(),abscomp<double>);

    int nreal=0;
    for(size_t i=0;i<roots.size();i++)
      if(fabs(roots[i].imag())<10*DBL_EPSILON)
	nreal++;

    /*
      printf("%i real roots:",nreal);
      for(size_t i=0;i<roots.size();i++)
      if(fabs(roots[i].imag())<10*DBL_EPSILON)
      printf(" (% e,% e)",roots[i].real(),roots[i].imag());
      printf("\n");
    */

    for(size_t i=0;i<roots.size();i++)
      if(roots[i].real()>sqrt(DBL_EPSILON) && fabs(roots[i].imag())<10*DBL_EPSILON) {
	// Root is real and positive. Is it smaller than the current minimum?
	if(roots[i].real()<step)
	  step=roots[i].real();
      }

    // Free workspace
    gsl_poly_complex_workspace_free(w);
  }
    
  return step;
}

double Unitary::polynomial_step_fdf(const arma::cx_mat & W) {
  // Step size
  const double deltaTmu=Tmu/(npoly-1);
  std::vector<arma::cx_mat> R(npoly);
  
  // Trial matrices
  R[0].eye(W.n_cols,W.n_cols);
  R[1]=get_rotation(deltaTmu);
  for(int i=2;i<npoly;i++)
    R[i]=R[i-1]*R[1];

  // Evaluate the first-order derivative of the cost function at the expansion points
  std::vector<double> f(npoly);
  std::vector<double> fp(npoly);
  for(int i=0;i<npoly;i++) {
    // Trial matrix is
    arma::cx_mat Wtr=R[i]*W;
    arma::cx_mat der;

    if(i==0) {
      f[i]=J;
      der=G[G.size()-1];
    } else
      cost_func_der(Wtr*W,f[i],der);

    // Compute the derivative
    fp[i]=sign*2.0*std::real(arma::trace(der*arma::trans(Wtr)*arma::trans(H)));
  }

  // Fit derivative to polynomial of order p: J'(mu) = a0 + a1*mu + ... + ap*mu^p
  const int p=2*(npoly-1);

  // Compute polynomial coefficients. We have 2*n values
  arma::vec jvec(2*npoly);
  for(int i=0;i<npoly;i++) {
    jvec(2*i)=f[i];
    jvec(2*i+1)=fp[i];
  }

  // Form mu matrix
  arma::mat mumat(2*npoly,p+2);
  mumat.zeros();
  for(int i=0;i<npoly;i++) {
    // Value of mu in the point is
    double mu=i*deltaTmu;

    // First row: J(mu)
    mumat(2*i,0)=1.0;
    for(int j=1;j<=p+1;j++)
      mumat(2*i,j)=pow(mu,j)/(j);
    // Second row: J'(mu)
    mumat(2*i+1,1)=1.0;
    for(int j=2;j<=p+1;j++)
      mumat(2*i+1,j)=pow(mu,j-1);
  }

  arma::vec aval;
  bool solveok=true;

  // Solve for coefficients - may not be stable numerically
  //solveok=arma::solve(aval,mumat,jvec);

  // Use inverse matrix
  {
    arma::mat invmu;
    solveok=arma::inv(invmu,mumat);

    if(solveok)
      aval=invmu*jvec;
  }

  if(!solveok) {
    mumat.print("Mu");
    arma::trans(jvec).print("Jvec");
    throw std::runtime_error("Error solving for coefficients a.\n");
  }

  // Find smallest positive root of a0 + a1*mu + ... + ap*mu^p = 0.
  double step=DBL_MAX;
  {
    // Coefficient of highest order term must be nonzero.
    int r=p;
    while(aval(r+1)==0.0)
      r--;

    // Coefficients
    double a[r+1];
    for(int i=0;i<r+1;i++)
      a[i]=aval(i+1);

    /*
      printf("Coefficients:");
      for(int i=0;i<r+1;i++)
      printf(" % e",a[i]);
      printf("\n");
    */

    // GSL routine workspace - r:th order polynomial has r+1 coefficients
    gsl_poly_complex_workspace *w=gsl_poly_complex_workspace_alloc(r+1);

    // Return values
    double z[2*r];
    int gslok=gsl_poly_complex_solve(a,r+1,w,z);

    if(gslok!=GSL_SUCCESS) {
      ERROR_INFO();
      fprintf(stderr,"Solution of polynomial root failed, error: \"%s\"\n",gsl_strerror(solveok));
      throw std::runtime_error("Error solving polynomial.\n");
    }

    // Get roots
    std::vector< std::complex<double> > roots(r);
    for(int i=0;i<r;i++) {
      roots[i].real()=z[2*i];
      roots[i].imag()=z[2*i+1];
    }
    // and order them into increasing absolute value
    std::stable_sort(roots.begin(),roots.end(),abscomp<double>);

    /*
      printf("Roots:");
      for(size_t i=0;i<roots.size();i++)
      printf(" (% e, % e)",z[2*i],z[2*i+1]);
      printf("\n");
    */

    int nreal=0;
    for(size_t i=0;i<roots.size();i++)
      if(fabs(roots[i].imag())<10*DBL_EPSILON)
	nreal++;

    /*
      printf("%i real roots:",nreal);
      for(size_t i=0;i<roots.size();i++)
      if(fabs(roots[i].imag())<10*DBL_EPSILON)
      printf(" (% e,% e)",roots[i].real(),roots[i].imag());
      printf("\n");
    */

    for(size_t i=0;i<roots.size();i++)
      if(roots[i].real()>sqrt(DBL_EPSILON) && fabs(roots[i].imag())<10*DBL_EPSILON) {
	// Root is real and positive. Is it smaller than the current minimum?
	if(roots[i].real()<step)
	  step=roots[i].real();
      }

    // Free workspace
    gsl_poly_complex_workspace_free(w);
  }

  // Sanity check
  if(step==DBL_MAX)
    step=0.0;
  
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

Brockett::Brockett(size_t N, unsigned long int seed) : Unitary(2, 1e-10, true, true) {
  // Get random complex matrix
  sigma=randn_mat(N,N,seed)+std::complex<double>(0.0,1.0)*randn_mat(N,N,seed+1);
  // Hermitize it
  sigma=sigma+arma::trans(sigma);
  // Get N matrix
  Nmat.zeros(N,N);
  for(size_t i=1;i<=N;i++)
    Nmat(i,i)=i;

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

void Brockett::print_progress(size_t k) const {
  printf("%4i % e % e % e % e\n",(int) k, J, bracket(G,G), diag, unit);

  fprintf(log,"%4i % e % e % e % e\n",(int) k, J, bracket(G,G), diag, unit);
  fflush(log);
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
  
  double norm=arma::norm(U-eye,"fro");
  return 10*log10(norm);
}

