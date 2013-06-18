#include "unitary.h"
#include "timer.h"
#include "mathf.h"
#include <cfloat>

extern "C" {
#include <gsl/gsl_poly.h>
#include <gsl/gsl_errno.h>
}

#define bracket(X,Y) (0.5*std::real(arma::trace(X*arma::trans(Y))))

Unitary::Unitary(int qv, bool max, bool ver) {
  q=qv;
  verbose=ver;

  /// Maximize or minimize?
  if(max)
    sign=1;
  else
    sign=-1;
}

Unitary::~Unitary() {
}

void Unitary::check_unitary(const arma::cx_mat & W) const {
  arma::cx_mat prod=arma::trans(W)*W;
  for(size_t i=0;i<prod.n_cols;i++)
    prod(i,i)-=1.0;
  double norm=rms_cnorm(prod);
  if(norm>=1e-10)
    throw std::runtime_error("Matrix is not unitary!\n");
}

double Unitary::optimize_poly(arma::cx_mat & W, int n, double eps, bool fdf) {
  Timer t;

  if(W.n_cols<2) {
    // No optimization is necessary.
    W.eye();
    return 0.0;
  }

  int k=0;
  double J=0;

  // G matrices
  std::vector<arma::cx_mat> G;
  // H matrix
  arma::cx_mat H;
  // Imaginary unit
  std::complex<double> imagI(0,1.0);

  while(true) {
    // Increase iteration number
    k++;

    // Compute the cost function and the euclidean derivative, Abrudan 2009 table 3 step 2
    arma::cx_mat Gammak;
    cost_func_der(W,J,Gammak);

    // Riemannian gradient, Abrudan 2009 table 3 step 2
    G.push_back(Gammak*arma::trans(W) - W*arma::trans(Gammak));
    // Remove old matrices from memory?
    if(G.size()>2)
      G.erase(G.begin());

    if(verbose) {
      printf("\t%4i\t%e\t%e\n",(int) k,J,bracket(G[G.size()-1],G[G.size()-1]));
      fflush(stdout);
    }

    // H matrix
    if(k==1) {
      // First iteration; initialize with gradient
      H=G[G.size()-1];
    } else {
      // Compute Polak-Ribière coefficient
      double gamma=bracket(G[G.size()-1] - G[G.size()-2], G[G.size()-1]) / bracket(G[G.size()-2],G[G.size()-2]);
      // Fletcher-Reeves
      //double gamma=bracket(G[G.size()-1], G[G.size()-1]) / bracket(G[G.size()-2],G[G.size()-2]);

      gamma=std::max(gamma,0.0);

      // Update H
      H=G[G.size()-1]+gamma*H;
    }

    // Check for convergence.
    if(bracket(G[G.size()-1],G[G.size()-1])<eps)
      break;

    arma::vec Hval;
    arma::cx_mat Hvec;
    double wmax;
    double Tmu;

    // Diagonalize iH to find eigenvalues purely imaginary
    // eigenvalues iw_i of -H; Abrudan 2009 table 3 step 1.
    bool diagok=arma::eig_sym(Hval,Hvec,imagI*H);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Unitary optimization: error diagonalizing H.\n");
    }

    // Find maximal eigenvalue
    wmax=0.0;
    for(size_t i=0;i<Hval.n_elem;i++)
      if(fabs(Hval(i))>wmax)
	wmax=fabs(Hval(i));
    if(wmax==0.0) {
      continue;
    }

    // Compute maximal step size.
    // Order of the cost function in the coefficients of W.
    Tmu=2.0*M_PI/(q*wmax);

    // Step size
    const double deltaTmu=Tmu/(n-1);
    std::vector<arma::cx_mat> R(n);

    // Trial matrices
    R[0].eye(W.n_cols,W.n_cols);
    R[1]=Hvec*arma::diagmat(arma::exp(sign*deltaTmu*imagI*Hval))*arma::trans(Hvec);
    for(int i=2;i<n;i++)
      R[i]=R[i-1]*R[1];

    // Step size to use
    double step=DBL_MAX;

    if(!fdf) {
      // Evaluate the first-order derivative of the cost function at the expansion points
      std::vector<double> Jprime(n);
      for(int i=0;i<n;i++) {
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
      const int p=n-1;

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
    } else {
      // Evaluate the first-order derivative of the cost function at the expansion points
      std::vector<double> f(n);
      std::vector<double> fp(n);
      for(int i=0;i<n;i++) {
	// Trial matrix is
	arma::cx_mat Wtr=R[i]*W;
	arma::cx_mat der;

	if(i==0)
	  der=G[G.size()-1];
	else
	  cost_func_der(Wtr*W,f[i],der);

	// Compute the derivative
	fp[i]=sign*2.0*std::real(arma::trace(der*arma::trans(Wtr)*arma::trans(H)));
      }

      // Fit derivative to polynomial of order p: J'(mu) = a0 + a1*mu + ... + ap*mu^p
      const int p=2*(n-1);

      // Compute polynomial coefficients. We have 2*n values
      arma::vec jvec(2*n);
      for(int i=0;i<n;i++) {
	jvec(2*i)=f[i];
	jvec(2*i+1)=fp[i];
      }

      // Form mu matrix
      arma::mat mumat(2*n,p+2);
      mumat.zeros();
      for(int i=0;i<n;i++) {
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
      double xmin=DBL_MAX;
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
	    if(roots[i].real()<xmin)
	      xmin=roots[i].real();
	  }

	// Free workspace
	gsl_poly_complex_workspace_free(w);
      }

      // Sanity check
      if(xmin==DBL_MAX)
	xmin=0.0;
      if(xmin<0.0) throw std::runtime_error("Negative step size!\n");

      // Store step size
      step=xmin;
    }

    // Step size is xmin. Update U
    if(step<0.0) throw std::runtime_error("Negative step size!\n");
    if(step!=0.0) {
      arma::cx_mat Ropt=Hvec*arma::diagmat(arma::exp(sign*step*imagI*Hval))*arma::trans(Hvec);
      W=Ropt*W;
    }
  }

  return J;
}


Boys::Boys(const BasisSet & basis, const arma::mat & C, bool ver) : Unitary(4,true,ver) {
  // Get R^2 matrix
  std::vector<arma::mat> momstack=basis.moment(2);
  rsq=momstack[getind(2,0,0)]+momstack[getind(0,2,0)]+momstack[getind(0,0,2)];

  // Get r matrices
  std::vector<arma::mat> rmat=basis.moment(1);

  // Convert matrices to MO basis
  rsq=arma::trans(C)*rsq*C;
  rx=arma::trans(C)*rmat[0]*C;
  ry=arma::trans(C)*rmat[1]*C;
  rz=arma::trans(C)*rmat[2]*C;
}

Boys::~Boys() {
}

double Boys::cost_func(const arma::cx_mat & W) {
  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != rsq.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("Matrix does not match size of problem!\n");
  }

  double B=0;

  // <i|r^2|i> terms
  arma::cx_mat rsm=rsq*W;
  for(size_t io=0;io<W.n_cols;io++)
    B+=std::real(arma::as_scalar(arma::trans(W.col(io))*rsm.col(io)));

  // <i|r|i>^2 terms
  arma::cx_mat rxw=rx*W;
  arma::cx_mat ryw=ry*W;
  arma::cx_mat rzw=rz*W;
    
  for(size_t io=0;io<W.n_cols;io++) {
    double xp=std::real(arma::as_scalar(arma::trans(W.col(io))*rxw.col(io)));
    double yp=std::real(arma::as_scalar(arma::trans(W.col(io))*ryw.col(io)));
    double zp=std::real(arma::as_scalar(arma::trans(W.col(io))*rzw.col(io)));
    B-=xp*xp + yp*yp + zp*zp;
  }
  
  return B;
}

arma::cx_mat Boys::cost_der(const arma::cx_mat & W) {
  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != rsq.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("Matrix does not match size of problem!\n");
  }

  // Returned matrix
  arma::cx_mat Bder(W.n_cols,W.n_cols);

  // r^2 terms
  for(size_t b=0;b<W.n_cols;b++)
    for(size_t a=0;a<W.n_cols;a++)
      Bder(a,b)=arma::as_scalar(rsq.row(a)*W.col(b));
    
  // r terms 
  arma::cx_mat rxw=rx*W;
  arma::cx_mat ryw=ry*W;
  arma::cx_mat rzw=rz*W;
  
  
  for(size_t b=0;b<W.n_cols;b++) {
    std::complex<double> xp=arma::as_scalar(arma::trans(W.col(b))*rxw.col(b));
    std::complex<double> yp=arma::as_scalar(arma::trans(W.col(b))*ryw.col(b));
    std::complex<double> zp=arma::as_scalar(arma::trans(W.col(b))*rzw.col(b));
    
    for(size_t a=0;a<W.n_cols;a++) {
      std::complex<double> dx=rxw(a,b);
      std::complex<double> dy=ryw(a,b);
      std::complex<double> dz=rzw(a,b);
      
      Bder(a,b)-=2.0*(xp*dx + yp*dy + zp*dz);
    }
  }
  
  return Bder;
}

void Boys::cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der) {
  f=cost_func(W);
  der=cost_der(W);
}
