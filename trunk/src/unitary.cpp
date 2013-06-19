#include "unitary.h"
#include "timer.h"
#include "mathf.h"
#include <cfloat>

extern "C" {
#include <gsl/gsl_poly.h>
#include <gsl/gsl_errno.h>
}

#define bracket(X,Y) (0.5*std::real(arma::trace(X*arma::trans(Y))))

#define MAXITER 1000

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

bool Unitary::converged() const {
  /// Dummy default function, just check norm of gradient
  return false;
}

double Unitary::optimize(arma::cx_mat & W, enum unitmethod met) {
  Timer t;

  // Clear derivative stack
  G.clear();

  if(W.n_cols<2) {
    // No optimization is necessary.
    W.eye();
    J=cost_func(W);
    return 0.0;
  }

  // Check matrix
  check_unitary(W);

  // Iteration number
  int k=0;

  J=0;

  while(true) {
    // Increase iteration number
    k++;

    // Store old value
    oldJ=J;

    // Compute the cost function and the euclidean derivative, Abrudan 2009 table 3 step 2
    arma::cx_mat Gammak;
    cost_func_der(W,J,Gammak);

    // Riemannian gradient, Abrudan 2009 table 3 step 2
    G.push_back(Gammak*arma::trans(W) - W*arma::trans(Gammak));
    // Remove old matrices from memory?
    if(G.size()>2)
      G.erase(G.begin());

    // Print progress
    if(verbose)
      print_progress(k);

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
    if(bracket(G[G.size()-1],G[G.size()-1])<eps || converged()) {
      
      if(verbose) {
	fprintf(stderr," %10.3f\n",t.get());
	fflush(stderr);
	
	printf(" %s\nConverged.\n",t.elapsed().c_str());
	fflush(stdout);

	// Print classification
	classify(W);
      }

      break;
    } else if(k==MAXITER) {
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
    if(met==POLY_DF)
      step=polynomial_step_df(W);
    else if(met==POLY_FDF)
      step=polynomial_step_fdf(W);
    else if(met==ARMIJO)
      step=armijo_step(W);
    else throw std::runtime_error("Method not implemented.\n");

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
  printf("\t%4i\t% e\t% e\t%e ",(int) k,J,J-oldJ,bracket(G[G.size()-1],G[G.size()-1]));
  fflush(stdout);
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
    while(J-J2 >= step*bracket(G[G.size()-1],H)) {
      // Increase step size.
      step*=2.0;
      R=get_rotation(step);

      // and re-evaluate J2
      J2=cost_func(R*R*W);
    }

    // Evaluate function at R
    double J1=cost_func(R*W);

    // Second condition: f(W) - f(R W) <= mu/2*<G,H>
    while(J-J1 < step/2.0*bracket(G[G.size()-1],H)) {
      // Decrease step size.
      step/=2.0;
      R=get_rotation(step);

      // and re-evaluate J1
      J1=cost_func(R*W);
    }

  } else if(sign==1) {
    // Maximization

    // First condition: f(W) - f(R^2 W) >= mu*<G,H>
    while(J-J2 <= -step*bracket(G[G.size()-1],H)) {
      // Increase step size.
      step*=2.0;
      R=get_rotation(step);

      // and re-evaluate J2
      J2=cost_func(R*R*W);
    }

    // Evaluate function at R
    double J1=cost_func(R*W);

    // Second condition: f(W) - f(R W) <= mu/2*<G,H>
    while(J-J1 > -step/2.0*bracket(G[G.size()-1],H)) {
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

Boys::Boys(const BasisSet & basis, const arma::mat & C, double thr, bool ver) : Unitary(4,thr,false,ver) {
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

PZSIC::PZSIC(SCF *solverp, dft_t dftp, DFTGrid * gridp, bool verb) : Unitary(4,0.0,true,verb) {
  solver=solverp;
  dft=dftp;
  grid=gridp;
}

PZSIC::~PZSIC() {
}

void PZSIC::set(const rscf_t & solp, double occ, double pz) {
  sol=solp;
  occnum=occ;
  pzcor=pz;
}

double PZSIC::cost_func(const arma::cx_mat & W) {
  // Evaluate SIC energy.

  arma::cx_mat der;
  double ESIC;
  cost_func_der(W,ESIC,der);
  return ESIC;
}

arma::cx_mat PZSIC::cost_der(const arma::cx_mat & W) {

  arma::cx_mat der;
  double ESIC;
  cost_func_der(W,ESIC,der);
  return der;
}

void PZSIC::cost_func_der(const arma::cx_mat & W, double & f, arma::cx_mat & der) {
  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != sol.C.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix does not match size of problem!\n");
  }

  // Get transformed orbitals
  arma::cx_mat Ctilde=sol.C*W;

  // Compute orbital-dependent Fock matrices
  //  solver->PZSIC_Fock(Forb,Eorb,Ctilde,occnum,dft,*grid);

  // and the total SIC contribution
  HSIC.zeros(Ctilde.n_rows,Ctilde.n_rows);
  for(size_t io=0;io<Ctilde.n_cols;io++) {
    arma::mat Pio=arma::real(Ctilde.col(io)*arma::trans(Ctilde.col(io)));
    
    HSIC+=Forb[io]*Pio*(solver->get_S());
  }
  
  // SI energy is
  f=arma::sum(Eorb);

  // Derivative is
  der.zeros(Ctilde.n_cols,Ctilde.n_cols);
  for(size_t io=0;io<Ctilde.n_cols;io++)
    for(size_t jo=0;jo<Ctilde.n_cols;jo++)
      der(io,jo)=arma::as_scalar(arma::trans(sol.C.col(io))*Forb[jo]*Ctilde.col(jo));

  // Kappa is
  kappa.zeros(Ctilde.n_cols,Ctilde.n_cols);
  for(size_t io=0;io<Ctilde.n_cols;io++)
    for(size_t jo=0;jo<Ctilde.n_cols;jo++)
      kappa(io,jo)=arma::as_scalar(arma::trans(Ctilde.col(io))*(Forb[jo]-Forb[io])*Ctilde.col(jo));
}

void PZSIC::print_progress(size_t k) const {
  double R, K;
  get_rk(R,K);
  
  fprintf(stderr,"\t%4i\t%e\t% e",(int) k,K/R,J);
  printf("\t%4i\t%e\t% e",(int) k,K/R,J);
  if(k>1) {
    fprintf(stderr,"\t% e", J-oldJ);
    printf("\t% e", J-oldJ);
  } else {
    fprintf(stderr,"\t%13s","");
    printf("\t%13s","");
  }
  
  fflush(stdout);
  fflush(stderr);
}

void PZSIC::get_rk(double & R, double & K) const {
  // Occupation numbers
  std::vector<double> occs;
  occs.assign(sol.C.n_cols,occnum);

  // Compute SIC density
  rscf_t sic(sol);
  sic.H-=pzcor*HSIC;
  diagonalize(solver->get_S(),solver->get_Sinvh(),sic);

  // Difference from self-consistency is
  R=rms_norm(sic.P-sol.P);
  // Difference from Pedersen condition is
  K=rms_cnorm(kappa);
}

bool PZSIC::converged() const {
  double R, K;
  get_rk(R,K);
  
  if(K<0.1*R)
    // Converged
    return true;
  else
    // Not converged
    return false;
}
