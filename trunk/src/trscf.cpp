#include "trscf.h"
#include "linalg.h"

TRSCF::TRSCF(const arma::mat & ovl, size_t m) {
  S=ovl;
  max=m;

  h=0.2;
}

TRSCF::~TRSCF() {
}

void TRSCF::push(double E, const arma::mat & D, const arma::mat & F) {
  // Check size
  if(Es.size()==max) {
    Es.erase(Es.begin());
    Ds.erase(Ds.begin());
    Fs.erase(Fs.begin());
  }

  Es.push_back(E);
  Ds.push_back(D);
  Fs.push_back(F);

  // Update index of minimal entry
  update_minind();
}

void TRSCF::clear() {
  Es.clear();
  Ds.clear();
  Fs.clear();
}

void TRSCF::update_minind() {
  minind=0;
  size_t minE=Es[0];
  for(size_t j=1;j<Es.size();j++) {
    if(Es[j]<minE) {
      minE=Es[j];
      minind=j;
    }
  }
}

arma::mat TRSCF::get_Dbar(const arma::vec & c) const {
  arma::mat Dbar=Ds[minind];
  size_t i=0;
  for(size_t ii=0;ii<Ds.size();ii++)
    if(ii!=minind) {
      Dbar+=c(i)*Ds[i];
      i++;
    }
  return Dbar;
}

arma::mat TRSCF::get_M() const {
  size_t N=Ds.size()-1;
  arma::mat M(N,N);

  size_t m=0, n=0;
  for(size_t mi=0;mi<Ds.size();mi++) {
    if(mi==minind)
      continue;

    n=0;
    for(size_t ni=0;ni<Ds.size();ni++) {
      if(ni==minind)
	continue;

      // JCP 121 eqn (34)
      M(m,n)=arma::trace(Ds[mi]*S*Ds[ni]*S);
      n++;
    }

    m++;
  }

  return M;
}
    

arma::mat TRSCF::get_Fbar(const arma::vec & c) const {
  arma::mat Fbar=Fs[minind];
  size_t i=0;
  for(size_t ii=0;ii<Fs.size();ii++)
    if(ii!=minind) {
      Fbar+=c(i)*Fs[i];
      i++;
    }
  return Fbar;
}

std::vector<arma::mat> TRSCF::get_Dn0() const {
  std::vector<arma::mat> ret(Ds.size()-1);
  size_t i=0;
  for(size_t ii=0;ii<Ds.size();ii++)
    if(ii!=minind) {
      ret[i]=Ds[i]-Ds[minind];
      i++;
    }
  return ret;
}

std::vector<arma::mat> TRSCF::get_Fn0() const {
  std::vector<arma::mat> ret(Fs.size()-1);
  size_t i=0;
  for(size_t ii=0;ii<Fs.size();ii++)
    if(ii!=minind) {
      ret[i]=Fs[i]-Fs[minind];
      i++;
    }
       
  return ret;
}

double TRSCF::E_DSM(const arma::vec & c) const {
  // Averaged matrices
  arma::mat Dbar=get_Dbar(c);
  arma::mat Fbar=get_Fbar(c);

  // Minimal matrices
  arma::mat F0=Fs[minind];
  arma::mat D0=Ds[minind];

  double E=Es[minind];
  E+=arma::trace(Dbar*F0)-arma::trace(D0*F0)-arma::trace(D0*Fbar);
  E+=arma::trace(6.0*Dbar*S*Dbar*Fbar)-arma::trace(4.0*Dbar*S*Dbar*S*Dbar*Fbar)-arma::trace(Dbar*Fbar);

    return E;
}

double TRSCF::dE(const arma::vec & g, const arma::mat & H, const arma::vec & c) const {
  return arma::dot(g,c)+0.5*arma::as_scalar(arma::trans(c)*H*c);
}

arma::vec TRSCF::get_g(const arma::vec & c) const {
  // Amount of entries is
  const size_t N=Es.size();

  // F0 and D0 are
  arma::mat F0=Fs[minind];
  arma::mat D0=Ds[minind];

  // Get average matrices and stacks
  arma::mat Fbar=get_Fbar(c);
  arma::mat Dbar=get_Dbar(c);

  // Difference matrices
  std::vector<arma::mat> Fn=get_Fn0();
  std::vector<arma::mat> Dn=get_Dn0();

  // Fill gradient vector
  arma::vec g(N-1);
  size_t n=0;
  for(size_t ni=0;ni<N;ni++) {
    if(ni==minind)
      continue;

    // Fill gradient vector
    g(n)=arma::trace(Dn[n]*F0)-arma::trace(D0*Fn[n])-arma::trace(Dn[n]*Fbar)-arma::trace(Dbar*Fn[n]);
    g(n)+=arma::trace(6.0*Dn[n]*S*Dbar*Fbar)+arma::trace(6.0*Dbar*S*Dn[n]*Fbar) + arma::trace(6.0*Dbar*S*Dbar*Fn[n]);
    g(n)+=-arma::trace(4.0*Dn[n]*S*Dbar*S*Dbar*Fbar)-arma::trace(4.0*Dbar*S*Dn[n]*S*Dbar*Fbar) - arma::trace(4.0*Dbar*S*Dbar*S*Dn[n]*Fbar) - arma::trace(4.0*Dbar*S*Dbar*S*Dbar*Fn[n]);

    n++;
  }
  return g;
}

arma::mat TRSCF::get_H(const arma::vec & c) const {
  // Amount of entries is
  const size_t N=Es.size();

  // F0 and D0 are
  arma::mat F0=Fs[minind];
  arma::mat D0=Ds[minind];

  // Get average matrices and stacks
  arma::mat Fbar=get_Fbar(c);
  arma::mat Dbar=get_Dbar(c);

  // Difference matrices
  std::vector<arma::mat> Fn=get_Fn0();
  std::vector<arma::mat> Dn=get_Dn0();

  // Fill in Hessian
  arma::mat H(N-1,N-1);
  size_t m=0, n=0;
  for(size_t mi=0;mi<N;mi++) {
    if(mi==minind)
      continue;

    n=0;
    for(size_t ni=0;ni<N;ni++) {
      if(ni==minind)
	continue;

      H(m,n)=-arma::trace(Dn[n]*Fn[m])-arma::trace(Dn[m]*Fn[n]);

      H(m,n)+=arma::trace(6.0*Dn[n]*S*Dn[m]*Fbar)+arma::trace(6.0*Dn[m]*S*Dn[n]*Fbar);
      H(m,n)+=arma::trace(6.0*Dn[n]*S*Dbar*Fn[m])+arma::trace(6.0*Dn[m]*S*Dbar*Fn[n]);
      H(m,n)+=arma::trace(6.0*Dbar*S*Dn[n]*Fn[m])+arma::trace(6.0*Dbar*S*Dn[m]*Fn[m]);

      H(m,n)+=
	-arma::trace(4.0*Dn[n]*S*Dn[m]*S*Dbar*Fbar)
	-arma::trace(4.0*Dn[m]*S*Dn[n]*S*Dbar*Fbar);
      H(m,n)+=
	-arma::trace(4.0*Dn[n]*S*Dbar*S*Dn[m]*Fbar)
	-arma::trace(4.0*Dn[m]*S*Dbar*S*Dn[n]*Fbar);
      H(m,n)+=
	-arma::trace(4.0*Dn[n]*S*Dbar*S*Dbar*Fn[m])
	-arma::trace(4.0*Dn[m]*S*Dbar*S*Dbar*Fn[n]);
      H(m,n)+=
	-arma::trace(4.0*Dbar*S*Dn[n]*S*Dn[m]*Fbar)
	-arma::trace(4.0*Dbar*S*Dn[m]*S*Dn[n]*Fbar);
      H(m,n)+=
	-arma::trace(4.0*Dbar*S*Dn[n]*S*Dbar*Fn[m])
	-arma::trace(4.0*Dbar*S*Dn[m]*S*Dbar*Fn[n]);
      H(m,n)+=
	-arma::trace(4.0*Dbar*S*Dbar*S*Dn[n]*Fn[m])
	-arma::trace(4.0*Dbar*S*Dbar*S*Dn[m]*Fn[n]);

      n++;
    }

    m++;
  }

  return H;
}

arma::vec TRSCF::solve_c() const {
  // The starting point is
  arma::vec c0(Es.size());
  c0.zeros();

  // Lagrange multiplier for the step length
  double mu=0.0;
  double oldmu=0.0;

  // The step vector
  arma::vec c;

  // Get the M matrix
  arma::mat M=get_M();

  // Eigenvalues of Hessian
  arma::vec Hval;
  // Eigenvectors of Hessian
  arma::mat Hvec;

  // Length of the step
  double l;

  // Loop over mu
  do {
    // Start at
    c=c0;
    
    // Solve c
    solve_c(c,mu,Hval,Hvec,M);

    // Compute step length
    l=step_len(c,M);
    
    // Increment mu if necessary
    if(l>h) {
      oldmu=mu;
      mu+=1.0;
    }
  } while(l>h);
  
  // Do we need to do a bisection to refine the step length?
  if(oldmu!=mu) {
    // Yes, we do. Left value is
    double lx=oldmu;
    // and right value is
    double rx=mu;
    
    // Correct shift is somewhere in between.
    do {
      // Middle value is
      double mx=(lx+rx)/2.0;

      // Solve c
      c=c0;
      solve_c(c,mx,Hval,Hvec,M);

      // and the length here is
      l=step_len(c,M);

      // Determine which limit to move
      if(l<h) // Move left limit
	lx=mx;
      if(l>h) // Move right limit
	rx=mx;
    } while(rx-lx>1e-3);
  }
  
  return c;
}


void TRSCF::solve_c(arma::vec & c, double mu, arma::vec & Hval, arma::mat & Hvec, const arma::mat & M) const {
  // Solve c self-consistently using fixed value of mu
  double l;

  // Old value
  arma::vec cold(c);
  // Difference from old value
  double d;

  size_t iit=0;

  do {
    // Get the Hessian in the current point
    arma::mat H=get_H(c);
    
    // Get the gradient in the current point
    arma::vec g=get_g(c);
    
    // The level-shifted Hessian is
    arma::mat hess=H-mu*M;
    eig_sym_ordered(Hval,Hvec,hess);
    
    // Solve (H - mu M) c = -g.
    cold=c;
    c.zeros();
    for(size_t i=0;i<Hval.n_elem;i++)
      if(Hval(i)>0.0) // Skip negative eigenvalues
	c+=arma::dot(Hvec.col(i),-g)/Hval(i)*Hvec.col(i);
    
    // Check that we are within the trust region
    l=step_len(c,M);

    
    printf("mu = %e: iteration %i, step length = %e.\n",mu,+iit,l);

    if(l>h)
      // Outside trust region.
      break;

    // Compute difference from old value of c
    d=step_len(c-cold,M);
  } while(d>1e-3);
}

double TRSCF::step_len(const arma::vec & c, const arma::mat & M) const {
  return sqrt(arma::as_scalar(arma::trans(c)*M*(c)));
}

arma::mat TRSCF::solve() const {
  // Solve the coefficients
  arma::vec c=solve_c();
  // and perform the averaging
  return get_Fbar(c);
}
