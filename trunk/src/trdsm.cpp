/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "trdsm.h"
#include "linalg.h"


// Debug
//#define DEBUG

#ifdef DEBUG
bool hessian=false;
#define STEPSIZE 1e-3
#endif

#define ETOL 1e-12

TRDSM::TRDSM(const arma::mat & ovl, size_t m) {
  S=ovl;
  max=m;
}

TRDSM::~TRDSM() {
}

void TRDSM::push(double E, const arma::mat & D, const arma::mat & F) {
  // Add to stack
  Es.push_back(E);
  Ds.push_back(D);
  Fs.push_back(F);

  // Update index of minimal entry
  update_minind();

  // Clear bad density matrices
  clean_density();

  // Check size
  if(Es.size()==max) {
    Es.erase(Es.begin());
    Ds.erase(Ds.begin());
    Fs.erase(Fs.begin());
    // Update once again index of minimal entry
    update_minind();
  }
}

void TRDSM::clear() {
  Es.clear();
  Ds.clear();
  Fs.clear();
}

void TRDSM::update_minind() {
  minind=0;
  size_t minE=Es[0];
  for(size_t j=1;j<Es.size();j++) {
    if(Es[j]<minE) {
      minE=Es[j];
      minind=j;
    }
  }
}

void TRDSM::clean_density() {
  const double mixfac=0.2;
  const arma::mat Dref=Ds[minind];

  std::vector<size_t> delind;

  /// Compare density matrices to current best estimate
  for(size_t ii=0;ii<Ds.size();ii++) {
    if(ii==minind)
      continue;

    // Form trial mixed density matrix
    arma::mat Dt=(1-mixfac)*Dref+mixfac*Ds[ii];

    // Compute purified density matrix
    arma::mat DSD=Dt*S*Dt;
    arma::mat DSDSD=DSD*S*Dt;

    // Deviation from purity is
    arma::mat Dd=3.0*DSD-2*DSDSD-Dt;

    // Compute norm of deviation from idempotency
    double deltanorm=arma::trace(Dd*S*Dd*S);
    // Compute norm of deviation from reference density
    arma::mat Dr=Dt-Dref;
    double diffnorm=arma::trace(Dr*S*Dr*S);
    // Remove matrix from stack?
    if(deltanorm/diffnorm>0.5)
      delind.push_back(ii);
  }

  // Remove the bad matrices.
  for(size_t i=delind.size()-1;i<delind.size();i--) {
      // Yes.
    //      printf("Erasing density matrix %i.\n",(int) delind[i]);
      fflush(stdout);
      Ds.erase(Ds.begin()+delind[i]);
      Es.erase(Es.begin()+delind[i]);
      Fs.erase(Fs.begin()+delind[i]);
  }

  // Update the minimal index once again
  update_minind();
}

arma::mat TRDSM::get_M() const {
  size_t N=Ds.size()-1;

  // Form S2 metric
  arma::mat S2(N+1,N+1);
  for(size_t mi=0;mi<Ds.size();mi++)
    for(size_t ni=0;ni<=mi;ni++) {
      S2(mi,ni)=arma::trace(Ds[mi]*S*Ds[ni]*S);
      // Symmetrize
      S2(ni,mi)=S2(mi,ni);
    }

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
      //      M(m,n)=arma::trace(Ds[mi]*S*Ds[ni]*S);

      // Taken from LSDALTON implementation, not documented anywhere(!)
      M(m,n)=S2(mi,ni)-S2(mi,minind)-S2(minind,ni)+S2(minind,minind);
      n++;
    }

    m++;
  }

  return M;
}

arma::mat TRDSM::get_Dbar(const arma::vec & c) const {
  arma::mat Dbar=Ds[minind];
  size_t i=0;
  for(size_t ii=0;ii<Ds.size();ii++)
    if(ii!=minind) {
      Dbar+=c(i)*(Ds[ii]-Ds[minind]);
      i++;
    }
  return Dbar;
}

arma::mat TRDSM::get_Fbar(const arma::vec & c) const {
  arma::mat Fbar=Fs[minind];
  size_t i=0;
  for(size_t ii=0;ii<Fs.size();ii++)
    if(ii!=minind) {
      Fbar+=c(i)*(Fs[ii]-Fs[minind]);
      i++;
    }
  return Fbar;
}

std::vector<arma::mat> TRDSM::get_Dn0() const {
  std::vector<arma::mat> ret(Ds.size()-1);
  size_t i=0;
  for(size_t ii=0;ii<Ds.size();ii++)
    if(ii!=minind) {
      ret[i]=Ds[ii]-Ds[minind];
      i++;
    }
  return ret;
}

std::vector<arma::mat> TRDSM::get_Fn0() const {
  std::vector<arma::mat> ret(Fs.size()-1);
  size_t i=0;
  for(size_t ii=0;ii<Fs.size();ii++)
    if(ii!=minind) {
      ret[i]=Fs[ii]-Fs[minind];
      i++;
    }

  return ret;
}

double TRDSM::E_DSM(const arma::vec & c) const {
  // Averaged matrices
  arma::mat Dbar=get_Dbar(c);
  arma::mat Fbar=get_Fbar(c);

  // Minimal matrices
  arma::mat F0=Fs[minind];
  arma::mat D0=Ds[minind];

  // Helper
  arma::mat DbarS=Dbar*S;
  arma::mat DbarFbar=Dbar*Fbar;

  double E=Es[minind];
  E+=arma::trace(Dbar*F0)-arma::trace(D0*F0)-arma::trace(DbarFbar)-arma::trace(D0*Fbar);
  E+=arma::trace(6.0*DbarS*DbarFbar)-arma::trace(4.0*DbarS*DbarS*DbarFbar);

  return E;
}

double TRDSM::dE_DSM(const arma::vec & g, const arma::mat & H, const arma::vec & c) const {
  return arma::dot(g,c)+0.5*arma::as_scalar(arma::trans(c)*H*c);
}

arma::vec TRDSM::get_gradient(const arma::vec & c) const {
  // Amount of entries is
  const size_t N=Es.size()-1;

  // F0 and D0 are
  arma::mat F0=Fs[minind];
  arma::mat D0=Ds[minind];

  // Get average matrices and stacks
  arma::mat Fbar=get_Fbar(c);
  arma::mat Dbar=get_Dbar(c);

  // Helpers
  arma::mat SDbar=S*Dbar;
  arma::mat SDbarFbar=SDbar*Fbar;
  arma::mat DbarS=Dbar*S;

  // Difference matrices
  std::vector<arma::mat> Fn=get_Fn0();
  std::vector<arma::mat> Dn=get_Dn0();

  // Fill gradient vector
  arma::vec g(N);
  for(size_t n=0;n<N;n++) {
    // Fill gradient vector
    g(n)=arma::trace(Dn[n]*(F0-Fbar))-arma::trace((Dbar+D0)*Fn[n]);
    g(n)+=6.0*(arma::trace(Dn[n]*SDbarFbar)+arma::trace(DbarS*Dn[n]*Fbar) + arma::trace(Dbar*SDbar*Fn[n]));
    g(n)+=-4.0*(arma::trace(Dn[n]*SDbar*SDbarFbar)+arma::trace(DbarS*Dn[n]*SDbarFbar) + arma::trace(Dbar*SDbar*S*Dn[n]*Fbar) + arma::trace(Dbar*SDbar*SDbar*Fn[n]));
  }

#ifdef DEBUG
  if(!hessian) {
    // Check the gradient.
    arma::vec ng(N);
    for(size_t n=0;n<N;n++) {
      // Step size
      double dx=STEPSIZE;

      // Left and right values of c
      arma::vec lc(c), rc(c);
      lc(n)-=dx;
      rc(n)+=dx;

      // Corresponding energies
      double lE=E_DSM(lc), rE=E_DSM(rc);
      // Gradient is
      ng(n)=(rE-lE)/(2.0*dx);

      printf("g(%i)=%e vs. %e (l %e r %e), diff %e %%\n",(int) n+1,g(n),ng(n),lE,rE,(ng(n)-g(n))/g(n)*100.0);
    }
  }
#endif

  return g;
}

void TRDSM::get_gradient_hessian(const arma::vec & c, arma::vec & g, arma::mat & H) const {
  // Amount of entries is
  const size_t N=Es.size()-1;

  // F0 and D0 are
  arma::mat F0=Fs[minind];
  arma::mat D0=Ds[minind];

  // Get average matrices and stacks
  arma::mat Fbar=get_Fbar(c);
  arma::mat Dbar=get_Dbar(c);

  // Helpers
  arma::mat SDbar=S*Dbar;
  arma::mat SDbarFbar=SDbar*Fbar;
  arma::mat DbarS=Dbar*S;

  // Difference matrices
  std::vector<arma::mat> Fn=get_Fn0();
  std::vector<arma::mat> Dn=get_Dn0();

  // Fill gradient vector
  g.zeros(N);
  for(size_t n=0;n<N;n++) {
    // Fill gradient vector
    g(n)=arma::trace(Dn[n]*(F0-Fbar))-arma::trace((Dbar+D0)*Fn[n]);
    g(n)+=6.0*(arma::trace(Dn[n]*SDbarFbar)+arma::trace(DbarS*Dn[n]*Fbar) + arma::trace(Dbar*SDbar*Fn[n]));
    g(n)+=-4.0*(arma::trace(Dn[n]*SDbar*SDbarFbar)+arma::trace(DbarS*Dn[n]*SDbarFbar) + arma::trace(Dbar*SDbar*S*Dn[n]*Fbar) + arma::trace(Dbar*SDbar*SDbar*Fn[n]));
  }

#ifdef DEBUG
  // Check the gradient.
  arma::vec ng(N);
  for(size_t n=0;n<N;n++) {
    // Step size
    double dx=STEPSIZE;

    // Left and right values of c
    arma::vec lc(c), rc(c);
    lc(n)-=dx;
    rc(n)+=dx;

    // Corresponding energies
    double lE=E_DSM(lc), rE=E_DSM(rc);
    // Gradient is
    ng(n)=(rE-lE)/(2.0*dx);

      printf("g(%i)=%e vs. %e (l %e r %e), diff %e %%\n",(int) n+1,g(n),ng(n),lE,rE,(ng(n)-g(n))/g(n)*100.0);
  }
#endif

  // Fill in Hessian
  H.zeros(N,N);
  for(size_t m=0;m<N;m++) {
    for(size_t n=0;n<=m;n++) {
      H(m,n)=
	-arma::trace(Dn[n]*Fn[m])
	-arma::trace(Dn[m]*Fn[n]);

      H(m,n)+=
	 arma::trace(6.0*Dn[n]*S*Dn[m]*Fbar)
	+arma::trace(6.0*Dn[m]*S*Dn[n]*Fbar);
      H(m,n)+=
	 arma::trace(6.0*Dn[n]*SDbar*Fn[m])
	+arma::trace(6.0*Dn[m]*SDbar*Fn[n]);
      H(m,n)+=
	 arma::trace(6.0*DbarS*Dn[n]*Fn[m])
	+arma::trace(6.0*DbarS*Dn[m]*Fn[n]);

      H(m,n)+=
	-arma::trace(4.0*Dn[n]*S*Dn[m]*SDbarFbar)
	-arma::trace(4.0*Dn[m]*S*Dn[n]*SDbarFbar);
      H(m,n)+=
	-arma::trace(4.0*Dn[n]*SDbar*S*Dn[m]*Fbar)
	-arma::trace(4.0*Dn[m]*SDbar*S*Dn[n]*Fbar);
      H(m,n)+=
	-arma::trace(4.0*Dn[n]*SDbar*SDbar*Fn[m])
	-arma::trace(4.0*Dn[m]*SDbar*SDbar*Fn[n]);
      H(m,n)+=
	-arma::trace(4.0*DbarS*Dn[n]*S*Dn[m]*Fbar)
	-arma::trace(4.0*DbarS*Dn[m]*S*Dn[n]*Fbar);
      H(m,n)+=
	-arma::trace(4.0*DbarS*Dn[n]*SDbar*Fn[m])
	-arma::trace(4.0*DbarS*Dn[m]*SDbar*Fn[n]);
      H(m,n)+=
	-arma::trace(4.0*DbarS*DbarS*Dn[n]*Fn[m])
	-arma::trace(4.0*DbarS*DbarS*Dn[m]*Fn[n]);

      // Symmetrize
      H(n,m)=H(m,n);
    }
  }

#ifdef DEBUG
  // Check the Hessian. Get the gradient
  hessian=true;
  arma::mat nH(N,N);
  for(size_t m=0;m<N;m++) {
    // Step size
    double dx=STEPSIZE;

    for(size_t n=0;n<N;n++) {
      // Left and right values of c
      arma::vec lc(c), rc(c);
      lc(m)-=dx;
      rc(m)+=dx;

      // Corresponding gradients are
      arma::vec lg=get_gradient(lc);
      arma::vec rg=get_gradient(rc);

      // Hessian element is
      nH(m,n)=(rg(n)-lg(n))/(2.0*dx);

      printf("H(%i,%i)=%e vs. %e, diff %e %%\n",(int) m+1, (int) n+1,H(m,n),nH(m,n),(nH(m,n)/H(m,n)-1.0)*100.0);
    }
  }
  hessian=false;
#endif
}

arma::vec TRDSM::solve_c_ls() const {
  size_t N=Es.size()-1;

  // Begin at
  arma::vec c(N);
  c.zeros();

  // Metric is
  arma::mat M=get_M();

  // Old c
  arma::vec cold;

  // Current energy
  double E=E_DSM(c);
  //  printf("Initial energy is %e.\n",E);

  // Accepted norm of coefficients
  const double h=1.2;

  // Old energy
  double Eold;

  size_t ig=0;

  // Do gradient descent
  while(true) {
    // Store old energy
    Eold=E;
    // and old c
    cold=c;

    // Compute gradient at c
    ig++;
    arma::vec g=get_gradient(c);
    size_t is=0;

    // Compute helpers
    double cdotc=arma::as_scalar(arma::trans(c)*M*c);
    double cdotg=arma::as_scalar(arma::trans(c)*M*g);
    double gdotg=arma::as_scalar(arma::trans(g)*M*g);

    // Compute step size
    double stepmax=(sqrt(cdotg*cdotg+gdotg*(h*h-cdotc))+cdotg)/gdotg;

    // Determine optimal step size within bracket
    double a=0.0;
    double b=stepmax;

    // Corresponding vectors
    arma::vec ca(c);
    arma::vec cb=c-b*g;

    // Check step size
    double la=step_len(ca,M);
    double lb=step_len(cb,M);
    //    printf("Current length is %e, maximum length is %e.\n",la,lb);

    // Golden ratio
    const double tau=2.0/(1.0+sqrt(5.0));
    // Compute trials
    do {
      is++;
      double l1=a+(1-tau)*(b-a);
      double l2=a+tau*(b-a);

      // Corresponding vectors and values
      arma::vec c1=c-l1*g;
      arma::vec c2=c-l2*g;

      double E1=E_DSM(c1);
      double E2=E_DSM(c2);

      double Emin=std::min(E1,E2);

      // Check that we aren't sitting on the minimum
      if(E<Emin) {
	// Move right limit to left value.
	b=l1;
	cb=c1;
      } else if(E1<E2) {
	// Move right limit
	b=l2;
	cb=c2;
      } else if(E1>E2) {
	// Move left limit
	a=l1;
	ca=c1;
      } else { // E1==E2
	// Move both limits
	a=l1;
	b=l2;
	ca=c1;
	cb=c2;
      }

      //      printf("Gradient %i step %i (length %e): minimal energy %e, change by %e.\n",(int) ig, (int) is,(a+b)/2.0,Emin,Emin-E);
      fflush(stdout);

      if(E-Emin<ETOL)
	// Converged
	break;

      E=Emin;
    } while(b-a > DBL_EPSILON);

    // Update c
    c=cold-(la+lb)/2.0*g;

    //    printf("Gradient %i decreased energy by %e, len(c)=%e.\n",(int) ig, Eold-E,step_len(c,M));

    // Check norm of c
    /*
    if(step_len(c,M)>k)
      return cold;
    */

    if(Eold-E<ETOL) {
      //      printf("Energy converged to %e precision.\n",Eold-E);
      fflush(stdout);
      break;
    }
  }

  return c;
}

arma::vec TRDSM::solve_c() const {
  // Lagrange multiplier for the step length
  double mu=0.0;
  double oldmu=0.0;

  // The step vector
  arma::vec c;

  // Accepted norm of coefficients
  const double h=1.2;

  // Get the M matrix
  arma::mat M=get_M();

  // Eigenvalues of Hessian
  arma::vec Hval;
  // Eigenvectors of Hessian
  arma::mat Hvec;

  // Length of the step
  double l;

  // Loop over mu
  while(true) {
    // Solve c
    solve_c(c,mu,Hval,Hvec,M);

    // Compute step length
    l=step_len(c,M);

    // Increment mu if necessary
    if(l>h) {
      oldmu=mu;
      mu+=1.0;
    } else
      break;
  }

  // Do we need to do a bisection to refine the step length?
  if(oldmu!=mu) {
    // Yes, we do. Left value is
    double lx=oldmu;
    // and right value is
    double rx=mu;

    // Correct shift is somewhere in between.
    double mx;
    while(rx-lx>10*DBL_EPSILON) {
      // Middle value is
      mx=(lx+rx)/2.0;

      // Solve c
      solve_c(c,mx,Hval,Hvec,M);

      // and the length here is
      l=step_len(c,M);

      // Determine which limit to move
      if(l>h) // Move left limit
	lx=mx;
      if(l<h) // Move right limit
	rx=mx;
    }

    mu=mx;
  }

  //  printf("Final value of mu is %e.\n",mu);

  return c;
}


void TRDSM::solve_c(arma::vec & c, double mu, arma::vec & Hval, arma::mat & Hvec, const arma::mat & M) const {
  // Solve c self-consistently using fixed value of mu
  double l;

  // The starting point is
  c.zeros(Es.size()-1);
  // Current energy
  double E=E_DSM(c);

  // Old value
  arma::vec cold(c);
  // Difference from old value
  double d;

  //  double Estart=E_DSM(c);

  // Trust radius
  double k=0.2;

  do {
    // Expand the energy locally as a parabolic function:
    // E(c) ~ E(c0) + dc.g(c0) + ½ dc.H(g0).dc

    // Get the gradient and the Hessian in the current point
    arma::vec g;
    arma::mat H;
    get_gradient_hessian(c,g,H);

    // The level-shifted Hessian is
    arma::mat hess=H-mu*M;
    eig_sym_ordered(Hval,Hvec,hess);

    // Solve (H - mu M) dc = -g.
    arma::vec dc(c);
    dc.zeros();
    for(size_t i=0;i<Hval.n_elem;i++)
      if(Hval(i)>0.0) // Skip negative eigenvalues
	dc+=arma::dot(Hvec.col(i),-g)/Hval(i)*Hvec.col(i);

    // Do step
    cold=c;
    c+=dc;

    // Check that we are within the trust region
    l=step_len(c,M);
    if(l>k) {
      // Outside trust region. Go to boundary using binary search
      double lx=0.0;
      double rx=1.0;

      double mx;
      do {
	mx=(lx+rx)/2.0;
	// New value of c
	c=cold+mx*dc;
	// Compute length
	l=step_len(c,M);

	// Check what to do
	if(l>k)
	  rx=mx;
	else if(l<k)
	  lx=mx;
	else break;
      } while(rx-lx > 10*DBL_EPSILON);
    }

    // Length of step is
    d=step_len(c-cold,M);

    // Compute new energy
    double Eold=E;
    E=E_DSM(c);

    // Trust region update. Predicted energy change is
    double dEpred=dE_DSM(g,H,c);
    // while actual change is
    double dEact=E-Eold;
    // Compute ratio
    double ratio;
    if(fabs(dEpred)>DBL_EPSILON)
      ratio=dEact/dEpred;
    else
      ratio=-1.0;

    // Do we accept the step?
    if(fabs(dEact)<ETOL || fabs(dEpred)<ETOL) {
      // Converged.
      break;
    }

    // Accept move?
    if(E<Eold) {
      //      printf("Move accepted, len(c)=%e, ratio is %.2f, change in DSM energy is %e (predicted %e), k=%e.\n",step_len(c,M),ratio,dEact,dEpred,k);
    } else {
      //      printf("Move refused, len(c)=%e, ratio is %.2f, change in DSM energy is %e (predicted %e), dc=%e, k=%e.\n",step_len(c,M),ratio,dEact,dEpred,step_len(c-cold,M),k);
      c=cold;
    }

    // Update trust radius?
    if(ratio>=0.75 && ratio <=1.25) {
      // Yes - model is good.

      // Increase trust radius
      k*=1.2;
    } else {
      // No, reduce trust radius
      k*=0.7;
    }
  } while(d>sqrt(DBL_EPSILON) && k>DBL_EPSILON);

  //  double Efinal=E_DSM(c);
  //  printf("Starting DSM energy=%e, final DSM energy=%e (reduction by %e).\n",Estart,Efinal,Estart-Efinal);
}

double TRDSM::step_len(const arma::vec & c, const arma::mat & M) const {
  return sqrt(arma::as_scalar(arma::trans(c)*M*(c)));
}

arma::mat TRDSM::solve() const {
  // Solve the coefficients
  arma::vec c;

  if(Ds.size()==1)
    // Trivial case
    c.zeros(1);
  else
    //    c=solve_c_ls();
    c=solve_c();

  // and perform the averaging
  return get_Fbar(c);
}
