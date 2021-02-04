/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#include <cfloat>
#include "diis.h"
#include "lbfgs.h"
#include "linalg.h"
#include "mathf.h"
#include "stringutil.h"

// Maximum allowed absolute weight for a Fock matrix
#define MAXWEIGHT 10.0
// Trigger cooloff if energy rises more than
#define COOLTHR 0.1

bool operator<(const diis_pol_entry_t & lhs, const diis_pol_entry_t & rhs) {
  return lhs.E < rhs.E;
}

bool operator<(const diis_unpol_entry_t & lhs, const diis_unpol_entry_t & rhs) {
  return lhs.E < rhs.E;
}

DIIS::DIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool usediis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) {
  S=S_;
  Sinvh=Sinvh_;
  usediis=usediis_;
  useadiis=useadiis_;
  verbose=verbose_;
  imax=imax_;

  // Start mixing in DIIS weight when error is
  diiseps=diiseps_;
  // and switch to full DIIS when error is
  diisthr=diisthr_;

  // No cooloff
  cooloff=0;
}

rDIIS::rDIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool usediis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) : DIIS(S_,Sinvh_,usediis_,diiseps_,diisthr_,useadiis_,verbose_,imax_) {
}

uDIIS::uDIIS(const arma::mat & S_, const arma::mat & Sinvh_, bool combine_, bool usediis_, double diiseps_, double diisthr_, bool useadiis_, bool verbose_, size_t imax_) : DIIS(S_,Sinvh_,usediis_,diiseps_,diisthr_,useadiis_,verbose_,imax_), combine(combine_) {
}

DIIS::~DIIS() {
}

rDIIS::~rDIIS() {
}

uDIIS::~uDIIS() {
}

void rDIIS::clear() {
  stack.clear();
}

void uDIIS::clear() {
  stack.clear();
}

void rDIIS::erase_last() {
  stack.erase(stack.begin());
}

void uDIIS::erase_last() {
  stack.erase(stack.begin());
}

void rDIIS::update(const arma::mat & F, const arma::mat & P, double E, double & error) {
  // New entry
  diis_unpol_entry_t hlp;
  hlp.F=F;
  hlp.P=P;
  hlp.E=E;

  // Compute error matrix
  arma::mat errmat(F*P*S);
  // FPS - SPF
  errmat-=arma::trans(errmat);
  // and transform it to the orthonormal basis (1982 paper, page 557)
  errmat=arma::trans(Sinvh)*errmat*Sinvh;
  // and store it
  hlp.err=arma::vectorise(errmat);

  // DIIS error is
  error=arma::max(arma::max(arma::abs(errmat)));

  // Is stack full?
  if(stack.size()==imax) {
    erase_last();
  }
  // Add to stack
  stack.push_back(hlp);

  // Update ADIIS helpers
  PiF_update();
}

void rDIIS::PiF_update() {
  const arma::mat & Fn=stack[stack.size()-1].F;
  const arma::mat & Pn=stack[stack.size()-1].P;

  // Update matrices
  PiF.zeros(stack.size());
  for(size_t i=0;i<stack.size();i++)
    PiF(i)=arma::trace((stack[i].P-Pn)*Fn);

  PiFj.zeros(stack.size(),stack.size());
  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<stack.size();j++)
      PiFj(i,j)=arma::trace((stack[i].P-Pn)*(stack[j].F-Fn));
}

void uDIIS::update(const arma::mat & Fa, const arma::mat & Fb, const arma::mat & Pa, const arma::mat & Pb, double E, double & error) {
  // New entry
  diis_pol_entry_t hlp;
  hlp.Fa=Fa;
  hlp.Fb=Fb;
  hlp.Pa=Pa;
  hlp.Pb=Pb;
  hlp.E=E;

  // Compute error matrices
  arma::mat errmata(Fa*Pa*S);
  arma::mat errmatb(Fb*Pb*S);
  // FPS - SPF
  errmata-=arma::trans(errmata);
  errmatb-=arma::trans(errmatb);
  // and transform them to the orthonormal basis (1982 paper, page 557)
  errmata=arma::trans(Sinvh)*errmata*Sinvh;
  errmatb=arma::trans(Sinvh)*errmatb*Sinvh;
  // and store it
  if(combine) {
    hlp.err=arma::vectorise(errmata+errmatb);
  } else {
    hlp.err.zeros(errmata.n_elem+errmatb.n_elem);
    hlp.err.subvec(0,errmata.n_elem-1)=arma::vectorise(errmata);
    hlp.err.subvec(errmata.n_elem,hlp.err.n_elem-1)=arma::vectorise(errmatb);
  }

  // DIIS error is
  error=arma::max(arma::abs(hlp.err));

  // Is stack full?
  if(stack.size()==imax) {
    erase_last();
  }
  // Add to stack
  stack.push_back(hlp);

  // Update ADIIS helpers
  PiF_update();
}

void uDIIS::PiF_update() {
  const arma::mat & Fan=stack[stack.size()-1].Fa;
  const arma::mat & Fbn=stack[stack.size()-1].Fb;
  const arma::mat & Pan=stack[stack.size()-1].Pa;
  const arma::mat & Pbn=stack[stack.size()-1].Pb;

  // Update matrices
  PiF.zeros(stack.size());
  for(size_t i=0;i<stack.size();i++)
    PiF(i)=arma::trace((stack[i].Pa-Pan)*Fan) + arma::trace((stack[i].Pb-Pbn)*Fbn);

  PiFj.zeros(stack.size(),stack.size());
  for(size_t i=0;i<stack.size();i++)
    for(size_t j=0;j<stack.size();j++)
      PiFj(i,j)=arma::trace((stack[i].Pa-Pan)*(stack[j].Fa-Fan))+arma::trace((stack[i].Pb-Pbn)*(stack[j].Fb-Fbn));
}

arma::vec rDIIS::get_energies() const {
  arma::vec E(stack.size());
  for(size_t i=0;i<stack.size();i++)
    E(i)=stack[i].E;
  return E;
}

arma::mat rDIIS::get_diis_error() const {
  arma::mat err(stack[0].err.n_elem,stack.size());
  for(size_t i=0;i<stack.size();i++)
    err.col(i)=stack[i].err;
  return err;
}

arma::vec uDIIS::get_energies() const {
  arma::vec E(stack.size());
  for(size_t i=0;i<stack.size();i++)
    E(i)=stack[i].E;
  return E;
}
arma::mat uDIIS::get_diis_error() const {
  arma::mat err(stack[0].err.n_elem,stack.size());
  for(size_t i=0;i<stack.size();i++)
    err.col(i)=stack[i].err;
  return err;
}

arma::vec DIIS::get_w_wrk() {
  // DIIS error
  arma::mat de=get_diis_error();
  double err=arma::max(arma::abs(de.col(de.n_cols-1)));

  // Weight
  arma::vec w;

  if(useadiis && !usediis) {
    w=get_w_adiis();
    if(verbose) {
      printf("ADIIS weights\n");
      print_mat(w.t(),"% .2e ");
    }
  } else if(!useadiis && usediis) {
    // Use DIIS only if error is smaller than threshold
    if(err>diisthr)
      throw std::runtime_error("DIIS error too large for only DIIS to converge wave function.\n");

    w=get_w_diis();

    if(verbose) {
      printf("DIIS weights\n");
      print_mat(w.t(),"% .2e ");
    }
  } else if(useadiis && usediis) {
    // Sliding scale: DIIS weight
    double diisw=std::max(std::min(1.0 - (err-diisthr)/(diiseps-diisthr), 1.0), 0.0);
    // ADIIS weght
    double adiisw=1.0-diisw;

    // Determine cooloff
    if(cooloff>0) {
      diisw=0.0;
      cooloff--;
    } else {
      // Check if energy has increased
      arma::vec E=get_energies();
      if(E.n_elem>1 &&  E(E.n_elem-1)-E(E.n_elem-2) > COOLTHR) {
	cooloff=2;
	diisw=0.0;
      }
    }

    w.zeros(de.n_cols);

    // DIIS and ADIIS weights
    arma::vec wd, wa;
    if(diisw!=0.0) {
      wd=get_w_diis();
      w+=diisw*wd;
    }
    if(adiisw!=0.0) {
      wa=get_w_adiis();
      w+=adiisw*wa;
    }

    if(verbose) {
      if(adiisw!=0.0) {
        printf("ADIIS weights\n");
        print_mat(wa.t(),"% .2e ");
      }
      if(diisw!=0.0) {
        printf("CDIIS weights\n");
        print_mat(wd.t(),"% .2e ");
      }
      if(adiisw!=0.0 && diisw!=0.0) {
        printf(" DIIS weights\n");
        print_mat(w.t(),"% .2e ");
      }
    }

  } else
    throw std::runtime_error("Nor DIIS or ADIIS has been turned on.\n");

  return w;
}

arma::vec DIIS::get_w() {
  arma::vec sol;
  while(true) {
    sol=get_w_wrk();
    if(std::abs(sol(sol.n_elem-1))<=sqrt(DBL_EPSILON)) {
      if(sol.n_elem > 2) {
        if(verbose) printf("Weight on last matrix too small, dropping the oldest matrix.\n");
        erase_last();
        PiF_update();
      } else {
        if(verbose) printf("Weight on last matrix still too small; switching to damping.\n");
        sol.zeros();

        double damp=0.1;
        sol(0)=1-damp;
        sol(1)=damp;
        break;
      }
    } else
      break;
  }

  return sol;
}

arma::vec DIIS::get_w_diis() const {
  arma::mat errs=get_diis_error();
  return get_w_diis_wrk(errs);
}

arma::vec DIIS::get_w_diis_wrk(const arma::mat & errs) const {
  // Size of LA problem
  int N=(int) errs.n_cols;

  // Array holding the errors
  arma::mat B(N,N);
  B.zeros();
  // Compute errors
  for(int i=0;i<N;i++)
    for(int j=0;j<N;j++) {
      B(i,j)=arma::dot(errs.col(i),errs.col(j));
    }

  /*
    The C1-DIIS method is equivalent to solving the group of linear
    equations
            B w = lambda 1       (1)

    where B is the error matrix, w are the DIIS weights, lambda is the
    Lagrange multiplier that guarantees that the weights sum to unity,
    and 1 stands for a unit vector (1 1 ... 1)^T.

    By rescaling the weights as w -> w/lambda, equation (1) is
    reverted to the form
            B w = 1              (2)

    which can easily be solved using SVD techniques.

    Finally, the weights are renormalized to satisfy
            \sum_i w_i = 1
    which takes care of the Lagrange multipliers.
  */

  // Right-hand side of equation is
  arma::vec rh(N);
  rh.ones();

  // Singular value decomposition
  arma::mat U, V;
  arma::vec sval;
  if(!arma::svd(U,sval,V,B,"std")) {
    throw std::logic_error("SVD failed in DIIS.\n");
  }
  //sval.print("Singular values");

  // Form solution vector
  arma::vec sol(N);
  sol.zeros();
  for(int i=0;i<N;i++) {
#if 0
    // Perform Tikhonov regularization for the singular
    // eigenvalues. This doesn't appear to work as well as just
    // cutting out the spectrum.
    double s(sval(i));
    double t=1e-4;
    double invs=s/(s*s + t*t);
    sol += arma::dot(U.col(i),rh)*invs * V.col(i);
#else
    // Just cut out the problematic part. Weirdly, it appears that any
    // kind of screening of the singular eigenvalues results in poorer
    // convergence behavior(!)
    if(sval(i) != 0.0)
      sol += arma::dot(U.col(i),rh)/sval(i) * V.col(i);
#endif
  }

  // Sanity check
  if(arma::sum(sol)==0.0)
    sol.ones();

  // Normalize solution
  //printf("Sum of weights is %e\n",arma::sum(sol));
  sol/=arma::sum(sol);

  return sol;
}

void rDIIS::solve_F(arma::mat & F) {
  arma::vec sol(get_w());
  // Form weighted Fock matrix
  F.zeros();
  for(size_t i=0;i<stack.size();i++)
    F+=sol(i)*stack[i].F;
}

void uDIIS::solve_F(arma::mat & Fa, arma::mat & Fb) {
  arma::vec sol(get_w());

  // Form weighted Fock matrix
  Fa.zeros();
  Fb.zeros();
  for(size_t i=0;i<stack.size();i++) {
    Fa+=sol(i)*stack[i].Fa;
    Fb+=sol(i)*stack[i].Fb;
  }
}

void rDIIS::solve_P(arma::mat & P) {
  arma::vec sol(get_w());

  // Form weighted density matrix
  P.zeros();
  for(size_t i=0;i<stack.size();i++)
    P+=sol(i)*stack[i].P;
}

void uDIIS::solve_P(arma::mat & Pa, arma::mat & Pb) {
  arma::vec sol(get_w());

  // Form weighted density matrix
  Pa.zeros();
  Pb.zeros();
  for(size_t i=0;i<stack.size();i++) {
    Pa+=sol(i)*stack[i].Pa;
    Pb+=sol(i)*stack[i].Pb;
  }
}

static void find_minE(const std::vector< std::pair<double,double> > & steps, double & Emin, size_t & imin) {
  Emin=steps[0].second;
  imin=0;
  for(size_t i=1;i<steps.size();i++)
    if(steps[i].second < Emin) {
      Emin=steps[i].second;
      imin=i;
    }
}

static arma::vec compute_c(const arma::vec & x) {
  // Compute contraction coefficients
  return x%x/arma::dot(x,x);
}

static arma::mat compute_jac(const arma::vec & x) {
  // Compute jacobian of transformation: jac(i,j) = dc_i / dx_j

  // Compute coefficients
  arma::vec c(compute_c(x));
  double xnorm=arma::dot(x,x);

  arma::mat jac(c.n_elem,c.n_elem);
  for(size_t i=0;i<c.n_elem;i++) {
    double ci=c(i);
    double xi=x(i);

    for(size_t j=0;j<c.n_elem;j++) {
      double xj=x(j);

      jac(i,j)=-ci*2.0*xj/xnorm;
    }

    // Extra term on diagonal
    jac(i,i)+=2.0*xi/xnorm;
  }

  return jac;
}

arma::vec DIIS::get_w_adiis() const {
  // Number of parameters
  size_t N=PiF.n_elem;

  if(N==1) {
    // Trivial case.
    arma::vec ret(1);
    ret.ones();
    return ret;
  }

  // Starting point: equal weights on all matrices
  arma::vec x=arma::ones<arma::vec>(N)/N;

  // BFGS accelerator
  LBFGS bfgs;

  // Step size
  double steplen=0.01, fac=2.0;

  for(size_t iiter=0;iiter<1000;iiter++) {
    // Get gradient
    //double E(get_E_adiis(x));
    arma::vec g(get_dEdx_adiis(x));
    if(arma::norm(g,2)<=1e-7) {
      break;
    }

    // Search direction
    bfgs.update(x,g);
    arma::vec sd(-bfgs.solve());

    // Do a line search on the search direction
    std::vector< std::pair<double, double> > steps;
    // First, we try a fraction of the current step length
    {
      std::pair<double, double> p;
      p.first=steplen/fac;
      p.second=get_E_adiis(x+sd*p.first);
      steps.push_back(p);
    }
    // Next, we try the current step length
    {
      std::pair<double, double> p;
      p.first=steplen;
      p.second=get_E_adiis(x+sd*p.first);
      steps.push_back(p);
    }

    // Minimum energy and index
    double Emin;
    size_t imin;

    while(true) {
      // Sort the steps in length
      std::sort(steps.begin(),steps.end());

      // Find the minimum energy
      find_minE(steps,Emin,imin);

      // Where is the minimum?
      if(imin==0 || imin==steps.size()-1) {
	// Need smaller step
	std::pair<double,double> p;
	if(imin==0) {
	  p.first=steps[imin].first/fac;
	  if(steps[imin].first<DBL_EPSILON)
	    break;
	} else {
	  p.first=steps[imin].first*fac;
	}
	p.second=get_E_adiis(x+sd*p.first);
	steps.push_back(p);
      } else {
	// Optimum is somewhere in the middle
	break;
      }
    }

    if((imin!=0) && (imin!=steps.size()-1)) {
      // Interpolate: A b = y
      arma::mat A(3,3);
      arma::vec y(3);
      for(size_t i=0;i<3;i++) {
	A(i,0)=1.0;
	A(i,1)=steps[imin+i-1].first;
	A(i,2)=std::pow(A(i,1),2);

	y(i)=steps[imin+i-1].second;
      }

      arma::mat b;
      if(arma::solve(b,A,y) && b(2)>sqrt(DBL_EPSILON)) {
	// Success in solution and parabola gives minimum.

	// The minimum of the parabola is at
	double x0=-b(1)/(2*b(2));

	// Is this an interpolation?
	if(A(0,1) < x0 && x0 < A(2,1)) {
	  // Do the calculation with the interpolated step
	  std::pair<double,double> p;
	  p.first=x0;
	  p.second=get_E_adiis(x+sd*p.first);
	  steps.push_back(p);

	  // Find the minimum energy
	  find_minE(steps,Emin,imin);
	}
      }
    }

    if(steps[imin].first<DBL_EPSILON)
      break;

    // Switch to the minimum geometry
    x+=steps[imin].first*sd;
    // Store optimal step length
    steplen=steps[imin].first;

    //printf("Step %i: energy decreased by %e, gradient norm %e\n",(int) iiter+1,steps[imin].second-E,arma::norm(g,2)); fflush(stdout);
  }

  // Calculate weights
  return compute_c(x);
}

double DIIS::get_E_adiis(const arma::vec & x) const {
  // Consistency check
  if(x.n_elem != PiF.n_elem) {
    throw std::domain_error("Incorrect number of parameters.\n");
  }

  arma::vec c(compute_c(x));

  // Compute energy
  double Eval=0.0;
  Eval+=2.0*arma::dot(c,PiF);
  Eval+=arma::as_scalar(arma::trans(c)*PiFj*c);

  return Eval;
}

arma::vec DIIS::get_dEdx_adiis(const arma::vec & x) const {
  // Compute contraction coefficients
  arma::vec c(compute_c(x));

  // Compute derivative of energy
  arma::vec dEdc=2.0*PiF + PiFj*c + arma::trans(PiFj)*c;

  // Compute jacobian of transformation: jac(i,j) = dc_i / dx_j
  arma::mat jac(compute_jac(x));

  // Finally, compute dEdx by plugging in Jacobian of transformation
  // dE/dx_i = dc_j/dx_i dE/dc_j
  return arma::trans(jac)*dEdc;
}
