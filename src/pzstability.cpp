/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include <cfloat>
#include "pzstability.h"
#include "checkpoint.h"
#include "stringutil.h"
#include "linalg.h"
#include "lbfgs.h"
#include "timer.h"
#include "mathf.h"
#include "dftfuncs.h"

// Threshold for a changed orbital
#define CHANGETHR (100*DBL_EPSILON)

// Form orbital density matrix
arma::cx_mat form_density(const arma::cx_mat & C) {
  return arma::conj(C)*arma::strans(C);
}

// Conversion to proper gradient
arma::cx_mat gradient_convert(const arma::cx_mat & M) {
  return 2.0*arma::real(M)*COMPLEX1 - 2.0*arma::imag(M)*COMPLEXI;
}

arma::cx_mat spread_ov(const arma::vec & x, size_t o, size_t v, bool real, bool imag) {
  // Sanity check
  if((real && !imag && x.n_elem != o*v) || (!real && imag && x.n_elem != o*v) || (real && imag && x.n_elem != 2*o*v))
    throw std::logic_error("Invalid vector length for ov rotation.\n");

  // Rotation matrix
  arma::cx_mat r(o,v);
  r.zeros();

  // Collect real part of rotation
  size_t ioff=0;
  if(real) {
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<v;j++)
	r(i,j)=x(i*v + j + ioff)*COMPLEX1;
    ioff+=o*v;
  }
  // Imaginary part
  if(imag) {
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<v;j++)
	r(i,j)+=x(i*v + j + ioff)*COMPLEXI;
    ioff+=o*v;
  }

  return r;
}

arma::vec gather_ov(const arma::cx_mat & Mov, bool real, bool imag) {
  // Matrix size
  size_t o(Mov.n_rows);
  size_t v(Mov.n_cols);

  // Returned parameters
  arma::vec x;
  if(real && imag)
    x.zeros(2*o*v);
  else
    x.zeros(o*v);

  size_t ioff=0;

  // Collect real part of rotation
  if(real) {
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<v;j++)
	x(i*v + j + ioff)=std::real(Mov(i,j));
    ioff+=o*v;
  }

  // Imaginary part
  if(imag) {
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<v;j++)
	x(i*v + j + ioff)=std::imag(Mov(i,j));
    ioff+=o*v;
  }

  return x;
}

arma::cx_mat spread_oo(const arma::vec & x, size_t o, bool real, bool imag) {
  // Sanity check
  if( (((real && !imag) || (!real && imag)) && x.size()!=o*(o-1)/2) || (real && imag && x.size()!=o*(o-1)) )
    throw std::logic_error("Invalid vector length for oo rotation.\n");

  // Rotation matrix
  arma::cx_mat R(o,o);
  R.zeros();

  // Collect real part of rotation
  size_t ioff=0;
  if(real) {
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<i;j++) {
	// Indexing requires j<i
	size_t idx=i*(i-1)/2 + j;
	R(j,i)= x(idx)*COMPLEX1;
	R(i,j)=-x(idx)*COMPLEX1;
      }
    ioff+=o*(o-1)/2;
  }

  // Imaginary part of rotation.
  if(imag) {
    for(size_t i=0;i<o;i++)
      // Diagonal part is just orbital phase which has no effect
      for(size_t j=0;j<i;j++) {
	// Indexing requires j<i
	size_t idx=ioff + i*(i-1)/2 + j;
	R(j,i)+=x(idx)*COMPLEXI;
	R(i,j)-=x(idx)*COMPLEXI;
      }
    ioff+=o*(o-1)/2;
  }

  return R;
}

arma::vec gather_oo(const arma::cx_mat & M, bool real, bool imag) {
  // Matrix size
  size_t o=M.n_cols;

  // Parameters
  arma::vec x;
  if(real && imag)
    x.zeros(o*(o-1));
  else
    x.zeros(o*(o-1)/2);
  size_t ioff=0;

  // Collect real part of rotation
  if(real) {
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<i;j++) {
	// Indexing requires j<i
	size_t idx=i*(i-1)/2 + j;
	x(idx + ioff)=std::real(M(j,i));
      }
    ioff+=o*(o-1)/2;
  }

  // Imaginary part of rotation.
  if(imag) {
    for(size_t i=0;i<o;i++)
      // Diagonal part is just orbital phase which has no effect
      for(size_t j=0;j<i;j++) {
	// Indexing requires j<i
	size_t idx=i*(i-1)/2 + j;
	x(idx + ioff)=std::imag(M(j,i));
      }
    ioff+=o*(o-1)/2;
  }

  return x;
}

FDHessian::FDHessian(bool ver) {
  ss_fd_=cbrt(DBL_EPSILON);
  ss_ls_=1e-4;
  verbose_=ver;
}

FDHessian::~FDHessian() {
}

arma::vec FDHessian::gradient() {
  arma::vec x0(count_params());
  x0.zeros();
  return gradient(x0);
}

arma::vec FDHessian::gradient(const arma::vec & x0) {
  // Amount of parameters
  size_t npar=count_params();

  // Compute gradient
  arma::vec g(npar);
  g.zeros();

  /* This loop isn't OpenMP parallel, because parallellization is
     already used in the energy evaluation. Parallellizing over trials
     here would require use of thread-local DFT grids, and possibly
     even thread-local SCF solver objects (for the Coulomb part).*/
  for(size_t i=0;i<npar;i++) {
    arma::vec x;

    // RHS value
    x=x0;
    x(i)+=ss_fd_;
    double yr=eval(x);

    // LHS value
    x=x0;
    x(i)-=ss_fd_;
    double yl=eval(x);

    // Derivative
    g(i)=(yr-yl)/(2.0*ss_fd_);

    if(std::isnan(g(i))) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Element " << i << " of gradient gives NaN.\n";
      oss << "Step size is " << ss_fd_ << ", and left and right values are " << yl << " and " << yr << ".\n";
      throw std::runtime_error(oss.str());
    }
  }

  return g;
}

typedef struct {
  size_t i;
  size_t j;
} loopidx_t;

arma::mat FDHessian::hessian() {
  // Amount of parameters
  size_t npar=count_params();

  // Compute gradient
  arma::mat h(npar,npar);
  h.zeros();

  std::vector<loopidx_t> idx;
  for(size_t i=0;i<npar;i++)
    for(size_t j=0;j<=i;j++) {
      loopidx_t t;
      t.i=i;
      t.j=j;
      idx.push_back(t);
    }

  /* This loop isn't OpenMP parallel, because parallellization is
     already used in the energy evaluation. Parallellizing over trials
     here would require use of thread-local DFT grids, and possibly
     even thread-local SCF solver objects (for the Coulomb part).*/
  for(size_t ii=0;ii<idx.size();ii++) {
    size_t i=idx[ii].i;
    size_t j=idx[ii].j;

    arma::vec x(npar);

    // RH,RH value
    x.zeros();
    x(i)+=ss_fd_;
    x(j)+=ss_fd_;
    double yrr=eval(x);

    // RH,LH
    x.zeros();
    x(i)+=ss_fd_;
    x(j)-=ss_fd_;
    double yrl=eval(x);

    // LH,RH
    x.zeros();
    x(i)-=ss_fd_;
    x(j)+=ss_fd_;
    double ylr=eval(x);

    // LH,LH
    x.zeros();
    x(i)-=ss_fd_;
    x(j)-=ss_fd_;
    double yll=eval(x);

    // Values
    h(i,j)=(yrr - yrl - ylr + yll)/(4.0*ss_fd_*ss_fd_);
    // Symmetrize
    h(j,i)=h(i,j);

    if(std::isnan(h(i,j))) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Element (" << i << "," << j << ") of Hessian gives NaN.\n";
      oss << "Step size is " << ss_fd_ << ". Stencil values\n";
      oss << "yrr = " << yrr << "\n";
      oss << "yrl = " << yrl << "\n";
      oss << "ylr = " << ylr << "\n";
      oss << "yll = " << yll << "\n";
      throw std::runtime_error(oss.str());
    }
  }

  return h;
}

void FDHessian::update(const arma::vec & x) {
  (void) x;
  throw std::runtime_error("Error - update function must be overloaded!\n");
}

void FDHessian::print_status(size_t iiter, const arma::vec & g, const Timer & t) const {
  if(verbose_)
    printf("\nIteration %i, gradient norm %e, max norm %e (%s)\n",(int) iiter,arma::norm(g,2),arma::max(arma::abs(g)),t.elapsed().c_str());
}

double FDHessian::optimize(size_t maxiter, double gthr, bool max) {
  arma::vec x0;
  if(!count_params())
    return 0.0;
  else
    x0.zeros(count_params());

  double ival=eval(x0);
  if(verbose_)
    printf("Initial value is % .10f\n",ival);

  // Current and previous gradient
  arma::vec g, gold;
  // Search direction
  arma::vec sd;

  for(size_t iiter=0;iiter<maxiter;iiter++) {
    // Evaluate gradient
    gold=g;
    {
      Timer t;
      g=gradient();
      print_status(iiter,g,t);
    }
    if(arma::norm(g,2)<gthr)
      break;

    // Initial step size
    double initstep=ss_ls_;
    // Factor for increase of step size
    double stepfac=2.0;

    // Do line search
    std::vector<double> step, val;

    // Update search direction
    arma::vec oldsd(sd);
    sd = max ? g : -g;

    if(iiter % std::min((size_t) round(sqrt(count_params())),(size_t) 5) !=0) {
      // Update factor
      double gamma;

      // Polak-Ribiere
      gamma=arma::dot(g,g-gold)/arma::dot(gold,gold);
      // Fletcher-Reeves
      //gamma=arma::dot(g,g)/arma::dot(gold,gold);

      // Update search direction
      arma::vec sdnew=sd+gamma*oldsd;

      // Check that new SD is sane
      if(iiter>=1 && arma::dot(g,gold)>=0.2*arma::dot(gold,gold)) {
	if(verbose_) printf("Powell restart - SD step\n");
      } else if(arma::dot(sdnew,sd)<=0) {
	// This would take us into the wrong direction!
	if(verbose_) printf("Bad CG direction. SD step\n");
      } else {
	// Update search direction
	sd=sdnew;
	if(verbose_) printf("CG step\n");
      }
    } else if(verbose_) printf("SD step\n");

    while(true) {
      step.push_back(std::pow(stepfac,step.size())*initstep);
      val.push_back(eval(step[step.size()-1]*sd));

      if(verbose_) {
	if(val.size()>=2)
	  printf(" %e % .10f % e % e\n",step[step.size()-1],val[val.size()-1],val[val.size()-1]-val[0],val[val.size()-1]-val[val.size()-2]);
	else
	  printf(" %e % .10f\n",step[step.size()-1],val[val.size()-1]);
      }
      double dval=val[val.size()-1]-val[val.size()-2];

      // Check if converged
      if(val.size()>=2) {
	if(max && dval<0)
	  break;
	else if(!max && dval>0)
	  break;
      }
    }

    // Get optimal value
    arma::vec vals=arma::conv_to<arma::vec>::from(val);
    arma::uword iopt;
    if(max)
      vals.max(iopt);
    else
      vals.min(iopt);
    if(verbose_) printf("Line search changed value by %e\n",val[iopt]-val[0]);

    // Optimal value is
    double optstep=step[iopt];
    // Update x
    update(optstep*sd);
  }

  double fval=eval(x0);
  if(verbose_) printf("Final value is % .10f; optimization changed value by %e\n",fval,fval-ival);

  // Return the change
  return fval-ival;
}

PZStability::PZStability(SCF * solver, bool ver) : FDHessian(ver) {
  solverp_=solver;
  solverp_->set_verbose(false);

  real_=true;
  imag_=true;
  cancheck_=false;
  oocheck_=true;

  // Init sizes
  restr_=true;
  oa_=ob_=0;
  va_=vb_=0;
}

PZStability::~PZStability() {
}

size_t PZStability::count_oo_params(size_t o) const {
  size_t n=0;
  if(real_)
    n+=o*(o-1)/2;
  if(imag_)
    n+=o*(o-1)/2;

  return n;
}

size_t PZStability::count_ov_params(size_t o, size_t v) const {
  size_t n=0;
  // Real part
  if(real_)
    n+=o*v;
  // Complex part
  if(imag_)
    n+=o*v;

  return n;
}

size_t PZStability::count_params(size_t o, size_t v) const {
  size_t n=0;

  // Check canonicals?
  if(cancheck_) {
    n+=count_ov_params(o,v);
  }

  // Check oo block?
  if(oocheck_) {
    n+=count_oo_params(o);
  }

  return n;
}

size_t PZStability::count_params() const {
  size_t npar=count_params(oa_,va_);
  if(!restr_)
    npar+=count_params(ob_,vb_);

  return npar;
}

std::vector<pz_rot_par_t> PZStability::classify() const {
  std::vector<pz_rot_par_t> ret;
  if(restr_ || ob_==0) {
    pz_rot_par_t ooreal;
    ooreal.name="real OO";
    pz_rot_par_t ooimag;
    ooimag.name="imag OO";
    pz_rot_par_t oo;
    oo.name="OO";

    pz_rot_par_t ovreal;
    ovreal.name="real OV";
    pz_rot_par_t ovimag;
    ovimag.name="imag OV";
    pz_rot_par_t ov;
    ov.name="OV";

    pz_rot_par_t rreal;
    rreal.name="real OO+OV";
    pz_rot_par_t rimag;
    rimag.name="imag OO+OV";
    pz_rot_par_t rfull;
    rfull.name="OO+OV";

    size_t ioff=0;
    if(cancheck_) {
      if(real_) {
	arma::uword np=oa_*va_;
	arma::uvec i(arma::linspace<arma::uvec>(ioff,ioff+np-1,np));
	ovreal.idx=i;
	ret.push_back(ovreal);
	ioff+=np;
      }
      if(imag_) {
	arma::uword np=oa_*va_;
	arma::uvec i(arma::linspace<arma::uvec>(ioff,ioff+np-1,np));
	ovimag.idx=i;
	ret.push_back(ovimag);
	ioff+=np;
      }
      if(ovreal.idx.n_elem>0 && ovimag.idx.n_elem>0) {
	ov.idx.zeros(ovreal.idx.n_elem+ovimag.idx.n_elem);
	if(ovreal.idx.n_elem)
	  ov.idx.subvec(0,ovreal.idx.n_elem-1)=ovreal.idx;
	if(ovimag.idx.n_elem)
	  ov.idx.subvec(ovreal.idx.n_elem,ov.idx.n_elem-1)=ovimag.idx;
	ret.push_back(ov);
      }
    }
    if(oocheck_) {
      if(real_) {
	arma::uword np=oa_*(oa_-1)/2;
	arma::uvec i(arma::linspace<arma::uvec>(ioff,ioff+np-1,np));
	ooreal.idx=i;
	if(np)
	  ret.push_back(ooreal);
	ioff+=np;
      }
      if(imag_) {
	arma::uword np=oa_*(oa_-1)/2;
	arma::uvec i(arma::linspace<arma::uvec>(ioff,ioff+np-1,np));
	ooimag.idx=i;
	if(np)
	  ret.push_back(ooimag);
	ioff+=np;
      }
      if(ooreal.idx.n_elem>0 && ooimag.idx.n_elem>0) {
	oo.idx.zeros(ooreal.idx.n_elem+ooimag.idx.n_elem);
	if(ooreal.idx.n_elem)
	  oo.idx.subvec(0,ooreal.idx.n_elem-1)=ooreal.idx;
	if(ooimag.idx.n_elem)
	  oo.idx.subvec(ooreal.idx.n_elem,oo.idx.n_elem-1)=ooimag.idx;
	ret.push_back(oo);
      }
    }
    if(cancheck_ && oocheck_) {
      if(ooreal.idx.n_elem>0 && ovreal.idx.n_elem>0) {
	rreal.idx.zeros(ooreal.idx.n_elem+ovreal.idx.n_elem);
	if(ooreal.idx.n_elem)
	  rreal.idx.subvec(0,ooreal.idx.n_elem-1)=ooreal.idx;
	if(ovreal.idx.n_elem)
	  rreal.idx.subvec(ooreal.idx.n_elem,rreal.idx.n_elem-1)=ovreal.idx;
	ret.push_back(rreal);
      }

      if(ooimag.idx.n_elem>0 && ovimag.idx.n_elem>0) {
	rimag.idx.zeros(ooimag.idx.n_elem+ovimag.idx.n_elem);
	if(ooimag.idx.n_elem)
	  rimag.idx.subvec(0,ooimag.idx.n_elem-1)=ooimag.idx;
	if(ovimag.idx.n_elem)
	  rimag.idx.subvec(ooimag.idx.n_elem,rimag.idx.n_elem-1)=ovimag.idx;
	ret.push_back(rimag);
      }

      if(rreal.idx.n_elem>0 && rimag.idx.n_elem>0) {
	rfull.idx.zeros(rreal.idx.n_elem+rimag.idx.n_elem);
	if(rreal.idx.n_elem)
	  rfull.idx.subvec(0,rreal.idx.n_elem-1)=rreal.idx;
	if(rimag.idx.n_elem)
	  rfull.idx.subvec(rreal.idx.n_elem,rfull.idx.n_elem-1)=rimag.idx;
	ret.push_back(rfull);
      }
    }

  } else {
    pz_rot_par_t ooareal;
    ooareal.name="real alpha OO";
    pz_rot_par_t ooaimag;
    ooaimag.name="imag alpha OO";
    pz_rot_par_t ooa;
    ooa.name="alpha OO";

    pz_rot_par_t oobreal;
    oobreal.name="real beta  OO";
    pz_rot_par_t oobimag;
    oobimag.name="imag beta  OO";
    pz_rot_par_t oob;
    oob.name="beta  OO";

    pz_rot_par_t ooreal;
    ooreal.name="real OO";
    pz_rot_par_t ooimag;
    ooimag.name="imag OO";
    pz_rot_par_t oo;
    oo.name="OO";

    pz_rot_par_t ovareal;
    ovareal.name="real alpha OV";
    pz_rot_par_t ovaimag;
    ovaimag.name="imag alpha OV";
    pz_rot_par_t ova;
    ova.name="alpha OV";

    pz_rot_par_t ovbreal;
    ovbreal.name="real beta  OV";
    pz_rot_par_t ovbimag;
    ovbimag.name="imag beta  OV";
    pz_rot_par_t ovb;
    ovb.name="beta  OV";

    pz_rot_par_t ovreal;
    ovreal.name="real OV";
    pz_rot_par_t ovimag;
    ovimag.name="imag OV";
    pz_rot_par_t ov;
    ov.name="OV";

    pz_rot_par_t rareal;
    rareal.name="real alpha O+V";
    pz_rot_par_t raimag;
    raimag.name="imag alpha O+V";
    pz_rot_par_t rafull;
    rafull.name="alpha O+V";

    pz_rot_par_t rbreal;
    rbreal.name="real beta  O+V";
    pz_rot_par_t rbimag;
    rbimag.name="imag beta  O+V";
    pz_rot_par_t rbfull;
    rbfull.name="beta  O+V";

    pz_rot_par_t rreal;
    rreal.name="real O+V";
    pz_rot_par_t rimag;
    rimag.name="imag O+V";
    pz_rot_par_t rfull;
    rfull.name="O+V";

    size_t ioff=0;
    if(cancheck_) {
      if(real_) {
	arma::uword np=oa_*va_;
	ovareal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	if(np)
	  ret.push_back(ovareal);
	ioff+=np;
      }

      if(imag_) {
	arma::uword np=oa_*va_;
	ovaimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	ret.push_back(ovaimag);
	ioff+=np;
      }

      if(real_) {
	arma::uword np=ob_*vb_;
	ovbreal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	if(np)
	  ret.push_back(ovbreal);
	ioff+=np;
      }

      if(imag_) {
	arma::uword np=ob_*vb_;
	ovbimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	ret.push_back(ovbimag);
	ioff+=np;
      }

      if(real_) {
	ovreal.idx.zeros(ovareal.idx.n_elem+ovbreal.idx.n_elem);
	if(ovareal.idx.n_elem)
	  ovreal.idx.subvec(0,ovareal.idx.n_elem-1)=ovareal.idx;
	if(ovbreal.idx.n_elem)
	  ovreal.idx.subvec(ovareal.idx.n_elem,ovreal.idx.n_elem-1)=ovbreal.idx;
	ret.push_back(ovreal);
      }

      if(imag_) {
	ovimag.idx.zeros(ovaimag.idx.n_elem+ovbimag.idx.n_elem);
	if(ovaimag.idx.n_elem)
	  ovimag.idx.subvec(0,ovaimag.idx.n_elem-1)=ovaimag.idx;
	if(ovbimag.idx.n_elem)
	  ovimag.idx.subvec(ovaimag.idx.n_elem,ovimag.idx.n_elem-1)=ovbimag.idx;
	ret.push_back(ovimag);
      }

      if(real_ && imag_) {
	ova.idx.zeros(ovareal.idx.n_elem+ovaimag.idx.n_elem);
	if(ovareal.idx.n_elem)
	  ova.idx.subvec(0,ovareal.idx.n_elem-1)=ovareal.idx;
	if(ovaimag.idx.n_elem)
	  ova.idx.subvec(ovareal.idx.n_elem,ova.idx.n_elem-1)=ovaimag.idx;
	ret.push_back(ova);

	ovb.idx.zeros(ovbreal.idx.n_elem+ovbimag.idx.n_elem);
	if(ovbreal.idx.n_elem)
	  ovb.idx.subvec(0,ovbreal.idx.n_elem-1)=ovbreal.idx;
	if(ovbimag.idx.n_elem)
	  ovb.idx.subvec(ovbreal.idx.n_elem,ovb.idx.n_elem-1)=ovbimag.idx;
	ret.push_back(ovb);

	ov.idx.zeros(ova.idx.n_elem+ovb.idx.n_elem);
	if(ova.idx.n_elem)
	  ov.idx.subvec(0,ova.idx.n_elem-1)=ova.idx;
	if(ovb.idx.n_elem)
	  ov.idx.subvec(ova.idx.n_elem,ov.idx.n_elem-1)=ovb.idx;
	ret.push_back(ov);
      }
    }

    if(oocheck_) {
      if(real_) {
	arma::uword np=oa_*(oa_-1)/2;
	if(np>0) {
	  ooareal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(ooareal);
	}
	ioff+=np;
      }

      if(imag_) {
	arma::uword np=oa_*(oa_-1)/2;
	if(np>0) {
	  ooaimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(ooaimag);
	}
	ioff+=np;
      }

      if(real_) {
	arma::uword np=ob_*(ob_-1)/2;
	if(np>0) {
	  oobreal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(oobreal);
	}
	ioff+=np;
      }

      if(imag_) {
	arma::uword np=ob_*(ob_-1)/2;
	if(np>0) {
	  oobimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(oobimag);
	}
	ioff+=np;
      }

      if(real_ && oa_>1 && ob_>1) {
	ooreal.idx.zeros(ooareal.idx.n_elem+oobreal.idx.n_elem);
	if(ooareal.idx.n_elem)
	  ooreal.idx.subvec(0,ooareal.idx.n_elem-1)=ooareal.idx;
	if(oobreal.idx.n_elem)
	  ooreal.idx.subvec(ooareal.idx.n_elem,ooreal.idx.n_elem-1)=oobreal.idx;
	ret.push_back(ooreal);
      }

      if(imag_ && oa_>1 && ob_>1) {
	ooimag.idx.zeros(ooaimag.idx.n_elem+oobimag.idx.n_elem);
	if(ooaimag.idx.n_elem)
	  ooimag.idx.subvec(0,ooaimag.idx.n_elem-1)=ooaimag.idx;
	if(oobimag.idx.n_elem)
	  ooimag.idx.subvec(ooaimag.idx.n_elem,ooimag.idx.n_elem-1)=oobimag.idx;
	ret.push_back(ooimag);
      }

      if(real_ && imag_) {
	ooa.idx.zeros(ooareal.idx.n_elem+ooaimag.idx.n_elem);
	if(ooareal.idx.n_elem)
	  ooa.idx.subvec(0,ooareal.idx.n_elem-1)=ooareal.idx;
	if(ooaimag.idx.n_elem)
	  ooa.idx.subvec(ooareal.idx.n_elem,ooa.idx.n_elem-1)=ooaimag.idx;
	ret.push_back(ooa);

	if(ob_>1) {
	  oob.idx.zeros(oobreal.idx.n_elem+oobimag.idx.n_elem);
	  if(oobreal.idx.n_elem)
	    oob.idx.subvec(0,oobreal.idx.n_elem-1)=oobreal.idx;
	  if(oobimag.idx.n_elem)
	    oob.idx.subvec(oobreal.idx.n_elem,oob.idx.n_elem-1)=oobimag.idx;
	  ret.push_back(oob);
	}

	oo.idx.zeros(ooa.idx.n_elem+oob.idx.n_elem);
	oo.idx.subvec(0,ooa.idx.n_elem-1)=ooa.idx;
	if(ob_>1) {
	  oo.idx.subvec(ooa.idx.n_elem,oo.idx.n_elem-1)=oob.idx;
	  ret.push_back(oo);
	}
      }
    }
    if(cancheck_ && oocheck_) {
      rareal.idx.zeros(ooareal.idx.n_elem+ovareal.idx.n_elem);
      if(ooareal.idx.n_elem)
	rareal.idx.subvec(0,ooareal.idx.n_elem-1)=ooareal.idx;
      if(ovareal.idx.n_elem)
	rareal.idx.subvec(ooareal.idx.n_elem,rareal.idx.n_elem-1)=ovareal.idx;
      if(real_ && imag_)
	ret.push_back(rareal);

      rbreal.idx.zeros(oobreal.idx.n_elem+ovbreal.idx.n_elem);
      if(oobreal.idx.n_elem)
	rbreal.idx.subvec(0,oobreal.idx.n_elem-1)=oobreal.idx;
      if(ovbreal.idx.n_elem)
	rbreal.idx.subvec(oobreal.idx.n_elem,rbreal.idx.n_elem-1)=ovbreal.idx;
      if(real_ && imag_)
	ret.push_back(rbreal);

      rreal.idx.zeros(rareal.idx.n_elem+rbreal.idx.n_elem);
      if(rareal.idx.n_elem)
	rreal.idx.subvec(0,rareal.idx.n_elem-1)=rareal.idx;
      if(rbreal.idx.n_elem)
	rreal.idx.subvec(rareal.idx.n_elem,rreal.idx.n_elem-1)=rbreal.idx;
      if(real_ && imag_)
	ret.push_back(rreal);

      raimag.idx.zeros(ooaimag.idx.n_elem+ovaimag.idx.n_elem);
      if(ooaimag.idx.n_elem)
	raimag.idx.subvec(0,ooaimag.idx.n_elem-1)=ooaimag.idx;
      if(ovaimag.idx.n_elem)
	raimag.idx.subvec(ooaimag.idx.n_elem,raimag.idx.n_elem-1)=ovaimag.idx;
      if(imag_ && imag_)
	ret.push_back(raimag);

      rbimag.idx.zeros(oobimag.idx.n_elem+ovbimag.idx.n_elem);
      if(oobimag.idx.n_elem)
	rbimag.idx.subvec(0,oobimag.idx.n_elem-1)=oobimag.idx;
      if(ovbimag.idx.n_elem)
	rbimag.idx.subvec(oobimag.idx.n_elem,rbimag.idx.n_elem-1)=ovbimag.idx;
      if(real_ && imag_)
	ret.push_back(rbimag);

      rimag.idx.zeros(raimag.idx.n_elem+rbimag.idx.n_elem);
      if(raimag.idx.n_elem)
	rimag.idx.subvec(0,raimag.idx.n_elem-1)=raimag.idx;
      if(rbimag.idx.n_elem)
	rimag.idx.subvec(raimag.idx.n_elem,rimag.idx.n_elem-1)=rbimag.idx;
      if(real_ && imag_)
	ret.push_back(rimag);

      rfull.idx.zeros(rreal.idx.n_elem+rimag.idx.n_elem);
      if(rreal.idx.n_elem)
	rfull.idx.subvec(0,rreal.idx.n_elem-1)=rreal.idx;
      if(rimag.idx.n_elem)
	rfull.idx.subvec(rreal.idx.n_elem,rfull.idx.n_elem-1)=rimag.idx;
      ret.push_back(rfull);
    }
  }

  return ret;
}

arma::cx_mat PZStability::unified_H(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::vec & worb, const arma::cx_mat & H0) const {
  // Build effective Fock operator
  arma::cx_mat H(H0*COMPLEX1);

  if(pzw_!=0.0) {
    arma::mat S(solverp_->get_S());
    for(size_t io=0;io<CO.n_cols;io++) {
      arma::cx_mat Porb(form_density(CO.col(io)));
      H-=worb(io)*S*Porb*Forb[io]*Porb*S;
    }

    if(CV.n_cols) {
      // Virtual space density matrix
      arma::cx_mat v(CV.n_rows,CV.n_rows);
      v.zeros();
      for(size_t io=0;io<CV.n_cols;io++)
	v+=form_density(CV.col(io));

      for(size_t io=0;io<CO.n_cols;io++) {
	arma::cx_mat Porb(form_density(CO.col(io)));
	H-=worb(io)*S*(v*Forb[io]*Porb + Porb*Forb[io]*v)*S;
      }
    }
  }

  return H;
}

arma::mat PZStability::centroids(const arma::cx_mat & CO) const {
  // Get moment matrix
  std::vector<arma::mat> mommat=basis_.moment(1);

  arma::mat cen(mommat.size(),CO.n_cols);
  for(size_t io=0;io<CO.n_cols;io++)
    for(size_t ic=0;ic<mommat.size();ic++)
      cen(ic,io)=arma::as_scalar(arma::real(CO.col(io).t()*mommat[ic]*CO.col(io)));

  return cen;
}

void PZStability::print_info(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0, const arma::vec & Eorb, const arma::vec & worb) {
  if(!verbose_) return;

  // Form unified Hamiltonian
  arma::cx_mat H(unified_H(CO,CV,Forb,worb,H0));

  // Occupied block
  bool diagok;

  arma::vec Eo;
  arma::cx_mat Co;
  arma::cx_mat Hoo;
  if(CO.n_cols) {
    Hoo=arma::trans(CO)*H*CO;
    diagok=arma::eig_sym(Eo,Co,Hoo);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Error diagonalizing H in occupied space.\n");
    }
  }

  arma::vec Ev;
  arma::cx_mat Cv;
  if(CV.n_cols) {
    arma::cx_mat Hvv(arma::trans(CV)*H*CV);
    diagok=arma::eig_sym(Ev,Cv,Hvv);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Error diagonalizing H in virtual space.\n");
    }
  }

  // Whole set of orbital energies
  arma::vec Efull(CO.n_cols+CV.n_cols);
  if(CO.n_cols)
    Efull.subvec(0,CO.n_cols-1)=Eo;
  if(CV.n_cols)
    Efull.subvec(CO.n_cols,CO.n_cols+CV.n_cols-1)=Ev;

  // Print out
  std::vector<double> occs(CO.n_cols,1.0);
  print_E(Efull,occs,false);

  if(pzw_!=0.0) {
    // Collect projected energies
    arma::vec Ep(CO.n_cols);
    for(size_t io=0;io<CO.n_cols;io++)
      Ep(io)=std::real(Hoo(io,io));

    // Print out optimal orbitals
    if(CO.n_cols) {
      printf("Decomposition of self-interaction energies:\n");
      printf("\t%4s\t%8s\t%8s\t%8s\n","io","E(orb)","E(SI)","Scaling");
      for(size_t io=0;io<CO.n_cols;io++)
	printf("\t%4i\t% 8.3f\t% 8.6f\t% 8.6f\n",(int) io+1,Ep(io),Eorb(io),worb(io));
      fflush(stdout);
    }
  }

  printf("Orbital centroids:\n");
  arma::mat cen(centroids(CO));
  // Convert to angstrom
  cen*=BOHRINANGSTROM;
  for(size_t io=0;io<cen.n_cols;io++)
    printf("%3i % .6f % .6f % .6f\n",(int) io+1, cen(0,io), cen(1,io), cen(2,io));
}

void PZStability::print_info() {
  if(!verbose_) return;

  arma::vec x(count_params());
  x.zeros();

  rscf_t rsl;
  uscf_t usl;

  if(restr_) {
    // Evaluate orbital matrices
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb, worb;
    eval(x,rsl,Forb,Eorb,worb,true,true,true);

    // Occupied orbitals
    arma::cx_mat CO=make_CO(rsl);
    arma::cx_mat CV=make_CV(rsl);

    // Diagonalize
    print_info(CO,CV,Forb,make_H(rsl),Eorb,worb);

    // Density matrix
    arma::mat P(arma::real(2.0*form_density(CO)));
    arma::vec dipmom(dipole_moment(P,basis_));
    printf("Dipole mu = (% 08.8f, % 08.8f, % 08.8f) D\n",dipmom(0)/AUINDEBYE,dipmom(1)/AUINDEBYE,dipmom(2)/AUINDEBYE);

  } else {
    // Evaluate orbital matrices
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    arma::vec worba, worbb;
    eval(x,usl,Forba,Eorba,worba,Forbb,Eorbb,worbb,true,true,true);

    // Occupied orbitals
    arma::cx_mat COa(make_CO(false,usl));
    arma::cx_mat COb(make_CO(true,usl));
    // Virtuals
    arma::cx_mat CVa(make_CV(false,usl));
    arma::cx_mat CVb(make_CV(true,usl));

    // Diagonalize
    printf("\n **** Alpha orbitals ****\n");
    print_info(COa,CVa,Forba,make_H(usl,false),Eorba,worba);
    printf("\n **** Beta  orbitals ****\n");
    print_info(COb,CVb,Forbb,make_H(usl,true),Eorbb,worbb);

    // Density matrix
    arma::mat P(arma::real(form_density(COa)));
    if(COb.n_cols)
      P += arma::real(form_density(COb));
    arma::vec dipmom(dipole_moment(P,basis_));
    printf("Dipole mu = (% 08.8f, % 08.8f, % 08.8f) D\n",dipmom(0)/AUINDEBYE,dipmom(1)/AUINDEBYE,dipmom(2)/AUINDEBYE);
  }

  // Print total energy and its components
  energy_t en = restr_ ? rsl.en : usl.en;
  printf("\n");
  printf("%-21s energy: % .16e\n","Kinetic",en.Ekin);
  printf("%-21s energy: % .16e\n","Nuclear attraction",en.Enuca);
  printf("%-21s energy: % .16e\n","Total one-electron",en.Eone);
  printf("%-21s energy: % .16e\n","Nuclear repulsion",en.Enucr);
  printf("%-21s energy: % .16e\n","Coulomb",en.Ecoul);
  printf("%-21s energy: % .16e\n","Exchange-correlation",en.Exc);
  printf("%-21s energy: % .16e\n","Non-local correlation",en.Enl);
  printf("%-21s energy: % .16e\n","SI correction",en.Esic);
  printf("-----------------------------------------------------\n");
  printf("%28s: % .16e\n","Total energy",en.E);
  printf("%28s: % .16e\n","Virial factor",-en.E/en.Ekin);
}

void PZStability::perturb(double h) {
  // Form update vector
  arma::vec x(count_params());
  x.randn();
  update(h*x);
}

void PZStability::update_step(const arma::vec & g) {
  // Collect derivatives
  if(restr_ || ob_==0) {
    arma::cx_mat G=rotation_pars(g,false);
    if(oocheck_ && !cancheck_)
      // Only doing OO block, so we can take the first subblock
      G=G.submat(0,0,oa_-1,oa_-1);

    // Calculate eigendecomposition
    arma::vec Gval;
    arma::cx_mat Gvec;
    bool diagok=arma::eig_sym(Gval,Gvec,-COMPLEXI*G);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Error diagonalizing G.\n");
    }

    // Calculate maximum step size; cost function is 4th order in parameters
    Tmu_=0.5*M_PI/arma::max(arma::abs(Gval));

  } else {
    arma::cx_mat Ga=rotation_pars(g,false);
    arma::cx_mat Gb=rotation_pars(g,true);
    if(oocheck_ && !cancheck_) {
      // Only doing OO block, so we can take the OO subblocks
      Ga=Ga.submat(0,0,oa_-1,oa_-1);
      Gb=Gb.submat(0,0,ob_-1,ob_-1);
    }

    // Calculate eigendecompositions
    arma::vec Gaval, Gbval;
    arma::cx_mat Gavec, Gbvec;
    bool diagok=arma::eig_sym(Gaval,Gavec,-COMPLEXI*Ga);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Error diagonalizing Ga.\n");
    }
    diagok=arma::eig_sym(Gbval,Gbvec,-COMPLEXI*Gb);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Error diagonalizing Gb.\n");
    }

    // Calculate maximum step size; cost function is 4th order in parameters
    Tmu_=0.5*M_PI/std::max(arma::max(arma::abs(Gaval)),arma::max(arma::abs(Gbval)));
  }
}


arma::vec PZStability::compute_worb(const arma::cx_mat & C) {
  arma::vec w(C.n_cols);
  w.ones();

  switch(scale_) {
  case(PZ_SCALE_CONSTANT):
    w*=pzw_;
    break;

  case(PZ_SCALE_DENSITY):
    {
      for(size_t io=0;io<C.n_cols;io++) {
	arma::mat S(grid_.eval_overlap(C,io,scaleexp_));
	w(io)=std::real(arma::as_scalar(arma::trans(C.col(io))*S*C.col(io)));
      }
      break;
    }

  case(PZ_SCALE_KINETIC):
    {
      arma::mat S(grid_.eval_tau_overlap(C,scaleexp_));
      for(size_t io=0;io<C.n_cols;io++)
	w(io)=std::real(arma::as_scalar(arma::trans(C.col(io))*S*C.col(io)));
      break;
    }

  default:
    throw std::logic_error("Not implemented\n");
  }

  //w.t().print("Orbital weights");

  return w;
}

void PZStability::scaling_gradient_oo(arma::cx_mat & gOO, const arma::cx_mat & CO, const arma::vec & Eorb) {
  switch(scale_) {
  case(PZ_SCALE_CONSTANT):
    return;

  case(PZ_SCALE_DENSITY):
    {
      // Calculate the overlap matrices
      std::vector<arma::mat> S(CO.n_cols);
      for(size_t io=0;io<CO.n_cols;io++)
	S[io]=grid_.eval_overlap(CO,io,scaleexp_);

      // Increment gOO
      for(size_t m=0;m<CO.n_cols;m++)
	for(size_t n=0;n<CO.n_cols;n++)
	  gOO(m,n) += (scaleexp_+1)*arma::as_scalar(arma::trans(CO.col(n))*(Eorb(m)*S[m]-Eorb(n)*S[n])*CO.col(m));

      return;
    }

  case(PZ_SCALE_KINETIC):
    {
      // Calculate the overlap
      arma::mat S(grid_.eval_tau_overlap(CO,scaleexp_));

      // Increment gOO
      for(size_t m=0;m<CO.n_cols;m++)
	for(size_t n=0;n<CO.n_cols;n++)
	  gOO(m,n) += (Eorb(m)-Eorb(n))*arma::as_scalar(arma::trans(CO.col(n))*S*CO.col(m));

      return;
    }

  default:
    throw std::logic_error("Not implemented\n");
  }
}

void PZStability::scaling_gradient_ov(arma::cx_mat & gOV, const arma::cx_mat & CO, const arma::vec & Eorb, const arma::cx_mat & CV) {
  switch(scale_) {
  case(PZ_SCALE_CONSTANT):
    return;

  case(PZ_SCALE_DENSITY):
    {
      // First part
      {
	// Calculate the overlap matrices
	std::vector<arma::mat> S(CO.n_cols);
	for(size_t io=0;io<CO.n_cols;io++)
	  S[io]=grid_.eval_overlap(CO,io,scaleexp_);

	// Increment gOV
	for(size_t m=0;m<CO.n_cols;m++)
	  for(size_t a=0;a<CV.n_cols;a++)
	    gOV(m,a) += (scaleexp_+1)*Eorb(m)*arma::as_scalar(arma::trans(CV.col(a))*S[m]*CO.col(m));
      }

      // Second part
      {
	// Calculate the weighted overlap
	arma::mat S(grid_.eval_overlap(CO,Eorb,scaleexp_+1));

	// Increment gOV
	for(size_t m=0;m<CO.n_cols;m++)
	  for(size_t a=0;a<CV.n_cols;a++)
	    gOV(m,a) -= scaleexp_*arma::as_scalar(arma::trans(CV.col(a))*S*CO.col(m));
      }

      return;
    }

  case(PZ_SCALE_KINETIC):
    {
      // First part
      {
	// Calculate the overlap matrix
	arma::mat S(grid_.eval_tau_overlap(CO,scaleexp_));

	// Increment gOV
	for(size_t m=0;m<CO.n_cols;m++)
	  for(size_t a=0;a<CV.n_cols;a++)
	    gOV(m,a) += Eorb(m)*arma::as_scalar(arma::trans(CV.col(a))*S*CO.col(m));
      }

      // Second part
      {
	// Calculate the weighted overlap
	arma::mat S(grid_.eval_tau_overlap_deriv(CO,Eorb,scaleexp_));

	// Increment gOV
	for(size_t m=0;m<CO.n_cols;m++)
	  for(size_t a=0;a<CV.n_cols;a++)
	    gOV(m,a) += arma::as_scalar(arma::trans(CV.col(a))*S*CO.col(m));
      }

      return;
    }

  default:
    throw std::logic_error("Not implemented\n");
  }
}

double PZStability::eval(const arma::vec & x, rscf_t & sol, std::vector<arma::cx_mat> & Forb, arma::vec & Eorb, arma::vec & worb, bool can, bool fock, bool useref) {
  // Use reference
  sol=rsol_;

  // List of changed orbitals
  std::vector<size_t> occlist, virtlist;
  if(arma::norm(x,2)!=0.0) {
    arma::cx_mat R(rotation(x,false));
    sol.cC=sol.cC*R;

    if(useref) {
      // Remove unity
      R-=arma::eye<arma::cx_mat>(sol.cC.n_cols,sol.cC.n_cols);
      // Find orbitals that have changed
      for(size_t ia=0;ia<oa_;ia++)
	for(size_t ja=ia+1;ja<R.n_cols;ja++)
	  if(std::norm(R(ia,ja))>=CHANGETHR) {
	    occlist.push_back(ia);
	    if(ja<oa_)
	      occlist.push_back(ja);
	    else
	      virtlist.push_back(ja);
	  }
    }
  }

  // Update density matrix
  arma::cx_mat P(2.0*form_density(sol.cC.cols(0,oa_-1)));
  // Debug
  sol.P=arma::real(P);
  sol.P_im=arma::imag(P);

  // Clear out any old data
  Forb.clear();
  Eorb.clear();
  worb.clear();

  // Dummy occupation vector
  std::vector<double> occs(oa_,2.0);

  // Build global Fock operator
  if(can && (!useref || (useref && virtlist.size()) ))
    solverp_->Fock_RDFT(sol,occs,ovmethod_,grid_,nlgrid_);

  if(pzw_==0.0)
    return sol.en.E;

  // Build the SI part
  arma::cx_mat CO;
  if(useref) {
    CO.zeros(sol.cC.n_rows,occlist.size());
    for(size_t i=0;i<occlist.size();i++)
      CO.col(i)=sol.cC.col(occlist[i]);

    std::vector<arma::cx_mat> Forb_hlp;
    arma::vec Eorb_hlp, worb_hlp;
    solverp_->PZSIC_Fock(Forb_hlp,Eorb_hlp,CO,oomethod_,grid_,nlgrid_,fock);
    Eorb=ref_Eorb_;
    for(size_t i=0;i<occlist.size();i++)
      Eorb(occlist[i])=Eorb_hlp(i);

    worb=ref_worb_;
    worb_hlp=compute_worb(CO);
    for(size_t i=0;i<occlist.size();i++)
      worb(occlist[i])=worb_hlp(i);

    if(fock) {
      Forb=ref_Forb_;
      for(size_t i=0;i<occlist.size();i++)
	Forb[occlist[i]]=Forb_hlp[i];
    }
  } else {
    CO=sol.cC.cols(0,oa_-1);
    worb=compute_worb(CO);
    solverp_->PZSIC_Fock(Forb,Eorb,CO,oomethod_,grid_,nlgrid_,fock);
  }

  sol.en.Esic=-2.0*arma::sum(worb%Eorb);
  sol.en.Eel=sol.en.Ecoul+sol.en.Exc+sol.en.Eone+sol.en.Enl+sol.en.Esic;
  sol.en.E=sol.en.Eel+sol.en.Enucr;

  return sol.en.E;
}

double PZStability::eval(const arma::vec & x, uscf_t & sol, std::vector<arma::cx_mat> & Forba, arma::vec & Eorba, arma::vec & worba, std::vector<arma::cx_mat> & Forbb, arma::vec & Eorbb, arma::vec & worbb, bool can, bool fock, bool useref) {
  // Use reference
  sol=usol_;

  // List of changed orbitals
  std::vector<size_t> occlista, occlistb, virtlista, virtlistb;
  if(arma::norm(x,2)!=0.0) {
    arma::cx_mat Ra(rotation(x,false));
    sol.cCa=sol.cCa*Ra;

    if(useref) {
      Ra-=arma::eye<arma::cx_mat>(sol.cCa.n_cols,sol.cCa.n_cols);
      for(size_t ia=0;ia<oa_;ia++)
	for(size_t ja=ia+1;ja<Ra.n_cols;ja++)
	  if(std::norm(Ra(ia,ja))>=CHANGETHR) {
	    occlista.push_back(ia);
	    if(ja<oa_)
	      occlista.push_back(ja);
	    else
	      virtlista.push_back(ja);
	  }
    }

    if(ob_) {
      arma::cx_mat Rb(rotation(x,true));
      sol.cCb=sol.cCb*Rb;
      if(useref) {
	Rb-=arma::eye<arma::cx_mat>(sol.cCb.n_cols,sol.cCb.n_cols);
	for(size_t ib=0;ib<ob_;ib++)
	  for(size_t jb=ib+1;jb<Rb.n_cols;jb++)
	    if(std::norm(Rb(ib,jb))>=CHANGETHR) {
	      occlistb.push_back(ib);
	      if(jb<ob_)
		occlistb.push_back(jb);
	      else
		virtlistb.push_back(jb);
	    }
      }
    }
  }

  // Update density matrix
  {
    arma::cx_mat Pa(form_density(sol.cCa.cols(0,oa_-1)));
    sol.Pa=arma::real(Pa);
    sol.Pa_im=arma::imag(Pa);
  }
  if(ob_) {
    arma::cx_mat Pb(form_density(sol.cCb.cols(0,ob_-1)));
    sol.Pb=arma::real(Pb);
    sol.Pb_im=arma::imag(Pb);
  } else {
    sol.Pb.zeros(sol.cCb.n_rows,sol.cCb.n_rows);
    sol.Pb_im.zeros(sol.cCb.n_rows,sol.cCb.n_rows);
  }
  sol.P=sol.Pa+sol.Pb;

  // Clear out any old data
  Forba.clear();
  Eorba.clear();
  worba.clear();
  Forbb.clear();
  Eorbb.clear();
  worbb.clear();

  // Dummy occupation vector
  std::vector<double> occa(oa_,1.0);
  std::vector<double> occb(ob_,1.0);

  // Build global Fock operator
  if(can && (!useref || (useref && (virtlista.size() || virtlistb.size())) ))
    solverp_->Fock_UDFT(sol,occa,occb,ovmethod_,grid_,nlgrid_);
  if(pzw_==0.0)
    return sol.en.E;

  // Build the SI part
  std::vector<arma::cx_mat> Forb;
  arma::vec Eorb;
  arma::vec worba_hlp, worbb_hlp;

  // Build the SI part
  arma::cx_mat CO;
  if(useref) {
    CO.zeros(sol.cCa.n_rows,occlista.size()+occlistb.size());
    for(size_t i=0;i<occlista.size();i++)
      CO.col(i)=sol.cCa.col(occlista[i]);
    for(size_t i=0;i<occlistb.size();i++)
      CO.col(i+occlista.size())=sol.cCb.col(occlistb[i]);

    solverp_->PZSIC_Fock(Forb,Eorb,CO,oomethod_,grid_,nlgrid_,fock);
    if(occlista.size())
      worba_hlp=compute_worb(CO.cols(0,occlista.size()-1));
    if(occlistb.size())
      worbb_hlp=compute_worb(CO.cols(occlista.size(),CO.n_cols-1));

    Eorba=ref_Eorba_;
    for(size_t i=0;i<occlista.size();i++)
      Eorba(occlista[i])=Eorb(i);
    Eorbb=ref_Eorbb_;
    for(size_t i=0;i<occlistb.size();i++)
      Eorbb(occlistb[i])=Eorb(i+occlista.size());

    worba=ref_worba_;
    for(size_t i=0;i<occlista.size();i++)
      worba(occlista[i])=worba_hlp(i);
    worbb=ref_worbb_;
    for(size_t i=0;i<occlistb.size();i++)
      worbb(occlistb[i])=worbb_hlp(i);

    if(fock) {
      Forba=ref_Forba_;
      for(size_t i=0;i<occlista.size();i++)
	Forba[occlista[i]]=Forb[i];
      Forbb=ref_Forbb_;
      for(size_t i=0;i<occlistb.size();i++)
	Forbb[occlistb[i]]=Forb[i+occlista.size()];
    }
  } else {
    CO.zeros(sol.cCa.n_rows,oa_+ob_);
    CO.cols(0,oa_-1)=sol.cCa.cols(0,oa_-1);
    if(ob_)
      CO.cols(oa_,oa_+ob_-1)=sol.cCb.cols(0,ob_-1);
    solverp_->PZSIC_Fock(Forb,Eorb,CO,oomethod_,grid_,nlgrid_,fock);

    Eorba=Eorb.subvec(0,oa_-1);
    if(ob_)
      Eorbb=Eorb.subvec(oa_,oa_+ob_-1);

    worba=compute_worb(sol.cCa.cols(0,oa_-1));
    if(ob_)
      worbb=compute_worb(sol.cCb.cols(0,ob_-1));

    if(fock) {
      Forba.resize(oa_);
      for(size_t i=0;i<oa_;i++)
	Forba[i]=Forb[i];
      if(ob_) {
	Forbb.resize(ob_);
	for(size_t i=0;i<ob_;i++)
	  Forbb[i]=Forb[i+oa_];
      }
    }
  }

  // Result is
  sol.en.Esic=-(arma::sum(worba%Eorba)+arma::sum(worbb%Eorbb));
  sol.en.Eel=sol.en.Ecoul+sol.en.Exc+sol.en.Eone+sol.en.Enl+sol.en.Esic;
  sol.en.E=sol.en.Eel+sol.en.Enucr;

  return sol.en.E;
}

arma::vec PZStability::gradient() {
  arma::vec x;
  x.zeros(count_params());
  return gradient(x, true);
}

static arma::mat precondition_matrix(const arma::mat & Ediff, double dH) {
  // Demand that all scalings are within this range
  double min=1e-6;
  double max=1/min;

  arma::mat ret(Ediff.n_rows,Ediff.n_cols);
  for(size_t io=0;io<ret.n_rows;io++)
    for(size_t iv=0;iv<ret.n_cols;iv++) {
      ret(io,iv)=1.0/(Ediff(io,iv)+dH);
      if(ret(io,iv)<min) ret(io,iv)=min;
      else if(ret(io,iv)>max) ret(io,iv)=max;
    }
  return ret;
}

static arma::mat precondition_matrix(const arma::vec & Eo, const arma::vec & Ev, double dH) {
  arma::mat Ediff(Eo.n_elem,Ev.n_elem);
  for(size_t io=0;io<Eo.n_elem;io++)
    for(size_t iv=0;iv<Ev.n_elem;iv++)
      Ediff(io,iv)=Ev(iv)-Eo(io);
  return precondition_matrix(Ediff,dH);
}


arma::cx_mat PZStability::make_CO(const rscf_t & sol) const {
  if(!restr_)
    throw std::logic_error("Called get_CO() using unrestricted orbitals!\n");

  return sol.cC.cols(0,oa_-1);
}

arma::cx_mat PZStability::make_CO() const {
  return make_CO(rsol_);
}

arma::cx_mat PZStability::make_CO(bool spin, const uscf_t & sol) const {
  if(restr_)
    throw std::logic_error("Called get_CO(spin) using restricted orbitals!\n");

  arma::cx_mat C;
  if(spin && ob_>0)
    C=sol.cCb.cols(0,ob_-1);
  else if(!spin)
    C=sol.cCa.cols(0,oa_-1);

  return C;
}

arma::cx_mat PZStability::make_CO(bool spin) const {
  return make_CO(spin,usol_);
}

arma::cx_mat PZStability::make_CV(const rscf_t & sol) const {
  if(!restr_)
    throw std::logic_error("Called get_CV() using unrestricted orbitals!\n");

  arma::cx_mat CV;
  if(sol.cC.n_cols>oa_)
    CV=sol.cC.cols(oa_,rsol_.cC.n_cols-1);
  return CV;
}

arma::cx_mat PZStability::make_CV() const {
  return make_CV(rsol_);
}

arma::cx_mat PZStability::make_CV(bool spin, const uscf_t & sol) const {
  if(restr_)
    throw std::logic_error("Called get_CV(spin) using restricted orbitals!\n");

  size_t No = spin ? ob_ : oa_;
  const arma::cx_mat & C = spin ? sol.cCb : sol.cCa;

  arma::cx_mat CV;
  if(C.n_cols > No)
    CV=C.cols(No,C.n_cols-1);

  return CV;
}

arma::cx_mat PZStability::make_CV(bool spin) const {
  return make_CV(spin,usol_);
}

arma::vec PZStability::precondition_unified(const arma::vec & g) const {
  // Search direction
  arma::vec sd(g);

  // Offset
  size_t ioff=0;

  if(restr_) {
    // Occupied orbitals
    arma::cx_mat CO(make_CO());
    // Virtual orbitals
    arma::cx_mat CV(make_CV());

    if(cancheck_ && va_) {
      // Form OV gradient
      arma::cx_mat gOV(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa_,va_)-1),oa_,va_,real_,imag_));
      // Check
      arma::vec gs(g.subvec(ioff,ioff+count_ov_params(oa_,va_)-1));
      arma::vec gt(gather_ov(gOV,real_,imag_));

      // Preconditioning. Form unified Hamiltonian
      arma::cx_mat H(unified_H(CO,CV,ref_Forb_,ref_worb_,make_H(rsol_)));

      arma::cx_mat Hoo(arma::trans(CO)*H*CO);
      arma::cx_mat Hvv(arma::trans(CV)*H*CV);

      arma::vec Eo;
      arma::cx_mat Co;
      eig_sym_ordered(Eo,Co,Hoo);

      arma::vec Ev;
      arma::cx_mat Cv;
      eig_sym_ordered(Ev,Cv,Hvv);

      // Minimum Hessian shift is
      double dH=std::max(arma::max(Eo)-arma::min(Ev),1e-4);

      // Transform OV gradient into pseudocanonical space
      arma::cx_mat GOV(arma::trans(Co)*gOV*Cv);
      // and perform preconditioning
      GOV=GOV%precondition_matrix(Eo,Ev,dH);

      // Transform back into the original frame
      GOV=Co*GOV*arma::trans(Cv);

      arma::vec POV(gather_ov(GOV,real_,imag_));
      if(POV.n_elem != count_ov_params(oa_,va_))
	throw std::logic_error("Amount of elements doesn't match!\n");
      sd.subvec(ioff,ioff+POV.n_elem-1)=POV;
      ioff+=POV.n_elem;
    }

  } else {
    arma::cx_mat COa(make_CO(false));
    arma::cx_mat COb(make_CO(true));
    arma::cx_mat CVa(make_CV(false));
    arma::cx_mat CVb(make_CV(true));

    if(cancheck_ && va_) {
      // Preconditioning. Form unified Hamiltonian
      arma::cx_mat Ha(unified_H(COa,CVa,ref_Forba_,ref_worba_,make_H(usol_,false)));
      arma::cx_mat Hb(unified_H(COb,CVb,ref_Forbb_,ref_worbb_,make_H(usol_,true)));

      arma::cx_mat Hooa(arma::trans(COa)*Ha*COa);
      arma::cx_mat Hvva(arma::trans(CVa)*Ha*CVa);
      arma::cx_mat Hoob;
      if(ob_)
	Hoob=arma::trans(COb)*Hb*COb;
      arma::cx_mat Hvvb(arma::trans(CVb)*Hb*CVb);

      arma::vec Eoa;
      arma::cx_mat Coa;
      eig_sym_ordered(Eoa,Coa,Hooa);

      arma::vec Eob;
      arma::cx_mat Cob;
      if(ob_)
	eig_sym_ordered(Eob,Cob,Hoob);

      arma::vec Eva;
      arma::cx_mat Cva;
      eig_sym_ordered(Eva,Cva,Hvva);

      arma::vec Evb;
      arma::cx_mat Cvb;
      eig_sym_ordered(Evb,Cvb,Hvvb);

      // Minimum Hessian shift is
      double dH=std::max(arma::max(Eoa)-arma::min(Eva),1e-4);
      if(ob_)
	dH=std::max(arma::max(Eob)-arma::min(Evb),dH);

      // Transform OV gradient into pseudocanonical space and perform preconditioning
      arma::cx_mat gOVa(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa_,va_)-1),oa_,va_,real_,imag_));
      arma::cx_mat GOVa(arma::trans(Coa)*gOVa*Cva);
      GOVa=GOVa%precondition_matrix(Eoa,Eva,dH);
      // Transform back into the original frame
      GOVa=Coa*GOVa*arma::trans(Cva);

      arma::vec POVa(gather_ov(GOVa,real_,imag_));
      sd.subvec(ioff,ioff+POVa.n_elem-1)=POVa;
      ioff+=POVa.n_elem;
      if(POVa.n_elem != count_ov_params(oa_,va_))
	throw std::logic_error("Amount of elements doesn't match!\n");

      if(ob_) {
	arma::cx_mat gOVb(spread_ov(g.subvec(ioff,ioff+count_ov_params(ob_,vb_)-1),ob_,vb_,real_,imag_));
	arma::cx_mat GOVb(arma::trans(Cob)*gOVb*Cvb);
	GOVb=GOVb%precondition_matrix(Eob,Evb,dH);
	// Transform back into the original frame
	GOVb=Cob*GOVb*arma::trans(Cvb);

	arma::vec POVb(gather_ov(GOVb,real_,imag_));
	sd.subvec(ioff,ioff+POVb.n_elem-1)=POVb;
	ioff+=POVb.n_elem;
	if(POVb.n_elem != count_ov_params(ob_,vb_))
	  throw std::logic_error("Amount of elements doesn't match!\n");
      }
    }
  }

  return sd;
}

arma::vec PZStability::precondition_orbital(const arma::vec & g) const {
  // Search direction
  arma::vec sd(g);

  // Offset
  size_t ioff=0;

  if(restr_) {
    // Occupied orbitals
    arma::cx_mat CO(make_CO());
    // Virtual orbitals
    arma::cx_mat CV(make_CV());

    if(cancheck_ && va_) {
      // OV orbital energy differences
      arma::mat dE(oa_,va_);
      for(size_t io=0;io<oa_;io++) {
	// Orbital Hamiltonian is
	arma::cx_mat Fo(make_H(rsol_));
	if(pzw_!=0.0) Fo-=ref_worb_(io)*ref_Forb_[io];
	// Occupied energy is
	double Eocc=std::real(arma::as_scalar(arma::trans(CO.col(io))*Fo*CO.col(io)));
	// Loop over virtuals
	for(size_t iv=0;iv<va_;iv++) {
	  // Virtual energy is
	  double Evirt=std::real(arma::as_scalar(arma::trans(CV.col(iv))*Fo*CV.col(iv)));
	  // Store
	  dE(io,iv)=Evirt-Eocc;
	}
      }

      // Hessian shift is
      double dH=std::max(-arma::min(arma::min(dE)),1e-4);

      // Form OV gradient
      arma::cx_mat gOV(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa_,va_)-1),oa_,va_,real_,imag_));

      // Run element-wise scaling
      arma::cx_mat GOV(gOV%precondition_matrix(dE,dH));

      arma::vec POV(gather_ov(GOV,real_,imag_));
      if(POV.n_elem != count_ov_params(oa_,va_))
	throw std::logic_error("Amount of elements doesn't match!\n");
      sd.subvec(ioff,ioff+POV.n_elem-1)=POV;
      ioff+=POV.n_elem;
    }
  } else {
    arma::cx_mat COa(make_CO(false));
    arma::cx_mat COb(make_CO(true));
    arma::cx_mat CVa(make_CV(false));
    arma::cx_mat CVb(make_CV(true));

    if(cancheck_ && va_) {
      // OV orbital energy differences
      arma::mat dEa(oa_,va_);
      for(size_t io=0;io<oa_;io++) {
	// Orbital Hamiltonian is
	arma::cx_mat Fo(make_H(usol_,false));
	if(pzw_!=0.0) Fo-=ref_worba_(io)*ref_Forba_[io];
	// Occupied energy is
	double Eocc=std::real(arma::as_scalar(arma::trans(COa.col(io))*Fo*COa.col(io)));
	// Loop over virtuals
	for(size_t iv=0;iv<va_;iv++) {
	  // Virtual energy is
	  double Evirt=std::real(arma::as_scalar(arma::trans(CVa.col(iv))*Fo*CVa.col(iv)));
	  // Store
	  dEa(io,iv)=Evirt-Eocc;
	}
      }

      arma::mat dEb;
      if(ob_) {
	dEb.zeros(ob_,vb_);
	for(size_t io=0;io<ob_;io++) {
	  // Orbital Hamiltonian is
	  arma::cx_mat Fo(make_H(usol_,true));
	  if(pzw_!=0.0) Fo-=ref_worbb_(io)*ref_Forbb_[io];
	  // Occupied energy is
	  double Eocc=std::real(arma::as_scalar(arma::trans(COb.col(io))*Fo*COb.col(io)));
	  // Loop over virtuals
	  for(size_t iv=0;iv<vb_;iv++) {
	    // Virtual energy is
	    double Evirt=std::real(arma::as_scalar(arma::trans(CVb.col(iv))*Fo*CVb.col(iv)));
	    // Store
	    dEb(io,iv)=Evirt-Eocc;
	  }
	}
      }

      // Minimal Hessian shift is
      double dH=std::max(-arma::min(arma::min(dEa)),1e-4);
      if(ob_)
	dH=std::max(dH,-arma::min(arma::min(dEb)));

      // Form OV gradient
      arma::cx_mat gOVa(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa_,va_)-1),oa_,va_,real_,imag_));

      // Run element-wise division
      arma::cx_mat GOVa(gOVa%precondition_matrix(dEa,dH));

      arma::vec POVa(gather_ov(GOVa,real_,imag_));
      if(POVa.n_elem != count_ov_params(oa_,va_))
	throw std::logic_error("Amount of elements doesn't match!\n");
      sd.subvec(ioff,ioff+POVa.n_elem-1)=POVa;
      ioff+=POVa.n_elem;

      if(ob_) {
	// Form OV gradient
	arma::cx_mat gOVb(spread_ov(g.subvec(ioff,ioff+count_ov_params(ob_,vb_)-1),ob_,vb_,real_,imag_));

	// Run element-wise division
	arma::cx_mat GOVb(gOVb%precondition_matrix(dEb,dH));

	arma::vec POVb(gather_ov(GOVb,real_,imag_));
	if(POVb.n_elem != count_ov_params(ob_,vb_))
	  throw std::logic_error("Amount of elements doesn't match!\n");
	sd.subvec(ioff,ioff+POVb.n_elem-1)=POVb;
	ioff+=POVb.n_elem;
      }
    }
  }

  return sd;
}

arma::vec PZStability::gradient(const arma::vec & x, bool ref) {
  arma::vec g(count_params());
  g.zeros();

  if(restr_) {
    size_t ioff=0;

    // Evaluate orbital matrices
    rscf_t sol;
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb, worb;
    eval(x,sol,Forb,Eorb,worb,cancheck_,true,ref);

    // Occupied orbitals
    arma::cx_mat CO(make_CO(sol));
    // Virtual orbitals
    arma::cx_mat CV(make_CV(sol));

    if(cancheck_ && va_) {
      // Hamiltonian is
      arma::cx_mat H(make_H(sol));
      // OV gradient is
      arma::cx_mat gOV(oa_,va_);
      if(pzw_==0.0)
	gOV=-arma::strans(arma::trans(CV.cols(0,va_-1))*arma::conj(H)*CO.cols(0,oa_-1));
      else {
	for(size_t i=0;i<oa_;i++) {
	  arma::cx_vec hlp(arma::conj(H-worb(i)*Forb[i])*CO.col(i));
	  for(size_t a=0;a<va_;a++)
	    gOV(i,a)=-arma::cdot(CV.col(a),hlp);
	}
	// Put in scaling gradient
	scaling_gradient_ov(gOV,CO,Eorb,CV);
      }

      // Convert to proper gradient
      gOV=gradient_convert(gOV);

      // Collect values
      arma::vec pOV(gather_ov(gOV,real_,imag_));
      g.subvec(ioff,ioff+pOV.n_elem-1)=pOV;
      ioff+=pOV.n_elem;
    }

    if(oocheck_ && oa_>1) {
      // OO gradient is
      arma::cx_mat gOO(oa_,oa_);
      if(pzw_!=0.0) {
	arma::cx_mat FO(CO.n_rows,oa_);
	for(size_t i=0;i<oa_;i++)
	  FO.col(i)=worb(i)*arma::conj(Forb[i])*CO.col(i);
	gOO=-arma::strans(-arma::trans(CO)*FO + arma::trans(FO)*CO);

	// Put in scaling gradient
	scaling_gradient_oo(gOO,CO,Eorb);
      } else
	gOO.zeros();

      // Convert to proper gradient
      gOO=gradient_convert(gOO);

      // Collect values
      arma::vec pOO(gather_oo(gOO,real_,imag_));
      g.subvec(ioff,ioff+pOO.n_elem-1)=pOO;
      ioff+=pOO.n_elem;
    }

    // Closed shell - two orbitals!
    g*=2.0;

  } else {
    // Evaluate orbital matrices
    uscf_t sol;
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    arma::vec worba, worbb;
    eval(x,sol,Forba,Eorba,worba,Forbb,Eorbb,worbb,cancheck_,true,ref);

    // Occupied orbitals
    arma::cx_mat COa(make_CO(false,sol));
    arma::cx_mat COb(make_CO(true,sol));
    arma::cx_mat CVa(make_CV(false,sol));
    arma::cx_mat CVb(make_CV(true,sol));

    size_t ioff=0;

    if(cancheck_ && va_) {
      // Hamiltonian is
      arma::cx_mat Ha(make_H(sol,false));

      // OV alpha gradient is
      arma::cx_mat gOVa(oa_,va_);
      if(pzw_==0.0)
	gOVa=-arma::strans(arma::trans(CVa.cols(0,va_-1))*arma::conj(Ha)*COa.cols(0,oa_-1));
      else {
	for(size_t i=0;i<oa_;i++) {
	  arma::cx_vec hlp(arma::conj(Ha-worba(i)*Forba[i])*COa.col(i));
	  for(size_t a=0;a<va_;a++)
	    gOVa(i,a)=-arma::cdot(CVa.col(a),hlp);
	}

	// Put in scaling gradient
	scaling_gradient_ov(gOVa,COa,Eorba,CVa);
      }

      // Convert to proper gradient
      gOVa=gradient_convert(gOVa);

      // Collect values
      arma::vec pOVa(gather_ov(gOVa,real_,imag_));
      g.subvec(ioff,ioff+pOVa.n_elem-1)=pOVa;
      ioff+=pOVa.n_elem;

      if(ob_ && vb_) {
	// Hamiltonian is
	arma::cx_mat Hb(make_H(sol,true));

	// OV beta gradient is
	arma::cx_mat gOVb(ob_,vb_);
	if(pzw_==0.0)
	  gOVb=-arma::strans(arma::trans(CVb.cols(0,vb_-1))*arma::conj(Hb)*COb.cols(0,ob_-1));
	else {
	  for(size_t i=0;i<ob_;i++) {
	    arma::cx_vec hlp(arma::conj(Hb-worbb(i)*Forbb[i])*COb.col(i));
	    for(size_t a=0;a<vb_;a++)
	      gOVb(i,a)=-arma::cdot(CVb.col(a),hlp);
	  }

	  // Put in scaling gradient
	  scaling_gradient_ov(gOVb,COb,Eorbb,CVb);
	}

	// Convert to proper gradient
	gOVb=gradient_convert(gOVb);

	// Collect values
	arma::vec pOVb(gather_ov(gOVb,real_,imag_));
	g.subvec(ioff,ioff+pOVb.n_elem-1)=pOVb;
	ioff+=pOVb.n_elem;
      }
    }

    if(oocheck_) {
      if(oa_>1) {
	// OO alpha gradient is
	arma::cx_mat gOOa(oa_,oa_);
	if(pzw_!=0.0) {
	  arma::cx_mat FOa(COa.n_rows,oa_);
	  for(size_t i=0;i<oa_;i++)
	    FOa.col(i)=worba(i)*arma::conj(Forba[i])*COa.col(i);
	  gOOa=-arma::strans(-arma::trans(COa)*FOa + arma::trans(FOa)*COa);

	  // Put in scaling gradient
	  scaling_gradient_oo(gOOa,COa,Eorba);
	} else
	  gOOa.zeros();

	// Convert to proper gradient
	gOOa=gradient_convert(gOOa);

	// Collect values
	arma::vec pOOa(gather_oo(gOOa,real_,imag_));
	g.subvec(ioff,ioff+pOOa.n_elem-1)=pOOa;
	ioff+=pOOa.n_elem;
      }

      if(ob_>1) {
	// OO beta gradient is
	arma::cx_mat gOOb(ob_,ob_);
	if(pzw_!=0.0) {
	  arma::cx_mat FOb(COb.n_rows,ob_);
	  for(size_t i=0;i<ob_;i++)
	    FOb.col(i)=worbb(i)*arma::conj(Forbb[i])*COb.col(i);
	  gOOb=-arma::strans(-arma::trans(COb)*FOb + arma::trans(FOb)*COb);

	  // Put in scaling gradient
	  scaling_gradient_oo(gOOb,COb,Eorbb);
	} else
	  gOOb.zeros();

	// Convert to proper gradient
	gOOb=gradient_convert(gOOb);

	// Collect values
	arma::vec pOOb(gather_oo(gOOb,real_,imag_));
	g.subvec(ioff,ioff+pOOb.n_elem-1)=pOOb;
	ioff+=pOOb.n_elem;
      }
    }
  }

  return g;
}

arma::mat PZStability::hessian() {
  // Amount of parameters
  size_t npar=count_params();

  // Compute Hessian
  arma::mat h(npar,npar);
  h.zeros();

  /* This loop isn't OpenMP parallel, because parallellization is
     already used in the energy evaluation. Parallellizing over trials
     here would require use of thread-local DFT grids, and possibly
     even thread-local SCF solver objects (for the Coulomb part).*/
  for(size_t i=0;i<npar;i++) {
    arma::vec x(npar);
    x.zeros();

    // RHS gradient
    x(i)=ss_fd_;
    arma::vec gr=gradient(x,true);

    // LHS value
    x(i)=-ss_fd_;
    arma::vec gl=gradient(x,true);

    // Finite difference derivative is
    for(size_t j=0;j<npar;j++) {
      h(i,j)=(gr(j)-gl(j))/(2.0*ss_fd_);

      if(std::isnan(h(i,j))) {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Element (" << i << "," << j <<") of hessian gives NaN.\n";
	oss << "Step size is " << ss_fd_ << ", and left and right values are " << gl(j) << " and " << gr(j) << ".\n";
	throw std::runtime_error(oss.str());
      }
    }
  }

  // Symmetrize Hessian to distribute numerical error evenly
  h=(h+arma::trans(h))/2.0;

  return h;
}

double PZStability::eval(const arma::vec & x) {
  if(restr_) {
    rscf_t sol;
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb, worb;
    return eval(x,sol,Forb,Eorb,worb,cancheck_,false,false);
  } else {
    uscf_t sol;
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    arma::vec worba, worbb;
    return eval(x,sol,Forba,Eorba,worba,Forbb,Eorbb,worbb,cancheck_,false,false);
  }
}

double PZStability::energy() {
  arma::vec x(count_params());
  x.zeros();
  return eval(x);
}

double PZStability::optimize(size_t maxiter, double gthr, double nrthr, double dEthr, int preconditioning) {
  arma::vec x0;
  if(!count_params())
    return 0.0;
  else
    x0.zeros(count_params());

  // Make sure all data is on the checkpoint file
  update(x0);
  // Update reference
  update_reference(true);
  // Print info
  print_info();

  // Evaluate energy
  double ival=eval(x0);
  if(verbose_) printf("Initial value is % .10f\n",ival);

  // Current and previous gradient
  arma::vec g, gold;
  // Search direction
  arma::vec sd;
  // Current value
  double E0(ival);
  LBFGS lbfgs;

  for(size_t iiter=0;iiter<maxiter;iiter++) {
    // Evaluate gradient
    gold=g;
    {
      Timer t;
      g=gradient();
      print_status(iiter,g,t);
    }
    if(arma::norm(g,2)<gthr)
      break;

    // Update BFGS
    lbfgs.update(x0,g);

    // Update search direction
    arma::vec oldsd(sd);
    switch(preconditioning) {
    case(0):
      sd = -g;
      break;

    case(1):
      sd = -precondition_unified(g);
      break;

    case(2):
      sd = -precondition_orbital(g);
      break;

    default:
      throw std::logic_error("Invalid value for PZprec.\n");
    }

    if(preconditioning && arma::norm_dot(sd,-g)<0.0) {
      if(verbose_) printf("Projection of preconditioned search direction on gradient is %e, not using preconditioning.\n",arma::norm_dot(sd,-g));
      sd=-g;
    }

    if(arma::norm(g,2) < nrthr && !cancheck_) {
      // Evaluate Hessian
      Timer tp;
      if(verbose_) {
	printf("Calculating Hessian ... ");
	fflush(stdout);
      }
      arma::mat h(hessian());
      if(verbose_) {
	printf("done (%s)\n",tp.elapsed().c_str());
	fflush(stdout);
      }

      // Run eigendecomposition
      arma::vec hval;
      arma::mat hvec;
      bool diagok=arma::eig_sym(hval,hvec,h);
      if(!diagok)
	throw std::runtime_error("Error diagonalizing orbital Hessian\n");
      if(verbose_) hval.t().print("Hessian eigenvalues");

      // Enforce positive defitiveness
      hval+=std::max(0.0,-arma::min(hval))+1e-4;

      // Form new search direction: sd = - H^-1 g
      sd.zeros(hvec.n_rows);
      for(size_t i=0;i<hvec.n_cols;i++)
	sd-=arma::dot(hvec.col(i),g)/hval(i)*hvec.col(i);

      // Backtracking line search
      double Etr=eval(sd);
      if(verbose_) printf(" %e % .10f\n",1.0,Etr);
      fflush(stdout);

      double tau=0.7;
      double Enew=eval(tau*sd);
      if(verbose_) printf(" %e % .10f\n",tau,Enew);
      fflush(stdout);

      double l=1.0;
      while(Enew<Etr) {
	Etr=Enew;
	l*=tau;
	Enew=eval(l*tau*sd);
	if(verbose_) printf(" %e % .10f backtrack\n",l*tau,Enew);
	fflush(stdout);
      }

      if(verbose_) printf("Newton step changed value by %e\n",Etr-E0);
      fflush(stdout);

      update(l*sd);
      x0+=l*sd;
      if(fabs(Etr-E0)<dEthr)
	break;

      // Accept move
      E0=Etr;
      parallel_transport(g,sd,l);
      continue;

    } else if(!cancheck_) { // Use BFGS in OO optimization
      // New search direction
      arma::vec sd0(sd);
      sd=-lbfgs.solve();

      // Check sanity
      if(arma::dot(sd,-g)<0) {
	if(verbose_) printf("Bad BFGS direction, dot product % e. BFGS reset\n",arma::dot(sd,-g)/arma::dot(g,g));
	lbfgs.clear();
	lbfgs.update(x0,g);
	sd=-lbfgs.solve();

      } else if(iiter>=1 && arma::dot(g,gold)>=0.2*arma::dot(gold,gold)) {
	if(verbose_) printf("Powell restart - SD step\n");
	sd=sd0;

      } else {
	if(verbose_) printf("BFGS step\n");
	if(verbose_) printf("Projection of search direction onto steepest descent direction is %e\n",arma::dot(sd,-g)/sqrt(arma::dot(sd,sd)*arma::dot(g,g)));
      }
    } else {
      if((iiter % std::min(count_params(), (size_t) 10)!=0)) {
	// Update factor
	double gamma;

	// Polak-Ribiere
	gamma=arma::dot(g,g-gold)/arma::dot(gold,gold);
	// Fletcher-Reeves
	//gamma=arma::dot(g,g)/arma::dot(gold,gold);

	// Update search direction
	arma::vec sdnew(sd+gamma*oldsd);

	// Check that new SD is sane
	if(arma::dot(sdnew,-g)<=0) {
	  // This would take us into the wrong direction!
	  if(verbose_) printf("Bad CG direction. SD step\n");
	} else {
	  // Update search direction
	  sd=sdnew;
	  if(verbose_) printf("CG step\n");
	}
      } else if(verbose_) printf("SD step\n");
    }

    // Derivative is
    double dE=arma::dot(sd,g);

    if(verbose_) printf(" %e % .10f\n",0.0,E0);
    fflush(stdout);

    // Update step size
    update_step(sd);

    // Initial step size. Don't go too far so that the parabolic
    // approximation is valid
    //double d= cancheck ? Tmu/25.0 : Tmu/5.0;
    double d=Tmu_/5.0;
    // Value at initial step
    double Ed=eval(d*sd);
    if(verbose_) printf(" %e % .10f\n",d,Ed);
    fflush(stdout);

    // Optimal step length
    double step;
    // Energy for optimal step
    double Es;
    // Was fit succesful?
    bool fitok;

    // Fit parabola
    double a=(Ed - dE*d - E0)/(d*d);
    // Predicted energy
    double Ep;
    fitok=a>0;
    if(fitok) {
      // The optimal step is at
      step=-dE/(2.0*a);
      // Predicted energy is
      Ep=a*step*step + dE*step + E0;
    }

    // Check step length
    if(fitok) {
      if(step>d || step<0.0)
	fitok=false;
    }

    // If step is not OK, just use the trial step
    if(!fitok) {
      step=d;
      Es=Ed;
    } else {
      // Evaluate energy at trial step
      Es=eval(step*sd);
      if(fitok) {
	if(verbose_) printf(" %e % .10f, % e difference from prediction\n",step,Es,Es-Ep);
	fflush(stdout);
      }
    }

    // Did the search work? If not, backtracking line search
    if(Es>=E0) {
      double tau=0.7;
      double Es0=Es;
      while(step>DBL_EPSILON) {
	step*=tau;
	Es0=Es;
	Es=eval(step*sd);
	if(verbose_) printf(" %e % .10f backtrack\n",step,Es);
	fflush(stdout);
	if(Es>Es0 && Es<E0)
	  break;
      }
      // Overstepped
      step/=tau;
      Es=Es0;
    }

    if(verbose_) printf("Line search changed value by %e\n",Es-E0);
    update(step*sd);
    x0+=step*sd;
    if(fabs(Es-E0)<dEthr)
      break;

    // Parallel transport the gradient in the search direction
    E0=Es;
    parallel_transport(g,sd,step);
  }

  if(verbose_) printf("Final value is % .10f; optimization changed value by %e\n",E0,E0-ival);
  // Update grid
  update_grid(false);
  // Update reference
  update_reference(true);
  // Print info
  print_info();

  // Return the change
  return E0-ival;
}


void PZStability::parallel_transport(arma::vec & gold, const arma::vec & sd, double step) const {
  if(restr_ || ob_==0) {
    // Form the rotation matrix
    arma::cx_mat R(rotation(sd*step,false));
    // Form the G matrix
    arma::cx_mat G(rotation_pars(gold,false));
    // Transform G
    G=arma::trans(R)*G*R;

    // Collect the parameters
    size_t ioff=0;
    if(cancheck_) {
      arma::vec pOV(gather_ov(G.submat(0,oa_,oa_-1,oa_+va_-1),real_,imag_));
      gold.subvec(ioff,ioff+pOV.n_elem-1)=pOV;
      ioff+=pOV.n_elem;
    }
    if(oocheck_) {
      arma::vec pOO(gather_oo(G.submat(0,0,oa_-1,oa_-1),real_,imag_));
      gold.subvec(ioff,ioff+pOO.n_elem-1)=pOO;
      ioff+=pOO.n_elem;
    }

  } else {
    // Form the rotation matrix
    arma::cx_mat Ra(rotation(sd*step,false));
    arma::cx_mat Rb(rotation(sd*step,true));
    // Form the G matrix
    arma::cx_mat Ga(rotation_pars(gold,false));
    arma::cx_mat Gb(rotation_pars(gold,true));
    // Transform G
    Ga=arma::trans(Ra)*Ga*Ra;
    Gb=arma::trans(Rb)*Gb*Rb;

    // Collect the parameters
    size_t ioff=0;
    if(cancheck_) {
      arma::vec pOVa(gather_ov(Ga.submat(0,oa_,oa_-1,oa_+va_-1),real_,imag_));
      gold.subvec(ioff,ioff+pOVa.n_elem-1)=pOVa;
      ioff+=pOVa.n_elem;
      arma::vec pOVb(gather_ov(Gb.submat(0,ob_,ob_-1,ob_+vb_-1),real_,imag_));
      gold.subvec(ioff,ioff+pOVb.n_elem-1)=pOVb;
      ioff+=pOVb.n_elem;
    }
    if(oocheck_) {
      arma::vec pOOa(gather_oo(Ga.submat(0,0,oa_-1,oa_-1),real_,imag_));
      gold.subvec(ioff,ioff+pOOa.n_elem-1)=pOOa;
      ioff+=pOOa.n_elem;
      if(ob_>1) {
	arma::vec pOOb(gather_oo(Gb.submat(0,0,ob_-1,ob_-1),real_,imag_));
	gold.subvec(ioff,ioff+pOOb.n_elem-1)=pOOb;
	ioff+=pOOb.n_elem;
      }
    }
  }
}

inline void orthonormalize(const arma::mat & S, arma::cx_mat & C, bool verbose) {
  // Orbital overlap
  arma::cx_mat So(arma::trans(C)*S*C);
  // Difference from orthonormality
  arma::cx_mat dS=So-arma::eye<arma::cx_mat>(So.n_rows,So.n_cols);
  double d=arma::norm(dS,2);
  if(d>=1e-9) {
    if(verbose) printf("Difference from orbital orthonormality is %e, orthonormalizing\n",d);
    orthonormalize(S,C);
  } else {
    //printf("Difference from orbital orthonormality is %e, OK\n",d);
  }
}

static void pseudocanonize(arma::mat & C, arma::vec & E, const arma::mat & H, size_t N) {
  // Occupied orbitals
  arma::mat Co(C.cols(0,N-1));
  // Projections
  arma::mat CoHCo(arma::trans(Co)*H*Co);
  // Eigenvectors and eigenvalues
  arma::vec eval;
  arma::mat evec;
  eig_sym_ordered(eval,evec,CoHCo);

  // Rotate orbitals
  Co=Co*evec;

  // Store orbitals and energies
  C.cols(0,N-1)=Co;
  E.subvec(0,N-1)=eval;

  if(N<C.n_cols) {
    arma::mat Cv=C.cols(N,C.n_cols-1);
    arma::mat CvHCv(arma::trans(Cv)*H*Cv);
    eig_sym_ordered(eval,evec,CvHCv);
    Cv=Cv*evec;
    C.cols(N,C.n_cols-1)=Cv;
    E.subvec(N,C.n_cols-1)=eval;
  }
}

static void diagonalize(arma::vec & E, arma::mat & C, const arma::mat & H, const arma::mat & Sinvh) {
  // Run eigendecomposition in orthonormal basis
  eig_sym_ordered(E,C,arma::trans(Sinvh)*H*Sinvh);
  // Back-transform the orbitals
  C=Sinvh*C;
}

void PZStability::update(const arma::vec & x) {
  if(arma::norm(x,2)!=0.0)  {
    if(restr_) {
      arma::cx_mat R=rotation(x,false);
      rsol_.cC=rsol_.cC*R;
    } else {
      arma::cx_mat Ra=rotation(x,false);
      usol_.cCa=usol_.cCa*Ra;
      if(ob_) {
	arma::cx_mat Rb=rotation(x,true);
	usol_.cCb=usol_.cCb*Rb;
      }
    }
  }

  // Check that orbitals are orthonormal and reorthonormalize if
  // necessary
  if(true) {
    arma::mat S(basis_.overlap());
    if(restr_) {
      orthonormalize(S,rsol_.cC,verbose_);
    } else {
      orthonormalize(S,usol_.cCa,verbose_);
      orthonormalize(S,usol_.cCb,verbose_);
    }
  }

  // Update reference, without sort
  update_reference(false);

  // Update orbitals in checkpoint file
  Checkpoint *chkptp=solverp_->get_checkpoint();

  // Orthogonalizing matrix
  arma::mat Sinvh;
  chkptp->read("Sinvh",Sinvh);

  if(restr_) {
    // Generate dummy orbitals and orbital energies
    arma::mat H(arma::real(unified_H(make_CO(),make_CV(),ref_Forb_,ref_worb_,make_H(rsol_))));
    arma::vec E;
    arma::mat C;
    ::diagonalize(E,C,H,Sinvh);

    std::vector<double> occs(C.n_cols,0);
    for(size_t i=0;i<oa_;i++)
      occs[i]=2.0;
    chkptp->write("occ",occs);

    chkptp->write(rsol_.en);
    chkptp->write("C",C);
    chkptp->write("E",E);
    chkptp->write("P",rsol_.P);
    chkptp->write("H",rsol_.H);

    if(imag_) {
      chkptp->write("P_im",rsol_.P_im);
      chkptp->write("K_im",rsol_.K_im);
    }

    if(imag_ || pzw_!=0.0)
      // Only save CW if PZ is in use or orbitals are complex
      chkptp->cwrite("CW",rsol_.cC);

  } else {
    // Generate dummy orbitals and orbital energies
    arma::mat Ha(arma::real(unified_H(make_CO(false),make_CV(false),ref_Forba_,ref_worba_,make_H(usol_,false))));
    arma::mat Hb(arma::real(unified_H(make_CO(true),make_CV(true),ref_Forbb_,ref_worbb_,make_H(usol_,true))));
    arma::vec Ea, Eb;
    arma::mat Ca, Cb;
    ::diagonalize(Ea,Ca,Ha,Sinvh);
    ::diagonalize(Eb,Cb,Hb,Sinvh);


    std::vector<double> occa(Ca.n_cols,0), occb(Cb.n_cols,0);
    for(size_t i=0;i<oa_;i++)
      occa[i]=1.0;
    for(size_t i=0;i<ob_;i++)
      occb[i]=1.0;
    chkptp->write("occa",occa);
    chkptp->write("occb",occb);

    chkptp->write(usol_.en);

    chkptp->write("Ca",Ca);
    chkptp->write("Cb",Cb);
    chkptp->write("Ea",Ea);
    chkptp->write("Eb",Eb);

    chkptp->write("Ha",usol_.Ha);
    chkptp->write("Hb",usol_.Hb);
    chkptp->write("Pa",usol_.Pa);
    chkptp->write("Pb",usol_.Pb);
    chkptp->write("P",usol_.P);
    if(imag_) {
      chkptp->write("Pa_im",usol_.Pa_im);
      chkptp->write("Pb_im",usol_.Pb_im);
      chkptp->write("Ka_im",usol_.Ka_im);
      chkptp->write("Kb_im",usol_.Kb_im);
    }

    if(imag_ || pzw_!=0.0) {
      // Only save CW if PZ is in use or orbitals are complex
      chkptp->cwrite("CWa",usol_.cCa);
      chkptp->cwrite("CWb",usol_.cCb);
    }
  }
}

arma::cx_mat PZStability::make_H(const rscf_t & sol) const {
  arma::cx_mat H=sol.H*COMPLEX1;
  if(sol.K_im.n_rows == sol.H.n_rows && sol.K_im.n_cols == sol.H.n_cols)
    H-=0.5*sol.K_im*COMPLEXI;
  return H;
}

arma::cx_mat PZStability::make_H(const uscf_t & sol, bool spin) const {
  if(!spin) {
    arma::cx_mat Ha=sol.Ha*COMPLEX1;
    if(sol.Ka_im.n_rows == sol.Ha.n_rows && sol.Ka_im.n_cols == sol.Ha.n_cols)
      Ha-=sol.Ka_im*COMPLEXI;
    return Ha;
  } else {
    arma::cx_mat Hb=sol.Hb*COMPLEX1;
    if(sol.Kb_im.n_rows == sol.Hb.n_rows && sol.Kb_im.n_cols == sol.Hb.n_cols)
      Hb-=sol.Kb_im*COMPLEXI;
    return Hb;
  }
}

void PZStability::update_reference(bool sort) {
  arma::vec x0(count_params());
  x0.zeros();

  if(verbose_) printf("Updating reference ... ");
  fflush(stdout);
  Timer t;

  if(restr_) {
    rscf_t sol;
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb, worb;
    eval(x0,sol,Forb,Eorb,worb,true,true,false);

    if(sort) {
      arma::cx_mat CO(make_CO(sol));
      arma::cx_mat CV(make_CV(sol));
      // Unified Hamiltonian
      arma::cx_mat H(unified_H(CO,CV,Forb,worb,make_H(sol)));

      // Calculate projected orbital energies
      arma::vec Eorbo=arma::real(arma::diagvec(arma::trans(CO)*H*CO));
      arma::vec Eorbv;
      if(CV.n_cols)
	Eorbv=arma::real(arma::diagvec(arma::trans(CV)*H*CV));
      // Sort in ascending order
      arma::uvec idxo=arma::stable_sort_index(Eorbo,"ascend");

      // Store reference
      rsol_=sol;
      for(arma::uword i=0;i<idxo.n_elem;i++)
	rsol_.cC.col(i)=CO.col(idxo(i));
      if(CV.n_cols) {
	arma::uvec idxv=arma::stable_sort_index(Eorbv,"ascend");
	for(arma::uword i=0;i<idxv.n_elem;i++)
	  rsol_.cC.col(i+oa_)=CV.col(idxv(i));
      }
      if(pzw_!=0.0) {
	ref_Eorb_.zeros(Eorb.n_elem);
	for(size_t i=0;i<idxo.n_elem;i++)
	  ref_Eorb_(i)=Eorb(idxo(i));

	ref_worb_.zeros(worb.n_elem);
	for(size_t i=0;i<idxo.n_elem;i++)
	  ref_worb_(i)=worb(idxo(i));

	ref_Forb_.resize(Forb.size());
	for(size_t i=0;i<idxo.n_elem;i++)
	  ref_Forb_[i]=Forb[idxo(i)];
      } else {
	ref_Eorb_.clear();
	ref_worb_.clear();
	ref_Forb_.clear();
      }
    } else {
      // Store reference
      rsol_=sol;
      ref_Eorb_=Eorb;
      ref_worb_=worb;
      ref_Forb_=Forb;
    }

  } else {
    uscf_t sol;
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    arma::vec worba, worbb;
    eval(x0,sol,Forba,Eorba,worba,Forbb,Eorbb,worbb,true,true,false);

    if(sort) {
      arma::cx_mat COa(make_CO(false,sol));
      arma::cx_mat COb(make_CO(true,sol));
      arma::cx_mat CVa(make_CV(false,sol));
      arma::cx_mat CVb(make_CV(true,sol));

      // Unified Hamiltonians
      arma::cx_mat Ha(unified_H(COa,CVa,Forba,worba,make_H(sol,false)));
      arma::cx_mat Hb(unified_H(COb,CVb,Forbb,worbb,make_H(sol,true)));

      // Calculate projected orbital energies
      arma::vec Eorbao=arma::real(arma::diagvec(arma::trans(COa)*Ha*COa));
      arma::vec Eorbav;
      if(CVa.n_cols)
	Eorbav=arma::real(arma::diagvec(arma::trans(CVa)*Ha*CVa));
      arma::vec Eorbbo;
      if(ob_)
	Eorbbo=arma::real(arma::diagvec(arma::trans(COb)*Hb*COb));
      arma::vec Eorbbv;
      if(CVb.n_cols)
	Eorbbv=arma::real(arma::diagvec(arma::trans(CVb)*Hb*CVb));

      // Sort in ascending order
      arma::uvec idxao=arma::stable_sort_index(Eorbao,"ascend");

      usol_=sol;
      for(size_t i=0;i<idxao.n_elem;i++)
	usol_.cCa.col(i)=COa.col(idxao(i));
      if(CVa.n_cols) {
	arma::uvec idxav=arma::stable_sort_index(Eorbav,"ascend");
	for(arma::uword i=0;i<idxav.n_elem;i++)
	  usol_.cCa.col(i+oa_)=CVa.col(idxav(i));
      }

      if(pzw_!=0.0) {
	ref_Eorba_.zeros(oa_);
	for(size_t i=0;i<idxao.n_elem;i++)
	  ref_Eorba_(i)=Eorba(idxao(i));

	ref_worba_.zeros(oa_);
	for(size_t i=0;i<idxao.n_elem;i++)
	  ref_worba_(i)=worba(idxao(i));

	ref_Forba_.resize(oa_);
	for(size_t i=0;i<idxao.n_elem;i++)
	  ref_Forba_[i]=Forba[idxao(i)];
      } else {
	ref_Eorba_.clear();
	ref_worba_.clear();
	ref_Forba_.clear();
      }

      if(ob_) {
	arma::uvec idxbo=arma::stable_sort_index(Eorbbo,"ascend");
	for(arma::uword i=0;i<idxbo.n_elem;i++)
	  usol_.cCb.col(i)=COb.col(idxbo(i));

	if(pzw_!=0.0) {
	  ref_Eorbb_.zeros(ob_);
	  for(size_t i=0;i<idxbo.n_elem;i++)
	    ref_Eorbb_(i)=Eorbb(idxbo(i));

	  ref_worbb_.zeros(ob_);
	  for(size_t i=0;i<idxbo.n_elem;i++)
	    ref_worbb_(i)=worbb(idxbo(i));

	  ref_Forbb_.resize(ob_);
	  for(size_t i=0;i<idxbo.n_elem;i++)
	    ref_Forbb_[i]=Forbb[idxbo(i)];
	} else {
	  ref_Eorbb_.clear();
	  ref_worbb_.clear();
	  ref_Forbb_.clear();
	}
      }
      if(CVb.n_cols) {
	arma::uvec idxbv=arma::stable_sort_index(Eorbbv,"ascend");
	for(arma::uword i=0;i<idxbv.n_elem;i++)
	  usol_.cCb.col(i+ob_)=CVb.col(idxbv(i));
      }
    } else {
      // Store reference
      usol_=sol;
      ref_Eorba_=Eorba;
      ref_worba_=worba;
      ref_Forba_=Forba;

      ref_Eorbb_=Eorbb;
      ref_worbb_=worbb;
      ref_Forbb_=Forbb;
    }
  }

  if(verbose_) printf("done (%s)\n",t.elapsed().c_str());
  fflush(stdout);
}

arma::cx_mat PZStability::rotation(const arma::vec & x, bool spin) const {
  // Get rotation matrix
  arma::cx_mat X(rotation_pars(x,spin));

  // Rotation matrix
  arma::cx_mat R(X);
  R.eye();
  if(oocheck_ && !cancheck_) {
    // It suffices to just exponentiate the OO block
    size_t o=spin ? ob_ : oa_;
    R.submat(0,0,o-1,o-1)=matexp(X.submat(0,0,o-1,o-1));
  } else
    // Need to exponentiate the whole thing
    R=matexp(X);

  return R;
}

arma::cx_mat PZStability::rotation_pars(const arma::vec & x, bool spin) const {
  if(x.n_elem != count_params()) {
    ERROR_INFO();
    throw std::runtime_error("Inconsistent parameter size.\n");
  }
  if(spin && restr_) {
    ERROR_INFO();
    throw std::runtime_error("Incompatible arguments.\n");
  }

  // Amount of occupied and virtual orbitals
  size_t o=oa_, v=va_;
  if(spin) {
    o=ob_;
    v=vb_;
  }

  // Construct full, padded rotation matrix
  arma::cx_mat R(o+v,o+v);
  R.zeros();

  // OV part
  if(cancheck_) {
    size_t ioff0=0;
    if(spin)
      ioff0=count_ov_params(oa_,va_);

    if(v) {
      arma::cx_mat r(spread_ov(x.subvec(ioff0,ioff0+count_ov_params(o,v)-1),o,v,real_,imag_));
      R.submat(0,o,o-1,o+v-1)=r;
      R.submat(o,0,o+v-1,o-1)=-arma::trans(r);
    }
  }

  // OO part
  if(oocheck_ && o>1) {
    size_t ioff0=0;
    // Canonical rotations
    if(cancheck_) {
      ioff0=count_ov_params(oa_,va_);
      if(!restr_)
	ioff0+=count_ov_params(ob_,vb_);
    }
    // Occupied rotations
    if(spin)
      ioff0+=count_oo_params(oa_);

    // Get the rotation matrix
    arma::cx_mat r(spread_oo(x.subvec(ioff0,ioff0+count_oo_params(o)-1),o,real_,imag_));
    R.submat(0,0,o-1,o-1)=r;
  }

  return R;
}

arma::cx_mat PZStability::matexp(const arma::cx_mat & R) const {
  // R is anti-hermitian. Get its eigenvalues and eigenvectors
  arma::cx_mat Rvec;
  arma::vec Rval;
  bool diagok=arma::eig_sym(Rval,Rvec,-COMPLEXI*R);
  if(!diagok) {
    arma::mat Rt;
    Rt=arma::real(R);
    Rt.save("R_re.dat",arma::raw_ascii);
    Rt=arma::imag(R);
    Rt.save("R_im.dat",arma::raw_ascii);

    ERROR_INFO();
    throw std::runtime_error("Unitary optimization: error diagonalizing R.\n");
  }

  // Rotation is
  arma::cx_mat rot(Rvec*arma::diagmat(arma::exp(COMPLEXI*Rval))*arma::trans(Rvec));

  arma::cx_mat prod=arma::trans(rot)*rot-arma::eye(rot.n_cols,rot.n_cols);
  double norm=rms_cnorm(prod);
  if(norm>=sqrt(DBL_EPSILON)) {
    arma::mat Rre(arma::real(R));
    Rre.save("R.real.dat",arma::raw_ascii);
    arma::mat Rim(arma::imag(R));
    Rim.save("R.imag.dat",arma::raw_ascii);

    arma::mat rotre(arma::real(rot));
    rotre.save("rotation.real.dat",arma::raw_ascii);
    arma::mat rotim(arma::imag(rot));
    rotim.save("rotation.imag.dat",arma::raw_ascii);
    std::ostringstream oss;
    oss << "Matrix is not unitary! RMS deviation from unitarity is " << norm << "!\n";
    throw std::runtime_error(oss.str());
  }

  return rot;
}

void PZStability::configure_method(const dft_t & ovmethod_v, const dft_t & oomethod_v, double pzw_v, pz_scaling_t scale_v, double scaleexp_v) {
  ovmethod_=ovmethod_v;
  oomethod_=oomethod_v;
  pzw_=pzw_v;
  scale_=scale_v;
  scaleexp_=scaleexp_v;
  if(scale_!=PZ_SCALE_CONSTANT)
    // Override dummy value
    pzw_=1.0;

  Checkpoint *chkptp=solverp_->get_checkpoint();
  chkptp->read(basis_);
  grid_=DFTGrid(&basis_,true,ovmethod_.lobatto);
  nlgrid_=DFTGrid(&basis_,false,ovmethod_.lobatto);

  // Range-separation constants
  double omega, kfull, kshort;
  range_separation(ovmethod_.x_func,omega,kfull,kshort);

  if(verbose_) {
    if(omega!=0.0) {
      printf("\nUsing range-separated exchange with range-separation constant omega = % .3f.\n",omega);
      printf("Using % .3f %% short-range and % .3f %% long-range exchange.\n",(kfull+kshort)*100,kfull*100);
    } else if(kfull!=0.0)
      printf("\nUsing hybrid exchange with % .3f %% of exact exchange.\n",kfull*100);
    else
      printf("\nA pure exchange functional used, no exact exchange.\n");
  }

  // Compute range-separated integrals if necessary
  if(is_range_separated(ovmethod_.x_func))
    solverp_->fill_rs(omega);
}

void PZStability::configure_dof(bool real_v, bool imag_v, bool can, bool oo) {
  real_=real_v;
  imag_=imag_v;
  cancheck_=can;
  oocheck_=oo;

  if(verbose_) {
    std::vector<std::string> truth(2);
    truth[0]="false";
    truth[1]="true";
    fprintf(stderr,"oo = %s, ov = %s, real = %s, imag = %s\n",truth[oocheck_].c_str(),truth[cancheck_].c_str(),truth[real_].c_str(),truth[imag_].c_str());
    fprintf(stderr,"There are %i parameters.\n",(int) count_params());
  }

  if(false) {
    // Check that gradient is valid
    arma::vec x0;
    x0.zeros(count_params());

    // Test gradient
    arma::vec g(gradient(x0,true));
    arma::vec gn(FDHessian::gradient(x0));
    double norm(arma::norm(g-gn,2));
    printf("Gradient error norm is  %e\n",norm);
    if(norm>=1e-6) {
      g.t().print("Analytic gradient");
      gn.t().print("Numerical gradient");
      fflush(stdout);
      throw std::logic_error("Gradient is wrong.\n");
    }
  }
}

void PZStability::set_reference(const rscf_t & sol) {
  Checkpoint *chkptp=solverp_->get_checkpoint();

  chkptp->read(basis_);

  // Update solution
  rsol_=sol;

  // Update size parameters
  restr_=true;
  int Na;
  chkptp->read("Nel-a",Na);
  ob_=oa_=Na;
  va_=vb_=rsol_.cC.n_cols-oa_;

  chkptp->write("Restricted",1);

  std::vector<std::string> truth(2);
  truth[0]="false";
  truth[1]="true";

  if(verbose_) fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i\n",(int) oa_, (int) ob_, (int) va_, (int) vb_);

  // Reconstruct DFT grid
  update_grid(true);
  // Update reference
  update_reference(true);
}

void PZStability::update_grid(bool init) {
  grid_.verbose(verbose_);
  nlgrid_.verbose(verbose_);
  if(ovmethod_.adaptive) {
    arma::cx_mat Ctilde;
    if(restr_)
      Ctilde=rsol_.cC.cols(0,oa_-1);
    else {
      Ctilde.zeros(usol_.cCa.n_rows,oa_+ob_);
      Ctilde.cols(0,oa_-1)=usol_.cCa.cols(0,oa_-1);
      if(ob_)
	Ctilde.cols(oa_,oa_+ob_-1)=usol_.cCb.cols(0,ob_-1);
    }
    if (ovmethod_.x_func>0 || ovmethod_.c_func>0)
      grid_.construct(Ctilde,ovmethod_.gridtol,ovmethod_.x_func,ovmethod_.c_func);
  } else if(init) {
    if (ovmethod_.x_func>0 || ovmethod_.c_func>0)
      grid_.construct(ovmethod_.nrad,ovmethod_.lmax,ovmethod_.x_func,ovmethod_.c_func);
    if(ovmethod_.nl)
      nlgrid_.construct(ovmethod_.nlnrad,ovmethod_.nllmax,true,false,false,true);
  }
}

void PZStability::set_reference(const uscf_t & sol) {
  Checkpoint *chkptp=solverp_->get_checkpoint();

  // Update solution
  usol_=sol;

  // Update size parameters
  restr_=false;
  int Na, Nb;
  chkptp->read("Nel-a",Na);
  chkptp->read("Nel-b",Nb);
  oa_=Na;
  ob_=Nb;
  va_=usol_.cCa.n_cols-oa_;
  vb_=usol_.cCb.n_cols-ob_;

  chkptp->write("Restricted",0);
  if(verbose_) fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i\n",(int) oa_, (int) ob_, (int) va_, (int) vb_);
  fflush(stderr);

  // Reconstruct DFT grid
  update_grid(true);
  // Update reference
  update_reference(true);
}

rscf_t PZStability::rsol() const {
  return rsol_;
}

uscf_t PZStability::usol() const {
  return usol_;
}

bool PZStability::check(bool stability, double cutoff, double dEthr) {
  Timer tfull;

  if(!count_params())
    return false;

  // Estimate runtime
  {
    double ttot=0.0;

    Timer t;
    arma::vec x(count_params());

    if(cancheck_) {
      // OV part
      if(cancheck_ && vb_) {
	x.zeros();
	x(0)=ss_fd_;
	t.set();
	gradient(x,true);
	double dt=t.get();

	// Total number of OV calculations is
	if(restr_)
	  ttot+=2*dt*count_ov_params(oa_,va_);
	else
	  ttot+=2*dt*(count_ov_params(oa_,va_)+count_ov_params(ob_,vb_));
      }
    }
    if(oocheck_ && oa_>1) {
      x.zeros();
      size_t ioff0=0;
      // Canonical rotations
      if(cancheck_) {
	ioff0=count_ov_params(oa_,va_);
	if(!restr_)
	  ioff0+=count_ov_params(ob_,vb_);
      }
      x(ioff0)=ss_fd_;
      t.set();
      gradient(x,true);
      double dt=t.get();

      // Total number of OO calculations is
      if(restr_)
	ttot+=2*dt*count_oo_params(oa_);
      else
	ttot+=2*dt*(count_oo_params(oa_)+count_oo_params(ob_));
    }

    // Total time is
    if(verbose_) {
      fprintf(stderr,"\nComputing the Hessian will take approximately %s\n",t.parse(ttot).c_str());
      fflush(stderr);
    }
  }

  // Evaluate Hessian
  Timer t;
  arma::mat h(hessian());
  if(verbose_) {
    printf("Hessian evaluated (%s)\n",t.elapsed().c_str()); fflush(stdout);
  }
  t.set();

  // Block the degrees of freedom
  std::vector<pz_rot_par_t> dof(classify());
  // Block-diagonalize Hessian
  if(verbose_)
    for(size_t i=0;i<dof.size();i++) {
      // Helpers
      Timer tdiag;
      arma::vec hval;
      bool diagok=arma::eig_sym(hval,h.submat(dof[i].idx,dof[i].idx));
      if(!diagok) {
	std::ostringstream oss;
	oss << "Error diagonalizing " << dof[i].name << " Hessian.\n";
	throw std::runtime_error(oss.str());
      }

      std::ostringstream oss;
      oss << "Eigenvalues in the " << dof[i].name << " block (" << tdiag.elapsed() << ")";
      hval.t().print(oss.str());
      fflush(stdout);
    }

  arma::mat I;
  if(stability) {
    Timer tdiag;
    arma::vec hval;
    arma::mat hvec;
    bool diagok=arma::eig_sym(hval,hvec,h);
    if(!diagok) {
      std::ostringstream oss;
      oss << "Error diagonalizing full Hessian.\n";
      throw std::runtime_error(oss.str());
    }
    if(verbose_) printf("Full Hessian diagonalized in %s.\n",tdiag.elapsed().c_str());

    // Find instabilities
    I=hvec.cols(arma::find(hval<cutoff));
    // Displace solution in the direction of the instabilities
    arma::vec x(count_params());
    x.zeros();
    // Current energy
    double E0=eval(x);
    // Initial energy
    double Ei=E0;

    // Form eigenvector
    if(I.n_cols) {
      // Just use the eigenvector corresponding to the smallest
      // eigenvalue, since the space is curved anyhow
      x=I.col(0);

      // Do line search
      double ds=ss_ls_;
      const double dfac=cbrt(10.0);

      double Enew=eval(x*ds);
      if(verbose_) printf("\t%e % .10f % e\n",ds,Enew,Enew-Ei);

      while(true) {
	ds*=dfac;
	E0=Enew;
	Enew=eval(x*ds);
	if(verbose_) printf("\t%e % .10f % e\n",ds,Enew,Enew-Ei);
	fflush(stdout);
	if(Enew>E0)
	  break;
      }
      // Overstepped
      ds/=dfac;

      if(E0-Ei<dEthr) {
	if(verbose_) printf("Stability analysis decreased energy by %e\n",E0-Ei);

	// Update solution
	x*=ds;

	// Update solution
	update(x);
      } else {
	I.clear();
	if(verbose_) printf("Stability analysis failed to decrease energy significantly, dE = %e\n",E0-Ei);
      }
    }
  }

  if(verbose_) fprintf(stderr,"Check completed in %s.\n",tfull.elapsed().c_str());

  // Found instabilities?
  return stability && I.n_cols>0;
}

void PZStability::print_status(size_t iiter, const arma::vec & g, const Timer & t) const {
  if(verbose_) printf("\nIteration %i, gradient norm (%s):\n",(int) iiter,t.elapsed().c_str());

  // Get decomposition
  std::vector<pz_rot_par_t> dof(classify());
  for(size_t i=0;i<dof.size();i++) {
    arma::vec gs(dof[i].idx.n_elem);
    for(size_t k=0;k<dof[i].idx.n_elem;k++)
      gs(k)=g(dof[i].idx(k));

    if(verbose_) printf("%20s %e %e\n",dof[i].name.c_str(),arma::norm(gs,2),arma::norm(gs,"inf"));
  }
}

void PZStability::linesearch(const std::string & fname, int prec, int Np) {
  // Get gradient
  arma::vec g(gradient());

  // Use preconditioned direction
  if(prec==1)
    g=precondition_unified(g);
  else if(prec==2)
    g=precondition_orbital(g);

  FILE *out=fopen(fname.c_str(),"w");
  // Do line search
  double dx=Tmu_/Np;
  for(int i=-Np;i<=Np;i++) {
    printf("x = %e\n",i*dx);
    fprintf(out,"%e % e\n",i*dx,eval(i*dx*g));
    fflush(out);
  }
  fclose(out);
}
