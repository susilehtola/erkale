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

FDHessian::FDHessian() {
  ss_fd=cbrt(DBL_EPSILON);
  ss_ls=1e-4;
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
    x(i)+=ss_fd;
    double yr=eval(x);

    // LHS value
    x=x0;
    x(i)-=ss_fd;
    double yl=eval(x);

    // Derivative
    g(i)=(yr-yl)/(2.0*ss_fd);

    if(std::isnan(g(i))) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Element " << i << " of gradient gives NaN.\n";
      oss << "Step size is " << ss_fd << ", and left and right values are " << yl << " and " << yr << ".\n";
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
    x(i)+=ss_fd;
    x(j)+=ss_fd;
    double yrr=eval(x);

    // RH,LH
    x.zeros();
    x(i)+=ss_fd;
    x(j)-=ss_fd;
    double yrl=eval(x);

    // LH,RH
    x.zeros();
    x(i)-=ss_fd;
    x(j)+=ss_fd;
    double ylr=eval(x);

    // LH,LH
    x.zeros();
    x(i)-=ss_fd;
    x(j)-=ss_fd;
    double yll=eval(x);

    // Values
    h(i,j)=(yrr - yrl - ylr + yll)/(4.0*ss_fd*ss_fd);
    // Symmetrize
    h(j,i)=h(i,j);

    if(std::isnan(h(i,j))) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Element (" << i << "," << j << ") of Hessian gives NaN.\n";
      oss << "Step size is " << ss_fd << ". Stencil values\n";
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
  printf("\nIteration %i, gradient norm %e, max norm %e (%s)\n",(int) iiter,arma::norm(g,2),arma::max(arma::abs(g)),t.elapsed().c_str());
}

double FDHessian::optimize(size_t maxiter, double gthr, bool max) {
  arma::vec x0;
  if(!count_params())
    return 0.0;
  else
    x0.zeros(count_params());

  double ival=eval(x0);
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
    double initstep=ss_ls;
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
	printf("Powell restart - SD step\n");
      } else if(arma::dot(sdnew,sd)<=0) {
	// This would take us into the wrong direction!
	printf("Bad CG direction. SD step\n");
      } else {
	// Update search direction
	sd=sdnew;
	printf("CG step\n");
      }
    } else printf("SD step\n");

    while(true) {
      step.push_back(std::pow(stepfac,step.size())*initstep);
      val.push_back(eval(step[step.size()-1]*sd));

      if(val.size()>=2)
	printf(" %e % .10f % e % e\n",step[step.size()-1],val[val.size()-1],val[val.size()-1]-val[0],val[val.size()-1]-val[val.size()-2]);
      else
      	printf(" %e % .10f\n",step[step.size()-1],val[val.size()-1]);

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
    printf("Line search changed value by %e\n",val[iopt]-val[0]);

    // Optimal value is
    double optstep=step[iopt];
    // Update x
    update(optstep*sd);
  }

  double fval=eval(x0);
  printf("Final value is % .10f; optimization changed value by %e\n",fval,fval-ival);

  // Return the change
  return fval-ival;
}

PZStability::PZStability(SCF * solver) {
  solverp=solver;
  solverp->set_verbose(false);

  imag=true;
  cancheck=false;
  oocheck=true;

  // Init sizes
  restr=true;
  oa=ob=0;
  va=vb=0;
}

PZStability::~PZStability() {
}

size_t PZStability::count_oo_params(size_t o) const {
  size_t n=0;
  if(real)
    n+=o*(o-1)/2;
  if(imag)
    n+=o*(o-1)/2;

  return n;
}

size_t PZStability::count_ov_params(size_t o, size_t v) const {
  size_t n=0;
  // Real part
  if(real)
    n+=o*v;
  // Complex part
  if(imag)
    n+=o*v;

  return n;
}

size_t PZStability::count_params(size_t o, size_t v) const {
  size_t n=0;

  // Check canonicals?
  if(cancheck) {
    n+=count_ov_params(o,v);
  }

  // Check oo block?
  if(oocheck) {
    n+=count_oo_params(o);
  }

  return n;
}

size_t PZStability::count_params() const {
  size_t npar=count_params(oa,va);
  if(!restr)
    npar+=count_params(ob,vb);

  return npar;
}

std::vector<pz_rot_par_t> PZStability::classify() const {
  std::vector<pz_rot_par_t> ret;
  if(restr || ob==0) {
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
    if(cancheck) {
      if(real) {
	arma::uword np=oa*va;
	arma::uvec i(arma::linspace<arma::uvec>(ioff,ioff+np-1,np));
	ovreal.idx=i;
	ret.push_back(ovreal);
	ioff+=np;
      }
      if(imag) {
	arma::uword np=oa*va;
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
    if(oocheck) {
      if(real) {
	arma::uword np=oa*(oa-1)/2;
	arma::uvec i(arma::linspace<arma::uvec>(ioff,ioff+np-1,np));
	ooreal.idx=i;
	if(np)
	  ret.push_back(ooreal);
	ioff+=np;
      }
      if(imag) {
	arma::uword np=oa*(oa-1)/2;
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
    if(cancheck && oocheck) {
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
    if(cancheck) {
      if(real) {
	arma::uword np=oa*va;
	ovareal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	if(np)
	  ret.push_back(ovareal);
	ioff+=np;
      }

      if(imag) {
	arma::uword np=oa*va;
	ovaimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	ret.push_back(ovaimag);
	ioff+=np;
      }

      if(real) {
	arma::uword np=ob*vb;
	ovbreal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	if(np)
	  ret.push_back(ovbreal);
	ioff+=np;
      }

      if(imag) {
	arma::uword np=ob*vb;
	ovbimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	ret.push_back(ovbimag);
	ioff+=np;
      }

      if(real) {
	ovreal.idx.zeros(ovareal.idx.n_elem+ovbreal.idx.n_elem);
	if(ovareal.idx.n_elem)
	  ovreal.idx.subvec(0,ovareal.idx.n_elem-1)=ovareal.idx;
	if(ovbreal.idx.n_elem)
	  ovreal.idx.subvec(ovareal.idx.n_elem,ovreal.idx.n_elem-1)=ovbreal.idx;
	ret.push_back(ovreal);
      }
      
      if(imag) {
	ovimag.idx.zeros(ovaimag.idx.n_elem+ovbimag.idx.n_elem);
	if(ovaimag.idx.n_elem)
	  ovimag.idx.subvec(0,ovaimag.idx.n_elem-1)=ovaimag.idx;
	if(ovbimag.idx.n_elem)
	  ovimag.idx.subvec(ovaimag.idx.n_elem,ovimag.idx.n_elem-1)=ovbimag.idx;
	ret.push_back(ovimag);
      }
      
      if(real && imag) {
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
    
    if(oocheck) {
      if(real) {
	arma::uword np=oa*(oa-1)/2;
	if(np>0) {
	  ooareal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(ooareal);
	}
	ioff+=np;
      }

      if(imag) {
	arma::uword np=oa*(oa-1)/2;
	if(np>0) {
	  ooaimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(ooaimag);
	}
	ioff+=np;
      }

      if(real) {
	arma::uword np=ob*(ob-1)/2;
	if(np>0) {
	  oobreal.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(oobreal);
	}
	ioff+=np;
      }

      if(imag) {
	arma::uword np=ob*(ob-1)/2;
	if(np>0) {
	  oobimag.idx=arma::linspace<arma::uvec>(ioff,ioff+np-1,np);
	  ret.push_back(oobimag);
	}
	ioff+=np;
      }

      if(real && oa>1 && ob>1) {
	ooreal.idx.zeros(ooareal.idx.n_elem+oobreal.idx.n_elem);
	if(ooareal.idx.n_elem)
	  ooreal.idx.subvec(0,ooareal.idx.n_elem-1)=ooareal.idx;
	if(oobreal.idx.n_elem)
	  ooreal.idx.subvec(ooareal.idx.n_elem,ooreal.idx.n_elem-1)=oobreal.idx;
	ret.push_back(ooreal);
      }

      if(imag && oa>1 && ob>1) {
	ooimag.idx.zeros(ooaimag.idx.n_elem+oobimag.idx.n_elem);
	if(ooaimag.idx.n_elem)
	  ooimag.idx.subvec(0,ooaimag.idx.n_elem-1)=ooaimag.idx;
	if(oobimag.idx.n_elem)
	  ooimag.idx.subvec(ooaimag.idx.n_elem,ooimag.idx.n_elem-1)=oobimag.idx;
	ret.push_back(ooimag);
      }

      if(real && imag) {
	ooa.idx.zeros(ooareal.idx.n_elem+ooaimag.idx.n_elem);
	if(ooareal.idx.n_elem)
	  ooa.idx.subvec(0,ooareal.idx.n_elem-1)=ooareal.idx;
	if(ooaimag.idx.n_elem)
	  ooa.idx.subvec(ooareal.idx.n_elem,ooa.idx.n_elem-1)=ooaimag.idx;
	ret.push_back(ooa);

	if(ob>1) {
	  oob.idx.zeros(oobreal.idx.n_elem+oobimag.idx.n_elem);
	  if(oobreal.idx.n_elem)
	    oob.idx.subvec(0,oobreal.idx.n_elem-1)=oobreal.idx;
	  if(oobimag.idx.n_elem)
	    oob.idx.subvec(oobreal.idx.n_elem,oob.idx.n_elem-1)=oobimag.idx;
	  ret.push_back(oob);
	}
	
	oo.idx.zeros(ooa.idx.n_elem+oob.idx.n_elem);
	oo.idx.subvec(0,ooa.idx.n_elem-1)=ooa.idx;
	if(ob>1) {
	  oo.idx.subvec(ooa.idx.n_elem,oo.idx.n_elem-1)=oob.idx;
	  ret.push_back(oo);
	}
      }
    }
    if(cancheck && oocheck) {
      rareal.idx.zeros(ooareal.idx.n_elem+ovareal.idx.n_elem);
      if(ooareal.idx.n_elem)
	rareal.idx.subvec(0,ooareal.idx.n_elem-1)=ooareal.idx;
      if(ovareal.idx.n_elem)
	rareal.idx.subvec(ooareal.idx.n_elem,rareal.idx.n_elem-1)=ovareal.idx;
      if(real && imag)
	ret.push_back(rareal);

      rbreal.idx.zeros(oobreal.idx.n_elem+ovbreal.idx.n_elem);
      if(oobreal.idx.n_elem)
	rbreal.idx.subvec(0,oobreal.idx.n_elem-1)=oobreal.idx;
      if(ovbreal.idx.n_elem)
	rbreal.idx.subvec(oobreal.idx.n_elem,rbreal.idx.n_elem-1)=ovbreal.idx;
      if(real && imag)
	ret.push_back(rbreal);

      rreal.idx.zeros(rareal.idx.n_elem+rbreal.idx.n_elem);
      if(rareal.idx.n_elem)
	rreal.idx.subvec(0,rareal.idx.n_elem-1)=rareal.idx;
      if(rbreal.idx.n_elem)
	rreal.idx.subvec(rareal.idx.n_elem,rreal.idx.n_elem-1)=rbreal.idx;
      if(real && imag)
	ret.push_back(rreal);

      raimag.idx.zeros(ooaimag.idx.n_elem+ovaimag.idx.n_elem);
      if(ooaimag.idx.n_elem)
	raimag.idx.subvec(0,ooaimag.idx.n_elem-1)=ooaimag.idx;
      if(ovaimag.idx.n_elem)
	raimag.idx.subvec(ooaimag.idx.n_elem,raimag.idx.n_elem-1)=ovaimag.idx;
      if(imag && imag)
	ret.push_back(raimag);

      rbimag.idx.zeros(oobimag.idx.n_elem+ovbimag.idx.n_elem);
      if(oobimag.idx.n_elem)
	rbimag.idx.subvec(0,oobimag.idx.n_elem-1)=oobimag.idx;
      if(ovbimag.idx.n_elem)
	rbimag.idx.subvec(oobimag.idx.n_elem,rbimag.idx.n_elem-1)=ovbimag.idx;
      if(real && imag)
	ret.push_back(rbimag);

      rimag.idx.zeros(raimag.idx.n_elem+rbimag.idx.n_elem);
      if(raimag.idx.n_elem)
	rimag.idx.subvec(0,raimag.idx.n_elem-1)=raimag.idx;
      if(rbimag.idx.n_elem)
	rimag.idx.subvec(raimag.idx.n_elem,rimag.idx.n_elem-1)=rbimag.idx;
      if(real && imag)
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

arma::cx_mat PZStability::unified_H(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0) const {
  // Build effective Fock operator
  arma::cx_mat H(H0*COMPLEX1);

  if(pzw!=0.0) {
    arma::mat S(solverp->get_S());
    for(size_t io=0;io<CO.n_cols;io++) {
      arma::cx_mat Porb(form_density(CO.col(io)));
      H-=pzw*S*Porb*Forb[io]*Porb*S;
    }

    if(CV.n_cols) {
      // Virtual space density matrix
      arma::cx_mat v(CV.n_rows,CV.n_rows);
      v.zeros();
      for(size_t io=0;io<CV.n_cols;io++)
	v+=form_density(CV.col(io));

      for(size_t io=0;io<CO.n_cols;io++) {
	arma::cx_mat Porb(form_density(CO.col(io)));
	H-=pzw*S*(v*Forb[io]*Porb + Porb*Forb[io]*v)*S;
      }
    }
  }

  return H;
}

void PZStability::print_info(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0, const arma::vec & Eorb) {
  // Form unified Hamiltonian
  arma::cx_mat H(unified_H(CO,CV,Forb,H0));

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

  if(pzw!=0.0) {
    // Collect projected energies
    arma::vec Ep(CO.n_cols);
    for(size_t io=0;io<CO.n_cols;io++)
      Ep(io)=std::real(Hoo(io,io));

    // Print out optimal orbitals
    if(CO.n_cols) {
      printf("Decomposition of self-interaction energies:\n");
      printf("\t%4s\t%8s\t%8s\n","io","E(orb)","E(SI)");
      for(size_t io=0;io<CO.n_cols;io++)
	printf("\t%4i\t% 8.3f\t% 8.3f\n",(int) io+1,Ep(io),Eorb(io));
      fflush(stdout);
    }
  }
}

void PZStability::print_info() {
  arma::vec x(count_params());
  x.zeros();

  rscf_t rsl;
  uscf_t usl;
  
  if(restr) {
    // Evaluate orbital matrices
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    eval(x,rsl,Forb,Eorb,true,true,pzw,true);

    // Occupied orbitals
    arma::cx_mat CO=rsl.cC.cols(0,oa-1);
    // Virtuals
    arma::cx_mat CV;
    if(va)
      CV=rsl.cC.cols(oa,oa+va-1);

    // Diagonalize
    print_info(CO,CV,Forb,get_H(rsl),Eorb);
  } else {
    // Evaluate orbital matrices
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    eval(x,usl,Forba,Eorba,Forbb,Eorbb,true,true,pzw,true);

    // Occupied orbitals
    arma::cx_mat COa=usl.cCa.cols(0,oa-1);
    arma::cx_mat COb;
    if(ob)
      COb=usl.cCb.cols(0,ob-1);
    // Virtuals
    arma::cx_mat CVa;
    if(va)
      CVa=usl.cCa.cols(oa,oa+va-1);
    arma::cx_mat CVb;
    if(vb)
      CVb=usl.cCb.cols(ob,ob+vb-1);

    // Diagonalize
    printf("\n **** Alpha orbitals ****\n");
    print_info(COa,CVa,Forba,get_H(usl,false),Eorba);
    printf("\n **** Beta  orbitals ****\n");
    print_info(COb,CVb,Forbb,get_H(usl,true),Eorbb);
  }

  // Print total energy and its components
  energy_t en = restr ? rsl.en : usl.en;
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
  if(restr || ob==0) {
    arma::cx_mat G=rotation_pars(g,false);
    if(oocheck && !cancheck)
      // Only doing OO block, so we can take the first subblock
      G=G.submat(0,0,oa-1,oa-1);

    // Calculate eigendecomposition
    arma::vec Gval;
    arma::cx_mat Gvec;
    bool diagok=arma::eig_sym(Gval,Gvec,-COMPLEXI*G);
    if(!diagok) {
      ERROR_INFO();
      throw std::runtime_error("Error diagonalizing G.\n");
    }

    // Calculate maximum step size; cost function is 4th order in parameters
    Tmu=0.5*M_PI/arma::max(arma::abs(Gval));

  } else {
    arma::cx_mat Ga=rotation_pars(g,false);
    arma::cx_mat Gb=rotation_pars(g,true);
    if(oocheck && !cancheck) {
      // Only doing OO block, so we can take the OO subblocks
      Ga=Ga.submat(0,0,oa-1,oa-1);
      Gb=Gb.submat(0,0,ob-1,ob-1);
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
    Tmu=0.5*M_PI/std::max(arma::max(arma::abs(Gaval)),arma::max(arma::abs(Gbval)));
  }
}

double PZStability::eval(const arma::vec & x, rscf_t & sol, std::vector<arma::cx_mat> & Forb, arma::vec & Eorb, bool can, bool fock, double pzweight, bool useref) {
  // Use reference
  sol=rsol;

  // List of changed orbitals
  std::vector<size_t> occlist, virtlist;
  if(arma::norm(x,2)!=0.0) {
    arma::cx_mat R(rotation(x,false));
    sol.cC=sol.cC*R;

    if(useref) {
      // Remove unity
      R-=arma::eye<arma::cx_mat>(sol.cC.n_cols,sol.cC.n_cols);
      // Find orbitals that have changed
      for(size_t ia=0;ia<oa;ia++)
	for(size_t ja=ia+1;ja<R.n_cols;ja++)
	  if(std::norm(R(ia,ja))>=CHANGETHR) {
	    occlist.push_back(ia);
	    if(ja<oa)
	      occlist.push_back(ja);
	    else
	      virtlist.push_back(ja);
	  }
    }
  }

  // Update density matrix
  arma::cx_mat P(2.0*form_density(sol.cC.cols(0,oa-1)));
  // Debug
  sol.P=arma::real(P);
  sol.P_im=arma::imag(P);

  // Clear out any old data
  Forb.clear();
  Eorb.clear();

  // Dummy occupation vector
  std::vector<double> occs(oa,2.0);

  // Build global Fock operator
  if(can && (!useref || (useref && virtlist.size()) ))
    solverp->Fock_RDFT(sol,occs,ovmethod,grid,nlgrid);

  if(pzweight==0.0)
    return sol.en.E;

  // Build the SI part
  arma::cx_mat CO;
  if(useref) {
    CO.zeros(sol.cC.n_rows,occlist.size());
    for(size_t i=0;i<occlist.size();i++)
      CO.col(i)=sol.cC.col(occlist[i]);

    std::vector<arma::cx_mat> Forb_hlp;
    arma::vec Eorb_hlp;
    solverp->PZSIC_Fock(Forb_hlp,Eorb_hlp,CO,oomethod,grid,nlgrid,fock);
    Eorb=ref_Eorb;
    for(size_t i=0;i<occlist.size();i++)
      Eorb(occlist[i])=Eorb_hlp(i);

    if(fock) {
      Forb=ref_Forb;
      for(size_t i=0;i<occlist.size();i++)
	Forb[occlist[i]]=Forb_hlp[i];
    }
  } else {
    CO=sol.cC.cols(0,oa-1);
    solverp->PZSIC_Fock(Forb,Eorb,CO,oomethod,grid,nlgrid,fock);
  }

  sol.en.Esic=-2.0*pzweight*arma::sum(Eorb);
  sol.en.E+=sol.en.Esic;
  
  return sol.en.E;
}

double PZStability::eval(const arma::vec & x, uscf_t & sol, std::vector<arma::cx_mat> & Forba, arma::vec & Eorba, std::vector<arma::cx_mat> & Forbb, arma::vec & Eorbb, bool can, bool fock, double pzweight, bool useref) {
  // Use reference
  sol=usol;

  // List of changed orbitals
  std::vector<size_t> occlista, occlistb, virtlista, virtlistb;
  if(arma::norm(x,2)!=0.0) {
    arma::cx_mat Ra(rotation(x,false));
    sol.cCa=sol.cCa*Ra;

    if(useref) {
      Ra-=arma::eye<arma::cx_mat>(sol.cCa.n_cols,sol.cCa.n_cols);
      for(size_t ia=0;ia<oa;ia++)
	for(size_t ja=ia+1;ja<Ra.n_cols;ja++)
	  if(std::norm(Ra(ia,ja))>=CHANGETHR) {
	    occlista.push_back(ia);
	    if(ja<oa)
	      occlista.push_back(ja);
	    else
	      virtlista.push_back(ja);
	  }
    }

    if(ob) {
      arma::cx_mat Rb(rotation(x,true));
      sol.cCb=sol.cCb*Rb;
      if(useref) {
	Rb-=arma::eye<arma::cx_mat>(sol.cCb.n_cols,sol.cCb.n_cols);
	for(size_t ib=0;ib<ob;ib++)
	  for(size_t jb=ib+1;jb<Rb.n_cols;jb++)
	    if(std::norm(Rb(ib,jb))>=CHANGETHR) {
	      occlistb.push_back(ib);
	      if(jb<ob)
		occlistb.push_back(jb);
	      else
		virtlistb.push_back(jb);
	    }
      }
    }
  }

  // Update density matrix
  {
    arma::cx_mat Pa(form_density(sol.cCa.cols(0,oa-1)));
    sol.Pa=arma::real(Pa);
    sol.Pa_im=arma::imag(Pa);
  }
  if(ob) {
    arma::cx_mat Pb(form_density(sol.cCb.cols(0,ob-1)));
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
  Forbb.clear();
  Eorbb.clear();

  // Dummy occupation vector
  std::vector<double> occa(oa,1.0);
  std::vector<double> occb(ob,1.0);

  // Build global Fock operator
  if(can && (!useref || (useref && (virtlista.size() || virtlistb.size())) ))
    solverp->Fock_UDFT(sol,occa,occb,ovmethod,grid,nlgrid);
  if(pzweight==0.0)
    return sol.en.E;

  // Build the SI part
  std::vector<arma::cx_mat> Forb;
  arma::vec Eorb;

  // Build the SI part
  arma::cx_mat CO;
  if(useref) {
    CO.zeros(sol.cCa.n_rows,occlista.size()+occlistb.size());
    for(size_t i=0;i<occlista.size();i++)
      CO.col(i)=sol.cCa.col(occlista[i]);
    for(size_t i=0;i<occlistb.size();i++)
      CO.col(i+occlista.size())=sol.cCb.col(occlistb[i]);

    solverp->PZSIC_Fock(Forb,Eorb,CO,oomethod,grid,nlgrid,fock);

    Eorba=ref_Eorba;
    for(size_t i=0;i<occlista.size();i++)
      Eorba(occlista[i])=Eorb(i);
    Eorbb=ref_Eorbb;
    for(size_t i=0;i<occlistb.size();i++)
      Eorbb(occlistb[i])=Eorb(i+occlista.size());

    if(fock) {
      Forba=ref_Forba;
      for(size_t i=0;i<occlista.size();i++)
	Forba[occlista[i]]=Forb[i];
      Forbb=ref_Forbb;
      for(size_t i=0;i<occlistb.size();i++)
	Forbb[occlistb[i]]=Forb[i+occlista.size()];
    }
  } else {
    CO.zeros(sol.cCa.n_rows,oa+ob);
    CO.cols(0,oa-1)=sol.cCa.cols(0,oa-1);
    if(ob)
      CO.cols(oa,oa+ob-1)=sol.cCb.cols(0,ob-1);
    solverp->PZSIC_Fock(Forb,Eorb,CO,oomethod,grid,nlgrid,fock);

    Eorba=Eorb.subvec(0,oa-1);
    if(ob)
      Eorbb=Eorb.subvec(oa,oa+ob-1);

    if(fock) {
      Forba.resize(oa);
      for(size_t i=0;i<oa;i++)
	Forba[i]=Forb[i];
      if(ob) {
	Forbb.resize(ob);
	for(size_t i=0;i<ob;i++)
	  Forbb[i]=Forb[i+oa];
      }
    }
  }

  // Result is
  sol.en.Esic=-pzweight*(arma::sum(Eorba)+arma::sum(Eorbb));
  sol.en.E+=sol.en.Esic;
  return sol.en.E;
}

arma::vec PZStability::gradient() {
  arma::vec x;
  x.zeros(count_params());
  return gradient(x, true);
}

arma::vec PZStability::precondition_unified(const arma::vec & g) const {
  // Search direction
  arma::vec sd(g);

  // Offset
  size_t ioff=0;

  if(restr) {
    // Occupied orbitals
    arma::cx_mat CO(rsol.cC.cols(0,oa-1));
    // Virtual orbitals
    arma::cx_mat CV;
    if(va)
      CV=rsol.cC.cols(oa,rsol.cC.n_cols-1);

    if(cancheck && va) {
      // Form OV gradient
      arma::cx_mat gOV(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa,va)-1),oa,va,real,imag));
      // Check
      arma::vec gs(g.subvec(ioff,ioff+count_ov_params(oa,va)-1));
      arma::vec gt(gather_ov(gOV,real,imag));

      // Preconditioning. Form unified Hamiltonian
      arma::cx_mat H(unified_H(CO,CV,ref_Forb,get_H(rsol)));

      arma::cx_mat Hoo(arma::trans(CO)*H*CO);
      arma::cx_mat Hvv(arma::trans(CV)*H*CV);

      arma::vec Eo;
      arma::cx_mat Co;
      eig_sym_ordered(Eo,Co,Hoo);

      arma::vec Ev;
      arma::cx_mat Cv;
      eig_sym_ordered(Ev,Cv,Hvv);

      // Minimum Hessian shift is
      double minH=1e-4;
      double dH=minH+std::max(arma::max(Eo)-arma::min(Ev),0.0);

      // Transform OV gradient into pseudocanonical space and perform preconditioning
      arma::cx_mat GOV(arma::trans(Co)*gOV*Cv);
      for(size_t io=0;io<CO.n_cols;io++)
	for(size_t iv=0;iv<CV.n_cols;iv++)
	  GOV(io,iv)/=Ev(iv)-Eo(io)+dH;

      // Transform back into the original frame
      GOV=Co*GOV*arma::trans(Cv);

      arma::vec POV(gather_ov(GOV,real,imag));
      if(POV.n_elem != count_ov_params(oa,va))
	throw std::logic_error("Amount of elements doesn't match!\n");
      sd.subvec(ioff,ioff+POV.n_elem-1)=POV;
      ioff+=POV.n_elem;
    }

  } else {
    arma::cx_mat COa(usol.cCa.cols(0,oa-1));
    arma::cx_mat COb;
    if(ob)
      COb=usol.cCb.cols(0,ob-1);
    arma::cx_mat CVa;
    if(usol.cCa.n_cols>oa)
      CVa=usol.cCa.cols(oa,usol.cCa.n_cols-1);
    arma::cx_mat CVb;
    if(usol.cCb.n_cols>ob)
      CVb=usol.cCb.cols(ob,usol.cCb.n_cols-1);

    if(cancheck && va) {
      // Preconditioning. Form unified Hamiltonian
      arma::cx_mat Ha(unified_H(COa,CVa,ref_Forba,get_H(usol,false)));
      arma::cx_mat Hb(unified_H(COb,CVb,ref_Forbb,get_H(usol,true)));

      arma::cx_mat Hooa(arma::trans(COa)*Ha*COa);
      arma::cx_mat Hvva(arma::trans(CVa)*Ha*CVa);
      arma::cx_mat Hoob;
      if(ob)
	Hoob=arma::trans(COb)*Hb*COb;
      arma::cx_mat Hvvb(arma::trans(CVb)*Hb*CVb);

      arma::vec Eoa;
      arma::cx_mat Coa;
      eig_sym_ordered(Eoa,Coa,Hooa);

      arma::vec Eob;
      arma::cx_mat Cob;
      if(ob)
	eig_sym_ordered(Eob,Cob,Hoob);

      arma::vec Eva;
      arma::cx_mat Cva;
      eig_sym_ordered(Eva,Cva,Hvva);

      arma::vec Evb;
      arma::cx_mat Cvb;
      eig_sym_ordered(Evb,Cvb,Hvvb);

      // Minimum Hessian shift is
      const double minH=1e-4;
      double dH=std::max(arma::max(Eoa)-arma::min(Eva),0.0);
      if(ob)
	dH=std::max(arma::max(Eob)-arma::min(Evb),dH);
      dH+=minH;

      // Transform OV gradient into pseudocanonical space and perform preconditioning
      arma::cx_mat gOVa(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa,va)-1),oa,va,real,imag));
      arma::cx_mat GOVa(arma::trans(Coa)*gOVa*Cva);
      for(size_t io=0;io<COa.n_cols;io++)
	for(size_t iv=0;iv<CVa.n_cols;iv++)
	  GOVa(io,iv)/=Eva(iv)-Eoa(io)+dH;
      // Transform back into the original frame
      GOVa=Coa*GOVa*arma::trans(Cva);

      arma::vec POVa(gather_ov(GOVa,real,imag));
      sd.subvec(ioff,ioff+POVa.n_elem-1)=POVa;
      ioff+=POVa.n_elem;
      if(POVa.n_elem != count_ov_params(oa,va))
	throw std::logic_error("Amount of elements doesn't match!\n");

      if(ob) {
	arma::cx_mat gOVb(spread_ov(g.subvec(ioff,ioff+count_ov_params(ob,vb)-1),ob,vb,real,imag));
	arma::cx_mat GOVb(arma::trans(Cob)*gOVb*Cvb);
	for(size_t io=0;io<COb.n_cols;io++)
	  for(size_t iv=0;iv<CVb.n_cols;iv++)
	    GOVb(io,iv)/=Evb(iv)-Eob(io)+dH;
	// Transform back into the original frame
	GOVb=Cob*GOVb*arma::trans(Cvb);

	arma::vec POVb(gather_ov(GOVb,real,imag));
	sd.subvec(ioff,ioff+POVb.n_elem-1)=POVb;
	ioff+=POVb.n_elem;
	if(POVb.n_elem != count_ov_params(ob,vb))
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

  if(restr) {
    // Occupied orbitals
    arma::cx_mat CO(rsol.cC.cols(0,oa-1));
    // Virtual orbitals
    arma::cx_mat CV;
    if(va)
      CV=rsol.cC.cols(oa,rsol.cC.n_cols-1);

    if(cancheck && va) {
      // OV orbital energy differences
      arma::mat dE(oa,va);
      for(size_t io=0;io<oa;io++) {
	// Orbital Hamiltonian is
	arma::cx_mat Fo(get_H(rsol));
	if(pzw!=0.0) Fo-=pzw*ref_Forb[io];
	// Occupied energy is
	double Eocc=std::real(arma::as_scalar(arma::trans(CO.col(io))*Fo*CO.col(io)));
	// Loop over virtuals
	for(size_t iv=0;iv<va;iv++) {
	  // Virtual energy is
	  double Evirt=std::real(arma::as_scalar(arma::trans(CV.col(iv))*Fo*CV.col(iv)));
	  // Store
	  dE(io,iv)=Evirt-Eocc;
	}
      }

      // Minimal Hessian shift is
      double minH=1e-4+std::max(-arma::min(arma::min(dE)),0.0);
      dE+=minH*arma::ones<arma::mat>(oa,va);

      // Form OV gradient
      arma::cx_mat gOV(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa,va)-1),oa,va,real,imag));

      // Run element-wise division
      arma::cx_mat GOV(gOV/dE);
      
      arma::vec POV(gather_ov(GOV,real,imag));
      if(POV.n_elem != count_ov_params(oa,va))
	throw std::logic_error("Amount of elements doesn't match!\n");
      sd.subvec(ioff,ioff+POV.n_elem-1)=POV;
      ioff+=POV.n_elem;
    }
  } else {
    arma::cx_mat COa(usol.cCa.cols(0,oa-1));
    arma::cx_mat COb;
    if(ob)
      COb=usol.cCb.cols(0,ob-1);
    arma::cx_mat CVa;
    if(usol.cCa.n_cols>oa)
      CVa=usol.cCa.cols(oa,usol.cCa.n_cols-1);
    arma::cx_mat CVb;
    if(usol.cCb.n_cols>ob)
      CVb=usol.cCb.cols(ob,usol.cCb.n_cols-1);

    if(cancheck && va) {
      // OV orbital energy differences
      arma::mat dEa(oa,va);
      for(size_t io=0;io<oa;io++) {
	// Orbital Hamiltonian is
	arma::cx_mat Fo(get_H(usol,false));
	if(pzw!=0.0) Fo-=pzw*ref_Forba[io];
	// Occupied energy is
	double Eocc=std::real(arma::as_scalar(arma::trans(COa.col(io))*Fo*COa.col(io)));
	// Loop over virtuals
	for(size_t iv=0;iv<va;iv++) {
	  // Virtual energy is
	  double Evirt=std::real(arma::as_scalar(arma::trans(CVa.col(iv))*Fo*CVa.col(iv)));
	  // Store
	  dEa(io,iv)=Evirt-Eocc;
	}
      }

      arma::mat dEb;
      if(ob) {
	dEb.zeros(ob,vb);
	for(size_t io=0;io<ob;io++) {
	  // Orbital Hamiltonian is
	  arma::cx_mat Fo(get_H(usol,true));
	  if(pzw!=0.0) Fo-=pzw*ref_Forbb[io];
	  // Occupied energy is
	  double Eocc=std::real(arma::as_scalar(arma::trans(COb.col(io))*Fo*COb.col(io)));
	  // Loop over virtuals
	  for(size_t iv=0;iv<vb;iv++) {
	    // Virtual energy is
	    double Evirt=std::real(arma::as_scalar(arma::trans(CVb.col(iv))*Fo*CVb.col(iv)));
	    // Store
	    dEb(io,iv)=Evirt-Eocc;
	  }
	}
      }
	    
      // Minimal Hessian shift is
      double minH=1e-4;
      if(ob)
	minH+=std::max(std::max(-arma::min(arma::min(dEa)),-arma::min(arma::min(dEb))),0.0);
      else
	minH+=std::max(-arma::min(arma::min(dEa)),0.0);

      // Shift
      dEa+=minH*arma::ones<arma::mat>(oa,va);
      if(ob)
	dEb+=minH*arma::ones<arma::mat>(ob,vb);

      // Form OV gradient
      arma::cx_mat gOVa(spread_ov(g.subvec(ioff,ioff+count_ov_params(oa,va)-1),oa,va,real,imag));

      // Run element-wise division
      arma::cx_mat GOVa(gOVa/dEa);
      
      arma::vec POVa(gather_ov(GOVa,real,imag));
      if(POVa.n_elem != count_ov_params(oa,va))
	throw std::logic_error("Amount of elements doesn't match!\n");
      sd.subvec(ioff,ioff+POVa.n_elem-1)=POVa;
      ioff+=POVa.n_elem;

      if(ob) {
	// Form OV gradient
	arma::cx_mat gOVb(spread_ov(g.subvec(ioff,ioff+count_ov_params(ob,vb)-1),ob,vb,real,imag));
	
	// Run element-wise division
	arma::cx_mat GOVb(gOVb/dEb);
	
	arma::vec POVb(gather_ov(GOVb,real,imag));
	if(POVb.n_elem != count_ov_params(ob,vb))
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

  if(restr) {
    size_t ioff=0;

    // Evaluate orbital matrices
    rscf_t sol;
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    eval(x,sol,Forb,Eorb,cancheck,true,pzw,ref);

    // Occupied orbitals
    arma::cx_mat CO(sol.cC.cols(0,oa-1));
    // Virtual orbitals
    arma::cx_mat CV;
    if(va)
      CV=sol.cC.cols(oa,sol.cC.n_cols-1);

    if(cancheck && va) {
      // Hamiltonian is
      arma::cx_mat H(get_H(sol));
      // OV gradient is
      arma::cx_mat gOV(oa,va);
      if(pzw==0.0)
	gOV=-arma::strans(arma::trans(CV.cols(0,va-1))*arma::conj(H)*CO.cols(0,oa-1));
      else
	for(size_t i=0;i<oa;i++) {
	  arma::cx_vec hlp(arma::conj(H-pzw*Forb[i])*CO.col(i));
	  for(size_t a=0;a<va;a++)
	    gOV(i,a)=-arma::cdot(CV.col(a),hlp);
	}
      
      // Convert to proper gradient
      gOV=gradient_convert(gOV);

      // Collect values
      arma::vec pOV(gather_ov(gOV,real,imag));
      g.subvec(ioff,ioff+pOV.n_elem-1)=pOV;
      ioff+=pOV.n_elem;
    }

    if(oocheck && oa>1) {
      // OO gradient is
      arma::cx_mat gOO(oa,oa);
      if(pzw!=0.0) {
	arma::cx_mat FO(CO.n_rows,oa);
	for(size_t i=0;i<oa;i++)
	  FO.col(i)=arma::conj(Forb[i])*CO.col(i);
	gOO=-pzw*arma::strans(-arma::trans(CO)*FO + arma::trans(FO)*CO);
      } else
	gOO.zeros();

      // Convert to proper gradient
      gOO=gradient_convert(gOO);

      // Collect values
      arma::vec pOO(gather_oo(gOO,real,imag));
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
    eval(x,sol,Forba,Eorba,Forbb,Eorbb,cancheck,true,pzw,ref);

    // Occupied orbitals
    arma::cx_mat COa=sol.cCa.cols(0,oa-1);
    arma::cx_mat COb;
    if(ob)
      COb=sol.cCb.cols(0,ob-1);
    // Virtuals
    arma::cx_mat CVa;
    if(va)
      CVa=sol.cCa.cols(oa,oa+va-1);
    arma::cx_mat CVb;
    if(vb)
      CVb=sol.cCb.cols(ob,ob+vb-1);

    size_t ioff=0;

    if(cancheck && va) {
      // Hamiltonian is
      arma::cx_mat Ha(get_H(sol,false));
      
      // OV alpha gradient is
      arma::cx_mat gOVa(oa,va);
      if(pzw==0.0)
	gOVa=-arma::strans(arma::trans(CVa.cols(0,va-1))*arma::conj(Ha)*COa.cols(0,oa-1));
      else
	for(size_t i=0;i<oa;i++) {
	  arma::cx_vec hlp(arma::conj(Ha-pzw*Forba[i])*COa.col(i));
	  for(size_t a=0;a<va;a++)
	    gOVa(i,a)=-arma::cdot(CVa.col(a),hlp);
	}

      // Convert to proper gradient
      gOVa=gradient_convert(gOVa);

      // Collect values
      arma::vec pOVa(gather_ov(gOVa,real,imag));
      g.subvec(ioff,ioff+pOVa.n_elem-1)=pOVa;
      ioff+=pOVa.n_elem;

      if(ob && vb) {
	// Hamiltonian is
	arma::cx_mat Hb(get_H(sol,true));
	
	// OV beta gradient is
	arma::cx_mat gOVb(ob,vb);
	if(pzw==0.0)
	  gOVb=-arma::strans(arma::trans(CVb.cols(0,vb-1))*arma::conj(Hb)*COb.cols(0,ob-1));
	else
	  for(size_t i=0;i<ob;i++) {
	    arma::cx_vec hlp(arma::conj(Hb-pzw*Forbb[i])*COb.col(i));
	    for(size_t a=0;a<vb;a++)
	      gOVb(i,a)=-arma::cdot(CVb.col(a),hlp);
	  }

	// Convert to proper gradient
	gOVb=gradient_convert(gOVb);

	// Collect values
	arma::vec pOVb(gather_ov(gOVb,real,imag));
	g.subvec(ioff,ioff+pOVb.n_elem-1)=pOVb;
	ioff+=pOVb.n_elem;
      }
    }

    if(oocheck) {
      if(oa>1) {
	// OO alpha gradient is
	arma::cx_mat gOOa(oa,oa);
	if(pzw!=0.0) {
	  arma::cx_mat FOa(COa.n_rows,oa);
	  for(size_t i=0;i<oa;i++)
	    FOa.col(i)=arma::conj(Forba[i])*COa.col(i);
	  gOOa=-pzw*arma::strans(-arma::trans(COa)*FOa + arma::trans(FOa)*COa);
	} else
	  gOOa.zeros();

	// Convert to proper gradient
	gOOa=gradient_convert(gOOa);

	// Collect values
	arma::vec pOOa(gather_oo(gOOa,real,imag));
	g.subvec(ioff,ioff+pOOa.n_elem-1)=pOOa;
	ioff+=pOOa.n_elem;
      }

      if(ob>1) {
	// OO beta gradient is
	arma::cx_mat gOOb(ob,ob);
	if(pzw!=0.0) {
	  arma::cx_mat FOb(COb.n_rows,ob);
	  for(size_t i=0;i<ob;i++)
	    FOb.col(i)=arma::conj(Forbb[i])*COb.col(i);
	  gOOb=-pzw*arma::strans(-arma::trans(COb)*FOb + arma::trans(FOb)*COb);
	} else
	  gOOb.zeros();

	// Convert to proper gradient
	gOOb=gradient_convert(gOOb);

	// Collect values
	arma::vec pOOb(gather_oo(gOOb,real,imag));
	g.subvec(ioff,ioff+pOOb.n_elem-1)=pOOb;
	ioff+=pOOb.n_elem;
      }
    }
  }

  /*
  arma::vec gn(FDHessian::gradient(x));
  fflush(stdout);
  if(rms_norm(g-gn)>1e-6) {
    g.t().print("Analytic gradient");
    gn.t().print("Numeric gradient");
    (g-gn).t().print("Difference");    
    printf("Problem in analytic gradient, difference norm %e.\n",arma::norm(g-gn,2));
    if(arma::norm(g+gn,2)<arma::norm(g-gn,2))
      printf("Derivative has wrong sign? |g+gn|=%e, |g-gn|=%e\n",arma::norm(g+gn,2),arma::norm(g-gn,2));
    //throw std::runtime_error("Problem in gradient\n");
  } else
    printf("Analytic gradient is OK, difference norm %e\n",arma::norm(g-gn,2));
  */
  
  // Use numerical gradient
  //g=gn;
  
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
    x(i)=ss_fd;
    arma::vec gr=gradient(x,true);

    // LHS value
    x(i)=-ss_fd;
    arma::vec gl=gradient(x,true);

    // Finite difference derivative is
    for(size_t j=0;j<npar;j++) {
      h(i,j)=(gr(j)-gl(j))/(2.0*ss_fd);

      if(std::isnan(h(i,j))) {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Element (" << i << "," << j <<") of hessian gives NaN.\n";
	oss << "Step size is " << ss_fd << ", and left and right values are " << gl(j) << " and " << gr(j) << ".\n";
	throw std::runtime_error(oss.str());
      }
    }
  }

  // Symmetrize Hessian to distribute numerical error evenly
  h=(h+arma::trans(h))/2.0;

  return h;
}

double PZStability::eval(const arma::vec & x) {
  if(restr) {
    rscf_t sol;
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    return eval(x,sol,Forb,Eorb,cancheck,false,pzw,false);
  } else {
    uscf_t sol;
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    return eval(x,sol,Forba,Eorba,Forbb,Eorbb,cancheck,false,pzw,false);
  }
}

double PZStability::get_E() {
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

  double ival=eval(x0);
  printf("Initial value is % .10f\n",ival);

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
      //printf("Projection of preconditioned search direction on gradient is %e.\n",arma::norm_dot(sd,-g));
      printf("Projection of preconditioned search direction on gradient is %e, not using preconditioning.\n",arma::norm_dot(sd,-g));
      sd=-g;
    }

    if(arma::norm(g,2) < nrthr && !cancheck) {
      // Evaluate Hessian
      Timer tp;
      printf("Calculating Hessian ... "); fflush(stdout);
      arma::mat h(hessian());
      printf("done (%s)\n",tp.elapsed().c_str()); fflush(stdout);

      // Run eigendecomposition
      arma::vec hval;
      arma::mat hvec;
      bool diagok=arma::eig_sym(hval,hvec,h);
      if(!diagok)
	throw std::runtime_error("Error diagonalizing orbital Hessian\n");
      hval.t().print("Hessian eigenvalues");

      // Enforce positive defitiveness
      hval+=std::max(0.0,-arma::min(hval))+1e-4;

      // Form new search direction: sd = - H^-1 g
      sd.zeros(hvec.n_rows);
      for(size_t i=0;i<hvec.n_cols;i++)
	sd-=arma::dot(hvec.col(i),g)/hval(i)*hvec.col(i);

      // Backtracking line search
      double Etr=eval(sd);
      printf(" %e % .10f\n",1.0,Etr);
      fflush(stdout);

      double tau=0.7;
      double Enew=eval(tau*sd);
      printf(" %e % .10f\n",tau,Enew);
      fflush(stdout);

      double l=1.0;
      while(Enew<Etr) {
	Etr=Enew;
	l*=tau;
	Enew=eval(l*tau*sd);
	printf(" %e % .10f backtrack\n",l*tau,Enew);
	fflush(stdout);
      }

      printf("Newton step changed value by %e\n",Etr-E0);
      fflush(stdout);

      update(l*sd);
      x0+=l*sd;
      if(fabs(Etr-E0)<dEthr)
	break;

      // Accept move
      E0=Etr;
      parallel_transport(g,sd,l);
      continue;

    } else if(!cancheck) { // Use BFGS in OO optimization
      // New search direction
      arma::vec sd0(sd);
      sd=-lbfgs.solve();

      // Check sanity
      if(arma::dot(sd,-g)<0) {
	printf("Bad BFGS direction, dot product % e. BFGS reset\n",arma::dot(sd,-g)/arma::dot(g,g));
	lbfgs.clear();
	lbfgs.update(x0,g);
	sd=-lbfgs.solve();

      } else if(iiter>=1 && arma::dot(g,gold)>=0.2*arma::dot(gold,gold)) {
	printf("Powell restart - SD step\n");
	sd=sd0;

      } else {
	printf("BFGS step\n");
	printf("Projection of search direction onto steepest descent direction is %e\n",arma::dot(sd,-g)/sqrt(arma::dot(sd,sd)*arma::dot(g,g)));
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
	if(arma::dot(sdnew,-g)<=0)
	  // This would take us into the wrong direction!
	  printf("Bad CG direction. SD step\n");
	else {
	  // Update search direction
	  sd=sdnew;
	  printf("CG step\n");
	}
      } else printf("SD step\n");
    }

    // Derivative is
    double dE=arma::dot(sd,g);

    printf(" %e % .10f\n",0.0,E0);
    fflush(stdout);

    // Update step size
    update_step(sd);
    
    // Initial step size. Don't go too far so that the parabolic
    // approximation is valid
    //double d= cancheck ? Tmu/25.0 : Tmu/5.0;
    double d=Tmu/5.0;
    // Value at initial step
    double Ed=eval(d*sd);
    printf(" %e % .10f\n",d,Ed);
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
	printf(" %e % .10f, % e difference from prediction\n",step,Es,Es-Ep);
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
	printf(" %e % .10f backtrack\n",step,Es);
	fflush(stdout);
	if(Es>Es0 && Es<E0)
	  break;
      }
      // Overstepped
      step/=tau;
      Es=Es0;
    }

    printf("Line search changed value by %e\n",Es-E0);
    update(step*sd);
    x0+=step*sd;
    if(fabs(Es-E0)<dEthr)
      break;

    // Parallel transport the gradient in the search direction
    E0=Es;
    parallel_transport(g,sd,step);
  }

  printf("Final value is % .10f; optimization changed value by %e\n",E0,E0-ival);
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
  if(restr || ob==0) {
    // Form the rotation matrix
    arma::cx_mat R(rotation(sd*step,false));
    // Form the G matrix
    arma::cx_mat G(rotation_pars(gold,false));
    // Transform G
    G=arma::trans(R)*G*R;

    // Collect the parameters
    size_t ioff=0;
    if(cancheck) {
      arma::vec pOV(gather_ov(G.submat(0,oa,oa-1,oa+va-1),real,imag));
      gold.subvec(ioff,ioff+pOV.n_elem-1)=pOV;
      ioff+=pOV.n_elem;
    }
    if(oocheck) {
      arma::vec pOO(gather_oo(G.submat(0,0,oa-1,oa-1),real,imag));
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
    if(cancheck) {
      arma::vec pOVa(gather_ov(Ga.submat(0,oa,oa-1,oa+va-1),real,imag));
      gold.subvec(ioff,ioff+pOVa.n_elem-1)=pOVa;
      ioff+=pOVa.n_elem;
      arma::vec pOVb(gather_ov(Gb.submat(0,ob,ob-1,ob+vb-1),real,imag));
      gold.subvec(ioff,ioff+pOVb.n_elem-1)=pOVb;
      ioff+=pOVb.n_elem;
    }
    if(oocheck) {
      arma::vec pOOa(gather_oo(Ga.submat(0,0,oa-1,oa-1),real,imag));
      gold.subvec(ioff,ioff+pOOa.n_elem-1)=pOOa;
      ioff+=pOOa.n_elem;
      if(ob>1) {
	arma::vec pOOb(gather_oo(Gb.submat(0,0,ob-1,ob-1),real,imag));
	gold.subvec(ioff,ioff+pOOb.n_elem-1)=pOOb;
	ioff+=pOOb.n_elem;
      }
    }
  }
}

inline void orthonormalize(const arma::mat & S, arma::cx_mat & C) {
  // Orbital overlap
  arma::cx_mat So(arma::trans(C)*S*C);
  // Difference from orthonormality
  arma::cx_mat dS=So-arma::eye<arma::cx_mat>(So.n_rows,So.n_cols);
  double d=arma::norm(dS,2);
  if(d>=1e-9) {
    printf("Difference from orbital orthonormality is %e, orthonormalizing\n",d);
    orthonormalize(S,C);
  } else {
    //printf("Difference from orbital orthonormality is %e, OK\n",d);
  }
}

      
void PZStability::update(const arma::vec & x) {
  if(arma::norm(x,2)!=0.0)  {
    if(restr) {
      arma::cx_mat R=rotation(x,false);
      rsol.cC=rsol.cC*R;
    } else {
      arma::cx_mat Ra=rotation(x,false);
      usol.cCa=usol.cCa*Ra;
      if(ob) {
	arma::cx_mat Rb=rotation(x,true);
	usol.cCb=usol.cCb*Rb;
      }
    }
  }

  // Check that orbitals are orthonormal and reorthonormalize if
  // necessary
  if(true) {
    arma::mat S(basis.overlap());
    if(restr) {
      orthonormalize(S,rsol.cC);
    } else {
      orthonormalize(S,usol.cCa);
      orthonormalize(S,usol.cCb);
    }
  }

  // Update reference, without sort
  update_reference(false);

  // Update orbitals in checkpoint file
  Checkpoint *chkptp=solverp->get_checkpoint();
  if(restr) {
    chkptp->cwrite("CW",rsol.cC);
    chkptp->write(rsol.en);
    chkptp->write("P",rsol.P);
    chkptp->write("P_im",rsol.P_im);
  } else {
    chkptp->cwrite("CWa",usol.cCa);
    chkptp->write(usol.en);
    if(ob)
      chkptp->cwrite("CWb",usol.cCb);
    chkptp->write("P",usol.P);
    chkptp->write("Pa",usol.Pa);
    chkptp->write("Pb",usol.Pb);

    arma::mat P_im(arma::imag(usol.Pa+usol.Pb));
    chkptp->write("P_im",P_im);
  }
}

arma::cx_mat PZStability::get_H(const rscf_t & sol) const {
  arma::cx_mat H=sol.H*COMPLEX1;
  if(sol.K_im.n_rows == sol.H.n_rows && sol.K_im.n_cols == sol.H.n_cols)
    H-=0.5*sol.K_im*COMPLEXI;
  return H;
}

arma::cx_mat PZStability::get_H(const uscf_t & sol, bool spin) const {
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

  printf("Updating reference ... ");
  fflush(stdout);
  Timer t;

  if(restr) {
    rscf_t sol;
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    eval(x0,sol,Forb,Eorb,true,true,pzw,false);
    
    if(sort) {
      arma::cx_mat CO(sol.cC.cols(0,oa-1));
      arma::cx_mat CV;
      if(sol.cC.n_cols>oa)
	CV=sol.cC.cols(oa,sol.cC.n_cols-1);
      // Unified Hamiltonian
      arma::cx_mat H(unified_H(CO,CV,Forb,get_H(sol)));

      // Calculate projected orbital energies
      arma::vec Eorbo=arma::real(arma::diagvec(arma::trans(CO)*H*CO));
      arma::vec Eorbv;
      if(CV.n_cols)
	Eorbv=arma::real(arma::diagvec(arma::trans(CV)*H*CV));
      // Sort in ascending order
      arma::uvec idxo=arma::stable_sort_index(Eorbo,"ascend");

      // Store reference
      rsol=sol;
      for(arma::uword i=0;i<idxo.n_elem;i++)
	rsol.cC.col(i)=CO.col(idxo(i));
      if(CV.n_cols) {
	arma::uvec idxv=arma::stable_sort_index(Eorbv,"ascend");
	for(arma::uword i=0;i<idxv.n_elem;i++)
	  rsol.cC.col(i+oa)=CV.col(idxv(i));
      }
      if(pzw!=0.0) {
	ref_Eorb.zeros(Eorb.n_elem);
	for(size_t i=0;i<idxo.n_elem;i++)
	  ref_Eorb(i)=Eorb(idxo(i));
	ref_Forb.resize(Forb.size());
	for(size_t i=0;i<idxo.n_elem;i++)
	  ref_Forb[i]=Forb[idxo(i)];
      } else {
	ref_Eorb.clear();
	ref_Forb.clear();
      }
    } else {
      // Store reference
      rsol=sol;
      ref_Eorb=Eorb;
      ref_Forb=Forb;
    }

  } else {
    uscf_t sol;
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    eval(x0,sol,Forba,Eorba,Forbb,Eorbb,true,true,pzw,false);

    if(sort) {
      arma::cx_mat COa(sol.cCa.cols(0,oa-1));
      arma::cx_mat COb;
      if(ob)
	COb=sol.cCb.cols(0,ob-1);
      arma::cx_mat CVa;
      if(sol.cCa.n_cols>oa)
	CVa=sol.cCa.cols(oa,sol.cCa.n_cols-1);
      arma::cx_mat CVb;
      if(sol.cCb.n_cols>ob)
	CVb=sol.cCb.cols(ob,sol.cCb.n_cols-1);

      // Unified Hamiltonians
      arma::cx_mat Ha(unified_H(COa,CVa,Forba,get_H(sol,false)));
      arma::cx_mat Hb(unified_H(COb,CVb,Forbb,get_H(sol,true)));

      // Calculate projected orbital energies
      arma::vec Eorbao=arma::real(arma::diagvec(arma::trans(COa)*Ha*COa));
      arma::vec Eorbav;
      if(CVa.n_cols)
	Eorbav=arma::real(arma::diagvec(arma::trans(CVa)*Ha*CVa));
      arma::vec Eorbbo;
      if(ob)
	Eorbbo=arma::real(arma::diagvec(arma::trans(COb)*Hb*COb));
      arma::vec Eorbbv;
      if(CVb.n_cols)
	Eorbbv=arma::real(arma::diagvec(arma::trans(CVb)*Hb*CVb));

      // Sort in ascending order
      arma::uvec idxao=arma::stable_sort_index(Eorbao,"ascend");

      usol=sol;
      for(size_t i=0;i<idxao.n_elem;i++)
	usol.cCa.col(i)=COa.col(idxao(i));
      if(CVa.n_cols) {
	arma::uvec idxav=arma::stable_sort_index(Eorbav,"ascend");
	for(arma::uword i=0;i<idxav.n_elem;i++)
	  usol.cCa.col(i+oa)=CVa.col(idxav(i));
      }

      if(pzw!=0.0) {
	ref_Eorba.zeros(oa);
	for(size_t i=0;i<idxao.n_elem;i++)
	  ref_Eorba(i)=Eorba(idxao(i));
	ref_Forba.resize(oa);
	for(size_t i=0;i<idxao.n_elem;i++)
	  ref_Forba[i]=Forba[idxao(i)];
      } else {
	ref_Eorba.clear();
	ref_Forba.clear();
      }

      if(ob) {
	arma::uvec idxbo=arma::stable_sort_index(Eorbbo,"ascend");
	for(arma::uword i=0;i<idxbo.n_elem;i++)
	  usol.cCb.col(i)=COb.col(idxbo(i));

	if(pzw!=0.0) {
	  ref_Eorbb.zeros(ob);
	  for(size_t i=0;i<idxbo.n_elem;i++)
	    ref_Eorbb(i)=Eorbb(idxbo(i));
	  ref_Forbb.resize(ob);
	  for(size_t i=0;i<idxbo.n_elem;i++)
	    ref_Forbb[i]=Forbb[idxbo(i)];
	} else {
	  ref_Eorbb.clear();
	  ref_Forbb.clear();
	}
      }
      if(CVb.n_cols) {
	arma::uvec idxbv=arma::stable_sort_index(Eorbbv,"ascend");
	for(arma::uword i=0;i<idxbv.n_elem;i++)
	  usol.cCb.col(i+ob)=CVb.col(idxbv(i));
      }
    } else {
      // Store reference
      usol=sol;
      ref_Eorba=Eorba;
      ref_Forba=Forba;
      ref_Eorbb=Eorbb;
      ref_Forbb=Forbb;
    }
  }

  printf("done (%s)\n",t.elapsed().c_str());
  fflush(stdout);
}

arma::cx_mat PZStability::rotation(const arma::vec & x, bool spin) const {
  // Get rotation matrix
  arma::cx_mat X(rotation_pars(x,spin));

  // Rotation matrix
  arma::cx_mat R(X);
  R.eye();
  if(oocheck && !cancheck) {
    // It suffices to just exponentiate the OO block
    size_t o=spin ? ob : oa;
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
  if(spin && restr) {
    ERROR_INFO();
    throw std::runtime_error("Incompatible arguments.\n");
  }

  // Amount of occupied and virtual orbitals
  size_t o=oa, v=va;
  if(spin) {
    o=ob;
    v=vb;
  }

  // Construct full, padded rotation matrix
  arma::cx_mat R(o+v,o+v);
  R.zeros();

  // OV part
  if(cancheck) {
    size_t ioff0=0;
    if(spin)
      ioff0=count_ov_params(oa,va);

    if(v) {
      arma::cx_mat r(spread_ov(x.subvec(ioff0,ioff0+count_ov_params(o,v)-1),o,v,real,imag));
      R.submat(0,o,o-1,o+v-1)=r;
      R.submat(o,0,o+v-1,o-1)=-arma::trans(r);
    }
  }

  // OO part
  if(oocheck && o>1) {
    size_t ioff0=0;
    // Canonical rotations
    if(cancheck) {
      ioff0=count_ov_params(oa,va);
      if(!restr)
	ioff0+=count_ov_params(ob,vb);
    }
    // Occupied rotations
    if(spin)
      ioff0+=count_oo_params(oa);

    // Get the rotation matrix
    arma::cx_mat r(spread_oo(x.subvec(ioff0,ioff0+count_oo_params(o)-1),o,real,imag));
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

void PZStability::set_method(const dft_t & ovmethod_, const dft_t & oomethod_, double pzw_) {
  ovmethod=ovmethod_;
  oomethod=oomethod_;
  pzw=pzw_;

  Checkpoint *chkptp=solverp->get_checkpoint();
  chkptp->read(basis);
  grid=DFTGrid(&basis,true,ovmethod.lobatto);
  nlgrid=DFTGrid(&basis,false,ovmethod.lobatto);

  // Range separation constants
  double omega, kfull, kshort;
  range_separation(ovmethod.x_func,omega,kfull,kshort);
  
  if(omega!=0.0) {
    printf("\nUsing range separated exchange with range separation constant omega = % .3f.\n",omega);
    printf("Using % .3f %% short range and % .3f %% long range exchange.\n",(kfull+kshort)*100,kfull*100);
  } else if(kfull!=0.0)
    printf("\nUsing hybrid exchange with % .3f %% of exact exchange.\n",kfull*100);
  else
    printf("\nA pure exchange functional used, no exact exchange.\n");

  // Compute range separated integrals if necessary
  if(is_range_separated(ovmethod.x_func))
    solverp->fill_rs(omega);
}

void PZStability::set_params(bool real_, bool imag_, bool can, bool oo) {
  real=real_;
  imag=imag_;
  cancheck=can;
  oocheck=oo;

  std::vector<std::string> truth(2);
  truth[0]="false";
  truth[1]="true";
  fprintf(stderr,"oo = %s, ov = %s, real = %s, imag = %s\n",truth[oocheck].c_str(),truth[cancheck].c_str(),truth[real].c_str(),truth[imag].c_str());
  fprintf(stderr,"There are %i parameters.\n",(int) count_params());
}

void PZStability::set(const rscf_t & sol) {
  Checkpoint *chkptp=solverp->get_checkpoint();

  chkptp->read(basis);

  // Update solution
  rsol=sol;

  // Update size parameters
  restr=true;
  int Na;
  chkptp->read("Nel-a",Na);
  ob=oa=Na;
  va=vb=rsol.cC.n_cols-oa;

  chkptp->write("Restricted",1);

  std::vector<std::string> truth(2);
  truth[0]="false";
  truth[1]="true";

  fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i\n",(int) oa, (int) ob, (int) va, (int) vb);

  // Reconstruct DFT grid
  update_grid(true);
  // Update reference
  update_reference(true);
}

void PZStability::update_grid(bool init) {
  if(ovmethod.adaptive) {
    arma::cx_mat Ctilde;
    if(restr)
      Ctilde=rsol.cC.cols(0,oa-1);
    else {
      Ctilde.zeros(usol.cCa.n_rows,oa+ob);
      Ctilde.cols(0,oa-1)=usol.cCa.cols(0,oa-1);
      if(ob)
	Ctilde.cols(oa,oa+ob-1)=usol.cCb.cols(0,ob-1);
    }
    if (ovmethod.x_func>0 || ovmethod.c_func>0)
      grid.construct(Ctilde,ovmethod.gridtol,ovmethod.x_func,ovmethod.c_func);
  } else if(init) {
    bool strict(solverp->get_strictint());
    if (ovmethod.x_func>0 || ovmethod.c_func>0)
      grid.construct(ovmethod.nrad,ovmethod.lmax,ovmethod.x_func,ovmethod.c_func,strict);
    if(ovmethod.nl)
      nlgrid.construct(ovmethod.nlnrad,ovmethod.nllmax,true,false,strict,true);
  }
}
    
void PZStability::set(const uscf_t & sol) {
  Checkpoint *chkptp=solverp->get_checkpoint();

  // Update solution
  usol=sol;

  // Update size parameters
  restr=false;
  int Na, Nb;
  chkptp->read("Nel-a",Na);
  chkptp->read("Nel-b",Nb);
  oa=Na;
  ob=Nb;
  va=usol.cCa.n_cols-oa;
  vb=usol.cCb.n_cols-ob;

  chkptp->write("Restricted",0);
  fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i\n",(int) oa, (int) ob, (int) va, (int) vb);
  fflush(stderr);

  // Reconstruct DFT grid
  update_grid(true);
  // Update reference
  update_reference(true);
}

rscf_t PZStability::get_rsol() const {
  return rsol;
}

uscf_t PZStability::get_usol() const {
  return usol;
}

bool PZStability::check(bool stability, double cutoff) {
  Timer tfull;

  if(!count_params())
    return false;
  
  // Estimate runtime
  {
    double ttot=0.0;

    Timer t;
    arma::vec x(count_params());

    if(cancheck) {
      // OV part
      if(cancheck && vb) {
        x.zeros();
	x(0)=ss_fd;
	t.set();
	gradient(x,true);
	double dt=t.get();

	// Total number of OV calculations is
	if(restr)
	  ttot+=2*dt*count_ov_params(oa,va);
	else
	  ttot+=2*dt*(count_ov_params(oa,va)+count_ov_params(ob,vb));
      }
    }
    if(oocheck && oa>1) {
      x.zeros();
      size_t ioff0=0;
      // Canonical rotations
      if(cancheck) {
	ioff0=count_ov_params(oa,va);
	if(!restr)
	  ioff0+=count_ov_params(ob,vb);
      }
      x(ioff0)=ss_fd;
      t.set();
      gradient(x,true);
      double dt=t.get();

      // Total number of OO calculations is
      if(restr)
	ttot+=2*dt*count_oo_params(oa);
      else
	ttot+=2*dt*(count_oo_params(oa)+count_oo_params(ob));
    }

    // Total time is
    fprintf(stderr,"\nComputing the Hessian will take approximately %s\n",t.parse(ttot).c_str());
    fflush(stderr);
  }

  // Evaluate Hessian
  Timer t;
  arma::mat h(hessian());
  printf("Hessian evaluated (%s)\n",t.elapsed().c_str()); fflush(stdout);
  t.set();

  // Block the degrees of freedom
  std::vector<pz_rot_par_t> dof(classify());
  // Block-diagonalize Hessian
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
    printf("Full Hessian diagonalized in %s.\n",tdiag.elapsed().c_str());

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
      double ds=ss_ls;
      const double dfac=cbrt(10.0);

      double Enew=eval(x*ds);
      printf("\t%e % .10f % e\n",ds,Enew,Enew-Ei);

      while(true) {
	ds*=dfac;
	E0=Enew;
	Enew=eval(x*ds);
	printf("\t%e % .10f % e\n",ds,Enew,Enew-Ei);
	fflush(stdout);
	if(Enew>E0)
	  break;
      }
      // Overstepped
      ds/=dfac;

      // Update solution
      x*=ds;

      // Update solution
      update(x);
      printf("Stability analysis decreased energy by %e\n",E0-Ei);
    }
  }

  fprintf(stderr,"Check completed in %s.\n",tfull.elapsed().c_str());

  // Found instabilities?
  return stability && I.n_cols>0;
}

void PZStability::print_status(size_t iiter, const arma::vec & g, const Timer & t) const {
  printf("\nIteration %i, gradient norm (%s):\n",(int) iiter,t.elapsed().c_str());

  // Get decomposition
  std::vector<pz_rot_par_t> dof(classify());
  for(size_t i=0;i<dof.size();i++) {
    arma::vec gs(dof[i].idx.n_elem);
    for(size_t k=0;k<dof[i].idx.n_elem;k++)
      gs(k)=g(dof[i].idx(k));

    printf("%20s %e %e\n",dof[i].name.c_str(),arma::norm(gs,2),arma::norm(gs,"inf"));
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
  double dx=Tmu/Np;
  for(int i=-Np;i<=Np;i++) {
    printf("x = %e\n",i*dx);
    fprintf(out,"%e % e\n",i*dx,eval(i*dx*g));
    fflush(out);
  }
  fclose(out);
}
