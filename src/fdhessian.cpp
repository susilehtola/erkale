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

#include "fdhessian.h"
#include <cfloat>

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
