/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "energy-opt.h"
#include "unitary.h"
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

EnergyOptimizer::EnergyOptimizer(const std::string & elv, const arma::ivec & am, const arma::uvec & nfv, const arma::uvec & nparv, const arma::uvec & optshv) {
  el=elv;
  sham=am;
  nf=nfv;
  npar=nparv;
  optsh=optshv;

  fd_h=1e-5;
  ls_h=1e-3;
#ifdef _OPENMP
  ntr=(size_t) omp_get_max_threads();
#else
  ntr=1;
#endif

  if(am.n_elem!=nf.n_elem)
    throw std::runtime_error("sham and nf are not of same size!\n");
  if(nf.n_elem!=npar.n_elem)
    throw std::runtime_error("nf and npar are not of same size!\n");
  if(nf.n_elem!=optsh.n_elem)
    throw std::runtime_error("nf and optsh are not of same size!\n");
  for(arma::uword i=0;i<npar.n_elem;i++) {
    if(npar(i)>nf(i)) {
      std::ostringstream oss;
      oss << "The " << shell_types[i] << " shell has more parameters than functions!\n";
      throw std::runtime_error(oss.str());
    }
    if(npar(i)==1 && nf(i)>1) {
      std::ostringstream oss;
      oss << "The " << shell_types[i] << " shell has more than one function but only a single parameter!\n";
      throw std::runtime_error(oss.str());
    }
  }
  for(arma::uword i=0;i<optsh.n_elem;i++)
    if(optsh(i))
      optsh(i)=1;
}

EnergyOptimizer::~EnergyOptimizer() {
}

std::vector<arma::vec> EnergyOptimizer::get_exps(const arma::vec & x) const {
  // Exponent vector
  std::vector<arma::vec> exps(arma::max(arma::abs(sham))+1);

  size_t ioff=0;
  for(arma::uword i=0;i<nf.n_elem;i++) {
    // Get exponents
    arma::vec shexp=legendre_set(x.subvec(ioff,ioff+npar(i)-1),nf(i));

    // Concatenate to full set
    size_t minam=sham(i);
    if(sham[i]<0)
      minam=0;

    for(size_t am=minam;am<=abs(sham(i));am++) {
      if(exps[am].n_elem) {
	// Get original exponents
	arma::vec ox=exps[am];

	// Resize list
	exps[am].zeros(ox.n_elem+shexp.n_elem);
	exps[am].subvec(0,ox.n_elem-1)=ox;
	exps[am].subvec(ox.n_elem,exps[am].n_elem-1)=shexp;

	exps[am]=sort(exps[am],"descend");
      } else
	exps[am]=shexp;
    }

    ioff+=npar(i);
  }

  return exps;
}

arma::uvec EnergyOptimizer::idx_vec() const {
  // Get a parameter index vector.
  // First count the amount of parameters
  arma::uword n=arma::dot(optsh,npar);

  // Index vector
  arma::uvec idx(n);
  arma::uword ioff=0;
  arma::uword j=0;
  for(arma::uword i=0;i<npar.n_elem;i++) {
    if(optsh(i)) {
      for(arma::uword ip=0;ip<npar(i);ip++)
	idx(j++)=ioff+ip;
    }
    ioff+=npar(i);
  }

  return idx;
}

arma::vec EnergyOptimizer::pad_vec(const arma::vec & sd) const {
  arma::vec x(arma::sum(npar));
  x.zeros();

  arma::uvec idx(idx_vec());
  for(arma::uword i=0;i<idx.n_elem;i++)
    x(idx(i))=sd(i);

  return x;
}

arma::vec EnergyOptimizer::calcG(const arma::vec & x) {
  // Parameter indices
  arma::uvec idx=idx_vec();

  // Gradient
  arma::vec g(idx.n_elem);

  // Form trials
  std::vector<BasisSetLibrary> trials;
  for(size_t i=0;i<g.n_elem;i++) {
    // RH value
    arma::vec xrh(x);
    xrh(idx(i))+=fd_h;
    trials.push_back(form_basis(xrh));

    // LH value
    arma::vec xlh(x);
    xlh(idx(i))-=fd_h;
    trials.push_back(form_basis(xlh));
  }

  // Get energies
  std::vector<double> Etr=calcE(trials);

  // Get gradients
  for(size_t i=0;i<g.n_elem;i++) {
    g(i)=(Etr[2*i]-Etr[2*i+1])/(2*fd_h);

    // Check that gradient is sane
    if(!std::isnormal(g(i)))
      g(i)=0.0;
  }

  return g;
}

BasisSetLibrary EnergyOptimizer::form_basis(const arma::vec & x) const {
  return form_basis(get_exps(x));
}

BasisSetLibrary EnergyOptimizer::form_basis(const std::vector<arma::vec> & exps) const {

  // Form element basis set
  ElementBasisSet elbas(el);
  for(size_t am=0;am<exps.size();am++)
    for(size_t ix=0;ix<exps[am].size();ix++) {
      FunctionShell sh(am);
      sh.add_exponent(1.0,exps[am][ix]);
      elbas.add_function(sh);
    }

  // Returned basis set
  BasisSetLibrary baslib;
  baslib.add_element(elbas);
  return baslib;
}


double EnergyOptimizer::optimize(arma::vec & x, size_t maxiter) {
  // Current and old gradient
  arma::vec g, gold;

  // Search direction
  arma::vec sd;

  // Energy value
  double Eval=0.0;
  // Initial energy
  double Einit=0.0;

  print_info(x,"Input parameters");

  // Minimization loop
  for(size_t iloop=0;iloop<maxiter;iloop++) {
    // Get gradient
    gold=g;
    g=calcG(x);

    printf("Iteration %i gradient, norm %e\n",(int) iloop+1,arma::norm(g,2));
    g.t().print();

    if(arma::norm(g,2)<1e-5)
      break;

    // Update search direction
    if(iloop % x.n_elem == 0) {
      // Steepest descent
      sd=-g;

      printf("Using SD step.\n");
    } else {
      // Update factor
      double gamma;

      // Polak-Ribiere
      gamma=arma::dot(g,g-gold)/arma::dot(gold,gold);
      // Fletcher-Reeves
      //gamma=arma::dot(g,g)/arma::dot(gold,gold);

      // Update search direction
      sd=-g + gamma*sd;

      printf("Using CG step.\n");
    }

    std::vector<double> step, E;

    // Trial step sizes
    std::vector<double> stepbatch;
    stepbatch.push_back(0.0);

    // Index of optimum
    size_t optidx;

    while(true) {
      // Generate new trials
      while(stepbatch.size() < ntr) {
	// Step size is
	double ss=std::pow(2.0,step.size()+stepbatch.size()-1)*ls_h;
	stepbatch.push_back(ss);
      }

      // Generate basis sets
      std::vector<BasisSetLibrary> basbatch(stepbatch.size());
      for(size_t i=0;i<stepbatch.size();i++)
	basbatch[i]=form_basis(x+stepbatch[i]*sd);

      // Get trial energies
      std::vector<double> batchE=calcE(basbatch);

      // Print out info
      double refE=(E.size()==0) ? batchE[0] : E[0];
      for(size_t i=0;i<batchE.size();i++)
	printf("%2i %e % .10e % e\n",(int) (i+step.size()),stepbatch[i],batchE[i],batchE[i]-refE);

      // Initial energy?
      if(E.size()==0)
	Einit=batchE[0];
      // Add to total list
      for(size_t i=0;i<batchE.size();i++) {
	step.push_back(stepbatch[i]);
	E.push_back(batchE[i]);
      }
      stepbatch.clear();

      // Check if converged
      bool convd=false;
      for(size_t i=1;i<E.size();i++)
	if(E[i]>E[i-1]) {
	  optidx=i-1;
	  //	  printf("Converged, optidx = %i\n",(int) optidx);
	  convd=true;
	  break;
	}
      if(convd)
	break;
    }

    // Search for best step size
    if(optidx==0) {
      printf("Error in line search - could not go downhill!\n");
      break;
    }

    // Step size to use
    double ss=0.0;

    bool fit=false;

    if(fit && (!std::isnormal(E[optidx+1]) || E[optidx+1]==DBL_MAX)) {
      printf("Odd energy detected\n");
      if(optidx==1) {
	fit=false;
      } else
	optidx--;
    }

    if(fit) {
      // Optimal energy
      Eval=E[optidx];

      // Do parabola fit
      arma::vec px(3), py(3);

      px(0)=step[optidx-1];
      px(1)=step[optidx];
      px(2)=step[optidx+1];

      py(0)=E[optidx-1];
      py(1)=E[optidx];
      py(2)=E[optidx+1];

      // Fit to polynomial of order p
      arma::vec coeff=fit_polynomial(px,py);

      // Find out zeros of the derivative polynomial
      arma::vec roots=solve_roots(derivative_coefficients(coeff));
      // and use the smallest positive one
      ss=smallest_positive(roots);

      // Check sanity
      if(ss<step[optidx-1]) {
	ss=step[optidx];
	printf("Renormalized step size to %e\n",ss);
      }
    } else {
	ss=step[optidx];
    }

    printf("Optimal step size is %e, reducing the energy by %e.\n",ss,E[optidx]-E[0]);

    x+=pad_vec(ss*sd);

    print_info(x,"Current parameters");
  }

  printf("Optimization reduced the energy by %e.\n",Eval-Einit);

  return Eval;
}

double EnergyOptimizer::scan(arma::vec & x, double min, double max, double dx) {
  // Polarization exponent search
  size_t Ntr=ceil((max-min)/dx);
  std::vector<BasisSetLibrary> trbas(Ntr);
  // Sanity check
  if(npar(npar.n_elem-1)!=1)
    throw std::runtime_error("Scan requires a single parameter on the last shell!\n");

  printf("Scanning from % e to % e, with %i trials.\n",min,max,(int) Ntr);

  for(size_t i=0;i<Ntr;i++) {
    // Dummy vector
    arma::vec xdum(x);
    xdum(xdum.n_elem-1)=min+i*dx;
    trbas[i]=form_basis(xdum);
  }
  // Value w/o polarization function
  {
    std::vector<arma::vec> exps=get_exps(x);
    exps.erase(exps.end()-1);
    trbas.push_back(form_basis(exps));
  }

  // Get energies
  std::vector<double> trE=calcE(trbas);

  printf("\nValue w/o polarization shell % .10e\n",trE[Ntr]);
  printf("Trial energies\n");
  for(size_t i=0;i<Ntr;i++)
    printf("%3i/%-3i % e % .10e % e\n",(int) i+1,(int) trE.size(),min+i*dx,trE[i],trE[i]-trE[Ntr]);

  // Get minimum
  arma::vec E=arma::conv_to<arma::vec>::from(trE);
  arma::uword minind;
  double Emin=E.min(minind);

  printf("Polarization shell reduces energy by % .10e to % .10e\n",Emin-trE[Ntr],Emin);
  printf("\n");

  // Store value
  x(x.n_elem-1)=min+minind*dx;

  return Emin-trE[Ntr];
}

void EnergyOptimizer::print_info(const arma::vec & x, const std::string & msg) const {
  x.t().print(msg);

  // Print current exponents
  std::vector<arma::vec> exps=get_exps(x);
  for(size_t am=0;am<exps.size();am++) {
    printf("%c exponents\n",shell_types[am]);
    arma::vec ame=arma::log10(exps[am]);

    for(arma::uword ix=0;ix<ame.n_elem;ix++)
      printf("% .3f ",ame(ix));
    printf("\n");
  }
  printf("\n");
}
