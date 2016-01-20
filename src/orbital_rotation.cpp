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
#include "orbital_rotation.h"
#include "checkpoint.h"
#include "stringutil.h"
#include "linalg.h"
#include "lbfgs.h"
#include "timer.h"
#include "mathf.h"
#include "dftfuncs.h"

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

OrbitalRotation::OrbitalRotation() {
}

OrbitalRotation::~OrbitalRotation() {
}

size_t OrbitalRotation::count_params(size_t i, size_t j) const {
  if(i<j)
    throw std::logic_error("i<j!\n");
  
  size_t o(orbgroups[i].n_elem);
  size_t v(orbgroups[j].n_elem);
  
  size_t n=0;
  if(i==j) {
    if(real)
      n+=o*(o-1)/2;
    if(imag)
      n+=o*(o-1)/2;
  } else {
    if(real)
      n+=o*v;
    if(imag)
      n+=o*v;
  }

  return n;
}

size_t OrbitalRotation::count_params() const {
  if(enabled.n_rows != orbgroups.size() || enabled.n_cols != orbgroups.size())
    throw std::logic_error("OrbitalRotation class not initialized properly.\n");
  
  // Amount of dofs
  size_t n=0;
  for(size_t i=0;i<orbgroups.size();i++)
    for(size_t j=i;j<orbgroups.size();j++) {
      if(!enabled(i,j))
	continue;
      
      n+=count_params(i,j);
    }
  
  return n;
}

arma::vec OrbitalRotation::collect(const std::vector<arma::cx_mat> & Gs) const {
  // Returned vector
  arma::vec g(count_params());
  // Running sum
  size_t ig=0;
  // G index
  size_t iG=0;

  for(size_t i=0;i<orbgroups.size();i++)
    for(size_t j=i;j<orbgroups.size();j++) {
      if(!enabled(i,j))
	continue;

      // Get matrix and convert it to gradient
      arma::cx_mat G(gradient_convert(Gs[iG++]));

      // Get parameters in block
      arma::vec p = (i==j) ? gather_oo(G,real,imag) : gather_ov(G,real,imag);

      // Store parameters
      g.subvec(ig,ig+p.n_elem-1) = p;
      ig+=p.n_elem;
    }

  return g;
}

std::vector<arma::cx_mat> OrbitalRotation::spread(const arma::vec & p) const {
  // Returned array
  std::vector<arma::cx_mat> Gs;

  // Parameter index offset
  size_t ip=0;
  
  for(size_t i=0;i<orbgroups.size();i++)
    for(size_t j=i;j<orbgroups.size();j++) {
      if(!enabled(i,j))
	continue;

      size_t o(orbgroups[i].n_elem);
      size_t v(orbgroups[j].n_elem);

      // Amount of parameters in this subblock
      size_t np(count_params(i,j));
      
      // Gradient matrix
      arma::cx_mat G= (i==j) ? spread_oo(p.subvec(ip,ip+np-1),o,real,imag) : spread_ov(p.subvec(ip,ip+np-1),o,v,real,imag);
      Gs.push_back(G);

      // Increment offset
      ip+=np;
    }

  return Gs;
}

void OrbitalRotation::update(const arma::vec & p) {
  C=rotate(p);
}

double OrbitalRotation::evaluate_step(const arma::vec & g, int n) const {
  arma::cx_mat G(rotation_pars(g));
  
  // Calculate eigendecomposition
  arma::vec Gval;
  arma::cx_mat Gvec;
  bool diagok=arma::eig_sym(Gval,Gvec,-COMPLEXI*G);
  if(!diagok) {
    ERROR_INFO();
    throw std::runtime_error("Error diagonalizing G.\n");
  }

  // Calculate maximum step size; cost function is n:th order in parameters
  return 2.0*M_PI/(n*arma::max(arma::abs(Gval)));
}

double OrbitalRotation::evaluate_step(const arma::vec & g) const {
  return evaluate_step(g,4);
}

std::vector<orb_rot_par_t> OrbitalRotation::classify() const {
  std::vector<orb_rot_par_t> ret;
  size_t n=0;
  
  for(size_t i=0;i<orbgroups.size();i++)
    for(size_t j=i;j<orbgroups.size();j++) {
      if(!enabled(i,j))
	continue;

      // Amount of parameters in block
      size_t np(count_params(i,j));
      if(!np)
	continue;
      
      orb_rot_par_t entry;
      entry.name=orblegend[i] + " - " + orblegend[j];

      if(real && !imag) {
	entry.name="real " + entry.name;
	entry.idx=arma::linspace<arma::uvec>(n,n+np-1,np);
	  ret.push_back(entry);
      } else if(!real && imag) {
	entry.name="imag " + entry.name;
	entry.idx=arma::linspace<arma::uvec>(n,n+np-1,np);
	ret.push_back(entry);
      } else if(real && imag) {
	orb_rot_par_t rentry;
	rentry.name="real " + entry.name;
	rentry.idx=arma::linspace<arma::uvec>(n,n+np/2-1,np/2);
	ret.push_back(rentry);

	orb_rot_par_t ientry;
	ientry.name="imag " + entry.name;
	ientry.idx=arma::linspace<arma::uvec>(n+np/2,n+np-1,np/2);
	ret.push_back(ientry);
	
	// Full
	entry.idx=arma::linspace<arma::uvec>(n,n+np-1,np);
	ret.push_back(entry);
      }
    }

  return ret;
}
  
arma::cx_mat OrbitalRotation::rotation_pars(const arma::vec & p) const {
  // Rotation parameter matrix
  arma::cx_mat R(C.n_cols,C.n_cols);
  R.zeros();
  
  // Fill out matrix
  std::vector<arma::cx_mat> Gs(spread(p));
  
  size_t iG=0;
  
  for(size_t i=0;i<orbgroups.size();i++)
    for(size_t j=i;j<orbgroups.size();j++) {
      if(!enabled(i,j))
	continue;
      
      R(orbgroups[i],orbgroups[j])=Gs[iG];
      if(i!=j)
	R(orbgroups[j],orbgroups[i])=-arma::trans(Gs[iG]);
      
      iG++;
    }

  return R;
}

arma::cx_mat OrbitalRotation::rotate(const arma::vec & p) const {
  // Rotation parameter matrix
  arma::cx_mat R(rotation_pars(p));
  
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
  
  return C*rot;
}

arma::vec OrbitalRotation::gradient() {
  return collect(block_gradient());
}

arma::vec OrbitalRotation::gradient(const arma::vec & x) {
  return collect(block_gradient(x));
}

arma::mat OrbitalRotation::hessian() {
  // Amount of parameters
  size_t npar(count_params());
  
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
    arma::vec gr(gradient(x));
    
    // LHS value
    x(i)=-ss_fd;
    arma::vec gl(gradient(x));
    
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

double OrbitalRotation::optimize(size_t maxiter, double gthr, double nrthr, double dEthr, int preconditioning) {
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

    // Update search direction
    arma::vec oldsd(sd);
    sd=precondition(-g);

    if(arma::norm_dot(sd,-g)<0.0) {
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
