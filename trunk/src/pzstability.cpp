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
#include "linalg.h"
#include "timer.h"
#include "mathf.h"

// Mode to evaluate gradient and hessian in: 0 for full calculation, 1 for wrt reference. (-1 inside eval is for reference update)
#define GHMODE 1

FDHessian::FDHessian() {
  // Rotation step size
  ss=cbrt(DBL_EPSILON);
}

FDHessian::~FDHessian() {
}

arma::vec FDHessian::gradient() {
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
    arma::vec x(npar);
    x.zeros();

    // LHS value
    x(i)=-ss;
    double yl=eval(x,GHMODE);

    // RHS value
    x(i)=ss;
    double yr=eval(x,GHMODE);

    // Derivative
    g(i)=(yr-yl)/(2*ss);

    if(std::isnan(g(i))) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Element " << i << " of gradient gives NaN.\n";
      oss << "Step size is " << ss << ", and left and right values are " << yl << " and " << yr << ".\n";
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
    x(i)+=ss;
    x(j)+=ss;
    double yrr=eval(x,GHMODE);

    // RH,LH
    x.zeros();
    x(i)+=ss;
    x(j)-=ss;
    double yrl=eval(x,GHMODE);

    // LH,RH
    x.zeros();
    x(i)-=ss;
    x(j)+=ss;
    double ylr=eval(x,GHMODE);

    // LH,LH
    x.zeros();
    x(i)-=ss;
    x(j)-=ss;
    double yll=eval(x,GHMODE);

    // Values
    h(i,j)=(yrr - yrl - ylr + yll)/(4.0*ss*ss);
    // Symmetrize
    h(j,i)=h(i,j);

    if(std::isnan(h(i,j))) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Element (" << i << "," << j << ") of Hessian gives NaN.\n";
      oss << "Step size is " << ss << ". Stencil values\n";
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
  printf("\nIteration %i, gradient norm %e (%s)\n",(int) iiter,arma::norm(g,2),t.elapsed().c_str());
}

void FDHessian::optimize(size_t maxiter, double gthr, bool max) {
  arma::vec x0(count_params());
  x0.zeros();

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
    double initstep=1e-4;
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
      sd+=gamma*oldsd;

      printf("CG step\n");
    } else printf("SD step\n");    

    while(true) {
      step.push_back(std::pow(stepfac,step.size())*initstep);
      val.push_back(eval(step[step.size()-1]*sd));

      if(val.size()>=2)
	printf(" %e % .10f % e\n",step[step.size()-1],val[val.size()-1],val[val.size()-1]-val[val.size()-2]);
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
}

PZStability::PZStability(SCF * solver, dft_t dft) {
  solverp=solver;
  solverp->set_verbose(false);

  method=dft;

  cplx=true;
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

  n+=o*(o-1)/2;
  if(cplx)
    n+=o*(o+1)/2;

  return n;
}

size_t PZStability::count_ov_params(size_t o, size_t v) const {
  size_t n=0;
  // Real part
  n+=o*v;
  if(cplx)
    // Complex part
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

void PZStability::real_imag_idx(arma::uvec & idxr, arma::uvec & idxi) const {
  if(!cplx) {
    ERROR_INFO();
    throw std::runtime_error("Should not call real_imag_idx for purely real calculation!\n");
  }
  
  // Count amount of parameters
  size_t nreal=0, nimag=0;
  if(cancheck) {
    nreal+=oa*va;
    nimag+=oa*va;
    if(!restr) {
      nreal+=ob*vb;
      nimag+=ob*vb;
    }
  }
  if(oocheck) {
    nreal+=oa*(oa-1)/2;
    if(cplx)
      nimag+=oa*(oa+1)/2;
    
    if(!restr) {
      nreal+=ob*(ob-1)/2;
      if(cplx)
	nimag+=ob*(ob+1)/2;
    }
  }

  // Sanity check
  if(nreal+nimag != count_params()) {
    ERROR_INFO();
    throw std::runtime_error("Parameter count is wrong!\n");
  }
  
  // Parameter indices
  idxr.zeros(nreal);
  idxi.zeros(nimag);
  
  // Fill indices.
  size_t ir=0, ii=0;
  
  // Offset
  size_t ioff=0;
  
  if(cancheck) {
    // First are the real parameters
    for(size_t irot=0;irot<oa*va;irot++) {
      idxr(ir++)=irot;
    }
    ioff+=oa*va;
    // followed by the imaginary parameters
    for(size_t irot=0;irot<oa*va;irot++)
      idxi(ii++)=irot+ioff;
    ioff+=oa*va;
    
    if(!restr) {
      // and then again the real parameters
      for(size_t irot=0;irot<ob*vb;irot++)
	idxr(ir++)=irot + ioff;
      ioff+=ob*vb;
      // followed by the imaginary parameters
      for(size_t irot=0;irot<ob*vb;irot++)
	idxi(ii++)=irot + ioff;
      ioff+=ob*vb;
    }
  }
    
  if(oocheck) {
    // First are the real parameters
    for(size_t irot=0;irot<oa*(oa-1)/2;irot++) {
      idxr(ir++)=irot+ioff;
    }
    ioff+=oa*(oa-1)/2;
    // and then the imaginary parameters
    for(size_t irot=0;irot<oa*(oa+1)/2;irot++)
      idxi(ii++)=irot + ioff;
    ioff+=oa*(oa+1)/2;

    if(!restr) {
      // First are the real parameters
      for(size_t irot=0;irot<ob*(ob-1)/2;irot++) {
	idxr(ir++)=irot+ioff;
      }
      ioff+=ob*(ob-1)/2;
      // and then the imaginary parameters
      for(size_t irot=0;irot<ob*(ob+1)/2;irot++)
	idxi(ii++)=irot + ioff;
      ioff+=ob*(ob+1)/2;
    }
  }

  // Sanity check
  arma::uvec idx(nreal+nimag);
  idx.subvec(0,nreal-1)=idxr;
  idx.subvec(nreal,nreal+nimag-1)=idxi;
  idx=arma::sort(idx,"ascending");
  for(size_t i=0;i<idx.n_elem;i++)
    if(idx(i)!=i) {
      std::ostringstream oss;
      oss << "Element " << i << " of compound index is wrong: " << idx(i) << "!\n";
      throw std::runtime_error(oss.str());
    }
}

double PZStability::eval(const arma::vec & x) {
  return eval(x,false);
}
  
std::vector<size_t> check_ov(const arma::cx_mat & Rov, double rotcut) {
  // Get list of changed occupied orbitals
  std::vector<size_t> chkorb;
  for(size_t o=0;o<Rov.n_rows;o++)
    for(size_t v=0;v<Rov.n_cols;v++)
      if(std::abs(Rov(o,v))>=rotcut) {
	chkorb.push_back(o);
	break;
      }
  return chkorb;
}

static void add_to_list(std::vector<size_t> & list, size_t num) {
  bool found=false;
  for(size_t i=0;i<list.size();i++)
    if(list[i]==num) {
      found=true;
      break;
    }
  if(!found)
    list.push_back(num);
}

static void check_oo(std::vector<size_t> & chkorb, const arma::cx_mat & Roo, double rotcut) {
  // Get list of changed occupied orbitals
  for(size_t o1=0;o1<Roo.n_rows;o1++)
    for(size_t o2=0;o2<o1;o2++)
      if(std::abs(Roo(o1,o2))>=rotcut) {
	// Add both orbitals to the list
	add_to_list(chkorb,o1);
	add_to_list(chkorb,o2);
      }
}

double PZStability::eval(const arma::vec & x, int mode) {
  double focktol=ROUGHTOL;

  // Rotation cutoff
  double rotcut=10*DBL_EPSILON;
  
  if(restr) {
    rscf_t tmp(rsol);

    // List of occupieds to check
    std::vector<size_t> chkorb;
    if(mode==-1 || mode == 0) {
      for(size_t o=0;o<oa;o++)
	chkorb.push_back(o);
    }
    
    // Get rotation matrix
    arma::cx_mat Rov;
    if(cancheck) {
      // ov rotation matrix
      Rov=ov_rotation(x,false);
      // Get list of changed occupied orbitals
      if(mode==1)
	chkorb=check_ov(Rov.submat(0,oa,oa-1,Rov.n_cols-1),rotcut);
    }
    
    // Do we need to do the reference part?
    if(chkorb.size() || mode==-1 || mode==0) {
      if(cancheck) {
	// Rotate orbitals
	tmp.cC=tmp.cC*Rov;
	
	// Update density matrix
	tmp.P=2.0*arma::real(tmp.cC.cols(0,oa-1)*arma::trans(tmp.cC.cols(0,oa-1)));
      }
      
      // Dummy occupation vector
      std::vector<double> occa(oa,2.0);

      // Build global Fock operator
      solverp->Fock_RDFT(tmp,occa,method,grid,nlgrid,focktol);

      // Update reference energy
      if(mode==-1)
	ref_E0=tmp.en.E;
      
    } else if(mode==1) {
      // No, we don't. Set energy
      tmp.en.E=ref_E0;
    } else {
      ERROR_INFO();
      throw std::runtime_error("Shouldn't be here!\n");
    }
    
    // Get oo rotation
    arma::cx_mat Roo;
    if(oocheck) {
      Roo=oo_rotation(x,false);
      if(mode==1)
	check_oo(chkorb,Roo,rotcut);
    } else
      Roo.eye(oa,oa);

    // Orbital SI energies
    arma::vec Eo(ref_Eo);

    // Do we need to do anything for the oo part?
    if(chkorb.size() || mode == 0 || mode == -1) {
      // Collect list of changed occupied orbitals
      arma::uvec orblist(arma::sort(arma::conv_to<arma::uvec>::from(chkorb)));
      
      // Transformed oo block
      arma::cx_mat Ct=tmp.cC.cols(0,oa-1)*Roo;
      // Dummy matrix
      arma::cx_mat Rdum=arma::eye(orblist.n_elem,orblist.n_elem)*COMPLEX1;

      // Build the SI part
      std::vector<arma::cx_mat> Forb;
      arma::vec Eorb;
      solverp->PZSIC_Fock(Forb,Eorb,Ct.cols(orblist),Rdum,method,grid,nlgrid,false);

      if(mode==1) {
	for(size_t i=0;i<orblist.n_elem;i++)
	  Eo(orblist(i))=Eorb(i);
      } else {
	Eo=Eorb;

	if(mode==-1)
	  // Update reference
	  ref_Eo=Eorb;
      }
    }

    // Account for spin
    return tmp.en.E - 2.0*arma::sum(Eo);

  } else {
    uscf_t tmp(usol);

    // List of occupieds to check
    std::vector<size_t> chkorba, chkorbb;
    if(mode==-1 || mode==0) {
      for(size_t o=0;o<oa;o++)
	chkorba.push_back(o);
      for(size_t o=0;o<ob;o++)
	chkorbb.push_back(o);
    }
      
    // Get rotation matrix
    arma::cx_mat Rova, Rovb;
    if(cancheck) {
      // ov rotation matrix
      Rova=ov_rotation(x,false);
      Rovb=ov_rotation(x,true);
      // Get list of changed occupied orbitals
      if(mode==1) {
	chkorba=check_ov(Rova.submat(0,oa,oa-1,oa+va-1),rotcut);
	chkorbb=check_ov(Rovb.submat(0,ob,ob-1,ob+vb-1),rotcut);
      }
    }
    
    // Do we need to do the reference part?
    if(chkorba.size() || chkorbb.size() || mode==-1 || mode==0) {
      if(cancheck) {
	// Rotate orbitals
	tmp.cCa=tmp.cCa*Rova;
	tmp.cCb=tmp.cCb*Rovb;
	
	// Update density matrix
	tmp.Pa=arma::real(tmp.cCa.cols(0,oa-1)*arma::trans(tmp.cCa.cols(0,oa-1)));
	tmp.Pb=arma::real(tmp.cCb.cols(0,ob-1)*arma::trans(tmp.cCb.cols(0,ob-1)));
	tmp.P=tmp.Pa+tmp.Pb;
      }
      
      // Dummy occupation vector
      std::vector<double> occa(oa,1.0);
      std::vector<double> occb(ob,1.0);
      
      // Build global Fock operator
      solverp->Fock_UDFT(tmp,occa,occb,method,grid,nlgrid,focktol);

      // Update reference energy
      if(mode==-1)
	ref_E0=tmp.en.E;
      
    } else if(mode==1) {
      // No, we don't. Set energy
      tmp.en.E=ref_E0;
    } else {
      ERROR_INFO();
      throw std::runtime_error("Shouldn't be here!\n");
    }
    
    // Get oo rotation
    arma::cx_mat Rooa, Roob;
    if(oocheck) {
      Rooa=oo_rotation(x,false);
      Roob=oo_rotation(x,true);

      if(mode==1) {
	check_oo(chkorba,Rooa,rotcut);
	check_oo(chkorbb,Roob,rotcut);
      }
    } else {
      Rooa.eye(oa,oa);
      Roob.eye(ob,ob);
    }
    
    // Orbital SI energies
    arma::vec Eoa(ref_Eoa), Eob(ref_Eob);

    // Do we need to do anything for the oo part?
    if(chkorba.size() && (mode == 0 || mode == -1)) {
      // Collect list of changed occupied orbitals
      arma::uvec orblist(arma::sort(arma::conv_to<arma::uvec>::from(chkorba)));
      
      // Transformed oo block
      arma::cx_mat Ct=tmp.cCa.cols(0,oa-1)*Rooa;
      // Dummy matrix
      arma::cx_mat Rdum=arma::eye(orblist.n_elem,orblist.n_elem)*COMPLEX1;
      
      // Build the SI part
      std::vector<arma::cx_mat> Forb;
      arma::vec Eorb;
      solverp->PZSIC_Fock(Forb,Eorb,Ct.cols(orblist),Rdum,method,grid,nlgrid,false);

      // Collect energies
      if(mode==1) {
	for(size_t i=0;i<orblist.n_elem;i++)
	  Eoa(orblist(i))=Eorb(i);
      } else {
	Eoa=Eorb;

	if(mode==-1)
	  // Update reference
	  ref_Eoa=Eorb;
      }
    }
    if(chkorbb.size() && (mode == 0 || mode == -1)) {
      // Collect list of changed occupied orbitals
      arma::uvec orblist(arma::sort(arma::conv_to<arma::uvec>::from(chkorbb)));
      
      // Transformed oo block
      arma::cx_mat Ct=tmp.cCb.cols(0,ob-1)*Roob;
      // Dummy matrix
      arma::cx_mat Rdum=arma::eye(orblist.n_elem,orblist.n_elem)*COMPLEX1;
      
      // Build the SI part
      std::vector<arma::cx_mat> Forb;
      arma::vec Eorb;
      solverp->PZSIC_Fock(Forb,Eorb,Ct.cols(orblist),Rdum,method,grid,nlgrid,false);
      
      // Collect energies
      if(mode==1) {
	for(size_t i=0;i<orblist.n_elem;i++)
	  Eob(orblist(i))=Eorb(i);
      } else {
	Eob=Eorb;
	
	if(mode==-1)
	  // Update reference
	  ref_Eob=Eorb;
      }
    }
    
    // Result is
    return tmp.en.E-arma::sum(Eoa)-arma::sum(Eob);
  }
}

void PZStability::update(const arma::vec & x) {
  if(cancheck) {
    // Perform ov rotation
    if(restr) {
      arma::cx_mat Rov=ov_rotation(x,false);
      rsol.cC=rsol.cC*Rov;
    } else {
      arma::cx_mat Rova=ov_rotation(x,false);
      arma::cx_mat Rovb=ov_rotation(x,true);
      usol.cCa=usol.cCa*Rova;
      usol.cCb=usol.cCb*Rovb;
    }
  }

  if(oocheck) {
    // Perform oo rotation
    if(restr) {
      arma::cx_mat Roo=oo_rotation(x,false);
      rsol.cC.cols(0,oa-1)=rsol.cC.cols(0,oa-1)*Roo;
    } else {
      arma::cx_mat Rooa=oo_rotation(x,false);
      arma::cx_mat Roob=oo_rotation(x,true);
      usol.cCa.cols(0,oa-1)=usol.cCa.cols(0,oa-1)*Rooa;
      usol.cCb.cols(0,ob-1)=usol.cCb.cols(0,ob-1)*Roob;
    }
  }

  // Update orbitals in checkpoint file
  Checkpoint *chkptp=solverp->get_checkpoint();
  if(restr)
    chkptp->cwrite("CW",rsol.cC.cols(0,oa-1));
  else {
    chkptp->cwrite("CWa",usol.cCa.cols(0,oa-1));
    chkptp->cwrite("CWb",usol.cCb.cols(0,ob-1));   
  }
  
  // Update reference
  arma::vec x0(count_params());
  x0.zeros();
  eval(x0,true);
}

arma::cx_mat PZStability::ov_rotation(const arma::vec & x, bool spin) const {
  if(x.n_elem != count_params()) {
    ERROR_INFO();
    throw std::runtime_error("Inconsistent parameter size.\n");
  }
  if(spin && restr) {
    ERROR_INFO();
    throw std::runtime_error("Incompatible arguments.\n");
  }
  if(!cancheck)
    throw std::runtime_error("ov_rotation called even though canonical orbitals are not supposed to be checked!\n");

  // Amount of occupied orbitals
  size_t o=oa, v=va;
  if(spin) {
    o=ob;
    v=vb;
  }
  // Rotation matrix
  arma::cx_mat rot(o,v);

  // Calculate offset
  size_t ioff0=0;
  if(spin)
    ioff0=count_ov_params(oa,va);

  // Collect real part of rotation
  arma::mat rr(o,v);
  rr.zeros();
  for(size_t i=0;i<o;i++)
    for(size_t j=0;j<v;j++) {
      rr(i,j)=x(i*v + j + ioff0);
    }

  // Full rotation
  arma::cx_mat r(o,v);
  if(!cplx) {
    r=rr*COMPLEX1;
  } else {
    // Imaginary part of rotation.
    arma::mat ir(o,v);
    ir.zeros();
    // Offset
    size_t ioff=o*v + ioff0;

    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<v;j++) {
	ir(i,j)=x(i*v + j + ioff);
      }

    // Matrix
    r=rr*COMPLEX1 + ir*COMPLEXI;
  }

  // Construct full, padded rotation matrix
  arma::cx_mat R(o+v,o+v);
  R.zeros();
  R.submat(0,o,o-1,o+v-1)=r;
  R.submat(o,0,o+v-1,o-1)=-arma::trans(r);

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
  arma::cx_mat Rov(Rvec*arma::diagmat(arma::exp(COMPLEXI*Rval))*arma::trans(Rvec));

  arma::cx_mat prod=arma::trans(Rov)*Rov-arma::eye(Rov.n_cols,Rov.n_cols);
  double norm=rms_cnorm(prod);
  if(norm>=sqrt(DBL_EPSILON))
    throw std::runtime_error("Matrix is not unitary!\n");

  return Rov;  
}

arma::cx_mat PZStability::oo_rotation(const arma::vec & x, bool spin) const {
  if(x.n_elem != count_params()) {
    ERROR_INFO();
    throw std::runtime_error("Inconsistent parameter size.\n");
  }
  if(spin && restr) {
    ERROR_INFO();
    throw std::runtime_error("Incompatible arguments.\n");
  }

  // Amount of occupied orbitals
  size_t o=oa;
  if(spin)
    o=ob;

  // Calculate offset
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

  // Collect real part of rotation
  arma::mat kappa(o,o);
  kappa.zeros();
  for(size_t i=0;i<o;i++)
    for(size_t j=0;j<i;j++) {
      size_t idx=i*(i-1)/2 + j + ioff0;
      kappa(i,j)=x(idx);
      kappa(j,i)=-x(idx);
    }

  // Rotation matrix
  arma::cx_mat R;
  if(!cplx)
    R=kappa*COMPLEX1;
  else {
    // Imaginary part of rotation.
    arma::mat lambda(o,o);
    lambda.zeros();
    // Offset
    size_t ioff=o*(o-1)/2 + ioff0;
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<=i;j++) {
	size_t idx=ioff + i*(i+1)/2 + j;
	lambda(i,j)=x(idx);
	lambda(j,i)=-x(idx);
      }

    // Matrix
    R=kappa*COMPLEX1 + lambda*COMPLEXI;
  }

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
  arma::cx_mat Roo(Rvec*arma::diagmat(arma::exp(COMPLEXI*Rval))*arma::trans(Rvec));

  // Check unitarity
  arma::cx_mat prod=arma::trans(Roo)*Roo-arma::eye(Roo.n_cols,Roo.n_cols);
  double norm=rms_cnorm(prod);
  if(norm>=sqrt(DBL_EPSILON))
    throw std::runtime_error("Matrix is not unitary!\n");

  return Roo;
}

void PZStability::set(const rscf_t & sol, const arma::uvec & drop, bool cplx_, bool can, bool oo) {
  cplx=cplx_;
  cancheck=can;
  oocheck=oo;

  Checkpoint *chkptp=solverp->get_checkpoint();
  arma::cx_mat CW;
  chkptp->cread("CW",CW);

  chkptp->read(basis);
  grid=DFTGrid(&basis,true,method.lobatto);
  nlgrid=DFTGrid(&basis,true,method.lobatto);

  // Update solution
  rsol=sol;
  rsol.cC.cols(0,CW.n_cols-1)=CW;

  // Drop orbitals
  if(!cancheck) {
    arma::uvec dr(arma::sort(drop,"descend"));
    for(size_t i=0;i<dr.n_elem;i++) {
      rsol.cC.shed_col(dr(i));
      CW.shed_col(0);
    }
  }

  // Update size parameters
  restr=true;
  oa=ob=CW.n_cols;
  va=vb=rsol.cC.n_cols-oa;

  fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i\n",(int) oa, (int) ob, (int) va, (int) vb);
  fprintf(stderr,"There are %i parameters.\n",(int) count_params());
  fflush(stdout);

  // Reconstruct DFT grid
  if(method.adaptive)
    grid.construct(CW,method.gridtol,method.x_func,method.c_func);
  else {
    bool strict(false);
    grid.construct(method.nrad,method.lmax,method.x_func,method.c_func,strict);
    if(method.nl)
      nlgrid.construct(method.nlnrad,method.nllmax,true,false,strict,true);
  }

  // Update reference
  arma::vec x(count_params());
  x.zeros();
  eval(x,-1);
}

void PZStability::set(const uscf_t & sol, const arma::uvec & dropa, const arma::uvec & dropb, bool cplx_, bool can, bool oo) {
  cplx=cplx_;
  cancheck=can;
  oocheck=oo;

  Checkpoint *chkptp=solverp->get_checkpoint();
  arma::cx_mat CWa, CWb;
  chkptp->cread("CWa",CWa);
  if(chkptp->exist("CWb"))
    chkptp->cread("CWb",CWb);
  
  chkptp->read(basis);
  grid=DFTGrid(&basis,true,method.lobatto);
  nlgrid=DFTGrid(&basis,true,method.lobatto);

  // Update solution
  usol=sol;
  usol.cCa.cols(0,CWa.n_cols-1)=CWa;
  if(CWb.n_cols)
    usol.cCb.cols(0,CWb.n_cols-1)=CWb;

  // Drop orbitals
  if(!cancheck) {
    arma::uvec dra(arma::sort(dropa,"descend"));
    for(size_t i=0;i<dra.n_elem;i++) {
      usol.cCa.shed_col(dra(i));
      CWa.shed_col(0);
    }
    arma::uvec drb(arma::sort(dropb,"descend"));
    for(size_t i=0;i<drb.n_elem;i++) {
      usol.cCb.shed_col(drb(i));
      CWb.shed_col(0);
    }
  }
  
  // Update size parameters
  restr=false;
  oa=CWa.n_cols;
  ob=CWb.n_cols;
  va=usol.cCa.n_cols-oa;
  vb=usol.cCb.n_cols-ob;

  fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i\n",(int) oa, (int) ob, (int) va, (int) vb);
  fprintf(stderr,"There are %i parameters.\n",(int) count_params());
  fflush(stdout);

  // Reconstruct DFT grid
  if(method.adaptive) {
    arma::cx_mat Ctilde(sol.Ca.n_rows,CWa.n_cols+CWb.n_cols);
    Ctilde.cols(0,oa-1)=CWa;
    if(ob)
      Ctilde.cols(oa,oa+ob-1)=CWb;
    grid.construct(Ctilde,method.gridtol,method.x_func,method.c_func);
  } else {
    bool strict(false);
    grid.construct(method.nrad,method.lmax,method.x_func,method.c_func,strict);
    if(method.nl)
      nlgrid.construct(method.nlnrad,method.nllmax,true,false,strict,true);
  }

  // Update reference
  arma::vec x(count_params());
  x.zeros();
  eval(x,-1);
}

void PZStability::check() {
  Timer tfull;
  
  // Estimate runtime
  {
    // Test value
    arma::vec x0(count_params());
    x0.zeros();
    if(x0.n_elem)
      x0(0)=0.1;
    if(x0.n_elem>=2)
      x0(1)=-0.1;
    
    Timer t;
    eval(x0,GHMODE);
    double dt=t.get();

    // Total time is
    double ttot=2*x0.n_elem*(x0.n_elem+1)*dt;
    fprintf(stderr,"\nComputing the Hessian will take approximately %s\n",t.parse(ttot).c_str());
    fflush(stderr);
  }

  // Get gradient
  Timer t;
  arma::vec g(gradient());
  printf("Gradient norm is %e (%s)\n",arma::norm(g,2),t.elapsed().c_str()); fflush(stdout);
  if(cancheck && oocheck) {
    size_t nov=count_ov_params(oa,va);
    if(!restr)
      nov+=count_ov_params(ob,vb);

    double ovnorm=arma::norm(g.subvec(0,nov-1),2);
    double oonorm=arma::norm(g.subvec(nov,g.n_elem-1),2);
    printf("OV norm %e, OO norm %e.\n",ovnorm,oonorm);
  }
  t.set();

  // Evaluate Hessian
  arma::mat h(hessian());
  printf("Hessian evaluated (%s)\n",t.elapsed().c_str()); fflush(stdout);
  t.set();

  // Helpers
  arma::mat hvec;
  arma::vec hval;

  if(cancheck && oocheck) {
    // Amount of parameters
    size_t nov=count_ov_params(oa,va);
    if(!restr)
      nov+=count_ov_params(ob,vb);

    // Stability of canonical orbitals
    arma::mat hcan(h.submat(0,0,nov-1,nov-1));
    eig_sym_ordered(hval,hvec,hcan);
    printf("\nOV Hessian diagonalized (%s)\n",t.elapsed().c_str()); fflush(stdout);
    hval.t().print("Canonical orbital stability");

    // Stability of optimal orbitals
    arma::mat hopt(h.submat(nov-1,nov-1,h.n_rows-1,h.n_cols-1));
    eig_sym_ordered(hval,hvec,hopt);
    printf("\nOO Hessian diagonalized (%s)\n",t.elapsed().c_str()); fflush(stdout);
    hval.t().print("Optimal orbital stability");
  }

  if(cplx) {
    // Collect real and imaginary parts of Hessian.
    arma::uvec rp, ip;
    real_imag_idx(rp,ip);

    arma::mat rh(rp.n_elem,rp.n_elem);
    for(size_t i=0;i<rp.n_elem;i++)
      for(size_t j=0;j<rp.n_elem;j++)
	rh(i,j)=h(rp(i),rp(j));

    arma::mat ih(ip.n_elem,ip.n_elem);
    for(size_t i=0;i<ip.n_elem;i++)
      for(size_t j=0;j<ip.n_elem;j++)
	ih(i,j)=h(ip(i),ip(j));

    eig_sym_ordered(hval,hvec,rh);
    printf("\nReal part of Hessian diagonalized (%s)\n",t.elapsed().c_str()); fflush(stdout);
    hval.t().print("Real orbital stability");
    
    eig_sym_ordered(hval,hvec,ih);
    printf("\nImaginary part of Hessian diagonalized (%s)\n",t.elapsed().c_str()); fflush(stdout);
    hval.t().print("Imaginary orbital stability");
  }
   
  // Total stability
  eig_sym_ordered(hval,hvec,h);
  printf("\nFull Hessian diagonalized (%s)\n",t.elapsed().c_str()); fflush(stdout);
  hval.t().print("Orbital stability");

  fprintf(stderr,"Check completed in %s.\n",tfull.elapsed().c_str());
}

void PZStability::print_status(size_t iiter, const arma::vec & g, const Timer & t) const {
  printf("\nIteration %i, gradient norm %e (%s)\n",(int) iiter,arma::norm(g,2),t.elapsed().c_str());

  if(cancheck && oocheck) {
    // Amount of parameters
    size_t nov=count_ov_params(oa,va);
    if(!restr)
      nov+=count_ov_params(ob,vb);

    double ovnorm=arma::norm(g.subvec(0,nov-1),2);
    double oonorm=arma::norm(g.subvec(nov,g.n_elem-1),2);
    printf("OV norm %e, OO norm %e.\n",ovnorm,oonorm);
  }
}
