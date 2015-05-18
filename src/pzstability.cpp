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
#include "timer.h"
#include "mathf.h"

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
	x(i*v + j + ioff)=2.0*std::real(Mov(i,j));
    ioff+=o*v;
  }
  
  // Imaginary part
  if(imag) {
    for(size_t i=0;i<o;i++)
      for(size_t j=0;j<v;j++)
	x(i*v + j + ioff)=-2.0*std::imag(Mov(i,j));
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
	R(i,j)=-x(idx)*COMPLEX1;
	R(j,i)=x(idx)*COMPLEX1;
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
	R(i,j)-=x(idx)*COMPLEXI;
	R(j,i)+=x(idx)*COMPLEXI;
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
	x(idx + ioff)=2.0*std::real(M(j,i));
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
	x(idx + ioff)=-2.0*std::imag(M(j,i));
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
    
    // RHS value
    x(i)=ss_fd;
    double yr=eval(x);
    
    // LHS value
    x(i)=-ss_fd;
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

void FDHessian::update(const arma::vec & x, bool ref) {
  (void) x;
  throw std::runtime_error("Error - update function must be overloaded!\n");
}

void FDHessian::print_status(size_t iiter, const arma::vec & g, const Timer & t) const {
  printf("\nIteration %i, gradient norm %e, max norm %e (%s)\n",(int) iiter,arma::norm(g,2),arma::max(arma::abs(g)),t.elapsed().c_str());
}

double FDHessian::optimize(size_t maxiter, double gthr, bool max) {
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
      sd+=gamma*oldsd;

      printf("CG step\n");
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

PZStability::PZStability(SCF * solver, dft_t dft) {
  solverp=solver;
  solverp->set_verbose(false);

  method=dft;

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

void PZStability::real_imag_idx(arma::uvec & idxr, arma::uvec & idxi) const {
  if(!imag) {
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
    if(imag)
      nimag+=oa*(oa-1)/2;
    
    if(!restr) {
      nreal+=ob*(ob-1)/2;
      if(imag)
	nimag+=ob*(ob-1)/2;
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
    for(size_t irot=0;irot<oa*(oa-1)/2;irot++)
      idxi(ii++)=irot + ioff;
    ioff+=oa*(oa-1)/2;

    if(!restr) {
      // First are the real parameters
      for(size_t irot=0;irot<ob*(ob-1)/2;irot++) {
	idxr(ir++)=irot+ioff;
      }
      ioff+=ob*(ob-1)/2;
      // and then the imaginary parameters
      for(size_t irot=0;irot<ob*(ob-1)/2;irot++)
	idxi(ii++)=irot + ioff;
      ioff+=ob*(ob-1)/2;
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

void PZStability::print_info(const arma::cx_mat & CO, const arma::cx_mat & CV, const std::vector<arma::cx_mat> & Forb, const arma::cx_mat & H0, const arma::vec & Eorb) {
  // Virtual space density matrix
  arma::cx_mat v(CV.n_rows,CV.n_rows);
  v.zeros();
  for(size_t io=oa;io<CV.n_cols;io++)
    v+=CV.col(io)*arma::trans(CV.col(io));
  
  // Build effective Fock operator
  arma::mat S(solverp->get_S());
  arma::cx_mat H(H0*COMPLEX1);
  for(size_t io=0;io<oa;io++) {
    arma::cx_mat Porb(CO.col(io)*arma::trans(CO.col(io)));
    H-=S*(Porb*Forb[io]*Porb + v*Forb[io]*Porb + Porb*Forb[io]*v)*S;
  }
  
  // Occupied block
  bool diagok;
  arma::cx_mat Hoo(arma::trans(CO)*H*CO);
  arma::cx_mat Hvv(arma::trans(CV)*H*CV);
  
  arma::vec Eo;
  arma::cx_mat Co;
  diagok=arma::eig_sym(Eo,Co,Hoo);
  if(!diagok) {
    ERROR_INFO();
    throw std::runtime_error("Error diagonalizing H in occupied space.\n");
  }
  
  arma::vec Ev;
  arma::cx_mat Cv;
  diagok=arma::eig_sym(Ev,Cv,Hvv);
  if(!diagok) {
    ERROR_INFO();
    throw std::runtime_error("Error diagonalizing H in virtual space.\n");
  }
  
  // Whole set of orbital energies
  arma::vec Efull(CO.n_cols+CV.n_cols);
  Efull.subvec(0,CO.n_cols-1)=Eo;
  Efull.subvec(CO.n_cols,CO.n_cols+CV.n_cols-1)=Ev;

  // Print out
  std::vector<double> occs(CO.n_cols,1.0);
  print_E(Efull,occs,false);

  // Collect projected energies
  arma::vec Ep(CO.n_cols);
  for(size_t io=0;io<CO.n_cols;io++)
    Ep(io)=std::real(Hoo(io,io));
		     
  // Print out optimal orbitals
  printf("Decomposition of self-interaction energies:\n");
  printf("\t%4s\t%8s\t%8s\n","io","E(orb)","E(SI)");
  for(size_t io=0;io<CO.n_cols;io++)
    printf("\t%4i\t% 8.3f\t% 8.3f\n",(int) io+1,Ep(io),Eorb(io));
  fflush(stdout);
  
}

void PZStability::print_info() {
  double focktol=ROUGHTOL;

  if(restr) {
    rscf_t tmp(rsol);
    
    // Occupied orbitals
    arma::cx_mat CO=tmp.cC.cols(0,oa-1);
    // Virtuals
    arma::cx_mat CV=tmp.cC.cols(oa,oa+va-1);

    // Dummy occupation vector
    std::vector<double> occa(oa,2.0);
    // Build global Fock operator
    solverp->Fock_RDFT(tmp,occa,method,grid,nlgrid,focktol);
    // Build the SI part
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    solverp->PZSIC_Fock(Forb,Eorb,CO,method,grid,nlgrid,true);

    // Diagonalize
    if(tmp.K_im.n_rows == tmp.H.n_rows && tmp.K_im.n_cols == tmp.H.n_cols)
      print_info(CO,CV,Forb,tmp.H*COMPLEX1 + tmp.K_im*COMPLEXI,Eorb);
    else
      print_info(CO,CV,Forb,tmp.H*COMPLEX1,Eorb);
    
  } else {
    uscf_t tmp(usol);
    
    // Dummy occupation vector
    std::vector<double> occa(oa,1.0);
    std::vector<double> occb(ob,1.0);
    
    // Build global Fock operator
    solverp->Fock_UDFT(tmp,occa,occb,method,grid,nlgrid,focktol);

    // Build the SI part
    std::vector<arma::cx_mat> Forba(oa), Forbb(ob);
    arma::vec Eorba, Eorbb;
    {
      arma::cx_mat Ct(tmp.cCa.n_rows,oa+ob);
      Ct.cols(0,oa-1)=tmp.cCa.cols(0,oa-1);
      if(ob)
	Ct.cols(oa,oa+ob-1)=tmp.cCb.cols(0,ob-1);
      
      // Build the SI part
      std::vector<arma::cx_mat> Forb;
      arma::vec Eorb;
      solverp->PZSIC_Fock(Forb,Eorb,Ct,method,grid,nlgrid,true);

      Eorba=Eorb.subvec(0,oa-1);
      if(ob)
	Eorbb=Eorb.subvec(oa,oa+ob-1);
      for(size_t i=0;i<oa;i++)
	Forba[i]=Forb[i];
      for(size_t i=0;i<ob;i++)
	Forbb[i]=Forb[i+oa];
    }

    // Occupied orbitals
    arma::cx_mat COa=tmp.cCa.cols(0,oa-1);
    arma::cx_mat COb;
    if(ob)
      COb=tmp.cCb.cols(0,ob-1);
    // Virtuals
    arma::cx_mat CVa=tmp.cCa.cols(oa,oa+va-1);
    arma::cx_mat CVb=tmp.cCb.cols(ob,ob+vb-1);

    // Diagonalize
    printf("\n **** Alpha orbitals ****\n");
    if(tmp.Ka_im.n_rows == tmp.Ha.n_rows && tmp.Ka_im.n_cols == tmp.Ha.n_cols)
      print_info(COa,CVa,Forba,tmp.Ha*COMPLEX1 + tmp.Ka_im*COMPLEXI,Eorba);
    else
      print_info(COa,CVa,Forba,tmp.Ha*COMPLEX1,Eorba);
    printf("\n **** Beta  orbitals ****\n");
    if(tmp.Kb_im.n_rows == tmp.Hb.n_rows && tmp.Kb_im.n_cols == tmp.Hb.n_cols)
      print_info(COb,CVb,Forbb,tmp.Hb*COMPLEX1 + tmp.Kb_im*COMPLEXI,Eorbb);
    else
      print_info(COb,CVb,Forbb,tmp.Hb*COMPLEX1,Eorbb);
  }
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

  // Follow Borghi et al and divide this further
  Tmu/=5;
}

arma::vec PZStability::gradient() {
  double focktol=ROUGHTOL;

  arma::vec g(count_params());
  g.zeros();

  if(restr) {
    rscf_t tmp(rsol);
    
    // Dummy occupation vector
    std::vector<double> occa(oa,2.0);
    // Build global Fock operator
    if(cancheck)
      solverp->Fock_RDFT(tmp,occa,method,grid,nlgrid,focktol);
    
    // Occupied orbitals
    arma::cx_mat CO=tmp.cC.cols(0,oa-1);
    arma::cx_mat CV=tmp.cC.cols(oa,oa+va-1);
    // Build the SI part
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    solverp->PZSIC_Fock(Forb,Eorb,CO,method,grid,nlgrid,true);
    
    size_t ioff=0;
    
    if(cancheck) {
      // OV gradient is
      arma::cx_mat gOV(oa,va);
      for(size_t i=0;i<oa;i++)
	for(size_t j=0;j<va;j++)
	  gOV(i,j)=-arma::as_scalar(arma::strans(CO.col(i))*(tmp.H-Forb[i])*arma::conj(CV.col(j)));
      
      // Collect values
      arma::vec pOV(gather_ov(gOV,real,imag));
      g.subvec(ioff,ioff+pOV.n_elem-1)=pOV;
      ioff+=pOV.n_elem;
    }

    if(oocheck) {
      // OO gradient is
      arma::cx_mat gOO(oa,oa);
      for(size_t i=0;i<oa;i++)
	for(size_t j=0;j<oa;j++)
	  gOO(i,j)=arma::as_scalar(arma::strans(CO.col(i))*(Forb[i]-Forb[j])*arma::conj(CO.col(j)));

      // Collect values
      arma::vec pOO(gather_oo(gOO,real,imag));
      g.subvec(ioff,ioff+pOO.n_elem-1)=pOO;
      ioff+=pOO.n_elem;
    }

    // Closed shell - two orbitals!
    g*=2.0;
    
  } else {
    uscf_t tmp(usol);
    
    // Dummy occupation vector
    std::vector<double> occa(oa,1.0);
    std::vector<double> occb(ob,1.0);
    
    // Build global Fock operator
    if(cancheck)
    solverp->Fock_UDFT(tmp,occa,occb,method,grid,nlgrid,focktol);

    // Build the SI part
    std::vector<arma::cx_mat> Forba(oa), Forbb(ob);
    arma::vec Eorba, Eorbb;
    {
      arma::cx_mat Ct(tmp.cCa.n_rows,oa+ob);
      Ct.cols(0,oa-1)=tmp.cCa.cols(0,oa-1);
      if(ob)
	Ct.cols(oa,oa+ob-1)=tmp.cCb.cols(0,ob-1);
      
      // Build the SI part
      std::vector<arma::cx_mat> Forb;
      arma::vec Eorb;
      solverp->PZSIC_Fock(Forb,Eorb,Ct,method,grid,nlgrid,true);

      Eorba=Eorb.subvec(0,oa-1);
      if(ob)
	Eorbb=Eorb.subvec(oa,oa+ob-1);
      for(size_t i=0;i<oa;i++)
	Forba[i]=Forb[i];
      for(size_t i=0;i<ob;i++)
	Forbb[i]=Forb[i+oa];
    }

    // Occupied orbitals
    arma::cx_mat COa=tmp.cCa.cols(0,oa-1);
    arma::cx_mat COb;
    if(ob)
      COb=tmp.cCb.cols(0,ob-1);
    // Virtuals
    arma::cx_mat CVa=tmp.cCa.cols(oa,oa+va-1);
    arma::cx_mat CVb=tmp.cCb.cols(ob,ob+vb-1);

    size_t ioff=0;
    
    if(cancheck) {
      // OV alpha gradient is
      arma::cx_mat gOVa(oa,va);
      for(size_t i=0;i<oa;i++)
	for(size_t j=0;j<va;j++)
	  gOVa(i,j)=-arma::as_scalar(arma::strans(COa.col(i))*(tmp.Ha-Forba[i])*arma::conj(CVa.col(j)));

      // Collect values
      arma::vec pOVa(gather_ov(gOVa,real,imag));
      g.subvec(ioff,ioff+pOVa.n_elem-1)=pOVa;
      ioff+=pOVa.n_elem;
      
      if(ob) {
	// OV beta gradient is
	arma::cx_mat gOVb(ob,vb);
	for(size_t i=0;i<ob;i++)
	  for(size_t j=0;j<vb;j++)
	    gOVb(i,j)=-arma::as_scalar(arma::strans(COb.col(i))*(tmp.Hb-Forbb[i])*arma::conj(CVb.col(j)));

	// Collect values
	arma::vec pOVb(gather_ov(gOVb,real,imag));
	g.subvec(ioff,ioff+pOVb.n_elem-1)=pOVb;
	ioff+=pOVb.n_elem;
      }
    }
    
    if(oocheck) {
      // OO alpha gradient is
      arma::cx_mat gOOa(oa,oa);
      for(size_t i=0;i<oa;i++)
	for(size_t j=0;j<oa;j++)
	  gOOa(i,j)=arma::as_scalar(arma::strans(COa.col(i))*(Forba[i]-Forba[j])*arma::conj(COa.col(j)));

      // Collect values
      arma::vec pOOa(gather_oo(gOOa,real,imag));
      g.subvec(ioff,ioff+pOOa.n_elem-1)=pOOa;
      ioff+=pOOa.n_elem;
      
      if(ob) {
	// OO beta gradient is
	arma::cx_mat gOOb(ob,ob);
	for(size_t i=0;i<ob;i++)
	  for(size_t j=0;j<ob;j++)
	    gOOb(i,j)=arma::as_scalar(arma::strans(COb.col(i))*(Forbb[i]-Forbb[j])*arma::conj(COb.col(j)));

	// Collect values
	arma::vec pOOb(gather_oo(gOOb,real,imag));
	g.subvec(ioff,ioff+pOOb.n_elem-1)=pOOb;
	ioff+=pOOb.n_elem;
      }
    }
  }

  /*
  arma::vec gn(FDHessian::gradient());
  if(rms_norm(gn-g)>1e-6) {
    gn.t().print("Numerical gradient");
    g.t().print("Analytic gradient");
    fflush(stdout);
    g.save("g.dat",arma::raw_ascii);
    gn.save("gn.dat",arma::raw_ascii);
    throw std::logic_error("Problem in gradient.\n");
  } else {
    printf("Analytic gradient is OK.\n");
  }
  */

  // Update step size
  update_step(g);
  
  return g;
}

arma::mat PZStability::hessian() {
  // Amount of parameters
  size_t npar=count_params();

  // Compute Hessian
  arma::mat h(npar,npar);
  h.zeros();

  // Get original references
  rscf_t rsol0(rsol);
  uscf_t usol0(usol);
  
  /* This loop isn't OpenMP parallel, because parallellization is
     already used in the energy evaluation. Parallellizing over trials
     here would require use of thread-local DFT grids, and possibly
     even thread-local SCF solver objects (for the Coulomb part).*/
  for(size_t i=0;i<npar;i++) {
    arma::vec x(npar);
    x.zeros();
    
    // RHS gradient
    x(i)=ss_fd;
    rsol=rsol0;
    usol=usol0;
    update(x,false);
    arma::vec gr=gradient();
    
    // LHS value
    x(i)=-ss_fd;
    rsol=rsol0;
    usol=usol0;
    update(x,false);
    arma::vec gl=gradient();
    
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
  
  rsol=rsol0;
  usol=usol0;

  // Symmetrize Hessian to distribute numerical error evenly
  h=(h+arma::trans(h))/2.0;
  
  return h;
}

double PZStability::eval(const arma::vec & x) {
  double focktol=ROUGHTOL;

  if(restr) {
    rscf_t tmp(rsol);

    // Rotate orbitals
    tmp.cC=tmp.cC*rotation(x,false);
    
    // Update density matrix
    tmp.P=2.0*arma::real(tmp.cC.cols(0,oa-1)*arma::trans(tmp.cC.cols(0,oa-1)));
    
    // Dummy occupation vector
    std::vector<double> occa(oa,2.0);
    
    // Build global Fock operator
    solverp->Fock_RDFT(tmp,occa,method,grid,nlgrid,focktol);
    
    // Build the SI part
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    solverp->PZSIC_Fock(Forb,Eorb,tmp.cC.cols(0,oa-1),method,grid,nlgrid,false);
    
    // Account for spin
    return tmp.en.E - 2.0*arma::sum(Eorb);
    
  } else {
    uscf_t tmp(usol);
    
    // Rotate orbitals
    tmp.cCa=tmp.cCa*rotation(x,false);
    if(ob)
      tmp.cCb=tmp.cCb*rotation(x,true);
    
    // Update density matrix
    tmp.Pa=arma::real(tmp.cCa.cols(0,oa-1)*arma::trans(tmp.cCa.cols(0,oa-1)));
    if(ob)
      tmp.Pb=arma::real(tmp.cCb.cols(0,ob-1)*arma::trans(tmp.cCb.cols(0,ob-1)));
    else
      tmp.Pb.zeros(tmp.cCb.n_rows,tmp.cCb.n_rows);
    
    tmp.P=tmp.Pa+tmp.Pb;
    
    // Dummy occupation vector
    std::vector<double> occa(oa,1.0);
    std::vector<double> occb(ob,1.0);
    
    // Build global Fock operator
    solverp->Fock_UDFT(tmp,occa,occb,method,grid,nlgrid,focktol);
    
    // Build the SI part
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    arma::cx_mat Ct(tmp.cCa.n_rows,oa+ob);
    Ct.cols(0,oa-1)=tmp.cCa.cols(0,oa-1);
    if(ob)
      Ct.cols(oa,oa+ob-1)=tmp.cCb.cols(0,ob-1);
    solverp->PZSIC_Fock(Forb,Eorb,Ct,method,grid,nlgrid,false);
    
    // Result is
    return tmp.en.E-arma::sum(Eorb);
  }
}

double PZStability::optimize(size_t maxiter, double gthr, bool max) {
  arma::vec x0(count_params());
  x0.zeros();

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
    sd = max ? g : -g;
   
    if(iiter % std::min(count_params(), (size_t) 5)!=0) {
      // Update factor
      double gamma;
      
      // Polak-Ribiere
      gamma=arma::dot(g,g-gold)/arma::dot(gold,gold);
      // Fletcher-Reeves
      //gamma=arma::dot(g,g)/arma::dot(gold,gold);
      
      // Update search direction
      arma::vec sdnew(sd+gamma*oldsd);

      // Check that new SD is sane
      if(arma::dot(sdnew,sd)<=0)
	// This would take us into the wrong direction!
	printf("Bad CG direction. SD step\n");
      else {
	// Update search direction
	sd=sdnew;
	printf("CG step\n");
      }
    } else printf("SD step\n");    

    // Derivative is
    double dE=arma::dot(sd,g);

    printf(" %e % .10f\n",0.0,E0);
    fflush(stdout);
	
    // Initial step size
    double d=Tmu;
    // Value at initial step
    double Ed=eval(d*sd);
    printf(" %e % .10f\n",d,Ed);
    fflush(stdout);

    // Optimal step length
    double step;
    // Energy for optimal step
    double Es;

    // Fit parabola
    double a=(Ed - dE*d - E0)/(d*d);
    bool fitok=a>0;
    if(fitok) {
      // The optimal step is at
      step=-dE/(2.0*a);

      // Step is OK if it is in the trust interval
      fitok=(step>0.0 && step<=Tmu);
    }
    // Reset step if it's not OK
    if(!fitok)
      step=Tmu;
    
    if(fitok) {
      // Evaluate energy at trial step
      Es=eval(step*sd);
      printf(" %e % .10f\n",step,Es);
      fflush(stdout);
      
      // Did the search work?
      if(Es>=E0)
	fitok=false;
    }

    // Did the search work? If not, fall back to a backtracking line search.
    if(!fitok) {
      double tau=0.7;
      if(step<0.0 || step>Tmu)
	step=Tmu;
      
      while(true) {
	step*=tau;
	Es=eval(step*sd);
	printf(" %e % .10f backtrack\n",step,Es);
	fflush(stdout);
	if(Es<E0)
	  break;
      }
    }
    
    printf("Line search changed value by %e\n",Es-E0);
    update(step*sd);
    E0=Es;
  }
  
  printf("Final value is % .10f; optimization changed value by %e\n",E0,E0-ival);
  print_info();
  
  // Return the change
  return E0-ival;
}

void PZStability::update(const arma::vec & x, bool ref) {
  if(restr) {
    arma::cx_mat R=rotation(x,false);
    rsol.cC=rsol.cC*R;
    rsol.P=2.0*arma::real(rsol.cC.cols(0,oa-1)*arma::trans(rsol.cC.cols(0,oa-1)));
  } else {
    arma::cx_mat Ra=rotation(x,false);
    usol.cCa=usol.cCa*Ra;
    usol.Pa=arma::real(usol.cCa.cols(0,oa-1)*arma::trans(usol.cCa.cols(0,oa-1)));
    if(ob) {
      arma::cx_mat Rb=rotation(x,true);
      usol.cCb=usol.cCb*Rb;
      usol.Pb=arma::real(usol.cCb.cols(0,ob-1)*arma::trans(usol.cCb.cols(0,ob-1)));
    } else
      usol.Pb.zeros(usol.cCb.n_rows,usol.cCb.n_rows);
    
      usol.P=usol.Pa+usol.Pb;
  }

  // Update orbitals in checkpoint file
  Checkpoint *chkptp=solverp->get_checkpoint();
  if(restr)
    chkptp->cwrite("CW",rsol.cC.cols(0,oa-1));
  else {
    chkptp->cwrite("CWa",usol.cCa.cols(0,oa-1));
    if(ob)
      chkptp->cwrite("CWb",usol.cCb.cols(0,ob-1));   
  }
}

arma::cx_mat PZStability::rotation(const arma::vec & x, bool spin) const {
  // Get rotation matrix
  arma::cx_mat X(rotation_pars(x,spin));

  // Rotation matrix
  arma::cx_mat R(X);
  R.eye();
  if(!cancheck) {
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

    arma::cx_mat r(spread_ov(x.subvec(ioff0,ioff0+count_ov_params(o,v)-1),o,v,real,imag));
    R.submat(0,o,o-1,o+v-1)=r;
    R.submat(o,0,o+v-1,o-1)=-arma::trans(r);
  }

  // OO part
  if(oocheck) {
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
  if(norm>=sqrt(DBL_EPSILON))
    throw std::runtime_error("Matrix is not unitary!\n");
  
  return rot;  
}

void PZStability::set(const rscf_t & sol, const arma::uvec & drop, bool real_, bool imag_, bool can, bool oo) {
  real=real_;
  imag=imag_;
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

  // Reconstruct DFT grid
  if(method.adaptive)
    grid.construct(CW,method.gridtol,method.x_func,method.c_func);
  else {
    bool strict(false);
    grid.construct(method.nrad,method.lmax,method.x_func,method.c_func,strict);
    if(method.nl)
      nlgrid.construct(method.nlnrad,method.nllmax,true,false,strict,true);
  }
}

void PZStability::set(const uscf_t & sol, const arma::uvec & dropa, const arma::uvec & dropb, bool real_, bool imag_, bool can, bool oo) {
  real=real_;
  imag=imag_;
  cancheck=can;
  oocheck=oo;

  Checkpoint *chkptp=solverp->get_checkpoint();
  arma::cx_mat CWa, CWb;
  chkptp->cread("CWa",CWa);
  if(chkptp->exist("CWb.re"))
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
  fflush(stderr);

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
}

rscf_t PZStability::get_rsol() const {
  return rsol;
}

uscf_t PZStability::get_usol() const {
  return usol;
}

void PZStability::check() {
  Timer tfull;
  
  // Estimate runtime
  {
    Timer t;
    gradient();
    double dt=t.get();

    // Total time is
    double ttot=2*count_params()*dt;
    fprintf(stderr,"\nComputing the Hessian will take approximately %s\n",t.parse(ttot).c_str());
    fflush(stderr);
  }

  // Evaluate Hessian
  Timer t;
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

  if(real && imag) {
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
  printf("\nIteration %i, gradient norm %e, max norm %e (%s)\n",(int) iiter,arma::norm(g,2),arma::max(arma::abs(g)),t.elapsed().c_str());

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

void PZStability::linesearch() {
  // Get gradient
  arma::vec g(gradient());

  FILE *out=fopen("pz_ls.dat","w");
  
  // Do line search
  double dx=1e-2;
  for(int i=-100;i<=100;i++) {
    printf("x = %e\n",i*dx);
    fprintf(out,"%e % e\n",i*dx,eval(i*dx*g));
  }
  fclose(out);
}
