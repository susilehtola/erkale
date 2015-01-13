/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include <cfloat>
#include "pzstability.h"
#include "linalg.h"
#include "timer.h"

#define COMPLEX1 std::complex<double>(1.0,0.0)
#define COMPLEXI std::complex<double>(0.0,1.0)

FDHessian::FDHessian() {
  // Rotation step size
  ss=1e-6;
}

FDHessian::~FDHessian() {
}

arma::vec FDHessian::gradient() {
  // Amount of parameters
  size_t npar=count_params();

  // Compute gradient
  arma::vec g(npar);
  g.zeros();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(size_t i=0;i<npar;i++) {
    arma::vec x(npar);
    x.zeros();

    // LHS value
    x(i)=-ss;
    double yl=eval(x);

    // RHS value
    x(i)=ss;
    double yr=eval(x);

    // Derivative
    g(i)=(yr-yl)/(2*ss);
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
  
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
  for(size_t ii=0;ii<idx.size();ii++) {
    size_t i=idx[ii].i;
    size_t j=idx[ii].j;
    
    arma::vec x(npar);

    // RH,RH value
    x.zeros();
    x(i)+=ss;
    x(j)+=ss;
    double yrr=eval(x);
    
    // RH,LH
    x.zeros();
    x(i)+=ss;
    x(j)-=ss;
    double yrl=eval(x);

    // LH,RH
    x.zeros();
    x(i)-=ss;
    x(j)+=ss;
    double ylr=eval(x);

    // LH,LH
    x.zeros();
    x(i)-=ss;
    x(j)-=ss;
    double yll=eval(x);

    // Values
    h(i,j)=(yrr - yrl - ylr + yll)/(4.0*ss*ss);
    // Symmetrize
    h(j,i)=h(i,j);
  }
  
  return h;
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
  N=0;

  check_lapack_thread();
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

double PZStability::eval(const arma::vec & x) {
  double focktol=ROUGHTOL;
  
  if(restr) {
    rscf_t tmp(rsol);

    // Perform ov rotation
    if(cancheck) {
      arma::cx_mat Rov=ov_rotation(x,false);
      tmp.cC=tmp.cC*Rov;
    }

    // Update density matrix
    tmp.P=2.0*arma::real(tmp.cC.cols(0,oa-1)*arma::trans(tmp.cC.cols(0,oa-1)));

    // Dummy occupation vector
    std::vector<double> occa(oa,1.0);
    
    // Build global Fock operator
    rscf_t dum;
    solverp->Fock_RDFT(tmp,occa,method,dum,*grid,focktol);

    // Get oo rotation
    arma::cx_mat Roo;
    if(oocheck)
      Roo=oo_rotation(x,false);
    else
      Roo.eye(oa,oa);

    // Build the SI part
    std::vector<arma::cx_mat> Forb;
    arma::vec Eorb;
    solverp->PZSIC_Fock(Forb,Eorb,tmp.cC.cols(0,oa-1),Roo,method,*grid,false);

    return tmp.en.E-arma::sum(Eorb);
  } else {
    uscf_t tmp(usol);

    // Perform ov rotation
    if(cancheck) {
      arma::cx_mat Rova=ov_rotation(x,false);
      tmp.cCa=tmp.cCa*Rova;
      
      arma::cx_mat Rovb=ov_rotation(x,true);
      tmp.cCb=tmp.cCb*Rovb;
    }

    // Update density matrices
    tmp.Pa=arma::real(tmp.cCa.cols(0,oa-1)*arma::trans(tmp.cCa.cols(0,oa-1)));
    tmp.Pb=arma::real(tmp.cCb.cols(0,ob-1)*arma::trans(tmp.cCb.cols(0,ob-1)));
    tmp.P=tmp.Pa+tmp.Pb;

    // Dummy occupation vector
    std::vector<double> occa(oa,1.0), occb(ob,1.0);
    
    // Build global Fock operator
    uscf_t dum;
    solverp->Fock_UDFT(tmp,occa,occb,method,dum,*grid,focktol);

    // Get oo rotation
    arma::cx_mat Rooa, Roob;
    if(oocheck) {
      Rooa=oo_rotation(x,false);
      Roob=oo_rotation(x,true);
    } else {
      Rooa.eye(oa,oa);
      Roob.eye(ob,ob);
    }
	
    // Build the SI part
    std::vector<arma::cx_mat> Forba, Forbb;
    arma::vec Eorba, Eorbb;
    solverp->PZSIC_Fock(Forba,Eorba,tmp.cCa.cols(0,oa-1),Rooa,method,*grid,false);
    solverp->PZSIC_Fock(Forbb,Eorbb,tmp.cCb.cols(0,ob-1),Roob,method,*grid,false);

    return tmp.en.E-arma::sum(Eorba)-arma::sum(Eorbb);
  }
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
  arma::cx_mat R(N,N);
  R.zeros();
  R.submat(0,o,o-1,N-1)=r;
  R.submat(o,0,N-1,o-1)=-arma::trans(r);

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
  return Rvec*arma::diagmat(arma::exp(COMPLEXI*Rval))*arma::trans(Rvec);
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
  return Rvec*arma::diagmat(arma::exp(COMPLEXI*Rval))*arma::trans(Rvec);
}

void PZStability::check(const rscf_t & sol, bool cplx_, bool can) {
  cplx=cplx_;
  cancheck=can;
  
  Checkpoint *chkptp=solverp->get_checkpoint();
  arma::cx_mat CW;
  chkptp->cread("CW",CW);

  BasisSet basis;
  chkptp->read(basis);
  grid=new DFTGrid(&basis,true,false);
  
  // Update size parameters
  restr=true;
  oa=ob=CW.n_cols;
  N=sol.C.n_rows;
  va=vb=N-oa;

  // Update solution
  rsol=sol;
  rsol.cC.cols(0,oa-1)=CW;

  fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i, N = %i\n",(int) oa, (int) ob, (int) va, (int) vb, (int) N);
  fprintf(stderr,"There are %i parameters.\n",(int) count_params());
  fflush(stdout);

  // Reconstruct DFT grid
  grid->construct(CW,method.gridtol,method.x_func,method.c_func);

  // Estimate runtime
  {
    // Test value
    arma::vec x0(count_params());
    x0.zeros();

    Timer t;
    eval(x0);
    double dt=t.get();

    // Total time is
    double ttot=2*x0.n_elem*(x0.n_elem+1)*dt;
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

  // Total stability
  eig_sym_ordered(hval,hvec,h);
  printf("\nFull Hessian diagonalized (%s)\n",t.elapsed().c_str()); fflush(stdout);
  hval.t().print("Orbital stability");

  delete grid;
}

void PZStability::check(const uscf_t & sol, bool cplx_, bool can) {
  cplx=cplx_;
  cancheck=can;

  Checkpoint *chkptp=solverp->get_checkpoint();
  arma::cx_mat CWa, CWb;
  chkptp->cread("CWa",CWa);
  chkptp->cread("CWb",CWb);

  BasisSet basis;
  chkptp->read(basis);
  grid=new DFTGrid(&basis,true,false);
  
  // Update size parameters
  restr=false;
  oa=CWa.n_cols;
  ob=CWb.n_cols;
  N=sol.Ca.n_rows;
  va=N-oa;
  vb=N-ob;
  
  // Update solution
  usol=sol;
  usol.cCa.cols(0,oa-1)=CWa;
  usol.cCb.cols(0,ob-1)=CWb;
  
  fprintf(stderr,"\noa = %i, ob = %i, va = %i, vb = %i, N = %i\n",(int) oa, (int) ob, (int) va, (int) vb, (int) N);
  fprintf(stderr,"There are %i parameters.\n",(int) count_params());
  fflush(stdout);
  
  // Reconstruct DFT grid
  arma::cx_mat Ctilde(sol.Ca.n_rows,CWa.n_cols+CWb.n_cols);
  Ctilde.cols(0,oa-1)=CWa;
  if(ob)
    Ctilde.cols(oa,oa+ob-1)=CWb;
  grid->construct(Ctilde,method.gridtol,method.x_func,method.c_func);
  
  // Estimate runtime
  {
    // Test value
    arma::vec x0(count_params());
    x0.zeros();

    Timer t;
    eval(x0);
    double dt=t.get();

    // Total time is
    double ttot=2*x0.n_elem*(x0.n_elem+1)*dt;
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
    size_t nov=count_ov_params(oa,va)+count_ov_params(ob,vb);

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

  // Total stability
  eig_sym_ordered(hval,hvec,h);
  printf("\nTotal Hessian diagonalized (%s)\n",t.elapsed().c_str()); fflush(stdout);
  hval.t().print("Orbital stability");

  delete grid;
}


