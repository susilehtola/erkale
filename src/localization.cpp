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

#include <armadillo>
#include <cstdio>
#include <cfloat>

#include "elements.h"
#include "dftfuncs.h"
#include "dftgrid.h"
#include "global.h"
#include "guess.h"
#include "hirshfeldi.h"
#include "linalg.h"
#include "localization.h"
#include "mathf.h"
#include "properties.h"
#include "stringutil.h"
#include "stockholder.h"
#include "timer.h"

void orbital_localization(enum locmet met0, const BasisSet & basis, const arma::mat & C, const arma::mat & P, double & measure, arma::cx_mat & W, bool verbose, bool real, int maxiter, double Gthr, double Fthr, enum unitmethod umet, enum unitacc uacc, bool delocalize, std::string fname, bool debug) {
  Timer t;

  // Optimizer
  UnitaryOptimizer opt(Gthr,Fthr,verbose,real);
  // Cost function
  UnitaryFunction *func;

  // Real operation?
  if(real)
    W=arma::real(W)*std::complex<double>(1.0,0.0);

  // Worker stack
  std::vector<enum locmet> metstack;

  switch(met0) {
  case(BOYS_2):
    // Initialize with Boys
    metstack.push_back(BOYS);
    metstack.push_back(BOYS_2);
    break;

  case(BOYS_3):
    // Initialize with Boys
    metstack.push_back(BOYS);
    metstack.push_back(BOYS_2);
    metstack.push_back(BOYS_3);
    break;

  case(BOYS_4):
    // Initialize with Boys
    metstack.push_back(BOYS);
    metstack.push_back(BOYS_2);
    metstack.push_back(BOYS_3);
    metstack.push_back(BOYS_4);
    break;

  case(FM_2):
    // Initialize with FM
    metstack.push_back(BOYS);
    metstack.push_back(FM_1);
    metstack.push_back(FM_2);
    break;

  case(FM_3):
    // Initialize with FM
    metstack.push_back(BOYS);
    metstack.push_back(FM_1);
    metstack.push_back(FM_2);
    metstack.push_back(FM_3);
    break;

  case(FM_4):
    // Initialize with FM
    metstack.push_back(BOYS);
    metstack.push_back(FM_1);
    metstack.push_back(FM_2);
    metstack.push_back(FM_3);
    metstack.push_back(FM_4);
    break;
    
  default:
    // Default - just do the one thing
    metstack.push_back(met0);
  }

  // Loop over localization
  for(size_t im=0;im<metstack.size();im++) {
    enum locmet met(metstack[im]);

    if(met==BOYS || met==BOYS_2 || met==BOYS_3 || met==BOYS_4) {
      int n=0;
      if(met==BOYS)
	n=1;
      else if(met==BOYS_2)
	n=2;
      else if(met==BOYS_3)
	n=3;
      else if(met==BOYS_4)
	n=4;
      
      func=new Boys(basis,C,n,verbose,delocalize);

    } else if(met==FM_1 || met==FM_2 || met==FM_3 || met==FM_4) {
      int n=0;
      if(met==FM_1)
	n=1;
      else if(met==FM_2)
	n=2;
      else if(met==FM_3)
	n=3;
      else if(met==FM_4)
	n=4;

      func=new FMLoc(basis,C,n,verbose,delocalize);

    } else if(met==PIPEK_MULLIKENH    ||	\
	      met==PIPEK_MULLIKEN2    ||	\
	      met==PIPEK_MULLIKEN4    ||	\
	      met==PIPEK_LOWDINH      ||	\
	      met==PIPEK_LOWDIN2      ||	\
	      met==PIPEK_LOWDIN4      ||	\
	      met==PIPEK_BADERH       ||	\
	      met==PIPEK_BADER2       ||	\
	      met==PIPEK_BADER4       ||	\
	      met==PIPEK_BECKEH       ||	\
	      met==PIPEK_BECKE2       ||	\
	      met==PIPEK_BECKE4       ||	\
	      met==PIPEK_HIRSHFELDH   ||	\
	      met==PIPEK_HIRSHFELD2   ||	\
	      met==PIPEK_HIRSHFELD4   ||	\
	      met==PIPEK_ITERHIRSHH   ||	\
	      met==PIPEK_ITERHIRSH2   ||	\
	      met==PIPEK_ITERHIRSH4   ||	\
	      met==PIPEK_IAOH         ||	\
	      met==PIPEK_IAO2         ||	\
	      met==PIPEK_IAO4         ||	\
	      met==PIPEK_STOCKHOLDERH ||	\
	      met==PIPEK_STOCKHOLDER2 ||	\
	      met==PIPEK_STOCKHOLDER4 ||	\
	      met==PIPEK_VORONOIH     ||	\
	      met==PIPEK_VORONOI2     ||	\
	      met==PIPEK_VORONOI4) {
      
      // Penalty exponent
      double p;
      enum chgmet chg;
      
      switch(met) {
      case(PIPEK_MULLIKENH):
	p=1.5;
	chg=MULLIKEN;
	break;
	
      case(PIPEK_MULLIKEN2):
	p=2.0;
	chg=MULLIKEN;
	break;
	
      case(PIPEK_MULLIKEN4):
	p=4.0;
	chg=MULLIKEN;
	break;
	
      case(PIPEK_LOWDINH):
	p=1.5;
	chg=LOWDIN;
	break;
	
      case(PIPEK_LOWDIN2):
	p=2.0;
	chg=LOWDIN;
	break;
	
      case(PIPEK_LOWDIN4):
	p=4.0;
	chg=LOWDIN;
	break;
	
      case(PIPEK_BADERH):
	p=1.5;
	chg=BADER;
	break;

      case(PIPEK_BADER2):
	p=2.0;
	chg=BADER;
	break;

      case(PIPEK_BADER4):
	p=4.0;
	chg=BADER;
	break;

      case(PIPEK_BECKEH):
	p=1.5;
	chg=BECKE;
	break;

      case(PIPEK_BECKE2):
	p=2.0;
	chg=BECKE;
	break;

      case(PIPEK_BECKE4):
	p=4.0;
	chg=BECKE;
	break;

      case(PIPEK_VORONOIH):
	p=1.5;
	chg=VORONOI;
	break;

      case(PIPEK_VORONOI2):
	p=2.0;
	chg=VORONOI;
	break;

      case(PIPEK_VORONOI4):
	p=4.0;
	chg=VORONOI;
	break;

      case(PIPEK_IAOH):
	p=1.5;
	chg=IAO;
	break;

      case(PIPEK_IAO2):
	p=2.0;
	chg=IAO;
	break;

      case(PIPEK_IAO4):
	p=4.0;
	chg=IAO;
	break;

      case(PIPEK_HIRSHFELDH):
	p=1.5;
	chg=HIRSHFELD;
	break;

      case(PIPEK_HIRSHFELD2):
	p=2.0;
	chg=HIRSHFELD;
	break;

      case(PIPEK_HIRSHFELD4):
	p=4.0;
	chg=HIRSHFELD;
	break;

      case(PIPEK_ITERHIRSHH):
	p=1.5;
	chg=ITERHIRSH;
	break;

      case(PIPEK_ITERHIRSH2):
	p=2.0;
	chg=ITERHIRSH;
	break;

      case(PIPEK_ITERHIRSH4):
	p=4.0;
	chg=ITERHIRSH;
	break;

      case(PIPEK_STOCKHOLDERH):
	p=1.5;
	chg=STOCKHOLDER;
	break;

      case(PIPEK_STOCKHOLDER2):
	p=2.0;
	chg=STOCKHOLDER;
	break;

      case(PIPEK_STOCKHOLDER4):
	p=4.0;
	chg=STOCKHOLDER;
	break;

      default:
	ERROR_INFO();
	throw std::runtime_error("Not implemented.\n");
      }
      
      // If only one nucleus - nothing to do!
      if(basis.get_Nnuc()==1)
	continue;

      func=new Pipek(chg,basis,C,P,p,verbose);
    
    } else if(met==EDMISTON) {
      func=new Edmiston(basis,C);

    } else {
      ERROR_INFO();
      throw std::runtime_error("Method not implemented.\n");
    }

    // Set matrix
    func->setW(W);
    // Log file?
    if(im==metstack.size()-1) {
      opt.open_log(fname);
      opt.set_debug(debug);
    }
    // Run optimization
    opt.optimize(func,umet,uacc,maxiter);
    // Get updated matrix
    W=func->getW();
    // and cost function value
    measure=func->getf();
    // Free worker
    delete func;
  }

  if(verbose) {
    printf("Localization done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

Boys::Boys(const BasisSet & basis, const arma::mat & C, int nv, bool ver, bool delocalize) : UnitaryFunction(4*nv,delocalize) {
  // Save n
  n=nv;

  Timer t;
  if(ver) {
    printf("Computing r^2 and dipole matrices ...");
    fflush(stdout);
  }

  // Get R^2 matrix
  std::vector<arma::mat> momstack=basis.moment(2);
  rsq=momstack[getind(2,0,0)]+momstack[getind(0,2,0)]+momstack[getind(0,0,2)];

  // Get r matrices
  std::vector<arma::mat> rmat=basis.moment(1);

  // Convert matrices to MO basis
  rsq=arma::trans(C)*rsq*C;
  rx=arma::trans(C)*rmat[0]*C;
  ry=arma::trans(C)*rmat[1]*C;
  rz=arma::trans(C)*rmat[2]*C;

  if(ver) {
    printf(" done (%s)\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

Boys::~Boys() {
}

Boys* Boys::copy() const {
  return new Boys(*this);
}

void Boys::set_n(int nv) {
  n=nv;

  // Set q accordingly
  q=4*(n+1);
}


double Boys::cost_func(const arma::cx_mat & Wv) {
  W=Wv;

  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != rsq.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << rsq.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  double B=0;

  // For <i|r^2|i> terms
  arma::cx_mat rsw=rsq*W;
  // For <i|r|i>^2 terms
  arma::cx_mat rxw=rx*W;
  arma::cx_mat ryw=ry*W;
  arma::cx_mat rzw=rz*W;

  // Loop over orbitals
#ifdef _OPENMP
#pragma omp parallel for reduction(+:B)
#endif
  for(size_t io=0;io<W.n_cols;io++) {
    // <r^2> term
    double w=std::real(arma::as_scalar(arma::trans(W.col(io))*rsw.col(io)));

    // <r>^2 terms
    double xp=std::real(arma::as_scalar(arma::trans(W.col(io))*rxw.col(io)));
    double yp=std::real(arma::as_scalar(arma::trans(W.col(io))*ryw.col(io)));
    double zp=std::real(arma::as_scalar(arma::trans(W.col(io))*rzw.col(io)));
    w-=xp*xp + yp*yp + zp*zp;

    // Add to total
    B+=pow(w,n);
  }
  f=B;
  
  return f;
}

arma::cx_mat Boys::cost_der(const arma::cx_mat & Wv) {
  W=Wv;

  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != rsq.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << rsq.n_cols << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Returned matrix
  arma::cx_mat Bder(W.n_cols,W.n_cols);
  arma::cx_mat rsw=rsq*W;
  arma::cx_mat rxw=rx*W;
  arma::cx_mat ryw=ry*W;
  arma::cx_mat rzw=rz*W;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t b=0;b<W.n_cols;b++) {
    // Helpers for r terms
    double xp=std::real(arma::as_scalar(arma::trans(W.col(b))*rxw.col(b)));
    double yp=std::real(arma::as_scalar(arma::trans(W.col(b))*ryw.col(b)));
    double zp=std::real(arma::as_scalar(arma::trans(W.col(b))*rzw.col(b)));

    // Normal Boys contribution
    double w=std::real(arma::as_scalar(arma::trans(W.col(b))*rsw.col(b)));
    w-=xp*xp + yp*yp + zp*zp;

    // r^2 terms
    for(size_t a=0;a<W.n_cols;a++) {
      // Compute derivative
      std::complex<double> dert=rsw(a,b) - 2.0*(xp*rxw(a,b) + yp*ryw(a,b) + zp*rzw(a,b));

      // Set derivative
      Bder(a,b)=n*pow(w,n-1)*dert;
    }
  }

  return Bder;
}

void Boys::cost_func_der(const arma::cx_mat & Wv, double & fv, arma::cx_mat & der) {
  fv=cost_func(Wv);
  der=cost_der(Wv);
}


FMLoc::FMLoc(const BasisSet & basis, const arma::mat & C, int nv, bool ver, bool delocalize) : UnitaryFunction(8*nv,delocalize) {
  // Save n
  n=nv;

  Timer t;
  if(ver) {
    printf("Computing r^4, r^3, r^2 and r matrices ...");
    fflush(stdout);
  }

  // Get the r_i^2 r_j^2 matrices
  std::vector<arma::mat> momstack=basis.moment(4);
  // Diagonal: x^4 + y^4 + z^4
  rfour=momstack[getind(4,0,0)] + momstack[getind(0,4,0)] + momstack[getind(0,0,4)] \
    // Off-diagonal: 2 x^2 y^2 + 2 x^2 z^2 + 2 y^2 z^2
    +2.0*(momstack[getind(2,2,0)]+momstack[getind(2,0,2)]+momstack[getind(0,2,2)]);
  // Convert to MO basis
  rfour=arma::trans(C)*rfour*C;

  // Get R^3 matrices
  momstack=basis.moment(3);
  rrsq.resize(3);
  // x^3 + xy^2 + xz^2
  rrsq[0]=momstack[getind(3,0,0)]+momstack[getind(1,2,0)]+momstack[getind(1,0,2)];
  // x^2y + y^3 + yz^2
  rrsq[1]=momstack[getind(2,1,0)]+momstack[getind(0,3,0)]+momstack[getind(0,1,2)];
  // x^2z + y^2z + z^3
  rrsq[2]=momstack[getind(2,0,1)]+momstack[getind(0,2,1)]+momstack[getind(0,0,3)];
  // and convert to the MO basis
  for(int ic=0;ic<3;ic++)
    rrsq[ic]=arma::trans(C)*rrsq[ic]*C;

  // Get R^2 matrix
  momstack=basis.moment(2);
  // and convert to the MO basis
  for(size_t i=0;i<momstack.size();i++) {
    momstack[i]=arma::trans(C)*momstack[i]*C;
  }
  rr.resize(3);
  for(int ic=0;ic<3;ic++)
    rr[ic].resize(3);

  // Diagonal
  rr[0][0]=momstack[getind(2,0,0)];
  rr[1][1]=momstack[getind(0,2,0)];
  rr[2][2]=momstack[getind(0,0,2)];

  // Off-diagonal
  rr[0][1]=momstack[getind(1,1,0)];
  rr[1][0]=rr[0][1];

  rr[0][2]=momstack[getind(1,0,1)];
  rr[2][0]=rr[0][2];

  rr[1][2]=momstack[getind(0,1,1)];
  rr[2][1]=rr[1][2];

  // and the rsq matrix
  rsq=rr[0][0]+rr[1][1]+rr[2][2];

  // Get r matrices
  rmat=basis.moment(1);
  // and convert to the MO basis
  for(size_t i=0;i<rmat.size();i++) {
    rmat[i]=arma::trans(C)*rmat[i]*C;
  }

  if(ver) {
    printf(" done (%s)\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

FMLoc::~FMLoc() {
}

FMLoc* FMLoc::copy() const {
  return new FMLoc(*this);
}

void FMLoc::set_n(int nv) {
  n=nv;

  // Set q accordingly
  q=8*(nv+1);
}

double FMLoc::cost_func(const arma::cx_mat & Wv) {
  W=Wv;
  
  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != rsq.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << rsq.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  double B=0;

  // For <i|r^4|i> terms
  arma::cx_mat rfw=rfour*W;
  // For <i|r^3|i> terms
  std::vector<arma::cx_mat> rrsqw(3);
  for(int ic=0;ic<3;ic++)
    rrsqw[ic]=rrsq[ic]*W;
  // For <i|r^2|i> terms
  std::vector< std::vector<arma::cx_mat> > rrw(3);
  for(int ic=0;ic<3;ic++) {
    rrw[ic].resize(3);
    for(int jc=0;jc<3;jc++)
      rrw[ic][jc]=rr[ic][jc]*W;
  }
  arma::cx_mat rsqw=rsq*W;
  // For <i|r|i> terms
  std::vector<arma::cx_mat> rw(3);
  for(int ic=0;ic<3;ic++)
    rw[ic]=rmat[ic]*W;

  // Loop over orbitals
#ifdef _OPENMP
#pragma omp parallel for reduction(+:B)
#endif
  for(size_t io=0;io<W.n_cols;io++) {
    // <r^4> term
    double rfour_t=std::real(arma::as_scalar(arma::trans(W.col(io))*rfw.col(io)));

    // rrsq
    arma::vec rrsq_t(3);
    for(int ic=0;ic<3;ic++)
      rrsq_t(ic)=std::real(arma::as_scalar(arma::trans(W.col(io))*rrsqw[ic].col(io)));

    // rr
    arma::mat rr_t(3,3);
    for(int ic=0;ic<3;ic++)
      for(int jc=0;jc<=ic;jc++) {
	rr_t(ic,jc)=std::real(arma::as_scalar(arma::trans(W.col(io))*rrw[ic][jc].col(io)));
	rr_t(jc,ic)=rr_t(ic,jc);
      }

    // rsq
    double rsq_t=std::real(arma::as_scalar(arma::trans(W.col(io))*rsqw.col(io)));

    // r
    arma::vec r_t(3);
    for(int ic=0;ic<3;ic++)
      r_t(ic)=std::real(arma::as_scalar(arma::trans(W.col(io))*rw[ic].col(io)));

    // Collect terms
    double w= rfour_t - 4.0*arma::dot(rrsq_t,r_t) + 2.0*rsq_t*arma::dot(r_t,r_t) + 4.0 * arma::as_scalar(arma::trans(r_t)*rr_t*r_t) - 3.0*std::pow(arma::dot(r_t,r_t),2);

    // Add to total
    B+=pow(w,n);
  }
  f=B;

  return B;
}

arma::cx_mat FMLoc::cost_der(const arma::cx_mat & Wv) {
  W=Wv;

  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != rsq.n_rows) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << rsq.n_rows << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Returned matrix
  arma::cx_mat Bder(W.n_cols,W.n_cols);

  // For <i|r^4|i> terms
  arma::cx_mat rfw=rfour*W;
  // For <i|r^3|i> terms
  std::vector<arma::cx_mat> rrsqw(3);
  for(int ic=0;ic<3;ic++)
    rrsqw[ic]=rrsq[ic]*W;
  // For <i|r^2|i> terms
  std::vector< std::vector<arma::cx_mat> > rrw(3);
  for(int ic=0;ic<3;ic++) {
    rrw[ic].resize(3);
    for(int jc=0;jc<3;jc++)
      rrw[ic][jc]=rr[ic][jc]*W;
  }
  arma::cx_mat rsqw=rsq*W;
  // For <i|r|i> terms
  std::vector<arma::cx_mat> rw(3);
  for(int ic=0;ic<3;ic++)
    rw[ic]=rmat[ic]*W;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t io=0;io<W.n_cols;io++) {
    // <r^4> term
    double rfour_t=std::real(arma::as_scalar(arma::trans(W.col(io))*rfw.col(io)));

    // rrsq
    arma::vec rrsq_t(3);
    for(int ic=0;ic<3;ic++)
      rrsq_t(ic)=std::real(arma::as_scalar(arma::trans(W.col(io))*rrsqw[ic].col(io)));

    // rr
    arma::mat rr_t(3,3);
    for(int ic=0;ic<3;ic++)
      for(int jc=0;jc<=ic;jc++) {
	rr_t(ic,jc)=std::real(arma::as_scalar(arma::trans(W.col(io))*rrw[ic][jc].col(io)));
	rr_t(jc,ic)=rr_t(ic,jc);
      }

    // rsq
    double rsq_t=std::real(arma::as_scalar(arma::trans(W.col(io))*rsqw.col(io)));

    // r
    arma::vec r_t(3);
    for(int ic=0;ic<3;ic++)
      r_t(ic)=std::real(arma::as_scalar(arma::trans(W.col(io))*rw[ic].col(io)));

    // Collect terms
    double w= rfour_t - 4.0*arma::dot(rrsq_t,r_t) + 2.0*rsq_t*arma::dot(r_t,r_t) + 4.0 * arma::as_scalar(arma::trans(r_t)*rr_t*r_t) - 3.0*std::pow(arma::dot(r_t,r_t),2);

    // Compute derivative
    for(size_t a=0;a<W.n_cols;a++) {

      // <r^4> term
      std::complex<double> rfour_d=rfw(a,io);

      // rrsq
      arma::cx_vec rrsq_d(3);
      for(int ic=0;ic<3;ic++)
	rrsq_d(ic)=rrsqw[ic](a,io);

      // rr
      arma::cx_mat rr_d(3,3);
      for(int ic=0;ic<3;ic++)
	for(int jc=0;jc<3;jc++) {
	  rr_d(ic,jc)=rrw[ic][jc](a,io);
	}

      // rsq
      std::complex<double> rsq_d=rsqw(a,io);

      // r
      arma::cx_vec r_d(3);
      for(int ic=0;ic<3;ic++)
	r_d(ic)=rw[ic](a,io);

      // Derivative is
      std::complex<double> one(1.0,0.0);
      std::complex<double> dert=rfour_d - 4.0*(arma::dot(one*rrsq_t,r_d)+arma::dot(rrsq_d,one*r_t)) + 2.0*rsq_d*arma::dot(r_t,r_t) + 4.0*rsq_t*arma::dot(one*r_t,r_d) + 8.0*arma::as_scalar((one*(arma::trans(r_t)*rr_t))*r_d) + 4.0*arma::as_scalar(arma::trans(one*r_t)*rr_d*(one*r_t)) - 12.0*arma::dot(r_t,r_t)*arma::dot(one*r_t,r_d);

      // Set derivative
      Bder(a,io)=n*pow(w,n-1)*dert;
    }
  }

  return Bder;
}

void FMLoc::cost_func_der(const arma::cx_mat & Wv, double & fv, arma::cx_mat & der) {
  fv=cost_func(Wv);
  der=cost_der(Wv);
}


Pipek::Pipek(enum chgmet chgv, const BasisSet & basis, const arma::mat & Cv, const arma::mat & P, double pv, bool ver, bool delocalize) : UnitaryFunction(2*pv,!delocalize) {
  // Store used method
  chg=chgv;
  C=Cv;
  // and penalty exponent
  p=pv;

  // Overlap matrix tolerance threshold
  double otol=1e-5;

  Timer t;
  if(ver) {
    printf("Initializing Pipek-Mezey calculation with ");
    if(chg==BADER)
      printf("Bader");
    else if(chg==BECKE)
      printf("Becke");
    else if(chg==HIRSHFELD)
      printf("Hirshfeld");
    else if(chg==ITERHIRSH)
      printf("iterative Hirshfeld");
    else if(chg==IAO)
      printf("IAO");
    else if(chg==LOWDIN)
      printf("Löwdin");
    else if(chg==MULLIKEN)
      printf("Mulliken");
    else if(chg==STOCKHOLDER)
      printf("Stockholder");
    else if(chg==VORONOI)
      printf("Voronoi");
    printf(" charges.\n");
    fflush(stdout);
  }

  if(chg==BADER || chg==VORONOI) {
    // Helper. Non-verbose operation
    bader.set(basis,ver);
    // Construct integration grid
    if(chg==BADER)
      bader.construct_bader(P,otol);
    else
      bader.construct_voronoi(otol);
    // Amount of regions
    N=bader.get_Nmax();

  } else if(chg==BECKE) {
    // Amount of regions
    N=basis.get_Nnuc();
    // Grid
    grid=DFTGrid(&basis,ver);
    // Construct integration grid
    grid.construct_becke(otol);

  } else if(chg==HIRSHFELD) {
    // Amount of regions
    N=basis.get_Nnuc();
    // We don't know method here so just use HF.
    hirsh.compute(basis,"HF");

    // Helper. Non-verbose operation
    //      DFTGrid intgrid(&basis,false);
    grid=DFTGrid(&basis,ver);
    // Construct integration grid
    grid.construct_hirshfeld(hirsh,otol);

  } else if(chg==ITERHIRSH) {
    // Amount of regions
    N=basis.get_Nnuc();
    // Iterative Hirshfeld atomic charges
    HirshfeldI hirshi;
    hirshi.compute(basis,P);
    // Helper
    hirsh=hirshi.get();

    // Helper. Non-verbose operation
    //      DFTGrid intgrid(&basis,false);
    grid=DFTGrid(&basis,ver);
    // Construct integration grid
    grid.construct_hirshfeld(hirsh,otol);

  } else if(chg==IAO) {
    // Amount of regions
    N=basis.get_Nnuc();

    basis.print();

    // Construct IAO orbitals
    C_iao=construct_IAO(basis,C,idx_iao);

    // Also need overlap matrix
    S=basis.overlap();

  } else if(chg==STOCKHOLDER) {
    // Amount of regions
    N=basis.get_Nnuc();
    // Stockholder atomic charges
    Stockholder stock(basis,P);
    // Helper
    hirsh=stock.get();

    // Helper. Non-verbose operation
    //      DFTGrid intgrid(&basis,false);
    grid=DFTGrid(&basis,ver);
    // Construct integration grid
    grid.construct_hirshfeld(hirsh,otol);

  } else if(chg==MULLIKEN) {
    // Amount of regions
    N=basis.get_Nnuc();
    // Get overlap matrix
    S=basis.overlap();
    // Get shells
    shells.resize(N);
    for(size_t i=0;i<N;i++)
      shells[i]=basis.get_funcs(i);

  } else if(chg==LOWDIN) {
    // Amount of regions
    N=basis.get_Nnuc();
    // Get overlap matrix
    S=basis.overlap();

    // Get S^1/2 (and S^-1/2)
    arma::mat Sinvh;
    S_half_invhalf(S,Sh,Sinvh);

    // Get shells
    shells.resize(N);
    for(size_t i=0;i<N;i++)
      shells[i]=basis.get_funcs(i);

  } else {
    ERROR_INFO();
    throw std::runtime_error("Charge method not implemented.\n");
  }

  if(ver) {
    printf("Initialization of Pipek-Mezey took %s\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

Pipek::~Pipek() {
}

Pipek* Pipek::copy() const {
  return new Pipek(*this);
}

arma::mat Pipek::get_charge(size_t iat) {
  arma::mat Q;

  if(chg==BADER || chg==VORONOI) {
    arma::mat Sat=bader.regional_overlap(iat);
    Q=arma::trans(C)*Sat*C;

  } else if(chg==BECKE) {
    arma::mat Sat=grid.eval_overlap(iat);
    Q=arma::trans(C)*Sat*C;

  } else if(chg==HIRSHFELD || chg==ITERHIRSH || chg==STOCKHOLDER) {
    arma::mat Sat=grid.eval_hirshfeld_overlap(hirsh,iat);
    Q=arma::trans(C)*Sat*C;

  } else if(chg==IAO) {

    // Construct IAO density matrix
    arma::mat Piao(C.n_rows, C.n_rows);
    Piao.zeros();
    for(size_t fi=0;fi<idx_iao[iat].size();fi++) {
      // Index of IAO is
      size_t io=idx_iao[iat][fi];
      // Add to IAO density
      Piao+=C_iao.col(io)*arma::trans(C_iao.col(io));
    }
    Q=arma::trans(C)*S*Piao*S*C;

  } else if(chg==LOWDIN) {
    Q.zeros(C.n_cols,C.n_cols);

    // Helper matrix
    arma::mat ShC=Sh*C;

    for(size_t io=0;io<C.n_cols;io++)
      for(size_t jo=0;jo<C.n_cols;jo++)
	for(size_t is=0;is<shells[iat].size();is++)
	  for(size_t fi=shells[iat][is].get_first_ind();fi<=shells[iat][is].get_last_ind();fi++)
	    Q(io,jo)+=ShC(fi,io)*ShC(fi,jo);

  } else if(chg==MULLIKEN) {
    Q.zeros(C.n_cols,C.n_cols);

    // Helper matrix
    arma::mat SC=S*C;

    // Increment charge
    for(size_t io=0;io<C.n_cols;io++)
      for(size_t jo=0;jo<C.n_cols;jo++)
	for(size_t is=0;is<shells[iat].size();is++)
	  for(size_t fi=shells[iat][is].get_first_ind();fi<=shells[iat][is].get_last_ind();fi++)
	    Q(io,jo)+=C(fi,io)*SC(fi,jo);

    // Symmetrize
    Q=(Q+arma::trans(Q))/2.0;

  } else {
    ERROR_INFO();
    throw std::runtime_error("Charge method not implemented.\n");
  }

  return Q;
}

double Pipek::cost_func(const arma::cx_mat & Wv) {
  W=Wv;

  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != C.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << C.n_cols << "!\n";
    throw std::runtime_error(oss.str());
  }

  double Dinv=0;

  // Compute sum
#ifdef _OPENMP
#pragma omp parallel for reduction(+:Dinv) schedule(dynamic,1)
#endif
  for(size_t iat=0;iat<N;iat++) {
      // Helper matrix
    arma::cx_mat qw=get_charge(iat)*W;
    for(size_t io=0;io<W.n_cols;io++) {
      std::complex<double> Qa=std::real(arma::as_scalar(arma::trans(W.col(io))*qw.col(io)));
      Dinv+=std::real(std::pow(Qa,p));
    }
  }
  f=Dinv;

  return Dinv;
}

arma::cx_mat Pipek::cost_der(const arma::cx_mat & Wv) {
  W=Wv;

  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != C.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << C.n_cols << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Returned matrix
  arma::cx_mat Dder(W.n_cols,W.n_cols);
  Dder.zeros();

  // Compute sum
#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    arma::cx_mat Dwrk(Dder);
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t iat=0;iat<N;iat++) {
      // Helper matrix
      arma::cx_mat qw=get_charge(iat)*W;

      for(size_t b=0;b<W.n_cols;b++) {
	std::complex<double> qwp=arma::as_scalar(arma::trans(W.col(b))*qw.col(b));
	std::complex<double> t=p*std::pow(qwp,p-1);

	for(size_t a=0;a<W.n_cols;a++) {
#ifdef _OPENMP
	  Dwrk(a,b)+=t*qw(a,b);
#else
	  Dder(a,b)+=t*qw(a,b);
#endif
	}
      }
    }

#ifdef _OPENMP
#pragma omp critical
    Dder+=Dwrk;
#endif
  }

  return Dder;
}

void Pipek::cost_func_der(const arma::cx_mat & Wv, double & Dinv, arma::cx_mat & Dder) {
  W=Wv;

  if(W.n_rows != W.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(W.n_rows != C.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << C.n_cols << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Returned matrix
  Dder.zeros(W.n_cols,W.n_cols);
  double D=0;

  // Compute sum
#ifdef _OPENMP
#pragma omp parallel reduction(+:D)
#endif
  {

#ifdef _OPENMP
    arma::cx_mat Dwrk(Dder);
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t iat=0;iat<N;iat++) {
      // Helper matrix
      arma::cx_mat qw=get_charge(iat)*W;

      for(size_t b=0;b<W.n_cols;b++) {
	std::complex<double> qwp=arma::as_scalar(arma::trans(W.col(b))*qw.col(b));
	std::complex<double> t=p*std::pow(qwp,p-1);
	D+=std::real(std::pow(qwp,p));

	for(size_t a=0;a<W.n_cols;a++) {
#ifdef _OPENMP
	  Dwrk(a,b)+=t*qw(a,b);
#else
	  Dder(a,b)+=t*qw(a,b);
#endif
	}
      }
    }

#ifdef _OPENMP
#pragma omp critical
    Dder+=Dwrk;
#endif
  }

  Dinv=D;
  f=D;
}

Edmiston::Edmiston(const BasisSet & basis, const arma::mat & Cv, bool delocalize) : UnitaryFunction(4,!delocalize) {
  // Store orbitals
  C=Cv;
  // Initialize fitting integrals. Direct computation, linear dependence threshold 1e-8, use Hartree-Fock routine since it has better tolerance for linear dependencies
  dfit.fill(basis,basis.density_fitting(),true,1e-8,false);
}

Edmiston::~Edmiston() {
}

Edmiston* Edmiston::copy() const {
  return new Edmiston(*this);
}

void Edmiston::setW(const arma::cx_mat & Wv) {
  // We need to update everything to match W
  arma::cx_mat der;
  cost_func_der(Wv,f,der);  
}

double Edmiston::cost_func(const arma::cx_mat & Wv) {
  arma::cx_mat der;
  cost_func_der(Wv,f,der);
  return f;
}

arma::cx_mat Edmiston::cost_der(const arma::cx_mat & Wv) {
  arma::cx_mat der;
  cost_func_der(Wv,f,der);
  return der;
}

void Edmiston::cost_func_der(const arma::cx_mat & Wv, double & fv, arma::cx_mat & der) {
  if(Wv.n_rows != Wv.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(Wv.n_rows != C.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << W.n_rows << " vs " << C.n_cols << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Get transformed orbitals
  arma::cx_mat Ctilde=C*Wv;

  // Orbital density matrices
  std::vector<arma::mat> Porb(Wv.n_cols);
  for(size_t io=0;io<Wv.n_cols;io++)
    Porb[io]=arma::real( Ctilde.col(io)*arma::trans(Ctilde.col(io)) );
  
  // Check if we need to do something
  if(W.n_rows != Wv.n_rows || W.n_cols != Wv.n_cols || rms_cnorm(W-Wv)>=DBL_EPSILON) {
    // Compute orbital-dependent Fock matrices
    W=Wv;
    Jorb=dfit.calc_J(Porb);
  }

  // Compute self-repulsion
  f=0.0;
  for(size_t io=0;io<W.n_cols;io++)
    f+=arma::trace(Porb[io]*Jorb[io]);
  fv=f;

  // Compute derivative
  der.zeros(W.n_cols,W.n_cols);
  for(size_t a=0;a<W.n_cols;a++)
    for(size_t b=0;b<W.n_cols;b++)
      der(a,b) =2.0 * arma::as_scalar( arma::trans(C.col(a))*Jorb[b]*Ctilde.col(b) );
}

PZSIC::PZSIC(SCF *solverp, dft_t dftp, DFTGrid * gridp, DFTGrid * nlgridp, double maxtolv, double rmstolv, enum pzham hm) : UnitaryFunction(4,true) {
  solver=solverp;
  dft=dftp;
  grid=gridp;
  nlgrid=nlgridp;

  // Convergence criteria
  rmstol=rmstolv;
  maxtol=maxtolv;

  // Hamiltonian
  ham=hm;
}

PZSIC::~PZSIC() {
}

PZSIC* PZSIC::copy() const {
  return new PZSIC(*this);
}

void PZSIC::set(const rscf_t & solp, double pz) {
  sol=solp;
  pzcor=pz;
}

void PZSIC::setW(const arma::cx_mat & Wv) {
  // We need to update everything to match W
  arma::cx_mat der;
  cost_func_der(Wv,f,der);  
}

double PZSIC::cost_func(const arma::cx_mat & Wv) {
  // Just do the derivative, too, since we need it anyhow to calculate kappa.
  double fv;
  arma::cx_mat der;
  cost_func_der(Wv,fv,der);
  return fv;
    
  // Evaluate SI energy.
  if(Wv.n_rows != Wv.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }
  
  if(Wv.n_rows > sol.cC.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << Wv.n_rows << " rotations vs " << sol.cC.n_cols << "CMOs!\n";
    throw std::runtime_error(oss.str());
  }

  // Check if we need to do something
  if(W.n_rows != Wv.n_rows || W.n_cols != Wv.n_cols || rms_cnorm(W-Wv)>=DBL_EPSILON) {
    // Compute orbital energies
    W=Wv;
    Forb.clear();
    solver->PZSIC_Fock(Forb,Eorb,sol.cC,Wv,dft,*grid,*nlgrid,false);
  }
  if(Eorb.size() != Wv.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Problem in PZSIC: have " << Wv.n_cols << " orbitals but " << Eorb.size() << " energies!\n";
    throw std::runtime_error(oss.str());
  }

  // SI energy is
  f=arma::sum(Eorb);
  return f;
}

arma::cx_mat PZSIC::cost_der(const arma::cx_mat & Wv) {
  // Evaluate derivative

  arma::cx_mat der;
  double ESIC;
  cost_func_der(Wv,ESIC,der);
  return der;
}

void PZSIC::cost_func_der(const arma::cx_mat & Wv, double & fv, arma::cx_mat & der) {
   if(Wv.n_rows != Wv.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Matrix is not square!\n");
  }

  if(Wv.n_rows > sol.cC.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Matrix does not match size of problem: " << Wv.n_rows << " rotations vs " << sol.cC.n_cols << "CMOs!\n";
    throw std::runtime_error(oss.str());
  }

  // Check if we need to do something
  if(W.n_rows != Wv.n_rows || W.n_cols != Wv.n_cols || rms_cnorm(W-Wv)>=DBL_EPSILON || Forb.size() != Wv.n_cols) {
    // Compute orbital-dependent Fock matrices
    W=Wv;
    solver->PZSIC_Fock(Forb,Eorb,sol.cC,Wv,dft,*grid,*nlgrid,true);
  }
  if(Eorb.size() != Wv.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Problem in PZSIC: have " << Wv.n_cols << " orbitals but " << Eorb.size() << " energies!\n";
    throw std::runtime_error(oss.str());
  }
  if(Forb.size() != Wv.n_cols) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Problem in PZSIC: have " << Wv.n_cols << " orbitals but " << Forb.size() << " potentials!\n";
    throw std::runtime_error(oss.str());
  }

  // Get transformed orbitals
  arma::cx_mat Ctilde=sol.cC.cols(0,Wv.n_cols-1)*Wv;
  // and the total SIC contribution.
  HSIC.zeros(Ctilde.n_rows,Ctilde.n_rows);
  if(ham==PZSYMM) {
    // Symmetrized operator
    arma::mat S=solver->get_S();
    for(size_t io=0;io<Ctilde.n_cols;io++) {
      arma::cx_mat Pio=Ctilde.col(io)*arma::trans(Ctilde.col(io));
      arma::cx_mat Fio=Forb[io];
      HSIC+=(Fio*Pio*S + S*Pio*Fio)/2.0;
    }

  } else if(ham==PZUNITED) {
    // Unified Hamiltonian. See
    // J. G. Harrison, R. A. Heaton, and C. C. Lin, "Self-interaction
    // correction to the local density Hartree-Fock atomic
    // calculations of excited and ground states", J. Phys. B:
    // At. Mol. Phys. 16, 2079 (1983).

    arma::mat S=solver->get_S();
    arma::mat Sinvh=solver->get_Sinvh();

    // Construct virtual space projector
    arma::cx_mat O(S*COMPLEX1);
    O.zeros();
    // Add in total possible density
    for(size_t i=0;i<Sinvh.n_cols;i++)
      O+=Sinvh.col(i)*arma::trans(Sinvh.col(i))*COMPLEX1;
    // and substract the occupied density
    for(size_t io=0;io<sol.cC.n_cols;io++)
      O-=sol.cC.col(io)*arma::trans(sol.cC.col(io));

    // Construct Hamiltonian
    for(size_t io=0;io<Ctilde.n_cols;io++) {
      arma::cx_mat Pio=Ctilde.col(io)*arma::trans(Ctilde.col(io));
      arma::cx_mat Fio=Forb[io];
      HSIC+=S*( Pio*Fio*Pio + O*Fio*Pio + Pio*Fio*O )*S;
    }

  } else {
    std::ostringstream oss;
    oss << "Unsupported PZ-SIC Hamiltonian!\n";
    throw std::runtime_error(oss.str());
  }

  // SI energy is
  fv=f=arma::sum(Eorb);

  // Derivative is
  der.zeros(Ctilde.n_cols,Ctilde.n_cols);
  for(size_t io=0;io<Ctilde.n_cols;io++)
    for(size_t jo=0;jo<Ctilde.n_cols;jo++)
      der(io,jo)=arma::as_scalar(arma::trans(sol.cC.col(io))*Forb[jo]*Ctilde.col(jo));

  // Kappa is
  kappa.zeros(Ctilde.n_cols,Ctilde.n_cols);
  for(size_t io=0;io<Ctilde.n_cols;io++)
    for(size_t jo=0;jo<Ctilde.n_cols;jo++)
      kappa(io,jo)=arma::as_scalar(arma::trans(Ctilde.col(io))*(Forb[jo]-Forb[io])*Ctilde.col(jo));
}

std::string PZSIC::legend() const {
  char leg[80];
  sprintf(leg,"%12s   %12s ","Kmax","Krms");
  return std::string(leg);
}

std::string PZSIC::status(bool lfmt) {
  double Krms, Kmax;
  get_k_rms_max(Krms,Kmax);

  char Km[80], Kr[80];
  char mc=(Kmax<maxtol) ? '*' : ' ';
  char rc=(Krms<rmstol) ? '*' : ' ';

  if(lfmt) {
    sprintf(Km,"%.16e",Kmax);
    sprintf(Kr,"%.16e",Krms);
  } else {
    sprintf(Km,"%e%c",Kmax,mc);
    sprintf(Kr,"%e%c",Krms,rc);
  }

  std::string stat=std::string(Km)+"  "+std::string(Kr);
  return stat;
}

void PZSIC::print_time(const Timer & t) const {
  printf("Iteration done in %s.\n",t.elapsed().c_str());
  fflush(stdout);

  fprintf(stderr,"\t%10.3f\n",t.get());
  fflush(stderr);
}

void PZSIC::get_k_rms_max(double & Krms, double & Kmax) {
  // Check kappa is up to date
  double fv;
  arma::cx_mat der;
  cost_func_der(W,fv,der);
  // Difference from Pedersen condition is
  Krms=rms_cnorm(kappa);
  Kmax=max_cabs(kappa);
}

bool PZSIC::converged() {
  double Krms, Kmax;
  get_k_rms_max(Krms,Kmax);

  if(Kmax<maxtol && Krms<rmstol)
    // Converged
    return true;
  else
    // Not converged
    return false;
}

double PZSIC::get_ESIC() const {
  return f;
}

arma::vec PZSIC::get_Eorb() const {
  return Eorb;
}

arma::cx_mat PZSIC::get_HSIC() {
  // Make sure HSIC is up to date
  double fv;
  arma::cx_mat der;
  cost_func_der(W,fv,der);
  return HSIC;
}

inline double complex_norm(double phi, const arma::mat & S, const arma::cx_vec & C) {
  // Get the imaginary norm of C rotated by exp(i phi)
  arma::vec c=arma::imag(C*exp(std::complex<double>(0.0,phi)));
  return as_scalar(arma::trans(c)*S*c);
}

double analyze_orbital(const arma::mat & S, const arma::cx_vec & C) {
  // Just do a full scan
  arma::vec phi(arma::linspace(0.0,2*M_PI,201));
  
  arma::vec cn(phi.n_elem);
  for(arma::uword i=0;i<cn.n_elem;i++)
    cn(i)=complex_norm(phi(i),S,C);
  
  return cn.min();
}

void analyze_orbitals(const BasisSet & basis, const arma::cx_mat & C) {
  arma::mat S(basis.overlap());
  arma::vec cnorms(C.n_cols);
  
  // Loop over orbitals
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(arma::uword i=0;i<C.n_cols;i++)
    cnorms(i)=analyze_orbital(S,C.col(i));

  for(arma::uword i=0;i<C.n_cols;i++)
    printf("Orbital %3i: norm of imaginary part %e\n",(int) i+1,cnorms(i));
}
