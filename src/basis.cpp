/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
// For exceptions
#include <sstream>
#include <stdexcept>

#include "basis.h"
#include "cintenv.h"
#include "eriworker.h"
#include "elements.h"
#include "integrals.h"
#include "linalg.h"
#include "mathf.h"
#include "settings.h"
#include "solidharmonics.h"
#include "stringutil.h"
#include "timer.h"

extern "C" {
#include <cint.h>
#include <cint_funcs.h>
}
// cint.h defines function-like atm() and bas() accessor macros, which
// mangle any same-named variable that is followed by a parenthesis
#undef atm
#undef bas


// Derivative operator
inline double _der1(const double x[], int l, double zeta) {
  double d=-2.0*zeta*x[l+1];
  if(l>0)
    d+=l*x[l-1];
  return d;
}

// Second derivative operator
inline double _der2(const double x[], int l, double zeta) {
  double d=4.0*zeta*zeta*x[l+2] - 2.0*zeta*(2*l+1)*x[l];
  if(l>1)
    d+=l*(l-1)*x[l-2];
  return d;
}

// Third derivative operator
inline double _der3(const double x[], int l, double zeta) {
  double d=-8.0*zeta*zeta*zeta*x[l+3] + 12.0*zeta*zeta*(l+1)*x[l+1];
  if(l>0)
    d-=6.0*zeta*l*l*x[l-1];
  if(l>2)
    d+=l*(l-1)*(l-2)*x[l-3];
  return d;
}

bool operator==(const nucleus_t & lhs, const nucleus_t & rhs) {
  return (lhs.ind == rhs.ind) && (lhs.r == rhs.r) && (lhs.Z == rhs.Z) && \
    (lhs.bsse == rhs.bsse) && (stricmp(lhs.symbol,rhs.symbol)==0);
}

bool operator==(const coords_t & lhs, const coords_t & rhs) {
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

arma::vec coords_to_vec(const coords_t & c) {
  arma::vec r(3);
  r(0)=c.x;
  r(1)=c.y;
  r(2)=c.z;
  return r;
}

coords_t vec_to_coords(const arma::vec & v) {
  if(v.n_elem != 3) {
    std::ostringstream oss;
    oss << "Expected a 3-element vector, got " << v.n_elem << "!\n";
    throw std::logic_error(oss.str());
  }
  coords_t r;
  r.x=v(0);
  r.y=v(1);
  r.z=v(2);
  return r;
}

// Operators for computing displacements
coords_t operator-(const coords_t & lhs, const coords_t & rhs) {
  coords_t ret;
  ret.x=lhs.x-rhs.x;
  ret.y=lhs.y-rhs.y;
  ret.z=lhs.z-rhs.z;
  return ret;
}

coords_t operator+(const coords_t & lhs, const coords_t& rhs) {
  coords_t ret;
  ret.x=lhs.x+rhs.x;
  ret.y=lhs.y+rhs.y;
  ret.z=lhs.z+rhs.z;
  return ret;
}

coords_t operator/(const coords_t & lhs, double fac) {
  coords_t ret;
  ret.x=lhs.x/fac;
  ret.y=lhs.y/fac;
  ret.z=lhs.z/fac;
  return ret;
}

coords_t operator*(const coords_t & lhs, double fac) {
  coords_t ret;
  ret.x=lhs.x*fac;
  ret.y=lhs.y*fac;
  ret.z=lhs.z*fac;
  return ret;
}

bool operator<(const contr_t & lhs, const contr_t & rhs) {
  // Decreasing order of exponents.
  return lhs.z>rhs.z;
}

bool operator==(const contr_t & lhs, const contr_t & rhs) {
  //  return (lhs.z==rhs.z) && (lhs.c==rhs.c);

  // Since this also needs to work for saved and reloaded basis sets, we need to relax the comparison.
  const double tol=sqrt(DBL_EPSILON);

  bool same=(fabs(lhs.z-rhs.z)<tol*std::max(1.0,fabs(lhs.z))) && (fabs(lhs.c-rhs.c)<tol*std::max(1.0,fabs(lhs.z)));

  /*
    if(!same) {
    fprintf(stderr,"Contractions differ: %e %e vs %e %e, diff %e %e!\n",lhs.c,lhs.z,rhs.c,rhs.z,rhs.c-lhs.c,rhs.z-lhs.z);
    }
  */

  return same;
}

GaussianShell::GaussianShell() {
  // Dummy constructor
}

GaussianShell::GaussianShell(int amv, bool lm, const std::vector<contr_t> & C) {
  // Construct a segmented shell of basis functions (one contraction).
  // A generally contracted shell is assembled by add_shells, which
  // fills the coefficient matrix cf with several columns.

  // Store contraction
  c_=C;
  // Sort the contraction
  sort();

  // A single contraction: the coefficient matrix is one column
  cf_.set_size(c_.size(),1);
  for(size_t i=0;i<c_.size();i++)
    cf_(i,0)=c_[i].c;

  // Set angular momentum
  am_=amv;
  // Use spherical harmonics?
  uselm_=lm;

  // If spherical harmonics are used, fill transformation matrix
  if(uselm_)
    transmat_=Ylm_transmat(am_);
  else {
    // Do away with uninitialized value warnings in valgrind
    transmat_=arma::mat(1,1);
    transmat_(0,0)=1.0/0.0; // Initialize to NaN
  }

  // Compute necessary amount of Cartesians
  size_t Ncart=(am_+1)*(am_+2)/2;
  // Allocate memory
  cart_.reserve(Ncart);
  cart_.resize(Ncart);
  // Initialize the shells

  int n=0;
  for(int i=0; i<=am_; i++) {
    int nx = am_ - i;
    for(int j=0; j<=i; j++) {
      int ny = i-j;
      int nz = j;

      cart_[n].l=nx;
      cart_[n].m=ny;
      cart_[n].n=nz;
      cart_[n].relnorm=1.0;
      n++;
    }
  }

  // Default values
  indstart_=0;
  cenind_=0;
  cen_.x=cen_.y=cen_.z=0.0;
}

GaussianShell::~GaussianShell() {
}

void GaussianShell::first_ind(size_t ind) {
  indstart_=ind;
}

void GaussianShell::set_center(const coords_t & cenv, size_t cenindv) {
  cen_=cenv;
  cenind_=cenindv;
}

void GaussianShell::sync_c() {
  // Mirror the first contraction's coefficients back into c, whose .z
  // fields carry the (authoritative) shared exponents
  for(size_t i=0;i<c_.size();i++)
    c_[i].c=cf_(i,0);
}

void GaussianShell::sort() {
  // Order the primitives by decreasing exponent. When the coefficient
  // matrix already exists (generally contracted shell) its rows follow
  // the same permutation.
  if(cf_.n_elem==0) {
    std::stable_sort(c_.begin(),c_.end());
    return;
  }

  std::vector<size_t> idx(c_.size());
  for(size_t i=0;i<idx.size();i++)
    idx[i]=i;
  std::stable_sort(idx.begin(),idx.end(),[this](size_t a, size_t b){ return c_[a]<c_[b]; });

  std::vector<contr_t> cnew(c_.size());
  arma::mat cfnew(cf_.n_rows,cf_.n_cols);
  for(size_t i=0;i<idx.size();i++) {
    cnew[i]=c_[idx[i]];
    cfnew.row(i)=cf_.row(idx[i]);
  }
  c_=cnew;
  cf_=cfnew;
}

void GaussianShell::convert_contraction() {
  // Convert contraction from contraction of normalized gaussians to
  // contraction of unnormalized gaussians.

  // Note - these refer to cartesian functions!
  double fac=pow(M_2_PI,0.75)*pow(2,am_)/sqrt(doublefact(2*am_-1));

  for(size_t i=0;i<c_.size();i++)
    cf_.row(i)*=fac*pow(c_[i].z,am_/2.0+0.75);
  sync_c();
}

void GaussianShell::convert_sap_contraction() {
  // Convert contraction from contraction of normalized density
  // gaussians to contraction of unnormalized gaussians.

  if(am_ != 0) throw std::logic_error("SAP basis should only have S functions!\n");
  for(size_t i=0;i<c_.size();i++)
    cf_.row(i)*=pow(c_[i].z/M_PI,1.5);
  sync_c();
}

void GaussianShell::normalize(bool coeffs) {
  // Normalize contraction of unnormalized primitives wrt first function on shell

  // Check for dummy shell
  if(c_.size()==1 && c_[0].z==0.0) {
    // Yes, this is a dummy.
    cf_(0,0)=1.0;
    sync_c();
    return;
  }

  if(coeffs) {
    const double angfac=pow(M_PI,1.5)*doublefact(2*am_-1)/pow(2.0,am_);

    // Normalize each contraction to unit self-overlap independently
    for(size_t ictr=0;ictr<cf_.n_cols;ictr++) {
      double fact=0.0;
      for(size_t i=0;i<c_.size();i++)
	for(size_t j=0;j<c_.size();j++)
	  fact+=cf_(i,ictr)*cf_(j,ictr)/pow(c_[i].z+c_[j].z,am_+1.5);
      fact*=angfac;
      fact=1.0/sqrt(fact);
      cf_.col(ictr)*=fact;
    }
    sync_c();
  }

  // FIXME: Do something more clever here.
  if(!uselm_) {
    // Compute relative normalization factors
    for(size_t i=0;i<cart_.size();i++)
      cart_[i].relnorm=sqrt(doublefact(2*am_-1)/(doublefact(2*cart_[i].l-1)*doublefact(2*cart_[i].m-1)*doublefact(2*cart_[i].n-1)));
  } else {
    // Compute self-overlap and scale the coefficients
    const arma::vec S=function_norms();
    for(size_t i=0;i<cart_.size();i++)
      cart_[i].relnorm/=sqrt(S(0));
  }
}

void GaussianShell::coulomb_normalize() {
  // Normalize functions using Coulomb norm
  size_t Ncart=cart_.size();
  size_t Nbf=this->Nbf();
  const size_t nctr=Nctr();

  // The Coulomb self-repulsion (i|j) of the functions on this shell.
  // The environment measures the current normalization of the shell, so
  // repeated calls compound as they did with the four-center integrals.
  CintEnv cenv(std::vector<GaussianShell>(1,*this),false);
  ERIWorker eri(cenv);
  eri.compute_2c(0,0);
  const std::vector<double> * erip=eri.getp();

  if(nctr==1) {
    if(!uselm_) {
      // Cartesian functions
      for(size_t i=0;i<Ncart;i++)
        cart_[i].relnorm*=1.0/sqrt((*erip)[i*Nbf+i]);
    } else {
      // Spherical normalization, need to distribute the normalization
      // coefficient among the cartesians, so all the functions of the
      // shell must have the same norm
      int diff=0;
      for(size_t i=1;i<Nbf;i++)
        if(fabs((*erip)[i*Nbf+i]-(*erip)[0])>sqrt(DBL_EPSILON)*(*erip)[0]) {
          printf("%e != %e, diff %e\n",(*erip)[i*Nbf+i],(*erip)[0],(*erip)[i*Nbf+i]-(*erip)[0]);
          fflush(stdout);
          diff++;
        }

      if(diff) {
        ERROR_INFO();
        std::ostringstream oss;
        oss << "\nSpherical functions have different norms!\n";
        throw std::runtime_error(oss.str());
      }

      // Scale coefficients
      for(size_t i=0;i<Ncart;i++)
        cart_[i].relnorm*=1.0/sqrt((*erip)[0]);
    }
    return;
  }

  // Generally contracted shell: the cartesian relnorm is shared by all
  // contractions, but the contractions have different Coulomb norms, so
  // normalize each one by scaling its coefficient column instead. Within
  // a single spherical contraction all 2l+1 functions have the same norm.
  if(!uselm_) {
    ERROR_INFO();
    throw std::runtime_error("Coulomb normalization of a generally contracted cartesian shell is not supported.\n");
  }
  const size_t Nfunc=Nbf/nctr;
  for(size_t ic=0;ic<nctr;ic++) {
    const size_t i0=ic*Nfunc;
    const double n0=(*erip)[i0*Nbf+i0];
    int diff=0;
    for(size_t i=1;i<Nfunc;i++)
      if(fabs((*erip)[(i0+i)*Nbf+(i0+i)]-n0)>sqrt(DBL_EPSILON)*n0) {
        printf("%e != %e, diff %e\n",(*erip)[(i0+i)*Nbf+(i0+i)],n0,(*erip)[(i0+i)*Nbf+(i0+i)]-n0);
        fflush(stdout);
        diff++;
      }
    if(diff) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "\nSpherical functions have different norms!\n";
      throw std::runtime_error(oss.str());
    }
    cf_.col(ic)*=1.0/sqrt(n0);
  }
  // Mirror the (scaled) first column back into the exponent carrier
  sync_c();
}

std::vector<contr_t> GaussianShell::contr() const {
  return c_;
}

const std::vector<contr_t> & GaussianShell::contr_ref() const {
  return c_;
}

std::vector<shellf_t> GaussianShell::cart() const {
  return cart_;
}

const std::vector<shellf_t> & GaussianShell::cart_ref() const {
  return cart_;
}

std::vector<contr_t> GaussianShell::contr_normalized() const {
  return contr_normalized(0);
}

std::vector<contr_t> GaussianShell::contr_normalized(size_t ictr) const {
  // The ictr'th contraction, its coefficients converted to those of
  // normalized primitives
  std::vector<contr_t> cn(contr(ictr));

  // Note - these refer to cartesian functions!
  double fac=pow(M_2_PI,0.75)*pow(2,am_)/sqrt(doublefact(2*am_-1));

  for(size_t i=0;i<cn.size();i++)
    cn[i].c/=fac*pow(cn[i].z,am_/2.0+0.75);

  return cn;
}

size_t GaussianShell::Nbf() const {
  // nctr angular blocks, one per contraction
  return Nctr() * (uselm_ ? Nlm() : Ncart());
}

size_t GaussianShell::Nctr() const {
  return cf_.n_cols;
}

bool GaussianShell::same_primitives(const GaussianShell & rhs) const {
  if(cenind_ != rhs.cenind_ || am_ != rhs.am_ || uselm_ != rhs.uselm_)
    return false;
  if(c_.size() != rhs.c_.size())
    return false;
  for(size_t i=0;i<c_.size();i++)
    if(c_[i].z != rhs.c_[i].z)
      return false;
  return true;
}

void GaussianShell::merge_contraction(const GaussianShell & rhs) {
  if(!same_primitives(rhs))
    throw std::logic_error("GaussianShell::merge_contraction: the shells do not share the same primitives.\n");
  // Append rhs's contraction columns after ours, preserving order
  cf_=arma::join_rows(cf_,rhs.cf_);
}

const arma::mat & GaussianShell::coefs() const {
  return cf_;
}

std::vector<contr_t> GaussianShell::contr(size_t ictr) const {
  std::vector<contr_t> ret(c_.size());
  for(size_t i=0;i<c_.size();i++) {
    ret[i].z=c_[i].z;
    ret[i].c=cf_(i,ictr);
  }
  return ret;
}

size_t GaussianShell::Nlm() const {
  return 2*am_+1;
}

double GaussianShell::range(double eps) const {
  double oldr;
  // Start at
  double r=1.0;

  double val;
  // Increase r so that value certainly has dropped below.
  do {
    // Increase value of r.
    oldr=r;
    r*=2.0;

    val=0.0;
    for(size_t i=0;i<c_.size();i++)
      val+=arma::max(arma::abs(cf_.row(i)))*exp(-c_[i].z*r*r);
    val*=pow(r,am_);
  } while(fabs(val)>eps);

  // OK, now the range lies in the range [oldr,r]. Use binary search to refine
  double left=oldr, right=r;
  double middle=(left+right)/2.0;

  while(right-left>10*DBL_EPSILON*right) {
    // Compute middle of interval
    middle=(left+right)/2.0;

    // Compute value in the middle
    val=0.0;
    for(size_t i=0;i<c_.size();i++)
      val+=arma::max(arma::abs(cf_.row(i)))*exp(-c_[i].z*middle*middle);
    val*=pow(middle,am_);

    // Switch values
    if(fabs(val)<eps) {
      // Switch right value
      right=middle;
    } else
      // Switch left value
      left=middle;
  }

  return middle;
}

bool GaussianShell::lm_in_use() const {
  return uselm_;
}

void GaussianShell::set_lm(bool lm) {
  uselm_=lm;

  if(uselm_)
    transmat_=Ylm_transmat(am_);
  else
    transmat_=arma::mat();
}

arma::mat GaussianShell::transmat() const {
  return transmat_;
}

size_t GaussianShell::Ncart() const {
  return cart_.size();
}

size_t GaussianShell::Ncontr() const {
  return c_.size();
}

int GaussianShell::am() const {
  return am_;
}

size_t GaussianShell::center_ind() const {
  //  return cen->ind;
  return cenind_;
}

coords_t GaussianShell::center() const {
  return cen_;
}

bool GaussianShell::operator<(const GaussianShell & rhs) const {
  // Sort first by nucleus
  if(cenind_ < rhs.cenind_)
    return true;
  else if(cenind_ == rhs.cenind_) {
    // Then by angular momentum
    if(am_<rhs.am_)
      return true;
    else if(am_==rhs.am_) {
      // Then by decreasing order of exponents
      if(c_.size() && rhs.c_.size())
	return c_[0].z>rhs.c_[0].z;
    }
  }

  return false;
}


bool GaussianShell::operator==(const GaussianShell & rhs) const {
  // Check first nucleus
  if(cenind_ != rhs.cenind_) {
    //    fprintf(stderr,"Center indices differ!\n");
    return false;
  }

  // Then, angular momentum
  if(am_!=rhs.am_) {
    //    fprintf(stderr,"Angular momentum differs!\n");
    return false;
  }

  // Then, by number of primitives and of contractions
  if(c_.size() != rhs.c_.size())
    return false;
  if(cf_.n_cols != rhs.cf_.n_cols)
    return false;

  // The exponents and the first contraction (contr_t carries a tolerance)
  for(size_t i=0;i<c_.size();i++)
    if(!(c_[i]==rhs.c_[i]))
      return false;

  // The remaining contraction columns
  const double tol=sqrt(DBL_EPSILON);
  for(size_t j=1;j<cf_.n_cols;j++)
    for(size_t i=0;i<cf_.n_rows;i++)
      if(std::fabs(cf_(i,j)-rhs.cf_(i,j)) > tol*std::max(1.0,std::fabs(cf_(i,j))))
        return false;

  return true;
}

size_t GaussianShell::first_ind() const {
  return indstart_;
}

size_t GaussianShell::last_ind() const {
  return indstart_+Nbf()-1;
}

void GaussianShell::print() const {

  printf("\t%c shell at nucleus %3i with with basis functions %4i-%-4i\n",shell_types[am_],(int) (center_ind()+1),(int) first_ind()+1,(int) last_ind()+1);
  printf("\t\tCenter of shell is at % 0.4f % 0.4f % 0.4f Å.\n",cen_.x/ANGSTROMINBOHR,cen_.y/ANGSTROMINBOHR,cen_.z/ANGSTROMINBOHR);

  // Get contraction of normalized primitives
  std::vector<contr_t> cn(contr_normalized());

  printf("\t\tExponential contraction is\n");
  printf("\t\t\tzeta\t\tprimitive coeff\ttotal coeff\n");
  for(size_t i=0;i<c_.size();i++)
    printf("\t\t\t%e\t% e\t% e\n",c_[i].z,cn[i].c,c_[i].c);
  if(uselm_) {
    printf("\t\tThe functions on this shell are:\n\t\t\t");
    for(int m=-am_;m<=am_;m++)
      printf(" (%i,%i)",am_,m);
    printf("\n");
  } else {
    printf("\t\tThe functions on this shell are:\n\t\t\t");
    for(size_t i=0;i<cart_.size();i++) {
      printf(" ");
      if(cart_[i].l+cart_[i].m+cart_[i].n==0)
	printf("1");
      else {
	for(int j=0;j<cart_[i].l;j++)
	  printf("x");
	for(int j=0;j<cart_[i].m;j++)
	  printf("y");
	for(int j=0;j<cart_[i].n;j++)
	  printf("z");
      }
    }
    printf("\n");
  }

  /*
    printf("\t\tThe cartesian functions on this shell are:\n");
    for(size_t i=0;i<cart.size();i++)
    printf("\t\t\t%i %i %i\t%0.6f\n",cart[i].l,cart[i].m,cart[i].n,cart[i].relnorm);
  */
}

arma::vec GaussianShell::eval_func(double x, double y, double z) const {
  // Evaluate basis functions at (x,y,z) via the fused evaluator, which
  // handles the generally contracted case
  arma::vec fval, lval;
  arma::mat gval, hval, lgval;
  eval_bf_derivs(x, y, z, fval, gval, lval, hval, lgval, false, false, false, false);
  return fval;
}

arma::mat GaussianShell::eval_grad(double x, double y, double z) const {
  // Thin wrapper around the fused evaluator: same inner-loop order
  // and per-iteration work as the original specialised
  // implementation, but the contracted-exponential evaluation and
  // power tables now share with eval_lapl/eval_hess/eval_laplgrad
  // when more than one is requested.
  arma::vec fval, lval;
  arma::mat gval, hval, lgval;
  eval_bf_derivs(x, y, z, fval, gval, lval, hval, lgval, true, false, false, false);
  return gval;
}

arma::vec GaussianShell::eval_lapl(double x, double y, double z) const {
  // Thin wrapper around the fused evaluator (cf. eval_grad).
  arma::vec fval, lval;
  arma::mat gval, hval, lgval;
  eval_bf_derivs(x, y, z, fval, gval, lval, hval, lgval, false, true, false, false);
  return lval;
}

void GaussianShell::eval_bf_derivs(double x, double y, double z,
                                   arma::vec & fval,
                                   arma::mat & gval,
                                   arma::vec & lval,
                                   arma::mat & hval,
                                   arma::mat & lgval,
                                   bool do_grad, bool do_lapl,
                                   bool do_hess, bool do_lgrad) const {
  // Evaluate the basis-function values plus any subset of gradient,
  // laplacian, Hessian and gradient-of-laplacian in one pass. The
  // five specialised eval_* siblings each rebuild xrel/yrel/zrel,
  // the power arrays, and the per-primitive exp(-z * rrelsq); a
  // fused pass amortises all of that across the requested outputs.

  const double xrel = x - cen_.x;
  const double yrel = y - cen_.y;
  const double zrel = z - cen_.z;
  const double rrelsq = xrel*xrel + yrel*yrel + zrel*zrel;

  // Power-array degree needed:
  //   func only           -> am
  //   grad                -> am + 1   (_der1 reads xr[l+1])
  //   lapl, hess          -> am + 2   (_der2 reads xr[l+2])
  //   laplgrad            -> am + 3   (_der3 reads xr[l+3])
  int xpow_max = am_;
  if(do_grad) xpow_max = std::max(xpow_max, am_ + 1);
  if(do_lapl || do_hess) xpow_max = std::max(xpow_max, am_ + 2);
  if(do_lgrad) xpow_max = std::max(xpow_max, am_ + 3);
  double xr[xpow_max+1], yr[xpow_max+1], zr[xpow_max+1];
  xr[0] = 1.0; yr[0] = 1.0; zr[0] = 1.0;
  if(xpow_max >= 1) {
    xr[1] = xrel; yr[1] = yrel; zr[1] = zrel;
    for(int i=2; i<=xpow_max; i++) {
      xr[i] = xr[i-1]*xrel;
      yr[i] = yr[i-1]*yrel;
      zr[i] = zr[i-1]*zrel;
    }
  }

  // Cartesian-basis accumulators (allocated only if requested). One
  // block of columns per contraction: the derivative components of
  // contraction ictr occupy columns [ictr*ncomp, (ictr+1)*ncomp). The
  // primitive exp() and the derivative factors are computed once (they
  // do not depend on the contraction); only the multiply-accumulate
  // into the columns scales with the number of contractions.
  const size_t nctr = cf_.n_cols;
  const size_t Ncart = cart_.size();
  arma::mat fbuf;  fbuf.zeros(Ncart, nctr);
  arma::mat gbuf;  if(do_grad) gbuf.zeros(Ncart, 3*nctr);
  arma::mat lbuf;  if(do_lapl) lbuf.zeros(Ncart, nctr);
  arma::mat hbuf;  if(do_hess) hbuf.zeros(Ncart, 9*nctr);
  arma::mat lgbuf; if(do_lgrad) lgbuf.zeros(Ncart, 3*nctr);

  for(size_t iexp=0; iexp<c_.size(); iexp++) {
    const double z_i = c_[iexp].z;
    // Bare Gaussian: no contraction coefficient (one exp per primitive)
    const double e_i = std::exp(-z_i * rrelsq);

    for(size_t icart=0; icart<Ncart; icart++) {
      const int l = cart_[icart].l;
      const int m = cart_[icart].m;
      const int n = cart_[icart].n;
      const double xl = xr[l];
      const double ym = yr[m];
      const double zn = zr[n];

      // Value term, bare
      const double v = xl * ym * zn * e_i;
      for(size_t ic=0; ic<nctr; ic++)
        fbuf(icart, ic) += cf_(iexp, ic) * v;

      // Derivative factors, computed once (independent of contraction)
      const bool need_d1 = do_grad || do_hess || do_lgrad;
      const bool need_d2 = do_lapl || do_hess || do_lgrad;
      const bool need_d3 = do_lgrad;
      const double d1x = need_d1 ? _der1(xr, l, z_i) : 0.0;
      const double d1y = need_d1 ? _der1(yr, m, z_i) : 0.0;
      const double d1z = need_d1 ? _der1(zr, n, z_i) : 0.0;
      const double d2x = need_d2 ? _der2(xr, l, z_i) : 0.0;
      const double d2y = need_d2 ? _der2(yr, m, z_i) : 0.0;
      const double d2z = need_d2 ? _der2(zr, n, z_i) : 0.0;
      const double d3x = need_d3 ? _der3(xr, l, z_i) : 0.0;
      const double d3y = need_d3 ? _der3(yr, m, z_i) : 0.0;
      const double d3z = need_d3 ? _der3(zr, n, z_i) : 0.0;

      if(do_grad) {
        const double gx = d1x * ym * zn * e_i;
        const double gy = xl  * d1y * zn * e_i;
        const double gz = xl  * ym * d1z * e_i;
        for(size_t ic=0; ic<nctr; ic++) {
          const double w = cf_(iexp, ic);
          gbuf(icart, 3*ic+0) += w * gx;
          gbuf(icart, 3*ic+1) += w * gy;
          gbuf(icart, 3*ic+2) += w * gz;
        }
      }

      if(do_lapl) {
        const double lp = (d2x * ym * zn + xl * d2y * zn + xl * ym * d2z) * e_i;
        for(size_t ic=0; ic<nctr; ic++)
          lbuf(icart, ic) += cf_(iexp, ic) * lp;
      }

      if(do_hess) {
        const double hxx = d2x * ym * zn * e_i;
        const double hyy = xl * d2y * zn * e_i;
        const double hzz = xl * ym * d2z * e_i;
        const double hxy = d1x * d1y * zn * e_i;
        const double hxz = d1x * ym  * d1z * e_i;
        const double hyz = xl  * d1y * d1z * e_i;
        for(size_t ic=0; ic<nctr; ic++) {
          const double w = cf_(iexp, ic);
          hbuf(icart, 9*ic+0) += w * hxx;
          hbuf(icart, 9*ic+4) += w * hyy;
          hbuf(icart, 9*ic+8) += w * hzz;
          hbuf(icart, 9*ic+1) += w * hxy;
          hbuf(icart, 9*ic+3) += w * hxy;
          hbuf(icart, 9*ic+2) += w * hxz;
          hbuf(icart, 9*ic+6) += w * hxz;
          hbuf(icart, 9*ic+5) += w * hyz;
          hbuf(icart, 9*ic+7) += w * hyz;
        }
      }

      if(do_lgrad) {
        const double lg0 = (d3x * ym * zn + d1x * d2y * zn + d1x * ym * d2z) * e_i;
        const double lg1 = (d2x * d1y * zn + xl * d3y * zn + xl * d1y * d2z) * e_i;
        const double lg2 = (d2x * ym * d1z + xl * d2y * d1z + xl * ym * d3z) * e_i;
        for(size_t ic=0; ic<nctr; ic++) {
          const double w = cf_(iexp, ic);
          lgbuf(icart, 3*ic+0) += w * lg0;
          lgbuf(icart, 3*ic+1) += w * lg1;
          lgbuf(icart, 3*ic+2) += w * lg2;
        }
      }
    }
  }

  // Plug in the per-cartesian normalisation constant (shared across contractions)
  for(size_t icart=0; icart<Ncart; icart++) {
    const double rn = cart_[icart].relnorm;
    fbuf.row(icart) *= rn;
    if(do_grad)  gbuf.row(icart)  *= rn;
    if(do_lapl)  lbuf.row(icart)  *= rn;
    if(do_hess)  hbuf.row(icart)  *= rn;
    if(do_lgrad) lgbuf.row(icart) *= rn;
  }

  // Project each contraction's cartesian block to the output, stacking
  // the contractions along the function (row) dimension: contraction
  // ictr occupies output rows [ictr*Nout, (ictr+1)*Nout).
  const size_t Nout = uselm_ ? Nlm() : Ncart;
  fval.set_size(nctr*Nout);
  if(do_grad)  gval.set_size(nctr*Nout, 3);
  if(do_lapl)  lval.set_size(nctr*Nout);
  if(do_hess)  hval.set_size(nctr*Nout, 9);
  if(do_lgrad) lgval.set_size(nctr*Nout, 3);

  for(size_t ic=0; ic<nctr; ic++) {
    const size_t r0=ic*Nout, r1=r0+Nout-1;
    if(uselm_) {
      fval.subvec(r0,r1) = transmat_ * fbuf.col(ic);
      if(do_grad)  gval.rows(r0,r1)  = transmat_ * gbuf.cols(3*ic,3*ic+2);
      if(do_lapl)  lval.subvec(r0,r1) = transmat_ * lbuf.col(ic);
      if(do_hess)  hval.rows(r0,r1)  = transmat_ * hbuf.cols(9*ic,9*ic+8);
      if(do_lgrad) lgval.rows(r0,r1) = transmat_ * lgbuf.cols(3*ic,3*ic+2);
    } else {
      fval.subvec(r0,r1) = fbuf.col(ic);
      if(do_grad)  gval.rows(r0,r1)  = gbuf.cols(3*ic,3*ic+2);
      if(do_lapl)  lval.subvec(r0,r1) = lbuf.col(ic);
      if(do_hess)  hval.rows(r0,r1)  = hbuf.cols(9*ic,9*ic+8);
      if(do_lgrad) lgval.rows(r0,r1) = lgbuf.cols(3*ic,3*ic+2);
    }
  }
}

arma::mat GaussianShell::eval_hess(double x, double y, double z) const {
  arma::vec fval, lval;
  arma::mat gval, hval, lgval;
  eval_bf_derivs(x, y, z, fval, gval, lval, hval, lgval,
                 false, false, true, false);
  return hval;
}

arma::mat GaussianShell::eval_laplgrad(double x, double y, double z) const {
  arma::vec fval, lval;
  arma::mat gval, hval, lgval;
  eval_bf_derivs(x, y, z, fval, gval, lval, hval, lgval,
                 false, false, false, true);
  return lgval;
}

// Calculate overlaps between basis functions
namespace {
  /// Force-convention derivative of the nuclear attraction integrals
  /// with respect to the centers of the two shells: components 0-2
  /// differentiate the bra, 3-5 the ket. The operator is -1/|r-C|, and
  /// the force convention is minus the geometric derivative, so the
  /// derivative of the bra is +iprinv and that of the ket is its
  /// transpose with the shells swapped.
  arma::vec nuclear_pulay_pair(Int1eWorker & w, size_t i, size_t j, const arma::mat & P, const double * orig) {
    std::vector<arma::mat> der(6);
    w.compute(CINT1E_IPRINV,i,j,orig);
    for(int ic=0;ic<3;ic++)
      der[ic]=-w.get_mat(ic,i,j);
    w.compute(CINT1E_IPRINV,j,i,orig);
    for(int ic=0;ic<3;ic++)
      der[ic+3]=-arma::trans(w.get_mat(ic,j,i));

    arma::vec ret(6);
    for(size_t ic=0;ic<6;ic++)
      ret(ic)=arma::trace(arma::trans(P)*der[ic]);
    return ret;
  }

  /// Hellmann-Feynman derivative of the nuclear attraction integrals
  /// with respect to the center of the operator, in the force
  /// convention. By translational invariance it is the sum of the two
  /// basis function center derivatives.
  arma::vec nuclear_der_pair(Int1eWorker & w, size_t i, size_t j, const arma::mat & P, const double * orig) {
    std::vector<arma::mat> der(3);
    w.compute(CINT1E_IPRINV,i,j,orig);
    for(int ic=0;ic<3;ic++)
      der[ic]=w.get_mat(ic,i,j);
    w.compute(CINT1E_IPRINV,j,i,orig);
    for(int ic=0;ic<3;ic++)
      der[ic]+=arma::trans(w.get_mat(ic,j,i));

    arma::vec ret(3);
    for(size_t ic=0;ic<3;ic++)
      ret(ic)=arma::trace(arma::trans(P)*der[ic]);
    return ret;
  }

  /// Derivative of a two-index operator with respect to the centers of
  /// the two shells, contracted with a matrix: components 0-2
  /// differentiate the bra, 3-5 the ket
  arma::vec pulay_pair(Int1eWorker & w, cint_1e_kernel_t bra, cint_1e_kernel_t ket,
                       size_t i, size_t j, const arma::mat & P, double sign) {
    arma::vec ret(6);
    w.compute(bra,i,j);
    for(int ic=0;ic<3;ic++)
      ret(ic)=sign*arma::trace(arma::trans(P)*w.get_mat(ic,i,j));
    w.compute(ket,i,j);
    for(int ic=0;ic<3;ic++)
      ret(ic+3)=sign*arma::trace(arma::trans(P)*w.get_mat(ic,i,j));
    return ret;
  }

  /// Moment integrals of order mom around the given origin, in ERKALE's
  /// order: the unique components with the powers of x decreasing. The
  /// libcint operators are the full (symmetric) tensors, so any index
  /// string with the right powers gives the component; the canonical
  /// sorted one is used.
  std::vector<arma::mat> moment_pair(Int1eWorker & w, int mom, size_t i, size_t j, const double * orig) {
    static const cint_1e_kernel_t rint[5]={CINT1E_OVLP, CINT1E_R, CINT1E_RR, CINT1E_RRR, CINT1E_RRRR};
    if(mom<0 || mom>4)
      throw std::runtime_error("Moment integrals are only available up to fourth order.\n");

    w.compute(rint[mom],i,j,NULL,orig);

    std::vector<arma::mat> ret;
    ret.reserve((mom+1)*(mom+2)/2);
    for(int ii=0; ii<=mom; ii++) {
      int nx=mom - ii;
      for(int jj=0; jj<=ii; jj++) {
        int ny=ii - jj;
        int nz=jj;

        int idx=0;
        for(int k=0;k<nx;k++) idx=idx*3+0;
        for(int k=0;k<ny;k++) idx=idx*3+1;
        for(int k=0;k<nz;k++) idx=idx*3+2;

        ret.push_back(w.get_mat(idx,i,j));
      }
    }
    return ret;
  }
}

arma::vec GaussianShell::function_norms() const {
  // Norms of every function of the shell, one contraction after the
  // other. Both functions of an overlap sit on the same center, so the
  // integral factorizes over the cartesian directions and vanishes
  // unless every direction has an even total power. Precompute the
  // per-primitive-pair, per-cartesian-pair angular factor once and
  // weight it by each contraction's coefficients.
  const size_t Nout = uselm_ ? Nlm() : cart_.size();
  arma::vec norms(Nctr()*Nout);

  for(size_t ictr=0;ictr<Nctr();ictr++) {
    arma::mat S(cart_.size(),cart_.size(),arma::fill::zeros);
    for(size_t ic=0;ic<cart_.size();ic++)
      for(size_t jc=0;jc<cart_.size();jc++) {
        const int L[3]={cart_[ic].l+cart_[jc].l, cart_[ic].m+cart_[jc].m, cart_[ic].n+cart_[jc].n};
        if(L[0]%2 || L[1]%2 || L[2]%2)
          continue;

        double val=0.0;
        for(size_t ip=0;ip<c_.size();ip++)
          for(size_t jp=0;jp<c_.size();jp++) {
            const double zeta=c_[ip].z+c_[jp].z;
            double term=cf_(ip,ictr)*cf_(jp,ictr)*pow(M_PI/zeta,1.5);
            for(int ix=0;ix<3;ix++)
              term*=doublefact(L[ix]-1)/pow(2.0*zeta,L[ix]/2);
            val+=term;
          }
        S(ic,jc)=cart_[ic].relnorm*cart_[jc].relnorm*val;
      }

    if(uselm_)
      S=transmat_*S*arma::trans(transmat_);

    norms.subvec(ictr*Nout, ictr*Nout+Nout-1)=S.diag();
  }

  return norms;
}

// Calculate overlaps between basis functions
arma::mat GaussianShell::coulomb_overlap(const GaussianShell & rhs) const {
  // Number of functions on the shells
  size_t Ni=Nbf();
  size_t Nj=rhs.Nbf();

  // Two-center Coulomb integrals over the shell pair
  std::vector<GaussianShell> shpair;
  shpair.push_back(*this);
  shpair.push_back(rhs);
  CintEnv cenv(shpair,false);
  ERIWorker eri(cenv);
  eri.compute_2c(0,1);
  const std::vector<double> * erip=eri.getp();

  // Fill overlap matrix
  arma::mat S(Ni,Nj);
  for(size_t i=0;i<Ni;i++)
    for(size_t j=0;j<Nj;j++)
      S(i,j)=(*erip)[i*Nj+j];

  return S;
}










arma::vec GaussianShell::integral() const {
  // Integral over each function of the shell, one contraction after the
  // other.
  const size_t Nout = uselm_ ? Nlm() : cart_.size();
  arma::vec out(Nctr()*Nout);

  for(size_t ictr=0;ictr<Nctr();ictr++) {
    arma::vec ints(cart_.size());
    ints.zeros();

    for(size_t ic=0;ic<cart_.size();ic++) {
      int l=cart_[ic].l;
      int m=cart_[ic].m;
      int n=cart_[ic].n;

      if(l%2 || m%2 || n%2)
        // Odd function - zero integral
        continue;

      for(size_t ix=0;ix<c_.size();ix++) {
        double zeta=c_[ix].z;
        double intx=2.0*pow(0.5/sqrt(zeta),l+1)*sqrt(M_PI);
        double inty=2.0*pow(0.5/sqrt(zeta),m+1)*sqrt(M_PI);
        double intz=2.0*pow(0.5/sqrt(zeta),n+1)*sqrt(M_PI);
        ints(ic)+=cf_(ix,ictr)*intx*inty*intz;
      }

      ints(ic)*=cart_[ic].relnorm;
    }

    if(uselm_)
      ints=transmat_*ints;

    out.subvec(ictr*Nout, ictr*Nout+Nout-1)=ints;
  }

  return out;
}


BasisSet::BasisSet() {
  // Use spherical harmonics and cartesian functions by default.
  uselm_=true;
  optlm_=true;
}

extern Settings settings;

BasisSet::BasisSet(size_t Nat) {
  // Use spherical harmonics?
  uselm_=settings.get_bool("UseLM");
  optlm_=settings.get_bool("OptLM");

  shells_.reserve(Nat);
  nuclei_.reserve(Nat);
}

BasisSet::~BasisSet() {
}

void BasisSet::add_nucleus(const nucleus_t & nuc) {
  nuclei_.push_back(nuc);
  // Clear list of functions
  nuclei_[nuclei_.size()-1].shells.clear();
  // Set nuclear index
  nuclei_[nuclei_.size()-1].ind=nuclei_.size()-1;
}

void BasisSet::add_shell(size_t nucind, const GaussianShell & sh, bool dosort) {
  if(nucind>=nuclei_.size()) {
    ERROR_INFO();
    throw std::runtime_error("Cannot add functions to nonexisting nucleus!\n");
  }

  // Add shell
  shells_.push_back(sh);
  // Set pointer to nucleus
  shells_[shells_.size()-1].set_center(nuclei_[nucind].r,nucind);

  // Sort the basis set, updating the nuclear list and basis function indices as well
  if(dosort)
    sort();
  else {
    // Just do the numbering and shell list updates.
    check_numbering();
    update_nuclear_shell_list();
  }
}

void BasisSet::add_shell(size_t nucind, int am, bool lm, const std::vector<contr_t> & C, bool dosort) {
  // Create new shell.
  GaussianShell sh=GaussianShell(am,lm,C);
  // Do the rest here
  add_shell(nucind,sh,dosort);
}

void BasisSet::add_shells(size_t nucind, ElementBasisSet el, bool dosort) {
  // Add basis functions at cen

  // Get the shells on the element
  std::vector<FunctionShell> bf=el.get_shells();

  // Loop over shells in element basis
  for(size_t i=0;i<bf.size();i++) {
    // Spherical harmonics for this shell?
    const bool lm = (!optlm_ || bf[i].get_am()>=2) ? uselm_ : false;

    // A library shell may be generally contracted: add one GaussianShell
    // per contraction. They share the exponents, so finalize's
    // merge_generally_contracted regroups them into a single native
    // generally contracted shell (segmented shells with nctr==1 add just
    // one, as before).
    for(size_t ic=0;ic<bf[i].get_Nctr();ic++)
      add_shell(nucind,GaussianShell(bf[i].get_am(),lm,bf[i].get_contr(ic)),dosort);
  }
}

void BasisSet::check_numbering() {
  // Renumber basis functions
  size_t ind=0;
  for(size_t i=0;i<shells_.size();i++) {
    shells_[i].first_ind(ind);
    ind=shells_[i].last_ind()+1;
  }
}

void BasisSet::update_nuclear_shell_list() {
  // First, clear the list on all nuclei.
  for(size_t inuc=0;inuc<nuclei_.size();inuc++)
    nuclei_[inuc].shells.clear();

  // Then, update the lists. Loop over shells
  for(size_t ish=0;ish<shells_.size();ish++) {
    // Find out nuclear index
    size_t inuc=shells_[ish].center_ind();
    // Add pointer to the nucleus
    nuclei_[inuc].shells.push_back(&shells_[ish]);
  }
}

void BasisSet::sort() {
  // Sort the shells first by increasing index of center, then by
  // increasing angular momentum and last by decreasing exponent.
  std::stable_sort(shells_.begin(),shells_.end());

  // Check the numbering of the basis functions
  check_numbering();

  // and since we probably have changed the order of the basis
  // functions, we need to update the list of functions on the nuclei.
  update_nuclear_shell_list();
}

void BasisSet::merge_generally_contracted() {
  // Group consecutive shells that share the same center, angular
  // momentum, harmonics flag and primitive exponents into one generally
  // contracted shell. The shells arrive already in their final order, in
  // which the segmented sort has made every generally contracted block
  // contiguous, so a single consecutive-merge pass captures all general
  // contraction without disturbing the basis function order (the merged
  // shell emits [ctr0][ctr1]... in the same span the segmented shells
  // occupied).
  if(shells_.empty())
    return;

  std::vector<GaussianShell> merged;
  merged.reserve(shells_.size());
  merged.push_back(shells_[0]);
  for(size_t i=1;i<shells_.size();i++) {
    if(merged.back().same_primitives(shells_[i]))
      merged.back().merge_contraction(shells_[i]);
    else
      merged.push_back(shells_[i]);
  }
  shells_=std::move(merged);

  // Renumber the basis functions and rebuild the per-nucleus shell lists.
  check_numbering();
  update_nuclear_shell_list();
}

void BasisSet::compute_nuclear_distances() {
  // Amount of nuclei
  size_t N=nuclei_.size();

  // Reserve memory
  nucleardist_=arma::mat(N,N);

  double d;

  // Fill table
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<=i;j++) {
      d=dist(nuclei_[i].r.x,nuclei_[i].r.y,nuclei_[i].r.z,nuclei_[j].r.x,nuclei_[j].r.y,nuclei_[j].r.z);

      nucleardist_(i,j)=d;
      nucleardist_(j,i)=d;
    }
}

double BasisSet::nuclear_distance(size_t i, size_t j) const {
  return nucleardist_(i,j);
}

arma::mat BasisSet::nuclear_distances() const {
  return nucleardist_;
}

bool operator<(const shellpair_t & lhs, const shellpair_t & rhs) {
  // Helper for ordering shellpairs by angular momentum
  return (lhs.li+lhs.lj)<(rhs.li+rhs.lj);
}

void BasisSet::form_unique_shellpairs() {
  // Drop list of existing pairs.
  shellpairs_.clear();

  // Form list of unique shell pairs.
  shellpair_t tmp;

  // Now, form list of unique shell pairs
  for(size_t i=0;i<shells_.size();i++) {
    for(size_t j=0;j<=i;j++) {
      // Have to set these in every iteration due to swap below
      tmp.is=i;
      tmp.js=j;

      // Order the pair with the higher angular momentum first
      if(shells_[j].am()>shells_[i].am())
	std::swap(tmp.is,tmp.js);

      // Set angular momenta
      tmp.li=shells_[tmp.is].am();
      tmp.lj=shells_[tmp.js].am();

      shellpairs_.push_back(tmp);
    }
  }

  // Sort list of unique shell pairs
  stable_sort(shellpairs_.begin(),shellpairs_.end());

  /*
  // Print list
  printf("\nList of unique shell pairs (%lu pairs):\n",shellpairs.size());
  for(size_t ind=0;ind<shellpairs.size();ind++) {
  size_t i=shellpairs[ind].is;
  size_t j=shellpairs[ind].js;

  int li=shells[i].get_am();
  int lj=shells[j].get_am();

  printf("%i\t%i\t%i\t%i\t%i\n",(int) i,(int) j,li,lj,li+lj);
  }
  */
}

std::vector<shellpair_t> BasisSet::unique_shellpairs() const {
  if(shells_.size() && !shellpairs_.size()) {
    throw std::runtime_error("shellpairs not initialized! Maybe you forgot to finalize?\n");
  }

  return shellpairs_;
}

ScreeningData BasisSet::compute_screening(double tol, double omega, double alpha, double beta, bool verbose) const {
  ScreeningData s;

  // Get the screening matrices
  eri_screening(s.Q,s.M,omega,alpha,beta);

  // Fill out list
  s.shpairs.resize(shellpairs_.size());
  for(size_t i=0;i<shellpairs_.size();i++) {
    s.shpairs[i].is=shellpairs_[i].is;
    s.shpairs[i].i0=shells_[shellpairs_[i].is].first_ind();
    s.shpairs[i].Ni=shells_[shellpairs_[i].is].Nbf();

    s.shpairs[i].js=shellpairs_[i].js;
    s.shpairs[i].j0=shells_[shellpairs_[i].js].first_ind();
    s.shpairs[i].Nj=shells_[shellpairs_[i].js].Nbf();

    s.shpairs[i].eri=s.Q(s.shpairs[i].is,s.shpairs[i].js);
  }
  // and sort it
  std::stable_sort(s.shpairs.begin(),s.shpairs.end());

  // Find out which pairs are negligible.
  size_t ulimit=s.shpairs.size()-1;
  // An integral (ij|kl) <= sqrt( (ij|ij) (kl|kl) ). Since the
  // integrals are in decreasing order, the integral threshold is
  double thr=tol/s.shpairs[0].eri;
  while(s.shpairs[ulimit].eri < thr)
    ulimit--;
  if(ulimit<s.shpairs.size())
    s.shpairs.resize(ulimit+1);

  (void) verbose;
  //if(verbose)
  // printf("%u shell pairs out of %u are significant.\n",(unsigned int) s.shpairs.size(),(unsigned int) shellpairs.size());

  return s;
}

bool operator<(const eripair_t & lhs, const eripair_t & rhs) {
  // Sort in decreasing order!
  return lhs.eri > rhs.eri;
}


void BasisSet::finalize(bool convert, bool donorm) {
  // Finalize basis set structure for use.

  // Group same-primitive shells into generally contracted shells before
  // anything downstream (ranges, contraction conversion, normalization,
  // shell pairs) is computed, so all of it sees the native generally
  // contracted shells. A segmented basis (no two shells share primitives)
  // is left untouched.
  merge_generally_contracted();

  // Compute nuclear distances.
  compute_nuclear_distances();

  // Compute ranges of shells
  compute_shell_ranges();

  // Convert contractions
  if(convert)
    convert_contractions();
  // Normalize contractions if requested, and compute cartesian norms
  normalize(donorm);

  // Form list of unique shell pairs
  form_unique_shellpairs();
  // and update the nuclear shell list (in case the basis set was
  // loaded from checkpoint)
  update_nuclear_shell_list();
}

int BasisSet::am(size_t ind) const {
  return shells_[ind].am();
}

int BasisSet::max_am() const {
  if(shells_.size()==0) {
    return -1;
  }

  int maxam=shells_[0].am();
  for(size_t i=1;i<shells_.size();i++)
    if(shells_[i].am()>maxam)
      maxam=shells_[i].am();
  return maxam;
}

size_t BasisSet::max_Ncontr() const {
  size_t maxc=shells_[0].Ncontr();
  for(size_t i=1;i<shells_.size();i++)
    if(shells_[i].Ncontr()>maxc)
      maxc=shells_[i].Ncontr();
  return maxc;
}

size_t BasisSet::Nbf() const {
  if(shells_.size())
    return shells_[shells_.size()-1].last_ind()+1;
  else
    return 0;
}

void BasisSet::compute_shell_ranges() {
  if(settings.is_double("DFTBasisThr"))
    compute_shell_ranges(settings.get_double("DFTBasisThr"));
}

void BasisSet::compute_shell_ranges(double eps) {
  shell_ranges_=shell_ranges(eps);
}

std::vector<double> BasisSet::shell_ranges() const {
  return shell_ranges_;
}

std::vector<double> BasisSet::shell_ranges(double eps) const {
  std::vector<double> shran(shells_.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<shells_.size();i++)
    shran[i]=shells_[i].range(eps);

  return shran;
}

std::vector<double> BasisSet::nuclear_distances(size_t inuc) const {
  std::vector<double> d(nucleardist_.n_cols);
  for(size_t i=0;i<nucleardist_.n_cols;i++)
    d[i]=nucleardist_(inuc,i);
  return d;
}

size_t BasisSet::Ncart() const {
  size_t n=0;
  for(size_t i=0;i<shells_.size();i++)
    n+=shells_[i].Ncart();
  return n;
}

size_t BasisSet::Nlm() const {
  size_t n=0;
  for(size_t i=0;i<shells_.size();i++)
    n+=shells_[i].Nlm();
  return n;
}

size_t BasisSet::Nbf(size_t ind) const {
  return shells_[ind].Nbf();
}

size_t BasisSet::Ncart(size_t ind) const {
  return shells_[ind].Ncart();
}

size_t BasisSet::last_ind() const {
  if(shells_.size())
    return shells_[shells_.size()-1].last_ind();
  else {
    std::ostringstream oss;
    oss << "\nError in function " << __FUNCTION__ << "(file " << __FILE__ << ", near line " << __LINE__ << "\nCannot get number of last basis function of an empty basis set!\n";
    throw std::domain_error(oss.str());
  }
}

size_t BasisSet::first_ind(size_t num) const {
  return shells_[num].first_ind();
}

size_t BasisSet::last_ind(size_t num) const {
  return shells_[num].last_ind();
}

arma::vec BasisSet::bf_Rsquared() const {
  arma::vec Rsq(Nbf());

  CintEnv cenv(*this,false);
  Int1eWorker w(cenv);

  for(size_t i=0;i<shells_.size();i++) {
    // First function on shell
    size_t i0=shells_[i].first_ind();
    // Number of functions
    size_t nbf=shells_[i].Nbf();

    // Calculate second moments around the center of the shell
    coords_t cen=shells_[i].center();
    const double orig[3]={cen.x, cen.y, cen.z};
    std::vector<arma::mat> mom2=moment_pair(w,2,i,i,orig);
    // Compute spatial extents
    for(size_t fi=0;fi<nbf;fi++)
      Rsq(i0+fi)=mom2[getind(2,0,0)](fi,fi)+mom2[getind(0,2,0)](fi,fi)+mom2[getind(0,0,2)](fi,fi);
  }

  return Rsq;
}

arma::uvec BasisSet::shell_indices() const {
  arma::uvec idx(Nbf());
  for(size_t i=0;i<shells_.size();i++)
    idx.subvec(shells_[i].first_ind(),shells_[i].last_ind())=i*arma::ones<arma::uvec>(shells_[i].Nbf());
  return idx;
}

size_t BasisSet::find_shell_ind(size_t find) const {
  // Find shell the function belongs to
  for(size_t i=0;i<shells_.size();i++)
    if(find>=shells_[i].first_ind() && find<=shells_[i].last_ind())
      return i;

  std::ostringstream oss;
  oss << "Basis function " << find << " not found in basis set!\n";
  throw std::runtime_error(oss.str());
}

size_t BasisSet::shell_center_ind(size_t num) const {
  return shells_[num].center_ind();
}

std::vector<GaussianShell> BasisSet::shells() const {
  return shells_;
}

const std::vector<GaussianShell> & BasisSet::shells_ref() const {
  return shells_;
}

GaussianShell BasisSet::shell(size_t ind) const {
  return shells_[ind];
}

coords_t BasisSet::shell_center(size_t num) const {
  return shells_[num].center();
}

std::vector<contr_t> BasisSet::contr(size_t ind) const {
  return shells_[ind].contr();
}

std::vector<contr_t> BasisSet::contr_normalized(size_t ind) const {
  return shells_[ind].contr_normalized();
}

std::vector<shellf_t> BasisSet::cart(size_t ind) const {
  return shells_[ind].cart();
}


bool BasisSet::is_lm_default() const {
  return uselm_;
}

bool BasisSet::lm_in_use(size_t num) const {
  return shells_[num].lm_in_use();
}

void BasisSet::set_lm(size_t num, bool lm) {
  // Set use of spherical harmonics
  shells_[num].set_lm(lm);
  // Check numbering of basis functions which may have changed
  check_numbering();
}

arma::ivec BasisSet::m_values() const {
  arma::ivec ret(Nbf());
  for(size_t is=0;is<Nshells();is++) {
    // Angular momentum is
    int am(this->am(is));

    // First function on shell
    size_t i0(first_ind(is));

    // Functions are -m, -m+1, ..., m-1, m
    if(lm_in_use(is)) {
      ret.subvec(i0,i0+2*am)=arma::linspace<arma::ivec>(-am,am,2*am+1);
    } else {
      if(am==0)
        ret(i0)=0;
      else if(am==1) {
        // Functions are in order x, y, z i.e. 1, -1, 0
        ret(i0)=1;
        ret(i0+1)=-1;
        ret(i0+2)=0;
      } else
        throw std::logic_error("Need to use spherical basis for linear symmetry!\n");
    }
  }

  return ret;
}

arma::ivec BasisSet::unique_m_values() const {
  // Find unique m values
  arma::ivec mval(m_values());
  arma::sword mmin=0;
  arma::sword mmax=arma::max(arma::abs(mval));

  std::vector<arma::sword> mvals;
  mvals.push_back(0);
  for(int am=1;am<=mmax;am++) {
    mvals.push_back(-am);
    mvals.push_back(am);
  }

  return arma::conv_to<arma::ivec>::from(mvals);
}

std::map<int, arma::uword> BasisSet::unique_m_map() const {
  arma::ivec muni(unique_m_values());
  std::map<int, arma::uword> mlook;
  for(arma::uword i=0;i<muni.size();i++)
    mlook[muni(i)]=i;
  return mlook;
}

arma::imat BasisSet::count_m_occupied(const arma::mat & C) const {
  arma::ivec mc(m_classify(C,m_values()));

  std::map<int, arma::uword> mlook(unique_m_map());

  arma::imat occ;
  occ.zeros(mlook.size(),3);
  for(size_t i=0;i<C.n_cols;i++)
    occ(mlook[mc(i)],0)++;
  occ.col(1)=occ.col(0);
  occ.col(2)=unique_m_values();
  return occ;
}

arma::imat BasisSet::count_m_occupied(const arma::mat & Ca, const arma::mat & Cb) const {
  arma::ivec mca(m_classify(Ca,m_values()));
  arma::ivec mcb(m_classify(Cb,m_values()));

  std::map<int, arma::uword> mlook(unique_m_map());

  arma::imat occ;
  occ.zeros(mlook.size(),3);
  for(size_t i=0;i<Ca.n_cols;i++)
    occ(mlook[mca(i)],0)++;
  for(size_t i=0;i<Cb.n_cols;i++)
    occ(mlook[mcb(i)],1)++;
  occ.col(2)=unique_m_values();
  return occ;
}

arma::uvec BasisSet::m_indices(int mwant) const {
  arma::ivec midx(m_values());
  return arma::find(midx==mwant);
}

arma::mat BasisSet::transmat(size_t ind) const {
  return shells_[ind].transmat();
}

size_t BasisSet::Nshells() const {
  return shells_.size();
}

size_t BasisSet::Nnuc() const {
  return nuclei_.size();
}

nucleus_t BasisSet::nucleus(size_t inuc) const {
  return nuclei_[inuc];
}

std::vector<nucleus_t> BasisSet::nuclei() const {
  return nuclei_;
}

arma::mat BasisSet::nuclear_coords() const {
  arma::mat coords(nuclei_.size(),3);
  for(size_t i=0;i<nuclei_.size();i++)
    coords.row(i)=arma::trans(coords_to_vec(nuclei_[i].r));

  return coords;
}

void BasisSet::set_nuclear_coords(const arma::mat & c) {
  if(c.n_rows != nuclei_.size() || c.n_cols != 3)
    throw std::logic_error("Coordinates matrix does not match nuclei!\n");

  for(size_t i=0;i<nuclei_.size();i++)
    nuclei_[i].r=vec_to_coords(arma::trans(c.row(i)));

  // Update shell centers
  for(size_t i=0;i<shells_.size();i++) {
    size_t icen=shells_[i].center_ind();
    shells_[i].set_center(nuclei_[icen].r,icen);
  }
  // Update listings
  finalize(false,false);
}

coords_t BasisSet::nuclear_coords(size_t inuc) const {
  return nuclei_[inuc].r;
}

int BasisSet::Z(size_t inuc) const {
  return nuclei_[inuc].Z;
}

std::string BasisSet::symbol(size_t inuc) const {
  return nuclei_[inuc].symbol;
}

std::string BasisSet::symbol_hr(size_t inuc) const {
  if(nuclei_[inuc].bsse)
    return nuclei_[inuc].symbol+"-Bq";
  else
    return nuclei_[inuc].symbol;
}

std::vector<GaussianShell> BasisSet::funcs(size_t inuc) const {
  std::vector<GaussianShell> ret;
  for(size_t i=0;i<nuclei_[inuc].shells.size();i++)
    ret.push_back(*(nuclei_[inuc].shells[i]));

  return ret;
}

std::vector<size_t> BasisSet::shell_inds(size_t inuc) const {
  std::vector<size_t> ret;
  for(size_t i=0;i<shells_.size();i++)
    if(shells_[i].center_ind()==inuc)
      ret.push_back(i);
  return ret;
}

arma::vec BasisSet::eval_func(double x, double y, double z) const {
  // Helper
  coords_t r;
  r.x=x;
  r.y=y;
  r.z=z;

  // Determine which shells might contribute
  std::vector<size_t> compute_shells;
  for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
    // Determine distance to nucleus
    double dist=norm(r-nuclei_[inuc].r);
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++)
      // Shell is relevant if range is larger than distance
      if(dist < shell_ranges_[shellinds[ish]])
	compute_shells.push_back(shellinds[ish]);
  }

  // Returned values
  arma::vec ret(Nbf());
  ret.zeros();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<compute_shells.size();i++) {
    size_t ish=compute_shells[i];

    // Evalute shell. Function values
    arma::vec shf=shells_[ish].eval_func(x,y,z);
    // First function on shell
    size_t f0=shells_[ish].first_ind();

    // and store the functions
    for(size_t fi=0;fi<shells_[ish].Nbf();fi++) {
      ret(f0+fi)=shf(fi);
    }
  }

  return ret;
}

arma::mat BasisSet::eval_grad(double x, double y, double z) const {
  // Helper
  coords_t r;
  r.x=x;
  r.y=y;
  r.z=z;

  // Determine which shells might contribute
  std::vector<size_t> compute_shells;
  for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
    // Determine distance to nucleus
    double dist=norm(r-nuclei_[inuc].r);
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++)
      // Shell is relevant if range is larger than distance
      if(dist < shell_ranges_[shellinds[ish]])
	compute_shells.push_back(shellinds[ish]);
  }

  // Returned values
  arma::mat ret(Nbf(),3);
  ret.zeros();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<compute_shells.size();i++) {
    size_t ish=compute_shells[i];

    // Evalute shell. Gradient values
    arma::mat gf=shells_[ish].eval_grad(x,y,z);
    // First function on shell
    size_t f0=shells_[ish].first_ind();

    // and store the functions
    for(size_t fi=0;fi<shells_[ish].Nbf();fi++) {
      ret.row(f0+fi)=gf.row(fi);
    }
  }

  return ret;
}

arma::mat BasisSet::eval_hess(double x, double y, double z) const {
  // Helper
  coords_t r;
  r.x=x;
  r.y=y;
  r.z=z;

  // Determine which shells might contribute
  std::vector<size_t> compute_shells;
  for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
    // Determine distance to nucleus
    double dist=norm(r-nuclei_[inuc].r);
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++)
      // Shell is relevant if range is larger than distance
      if(dist < shell_ranges_[shellinds[ish]])
	compute_shells.push_back(shellinds[ish]);
  }

  // Returned values
  arma::mat ret(Nbf(),9);
  ret.zeros();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<compute_shells.size();i++) {
    size_t ish=compute_shells[i];

    // Evalute shell. Gradient values
    arma::mat gf=shells_[ish].eval_hess(x,y,z);
    // First function on shell
    size_t f0=shells_[ish].first_ind();

    // and store the functions
    for(size_t fi=0;fi<shells_[ish].Nbf();fi++) {
      ret.row(f0+fi)=gf.row(fi);
    }
  }

  return ret;
}

arma::vec BasisSet::eval_func(size_t ish, double x, double y, double z) const {
  return shells_[ish].eval_func(x,y,z);
}

arma::mat BasisSet::eval_grad(size_t ish, double x, double y, double z) const {
  return shells_[ish].eval_grad(x,y,z);
}

arma::vec BasisSet::eval_lapl(size_t ish, double x, double y, double z) const {
  return shells_[ish].eval_lapl(x,y,z);
}

arma::mat BasisSet::eval_hess(size_t ish, double x, double y, double z) const {
  return shells_[ish].eval_hess(x,y,z);
}

arma::mat BasisSet::eval_laplgrad(size_t ish, double x, double y, double z) const {
  return shells_[ish].eval_laplgrad(x,y,z);
}

void BasisSet::eval_bf_derivs(size_t ish, double x, double y, double z,
                              arma::vec & fval,
                              arma::mat & gval,
                              arma::vec & lval,
                              arma::mat & hval,
                              arma::mat & lgval,
                              bool do_grad, bool do_lapl,
                              bool do_hess, bool do_lgrad) const {
  shells_[ish].eval_bf_derivs(x, y, z, fval, gval, lval, hval, lgval,
                             do_grad, do_lapl, do_hess, do_lgrad);
}

void BasisSet::convert_contractions() {
  for(size_t i=0;i<shells_.size();i++)
    shells_[i].convert_contraction();
}

void BasisSet::convert_contraction(size_t ind) {
  shells_[ind].convert_contraction();
}

void BasisSet::normalize(bool coeffs) {
  for(size_t i=0;i<shells_.size();i++)
    shells_[i].normalize(coeffs);
}

void BasisSet::coulomb_normalize() {
  for(size_t i=0;i<shells_.size();i++)
    shells_[i].coulomb_normalize();
}

void BasisSet::print(bool verbose) const {
  printf("There are %i shells and %i nuclei in the basis set.\n\n",(int) shells_.size(),(int) nuclei_.size());

  printf("List of nuclei, geometry in Ångström with three decimal places:\n");

  printf("\t\t Z\t    x\t    y\t    z\n");
  for(size_t i=0;i<nuclei_.size();i++) {
    if(nuclei_[i].bsse)
      printf("%i\t%s\t*%i\t% 7.3f\t% 7.3f\t% 7.3f\n",(int) i+1,nuclei_[i].symbol.c_str(),nuclei_[i].Z,nuclei_[i].r.x/ANGSTROMINBOHR,nuclei_[i].r.y/ANGSTROMINBOHR,nuclei_[i].r.z/ANGSTROMINBOHR);
    else
      printf("%i\t%s\t %i\t% 7.3f\t% 7.3f\t% 7.3f\n",(int) i+1,nuclei_[i].symbol.c_str(),nuclei_[i].Z,nuclei_[i].r.x/ANGSTROMINBOHR,nuclei_[i].r.y/ANGSTROMINBOHR,nuclei_[i].r.z/ANGSTROMINBOHR);
  }

  if(nuclei_.size()>1 && nuclei_.size()<=13) {
    // Legend length is 7 + 6*(N-1) chars

    // Print legend
    printf("\nInteratomic distance matrix:\n%7s","");
    for(size_t i=0;i<nuclei_.size()-1;i++)
      printf(" %3i%-2s",(int) i+1,nuclei_[i].symbol.c_str());
    printf("\n");

    // Print atomic entries
    for(size_t i=1;i<nuclei_.size();i++) {
      printf(" %3i%-2s",(int) i+1,nuclei_[i].symbol.c_str());
      for(size_t j=0;j<i;j++)
	printf(" %5.3f",norm(nuclei_[i].r-nuclei_[j].r)/ANGSTROMINBOHR);
      printf("\n");
    }
  }

  printf("\nList of basis functions:\n");

  if(verbose) {
    for(size_t i=0;i<shells_.size();i++) {
      printf("Shell %4i",(int) i);
      shells_[i].print();
    }
  } else {
    for(size_t i=0;i<shells_.size();i++) {
      // Type of shell - spherical harmonics or cartesians
      std::string type;
      if(shells_[i].lm_in_use())
	type="sph";
      else
	type="cart";


      printf("Shell %4i",(int) i+1);
      printf("\t%c %4s shell at nucleus %3i with with basis functions %4i-%-4i\n",shell_types[shells_[i].am()],type.c_str(),(int) (shells_[i].center_ind()+1),(int) shells_[i].first_ind()+1,(int) shells_[i].last_ind()+1);
    }
  }


  printf("\nBasis set contains %i functions, maximum angular momentum is %i.\\
n",(int) Nbf(),max_am());
  if(is_lm_default())
    printf("Spherical harmonic Gaussians are used by default, there are %i cartesians.\n",(int) Ncart());
  else
    printf("Cartesian Gaussians are used by default.\n");
}

arma::mat BasisSet::cart_to_sph_trans() const {
  // Form transformation matrix to spherical harmonics

  const size_t Nlm=this->Nlm();
  const size_t Ncart=this->Ncart();

  // Returned matrix
  arma::mat trans(Nlm,Ncart);
  trans.zeros();

  // Bookkeeping indices
  size_t n=0, l=0;

  // Helper matrix
  arma::mat tmp;

  for(size_t i=0;i<shells_.size();i++) {
    // Get angular momentum of shell
    int am=shells_[i].am();

    // Number of cartesians and harmonics on shell
    int Nc=(am+1)*(am+2)/2;
    int Nl=2*am+1;

    // Get transformation matrix
    tmp=Ylm_transmat(am);

    // Store transformation matrix
    trans.submat(l,n,l+Nl-1,n+Nc-1)=tmp;
    n+=Nc;
    l+=Nl;
  }

  return trans;
}

arma::mat BasisSet::sph_to_cart_trans() const {
  // Form transformation matrix to cartesians

  return inv(cart_to_sph_trans());
}


arma::mat BasisSet::overlap() const {
  // Form overlap matrix
  const size_t N=Nbf();
  arma::mat S(N,N);
  S.zeros();

  CintEnv cenv(*this,false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shellpairs_.size();ip++) {
      size_t i=shellpairs_[ip].is;
      size_t j=shellpairs_[ip].js;

      w.compute(CINT1E_OVLP,i,j);
      arma::mat tmp=w.get_mat(0,i,j);

      S.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind())=tmp;
      S.submat(shells_[j].first_ind(),shells_[i].first_ind(),shells_[j].last_ind(),shells_[i].last_ind())=arma::trans(tmp);
    }
  }

  return S;
}

arma::mat BasisSet::coulomb_overlap() const {
  // Form overlap matrix

  // Size of basis set
  const size_t N=Nbf();

  // Initialize matrix
  arma::mat S(N,N);
  S.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs_.size();ip++) {
    // Shells in pair
    size_t i=shellpairs_[ip].is;
    size_t j=shellpairs_[ip].js;

    arma::mat tmp=shells_[i].coulomb_overlap(shells_[j]);

    // Store overlap
    S.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind())=tmp;
    S.submat(shells_[j].first_ind(),shells_[i].first_ind(),shells_[j].last_ind(),shells_[i].last_ind())=arma::trans(tmp);
  }

  return S;
}

arma::mat BasisSet::overlap(const BasisSet & rhs) const {
  // Form overlap wrt to other basis set

  // Size of this basis set
  const size_t Nl=Nbf();
  // Size of rhs basis
  const size_t Nr=rhs.Nbf();

  // Initialize matrix
  arma::mat S12(Nl,Nr);
  S12.zeros();

  // The shells of the other basis follow ours in the environment
  CintEnv cenv(*this,rhs,false);
  const size_t Nsh=shells_.size();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t i=0;i<shells_.size();i++) {
      for(size_t j=0;j<rhs.shells_.size();j++) {
        w.compute(CINT1E_OVLP,i,Nsh+j);
        S12.submat(shells_[i].first_ind(),rhs.shells_[j].first_ind(),
                   shells_[i].last_ind() ,rhs.shells_[j].last_ind() )=w.get_mat(0,i,Nsh+j);
      }
    }
  }
  return S12;
}

arma::mat BasisSet::coulomb_overlap(const BasisSet & rhs) const {
  // Form overlap wrt to other basis set

  // Size of this basis set
  const size_t Nl=Nbf();
  // Size of rhs basis
  const size_t Nr=rhs.Nbf();

  // Initialize matrix
  arma::mat S12(Nl,Nr);
  S12.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<shells_.size();i++) {
    for(size_t j=0;j<rhs.shells_.size();j++) {
      S12.submat(shells_[i].first_ind(),rhs.shells_[j].first_ind(),
		 shells_[i].last_ind() ,rhs.shells_[j].last_ind() )=shells_[i].coulomb_overlap(rhs.shells_[j]);;
    }
  }
  return S12;
}


arma::mat BasisSet::kinetic() const {
  // Form kinetic energy matrix
  size_t N=Nbf();
  arma::mat T(N,N);
  T.zeros();

  CintEnv cenv(*this,false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shellpairs_.size();ip++) {
      size_t i=shellpairs_[ip].is;
      size_t j=shellpairs_[ip].js;

      w.compute(CINT1E_KIN,i,j);
      arma::mat tmp=w.get_mat(0,i,j);

      T.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind())=tmp;
      T.submat(shells_[j].first_ind(),shells_[i].first_ind(),shells_[j].last_ind(),shells_[i].last_ind())=arma::trans(tmp);
    }
  }

  return T;
}

std::vector<arma::mat> BasisSet::gradient_integral() const {
  // Form the <mu|nabla|nu> matrix
  size_t N=Nbf();
  std::vector<arma::mat> T(3);
  for(size_t ic=0; ic<3;ic++)
    T[ic].zeros(N,N);

  CintEnv cenv(*this,false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shellpairs_.size();ip++) {
      size_t i=shellpairs_[ip].is;
      size_t j=shellpairs_[ip].js;

      // The derivative acts on the ket
      w.compute(CINT1E_OVLPIP,i,j);

      // The operator is antisymmetric
      for(size_t ic=0;ic<3;ic++) {
        arma::mat tmp=w.get_mat(ic,i,j);
        T[ic].submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind())=tmp;
        if(i!=j)
          T[ic].submat(shells_[j].first_ind(),shells_[i].first_ind(),shells_[j].last_ind(),shells_[i].last_ind())=-arma::trans(tmp);
      }
    }
  }

  return T;
}

arma::mat BasisSet::nuclear() const {
  std::vector<std::tuple<int,double,double,double>> nuclear_data;
  for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
    if(nuclei_[inuc].bsse)
      continue;
    // Nuclear charge
    int Z=nuclei_[inuc].Z;

    // Coordinates of nucleus
    double cx=nuclei_[inuc].r.x;
    double cy=nuclei_[inuc].r.y;
    double cz=nuclei_[inuc].r.z;
    nuclear_data.push_back(std::make_tuple(Z,cx,cy,cz));
  }
  return nuclear(nuclear_data);
}

arma::mat BasisSet::nuclear(const std::vector<std::tuple<int,double,double,double>> & nuclear_data) const {

  // Size of basis set
  size_t N=Nbf();

  // Initialize matrix
  arma::mat Vnuc(N,N);
  Vnuc.zeros();

  CintEnv cenv(*this,false);

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs_.size();ip++)
    for(size_t inuc=0;inuc<nuclear_data.size();inuc++) {

      auto [Z, cx, cy, cz] = nuclear_data[inuc];

      // Shells in pair
      size_t i=shellpairs_[ip].is;
      size_t j=shellpairs_[ip].js;

      // Get subblock. The attraction operator is -Z/|r-C|.
      const double orig[3]={cx, cy, cz};
      w.compute(CINT1E_RINV,i,j,orig);
      arma::mat tmp=-Z*w.get_mat(0,i,j);

      // On the off diagonal we fill out both sides of the matrix
      if(i!=j) {
	Vnuc.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind())+=tmp;
	Vnuc.submat(shells_[j].first_ind(),shells_[i].first_ind(),shells_[j].last_ind(),shells_[i].last_ind())+=arma::trans(tmp);
      } else
	// On the diagonal we just get it once
	Vnuc.submat(shells_[i].first_ind(),shells_[i].first_ind(),shells_[i].last_ind(),shells_[i].last_ind())+=arma::trans(tmp);
    }
  }

  return Vnuc;
}

arma::mat BasisSet::potential(coords_t r) const {
  // Form nuclear attraction matrix

  // Size of basis set
  size_t N=Nbf();

  // Initialize matrix
  arma::mat V(N,N);
  V.zeros();

  CintEnv cenv(*this,false);

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);
    const double orig[3]={r.x, r.y, r.z};

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs_.size();ip++) {
    // Shells in pair
    size_t i=shellpairs_[ip].is;
    size_t j=shellpairs_[ip].js;

    // Get subblock. The operator is -1/|r-C|.
    w.compute(CINT1E_RINV,i,j,orig);
    arma::mat tmp=-w.get_mat(0,i,j);

    // On the off diagonal we fill out both sides of the matrix
    if(i!=j) {
      V.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind())=tmp;
      V.submat(shells_[j].first_ind(),shells_[i].first_ind(),shells_[j].last_ind(),shells_[i].last_ind())=arma::trans(tmp);
    } else
      // On the diagonal we just get it once
      V.submat(shells_[i].first_ind(),shells_[i].first_ind(),shells_[i].last_ind(),shells_[i].last_ind())=arma::trans(tmp);
  }
  }

  return V;
}

arma::mat BasisSet::sap_potential(const BasisSetLibrary & sapfit) const {
  bool verbose=settings.get_bool("Verbose");
  double intthr=settings.get_double("IntegralThresh");

  Timer t;

  // Get shells in orbital basis
  std::vector<GaussianShell> shells=this->shells();
  // Get list of shell pairs
  double omega=0.0;
  double alpha=1.0;
  double beta=0.0;
  ScreeningData scr=compute_screening(intthr,omega,alpha,beta,false);
  const arma::mat & Q = scr.Q;
  const std::vector<eripair_t> & shpairs = scr.shpairs;
  // and nuclei
  std::vector<nucleus_t> nuclei=this->nuclei();
  if(verbose) {
    printf("%i shell pairs and %i nuclei\n",(int) shpairs.size(), (int) nuclei.size());
    fflush(stdout);
  }

  // Form the SAP shell of each nucleus. The potential is a sum of the
  // atomic screened potentials, each a single contracted s function, so
  // the SAP potential is a three-center integral (mu nu | sap).
  std::vector<GaussianShell> sapshells;
  std::vector<size_t> sapnuc;
  for(size_t inuc=0;inuc<nuclei.size();inuc++) {
    if(nuclei[inuc].bsse)
      continue;

    // Get the SAP basis for the element
    ElementBasisSet sapbas;
    try {
      // Check first if a special set is wanted for given center
      sapbas=sapfit.get_element(nuclei[inuc].symbol,inuc+1);
    } catch(std::runtime_error & err) {
      // Did not find a special basis, use the general one instead.
      sapbas=sapfit.get_element(nuclei[inuc].symbol,0);
    }

    // Get the shells on the element
    std::vector<FunctionShell> bf=sapbas.get_shells();
    if(bf.size() != 1 || bf[0].get_am() != 0)
      throw std::logic_error("SAP basis should only have a single contracted S function per element!\n");
    // Check sum rule
    std::vector<contr_t> contr(bf[0].get_contr());
    double Zsap=0.0;
    for(size_t i=0;i<contr.size();i++)
      Zsap-=contr[i].c;
    if(std::abs(Zsap - nuclei[inuc].Z) >= 1e-3) {
      std::ostringstream oss;
      oss << "SAP basis on nucleus " << inuc+1 << " violates sum rule: " << Zsap << " instead of expected " << nuclei[inuc].Z << "!\n";
      throw std::logic_error(oss.str());
    }

    // Form the SAP shell
    GaussianShell sapsh(GaussianShell(bf[0].get_am(),false,bf[0].get_contr()));
    // and set its center
    sapsh.set_center(nuclei[inuc].r,inuc);
    // Convert the contraction to unnormalized primitives
    sapsh.convert_sap_contraction();

    sapshells.push_back(sapsh);
    sapnuc.push_back(inuc);
  }

  // libcint environment: the orbital shells, followed by the SAP shells
  std::vector<GaussianShell> allshells(shells);
  allshells.insert(allshells.end(),sapshells.begin(),sapshells.end());
  CintEnv cenv(allshells);
  const size_t Nsh=shells.size();

  // Construct repulsive potential
  arma::mat Jx(Nbf(),Nbf());
  Jx.zeros();
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker eri(cenv);
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shpairs.size();ip++) {
      size_t is=shpairs[ip].is;
      size_t js=shpairs[ip].js;

      // We have already computed the Schwarz screening
      double QQ=Q(is,js)*Q(is,js);
      if(QQ<intthr)
        // Small integral
        continue;

      // Loop over the SAP shells
      for(size_t isap=0;isap<sapshells.size();isap++) {
        // Compute integrals
        eri.compute_3c(is,js,Nsh+isap);
        erip=eri.getp();

        // and store them
        size_t Ni(shells[is].Nbf());
        size_t Nj(shells[js].Nbf());
        size_t i0(shells[is].first_ind());
        size_t j0(shells[js].first_ind());

        // Remember minus sign from V(r)=-Z(r)/r
        for(size_t ii=0;ii<Ni;ii++)
          for(size_t jj=0;jj<Nj;jj++) {
            size_t i=i0+ii;
            size_t j=j0+jj;
            Jx(i,j) -= (*erip)[ii*Nj+jj];
          }
        if(is != js) {
          // Symmetrize
          for(size_t ii=0;ii<Ni;ii++)
            for(size_t jj=0;jj<Nj;jj++) {
              size_t i=i0+ii;
              size_t j=j0+jj;
              Jx(j,i) -= (*erip)[ii*Nj+jj];
            }
        }
      }
    }
  }

  if(verbose) {
    printf("SAP potential formed in %.3f s.\n",t.get());
    fflush(stdout);
  }

  return Jx;
}

void BasisSet::eri_screening(arma::mat & Q, arma::mat & M, double omega, double alpha, double beta) const {
  // Get unique pairs
  std::vector<shellpair_t> pairs=unique_shellpairs();

  Q.zeros(shells_.size(),shells_.size());
  M.zeros(shells_.size(),shells_.size());

  // libcint description of the basis
  CintEnv cenv(*this);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri = make_eri_worker(cenv, omega, alpha, beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<pairs.size();ip++) {
      size_t i=pairs[ip].is;
      size_t j=pairs[ip].js;

      // Compute (ij|ij) integrals
      {
        eri->compute(i,j,i,j);
        erip=eri->getp();
        // Get maximum value
        double m=0.0;
        for(size_t k=0;k<erip->size();k++)
          m=std::max(m,std::abs((*erip)[k]));
        m=sqrt(m);
        Q(i,j)=m;
        Q(j,i)=m;
      }

      // Compute (ii|jj) integrals
      {
        eri->compute(i,i,j,j);
        erip=eri->getp();
        // Get maximum value
        double m=0.0;
        for(size_t k=0;k<erip->size();k++)
          m=std::max(m,std::abs((*erip)[k]));
        m=sqrt(m);
        M(i,j)=m;
        M(j,i)=m;
      }
    }
  }
}

arma::vec BasisSet::nuclear_pulay(const arma::mat & P) const {
  arma::vec f(3*nuclei_.size());
  f.zeros();

  CintEnv cenv(*this,false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);
    // Loop over shells
#ifdef _OPENMP
    arma::vec fwrk(3*nuclei_.size());
    fwrk.zeros();

#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shellpairs_.size();ip++)
      for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
	// If BSSE nucleus, do nothing
	if(nuclei_[inuc].bsse)
	  continue;

	// Shells in pair
	size_t i=shellpairs_[ip].is;
	size_t j=shellpairs_[ip].js;

	// Nuclear charge
	int Z=nuclei_[inuc].Z;

	// Coordinates of nucleus
	double cx=nuclei_[inuc].r.x;
	double cy=nuclei_[inuc].r.y;
	double cz=nuclei_[inuc].r.z;

	// Density matrix for the pair
	arma::mat Pmat=P.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind());

	// Get the forces
	const double orig[3]={cx, cy, cz};
	arma::vec tmp=Z*nuclear_pulay_pair(w,i,j,Pmat,orig);

	// Off-diagonal?
	if(i!=j)
	  tmp*=2.0;

	// and increment the nuclear force.
#ifdef _OPENMP
	fwrk.subvec(3*shells_[i].center_ind(),3*shells_[i].center_ind()+2)+=tmp.subvec(0,2);
	fwrk.subvec(3*shells_[j].center_ind(),3*shells_[j].center_ind()+2)+=tmp.subvec(3,5);
#else
	f.subvec(3*shells[i].get_center_ind(),3*shells[i].get_center_ind()+2)+=tmp.subvec(0,2);
	f.subvec(3*shells[j].get_center_ind(),3*shells[j].get_center_ind()+2)+=tmp.subvec(3,5);
#endif
      }

#ifdef _OPENMP
#pragma omp critical
    f+=fwrk;
#endif
  }

  return f;
}

arma::vec BasisSet::nuclear_der(const arma::mat & P) const {
  arma::vec f(3*nuclei_.size());
  f.zeros();

  CintEnv cenv(*this,false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);
    // Loop over shells
#ifdef _OPENMP
    arma::vec fwrk(3*nuclei_.size());
    fwrk.zeros();

#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shellpairs_.size();ip++)
      for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
	// If BSSE nucleus, do nothing
	if(nuclei_[inuc].bsse)
	  continue;

	// Shells in pair
	size_t i=shellpairs_[ip].is;
	size_t j=shellpairs_[ip].js;

	// Nuclear charge
	int Z=nuclei_[inuc].Z;

	// Coordinates of nucleus
	double cx=nuclei_[inuc].r.x;
	double cy=nuclei_[inuc].r.y;
	double cz=nuclei_[inuc].r.z;

	// Density matrix for the pair
	arma::mat Pmat=P.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind());

	// Get the forces
	const double orig[3]={cx, cy, cz};
	arma::vec tmp=Z*nuclear_der_pair(w,i,j,Pmat,orig);

	// Off-diagonal?
	if(i!=j)
	  tmp*=2.0;

	// and increment the nuclear force.
#ifdef _OPENMP
	fwrk.subvec(3*inuc,3*inuc+2)+=tmp;
#else
	f.subvec(3*inuc,3*inuc+2)+=tmp;
#endif
      }

#ifdef _OPENMP
#pragma omp critical
    f+=fwrk;
#endif
  }

  return f;
}

arma::vec BasisSet::kinetic_pulay(const arma::mat & P) const {
  arma::vec f(3*nuclei_.size());
  f.zeros();

  CintEnv cenv(*this,false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);
    // Loop over shells
#ifdef _OPENMP
    arma::vec fwrk(3*nuclei_.size());
    fwrk.zeros();

#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shellpairs_.size();ip++) {
      // Shells in pair
      size_t i=shellpairs_[ip].is;
      size_t j=shellpairs_[ip].js;

      // Density matrix for the pair
      arma::mat Pmat=P.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind());

      // Get the forces
      arma::vec tmp=pulay_pair(w,CINT1E_IPKIN,CINT1E_KINIP,i,j,Pmat,1.0);

      // Off-diagonal?
      if(i!=j)
	tmp*=2.0;

      // and increment the nuclear force.
#ifdef _OPENMP
      fwrk.subvec(3*shells_[i].center_ind(),3*shells_[i].center_ind()+2)+=tmp.subvec(0,2);
      fwrk.subvec(3*shells_[j].center_ind(),3*shells_[j].center_ind()+2)+=tmp.subvec(3,5);
#else
      f.subvec(3*shells[i].get_center_ind(),3*shells[i].get_center_ind()+2)+=tmp.subvec(0,2);
      f.subvec(3*shells[j].get_center_ind(),3*shells[j].get_center_ind()+2)+=tmp.subvec(3,5);
#endif
    }

#ifdef _OPENMP
#pragma omp critical
    f+=fwrk;
#endif
  }

  return f;
}

arma::vec BasisSet::overlap_der(const arma::mat & P) const {
  arma::vec f(3*nuclei_.size());
  f.zeros();

  CintEnv cenv(*this,false);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);
    // Loop over shells
#ifdef _OPENMP
    arma::vec fwrk(3*nuclei_.size());
    fwrk.zeros();

#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<shellpairs_.size();ip++) {

      // Shells in pair
      size_t i=shellpairs_[ip].is;
      size_t j=shellpairs_[ip].js;

      // Density matrix for the pair
      arma::mat Pmat=P.submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind());

      // Get the forces
      arma::vec tmp=pulay_pair(w,CINT1E_IPOVLP,CINT1E_OVLPIP,i,j,Pmat,-1.0);

      // Off-diagonal?
      if(i!=j)
	tmp*=2.0;

      // and increment the nuclear force.
#ifdef _OPENMP
      fwrk.subvec(3*shells_[i].center_ind(),3*shells_[i].center_ind()+2)+=tmp.subvec(0,2);
      fwrk.subvec(3*shells_[j].center_ind(),3*shells_[j].center_ind()+2)+=tmp.subvec(3,5);
#else
      f.subvec(3*shells[i].get_center_ind(),3*shells[i].get_center_ind()+2)+=tmp.subvec(0,2);
      f.subvec(3*shells[j].get_center_ind(),3*shells[j].get_center_ind()+2)+=tmp.subvec(3,5);
#endif
    }

#ifdef _OPENMP
#pragma omp critical
    f+=fwrk;
#endif
  }

  return f;
}

arma::vec BasisSet::nuclear_force() const {
  arma::vec f(3*nuclei_.size());
  f.zeros();

  for(size_t i=0;i<nuclei_.size();i++) {
    if(nuclei_[i].bsse)
      continue;

    for(size_t j=0;j<i;j++) {
      if(nuclei_[j].bsse)
	continue;

      // Calculate distance
      coords_t rij=nuclei_[i].r-nuclei_[j].r;
      // and its third power
      double rcb=pow(norm(rij),3);

      // Force is
      arma::vec F(3);
      F(0)=nuclei_[i].Z*nuclei_[j].Z/rcb*rij.x;
      F(1)=nuclei_[i].Z*nuclei_[j].Z/rcb*rij.y;
      F(2)=nuclei_[i].Z*nuclei_[j].Z/rcb*rij.z;

      f.subvec(3*i,3*i+2)+=F;
      f.subvec(3*j,3*j+2)-=F;
    }
  }

  return f;
}

std::vector<arma::mat> BasisSet::moment(int mom, double x, double y, double z) const {
  // Compute moment integrals around (x,y,z);

  // Number of moments to compute is
  size_t Nmom=(mom+1)*(mom+2)/2;
  // Amount of basis functions is
  size_t Nbf=this->Nbf();

  // Returned array, holding the moment integrals
  std::vector<arma::mat> ret;
  ret.reserve(Nmom);

  // Initialize arrays
  for(size_t i=0;i<Nmom;i++) {
    ret.push_back(arma::mat(Nbf,Nbf));
    ret[i].zeros();
  }

  CintEnv cenv(*this,false);

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    Int1eWorker w(cenv);
    const double orig[3]={x, y, z};

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs_.size();ip++) {
    // Shells in pair
    size_t i=shellpairs_[ip].is;
    size_t j=shellpairs_[ip].js;

    // Compute moment integral over shells
    std::vector<arma::mat> ints=moment_pair(w,mom,i,j,orig);

    // Store moments
    if(i!=j) {
      for(size_t m=0;m<Nmom;m++) {
	ret[m].submat(shells_[i].first_ind(),shells_[j].first_ind(),shells_[i].last_ind(),shells_[j].last_ind())=ints[m];
	ret[m].submat(shells_[j].first_ind(),shells_[i].first_ind(),shells_[j].last_ind(),shells_[i].last_ind())=arma::trans(ints[m]);
      }
    } else {
      for(size_t m=0;m<Nmom;m++)
	ret[m].submat(shells_[i].first_ind(),shells_[i].first_ind(),shells_[i].last_ind(),shells_[i].last_ind())=ints[m];
    }
  }
  }

  return ret;
}

arma::vec BasisSet::integral() const {
  arma::vec ints(Nbf());
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t is=0;is<shells_.size();is++)
    ints.subvec(shells_[is].first_ind(),shells_[is].last_ind())=shells_[is].integral();

  return ints;
}



int BasisSet::Ztot() const {
  int Zt=0;
  for(size_t i=0;i<nuclei_.size();i++) {
    if(nuclei_[i].bsse)
      continue;
    Zt+=nuclei_[i].Z;
  }
  return Zt;
}

double BasisSet::Enuc() const {
  double En=0.0;

  for(size_t i=0;i<nuclei_.size();i++) {
    if(nuclei_[i].bsse)
      continue;

    int Zi=nuclei_[i].Z;

    for(size_t j=0;j<i;j++) {
      if(nuclei_[j].bsse)
	continue;
      int Zj=nuclei_[j].Z;

      En+=Zi*Zj/nucleardist_(i,j);
    }
  }

  return En;
}

void BasisSet::projectMOs(const BasisSet & oldbas, const arma::vec & oldE, const arma::mat & oldMOs, arma::vec & E, arma::mat & MOs, size_t nocc) const {
  if(*this == oldbas) {
    MOs=oldMOs;
    E=oldE;
    return;
  }

  if(oldMOs.n_cols < nocc) {
    oldbas.print();
    fflush(stdout);

    std::ostringstream oss;
    oss << "Old basis doesn't have enough occupied orbitals: " << oldbas.Nbf() << " basis functions but " << nocc << " orbitals wanted!\n";
    throw std::runtime_error(oss.str());
  }

  BasisSet transbas(oldbas);
  transbas.nuclei_=nuclei_;
  // Store new geometries
  for(size_t i=0;i<oldbas.shells_.size();i++) {
    // Index of center in the old basis is
    size_t idx=oldbas.shells_[i].center_ind();
    // Set new coordinates
    transbas.shells_[i].set_center(nuclei_[idx].r,idx);
  }
  transbas.finalize(false,false);

  // Get overlap matrix
  arma::mat S11=overlap();
  // and overlap with old basis
  arma::mat S12=overlap(transbas);

  // Form orthogonalizing matrix
  arma::mat Svec;
  arma::vec Sval;
  eig_sym_ordered(Sval,Svec,S11);

  // Get number of basis functions
  const size_t Nbf=this->Nbf();

  // Count number of eigenvalues that are above cutoff
  size_t Nind=0;
  for(size_t i=0;i<Nbf;i++)
    if(Sval(i)>=settings.get_double("LinDepThresh"))
      Nind++;
  // Number of linearly dependent basis functions
  const size_t Ndep=Nbf-Nind;
  if(Nind<nocc) {
    print();
    fflush(stdout);

    std::ostringstream oss;
    oss << "Basis set too small for occupied orbitals: " << Nind << " independent functions but " << nocc << " orbitals!\n";
    throw std::runtime_error(oss.str());
  }

  // Get rid of linearly dependent eigenvalues and eigenvectors
  Sval=Sval.subvec(Ndep,Nbf-1);
  Svec=Svec.cols(Ndep,Nbf-1);

  // Form canonical orthonormalization matrix
  arma::mat Sinvh(Nbf,Nind);
  for(size_t i=0;i<Nind;i++)
    Sinvh.col(i)=Svec.col(i)/sqrt(Sval(i));

  // and the real S^-1
  arma::mat Sinv=Sinvh*arma::trans(Sinvh);

  // New orbitals and orbital energies
  MOs.zeros(Sinvh.n_rows,Nind);
  E.zeros(Nind);

  if(!nocc) {
    MOs=Sinvh;
    return;
  }

  // Projected orbitals
  MOs.cols(0,nocc-1)=Sinv*S12*oldMOs.cols(0,nocc-1);
  // and energies. PZ calculations might not have energies stored!
  size_t nE=std::min((arma::uword) nocc,oldE.n_elem);
  if(nE>0)
    E.subvec(0,nE-1)=oldE.subvec(0,nE-1);
  // Overlap of projected orbitals is
  arma::mat SMO=arma::trans(MOs.cols(0,nocc-1))*S11*MOs.cols(0,nocc-1);

  // Do eigendecomposition
  arma::mat SMOvec;
  arma::vec SMOval;
  bool ok=arma::eig_sym(SMOval,SMOvec,SMO);
  if(!ok)
    throw std::runtime_error("Failed to diagonalize orbital overlap\n");

  // Orthogonalizing matrix is
  arma::mat orthmat=SMOvec*arma::diagmat(1.0/sqrt(SMOval))*trans(SMOvec);

  // Orthonormal projected orbitals are
  MOs.cols(0,nocc-1)=MOs.cols(0,nocc-1)*orthmat;

  // Form virtual orbitals
  if(nocc<Nind) {
    // MO submatrix
    arma::mat C=MOs.cols(0,nocc-1);

    // Generate other vectors. Compute overlap of functions
    arma::mat X=arma::trans(Sinvh)*S11*C;
    // and perform SVD
    arma::mat U, V;
    arma::vec s;
    bool svdok=arma::svd(U,s,V,X);
    if(!svdok)
      throw std::runtime_error("SVD decomposition failed!\n");

    // Rotate eigenvectors.
    arma::mat Cnew(Sinvh*U);

    // Now, the occupied subspace is found in the first nocc
    // eigenvectors.
    MOs.cols(nocc,Nind-1)=Cnew.cols(nocc,Nind-1);
    // Dummy energies
    if(nE==nocc)
      E.subvec(nocc,Nind-1)=1.1*std::max(E(nocc-1),0.0)*arma::ones(Nind-nocc,1);
  }

  // Failsafe
  try {
    // Check orthogonality of orbitals
    check_orth(MOs,S11,false);
  } catch(std::runtime_error & err) {
    std::ostringstream oss;
    oss << "Orbitals generated by MO_project are not orthonormal.\n";
    throw std::runtime_error(oss.str());
  }
}

void BasisSet::projectOMOs(const BasisSet & oldbas, const arma::cx_mat & oldOMOs, arma::cx_mat & OMOs, size_t nocc) const {
  if(*this == oldbas) {
    OMOs=oldOMOs;
    return;
  }

  if(oldOMOs.n_cols < nocc) {
    oldbas.print();
    fflush(stdout);

    std::ostringstream oss;
    oss << "Old basis doesn't have enough occupied orbitals: " << oldbas.Nbf() << " basis functions but " << nocc << " orbitals wanted!\n";
    throw std::runtime_error(oss.str());
  }

  BasisSet transbas(oldbas);
  transbas.nuclei_=nuclei_;
  // Store new geometries
  for(size_t i=0;i<oldbas.shells_.size();i++) {
    // Index of center in the old basis is
    size_t idx=oldbas.shells_[i].center_ind();
    // Set new coordinates
    transbas.shells_[i].set_center(nuclei_[idx].r,idx);
  }
  transbas.finalize(false,false);

  // Get overlap matrix
  arma::mat S11=overlap();
  // and overlap with old basis
  arma::mat S12=overlap(transbas);

  // Form orthogonalizing matrix
  arma::mat Svec;
  arma::vec Sval;
  eig_sym_ordered(Sval,Svec,S11);

  // Get number of basis functions
  const size_t Nbf=this->Nbf();

  // Count number of eigenvalues that are above cutoff
  size_t Nind=0;
  for(size_t i=0;i<Nbf;i++)
    if(Sval(i)>=settings.get_double("LinDepThresh"))
      Nind++;
  // Number of linearly dependent basis functions
  const size_t Ndep=Nbf-Nind;
  if(Nind<nocc) {
    print();
    fflush(stdout);

    std::ostringstream oss;
    oss << "Basis set too small for occupied orbitals: " << Nind << " independent functions but " << nocc << " orbitals!\n";
    throw std::runtime_error(oss.str());
  }

  // Get rid of linearly dependent eigenvalues and eigenvectors
  Sval=Sval.subvec(Ndep,Nbf-1);
  Svec=Svec.cols(Ndep,Nbf-1);

  // Form canonical orthonormalization matrix
  arma::mat Sinvh(Nbf,Nind);
  for(size_t i=0;i<Nind;i++)
    Sinvh.col(i)=Svec.col(i)/sqrt(Sval(i));

  // and the real S^-1
  arma::mat Sinv=Sinvh*arma::trans(Sinvh);

  // New orbitals and orbital energies
  OMOs.zeros(Sinvh.n_rows,Nind);

  if(!nocc) {
    OMOs=Sinvh*COMPLEX1;
    return;
  }

  // Projected orbitals
  OMOs.cols(0,nocc-1)=Sinv*S12*oldOMOs.cols(0,nocc-1);

  // Overlap of projected orbitals is
  arma::cx_mat SMO=arma::trans(OMOs.cols(0,nocc-1))*S11*OMOs.cols(0,nocc-1);

  // Do eigendecomposition
  arma::cx_mat SMOvec;
  arma::vec SMOval;
  bool ok=arma::eig_sym(SMOval,SMOvec,SMO);
  if(!ok)
    throw std::runtime_error("Failed to diagonalize orbital overlap\n");

  // Orthogonalizing matrix is
  arma::cx_mat orthmat=SMOvec*arma::diagmat(1.0/sqrt(SMOval))*trans(SMOvec);

  // Orthonormal projected orbitals are
  OMOs.cols(0,nocc-1)=OMOs.cols(0,nocc-1)*orthmat;

  // Form virtual orbitals
  if(nocc<Nind) {
    // MO submatrix
    arma::cx_mat C=OMOs.cols(0,nocc-1);

    // Generate other vectors. Compute overlap of functions
    arma::cx_mat X=arma::trans(Sinvh)*S11*C;
    // and perform SVD
    arma::cx_mat U, V;
    arma::vec s;
    bool svdok=arma::svd(U,s,V,X);
    if(!svdok)
      throw std::runtime_error("SVD decomposition failed!\n");

    // Rotate eigenvectors.
    arma::cx_mat Cnew(Sinvh*U);

    // Now, the occupied subspace is found in the first nocc
    // eigenvectors.
    OMOs.cols(nocc,Nind-1)=Cnew.cols(nocc,Nind-1);
  }

  // Failsafe
  try {
    // Check orthogonality of orbitals
    check_orth(OMOs,S11,false);
  } catch(std::runtime_error & err) {
    std::ostringstream oss;
    oss << "Orbitals generated by OMO_project are not orthonormal.\n";
    throw std::runtime_error(oss.str());
  }
}

bool exponent_compare(const GaussianShell & lhs, const GaussianShell & rhs) {
  return lhs.contr()[0].z>rhs.contr()[0].z;
}

BasisSet BasisSet::density_fitting(double fsam, int lmaxinc) const {
  // Automatically generate density fitting basis.

  // R. Yang, A. P. Rendell and M. J. Frisch, "Automatically generated
  // Coulomb fitting basis sets: Design and accuracy for systems
  // containing H to Kr", J. Chem. Phys. 127 (2007), 074102

  bool uselm0(settings.get_bool("UseLM"));
  settings.set_bool("UseLM",true);
  // Density fitting basis set
  BasisSet dfit(1);
  settings.set_bool("UseLM",uselm0);

  // Loop over nuclei
  for(size_t in=0;in<nuclei_.size();in++) {
    // Add nucleus to fitting set
    dfit.add_nucleus(nuclei_[in]);
    // Dummy nucleus
    nucleus_t nuc=nuclei_[in];

    // Define lval - (1) in YRF
    int lval;
    if(nuclei_[in].Z<3)
      lval=0;
    else if(nuclei_[in].Z<19)
      lval=1;
    else if(nuclei_[in].Z<55)
      lval=2;
    else
      lval=3;

    // Get shells corresponding to this nucleus
    std::vector<GaussianShell> shs=funcs(in);

    // Form candidate set - (2), (3) and (6) in YRF
    std::vector<GaussianShell> cand;
    for(size_t i=0;i<shs.size();i++) {
      // Get angular momentum
      int am=2*shs[i].am();
      // Get exponents
      std::vector<contr_t> contr=shs[i].contr();

      // Dummy contraction
      std::vector<contr_t> C(1);
      C[0].c=1.0;

      for(size_t j=0;j<contr.size();j++) {
	// Set exponent
	C[0].z=2.0*contr[j].z;

	// Check that candidate set doesn't already contain the same function
	bool found=0;
	for(size_t k=0;k<cand.size();k++)
	  if((cand[k].am()==am) && (cand[k].contr()[0]==C[0])) {
	    found=1;
	    break;
	  }

	// Add function
	if(!found) {
	  cand.push_back(GaussianShell(am,true,C));
	  cand[cand.size()-1].set_center(nuc.r,in);
	}
      }
    }

    // Sort trial set in order of decreasing exponents (don't care
    // about angular momentum) - (4) in YRF
    std::stable_sort(cand.begin(),cand.end(),exponent_compare);

    // Define maximum angular momentum for candidate functions and for
    // density fitting - (5) in YRF
    int lmax_obs=0;
    for(size_t i=0;i<shs.size();i++)
      if(shs[i].am()>lmax_obs)
	lmax_obs=shs[i].am();
    int lmax_abs=std::max(lmax_obs+lmaxinc,2*lval);

    // (6) was already above.

    while(cand.size()>0) {
      // Generate trial set
      std::vector<GaussianShell> trial;

      // Function with largest exponent is moved to the trial set and
      // its exponent is set as the reference value - (7) in YRF
      double ref=(cand[0].contr())[0].z;
      trial.push_back(cand[0]);
      cand.erase(cand.begin());

      if(cand.size()>0) {
	// More functions remaining, move all for which ratio of
	// reference to exponent is smaller than fsam - (8) in YRF
	for(size_t i=cand.size()-1;i<cand.size();i--)
	  if(ref/((cand[i].contr())[0].z)<fsam) {
	    trial.push_back(cand[i]);
	    cand.erase(cand.begin()+i);
	  }

	// Compute geometric average of exponents - (9) in YRF
	double geomav=1.0;
	for(size_t i=0;i<trial.size();i++)
	  geomav*=trial[i].contr()[0].z;
	geomav=pow(geomav,1.0/trial.size());

	//	printf("Geometric average of %i functions is %e.\n",(int) trial.size(),geomav);

	// Form list of angular momentum values
	// Compute maximum angular moment of current trial set
	int ltrial=0;
	for(size_t i=0;i<trial.size();i++)
	  if(trial[i].am()>ltrial)
	    ltrial=trial[i].am();

	// If this is larger than allowed, renormalize
	if(ltrial>lmax_abs)
	  ltrial=lmax_abs;

	// Form list of angular momentum already used in ABS
	std::vector<int> lvals(::max_am+1);
	for(int i=0;i<=::max_am;i++)
	  lvals[i]=0;

	// Maximum angular momentum of trial functions is
	lvals[ltrial]++;
	// Get shells on current center
	std::vector<GaussianShell> cur_shells=dfit.funcs(in);
	for(size_t i=0;i<cur_shells.size();i++)
	  lvals[cur_shells[i].am()]++;

	// Check that there are no gaps in lvals
	bool fill=0;
	for(size_t i=lvals.size()-1;i<lvals.size();i--) {
	  // Fill down from here below
	  if(!fill && lvals[i]>0)
	    fill=1;
	  if(fill && !lvals[i])
	    lvals[i]++;
	}

	// Add density fitting functions
	std::vector<contr_t> C(1);
	C[0].c=1.0;
	C[0].z=geomav;
	for(int l=0;l<=::max_am;l++)
	  if(lvals[l]>0) {
	    // Pure spherical functions used
	    dfit.add_shell(in,l,true,C);
	  }
      }
    } // (10) in YRF
  } // (11) in YRF

  // Normalize basis set
  dfit.coulomb_normalize();
  // Form list of unique shell pairs
  dfit.form_unique_shellpairs();

  return dfit;
}


BasisSet BasisSet::exchange_fitting() const {
  // Exchange fitting basis set

  bool uselm0(settings.get_bool("UseLM"));
  settings.set_bool("UseLM",true);
  // Density fitting basis set
  BasisSet fit(nuclei_.size());
  settings.set_bool("UseLM",uselm0);

  const int maxam=max_am();

  // Loop over nuclei
  for(size_t in=0;in<nuclei_.size();in++) {
    // Get shells corresponding to this nucleus
    std::vector<GaussianShell> shs=funcs(in);

    // Sort shells in increasing angular momentum
    std::sort(shs.begin(),shs.end());

    // Determine amount of functions on current atom and minimum and maximum exponents
    std::vector<int> nfunc(2*maxam+1);
    std::vector<double> mine(2*maxam+1);
    std::vector<double> maxe(2*maxam+1);
    int lmax=0;

    // Initialize arrays
    for(int l=0;l<=2*maxam;l++) {
      nfunc[l]=0;
      mine[l]=DBL_MAX;
      maxe[l]=0.0;
    }

    // Loop over shells of current nucleus
    for(size_t ish=0;ish<shs.size();ish++)
      // Second loop over shells of current nucleus
      for(size_t jsh=0;jsh<shs.size();jsh++) {

	// Current angular momentum
	int l=shs[ish].am()+shs[jsh].am();

	// Update maximum value
	if(l>lmax)
	  lmax=l;

	// Increase amount of functions
	nfunc[l]++;

	// Get exponential contractions
	std::vector<contr_t> icontr=shs[ish].contr();
	std::vector<contr_t> jcontr=shs[jsh].contr();

	// Minimum exponent
	double mi=icontr[icontr.size()-1].z+jcontr[jcontr.size()-1].z;
	// Maximum exponent
	double ma=icontr[0].z+jcontr[0].z;

	// Check global minimum and maximum
	if(mi<mine[l])
	  mine[l]=mi;
	if(ma>maxe[l])
	  maxe[l]=ma;
      }

    // Add functions to fitting basis set
    for(int l=0;l<=lmax;l++) {
      std::vector<contr_t> C(1);
      C[0].c=1.0;

      // Compute even-tempered formula
      double alpha=mine[l];
      double beta;
      if(nfunc[l]>1)
	beta=pow(maxe[l]/mine[l],1.0/(nfunc[l]-1));
      else
	beta=1.0;

      // Add even-tempered functions
      for(int n=0;n<nfunc[l];n++) {
	// Compute exponent
	C[0].z=alpha*pow(beta,n);
	fit.add_shell(in,l,true,C);
      }
    }
  }

  // Normalize basis set
  fit.coulomb_normalize();
  // Form list of unique shell pairs
  fit.form_unique_shellpairs();

  return fit;
}

BasisSet BasisSet::cholesky_aux_basis(double thr, int linc) const {
  // Per-nucleus atomic Cholesky decomposition. A given element can
  // carry different orbital bases on different centers (mixed-basis
  // calculations), so we run cholesky_set per nucleus and tag each
  // returned ElementBasisSet with the atom number; construct_basis
  // then picks up the per-atom override via baslib.get_element(el,
  // num+1) at line 3798.

  BasisSetLibrary aux_lib;
  for(size_t inuc=0; inuc<nuclei_.size(); inuc++) {
    // Build the orbital ElementBasisSet for this nucleus
    ElementBasisSet el(nuclei_[inuc].symbol);
    std::vector<GaussianShell> shs(funcs(inuc));
    for(size_t ish=0; ish<shs.size(); ish++)
      el.add_function(FunctionShell(shs[ish].am(), shs[ish].contr()));

    // metric=0  -> Coulomb metric (the right one for ERI fitting)
    // full=false -> use one-step ERIchol pivoting per atom to skip
    //              insignificant aux candidates before secondary CD
    ElementBasisSet aux_el(el.cholesky_set(thr, false, 0));
    // Optionally prune high-angular-momentum aux shells (Lehtola JCTC 19,
    // 6242 (2023)); l_obs is this center's orbital l_max. The aux basis is
    // left uncontracted.
    if(linc>=0) {
      int Z=::get_Z(el.get_symbol());
      int l_keep=aux_lmax_keep(Z, el.get_max_am(), linc);
      aux_el.truncate_shells(std::map<int,int>{{Z, l_keep}});
    }
    // Tag with atom number so construct_basis routes the right aux
    // basis to the right center even when atoms of the same element
    // carry different orbital primitives.
    aux_el.set_number(nuclei_[inuc].ind + 1);
    aux_lib.add_element(aux_el);
  }

  // Aux basis is always spherical-harmonic
  bool uselm0 = settings.get_bool("UseLM");
  settings.set_bool("UseLM", true);
  BasisSet aux(1);
  construct_basis(aux, nuclei_, aux_lib);
  aux.coulomb_normalize();
  settings.set_bool("UseLM", uselm0);

  return aux;
}

bool BasisSet::same_geometry(const BasisSet & rhs) const {
  if(nuclei_.size() != rhs.nuclei_.size())
    return false;

  for(size_t i=0;i<nuclei_.size();i++)
    if(!(nuclei_[i]==rhs.nuclei_[i])) {
      //      fprintf(stderr,"Nuclei %i differ!\n",(int) i);
      return false;
    }

  return true;
}

bool BasisSet::same_shells(const BasisSet & rhs) const {
  if(shells_.size() != rhs.shells_.size())
    return false;

  for(size_t i=0;i<shells_.size();i++)
    if(!(shells_[i]==rhs.shells_[i])) {
      //      fprintf(stderr,"Shells %i differ!\n",(int) i);
      return false;
    }

  return true;
}

bool BasisSet::operator==(const BasisSet & rhs) const {
  return same_geometry(rhs) && same_shells(rhs);
}

BasisSet BasisSet::decontract(arma::mat & m) const {
  // Decontract basis set. m maps old basis functions to new ones

  // Contraction schemes for the nuclei
  std::vector< std::vector<arma::mat> > coeffs(nuclei_.size());
  std::vector< std::vector<arma::vec> > exps(nuclei_.size());
  // Is puream used on the shell?
  std::vector< std::vector<bool> > puream(nuclei_.size());

  // Amount of new basis functions
  size_t Nbfnew=0;

  // Collect the schemes. Loop over the nuclei.
  for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
    // Construct an elemental basis set for the nucleus
    ElementBasisSet elbas(symbol(inuc));

    // Get the shells belonging to this nucleus
    std::vector<GaussianShell> shs=funcs(inuc);

    // and add the contractions to the elemental basis set
    for(size_t ish=0;ish<shs.size();ish++) {
      // Angular momentum is
      int am=shs[ish].am();
      // Normalized contraction coefficients
      std::vector<contr_t> c=shs[ish].contr_normalized();
      FunctionShell fsh(am,c);
      elbas.add_function(fsh);
    }

    // Sanity check - puream must be the same for all shells of the current nucleus with the same am
    if(shs.size()>0) {
      std::vector<int> pam;
      for(int am=0;am<=elbas.get_max_am();am++) {
	// Initialization value
	pam.push_back(-1);

	for(size_t ish=0;ish<shs.size();ish++) {
	  // Skip if am is not the same
	  if(shs[ish].am()!=am)
	    continue;

	  // Is this the first shell of the type?
	  if(pam[am]==-1)
	    pam[am]=shs[ish].lm_in_use();
	  else if(shs[ish].lm_in_use()!=pam[am]) {
	    ERROR_INFO();
	    throw std::runtime_error("BasisSet::decontract not implemented for mixed pure am on the same center.\n");
	  }
	}

	// Store the value
	puream[inuc].push_back(pam[am]==1);
      }
    }

    // Exponents and contraction schemes
    for(int am=0;am<=elbas.get_max_am();am++) {
      arma::vec z;
      arma::mat c;
      elbas.get_primitives(z,c,am);
      coeffs[inuc].push_back(c);
      exps[inuc].push_back(z);

      if(puream[inuc][am])
	Nbfnew+=(2*am+1)*z.size();
      else
	Nbfnew+=(am+1)*(am+2)/2*z.size();
    }
  }

  // Now form the new, decontracted basis set.
  BasisSet dec;
  // Initialize transformation matrix
  m.zeros(Nbfnew,Nbf());

  // Add the nuclei
  for(size_t i=0;i<nuclei_.size();i++)
    dec.add_nucleus(nuclei_[i]);

  // and the shells.
  for(size_t inuc=0;inuc<nuclei_.size();inuc++) {
    // Get the shells belonging to this nucleus
    std::vector<GaussianShell> shs=funcs(inuc);

    // Generate the new basis functions. Loop over am
    for(int am=0;am<(int) coeffs[inuc].size();am++) {
      // First functions with the exponents are
      std::vector<size_t> ind0;

      // Add the new shells
      for(size_t iz=0;iz<exps[inuc][am].size();iz++) {
	// Index of first function is
	ind0.push_back(dec.Nbf());
	// Add the shell
	std::vector<contr_t> hlp(1);
	hlp[0].c=1.0;
	hlp[0].z=exps[inuc][am][iz];
	dec.add_shell(inuc,am,puream[inuc][am],hlp,false);
      }

      // and store the coefficients
      for(size_t ish=0;ish<shs.size();ish++)
	if(shs[ish].am()==am) {
	  // Get the normalized contraction on the shell
	  std::vector<contr_t> ct=shs[ish].contr_normalized();
	  // and loop over the exponents
	  for(size_t ic=0;ic<ct.size();ic++) {

	    // Find out where the exponent is in the new basis set
	    size_t ix;
	    for(ix=0;ix<exps[inuc][am].size();ix++)
	      if(exps[inuc][am][ix]==ct[ic].z)
		// Found exponent
		break;

	    // Now that we know where the exponent is in the new basis
	    // set, we can just store the coefficients. So, loop over
	    // the functions on the shell
	    for(size_t ibf=0;ibf<shs[ish].Nbf();ibf++)
	      m(ind0[ix]+ibf,shs[ish].first_ind()+ibf)=ct[ic].c;
	  }
	}
    }
  }

  // Finalize the basis
  dec.finalize();

  return dec;
}

GaussianShell dummyshell() {
  // Set center
  coords_t r;
  r.x=0.0;
  r.y=0.0;
  r.z=0.0;

  std::vector<contr_t> C(1);
  C[0].c=1.0;
  C[0].z=0.0;

  GaussianShell sh(0,false,C);
  sh.set_center(r,0);

  return sh;
}

std::vector<size_t> i_idx(size_t N) {
  std::vector<size_t> ret;
  ret.reserve(N);
  ret.resize(N);
  for(size_t i=0;i<N;i++)
    ret[i]=(i*(i+1))/2;
  return ret;
}

void construct_basis(BasisSet & basis, const std::vector<nucleus_t> & nuclei, const BasisSetLibrary & baslib) {
  std::vector<atom_t> atoms(nuclei.size());
  for(size_t i=0;i<nuclei.size();i++) {
    atoms[i].x=nuclei[i].r.x;
    atoms[i].y=nuclei[i].r.y;
    atoms[i].z=nuclei[i].r.z;
    atoms[i].Q=nuclei[i].Q;
    atoms[i].num=nuclei[i].ind;
    atoms[i].el=nuclei[i].symbol;
  }

  construct_basis(basis,atoms,baslib);
}

void construct_basis(BasisSet & basis, const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib) {
  // Number of atoms is
  size_t Nat=atoms.size();

  // Indices of atoms to decontract basis set for
  std::vector<size_t> dec;
  bool decall=false;
  if(stricmp(settings.get_string("Decontract"),"")!=0) {
    // Check for '*'
    std::string str=settings.get_string("Decontract");
    if(str.size()==1 && str[0]=='*')
      decall=true;
    else
      // Parse and convert to C++ indexing
      dec=parse_range(settings.get_string("Decontract"),true);
  }

  // Rotation?
  bool rotate=settings.get_bool("BasisRotate");
  double cutoff=settings.get_double("BasisCutoff");

  // Create basis set
  basis=BasisSet(Nat);
  // and add atoms to basis set
  for(size_t i=0;i<Nat;i++) {
    // First we need to add the nucleus itself.
    nucleus_t nuc;

    // Get center
    nuc.r.x=atoms[i].x;
    nuc.r.y=atoms[i].y;
    nuc.r.z=atoms[i].z;
    // Charge status
    nuc.Q=atoms[i].Q;

    // Get symbol in raw form
    std::string el=atoms[i].el;

    // Determine if nucleus is BSSE or not
    nuc.bsse=false;
    if(el.size()>3 && el.substr(el.size()-3,3)=="-Bq") {
      // Yes, this is a BSSE nucleus
      nuc.bsse=true;
      el=el.substr(0,el.size()-3);
    }

    // Set symbol
    nuc.symbol=el;
    // Set charge
    nuc.Z=get_Z(el);
    // and add the nucleus.
    basis.add_nucleus(nuc);

    // Now add the basis functions.
    ElementBasisSet elbas;
    try {
      // Check first if a special set is wanted for given center
      elbas=baslib.get_element(el,atoms[i].num+1);
    } catch(std::runtime_error & err) {
      // Did not find a special basis, use the general one instead.
      elbas=baslib.get_element(el,0);
    }

    // ERKALE is all-electron: refuse a basis that carries an effective
    // core potential for a used element (e.g. a BSE JSON set that pairs
    // valence shells with ecp_potentials). Reading only the valence
    // shells and dropping the ECP would be physically wrong, so fail
    // loudly rather than silently.
    if(elbas.has_ecp()) {
      std::ostringstream oss;
      oss << "The basis set for element " << el << " carries an effective core potential (ecp_potentials), which ERKALE does not support.\n";
      throw std::runtime_error(oss.str());
    }

    // Decontract set?
    bool decon=false;
    if(decall)
      // All functions decontracted
      decon=true;
    else
      // Check if this center is decontracted
      for(size_t j=0;j<dec.size();j++)
	if(i==dec[j])
	  decon=true;

    if(decon)
      elbas.decontract();
    else if(rotate)
      elbas.P_orthogonalize(cutoff);

    basis.add_shells(i,elbas);
  }

  // Finalize basis set and convert contractions
  basis.finalize(true);
}

arma::vec compute_orbitals(const arma::mat & C, const BasisSet & bas, const coords_t & r) {
  // Evaluate basis functions
  arma::vec bf=bas.eval_func(r.x,r.y,r.z);

  // Orbitals are
  arma::rowvec orbs=arma::trans(bf)*C;

  return arma::trans(orbs);
}

arma::mat fermi_lowdin_orbitals(const arma::mat & C, const BasisSet & bas, const arma::mat & r) {
  if(r.n_cols!=3)
    throw std::logic_error("r should have three columns for x, y, z!\n");
  if(r.n_rows != C.n_cols)
    throw std::logic_error("r should have as many rows as there are orbitals to localize!\n");
  if(C.n_rows != bas.Nbf())
    throw std::logic_error("C does not correspond to basis set!\n");

  // Evaluate basis function matrix: Nbf x nFOD
  arma::mat bf(bas.Nbf(), r.n_rows);
  for(size_t i=0;i<r.n_rows;i++)
    bf.col(i)=bas.eval_func(r(i,0),r(i,1),r(i,2));
  // Compute the values of the orbitals at the FODs
  arma::mat g(bf.t()*C); // nFOD x Nmo
  g.print("Orbitals' values at FODs");
  // Value of electron density at the FODs
  arma::vec n(arma::sqrt(arma::sum(arma::pow(g.t(),2))).t());

  // T matrix (T_{ij}=g_j (r_i) / sqrt(n(r_i)/2))
  arma::mat T(r.n_rows, r.n_rows);
  for(size_t j=0;j<r.n_rows;j++)
    T.col(j)=g.col(j)/n;

  arma::square(n).print("Electron density at FODs");

  // Form unitary transform
  arma::mat TTt(T*T.t());
  arma::vec tval;
  arma::mat tvec;
  arma::eig_sym(tval,tvec,TTt);

  arma::mat invsqrtTTt(tvec*arma::diagmat(arma::pow(tval,-0.5))*tvec.t());
  arma::mat U(invsqrtTTt*T);

  // Orbitals are rotated as C -> C U^T
  U=U.t();

  arma::mat grot(g*U);
  grot.print("FLO values at FODs");

  return U;
}

double compute_density(const arma::mat & P, const BasisSet & bas, const coords_t & r) {
  // Evaluate basis functions
  arma::vec bf=bas.eval_func(r.x,r.y,r.z);

  // Density is
  return arma::as_scalar(arma::trans(bf)*P*bf);
}

void compute_density_gradient(const arma::mat & P, const BasisSet & bas, const coords_t & r, double & d, arma::vec & g) {
  // Evaluate basis functions
  arma::vec bf=bas.eval_func(r.x,r.y,r.z);
  // and gradients
  arma::mat grad=bas.eval_grad(r.x,r.y,r.z);

  // Density is
  d=arma::as_scalar(arma::trans(bf)*P*bf);
  // and the gradient
  g=arma::trans(arma::trans(bf)*P*grad);
}

void compute_density_gradient_hessian(const arma::mat & P, const BasisSet & bas, const coords_t & r, double & d, arma::vec & g, arma::mat & h) {
  // Evaluate basis functions
  arma::vec bf=bas.eval_func(r.x,r.y,r.z);
  // and gradients
  arma::mat grad=bas.eval_grad(r.x,r.y,r.z);
  // and hessians
  arma::mat hess=bas.eval_hess(r.x,r.y,r.z);

  // Density is
  d=arma::as_scalar(arma::trans(bf)*P*bf);
  // and the gradient
  g=arma::trans(arma::trans(bf)*P*grad);

  // First part of hessian is
  arma::vec hf=arma::trans(bf)*P*hess;
  // and second part
  arma::mat hs=arma::trans(grad)*P*grad;

  // Convert to matrix form
  h=2.0*(arma::reshape(hf,3,3)+hs);
}

double compute_potential(const arma::mat & P, const BasisSet & bas, const coords_t & r) {
  // Compute nuclear contribution
  std::vector<nucleus_t> nucs=bas.nuclei();
  double nucphi=0.0;
  for(size_t i=0;i<nucs.size();i++)
    if(!nucs[i].bsse)
      nucphi+=nucs[i].Z/norm(r - nucs[i].r);

  // Get potential energy matrix
  arma::mat V=bas.potential(r);
  // Electronic contribution is (minus sign is already in the definition of the potential matrix)
  double elphi=arma::trace(P*V);

  return nucphi+elphi;
}

double compute_elf(const arma::mat & P, const BasisSet & bas, const coords_t & r) {
  // Evaluate basis functions
  arma::vec bf=bas.eval_func(r.x,r.y,r.z);
  // and gradients
  arma::mat grad=bas.eval_grad(r.x,r.y,r.z);

  // Compute kinetic energy term (eqn 9)
  double tau=arma::trace(arma::trans(grad)*P*grad);

  // Compute rho and grad rho
  double rho=arma::as_scalar(arma::trans(bf)*P*bf);
  // and the gradient
  arma::vec grho=arma::trans(arma::trans(bf)*P*grad);

  // The D value (eqn 10) is
  double D  = tau - 0.25*arma::dot(grho,grho)/rho;
  // and the corresponding isotropic D is (eqn 13)
  double D0 = 3.0/5.0 * std::pow(6.0*M_PI*M_PI,2.0/3.0) * std::pow(rho,5.0/3.0);

  // from which the chi value is
  double chi = D/D0;

  // making the ELF
  return 1.0 / (1.0 + chi*chi);
}

std::vector< std::vector<size_t> > BasisSet::find_identical_shells() const {
  // Returned list of identical basis functions
  std::vector< std::vector<size_t> > ret;

  // Loop over shells
  for(size_t ish=0;ish<shells_.size();ish++) {
    // Get exponents, contractions and cartesian functions on shell
    std::vector<contr_t> shell_contr=shells_[ish].contr();
    std::vector<shellf_t> shell_cart=shells_[ish].cart();

    // Try to find the shell on the current list of identicals
    bool found=false;
    for(size_t iident=0;iident<ret.size();iident++) {

      // Check first cartesian part.
      std::vector<shellf_t> cmp_cart=shells_[ret[iident][0]].cart();

      if(shell_cart.size()==cmp_cart.size()) {
	// Default value
	found=true;

	for(size_t icart=0;icart<shell_cart.size();icart++)
	  if(shell_cart[icart].l!=cmp_cart[icart].l || shell_cart[icart].m!=cmp_cart[icart].m || shell_cart[icart].n!=cmp_cart[icart].n)
	    found=false;

	// Check that usage of spherical harmonics matches, too
	if(shells_[ish].lm_in_use() != shells_[ret[iident][0]].lm_in_use())
	  found=false;

	// If cartesian parts match, check also exponents and contraction coefficients
	if(found) {
	  // Get exponents
	  std::vector<contr_t> cmp_contr=shells_[ret[iident][0]].contr();

	  // Check exponents
	  if(shell_contr.size()==cmp_contr.size()) {
	    for(size_t ic=0;ic<shell_contr.size();ic++)
	      if(!(shell_contr[ic]==cmp_contr[ic]))
		found=false;
	  } else
	    found=false;
	}

	// If everything matches, add the function to the current list.
	if(found) {
	  ret[iident].push_back(ish);
	  // Stop iteration over list of identical functions
	  break;
	}
      }
    }

    // If the shell was not found on the list of identicals, add it
    if(!found) {
      std::vector<size_t> hlp;
      hlp.push_back(ish);
      ret.push_back(hlp);
    }
  }

  return ret;
}

double orth_diff(const arma::mat & C, const arma::mat & S) {
  // Compute difference from unit overlap
  arma::mat d(arma::abs(arma::trans(C)*S*C-arma::eye<arma::mat>(C.n_cols,C.n_cols)));
  // Get maximum error
  return arma::max(arma::max(d));
}

double orth_diff(const arma::cx_mat & C, const arma::mat & S) {
  // Compute difference from unit overlap
  arma::mat d(arma::abs(arma::trans(C)*S*C - arma::eye<arma::mat>(C.n_cols,C.n_cols)));
  // Get maximum error
  return arma::max(arma::max(d));
}

void check_orth(const arma::mat & C, const arma::mat & S, bool verbose, double thr) {
  if(!C.n_cols)
    throw std::logic_error("Error in check_orth: no orbitals!\n");
  if(C.n_rows != S.n_rows) {
    std::ostringstream oss;
    oss << "Error in check_orth: got " << C.n_rows << " x " << C.n_cols << " C and " << S.n_rows << " x " << S.n_cols << " S!\n";
    throw std::logic_error(oss.str());
  }

  // Compute difference from unit overlap
  arma::mat d(arma::abs(arma::trans(C)*S*C-arma::eye<arma::mat>(C.n_cols,C.n_cols)));
  // Get maximum error
  double maxerr(arma::max(arma::max(d)));

  if(verbose) {
    printf("Maximum deviation from orthogonality is %e.\n",maxerr);
    fflush(stdout);
  }

  if(maxerr>thr) {
    // Clean up
    for(size_t j=0;j<d.n_cols;j++)
      for(size_t i=0;i<d.n_cols;i++)
	if(fabs(d(i,j))<10*DBL_EPSILON)
	  d(i,j)=0.0;

    d.save("MOovl_diff.dat",arma::raw_ascii);

    std::ostringstream oss;
    oss << "Generated orbitals are not orthonormal! Maximum deviation from orthonormality is " << maxerr <<".\nCheck the used LAPACK implementation.\n";
    throw std::runtime_error(oss.str());
  }
}

void check_orth(const arma::cx_mat & C, const arma::mat & S, bool verbose, double thr) {
  if(!C.n_cols)
    throw std::logic_error("Error in check_orth: no orbitals!\n");
  if(C.n_rows != S.n_rows) {
    std::ostringstream oss;
    oss << "Error in check_orth: got " << C.n_rows << " x " << C.n_cols << " C and " << S.n_rows << " x " << S.n_cols << " S!\n";
    throw std::logic_error(oss.str());
  }

  arma::mat d(arma::abs(arma::trans(C)*S*C - arma::eye<arma::cx_mat>(C.n_cols,C.n_cols)));
  double maxerr(arma::max(arma::max(d)));

  if(verbose) {
    printf("Maximum deviation from orthogonality is %e.\n",maxerr);
    fflush(stdout);
  }

  if(maxerr>thr) {
    // Clean up
    for(size_t i=0;i<d.n_cols;i++)
      for(size_t j=0;j<d.n_cols;j++)
	if(std::abs(d(i,j))<10*DBL_EPSILON)
	  d(i,j)=0.0;

    d.save("OMOovl_diff.dat",arma::raw_ascii);

    std::ostringstream oss;
    oss << "Generated orbitals are not orthonormal! Maximum deviation from orthonormality is " << maxerr <<".\nCheck the used LAPACK implementation.\n";
    throw std::runtime_error(oss.str());
  }
}

template<typename T> static arma::Mat<T> construct_IAO_wrk(const BasisSet & basis, const arma::Mat<T> & C, std::vector< std::vector<size_t> > & idx, bool verbose, std::string minbaslib) {
  // Get minao library
  BasisSetLibrary minao;
  minao.load_basis(minbaslib);
  // Default settings
  Settings set;
  set.add_scf_settings();
  set.set_bool("Verbose",verbose);
  // Can't rotate basis or it will break the contractions
  set.set_bool("BasisRotate",false);

  // Construct minimal basis set
  BasisSet minbas;
  construct_basis(minbas,basis.nuclei(),minao);

  // Get indices
  idx.clear();
  idx.resize(minbas.Nnuc());
  for(size_t inuc=0;inuc<minbas.Nnuc();inuc++) {
    // Get shells on nucleus
    std::vector<GaussianShell> sh=minbas.funcs(inuc);
    // Store indices
    for(size_t si=0;si<sh.size();si++)
      for(size_t fi=0;fi<sh[si].Nbf();fi++)
	idx[inuc].push_back(sh[si].first_ind()+fi);
  }

  // Calculate S1, S12, S2, and S21
  arma::mat S1(basis.overlap());
  arma::mat S2(minbas.overlap());

  arma::mat S12(basis.overlap(minbas));
  arma::mat S21(arma::trans(S12));

  // and inverse matrices
  arma::mat S1inv_h(BasOrth(S1));
  arma::mat S2inv_h(BasOrth(S2));
  // Need to be OK for canonical as well
  arma::mat S1inv(S1inv_h*arma::trans(S1inv_h));
  arma::mat S2inv(S2inv_h*arma::trans(S2inv_h));

  // Compute Ctilde.
  arma::Mat<T> Ctilde(S1inv*S12*S2inv*S21*C);

  // and orthonormalize it
  Ctilde=orthonormalize(S1,Ctilde);

  // "Density matrices"
  arma::Mat<T> P(C*arma::trans(C));
  arma::Mat<T> Pt(Ctilde*arma::trans(Ctilde));

  // Identity matrix
  arma::mat unit(S1.n_rows,S1.n_cols);
  unit.eye();

  // Compute the non-orthonormal IAOs.
  arma::Mat<T> A=P*S1*Pt*S12 + (unit-P*S1)*(unit-Pt*S1)*S1inv*S12;

  // and orthonormalize them
  return orthonormalize(S1,A);
}
arma::mat construct_IAO(const BasisSet & basis, const arma::mat & C, std::vector< std::vector<size_t> > & idx, bool verbose, std::string minbaslib) {
  return construct_IAO_wrk<double>(basis,C,idx,verbose,minbaslib);
}
arma::cx_mat construct_IAO(const BasisSet & basis, const arma::cx_mat & C, std::vector< std::vector<size_t> > & idx, bool verbose, std::string minbaslib) {
  return construct_IAO_wrk< std::complex<double> >(basis,C,idx,verbose,minbaslib);
}

arma::mat block_m(const arma::mat & F, const arma::ivec & mv) {
  arma::mat Fnew(F);
  Fnew.zeros();
  for(arma::sword m=0;m<=mv.max();m++) {
    if(m==0) {
      // Indices are
      arma::uvec idx(arma::find(mv==m));
      Fnew(idx,idx)=F(idx,idx);
    } else {
      // Indices for plus and minus values are
      arma::uvec pidx(arma::find(mv==m));
      arma::uvec nidx(arma::find(mv==-m));
      Fnew(pidx,pidx)=F(pidx,pidx);
      Fnew(nidx,nidx)=F(nidx,nidx);
    }
  }

  return Fnew;
}

arma::mat m_norm(const arma::mat & C, const arma::ivec & mv) {
  arma::mat osym(mv.max()-mv.min()+1,C.n_cols);
  for(arma::sword m=mv.min();m<=mv.max();m++) {
    arma::uvec idx(arma::find(mv==m));
    for(size_t io=0;io<C.n_cols;io++) {
      arma::vec cv(C.col(io));
      osym(m-mv.min(),io)=arma::norm(cv(idx),"fro");
    }
  }

  return osym;
}

arma::ivec m_classify(const arma::mat & C, const arma::ivec & mv) {
  // Orbital class
  arma::ivec oclass;
  if(C.n_cols == 0)
    return oclass;
  oclass.zeros(C.n_cols);

  // Get symmetries
  arma::mat osym(m_norm(C,mv));

  //osym.print("Orbital symmetry");

  // Maximum angular momentum is
  if(osym.n_rows%2 != 1) throw std::logic_error("Invalid number of rows!\n");
  int maxam((osym.n_rows-1)/2);

  for(size_t io=0;io<C.n_cols;io++) {
    arma::vec s(osym.col(io));

    // Get maximum
    arma::uword idx;
    s.max(idx);

    // This corresponds to the m value
    int m=idx;
    m-=maxam;

    oclass(io)=m;
  }

  return oclass;
}

std::vector< std::vector<size_t> > BasisSet::find_identical_nuclei() const {
  // Index list
  std::vector< std::vector<size_t> > ret;

  // Loop over nuclei
  for(size_t i=0;i<Nnuc();i++) {
    // Check that nucleus isn't BSSE
    nucleus_t nuc=nucleus(i);
    if(nuc.bsse)
      continue;

    // Get the shells on the nucleus
    std::vector<GaussianShell> shi=funcs(i);

    // Check if there something already on the list
    bool found=false;
    for(size_t j=0;j<ret.size();j++) {
      std::vector<GaussianShell> shj=funcs(ret[j][0]);

      // Check nuclear type
      if(symbol(i).compare(symbol(ret[j][0]))!=0)
	continue;
      // Check charge status
      if(nucleus(i).Q != nucleus(ret[j][0]).Q)
	continue;

      // Do comparison
      if(shi.size()!=shj.size())
	continue;
      else {

	bool same=true;
	for(size_t ii=0;ii<shi.size();ii++) {
	  // Check angular momentum
	  if(shi[ii].am()!=shj[ii].am()) {
	    same=false;
	    break;
	  }

	  // and exponents
	  std::vector<contr_t> lhc=shi[ii].contr();
	  std::vector<contr_t> rhc=shj[ii].contr();

	  if(lhc.size() != rhc.size()) {
	    same=false;
	    break;
	  }
	  for(size_t ic=0;ic<lhc.size();ic++) {
	    if(!(lhc[ic]==rhc[ic])) {
	      same=false;
	      break;
	    }
	  }

	  if(!same)
	    break;
	}

	if(same) {
	  // Found identical atom.
	  found=true;

	  // Add it to the list.
	  ret[j].push_back(i);
	}
      }
    }

    if(!found) {
      // Didn't find the atom, add it to the list.
      std::vector<size_t> tmp;
      tmp.push_back(i);

      ret.push_back(tmp);
    }
  }

  return ret;
}
