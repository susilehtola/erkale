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



#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
// For exceptions
#include <sstream>
#include <stdexcept>

#include "basis.h"
#include "elements.h"
#include "integrals.h"
#include "linalg.h"
#include "mathf.h"
#include "obara-saika.h"
#include "solidharmonics.h"
#include "stringutil.h"

// Debug LIBINT routines against Huzinaga integrals?
//#define LIBINTDEBUG

bool operator==(const nucleus_t & lhs, const nucleus_t & rhs) {
  return (lhs.ind == rhs.ind) && (lhs.r == rhs.r) && (lhs.Z == rhs.Z) && \
    (lhs.bsse == rhs.bsse) && (stricmp(lhs.symbol,rhs.symbol)==0);
}

bool operator==(const coords_t & lhs, const coords_t & rhs) {
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
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


double normsq(const coords_t & r) {
  return r.x*r.x + r.y*r.y + r.z*r.z;
}

double norm(const coords_t & r) {
  return sqrt(normsq(r));
}

bool operator<(const contr_t & lhs, const contr_t & rhs) {
  // Decreasing order of exponents.
  return lhs.z>rhs.z;
}

bool operator==(const contr_t & lhs, const contr_t & rhs) {
  //  return (lhs.z==rhs.z) && (lhs.c==rhs.c);

  // Since this also needs to work for saved and reloaded basis sets, we need to relax the comparison.
  const double tol=1e3*DBL_EPSILON;

  bool same=(fabs(lhs.z-rhs.z)<tol) && (fabs(lhs.c-rhs.c)<tol);

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
  // Construct shell of basis functions

  // Store contraction
  c=C;
  // Sort the contraction
  sort();

  // Set angular momentum
  am=amv;
  // Use spherical harmonics?
  uselm=lm;

  // If spherical harmonics are used, fill transformation matrix
  if(uselm)
    transmat=Ylm_transmat(am);
  else {
    // Do away with uninitialized value warnings in valgrind
    transmat=arma::mat(1,1);
    transmat(0,0)=1.0/0.0; // Initialize to NaN
  }

  // Compute necessary amount of Cartesians
  size_t Ncart=(am+1)*(am+2)/2;
  // Allocate memory
  cart.reserve(Ncart);
  cart.resize(Ncart);
  // Initialize the shells

  int n=0;
  for(int i=0; i<=am; i++) {
    int nx = am - i;
    for(int j=0; j<=i; j++) {
      int ny = i-j;
      int nz = j;

      cart[n].l=nx;
      cart[n].m=ny;
      cart[n].n=nz;
      cart[n].relnorm=1.0;
      n++;
    }
  }
}

GaussianShell::~GaussianShell() {
}

void GaussianShell::set_first_ind(size_t ind) {
  indstart=ind;
}

void GaussianShell::set_center(const coords_t & cenv, size_t cenindv) {
  cen=cenv;
  cenind=cenindv;
}

void GaussianShell::sort() {
  std::stable_sort(c.begin(),c.end());
}

void GaussianShell::convert_contraction() {
  // Convert contraction from contraction of normalized gaussians to
  // contraction of unnormalized gaussians.

  // Note - these refer to cartesian functions!
  double fac=pow(M_2_PI,0.75)*pow(2,am)/sqrt(doublefact(2*am-1));

  for(size_t i=0;i<c.size();i++)
    c[i].c*=fac*pow(c[i].z,am/2.0+0.75);
}

void GaussianShell::normalize() {
  // Normalize contraction of unnormalized primitives wrt first function on shell

  // Check for dummy shell
  if(c.size()==1 && c[0].z==0.0) {
    // Yes, this is a dummy.
    c[0].c=1.0;
    return;
  }

  double fact=0.0;

  // Calculate overlap of exponents
  for(size_t i=0;i<c.size();i++)
    for(size_t j=0;j<c.size();j++)
      fact+=c[i].c*c[j].c/pow(c[i].z+c[j].z,am+1.5);

  // Add constant part
  fact*=pow(M_PI,1.5)*doublefact(2*am-1)/pow(2.0,am);

  // The coefficients must be scaled by 1/sqrt(fact)
  fact=1.0/sqrt(fact);
  for(size_t i=0;i<c.size();i++)
    c[i].c*=fact;

  // FIXME: Do something more clever here.
  if(!uselm) {
    // Compute relative normalization factors
    for(size_t i=0;i<cart.size();i++)
      cart[i].relnorm=sqrt(doublefact(2*am-1)/(doublefact(2*cart[i].l-1)*doublefact(2*cart[i].m-1)*doublefact(2*cart[i].n-1)));
  } else {
    // Compute self-overlap
    arma::mat S=overlap(*this);
    // and scale coefficients.
    for(size_t i=0;i<cart.size();i++)
      cart[i].relnorm/=sqrt(S(0,0));
  }
}

void GaussianShell::coulomb_normalize() {
  // Normalize functions using Coulomb norm
  size_t Ncart=cart.size();
  size_t Nbf=get_Nbf();

  // Dummy shell
  GaussianShell dummy;
  dummy=dummyshell();

  // Compute ERI
  ERIWorker eri(get_am(),get_Ncontr());
  std::vector<double> eris(get_Nbf()*get_Nbf());
  eri.compute(this,&dummy,this,&dummy,eris);

  if(!uselm) {
    // Cartesian functions
    for(size_t i=0;i<Ncart;i++)
      cart[i].relnorm*=1.0/sqrt(eris[i*Nbf+i]);
  } else {
    // Spherical normalization, need to distribute
    // normalization coefficient among cartesians

    // Spherical ERI is
    // ERI = transmat * ERI_cart * trans(transmat)

    // FIXME - Do something more clever here
    // Check that all factors are the same
    int diff=0;
    for(size_t i=1;i<Nbf;i++)
      if(fabs(eris[i*Nbf+i]-eris[0])>1000*DBL_EPSILON*eris[0]) {
	printf("%e != %e, diff %e\n",eris[i*Nbf+i],eris[0],eris[i*Nbf+i]-eris[0]);
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
      cart[i].relnorm*=1.0/sqrt(eris[0]);
  }
}

std::vector<contr_t> GaussianShell::get_contr() const {
  return c;
}

std::vector<shellf_t> GaussianShell::get_cart() const {
  return cart;
}

std::vector<contr_t> GaussianShell::get_contr_normalized() const {
  // Returned array
  std::vector<contr_t> cn=c;

  // Note - these refer to cartesian functions!
  double fac=pow(M_2_PI,0.75)*pow(2,am)/sqrt(doublefact(2*am-1));

  // Convert coefficients to those of normalized primitives
  for(size_t i=0;i<cn.size();i++)
    cn[i].c/=fac*pow(cn[i].z,am/2.0+0.75);

  return cn;
}

size_t GaussianShell::get_Nbf() const {
  if(uselm)
    return get_Nlm();
  else
    return get_Ncart();
}

size_t GaussianShell::get_Nlm() const {
  return 2*am+1;
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
    for(size_t i=0;i<c.size();i++)
      val+=c[i].c*exp(-c[i].z*r*r);
    val*=pow(r,am);
  } while(fabs(val)>eps);

  // OK, now the range lies in the range [oldr,r]. Use binary search to refine
  double left=oldr, right=r;
  double middle=(left+right)/2.0;

  while(right-left>10*DBL_EPSILON*right) {
    // Compute middle of interval
    middle=(left+right)/2.0;

    // Compute value in the middle
    val=0.0;
    for(size_t i=0;i<c.size();i++)
      val+=c[i].c*exp(-c[i].z*middle*middle);
    val*=pow(middle,am);

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
  return uselm;
}

void GaussianShell::set_lm(bool lm) {
  uselm=lm;

  if(uselm)
    transmat=Ylm_transmat(am);
  else
    transmat=arma::mat();
}

arma::mat GaussianShell::get_trans() const {
  return transmat;
}

size_t GaussianShell::get_Ncart() const {
  return cart.size();
}

size_t GaussianShell::get_Ncontr() const {
  return c.size();
}

int GaussianShell::get_am() const {
  return am;
}

size_t GaussianShell::get_center_ind() const {
  //  return cen->ind;
  return cenind;
}

coords_t GaussianShell::get_center() const {
  return cen;
}

bool GaussianShell::operator<(const GaussianShell & rhs) const {
  // Sort first by nucleus
  if(cenind < rhs.cenind)
    return true;
  else if(cenind == rhs.cenind) {
    // Then by angular momentum
    if(am<rhs.am)
      return true;
    else if(am==rhs.am) {
      // Then by decreasing order of exponents
      if(c.size() && rhs.c.size())
	return c[0].z>rhs.c[0].z;
    }
  }

  return false;
}


bool GaussianShell::operator==(const GaussianShell & rhs) const {
  // Check first nucleus
  if(cenind != rhs.cenind) {
    //    fprintf(stderr,"Center indices differ!\n");
    return false;
  }

  // Then, angular momentum
  if(am!=rhs.am) {
    //    fprintf(stderr,"Angular momentum differs!\n");
    return false;
  }

  // Then, by exponents
  if(c.size() != rhs.c.size()) {
    //    fprintf(stderr,"Contraction size differs!\n");
    return false;
  }

  for(size_t i=0;i<c.size();i++) {
    if(!(c[i]==rhs.c[i])) {
      //      fprintf(stderr,"%i:th contraction differs!\n",(int) i+1);
      return false;
    }
  }

  return true;
}

size_t GaussianShell::get_first_ind() const {
  return indstart;
}

size_t GaussianShell::get_last_ind() const {
  return indstart+get_Nbf()-1;
}

void GaussianShell::print() const {

  printf("\t%c shell at nucleus %3i with with basis functions %4i-%-4i\n",shell_types[am],(int) (get_center_ind()+1),(int) get_first_ind()+1,(int) get_last_ind()+1);
  printf("\t\tCenter of shell is at % 0.4f % 0.4f % 0.4f Å.\n",cen.x/ANGSTROMINBOHR,cen.y/ANGSTROMINBOHR,cen.z/ANGSTROMINBOHR);

  // Get contraction of normalized primitives
  std::vector<contr_t> cn=get_contr_normalized();

  printf("\t\tExponential contraction is\n");
  printf("\t\t\tzeta\t\tprimitive coeff\ttotal coeff\n");
  for(size_t i=0;i<c.size();i++)
    printf("\t\t\t%e\t% e\t% e\n",c[i].z,cn[i].c,c[i].c);
  if(uselm) {
    printf("\t\tThe functions on this shell are:\n\t\t\t");
    for(int m=-am;m<=am;m++)
      printf(" (%i,%i)",am,m);
    printf("\n");
  } else {
    printf("\t\tThe functions on this shell are:\n\t\t\t");
    for(size_t i=0;i<cart.size();i++) {
      printf(" ");
      if(cart[i].l+cart[i].m+cart[i].n==0)
	printf("1");
      else {
	for(int j=0;j<cart[i].l;j++)
	  printf("x");
	for(int j=0;j<cart[i].m;j++)
	  printf("y");
	for(int j=0;j<cart[i].n;j++)
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
  // Evaluate basis functions at (x,y,z)

  // Compute coordinates relative to center
  double xrel=x-cen.x;
  double yrel=y-cen.y;
  double zrel=z-cen.z;

  double rrelsq=xrel*xrel+yrel*yrel+zrel*zrel;

  // Evaluate exponential factor
  double expfac=0;
  for(size_t i=0;i<c.size();i++)
    expfac+=c[i].c*exp(-c[i].z*rrelsq);

  // Power arrays, x^l, y^l, z^l
  double xr[am+1], yr[am+1], zr[am+1];

  xr[0]=1.0;
  yr[0]=1.0;
  zr[0]=1.0;

  if(am) {
    xr[1]=xrel;
    yr[1]=yrel;
    zr[1]=zrel;

    for(int i=2;i<=am;i++) {
      xr[i]=xr[i-1]*xrel;
      yr[i]=yr[i-1]*yrel;
      zr[i]=zr[i-1]*zrel;
    }
  }

  // Values of functions
  arma::vec ret(cart.size());

  // Loop over functions
  for(size_t i=0;i<cart.size();i++) {
    // Value of function at (x,y,z) is
    ret[i]=cart[i].relnorm*xr[cart[i].l]*yr[cart[i].m]*zr[cart[i].n]*expfac;
  }

  if(uselm)
    // Transform into spherical harmonics
    return transmat*ret;
  else
    return ret;
}

arma::mat GaussianShell::eval_grad(double x, double y, double z) const {
  // Evaluate gradients of functions at (x,y,z)

  // Compute coordinates relative to center
  double xrel=x-cen.x;
  double yrel=y-cen.y;
  double zrel=z-cen.z;

  double rrelsq=xrel*xrel+yrel*yrel+zrel*zrel;

  // Power arrays, x^l, y^l, z^l
  double xr[am+2], yr[am+2], zr[am+2];

  xr[0]=1.0;
  yr[0]=1.0;
  zr[0]=1.0;

  xr[1]=xrel;
  yr[1]=yrel;
  zr[1]=zrel;

  for(int i=2;i<=am+1;i++) {
    xr[i]=xr[i-1]*xrel;
    yr[i]=yr[i-1]*yrel;
    zr[i]=zr[i-1]*zrel;
  }

  // Gradient array, N_cart x 3
  arma::mat ret(cart.size(),3);
  ret.zeros();

  // Helper variables
  double tmp, expf;

  // Loop over functions
  for(size_t icart=0;icart<cart.size();icart++) {
    // Get types
    int l=cart[icart].l;
    int m=cart[icart].m;
    int n=cart[icart].n;

    // Loop over exponential contraction
    for(size_t iexp=0;iexp<c.size();iexp++) {
      // Contracted exponent
      expf=c[iexp].c*exp(-c[iexp].z*rrelsq);

      // x component of gradient:
      tmp=-2.0*c[iexp].z*xr[l+1];
      if(l>0)
	tmp+=l*xr[l-1];
      ret(icart,0)+=tmp*yr[m]*zr[n]*expf;

      // y component
      tmp=-2.0*c[iexp].z*yr[m+1];
      if(m>0)
	tmp+=m*yr[m-1];
      ret(icart,1)+=tmp*xr[l]*zr[n]*expf;

      // z component
      tmp=-2.0*c[iexp].z*zr[n+1];
      if(n>0)
	tmp+=n*zr[n-1];
      ret(icart,2)+=tmp*xr[l]*yr[m]*expf;
    }

    // Plug in normalization constant
    ret(icart,0)*=cart[icart].relnorm;
    ret(icart,1)*=cart[icart].relnorm;
    ret(icart,2)*=cart[icart].relnorm;
  }


  if(uselm)
    // Need to transform into spherical harmonics
    return transmat*ret;
  else
    return ret;
}

arma::vec GaussianShell::eval_lapl(double x, double y, double z) const {
  // Evaluate laplacian of basis functions at (x,y,z)

  // Compute coordinates relative to center
  double xrel=x-cen.x;
  double yrel=y-cen.y;
  double zrel=z-cen.z;

  double rrelsq=xrel*xrel+yrel*yrel+zrel*zrel;

  // Power arrays, x^l, y^l, z^l
  double xr[am+3], yr[am+3], zr[am+3];

  xr[0]=1.0;
  yr[0]=1.0;
  zr[0]=1.0;

  xr[1]=xrel;
  yr[1]=yrel;
  zr[1]=zrel;

  for(int i=2;i<=am+2;i++) {
    xr[i]=xr[i-1]*xrel;
    yr[i]=yr[i-1]*yrel;
    zr[i]=zr[i-1]*zrel;
  }

  // Values of laplacians of functions
  arma::vec ret(cart.size());
  ret.zeros();

  // Helper variables
  double tmp, expf;

  // Loop over functions
  for(size_t icart=0;icart<cart.size();icart++) {
    // Get types
    int l=cart[icart].l;
    int m=cart[icart].m;
    int n=cart[icart].n;

    // Loop over exponential contraction
    for(size_t iexp=0;iexp<c.size();iexp++) {
      // Contracted exponent
      expf=c[iexp].c*exp(-c[iexp].z*rrelsq);

      // Derivative wrt x
      tmp=4.0*c[iexp].z*c[iexp].z*xr[l+2]-2.0*(2*l+1)*c[iexp].z*xr[l];
      if(l>1)
	tmp+=l*(l-1)*xr[l-2];
      ret(icart)+=tmp*yr[m]*zr[n]*expf;

      // Derivative wrt y
      tmp=4.0*c[iexp].z*c[iexp].z*yr[m+2]-2.0*(2*m+1)*c[iexp].z*yr[m];
      if(m>1)
	tmp+=m*(m-1)*yr[m-2];
      ret(icart)+=tmp*xr[l]*zr[n]*expf;

      // Derivative wrt z
      tmp=4.0*c[iexp].z*c[iexp].z*zr[n+2]-2.0*(2*n+1)*c[iexp].z*zr[n];
      if(n>1)
	tmp+=n*(n-1)*zr[n-2];
      ret(icart)+=tmp*xr[l]*yr[m]*expf;
    }

    // Plug in normalization constant
    ret(icart)*=cart[icart].relnorm;
  }

  if(uselm)
    // Transform into spherical harmonics
    return transmat*ret;
  else
    return ret;
}

// Calculate overlaps between basis functions
arma::mat GaussianShell::overlap(const GaussianShell & rhs) const {

  // Overlap matrix
  arma::mat S(cart.size(),rhs.cart.size());
  S.zeros();

  // Coordinates
  double xa=cen.x;
  double ya=cen.y;
  double za=cen.z;

  double xb=rhs.cen.x;
  double yb=rhs.cen.y;
  double zb=rhs.cen.z;

#ifdef OBARASAIKA
  for(size_t ixl=0;ixl<c.size();ixl++)
    for(size_t ixr=0;ixr<rhs.c.size();ixr++)
      S+=c[ixl].c*rhs.c[ixr].c*overlap_int_os(xa,ya,za,c[ixl].z,cart,xb,yb,zb,rhs.c[ixr].z,rhs.cart);
#else
  // Loop over shells
  for(size_t icl=0;icl<cart.size();icl++)
    for(size_t icr=0;icr<rhs.cart.size();icr++) {
      // Angular momenta of shells
      int la=cart[icl].l;
      int ma=cart[icl].m;
      int na=cart[icl].n;

      int lb=rhs.cart[icr].l;
      int mb=rhs.cart[icr].m;
      int nb=rhs.cart[icr].n;

      // Helper variable
      double tmp=0.0;

      // Loop over exponents
      for(size_t ixl=0;ixl<zeta.size();ixl++)
	for(size_t ixr=0;ixr<rhs.zeta.size();ixr++)
	  tmp+=c[ixl]*rhs.c[ixr]*overlap_int(xa,ya,za,zeta[ixl],la,ma,na,xb,yb,zb,rhs.zeta[ixr],lb,mb,nb);

      // Set overlap
      S(icl,icr)=tmp*cart[icl].relnorm*rhs.cart[icr].relnorm;
    }
#endif

  // Transformation to spherical harmonics. Left side:
  if(uselm) {
    S=transmat*S;
  }
  // Right side
  if(rhs.uselm) {
    S=S*arma::trans(rhs.transmat);
  }

  return S;
}

// Calculate overlaps between basis functions
arma::mat GaussianShell::coulomb_overlap(const GaussianShell & rhs) const {

  // Number of functions on shells
  size_t Ni=get_Nbf();
  size_t Nj=rhs.get_Nbf();

  // Compute ERI
  GaussianShell dummy=dummyshell();
  std::vector<double> eris(get_Nbf()*rhs.get_Nbf());
  int maxam=std::max(get_am(),rhs.get_am());
  int maxcontr=std::max(get_Ncontr(),rhs.get_Ncontr());
  ERIWorker eri(maxam,maxcontr);
  eri.compute(this,&dummy,&rhs,&dummy,eris);
    
  // Fill overlap matrix
  arma::mat S(Ni,Nj);
  for(size_t i=0;i<Ni;i++)
    for(size_t j=0;j<Nj;j++)
      S(i,j)=eris[i*Nj+j];

  return S;
}

// Calculate kinetic energy matrix element between basis functions
arma::mat GaussianShell::kinetic(const GaussianShell & rhs) const {

  // Kinetic energy matrix
  arma::mat T(cart.size(),rhs.cart.size());
  T.zeros();

  // Coordinates
  double xa=cen.x;
  double ya=cen.y;
  double za=cen.z;

  double xb=rhs.cen.x;
  double yb=rhs.cen.y;
  double zb=rhs.cen.z;


#ifdef OBARASAIKA
  for(size_t ixl=0;ixl<c.size();ixl++)
    for(size_t ixr=0;ixr<rhs.c.size();ixr++)
      T+=c[ixl].c*rhs.c[ixr].c*kinetic_int_os(xa,ya,za,c[ixl].z,cart,xb,yb,zb,rhs.c[ixr].z,rhs.cart);

#else
  // Loop over shells
  for(size_t icl=0;icl<cart.size();icl++)
    for(size_t icr=0;icr<rhs.cart.size();icr++) {
      // Angular momenta of shells
      int la=cart[icl].l;
      int ma=cart[icl].m;
      int na=cart[icl].n;

      int lb=rhs.cart[icr].l;
      int mb=rhs.cart[icr].m;
      int nb=rhs.cart[icr].n;

      // Helper variable
      double tmp=0.0;

      // Loop over exponents
      for(size_t ixl=0;ixl<zeta.size();ixl++)
	for(size_t ixr=0;ixr<rhs.zeta.size();ixr++)
	  tmp+=c[ixl]*rhs.c[ixr]*kinetic_int(xa,ya,za,zeta[ixl],la,ma,na,xb,yb,zb,rhs.zeta[ixr],lb,mb,nb);

      // Set matrix element
      T(icl,icr)=tmp*cart[icl].relnorm*rhs.cart[icr].relnorm;
    }
#endif

  // Transformation to spherical harmonics. Left side:
  if(uselm) {
    T=transmat*T;
  }
  // Right side
  if(rhs.uselm) {
    T=T*arma::trans(rhs.transmat);
  }

  return T;
}


// Calculate nuclear attraction matrix element between basis functions
arma::mat GaussianShell::nuclear(double cx, double cy, double cz, const GaussianShell & rhs) const {

  // Matrix element of nuclear attraction operator
  arma::mat Vnuc(cart.size(),rhs.cart.size());
  Vnuc.zeros();

  // Coordinates
  double xa=cen.x;
  double ya=cen.y;
  double za=cen.z;

  double xb=rhs.cen.x;
  double yb=rhs.cen.y;
  double zb=rhs.cen.z;

#ifdef OBARASAIKA
  for(size_t ixl=0;ixl<c.size();ixl++)
    for(size_t ixr=0;ixr<rhs.c.size();ixr++)
      Vnuc+=c[ixl].c*rhs.c[ixr].c*nuclear_int_os(xa,ya,za,c[ixl].z,cart,cx,cy,cz,xb,yb,zb,rhs.c[ixr].z,rhs.cart);
#else

  // Loop over shells
  for(size_t icl=0;icl<cart.size();icl++) {
    // Angular momentum of shell
    int la=cart[icl].l;
    int ma=cart[icl].m;
    int na=cart[icl].n;

    for(size_t icr=0;icr<rhs.cart.size();icr++) {

      // Angular momentum of shell
      int lb=rhs.cart[icr].l;
      int mb=rhs.cart[icr].m;
      int nb=rhs.cart[icr].n;

      // Helper variable
      double tmp=0.0;

      // Loop over exponents
      for(size_t ixl=0;ixl<zeta.size();ixl++) {
	double ca=c[ixl];
	double zetaa=zeta[ixl];

	for(size_t ixr=0;ixr<rhs.zeta.size();ixr++) {
	  double cb=rhs.c[ixr];
	  double zetab=rhs.zeta[ixr];

	  tmp+=ca*cb*nuclear_int(xa,ya,za,zetaa,la,ma,na,cx,cy,cz,xb,yb,zb,zetab,lb,mb,nb);

	}
      }

      // Set matrix element
      Vnuc(icl,icr)=tmp*cart[icl].relnorm*rhs.cart[icr].relnorm;
    }
  }
#endif

  // Transformation to spherical harmonics. Left side:
  if(uselm) {
    Vnuc=transmat*Vnuc;
  }
  // Right side
  if(rhs.uselm) {
    Vnuc=Vnuc*arma::trans(rhs.transmat);
  }

  return Vnuc;
}

std::vector<arma::mat> GaussianShell::moment(int momam, double x, double y, double z, const GaussianShell & rhs) const {
  // Calculate moment integrals around (x,y,z) between shells

  // Amount of moments is
  size_t Nmom=(momam+1)*(momam+2)/2;

  // Moments to compute:
  std::vector<shellf_t> mom;
  mom.reserve(Nmom);
  for(int ii=0; ii<=momam; ii++) {
    int lc=momam - ii;
    for(int jj=0; jj<=ii; jj++) {
      int mc=ii - jj;
      int nc=jj;

      shellf_t tmp;
      tmp.l=lc;
      tmp.m=mc;
      tmp.n=nc;
      tmp.relnorm=1.0;
      mom.push_back(tmp);
    }
  }

  // Temporary array, place moment last so we can use slice()
  arma::cube wrk(cart.size(),rhs.cart.size(),Nmom);
  wrk.zeros();

  // Coordinates
  double xa=cen.x;
  double ya=cen.y;
  double za=cen.z;
  double xb=rhs.cen.x;
  double yb=rhs.cen.y;
  double zb=rhs.cen.z;

  // Compute moment integrals
  for(size_t ixl=0;ixl<c.size();ixl++) {
    double ca=c[ixl].c;
    double zetaa=c[ixl].z;

    for(size_t ixr=0;ixr<rhs.c.size();ixr++) {
      double cb=rhs.c[ixr].c;
      double zetab=rhs.c[ixr].z;

      wrk+=ca*cb*three_overlap_int_os(xa,ya,za,xb,yb,zb,x,y,z,zetaa,zetab,0.0,cart,rhs.cart,mom);
    }
  }

  // Collect the results
  std::vector<arma::mat> ret;
  ret.reserve(Nmom);
  for(size_t m=0;m<Nmom;m++) {
    // The matrix for this moment is
    arma::mat momval=wrk.slice(m);

    // Convert to spherical basis if necessary
    if(uselm) {
      momval=transmat*momval;
    }
    // Right side
    if(rhs.uselm) {
      momval=momval*arma::trans(rhs.transmat);
    }

    // Add it to the stack
    ret.push_back(momval);
  }

  return ret;
}

arma::vec GaussianShell::integral() const {
  // Compute integrals over the cartesian functions
  arma::vec ints(cart.size());
  ints.zeros();

  // Loop over cartesians
  for(size_t ic=0;ic<cart.size();ic++) {
    int l=cart[ic].l;
    int m=cart[ic].m;
    int n=cart[ic].n;

    if(l%2 || m%2 || n%2)
      // Odd function - zero integral
      continue;

    // Loop over exponents
    for(size_t ix=0;ix<c.size();ix++) {
      double zeta=c[ix].z;
      
      // Integral over x gives
      double intx=2.0*pow(0.5/sqrt(zeta),l+1)*sqrt(M_PI);
      // Integral over y
      double inty=2.0*pow(0.5/sqrt(zeta),m+1)*sqrt(M_PI);
      // Integral over z
      double intz=2.0*pow(0.5/sqrt(zeta),n+1)*sqrt(M_PI);

      // Increment total integral
      ints(ic)+=c[ix].c*intx*inty*intz;
    }

    // Plug in relative norm
    ints(ic)*=cart[ic].relnorm;
  }

  // Do conversion to spherical basis
  if(uselm)
    ints=transmat*ints;

  return ints;
}


BasisSet::BasisSet() {
  // Use spherical harmonics by default.
  uselm=1;
}

BasisSet::BasisSet(size_t Nat, const Settings & set) {
  // Use spherical harmonics?
  uselm=set.get_bool("UseLM");

  shells.reserve(Nat);
  nuclei.reserve(Nat);
}

BasisSet::~BasisSet() {
}

void BasisSet::add_nucleus(const nucleus_t & nuc) {
  nuclei.push_back(nuc);
  // Clear list of functions
  nuclei[nuclei.size()-1].shells.clear();
  // Set nuclear index
  nuclei[nuclei.size()-1].ind=nuclei.size()-1;
}

void BasisSet::add_shell(size_t nucind, const GaussianShell & sh, bool dosort) {
  if(nucind>=nuclei.size()) {
    ERROR_INFO();
    throw std::runtime_error("Cannot add functions to nonexisting nucleus!\n");
  }

  // Add shell
  shells.push_back(sh);
  // Set pointer to nucleus
  shells[shells.size()-1].set_center(nuclei[nucind].r,nucind);

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
    // Create shell
    GaussianShell sh;
    if(bf[i].get_am()>=2)
      sh=GaussianShell(bf[i].get_am(),uselm,bf[i].get_contr());
    else
      sh=GaussianShell(bf[i].get_am(),false,bf[i].get_contr());

    // and add it
    add_shell(nucind,sh,dosort);
  }
}

void BasisSet::check_numbering() {
  // Renumber basis functions
  size_t ind=0;
  for(size_t i=0;i<shells.size();i++) {
    shells[i].set_first_ind(ind);
    ind=shells[i].get_last_ind()+1;
  }
}

void BasisSet::update_nuclear_shell_list() {
  // First, clear the list on all nuclei.
  for(size_t inuc=0;inuc<nuclei.size();inuc++)
    nuclei[inuc].shells.clear();

  // Then, update the lists. Loop over shells
  for(size_t ish=0;ish<shells.size();ish++) {
    // Find out nuclear index
    size_t inuc=shells[ish].get_center_ind();
    // Add pointer to the nucleus
    nuclei[inuc].shells.push_back(&shells[ish]);
  }
}

void BasisSet::sort() {
  // Sort the shells first by increasing index of center, then by
  // increasing angular momentum and last by decreasing exponent.
  std::stable_sort(shells.begin(),shells.end());

  // Check the numbering of the basis functions
  check_numbering();

  // and since we probably have changed the order of the basis
  // functions, we need to update the list of functions on the nuclei.
  update_nuclear_shell_list();
}

void BasisSet::compute_nuclear_distances() {
  // Amount of nuclei
  size_t N=nuclei.size();

  // Reserve memory
  nucleardist=arma::mat(N,N);

  double d;

  // Fill table
  for(size_t i=0;i<N;i++)
    for(size_t j=0;j<=i;j++) {
      d=dist(nuclei[i].r.x,nuclei[i].r.y,nuclei[i].r.z,nuclei[j].r.x,nuclei[j].r.y,nuclei[j].r.z);

      nucleardist(i,j)=d;
      nucleardist(j,i)=d;
    }
}

double BasisSet::nuclear_distance(size_t i, size_t j) const {
  return nucleardist(i,j);
}

bool operator<(const shellpair_t & lhs, const shellpair_t & rhs) {
  // Helper for ordering shellpairs into libint order
  return (lhs.li+lhs.lj)<(rhs.li+rhs.lj);
}

void BasisSet::form_unique_shellpairs() {
  // Drop list of existing pairs.
  shellpairs.clear();

  // Form list of unique shell pairs.
  shellpair_t tmp;

  // Now, form list of unique shell pairs
  for(size_t i=0;i<shells.size();i++) {
    for(size_t j=0;j<=i;j++) {
      // Have to set these in every iteration due to swap below
      tmp.is=i;
      tmp.js=j;

      // Check that libint's angular momentum rules are satisfied
      if(shells[j].get_am()>shells[i].get_am())
	std::swap(tmp.is,tmp.js);

      // Set angular momenta
      tmp.li=shells[tmp.is].get_am();
      tmp.lj=shells[tmp.js].get_am();

      shellpairs.push_back(tmp);
    }
  }

  // Sort list of unique shell pairs
  stable_sort(shellpairs.begin(),shellpairs.end());

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

std::vector<shellpair_t> BasisSet::get_unique_shellpairs() const {
  if(shells.size() && !shellpairs.size()) {
    throw std::runtime_error("shellpairs not initialized! Maybe you forgot to finalize?\n");
  }

  return shellpairs;
}

void BasisSet::finalize(bool convert, bool donorm) {
  // Finalize basis set structure for use.

  // Compute nuclear distances.
  compute_nuclear_distances();

  // Compute ranges of shells
  compute_shell_ranges();

  // Convert contractions
  if(convert)
    convert_contractions();
  // Normalize contractions
  if(donorm)
    normalize();

  // Form list of unique shell pairs
  form_unique_shellpairs();
}

int BasisSet::get_am(size_t ind) const {
  return shells[ind].get_am();
}

int BasisSet::get_max_am() const {
  if(shells.size()==0) {
    ERROR_INFO();
    throw std::domain_error("Cannot get maximum angular momentum of an empty basis set!\n");
  }

  int maxam=shells[0].get_am();
  for(size_t i=1;i<shells.size();i++)
    if(shells[i].get_am()>maxam)
      maxam=shells[i].get_am();
  return maxam;
}

size_t BasisSet::get_max_Ncontr() const {
  size_t maxc=shells[0].get_Ncontr();
  for(size_t i=1;i<shells.size();i++)
    if(shells[i].get_Ncontr()>maxc)
      maxc=shells[i].get_Ncontr();
  return maxc;
}

size_t BasisSet::get_Nbf() const {
  if(shells.size())
    return shells[shells.size()-1].get_last_ind()+1;
  else
    return 0;
}

void BasisSet::compute_shell_ranges(double eps) {
  shell_ranges.reserve(shells.size());
  shell_ranges.resize(shells.size());
  for(size_t i=0;i<shells.size();i++)
    shell_ranges[i]=shells[i].range(eps);
}

std::vector<double> BasisSet::get_shell_ranges() const {
  return shell_ranges;
}

std::vector<double> BasisSet::get_nuclear_distances(size_t inuc) const {
  std::vector<double> d;
  d.reserve(nucleardist.n_cols);
  d.resize(nucleardist.n_cols);
  for(size_t i=0;i<nucleardist.n_cols;i++)
    d[i]=nucleardist(inuc,i);
  return d;
}

size_t BasisSet::get_Ncart() const {
  size_t n=0;
  for(size_t i=0;i<shells.size();i++)
    n+=shells[i].get_Ncart();
  return n;
}

size_t BasisSet::get_Nlm() const {
  size_t n=0;
  for(size_t i=0;i<shells.size();i++)
    n+=shells[i].get_Nlm();
  return n;
}

size_t BasisSet::get_Nbf(size_t ind) const {
  return shells[ind].get_Nbf();
}

size_t BasisSet::get_Ncart(size_t ind) const {
  return shells[ind].get_Ncart();
}

size_t BasisSet::get_last_ind() const {
  if(shells.size())
    return shells[shells.size()-1].get_last_ind();
  else {
    std::ostringstream oss;
    oss << "\nError in function " << __FUNCTION__ << "(file " << __FILE__ << ", near line " << __LINE__ << "\nCannot get number of last basis function of an empty basis set!\n";
    throw std::domain_error(oss.str());
  }
}

size_t BasisSet::get_first_ind(size_t num) const {
  return shells[num].get_first_ind();
}

size_t BasisSet::get_last_ind(size_t num) const {
  return shells[num].get_last_ind();
}

size_t BasisSet::get_center_ind(size_t num) const {
  return shells[num].get_center_ind();
}

std::vector<GaussianShell> BasisSet::get_shells() const {
  return shells;
}

std::vector<GaussianShell> BasisSet::get_shells(size_t inuc) const {
  std::vector<GaussianShell> ret;
  for(size_t ish=0;ish<shells.size();ish++)
    if(shells[ish].get_center_ind()==inuc)
      ret.push_back(shells[ish]);
  return ret;
}

GaussianShell BasisSet::get_shell(size_t ind) const {
  return shells[ind];
}

coords_t BasisSet::get_center(size_t num) const {
  return shells[num].get_center();
}

std::vector<contr_t> BasisSet::get_contr(size_t ind) const {
  return shells[ind].get_contr();
}

std::vector<contr_t> BasisSet::get_contr_normalized(size_t ind) const {
  return shells[ind].get_contr_normalized();
}

std::vector<shellf_t> BasisSet::get_cart(size_t ind) const {
  return shells[ind].get_cart();
}


bool BasisSet::is_lm_default() const {
  return uselm;
}

bool BasisSet::lm_in_use(size_t num) const {
  return shells[num].lm_in_use();
}

void BasisSet::set_lm(size_t num, bool lm) {
  // Set use of spherical harmonics
  shells[num].set_lm(lm);
  // Check numbering of basis functions which may have changed
  check_numbering();
}

arma::mat BasisSet::get_trans(size_t ind) const {
  return shells[ind].get_trans();
}

size_t BasisSet::get_Nshells() const {
  return shells.size();
}

size_t BasisSet::get_Nnuc() const {
  return nuclei.size();
}

nucleus_t BasisSet::get_nucleus(size_t inuc) const {
  return nuclei[inuc];
}

std::vector<nucleus_t> BasisSet::get_nuclei() const {
  return nuclei;
}

coords_t BasisSet::get_coords(size_t inuc) const {
  return nuclei[inuc].r;
}

int BasisSet::get_Z(size_t inuc) const {
  return nuclei[inuc].Z;
}

std::string BasisSet::get_symbol(size_t inuc) const {
  return nuclei[inuc].symbol;
}

std::vector<GaussianShell> BasisSet::get_funcs(size_t inuc) const {
  std::vector<GaussianShell> ret;
  for(size_t i=0;i<nuclei[inuc].shells.size();i++)
    ret.push_back(*(nuclei[inuc].shells[i]));

  return ret;
}

std::vector<size_t> BasisSet::get_shell_inds(size_t inuc) const {
  std::vector<size_t> ret;
  for(size_t i=0;i<shells.size();i++)
    if(shells[i].get_center_ind()==inuc)
      ret.push_back(i);
  return ret;
}

arma::vec BasisSet::eval_func(size_t ish, double x, double y, double z) const {
  return shells[ish].eval_func(x,y,z);
}

arma::mat BasisSet::eval_grad(size_t ish, double x, double y, double z) const {
  return shells[ish].eval_grad(x,y,z);
}

arma::vec BasisSet::eval_lapl(size_t ish, double x, double y, double z) const {
  return shells[ish].eval_lapl(x,y,z);
}

void BasisSet::convert_contractions() {
  for(size_t i=0;i<shells.size();i++)
    shells[i].convert_contraction();
}

void BasisSet::convert_contraction(size_t ind) {
  shells[ind].convert_contraction();
}

void BasisSet::normalize() {
  for(size_t i=0;i<shells.size();i++)
    shells[i].normalize();
}

void BasisSet::coulomb_normalize() {
  for(size_t i=0;i<shells.size();i++)
    shells[i].coulomb_normalize();
}

void BasisSet::print(bool verbose) const {
  printf("There are %i shells and %i nuclei in the basis set.\n\n",(int) shells.size(),(int) nuclei.size());

  printf("List of nuclei, geometry in Ångström with three decimal places:\n");

  printf("\t\t Z\t    x\t    y\t    z\n");
  for(size_t i=0;i<nuclei.size();i++) {
    if(nuclei[i].bsse)
      printf("%i\t%s\t*%i\t% 7.3f\t% 7.3f\t% 7.3f\n",(int) i+1,nuclei[i].symbol.c_str(),nuclei[i].Z,nuclei[i].r.x/ANGSTROMINBOHR,nuclei[i].r.y/ANGSTROMINBOHR,nuclei[i].r.z/ANGSTROMINBOHR);
    else
      printf("%i\t%s\t %i\t% 7.3f\t% 7.3f\t% 7.3f\n",(int) i+1,nuclei[i].symbol.c_str(),nuclei[i].Z,nuclei[i].r.x/ANGSTROMINBOHR,nuclei[i].r.y/ANGSTROMINBOHR,nuclei[i].r.z/ANGSTROMINBOHR);
  }
  printf("\nList of basis functions:\n");

  if(verbose) {
    for(size_t i=0;i<shells.size();i++) {
      printf("Shell %4i",(int) i);
      shells[i].print();
    }
  } else {
    for(size_t i=0;i<shells.size();i++) {
      // Type of shell - spherical harmonics or cartesians
      std::string type;
      if(shells[i].lm_in_use())
	type="sph";
      else
	type="cart";


      printf("Shell %4i",(int) i+1);
      printf("\t%c %4s shell at nucleus %3i with with basis functions %4i-%-4i\n",shell_types[shells[i].get_am()],type.c_str(),(int) (shells[i].get_center_ind()+1),(int) shells[i].get_first_ind()+1,(int) shells[i].get_last_ind()+1);
    }
  }


  printf("\nBasis set contains %i functions, maximum angular momentum is %i.\\
n",(int) get_Nbf(),get_max_am());
  if(is_lm_default())
    printf("Spherical harmonic Gaussians are used by default, there are %i cartesians.\n",(int) get_Ncart());
  else
    printf("Cartesian Gaussians are used by default.\n");
}

arma::mat BasisSet::cart_to_sph_trans() const {
  // Form transformation matrix to spherical harmonics

  const size_t Nlm=get_Nlm();
  const size_t Ncart=get_Ncart();

  // Returned matrix
  arma::mat trans(Nlm,Ncart);
  trans.zeros();

  // Bookkeeping indices
  size_t n=0, l=0;

  // Helper matrix
  arma::mat tmp;

  for(size_t i=0;i<shells.size();i++) {
    // Get angular momentum of shell
    int am=shells[i].get_am();

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

  // Size of basis set
  const size_t N=get_Nbf();

  // Initialize matrix
  arma::mat S(N,N);
  S.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs.size();ip++) {
    // Shells in pair
    size_t i=shellpairs[ip].is;
    size_t j=shellpairs[ip].js;

    // Get overlap between shells
    arma::mat tmp=shells[i].overlap(shells[j]);
    
    // Store overlap
    S.submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())=tmp;
    S.submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())=arma::trans(tmp);
  }

  return S;
}

arma::mat BasisSet::coulomb_overlap() const {
  // Form overlap matrix

  // Size of basis set
  const size_t N=get_Nbf();

  // Initialize matrix
  arma::mat S(N,N);
  S.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs.size();ip++) {
    // Shells in pair
    size_t i=shellpairs[ip].is;
    size_t j=shellpairs[ip].js;
    
    arma::mat tmp=shells[i].coulomb_overlap(shells[j]);
    
    // Store overlap
    S.submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())=tmp;
    S.submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())=arma::trans(tmp);
  }
  
  return S;
}

arma::mat BasisSet::overlap(const BasisSet & rhs) const {
  // Form overlap wrt to other basis set

  // Size of this basis set
  const size_t Nl=get_Nbf();
  // Size of rhs basis
  const size_t Nr=rhs.get_Nbf();

  // Initialize matrix
  arma::mat S12(Nl,Nr);
  S12.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<shells.size();i++) {
    for(size_t j=0;j<rhs.shells.size();j++) {
      S12.submat(shells[i].get_first_ind(),rhs.shells[j].get_first_ind(),
		 shells[i].get_last_ind() ,rhs.shells[j].get_last_ind() )=shells[i].overlap(rhs.shells[j]);;
    }
  }
  return S12;
}

arma::mat BasisSet::coulomb_overlap(const BasisSet & rhs) const {
  // Form overlap wrt to other basis set

  // Size of this basis set
  const size_t Nl=get_Nbf();
  // Size of rhs basis
  const size_t Nr=rhs.get_Nbf();

  // Initialize matrix
  arma::mat S12(Nl,Nr);
  S12.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<shells.size();i++) {
    for(size_t j=0;j<rhs.shells.size();j++) {
      S12.submat(shells[i].get_first_ind(),rhs.shells[j].get_first_ind(),
		 shells[i].get_last_ind() ,rhs.shells[j].get_last_ind() )=shells[i].coulomb_overlap(rhs.shells[j]);;
    }
  }
  return S12;
}


arma::mat BasisSet::kinetic() const {
  // Form kinetic energy matrix

  // Size of basis set
  size_t N=get_Nbf();

  // Initialize matrix
  arma::mat T(N,N);
  T.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs.size();ip++) {
    // Shells in pair
    size_t i=shellpairs[ip].is;
    size_t j=shellpairs[ip].js;
    
    // Get partial kinetic energy matrix
    arma::mat tmp=shells[i].kinetic(shells[j]);
    
    // Store result
    T.submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())=tmp;
    T.submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())=arma::trans(tmp);
  }
  
  return T;
}

arma::mat BasisSet::nuclear() const {
  // Form nuclear attraction matrix

  // Size of basis set
  size_t N=get_Nbf();

  // Initialize matrix
  arma::mat Vnuc(N,N);
  Vnuc.zeros();

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs.size();ip++)
    for(size_t inuc=0;inuc<nuclei.size();inuc++) {
      // If BSSE nucleus, do nothing
      if(nuclei[inuc].bsse)
	continue;

      // Nuclear charge
      int Z=nuclei[inuc].Z;

      // Coordinates of nucleus
      double cx=nuclei[inuc].r.x;
      double cy=nuclei[inuc].r.y;
      double cz=nuclei[inuc].r.z;

      // Shells in pair
      size_t i=shellpairs[ip].is;
      size_t j=shellpairs[ip].js;

      // Get subblock
      arma::mat tmp=Z*shells[i].nuclear(cx,cy,cz,shells[j]);

      // On the off diagonal we fill out both sides of the matrix
      if(i!=j) {
	Vnuc.submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())+=tmp;
	Vnuc.submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())+=arma::trans(tmp);
      } else
	// On the diagonal we just get it once
	Vnuc.submat(shells[i].get_first_ind(),shells[i].get_first_ind(),shells[i].get_last_ind(),shells[i].get_last_ind())+=arma::trans(tmp);
    }

  return Vnuc;
}

std::vector<arma::mat> BasisSet::moment(int mom, double x, double y, double z) const {
  // Compute moment integrals around (x,y,z);

  // Number of moments to compute is
  size_t Nmom=(mom+1)*(mom+2)/2;
  // Amount of basis functions is
  size_t Nbf=get_Nbf();

  // Returned array, holding the moment integrals
  std::vector<arma::mat> ret;
  ret.reserve(Nmom);

  // Initialize arrays
  for(size_t i=0;i<Nmom;i++) {
    ret.push_back(arma::mat(Nbf,Nbf));
    ret[i].zeros();
  }

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<shellpairs.size();ip++) {
    // Shells in pair
    size_t i=shellpairs[ip].is;
    size_t j=shellpairs[ip].js;


    // Compute moment integral over shells
    std::vector<arma::mat> ints=shells[i].moment(mom,x,y,z,shells[j]);

    // Store moments
    if(i!=j) {
      for(size_t m=0;m<Nmom;m++) {
	ret[m].submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())=ints[m];
	ret[m].submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())=arma::trans(ints[m]);
      }
    } else {
      for(size_t m=0;m<Nmom;m++)
	ret[m].submat(shells[i].get_first_ind(),shells[i].get_first_ind(),shells[i].get_last_ind(),shells[i].get_last_ind())=ints[m];
    }
  }

  return ret;
}

arma::vec BasisSet::integral() const {
  arma::vec ints(get_Nbf());
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t is=0;is<shells.size();is++)
    ints.subvec(shells[is].get_first_ind(),shells[is].get_last_ind())=shells[is].integral();

  return ints;
}

arma::cube three_overlap(const GaussianShell *is, const GaussianShell *js, const GaussianShell*ks) {
  // First, compute the cartesian integrals. Centers of shells
  coords_t icen=is->get_center();
  coords_t jcen=js->get_center();
  coords_t kcen=ks->get_center();

  // Cartesian functions
  std::vector<shellf_t> icart=is->get_cart();
  std::vector<shellf_t> jcart=js->get_cart();
  std::vector<shellf_t> kcart=ks->get_cart();

  // Exponential contractions
  std::vector<contr_t> icontr=is->get_contr();
  std::vector<contr_t> jcontr=js->get_contr();
  std::vector<contr_t> kcontr=ks->get_contr();

  // Cartesian integrals
  arma::cube cartint(icart.size(),jcart.size(),kcart.size());
  cartint.zeros();
  for(size_t ix=0;ix<icontr.size();ix++) {
    double iz=icontr[ix].z;
    double ic=icontr[ix].c;

    for(size_t jx=0;jx<jcontr.size();jx++) {
      double jz=jcontr[jx].z;
      double jc=jcontr[jx].c;

      for(size_t kx=0;kx<kcontr.size();kx++) {
	double kz=kcontr[kx].z;
	double kc=kcontr[kx].c;
	
	cartint+=ic*jc*kc*three_overlap_int_os(icen.x,icen.y,icen.z,jcen.x,jcen.y,jcen.z,kcen.x,kcen.y,kcen.z,iz,jz,kz,icart,jcart,kcart);
      }
    }
  }

  // Convert first two indices to spherical basis
  arma::cube twints(is->get_Nbf(),js->get_Nbf(),kcart.size());
  twints.zeros();
  for(size_t ik=0;ik<kcart.size();ik++) {
    // ({i}|{j}|ks) is
    arma::mat momval=cartint.slice(ik);
    
    // Do conversion
    if(is->lm_in_use())
      momval=is->get_trans()*momval;
    if(js->lm_in_use())
      momval=momval*arma::trans(js->get_trans());

    twints.slice(ik)=momval;
  }

  // Convert last index to spherical basis
  if(! ks->lm_in_use())
    return twints;

  // Transformation matrix
  arma::mat ktrans=ks->get_trans();

  // Final integrals
  arma::cube ints(is->get_Nbf(),js->get_Nbf(),ks->get_Nbf());
  ints.zeros();

  // Loop over spherical basis functions
  for(size_t ik=0;ik<ks->get_Nbf();ik++)
    // Loop over cartesians
    for(size_t ick=0;ick<kcart.size();ick++)
      ints.slice(ik)+=ktrans(ik,ick)*twints.slice(ick);
  
  return ints;
}


int BasisSet::Ztot() const {
  int Zt=0;
  for(size_t i=0;i<nuclei.size();i++) {
    if(nuclei[i].bsse)
      continue;
    Zt+=nuclei[i].Z;
  }
  return Zt;
}

double BasisSet::Enuc() const {
  double En=0.0;

  for(size_t i=0;i<nuclei.size();i++) {
    if(nuclei[i].bsse)
      continue;

    int Zi=nuclei[i].Z;

    for(size_t j=0;j<i;j++) {
      if(nuclei[j].bsse)
	continue;
      int Zj=nuclei[j].Z;

      En+=Zi*Zj/nucleardist(i,j);
    }
  }

  return En;
}

void BasisSet::projectMOs(const BasisSet & oldbas, const arma::colvec & oldE, const arma::mat & oldMOs, arma::colvec & E, arma::mat & MOs) const {
  // Project MOs from old basis to new basis (this one)

  // Get number of basis functions
  const size_t Nbf=get_Nbf();
  // Cutoff
  const double cutoff=LINTHRES;

  // Get overlap matrix
  arma::mat S11=overlap();

  // and form orthogonalizing matrix
  arma::mat Svec;
  arma::vec Sval;
  eig_sym_ordered(Sval,Svec,S11);

  // Count number of eigenvalues that are above cutoff
  size_t Nind=0;
  for(size_t i=0;i<Nbf;i++)
    if(Sval(i)>=cutoff)
      Nind++;
  // Number of linearly dependent basis functions
  const size_t Ndep=Nbf-Nind;
  // Get rid of linearly dependent eigenvalues and eigenvectors
  Sval=Sval.subvec(Ndep,Nbf-1);
  Svec=Svec.submat(0,Ndep,Nbf-1,Nbf-1);

  // Form canonical orthonormalization matrix
  arma::mat Sinvh(Nbf,Nind);
  for(size_t i=0;i<Nind;i++)
    Sinvh.col(i)=Svec.col(i)/sqrt(Sval(i));

  // and the real S^-1
  arma::mat Sinv=Sinvh*arma::trans(Sinvh);

  // Get overlap with old basis
  arma::mat S12=overlap(oldbas);

  // Linearly independent size of old basis set
  const size_t Nbfo=oldMOs.n_cols;

  if(Nbfo==0)
    throw std::runtime_error("No orbitals to project!\n");

  // How many MOs do we transform?
  size_t Nmo=Nbfo;
  if(Nind<Nmo) // New basis is smaller than old one
    Nmo=Nind;

  // OK, now we are ready to calculate the projections.

  // Initialize MO matrix
  MOs=arma::mat(Sinvh.n_rows,Nind);
  MOs.zeros();
  // and energy
  E=arma::colvec(Nind);
  // and fill them
  for(size_t i=0;i<Nmo;i++) {

    // We have the orbitals as |a> = \sum_n c_n |b_n>, where |b_n> are
    // the basis functions of the old basis.

    // We project the old orbitals with the (approximate identity)
    // operator \sum_N |B_N> S^-1 <B_N|, where {B_N} are the basis
    // functions of the new basis.

    // This gives
    // \sum_N |B_N> S^-1 <B_N| [\sum_n c_n |b_n>]
    // = \sum_N |B_N> S^-1 \sum_n <B_N|b_n> c_n

    MOs.col(i)=Sinv*S12*oldMOs.col(i);
    E(i)=oldE(i);
  }

  // Assure that orbitals are orthonormal
  for(size_t i=0;i<Nmo;i++) {
    // Remove overlap with other orbitals
    for(size_t j=0;j<i;j++) {
      MOs.col(i)-=arma::as_scalar(arma::trans(MOs.col(j))*S11*MOs.col(i))*MOs.col(j);
    }
    // Calculate norm
    double norm=sqrt(arma::as_scalar(arma::trans(MOs.col(i))*S11*MOs.col(i)));
    // Normalize
    MOs.col(i)/=norm;
  }

  // If the old basis had less functions than the new basis, then we
  // need to form the rest of the orbitals. To do this we consider all
  // of the linearly independent eigenvectors of the new basis, and
  // remove the Nmo with the maximum absolute overlap with the
  // MOs. The leftovers are then orthonormalized with respect to the
  // MOs.

  if(Nmo<Nind) {
    // Normalize the eigenvectors
    for(size_t i=0;i<Nind;i++)
      Svec.col(i)/=sqrt(Sval(i));

    // Index vector
    std::vector<size_t> idx(Nind);
    for(size_t i=0;i<Nind;i++)
      idx[i]=i;

    // Remove Nmo eigenvectors with largest overlap
    for(size_t io=0;io<Nmo;io++) {
      // Helper
      arma::vec hlp=S11*MOs.col(io);

      // Compute overlap
      double maxovl=0.0;
      size_t indmax=0;
      for(size_t i=0;i<idx.size();i++) {
	double ovl=fabs(arma::dot(Svec.col(idx[i]),hlp));
	if(ovl>maxovl) {
	  maxovl=ovl;
	  indmax=i;
	}
      }

      // Remove vector with maximum overlap
      idx.erase(idx.begin()+indmax);
    }

    // Set the remaining orbitals
    for(size_t io=0;io<idx.size();io++) {
      MOs.col(Nmo+io)=Svec.col(idx[io]);
      E(Nmo+io)=E(Nmo-1);
    }

    // Reorthogonalize the new functions against the projected
    // orbitals
    for(size_t i=Nmo;i<Nind;i++) {
      // Remove overlap with other orbitals
      for(size_t j=0;j<i;j++)
	MOs.col(i)-=arma::as_scalar(arma::trans(MOs.col(j))*S11*MOs.col(i))*MOs.col(j);
      // Calculate norm
      double norm=sqrt(arma::as_scalar(arma::trans(MOs.col(i))*S11*MOs.col(i)));
      // Normalize
      MOs.col(i)/=norm;
    }
  }

  // Failsafe
  try {
    // Check orthogonality of orbitals
    check_orth(MOs,S11,false);
  } catch(std::runtime_error err) {
    std::ostringstream oss;
    oss << "Orbitals generated by projectMOs are not orthonormal. Please try without projection.\n";
    throw std::runtime_error(oss.str());
  }
}

bool exponent_compare(const GaussianShell & lhs, const GaussianShell & rhs) {
  return lhs.get_contr()[0].z>rhs.get_contr()[0].z;
}

BasisSet BasisSet::density_fitting(double fsam, int lmaxinc) const {
  // Automatically generate density fitting basis.

  // R. Yang, A. P. Rendell and M. J. Frisch, "Automatically generated
  // Coulomb fitting basis sets: Design and accuracy for systems
  // containing H to Kr", J. Chem. Phys. 127 (2007), 074102

  Settings set;
  set.add_scf_settings();
  set.add_dft_settings();
  // Density fitting basis set
  BasisSet dfit(1,set);

  // Loop over nuclei
  for(size_t in=0;in<nuclei.size();in++) {
    // Add nucleus to fitting set
    dfit.add_nucleus(nuclei[in]);
    // Dummy nucleus
    nucleus_t nuc=nuclei[in];

    // Define lval - (1) in YRF
    int lval;
    if(nuclei[in].Z<3)
      lval=0;
    else if(nuclei[in].Z<19)
      lval=1;
    else if(nuclei[in].Z<55)
      lval=2;
    else
      lval=3;

    // Get shells corresponding to this nucleus
    std::vector<GaussianShell> shs=get_funcs(in);

    // Form candidate set - (2), (3) and (6) in YRF
    std::vector<GaussianShell> cand;
    for(size_t i=0;i<shs.size();i++) {
      // Get angular momentum
      int am=2*shs[i].get_am();
      // Get exponents
      std::vector<contr_t> contr=shs[i].get_contr();

      // Dummy contraction
      std::vector<contr_t> C(1);
      C[0].c=1.0;

      for(size_t j=0;j<contr.size();j++) {
	// Set exponent
	C[0].z=2.0*contr[j].z;

	// Check that candidate set doesn't already contain the same function
	bool found=0;
	for(size_t k=0;k<cand.size();k++)
	  if((cand[k].get_am()==am) && (cand[k].get_contr()[0]==C[0])) {
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
      if(shs[i].get_am()>lmax_obs)
	lmax_obs=shs[i].get_am();
    int lmax_abs=std::max(lmax_obs+lmaxinc,2*lval);

    // (6) was already above.

    while(cand.size()>0) {
      // Generate trial set
      std::vector<GaussianShell> trial;

      // Function with largest exponent is moved to the trial set and
      // its exponent is set as the reference value - (7) in YRF
      double ref=(cand[0].get_contr())[0].z;
      trial.push_back(cand[0]);
      cand.erase(cand.begin());

      if(cand.size()>0) {
	// More functions remaining, move all for which ratio of
	// reference to exponent is smaller than fsam - (8) in YRF
	for(size_t i=cand.size()-1;i<cand.size();i--)
	  if(ref/((cand[i].get_contr())[0].z)<fsam) {
	    trial.push_back(cand[i]);
	    cand.erase(cand.begin()+i);
	  }

	// Compute geometric average of exponents - (9) in YRF
	double geomav=1.0;
	for(size_t i=0;i<trial.size();i++)
	  geomav*=trial[i].get_contr()[0].z;
	geomav=pow(geomav,1.0/trial.size());

	//	printf("Geometric average of %i functions is %e.\n",(int) trial.size(),geomav);

	// Form list of angular momentum values
	// Compute maximum angular moment of current trial set
	int ltrial=0;
	for(size_t i=0;i<trial.size();i++)
	  if(trial[i].get_am()>ltrial)
	    ltrial=trial[i].get_am();

	// If this is larger than allowed, renormalize
	if(ltrial>lmax_abs)
	  ltrial=lmax_abs;

	// Form list of angular momentum already used in ABS
	std::vector<int> lvals(max_am+1);
	for(int i=0;i<=max_am;i++)
	  lvals[i]=0;

	// Maximum angular momentum of trial functions is
	lvals[ltrial]++;
	// Get shells on current center
	std::vector<GaussianShell> cur_shells=dfit.get_funcs(in);
	for(size_t i=0;i<cur_shells.size();i++)
	  lvals[cur_shells[i].get_am()]++;

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
	for(int l=0;l<=max_am;l++)
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

  Settings set;
  BasisSet fit(nuclei.size(),set);

  const int maxam=get_max_am();

  // Loop over nuclei
  for(size_t in=0;in<nuclei.size();in++) {
    // Get shells corresponding to this nucleus
    std::vector<GaussianShell> shs=get_funcs(in);

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
	int l=shs[ish].get_am()+shs[jsh].get_am();

	// Update maximum value
	if(l>lmax)
	  lmax=l;

	// Increase amount of functions
	nfunc[l]++;

	// Get exponential contractions
	std::vector<contr_t> icontr=shs[ish].get_contr();
	std::vector<contr_t> jcontr=shs[jsh].get_contr();

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

bool BasisSet::operator==(const BasisSet & rhs) const {
  if(nuclei.size() != rhs.nuclei.size())
    return false;

  for(size_t i=0;i<nuclei.size();i++)
    if(!(nuclei[i]==rhs.nuclei[i])) {
      //      fprintf(stderr,"Nuclei %i differ!\n",(int) i);
      return false;
    }

  if(shells.size() != rhs.shells.size())
    return false;

  for(size_t i=0;i<shells.size();i++)
    if(!(shells[i]==rhs.shells[i])) {
      //      fprintf(stderr,"Shells %i differ!\n",(int) i);
      return false;
    }

  return true;
}

BasisSet BasisSet::decontract(arma::mat & m) const {
  // Decontract basis set. m maps old basis functions to new ones
  
  // Contraction schemes for the nuclei
  std::vector< std::vector<arma::mat> > coeffs(nuclei.size());
  std::vector< std::vector< std::vector<double> > > exps(nuclei.size());
  // Is puream used on the shell?
  std::vector< std::vector<bool> > puream(nuclei.size());

  // Amount of new basis functions
  size_t Nbfnew=0;

  // Collect the schemes. Loop over the nuclei.
  for(size_t inuc=0;inuc<nuclei.size();inuc++) {
    // Construct an elemental basis set for the nucleus
    ElementBasisSet elbas(get_symbol(inuc));

    // Get the shells belonging to this nucleus
    std::vector<GaussianShell> shs=get_shells(inuc);

    // and add the contractions to the elemental basis set
    for(size_t ish=0;ish<shs.size();ish++) {
      // Angular momentum is
      int am=shs[ish].get_am();
      // Normalized contraction coefficients
      std::vector<contr_t> c=shs[ish].get_contr_normalized();
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
	  if(shs[ish].get_am()!=am)
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
      std::vector<double> z;
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
  m.zeros(Nbfnew,get_Nbf());

  // Add the nuclei
  for(size_t i=0;i<nuclei.size();i++)
    dec.add_nucleus(nuclei[i]);

  // and the shells.
  for(size_t inuc=0;inuc<nuclei.size();inuc++) {
    // Get the shells belonging to this nucleus
    std::vector<GaussianShell> shs=get_shells(inuc);

    // Generate the new basis functions. Loop over am
    for(int am=0;am<(int) coeffs[inuc].size();am++) {
      // First functions with the exponents are
      std::vector<size_t> ind0;

      // Add the new shells
      for(size_t iz=0;iz<exps[inuc][am].size();iz++) {
	// Index of first function is
	ind0.push_back(dec.get_Nbf());
	// Add the shell
	std::vector<contr_t> hlp(1);
	hlp[0].c=1.0;
	hlp[0].z=exps[inuc][am][iz];
	dec.add_shell(inuc,am,puream[inuc][am],hlp,false);
      }

      // and store the coefficients
      for(size_t ish=0;ish<shs.size();ish++)
	if(shs[ish].get_am()==am) {
	  // Get the normalized contraction on the shell
	  std::vector<contr_t> ct=shs[ish].get_contr_normalized();
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
	    for(size_t ibf=0;ibf<shs[ish].get_Nbf();ibf++)
	      m(ind0[ix]+ibf,shs[ish].get_first_ind()+ibf)=ct[ic].c;
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

BasisSet construct_basis(const std::vector<nucleus_t> & nuclei, const BasisSetLibrary & baslib, const Settings & set) {
  std::vector<atom_t> atoms(nuclei.size());
  for(size_t i=0;i<nuclei.size();i++) {
    atoms[i].x=nuclei[i].r.x;
    atoms[i].y=nuclei[i].r.y;
    atoms[i].z=nuclei[i].r.z;
    atoms[i].num=nuclei[i].ind;
    atoms[i].el=nuclei[i].symbol;
  }

  return construct_basis(atoms,baslib,set);
}

BasisSet construct_basis(const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib, const Settings & set) {
  // Number of atoms is
  size_t Nat=atoms.size();

  // Indices of atoms to decontract basis set for
  std::vector<size_t> dec;
  bool decall=false;
  if(stricmp(set.get_string("Decontract"),"")!=0) {
    // Check for '*'
    std::string str=set.get_string("Decontract");
    if(str.size()==1 && str[0]=='*')
      decall=true;
    else
      dec=parse_range(set.get_string("Decontract"));
  }
  // Convert to C++ indexing
  for(size_t i=0;i<dec.size();i++)
    dec[i]--;

  // Create basis set
  BasisSet basis(Nat,set);
  // and add atoms to basis set
  for(size_t i=0;i<Nat;i++) {
    // First we need to add the nucleus itself.
    nucleus_t nuc;

    // Get center
    nuc.r.x=atoms[i].x;
    nuc.r.y=atoms[i].y;
    nuc.r.z=atoms[i].z;

    // Get symbol in raw form
    std::string el=atoms[i].el;

    // Determine if nucleus is BSSE or not
    nuc.bsse=0;
    if(el.size()>3 && el.substr(el.size()-3,3)=="-Bq") {
      // Yes, this is a BSSE nucleus
      nuc.bsse=1;
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
    } catch(std::runtime_error err) {
      // Did not find a special basis, use the general one instead.
      elbas=baslib.get_element(el,0);
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

    basis.add_shells(i,elbas);
  }

  // Finalize basis set and convert contractions
  basis.finalize(true);

  return basis;
}



std::vector<double> compute_orbitals(const arma::mat & C, const BasisSet & bas, const coords_t & r) {
  // Get ranges of shells
  std::vector<double> shran=bas.get_shell_ranges();

  // Indices of shells to compute
  std::vector<size_t> compute_shells;

  // Determine which shells might contribute
  for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
    // Determine distance of nucleus
    double dist=norm(r-bas.get_coords(inuc));
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=bas.get_shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++) {
      // Shell is relevant if range is larger than minimal distance
      if(dist<shran[shellinds[ish]]) {
        // Add shell to list of shells to compute
        compute_shells.push_back(shellinds[ish]);
      }
    }
  }

  // Values of orbitals
  std::vector<double> orbs(C.n_cols);
  for(size_t io=0;io<C.n_cols;io++)
    orbs[io]=0.0;

  // Loop over shells
  for(size_t ish=0;ish<compute_shells.size();ish++) {
    // Index of first function on shell is
    size_t ind0=bas.get_first_ind(compute_shells[ish]);
    // Compute values of basis functions on shell
    arma::vec fval=bas.eval_func(compute_shells[ish],r.x,r.y,r.z);

    // Loop over orbitals
    for(size_t io=0;io<C.n_cols;io++)
      // Loop over functions
      for(size_t ibf=0;ibf<fval.n_elem;ibf++)
        orbs[io]+=C(ind0+ibf,io)*fval(ibf);
  }

  return orbs;
}

double compute_density(const arma::mat & P, const BasisSet & bas, const coords_t & r) {
  // Get ranges of shells
  std::vector<double> shran=bas.get_shell_ranges();

  // Indices of shells to compute
  std::vector<size_t> compute_shells;

  // Determine which shells might contribute
  for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
    // Determine distance of nucleus
    double dist=norm(r-bas.get_coords(inuc));
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=bas.get_shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++) {
      // Shell is relevant if range is larger than minimal distance
      if(dist<shran[shellinds[ish]]) {
        // Add shell to list of shells to compute
        compute_shells.push_back(shellinds[ish]);
      }
    }
  }

  // Compute necessary function values
  std::vector<double> f; // Value of function
  std::vector<size_t> ind; // Index of function

  // Loop over shells
  for(size_t ish=0;ish<compute_shells.size();ish++) {
    // Compute values of functions on this shell
    arma::vec fval=bas.eval_func(compute_shells[ish],r.x,r.y,r.z);
    // Index of first function on shell is
    size_t i0=bas.get_first_ind(compute_shells[ish]);

    // Store values and indices
    for(size_t ibf=0;ibf<fval.n_elem;ibf++) {
      f.push_back(fval(ibf));
      ind.push_back(i0+ibf);
    }
  }

  // Value of density at point
  double dens=0.0;

  // Loop over function values
  for(size_t ii=0;ii<f.size();ii++) {
    // Do off-diagonal first
    for(size_t jj=0;jj<ii;jj++)
      dens+=2.0*P(ind[ii],ind[jj])*f[ii]*f[jj];
    // and then diagonal
    dens+=P(ind[ii],ind[ii])*f[ii]*f[ii];
  }

  return dens;
}


std::vector< std::vector<size_t> > BasisSet::find_identical_shells() const {
  // Returned list of identical basis functions
  std::vector< std::vector<size_t> > ret;

  // Loop over shells
  for(size_t ish=0;ish<shells.size();ish++) {
    // Get exponents, contractions and cartesian functions on shell
    std::vector<contr_t> shell_contr=shells[ish].get_contr();
    std::vector<shellf_t> shell_cart=shells[ish].get_cart();

    // Try to find the shell on the current list of identicals
    bool found=0;
    for(size_t iident=0;iident<ret.size();iident++) {

      // Check first cartesian part.
      std::vector<shellf_t> cmp_cart=shells[ret[iident][0]].get_cart();

      if(shell_cart.size()==cmp_cart.size()) {
	// Default value
	found=1;

	for(size_t icart=0;icart<shell_cart.size();icart++)
	  if(shell_cart[icart].l!=cmp_cart[icart].l || shell_cart[icart].m!=cmp_cart[icart].m || shell_cart[icart].n!=cmp_cart[icart].n)
	    found=0;

	// Check that usage of spherical harmonics matches, too
	if(shells[ish].lm_in_use() != shells[ret[iident][0]].lm_in_use())
	  found=0;

	// If cartesian parts match, check also exponents and contraction coefficients
	if(found) {
	  // Get exponents
	  std::vector<contr_t> cmp_contr=shells[ret[iident][0]].get_contr();

	  // Check exponents
	  if(shell_contr.size()==cmp_contr.size()) {
	    for(size_t ic=0;ic<shell_contr.size();ic++)
	      if(!(shell_contr[ic]==cmp_contr[ic]))
		found=0;
	  } else
	    found=0;
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

double check_orth(const arma::mat & C, const arma::mat & S, bool verbose) {
  double maxerr=0.0;
  size_t maxi=0, maxj=0;
  arma::mat MOovl=arma::trans(C)*S*C;
  for(size_t i=0;i<MOovl.n_cols;i++) {
    for(size_t j=0;j<i;j++)
      if(fabs(MOovl(i,j))>maxerr) {
        maxerr=fabs(MOovl(i,j));
        maxi=i;
        maxj=j;
      }
    if(fabs(MOovl(i,i)-1)>maxerr) {
      maxerr=fabs(MOovl(i,i)-1);
      maxi=i;
      maxj=i;
    }
  }
  if(verbose) {
    printf("Maximum deviation from orthogonality is %e, occurring at %i %i.\n",maxerr,(int) maxi, (int) maxj);
    fflush(stdout);
  }

  if(maxerr>=1e-8) {
    std::ostringstream oss;
    oss << "Generated orbitals are not orthonormal! Maximum deviation from orthonormality at " << maxi+1 << "," << maxj+1 <<": " << maxerr <<".\nCheck the used LAPACK implementation.\n";
    throw std::runtime_error(oss.str());
  }

  return maxerr;
}
