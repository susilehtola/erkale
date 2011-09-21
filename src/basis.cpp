/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
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

// Debug LIBINT routines against Huzinaga integrals?
//#define LIBINTDEBUG

// Operators for computing displacements
coords_t operator-(const coords_t & lhs, const coords_t& rhs) {
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
  return (lhs.z==rhs.z) && (lhs.c==rhs.c);
}

GaussianShell::GaussianShell(bool lm) {
  // Dummy constructor
  indstart=(size_t) -1;
  am=-1;
  uselm=lm;
}

GaussianShell::GaussianShell(size_t indstartv, int amv, bool lm, int atindv, coords_t cenv, const std::vector<contr_t> & cv) {
  // Construct shell of basis functions

  indstart=indstartv;

  atind=atindv;
  cen=cenv;

  // Store contraction
  c=cv;
  std::sort(c.begin(),c.end());

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

void GaussianShell::convert_contraction() {
  // Convert contraction from contraction of normalized gaussians to
  // contraction of unnormalized gaussians.

  double fac=pow(M_2_PI,0.75)*pow(2,am)/sqrt(doublefact(2*am-1));

  for(size_t i=0;i<c.size();i++)
    c[i].c*=fac*pow(c[i].z,am/2.0+0.75);
}

void GaussianShell::normalize() {
  // Normalize contraction of unnormalized primitives wrt first function on shell
  
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

  // Compute relative normalization factors
  for(size_t i=0;i<cart.size();i++)
    cart[i].relnorm=sqrt(doublefact(2*am-1)/(doublefact(2*cart[i].l-1)*doublefact(2*cart[i].m-1)*doublefact(2*cart[i].n-1)));
}

void GaussianShell::coulomb_normalize() {
  // Normalize functions using Coulomb norm

  std::vector<double> eris;
  size_t Ncart=cart.size();
  size_t Nbf=get_Nbf();

  // Dummy shell
  coords_t cen={0.0, 0.0, 0.0};
  std::vector<contr_t> C(1);
  C[0].c=1.0;
  C[0].z=0.0;
	      
  GaussianShell dummyshell(0,0,0,0,cen,C);

  // Compute ERI
  eris=ERI(this,&dummyshell,this,&dummyshell);

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

  /*
  eris=ERI(this,&dummyshell,this,&dummyshell);
  printf("After normalization\n");
  for(size_t i=0;i<Nbf;i++) {
    for(size_t j=0;j<Nbf;j++) 
      printf(" % e",eris[i*Nbf+j]);
    printf("\n");
  }
  */
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

size_t GaussianShell::get_inuc() const {
  return atind;
}

coords_t GaussianShell::get_coords() const {
  return cen;
}

bool GaussianShell::operator<(const GaussianShell & rhs) const {
  if(atind<rhs.atind)
    return 1;
  else if(atind==rhs.atind) {
    if(am<rhs.am)
      return 1;
    else if(am==rhs.am) {
      // Decreasing order of exponents.
      if(c.size() && rhs.c.size())
	return c[0].z>rhs.c[0].z;
    }
  }

  return 0;
}

size_t GaussianShell::get_first_ind() const {
  return indstart;
}

size_t GaussianShell::get_last_ind() const {
  return indstart+get_Nbf()-1;
}

void GaussianShell::set_first_ind(size_t ind) {
  indstart=ind;
}

void GaussianShell::set_center_ind(size_t inuc) {
  atind=inuc;
}

void GaussianShell::print() const {

  printf("\t%c shell at nucleus %i with with basis functions %4i-%-4i\n",shell_types[am],(int) atind+1,(int) get_first_ind()+1,(int) get_last_ind()+1);
  printf("\t\tCenter of shell is at % 0.4f % 0.4f % 0.4f Å.\n",cen.x/ANGSTROMINBOHR,cen.y/ANGSTROMINBOHR,cen.z/ANGSTROMINBOHR);

  // Get contraction of normalized primitives
  std::vector<contr_t> cn=get_contr_normalized();

  printf("\t\tExponential contraction is\n");
  printf("\t\t\tzeta\t\tprimitive coeff\ttotal coeff\n");
  for(size_t i=0;i<c.size();i++)
    printf("\t\t\t%e\t%e\t%e\n",c[i].z,cn[i].c,c[i].c);
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

std::vector<arma::mat> GaussianShell::moment(int am, double x, double y, double z, const GaussianShell & rhs) const {
  // Calculate moment integrals around (x,y,z) between shells

  // Amount of moments is
  size_t Nmom=(am+1)*(am+2)/2;

  // Moments to compute:
  std::vector<shellf_t> mom;
  mom.reserve(Nmom);
  for(int ii=0; ii<=am; ii++) {
    int lc=am - ii;
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


BasisSet::BasisSet() {
}

BasisSet::BasisSet(size_t Nat, const Settings & set) {
  // Use spherical harmonics?
  uselm=set.get_bool("UseLM");

  shells.reserve(Nat);
  nuclei.reserve(Nat);
#ifdef LIBINT
  libintok=0;
#endif
}

BasisSet::~BasisSet() {
}

void BasisSet::add_functions(int atind, coords_t cen, ElementBasisSet el) {
  // Add basis functions at cen

  // Get the shells on the element
  std::vector<FunctionShell> bf=el.get_shells();

  // Allocate memory for basis functions
  shells.reserve(shells.size()+bf.size());

  // Index for basis function
  size_t ind;

  // Loop over shells in element basis
  for(size_t i=0;i<bf.size();i++) {

    // Determine index of next basis function
    try {
      ind=get_last_ind()+1;
    } catch(std::domain_error) {
      // Basis set is empty, so index of first function on the shell will be
      ind=0;
    }

    // Get contraction

    // Use spherical harmonics even if the number of functions is not reduced
    // (unnecessary transformations of integrals)
    //    shells.push_back(GaussianShell(ind,el.bf[i].am,uselm,atind,cen,el.bf[i].C,el.bf[i].z));

    // Add functions. Use spherical harmonics only when it's beneficial, i.e. if am>=2
    if(bf[i].get_am()>=2)
      shells.push_back(GaussianShell(ind,bf[i].get_am(),uselm,atind,cen,bf[i].get_contr()));
    else
      shells.push_back(GaussianShell(ind,bf[i].get_am(),false,atind,cen,bf[i].get_contr()));

  }
}

void BasisSet::add_functions(int atind, coords_t cen, int am, const std::vector<contr_t> & C) {
  // Add basis functions at cen

  // Index for basis function
  size_t ind;

  // Determine index of next basis function
  try {
    ind=get_last_ind()+1;
  } catch(std::domain_error) {
    // Basis set is empty, so index of first function on the shell will be
    ind=0;
  }

  if(am>=2)
    shells.push_back(GaussianShell(ind,am,uselm,atind,cen,C));
  else
    shells.push_back(GaussianShell(ind,am,0,atind,cen,C));

  // Sort basis set.
  sort();
}

void BasisSet::add_shell(GaussianShell sh) {
  // Add shell
  shells.push_back(sh);
  // Check numbering
  check_numbering();
  // Sort basis set
  sort();
}

void BasisSet::add_nucleus(nucleus_t nuc) {
  nuclei.push_back(nuc);
}

void BasisSet::add_nucleus(int atind, coords_t cen, int Z, std::string sym, bool bsse) {
  nucleus_t nuc;
  nuc.atind=atind;
  nuc.x=cen.x;
  nuc.y=cen.y;
  nuc.z=cen.z;
  nuc.Z=Z;
  nuc.symbol=sym;
  nuc.bsse=bsse;

  add_nucleus(nuc);
}

void BasisSet::check_numbering() {
  // Renumber basis functions
  size_t ind=0;
  for(size_t i=0;i<shells.size();i++) {
    shells[i].set_first_ind(ind);
    ind=shells[i].get_last_ind()+1;
  }
}

void BasisSet::sort() {
  // Sort shells in increasing nuclear number, then in increasing
  // angular momentum, then in decreasing exponent
  stable_sort(shells.begin(),shells.end());

  // Check numbering
  check_numbering();
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
      d=dist(nuclei[i].x,nuclei[i].y,nuclei[i].z,nuclei[j].x,nuclei[j].y,nuclei[j].z);

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
  // Form list of unique shell pairs.
  shellpair_t tmp;

  // Now, form list of unique shell pairs
  for(size_t i=0;i<shells.size();i++) {
    for(size_t j=0;j<=i;j++) {
      // Have to set these in every iteartion due to swap below
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

size_t BasisSet::find_pair(size_t is, size_t js) const {
  for(size_t i=0;i<shellpairs.size();i++)
    if((shellpairs[i].is==is || shellpairs[i].js==is) && (shellpairs[i].is==js || shellpairs[i].js==js))
      return i;

  ERROR_INFO();
  std::ostringstream oss;
  oss << "Pair "<<is<<", " <<js<<" not found!"; 
  throw std::runtime_error(oss.str());

  return 0;
}    

std::vector<shellpair_t> BasisSet::get_unique_shellpairs() const {
  return shellpairs;
}

#ifdef LIBINT
void BasisSet::finalize(bool convert, bool libintok)
#else
void BasisSet::finalize(bool convert)
#endif
{
  // Finalize basis set structure for use.

  // Compute nuclear distances.
  compute_nuclear_distances();

  // Compute ranges of shells
  compute_shell_ranges();

  // Convert contractions
  if(convert)
    convert_contractions();
  // Normalize contractions
  normalize();

  // Initialize libint if necessary
#ifdef LIBINT
  if(libintok)
    set_libint_ok();
  else
    libint_init();
#endif

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
  return shells[num].get_inuc();
}

std::vector<GaussianShell> BasisSet::get_shells() const {
  return shells;
}

GaussianShell BasisSet::get_shell(size_t ind) const {
  return shells[ind];
}

coords_t BasisSet::get_shell_coords(size_t num) const {
  return shells[num].get_coords();
}

std::vector<contr_t> BasisSet::get_contr(size_t ind) const {
  return shells[ind].get_contr();
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

nucleus_t BasisSet::get_nuc(size_t inuc) const {
  return nuclei[inuc];
}

coords_t BasisSet::get_nuclear_coords(size_t inuc) const {
  coords_t r;
  r.x=nuclei[inuc].x;
  r.y=nuclei[inuc].y;
  r.z=nuclei[inuc].z;
  return r;
}

int BasisSet::get_Z(size_t inuc) const {
  return nuclei[inuc].Z;
}

std::string BasisSet::get_symbol(size_t inuc) const {
  return nuclei[inuc].symbol;
}

std::vector<GaussianShell> BasisSet::get_funcs(size_t inuc) const {

  std::vector<GaussianShell> ret;
  for(size_t i=0;i<shells.size();i++)
    if(shells[i].get_inuc()==inuc)
      ret.push_back(shells[i]);

  return ret;
}

std::vector<size_t> BasisSet::get_shell_inds(size_t inuc) const {
  std::vector<size_t> ret;
  for(size_t i=0;i<shells.size();i++)
    if(shells[i].get_inuc()==inuc)
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

void BasisSet::normalize() {
  for(size_t i=0;i<shells.size();i++)
    shells[i].normalize();
}

void BasisSet::coulomb_normalize() {
  for(size_t i=0;i<shells.size();i++)
    shells[i].coulomb_normalize();
}

void BasisSet::print() const {
  printf("There are %i shells and %i nuclei in the basis set.\n\n",(int) shells.size(),(int) nuclei.size());

  printf("List of nuclei, geometry in Ångström with three decimal places:\n");

  printf("ind\tZ\t    x\t    y\t    z\n");
  for(size_t i=0;i<nuclei.size();i++) {
    printf("%i\t%i\t% 7.3f\t% 7.3f\t% 7.3f\n",(int) i+1,nuclei[i].Z,nuclei[i].x/ANGSTROMINBOHR,nuclei[i].y/ANGSTROMINBOHR,nuclei[i].z/ANGSTROMINBOHR);
  }
  printf("\nList of basis functions:\n");

  /*
  for(size_t i=0;i<shells.size();i++) {
    printf("Shell %4i",(int) i);
    shells[i].print();
  }
  */

  for(size_t i=0;i<shells.size();i++) {
    // Type of shell - spherical harmonics or cartesians
    std::string type;
    if(shells[i].lm_in_use())
      type="sph";
    else
      type="cart";

    printf("Shell %4i",(int) i+1);
    printf("\t%c %4s shell at nucleus %i with with basis functions %4i-%-4i\n",shell_types[shells[i].get_am()],type.c_str(),(int) shells[i].get_inuc()+1,(int) shells[i].get_first_ind()+1,(int) shells[i].get_last_ind()+1);
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
    int Ncart=(am+1)*(am+2)/2;
    int Nlm=2*am+1;

    // Get transformation matrix
    tmp=Ylm_transmat(am);

    // Store transformation matrix
    trans.submat(l,n,l+Nlm-1,n+Ncart-1)=tmp;
    n+=Ncart;
    l+=Nlm;
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
  for(size_t i=0;i<shells.size();i++)
    for(size_t j=0;j<=i;j++) {
      // Get overlap between shells
      arma::mat tmp=shells[i].overlap(shells[j]);

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
  for(size_t i=0;i<shells.size();i++) {
    for(size_t j=0;j<rhs.shells.size();j++) {
      S12.submat(shells[i].get_first_ind(),rhs.shells[j].get_first_ind(),
		 shells[i].get_last_ind() ,rhs.shells[j].get_last_ind() )=shells[i].overlap(rhs.shells[j]);;
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
  for(size_t i=0;i<shells.size();i++)
    for(size_t j=0;j<=i;j++) {

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

  // Loop over nuclei
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(size_t inuc=0;inuc<nuclei.size();inuc++) {
    // If BSSE nucleus, do nothing
    if(nuclei[inuc].bsse)
      continue;
    
    // Nuclear charge
    int Z=nuclei[inuc].Z;
    
    // Coordinates of nucleus
    double cx=nuclei[inuc].x;
    double cy=nuclei[inuc].y;
    double cz=nuclei[inuc].z;
    
    // Loop over shells
    for(size_t i=0;i<shells.size();i++) {
      for(size_t j=0;j<i;j++) {
	
	// Get subblock
	arma::mat tmp=Z*shells[i].nuclear(cx,cy,cz,shells[j]);
	// On the off diagonal we fill out both sides of the matrix
	//	Vnuc.submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())+=tmp;
	//	Vnuc.submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())+=arma::trans(tmp);
#ifdef _OPENMP
#pragma omp critical
#endif
	{
	  Vnuc.submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())+=tmp;
	  Vnuc.submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())+=arma::trans(tmp);
	}
      }
      
      arma::mat tmp=Z*shells[i].nuclear(cx,cy,cz,shells[i]); 
      // Add only once on diagonal
      //      Vnuc.submat(shells[i].get_first_ind(),shells[i].get_first_ind(),shells[i].get_last_ind(),shells[i].get_last_ind())+=tmp;
#ifdef _OPENMP
#pragma omp critical
#endif
      Vnuc.submat(shells[i].get_first_ind(),shells[i].get_first_ind(),shells[i].get_last_ind(),shells[i].get_last_ind())+=tmp;
    }
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
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(size_t i=0;i<shells.size();i++) {
    // Off-diagonal
    for(size_t j=0;j<i;j++) {
      // Compute moment integral over shells
      std::vector<arma::mat> ints=shells[i].moment(mom,x,y,z,shells[j]);
      
      // Store moments
      for(size_t m=0;m<Nmom;m++) {
	ret[m].submat(shells[i].get_first_ind(),shells[j].get_first_ind(),shells[i].get_last_ind(),shells[j].get_last_ind())=ints[m];
	ret[m].submat(shells[j].get_first_ind(),shells[i].get_first_ind(),shells[j].get_last_ind(),shells[i].get_last_ind())=arma::trans(ints[m]);
      }
    }
    
    // Diagonal
    std::vector<arma::mat> ints=shells[i].moment(mom,x,y,z,shells[i]);
    for(size_t m=0;m<Nmom;m++)
      ret[m].submat(shells[i].get_first_ind(),shells[i].get_first_ind(),shells[i].get_last_ind(),shells[i].get_last_ind())=ints[m];
  }

  return ret;
}


double BasisSet::ERI_cart(size_t is, size_t ii, size_t js, size_t jj, size_t ks, size_t kk, size_t ls, size_t ll) const {
  // Calculate cartesian ERI from functions ii, jj, kk, and ll on shells is, js, ks and ls.

  // Angular momentum indices
  int la=shells[is].cart[ii].l;
  int ma=shells[is].cart[ii].m;
  int na=shells[is].cart[ii].n;

  int lb=shells[js].cart[jj].l;
  int mb=shells[js].cart[jj].m;
  int nb=shells[js].cart[jj].n;

  int lc=shells[ks].cart[kk].l;
  int mc=shells[ks].cart[kk].m;
  int nc=shells[ks].cart[kk].n;

  int ld=shells[ls].cart[ll].l;
  int md=shells[ls].cart[ll].m;
  int nd=shells[ls].cart[ll].n;

  // Coordinates of centers
  coords_t cena=shells[is].cen;
  coords_t cenb=shells[js].cen;
  coords_t cenc=shells[ks].cen;
  coords_t cend=shells[ls].cen;

  // Result
  double eri=0.0;

  // Exponentials and contraction coefficients
  double za, zb, zc, zd;
  double ca, cb, cc, cd;

  // Loop over exponential contractions
  for(size_t ix=0;ix<shells[is].c.size();ix++) {
    za=shells[is].c[ix].z;
    ca=shells[is].c[ix].c;

    for(size_t jx=0;jx<shells[js].c.size();jx++) {
      zb=shells[js].c[jx].z;
      cb=shells[js].c[jx].c;

      for(size_t kx=0;kx<shells[ks].c.size();kx++) {
	zc=shells[ks].c[kx].z;
	cc=shells[ks].c[kx].c;

	for(size_t lx=0;lx<shells[ls].c.size();lx++) {

	  zd=shells[ls].c[lx].z;
	  cd=shells[ls].c[lx].c;
	  
	  // ERI of these functions is
	  eri+=ca*cb*cc*cd*ERI_int(la,ma,na,cena.x,cena.y,cena.z,za,
				   lb,mb,nb,cenb.x,cenb.y,cenb.z,zb,
				   lc,mc,nc,cenc.x,cenc.y,cenc.z,zc,
				   ld,md,nd,cend.x,cend.y,cend.z,zd);
	}
      }
    }
  }

  // Finally, we plug in the relative normalization factors here
  eri*=shells[is].cart[ii].relnorm*shells[js].cart[jj].relnorm*shells[ks].cart[kk].relnorm*shells[ls].cart[ll].relnorm;

  return eri;
}

#ifdef LIBINT
void BasisSet::libint_init() {
  if(!libintok) {
    libintok=1;
    init_libint_base();
  }
}

void BasisSet::set_libint_ok() {
  libintok=1;
}

std::vector<double> BasisSet::ERI_cart(size_t is, size_t js, size_t ks, size_t ls) const {
  // Compute shell of cartesian ERIs using libint

  std::vector<double> ret=ERI_cart_wrap(&shells[is],&shells[js],&shells[ks],&shells[ls]);

#ifdef LIBINTDEBUG
  // Check the integrals against Huzinaga
  std::vector<double> huz;

  // Allocate memory
  huz.reserve(N);
  huz.resize(N);

  // Numbers of functions on each shell
  const size_t Ni=shells[is].get_Ncart();
  const size_t Nj=shells[js].get_Ncart();
  const size_t Nk=shells[ks].get_Ncart();
  const size_t Nl=shells[ls].get_Ncart();

  size_t ind;
  for(size_t ii=0;ii<Ni;ii++)
    for(size_t ji=0;ji<Nj;ji++)
      for(size_t ki=0;ki<Nk;ki++)
	for(size_t li=0;li<Nl;li++) {
	  // Index in return table is
	  ind=((ii*Nj+ji)*Nk+ki)*Nl+li;
	  // Compute ERI
	  huz[ind]=ERI_cart(is,ii,js,ji,ks,ki,ls,li);
	}

  ind=0;
  for(size_t ii=0;ii<Ni;ii++)
    for(size_t ji=0;ji<Nj;ji++)
      for(size_t ki=0;ki<Nk;ki++)
	for(size_t li=0;li<Nl;li++) {
	  // and compare it with the libint result
	  if(fabs(huz[ind]-ret[ind])>100*DBL_EPSILON*std::max(fabs(huz[ind]),fabs(ret[ind])) && std::max(fabs(ret[ind]),fabs(huz[ind]))>=DBL_EPSILON*10) {

	    int indi=shells[is].get_first_ind()+ii;
	    int indj=shells[js].get_first_ind()+ji;
	    int indk=shells[ks].get_first_ind()+ki;
	    int indl=shells[ls].get_first_ind()+li;
	        
	    printf("Integral %i %i %i %i gives %.16e with Huzinaga and %.16e with libint, relative difference is %e.\n",indi,indj,indk,indl,huz[ind],ret[ind],(huz[ind]-ret[ind])/std::max(huz[ind],ret[ind]));
	  }

	  ind++;
	}
#endif

  // Return integrals
  return ret;
}


std::vector<double> ERI_cart(const GaussianShell *is_orig, const GaussianShell *js_orig, const GaussianShell *ks_orig, const GaussianShell *ls_orig) {
  // Compute shell of cartesian ERIs using libint

  // Libint computes (ab|cd) for 
  // l(a)>=l(b), l(c)>=l(d) and l(a)+l(b)>=l(c)+l(d)
  // where l(a) is the angular momentum type of the a shell,
  // thus it is possible that the order of shells needs to be swapped.

  // Helpers
  const GaussianShell *is=is_orig;
  const GaussianShell *js=js_orig;
  const GaussianShell *ks=ks_orig;
  const GaussianShell *ls=ls_orig;

  // Figure out maximum angular momentum
  int max_am=max4(is->get_am(),js->get_am(),ks->get_am(),ls->get_am());
  // and the sum of angular momenta
  int mmax=is->get_am()+js->get_am()+ks->get_am()+ls->get_am();

  // Figure out the number of contractions
  size_t Ncomb=is->get_Ncontr()*js->get_Ncontr()*ks->get_Ncontr()*ls->get_Ncontr();

  // Check angular momentum
  if(max_am>=LIBINT_MAX_AM) {
    ERROR_INFO();
    throw std::domain_error("You need a version of LIBINT that supports larger angular momentum.\n");
  }

  // Evaluator object
  Libint_t libint;
  // Initialize evaluator object
  init_libint(&libint,max_am,Ncomb);

  // Did we need to swap the indices?
  bool swap_ij=0;
  bool swap_kl=0;
  bool swap_ijkl=0;

  // Check order and swap shells if necessary
  if(is->get_am()<js->get_am()) {
    swap_ij=1;
    std::swap(is,js);
  }
  
  if(ks->get_am()<ls->get_am()) {
    swap_kl=1;
    std::swap(ks,ls);
  }
  
  if( (is->get_am()+js->get_am()) > (ks->get_am()+ls->get_am())) {
    swap_ijkl=1;
    std::swap(is,ks);
    std::swap(js,ls);
  }

  // Compute data for LIBINT
  compute_libint_data(libint,is,js,ks,ls);

  // Numbers of functions on each shell
  const size_t Ni=is->get_Ncart();
  const size_t Nj=js->get_Ncart();
  const size_t Nk=ks->get_Ncart();
  const size_t Nl=ls->get_Ncart();

  // The number of integrals is
  const size_t N=Ni*Nj*Nk*Nl;
  // Pointer to integrals table
  double *ints;

  // Allocate memory for return
  std::vector<double> ret;

  // Allocate memory
  ret.reserve(N);
  ret.resize(N);

  // Special handling of (ss|ss) integrals:
  if(mmax==0) {
    double tmp=0.0;
    for(size_t i=0;i<Ncomb;i++)
      tmp+=libint.PrimQuartet[i].F[0];

    // Plug in normalizations
    tmp*=is->get_cart()[0].relnorm;
    tmp*=js->get_cart()[0].relnorm;
    tmp*=ks->get_cart()[0].relnorm;
    tmp*=ls->get_cart()[0].relnorm;

    ret[0]=tmp;
  } else {  
    //    printf("Computing shell %i %i %i %i",shells[is].get_am(),shells[js].get_am(),shells[ks].get_am(),shells[ls].get_am());
    //    printf("which consists of basis functions (%i-%i)x(%i-%i)x(%i-%i)x(%i-%i).\n",(int) shells[is].get_first_ind(),(int) shells[is].get_last_ind(),(int) shells[js].get_first_ind(),(int) shells[js].get_last_ind(),(int) shells[ks].get_first_ind(),(int) shells[ks].get_last_ind(),(int) shells[ls].get_first_ind(),(int) shells[ls].get_last_ind());

    // Now we can compute the integrals using libint:
    ints=build_eri[is->get_am()][js->get_am()][ks->get_am()][ls->get_am()](&libint,Ncomb);

    // Normalize and collect the integrals
    libint_collect(ret,ints,is,js,ks,ls,swap_ij,swap_kl,swap_ijkl);
  }

  // Free memory
  free_libint(&libint);

  // Return integrals
  return ret;
}

void compute_libint_data(Libint_t & libint, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {  
  // Compute data necessary for libint.

  // Sum of angular momenta
  int mmax=is->get_am()+js->get_am()+ks->get_am()+ls->get_am();

  // Coordinates of centers
  double A[3], B[3], C[3], D[3];
  
  A[0]=is->cen.x;
  A[1]=is->cen.y;
  A[2]=is->cen.z;
  
  B[0]=js->cen.x;
  B[1]=js->cen.y;
  B[2]=js->cen.z;
  
  C[0]=ks->cen.x;
  C[1]=ks->cen.y;
  C[2]=ks->cen.z;
  
  D[0]=ls->cen.x;
  D[1]=ls->cen.y;
  D[2]=ls->cen.z;

  // Store AB and CD
  for(int i=0;i<3;i++) {
    libint.AB[i]=A[i]-B[i];
    libint.CD[i]=C[i]-D[i];
  }

  // Distances
  double rabsq=0;
  double rcdsq=0;
  for(int i=0;i<3;i++) {
    rabsq+=(A[i]-B[i])*(A[i]-B[i]);
    rcdsq+=(C[i]-D[i])*(C[i]-D[i]);
  }

  size_t ind=0;

  // Helper variable
  prim_data data;

  // Exponents
  double zetaa, zetab, zetac, zetad;
  // Contraction coefficients
  double c_p, c_pr, c_prs, c;

  // Compute primitive data
  for(size_t p=0;p<is->c.size();p++) {
    zetaa=is->c[p].z;
    c_p=is->c[p].c;

    for(size_t r=0;r<js->c.size();r++) {
      zetab=js->c[r].z;
      c_pr=c_p*js->c[r].c;

      // Reduced exponent
      double zeta=zetaa+zetab;

      for(size_t s=0;s<ks->c.size();s++) {
	zetac=ks->c[s].z;
	c_prs=c_pr*ks->c[s].c;

	for(size_t t=0;t<ls->c.size();t++) {
	  zetad=ls->c[t].z;

	  // Product of contraction coefficients
	  c=c_prs*ls->c[t].c;

	  // Reduced exponents
	  double eta=zetac+zetad;
	  double rho=zeta*eta/(zeta+eta);

	  // Geometrical quantities
	  double P[3], Q[3], W[3];
	  double rpqsq;
	  
	  for(int i=0;i<3;i++) {
	    P[i]=(zetaa*A[i]+zetab*B[i])/zeta;
	    Q[i]=(zetac*C[i]+zetad*D[i])/eta;
	    W[i]=(P[i]*zeta+Q[i]*eta)/(zeta+eta);
	  }

	  // Compute (PQ)^2
	  rpqsq=0;
	  for(int i=0;i<3;i++)
	    rpqsq+=(P[i]-Q[i])*(P[i]-Q[i]);

	  // Compute and store PA, QC, WP and WQ
	  for(int i=0;i<3;i++) {
	    data.U[0][i]=P[i]-A[i];
	    data.U[2][i]=Q[i]-C[i];
	    data.U[4][i]=W[i]-P[i];
	    data.U[5][i]=W[i]-Q[i];
	  }

	  // Store exponents
	  data.oo2z=0.5/zeta;
	  data.oo2n=0.5/eta;
	  data.oo2zn=0.5/(eta+zeta);
	  data.oo2p=0.5/rho;
	  data.poz=rho/zeta;
	  data.pon=rho/eta;

	  // Compute overlaps for auxiliary integrals
	  double S12=(M_PI/zeta)*sqrt(M_PI/zeta)*exp(-zetaa*zetab/zeta*rabsq);
	  double S34=(M_PI/eta)*sqrt(M_PI/eta)*exp(-zetac*zetad/eta*rcdsq);

	  // Prefactor of Boys' function is
	  double prefac=2.0*sqrt(rho/M_PI)*S12*S34*c;
	  // and its argument is
	  double boysarg=rho*rpqsq;
	  // Evaluate Boys' function
	  std::vector<double> bf=boysF_arr(mmax,boysarg);

	  // Store auxiliary integrals
	  for(int i=0;i<=mmax;i++)
	    data.F[i]=prefac*bf[i];
	  
	  // We have all necessary data; store quartet.
	  libint.PrimQuartet[ind++]=data;
	}
      }
    }
  }
}

void libint_collect(std::vector<double> & ret, const double * ints, const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls, bool swap_ij, bool swap_kl, bool swap_ijkl) {
  // Normalize and collect libint integrals
  size_t ind_i, ind_ij, ind_ijk, ind;
  size_t indout;
  double norm_i, norm_ij, norm_ijk, norm;
  
  // Numbers of functions on each shell
  const size_t Ni=is->get_Ncart();
  const size_t Nj=js->get_Ncart();
  const size_t Nk=ks->get_Ncart();
  const size_t Nl=ls->get_Ncart();

  for(size_t ii=0;ii<Ni;ii++) {
    ind_i=ii*Nj;
    norm_i=is->cart[ii].relnorm;
    for(size_t ji=0;ji<Nj;ji++) {
      ind_ij=(ind_i+ji)*Nk;
      norm_ij=norm_i*js->cart[ji].relnorm;
      for(size_t ki=0;ki<Nk;ki++) {
	ind_ijk=(ind_ij+ki)*Nl;
	norm_ijk=norm_ij*ks->cart[ki].relnorm;
	for(size_t li=0;li<Nl;li++) {
	  // Index in computed integrals table
	  ind=ind_ijk+li;
	  // Total norm factor
	  norm=norm_ijk*ls->cart[li].relnorm;
	  // Compute output index
	  indout=get_swapped_ind(ii,Ni,ji,Nj,ki,Nk,li,Nl,swap_ij,swap_kl,swap_ijkl);
	  ret[indout]=norm*ints[ind];
	}
      }
    }
  }
}


#else

std::vector<double> BasisSet::ERI_cart(size_t is, size_t js, size_t ks, size_t ls) const {
  // Compute shell of cartesian ERIs.

  // Allocate memory for return
  std::vector<double> ret;

  // Numbers of functions on each shell
  size_t Ni=shells[is].get_Nbf();
  size_t Nj=shells[js].get_Nbf();
  size_t Nk=shells[ks].get_Nbf();
  size_t Nl=shells[ls].get_Nbf();

  // The number of integrals is
  const size_t N=Ni*Nj*Nk*Nl;

  // Allocate memory
  ret.reserve(N);
  ret.resize(N);

  // Index in table
  size_t ind;

  // Calculate integrals
  for(size_t ii=0;ii<Ni;ii++)
    for(size_t ji=0;ji<Nj;ji++)
      for(size_t ki=0;ki<Nk;ki++)
	for(size_t li=0;li<Nl;li++) {
	  // Index in return table is
	  ind=((ii*Nj+ji)*Nk+ki)*Nl+li;
	  // Compute ERI
	  ret[ind]=ERI(is,ii,js,ji,ks,ki,ls,li);
	}

  bool is_lm=shells[is].lm_in_use();
  bool js_lm=shells[js].lm_in_use();
  bool ks_lm=shells[ks].lm_in_use();
  bool ls_lm=shells[ls].lm_in_use();

  // Return integrals
  return ret;
}
#endif

std::vector<double> ERI_cart_wrap(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  return ERI_cart(is,js,ks,ls);
}

std::vector<double> BasisSet::ERI(size_t is, size_t js, size_t ks, size_t ls) const {
  // Calculate ERIs and transform them to spherical harmonics basis, if necessary.

  return ERI_wrap(&(shells[is]),&(shells[js]),&(shells[ks]),&(shells[ls]));
}

std::vector<double> ERI_wrap(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  return ERI(is,js,ks,ls);
}

std::vector<double> ERI(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  // Calculate ERIs and transform them to spherical harmonics basis, if necessary.

  // Get the cartesian ERIs
  std::vector<double> eris=ERI_cart(is,js,ks,ls);

  // Are the shells in question using spherical harmonics?
  const bool is_lm=is->lm_in_use();
  const bool js_lm=js->lm_in_use();
  const bool ks_lm=ks->lm_in_use();
  const bool ls_lm=ls->lm_in_use();

  // If not, return the cartesian ERIs.
  if(!is_lm && !js_lm && !ks_lm && !ls_lm)
    return eris;

  // Otherwise we need to compute the transformation of the ERIs into the
  // spherical basis. The transformation is made one shell at a time;
  // doing all at once would be an N^8 operation.

  // Transformation matrices
  arma::mat trans_i;
  if(is_lm)
    trans_i=is->get_trans();

  arma::mat trans_j;
  if(js_lm)
    trans_j=js->get_trans();

  arma::mat trans_k;
  if(ks_lm)
    trans_k=ks->get_trans();

  arma::mat trans_l;
  if(ls_lm)
    trans_l=ls->get_trans();
  
  // Amount of cartesians on shells (input)
  const size_t Ni_cart=is->get_Ncart();
  const size_t Nj_cart=js->get_Ncart();
  const size_t Nk_cart=ks->get_Ncart();
  const size_t Nl_cart=ls->get_Ncart();

  // Amount of target functions (may be Ncart or Nsph, depending on the usage
  // of spherical harmonics on the shell or not.
  const size_t Ni_tgt=is->get_Nbf();
  const size_t Nj_tgt=js->get_Nbf();
  const size_t Nk_tgt=ks->get_Nbf();
  const size_t Nl_tgt=ls->get_Nbf();
  
  // Sizes after transformations
  const size_t N_l  =Ni_cart*Nj_cart*Nk_cart*Nl_tgt;
  const size_t N_kl =Ni_cart*Nj_cart*Nk_tgt *Nl_tgt;
  const size_t N_jkl=Ni_cart*Nj_tgt *Nk_tgt *Nl_tgt;
  // The number of target integrals is
  const size_t N_tgt=Ni_tgt*Nj_tgt*Nk_tgt*Nl_tgt;    

  // Helpers for computing indices
  size_t indout_i, indout_j, indout_k, indout;
  size_t indin_i, indin_j, indin_k, indin;
  size_t indinout_i, indinout_j;

  // Helper array
  std::vector<double> tmp(N_l);


  /*
  // First, transform over l.
  if(ls_lm)
    for(size_t iic=0;iic<Ni_cart;iic++)
      for(size_t jjc=0;jjc<Nj_cart;jjc++)
	for(size_t kkc=0;kkc<Nk_cart;kkc++)
	  for(size_t lls=0;lls<Nl_tgt;lls++) {
	    // Output index
	    indout=((iic*Nj_cart+jjc)*Nk_cart+kkc)*Nl_tgt+lls;
	    
	    // Zero output
	    tmp[indout]=0.0;
	    
	    // Compute transform
	    for(size_t llc=0;llc<Nl_cart;llc++) {
	      indin=((iic*Nj_cart+jjc)*Nk_cart+kkc)*Nl_cart+llc;
	      tmp[indout]+=trans_l(lls,llc)*eris[indin];
	    }
	  }
  else
    tmp=eris;
  
  // Then, transform over k.
  if(ks_lm) {
    eris.resize(N_kl);

    for(size_t iic=0;iic<Ni_cart;iic++)
      for(size_t jjc=0;jjc<Nj_cart;jjc++)
	for(size_t lls=0;lls<Nl_tgt;lls++)

	  for(size_t kks=0;kks<Nk_tgt;kks++) {
	    indout=((iic*Nj_cart+jjc)*Nk_tgt+kks)*Nl_tgt+lls;
	    eris[indout]=0.0;

	    for(size_t kkc=0;kkc<Nk_cart;kkc++) {
	      indin=((iic*Nj_cart+jjc)*Nk_cart+kkc)*Nl_tgt+lls;
	      eris[indout]+=trans_k(kks,kkc)*tmp[indin];
	    }
	  }
  } else
    eris=tmp;

  // Then, over j
  if(js_lm) {
    eris.resize(N_jkl);

    for(size_t iic=0;iic<Ni_cart;iic++)
      for(size_t lls=0;lls<Nl_tgt;lls++)
	for(size_t kks=0;kks<Nk_tgt;kks++) 
	  for(size_t jjs=0;jjs<Nj_tgt;jjs++) {
	    indout=((iic*Nj_tgt+jjs)*Nk_tgt+kks)*Nl_tgt+lls;

	    tmp[indout]=0.0;
	    for(size_t jjc=0;jjc<Nj_cart;jjc++) {
	      indin=((iic*Nj_cart+jjc)*Nk_tgt+kks)*Nl_tgt+lls;
	      tmp[indout]+=trans_j(jjs,jjc)*eris[indin];
	    }
	  }
  } else
    tmp=eris;

  // Finally, over i
  if(is_lm) {
    eris.resize(N_tgt);

    for(size_t lls=0;lls<Nl_tgt;lls++)
      for(size_t kks=0;kks<Nk_tgt;kks++) 
	for(size_t jjs=0;jjs<Nj_tgt;jjs++)
	  for(size_t iis=0;iis<Ni_tgt;iis++) {
	    indout=((iis*Nj_tgt+jjs)*Nk_tgt+kks)*Nl_tgt+lls;
	    eris[indout]=0.0;
	    for(size_t iic=0;iic<Ni_cart;iic++) {
	      indin=((iic*Nj_tgt+jjs)*Nk_tgt+kks)*Nl_tgt+lls;
	      eris[indout]+=trans_i(iis,iic)*tmp[indin];
	    }
	  }
  } else
    eris=tmp;

  return eris;
  */
	    

  // Transform over l
  if(ls_lm) {
    for(size_t iic=0;iic<Ni_cart;iic++) {
      indinout_i=iic*Nj_cart;
      for(size_t jjc=0;jjc<Nj_cart;jjc++) {
	indinout_j=(indinout_i+jjc)*Nk_cart;
	for(size_t kkc=0;kkc<Nk_cart;kkc++) {
	  indout_k=(indinout_j+kkc)*Nl_tgt;
	  indin_k=(indinout_j+kkc)*Nl_cart;
	  
	  for(size_t ll=0;ll<Nl_tgt;ll++) {
	    //	  indout=((iic*Nj_cart+jjc)*Nk_cart+kkc)*Nl_sph+ll;
	    indout=indout_k+ll;

	    tmp[indout]=0.0;
	    
	    for(size_t llc=0;llc<Nl_cart;llc++) {
	      //	    indin=((iic*Nj_cart+jjc)*Nk_cart+kkc)*Nl_cart+llc;
	      indin=indin_k+llc;
	      tmp[indout]+=trans_l(ll,llc)*eris[indin];
	    }
	  }
	}
      }
    }
  } else
    tmp=eris;

  // Next, transform over k
  if(ks_lm) {
    eris.resize(N_kl);
    for(size_t iic=0;iic<Ni_cart;iic++) {
      indinout_i=iic*Nj_cart;
      
      for(size_t jjc=0;jjc<Nj_cart;jjc++) {
	indout_j=(indinout_i+jjc)*Nk_tgt;
	indin_j=(indinout_i+jjc)*Nk_cart;
	
	for(size_t kk=0;kk<Nk_tgt;kk++) {
	  indout_k=(indout_j+kk)*Nl_tgt;
	  
	  for(size_t ll=0;ll<Nl_tgt;ll++) {
	    //	    indout=((iic*Nj_cart+jjc)*Nk_tgt+kk)*Nl_tgt+ll;
	    indout=indout_k+ll;
	    
	    eris[indout]=0.0;
	    for(size_t kkc=0;kkc<Nk_cart;kkc++) {
	      //indin=((iic*Nj_cart+jjc)*Nk_cart+kkc)*Nl_tgt+ll;
	      indin=(indin_j+kkc)*Nl_tgt+ll;
	      eris[indout]+=trans_k(kk,kkc)*tmp[indin];
	    }
	  }
	}
      }
    }
  } else
    eris=tmp;

  // And over j
  if(js_lm) {
    tmp.resize(N_jkl);
    for(size_t iic=0;iic<Ni_cart;iic++) {
      indout_i=iic*Nj_tgt;
      indin_i=iic*Nj_cart;
      
      for(size_t jj=0;jj<Nj_tgt;jj++) {
	indout_j=(indout_i+jj)*Nk_tgt;
	
	for(size_t kk=0;kk<Nk_tgt;kk++) {
	  indout_k=(indout_j+kk)*Nl_tgt;
	  
	  for(size_t ll=0;ll<Nl_tgt;ll++) {
	    //	  indout=((iic*Nj_tgt+jj)*Nk_tgt+kk)*Nl_tgt+ll;
	    indout=indout_k+ll;

	    tmp[indout]=0.0;
	    for(size_t jjc=0;jjc<Nj_cart;jjc++) {
	      //	    indin=((iic*Nj_cart+jjc)*Nk_tgt+kk)*Nl_tgt+ll;
	      indin=((indin_i+jjc)*Nk_tgt+kk)*Nl_tgt+ll;
	      tmp[indout]+=trans_j(jj,jjc)*eris[indin];
	    }
	  }
	}
      }
    }
  } else
    tmp=eris;

  // Finally, transform over i
  if(is_lm) {
    eris.resize(N_tgt);
    for(size_t ii=0;ii<Ni_tgt;ii++) {
      indout_i=ii*Nj_tgt;
      
      for(size_t jj=0;jj<Nj_tgt;jj++) {
	indout_j=(indout_i+jj)*Nk_tgt;
	
	for(size_t kk=0;kk<Nk_tgt;kk++) {
	  indout_k=(indout_j+kk)*Nl_tgt;
	  
	  for(size_t ll=0;ll<Nl_tgt;ll++) {
	    //	  indout=((ii*Nj_sph+jj)*Nk_sph+kk)*Nl_sph+ll;
	    indout=indout_k+ll;

	    eris[indout]=0.0;
	    for(size_t iic=0;iic<Ni_cart;iic++) {
	      indin=((iic*Nj_tgt+jj)*Nk_tgt+kk)*Nl_tgt+ll;
	      eris[indout]+=trans_i(ii,iic)*tmp[indin];
	    }
	  }
	}
      }
    }

    return eris;
  } else 
    // No need for i transform
    return tmp;
}

int BasisSet::Ztot() const {
  int Ztot=0;
  for(size_t i=0;i<nuclei.size();i++) {
    if(nuclei[i].bsse)
      continue;
    Ztot+=nuclei[i].Z;
  }
  return Ztot;
}

double BasisSet::Enuc() const {
  double Enuc=0.0;

  for(size_t i=0;i<nuclei.size();i++) {
    if(nuclei[i].bsse)
      continue;

    int Zi=nuclei[i].Z;
    
    for(size_t j=0;j<i;j++) {
      if(nuclei[j].bsse)
	continue;
      int Zj=nuclei[j].Z;
      
      Enuc+=Zi*Zj/nucleardist(i,j);
    }
  }
  
  return Enuc;
}

void BasisSet::projectMOs(const BasisSet & oldbas, const arma::colvec & oldE, const arma::mat & oldMOs, arma::colvec & E, arma::mat & MOs) const {
  // Project MOs from old basis to new basis (this one)


  // Get overlap matrix
  arma::mat S11=overlap();

  // and form orthogonalizing matrix
  arma::mat Sinvh=CanonicalOrth(S11);
  // and the real S^-1
  arma::mat Sinv=Sinvh*arma::trans(Sinvh);

  // Get overlap with old basis
  arma::mat S12=overlap(oldbas);

  // Sizes of linearly independent basis sets
  const size_t Nbfo=oldMOs.n_cols;
  const size_t Nbfn=Sinvh.n_cols;

  // How many MOs do we transform?
  size_t Nmo=Nbfo;
  if(Nbfn<Nmo)
    Nmo=Nbfn;

  // OK, now we are ready to calculate the projections.

  // Initialize MO matrix
  MOs=arma::mat(Sinvh.n_rows,Nmo);
  MOs.zeros();
  // and energy
  E=arma::colvec(Nmo);
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

  // This is probably not necessary in sane cases, but we do it anyway;
  // it may be important if spilling occurs.
  for(size_t i=0;i<Nmo;i++) {
    arma::vec mo=MOs.col(i);

    // Remove overlap with other orbitals
    arma::vec hlp=S11*mo;
    for(size_t j=0;j<i;j++)
      mo-=arma::dot(MOs.col(j),hlp)*MOs.col(j);
    // Calculate norm
    double norm=sqrt(arma::as_scalar(arma::trans(mo)*S11*mo));
    // Normalize
    mo/=norm;
    // and store
    MOs.col(i)=mo;
  }
}

bool exponent_compare(const GaussianShell & lhs, const GaussianShell & rhs) {
  return lhs.get_contr()[0].z>rhs.get_contr()[0].z;
}


#ifdef DFT_ENABLED
BasisSet BasisSet::density_fitting(double fsam, int lmaxinc) const {
  // Automatically generate density fitting basis.

  // R. Yang, A. P. Rendell and M. J. Frisch, "Automatically generated
  // Coulomb fitting basis sets: Design and accuracy for systems
  // containing H to Kr", J. Chem. Phys. 127 (2007), 074102

  Settings set;
  set.add_dft_settings();
  // Density fitting basis set
  BasisSet dfit(1,set);

  // Loop over nuclei
  for(size_t in=0;in<nuclei.size();in++) {
    // Center of nucleus
    coords_t cen=get_nuclear_coords(in);

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
    std::vector<GaussianShell> shells=get_funcs(in);

    // Form candidate set - (2), (3) and (6) in YRF
    std::vector<GaussianShell> cand;
    for(size_t i=0;i<shells.size();i++) {
      // Get angular momentum
      int am=2*shells[i].get_am();
      // Get exponents
      std::vector<contr_t> contr=shells[i].get_contr();

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
	if(!found)
	  cand.push_back(GaussianShell(0,am,true,in,cen,C));
      }
    }

    // Sort trial set in order of decreasing exponents (don't care
    // about angular momentum) - (4) in YRF
    std::stable_sort(cand.begin(),cand.end(),exponent_compare);

    // Define maximum angular momentum for candidate functions and for
    // density fitting - (5) in YRF
    int lmax_obs=0;
    for(size_t i=0;i<shells.size();i++)
      if(shells[i].get_am()>lmax_obs)
	lmax_obs=shells[i].get_am();
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
	  if(lvals[l]>0)
	    dfit.add_functions(in,cen,l,C);
      }
    } // (10) in YRF
  } // (11) in YRF
 
  // Normalize basis set
  dfit.coulomb_normalize();

  return dfit;
}
#endif

BasisSet BasisSet::exchange_fitting() const {
  // Exchange fitting basis set

  Settings set;
  BasisSet fit(nuclei.size(),set);

  const int maxam=get_max_am();

  // Loop over nuclei
  for(size_t in=0;in<nuclei.size();in++) {
    // Center of nucleus
    coords_t cen=get_nuclear_coords(in);

    // Get shells corresponding to this nucleus
    std::vector<GaussianShell> shells=get_funcs(in);

    // Sort shells in increasing angular momentum
    std::sort(shells.begin(),shells.end());

    // Determine amount of functions on current atom and minimum and maximum exponents
    std::vector<int> nfunc(maxam+1);
    std::vector<double> mine(maxam+1);
    std::vector<double> maxe(maxam+1);
    int lmax=0;

    // Initialize arrays
    for(int l=0;l<=maxam;l++) {
      nfunc[l]=0;
      mine[l]=DBL_MAX;
      maxe[l]=0.0;
    }

    // Loop over shells of current nucleus
    for(size_t ish=0;ish<shells.size();ish++) {
      // Current angular momentum
      int l=shells[ish].get_am();

      // Update maximum value
      if(l>lmax)
	lmax=l;

      // Increase amount of functions
      nfunc[l]++;

      // Get exponential contraction
      std::vector<contr_t> contr=shells[ish].get_contr();

      // Check exponent ranges
      if(mine[l]>contr[contr.size()-1].z)
	mine[l]=contr[contr.size()-1].z;
      
      if(maxe[l]<contr[0].z)
	maxe[l]=contr[0].z;
    }

    // Add functions to fitting basis set
    for(int l=0;l<=lmax;l++) {
      // Add density fitting functions                                                                                                                                                      
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
	fit.add_functions(in,cen,l,C);
      }
    }
  }
 
  // Normalize basis set
  fit.coulomb_normalize();

  return fit;
}

GaussianShell dummyshell() {
  coords_t cen={0.0, 0.0, 0.0};
  std::vector<contr_t> C(1);
  C[0].c=1.0;
  C[0].z=0.0;
  return GaussianShell(0,0,0,0,cen,C);
}

size_t get_swapped_ind(size_t i, size_t Ni, size_t j, size_t Nj, size_t k, size_t Nk, size_t l, size_t Nl, bool swap_ij, bool swap_kl, bool swap_ijkl) {
  // Compute indices of swapped integrals.

  // First, swap ij-kl if necessary.
  if(swap_ijkl) {
    std::swap(i,k);
    std::swap(Ni,Nk);
    std::swap(j,l);
    std::swap(Nj,Nl);
  }
    
  // Then, swap k-l if necessary.
  if(swap_kl) {
    std::swap(k,l);
    std::swap(Nk,Nl);
  }

  // Finally, swap i-j if necessary.
  if(swap_ij) {
    std::swap(i,j);
    std::swap(Ni,Nj);
  }

  // Now, compute the index
  return ((i * Nj + j) * Nk + k) * Nl + l;
}

std::vector<size_t> i_idx(size_t N) {
  std::vector<size_t> ret;
  ret.reserve(N);
  ret.resize(N);
  for(size_t i=0;i<N;i++)
    ret[i]=i*(i+1)/2;
  return ret;
}

#ifdef LIBINT
BasisSet construct_basis(const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib, const Settings & set, bool libintok)
#else
BasisSet construct_basis(const std::vector<atom_t> & atoms, const BasisSetLibrary & baslib, const Settings & set)
#endif
{
  // Number of atoms is
  size_t Nat=atoms.size();

  // Create basis set
  BasisSet basis(Nat,set);
  // and add atoms to basis set
  for(size_t i=0;i<Nat;i++) {
    // Get center
    coords_t cen;
    cen.x=atoms[i].x;
    cen.y=atoms[i].y;
    cen.z=atoms[i].z;

    // Determine if nucleus is BSSE or not
    bool bsse=0;
    std::string el=atoms[i].el;

    if(el.size()>3 && el.substr(el.size()-3,3)=="-Bq") {
      // Yes, this is a BSSE nucleus
      bsse=1;
      el=el.substr(0,el.size()-3);
    }

    // Get functions belonging to nucleus
    ElementBasisSet elbas;
    try {
      // Check first if a special set is wanted for given center
      elbas=baslib.get_element(el,atoms[i].num+1);
    } catch(std::runtime_error err) {
      // Did not find a special basis, use the general one instead.
      elbas=baslib.get_element(el,0);
    }

    basis.add_functions(i,cen,elbas);
    // and the nucleus
    basis.add_nucleus(i,cen,get_Z(el),el,bsse);
  }

  // Finalize basis set
#ifdef LIBINT
  basis.finalize(1,libintok);
#else
  basis.finalize(1);
#endif

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
    double dist=norm(r-bas.get_nuclear_coords(inuc));
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
    double dist=norm(r-bas.get_nuclear_coords(inuc));
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
