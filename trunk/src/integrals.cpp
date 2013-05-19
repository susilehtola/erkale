/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010
 * Copyright (c) 2010, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * The routines in this file are based on the classical paper
 *    "Gaussian-Expansion Methods for Molecular Integrals"
 * by H. Taketa, S. Huzinaga and K. O-Ohata, in the
 * Journal of the Physical Society of Japan, Vol 21, No 11, November 1966
 */



#include "integrals.h"
#include "mathf.h"
#include <vector>
// For exceptions
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdio>

// Type for looping over vector-type arrays
#define size_type std::vector<double>::size_type


double normconst(double zeta, int l, int m, int n) {
  // Calculate normalization coefficient of Gaussian primitive
  // THO 2.2

  double N;

  N=pow(zeta,l+m+n+1.5)*(2.0/M_PI)*sqrt(2.0/M_PI);
  N/=doublefact(2*l-1);
  N/=doublefact(2*m-1);
  N/=doublefact(2*n-1);
  N=pow(2.0,l+m+n)*sqrt(N);

  return N;
}

// Calculate expansion coefficient of x^j in (x+a)^l (x+b)^m
double fj(int j, int l, int m, double a, double b) {

  // In the expansion we have 0 <= im <= m, 0 <= il <= l.

  // For the factor of x^j we must have im + il == j, thus we require
  // that il = j - im. Obviously 0 <= j <= m + l.

  // Sanity check
  if(j<0 || j>m+l) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Trying to compute fj for j="<<j<<", l="<<l<<", m="<<m<<"!";
    throw std::runtime_error(oss.str());
  }

  // Now, requiring
  // 0 <= im <= m
  // 0 <= il <= l
  // we get
  // 0 <= im <= m
  // 0 <= j - im <= l

  // and we get the requirements
  // 0 <= im <= m
  // im <= j <= l+im
  // and we get the loop limits as
  // max(0,j-l) <= im <= min(m,j)

  // Lower and upper limit
  int low;
  if(j-l>0)
    low=j-l;
  else
    low=0;

  int high;
  if(m<j)
    high=m;
  else
    high=j;

  double ret=0.0;
  for(int im=low;im<=high;im++) {
    // This would be
    // ret+=choose(m,im)*pow(b,m-im)*choose(l,il)*pow(a,l-il);
    // but now il=j-im, thus
    ret+=choose(m,im)*pow(b,m-im)*choose(l,j-im)*pow(a,l-j+im);
  }
  return ret;
}

double center_1d(double zetaa, double xa, double zetab, double xb) {
  // Compute center of r_A and r_B
  return (zetaa*xa+zetab*xb)/(zetaa+zetab);
}

double distsq(double xa, double ya, double za, double xb, double yb, double zb) {
  // Compute distance squared of r_A and r_B
  return (xa-xb)*(xa-xb)+(ya-yb)*(ya-yb)+(za-zb)*(za-zb);
}

double dist(double xa, double ya, double za, double xb, double yb, double zb) {
  // Compute distance of r_A and r_B
  return sqrt(distsq(xa,ya,za,xb,yb,zb));
}

double overlap_int(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xb, double yb, double zb, double zetab, int lb, int mb, int nb) {
  // Compute overlap of unnormalized primitives at r_A and r_B

  // Distance between centers
  double absq=distsq(xa,ya,za,xb,yb,zb);

  // Compute center of product
  double xc=center_1d(zetaa,xa,zetab,xb);
  double yc=center_1d(zetaa,ya,zetab,yb);
  double zc=center_1d(zetaa,za,zetab,zb);

  // Sum of the exponentials
  double zeta=zetaa+zetab;
  double twozeta=2.0*zeta;

  // Compute overlap in x
  double Ox=0.0;
  for(int i=0;i<=(la+lb)/2;i++)
    Ox+=fj(2*i,la,lb,xc-xa,xc-xb)*doublefact(2*i-1)/pow(twozeta,i);

  // and in y
  double Oy=0.0;
  for(int i=0;i<=(ma+mb)/2;i++)
    Oy+=fj(2*i,ma,mb,yc-ya,yc-yb)*doublefact(2*i-1)/pow(twozeta,i);

  // and in z
  double Oz=0.0;
  for(int i=0;i<=(na+nb)/2;i++)
    Oz+=fj(2*i,na,nb,zc-za,zc-zb)*doublefact(2*i-1)/pow(twozeta,i);

  // End result is
  return pow(M_PI/zeta,1.5)*exp(-zetaa*zetab/zeta*absq)*Ox*Oy*Oz;
}


double kinetic_int(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xb, double yb, double zb, double zetab, int lb, int mb, int nb) {
  // Kinetic energy integral
  // THO 2.14

  // First term
  double ke1=zetab*(2*(lb+mb+nb)+3)*overlap_int(xa,ya,za,zetaa,la,ma,na,xb,yb,zb,zetab,lb,mb,nb);
  // Second term
  double ke2=-2.0*zetab*zetab*(
 			      overlap_int(xa,ya,za,zetaa,la,ma,na,xb,yb,zb,zetab,lb+2,mb,nb)
			     +overlap_int(xa,ya,za,zetaa,la,ma,na,xb,yb,zb,zetab,lb,mb+2,nb)
			     +overlap_int(xa,ya,za,zetaa,la,ma,na,xb,yb,zb,zetab,lb,mb,nb+2));
  // Third term
  double ke3=0.0;

  if(lb>=2)
    ke3+=-0.5*lb*(lb-1)*overlap_int(xa,ya,za,zetaa,la,ma,na,xb,yb,zb,zetab,lb-2,mb,nb);
  if(mb>=2)
    ke3+=-0.5*mb*(mb-1)*overlap_int(xa,ya,za,zetaa,la,ma,na,xb,yb,zb,zetab,lb,mb-2,nb);
  if(nb>=2)
    ke3+=-0.5*nb*(nb-1)*overlap_int(xa,ya,za,zetaa,la,ma,na,xb,yb,zb,zetab,lb,mb,nb-2);

  return ke1+ke2+ke3;
}


// Compute A array for nucler attraction integral (THO 2.18)
std::vector<double> A_array(int la, int lb, double PAx, double PBx, double PCx, double zeta) {
  // The idea behind the array is that although in principle it has three
  // indices (i, r and u), only the combination i-2r-u is used in THO 2.17.

  // Size of array
  int N=la+lb+1;

  std::vector<double> ret;
  // Make sure of sufficient memory allocation
  ret.reserve(N);
  // Resize array
  ret.resize(N);

  // Zero out array
  for(int i=0;i<N;i++)
    ret[i]=0.0;

  // Loop over indices i, r, and u
  int indx;
  double incr;

  for(int i=0;i<=la+lb;i++)
    for(int r=0;r<=i/2;r++)
      for(int u=0;u<=(i-2*r)/2;u++) {
	// Current index in array is
	indx=i-2*r-u;
	// Increment element
	incr=pow(-1.0,i+u)*fj(i,la,lb,PAx,PBx)*fact(i)*pow(PCx,i-2*r-2*u)*pow(4.0*zeta,-r-u)/(fact(r)*fact(u)*fact(i-2*r-2*u));

	ret[indx]+=incr;
      }
  return ret;
}

double nuclear_int(double xa, double ya, double za, double zetaa, int la, int ma, int na, double xnuc, double ynuc, double znuc, double xb, double yb, double zb, double zetab, int lb, int mb, int nb) {
  // Compute nuclear attraction integral <r_A|r_nuc|r_B>

  // Distance between centers
  double ABsq=distsq(xa,ya,za,xb,yb,zb);

  // Compute center of product
  double xp=center_1d(zetaa,xa,zetab,xb);
  double yp=center_1d(zetaa,ya,zetab,yb);
  double zp=center_1d(zetaa,za,zetab,zb);

  // Sum of the exponentials
  double zeta=zetaa+zetab;

  // Distances to center
  double PAx=xp-xa;
  double PAy=yp-ya;
  double PAz=zp-za;

  double PBx=xp-xb;
  double PBy=yp-yb;
  double PBz=zp-zb;

  /* There seems to be some confusion in THO, where after 2.15 {\bf p} is
     defined as {\bf P} - {\bf C} and p_x^(i-2r-2u) appears in the intermediate
     formulas, but CPx is in 2.18 instead of PCx. However, only the
     latter version works.

     "Handbook of Computational Chemistry" by David B. Cook also
     users PCx, not CPx.
 */

  double PCx=xp-xnuc;
  double PCy=yp-ynuc;
  double PCz=zp-znuc;

  double PCsq=PCx*PCx+PCy*PCy+PCz*PCz;

  // The argument of the Boys function is
  double boysarg=zeta*PCsq;

  // Get arrays in x, y and z
  std::vector<double> Ax=A_array(la,lb,PAx,PBx,PCx,zeta);
  std::vector<double> Ay=A_array(ma,mb,PAy,PBy,PCy,zeta);
  std::vector<double> Az=A_array(na,nb,PAz,PBz,PCz,zeta);

  /*  for(size_t i=0;i<Ax.size();i++)
    printf("Ax[%i]=%e\n",(int) i,Ax[i]);
  for(size_t i=0;i<Ay.size();i++)
    printf("Ay[%i]=%e\n",(int) i,Ay[i]);
  for(size_t i=0;i<Az.size();i++)
  printf("Az[%i]=%e\n",(int) i,Az[i]);*/

  // Total product array
  std::vector<double> A;
  // Size of product array
  size_type N=Ax.size()+Ay.size()+Az.size();
  A.reserve(N);
  A.resize(N);

  // Zero out array
  for(size_type i=0;i<N;i++)
    A[i]=0.0;

  // Compute products
  for(size_type ix=0;ix<Ax.size();ix++)
    for(size_type iy=0;iy<Ay.size();iy++)
      for(size_type iz=0;iz<Az.size();iz++)
	A[ix+iy+iz]+=Ax[ix]*Ay[iy]*Az[iz];


  // Now, the NAI is
  double nai=0.0;
  double bf;
  for(size_type i=0;i<N;i++) {
    bf=boysF(i,boysarg);
    nai+=A[i]*bf;
  }
  // and we plug in the constant factor to get
  nai*=-2.0*M_PI/zeta*exp(-zetaa*zetab*ABsq/zeta);

  //  printf("Nuclear attraction from (%e,%e,%e), wrt to l=%i,m=%i,n=%i,zeta=%e at (%e,%e,%e) and l=%i,m=%i,n=%i,zeta=%e at (%e,%e,%e) is %e.\n",xnuc,ynuc,znuc,la,ma,na,zetaa,xa,ya,za,lb,mb,nb,zetab,xb,yb,zb,nai);

  return nai;
}

// The factor that appears in the B array
double B_theta(int l, int l1, int l2, double a, double b, int r, double zeta) {
  return fj(l,l1,l2,a,b)*fact_ratio(l,r)*pow(zeta,r-l);
}

// Compute B array for electron repulsion integral (THO 2.22).
std::vector<double> B_array(int la, int lb, double Ax, double Bx, double Px, double zetaab, int lc, int ld, double Cx, double Dx, double Qx, double zetacd) {
  // The idea here is as before in the A array - even though there are a lot
  // of indices, they can be compactified into one

  std::vector<double> ret;
  // Size of returned array
  size_type N=la+lb+lc+ld+1;
  ret.reserve(N);
  ret.resize(N);

  // Combined exponential
  const double delta=1.0/(4.0*zetaab)+1.0/(4.0*zetacd);

  // Displacements
  const double PAx=Px-Ax;
  const double PBx=Px-Bx;
  const double QCx=Qx-Cx;
  const double QDx=Qx-Dx;

  const double QPx=Qx-Px;

  // Zero array
  for(size_type i=0;i<N;i++)
    ret[i]=0.0;

  size_type indx;

  // Loop over sum indices
  for(int i1=0;i1<=la+lb;i1++)
    for(int i2=0;i2<=lc+ld;i2++)
      for(int r1=0;r1<=i1/2;r1++)
	for(int r2=0;r2<=i2/2;r2++)
	  for(int u=0;u<=(i1+i2)/2-r1-r2;u++) {
	    // Index in array is
	    indx=i1+i2-2*(r1+r2)-u;
	    // Increment value
	    /*	    ret[indx]+=pow(-1.0,i2+u)*
	      fj(i1,la,lb,PAx,PBx)*fact_ratio(i1,r1)*pow(4.0*zetaab,r1-i1)*
	      fj(i2,lc,ld,QCx,QDx)*fact_ratio(i2,r2)*pow(4.0*zetacd,r2-i2)*
	      fact_ratio(i1+i2-2*(r1+r2),u)*pow(QPx,i1+i2-2*(r1+r2)-2*u)
	      // THO actually reads
	      *pow(delta,-i1-i2+u+r1+r2);
	      // but only the following
	      //	      	      *pow(delta,-i1-i2+2*(r1+r2)+u);
	      */

	    // Cook p. 249
	    ret[indx]+=pow(-1.0,i2+u)
	      *B_theta(i1,la,lb,PAx,PBx,r1,zetaab)
	      *B_theta(i2,lc,ld,QCx,QDx,r2,zetacd)
	      *pow(4.0,r1+r2-i1-i2)*pow(delta,2*(r1+r2)-i1-i2+u)
	      *fact_ratio(i1+i2-2*(r1+r2),u)*pow(QPx,i1+i2-2*(r1+r2+u));
	  }
  return ret;
}

double ERI_int(int la, int ma, int na, double Ax, double Ay, double Az, double zetaa, int lb, int mb, int nb, double Bx, double By, double Bz, double zetab, int lc, int mc, int nc, double Cx, double Cy, double Cz, double zetac, int ld, int md, int nd, double Dx, double Dy, double Dz, double zetad) {
  // Compute electron repulsion integral <ab|cd>

  // Compute exponents
  double zetaab=zetaa+zetab;
  double zetacd=zetac+zetad;
  double fourdelta=1.0/zetaab+1.0/zetacd;

  // Compute centers
  double Px=center_1d(zetaa,Ax,zetab,Bx);
  double Py=center_1d(zetaa,Ay,zetab,By);
  double Pz=center_1d(zetaa,Az,zetab,Bz);

  double Qx=center_1d(zetac,Cx,zetad,Dx);
  double Qy=center_1d(zetac,Cy,zetad,Dy);
  double Qz=center_1d(zetac,Cz,zetad,Dz);

  double ABsq=distsq(Ax, Ay, Az, Bx, By, Bz);
  double CDsq=distsq(Cx, Cy, Cz, Dx, Dy, Dz);

  // Get arrays
  std::vector<double> Barrx=B_array(la,lb,Ax,Bx,Px,zetaab,lc,ld,Cx,Dx,Qx,zetacd);
  std::vector<double> Barry=B_array(ma,mb,Ay,By,Py,zetaab,mc,md,Cy,Dy,Qy,zetacd);
  std::vector<double> Barrz=B_array(na,nb,Az,Bz,Pz,zetaab,nc,nd,Cz,Dz,Qz,zetacd);

  // Debug: print arrays
  /*  printf("\n\n");
  for(size_t i=0;i<Barrx.size();i++)
    printf("Bx[%i]=%e\n",(int) i,Barrx[i]);
  for(size_t i=0;i<Barry.size();i++)
    printf("By[%i]=%e\n",(int) i,Barry[i]);
  for(size_t i=0;i<Barrz.size();i++)
  printf("Bz[%i]=%e\n",(int) i,Barrz[i]);*/


  // Form product array
  size_type N, Nx, Ny, Nz;
  Nx=Barrx.size();
  Ny=Barry.size();
  Nz=Barrz.size();
  N=Nx+Ny+Nz;

  std::vector<double> B;
  B.reserve(N);
  B.resize(N);

  // Zero out array
  for(size_type i=0;i<N;i++)
    B[i]=0.0;

  // Loop over arrays in x, y and z
  for(size_type ix=0;ix<Nx;ix++)
    for(size_type iy=0;iy<Ny;iy++)
      for(size_type iz=0;iz<Nz;iz++)
	B[ix+iy+iz]+=Barrx[ix]*Barry[iy]*Barrz[iz];

  // Argument of Boys' function is
  double boysarg=distsq(Px, Py, Pz, Qx, Qy, Qz)/fourdelta;
  double bf;
  //  printf("Argument of Boys' function is %e\n",boysarg);

  // Result of ERI is
  double eri=0.0;
  for(size_type i=0;i<N;i++) {
    bf=boysF(i,boysarg);
    eri+=B[i]*bf;
  }

  //  printf("Sum is %e\n",eri);

  // Prefactor is
  double prefact=2.0*pow(M_PI,2.5)/(zetaab*zetacd*sqrt(zetaab+zetacd))*exp(-zetaa*zetab*ABsq/zetaab-zetac*zetad*CDsq/zetacd);

  /*  printf("Prefactor is %e\n",prefact);
  printf("Ax=%e, Bx=%e, Cx=%e, Dx=%e\n",Ax,Bx,Cx,Dx);
  printf("zetaa=%e, zetab=%e, zetac=%e, zetad=%e, zetaab=%e, zetacd=%e, ABsq=%e, CDsq=%e\n\n\n",zetaa,zetab,zetac,zetad,zetaab,zetacd,ABsq,CDsq); */

  return prefact*eri;
}
