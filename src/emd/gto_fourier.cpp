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

#include "gto_fourier.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

bool operator<(const poly1d_t & lhs, const poly1d_t & rhs) {
  return lhs.l<rhs.l;
}

bool operator==(const poly1d_t& lhs, const poly1d_t & rhs) {
  return lhs.l==rhs.l;
}

FourierPoly_1D::FourierPoly_1D() {
}

FourierPoly_1D::FourierPoly_1D(int l, double zeta) {
  // Construct polynomial using recursion
  *this=formpoly(l,zeta);

  // Now, add in the normalization factor
  std::complex<double> normfac(pow(2.0*zeta,-0.5-l),0.0);
  *this=normfac*(*this);
}

FourierPoly_1D FourierPoly_1D::formpoly(int l, double zeta) {
  // Construct polynomial w/o normalization factor

  FourierPoly_1D ret;

  poly1d_t term;

  if(l==0) {
    // R_0 (p_i, zeta) = 1.0
    term.c=1.0;
    term.l=0;

    ret.poly.push_back(term);
  } else if(l==1) {
    // R_1 (p_i, zeta) = - i p_i

    term.c=std::complex<double>(0.0,-1.0);
    term.l=1;

    ret.poly.push_back(term);
  } else {
    // Use recursion to formulate.

    FourierPoly_1D lm1=formpoly(l-1,zeta);
    FourierPoly_1D lm2=formpoly(l-2,zeta);

    std::complex<double> fac;

    // Add first the second term
    fac=std::complex<double>(2*zeta*(l-1),0.0);
    ret=fac*lm2;

    // We add the first term separately, since it conserves the value of angular momentum.
    fac=std::complex<double>(0.0,-1.0);
    for(size_t i=0;i<lm1.getN();i++) {
      term.c=fac*lm1.getc(i);
      term.l=lm1.getl(i)+1;
      ret.addterm(term);
    }
  }
  
  return ret;
}

FourierPoly_1D::~FourierPoly_1D() {
}

void FourierPoly_1D::addterm(const poly1d_t & t) {
  if(poly.size()==0) {
    poly.push_back(t);
  } else {
    // Get upper bound
    std::vector<poly1d_t>::iterator high;
    high=std::upper_bound(poly.begin(),poly.end(),t);
    
    // Corresponding index is
    size_t ind=high-poly.begin();
    
    if(ind>0 && poly[ind-1]==t)
	// Found it.
      poly[ind-1].c+=t.c;
    else {
      // Term does not exist, add it
      poly.insert(high,t);
    }
  }
}

FourierPoly_1D FourierPoly_1D::operator+(const FourierPoly_1D & rhs) const {
  FourierPoly_1D ret;

  ret=*this;
  for(size_t i=0;i<rhs.poly.size();i++)
    ret.addterm(rhs.poly[i]);

  return ret;
}

size_t FourierPoly_1D::getN() const {
  return poly.size();
}

std::complex<double> FourierPoly_1D::getc(size_t i) const {
  return poly[i].c;
}

int FourierPoly_1D::getl(size_t i) const {
  return poly[i].l;
}

void FourierPoly_1D::print() const {
  for(size_t i=0;i<poly.size();i++) {
    printf("(%e,%e)p^%i\n",poly[i].c.real(),poly[i].c.imag(),poly[i].l);
    if(i<poly.size()-1)
      printf(" + ");
  }
  printf("\n");
}

FourierPoly_1D operator*(std::complex<double> fac, const FourierPoly_1D & rhs) {
  FourierPoly_1D ret(rhs);
  
  for(size_t i=0;i<ret.poly.size();i++)
    ret.poly[i].c*=fac;
  
  return ret;
}

bool operator<(const trans3d_t & lhs, const trans3d_t& rhs) {
  // Sort first by angular momentum.
  if(lhs.l+lhs.m+lhs.n<rhs.l+rhs.m+rhs.n)
    return 1;
  else if(lhs.l+lhs.m+lhs.n==rhs.l+rhs.m+rhs.n) {
    // Then by x component
    if(lhs.l<rhs.l)
      return 1;
    else if(lhs.l==rhs.l) {
      // Then by y component
      if(lhs.m<rhs.m)
	return 1;
      else if(lhs.m==rhs.m) {
	// Then by z component
	if(lhs.n<rhs.n)
	  return 1;
	else if(lhs.n==rhs.n) {
	  // and finally by exponent
	  return lhs.z<rhs.z;
	}
      }
    }
  }

  return 0;
}

bool operator==(const trans3d_t & lhs, const trans3d_t& rhs) {
  return (lhs.l==rhs.l) && (lhs.m==rhs.m) && (lhs.n==rhs.n) && (lhs.z==rhs.z);
}

GTO_Fourier::GTO_Fourier() {
}

GTO_Fourier::GTO_Fourier(int l, int m, int n, double zeta) {

  // Create polynomials in px, py and pz
  FourierPoly_1D px(l,zeta), py(m,zeta), pz(n,zeta);

  std::complex<double> facx, facy, facz;
  int lx, ly, lz;

  std::complex<double> facxy;

  // Loop over the individual polynomials
  for(size_t ix=0;ix<px.getN();ix++) {
    facx=px.getc(ix);
    lx=px.getl(ix);

    for(size_t iy=0;iy<py.getN();iy++) {
      facy=py.getc(iy);
      ly=py.getl(iy);

      facxy=facx*facy;

      for(size_t iz=0;iz<pz.getN();iz++) {
	facz=pz.getc(iz);
	lz=pz.getl(iz);

	// Add the corresponding term
	trans3d_t term;
	term.l=lx;
	term.m=ly;
	term.n=lz;
	term.z=1.0/(4.0*zeta);
	term.c=facxy*facz;
	addterm(term);
      }
    }
  }
}

GTO_Fourier::~GTO_Fourier() {
}

void GTO_Fourier::addterm(const trans3d_t & t) {
  if(trans.size()==0) {
    trans.push_back(t);
  } else {
    // Get upper bound
    std::vector<trans3d_t>::iterator high;
    high=std::upper_bound(trans.begin(),trans.end(),t);
    
    // Corresponding index is
    size_t ind=high-trans.begin();
    
    if(ind>0 && trans[ind-1]==t)
	// Found it.
      trans[ind-1].c+=t.c;
    else {
      // Term does not exist, add it
      trans.insert(high,t);
    }
  }
}

GTO_Fourier GTO_Fourier::operator+(const GTO_Fourier & rhs) const {
  GTO_Fourier ret=*this;

  for(size_t i=0;i<rhs.trans.size();i++)
    ret.addterm(rhs.trans[i]);

  return ret;
}

GTO_Fourier & GTO_Fourier::operator+=(const GTO_Fourier & rhs) {
  for(size_t i=0;i<rhs.trans.size();i++)
    addterm(rhs.trans[i]);

  return *this;
}

std::vector<trans3d_t> GTO_Fourier::get() const {
  return trans;
}

std::complex<double> GTO_Fourier::eval(double px, double py, double pz) const {
  // Value of the transform
  std::complex<double> ret=0.0;

  // Momentum squared
  double psq=px*px+py*py+pz*pz;

  // Evaluate
  for(size_t i=0;i<trans.size();i++)
    ret+=trans[i].c*pow(px,trans[i].l)*pow(py,trans[i].m)*pow(pz,trans[i].n)*exp(-trans[i].z*psq);

  return ret;
}

GTO_Fourier operator*(std::complex<double> fac, const GTO_Fourier & rhs) {
  GTO_Fourier ret=rhs;

  for(size_t i=0;i<ret.trans.size();i++)
    ret.trans[i].c*=fac;

  return ret;
}

GTO_Fourier operator*(double fac, const GTO_Fourier & rhs) {
  GTO_Fourier ret=rhs;

  for(size_t i=0;i<ret.trans.size();i++)
    ret.trans[i].c*=fac;

  return ret;
}

void GTO_Fourier::print() const {
  for(size_t i=0;i<trans.size();i++)
    printf("(%e,%e) px^%i py^%i pz^%i exp(-%e p^2)\n",trans[i].c.real(),trans[i].c.imag(),trans[i].l,trans[i].m,trans[i].n,trans[i].z);
}

void GTO_Fourier::clean() {
  for(size_t i=trans.size()-1;i<trans.size();i--)
    if(norm(trans[i].c) == 0.0)
      trans.erase(trans.begin()+i);
}
