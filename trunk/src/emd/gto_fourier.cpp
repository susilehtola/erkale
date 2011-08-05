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
#include <cmath>
#include <cstdio>

FourierPoly_1D::FourierPoly_1D() {
}

FourierPoly_1D::FourierPoly_1D(int l, double zeta) {
  // Construct polynomial using recursion
  *this=formpoly(l,zeta);

  // Now, add in the normalization factor
  complex normfac;
  normfac.re=pow(2.0*zeta,-0.5-l);
  normfac.im=0.0;
  *this=normfac*(*this);
}

FourierPoly_1D FourierPoly_1D::formpoly(int l, double zeta) {
  // Construct polynomial w/o normalization factor

  FourierPoly_1D ret;

  poly1d_t term;

  if(l==0) {
    // R_0 (p_i, zeta) = 1.0

    term.c.re=1.0;
    term.c.im=0.0;
    term.l=0;

    ret.poly.push_back(term);
  } else if(l==1) {
    // R_1 (p_i, zeta) = - i p_i

    term.c.re=0.0;
    term.c.im=-1.0;
    term.l=1;

    ret.poly.push_back(term);
  } else {
    // Use recursion to formulate.

    FourierPoly_1D lm1=formpoly(l-1,zeta);
    FourierPoly_1D lm2=formpoly(l-2,zeta);

    complex fac;

    // Add first the second term
    fac.re=2*zeta*(l-1);
    fac.im=0.0;
    ret=fac*lm2;

    // We add the first term separately, since it conserves the value of angular momentum.
    fac.re=0.0;
    fac.im=-1.0;
    for(size_t i=0;i<lm1.getN();i++)
      ret.addterm(cmult(fac,lm1.getc(i)),lm1.getl(i)+1);
  }
  
  return ret;
}

FourierPoly_1D::~FourierPoly_1D() {
}

void FourierPoly_1D::addterm(complex c, int l) {
  bool found=0;
  // First, see if the term is already in the contraction.
  for(size_t i=0;i<poly.size();i++)
    if(poly[i].l==l) {
      found=1;
      poly[i].c=cadd(poly[i].c,c);
      break;
    }
  
  // If it was not found, add the new term
  if(!found) {
    poly1d_t help;
    help.c=c;
    help.l=l;
    
    poly.push_back(help);
  }
}

FourierPoly_1D FourierPoly_1D::operator+(const FourierPoly_1D & rhs) const {
  FourierPoly_1D ret;

  ret=*this;
  for(size_t i=0;i<rhs.poly.size();i++)
    ret.addterm(rhs.poly[i].c,rhs.poly[i].l);

  return ret;
}

size_t FourierPoly_1D::getN() const {
  return poly.size();
}

complex FourierPoly_1D::getc(size_t i) const {
  return poly[i].c;
}

int FourierPoly_1D::getl(size_t i) const {
  return poly[i].l;
}

void FourierPoly_1D::print() const {
  for(size_t i=0;i<poly.size();i++) {
    printf("(%e,%e)p^%i\n",poly[i].c.re,poly[i].c.im,poly[i].l);
    if(i<poly.size()-1)
      printf(" + ");
  }
  printf("\n");
}

FourierPoly_1D operator*(complex fac, const FourierPoly_1D & rhs) {
  FourierPoly_1D ret(rhs);
  
  for(size_t i=0;i<ret.poly.size();i++) {
    ret.poly[i].c=cmult(fac,ret.poly[i].c);
  }
  
  return ret;
}

GTO_Fourier::GTO_Fourier() {
}

GTO_Fourier::GTO_Fourier(int l, int m, int n, double zeta) {

  // Create polynomials in px, py and pz
  FourierPoly_1D px(l,zeta), py(m,zeta), pz(n,zeta);

  complex facx, facy, facz;
  int lx, ly, lz;

  complex facxy;

  // Loop over the individual polynomials
  for(size_t ix=0;ix<px.getN();ix++) {
    facx=px.getc(ix);
    lx=px.getl(ix);

    for(size_t iy=0;iy<py.getN();iy++) {
      facy=py.getc(iy);
      ly=py.getl(iy);

      facxy=cmult(facx,facy);

      for(size_t iz=0;iz<pz.getN();iz++) {
	facz=pz.getc(iz);
	lz=pz.getl(iz);

         // Add the term
	addterm(cmult(facxy,facz),lx,ly,lz,1.0/(4.0*zeta));
      }
    }
  }
}

GTO_Fourier::~GTO_Fourier() {
}

void GTO_Fourier::addterm(complex c, int l, int m, int n, double z) {
  // Add term to transform

  bool found=0;
  // First, check if the same kind of term already exists
  for(size_t i=0;i<trans.size();i++)
    if(trans[i].l==l && trans[i].m==m && trans[i].n==n && trans[i].z==z) {
      trans[i].c=cadd(trans[i].c,c);
      found=1;
      break;
    }

  if(!found) {
    // Else, add the term
    trans3d_t help;
    help.c=c;
    help.l=l;
    help.m=m;
    help.n=n;
    help.z=z;

    trans.push_back(help);
  }
}

GTO_Fourier GTO_Fourier::operator+(const GTO_Fourier & rhs) const {
  GTO_Fourier ret=*this;

  for(size_t i=0;i<rhs.trans.size();i++)
    ret.addterm(rhs.trans[i].c,rhs.trans[i].l,rhs.trans[i].m,rhs.trans[i].n,rhs.trans[i].z);

  return ret;
}

GTO_Fourier & GTO_Fourier::operator+=(const GTO_Fourier & rhs) {
  for(size_t i=0;i<rhs.trans.size();i++)
    addterm(rhs.trans[i].c,rhs.trans[i].l,rhs.trans[i].m,rhs.trans[i].n,rhs.trans[i].z);

  return *this;
}

std::vector<trans3d_t> GTO_Fourier::get() const {
  return trans;
}

GTO_Fourier operator*(complex fac, const GTO_Fourier & rhs) {
  GTO_Fourier ret=rhs;

  for(size_t i=0;i<ret.trans.size();i++)
    ret.trans[i].c=cmult(fac,ret.trans[i].c);

  return ret;
}

GTO_Fourier operator*(double fac, const GTO_Fourier & rhs) {
  GTO_Fourier ret=rhs;

  for(size_t i=0;i<ret.trans.size();i++) {
    ret.trans[i].c.re*=fac;
    ret.trans[i].c.im*=fac;
  }

  return ret;
}

void GTO_Fourier::print() const {
  for(size_t i=0;i<trans.size();i++)
    printf("(%e,%e) px^%i py^%i pz^%i exp(-%e p^2)\n",trans[i].c.re,trans[i].c.im,trans[i].l,trans[i].m,trans[i].n,trans[i].z);
}

void GTO_Fourier::clean() {
  for(size_t i=trans.size()-1;i<trans.size();i--)
    if(trans[i].c.re==0 && trans[i].c.im==0.0)
      trans.erase(trans.begin()+i);
}
