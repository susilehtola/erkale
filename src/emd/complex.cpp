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



#include "complex.h"
#include <cmath>
#include <cstdio>

complex cconj(const complex no) {
  complex ret;
  ret.re=no.re;
  ret.im=-no.im;
  return ret;
}

complex cneg(const complex no) {
  complex ret;
  ret.re=-no.re;
  ret.im=-no.im;
  return ret;
}

complex cexp(const complex no) {
  complex ret;
  ret.re=exp(no.re)*cos(no.im);
  ret.im=exp(no.re)*sin(no.im);
  return ret;
}

complex csin(const complex no) {
  complex ret;
  ret.re=sin(no.re)*cosh(no.im);
  ret.im=cos(no.re)*sinh(no.im);
  return ret;
}

complex ccos(const complex no) {
  complex ret;
  ret.re=cos(no.re)*cosh(no.im);
  ret.im=-sin(no.re)*sinh(no.im);
  return ret;
}

complex csinh(const complex no) {
  complex ret;
  ret.re=sinh(no.re)*cos(no.im);
  ret.im=cosh(no.re)*sin(no.im);
  return ret;
}

complex ccosh(const complex no) {
  complex ret;
  ret.re=cosh(no.re)*cos(no.im);
  ret.im=sinh(no.re)*sin(no.im);
  return ret;
}

complex cmult(const complex lhs, const complex rhs) {
  complex ret;
  ret.re=lhs.re*rhs.re-lhs.im*rhs.im;
  ret.im=lhs.re*rhs.im+lhs.im*rhs.re;
  return ret;
}

complex cdiv(const complex lhs, const complex rhs) {
  complex ret;
  ret.re=(lhs.re*rhs.re+lhs.im*rhs.im)/cnormsq(rhs);
  ret.im=(-lhs.re*rhs.im+lhs.im*rhs.re)/cnormsq(rhs);
  return ret;
}

complex cadd(const complex lhs, const complex rhs) {
  complex ret;
  ret.re=lhs.re+rhs.re;
  ret.im=lhs.im+rhs.im;
  return ret;
}

complex csub(const complex lhs, const complex rhs) {
  complex ret;
  ret.re=lhs.re-rhs.re;
  ret.im=lhs.im-rhs.im;
  return ret;
}

complex cpow(complex no, int m) {
  int i;

  /* Helper variable */
  complex ret;
  ret.re=1.0;
  ret.im=0.0;

  /* If m<0 we need to do division */
  if(m<0) {
    no=cdiv(ret,no);
    m=-m;
  }

  /* Calculate multiplication */
  for(i=0;i<m;i++)
    ret=cmult(ret,no);

  return ret;
}


complex cconjmult(complex conj, const complex no) {
  complex ret;
  ret.re=conj.re*no.re+conj.im*no.im;
  ret.im=no.im*conj.re-conj.im*no.re;
  return ret;
}

double cnormsq(const complex no) {
  return no.re*no.re+no.im*no.im;
}

double cnorm(const complex no) {
  return sqrt(no.re*no.re+no.im*no.im);
}

complex cscale(const complex no, const double fac) {
  complex ret;
  ret.re=no.re*fac;
  ret.im=no.im*fac;
  return ret;
}

void cprint(const complex no) {
  printf("(%g,%g)",no.re,no.im);
}

complex operator+(const complex & lhs, const complex & rhs) {
  return cadd(lhs,rhs);
}

complex & operator+=(complex & lhs, const complex & rhs) {
  lhs.re+=rhs.re;
  lhs.im+=rhs.im;
  return lhs;
}

complex operator*(const complex & lhs, const complex & rhs) {
  return cmult(lhs,rhs);
}

complex & operator*=(complex & lhs, const complex & rhs) {
  lhs=cmult(lhs,rhs);
  return lhs;
}

complex operator*(const double & lhs, const complex & rhs) {
  return cscale(rhs,lhs);
}

complex operator*(const complex & lhs, const double & rhs) {
  return cscale(lhs,rhs);
}
  
complex & operator*=(complex & lhs, const double & rhs) {
  lhs.re*=rhs;
  lhs.im*=rhs;
  return lhs;
}  
