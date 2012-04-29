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
#include <cmath>
#include <cstdio>
// For exceptions
#include <sstream>
#include <stdexcept>

#include "gto_fourier.h"
#include "spherical_expansion.h"

#include "../mathf.h"
#include "../timer.h"

extern "C" {
// 3j symbols
#include <gsl/gsl_sf_coupling.h>
}

// Index of (l,m) in table
#define lmind(l,m) ((l)*(l)+l+m)
// Location in multiplication table
#define multloc(l1,m1,l2,m2) (lmind(maxam+1,maxam+1)*lmind(l1,m1)+lmind(l2,m2))

bool operator<(const ylmcoeff_t & lhs, const ylmcoeff_t & rhs) {
  if(lhs.l<rhs.l)
    return 1;
  else if(lhs.l==rhs.l)
    return lhs.m<rhs.m;

  return 0;
}

bool operator==(const ylmcoeff_t & lhs, const ylmcoeff_t & rhs) {
  return (lhs.l==rhs.l) && (lhs.m==rhs.m);
}

SphericalExpansion::SphericalExpansion() {
}

SphericalExpansion::~SphericalExpansion() {
}

void SphericalExpansion::add(const ylmcoeff_t & t) {
  if(comb.size()==0) {
    comb.push_back(t);
  } else {
    // Get upper bound
    std::vector<ylmcoeff_t>::iterator high;
    high=std::upper_bound(comb.begin(),comb.end(),t);

    // Corresponding index is
    size_t ind=high-comb.begin();

    if(ind>0 && comb[ind-1]==t)
      comb[ind-1].c+=t.c;
    else {
      // Term does not exist, add it
      comb.insert(high,t);
    }
  }
}

void SphericalExpansion::addylm(int l, int m, std::complex<double> c) {
  ylmcoeff_t hlp;
  hlp.l=l;
  hlp.m=m;
  hlp.c=c;
  add(hlp);
}

void SphericalExpansion::addylm(int l, int m, double d) {
  std::complex<double> c(d,0);
  addylm(l,m,c);
}

void SphericalExpansion::clean() {
  // Clean out the table of linear combinations by removing elements with zero weight
  int ok;

  do {
    // Default value
    ok=1;

    for(size_t i=0;i<comb.size();i++)
      if(norm(comb[i].c) == 0.0) { // If there is an element with no weight
	comb.erase(comb.begin()+i); // Erase the element
	ok=0; // Redo while loop
	break; // Break for loop
      }
  } while(!ok);
}


void SphericalExpansion::clear() {
  // Clear out everything
  comb.clear();
}


SphericalExpansion SphericalExpansion::conjugate() const {
  // Complex conjugate the expansion
  SphericalExpansion ret=*this;

  for(size_t i=0;i<ret.comb.size();i++) {
    // The expansion coefficient changes to (-1)^m times its complex conjugate
    ret.comb[i].c=conj(ret.comb[i].c)*pow(-1.0,ret.comb[i].m);
    // and the sign of m changes
    ret.comb[i].m=-ret.comb[i].m;
  }

  // Finally, re-sort the list
  ret.sort();

  return ret;
}

void SphericalExpansion::print() const {
  // Print out the list of combinations
  for(size_t i=0;i<comb.size();i++) {
    printf("\t%i\t%i\t(%e, %e)\n",comb[i].l,comb[i].m,comb[i].c.real(),comb[i].c.imag());
  }
}

void SphericalExpansion::sort() {
  // Sort out the linear combination in increasing l, increasing m

  int ok;
  ylmcoeff_t temp;

  do {
    ok=1;

    for(size_t i=0;i<comb.size();i++)
      for(size_t j=0;j<i;j++)
	if( (comb[j].l>comb[i].l) || (comb[j].l==comb[i].l && comb[j].m>comb[i].m) ) {
	  ok=0;
	  temp=comb[j];
	  comb[j]=comb[i];
	  comb[i]=temp;
	}
  } while(!ok);
}


size_t SphericalExpansion::getN() const {
  return comb.size();
}

ylmcoeff_t SphericalExpansion::getcoeff(size_t i) const {
  return comb[i];
}

std::vector<ylmcoeff_t> SphericalExpansion::getcoeffs() const {
  return comb;
}

int SphericalExpansion::getmaxl() const {
  int maxl=0;
  for(size_t i=0;i<comb.size();i++)
    if(comb[i].l>maxl)
      maxl=comb[i].l;
  return maxl;
}

SphericalExpansion SphericalExpansion::operator+(const SphericalExpansion & rhs) const {
  // Addition of two linear combinations of spherical harmonics
  SphericalExpansion ret=*this;
  for(size_t i=0;i<rhs.comb.size();i++)
      ret.addylm(rhs.comb[i].l,rhs.comb[i].m,rhs.comb[i].c);
  return ret;
}

SphericalExpansion & SphericalExpansion::operator+=(const SphericalExpansion & rhs) {
  // Addition of two linear combinations of spherical harmonics
  for(size_t i=0;i<rhs.comb.size();i++)
      addylm(rhs.comb[i].l,rhs.comb[i].m,rhs.comb[i].c);
  return *this;
}

SphericalExpansion SphericalExpansion::operator-() const {
  SphericalExpansion ret=*this;
  for(size_t i=0;i<comb.size();i++)
    ret.comb[i].c*=-1.0;
  return ret;
}

SphericalExpansion SphericalExpansion::operator-(const SphericalExpansion & rhs) const {
  // Substraction of linear combinations of spherical harmonics
  SphericalExpansion ret=*this;
  for(size_t i=0;i<rhs.comb.size();i++) {
    ret.addylm(rhs.comb[i].l,rhs.comb[i].m,-rhs.comb[i].c);
  }
  return ret;
}

SphericalExpansion & SphericalExpansion::operator-=(const SphericalExpansion & rhs) {
  // Substraction of linear combinations of spherical harmonics
  for(size_t i=0;i<rhs.comb.size();i++) {
    addylm(rhs.comb[i].l,rhs.comb[i].m,-rhs.comb[i].c);
  }
  return *this;
}

SphericalExpansion SphericalExpansion::operator*(const SphericalExpansion & rhs) const {
  // Reduce multiplication of two spherical harmonics to a linear combination of spherical harmonics

  // New combination
  SphericalExpansion newcomb;
  // Allocate enough memory
  newcomb.comb.reserve(comb.size()+rhs.comb.size());

  // Maximum and minimum l in combination
  int lmin, lmax;

  // New coefficient
  double dc;
  std::complex<double> c;

  // Loop over combinations
  for(size_t i=0;i<comb.size();i++)
    for(size_t j=0;j<rhs.comb.size();j++) {

      // Lower and upper limit for l in loop
      if(comb[i].l>rhs.comb[j].l)
	lmin=comb[i].l-rhs.comb[j].l;
      else
	lmin=rhs.comb[j].l-comb[i].l;

      lmax=comb[i].l+rhs.comb[j].l;


      // Loop over new angular momentum values
      for(int l=lmin;l<=lmax;l++)
	// Loop over z component values
	for(int m=-l;m<=l;m++) {
	  // Calculate new coefficient
	  c=comb[i].c*rhs.comb[j].c;

	  // If coefficient is zero don't do anything
	  if(norm(c)==0.0)
	    continue;

	  // Real scaling factor: \sqrt{ \frac {(2j_1+1)(2j_2+1)(2j+1)} {4 \pi} } (-1)^m
	  dc=sqrt((2.0*comb[i].l+1.0)*(2.0*rhs.comb[j].l+1.0)*(2.0*l+1.0)/(4.0*M_PI))*pow(-1.0,m);
	  // Put in the 3j factors - GSL uses them as half integer units so multiply by two
	  dc*=gsl_sf_coupling_3j(2*comb[i].l,2*rhs.comb[j].l,2*l,2*comb[i].m,2*rhs.comb[j].m,-2*m);
	  dc*=gsl_sf_coupling_3j(2*comb[i].l,2*rhs.comb[j].l,2*l,0,0,0);

	  // Add it to the list if the scaling factor is not zero
	  if(dc!=0)
	    newcomb.addylm(l,m,c*dc);
	}
    }

  // Clean out the new combination table (in case of cancellations)
  newcomb.clean();

  // Sort the list
  newcomb.sort();

  return newcomb;
}

SphericalExpansion & SphericalExpansion::operator*=(const SphericalExpansion & rhs) {
  SphericalExpansion ret=(*this)*rhs;
  *this=ret;
  return *this;
}

SphericalExpansion & SphericalExpansion::operator*=(std::complex<double> fac) {
  // Scale the combination
  for(size_t i=0;i<comb.size();i++)
    comb[i].c*=fac;
  return *this;
}

SphericalExpansion & SphericalExpansion::operator*=(double fac) {
  // Scale the combination
  for(size_t i=0;i<comb.size();i++)
    comb[i].c*=fac;
  return *this;
}

SphericalExpansion operator*(std::complex<double> fac, const SphericalExpansion & func) {
  // Scale the combination
  SphericalExpansion ret(func);
  for(size_t i=0;i<ret.comb.size();i++)
    ret.comb[i].c*=fac;
  return ret;
}

SphericalExpansion operator*(double fac, const SphericalExpansion & func) {
  // Scale the combination
  SphericalExpansion ret=func;
  for(size_t i=0;i<ret.comb.size();i++)
    ret.comb[i].c*=fac;
  return ret;
}

// Multiplication table for spherical harmonics
SphericalExpansionMultiplicationTable::SphericalExpansionMultiplicationTable(int am) {
  maxam=am;
  table.resize(multloc(maxam,maxam,maxam,maxam)+1);

  // Left and right values
  for(int lleft=0;lleft<=maxam;lleft++)
    for(int mleft=-lleft;mleft<=lleft;mleft++) {
      SphericalExpansion left;
      left.addylm(lleft,mleft,1.0);

      for(int lright=0;lright<=maxam;lright++)
        for(int mright=-lright;mright<=lright;mright++) {
          SphericalExpansion right;
          right.addylm(lright,mright,1.0);

          table[multloc(lleft,mleft,lright,mright)]=left*right;
        }
    }
}

SphericalExpansionMultiplicationTable::~SphericalExpansionMultiplicationTable() {
}

void SphericalExpansionMultiplicationTable::print() const {
  for(int lleft=0;lleft<=maxam;lleft++)
    for(int mleft=-lleft;mleft<=lleft;mleft++)
      for(int lright=0;lright<=maxam;lright++)
        for(int mright=-lright;mright<=lright;mright++) {
          printf("The product of (%i,%i) with (%i,%i) is:\n",lleft,mleft,lright,mright);
          table[multloc(lleft,mleft,lright,mright)].print();
        }
}

SphericalExpansion SphericalExpansionMultiplicationTable::mult(const SphericalExpansion & lhs, const SphericalExpansion & rhs) const {
  // Returned expansion
  SphericalExpansion ret;

  // Check that table is big enough
  if(lhs.getmaxl()>maxam || rhs.getmaxl()>maxam) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Table not big enough: maxam = " << maxam << " but am_lhs = " << lhs.getmaxl() << " and am_rhs = " << rhs.getmaxl() << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Continue with multiplication. Loop over terms:
  for(size_t i=0;i<lhs.comb.size();i++)
    for(size_t j=0;j<rhs.comb.size();j++) {
      ret+=lhs.comb[i].c*rhs.comb[j].c*table[multloc(lhs.comb[i].l,lhs.comb[i].m,rhs.comb[j].l,rhs.comb[j].m)];
    }

  return ret;
}

CartesianExpansion::CartesianExpansion(int maxam) {
  // Reserve memory for table
  table.resize(maxam+1);
  // and for individual results
  for(int am=0;am<=maxam;am++)
    table[am].resize((am+1)*(am+2)/2);

  // Compute spherical harmonics expansions of px^l, py^m and pz^n
  std::vector<SphericalExpansion> px, py, pz;
  
  px.resize(maxam+1);
  py.resize(maxam+1);
  pz.resize(maxam+1);
  
  // p_i^0 = 1 = \sqrt{4 \pi} Y_0^0
  px[0].addylm(0,0,sqrt(4.0*M_PI));
  py[0].addylm(0,0,sqrt(4.0*M_PI));
  pz[0].addylm(0,0,sqrt(4.0*M_PI));

  // px = p * sqrt{ 2 \pi / 3} * ( Y_1^{-1} - Y_1^1)
  if(maxam>0) {
    px[1].addylm(1,-1, sqrt(2.0*M_PI/3.0));
    px[1].addylm(1, 1,-sqrt(2.0*M_PI/3.0));
  }
  // py = ip * sqrt{ 2 \pi / 3} * ( Y_1^{-1} + Y_1^1 )
  if(maxam>0) {
    std::complex<double> hlp(0.0,sqrt(2.0*M_PI/3.0));
    py[1].addylm(1,-1,hlp);
    py[1].addylm(1, 1,hlp);
  }
  // pz = p * sqrt{4 \pi / 3} Y_1^0
  if(maxam>0)
    pz[1].addylm(1,0,sqrt(4.0*M_PI/3.0));

  // Form the rest of the transforms
  for(int il=2;il<=maxam;il++)
    px[il]=px[il-1]*px[1];

  for(int im=2;im<=maxam;im++)
    py[im]=py[im-1]*py[1];

  for(int in=2;in<=maxam;in++)
    pz[in]=pz[in-1]*pz[1];

  // Fill table. Loop over angular momentum
  for(int am=0;am<=maxam;am++) {

    // Loop over functions
    size_t idx=0;
    for(int ii=0; ii<=am; ii++) {
      int l = am - ii;
      for(int jj=0; jj<=ii; jj++) {
	int m = ii-jj;
	int n = jj;

	table[am][idx++]=px[l]*py[m]*pz[n];
      }
    }
  }
}

CartesianExpansion::~CartesianExpansion() {
}

SphericalExpansion CartesianExpansion::get(int l, int m, int n) const {
  if(l+m+n >= (int) table.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Cartesian expansion table not big enough: maxam = " << (int) table.size()-1 << " am = " << l+m+n << " requested!\n";
    throw std::runtime_error(oss.str());
  }

  return table[l+m+n][getind(l,m,n)];
}
