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
      comb[ind-1].c=cadd(comb[ind-1].c,t.c);
    else {
      // Term does not exist, add it
      comb.insert(high,t);
    }
  }
}

void SphericalExpansion::addylm(int l, int m, complex c) {
  ylmcoeff_t hlp;
  hlp.l=l;
  hlp.m=m;
  hlp.c=c;
  add(hlp);
}

void SphericalExpansion::addylm(int l, int m, double d) {
  complex c;
  c.re=d;
  c.im=0;

  addylm(l,m,c);
}

void SphericalExpansion::clean() {
  // Clean out the table of linear combinations by removing elements with zero weight
  int ok;

  do {
    // Default value
    ok=1;

    for(size_t i=0;i<comb.size();i++)
      if(comb[i].c.re==0 && comb[i].c.im==0) { // If there is an element with no weight
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
    ret.comb[i].c=cscale(cconj(ret.comb[i].c),pow(-1.0,ret.comb[i].m));
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
    printf("\t%i\t%i\t(%e, %e)\n",comb[i].l,comb[i].m,comb[i].c.re,comb[i].c.im);
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
    ret.comb[i].c=cneg(ret.comb[i].c);
  return ret;
}

SphericalExpansion SphericalExpansion::operator-(const SphericalExpansion & rhs) const {
  // Substraction of linear combinations of spherical harmonics
  SphericalExpansion ret=*this;
  for(size_t i=0;i<rhs.comb.size();i++) {
    ret.addylm(rhs.comb[i].l,rhs.comb[i].m,cneg(rhs.comb[i].c));
  }
  return ret;
}

SphericalExpansion & SphericalExpansion::operator-=(const SphericalExpansion & rhs) {
  // Substraction of linear combinations of spherical harmonics
  for(size_t i=0;i<rhs.comb.size();i++) {
    addylm(rhs.comb[i].l,rhs.comb[i].m,cneg(rhs.comb[i].c));
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
  complex c;

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
	  c=cmult(comb[i].c,rhs.comb[j].c); // Complex part

	  // If coefficient is zero don't do anything
	  if(!c.re && !c.im)
	    continue;

	  // Real scaling factor: \sqrt{ \frac {(2j_1+1)(2j_2+1)(2j+1)} {4 \pi} } (-1)^m
	  dc=sqrt((2.0*comb[i].l+1.0)*(2.0*rhs.comb[j].l+1.0)*(2.0*l+1.0)/(4.0*M_PI))*pow(-1.0,m);
	  // Put in the 3j factors - GSL uses them as half integer units so multiply by two
	  dc*=gsl_sf_coupling_3j(2*comb[i].l,2*rhs.comb[j].l,2*l,2*comb[i].m,2*rhs.comb[j].m,-2*m);
	  dc*=gsl_sf_coupling_3j(2*comb[i].l,2*rhs.comb[j].l,2*l,0,0,0);

	  // Add it to the list if the scaling factor is not zero
	  if(dc!=0)
	    newcomb.addylm(l,m,cscale(c,dc));
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

SphericalExpansion & SphericalExpansion::operator*=(complex fac) {
  // Scale the combination
  for(size_t i=0;i<comb.size();i++)
    comb[i].c=cmult(comb[i].c,fac);
  return *this;
}

SphericalExpansion & SphericalExpansion::operator*=(double fac) {
  // Scale the combination
  for(size_t i=0;i<comb.size();i++) {
    comb[i].c.im*=fac;
    comb[i].c.re*=fac;
  }
  return *this;
}

SphericalExpansion operator*(complex fac, const SphericalExpansion & func) {
  // Scale the combination
  SphericalExpansion ret=func;
  for(size_t i=0;i<ret.comb.size();i++)
    ret.comb[i].c=cmult(ret.comb[i].c,fac);
  return ret;
}

SphericalExpansion operator*(double fac, const SphericalExpansion & func) {
  // Scale the combination
  SphericalExpansion ret=func;
  for(size_t i=0;i<ret.comb.size();i++) {
    ret.comb[i].c.re*=fac;
    ret.comb[i].c.im*=fac;
  }
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
  if(lhs.getmaxl()>maxam || rhs.getmaxl()>maxam)
    throw std::domain_error("Multiplication table is not big enough for computing the wanted multiplication!\n");

  // Continue with multiplication. Loop over terms:
  for(size_t i=0;i<lhs.comb.size();i++)
    for(size_t j=0;j<rhs.comb.size();j++) {
      ret+=cmult(lhs.comb[i].c,rhs.comb[j].c)*table[multloc(lhs.comb[i].l,lhs.comb[i].m,rhs.comb[j].l,rhs.comb[j].m)];
    }

  return ret;
}

GTO_Fourier_Ylm SphericalExpansionMultiplicationTable::mult(const GTO_Fourier_Ylm & lhs, const GTO_Fourier_Ylm & rhs) const {
  // Calculate product of two expansions
  GTO_Fourier_Ylm ret;

  for(size_t i=0;i<lhs.sphexp.size();i++)
    for(size_t j=0;j<rhs.sphexp.size();j++)
      ret.addterm(mult(lhs.sphexp[i].ang,rhs.sphexp[j].ang),lhs.sphexp[i].pm+rhs.sphexp[j].pm,lhs.sphexp[i].z+rhs.sphexp[j].z);
  return ret;
}

GTO_Fourier_Ylm::GTO_Fourier_Ylm() {
}

GTO_Fourier_Ylm::GTO_Fourier_Ylm(int l, int m, int n, double zeta) {
  // The Fourier transform of the basis function
  GTO_Fourier transform(l,m,n,zeta);
  
  // Get the result of the Fourier transform
  std::vector<trans3d_t> trans(transform.get());

  // Compute spherical harmonics expansions of px^l, py^m and pz^n
  std::vector<SphericalExpansion> px, py, pz;

  px.resize(l+1);
  py.resize(m+1);
  pz.resize(n+1);

  // p_i^0 = 1 = \sqrt{4 \pi} Y_0^0
  px[0].addylm(0,0,sqrt(4.0*M_PI));
  py[0].addylm(0,0,sqrt(4.0*M_PI));
  pz[0].addylm(0,0,sqrt(4.0*M_PI));

  // px = p * sqrt{ 2 \pi / 3} * ( Y_1^{-1} - Y_1^1)
  if(l>0) {
    px[1].addylm(1,-1,sqrt(2.0*M_PI/3.0));
    px[1].addylm(1,1,-sqrt(2.0*M_PI/3.0));
  }
  // py = ip * sqrt{ 2 \pi / 3} * ( Y_1^{-1} + Y_1^1 )
  if(m>0) {
    complex hlp;
    hlp.re=0.0;
    hlp.im=sqrt(2.0*M_PI/3.0);
    py[1].addylm(1,-1,hlp);
    py[1].addylm(1,1,hlp);
  }
  // pz = p * sqrt{4 \pi / 3} Y_1^0
  if(n>0)
    pz[1].addylm(1,0,sqrt(4.0*M_PI/3.0));

  // Form the rest of the transforms
  for(int il=2;il<=l;il++)
    px[il]=px[il-1]*px[1];
  for(int im=2;im<=m;im++)
    py[im]=py[im-1]*py[1];
  for(int in=2;in<=n;in++)
    pz[in]=pz[in-1]*pz[1];

  // Now, add all necessary terms
  for(size_t i=0;i<trans.size();i++) {
    // Angular part is
    SphericalExpansion ang=trans[i].c*px[trans[i].l]*py[trans[i].m]*pz[trans[i].n];
    // Add the relevant term
    addterm(ang,trans[i].l+trans[i].m+trans[i].n,trans[i].z);
  }
}

GTO_Fourier_Ylm::~GTO_Fourier_Ylm() {
}

void GTO_Fourier_Ylm::addterm(const SphericalExpansion & ang, int pm, double z) {
  // Add term to contraction

  bool found=0;
  // See first, if there is a corresponding term yet
  for(size_t i=0;i<sphexp.size();i++)
    if(sphexp[i].pm==pm && sphexp[i].z==z) {
      sphexp[i].ang+=ang;
      found=1;
      break;
    }

  if(!found) {
    // Not on the list, add it

    GTO_Fourier_Ylm_t help;
    help.ang=ang;
    help.pm=pm;
    help.z=z;
   
    sphexp.push_back(help);
  }
}


void GTO_Fourier_Ylm::clean() {
  for(size_t i=sphexp.size()-1;i<sphexp.size();i--) {
    sphexp[i].ang.clean();
    if(sphexp[i].ang.getN()==0) {
      // Empty term, remove it.
      sphexp.erase(sphexp.begin()+i);
    }
  }
}


void GTO_Fourier_Ylm::print() const {
  for(size_t i=0;i<sphexp.size();i++) {
    printf("Term %lu: p^%i exp(-%e p^2), angular part\n",i,sphexp[i].pm,sphexp[i].z);
    sphexp[i].ang.print();
  }
}

GTO_Fourier_Ylm GTO_Fourier_Ylm::conjugate() const {
  // Returned combination
  GTO_Fourier_Ylm ret(*this);

  // Complex conjugate everything
  for(size_t i=0;i<ret.sphexp.size();i++)
    ret.sphexp[i].ang=ret.sphexp[i].ang.conjugate();

  return ret;
}

std::vector<GTO_Fourier_Ylm_t> GTO_Fourier_Ylm::getexp() const {
  return sphexp;
}

GTO_Fourier_Ylm GTO_Fourier_Ylm::operator*(const GTO_Fourier_Ylm & rhs) const {
  // Returned object
  GTO_Fourier_Ylm ret;

  // Do multiplication
  for(size_t i=0;i<sphexp.size();i++)
    for(size_t j=0;j<rhs.sphexp.size();j++) 
      ret.addterm(sphexp[i].ang*rhs.sphexp[j].ang,sphexp[i].pm+rhs.sphexp[j].pm,sphexp[i].z+rhs.sphexp[j].z);

  return ret;
}

GTO_Fourier_Ylm GTO_Fourier_Ylm::operator+(const GTO_Fourier_Ylm & rhs) const {
  // Returned combination
  GTO_Fourier_Ylm ret(*this);

  // Add terms
  for(size_t i=0;i<rhs.sphexp.size();i++)
    ret.addterm(rhs.sphexp[i].ang,rhs.sphexp[i].pm,rhs.sphexp[i].z);

  return ret;
}

GTO_Fourier_Ylm & GTO_Fourier_Ylm::operator+=(const GTO_Fourier_Ylm & rhs) {
  // Add terms
  for(size_t i=0;i<rhs.sphexp.size();i++)
    addterm(rhs.sphexp[i].ang,rhs.sphexp[i].pm,rhs.sphexp[i].z);

  return *this;
}

GTO_Fourier_Ylm operator*(complex fac, const GTO_Fourier_Ylm & func) {
  // Returned value
  GTO_Fourier_Ylm ret(func);

  for(size_t i=0;i<ret.sphexp.size();i++)
    ret.sphexp[i].ang*=fac;

  return ret;
}

GTO_Fourier_Ylm operator*(double fac, const GTO_Fourier_Ylm & func) {
  // Returned value
  GTO_Fourier_Ylm ret(func);

  for(size_t i=0;i<ret.sphexp.size();i++)
    ret.sphexp[i].ang*=fac;

  return ret;
}


CartesianExpansion::CartesianExpansion(int maxam) {
  // Amount of elements per side of table is
  N=maxam+1;
  // Reserve space for elements
  table.resize(N*N*N);

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
    px[1].addylm(1,-1,sqrt(2.0*M_PI/3.0));
    px[1].addylm(1,1,-sqrt(2.0*M_PI/3.0));
  }
  // py = ip * sqrt{ 2 \pi / 3} * ( Y_1^{-1} + Y_1^1 )
  if(maxam>0) {
    complex hlp;
    hlp.re=0.0;
    hlp.im=sqrt(2.0*M_PI/3.0);
    py[1].addylm(1,-1,hlp);
    py[1].addylm(1,1,hlp);
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

  // Fill table
  for(int l=0;l<=maxam;l++)
    for(int m=0;m<=maxam;m++)
      for(int n=0;n<=maxam;n++)
	table[ind(l,m,n)]=px[l]*py[m]*pz[n];
}

size_t CartesianExpansion::ind(int l, int m, int n) const {
  return (l*N+m)*N+n;
}

CartesianExpansion::~CartesianExpansion() {
}

SphericalExpansion CartesianExpansion::get(int l, int m, int n) const {
  return table[ind(l,m,n)];
}
