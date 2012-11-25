/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
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
#include <cstdio>
#include "bfprod.h"
#include "mathf.h"

bool operator<(const prod_gaussian_1d_contr_t &lhs, const prod_gaussian_1d_contr_t &rhs) {
  return lhs.m < rhs.m;
}

bool operator==(const prod_gaussian_1d_contr_t &lhs, const prod_gaussian_1d_contr_t &rhs) {
  return lhs.m == rhs.m;
}

bool operator<(const prod_gaussian_1d_t & lhs, const prod_gaussian_1d_t & rhs) {
  // Sort first by center
  if(lhs.xp<rhs.xp)
    return 1;
  else if(lhs.xp==rhs.xp) {
    // Then, sort by exponent
    if(lhs.zeta<rhs.zeta)
      return 1;
    else if(lhs.zeta==rhs.zeta) {
      // Last, sort by polynomials
      if(lhs.c[0].m<rhs.c[0].m)
	return 1;
    }
  }

  return 0;
}

bool operator==(const prod_gaussian_1d_t & lhs, const prod_gaussian_1d_t & rhs) {
  return (lhs.xp==rhs.xp) && (lhs.zeta==rhs.zeta);
}

prod_gaussian_1d::~prod_gaussian_1d() {
}

prod_gaussian_1d::prod_gaussian_1d(double xa, double xb, int la, int lb, double zetaa, double zetab) {
  // Helper to push to stack
  prod_gaussian_1d_t hlp;

  // Compute reduced exponents zeta and eta
  hlp.zeta=zetaa+zetab;
  double eta=zetaa*zetab/hlp.zeta;

  // Center of product Gaussian is at
  hlp.xp=(zetaa*xa+zetab*xb)/hlp.zeta;

  // Add term to list
  p.push_back(hlp);

  // Global scale factor is
  double scale=exp(-eta*(xa-xb)*(xa-xb));

  // Combinatorial factors
  double afac[la+1];
  for(int ia=0;ia<=la;ia++)
    afac[ia]=choose(la,ia)*pow(hlp.xp-xa,la-ia);

  double bfac[lb+1];
  for(int ib=0;ib<=lb;ib++)
    bfac[ib]=choose(lb,ib)*pow(hlp.xp-xb,lb-ib);

  // Generate products
  for(int ia=0;ia<=la;ia++)
    for(int ib=0;ib<=lb;ib++) {
      prod_gaussian_1d_contr_t tmp;
      tmp.m=ia+ib;
      tmp.c=scale*afac[ia]*bfac[ib];

      add_contr(0,tmp);
    }
}

prod_gaussian_1d prod_gaussian_1d::operator+(const prod_gaussian_1d & rhs) const {
  // Returned value
  prod_gaussian_1d ret(*this);

  // Add rhs terms
  for(size_t i=0;i<rhs.p.size();i++)
    ret.add_term(rhs.p[i]);

  return ret;
}

prod_gaussian_1d & prod_gaussian_1d::operator+=(const prod_gaussian_1d & rhs) {
  // Add rhs terms
  for(size_t i=0;i<rhs.p.size();i++)
    add_term(rhs.p[i]);

  return *this;
}

void prod_gaussian_1d::add_term(const prod_gaussian_1d_t & t) {
  if(p.size()==0) {
    p.push_back(t);
  } else {
    // Get upper bound
    std::vector<prod_gaussian_1d_t>::iterator high;
    high=std::upper_bound(p.begin(),p.end(),t);

    // Corresponding index is
    size_t ind=high-p.begin();

    if(ind>0 && p[ind-1]==t)
      // Need to find place in p[ind-1] to add terms.
      // Loop over terms in t
      for(size_t it=0;it<t.c.size();it++)
	add_contr(ind-1,t.c[it]);
    else {
      // Term does not exist, add it
      p.insert(high,t);
    }
  }
}

void prod_gaussian_1d::add_contr(size_t ind, const prod_gaussian_1d_contr_t & t) {
  if(p[ind].c.size()==0) {
    p[ind].c.push_back(t);
  } else {

    // Get upper bound
    std::vector<prod_gaussian_1d_contr_t>::iterator hi;
    hi=std::upper_bound(p[ind].c.begin(),p[ind].c.end(),t);

    size_t indt=hi-p[ind].c.begin();
    if(indt>0 && p[ind].c[indt-1] == t)
      // Found it!
      p[ind].c[indt-1].c += t.c;
    else
      // Need to add the term.
      p[ind].c.insert(hi,t);
  }
}

void prod_gaussian_1d::print() const {
  for(size_t i=0;i<p.size();i++) {
    printf("Product gaussian at %e with exponent %e, contains %i terms:\n",p[i].xp,p[i].zeta, (int) p[i].c.size());

    for(size_t j=0;j<p[i].c.size();j++)
      printf(" %+e x^%i",p[i].c[j].c,p[i].c[j].m);
    printf("\n");
  }
}

std::vector<prod_gaussian_1d_t> prod_gaussian_1d::get() const {
  return p;
}

bool operator<(const prod_gaussian_3d_contr_t &lhs, const prod_gaussian_3d_contr_t &rhs) {
  // Sort first by total am
  if(lhs.l+lhs.m+lhs.n < rhs.l+rhs.m+rhs.n)
    return 1;
  else if(lhs.l+lhs.m+lhs.n == rhs.l+rhs.m+rhs.n) {

    // Then, by l
    if(lhs.l < rhs.l)
      return 1;
    else if(lhs.l == rhs.l) {
      // Then, by m
      if(lhs.m < rhs.m)
	return 1;
      else if(lhs.m == rhs.m) {
	// Then, by n
	return lhs.n<rhs.n;
      }
    }
  }

  return 0;
}

bool operator==(const prod_gaussian_3d_contr_t &lhs, const prod_gaussian_3d_contr_t &rhs) {
  return (lhs.l == rhs.l) && (lhs.m == rhs.m) && (lhs.n == rhs.n);
}

bool operator<(const prod_gaussian_3d_t & lhs, const prod_gaussian_3d_t & rhs) {
  // Sort first by exponent
  if(lhs.zeta<rhs.zeta)
    return 1;
  else if(lhs.zeta==rhs.zeta) {

    // then by center
    if(lhs.xp<rhs.xp)
      return 1;
    else if(lhs.xp == rhs.xp) {

      if(lhs.yp<rhs.yp)
	return 1;
      else if(lhs.yp == rhs.yp) {

	if(lhs.zp<rhs.zp)
	  return 1;
	else if(lhs.zp==rhs.zp) {

	  // Last, sort by polynomials
	  if(lhs.c[lhs.c.size()-1].l+lhs.c[lhs.c.size()-1].m+lhs.c[lhs.c.size()-1].n<rhs.c[rhs.c.size()-1].l+rhs.c[rhs.c.size()-1].m+rhs.c[rhs.c.size()-1].n)
	    return 1;
	}
      }
    }
  }

  return 0;
}

bool operator==(const prod_gaussian_3d_t & lhs, const prod_gaussian_3d_t & rhs) {
  return (lhs.xp==rhs.xp) && (lhs.yp==rhs.yp) && (lhs.zp==rhs.zp) && (lhs.zeta==rhs.zeta);
}

prod_gaussian_3d::prod_gaussian_3d() {
}

prod_gaussian_3d::~prod_gaussian_3d() {
}

prod_gaussian_3d::prod_gaussian_3d(double xa, double xb, double ya, double yb, double za, double zb, int la, int lb, int ma, int mb, int na, int nb, double zetaa, double zetab) {
  // Form 1d transforms
  prod_gaussian_1d xp(xa,xb,la,lb,zetaa,zetab);
  prod_gaussian_1d yp(ya,yb,ma,mb,zetaa,zetab);
  prod_gaussian_1d zp(za,zb,na,nb,zetaa,zetab);

  // and get them
  std::vector<prod_gaussian_1d_t> x=xp.get();
  std::vector<prod_gaussian_1d_t> y=yp.get();
  std::vector<prod_gaussian_1d_t> z=zp.get();

  // Initialize list:
  prod_gaussian_3d_t hlp;
  // Get center. We only have a single set of coordinates and a single
  // exponent, so the size of x, y and z is always 1.
  hlp.xp=x[0].xp;
  hlp.yp=y[0].xp;
  hlp.zp=z[0].xp;
  // and reduced exponent
  hlp.zeta=zetaa+zetab;
  // Add to stack
  p.push_back(hlp);

  // Now, add the terms in the product.
  for(size_t i=0;i<x[0].c.size();i++)
    for(size_t j=0;j<y[0].c.size();j++)
      for(size_t k=0;k<z[0].c.size();k++) {
	prod_gaussian_3d_contr_t tmp;
	tmp.c=x[0].c[i].c*y[0].c[j].c*z[0].c[k].c;
	// Angular moment
	tmp.l=x[0].c[i].m;
	tmp.m=y[0].c[j].m;
	tmp.n=z[0].c[k].m;

	add_contr(0,tmp);
      }
}


prod_gaussian_3d prod_gaussian_3d::operator+(const prod_gaussian_3d & rhs) const {
  // Returned value
  prod_gaussian_3d ret(*this);
  ret+=rhs;
  return ret;
}

prod_gaussian_3d & prod_gaussian_3d::operator+=(const prod_gaussian_3d & rhs) {
  // Add rhs terms
  for(size_t i=0;i<rhs.p.size();i++)
    add_term(rhs.p[i]);

  return *this;
}

prod_gaussian_3d prod_gaussian_3d::operator*(double fac) const {
  prod_gaussian_3d ret=*this;

  for(size_t i=0;i<ret.p.size();i++)
    for(size_t j=0;j<ret.p[i].c.size();j++)
      ret.p[i].c[j].c*=fac;

  return ret;
}

void prod_gaussian_3d::add_term(const prod_gaussian_3d_t & t) {
  if(p.size()==0) {
    p.push_back(t);
  } else {
    // Get upper bound
    std::vector<prod_gaussian_3d_t>::iterator high;
    high=std::upper_bound(p.begin(),p.end(),t);

    // Corresponding index is
    size_t ind=high-p.begin();

    if(ind>0 && p[ind-1]==t)
      // Need to find place in p[ind-1] to add terms.
      // Loop over terms in t
      for(size_t it=0;it<t.c.size();it++)
	add_contr(ind-1,t.c[it]);
    else {
      // Term does not exist, add it
      p.insert(high,t);
    }
  }
}

void prod_gaussian_3d::add_contr(size_t ind, const prod_gaussian_3d_contr_t & t) {
  if(p[ind].c.size()==0) {
    p[ind].c.push_back(t);
  } else {

    // Get upper bound
    std::vector<prod_gaussian_3d_contr_t>::iterator hi;
    hi=std::upper_bound(p[ind].c.begin(),p[ind].c.end(),t);

    size_t indt=hi-p[ind].c.begin();
    if(indt>0 && p[ind].c[indt-1] == t)
      // Found it!
      p[ind].c[indt-1].c += t.c;
    else
      // Need to add the term.
      p[ind].c.insert(hi,t);
  }
}

void prod_gaussian_3d::clean() {
  for(size_t i=0;i<p.size();i++)
    for(size_t j=p[i].c.size()-1;j<p[i].c.size();j--)
      if(p[i].c[j].c==0.0) {
	//	printf("Erasing p[%i].c[%i], for which c=%e.\n",i,j,p[i].c[j].c);
	p[i].c.erase(p[i].c.begin()+j);
      }
}

double prod_gaussian_3d::integral() const {
  double res=0.0;
  for(size_t i=0;i<p.size();i++) { // Loop over centers
    // Exponent is
    double zeta=p[i].zeta;

    // Loop over contraction
    for(size_t j=0;j<p[i].c.size();j++) {
      // Exponents are
      int l=p[i].c[j].l;
      int m=p[i].c[j].m;
      int n=p[i].c[j].n;

      // Check that these are even
      if((l%2==1) || (m%2==1) || (n%2==1))
	continue;

      // Get the halves
      int lh=l/2;
      int mh=m/2;
      int nh=n/2;

      // Contraction coefficient is
      double c=p[i].c[j].c;

      // The contribution is
      res+=c*pow(M_PI,3.0/2.0)*doublefact(l-1)*doublefact(m-1)*doublefact(n-1)*pow(2.0,-(lh+mh+nh))/(pow(zeta,lh+mh+nh)*pow(sqrt(zeta),3));
    }
  }

  return res;
}

std::vector<prod_gaussian_3d_t> prod_gaussian_3d::get() const {
  return p;
}

void prod_gaussian_3d::print() const {
  for(size_t i=0;i<p.size();i++) {
    printf("Product gaussian at (% e,% e,% e) with exponent %e, contains %i terms:\n",p[i].xp,p[i].yp,p[i].zp,p[i].zeta,(int) p[i].c.size());
    for(size_t j=0;j<p[i].c.size();j++)
      printf("\t%+e x^%i y^%i z^%i\n",p[i].c[j].c,p[i].c[j].l,p[i].c[j].m,p[i].c[j].n);
  }
}


std::vector<prod_gaussian_3d> compute_product(const BasisSet & bas, size_t is, size_t js) {
  // Contractions on shells
  std::vector<contr_t> icontr=bas.get_contr(is);
  std::vector<contr_t> jcontr=bas.get_contr(js);

  // Cartesian functions on shells
  std::vector<shellf_t> icart=bas.get_cart(is);
  std::vector<shellf_t> jcart=bas.get_cart(js);

  // Centers of shells
  coords_t icen=bas.get_center(is);
  coords_t jcen=bas.get_center(js);

  // Returned array
  std::vector<prod_gaussian_3d> ret;
  ret.reserve(icart.size()*jcart.size());

  // Form products
  for(size_t ii=0;ii<icart.size();ii++)
    for(size_t jj=0;jj<jcart.size();jj++) {

      // Result;
      prod_gaussian_3d tmp;

      // Loop over exponents
      for(size_t ix=0;ix<icontr.size();ix++)
	for(size_t jx=0;jx<jcontr.size();jx++) {
	  // Compute product
	  prod_gaussian_3d term(icen.x,jcen.x,icen.y,jcen.y,icen.z,jcen.z,icart[ii].l,jcart[jj].l,icart[ii].m,jcart[jj].m,icart[ii].n,jcart[jj].n,icontr[ix].z,jcontr[jx].z);
	  // Add to partial result
	  tmp+=term*(icontr[ix].c*jcontr[jx].c);

	  /*
	  printf("Product gaussian of (%i,%i,%i) centered at (% e,% e,% e) and (%i,%i,%i) centered at (% e,% e,% e) is\n",icart[ii].l,icart[ii].m,icart[ii].n,icen.x,icen.y,icen.z,jcart[jj].l,jcart[jj].m,jcart[jj].n,jcen.x,jcen.y,jcen.z);
	  term.print();
	  */
	}

      // Plug in normalization factors
      tmp=tmp*(icart[ii].relnorm*jcart[jj].relnorm);

      // Add to stack
      ret.push_back(tmp);
    }

  // Transform into spherical basis if necessary
  if(bas.lm_in_use(is) || bas.lm_in_use(js))
    return spherical_transform(bas,is,js,ret);
  else {
    // Clean out terms with zero contribution
    for(size_t i=0;i<ret.size();i++)
      ret[i].clean();
    // Return result
    return ret;
  }
}

std::vector<prod_gaussian_3d> spherical_transform(const BasisSet & bas, size_t is, size_t js, std::vector<prod_gaussian_3d> & res) {
  bool lm_i=bas.lm_in_use(is);
  bool lm_j=bas.lm_in_use(js);

  const size_t Ni_cart=bas.get_Ncart(is);
  const size_t Nj_cart=bas.get_Ncart(js);

  const size_t Ni_tgt=bas.get_Nbf(is);
  const size_t Nj_tgt=bas.get_Nbf(js);

  // First, transform over j. Helper array
  std::vector<prod_gaussian_3d> tmp(Ni_cart*Nj_tgt);

  if(lm_j) {
    // Get transformation matrix
    arma::mat trans_j=bas.get_trans(js);

    // Loop over functions
    for(size_t iic=0;iic<Ni_cart;iic++)
      for(size_t jjs=0;jjs<Nj_tgt;jjs++)
	for(size_t jjc=0;jjc<Nj_cart;jjc++)
	  tmp[iic*Nj_tgt+jjs]+=res[iic*Nj_cart+jjc]*trans_j(jjs,jjc);
  } else
    // No transformation necessary.
    tmp=res;

  if(lm_i) {
    // Get transformation matrix
    arma::mat trans_i=bas.get_trans(is);

    // Resize output vector
    res.resize(Ni_tgt*Nj_tgt);

    // Loop over functions
    for(size_t jjs=0;jjs<Nj_tgt;jjs++)
      for(size_t iis=0;iis<Ni_tgt;iis++) {
	// Clean output
	res[iis*Nj_tgt+jjs]=prod_gaussian_3d();
	// Compute transform
	for(size_t iic=0;iic<Ni_cart;iic++)
	  res[iis*Nj_tgt+jjs]+=tmp[iic*Nj_tgt+jjs]*trans_i(iis,iic);
      }

    // Clean out terms with zero contribution
    for(size_t i=0;i<res.size();i++)
      res[i].clean();
    return res;
  } else {
    // No transformation necessary.
    for(size_t i=0;i<tmp.size();i++)
      tmp[i].clean();
    return tmp;
  }
}


std::vector<prod_gaussian_3d> compute_products(const BasisSet & bas) {
  // Amount of basis functions is
  size_t Nbf=bas.get_Nbf();

  // .. so the size of the returned array is
  std::vector<prod_gaussian_3d> ret(Nbf*(Nbf+1)/2);

  // Get shells in the basis set.
  std::vector<GaussianShell> shells=bas.get_shells();

  // Amount of functions on shells
  std::vector<size_t> nbf(shells.size());
  for(size_t is=0;is<shells.size();is++)
    nbf[is]=shells[is].get_Nbf();

  // First functions on shells
  std::vector<size_t> ind0(shells.size());
  for(size_t is=0;is<shells.size();is++)
    ind0[is]=shells[is].get_first_ind();


  // Form products
  for(size_t is=0;is<shells.size();is++) {
    // Do off-diagonal first
    for(size_t js=0;js<is;js++) {
      // Compute products
      std::vector<prod_gaussian_3d> prod=compute_product(bas,is,js);

      // Store output
      for(size_t ii=0;ii<nbf[is];ii++)
	for(size_t jj=0;jj<nbf[js];jj++) {
	  size_t i=ind0[is]+ii;
	  size_t j=ind0[js]+jj;

	  // Since is>js we know i>j
	  ret[(i*(i+1))/2 + j]=prod[ii*nbf[js]+jj];
	}
    }

    // Then, do diagonal
    std::vector<prod_gaussian_3d> prod=compute_product(bas,is,is);
    for(size_t ii=0;ii<nbf[is];ii++)
      for(size_t jj=0;jj<=ii;jj++) {
	size_t i=ind0[is]+ii;
	size_t j=ind0[is]+jj;

	// Here we know as well that i>=j
	ret[(i*(i+1))/2 + j]=prod[ii*nbf[is]+jj];
      }
  }

  return ret;
}
