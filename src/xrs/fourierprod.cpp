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


#include "fourierprod.h"
#include "../lmgrid.h"
#include "../mathf.h"
#include "../emd/gto_fourier.h"

bool operator<(const prod_fourier_contr_t & lhs, const prod_fourier_contr_t & rhs) {
  // First, sort by l
  if(lhs.l<rhs.l)
    return 1;
  else if(lhs.l==rhs.l) {

    // Then, by m
    if(lhs.m<rhs.m)
      return 1;
    else if(lhs.m==rhs.m) {

      // Then, by n
      if(lhs.n<rhs.n)
	return 1;
    }
  }

  return 0;
}

bool operator==(const prod_fourier_contr_t & lhs, const prod_fourier_contr_t & rhs) {
  return (lhs.l==rhs.l) && (lhs.m==rhs.m) && (lhs.n==rhs.n);
}

bool operator<(const prod_fourier_t & lhs, const prod_fourier_t & rhs) {
  // First, sort by x coordinate of center
  // Sort first by center
  if(lhs.xp<rhs.xp)
    return 1;
  else if(lhs.xp == rhs.xp) {

    if(lhs.yp<rhs.yp)
      return 1;
    else if(lhs.yp == rhs.yp) {

      if(lhs.zp<rhs.zp)
        return 1;
      else if(lhs.zp==rhs.zp) {

        // Then, sort by exponent
        if(lhs.zeta<rhs.zeta)
          return 1;
      }
    }
  }

  return 0;
}

bool operator==(const prod_fourier_t & lhs, const prod_fourier_t & rhs) {
  return (lhs.xp==rhs.xp) && (lhs.yp==rhs.yp) && (lhs.zp==rhs.zp) && (lhs.zeta==rhs.zeta);
}

prod_fourier::prod_fourier() {
}

prod_fourier::prod_fourier(const prod_gaussian_3d & prod) {
  // Get the expansion from p
  std::vector<prod_gaussian_3d_t> expn=prod.get();

  // Loop over the centers in the product
  for(size_t i=0;i<expn.size();i++) {
    // New term to add in the total transform
    prod_fourier_t hlp;

    // Store the center
    hlp.xp=expn[i].xp;
    hlp.yp=expn[i].yp;
    hlp.zp=expn[i].zp;
    // and the exponent (take Fourier transform into account)
    hlp.zeta=1.0/(4.0*expn[i].zeta);

    // Form the Fourier polynomial at this center
    GTO_Fourier ft;
    for(size_t j=0;j<expn[i].c.size();j++) {
      GTO_Fourier term=GTO_Fourier(expn[i].c[j].l,expn[i].c[j].m,expn[i].c[j].n,expn[i].zeta);
      // GTO_Fourier places the normalization factor \f$ 1/(2 \pi)^{3/2} \f$, but it isn't needed here.
      ft+=(pow(2.0*M_PI,3.0/2.0)*expn[i].c[j].c)*term;
    }
    // Clean transform
    ft.clean();

    // Store the polynomial
    std::vector<trans3d_t> fterms=ft.get();
    for(size_t j=0;j<fterms.size();j++) {
      // Form term
      prod_fourier_contr_t tmp;
      tmp.l=fterms[j].l;
      tmp.m=fterms[j].m;
      tmp.n=fterms[j].n;
      tmp.c=fterms[j].c;

      // Add term to polynomial stack
      hlp.c.push_back(tmp);
    }

    // Add transform to stack
    p.push_back(hlp);
  }

  // We want to evaluate
  // \f$ \langle \mu | \exp i {\bf q} \cdot {\bf r} | \nu \rangle \f$,
  // but transform is wrt -q => do complex conjugate
  *this=conjugate();
}

prod_fourier::~prod_fourier() {
}

void prod_fourier::print() const {
  for(size_t i=0;i<p.size();i++) {
    printf("Fourier transform of function centered at (% e,% e,% e) with exponent %e (%e) is\n",p[i].xp,p[i].yp,p[i].zp,p[i].zeta,1.0/(4.0*p[i].zeta));
    for(size_t j=0;j<p[i].c.size();j++)
      printf(" (% e,% e) px^%i py^%i pz^%i\n",p[i].c[j].c.real(),p[i].c[j].c.imag(),p[i].c[j].l,p[i].c[j].m,p[i].c[j].n);
  }
}

prod_fourier prod_fourier::conjugate() const {
  // Returned object
  prod_fourier ret=*this;

  // Perform complex conjugation.
  for(size_t i=0;i<ret.p.size();i++) {
    // Phase factor changes sign.
    ret.p[i].xp*=-1.0;
    ret.p[i].yp*=-1.0;
    ret.p[i].zp*=-1.0;

    for(size_t j=0;j<ret.p[i].c.size();j++) {
      // Complex part of is also conjugated
      ret.p[i].c[j].c=std::conj(ret.p[i].c[j].c);
    }
  }

  return ret;
}

prod_fourier prod_fourier::operator*(const prod_fourier & rhs) const {
  // Returned object
  prod_fourier ret;

  // Loop over terms
  for(size_t i=0;i<p.size();i++)
    for(size_t j=0;j<rhs.p.size();j++) {

      // Term to add is
      prod_fourier term;
      prod_fourier_t hlp;

      // Phase factors are summed together
      hlp.xp=p[i].xp+rhs.p[j].xp;
      hlp.yp=p[i].yp+rhs.p[j].yp;
      hlp.zp=p[i].zp+rhs.p[j].zp;
      // and so are the exponents
      hlp.zeta=p[i].zeta+rhs.p[j].zeta;
      // Initialize term
      term.p.push_back(hlp);

      // Loop over cartesians
      for(size_t k=0;k<p[i].c.size();k++)
	for(size_t l=0;l<rhs.p[j].c.size();l++) {
	  // Resulting term is
	  prod_fourier_contr_t tmp;
	  tmp.l=p[i].c[k].l+rhs.p[j].c[l].l;
	  tmp.m=p[i].c[k].m+rhs.p[j].c[l].m;
	  tmp.n=p[i].c[k].n+rhs.p[j].c[l].n;
	  tmp.c=p[i].c[k].c*rhs.p[j].c[l].c;

	  // Add the term
	  term.add_contr(0,tmp);
	}

      // Add to result
      ret.add_term(term.p[0]);
    }

  return ret;
}

prod_fourier prod_fourier::operator*(double fac) const {
  prod_fourier ret(*this);

  for(size_t i=0;i<ret.p.size();i++)
    for(size_t j=0;j<ret.p[i].c.size();j++) {
      ret.p[i].c[j].c*=fac;
    }

  return ret;
}


prod_fourier & prod_fourier::operator+=(const prod_fourier & rhs) {
  for(size_t i=0;i<rhs.p.size();i++)
    add_term(rhs.p[i]);
  return *this;
}

void prod_fourier::add_term(const prod_fourier_t & t) {
  if(p.size()==0) {
    p.push_back(t);
  } else {
    // Get upper bound
    std::vector<prod_fourier_t>::iterator high;
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

void prod_fourier::add_contr(size_t ind, const prod_fourier_contr_t & t) {
  if(p[ind].c.size()==0) {
    p[ind].c.push_back(t);
  } else {

    // Get upper bound
    std::vector<prod_fourier_contr_t>::iterator hi;
    hi=std::upper_bound(p[ind].c.begin(),p[ind].c.end(),t);

    size_t indt=hi-p[ind].c.begin();
    if(indt>0 && p[ind].c[indt-1] == t)
      // Found it!
      p[ind].c[indt-1].c+=t.c;
    else
      // Need to add the term.
      p[ind].c.insert(hi,t);
  }
}

std::vector<prod_fourier_t> prod_fourier::get() const {
  return p;
}

std::complex<double> prod_fourier::eval(double qx, double qy, double qz) const {
  std::complex<double> res=0.0;

  // q squared
  double qsq=qx*qx+qy*qy+qz*qz;

  // Loop over terms
  for(size_t i=0;i<p.size();i++) {
    // Evaluate contraction
    std::complex<double> con=0.0;
    for(size_t j=0;j<p[i].c.size();j++)
      con+=p[i].c[j].c*pow(qx,p[i].c[j].l)*pow(qy,p[i].c[j].m)*pow(qz,p[i].c[j].n);

    // Argument of the exponential is
    std::complex<double> exparg(-qsq*p[i].zeta,-qx*p[i].xp -qy*p[i].yp -qz*p[i].zp);

    // Plug in exponential and phase factor
    res+=con*exp(exparg);
  }

  return res;
}

std::vector<prod_fourier> fourier_transform(const std::vector<prod_gaussian_3d> & prod) {
  std::vector<prod_fourier> ret(prod.size());
  for(size_t i=0;i<prod.size();i++)
    ret[i]=prod_fourier(prod[i]);
  return ret;
}

arma::cx_mat momentum_transfer(const std::vector<prod_fourier> & fprod, size_t Nbf, const arma::vec & q) {
  // Check that Nbf corresponds to fprod
  if(fprod.size() != ((Nbf*(Nbf+1))/2))
    throw std::runtime_error("Nbf does not correspond to size of fprod!\n");

  // Returned array
  arma::cx_mat ret(Nbf,Nbf);
  ret.zeros();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<Nbf;i++)
    for(size_t j=0;j<=i;j++) {
      std::complex<double> tmp=fprod[(i*(i+1))/2+j].eval(q(0),q(1),q(2));
      ret(i,j)=tmp;
      ret(j,i)=tmp;
    }

  return ret;
}
