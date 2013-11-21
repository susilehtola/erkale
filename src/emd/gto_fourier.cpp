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

std::vector< std::vector<GTO_Fourier> > fourier_expand(const BasisSet & bas, std::vector< std::vector<size_t> > & idents) {
  // Find out identical shells in basis set.
  idents=bas.find_identical_shells();

  // Compute the expansions of the non-identical shells
  std::vector< std::vector<GTO_Fourier> > fourier;
  for(size_t i=0;i<idents.size();i++) {
    // Get exponents, contraction coefficients and cartesians
    std::vector<contr_t> contr=bas.get_contr(idents[i][0]);
    std::vector<shellf_t> cart=bas.get_cart(idents[i][0]);

    // Compute expansion of basis functions on shell
    // Form expansions of cartesian functions
    std::vector<GTO_Fourier> cart_expansion;
    for(size_t icart=0;icart<cart.size();icart++) {
      // Expansion of current function
      GTO_Fourier func;
      for(size_t iexp=0;iexp<contr.size();iexp++)
        func+=contr[iexp].c*GTO_Fourier(cart[icart].l,cart[icart].m,cart[icart].n,contr[iexp].z);
      // Plug in the normalization factor
      func=cart[icart].relnorm*func;
      // Clean out terms with zero contribution
      func.clean();
      // Add to cartesian expansion
      cart_expansion.push_back(func);
    }

    // If spherical harmonics are used, we need to transform the
    // functions into the spherical harmonics basis.
    if(bas.lm_in_use(idents[i][0])) {
      std::vector<GTO_Fourier> sph_expansion;
      // Get transformation matrix
      arma::mat transmat=bas.get_trans(idents[i][0]);
      // Form expansion
      int l=bas.get_am(idents[i][0]);
      for(int m=-l;m<=l;m++) {
        // Expansion for current term
        GTO_Fourier mcomp;
        // Form expansion
        for(size_t icart=0;icart<transmat.n_cols;icart++)
          mcomp+=transmat(l+m,icart)*cart_expansion[icart];
        // clean it
        mcomp.clean();
        // and add it to the stack
        sph_expansion.push_back(mcomp);
      }
      // Now we have all components, add everything to the stack
      fourier.push_back(sph_expansion);
    } else
      // No need to transform, cartesians are used.
      fourier.push_back(cart_expansion);
  }

  return fourier;
}

double eval_emd(const BasisSet & bas, const arma::mat & P, const std::vector< std::vector<GTO_Fourier> > & fourier, const std::vector< std::vector<size_t> > & idents, double px, double py, double pz) {
  if(fourier.size() != idents.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Error - size of fourier array " << fourier.size() << " does not match that of idents " << idents.size() << "!\n";
    throw std::runtime_error(oss.str());
  }

  // Values of the Fourier polynomial part of the basis functions: [nident][nfuncs]
  std::vector< std::vector< std::complex<double> > > fpoly(fourier.size());
  for(size_t i=0;i<fourier.size();i++)
    fpoly[i].resize(fourier[i].size());
  
  // Amount of basis functions
  const size_t Nbf=bas.get_Nbf();
  // Values of the basis functions, i.e. the above with the additional phase factor
  std::vector< std::complex<double> > fvals(Nbf);
  
  // Compute values of Fourier polynomials at current value of p.
  for(size_t iid=0;iid<fourier.size();iid++)
    // Loop over the functions on the identical shells.
    for(size_t fi=0;fi<fourier[iid].size();fi++)
      fpoly[iid][fi]=fourier[iid][fi].eval(px,py,pz);
  
  // Compute the values of the basis functions themselves.
  // Loop over list of groups of identical shells
  for(size_t ii=0;ii<idents.size();ii++)
    // and over the shells of this type
    for(size_t jj=0;jj<idents[ii].size();jj++) {
      // The current shell is
      size_t is=idents[ii][jj];
      // and it is centered at
      coords_t cen=bas.get_shell_center(is);
      // thus the phase factor we get is
      std::complex<double> phase=exp(std::complex<double>(0.0,-(px*cen.x+py*cen.y+pz*cen.z)));
      
      // Now we just store the individual function values.
      size_t i0=bas.get_first_ind(is);
      size_t Ni=bas.get_Nbf(is);
      for(size_t fi=0;fi<Ni;fi++)
	fvals[i0+fi]=phase*fpoly[ii][fi];
    }
  
  // and now it's only a simple matter to compute the momentum density.
  double emd=0.0;
  for(size_t i=0;i<Nbf;i++) {
    // Off-diagonal
    for(size_t j=0;j<i;j++)
      emd+=2.0*P(i,j)*std::real(std::conj(fvals[i])*fvals[j]);
    // Diagonal
    emd+=P(i,i)*std::norm(fvals[i]);
  }
  
  return emd;
}
