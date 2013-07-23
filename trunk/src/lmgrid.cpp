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
#include "lmgrid.h"

// Compute values of spherical harmonics
#include "spherical_harmonics.h"
#include "timer.h"

// Check orthogonality of spherical harmonics
//#define CHECKORTHO
// Tolerance for deviation from orthonomality
#define ORTHTOL 50.0*DBL_EPSILON

std::vector<radial_grid_t> form_radial_grid(int nrad) {
  // Returned array
  std::vector<radial_grid_t> ret(nrad);

  // Get Chebyshev nodes and weights
  std::vector<double> xc, wc;
  chebyshev(nrad,xc,wc);

  // Reverse xc and wc so radii are in increasing order.
  std::reverse(xc.begin(),xc.end());
  std::reverse(wc.begin(),wc.end());

  // Loop over radii
  double jac;
  for(size_t ir=0;ir<xc.size();ir++) {
    // Calculate value of radius
    ret[ir].r=1.0/M_LN2*log(2.0/(1.0-xc[ir]));

    // Jacobian of transformation is
    jac=1.0/M_LN2/(1.0-xc[ir]);
    // so total quadrature weight is
    ret[ir].w=wc[ir]*ret[ir].r*ret[ir].r*jac;
  }


  /*
  // Let maximum distance be 10 Å
  const double maxr=18.897261;
  double dr=maxr/(nrad-1);
  for(int i=0;i<nrad;i++) {
    ret[i].r=i*dr;
    ret[i].w=dr*ret[i].r*ret[i].r;
  };
  ret[0].w/=2.0;
  ret[nrad-1].w/=2.0;
  */

  return ret;
}

std::vector<angular_grid_t> form_angular_grid(int lmax) {
  // Number of points in theta
  int nth=(lmax+3)/2;

  // Get corresponding Lobatto quadrature rule points in theta
  std::vector<double> xl, wl;
  lobatto_compute(nth,xl,wl);

  // Form quadrature points
  std::vector<angular_grid_t> grid;
  for(int ith=0;ith<nth;ith++) {
    // Compute cos(th) and sin(th);

    double cth=xl[ith];
    double sth=sqrt(1-cth*cth);
    // Determine number of points in phi, defined by smallest integer for which
    // sin^{nphi} \theta < THR
    double thr;
    if(lmax<=50)
      thr=1e-8;
    else
      thr=1e-9;

    // Calculate nphi
    int nphi=1;
    double t=sth;
    while(t>=thr && nphi<=lmax+1) {
      nphi++;
      t*=sth;
    }

    // Use an offset in phi?
    double phioff=0.0;
    if(ith%2)
      phioff=M_PI/nphi;

    // Now, generate the points.
    angular_grid_t point;
    double phi, dphi; // Value of phi and the increment
    double cph, sph; // Sine and cosine of phi

    dphi=2.0*M_PI/nphi;

    // Angular weight of points on this ring is
    point.w=2.0*M_PI*wl[ith]/nphi;

    for(int iphi=0;iphi<nphi;iphi++) {
      // Value of phi is
      phi=iphi*dphi+phioff;
      // and the sine and cosine are
      sph=sin(phi);
      cph=cos(phi);

      // Compute x, y and z
      point.r.x=sth*cph;
      point.r.y=sth*sph;
      point.r.z=cth;

      // Add point
      grid.push_back(point);
    }
  }

  return grid;
}

std::vector< std::vector< std::complex<double> > > compute_spherical_harmonics(const std::vector<angular_grid_t> & grid, int lmax) {
  // Values of spherical harmonics: Ylm[ngrid][l,m]
  std::vector< std::vector< std::complex<double> > > Ylm;
  Ylm.resize(grid.size());
  for(size_t i=0;i<grid.size();i++)
      Ylm[i].resize(lmind(lmax,lmax)+1);

  // Loop over grid points
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t i=0;i<grid.size();i++) {
    // Compute phi and cos(theta)
    double phi=atan2(grid[i].r.y,grid[i].r.x);
    double cth=grid[i].r.z;

    // Compute spherical harmonics
    for(int l=0;l<=lmax;l++)
      for(int m=-l;m<=l;m++)
	Ylm[i][lmind(l,m)]=spherical_harmonics(l,m,cth,phi);
  }

  return Ylm;
}

// Compute expansion of orbitals around cen, return clm[norbs][l,m][nrad]
expansion_t expand_orbitals(const arma::mat & C, const BasisSet & bas, const coords_t & cen, size_t Nrad, int lmax, int lquad) {

  // Returned expansion
  expansion_t ret;

  Timer t;

  // Form angular grid
  std::vector<angular_grid_t> grid=form_angular_grid(lquad);

  // Compute values of spherical harmonics and complex conjugate them
  std::vector< std::vector< std::complex<double> > > Ylm_conj=compute_spherical_harmonics(grid,lmax);
  for(size_t i=0;i<Ylm_conj.size();i++) {
    for(size_t lm=0;lm<Ylm_conj[i].size();lm++)
      Ylm_conj[i][lm].imag()*=-1.0;
  }

  printf("Formed angular grid and computed spherical harmonics in %s.\n",t.elapsed().c_str());
  t.set();

#ifdef CHECKORTHO
  // Check orthogonality of spherical harmonics

  size_t nfail=0, nsucc=0;

  for(int l1=0;l1<=lmax;l1++)
    for(int m1=-l1;m1<=l1;m1++)
      for(int l2=0;l2<=l1;l2++)
	for(int m2=-l2;m2<=l2;m2++) {
	  // Perform quadrature over the sphere
	  std::complex<double> res=0.0;

	  for(size_t ip=0;ip<grid.size();ip++)
	    res+=grid[ip].w*Ylm_conj[ip][lmind(l2,m2)]*conj(Ylm_conj[ip][lmind(l1,m1)]);

	  //	  printf("(%i,%i) - (%i,%i) -> (%e,%e)\n",l1,m1,l2,m2,res.real(),res.imag());
	  if( (l1==l2) && (m1==m2) ) {
	    if (fabs(abs(res)-1.0)>ORTHTOL ) {
	      printf("(%i,%i) - (%i,%i) -> (%e,%e)\n",l1,m1,l2,m2,res.real(),res.imag());
	      nfail++;
	    } else
	      nsucc++;
	  } else {
	    if (abs(res)>ORTHTOL ) {
	      printf("(%i,%i) - (%i,%i) -> (%e,%e)\n",l1,m1,l2,m2,res.real(),res.imag());
	      nfail++;
	    } else
	      nsucc++;
	  }
	}

  printf("Checked the orthonormality of spherical harmonics: %lu succ, %lu fail.\n",nsucc,nfail);

#endif

  // Form radial grid
  ret.grid=form_radial_grid(Nrad);

  // Coefficients of expansion
  ret.clm.resize(C.n_cols);
  for(size_t iorb=0;iorb<C.n_cols;iorb++) {
    ret.clm[iorb].resize(Ylm_conj[0].size());
    for(size_t lm=0;lm<Ylm_conj[0].size();lm++) {
      ret.clm[iorb][lm].resize(Nrad);
      for(size_t irad=0;irad<Nrad;irad++)
	ret.clm[iorb][lm][irad]=0.0;
    }
  }

  // Loop over radii
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t irad=0;irad<ret.grid.size();irad++) {
    // Loop over angular grid
    for(size_t iang=0;iang<grid.size();iang++) {

      // Compute coordinates of grid point
      coords_t gp=grid[iang].r*ret.grid[irad].r+cen;
      // Evaluate values of orbitals at grid point
      arma::vec orbs=compute_orbitals(C,bas,gp);

      // Do quadrature step
      for(size_t lm=0;lm<Ylm_conj[iang].size();lm++)
	for(size_t iorb=0;iorb<orbs.n_elem;iorb++)
	  ret.clm[iorb][lm][irad]+=orbs(iorb)*grid[iang].w*Ylm_conj[iang][lm];
    }
  }

  printf("Computed spherical harmonics expansion of orbitals in %s.\n",t.elapsed().c_str());

  return ret;
}



