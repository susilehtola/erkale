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

#include "emdcube.h"
#include "gto_fourier.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void emd_cube(const BasisSet & bas, const arma::mat & P, const std::vector<double> & px_arr, const std::vector<double> & py_arr, const std::vector<double> & pz_arr) {
  // Find out identical shells in basis set.
  std::vector< std::vector<size_t> > idents=bas.find_identical_shells();

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

  // Open output file.
  FILE *out=fopen("emdcube.dat","w");

  // Compute the norm (assumes evenly spaced grid!)
  double norm=0.0;

  // Compute the momentum densities in batches, allowing
  // parallellization.
#ifdef _OPENMP
  // The number of used threads
  const int nth=omp_get_max_threads();
  // The number of points per batch
  const size_t Nbatch_p=100*nth;
#else
  const int nth=1;
  const size_t Nbatch_p=100;
#endif

  // The total number of point is
  const size_t N=px_arr.size()*py_arr.size()*pz_arr.size();
  // The necessary amount of batches is
  size_t Nbatch=N/Nbatch_p;
  if(N%Nbatch_p!=0)
    Nbatch++;

  // The values of momentum in the batch
  coords_t p[Nbatch_p];
  // The values of EMD in the batch
  double emd[Nbatch_p];

  // Number of points to compute in the batch
  size_t np;
  // Total number of points computed
  size_t ntot=0;
  // Indices of x, y and z
  size_t xind=0, yind=0, zind=0;

  // Values of the Fourier polynomial part of the basis functions: [nth][nident][nfuncs]
  std::vector< std::vector< std::vector< std::complex<double> > > > fpoly(nth);
  for(int ith=0;ith<nth;ith++) {
    fpoly[ith].resize(idents.size());
    for(size_t i=0;i<fourier.size();i++)
      fpoly[ith][i].resize(fourier[i].size());
  }

  // Amount of basis functions
  const size_t Nbf=bas.get_Nbf();
  // Values of the basis functions, i.e. the above with the additional phase factor
  std::vector< std::vector< std::complex<double> > > fvals(nth);
  for(int ith=0;ith<nth;ith++)
    fvals[ith].resize(Nbf);

  // Loop over batches.
  for(size_t ib=0;ib<Nbatch;ib++) {
    // Zero amount of points in current batch.
    np=0;

    // Form list of points to compute.
    while(np<Nbatch_p && ntot+np<N) {
      p[np].x=px_arr[xind];
      p[np].y=py_arr[yind];
      p[np].z=pz_arr[zind];

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<pz_arr.size())
	zind++;
      else {
	zind=0;

	if(yind+1<py_arr.size())
	  yind++;
	else {
	  yind=0;
	  xind++;
	}
      }
    }

    // Begin parallel section
#ifdef _OPENMP
#pragma omp parallel
#endif
    {

#ifdef _OPENMP
      // Get thread index
      int ith=omp_get_thread_num();
#else
      int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for
#endif
      // Loop over the points in the batch
      for(size_t ip=0;ip<np;ip++) {
	// Current value of p is
	double px=p[ip].x;
	double py=p[ip].y;
	double pz=p[ip].z;

	// Compute values of Fourier polynomials at current value of p.
	for(size_t iid=0;iid<idents.size();iid++)
	  // Loop over the functions on the identical shells.
	  for(size_t fi=0;fi<fourier[iid].size();fi++)
	    fpoly[ith][iid][fi]=fourier[iid][fi].eval(px,py,pz);

	// Compute the values of the basis functions themselves.
	// Loop over list of groups of identical shells
	for(size_t ii=0;ii<idents.size();ii++)
	  // and over the shells of this type
	  for(size_t jj=0;jj<idents[ii].size();jj++) {
	    // The current shell is
	    size_t is=idents[ii][jj];
	    // and it is centered at
	    coords_t cen=bas.get_center(is);
	    // thus the phase factor we get is
	    std::complex<double> phase=exp(std::complex<double>(0.0,-(px*cen.x+py*cen.y+pz*cen.z)));

	    // Now we just store the individual function values.
	    size_t i0=bas.get_first_ind(is);
	    size_t Ni=bas.get_Nbf(is);
	    for(size_t fi=0;fi<Ni;fi++)
	      fvals[ith][i0+fi]=phase*fpoly[ith][ii][fi];
	  }

	// and now it's only a simple matter to compute the momentum density.
	emd[ip]=0.0;
	for(size_t i=0;i<Nbf;i++) {
	  // Off-diagonal
	  for(size_t j=0;j<i;j++)
	    emd[ip]+=2.0*std::real(P(i,j)*std::conj(fvals[ith][i])*fvals[ith][j]);
	  // Diagonal
	  emd[ip]+=std::real(P(i,i)*std::conj(fvals[ith][i])*fvals[ith][i]);
	}
      }
    } // end parallel region

    // Save computed value of EMD and increment norm
    for(size_t ip=0;ip<np;ip++) {
      fprintf(out,"%e\t%e\t%e\t%e\n",p[ip].x,p[ip].y,p[ip].z,emd[ip]);
      norm+=emd[ip];
    }

    // Increment number of computed points
    ntot+=np;
  }
  // Close output file.
  fclose(out);

  // Plug in the spacing in the integral
  double dx=(px_arr[px_arr.size()-1]-px_arr[0])/px_arr.size();
  double dy=(py_arr[py_arr.size()-1]-py_arr[0])/py_arr.size();
  double dz=(pz_arr[pz_arr.size()-1]-pz_arr[0])/pz_arr.size();
  norm*=dx*dy*dz;

  // Print norm
  printf("The norm of the EMD on the cube is %e.\n",norm);
}
