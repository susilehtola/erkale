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
  // Get expansions of functions
  std::vector< std::vector<size_t> > idents;
  std::vector< std::vector<GTO_Fourier> > fourier=fourier_expand(bas,idents);
  
  // Open output file.
  FILE *out=fopen("emdcube.dat","w");

  // Compute the norm (assumes evenly spaced grid!)
  double norm=0.0;

  // Compute the momentum densities in batches, allowing
  // parallellization.
#ifdef _OPENMP
  // The number of points per batch
  const size_t Nbatch_p=100*omp_get_max_threads();
#else
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
#pragma omp for
#endif
      // Loop over the points in the batch
      for(size_t ip=0;ip<np;ip++)
	// Evaluate EMD
	emd[ip]=eval_emd(bas,P,fourier,idents,p[ip].x,p[ip].y,p[ip].z);
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
