/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "global.h"
#include "basis.h"
#include "checkpoint.h"
#include "stringutil.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void density_cube(const BasisSet & bas, const arma::mat & P, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr) {
  // Open output file.
  FILE *out=fopen("denscube.dat","w");

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
  const size_t N=x_arr.size()*y_arr.size()*z_arr.size();
  // The necessary amount of batches is
  size_t Nbatch=N/Nbatch_p;
  if(N%Nbatch_p!=0)
    Nbatch++;

  // The values of momentum in the batch
  coords_t r[Nbatch_p];
  // The values of the density in the batch
  double rho[Nbatch_p];

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
      r[np].x=x_arr[xind];
      r[np].y=y_arr[yind];
      r[np].z=z_arr[zind];

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	zind=0;

	if(yind+1<y_arr.size())
	  yind++;
	else {
	  yind=0;
	  xind++;
	}
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    // Loop over the points in the batch
    for(size_t ip=0;ip<np;ip++)
      rho[ip]=compute_density(P,bas,r[ip]);

    // Save computed value of density and increment norm
    for(size_t ip=0;ip<np;ip++) {
      fprintf(out,"%e\t%e\t%e\t%e\n",r[ip].x,r[ip].y,r[ip].z,rho[ip]);
      norm+=rho[ip];
    }

    // Increment number of computed points
    ntot+=np;
  }
  // Close output file.
  fclose(out);

  // Plug in the spacing in the integral
  double dx=(x_arr[x_arr.size()-1]-x_arr[0])/x_arr.size();
  double dy=(y_arr[y_arr.size()-1]-y_arr[0])/y_arr.size();
  double dz=(z_arr[z_arr.size()-1]-z_arr[0])/z_arr.size();
  norm*=dx*dy*dz;

  // Print norm
  printf("The norm of the density on the cube is %e.\n",norm);
}

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Cubes from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Cubes from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();

  // Parse settings
  Settings set;
  set.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  set.add_string("DensCube", "Calculate electron density on a cube? e.g. -10:.3:10 -5:.2:4 -2:.1:3", "");

  if(argc==2)
    set.parse(argv[1]);
  else
    printf("Using default settings.\n");

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Load density matrix
  arma::mat P;
  chkpt.read("P",P);

  // Form grid in p space.
  std::vector<double> x, y, z;
  parse_cube(set.get_string("DensCube"),x,y,z);

  // Calculate density on cube
  density_cube(basis,P,x,y,z);

  return 0;
}
