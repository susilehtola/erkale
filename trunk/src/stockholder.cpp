/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#include "stockholder.h"
#include "timer.h"
#include "dftgrid.h"

StockholderAtom::StockholderAtom() {
}

StockholderAtom::~StockholderAtom() {
}

void StockholderAtom::fill(const BasisSet & basis, const arma::mat & P, size_t atindv, double dr, int nrad, int lmax) {
  // Allocate memory for radial shells
  rho.resize(nrad);
  weights.resize(nrad);
  grid.resize(nrad);

  // Store atom index
  atind=atindv;
  // Nuclear coordinate
  coords_t nuc=basis.get_nucleus(atind).r;

  // Get Lebedev rule
  std::vector<lebedev_point_t> leb=lebedev_sphere(lmax);

  // Nuclear distances
  std::vector<double> nucdist=basis.get_nuclear_distances(atind);
  // Shell ranges
  std::vector<double> shran=basis.get_shell_ranges();

  // Add points
  for(int irad=0;irad<nrad;irad++) {
    // Resize points and weights
    rho[irad].resize(leb.size());
    weights[irad].resize(leb.size());
    grid[irad].resize(leb.size());

    // Current radius
    double rad=irad*dr;

    // Indices of shells to compute
    std::vector<size_t> compute_shells;
    
    // Determine which shells might contribute to this radial shell
    for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
      // Determine closest distance of nucleus
      double dist=fabs(nucdist[inuc]-rad);
      // Get indices of shells centered on nucleus
      std::vector<size_t> shellinds=basis.get_shell_inds(inuc);
      
      // Loop over shells on nucleus
      for(size_t ish=0;ish<shellinds.size();ish++) {
	
	// Shell is relevant if range is larger than minimal distance
	if(dist<=shran[shellinds[ish]]) {
	  // Add shell to list of shells to compute
	  compute_shells.push_back(shellinds[ish]);
	}
      }
    }


#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(size_t ip=0;ip<grid[irad].size();ip++) {
      // Store weight
      weights[irad][ip]=leb[ip].w;

      // and coordinates
      grid[irad][ip].x=nuc.x+rad*leb[ip].x;
      grid[irad][ip].y=nuc.y+rad*leb[ip].y;
      grid[irad][ip].z=nuc.z+rad*leb[ip].z;
      
      // Compute functions at grid point
      std::vector<bf_f_t> flist;
      for(size_t i=0;i<compute_shells.size();i++) {
	// Shell is
	size_t ish=compute_shells[i];
	
        // Center of shell is
        coords_t shell_center=basis.get_shell_center(ish);
        // Compute distance of point to center of shell
        double shell_dist=norm(shell_center-grid[irad][ip]);

        // Add shell to point if it is within the range of the shell
        if(shell_dist<shran[ish]) {
          // Index of first function on shell is
          size_t ind0=basis.get_first_ind(ish);

          // Compute values of basis functions
	  arma::vec fval=basis.eval_func(ish,grid[irad][ip].x,grid[irad][ip].y,grid[irad][ip].z);

          // and add them to the list
          bf_f_t hlp;
          for(size_t ifunc=0;ifunc<fval.n_elem;ifunc++) {
            // Index of function is
            hlp.ind=ind0+ifunc;
            // Value is
            hlp.f=fval(ifunc);
            // Add to stack
            flist.push_back(hlp);
          }
        }
      }

      // Compute density
      double d=0.0;
      for(size_t ii=0;ii<flist.size();ii++) {
	// Index of function is
	size_t i=flist[ii].ind;
	for(size_t jj=0;jj<ii;jj++) {
	  // Index of function is
	  size_t j=flist[jj].ind;
	  d+=2.0*P(i,j)*flist[ii].f*flist[jj].f;
	}
	d+=P(i,i)*flist[ii].f*flist[ii].f;
      }

      // Store the density
      rho[irad][ip]=d;
    }
  }
}

void StockholderAtom::update(const Hirshfeld & hirsh, std::vector<double> & dens) {
  // Returned densities
  dens.resize(rho.size());

  // Loop over radii
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t irad=0;irad<grid.size();irad++) {
    // Initialize density
    double d=0.0;
    // and total weight
    double w=0.0;

    // Loop over grid
    for(size_t ip=0;ip<grid[irad].size();ip++) {
      // Increment total angular weight
      w+=weights[irad][ip];
      // and spherically averaged density
      double c=weights[irad][ip]*rho[irad][ip]*hirsh.get_weight(atind,grid[irad][ip]);
      d+=c;
    }

    // Store spherically averaged density
    dens[irad]=d/w;
  }
}

Stockholder::Stockholder(const BasisSet & basis, const arma::mat & P, double tol, double dr, int nrad, int lmax, bool verbose) {
  Timer t;

  // Allocate atomic grids
  atoms.resize(basis.get_Nnuc());

  // Get centers
  cen.resize(basis.get_Nnuc());
  for(size_t i=0;i<basis.get_Nnuc();i++)
    cen[i]=basis.get_nuclear_coords(i);

  // Compute molecular density
  if(verbose) {
    printf("Filling Stockholder molecular density grid ... ");
    fflush(stdout);
  }

  for(size_t i=0;i<basis.get_Nnuc();i++)
    atoms[i].fill(basis,P,i,dr,nrad,lmax);

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  // Initial weight
  std::vector<double> w0(nrad,1.0);
  std::vector< std::vector<double> > oldrho(cen.size(),w0);
  std::vector< std::vector<double> > newrho(cen.size(),w0);

  // Loop
  if(verbose) {
    printf("\nStockholder iteration\n");
    printf("%5s  %12s  %12s\n","iter","max","mean");
    fflush(stdout);
  }

  for(size_t iiter=0;iiter<10000;iiter++) {
    // Update weights
    ISA.set(cen,dr,oldrho);

    // Evaluate new spherically averaged densities
    for(size_t i=0;i<atoms.size();i++)
      atoms[i].update(ISA,newrho[i]);
    
    // Check for convergence
    arma::vec diff(atoms.size());
    diff.zeros();
    
    for(size_t irad=0;irad<newrho[0].size();irad++) {
      // Radius
      double r=irad*dr;
      
      // Compute density difference
      for(size_t iat=0;iat<atoms.size();iat++) {
	double d=fabs(newrho[iat][irad]-oldrho[iat][irad]);
	diff(iat)+=r*r*d*dr;
      }
    }
    
    /*
    // Print out densities
    std::ostringstream fname;
    fname << "atoms_" << iiter << ".dat";
    FILE *out=fopen(fname.str().c_str(),"w");
    for(size_t irad=0;irad<newrho[0].size();irad++) {
      fprintf(out,"%e",irad*dr);
      for(size_t iat=0;iat<newrho.size();iat++)
	fprintf(out," %e",newrho[iat][irad]);
      fprintf(out,"\n");
    }
    fclose(out);
    */

    // Swap densities
    std::swap(newrho,oldrho);

    double maxdiff=arma::max(diff);
    double meandiff=arma::mean(diff);

    if(verbose) {
      printf("%5i  %e  %e\n",(int) iiter+1,maxdiff,meandiff);	     
      fflush(stdout);
    }

    if(maxdiff<tol)
      break;
  }

  printf("Iteration converged within %e in %s.\n",tol,t.elapsed().c_str());

}

Stockholder::~Stockholder() {
}

Hirshfeld Stockholder::get() const {
  return ISA;
}
