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
#include <cfloat>

StockholderAtom::StockholderAtom() {
}

StockholderAtom::~StockholderAtom() {
}

void StockholderAtom::compute(const BasisSet & basis, const arma::mat & P, const std::vector<double> & shran, const std::vector<size_t> & compute_shells, double dr, size_t irad, int lmax) {
  // Current radius
  double rad=irad*dr;

  // Get rule
  std::vector<lebedev_point_t> leb=lebedev_sphere(lmax);

  // Nuclear coordinate
  coords_t nuc=basis.get_nucleus(atind).r;

  // Allocate memory
  rho[irad].resize(leb.size());
  weights[irad].resize(leb.size());
  grid[irad].resize(leb.size());

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

void StockholderAtom::fill_adaptive(const BasisSet & basis, const arma::mat & P, const Hirshfeld & hirsh, size_t atindv, double dr, int nrad, int lmax, double tol, bool verbose) {
  // Allocate memory for radial shells
  rho.resize(nrad);
  weights.resize(nrad);
  grid.resize(nrad);

  // Store atom index
  atind=atindv;

  // Nuclear distances
  std::vector<double> nucdist=basis.get_nuclear_distances(atind);
  // Shell ranges
  std::vector<double> shran=basis.get_shell_ranges();

  // Add points
  for(int irad=0;irad<nrad;irad++) {
    // Radius is
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

    // Initial order
    int l=3;
    // Compute radial shell
    compute(basis,P,shran,compute_shells,dr,irad,l);
    // and get the average
    double avg=average(hirsh,irad);

    // Next rule and new average
    int lnext;
    double avgnext;

    std::vector<double> oldrho, oldweights;
    std::vector<coords_t> oldgrid;

    while(l<=lmax) {
      // Get old rho, weight and grid
      oldrho=rho[irad];
      oldweights=weights[irad];
      oldgrid=grid[irad];

      // Increment rule
      lnext=next_lebedev(l);
      compute(basis,P,shran,compute_shells,dr,irad,lnext);
      avgnext=average(hirsh,irad);
      
      // Difference is
      double diff=rad*rad*fabs(avg-avgnext)*dr;

      // Check if convergence is fulfilled.
      if(diff < tol/nrad)
	break;
      else {
	// Switch variables
	l=lnext;
	avg=avgnext;
      }
    }

    // Revert to converged grid
    rho[irad]=oldrho;
    weights[irad]=oldweights;
    grid[irad]=oldgrid;
  }

  // Count size of grid
  size_t N=0;
  for(size_t ir=0;ir<grid.size();ir++)
    N+=grid[ir].size();

  if(verbose)
    printf("%4i %7i\n",(int) atind+1,(int) N);
}

void StockholderAtom::fill_static(const BasisSet & basis, const arma::mat & P, size_t atindv, double dr, int nrad, int l, bool verbose) {
  // Allocate memory for radial shells
  rho.resize(nrad);
  weights.resize(nrad);
  grid.resize(nrad);

  // Store atom index
  atind=atindv;

  // Nuclear distances
  std::vector<double> nucdist=basis.get_nuclear_distances(atind);
  // Shell ranges
  std::vector<double> shran=basis.get_shell_ranges();

  // Add points
  for(int irad=0;irad<nrad;irad++) {
    // Radius is
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

    // Compute radial shell
    compute(basis,P,shran,compute_shells,dr,irad,l);
  }
  
  // Count size of grid
  size_t N=0;
  for(size_t ir=0;ir<grid.size();ir++)
    N+=grid[ir].size();
  
  if(verbose)
    printf("%4i %7i\n",(int) atind+1,(int) N);
}

double StockholderAtom::average(const Hirshfeld & hirsh, size_t irad) const {
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
  return d/w;
}

void StockholderAtom::update(const Hirshfeld & hirsh, std::vector<double> & dens) {
  // Returned densities
  dens.resize(rho.size());

  // Loop over radii
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t irad=0;irad<grid.size();irad++) {
    dens[irad]=average(hirsh,irad);
  }
}

Stockholder::Stockholder(const BasisSet & basis, const arma::mat & P, double finaltol, double dr, int nrad, int l0, int lmax, bool verbose) {
  Timer t, ttot;

  // Allocate atomic grids
  atoms.resize(basis.get_Nnuc());

  // Get centers
  cen.resize(basis.get_Nnuc());
  for(size_t i=0;i<basis.get_Nnuc();i++)
    cen[i]=basis.get_nuclear_coords(i);

  // Initial weight.
  std::vector<double> w0(nrad,1.0);
  for(int ir=0;ir<nrad;ir++)
    w0[ir]=exp(-ir*dr);

  std::vector< std::vector<double> > oldrho(cen.size(),w0);
  std::vector< std::vector<double> > newrho(cen.size(),w0);

  // Update weights
  ISA.set(cen,dr,oldrho);
  (void) l0;

  // Current tolerance
  double tol=1e-2;

  if(verbose) {
    printf("Filling initial Stockholder molecular density grid.\n");
    printf("%4s %7s\n","atom","Npoints");
    fflush(stdout);
  }
  for(size_t i=0;i<basis.get_Nnuc();i++)
    atoms[i].fill_static(basis,P,i,dr,nrad,l0,verbose);
  if(verbose) {
    printf("Initial fill done in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }
  
  // Evaluate new spherically averaged densities
  for(size_t i=0;i<atoms.size();i++)
    atoms[i].update(ISA,oldrho[i]);

  if(verbose) {
    printf("Spherically averaged densities updated in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }


  /*
  {
    // Print out densities
    std::ostringstream fname;
    fname << "atoms_start.dat";
    FILE *out=fopen(fname.str().c_str(),"w");
    for(size_t irad=0;irad<oldrho[0].size();irad++) {
      fprintf(out,"%e",irad*dr);
      for(size_t iat=0;iat<oldrho.size();iat++)
	fprintf(out," %e",oldrho[iat][irad]);
      fprintf(out,"\n");
    }
    fclose(out);
  }
  */

  // and use these as starting weights for the self-consistent iteration
  ISA.set(cen,dr,oldrho);
  
  while(true) {
    // Compute molecular density
    if(verbose) {
      printf("\nFilling Stockholder molecular density grid.\n");
      printf("%4s %7s\n","atom","Npoints");
      fflush(stdout);
    }

    // Adaptive generation of grid
    for(size_t i=0;i<basis.get_Nnuc();i++)
      atoms[i].fill_adaptive(basis,P,ISA,i,dr,nrad,lmax,tol,verbose);
    
    if(verbose) {
      printf("Grid filled in %s. Grid iteration\n",t.elapsed().c_str());
      printf("%5s  %12s  %12s\n","iter","max","mean");
      fflush(stdout);
      t.set();
    }
    
    for(size_t iiter=0;iiter<10000;iiter++) {
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

      // Update weights
      ISA.set(cen,dr,oldrho);
    }

    if(verbose) {
      printf("Iteration converged within %e in %s.\n",tol,t.elapsed().c_str());
      fflush(stdout);

      /*
      // Print out densities
      char fname[80];
      sprintf(fname,"atoms_%.3e.dat",tol);
      FILE *out=fopen(fname,"w");
      for(size_t irad=0;irad<newrho[0].size();irad++) {
	fprintf(out,"%e",irad*dr);
	for(size_t iat=0;iat<oldrho.size();iat++)
	  fprintf(out," %e",oldrho[iat][irad]);
	fprintf(out,"\n");
      }
      fclose(out);
      */
    }	

    // Reduce tolerance and check convergence
    tol/=sqrt(10.0);
    if(tol < (1-sqrt(DBL_EPSILON))*finaltol)
      break;

    printf("tol = %e, finaltol = %e, diff %e\n",tol,finaltol,tol-finaltol);
  }
  
  if(verbose) {
    printf("Stockholder atoms solved in %s.\n",ttot.elapsed().c_str());
    fflush(stdout);
  }
}

Stockholder::~Stockholder() {
}

Hirshfeld Stockholder::get() const {
  return ISA;
}
