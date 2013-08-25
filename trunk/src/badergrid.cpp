/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2013
 * Copyright (c) 2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include <cfloat>
#include "badergrid.h"
#include "timer.h"
#include "stringutil.h"
#include "unitary.h"

#ifdef _OPENMP
#include <omp.h>
#endif

//#define BADERDEBUG

// Threshold for vanishingly small density
#define SMALLDENSITY 1e-8

// Threshold for converged density gradient
#define SMALLGRADIENT 1e-30
// Threshold for separation of maxima
#define SAMEMAXIMUM 0.1

// Minimum step size
#define MINSTEPSIZE 1e-6

// Threshold for convergence at nucleus
#define NUCLEARTHRESHOLD 1e-4

BaderAtom::BaderAtom(bool lobatto, double tolv) : AtomGrid(lobatto,tolv) {
}

BaderAtom::~BaderAtom() {
}

std::vector<arma::sword> BaderAtom::classify(const BasisSet & basis, const arma::mat & P, std::vector<coords_t> & maxima, size_t & ndens, size_t & ngrad) {
  // Returned classifications
  std::vector<arma::sword> region;
  region.assign(grid.size(),-1);
  
  // Nuclear coordinates
  arma::mat nuccoord=basis.get_nuclear_coords();

  size_t nd=0, ng=0;
  
  // Loop over points
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nd,ng)
#endif
  for(size_t ip=0;ip<grid.size();ip++) {
    if(ip>0 && false)
      throw std::runtime_error("Debug mode");
    
    // Track density to maximum.
    coords_t r=grid[ip].r;
    size_t iiter=0;
    
    // Initial step size to use
    const double steplen=0.1;
    double dr(steplen);
    
    // Density and gradient
    double d;
    arma::vec g;
    
    while(true) {
      // Iteration number
      iiter++;
      
      // Compute density and gradient
      compute_density_gradient(P,basis,r,d,g);
      double gnorm=arma::norm(g,2);
      nd++; ng++;
     
      // Convergence check
      if(d<=SMALLDENSITY) {
	// Zero density.
#ifdef BADERDEBUG
	printf("Point %4i at % f % f % f has small density %e, stopping. %e\n",(int) ip+1,r.x,r.y,r.z,d,compute_density(P,basis,r));
#endif
	region[ip]=0;
	break;
	
      } else if(gnorm<=SMALLGRADIENT)
	break;

      // Normalize gradient and preform line search
      coords_t gn;
      gn.x=g(0)/gnorm;
      gn.y=g(1)/gnorm;
      gn.z=g(2)/gnorm;
      std::vector<double> len, dens;
#ifdef BADERDEBUG
      printf("Gradient norm %e. Normalized gradient at % f % f % f is % f % f % f\n",gnorm,r.x,r.y,r.z,gn.x,gn.y,gn.z);
#endif

      // Determine step size to use by finding out minimal distance to nuclei
      arma::rowvec rv(3);
      rv(0)=r.x; rv(1)=r.y; rv(2)=r.z;
      // and the closest nucleus
      double mindist=arma::norm(rv-nuccoord.row(0),2);
      arma::vec closenuc=nuccoord.row(0);
      for(size_t i=1;i<nuccoord.n_rows;i++) {
	double t=arma::norm(rv-nuccoord.row(i),2);
	if(t<mindist) {
	  mindist=t;
	  closenuc=nuccoord.row(i);
	}
      }
      dr=std::min(dr,mindist/2.0);
      //      printf("Minimal distance is to nucleus %i at %e.\n",(int) closeind+1,mindist);

      // Starting point
      len.push_back(0.0);
      dens.push_back(d);

#ifdef BADERDEBUG
      printf("Step length %e: % f % f % f, density %e, difference %e\n",len[0],r.x,r.y,r.z,dens[0],0.0);
#endif

      // Trace until density does not increase any more.
      do {
	// Increase step size
	len.push_back(pow(2,len.size())*dr);
	// New point
	coords_t pt=r+gn*len[len.size()-1];
	// and density
	dens.push_back(compute_density(P,basis,pt));
	nd++;

#ifdef BADERDEBUG	
	printf("Step length %e: % f % f % f, density %e, difference %e\n",len[len.size()-1],pt.x,pt.y,pt.z,dens[dens.size()-1],dens[dens.size()-1]-dens[0]);
#endif
	
      } while(dens[dens.size()-1]>dens[dens.size()-2]);

      // Now we know where the optimal line length is. Do interpolation
      arma::vec ilen(3), idens(3);

      if(dens.size()==2) {
	ilen(0)=len[len.size()-2];
	ilen(2)=len[len.size()-1];
	ilen(1)=(ilen(0)+ilen(2))/2.0;
	
	idens(0)=dens[dens.size()-2];
	idens(2)=dens[dens.size()-1];
	idens(1)=compute_density(P,basis,r+gn*ilen(1));
	nd++;
      } else {
	ilen(0)=len[len.size()-3];
	ilen(2)=len[len.size()-1];
	ilen(1)=(ilen(0)+ilen(2))/2.0;
	
	idens(0)=dens[dens.size()-3];
	idens(2)=dens[dens.size()-1];
	idens(1)=compute_density(P,basis,r+gn*ilen(1));
	nd++;
      }

#ifdef BADERDEBUG	
      arma::trans(ilen).print("Step lengths");
      arma::trans(idens).print("Densities");
#endif

      // Fit polynomial
      arma::vec p=fit_polynomial(ilen,idens);

      // and solve for the roots of its derivative
      arma::vec roots=solve_roots(derivative_coefficients(p));

      // The optimal step length is
      double optlen=0.0;
      for(size_t i=0;i<roots.n_elem;i++)
	if(roots(i)>=ilen(0) && roots(i)<=ilen(2)) {
	  optlen=roots(i);
	  break;
	}

#ifdef BADERDEBUG      
      printf("Optimal step length is %e.\n",optlen);
#endif
      if(optlen==0.0) {
	if(dr>=MINSTEPSIZE) {
	  // Reduce step length.
	  dr/=10.0;
	  continue;
	} else {
	  /*
	  arma::trans(ilen).print("Points");
	  arma::trans(idens).print("Densities");
	  arma::trans(idens-idens(0)).print("Density difference");
	  arma::trans(roots).print("Roots");
	  throw std::runtime_error("Zero step length!\n");
	  */

	  break;
	}
      } else {

	// Check for convergence
	if(std::min(optlen,mindist)<=NUCLEARTHRESHOLD) {
	  // Converged at nucleus.
	  r.x=closenuc(0);
	  r.y=closenuc(1);
	  r.z=closenuc(2);

#ifdef BADERDEBUG
	  printf("Point %4i converged to nucleus.\n",(int) ip+1);
#endif
	  break;
	}
      }

      // Update point
      if(optlen>dr)
	dr=optlen;
      r=r+gn*optlen;
    }

    // No need to classify zero region
    if(region[ip]==0) {
#ifdef BADERDEBUG
      printf("Point %4i with density %e skipped.\n\n",(int) ip+1,compute_density(P,basis,grid[ip].r));
#endif
      continue;
    }

    // Maximum tracked. First check if the maximum is on the list of known maxima
    bool found=false;
    for(size_t i=0;i<maxima.size();i++)
      if(norm(r-maxima[i])<=SAMEMAXIMUM) {
	found=true;
	region[ip]=i+1;
#ifdef BADERDEBUG
	printf("Point %4i with density %e ended up at maximum %i.\n\n",(int) ip+1,compute_density(P,basis,grid[ip].r),(int) i+1);
#endif
	break;
      }

    // Maximum was not found, add it to the list
    if(!found)
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	maxima.push_back(r);
	region[ip]=maxima.size();
#ifdef BADERDEBUG
	printf("Point %4i with density %e found maximum %i at % f % f % f.\n\n",(int) ip+1,compute_density(P,basis,grid[ip].r),(int) maxima.size(),r.x,r.y,r.z);
#endif
      }
  }

  ndens+=nd;
  ngrad+=ng;

  return region;
}

void BaderAtom::charge(const BasisSet & basis, const arma::mat & P, const std::vector<arma::sword> & region, arma::vec & q) const {
  // Loop over points
#ifndef _OPENMP
  for(size_t ip=0;ip<grid.size();ip++) {
    if(region[ip]>0)
      q(region[ip]-1)+=grid[ip].w*compute_density(P,basis,grid[ip].r);
  }
#else
#pragma omp parallel for
  for(arma::sword ireg=0;ireg<(arma::sword) q.n_elem;ireg++)
    for(size_t ip=0;ip<grid.size();ip++)
      if(region[ip]-1==ireg)
	q(ireg)+=grid[ip].w*compute_density(P,basis,grid[ip].r);
#endif
}

void BaderAtom::regional_overlap(const std::vector<arma::sword> & region, std::vector<arma::mat> & stack) const {
#ifndef _OPENMP
  for(size_t ip=0;ip<grid.size();ip++)
    if(region[ip]>0) {
      // Loop over functions on grid point
      size_t first=grid[ip].f0;
      size_t last=first+grid[ip].nf;
      
      size_t i, j;
      for(size_t ii=first;ii<last;ii++) {
	// Index of function is
	i=flist[ii].ind;
	for(size_t jj=first;jj<last;jj++) {
	  j=flist[jj].ind;
	  
	  stack[region[ip]-1](i,j)+=grid[ip].w*flist[ii].f*flist[jj].f;
	}
      }
    }
#else
#pragma omp parallel for schedule(dynamic,1)
  for(arma::sword ireg=0;ireg<(arma::sword) stack.size();ireg++)
    for(size_t ip=0;ip<grid.size();ip++)
      if(region[ip]-1==ireg) {
	// Loop over functions on grid point
	size_t first=grid[ip].f0;
	size_t last=first+grid[ip].nf;
	
	size_t i, j;
	for(size_t ii=first;ii<last;ii++) {
	  // Index of function is
	  i=flist[ii].ind;
	  for(size_t jj=first;jj<last;jj++) {
	    j=flist[jj].ind;
	    
	    stack[ireg](i,j)+=grid[ip].w*flist[ii].f*flist[jj].f;
	  }
	}
      }
#endif
}

BaderGrid::BaderGrid(const BasisSet * bas, bool ver, bool lobatto) {
  basp=bas;
  verbose=ver;
  use_lobatto=lobatto;

  grids.resize(basp->get_Nnuc());

  // Allocate work grids
#ifdef _OPENMP
  int nth=omp_get_max_threads();
  for(int i=0;i<nth;i++)
    wrk.push_back(BaderAtom(lobatto));
#else
  wrk.push_back(BaderAtom(lobatto));
#endif
}

BaderGrid::~BaderGrid() {
}

void BaderGrid::construct(double tol) {
  
  // Add all atoms
  if(verbose) {
    printf("Constructing Bader grid.\n");
    printf("\t%4s  %7s  %10s  %s\n","atom","Npoints","Nfuncs","t");
    fflush(stdout);
  }
  
  // Set tolerances
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_tolerance(tol);

  Timer t;
  size_t Nat=basp->get_Nnuc();

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel section

#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<Nat;i++) {
      grids[i]=wrk[ith].construct_becke(*basp,i,verbose);
    }

  }   // End parallel section

  if(verbose) {
    printf("Bader grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}


void BaderGrid::classify(const arma::mat & P) {
  Timer t;

  // Destroy old classifications
  regions.clear();
  regions.resize(grids.size());
  // and maxima
  maxima.clear();

  size_t denstot=0;
  size_t gradtot=0;

  if(verbose) {
    printf("Running Bader classification.\n");
    printf("\t%4s  %7s  %7s\n","atom","Ndens","Ngrad");
    fflush(stdout);
  }

  for(size_t i=0;i<grids.size();i++) {
    // Form the grid
    wrk[0].form_grid(*basp,grids[i]);
    // and run the classification
    size_t ndens=0, ngrad=0;
    regions[i]=wrk[0].classify(*basp,P,maxima,ndens,ngrad);
    if(verbose) {
      printf("\t%4i  %7s  %7s\n",(int) i+1,space_number(ndens).c_str(),space_number(ngrad).c_str());
      fflush(stdout);
    }
    denstot+=ndens;
    gradtot+=ngrad;
  }

  printf("Bader analysis done in %s.\nUsed %i density and %i gradient evaluations.\n",t.elapsed().c_str(),(int) denstot, (int) gradtot);
  printf("Found %i maxima.\n",(int) maxima.size());
  for(size_t i=0;i<maxima.size();i++)
    printf("%4i % f % f % f\n",(int) i+1,maxima[i].x,maxima[i].y,maxima[i].z);
}

arma::vec BaderGrid::regional_charges(const arma::mat & P) {
  arma::vec q(maxima.size());
  q.zeros();
  for(size_t i=0;i<grids.size();i++) {
    wrk[0].form_grid(*basp,grids[i]);
    wrk[0].charge(*basp,P,regions[i],q);
  }

  return -q;
}

arma::vec BaderGrid::nuclear_charges(const arma::mat & P) {
  // Get regional charges
  arma::vec qr=regional_charges(P);
  arma::trans(qr).print("Regional charges");

  // Nuclear charges
  arma::vec q(basp->get_Nnuc());
  q.zeros();
  for(size_t i=0;i<grids.size();i++) {
    if(regions[i][0]>0)
      q[i]+=qr(regions[i][0]-1);
  }

  arma::trans(q).print("Nuclear charges");

  return q;
}

std::vector<arma::mat> BaderGrid::regional_overlap() {
  Timer t;
  if(verbose) {
    printf("Computing regional overlap matrices ... ");
    fflush(stdout);
  }

  std::vector<arma::mat> stack(maxima.size());
  for(size_t i=0;i<stack.size();i++)
    stack[i].zeros(basp->get_Nbf(),basp->get_Nbf());

  for(size_t i=0;i<grids.size();i++) {
    wrk[0].form_grid(*basp,grids[i]);
    wrk[0].compute_bf(*basp,grids[i]);
    wrk[0].regional_overlap(regions[i],stack);
  }

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());
    fflush(stdout);
  }

  return stack;
}
