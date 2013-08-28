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
#include "properties.h"

#include "elements.h"
#include "mathf.h"
// Lobatto or Lebedev for angular
#include "lobatto.h"
#include "lebedev.h"
// Gauss-Chebyshev for radial integral
#include "chebyshev.h"

#ifdef _OPENMP
#include <omp.h>
#endif

//#define BADERDEBUG

// Threshold for vanishingly small density
#define SMALLDENSITY 1e-10

// Convergence threshold of update
#define CONVTHR 1e-3
// Threshold distance for separation of maxima
#define SAMEMAXIMUM (10*CONVTHR)

BaderAtom::BaderAtom(bool lobatto, double tolv) : AtomGrid(lobatto,tolv) {
}

BaderAtom::~BaderAtom() {
}

atomgrid_t BaderAtom::construct(const BasisSet & basis, const arma::mat & P, size_t cenind, bool verbose) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Krack 1998 for details

  // Returned info
  atomgrid_t ret;
  ret.ngrid=0;
  ret.nfunc=0;

  Timer t;

  // Store index of center
  ret.atind=cenind;
  // and its coordinates
  ret.cen=basis.get_nuclear_coords(cenind);

  // Compute necessary number of radial points
  size_t nrad=std::max(20,(int) round(-5*(3*log10(tol)+8-element_row[basis.get_Z(ret.atind)])));

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> xc, wc;
  chebyshev(nrad,xc,wc);

  // Allocate memory
  ret.sh.resize(nrad);

  // Loop over radii
  double rad, jac;
  for(size_t ir=0;ir<xc.size();ir++) {
    // Calculate value of radius
    double ixc=xc.size()-1-ir;
    rad=1.0/M_LN2*log(2.0/(1.0-xc[ixc]));

    // Jacobian of transformation is
    jac=1.0/M_LN2/(1.0-xc[ixc]);
    // so total quadrature weight is
    double weight=wc[ixc]*rad*rad*jac;

    // Store shell data
    ret.sh[ir].r=rad;
    ret.sh[ir].w=weight;
    ret.sh[ir].l=3;
  }

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(tol));

  // List of maxima
  std::vector<coords_t> maxima;
  // Amount of density and gradient evaluations
  size_t ngrad=0, ndens=0;

  // Old and new diagonal elements of regional overlap
  arma::mat Sold, Snew;

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits shell by shell
  for(size_t ir=0;ir<ret.sh.size();ir++) {

    // Radial shell converged?
    bool convd;

    do {
      // Clear current grid points and function values
      free();

      // Form radial shell
      if(use_lobatto)
	add_lobatto_shell(ret,ir);
      else
	add_lebedev_shell(ret,ir);
      // Compute Becke weights for radial shell
      becke_weights(basis,ret,ir);
      // Prune points with small weight
      prune_points(1e-8*tol,ret.sh[ir]);

      // Compute values of basis functions
      compute_bf(basis,ret,ir);

      // Run classification
      std::vector<arma::sword> regions=classify(basis,P,maxima,ndens,ngrad);

      // Compute new diagonal overlap
      Snew.zeros(basis.get_Nbf(),maxima.size());
      for(size_t ip=0;ip<grid.size();ip++)
	if(regions[ip]>0) {
	  // Loop over functions on grid point
	  size_t first=grid[ip].f0;
	  size_t last=first+grid[ip].nf;
	  
	  for(size_t ii=first;ii<last;ii++) {
	    // Get index of function
	    size_t i=flist[ii].ind;
	    
	    Snew(i,regions[ip]-1)+=grid[ip].w*flist[ii].f*flist[ii].f;
	  }
	}

      // Check if amount of regions has changed / convergence has been achieved
      convd=true;
      if(Snew.n_cols!=Sold.n_cols)
	convd=false;
      else {
	// Compute maximum difference of diagonal elements,
	// normalizing to the amount of maxima
	maxdiff=arma::max(arma::max(arma::abs(Snew-Sold)));
	convd=(maxdiff<tol/xc.size());
      }
      
      // Swap matrices
      std::swap(Snew,Sold);

      // Increment order if tolerance not achieved.
      if(!convd) {
	if(use_lobatto)
	  ret.sh[ir].l+=2;
	else {
	  // Need to determine what is next order of Lebedev
	  // quadrature that is supported.
	  ret.sh[ir].l=next_lebedev(ret.sh[ir].l);
	}
      }
    } while(!convd && ret.sh[ir].l<=lmax);
    
    // Increase number of points and function values
    ret.ngrid+=grid.size();
    ret.nfunc+=flist.size();
  }

  // Free memory once more
  free();

  if(verbose) {
    //    printf("\t%4u  %7s  %10s  %10s  %s\n",(unsigned int) ret.atind+1,space_number(ret.ngrid).c_str(),space_number(ret.nfunc).c_str(),space_number(ndens).c_str(),t.elapsed().c_str());
    printf("\t%4u  %7u  %10u  %10u  %s\n",(unsigned int) ret.atind+1,(unsigned int) ret.ngrid,(unsigned int) ret.nfunc,(unsigned int) ndens,t.elapsed().c_str());
    fflush(stdout);
  }

  return ret;
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
    // Convergence check                                                                                                                                                                                         
    if(compute_density(P,basis,grid[ip].r)<=SMALLDENSITY) {
      // Zero density.                                                                                                                                                                                           
#ifdef BADERDEBUG
      printf("Point %4i at % f % f % f has small density %e, stopping.\n",(int) ip+1,grid[ip].r.x,grid[ip].r.y,grid[ip].r.z,compute_density(P,basis,grid[ip].r));
#endif
      region[ip]=0;
      continue;
    }
    
    // Otherwise, track the maximum
    coords_t r=track_to_maximum(basis,P,grid[ip].r,nd,ng);

    // Now that we have the maximum, check if the maximum is on the list of known maxima
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

void BaderGrid::construct(const arma::mat & P, double tol, bool regional) {
  
  // Add all atoms
  if(verbose) {
    printf("Constructing Bader grid.\n");
    if(regional)
      printf("\t%4s  %7s  %10s  %10s  %s\n","atom","Npoints","Nfuncs","Ndens","t");
    else
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
      if(regional)
	grids[i]=wrk[ith].construct(*basp,P,i,verbose);
      else
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
    printf("\t%4s  %7s  %7s  %5s\n","atom","Ngrad","Ndens","Avg");
    fflush(stdout);
  }

  for(size_t i=0;i<grids.size();i++) {
    // Form the grid
    wrk[0].form_grid(*basp,grids[i]);
    // and run the classification
    size_t ndens=0, ngrad=0;
    regions[i]=wrk[0].classify(*basp,P,maxima,ndens,ngrad);
    double densave=ndens*1.0/grids[i].ngrid;
    if(verbose) {
      //      printf("\t%4i  %7s  %7s  %4.2f\n",(int) i+1,space_number(ngrad).c_str(),space_number(ndens).c_str(),densave);
      printf("\t%4i  %7u  %7u  %4.2f\n",(int) i+1,(unsigned int) ngrad,(unsigned int) ndens,densave);
      fflush(stdout);
    }
    denstot+=ndens;
    gradtot+=ngrad;
  }

  printf("Bader analysis done in %s.\nUsed %i density and %i gradient evaluations.\n",t.elapsed().c_str(),(int) denstot, (int) gradtot);
  print_maxima();
}

void BaderGrid::print_maxima() const {
  // Classify into nuclear and nonnuclear maxima
  std::vector<coords_t> nuclear, nonnuc;
  std::vector<size_t> nuci;
  for(size_t i=0;i<maxima.size();i++) {
    // Check if it's a nuclear maximum
    bool nuc=false;
    for(size_t j=0;j<basp->get_Nnuc();j++)
      if(norm(basp->get_nuclear_coords(j)-maxima[i])<=SAMEMAXIMUM) {
	nuclear.push_back(maxima[i]);
	nuci.push_back(j);
	nuc=true;
	break;
      }

    if(!nuc)
      nonnuc.push_back(maxima[i]);
  }
  
  printf("Found %i nuclear maxima.\n",(int) nuclear.size());
  for(size_t i=0;i<nuclear.size();i++)
    printf("%4i %4i %-2s % f % f % f\n",(int) i+1,(int) nuci[i]+1,basp->get_symbol(nuci[i]).c_str(),nuclear[i].x,nuclear[i].y,nuclear[i].z);
  if(nonnuc.size()) {
    printf("Found %i non-nuclear maxima.\n",(int) nonnuc.size());
    for(size_t i=0;i<nonnuc.size();i++)
      printf("%4i % f % f % f\n",(int) i+1,nonnuc[i].x,nonnuc[i].y,nonnuc[i].z);
  }
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

  // Nuclear charges
  arma::vec q(basp->get_Nnuc());
  q.zeros();
  for(size_t i=0;i<grids.size();i++) {
    if(regions[i][0]>0)
      q[i]+=qr(regions[i][0]-1);
  }

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

coords_t track_to_maximum(const BasisSet & basis, const arma::mat & P, const coords_t r0, size_t & nd, size_t & ng) {
  // Track density to maximum.
  coords_t r(r0);
  size_t iiter=0;

  // Amount of density and gradient evaluations
  size_t ndens=0;
  size_t ngrad=0;
  
  // Nuclear coordinates
  arma::mat nuccoord=basis.get_nuclear_coords();
    
  // Initial step size to use
  const double steplen=0.1;
  double dr(steplen);
  // Maximum amount of steps to take in line search
  const int nline=5;
    
  // Density and gradient
  double d;
  arma::vec g;
    
  while(true) {
    // Iteration number
    iiter++;
      
    // Compute density and gradient
    compute_density_gradient(P,basis,r,d,g);
    double gnorm=arma::norm(g,2);
    ndens++; ngrad++;
     
    // Normalize gradient and perform line search
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
    //      printf("Minimal distance to nucleus is %e.\n",mindist);

    // Starting point
    len.push_back(0.0);
    dens.push_back(d);

#ifdef BADERDEBUG
    printf("Step length %e: % f % f % f, density %e, difference %e\n",len[0],r.x,r.y,r.z,dens[0],0.0);
#endif

    // Trace until density does not increase any more.
    do {
      // Increase step size
      len.push_back(len.size()*dr);
      // New point
      coords_t pt=r+gn*len[len.size()-1];
      // and density
      dens.push_back(compute_density(P,basis,pt));
      ndens++;

#ifdef BADERDEBUG	
      printf("Step length %e: % f % f % f, density %e, difference %e\n",len[len.size()-1],pt.x,pt.y,pt.z,dens[dens.size()-1],dens[dens.size()-1]-dens[0]);
#endif
	
    } while(dens[dens.size()-1]>dens[dens.size()-2] && dens.size()<nline);

    // Optimal line length
    double optlen=0.0;
    if(dens[dens.size()-1]>=dens[dens.size()-2])
      // Maximum allowed
      optlen=len[len.size()-1];

    else {
      // Interpolate
      arma::vec ilen(3), idens(3);
      
      if(dens.size()==2) {
	ilen(0)=len[len.size()-2];
	ilen(2)=len[len.size()-1];
	ilen(1)=(ilen(0)+ilen(2))/2.0;
	
	idens(0)=dens[dens.size()-2];
	idens(2)=dens[dens.size()-1];
	idens(1)=compute_density(P,basis,r+gn*ilen(1));
	ndens++;
      } else {
	ilen(0)=len[len.size()-3];
	ilen(1)=len[len.size()-2];
	ilen(2)=len[len.size()-1];
	
	idens(0)=dens[dens.size()-3];
	idens(1)=dens[dens.size()-2];
	idens(2)=dens[dens.size()-1];
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
      for(size_t i=0;i<roots.n_elem;i++)
	if(roots(i)>=ilen(0) && roots(i)<=ilen(2)) {
	  optlen=roots(i);
	  break;
	}
    }

#ifdef BADERDEBUG      
    printf("Optimal step length is %e.\n",optlen);
#endif

    if(std::min(optlen,mindist)<=CONVTHR) {
      // Converged at nucleus.
      r.x=closenuc(0);
      r.y=closenuc(1);
      r.z=closenuc(2);
    }

    if(optlen==0.0) {
      if(dr>=CONVTHR) {
	// Reduce step length.
	dr/=10.0;
	continue;
      } else {
	// Converged
	break;
      }
    } else if(optlen<=CONVTHR)
      // Converged
      break;
      
    // Update point
    r=r+gn*optlen;
  }

  nd+=ndens;
  ng+=ngrad;

#ifdef BADERDEBUG
  printf("Point % .3f % .3f %.3f tracked to maximum at % .3f % .3f % .3f with %s density evaluations.\n",r0.x,r0.y,r0.z,r.x,r.y,r.z,space_number(ndens).c_str());
#endif  
  
  return r;
}
