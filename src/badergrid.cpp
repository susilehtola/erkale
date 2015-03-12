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
#include "chebyshev.h"
#include "timer.h"
#include "stringutil.h"
#include "unitary.h"
#include "properties.h"
#include "elements.h"
#include "mathf.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Threshold for vanishingly small density
#define SMALLDENSITY 1e-10

// Convergence threshold of update
#define CONVTHR 1e-3
// Threshold distance for separation of maxima
#define SAMEMAXIMUM (10*CONVTHR)
// Radius small enough to force classification to nuclear region
#define TRUSTRAD 0.1

BaderGrid::BaderGrid() {
}
  
BaderGrid::~BaderGrid() {
}

void BaderGrid::set(const BasisSet & basis, bool ver, bool lobatto) {
  wrk=AngularGrid(lobatto);
  wrk.set_basis(basis);
  basp=&basis;
  // Only need function values
  wrk.set_grad_lapl(false,false);

  verbose=ver;
  basp=&basis;
}

void BaderGrid::construct_bader(const arma::mat & P, double otoler) {
  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());
  
  Timer t;

  size_t nd=0, ng=0;
  
  // Form radial shells
  std::vector<angshell_t> grids;
  for(size_t iat=0;iat<basp->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp->get_nuclear_coords(iat);
    sh.tol=otoler*PRUNETHR;
    
    // Compute necessary number of radial points for atom
    size_t nr=std::max(20,(int) round(-5*(3*log10(otoler)+8-element_row[basp->get_Z(iat)])));

    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev_jac(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;
    
    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      grids.push_back(sh);
    }
  }

  // List of grid points
  std::vector<gridpoint_t> points;
  // Initialize list of maxima
  maxima.clear();
  reggrid.clear();
  for(size_t i=0;i<basp->get_Nnuc();i++) {
    nucleus_t nuc(basp->get_nucleus(i));
    if(!nuc.bsse) {
      // Add to list
      maxima.push_back(nuc.r);
      std::vector<gridpoint_t> ghlp;
      reggrid.push_back(ghlp);
    }
  }  
  Nnuc=maxima.size();

  // Block inside classification?
  std::vector<bool> block(maxima.size(),false);
  
  // Index of last treated atom
  size_t oldatom=-1;
  for(size_t ig=0;ig<grids.size();ig++) {
    // Construct the shell
    wrk.set_grid(grids[ig]);
    grids[ig]=wrk.construct_becke(otoler/nrad[grids[ig].atind]);
    // Form the grid again
    wrk.form_grid();
    
    // Extract the points on the shell
    std::vector<gridpoint_t> shellpoints(wrk.get_grid());
    if(!shellpoints.size())
      continue;

    // Are we inside an established trust radius, or are we close enough to a real nucleus?
    bool inside=false;
    if(grids[ig].R<=TRUSTRAD && !(basp->get_nucleus(grids[ig].atind).bsse))
      inside=true;
    
    else if(!block[grids[ig].atind] && oldatom==grids[ig].atind) {
      // Compute projection of density gradient of points on shell
      arma::vec proj(shellpoints.size());
      coords_t nuccoord(basp->get_nuclear_coords(grids[ig].atind));
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(size_t ip=0;ip<shellpoints.size();ip++) {
	// Compute density gradient
	double d;
	arma::vec g;
	compute_density_gradient(P,*basp,shellpoints[ip].r,d,g);
	
	// Vector pointing to nucleus
	coords_t dRc=nuccoord-shellpoints[ip].r;
	arma::vec dR(3);
	dR(0)=dRc.x;
	dR(1)=dRc.y;
	dR(2)=dRc.z;
	// Compute dot product with gradient
	proj(ip)=arma::norm_dot(dR,g);
      }
      // Increment amount of gradient evaluations
      ng+=shellpoints.size();
      
      // Check if all points are inside
      const double cthcrit=cos(M_PI/4.0);
      inside=(arma::min(proj) >= cthcrit);
    }
    
    // If we are not inside, we need to run a point by point classification.
    if(!inside) {
      Timer tc;
      
      // Reset the trust atom
      oldatom=-1;
      // and the current atom
      block[grids[ig].atind]=true;	
      
      // Loop over points
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for(size_t ip=0;ip<shellpoints.size();ip++) {
	if(compute_density(P,*basp,shellpoints[ip].r)<=SMALLDENSITY) {
	  // Zero density - skip point
	  continue;
	}
	
	// Track the density to its maximum
	coords_t r=track_to_maximum(*basp,P,shellpoints[ip].r,nd,ng);

#ifdef _OPENMP
#pragma omp critical
#endif
	{
	  // Now that we have the maximum, check if it is on the list of known maxima
	  bool found=false;
	  for(size_t im=0;im<maxima.size();im++)
	    if(norm(r-maxima[im])<=SAMEMAXIMUM) {
	      found=true;
	      reggrid[im].push_back(shellpoints[ip]);
	      break;
	    }
	  
	  // Maximum was not found, add it to the list
	  if(!found) {
	    maxima.push_back(r);
	    std::vector<gridpoint_t> ghlp;
	    ghlp.push_back(shellpoints[ip]);
	    reggrid.push_back(ghlp);
	  }
	}
      }

      // Continue with the next radial shell
      continue;
	
    } else {
      // If we are here, then all points belong to this nuclear maximum
      oldatom=grids[ig].atind;
      reggrid[ grids[ig].atind ].insert(reggrid[ grids[ig].atind ].end(), shellpoints.begin(), shellpoints.end());
    }
  }

  
  if(verbose) {
    printf("Bader grid constructed in %s, taking %i density and %i gradient evaluations.\n",t.elapsed().c_str(),(int) nd, (int) ng);
    print_maxima();

    // Amount of integration points
    arma::uvec np(basp->get_Nnuc());
    np.zeros();
    // Amount of function values
    arma::uvec nf(basp->get_Nnuc());
    nf.zeros();
    
    for(size_t i=0;i<grids.size();i++) {
      np(grids[i].atind)+=grids[i].np;
      nf(grids[i].atind)+=grids[i].nfunc;
    }
    printf("Composition of atomic integration grid:\n %7s %7s %10s\n","atom","Npoints","Nfuncs");
    for(size_t i=0;i<basp->get_Nnuc();i++)
      printf(" %4i %-2s %7i %10i\n",(int) i+1, basp->get_symbol(i).c_str(), (int) np(i), (int) nf(i));
    printf("\nAmount of grid points in the regions:\n %7s %7s\n","region","Npoints");
    for(size_t i=0;i<reggrid.size();i++)
      printf(" %4i %7i\n",(int) i+1, (int) reggrid[i].size());
    fflush(stdout);
  }
}

void BaderGrid::construct_voronoi(double otoler) {
  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());
  
  Timer t;
  
  // Form radial shells
  std::vector<angshell_t> grids;
  for(size_t iat=0;iat<basp->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp->get_nuclear_coords(iat);
    sh.tol=otoler*PRUNETHR;
    
    // Compute necessary number of radial points for atom
    size_t nr=std::max(20,(int) round(-5*(3*log10(otoler)+8-element_row[basp->get_Z(iat)])));
    
    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev_jac(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;
    
    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      grids.push_back(sh);
    }
  }

  // List of grid points
  std::vector<gridpoint_t> points;
  // Initialize list of maxima
  maxima.clear();
  reggrid.clear();
  for(size_t i=0;i<basp->get_Nnuc();i++) {
    nucleus_t nuc(basp->get_nucleus(i));
    if(!nuc.bsse) {
      // Add to list
      maxima.push_back(nuc.r);
      std::vector<gridpoint_t> ghlp;
      reggrid.push_back(ghlp);
    }
  }
  Nnuc=maxima.size();
  
  for(size_t ig=0;ig<grids.size();ig++) {
    // Construct the shell
    wrk.set_grid(grids[ig]);
    grids[ig]=wrk.construct_becke(otoler/nrad[grids[ig].atind]);
    // Form the grid again
    wrk.form_grid();
    
    // Extract the points on the shell
    std::vector<gridpoint_t> shellpoints(wrk.get_grid());

    // Loop over the points on the shell
    for(size_t ip=0;ip<shellpoints.size();ip++) {
      // Compute distances to atoms
      arma::vec dist(maxima.size());
      for(size_t ia=0;ia<maxima.size();ia++)
	dist(ia)=normsq(shellpoints[ip].r-maxima[ia]);
      // Region is
      arma::uword idx;
      dist.min(idx);    
      // Assign point to atom
      reggrid[idx].push_back(shellpoints[ip]);
    }
  }

  if(verbose) {
    printf("Voronoi grid constructed in %s.\n",t.elapsed().c_str());
    // Amount of integration points
    arma::uvec np(basp->get_Nnuc());
    np.zeros();
    // Amount of function values
    arma::uvec nf(basp->get_Nnuc());
    nf.zeros();
    
    for(size_t i=0;i<grids.size();i++) {
      np(grids[i].atind)+=grids[i].np;
      nf(grids[i].atind)+=grids[i].nfunc;
    }
    printf("Composition of atomic integration grid:\n %7s %7s %10s\n","atom","Npoints","Nfuncs");
    for(size_t i=0;i<basp->get_Nnuc();i++)
      printf(" %4i %-2s %7i %10i\n",(int) i+1, basp->get_symbol(i).c_str(), (int) np(i), (int) nf(i));
    printf("\nAmount of grid points in the atomic regions:\n %7s %7s\n","region","Npoints");
    for(size_t i=0;i<reggrid.size();i++)
      printf(" %4i %7i\n",(int) i+1, (int) reggrid[i].size());
    fflush(stdout);
  }
}

size_t BaderGrid::get_Nmax() const {
  return maxima.size();
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

arma::mat BaderGrid::regional_overlap(size_t ireg) {
  if(ireg>=maxima.size()) {
    ERROR_INFO();
    throw std::runtime_error("Invalid region!\n");
  }

  // Function values in grid points
  arma::mat bf(basp->get_Nbf(),reggrid[ireg].size());
  arma::vec w(reggrid[ireg].size());
  for(size_t ip=0;ip<reggrid[ireg].size();ip++) {
    // Weight is
    w(ip)=reggrid[ireg][ip].w;
    // Basis function values are
    bf.col(ip)=basp->eval_func(reggrid[ireg][ip].r.x,reggrid[ireg][ip].r.y,reggrid[ireg][ip].r.z);
  }
  
  // Overlap matrix is
  arma::mat Sreg(basp->get_Nbf(),basp->get_Nbf());
  Sreg.zeros();
  increment_lda<double>(Sreg,w,bf);

  return Sreg;
}

arma::vec BaderGrid::regional_charges(const arma::mat & P) {
  arma::vec q(maxima.size());
  q.zeros();
  for(size_t i=0;i<maxima.size();i++) {
    arma::mat Sat(regional_overlap(i));
    q(i)=arma::trace(P*Sat);
  }

  return -q;
}

arma::vec BaderGrid::nuclear_charges(const arma::mat & P) {
  // Get regional charges
  arma::vec qr=regional_charges(P);
  return qr.subvec(0,Nnuc-1);
}

std::vector<arma::mat> BaderGrid::regional_overlap() {
  Timer t;
  if(verbose) {
    printf("Computing regional overlap matrices ... ");
    fflush(stdout);
  }

  std::vector<arma::mat> stack(maxima.size());
  for(size_t i=0;i<stack.size();i++)
    stack[i]=regional_overlap(i);

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
  const size_t nline=5;
    
  // Density and gradient
  double d;
  arma::vec g;
    
  while(true) {
    // Iteration number
    iiter++;
      
    // Compute density and gradient
    compute_density_gradient(P,basis,r,d,g);
    double gnorm=arma::norm(g,2);
    fflush(stdout);
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
    arma::rowvec closenuc=nuccoord.row(0);
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
