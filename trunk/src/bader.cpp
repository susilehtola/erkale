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

#include "bader.h"
#include "timer.h"

Bader::Bader() {
}

Bader::~Bader() {
}

void Bader::fill(const BasisSet & basis, const arma::mat & P, double space, double padding) {
  Timer t;
  printf("Filling Bader grid ... ");
  fflush(stdout);

  // Get coordinate matrix
  nuclei=basis.get_nuclear_coords();
  
  // Minimum and maximum coordinates
  start=arma::min(nuclei)-padding;
  arma::vec maxc=arma::max(nuclei)+padding;
  
  // Round to spacing
  for(int ic=0;ic<3;ic++) {
    start(ic)=floor(start(ic)/space)*space;
    maxc(ic)=ceil(maxc(ic)/space)*space;
  }

  // Store spacing
  spacing=space*arma::ones(3);

  // Compute size of array
  size_t Nx=(maxc(0)-start(0))/space;
  size_t Ny=(maxc(1)-start(1))/space;
  size_t Nz=(maxc(2)-start(2))/space;

  // Density array
  dens.zeros(Nx,Ny,Nz);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t iz=0;iz<Nz;iz++) // Loop over slices first, since they're stored continguously in memory
    for(size_t ix=0;ix<Nx;ix++)
      for(size_t iy=0;iy<Ny;iy++) {
	coords_t tmp;
	tmp.x=start(0)+ix*spacing(0);
	tmp.y=start(1)+iy*spacing(1);
	tmp.z=start(2)+iz*spacing(2);
	dens(ix,iy,iz)=compute_density(P,basis,tmp);
      }

  printf("done (%s). Grid has %u points.\n",t.elapsed().c_str(),dens.n_elem);
}

bool Bader::in_cube(const arma::ivec & p) const {
  if( p(0) < 0 || p(0) >= (arma::sword) dens.n_rows )
    return false;
  if( p(1) < 0 || p(1) >= (arma::sword) dens.n_cols )
    return false;
  if( p(2) < 0 || p(2) >= (arma::sword) dens.n_slices )
    return false;
  
  return true;
}

bool Bader::neighbors_assigned(const arma::ivec & p) const {
  // Do its neighbors have the same assignment? 2.3(v)
  bool assigned=true;
  for(int dx=-1;dx<=1;dx++)
    for(int dy=-1;dy<=1;dy++)
      for(int dz=-1;dz<=1;dz++) {
	// Skip current point
	if(dx==0 && dy==0 && dz==0)
	  continue;

	arma::ivec dp(3);
	dp(0)=dx;
	dp(1)=dy;
	dp(2)=dz;

	// Check that we don't run over
	arma::ivec np=p+dp;
	if(!in_cube(np))
	  continue;
	// Check assignment
	if(region(np(0),np(1),np(2))==region(p(0),p(1),p(2)))
	  assigned=false;
      }
  
  return assigned;
}

bool Bader::local_maximum(const arma::ivec & p) const {
  double maxd=0.0;
  for(int dx=-1;dx<=1;dx++)
    for(int dy=-1;dy<=1;dy++)
      for(int dz=-1;dz<=1;dz++) {
	// Skip current point
	if(dx==0 && dy==0 && dz==0)
	  continue;

	arma::ivec dp(3);
	dp(0)=dx;
	dp(1)=dy;
	dp(2)=dz;

	// Check that we don't run over
	arma::ivec np=p+dp;
	if(!in_cube(np))
	  continue;

	// Check for largest density value
	if(dens(p(0)+dx,p(1)+dy,p(2)+dz)>maxd)
	  maxd=dens(p(0)+dx,p(1)+dy,p(2)+dz);
      }
  
  // Are we at a local maximum?
  return maxd<=dens(p(0),p(1),p(2));
}

bool Bader::on_boundary(const arma::ivec & p) const {
  return !neighbors_assigned(p);
}

std::vector<arma::ivec> Bader::classify(arma::ivec p) {
  // Original point
  const arma::ivec p0(p);

  // List of points treated in current path.
  std::vector<arma::ivec> points;
  // Correction step
  arma::vec dr;
  dr.zeros(3);

  arma::trans(p).print("Start");
  
  // Loop over trajectory
  while(true) {
    // First, check if the current point has already a region assignment
    if(region(p(0),p(1),p(2))!=-1 && neighbors_assigned(p)) {
      // Yes. Assign the original point to this point
      region(p0(0),p0(1),p0(2))=region(p(0),p(1),p(2));
      // Stop processing here.
      break;
    }
    
    // Next, check if the current point is a local maximum. 2.3(v)
    if(local_maximum(p)) {
      // Yes, do assignment for the original point
      region(p0(0),p0(1),p0(2))=index;
      // Increment index
      index++;
      break;
    }

    arma::trans(p).print("Next p");
    
    // If we're here, we need to move on the grid.
    // Compute the gradient in the current point
    arma::vec rgrad(3);
    rgrad(0)=(dens(p(0)+1,p(1),p(2))-dens(p(0)-1,p(1),p(2)))/(2*spacing(0));
    rgrad(1)=(dens(p(0),p(1)+1,p(2))-dens(p(0),p(1)-1,p(2)))/(2*spacing(1));
    rgrad(2)=(dens(p(0),p(1),p(2)+1)-dens(p(0),p(1),p(2)-1))/(2*spacing(2));
    
    // Determine step length by allowing at maximum displacement by one grid point in any direction.
    rgrad*=arma::min(spacing/rgrad);
    
    // Determine what is the closest point on the grid to the displacement by rgrad
    arma::ivec dgrid(3);
    for(int ic=0;ic<3;ic++) {
      dgrid(ic)=(int) round(start(ic)/spacing(ic));
    }
    
    // Perform the move on the grid
    p(0)+=dgrid(0);
    p(1)+=dgrid(1);
    p(2)+=dgrid(2);
    
    // Update the correction vector
    dr+=rgrad - dgrid%spacing;
    
    // Check that the correction is smaller than the grid spacing
    for(int ic=0;ic<3;ic++) {
      while(dr(ic) < -0.5*spacing(ic)) {
	p(ic)++;
	dr(ic)+=spacing(ic);
      }
      while(dr(ic) > 0.5*spacing(ic)) {
	p(ic)--;
	dr(ic)-=spacing(ic);
      }
    }
  } // End loop over trajectory

  return points;
}

void Bader::analysis() {
  Timer t;
  printf("Performing Bader analysis ... ");
  fflush(stdout);

  // Index for next region
  index=0;
  
  // Initialize assignment array
  region.ones(dens.n_rows,dens.n_cols,dens.n_slices);
  region*=-1;
  
  // Loop over inside of grid
  for(size_t iiz=1;iiz<dens.n_slices-1;iiz++) // Loop over slices first, since they're stored continguously in memory
    for(size_t iix=1;iix<dens.n_rows-1;iix++)
      for(size_t iiy=1;iiy<dens.n_cols-1;iiy++) {
	
	// Check if point has already been classified
	if(region(iix,iiy,iiz)!=-1)
	  continue;
	
	// Otherwise, continue with the trajectory analysis.
	// Current point is
	arma::ivec p(3);
	p(0)=iix;
	p(1)=iiy;
	p(2)=iiz;
	
	// Get the trajectory
	std::vector<arma::ivec> points=classify(p);
	// and assign the points in the trajectory to the current point
	for(size_t ip=0;ip<points.size();ip++)
	  region(points[ip](0),points[ip](1),points[ip](2))=region(p(0),p(1),p(2));
	
      } // End loop over grid
  printf("done (%s)\n",t.elapsed().c_str());
  
  // Loop over edge points. This can be done in a rough fashion.
  for(size_t iiz=0;iiz<dens.n_slices;iiz+=dens.n_slices-1) // Loop over slices first, since they're stored continguously in memory
    for(size_t iix=0;iix<dens.n_rows;iix+=dens.n_rows-1)
      for(size_t iiy=0;iiy<dens.n_cols;iiy+=dens.n_cols-1) {
	
	// Region has already been assigned?!
	if(region(iix,iiy,iiz)!=-1) {
	  printf("Grid point (%e, %e, %e) already has an assignment?!\n",start(0)+iix*spacing(0),start(1)+iiy*spacing(1),start(2)+iiz*spacing(2));
	  continue;
	}
	
	// Current point
	arma::ivec p(3);
	p(0)=iix;
	p(1)=iiy;
	p(2)=iiz;
	
	// Find maximum density of neighbors.
	double maxd=0.0; // Max density
	arma::sword reg=-1; // Assignment
	for(int dx=-1;dx<=1;dx++)
	  for(int dy=-1;dy<=1;dy++)
	    for(int dz=-1;dz<=1;dz++) {
	      // Skip current point
	      if(dx==0 && dy==0 && dz==0)
		continue;
	      
	      arma::ivec dp(3);
	      dp(0)=dx;
	      dp(1)=dy;
	      dp(2)=dz;
	      
	      // Check that we don't run over
	      arma::ivec np=p+dp;
	      if(!in_cube(np))
		continue;
	      // Check density
	      if(dens(np(0),np(1),np(2))>maxd) {
		maxd=dens(np(0),np(1),np(2));
		reg=region(np(0),np(1),np(2));
	      }
	    }
	
	// Store the region
	region(p(0),p(1),p(2))=reg;
      }	


  printf("Refining analysis ... ");
  fflush(stdout);
  t.set();

  // Finally, analyze which points are on the boundary.
  std::vector<arma::ivec> points;
  for(size_t iiz=0;iiz<dens.n_slices;iiz+=dens.n_slices-1) // Loop over slices first, since they're stored continguously in memory
    for(size_t iix=0;iix<dens.n_rows;iix+=dens.n_rows-1)
      for(size_t iiy=0;iiy<dens.n_cols;iiy+=dens.n_cols-1) {
	
	// Current point
	arma::ivec p(3);
	p(0)=iix;
	p(1)=iiy;
	p(2)=iiz;
	
	// Is the point on a boundary?
	if(on_boundary(p)) {
	  // Add it to the list
	  points.push_back(p);
	  // and clear its assignment
	  region(p(0),p(1),p(2))=-1;
	}
      }
  
  // ... and rerun the analysis for the boundary points
  for(size_t ip=0;ip<points.size();ip++)
    classify(points[ip]);

  printf("done (%s).%i regions found.\n",t.elapsed().c_str(),index);

  // Save regions to file
  region.save("region.dat",arma::raw_ascii);
}

arma::ivec Bader::nuclear_regions() const {
  // Regions of the nuclei
  arma::ivec nucreg(nuclei.n_rows);
  for(size_t inuc=0;inuc<nuclei.n_rows;inuc++) {
    // Determine the grid point in which the nucleus resides
    arma::vec pv=(arma::trans(nuclei.row(inuc))-start)/spacing;

    // The region is
    nucreg(inuc) = region((arma::uword) round(pv(0)),(arma::uword) round(pv(1)), (arma::uword) round(pv(2)));
  }

  return nucreg;
}

arma::vec Bader::charges() const {
  // Charges in the Bader regions
  arma::vec q(index);
  q.zeros();

  // Perform integration
  for(size_t iiz=0;iiz<dens.n_slices;iiz+=dens.n_slices-1) // Loop over slices first, since they're stored continguously in memory
    for(size_t iix=0;iix<dens.n_rows;iix+=dens.n_rows-1)
      for(size_t iiy=0;iiy<dens.n_cols;iiy+=dens.n_cols-1)
	q(region(iix,iiy,iiz))+=dens(iix,iiy,iiz);

  arma::trans(q).print("Raw charges");

  // Plug in the spacing
  q*=spacing(0)*spacing(1)*spacing(2);

  arma::trans(q).print("Normalized charges");

  // Get the nuclear regions
  arma::ivec nucreg=nuclear_regions();
  arma::vec qnuc(nucreg.n_elem);
  for(size_t i=0;i<nucreg.n_elem;i++)
    qnuc(i)=q(nucreg(i));

  return qnuc;
}
