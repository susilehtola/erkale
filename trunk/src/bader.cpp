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
#include "stringutil.h"

// Debug printout?
//#define BADERDEBUG

Bader::Bader(bool ver) {
  verbose=ver;
}

Bader::~Bader() {
}

void Bader::analyse(const BasisSet & basis, const arma::mat & P, double space, double padding) {
  Timer t;

  // Get nuclei and nuclear coordinate matrix
  nuclei=basis.get_nuclei();
  nucc=basis.get_nuclear_coords();

  // Minimum and maximum coordinates
  start=arma::trans(arma::min(nucc)-padding);
  arma::vec maxc=arma::max(nucc)+padding;

  // Round to spacing
  for(int ic=0;ic<3;ic++) {
    start(ic)=floor(start(ic)/space)*space;
    maxc(ic)=ceil(maxc(ic)/space)*space;
  }

  // Store spacing
  spacing=space*arma::ones(3);

  // Compute size of array
  array_size.zeros(3);
  array_size(0)=(maxc(0)-start(0))/space;
  array_size(1)=(maxc(1)-start(1))/space;
  array_size(2)=(maxc(2)-start(2))/space;

  // Size in memory is
  size_t Nmem=array_size(0)*array_size(1)*array_size(2)*(sizeof(double)+sizeof(arma::sword));

  if(verbose) {
    printf("\nBader grid is %i x %i x %i, totalling %s points.\n",array_size(0),array_size(1),array_size(2),space_number(array_size(0)*array_size(1)*array_size(2)).c_str());
    printf("Grid will require %s of memory.\n",memory_size(Nmem).c_str());

#ifdef BADERDEBUG
    arma::trans(start).print("Grid start");
    arma::vec end=start+(array_size-1)%spacing;
    arma::trans(end).print("Grid end");
#endif

    printf("Filling grid ... ");
    fflush(stdout);
  }

  // Integrated charge
  double Q=0.0;
  
  // Allocate memory for density
  dens.zeros(array_size(0),array_size(1),array_size(2));
  // Initialize assignment array
  region.zeros(array_size(0),array_size(1),array_size(2));
  region*=-1;
  
  // Fill array
#ifdef _OPENMP
#pragma omp parallel for reduction(+:Q)
#endif
  for(size_t iz=0;iz<dens.n_slices;iz++) // Loop over slices first, since they're stored continguously in memory
    for(size_t ix=0;ix<dens.n_rows;ix++)
      for(size_t iy=0;iy<dens.n_cols;iy++) {
	coords_t tmp;
	tmp.x=start(0)+ix*spacing(0);
	tmp.y=start(1)+iy*spacing(1);
	tmp.z=start(2)+iz*spacing(2);
	dens(ix,iy,iz)=compute_density(P,basis,tmp);
	Q+=dens(ix,iy,iz);
      }
  Q*=spacing(0)*spacing(1)*spacing(2);

  if(verbose)
    printf("done (%s).\n",t.elapsed().c_str());
  
  double PS=arma::trace(P*basis.overlap());
  if(verbose)
    printf("Integral of charge over grid is %e, trace of density matrix is %e, difference %e.\n",Q,PS,Q-PS);

  // Renormalize density to account for inaccurate quadrature
  dens*=PS/Q;
#ifdef BADERDEBUG
  if(verbose)
    printf("Renormalized density on grid to correct norm.\n");
#endif

  // Run analysis
  analysis();
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

bool Bader::on_edge(const arma::ivec & p) const {
  if(p(0) == 0 || p(0) == (arma::sword) dens.n_rows-1 )
    return true;
  if( p(1) == 0 || p(1) == (arma::sword) dens.n_cols-1 )
    return true;
  if( p(2) == 0 || p(2) == (arma::sword) dens.n_slices-1 )
    return true;

  return false;
}

void Bader::check_regions(std::string msg) const {
  size_t nfail=0;
  for(size_t iz=0;iz<dens.n_slices;iz++) // Loop over slices first, since they're stored continguously in memory
    for(size_t ix=0;ix<dens.n_rows;ix++)
      for(size_t iy=0;iy<dens.n_cols;iy++)
	if(region(ix,iy,iz)<0) {
	  nfail++;
	  fprintf(stderr,"Point %u %u %u is in region %i.\n",(unsigned) ix, (unsigned) iy, (unsigned) iz, region(ix,iy,iz));
	}

  if(nfail) {
    std::ostringstream oss;
    oss << "Some points were not classified";
    if(msg.size()) {
      oss << " in " << msg;
    }
    oss << "!\n";
    throw std::runtime_error(oss.str());
  }
}

bool Bader::neighbors_assigned(const arma::ivec & p, int nnei) const {
  // Do its neighbors have the same assignment as p? 2.3(v)

  // If p doesn't have an assignment, this is automatically false.
  if(region(p(0),p(1),p(2))==-1)
    return false;

  bool assigned=true;

#ifdef BADERDEBUG
  if(!in_cube(p)) {
    arma::trans(p).print("Point");
    throw std::runtime_error("The point is not in the cube!\n");
  }
#endif

  // On an edge
  if(on_edge(p)) {
    for(int dx=-nnei;dx<=nnei;dx++)
      for(int dy=-nnei;dy<=nnei;dy++)
	for(int dz=-nnei;dz<=nnei;dz++) {
	  arma::ivec dp(3);
	  dp(0)=dx;
	  dp(1)=dy;
	  dp(2)=dz;

	  // Is the point still in the cube?
	  arma::ivec np=p+dp;
	  if(!in_cube(np))
	    continue;

	  // Check if assignment is same
	  if(region(np(0),np(1),np(2))!=region(p(0),p(1),p(2)))
	    assigned=false;
	}

    // Inside of cube
  } else {
    for(int dx=-1;dx<=1;dx++)
      for(int dy=-1;dy<=1;dy++)
	for(int dz=-1;dz<=1;dz++) {
	  arma::ivec dp(3);
	  dp(0)=dx;
	  dp(1)=dy;
	  dp(2)=dz;

	  // Check if assignment is same
	  arma::ivec np=p+dp;
	  if(region(np(0),np(1),np(2))!=region(p(0),p(1),p(2)))
	    assigned=false;
	}
  }

  return assigned;
}

bool Bader::local_maximum(const arma::ivec & p) const {
  double maxd=0.0;

#ifdef BADERDEBUG
  if(!in_cube(p)) {
    arma::trans(p).print("Point");
    throw std::runtime_error("The point is not in the cube!\n");
  }
#endif

  if(on_edge(p)) {
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

	  // Check maximum value making sure that we don't run over
	  arma::ivec np=p+dp;
	  if(in_cube(np) && dens(p(0)+dx,p(1)+dy,p(2)+dz)>maxd)
	    maxd=dens(p(0)+dx,p(1)+dy,p(2)+dz);
	}

  } else {
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

	  // Check for largest density value
	  arma::ivec np=p+dp;
	  if(dens(p(0)+dx,p(1)+dy,p(2)+dz)>maxd)
	    maxd=dens(p(0)+dx,p(1)+dy,p(2)+dz);
	}
  }

  // Are we at a local maximum?
  return maxd<=dens(p(0),p(1),p(2));
}

bool Bader::on_boundary(const arma::ivec & p, int nnei) const {
  // We are on a boundary, if one or more of the neighboring points
  // doesn't share the same classification. 

  return !neighbors_assigned(p,nnei);
}

arma::vec Bader::gradient(const arma::ivec & p) const {

#ifdef BADERDEBUG
  if(!in_cube(p)) {
    arma::trans(p).print("Point");
    throw std::runtime_error("The point is not in the cube!\n");
  }
#endif

  arma::vec g(3);

  if(on_edge(p)) {
    // Need to account for edges
    if(p(0)==0)
      g(0)=(dens(p(0)+1,p(1),p(2))-dens(p(0)  ,p(1),p(2)))/spacing(0);
    else if(p(0)==(arma::sword) (dens.n_rows-1))
      g(0)=(dens(p(0)  ,p(1),p(2))-dens(p(0)-1,p(1),p(2)))/spacing(0);
    else
      g(0)=(dens(p(0)+1,p(1),p(2))-dens(p(0)-1,p(1),p(2)))/(2*spacing(0));

    if(p(1)==0)
      g(1)=(dens(p(0),p(1)+1,p(2))-dens(p(0),p(1)  ,p(2)))/spacing(1);
    else if(p(1)==(arma::sword) (dens.n_cols-1))
      g(1)=(dens(p(0),p(1)  ,p(2))-dens(p(0),p(1)-1,p(2)))/spacing(1);
    else
      g(1)=(dens(p(0),p(1)+1,p(2))-dens(p(0),p(1)-1,p(2)))/(2*spacing(1));

    if(p(2)==0)
      g(2)=(dens(p(0),p(1),p(2)+1)-dens(p(0),p(1),p(2)  ))/spacing(2);
    else if(p(2)==(arma::sword) (dens.n_slices-1))
      g(2)=(dens(p(0),p(1),p(2)  )-dens(p(0),p(1),p(2)-1))/spacing(2);
    else
      g(2)=(dens(p(0),p(1),p(2)+1)-dens(p(0),p(1),p(2)-1))/(2*spacing(2));

  } else {

    // Safely inside of cube
    g(0)=(dens(p(0)+1,p(1),p(2))-dens(p(0)-1,p(1),p(2)))/(2*spacing(0));
    g(1)=(dens(p(0),p(1)+1,p(2))-dens(p(0),p(1)-1,p(2)))/(2*spacing(1));
    g(2)=(dens(p(0),p(1),p(2)+1)-dens(p(0),p(1),p(2)-1))/(2*spacing(2));
  }

  return g;
}

void Bader::print_neighbors(const arma::ivec & p) const {
  printf("\nNeighbors of point %i %i %i in region %i\n",p(0),p(1),p(2),region(p(0),p(1),p(2)));
  
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
	
	arma::ivec np=p+dp;
	if(!in_cube(np))
	  continue;
	
	// Check maximum value making sure that we don't run over
	printf("\t%i %i %i region %i\n",np(0),np(1),np(2),region(np(0),np(1),np(2)));
      }
}

std::vector<arma::ivec> Bader::classify(arma::ivec p) const {
  // Original point
  const arma::ivec p0(p);

  // List of points treated in current path.
  std::vector<arma::ivec> points;
  points.push_back(p);
  // Correction step
  arma::vec dr;
  dr.zeros(3);

  size_t iiter=0;

  // Loop over trajectory
  while(true) {

    // Check if the current point is a local maximum. 2.3(v)
    if(local_maximum(p)) {
      //      fprintf(stderr,"%i %i %i is local maximum.\n",p(0),p(1),p(2)); fflush(stderr);
      break;
    }

    // Next, check if the current point and its neighbors are already assigned
    if(region(p(0),p(1),p(2))!=-1 && neighbors_assigned(p)) {
      //      fprintf(stderr,"%i %i %i is inside classified region.\n",p(0),p(1),p(2)); fflush(stderr);
      // Stop processing here.
      break;
    }

    // If we're here, we need to move on the grid.
    // Compute the gradient in the current point
    arma::vec rgrad=gradient(p);

    // Determine step length by allowing at maximum displacement by one grid point in any direction.
    rgrad*=arma::min(spacing/arma::abs(rgrad));

    // Determine what is the closest point on the grid to the displacement by rgrad
    arma::ivec dgrid(3);
    for(int ic=0;ic<3;ic++) {
      dgrid(ic)=(arma::sword) round(rgrad(ic)/spacing(ic));
    }

    // Perform the move on the grid
    for(int ic=0;ic<3;ic++)
      if(p(ic)+dgrid(ic)>=0 && p(ic)+dgrid(ic)<array_size(ic))
	// Move is OK.
	p(ic)+=dgrid(ic);
      else
	// We would move outside the grid - don't perform the move
	dgrid(ic)=0;

    // Update the correction vector
    dr+=rgrad - dgrid%spacing;

    // Check that the correction is smaller than the grid spacing
    for(int ic=0;ic<3;ic++) {
      while(dr(ic) < -0.5*spacing(ic) && p(ic)<array_size(ic)-1) {
	p(ic)++;
	dr(ic)+=spacing(ic);
      }
      while(dr(ic) > 0.5*spacing(ic) && p(ic)>0) {
	p(ic)--;
	dr(ic)-=spacing(ic);
      }
    }

    // Add to points list
    points.push_back(p);
    iiter++;

  } // End loop over trajectory

  return points;
}

void Bader::analysis() {
  Timer t;
  if(verbose) {
    printf("Performing Bader analysis ... ");
    fflush(stdout);
  }

  // Reset amount of regions
  Nregions=0;
  
  // Initialize region assignment
  region.ones(array_size(0),array_size(1),array_size(2));
  region*=-1;

  // Loop over inside of grid
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t iiz=0;iiz<dens.n_slices;iiz++) // Loop over slices first, since they're stored continguously in memory
    for(size_t iix=0;iix<dens.n_rows;iix++)
      for(size_t iiy=0;iiy<dens.n_cols;iiy++) {

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

#ifdef _OPENMP
#pragma omp critical
#endif
	{
	  // Assign classification to the points. Ending point of trajectory is
	  arma::ivec pend=points[points.size()-1];

	  // Ended up at a local maximum?
	  if(local_maximum(pend)) {

	    arma::vec maxloc=start+pend%spacing;

	    // Maximum already classified?
	    if(region(pend(0),pend(1),pend(2))!=-1) {
#ifdef BADERDEBUG
	      //	      printf("Ended up at maximum %i at %4i %4i %4i, i.e. at % e % e %e.\n",region(pend(0),pend(1),pend(2)),pend(0),pend(1),pend(2),maxloc(0),maxloc(1),maxloc(2));
#endif
	      for(size_t ip=0;ip<points.size()-1;ip++)
		// Only set classification on points that don't have a
		// classification as of yet, since otherwise the
		// classification of boundary points can fluctuate
		// during the grid assignment.
		if(region(points[ip](0),points[ip](1),points[ip](2))==-1)
		  region(points[ip](0),points[ip](1),points[ip](2))=region(pend(0),pend(1),pend(2));


	      // New region
	    } else {
#ifdef BADERDEBUG
	      if(verbose)
		printf("Maximum %i = %e found at %4i %4i %4i, i.e. at % e % e %e.\n",Nregions,dens(pend(0),pend(1),pend(2)),pend(0),pend(1),pend(2),maxloc(0),maxloc(1),maxloc(2));
#endif

	      // Check that the maximum doesn't vanish
	      if(dens(pend(0),pend(1),pend(2))==0.0) {
		std::ostringstream oss;
		oss << "Zero-density maximum encountered at grid point (" << pend(0) << "," << pend(1) << "," << pend(2) << ").\n";
		oss << "Check the padding, or use an augmented basis set.\n";
		throw std::runtime_error(oss.str());
	      }

	      // Classify trajectory
	      for(size_t ip=0;ip<points.size();ip++)
		if(region(points[ip](0),points[ip](1),points[ip](2))==-1)
		  region(points[ip](0),points[ip](1),points[ip](2))=Nregions;
	      // Increment running number of regions
	      Nregions++;
	    }

	    // Ended up in a point surrounded by points with the same classification
	  } else if(region(pend(0),pend(1),pend(2))!=-1 && neighbors_assigned(pend)) {
	    
	    for(size_t ip=0;ip<points.size()-1;ip++)
	      if(region(points[ip](0),points[ip](1),points[ip](2))==-1)
		region(points[ip](0),points[ip](1),points[ip](2))=region(pend(0),pend(1),pend(2));

	  } else {
	    fprintf(stderr,"%i %i %i is not classified!.\n",pend(0),pend(1),pend(2)); fflush(stderr);
	    print_neighbors(pend);

	    ERROR_INFO();
	    throw std::runtime_error("Should not end here!\n");
	  }
	} // End critical region

      } // End loop over grid

  if(verbose) {
    printf("done (%s). %i regions found.\n",t.elapsed().c_str(),Nregions);
    fflush(stdout);
  }

#ifdef BADERDEBUG
  check_regions("the initial analysis");
#endif

  if(verbose) {
    printf("Refining analysis on ");
    fflush(stdout);
    t.set();
  }

  // Finally, analyze which points are on the boundary. 2.2(vii)
  std::vector<arma::ivec> points;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t iiz=0;iiz<dens.n_slices;iiz++) // Loop over slices first, since they're stored continguously in memory
    for(size_t iix=0;iix<dens.n_rows;iix++)
      for(size_t iiy=0;iiy<dens.n_cols;iiy++) {

	// Current point
	arma::ivec p(3);
	p(0)=iix;
	p(1)=iiy;
	p(2)=iiz;

	// Is the point on a boundary? Since the initial
	// classification is done by trajectories, the points on the
	// boundary may be misclassified.
	// 
	// However, if the point is a local maximum,
	// there is no need for reclassification.
	if(!local_maximum(p) && on_boundary(p)) {
	  // Add it to the list
#ifdef _OPENMP
#pragma omp critical
#endif
	  points.push_back(p);
	}
      }

  // ... reset their classification
  for(size_t ip=0;ip<points.size();ip++)
    region(points[ip](0),points[ip](1),points[ip](2))=-1;

  if(verbose) {
    printf("%s boundary points ... ",space_number(points.size()).c_str());
    fflush(stdout);
  }

  // ... and rerun the analysis
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t ip=0;ip<points.size();ip++) {

    // Get the trajectory from the point
    std::vector<arma::ivec> traj=classify(points[ip]);

    // ... and assign only the initial point.
    arma::ivec p0=traj[0];
    arma::ivec pend=traj[traj.size()-1];
    region(p0(0),p0(1),p0(2))=region(pend(0),pend(1),pend(2));

#ifdef BADERDEBUG
    if(region(pend(0),pend(1),pend(2))==-1) {
      std::ostringstream oss;
      oss << "Reclassification trajectory ends at non-classified point " << pend;
      throw std::runtime_error(oss.str());
    }
#endif
  }

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());
    fflush(stdout);
  }

#ifdef BADERDEBUG
  check_regions("the refinement analysis");
#endif

  // Reorder regions
  reorder();

  // Save regions to file
#ifdef BADERDEBUG
  print_regions();
  print_individual_regions();
#endif
}

arma::ivec Bader::nuclear_regions() const {
#ifdef BADERDEBUG
  check_regions("before nuclear regions");
#endif

  // Regions of the nucc
  arma::ivec nucreg(nucc.n_rows);
  for(size_t inuc=0;inuc<nucc.n_rows;inuc++) {
    // Determine the grid point in which the nucleus resides.
    arma::vec gpf=(arma::trans(nucc.row(inuc))-start)/spacing;
    // Round up to the closest grid point
    arma::ivec gp(3);
    for(int ic=0;ic<3;ic++) {
      gp(ic)=(arma::sword) round(gpf(ic));

      // Check that we don't go outside the array
      if(gp(ic)<0)
	gp(ic)=0;
      else if(gp(ic)>=array_size(ic))
	gp(ic)=array_size(ic)-1;
    }

    // The region is
    nucreg(inuc) = region(gp(0),gp(1),gp(2));

    // Coordinates of the grid point
    arma::vec gpc=start+gp%spacing;
    // Distance from nucleus
    arma::vec gp_dist=gpc-arma::trans(nucc.row(inuc));
    //    printf("Nucleus %i is in region %i, distance to grid point is %e.\n",(int) inuc+1,nucreg(inuc)+1,arma::norm(gp_dist,2));
  }

  return nucreg;
}

void Bader::reorder() {
  // Translation map
  arma::uvec map(Nregions);
  for(arma::uword i=0;i<map.n_elem;i++)
    map(i)=i;

  // Get the nuclear regions
  arma::ivec nucreg=nuclear_regions();

  // Loop over nuclei
  for(arma::uword inuc=0;inuc<nucreg.n_elem;inuc++)
    // The mapping should take the region of the nucleus to inuc.
    if(map(nucreg(inuc))!=inuc) {

      // Find out what entry of map is inuc
      arma::uword imap;
      for(imap=0;imap<map.n_elem;imap++)
	if(map(imap)==inuc)
	  break;

      // Swap the entries
      std::swap(map(nucreg(inuc)),map(imap));
    }

  // Perform remapping
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t iiz=0;iiz<dens.n_slices;iiz++) // Loop over slices first, since they're stored continguously in memory
    for(size_t iix=0;iix<dens.n_rows;iix++)
      for(size_t iiy=0;iiy<dens.n_cols;iiy++)
	region(iix,iiy,iiz)=map(region(iix,iiy,iiz));
}


arma::vec Bader::regional_charges() const {
  // Charges in the Bader regions
  arma::vec q(Nregions);
  q.zeros();

  // Perform integration
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    arma::vec qwrk(q);
#pragma omp for
#endif
    for(size_t iiz=0;iiz<dens.n_slices;iiz++) // Loop over slices first, since they're stored continguously in memory
      for(size_t iix=0;iix<dens.n_rows;iix++)
	for(size_t iiy=0;iiy<dens.n_cols;iiy++) {
#ifdef _OPENMP
	  qwrk(region(iix,iiy,iiz))+=dens(iix,iiy,iiz);
#else
	  q(region(iix,iiy,iiz))+=dens(iix,iiy,iiz);
#endif
	}

#ifdef _OPENMP
#pragma omp critical
    q+=qwrk;
#endif
  }

  // Plug in the spacing, and convert sign
  q*=-spacing(0)*spacing(1)*spacing(2);
  return q;
}

arma::vec Bader::nuclear_charges() const {
  arma::vec q=regional_charges();

  // Get the nuclear regions
  arma::ivec nucreg=nuclear_regions();
  arma::vec qnuc(nucreg.n_elem);
  for(size_t i=0;i<nucreg.n_elem;i++)
    qnuc(i)=q(nucreg(i));

  return qnuc;
}

void Bader::print_regions() const {
  // Open output file.
  FILE *out=fopen("bader_regions.cube","w");

  // Write out comment fields
  Timer t;
  fprintf(out,"ERKALE Bader regions\n");
  fprintf(out,"Generated on %s.\n",t.current_time().c_str());

  // Write out starting point
  fprintf(out,"%7i % g % g % g\n",(int) nuclei.size(),start(0),start(1),start(2));
  // Print amount of points and step sizes in the directions
  fprintf(out,"%7i % g % g % g\n",(int) dens.n_rows,spacing(0),0.0,0.0);
  fprintf(out,"%7i % g % g % g\n",(int) dens.n_cols,0.0,spacing(1),0.0);
  fprintf(out,"%7i % g % g % g\n",(int) dens.n_slices,0.0,0.0,spacing(2));
  // Print out atoms
  for(size_t i=0;i<nuclei.size();i++) {
    nucleus_t nuc=nuclei[i];
    fprintf(out,"%7i %g % g % g % g\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
  }

  // Index of points written
  size_t idx=0;

  // Cube ordering - innermost loop wrt z, inner wrt y and outer wrt x
  for(size_t iix=0;iix<dens.n_rows;iix++)
    for(size_t iiy=0;iiy<dens.n_cols;iiy++) {
      for(size_t iiz=0;iiz<dens.n_slices;iiz++) {
	// Point
	arma::ivec p(3);
	p(0)=iix;
	p(1)=iiy;
	p(2)=iiz;

	// Is the point on a boundary?
	if(on_boundary(p))
	  fprintf(out," % .5e",1.0);
	else
	  fprintf(out," % .5e",0.0);
	idx++;
	if(idx==6) {
	  idx=0;
	  fprintf(out,"\n");
	}
      }
      // y value changes
      if(idx!=0)
	fprintf(out,"\n");
    }
  // Close file
  fclose(out);
}

void Bader::print_individual_regions() const {
  // Open output file.
  FILE *out=fopen("individual_bader_regions.cube","w");

  // Write out comment fields
  Timer t;
  fprintf(out,"ERKALE individual Bader regions\n");
  fprintf(out,"Generated on %s.\n",t.current_time().c_str());

  // Write out starting point. Amount of nuclei as negative, because we use MO format
  fprintf(out,"%7i % g % g % g\n",-(int) nuclei.size(),start(0),start(1),start(2));
  // Print amount of points and step sizes in the directions
  fprintf(out,"%7i % g % g % g\n",(int) dens.n_rows,spacing(0),0.0,0.0);
  fprintf(out,"%7i % g % g % g\n",(int) dens.n_cols,0.0,spacing(1),0.0);
  fprintf(out,"%7i % g % g % g\n",(int) dens.n_slices,0.0,0.0,spacing(2));
  // Print out atoms
  for(size_t i=0;i<nuclei.size();i++) {
    nucleus_t nuc=nuclei[i];
    fprintf(out,"%7i %g % g % g % g\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
  }

  // Index of points written
  size_t idx=1;

  // Print out region indices
  fprintf(out,"%5i", Nregions);
  for(arma::sword ireg=0;ireg<Nregions;ireg++) {
    fprintf(out,"%5i", ireg+1);
    idx++;
    if(idx==10) {
      idx=0;
      fprintf(out,"\n");
    } else if(ireg+1 != Nregions)
      fprintf(out," ");
  }
  if(idx!=0) {
    fprintf(out,"\n");
    idx=0;
  }


  // Cube ordering - innermost loop wrt z, inner wrt y and outer wrt x
  for(size_t iix=0;iix<dens.n_rows;iix++)
    for(size_t iiy=0;iiy<dens.n_cols;iiy++) {
      for(size_t iiz=0;iiz<dens.n_slices;iiz++) {
	// Point
	arma::ivec p(3);
	p(0)=iix;
	p(1)=iiy;
	p(2)=iiz;

	// Is the point in the region?
	for(arma::sword ireg=0;ireg<Nregions;ireg++) {
	  if(region(p(0),p(1),p(2))==ireg)
	    fprintf(out," % .5e",1.0);
	  else
	    fprintf(out," % .5e",0.0);
	  idx++;
	  if(idx==6) {
	    idx=0;
	    fprintf(out,"\n");
	  }
	}
      }
      // y value changes
      if(idx!=0)
	fprintf(out,"\n");
    }
  // Close file
  fclose(out);
}

arma::vec Bader::regional_charges(const BasisSet & basis, const arma::mat & P) const {
  
  // Charges in the Bader regions
  arma::vec q(Nregions);
  q.zeros();

  // Perform integration
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    arma::vec qwrk(q);
#pragma omp for
#endif
    for(size_t iiz=0;iiz<dens.n_slices;iiz++) // Loop over slices first, since they're stored continguously in memory
      for(size_t iix=0;iix<dens.n_rows;iix++)
	for(size_t iiy=0;iiy<dens.n_cols;iiy++) {
	  
	  coords_t tmp;
	  tmp.x=start(0)+iix*spacing(0);
	  tmp.y=start(1)+iiy*spacing(1);
	  tmp.z=start(2)+iiz*spacing(2);
	  double d=compute_density(P,basis,tmp);
	  
#ifdef _OPENMP
	  qwrk(region(iix,iiy,iiz))+=d;
#else
	  q(region(iix,iiy,iiz))+=d;
#endif
	}
    
#ifdef _OPENMP
#pragma omp critical
    q+=qwrk;
#endif
  }
  
  // Plug in the spacing and convert sign
  q*=-spacing(0)*spacing(1)*spacing(2);
  return q;
}

arma::vec Bader::nuclear_charges(const BasisSet & basis, const arma::mat & P) const {
  arma::vec q=regional_charges(basis,P);

  // Get the nuclear regions
  arma::ivec nucreg=nuclear_regions();
  arma::vec qnuc(nucreg.n_elem);
  for(size_t i=0;i<nucreg.n_elem;i++)
    qnuc(i)=q(nucreg(i));
  
  return qnuc;
}

std::vector<arma::mat> Bader::regional_overlap(const BasisSet & basis) const {
  std::vector<arma::mat> Sat(Nregions);

  Timer t;
  if(verbose) {
    printf("Computing regional overlap matrices ... ");
    fflush(stdout);
  }

  // Calculate matrices
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(arma::sword ireg=0;ireg<Nregions;ireg++) {
    // Initialize
    Sat[ireg].zeros(basis.get_Nbf(),basis.get_Nbf());
    
    // Loop over grid
    for(size_t iiz=0;iiz<dens.n_slices;iiz++) // Loop over slices first, since they're stored continguously in memory
      for(size_t iix=0;iix<dens.n_rows;iix++)
	for(size_t iiy=0;iiy<dens.n_cols;iiy++)
	  if(region(iix,iiy,iiz)==ireg) {
	    // Evaluate basis function at the grid point
	    arma::vec bf=basis.eval_func(start(0)+iix*spacing(0),start(1)+iiy*spacing(1),start(2)+iiz*spacing(2));
	    // Add to overlap matrix
	    Sat[ireg]+=bf*arma::trans(bf);
	  }

    // Plug in normalization
    Sat[ireg]*=spacing(0)*spacing(1)*spacing(2);
  }
  
  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());
    fflush(stdout);
  }

  return Sat;
}
