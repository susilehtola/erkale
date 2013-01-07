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


#include <cfloat>
#include <cmath>
#include <cstdio>
// LibXC
#include <xc.h>

#include "dftfuncs.h"
#include "xcfit.h"
#include "elements.h"
#include "mathf.h"
// Lobatto or Lebedev for angular
#include "lobatto.h"
#include "lebedev.h"
// Gauss-Chebyshev for radial integral
#include "chebyshev.h"

#include "stringutil.h"
#include "timer.h"

// OpenMP parallellization for XC calculations
#ifdef _OPENMP
#include <omp.h>
#endif

// Compute closed-shell result from open-shell result
//#define CONSISTENCYCHECK

/* Partitioning functions */

void XCAtomGrid::add_lobatto_shell(atomgrid_t & g, size_t ir) {
  // Add points on ind:th radial shell.

  // Radius
  double rad=g.sh[ir].r;
  // Order of quadrature rule
  int l=g.sh[ir].l;
  // Radial weight
  double wrad=g.sh[ir].w;

  // Number of points in theta
  int nth=(l+3)/2;

  // Get corresponding Lobatto quadrature rule points in theta
  std::vector<double> xl, wl;
  lobatto_compute(nth,xl,wl);

  // Store index of first point
  g.sh[ir].ind0=grid.size();
  // Number of points on this shell
  size_t np=0;

  // Loop over points in theta
  for(int ith=0;ith<nth;ith++) {
    // Compute cos(th) and sin(th);

    double cth=xl[ith];
    double sth=sqrt(1-cth*cth);

    // Determine number of points in phi, defined by smallest integer for which
    // sin^{nphi} \theta < THR
    double thr;
    if(l<=50)
      thr=1e-8;
    else
      thr=1e-9;

    // Calculate nphi
    int nphi=1;
    double t=sth;
    while(t>=thr && nphi<=l+1) {
      nphi++;
      t*=sth;
    }

    // Use an offset in phi?
    double phioff=0.0;
    if(ith%2)
      phioff=M_PI/nphi;

    // Now, generate the points.
    gridpoint_t point;
    double phi, dphi; // Value of phi and the increment
    double cph, sph; // Sine and cosine of phi

    dphi=2.0*M_PI/nphi;

    // Total weight of points on this ring is
    // (disregarding weight from Becke partitioning)
    point.w=2.0*M_PI*wl[ith]/nphi*wrad;
    
    for(int iphi=0;iphi<nphi;iphi++) {
      // Value of phi is
      phi=iphi*dphi+phioff;
      // and the sine and cosine are
      sph=sin(phi);
      cph=cos(phi);

      // Compute x, y and z
      point.r.x=rad*sth*cph;
      point.r.y=rad*sth*sph;
      point.r.z=rad*cth;
      
      // Displace to center
      point.r=point.r+g.cen;

      // Add point
      grid.push_back(point);
      // Increment number of points
      np++;
    }
  }

  // Store number of points on this shell
  g.sh[ir].np=np;
}

void XCAtomGrid::add_lebedev_shell(atomgrid_t & g, size_t ir) {
  // Add points on ind:th radial shell.

  // Radius
  double rad=g.sh[ir].r;
  // Order of quadrature rule
  int l=g.sh[ir].l;
  // Radial weight
  double wrad=g.sh[ir].w;

  // Get quadrature rule
  std::vector<lebedev_point_t> points=lebedev_sphere(l);

  // Store index of first point
  g.sh[ir].ind0=grid.size();
  // Number of points on this shell
  size_t np=points.size();

  // Loop over points
  for(size_t i=0;i<points.size();i++) {
    gridpoint_t point;

    point.r.x=rad*points[i].x;
    point.r.y=rad*points[i].y;
    point.r.z=rad*points[i].z;
    // Displace to center
    point.r=point.r+g.cen;

    // Compute quadrature weight
    // (Becke weight not included)
    point.w=wrad*points[i].w;

    // Add point
    grid.push_back(point);
  }
  
  // Store number of points on this shell
  g.sh[ir].np=np;
}

void XCAtomGrid::becke_weights(const BasisSet & bas, const atomgrid_t & g, size_t ir) {
  // Compute weights of points.

  // Number of atoms in system
  const size_t Nat=bas.get_Nnuc();

  // Helper arrays
  std::vector<double> atom_dist;
  std::vector<double> atom_weight;
  std::vector< std::vector<double> > mu_ab;
  std::vector< std::vector<double> > smu_ab;

  // Initialize memory
  atom_dist.resize(Nat);
  atom_weight.resize(Nat);
  mu_ab.resize(Nat);
  smu_ab.resize(Nat);
  for(size_t i=0;i<Nat;i++) {
    mu_ab[i].resize(Nat);
    smu_ab[i].resize(Nat);
  }

  // Loop over points on wanted radial shell
  for(size_t ip=g.sh[ir].ind0;ip<g.sh[ir].ind0+g.sh[ir].np;ip++) {
    // Coordinates of the point are
    coords_t coord_p=grid[ip].r;
    
    // Compute distance of point to atoms
    for(size_t iat=0;iat<Nat;iat++)
      atom_dist[iat]=norm(bas.get_coords(iat)-coord_p);
    
    // Compute mu_ab
    for(size_t iat=0;iat<Nat;iat++) {
      // Diagonal
      mu_ab[iat][iat]=0.0;
      // Off-diagonal
      for(size_t jat=0;jat<iat;jat++) {
	mu_ab[iat][jat]=(atom_dist[iat]-atom_dist[jat])/bas.nuclear_distance(iat,jat);
	mu_ab[jat][iat]=-mu_ab[iat][jat];
      }
    }

    // Compute s(mu_ab)
    for(size_t iat=0;iat<Nat;iat++)
      for(size_t jat=0;jat<Nat;jat++) {
	smu_ab[iat][jat]=f_s(mu_ab[iat][jat],0.7);
      }    

    // Then, compute atomic weights
    for(size_t iat=0;iat<Nat;iat++) {
      atom_weight[iat]=1.0;

      for(size_t jat=0;jat<iat;jat++)
	atom_weight[iat]*=smu_ab[iat][jat];
      for(size_t jat=iat+1;jat<Nat;jat++)
	atom_weight[iat]*=smu_ab[iat][jat];
    }

    // Compute sum of weights
    double awsum=0.0;
    for(size_t iat=0;iat<Nat;iat++)
      awsum+=atom_weight[iat];

    // The Becke weight is
    grid[ip].w*=atom_weight[g.atind]/awsum;
  }
}

void XCAtomGrid::prune_points(double tolv, const radshell_t & rg) {
  // Prune points with small weight.

  // First point on radial shell
  size_t ifirst=rg.ind0;
  // Last point on radial shell
  size_t ilast=ifirst+rg.np;

  for(size_t i=ilast;(i>=ifirst && i<grid.size());i--)
    if(grid[i].w<tolv)
      grid.erase(grid.begin()+i);
}


void XCAtomGrid::free() {
  // Free integration points
  grid.clear();

  // Free values of basis functions
  flist.clear();
  glist.clear();

  // Free LDA stuff
  rho.clear();
  exc.clear();
  vxc.clear();

  // Free GGA stuff
  sigma.clear();
  vsigma.clear();
}

void XCAtomGrid::update_density(const arma::vec & gamma) {
  // Update values of densitty 

  if(!gamma.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - density matrix is empty!\n");
  }

  // Non-polarized calculation.
  polarized=false;

  // Check consistency of allocation
  if(rho.size()!=grid.size())
    rho.resize(grid.size());

  // Loop over points
  for(size_t ip=0;ip<grid.size();ip++) {
    // Calculate density
    rho[ip]=eval_dens(gamma,ip);
  }

  if(do_grad) {
    // Adjust size of grid
    if(grho.size()!=3*rho.size())
      grho.resize(3*rho.size());
    if(sigma.size()!=grid.size())
      sigma.resize(grid.size());

    double grad[3];
    for(size_t ip=0;ip<grid.size();ip++) {
      // Calculate gradient
      eval_grad(gamma,ip,grad);
      // Store it
      for(int ic=0;ic<3;ic++)
	grho[3*ip+ic]=grad[ic];

      // Compute sigma as well
      sigma[ip]=grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2];
    }
  }
}

void XCAtomGrid::update_density(const arma::vec & gammaa, const arma::vec & gammab) {
  // Update values of densitty 

  if(!gammaa.n_elem || !gammab.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - density matrix is empty!\n");
  }

  // Polarized calculation.
  polarized=true;

  // Check consistency of allocation
  if(rho.size()!=2*grid.size())
    rho.resize(2*grid.size());

  // Loop over points
  for(size_t ip=0;ip<grid.size();ip++) {
    // Compute densities
    rho[2*ip]=eval_dens(gammaa,ip);
    rho[2*ip+1]=eval_dens(gammab,ip);
  }

  if(do_grad) {
    if(grho.size()!=6*rho.size())
      grho.resize(6*rho.size());
    if(sigma.size()!=3*grid.size())
      sigma.resize(3*grid.size());

    double grada[3];
    double gradb[3];

    for(size_t ip=0;ip<grid.size();ip++) {
      // Compute gradients
      eval_grad(gammaa,ip,grada);
      eval_grad(gammab,ip,gradb);
      // and store them
      for(int ic=0;ic<3;ic++) {
	grho[6*ip+ic]=grada[ic];
	grho[6*ip+3+ic]=gradb[ic];
      }
      
      // Compute values of sigma
      sigma[3*ip]  =grada[0]*grada[0] + grada[1]*grada[1] + grada[2]*grada[2];
      sigma[3*ip+1]=grada[0]*gradb[0] + grada[1]*gradb[1] + grada[2]*gradb[2];
      sigma[3*ip+2]=gradb[0]*gradb[0] + gradb[1]*gradb[1] + gradb[2]*gradb[2];
    }
  }
}

double XCAtomGrid::eval_dens(const arma::vec & gamma, size_t ip) const {
  // Density
  double d=0.0;
  
  // Loop over functions on point
  size_t first=grid[ip].f0;
  size_t last=first+grid[ip].nf;

  size_t i;

  for(size_t ii=first;ii<last;ii++) {
    // Index of function is
    i=flist[ii].ind;
    d+=gamma(i)*flist[ii].f;
  }

  // Normalize negative densities (may cause problems in libxc)
  if(d<0.0)
    d=0.0;
  
  return d;
}

void XCAtomGrid::eval_grad(const arma::vec & gamma, size_t ip, double *g) const {
  // Initialize gradient
  for(int i=0;i<3;i++)
    g[i]=0.0;

  // Loop over functions on point
  size_t first=grid[ip].f0;
  size_t last=first+grid[ip].nf;

  size_t i;

  for(size_t ii=first;ii<last;ii++) {
    // Index of function is
    i=flist[ii].ind;
    
    for(int ic=0;ic<3;ic++)
      g[ic]+=gamma(i)*glist[3*ii+ic];
  }
}

double XCAtomGrid::compute_Nel() const {
  double nel=0.0;

  if(!polarized)
    for(size_t ip=0;ip<grid.size();ip++)
      nel+=grid[ip].w*rho[ip];
  else
    for(size_t ip=0;ip<grid.size();ip++)
      nel+=grid[ip].w*(rho[2*ip]+rho[2*ip+1]);

  return nel;
}

void XCAtomGrid::init_xc() {
  // Size of grid.
  const size_t N=grid.size();

  // Check allocation of arrays.
  if(exc.size()!=N)
    exc.resize(N);

  if(!polarized) {
    // Restricted case - all arrays of length N

    if(vxc.size()!=N)
      vxc.resize(N);

    if(do_grad) {
      if(vsigma.size()!=N)
	vsigma.resize(N);
    }
  } else {
    // Unrestricted case - arrays of length 2N or 3N
    if(vxc.size()!=2*N)
      vxc.resize(2*N);

    if(do_grad) {
      if(vsigma.size()!=3*N)
	vsigma.resize(3*N);
    }
  }

  // Initial values
  do_gga=false;

  // Fill arrays with zeros.
  for(size_t i=0;i<exc.size();i++)
    exc[i]=0.0;
  for(size_t i=0;i<vxc.size();i++)
    vxc[i]=0.0;
  for(size_t i=0;i<vsigma.size();i++)
    vsigma[i]=0.0;
}

void XCAtomGrid::compute_xc(int func_id) {
  // Compute exchange-correlation functional

  int nspin;
  if(!polarized)
    nspin=XC_UNPOLARIZED;
  else
    nspin=XC_POLARIZED;
  
  // Correlation and exchange functionals
  xc_func_type func;
  if(xc_func_init(&func, func_id, nspin) != 0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Functional "<<func_id<<" not found!"; 
    throw std::runtime_error(oss.str());
  }

  // Which functional is in question?
  bool gga=false;
  
  // Determine the family
  switch(func.info->family)
    {
    case XC_FAMILY_LDA:
      gga=false;
      break;
      
    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
      gga=true;
      break;
      
    default:
      {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Functional family " << func.info->family << " not currently supported in XC fitting!\n";
	throw std::runtime_error(oss.str());
      }
    }
  
  // Update controlling flags for eval_Fxc (exchange and correlation
  // parts might be of different type)
  do_gga=do_gga || gga;

  // Amount of grid points
  const size_t N=grid.size();

  // Work arrays - exchange and correlation are computed separately
  std::vector<double> exc_wrk(exc);
  std::vector<double> vxc_wrk(vxc);
  std::vector<double> vsigma_wrk(vsigma);

  // Evaluate functionals.
  if(gga) // GGA
    xc_gga_exc_vxc(&func, N, &rho[0], &sigma[0], &exc_wrk[0], &vxc_wrk[0], &vsigma_wrk[0]);
  else // LDA
    xc_lda_exc_vxc(&func, N, &rho[0], &exc_wrk[0], &vxc_wrk[0]);

  // Sum to total arrays containing both exchange and correlation
  for(size_t i=0;i<exc.size();i++)
    exc[i]+=exc_wrk[i];
  for(size_t i=0;i<vxc.size();i++)
    vxc[i]+=vxc_wrk[i];
  for(size_t i=0;i<vsigma.size();i++)
    vsigma[i]+=vsigma_wrk[i];

  /*
  // Sanity check
  size_t nerr=0;
  for(size_t i=0;i<exc.size();i++)
    if(std::isnan(exc_wrk[i])) {
      nerr++;
      fprintf(stderr,"exc[%i]=%e\n",(int) i, exc_wrk[i]);
    }
  for(size_t i=0;i<vxc.size();i++)
    if(std::isnan(vxc_wrk[i])) {
      nerr++;
      fprintf(stderr,"rho[%i]=%e, vxc[%i]=%e\n",(int) i, rho[i],(int) i, vxc_wrk[i]);
    }
  for(size_t i=0;i<vsigma.size();i++)
    if(std::isnan(vsigma_wrk[i])) {
      nerr++;
      fprintf(stderr,"sigma[%i]=%e, vsigma[%i]=%e\n",(int) i, sigma[i],(int) i, vsigma_wrk[i]);
    }

  if(nerr!=0) {
    fprintf(stderr,"%u errors with funcid=%i.\n",(unsigned int) nerr, func_id);
    throw std::runtime_error("NaN error\n");
  }
  */    

  // Free functional
  xc_func_end(&func);
}

double XCAtomGrid::eval_Exc() const {
  double Exc=0.0;

  if(!polarized)
    for(size_t i=0;i<grid.size();i++)
      Exc+=exc[i]*rho[i]*grid[i].w;
  else
    for(size_t i=0;i<grid.size();i++)
      Exc+=exc[i]*(rho[2*i]+rho[2*i+1])*grid[i].w;
  
  return Exc;
}

void XCAtomGrid::eval_Fxc(arma::vec & H) const {
  double xcfac;

  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
  }

  // LDA part
  for(size_t ip=0;ip<grid.size();ip++) {
    // Factor in common for basis functions
    xcfac=grid[ip].w*vxc[ip];

    // Loop over functions on grid point
    size_t first=grid[ip].f0;
    size_t last=first+grid[ip].nf;

    for(size_t ii=first;ii<last;ii++) {
      // Get index of function
      size_t i=flist[ii].ind;
      H(i)+=xcfac*flist[ii].f;
    }
  }
  
  // GGA part
  if(do_gga) {
    double xcvec[3];

    for(size_t ip=0;ip<grid.size();ip++) {
      // Factor in common for basis functions
      for(int ic=0;ic<3;ic++)
	xcvec[ic]=2.0*grid[ip].w*vsigma[ip]*grho[3*ip+ic];
      
      // Loop over functions on grid point
      size_t first=grid[ip].f0;
      size_t last=first+grid[ip].nf;
      
      for(size_t ii=first;ii<last;ii++) {
	// Get index of function
	size_t i=flist[ii].ind;
	for(int ic=0;ic<3;ic++)
	  H(i)+=xcvec[ic]*glist[3*ii+ic];
      }
    }
  }
}

void XCAtomGrid::eval_Fxc(arma::vec & Ha, arma::vec & Hb) const {
  double xcfaca, xcfacb;

  if(!polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
  }

  // LDA part
  for(size_t ip=0;ip<grid.size();ip++) {
    // Factor in common for basis functions
    xcfaca=grid[ip].w*vxc[2*ip];
    xcfacb=grid[ip].w*vxc[2*ip+1];

    // Loop over functions on grid point
    size_t first=grid[ip].f0;
    size_t last=first+grid[ip].nf;

    for(size_t ii=first;ii<last;ii++) {
      // Get index of function
      size_t i=flist[ii].ind;
      Ha(i)+=xcfaca*flist[ii].f;
      Hb(i)+=xcfacb*flist[ii].f;
    }
  }

  // GGA part
  if(do_gga) {
    double xcveca[3], xcvecb[3];
    
    for(size_t ip=0;ip<grid.size();ip++) {
      // Factor in common for basis functions
      for(int ic=0;ic<3;ic++) {
	xcveca[ic]=grid[ip].w*(2.0*vsigma[3*ip]  *grho[6*ip+ic]   + vsigma[3*ip+1]*grho[6*ip+3+ic]);
	xcvecb[ic]=grid[ip].w*(2.0*vsigma[3*ip+2]*grho[6*ip+3+ic] + vsigma[3*ip+1]*grho[6*ip+ic]);
      }

      // Loop over functions on grid point
      size_t first=grid[ip].f0;
      size_t last=first+grid[ip].nf;
      
      for(size_t ii=first;ii<last;ii++) {
	// Get index of function
	size_t i=flist[ii].ind;

	for(int ic=0;ic<3;ic++) {	  
	  Ha(i)+=xcveca[ic]*glist[3*ii+ic];
	  Hb(i)+=xcvecb[ic]*glist[3*ii+ic];
	}
      }
    }
  }
}

XCAtomGrid::XCAtomGrid(bool lobatto, double toler) {
  use_lobatto=lobatto;

  // These should really be set separately using the routines below.
  tol=toler;
  do_grad=false;
}  

void XCAtomGrid::set_tolerance(double toler) {
  tol=toler;
}

void XCAtomGrid::check_grad(int x_func, int c_func) {
  // Do we need gradients?
  do_grad=false;
  if(x_func>0)
    do_grad=do_grad || gradient_needed(x_func);
  if(c_func>0)
    do_grad=do_grad || gradient_needed(c_func);
}

atomgrid_t XCAtomGrid::construct(const BasisSet & bas, const arma::vec & gamma, size_t cenind, int x_func, int c_func, bool verbose, const DensityFit & dfit) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells                         
  // See Köster et al for specifics.

  // Returned info
  atomgrid_t ret;
  ret.ngrid=0;
  ret.nfunc=0;

  Timer t;

  // Store index of center
  ret.atind=cenind;
  // and its coordinates
  ret.cen=bas.get_coords(cenind);

  // Compute necessary number of radial points
  size_t nrad=std::max(20,(int) round(-5*(3*log10(tol)+6-element_row[bas.get_Z(ret.atind)])));

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> xc, wc;
  chebyshev(nrad,xc,wc);

  // Allocate memory
  ret.sh.resize(nrad);

  // Loop over radii
  double rad, jac;
  for(size_t ir=0;ir<xc.size();ir++) {
    // Calculate value of radius
    rad=1.0/M_LN2*log(2.0/(1.0-xc[ir]));

    // Jacobian of transformation is
    jac=1.0/M_LN2/(1.0-xc[ir]);
    // so total quadrature weight is
    double weight=wc[ir]*rad*rad*jac;

    // Store shell data
    ret.sh[ir].r=rad;
    ret.sh[ir].w=weight;
    ret.sh[ir].l=3;
  }
  
  // Number of basis functions
  size_t Naux=dfit.get_Naux();
  size_t Norb=dfit.get_Norb();

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(tol));

  // Old and new diagonal elements of Hamiltonian
  arma::vec Hold(Norb), Hnew(Norb);
  Hold.zeros();

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits shell by shell
  for(size_t ir=0;ir<ret.sh.size();ir++) {

    do {
      // Clear current grid points and function values
      free();

      // Form radial shell
      if(use_lobatto)
	add_lobatto_shell(ret,ir);
      else
	add_lebedev_shell(ret,ir);
      // Compute Becke weights for radial shell
      becke_weights(bas,ret,ir);
      // Prune points with small weight
      prune_points(1e-8*tol,ret.sh[ir]);

      // Compute values of basis functions
      compute_bf(bas,ret,ir);

      // Compute density
      update_density(gamma);

      // Clean out Hamiltonian
      Hnew.zeros();

      // Compute exchange and correlation.
      init_xc();
      // Compute the functionals
      if(x_func>0)
	compute_xc(x_func);
      if(c_func>0)
	compute_xc(c_func);

      // Construct the Fock matrix
      arma::vec Fvec(Naux);
      Fvec.zeros();
      eval_Fxc(Fvec);
      Hnew=dfit.invert_expansion_diag(Fvec);
      
      // Compute maximum difference of diagonal elements of Fock matrix
      maxdiff=0.0;
      for(size_t i=0;i<Norb;i++)
	if(fabs(Hold[i]-Hnew[i])>maxdiff)
	  maxdiff=fabs(Hold[i]-Hnew[i]);

      // Copy contents
      Hold=Hnew;

      // Increment order if tolerance not achieved.
      if(maxdiff>tol/xc.size()) {
	if(use_lobatto)
	  ret.sh[ir].l+=2;
	else {
	  // Need to determine what is next order of Lebedev
	  // quadrature that is supported.
	  ret.sh[ir].l=next_lebedev(ret.sh[ir].l);
	}
      }
    } while(maxdiff>tol/xc.size() && ret.sh[ir].l<=lmax);

    // Increase number of points and function values
    ret.ngrid+=grid.size();
    ret.nfunc+=flist.size();
  }
  
  // Free memory once more
  free();
  
  if(verbose) {
    printf("\t%4u %7u %8u %s\n",(unsigned int) ret.atind+1,(unsigned int) ret.ngrid,(unsigned int) ret.nfunc,t.elapsed().c_str());
    fflush(stdout);
  }

  return ret;
}

atomgrid_t XCAtomGrid::construct(const BasisSet & bas, const arma::vec & gammaa, const arma::vec & gammab, size_t cenind, int x_func, int c_func, bool verbose, const DensityFit & dfit) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells                         
  // See Köster et al for specifics.

  Timer t;

  // Returned info
  atomgrid_t ret;
  ret.ngrid=0;
  ret.nfunc=0;

  // Store index of center
  ret.atind=cenind;
  // and its coordinates
  ret.cen=bas.get_coords(cenind);

  // Compute necessary number of radial points
  size_t nrad=std::max(20,(int) round(-5*(3*log10(tol)+6-element_row[bas.get_Z(ret.atind)])));

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> xc, wc;
  chebyshev(nrad,xc,wc);

  // Allocate memory
  ret.sh.resize(nrad);

  // Loop over radii
  double rad, jac;
  for(size_t ir=0;ir<xc.size();ir++) {
    // Calculate value of radius
    rad=1.0/M_LN2*log(2.0/(1.0-xc[ir]));

    // Jacobian of transformation is
    jac=1.0/M_LN2/(1.0-xc[ir]);
    // so total quadrature weight is
    double weight=wc[ir]*rad*rad*jac;

    // Store shell data
    ret.sh[ir].r=rad;
    ret.sh[ir].w=weight;
    ret.sh[ir].l=3;
  }
  
  // Number of basis functions
  size_t Naux=dfit.get_Naux();
  size_t Norb=dfit.get_Norb();

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(tol));

  // Old and new diagonal elements of Hamiltonian
  arma::vec Haold(Norb), Hanew(Norb);
  arma::vec Hbold(Norb), Hbnew(Norb);

  Haold.zeros();
  Hanew.zeros();
  Hbold.zeros();
  Hbnew.zeros();

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits shell by shell
  for(size_t ir=0;ir<ret.sh.size();ir++) {

    do {
      // Clear current grid points and function values
      free();

      // Form radial shell
      if(use_lobatto)
	add_lobatto_shell(ret,ir);
      else
	add_lebedev_shell(ret,ir);
      // Compute Becke weights for radial shell
      becke_weights(bas,ret,ir);
      // Prune points with small weight
      prune_points(1e-8*tol,ret.sh[ir]);

      // Compute values of basis functions
      compute_bf(bas,ret,ir);

      // Compute density
      update_density(gammaa,gammab);

      // Compute exchange and correlation.
      init_xc();
      // Compute the functionals
      if(x_func>0)
	compute_xc(x_func);
      if(c_func>0)
	compute_xc(c_func);
      // and construct the Fock matrices
      arma::vec Favec(Naux);
      arma::vec Fbvec(Naux);
      Favec.zeros();
      Fbvec.zeros();
      eval_Fxc(Favec,Fbvec);
      Hanew=dfit.invert_expansion_diag(Favec);
      Hbnew=dfit.invert_expansion_diag(Fbvec);
      
      // Compute maximum difference of diagonal elements of Fock matrix
      maxdiff=0.0;
      for(size_t i=0;i<Norb;i++) {
	double tmp=std::max(fabs(Hanew[i]-Haold[i]),fabs(Hbnew[i]-Hbold[i]));
	if(tmp>maxdiff)
	  maxdiff=tmp;
      }
      
      // Copy values
      Haold=Hanew;
      Hbold=Hbnew;

      // Increment order if tolerance not achieved.
      if(maxdiff>tol/xc.size()) {
	if(use_lobatto)
	  ret.sh[ir].l+=2;
	else {
	  // Need to determine what is next order of Lebedev
	  // quadrature that is supported.
	  ret.sh[ir].l=next_lebedev(ret.sh[ir].l);
	}
      }
    } while(maxdiff>tol/xc.size() && ret.sh[ir].l<=lmax);

    // Increase number of points and function values
    ret.ngrid+=grid.size();
    ret.nfunc+=flist.size();
  }

  // Free memory once more
  free();
  
  if(verbose) {
    printf("\t%4u %7u %8u %s\n",(unsigned int) ret.atind+1,(unsigned int) ret.ngrid,(unsigned int) ret.nfunc,t.elapsed().c_str());
    fflush(stdout);
  }

  return ret;
}

XCAtomGrid::~XCAtomGrid() {
}

void XCAtomGrid::form_grid(const BasisSet & bas, atomgrid_t & g) {
  // Clear anything that already exists
  free();

  // Check allocation
  grid.reserve(g.ngrid);

  // Loop over radial shells
  for(size_t ir=0;ir<g.sh.size();ir++) {
    // Add grid points
    if(use_lobatto)
      add_lobatto_shell(g,ir);			
    else
      add_lebedev_shell(g,ir);
    // Do Becke weights
    becke_weights(bas,g,ir);
    // Prune points with small weight
    prune_points(1e-8*tol,g.sh[ir]);
  }
}

void XCAtomGrid::compute_bf(const BasisSet & bas, const atomgrid_t & g) {
  // Check allocation
  flist.reserve(g.nfunc);
  if(do_grad)
    glist.reserve(3*g.nfunc);

  // Loop over radial shells
  for(size_t ir=0;ir<g.sh.size();ir++) {
    compute_bf(bas,g,ir);
  }
}
  

void XCAtomGrid::compute_bf(const BasisSet & bas, const atomgrid_t & g, size_t irad) {
  // Compute values of relevant basis functions on irad:th shell

  //  fprintf(stderr,"Computing bf of rad shell %i of atom %i\n",(int) g.atind,(int) irad);

  // Get distances of other nuclei
  std::vector<double> nucdist=bas.get_nuclear_distances(g.atind);

  // Current radius
  double rad=g.sh[irad].r;

  // Get ranges of shells
  std::vector<double> shran=bas.get_shell_ranges();

  // Indices of shells to compute
  std::vector<size_t> compute_shells;

  // Determine which shells might contribute
  for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
    // Determine closest distance of nucleus
    double dist=fabs(nucdist[inuc]-rad);
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=bas.get_shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++) {
      
      // Shell is relevant if range is larger than minimal distance
      if(dist<=shran[shellinds[ish]]) {
	// Add shell to list of shells to compute
	compute_shells.push_back(shellinds[ish]);
      }
    }
  }

  if(do_grad) {
    // Loop over points
    for(size_t ip=g.sh[irad].ind0;ip<g.sh[irad].ind0+g.sh[irad].np;ip++) {
      // Store index of first function on grid point
      grid[ip].f0=flist.size();
      // Number of functions on point
      grid[ip].nf=0;
      
      // Loop over shells
      for(size_t ish=0;ish<compute_shells.size();ish++) {
	// Center of shell is
	coords_t shell_center=bas.get_center(compute_shells[ish]);
	// Compute distance of point to center of shell
	double shell_dist=norm(shell_center-grid[ip].r);
	// Add shell to point if it is within the range of the shell
	if(shell_dist<shran[compute_shells[ish]]) {
	  // Index of first function on shell is
	  size_t ind0=bas.get_first_ind(compute_shells[ish]);
	  
	  // Compute values of basis functions
	  arma::vec fval=bas.eval_func(compute_shells[ish],grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // and their gradients
	  arma::mat gval=bas.eval_grad(compute_shells[ish],grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  
	  // and add them to the list
	  bf_f_t hlp;
	  for(size_t ifunc=0;ifunc<fval.n_elem;ifunc++) {
	    // Index of function is
	    hlp.ind=ind0+ifunc;
	    // Value is
	    hlp.f=fval(ifunc);
	    // Add to stack
	    flist.push_back(hlp);
	    // and gradient, too
	    for(int ic=0;ic<3;ic++)
	      glist.push_back(gval(ifunc,ic));
	    
	    // Increment number of functions in point
	    grid[ip].nf++;
	  }
	}
      }
    }
  } else {
    // Loop over points
    for(size_t ip=g.sh[irad].ind0;ip<g.sh[irad].ind0+g.sh[irad].np;ip++) {
      // Store index of first function on grid point
      grid[ip].f0=flist.size();
      // Number of functions on point
      grid[ip].nf=0;
      
      // Loop over shells
      for(size_t ish=0;ish<compute_shells.size();ish++) {
	// Center of shell is
	coords_t shell_center=bas.get_center(compute_shells[ish]);
	// Compute distance of point to center of shell
	double shell_dist=norm(shell_center-grid[ip].r);
	// Add shell to point if it is within the range of the shell
	if(shell_dist<shran[compute_shells[ish]]) {
	  // Index of first function on shell is
	  size_t ind0=bas.get_first_ind(compute_shells[ish]);

	  // Compute values of basis functions
	  arma::vec fval=bas.eval_func(compute_shells[ish],grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  
	  // and add them to the list
	  bf_f_t hlp;
	  for(size_t ifunc=0;ifunc<fval.n_elem;ifunc++) {
	    // Index of function is
	    hlp.ind=ind0+ifunc;
	    // Value is
	    hlp.f=fval(ifunc);

	    // Add to stack
	    flist.push_back(hlp);
	    // Increment number of functions in point
	    grid[ip].nf++;
	  }
	}
      }
    }
  }
}

XCGrid::XCGrid(const BasisSet * fitbas, const DensityFit * dfit, bool ver, bool lobatto) {
  fitbasp=fitbas;
  dfitp=dfit;
  verbose=ver;
  use_lobatto=lobatto;

  // Allocate atomic grids
  grids.resize(fitbas->get_Nnuc());

  // Allocate work grids
#ifdef _OPENMP
  int nth=omp_get_max_threads();
  for(int i=0;i<nth;i++)
    wrk.push_back(XCAtomGrid(lobatto));
#else
  wrk.push_back(XCAtomGrid(lobatto));
#endif
}

XCGrid::~XCGrid() {
}

arma::vec XCGrid::expand(const arma::mat & P) const {
  // Do the fitting
  arma::vec gamma=dfitp->compute_expansion(P);

  /*
  // Compute norm
  arma::vec fitint=fitbasp->integral();
  
  double fitnorm=arma::dot(gamma,fitint);
  fprintf(stderr,"Fitted norm is %e.\n",fitnorm);
  */

  return gamma;
}

arma::mat XCGrid::invert(const arma::vec & gamma) const {
  arma::mat H=dfitp->invert_expansion(gamma);

  return H;
}

void XCGrid::construct(const arma::mat & P, double tol, int x_func, int c_func) {

  // Add all atoms
  if(verbose) {
    printf("Constructing DFT grid.\n");
    printf("\t%4s %7s %8s %s\n","atom","Npoints","Nfuncs","t");
    fflush(stdout);
  }

  // Set tolerances
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_tolerance(tol);
  // Check necessity of gradients
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad(x_func,c_func);

  Timer t;

  size_t Nat=fitbasp->get_Nnuc();

  // Do fitting
  arma::vec gamma=expand(P);

#ifdef _OPENMP
#pragma omp parallel
  {
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
    for(size_t i=0;i<Nat;i++) {
      grids[i]=wrk[ith].construct(*fitbasp,gamma,i,x_func,c_func,verbose,*dfitp);
    }
  }
#else
  for(size_t i=0;i<Nat;i++)
    grids[i]=wrk[0].construct(*fitbasp,gamma,i,x_func,c_func,verbose,*dfitp);
#endif
  
  if(verbose) {
    printf("DFT grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

void XCGrid::construct(const arma::mat & Pa, const arma::mat & Pb, double tol, int x_func, int c_func) {
  // Add all atoms
  if(verbose) {
    printf("\t%4s %7s %8s %s\n","atom","Npoints","Nfuncs","t");
    fflush(stdout);
  }

  // Set tolerances
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_tolerance(tol);
  // Check necessity of gradients
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad(x_func,c_func);

  Timer t;

  size_t Nat=fitbasp->get_Nnuc();

  // Do fitting
  arma::vec gammaa=expand(Pa);
  arma::vec gammab=expand(Pb);

#ifdef _OPENMP
#pragma omp parallel
  {
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
    for(size_t i=0;i<Nat;i++)
      grids[i]=wrk[ith].construct(*fitbasp,gammaa,gammab,i,x_func,c_func,verbose,*dfitp);
  }
#else
  for(size_t i=0;i<Nat;i++)
    grids[i]=wrk[0].construct(*fitbasp,gammaa,gammab,i,x_func,c_func,verbose,*dfitp);
#endif  

  if(verbose) {
    printf("DFT grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

size_t XCGrid::get_Npoints() const {
  size_t np=0;
  for(size_t i=0;i<grids.size();i++)
    np+=grids[i].ngrid;
  return np;
}

size_t XCGrid::get_Nfuncs() const {
  size_t nf=0;
  for(size_t i=0;i<grids.size();i++)
    nf+=grids[i].nfunc;
  return nf;
}

#ifdef CONSISTENCYCHECK
void XCGrid::eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Exc, double & Nel) {

  // Get expansion
  arma::vec gamma=expand(P);

  // Work vectors
  arma::vec Ha(gamma), Hb(gamma);
  Ha.zeros();
  Hb.zeros();

  eval_Fxc(x_func,c_func,gamma/2.0,gamma/2.0,Ha,Hb,Exc,Nel);
  arma::vec Hv=(Ha+Hb)/2.0;

  // Invert the expansion
  H=invert(Hv);
}
#else
void XCGrid::eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Exc, double & Nel) {
  // Clear exchange-correlation energy
  Exc=0.0;
  // Clear number of electrons
  Nel=0.0;

  arma::vec Hv(fitbasp->get_Nbf());
  Hv.zeros();

  // Do fitting
  arma::vec gamma=expand(P);

#ifdef _OPENMP
  // Get (maximum) number of threads
  int maxt=omp_get_max_threads();

  // Stack of work arrays
  std::vector<arma::vec> Hwrk;
  std::vector<double> Nelwrk;
  std::vector<double> Excwrk;

  for(int i=0;i<maxt;i++) {
    Hwrk.push_back(arma::vec(fitbasp->get_Nbf()));
    Hwrk[i].zeros();
    Nelwrk.push_back(0.0);
    Excwrk.push_back(0.0);
  }

#pragma omp parallel shared(Hwrk,Nelwrk,Excwrk)
  { // Begin parallel region
    
    // Current thread is
    int ith=omp_get_thread_num();

#pragma omp for schedule(dynamic,1)
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].form_grid(*fitbasp,grids[i]);
      // Compute basis functions
      wrk[ith].compute_bf(*fitbasp,grids[i]);
      
      // Update density
      wrk[ith].update_density(gamma);
      // Update number of electrons
      Nelwrk[ith]+=wrk[ith].compute_Nel();
      
      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk[ith].compute_xc(x_func);
      if(c_func>0)
	wrk[ith].compute_xc(c_func);

      // Evaluate the energy
      Excwrk[ith]+=wrk[ith].eval_Exc();
      // and construct the Fock matrices
      wrk[ith].eval_Fxc(Hwrk[ith]);
      // Free memory
      wrk[ith].free();
    }
  } // End parallel region

  // Sum results
  for(int i=0;i<maxt;i++) {
    Hv+=Hwrk[i];
    Nel+=Nelwrk[i];
    Exc+=Excwrk[i];
  }
#else
  for(size_t i=0;i<grids.size();i++) {
    // Change atom and create grid
    wrk[0].form_grid(*fitbasp,grids[i]);
    // Compute basis functions
    wrk[0].compute_bf(*fitbasp,grids[i]);

    // Update density
    wrk[0].update_density(gamma);
    // Update number of electrons
    Nel+=wrk[0].compute_Nel();
    
    // Initialize the arrays
    wrk[0].init_xc();
    // Compute the functionals
    if(x_func>0)
      wrk[0].compute_xc(x_func);
    if(c_func>0)
      wrk[0].compute_xc(c_func);

    // Evaluate the energy
    Exc+=wrk[0].eval_Exc();
    // and construct the Fock matrices
    wrk[0].eval_Fxc(Hv);
    // Free memory
    wrk[0].free();
  }
#endif

  // Compute the expansion
  H=invert(Hv);
}
#endif

void XCGrid::eval_Fxc(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ha, arma::mat & Hb, double & Exc, double & Nel) {
  // Clear exchange-correlation energy
  Exc=0.0;
  // Clear number of electrons
  Nel=0.0;
  // Fock vectors
  arma::vec Hav(fitbasp->get_Nbf());
  arma::vec Hbv(fitbasp->get_Nbf());
  Hav.zeros();
  Hbv.zeros();

  // Do the fitting
  arma::vec gammaa=expand(Pa);
  arma::vec gammab=expand(Pb);


#ifdef _OPENMP
  // Get (maximum) number of threads
  int maxt=omp_get_max_threads();

  // Stack of work arrays
  std::vector<arma::vec> Hawrk, Hbwrk;
  std::vector<double> Nelwrk;
  std::vector<double> Excwrk;

  for(int i=0;i<maxt;i++) {
    Hawrk.push_back(arma::vec(fitbasp->get_Nbf()));
    Hawrk[i].zeros();

    Hbwrk.push_back(arma::vec(fitbasp->get_Nbf()));
    Hbwrk[i].zeros();

    Nelwrk.push_back(0.0);
    Excwrk.push_back(0.0);
  }

#pragma omp parallel shared(Hawrk,Hbwrk,Nelwrk,Excwrk)
  { // Begin parallel region
    
    // Current thread is
    int ith=omp_get_thread_num();
    
#pragma omp for schedule(dynamic,1)
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {

      // Change atom and create grid
      wrk[ith].form_grid(*fitbasp,grids[i]);
      // Compute basis functions
      wrk[ith].compute_bf(*fitbasp,grids[i]);
      
      // Update density
      wrk[ith].update_density(gammaa,gammab);
      // Update number of electrons
      Nel+=wrk[ith].compute_Nel();

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
        wrk[ith].compute_xc(x_func);
      if(c_func>0)
        wrk[ith].compute_xc(c_func);

      // Evaluate the energy
      Exc+=wrk[ith].eval_Exc();
      // and construct the Fock matrices
      wrk[ith].eval_Fxc(Hawrk[ith],Hbwrk[ith]);
           
      // Free memory
      wrk[ith].free();
    }
  } // End parallel region
  
  // Sum results
  for(int i=0;i<maxt;i++) {
    Hav+=Hawrk[i];
    Hbv+=Hbwrk[i];
    Nel+=Nelwrk[i];
    Exc+=Excwrk[i];
  }
#else
  // Loop over atoms
  for(size_t i=0;i<grids.size();i++) {
    // Change atom and create grid
    wrk[0].form_grid(*fitbasp,grids[i]);
    // Compute basis functions
    wrk[0].compute_bf(*fitbasp,grids[i]);
    
    // Update density
    wrk[0].update_density(gammaa,gammab);
    // Update number of electrons
    Nel+=wrk[0].compute_Nel();

    // Initialize the arrays
    wrk[0].init_xc();
    // Compute the functionals
    if(x_func>0)
      wrk[0].compute_xc(x_func);
    if(c_func>0)
      wrk[0].compute_xc(c_func);

    // Evaluate the energy
    Exc+=wrk[0].eval_Exc();
    // and construct the Fock matrices
    wrk[0].eval_Fxc(Hav,Hbv);
 
    // Free memory
    wrk[0].free();
  }
#endif


  // Compute the expansion
  Ha=invert(Hav);
  Hb=invert(Hbv);
}
