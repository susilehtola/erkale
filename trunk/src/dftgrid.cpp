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
#include "dftgrid.h"
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

// Check libxc output for sanity
//#define LIBXCCHECK

bool operator<(const dens_list_t &lhs, const dens_list_t & rhs) {
  // Sort in decreasing order
  return lhs.d > rhs.d;
}

void AtomGrid::add_lobatto_shell(atomgrid_t & g, size_t ir) {
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

void AtomGrid::add_lebedev_shell(atomgrid_t & g, size_t ir) {
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

void AtomGrid::becke_weights(const BasisSet & bas, const atomgrid_t & g, size_t ir, double a) {
  // Compute weights of points.

  // Number of atoms in system
  const size_t Nat=bas.get_Nnuc();

  // Helper arrays
  arma::vec atom_dist;
  arma::vec atom_weight;
  arma::mat mu_ab;
  arma::mat smu_ab;
  // Indices to consider
  std::vector<size_t> cmpidx;

  // Initialize memory
  atom_dist.zeros(Nat);
  atom_weight.zeros(Nat);
  mu_ab.zeros(Nat,Nat);
  smu_ab.zeros(Nat,Nat);

  // Get nuclei
  std::vector<nucleus_t> nuccoords=bas.get_nuclei();

  // Get nuclear distances
  arma::mat nucdist=bas.nuclear_distances();
  // Compute closest distance to other atoms
  double Rin=DBL_MAX;
  for(size_t i=0;i<g.atind;i++)
    if(nucdist(g.atind,i)<Rin)
      Rin=nucdist(g.atind,i);
  for(size_t i=g.atind+1;i<Nat;i++)
    if(nucdist(g.atind,i)<Rin)
      Rin=nucdist(g.atind,i);

  double scrthr=std::pow(0.5*(1-a)*Rin,2);

  // Loop over points on wanted radial shell
  for(size_t ip=g.sh[ir].ind0;ip<g.sh[ir].ind0+g.sh[ir].np;ip++) {
    // Coordinates of the point are
    coords_t coord_p=grid[ip].r;

    // Prescreen - is the weight unity?
    if(normsq(nuccoords[g.atind].r-coord_p) < scrthr)
      // Yes - nothing to do.
      continue;

    // Compute distance of point to atoms
    for(size_t iat=0;iat<Nat;iat++)
      atom_dist(iat)=norm(nuccoords[iat].r-coord_p);

    // Compute mu_ab
    for(size_t iat=0;iat<Nat;iat++) {
      // Diagonal
      mu_ab(iat,iat)=0.0;
      // Off-diagonal
      for(size_t jat=0;jat<iat;jat++) {
	mu_ab(iat,jat)=(atom_dist(iat)-atom_dist(jat))/nucdist(iat,jat);
	mu_ab(jat,iat)=-mu_ab(iat,jat);
      }
    }

    // Compute s(mu_ab)
    for(size_t iat=0;iat<Nat;iat++)
      for(size_t jat=0;jat<Nat;jat++) {
	smu_ab(iat,jat)=f_s(mu_ab(iat,jat),a);
      }

    // Then, compute atomic weights
    for(size_t iat=0;iat<Nat;iat++) {
      atom_weight(iat)=1.0;

      for(size_t jat=0;jat<iat;jat++)
	atom_weight(iat)*=smu_ab(iat,jat);
      for(size_t jat=iat+1;jat<Nat;jat++)
	atom_weight(iat)*=smu_ab(iat,jat);
    }

    // The Becke weight is
    grid[ip].w*=atom_weight(g.atind)/arma::sum(atom_weight);
  }
}

void AtomGrid::hirshfeld_weights(const Hirshfeld & hirsh, const atomgrid_t & g, size_t ir) {
  // Compute weights of points.

  // Loop over points on wanted radial shell
  for(size_t ip=g.sh[ir].ind0;ip<g.sh[ir].ind0+g.sh[ir].np;ip++) {
    // The Hirshfeld weight is
    grid[ip].w*=hirsh.get_weight(g.atind,grid[ip].r);
  }
}

void AtomGrid::prune_points(double tolv, const radshell_t & rg) {
  // Prune points with small weight.

  // First point on radial shell
  size_t ifirst=rg.ind0;
  // Last point on radial shell
  size_t ilast=ifirst+rg.np;

  for(size_t i=ilast;(i>=ifirst && i<grid.size());i--)
    if(grid[i].w<tolv)
      grid.erase(grid.begin()+i);
}


void AtomGrid::free() {
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

  // Free mGGA stuff
  lapl_rho.clear();
  vlapl.clear();
  tau.clear();
  vtau.clear();
}

void AtomGrid::update_density(const arma::mat & P) {
  // Update values of densitty

  if(!P.n_elem) {
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
    rho[ip]=eval_dens(P,ip);
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
      eval_grad(P,ip,grad);
      // Store it
      for(int ic=0;ic<3;ic++)
	grho[3*ip+ic]=grad[ic];

      // Compute sigma as well
      sigma[ip]=grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2];
    }
  }

  if(do_lapl) {
    // Adjust size of grid
    if(lapl_rho.size()!=rho.size())
      lapl_rho.resize(rho.size());
    if(tau.size()!=rho.size())
      tau.resize(rho.size());

    double lapl, kin;

    for(size_t ip=0;ip<grid.size();ip++) {
      // Evaluate laplacian and kinetic energy densities
      eval_lapl_kin_dens(P,ip,lapl,kin);
      // and add them to the stack
      lapl_rho[ip]=lapl;
      tau[ip]=kin;
    }
  }
}

void AtomGrid::update_density(const arma::mat & Pa, const arma::mat & Pb) {
  // Update values of densitty

  if(!Pa.n_elem || !Pb.n_elem) {
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
    rho[2*ip]=eval_dens(Pa,ip);
    rho[2*ip+1]=eval_dens(Pb,ip);
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
      eval_grad(Pa,ip,grada);
      eval_grad(Pb,ip,gradb);
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

  if(do_lapl) {
    // Adjust size of grid
    if(lapl_rho.size()!=rho.size())
      lapl_rho.resize(rho.size());
    if(tau.size()!=rho.size())
      tau.resize(rho.size());

    double lapla, kina;
    double laplb, kinb;

    for(size_t ip=0;ip<grid.size();ip++) {
      // Evaluate laplacian and kinetic energy densities
      eval_lapl_kin_dens(Pa,ip,lapla,kina);
      eval_lapl_kin_dens(Pb,ip,laplb,kinb);
      // and add them to the stack
      lapl_rho[2*ip]=lapla;
      lapl_rho[2*ip+1]=laplb;

      tau[2*ip]=kina;
      tau[2*ip+1]=kinb;
    }
  }
}

double AtomGrid::eval_dens(const arma::mat & P, size_t ip) const {
  // Density
  double d=0.0;

  // Loop over functions on point
  size_t first=grid[ip].f0;
  size_t last=first+grid[ip].nf;

  size_t i, j;

  for(size_t ii=first;ii<last;ii++) {
    // Index of function is
    i=flist[ii].ind;
    for(size_t jj=first;jj<last;jj++) {
      j=flist[jj].ind;
      d+=P(i,j)*flist[ii].f*flist[jj].f;
    }
  }

  return d;
}

void AtomGrid::eval_grad(const arma::mat & P, size_t ip, double *g) const {
  // Initialize gradient
  for(int i=0;i<3;i++)
    g[i]=0.0;

  // Loop over functions on point
  size_t first=grid[ip].f0;
  size_t last=first+grid[ip].nf;

  size_t i, j;

  // grad rho = P(i,j) [ (grad i) j + i (grad j) ]
  //          = 2 P(i,j) i (grad j)
  for(size_t ii=first;ii<last;ii++) {
    // Index of function is
    i=flist[ii].ind;

    for(size_t jj=first;jj<last;jj++) {
      j=flist[jj].ind;

      g[0]+=P(i,j)*flist[ii].f*glist[3*jj  ];
      g[1]+=P(i,j)*flist[ii].f*glist[3*jj+1];
      g[2]+=P(i,j)*flist[ii].f*glist[3*jj+2];
    }
  }

  for(int ic=0;ic<3;ic++)
    g[ic]*=2.0;
}

void AtomGrid::eval_lapl_kin_dens(const arma::mat & P, size_t ip, double & lapl, double & kin) const {
  // Loop over functions on point
  size_t first=grid[ip].f0;
  size_t last=first+grid[ip].nf;

  size_t i, j;

  double bf_gdot;
  double bf_lapl;

   // Initialize output
  lapl=0.0;
  kin=0.0;

  for(size_t ii=first;ii<last;ii++) {
    // Index of function is
    i=flist[ii].ind;
    for(size_t jj=first;jj<last;jj++) {
      j=flist[jj].ind;

      // Dot product of gradients of basis functions
      bf_gdot=glist[3*ii]*glist[3*jj] + glist[3*ii+1]*glist[3*jj+1] + glist[3*ii+2]*glist[3*jj+2];
      // Laplacian
      bf_lapl=flist[ii].f*llist[jj]  + llist[ii]*flist[jj].f + 2.0*bf_gdot;

      // Increment output.
      lapl+=P(i,j)*bf_lapl;

      // libxc prior to version 2.0.0: without factor 0.5
      //kin+=P(i,j)*bf_gdot;
      // Since version 2.0.0:
      kin+=0.5*P(i,j)*bf_gdot;
    }
  }
}

void AtomGrid::eval_dens(const arma::mat & P, std::vector<dens_list_t> & list) const {
  for(size_t ip=0;ip<grid.size();ip++) {
    // Get density at grid point
    dens_list_t hlp;
    hlp.d=eval_dens(P,ip);
    hlp.w=grid[ip].w;
    list.push_back(hlp);
  }
}

double AtomGrid::compute_Nel() const {
  double nel=0.0;

  if(!polarized)
    for(size_t ip=0;ip<grid.size();ip++)
      nel+=grid[ip].w*rho[ip];
  else
    for(size_t ip=0;ip<grid.size();ip++)
      nel+=grid[ip].w*(rho[2*ip]+rho[2*ip+1]);

  return nel;
}

void AtomGrid::print_grid() const {
  for(size_t ip=0;ip<grid.size();ip++)
    printf("%5i % f % f % f %e\n",(int) ip+1,grid[ip].r.x,grid[ip].r.y,grid[ip].r.z,grid[ip].w);
}

void AtomGrid::init_xc() {
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

    if(do_lapl) {
      if(vtau.size()!=N)
	vtau.resize(N);
      if(vlapl.size()!=N)
	vlapl.resize(N);
    }
  } else {
    // Unrestricted case - arrays of length 2N or 3N
    if(vxc.size()!=2*N)
      vxc.resize(2*N);

    if(do_grad) {
      if(vsigma.size()!=3*N)
	vsigma.resize(3*N);
    }

    if(do_lapl) {
      if(vlapl.size()!=2*N)
	vlapl.resize(2*N);
      if(vtau.size()!=2*N)
	vtau.resize(2*N);
    }
  }

  // Initial values
  do_gga=false;
  do_mgga=false;

  // Fill arrays with zeros.
  for(size_t i=0;i<exc.size();i++)
    exc[i]=0.0;
  for(size_t i=0;i<vxc.size();i++)
    vxc[i]=0.0;
  for(size_t i=0;i<vsigma.size();i++)
    vsigma[i]=0.0;
  for(size_t i=0;i<vlapl.size();i++)
    vlapl[i]=0.0;
  for(size_t i=0;i<vtau.size();i++)
    vtau[i]=0.0;
}

void check_array(const std::vector<double> & x, size_t n, std::vector<size_t> & idx) {
  if(x.size()%n!=0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Size of array " << x.size() << " is not divisible by " << n << "!\n";
    throw std::runtime_error(oss.str());
  }

  for(size_t i=0;i<x.size()/n;i++) {
    // Check for failed entry
    bool fail=false;
    for(size_t j=0;j<n;j++)
      if(!std::isfinite(x[i*n+j]))
	fail=true;

    // If failed i is not in the list, add it
    if(fail) {
      if (!std::binary_search (idx.begin(), idx.end(), i)) {
	idx.push_back(i);
	std::sort(idx.begin(),idx.end());
      }
    }
  }
}

void AtomGrid::compute_xc(int func_id) {
  // Compute exchange-correlation functional

  // Which functional is in question?
  bool gga, mgga;
  is_gga_mgga(func_id,gga,mgga);

  // Update controlling flags for eval_Fxc (exchange and correlation
  // parts might be of different type)
  do_gga=do_gga || gga || mgga;
  do_mgga=do_mgga || mgga;

  // Amount of grid points
  const size_t N=grid.size();

  // Work arrays - exchange and correlation are computed separately
  std::vector<double> exc_wrk(exc);
  std::vector<double> vxc_wrk(vxc);
  std::vector<double> vsigma_wrk(vsigma);
  std::vector<double> vlapl_wrk(vlapl);
  std::vector<double> vtau_wrk(vtau);

  // Spin variable for libxc
  int nspin;
  if(!polarized)
    nspin=XC_UNPOLARIZED;
  else
    nspin=XC_POLARIZED;

  // Initialize libxc worker
  xc_func_type func;
  if(xc_func_init(&func, func_id, nspin) != 0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Functional "<<func_id<<" not found!";
    throw std::runtime_error(oss.str());
  }

  // Evaluate functionals.
  if(mgga) // meta-GGA
    xc_mgga_exc_vxc(&func, N, &rho[0], &sigma[0], &lapl_rho[0], &tau[0], &exc_wrk[0], &vxc_wrk[0], &vsigma_wrk[0], &vlapl_wrk[0], &vtau_wrk[0]);
  else if(gga) // GGA
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
  for(size_t i=0;i<vlapl.size();i++)
    vlapl[i]+=vlapl_wrk[i];
  for(size_t i=0;i<vtau.size();i++)
    vtau[i]+=vtau_wrk[i];

  // Free functional
  xc_func_end(&func);
}

void AtomGrid::print_density(FILE *f) const {
  // Loop over grid points
  for(size_t i=0;i<grid.size();i++) {
    // Get data in point
    libxc_dens_t d=get_dens(i);
    
    // Print out data
    fprintf(f,"% .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e\n",d.rhoa,d.rhob,d.sigmaaa,d.sigmaab,d.sigmabb,d.lapla,d.laplb,d.taua,d.taub);
  }
}

void AtomGrid::print_potential(int func_id, FILE *f) const {
  // Loop over grid points
  for(size_t i=0;i<grid.size();i++) {
    // Get data in point
    libxc_pot_t d=get_pot(i);

    int nspin;
    if(polarized)
      nspin=XC_POLARIZED;
    else
      nspin=XC_POLARIZED;
    
    // Print out data
    fprintf(f, "%3i %2i % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e\n",func_id,nspin,d.vrhoa,d.vrhob,d.vsigmaaa,d.vsigmaab,d.vsigmabb,d.vlapla,d.vlaplb,d.vtaua,d.vtaub);
  }
}

libxc_dens_t AtomGrid::get_dens(size_t idx) const {
  libxc_dens_t ret;

  // Alpha and beta density
  ret.rhoa=0.0;
  ret.rhob=0.0;
  
  // Sigma variables
  ret.sigmaaa=0.0;
  ret.sigmaab=0.0;
  ret.sigmabb=0.0;
      
  // Laplacians
  ret.lapla=0.0;
  ret.laplb=0.0;
      
  // Kinetic energy density
  ret.taua=0.0;
  ret.taub=0.0;

  if(do_mgga) {
    if(polarized) {
      ret.lapla=lapl_rho[2*idx];
      ret.laplb=lapl_rho[2*idx+1];
      ret.taua=tau[2*idx];
      ret.taub=tau[2*idx+1];
      
      ret.sigmaaa=sigma[3*idx];
      ret.sigmaab=sigma[3*idx+1];
      ret.sigmabb=sigma[3*idx+2];
	  
      ret.rhoa=rho[2*idx];
      ret.rhob=rho[2*idx+1];
    } else {
      ret.lapla=ret.laplb=lapl_rho[idx]/2.0;
      ret.taua=ret.taub=tau[idx]/2.0;
      ret.sigmaaa=ret.sigmaab=ret.sigmabb=sigma[idx]/4.0;
      ret.rhoa=ret.rhob=rho[idx]/2.0;
    }
  } else if(do_gga) {
    if(polarized) {
      ret.sigmaaa=sigma[3*idx];
      ret.sigmaab=sigma[3*idx+1];
      ret.sigmabb=sigma[3*idx+2];
      
      ret.rhoa=rho[2*idx];
      ret.rhob=rho[2*idx+1];
    } else {
      ret.sigmaaa=ret.sigmaab=ret.sigmabb=sigma[idx]/4.0;
      ret.rhoa=ret.rhob=rho[idx]/2.0;
    }
  } else {
    if(polarized) {
      ret.rhoa=rho[2*idx];
      ret.rhob=rho[2*idx+1];
    } else {
      ret.rhoa=ret.rhob=rho[idx]/2.0;
    }
  }

  return ret;
}

libxc_debug_t AtomGrid::get_data(size_t idx) const {
  libxc_debug_t d;
  d.dens=get_dens(idx);
  d.pot=get_pot(idx);
  return d;
}

libxc_pot_t AtomGrid::get_pot(size_t idx) const {
  libxc_pot_t ret;

  // Alpha and beta potential
  ret.vrhoa=0.0;
  ret.vrhob=0.0;
  
  // Sigma variables
  ret.vsigmaaa=0.0;
  ret.vsigmaab=0.0;
  ret.vsigmabb=0.0;
  
  // Laplacians
  ret.vlapla=0.0;
  ret.vlaplb=0.0;
      
  // Kinetic energy density
  ret.vtaua=0.0;
  ret.vtaub=0.0;
  
  if(do_mgga) {
    if(polarized) {
      ret.vlapla=vlapl[2*idx];
      ret.vlaplb=vlapl[2*idx+1];
      ret.vtaua=vtau[2*idx];
      ret.vtaub=vtau[2*idx+1];

      ret.vsigmaaa=vsigma[3*idx];
      ret.vsigmaab=vsigma[3*idx+1];
      ret.vsigmabb=vsigma[3*idx+2];
	  
      ret.vrhoa=vxc[2*idx];
      ret.vrhob=vxc[2*idx+1];
    } else {
      ret.vlapla=ret.vlaplb=vlapl[idx];
      ret.vtaua=ret.vtaub=vtau[idx];
      ret.vsigmaaa=ret.vsigmaab=ret.vsigmabb=vsigma[idx];
      ret.vrhoa=ret.vrhob=vxc[idx];
    }
  } else if(do_gga) {
    if(polarized) {
      ret.vsigmaaa=vsigma[3*idx];
      ret.vsigmaab=vsigma[3*idx+1];
      ret.vsigmabb=vsigma[3*idx+2];
	  
      ret.vrhoa=vxc[2*idx];
      ret.vrhob=vxc[2*idx+1];
    } else {
      ret.vsigmaaa=ret.vsigmaab=ret.vsigmabb=vsigma[idx];
      ret.vrhoa=ret.vrhob=vxc[idx];
    }
  } else {
    if(polarized) {
      ret.vrhoa=vxc[2*idx];
      ret.vrhob=vxc[2*idx+1];
    } else {
      ret.vrhoa=ret.vrhob=vxc[idx];
    }
  }

  return ret;
}

double AtomGrid::eval_Exc() const {
  double Exc=0.0;

  if(!polarized)
    for(size_t i=0;i<grid.size();i++)
      Exc+=exc[i]*rho[i]*grid[i].w;
  else
    for(size_t i=0;i<grid.size();i++)
      Exc+=exc[i]*(rho[2*i]+rho[2*i+1])*grid[i].w;

  return Exc;
}

void AtomGrid::eval_overlap(arma::mat & S) const {
  for(size_t ip=0;ip<grid.size();ip++) {

    // Loop over functions on grid point
    size_t first=grid[ip].f0;
    size_t last=first+grid[ip].nf;

    for(size_t ii=first;ii<last;ii++) {
      // Get index of function
      size_t i=flist[ii].ind;
      for(size_t jj=first;jj<last;jj++) {
	size_t j=flist[jj].ind;

	S(i,j)+=grid[ip].w*flist[ii].f*flist[jj].f;
      }
    }
  }
}

void AtomGrid::eval_diag_overlap(arma::vec & S) const {
  for(size_t ip=0;ip<grid.size();ip++) {
    // Loop over functions on grid point
    size_t first=grid[ip].f0;
    size_t last=first+grid[ip].nf;

    for(size_t ii=first;ii<last;ii++) {
      // Get index of function
      size_t i=flist[ii].ind;

      S(i)+=grid[ip].w*flist[ii].f*flist[ii].f;
    }
  }
}

void AtomGrid::eval_Fxc(arma::mat & H) const {
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
      for(size_t jj=first;jj<last;jj++) {
	size_t j=flist[jj].ind;

	H(i,j)+=xcfac*flist[ii].f*flist[jj].f;
      }
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

	for(size_t jj=first;jj<last;jj++) {
	  size_t j=flist[jj].ind;

	  H(i,j)+=xcvec[0]*(glist[3*ii]*flist[jj].f + flist[ii].f*glist[3*jj])
	    +     xcvec[1]*(glist[3*ii+1]*flist[jj].f + flist[ii].f*glist[3*jj+1])
	    +     xcvec[2]*(glist[3*ii+2]*flist[jj].f + flist[ii].f*glist[3*jj+2]);
	}
      }
    }
  }

  // Meta-GGA
  if(do_mgga) {
    double bf_lapl;
    double bf_gdot;

    double lfac, kfac;

    for(size_t ip=0;ip<grid.size();ip++) {
      // Factors in common for basis functions
      kfac=grid[ip].w*vtau[ip];
      lfac=grid[ip].w*vlapl[ip];

      // Loop over functions on grid point
      size_t first=grid[ip].f0;
      size_t last=first+grid[ip].nf;

      for(size_t ii=first;ii<last;ii++) {
	// Get index of function
	size_t i=flist[ii].ind;
	for(size_t jj=first;jj<last;jj++) {
	  size_t j=flist[jj].ind;

	  // Laplacian and kinetic energy density
	  bf_gdot=glist[3*ii]*glist[3*jj] + glist[3*ii+1]*glist[3*jj+1] + glist[3*ii+2]*glist[3*jj+2];
	  bf_lapl=flist[ii].f*llist[jj] + llist[ii]*flist[jj].f + 2.0*bf_gdot;

	  // Contribution is
	  H(i,j)+=0.5*kfac*bf_gdot + lfac*bf_lapl;
	}
      }
    }
  }
}

void AtomGrid::eval_diag_Fxc(arma::vec & H) const {
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

      H[i]+=xcfac*flist[ii].f*flist[ii].f;
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

	for(int ic=0;ic<3;ic++) {
	  H[i]+=xcvec[ic]*2.0*glist[3*ii+ic]*flist[ii].f;
	}
      }
    }
  }

  // Meta-GGA
  if(do_mgga) {
    double bf_gdot;
    double bf_lapl;

    double kfac;
    double lfac;

    for(size_t ip=0;ip<grid.size();ip++) {
      // Factors in common for basis functions
      lfac=grid[ip].w*vlapl[ip];
      kfac=grid[ip].w*vtau[ip];

      // Loop over functions on grid point
      size_t first=grid[ip].f0;
      size_t last=first+grid[ip].nf;

      for(size_t ii=first;ii<last;ii++) {
	// Get index of function
	size_t i=flist[ii].ind;

	// Laplacian and kinetic energy density
	bf_gdot=glist[3*ii]*glist[3*ii] + glist[3*ii+1]*glist[3*ii+1] + glist[3*ii+2]*glist[3*ii+2];
	bf_lapl=2.0*flist[ii].f*llist[ii] + 2.0*bf_gdot;

	H[i]+=0.5*kfac*bf_gdot + lfac*bf_lapl;
      }
    }
  }
}

void AtomGrid::eval_Fxc(arma::mat & Ha, arma::mat & Hb) const {
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
      for(size_t jj=first;jj<last;jj++) {
	size_t j=flist[jj].ind;

	Ha(i,j)+=xcfaca*flist[ii].f*flist[jj].f;
	Hb(i,j)+=xcfacb*flist[ii].f*flist[jj].f;
      }
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
	for(size_t jj=first;jj<last;jj++) {
	  size_t j=flist[jj].ind;

	  for(int ic=0;ic<3;ic++) {
	    Ha(i,j)+=xcveca[ic]*(glist[3*ii+ic]*flist[jj].f + flist[ii].f*glist[3*jj+ic]);
	    Hb(i,j)+=xcvecb[ic]*(glist[3*ii+ic]*flist[jj].f + flist[ii].f*glist[3*jj+ic]);
	  }
	}
      }
    }
  }

  // Meta-GGA
  if(do_mgga) {
    double bf_gdot;
    double bf_lapl;

    double kfaca, kfacb;
    double lfaca, lfacb;

    for(size_t ip=0;ip<grid.size();ip++) {
      // Factors in common for basis functions
      lfaca=grid[ip].w*vlapl[2*ip];
      lfacb=grid[ip].w*vlapl[2*ip+1];

      kfaca=grid[ip].w*vtau[2*ip];
      kfacb=grid[ip].w*vtau[2*ip+1];

      // Loop over functions on grid point
      size_t first=grid[ip].f0;
      size_t last=first+grid[ip].nf;

      for(size_t ii=first;ii<last;ii++) {
	// Get index of function
	size_t i=flist[ii].ind;
	for(size_t jj=first;jj<last;jj++) {
	  size_t j=flist[jj].ind;

	  // Laplacian and kinetic energy density
	  bf_gdot=glist[3*ii]*glist[3*jj] + glist[3*ii+1]*glist[3*jj+1] + glist[3*ii+2]*glist[3*jj+2];
	  bf_lapl=flist[ii].f*llist[jj] + llist[ii]*flist[jj].f + 2.0*bf_gdot;

	  Ha(i,j)+=0.5*kfaca*bf_gdot + lfaca*bf_lapl;
	  Hb(i,j)+=0.5*kfacb*bf_gdot + lfacb*bf_lapl;
	}
      }
    }
  }
}

void AtomGrid::eval_diag_Fxc(arma::vec & Ha, arma::vec & Hb) const {
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

      Ha[i]+=xcfaca*flist[ii].f*flist[ii].f;
      Hb[i]+=xcfacb*flist[ii].f*flist[ii].f;
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
	  Ha[i]+=xcveca[ic]*2.0*glist[3*ii+ic]*flist[ii].f;
	  Hb[i]+=xcvecb[ic]*2.0*glist[3*ii+ic]*flist[ii].f;
	}
      }
    }
  }

  // Meta-GGA
  if(do_mgga) {
    double bf_lapl;
    double bf_gdot;

    double kfaca, kfacb;
    double lfaca, lfacb;

    for(size_t ip=0;ip<grid.size();ip++) {
      // Factors in common for basis functions
      lfaca=grid[ip].w*vlapl[2*ip];
      lfacb=grid[ip].w*vlapl[2*ip+1];

      kfaca=grid[ip].w*vtau[2*ip];
      kfacb=grid[ip].w*vtau[2*ip+1];

      // Loop over functions on grid point
      size_t first=grid[ip].f0;
      size_t last=first+grid[ip].nf;

      for(size_t ii=first;ii<last;ii++) {
	// Get index of function
	size_t i=flist[ii].ind;

	// Laplacian and kinetic energy density
	bf_gdot=glist[3*ii]*glist[3*ii] + glist[3*ii+1]*glist[3*ii+1] + glist[3*ii+2]*glist[3*ii+2];
	bf_lapl=2.0*flist[ii].f*llist[ii] + 2.0*bf_gdot;

	Ha[i]+=0.5*kfaca*bf_gdot + lfaca*bf_lapl;
	Hb[i]+=0.5*kfacb*bf_gdot + lfacb*bf_lapl;
      }
    }
  }
}

arma::vec AtomGrid::eval_force(const BasisSet & bas, const arma::mat & Pa, const arma::mat & Pb) const {
  if(!polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted force with restricted density.\n");
  }

  arma::rowvec f(3*bas.get_Nnuc());
  f.zeros();

  // Loop over nuclei
  for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {

    // Get functions centered on the atom
    std::vector<GaussianShell> shells=bas.get_funcs(inuc);

    // LDA part. Loop over grid
    for(size_t ip=0;ip<grid.size();ip++) {

      // Grad rho in current point
      arma::rowvec gradrhoa(3);
      gradrhoa.zeros();
      arma::rowvec gradrhob(3);
      gradrhob.zeros();

      // Loop over shells on the nucleus
      for(size_t ish=0;ish<shells.size();ish++) {

	// First function on shell is
	size_t mu0=shells[ish].get_first_ind();

	// Evaluate the gradient in the current grid point
	arma::mat grad=shells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	// Increment sum. Loop over mu
	for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	  // Function index is
	  size_t mu=mu0+imu;

	  // Loop over the functions on the grid point
	  size_t first=grid[ip].f0;
	  size_t last=first+grid[ip].nf;

	  for(size_t inu=first;inu<last;inu++) {
	    // Get index of function
	    size_t nu=flist[inu].ind;

	    gradrhoa+=Pa(mu,nu)*grad.row(imu)*flist[inu].f;
	    gradrhob+=Pb(mu,nu)*grad.row(imu)*flist[inu].f;
	  }
	}
      }
      // Plug in the factor 2 to get the total gradient
      gradrhoa*=2.0;
      gradrhob*=2.0;

      // Increment total force
      f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*(vxc[2*ip]*gradrhoa + vxc[2*ip+1]*gradrhob);
    }
  }

  // GGA part
  if(do_gga) {

    // Loop over nuclei
    for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
      // Get functions centered on the atom
      std::vector<GaussianShell> shells=bas.get_funcs(inuc);

      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// X in the current grid point
	arma::mat Xa(3,3);
	Xa.zeros();

	arma::mat Xb(3,3);
	Xb.zeros();

	// Loop over shells on the nucleus
	for(size_t ish=0;ish<shells.size();ish++) {

	  // First function on shell is
	  size_t mu0=shells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=shells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=shells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    // Loop over the functions on the grid point
	    size_t first=grid[ip].f0;
	    size_t last=first+grid[ip].nf;

	    for(size_t inu=first;inu<last;inu++) {
	      // Get index of function
	      size_t nu=flist[inu].ind;

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  Xa(ic,jc)+=Pa(mu,nu)*(flist[inu].f*hess(imu,3*ic+jc) + glist[3*inu+jc]*grad(imu,ic));
	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  Xb(ic,jc)+=Pb(mu,nu)*(flist[inu].f*hess(imu,3*ic+jc) + glist[3*inu+jc]*grad(imu,ic));
	    }
	  }
	}
	Xa*=2.0;
	Xb*=2.0;

	// The xc "vector" is
	arma::vec xca(3);
	for(int ic=0;ic<3;ic++)
	  xca(ic)=2.0*vsigma[3*ip  ]*grho[6*ip  +ic] + vsigma[3*ip+1]*grho[6*ip+3+ic];
	arma::vec xcb(3);
	for(int ic=0;ic<3;ic++)
	  xcb(ic)=2.0*vsigma[3*ip+2]*grho[6*ip+3+ic] + vsigma[3*ip+1]*grho[6*ip  +ic];

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*arma::trans(Xa*xca+Xb*xcb);
      }
    }
  }

  // meta-GGA part
  if(do_mgga) {

    // First part. Loop over nuclei
    for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
      // Get functions centered on the atom
      std::vector<GaussianShell> shells=bas.get_funcs(inuc);

      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// y in the current grid point
	arma::vec ya(3);
	ya.zeros();
	arma::vec yb(3);
	yb.zeros();

	// Loop over shells on the nucleus
	for(size_t ish=0;ish<shells.size();ish++) {

	  // First function on shell is
	  size_t mu0=shells[ish].get_first_ind();

	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=shells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    // Loop over the functions on the grid point
	    size_t first=grid[ip].f0;
	    size_t last=first+grid[ip].nf;

	    for(size_t inu=first;inu<last;inu++) {
	      // Get index of function
	      size_t nu=flist[inu].ind;

	      // Collect grad nu
	      arma::vec gnu(3);
	      for(int ic=0;ic<3;ic++)
		gnu(ic)=glist[3*inu+ic];

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++) {
		  ya(ic)+=Pa(mu,nu)*hess(imu,3*ic+jc)*gnu(jc);
		  yb(ic)+=Pb(mu,nu)*hess(imu,3*ic+jc)*gnu(jc);
		}
	    }
	  }
	}

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*arma::trans((vtau[2*ip]+2.0*vlapl[2*ip])*ya + (vtau[2*ip+1]+2.0*vlapl[2*ip+1])*yb);
      }
    }

    // Second part. Loop over nuclei
    for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
      // Get functions centered on the atom
      std::vector<GaussianShell> shells=bas.get_funcs(inuc);

      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// z in the current grid point
	arma::rowvec za(3);
	za.zeros();
	arma::rowvec zb(3);
	zb.zeros();

	// Loop over shells on the nucleus
	for(size_t ish=0;ish<shells.size();ish++) {

	  // First function on shell is
	  size_t mu0=shells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=shells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the laplacian of the gradient in the current grid point
	  arma::mat laplgrad=shells[ish].eval_laplgrad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    // Loop over the functions on the grid point
	    size_t first=grid[ip].f0;
	    size_t last=first+grid[ip].nf;

	    for(size_t inu=first;inu<last;inu++) {
	      // Get index of function
	      size_t nu=flist[inu].ind;

	      za+=Pa(mu,nu)*(llist[inu]*grad.row(imu)+flist[inu].f*laplgrad.row(imu));
	      zb+=Pb(mu,nu)*(llist[inu]*grad.row(imu)+flist[inu].f*laplgrad.row(imu));
	    }
	  }
	}
	za*=2.0;
	zb*=2.0;

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*(vlapl[2*ip]*za + vlapl[2*ip+1]*zb);
      }
    }
  }


  return arma::trans(f);
}


arma::vec AtomGrid::eval_force(const BasisSet & bas, const arma::mat & P) const {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted force with unrestricted density.\n");
  }

  // Initialize force
  arma::rowvec f(3*bas.get_Nnuc());
  f.zeros();

  // Loop over nuclei
  for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
    // Get functions centered on the atom
    std::vector<GaussianShell> shells=bas.get_funcs(inuc);

    // LDA part. Loop over grid
    for(size_t ip=0;ip<grid.size();ip++) {

      // Grad rho in current point
      arma::rowvec gradrho(3);
      gradrho.zeros();

      // Loop over shells on the nucleus
      for(size_t ish=0;ish<shells.size();ish++) {

	// First function on shell is
	size_t mu0=shells[ish].get_first_ind();

	// Evaluate the gradient in the current grid point
	arma::mat grad=shells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	// Increment sum. Loop over mu
	for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	  // Function index is
	  size_t mu=mu0+imu;

	  // Loop over the functions on the grid point
	  size_t first=grid[ip].f0;
	  size_t last=first+grid[ip].nf;

	  for(size_t inu=first;inu<last;inu++) {
	    // Get index of function
	    size_t nu=flist[inu].ind;

	    gradrho+=P(mu,nu)*grad.row(imu)*flist[inu].f;
	  }
	}
      }
      // Plug in the factor 2 to get the total gradient
      gradrho*=2.0;

      // Increment total force
      f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*vxc[ip]*gradrho;
    }
  }

  // GGA part
  if(do_gga) {
    // Loop over nuclei
    for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
      // Get functions centered on the atom
      std::vector<GaussianShell> shells=bas.get_funcs(inuc);

      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// X in the current grid point
	arma::mat X(3,3);
	X.zeros();

	// Loop over shells on the nucleus
	for(size_t ish=0;ish<shells.size();ish++) {

	  // First function on shell is
	  size_t mu0=shells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=shells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=shells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    // Loop over the functions on the grid point
	    size_t first=grid[ip].f0;
	    size_t last=first+grid[ip].nf;

	    for(size_t inu=first;inu<last;inu++) {
	      // Get index of function
	      size_t nu=flist[inu].ind;

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  X(ic,jc)+=P(mu,nu)*(flist[inu].f*hess(imu,3*ic+jc) + glist[3*inu+jc]*grad(imu,ic));
	    }
	  }
	}
	X*=2.0;

	// The xc "vector" is
	arma::vec xc(3);
	for(int ic=0;ic<3;ic++)
	  xc(ic)=2.0*vsigma[ip]*grho[3*ip+ic];

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*arma::trans(X*xc);
      }
    }
  }

  // meta-GGA part
  if(do_mgga) {

    // First part. Loop over nuclei
    for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
      // Get functions centered on the atom
      std::vector<GaussianShell> shells=bas.get_funcs(inuc);

      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// y in the current grid point
	arma::vec y(3);
	y.zeros();

	// Loop over shells on the nucleus
	for(size_t ish=0;ish<shells.size();ish++) {

	  // First function on shell is
	  size_t mu0=shells[ish].get_first_ind();

	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=shells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    // Loop over the functions on the grid point
	    size_t first=grid[ip].f0;
	    size_t last=first+grid[ip].nf;

	    for(size_t inu=first;inu<last;inu++) {
	      // Get index of function
	      size_t nu=flist[inu].ind;

	      // Collect grad nu
	      arma::vec gnu(3);
	      for(int ic=0;ic<3;ic++)
		gnu(ic)=glist[3*inu+ic];

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  y(ic)+=P(mu,nu)*hess(imu,3*ic+jc)*gnu(jc);
	    }
	  }
	}

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*(vtau[ip]+2.0*vlapl[ip])*arma::trans(y);
      }
    }

    // Second part. Loop over nuclei
    for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
      // Get functions centered on the atom
      std::vector<GaussianShell> shells=bas.get_funcs(inuc);

      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// z in the current grid point
	arma::rowvec z(3);
	z.zeros();

	// Loop over shells on the nucleus
	for(size_t ish=0;ish<shells.size();ish++) {

	  // First function on shell is
	  size_t mu0=shells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=shells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the laplacian of the gradient in the current grid point
	  arma::mat laplgrad=shells[ish].eval_laplgrad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<shells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    // Loop over the functions on the grid point
	    size_t first=grid[ip].f0;
	    size_t last=first+grid[ip].nf;

	    for(size_t inu=first;inu<last;inu++) {
	      // Get index of function
	      size_t nu=flist[inu].ind;

	      z+=P(mu,nu)*(llist[inu]*grad.row(imu)+flist[inu].f*laplgrad.row(imu));
	    }
	  }
	}
	z*=2.0;

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*vlapl[ip]*z;
      }
    }
  }

  return arma::trans(f);
}

AtomGrid::AtomGrid(bool lobatto, double toler) {
  use_lobatto=lobatto;

  // These should really be set separately using the routines below.
  tol=toler;
  do_grad=false;
  do_lapl=false;
}

void AtomGrid::set_tolerance(double toler) {
  tol=toler;
}

void AtomGrid::check_grad_lapl(int x_func, int c_func) {
  // Do we need gradients?
  do_grad=false;
  if(x_func>0)
    do_grad=do_grad || gradient_needed(x_func);
  if(c_func>0)
    do_grad=do_grad || gradient_needed(c_func);

  // Do we need laplacians?
  do_lapl=false;
  if(x_func>0)
    do_lapl=do_lapl || laplacian_needed(x_func);
  if(c_func>0)
    do_lapl=do_lapl || laplacian_needed(c_func);
}

atomgrid_t AtomGrid::construct(const BasisSet & bas, size_t cenind, int nrad, int lmax, bool verbose) {
  // Returned info
  atomgrid_t ret;

  // Store index of center
  ret.atind=cenind;
  // and its coordinates
  ret.cen=bas.get_nuclear_coords(cenind);
  ret.ngrid=0;
  ret.nfunc=0;

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev(nrad,rad,wrad);
  nrad=rad.size(); // Sanity check

  // Allocate memory
  ret.sh.resize(nrad);

  // Loop over radii
  for(size_t ir=0;ir<rad.size();ir++) {
    // Store shell data
    ret.sh[ir].r=rad[ir];
    ret.sh[ir].w=wrad[ir]*rad[ir]*rad[ir];
    ret.sh[ir].l=lmax;
  }

  // Add shells
  for(size_t ir=0;ir<ret.sh.size();ir++)
    // Form radial shell
    if(use_lobatto)
      add_lobatto_shell(ret,ir);
    else
      add_lebedev_shell(ret,ir);

  // Form grid
  form_grid(bas,ret);
  // Compute values of basis functions
  compute_bf(bas,ret);

  // Store amount of grid and
  ret.ngrid=grid.size();
  ret.nfunc=flist.size();

  // Free memory
  free();

  if(verbose) {
    //    printf("\t%4u  %7s  %10s\n",(unsigned int) ret.atind+1,space_number(ret.ngrid).c_str(),space_number(ret.nfunc).c_str());
    printf("\t%4u  %7u  %10u\n",(unsigned int) ret.atind+1,(unsigned int) ret.ngrid,(unsigned int) ret.nfunc);
    fflush(stdout);
  }

  return ret;
}

atomgrid_t AtomGrid::construct(const BasisSet & bas, const arma::mat & P, size_t cenind, int x_func, int c_func, bool verbose) {
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
  ret.cen=bas.get_nuclear_coords(cenind);

  // Compute necessary number of radial points
  size_t nrad=std::max(20,(int) round(-5*(3*log10(tol)+6-element_row[bas.get_Z(ret.atind)])));

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev(nrad,rad,wrad);
  nrad=rad.size(); // Sanity check

  // Allocate memory
  ret.sh.resize(nrad);
  // Loop over radii
  for(size_t ir=0;ir<rad.size();ir++) {
    // Store shell data
    ret.sh[ir].r=rad[ir];
    ret.sh[ir].w=wrad[ir]*rad[ir]*rad[ir];
    ret.sh[ir].l=3;
  }

  // Number of basis functions
  size_t Nbf=bas.get_Nbf();

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(tol));

  // Old and new diagonal elements of Hamiltonian
  arma::vec Hold(Nbf), Hnew(Nbf);
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
      update_density(P);

      // Compute exchange and correlation.
      init_xc();
      // Compute the functionals
      if(x_func>0)
	compute_xc(x_func);
      if(c_func>0)
	compute_xc(c_func);

      // Construct the Fock matrix
      Hnew.zeros();
      eval_diag_Fxc(Hnew);

      // Compute maximum difference of diagonal elements of Fock matrix
      maxdiff=arma::max(arma::abs(Hold-Hnew));

      // Switch contents
      std::swap(Hold,Hnew);

      // Increment order if tolerance not achieved.
      if(maxdiff>tol/rad.size()) {
	if(use_lobatto)
	  ret.sh[ir].l+=2;
	else {
	  // Need to determine what is next order of Lebedev
	  // quadrature that is supported.
	  ret.sh[ir].l=next_lebedev(ret.sh[ir].l);
	}
      }
    } while(maxdiff>tol/rad.size() && ret.sh[ir].l<=lmax);

    // Increase number of points and function values
    ret.ngrid+=grid.size();
    ret.nfunc+=flist.size();
  }

  // Free memory once more
  free();

  if(verbose) {
    //    printf("\t%4u  %7s  %10s  %s\n",(unsigned int) ret.atind+1,space_number(ret.ngrid).c_str(),space_number(ret.nfunc).c_str(),t.elapsed().c_str());
    printf("\t%4u  %7u  %10u  %s\n",(unsigned int) ret.atind+1,(unsigned int) ret.ngrid,(unsigned int) ret.nfunc,t.elapsed().c_str());
    fflush(stdout);
  }

  return ret;
}

atomgrid_t AtomGrid::construct(const BasisSet & bas, const arma::mat & Pa, const arma::mat & Pb, size_t cenind, int x_func, int c_func, bool verbose) {
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
  ret.cen=bas.get_nuclear_coords(cenind);

  // Compute necessary number of radial points
  size_t nrad=std::max(20,(int) round(-5*(3*log10(tol)+6-element_row[bas.get_Z(ret.atind)])));

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev(nrad,rad,wrad);
  nrad=rad.size(); // Sanity check

  // Allocate memory
  ret.sh.resize(nrad);
  // Loop over radii
  for(size_t ir=0;ir<rad.size();ir++) {
    // Store shell data
    ret.sh[ir].r=rad[ir];
    ret.sh[ir].w=wrad[ir]*rad[ir]*rad[ir];
    ret.sh[ir].l=3;
  }

  // Number of basis functions
  size_t Nbf=bas.get_Nbf();

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(tol));

  // Old and new diagonal elements of Hamiltonian
  arma::vec Haold(Nbf), Hbold(Nbf);
  arma::vec Hanew(Nbf), Hbnew(Nbf);

  Haold.zeros();
  Hbold.zeros();

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
      update_density(Pa,Pb);

      // Compute exchange and correlation.
      init_xc();
      // Compute the functionals
      if(x_func>0)
	compute_xc(x_func);
      if(c_func>0)
	compute_xc(c_func);
      // and construct the Fock matrices
      Hanew.zeros();
      Hbnew.zeros();
      eval_diag_Fxc(Hanew,Hbnew);

      // Compute maximum difference of diagonal elements of Fock matrix
      maxdiff=std::max(arma::max(arma::abs(Haold-Hanew)),arma::max(arma::abs(Hbold-Hbnew)));

      // Copy contents
      std::swap(Haold,Hanew);
      std::swap(Hbold,Hbnew);

      // Increment order if tolerance not achieved.
      if(maxdiff>tol/rad.size()) {
	if(use_lobatto)
	  ret.sh[ir].l+=2;
	else {
	  // Need to determine what is next order of Lebedev
	  // quadrature that is supported.
	  ret.sh[ir].l=next_lebedev(ret.sh[ir].l);
	}
      }
    } while(maxdiff>tol/rad.size() && ret.sh[ir].l<=lmax);

    // Increase number of points and function values
    ret.ngrid+=grid.size();
    ret.nfunc+=flist.size();
  }

  // Free memory once more
  free();

  if(verbose) {
    //    printf("\t%4u  %7s  %10s  %s\n",(unsigned int) ret.atind+1,space_number(ret.ngrid).c_str(),space_number(ret.nfunc).c_str(),t.elapsed().c_str());
    printf("\t%4u  %7u  %10u  %s\n",(unsigned int) ret.atind+1,(unsigned int)ret.ngrid,(unsigned int) ret.nfunc,t.elapsed().c_str());
    fflush(stdout);
  }

  return ret;
}

atomgrid_t AtomGrid::construct_becke(const BasisSet & bas, size_t cenind, bool verbose) {
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
  ret.cen=bas.get_nuclear_coords(cenind);

  // Compute necessary number of radial points
  size_t nrad=std::max(20,(int) round(-5*(3*log10(tol)+8-element_row[bas.get_Z(ret.atind)])));

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev(nrad,rad,wrad);
  nrad=rad.size(); // Sanity check

  // Allocate memory
  ret.sh.resize(nrad);

  // Loop over radii
  for(size_t ir=0;ir<rad.size();ir++) {
    // Store shell data
    ret.sh[ir].r=rad[ir];
    ret.sh[ir].w=wrad[ir]*rad[ir]*rad[ir];
    ret.sh[ir].l=3;
  }

  // Number of basis functions
  size_t Nbf=bas.get_Nbf();

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(tol));

  // Old and new diagonal elements of overlap
  arma::vec Sold(Nbf), Snew(Nbf);
  Sold.zeros();

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

      // Compute new overlap
      Snew.zeros();
      eval_diag_overlap(Snew);

      // Compute maximum difference of diagonal elements
      maxdiff=arma::max(arma::abs(Snew-Sold));

      // Copy contents
      std::swap(Snew,Sold);

      // Increment order if tolerance not achieved.
      if(maxdiff>tol/rad.size()) {
	if(use_lobatto)
	  ret.sh[ir].l+=2;
	else {
	  // Need to determine what is next order of Lebedev
	  // quadrature that is supported.
	  ret.sh[ir].l=next_lebedev(ret.sh[ir].l);
	}
      }
    } while(maxdiff>tol/rad.size() && ret.sh[ir].l<=lmax);

    // Increase number of points and function values
    ret.ngrid+=grid.size();
    ret.nfunc+=flist.size();
  }

  // Free memory once more
  free();

  if(verbose) {
    //    printf("\t%4u  %7s  %10s  %s\n",(unsigned int) ret.atind+1,space_number(ret.ngrid).c_str(),space_number(ret.nfunc).c_str(),t.elapsed().c_str());
    printf("\t%4u  %7u  %10u  %s\n",(unsigned int) ret.atind+1,(unsigned int) ret.ngrid,(unsigned int) ret.nfunc,t.elapsed().c_str());
    fflush(stdout);
  }

  return ret;
}

atomgrid_t AtomGrid::construct_hirshfeld(const BasisSet & bas, size_t cenind, const Hirshfeld & hirsh, bool verbose) {
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
  ret.cen=bas.get_nuclear_coords(cenind);

  // Compute necessary number of radial points
  size_t nrad=std::max(20,(int) round(-5*(3*log10(tol)+8-element_row[bas.get_Z(ret.atind)])));

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev(nrad,rad,wrad);
  nrad=rad.size(); // Sanity check

  // Allocate memory
  ret.sh.resize(nrad);

  // Loop over radii
  for(size_t ir=0;ir<rad.size();ir++) {
    // Store shell data
    ret.sh[ir].r=rad[ir];
    ret.sh[ir].w=wrad[ir]*rad[ir]*rad[ir];
    ret.sh[ir].l=3;
  }

  // Number of basis functions
  size_t Nbf=bas.get_Nbf();

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(tol));

  // Old and new diagonal elements of overlap
  arma::vec Sold(Nbf), Snew(Nbf);
  Sold.zeros();

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
      // Compute Hirshfeld weights for radial shell
      hirshfeld_weights(hirsh,ret,ir);
      // Prune points with small weight
      prune_points(1e-8*tol,ret.sh[ir]);

      // Compute values of basis functions
      compute_bf(bas,ret,ir);

      // Compute new overlap
      Snew.zeros();
      eval_diag_overlap(Snew);

      // Compute maximum difference of diagonal elements
      maxdiff=arma::max(arma::abs(Snew-Sold));

      // Copy contents
      std::swap(Snew,Sold);

      // Increment order if tolerance not achieved.
      if(maxdiff>tol/rad.size()) {
	if(use_lobatto)
	  ret.sh[ir].l+=2;
	else {
	  // Need to determine what is next order of Lebedev
	  // quadrature that is supported.
	  ret.sh[ir].l=next_lebedev(ret.sh[ir].l);
	}
      }
    } while(maxdiff>tol/rad.size() && ret.sh[ir].l<=lmax);

    // Increase number of points and function values
    ret.ngrid+=grid.size();
    ret.nfunc+=flist.size();
  }

  // Free memory once more
  free();

  if(verbose) {
    //    printf("\t%4u  %7s  %10s  %s\n",(unsigned int) ret.atind+1,space_number(ret.ngrid).c_str(),space_number(ret.nfunc).c_str(),t.elapsed().c_str());
    printf("\t%4u  %7u  %10u  %s\n",(unsigned int) ret.atind+1,(unsigned int) ret.ngrid,(unsigned int) ret.nfunc,t.elapsed().c_str());
    fflush(stdout);
  }

  return ret;
}

AtomGrid::~AtomGrid() {
}

void AtomGrid::form_grid(const BasisSet & bas, atomgrid_t & g) {
  // Clear anything that already exists
  free();

  // Check allocation
  if(g.ngrid>grid.capacity())
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

  // Store number of points
  g.ngrid=grid.size();
}

void AtomGrid::form_hirshfeld_grid(const Hirshfeld & hirsh, atomgrid_t & g) {
  // Clear anything that already exists
  free();

  // Check allocation
  if(g.ngrid>grid.capacity())
    grid.reserve(g.ngrid);

  // Loop over radial shells
  for(size_t ir=0;ir<g.sh.size();ir++) {
    // Add grid points
    if(use_lobatto)
      add_lobatto_shell(g,ir);
    else
      add_lebedev_shell(g,ir);
    // Do Becke weights
    hirshfeld_weights(hirsh,g,ir);
    // Prune points with small weight
    prune_points(1e-8*tol,g.sh[ir]);
  }

  // Store number of points
  g.ngrid=grid.size();
}

void AtomGrid::compute_bf(const BasisSet & bas, atomgrid_t & g) {
  // Check allocation
  if(g.nfunc>flist.capacity())
    flist.reserve(g.nfunc);
  if(do_grad && (3*g.nfunc>glist.capacity()))
    glist.reserve(3*g.nfunc);
  if(do_lapl && g.nfunc>llist.capacity())
    llist.reserve(g.nfunc);

  // Loop over radial shells
  for(size_t ir=0;ir<g.sh.size();ir++) {
    compute_bf(bas,g,ir);
  }

  // Store number of function values
  g.nfunc=flist.size();
}

void AtomGrid::compute_bf(const BasisSet & bas, const atomgrid_t & g, size_t irad) {
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

  if(do_lapl) {
    // Loop over points
    for(size_t ip=g.sh[irad].ind0;ip<g.sh[irad].ind0+g.sh[irad].np;ip++) {
      // Store index of first function on grid point
      grid[ip].f0=flist.size();
      // Number of functions on point
      grid[ip].nf=0;

      // Loop over shells
      for(size_t ish=0;ish<compute_shells.size();ish++) {
	// Center of shell is
	coords_t shell_center=bas.get_shell_center(compute_shells[ish]);
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
	  // and their laplacians
	  arma::vec lval=bas.eval_lapl(compute_shells[ish],grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // and add them to the list
	  bf_f_t hlp;
	  for(size_t ifunc=0;ifunc<fval.n_elem;ifunc++) {
	    // Index of function is
	    hlp.ind=ind0+ifunc;
	    // Value is
	    hlp.f=fval(ifunc);
	    // Add to stack
	    flist.push_back(hlp);
	    // and gradient
	    for(int ic=0;ic<3;ic++)
	      glist.push_back(gval(ifunc,ic));
	    // and laplacian
	    llist.push_back(lval(ifunc));

	    // Increment number of functions in point
	    grid[ip].nf++;
	  }
	}
      }
    }
  } else if(do_grad) {
    // Loop over points
    for(size_t ip=g.sh[irad].ind0;ip<g.sh[irad].ind0+g.sh[irad].np;ip++) {
      // Store index of first function on grid point
      grid[ip].f0=flist.size();
      // Number of functions on point
      grid[ip].nf=0;

      // Loop over shells
      for(size_t ish=0;ish<compute_shells.size();ish++) {
	// Center of shell is
	coords_t shell_center=bas.get_shell_center(compute_shells[ish]);
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
	coords_t shell_center=bas.get_shell_center(compute_shells[ish]);
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

DFTGrid::DFTGrid() {
}

DFTGrid::DFTGrid(const BasisSet * bas, bool ver, bool lobatto) {
  basp=bas;
  verbose=ver;
  use_lobatto=lobatto;

  // Allocate atomic grids
  grids.resize(bas->get_Nnuc());

  // Allocate work grids
#ifdef _OPENMP
  int nth=omp_get_max_threads();
  for(int i=0;i<nth;i++)
    wrk.push_back(AtomGrid(lobatto));
#else
  wrk.push_back(AtomGrid(lobatto));
#endif
}

DFTGrid::~DFTGrid() {
}

void DFTGrid::construct(int nrad, int lmax, int x_func, int c_func) {
  if(verbose) {
    printf("Composition of static DFT grid:\n");
    printf("\t%4s  %7s  %10s\n","atom","Npoints","Nfuncs");
    fflush(stdout);
  }

  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  // Construct grids
  size_t Nat=basp->get_Nnuc();
#ifdef _OPENMP
#pragma omp parallel
  {
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
    for(size_t i=0;i<Nat;i++) {
      grids[i]=wrk[ith].construct(*basp,i,nrad,lmax,verbose);
    }
  }
#else
  for(size_t i=0;i<Nat;i++)
    grids[i]=wrk[0].construct(*basp,i,nrad,lmax,verbose);
#endif


}

void DFTGrid::construct(const arma::mat & P, double tol, int x_func, int c_func) {

  // Add all atoms
  if(verbose) {
    printf("Constructing DFT grid.\n");
    printf("\t%4s  %7s  %10s  %s\n","atom","Npoints","Nfuncs","t");
    fflush(stdout);
  }

  // Set tolerances
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_tolerance(tol);
  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  Timer t;

  size_t Nat=basp->get_Nnuc();

#ifdef _OPENMP
#pragma omp parallel
  {
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
    for(size_t i=0;i<Nat;i++) {
      grids[i]=wrk[ith].construct(*basp,P,i,x_func,c_func,verbose);
    }
  }
#else
  for(size_t i=0;i<Nat;i++)
    grids[i]=wrk[0].construct(*basp,P,i,x_func,c_func,verbose);
#endif

  if(verbose) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

void DFTGrid::construct(const arma::mat & Pa, const arma::mat & Pb, double tol, int x_func, int c_func) {
  // Add all atoms
  if(verbose) {
    printf("\t%4s  %7s  %10s  %s\n","atom","Npoints","Nfuncs","t");
    fflush(stdout);
  }

  // Set tolerances
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_tolerance(tol);
  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  Timer t;

  size_t Nat=basp->get_Nnuc();

#ifdef _OPENMP
#pragma omp parallel
  {
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
    for(size_t i=0;i<Nat;i++)
      grids[i]=wrk[ith].construct(*basp,Pa,Pb,i,x_func,c_func,verbose);
  }
#else
  for(size_t i=0;i<Nat;i++)
    grids[i]=wrk[0].construct(*basp,Pa,Pb,i,x_func,c_func,verbose);
#endif

  if(verbose) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

void DFTGrid::construct(const std::vector<arma::mat> & P, double tol, int x_func, int c_func) {
  // Add all atoms
  if(verbose) {
    printf("\t%4s  %4s  %7s  %10s  %s\n","atom","orb","Npoints","Nfuncs","t");
    fflush(stdout);
  }

  // Set tolerances
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_tolerance(tol);
  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  Timer t;

  const size_t Nat=basp->get_Nnuc();
  const size_t Norb=P.size();

  // Orbital grid lists
  std::vector< std::vector<atomgrid_t> > orbgrid(Norb);
  for(size_t iorb=0;iorb<Norb;iorb++)
    orbgrid[iorb].resize(Nat);

  // Dummy density matrix
  arma::mat Pdum(P[0]);
  Pdum.zeros();

#ifdef _OPENMP
#pragma omp parallel
  {
    int ith=omp_get_thread_num();
    // Collapse statement introduced in OpenMP 3.0 in May 2008
#if _OPENMP >= 200805
#pragma omp for schedule(dynamic,1) collapse(2)
    for(size_t iat=0;iat<Nat;iat++)
      for(size_t iorb=0;iorb<Norb;iorb++) {
#else
#pragma omp for schedule(dynamic,1)
    for(size_t iorb=0;iorb<Norb;iorb++)
      for(size_t iat=0;iat<Nat;iat++) {
#endif
	Timer toa;

	// Construct the grid
	orbgrid[iorb][iat]=wrk[ith].construct(*basp,P[iorb],Pdum,iat,x_func,c_func,false);

	// Print out info
	if(verbose) {
	  printf("\t%4u  %4u  %7u  %10u  %s\n",(unsigned int) iat+1,(unsigned int) iorb+1,(unsigned int) orbgrid[iorb][iat].ngrid,(unsigned int) orbgrid[iorb][iat].nfunc,toa.elapsed().c_str());
	  fflush(stdout);
	}
      }
  }
#else
  for(size_t iorb=0;iorb<Norb;iorb++)
    for(size_t iat=0;iat<Nat;iat++) {
      Timer toa;

      orbgrid[iorb][iat]=wrk[0].construct(*basp,P[iorb],Pdum,iat,x_func,c_func,false);

      // Print out info
      if(verbose) {
	printf("\t%4u  %4u  %7u  %10u  %s\n",(unsigned int) iat+1,(unsigned int) iorb+1,(unsigned int) orbgrid[iorb][iat].ngrid,(unsigned int) orbgrid[iorb][iat].nfunc,toa.elapsed().c_str());
	fflush(stdout);
      }
    }
#endif

  // Collect orbital grids
  grids.resize(orbgrid[0].size());
  for(size_t iat=0;iat<orbgrid[0].size();iat++) {
    // Initialize atomic grid
    grids[iat]=orbgrid[0][iat];

    // Loop over radial shells
    for(size_t irad=0;irad<grids[iat].sh.size();irad++) {
      // Rule order
      int l=orbgrid[0][iat].sh[irad].l;

      // Loop over orbitals
      for(size_t iorb=1;iorb<orbgrid.size();iorb++)
	l=std::max(l,orbgrid[iorb][iat].sh[irad].l);

      // Store l
      grids[iat].sh[irad].l=l;
    }
  }

  // Update grid sizes
#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);
      // Free memory
      wrk[ith].free();
    }
  }


  // Print out info
  if(verbose) {
    printf("Final grid size\n\t%4s  %7s  %10s\n","atom","Npoints","Nfuncs");
    for(size_t iat=0;iat<grids.size();iat++)
      printf("\t%4u  %7u  %10u\n",(unsigned int) iat+1,(unsigned int) grids[iat].ngrid,(unsigned int) grids[iat].nfunc);
    printf("SIC-DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

void DFTGrid::construct_becke(double tol) {

  // Add all atoms
  if(verbose) {
    printf("Constructing Becke grid.\n");
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
    printf("Becke grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

void DFTGrid::construct_hirshfeld(const Hirshfeld & hirsh, double tol) {

  // Add all atoms
  if(verbose) {
    printf("Constructing Hirshfeld grid.\n");
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
      grids[i]=wrk[ith].construct_hirshfeld(*basp,i,hirsh,verbose);
    }

  }   // End parallel section

  if(verbose) {
    printf("Hirshfeld grid constructed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
  }
}

size_t DFTGrid::get_Npoints() const {
  size_t np=0;
  for(size_t i=0;i<grids.size();i++)
    np+=grids[i].ngrid;
  return np;
}

size_t DFTGrid::get_Nfuncs() const {
  size_t nf=0;
  for(size_t i=0;i<grids.size();i++)
    nf+=grids[i].nfunc;
  return nf;
}

arma::mat DFTGrid::eval_overlap() {
  std::vector<arma::mat> Sat=eval_overlaps();
  arma::mat S=Sat[0];
  for(size_t inuc=1;inuc<Sat.size();inuc++)
    S+=Sat[inuc];

  return S;
}

arma::mat DFTGrid::eval_overlap(size_t inuc) {
  // Amount of basis functions
  size_t N=basp->get_Nbf();

  // Returned matrix
  arma::mat Sat(N,N);
  Sat.zeros();

#ifdef _OPENMP
  int ith=omp_get_thread_num();
#else
  int ith=0;
#endif

  // Change atom and create grid
  wrk[ith].form_grid(*basp,grids[inuc]);
  // Compute basis functions
  wrk[ith].compute_bf(*basp,grids[inuc]);
  // Evaluate overlap
  wrk[ith].eval_overlap(Sat);
  // Free memory
  wrk[ith].free();

  return Sat;
}

std::vector<arma::mat> DFTGrid::eval_overlaps() {
  // Amount of basis functions
  size_t N=basp->get_Nbf();

  // Returned matrices
  std::vector<arma::mat> Sat(basp->get_Nnuc());
  for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++)
    Sat[inuc].zeros(N,N);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

  // Add atomic contributions
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);
      // Evaluate overlap
      wrk[ith].eval_overlap(Sat[inuc]);
      // Free memory
      wrk[ith].free();
    }
  }

  return Sat;
}

arma::mat DFTGrid::eval_hirshfeld_overlap(const Hirshfeld & hirsh, size_t inuc) {
  // Amount of basis functions
  size_t N=basp->get_Nbf();

  // Returned matrices
  arma::mat Sat(N,N);
  Sat.zeros();

#ifdef _OPENMP
  int ith=omp_get_thread_num();
#else
  int ith=0;
#endif

  // Change atom and create grid
  wrk[ith].form_hirshfeld_grid(hirsh,grids[inuc]);
  // Compute basis functions
  wrk[ith].compute_bf(*basp,grids[inuc]);
  // Evaluate overlap
  wrk[ith].eval_overlap(Sat);
  // Free memory
  wrk[ith].free();

  return Sat;
}

std::vector<arma::mat> DFTGrid::eval_hirshfeld_overlaps(const Hirshfeld & hirsh) {
  // Amount of basis functions
  size_t N=basp->get_Nbf();

  // Returned matrices
  std::vector<arma::mat> Sat(basp->get_Nnuc());
  for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++)
    Sat[inuc].zeros(N,N);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

  // Add atomic contributions
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_hirshfeld_grid(hirsh,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);
      // Evaluate overlap
      wrk[ith].eval_overlap(Sat[inuc]);
      // Free memory
      wrk[ith].free();
    }
  }

  return Sat;
}

std::vector<dens_list_t> DFTGrid::eval_dens_list(const arma::mat & P) {
  std::vector<dens_list_t> list;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
    std::vector<dens_list_t> hlp;
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    // Loop over integral grid
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);

#ifndef _OPENMP
      // Compute density
      wrk[ith].eval_dens(P,list);
#else
      // Compute helper
      hlp.clear();
      wrk[ith].eval_dens(P,hlp);
#pragma omp critical
      // Add to full list
      list.insert(list.end(),hlp.begin(),hlp.end());
#endif
      // Free memory
      wrk[ith].free();
    }
  }

  // Sort the list
  std::sort(list.begin(),list.end());

  return list;
}

double DFTGrid::compute_Nel(const arma::mat & P) {
  double Nel=0.0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:Nel)
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    // Loop over integral grid
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);
      // Update density
      wrk[ith].update_density(P);
      // Integrate electrons
      Nel+=wrk[ith].compute_Nel();
      // Free memory
      wrk[ith].free();
    }
  }

  return Nel;
}

double DFTGrid::compute_Nel(const arma::mat & Pa, const arma::mat & Pb) {
  double Nel=0.0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:Nel)
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    // Loop over integral grid
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);
      // Update density
      wrk[ith].update_density(Pa,Pb);
      // Integrate electrons
      Nel+=wrk[ith].compute_Nel();
      // Free memory
      wrk[ith].free();
    }
  }

  return Nel;
}

arma::vec DFTGrid::compute_atomic_Nel(const arma::mat & P) {
   arma::vec Nel(grids.size());

#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    // Loop over integral grid
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);
      // Update density
      wrk[ith].update_density(P);
      // Integrate electrons
      Nel[inuc]=wrk[ith].compute_Nel();
      // Free memory
      wrk[ith].free();
    }
  }
  
  return Nel;
}


arma::vec DFTGrid::compute_atomic_Nel(const Hirshfeld & hirsh, const arma::mat & P) {
   arma::vec Nel(grids.size());

#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    // Loop over integral grid
    for(size_t inuc=0;inuc<grids.size();inuc++) {
      // Change atom and create grid
      wrk[ith].form_hirshfeld_grid(hirsh,grids[inuc]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[inuc]);
      // Update density
      wrk[ith].update_density(P);
      // Integrate electrons
      Nel[inuc]=wrk[ith].compute_Nel();
      // Free memory
      wrk[ith].free();
    }
  }
  
  return Nel;
}


#ifdef CONSISTENCYCHECK
void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Exc, double & Nel) {
  // Work arrays
  arma::mat Ha, Hb;
  Ha=H;
  Hb=H;

  Ha.zeros(P.n_rows,P.n_cols);
  Hb.zeros(P.n_rows,P.n_cols);

  eval_Fxc(x_func,c_func,P/2.0,P/2.0,Ha,Hb,Exc,Nel);
  H=(Ha+Hb)/2.0;
}
#else
void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Excv, double & Nelv) {
  // Clear Hamiltonian
  H.zeros(P.n_rows,P.n_cols);
  // Clear exchange-correlation energy
  double Exc=0.0;
  // Clear number of electrons
  double Nel=0.0;

#ifdef _OPENMP
  // Get (maximum) number of threads
  int maxt=omp_get_max_threads();

  // Stack of work arrays
  std::vector<arma::mat> Hwrk;

  for(int i=0;i<maxt;i++) {
    Hwrk.push_back(arma::mat(H.n_rows,H.n_cols));
    Hwrk[i].zeros();
  }

#pragma omp parallel shared(Hwrk) reduction(+:Nel,Exc)
#endif
  { // Begin parallel region

#ifdef _OPENMP
    // Current thread is
    int ith=omp_get_thread_num();

#pragma omp for schedule(dynamic,1)
#else
    int ith=0;
#endif
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[i]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[i]);

      // Update density
      wrk[ith].update_density(P);
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
#ifdef _OPENMP
      wrk[ith].eval_Fxc(Hwrk[ith]);
#else
      wrk[ith].eval_Fxc(H);
#endif
      // Free memory
      wrk[ith].free();
    }
  } // End parallel region

#ifdef _OPENMP
  // Sum results
  for(int i=0;i<maxt;i++)
    H+=Hwrk[i];
#endif

  Excv=Exc;
  Nelv=Nel;
}
#endif // CONSISTENCYCHECK

void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ha, arma::mat & Hb, double & Excv, double & Nelv) {
  // Clear Hamiltonian
  Ha.zeros(Pa.n_rows,Pa.n_cols);
  Hb.zeros(Pb.n_rows,Pb.n_cols);
  // Clear exchange-correlation energy
  double Exc=0.0;
  // Clear number of electrons
  double Nel=0.0;

#ifdef _OPENMP
  // Get (maximum) number of threads
  int maxt=omp_get_max_threads();

  // Stack of work arrays
  std::vector<arma::mat> Hawrk, Hbwrk;

  for(int i=0;i<maxt;i++) {
    Hawrk.push_back(arma::mat(Ha.n_rows,Ha.n_cols));
    Hawrk[i].zeros();

    Hbwrk.push_back(arma::mat(Hb.n_rows,Hb.n_cols));
    Hbwrk[i].zeros();
  }

#pragma omp parallel shared(Hawrk,Hbwrk) reduction(+:Nel,Exc)
#endif
  { // Begin parallel region

#ifdef _OPENMP
    // Current thread is
    int ith=omp_get_thread_num();

#pragma omp for schedule(dynamic,1)
#else
    int ith=0;
#endif
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {

      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[i]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[i]);

      // Update density
      wrk[ith].update_density(Pa,Pb);
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
#ifdef _OPENMP
      wrk[ith].eval_Fxc(Hawrk[ith],Hbwrk[ith]);
#else
      wrk[ith].eval_Fxc(Ha,Hb);
#endif

      // Free memory
      wrk[ith].free();
    }
  } // End parallel region

#ifdef _OPENMP
  // Sum results
  for(int i=0;i<maxt;i++) {
    Ha+=Hawrk[i];
    Hb+=Hbwrk[i];
  }
#endif

  Excv=Exc;
  Nelv=Nel;
}


void DFTGrid::eval_Fxc(int x_func, int c_func, const std::vector<arma::mat> & Pa, std::vector<arma::mat> & Ha, std::vector<double> & Exc, std::vector<double> & Nel) {
  // Clear Hamiltonian
  Ha.resize(Pa.size());
  for(size_t ip=0;ip<Pa.size();ip++)
    Ha[ip].zeros(Pa[ip].n_rows,Pa[ip].n_cols);

  // Clear exchange-correlation energy
  Exc.assign(Pa.size(),0.0);
  // Clear number of electrons
  Nel.assign(Pa.size(),0.0);

#ifdef _OPENMP
#pragma omp parallel
  { // Begin parallel region

    arma::mat Hdum(Pa[0]);
    Hdum.zeros();
    arma::mat Pdum(Pa[0]);
    Pdum.zeros();

    arma::mat Hwrk(Hdum);
    Hwrk.zeros();

    // Current thread is
    int ith=omp_get_thread_num();

#pragma omp for schedule(dynamic,1)
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {

      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[i]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[i]);

      // Loop over densities
      for(size_t ip=0;ip<Pa.size();ip++) {
	// Update density
	wrk[ith].update_density(Pa[ip],Pdum);
	// Update number of electrons
#pragma omp atomic
	Nel[ip]+=wrk[ith].compute_Nel();

	// Initialize the arrays
	wrk[ith].init_xc();
	// Compute the functionals
	if(x_func>0)
	  wrk[ith].compute_xc(x_func);
	if(c_func>0)
	  wrk[ith].compute_xc(c_func);

	// Evaluate the energy
#pragma omp atomic
	Exc[ip]+=wrk[ith].eval_Exc();
	// and construct the Fock matrices
	Hwrk.zeros(); // need to clear this here
	wrk[ith].eval_Fxc(Hwrk,Hdum);

	// Accumulate Fock matrix
#pragma omp critical
	Ha[ip]+=Hwrk;
      }
    }

    // Free memory
    wrk[ith].free();

  } // End parallel region
#else

  arma::mat Hdum(Pa[0]);
  Hdum.zeros();
  arma::mat Pdum(Pa[0]);
  Pdum.zeros();

  // Loop over atoms
  for(size_t i=0;i<grids.size();i++) {
    // Change atom and create grid
    wrk[0].form_grid(*basp,grids[i]);
    // Compute basis functions
    wrk[0].compute_bf(*basp,grids[i]);

    // Loop over densities
    for(size_t ip=0;ip<Pa.size();ip++) {
	// Update density
	wrk[0].update_density(Pa[ip],Pdum);
	// Update number of electrons
	Nel[ip]+=wrk[0].compute_Nel();

	// Initialize the arrays
	wrk[0].init_xc();
	// Compute the functionals
	if(x_func>0)
	  wrk[0].compute_xc(x_func);
	if(c_func>0)
	  wrk[0].compute_xc(c_func);

	// Evaluate the energy
	Exc[ip]+=wrk[0].eval_Exc();
	// and construct the Fock matrices
	wrk[0].eval_Fxc(Ha[ip],Hdum);
    }

    // Free memory
    wrk[0].free();
  }
#endif
}

arma::vec DFTGrid::eval_force(int x_func, int c_func, const arma::mat & P) {
  arma::vec f(3*basp->get_Nnuc());
  f.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region

#ifndef _OPENMP
    int ith=0;
#else
    // Current thread is
    int ith=omp_get_thread_num();

    // Helper
    arma::vec fwrk(f);

#pragma omp for schedule(dynamic,1)
#endif
    // Loop over atoms
    for(size_t iat=0;iat<grids.size();iat++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[iat]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[iat]);

      // Update density
      wrk[ith].update_density(P);

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk[ith].compute_xc(x_func);
      if(c_func>0)
	wrk[ith].compute_xc(c_func);

      // Calculate the force on the atom
#ifdef _OPENMP
      fwrk+=wrk[ith].eval_force(*basp,P);
#else
      f+=wrk[ith].eval_force(*basp,P);
#endif

      // Free memory
      wrk[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    f+=fwrk;
#endif
  } // End parallel region

  return f;
}

arma::vec DFTGrid::eval_force(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb) {
  arma::vec f(3*basp->get_Nnuc());
  f.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region

#ifndef _OPENMP
    int ith=0;
#else
    // Current thread is
    int ith=omp_get_thread_num();

    // Helper
    arma::vec fwrk(f);

#pragma omp for schedule(dynamic,1)
#endif
    // Loop over atoms
    for(size_t iat=0;iat<grids.size();iat++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[iat]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[iat]);

      // Update density
      wrk[ith].update_density(Pa,Pb);

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk[ith].compute_xc(x_func);
      if(c_func>0)
	wrk[ith].compute_xc(c_func);

      // Calculate the force on the atom
#ifdef _OPENMP
      fwrk+=wrk[ith].eval_force(*basp,Pa,Pb);
#else
      f+=wrk[ith].eval_force(*basp,Pa,Pb);
#endif

      // Free memory
      wrk[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    f+=fwrk;
#endif
  } // End parallel region

  return f;
}

void DFTGrid::print_density_potential(int func_id, const arma::mat & Pa, const arma::mat & Pb, std::string densname, std::string potname) {

  // Open output files
  FILE *dens=fopen(densname.c_str(),"w");
  FILE *pot=fopen(potname.c_str(),"w");

  Timer t;
  if(verbose) {
    printf("\nSaving density and potential data in %s and %s ... ",densname.c_str(),potname.c_str());
    fflush(stdout);
  } 
  

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region
    
#ifdef _OPENMP
    // Current thread is
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#else
    int ith=0;
#endif
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {

      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[i]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[i]);

      // Update density
      wrk[ith].update_density(Pa,Pb);

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(func_id>0)
        wrk[ith].compute_xc(func_id);

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk[ith].print_density(dens);
	wrk[ith].print_potential(func_id,pot);
      }

      // Free memory
      wrk[ith].free();
    }
  } // End parallel region

  // Close output files
  fclose(dens);
  fclose(pot);

  printf("done (%s)\n",t.elapsed().c_str());
}

void DFTGrid::print_density_potential(int func_id, const arma::mat & P, std::string densname, std::string potname) {
  // Open output files
  FILE *dens=fopen(densname.c_str(),"w");
  FILE *pot=fopen(potname.c_str(),"w");

  Timer t;
  if(verbose) {
    printf("\nSaving density and potential data in %s and %s ... ",densname.c_str(),potname.c_str());
    fflush(stdout);
  } 

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region

#ifdef _OPENMP
    // Current thread is
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#else
    int ith=0;
#endif
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].form_grid(*basp,grids[i]);
      // Compute basis functions
      wrk[ith].compute_bf(*basp,grids[i]);

      // Update density
      wrk[ith].update_density(P);

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(func_id>0)
	wrk[ith].compute_xc(func_id);

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk[ith].print_density(dens);
	wrk[ith].print_potential(func_id,pot);
      }
      
      // Free memory
      wrk[ith].free();
    }
  } // End parallel region

  // Close output files
  fclose(dens);
  fclose(pot);

  printf("done (%s)\n",t.elapsed().c_str());
}

