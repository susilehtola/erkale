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
#include "hirshfeld.h"
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

/* Partitioning functions */
inline double f_p(double mu) {
  return 1.5*mu-0.5*mu*mu*mu;
}

inline double f_q(double mu, double a) {
  if(mu<-a)
    return -1.0;
  else if(mu<a)
    return f_p(mu/a);
  else
    return 1.0;
}

inline double f_s(double mu, double a) {
  return 0.5*(1.0-f_p(f_p(f_q(mu,a))));
}

bool operator<(const dens_list_t &lhs, const dens_list_t & rhs) {
  // Sort in decreasing order
  return lhs.d > rhs.d;
}

AngularGrid::AngularGrid(bool lobatto_) : use_lobatto(lobatto_) {
}

AngularGrid::~AngularGrid() {
}

void AngularGrid::set_basis(const BasisSet & basis) {
  basp=&basis;
}

void AngularGrid::set_grid(const angshell_t & sh) {
  info=sh;
}

std::vector<gridpoint_t> AngularGrid::get_grid() const {
  return grid;
}

void AngularGrid::lobatto_shell() {
  // Add points on ind:th radial shell.

  // Get parameters of shell
  const double rad(info.R);
  const int l(info.l);
  const double wrad(info.w);

  // Number of points in theta
  int nth=(l+3)/2;

  // Get corresponding Lobatto quadrature rule points in theta
  std::vector<double> xl, wl;
  lobatto_compute(nth,xl,wl);

  // Clear grid
  grid.clear();

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
      point.r=point.r+info.cen;

      // Add point
      grid.push_back(point);
      // Increment number of points
      np++;
    }
  }

  // Store number of points on this shell
  info.np=np;
}

void AngularGrid::lebedev_shell() {
  // Parameters of shell
  const double rad(info.R);
  const int l(info.l);
  const double wrad(info.w);

  // Get quadrature rule
  std::vector<lebedev_point_t> points=lebedev_sphere(l);

  // Number of points on this shell
  size_t np=points.size();

  // Loop over points
  for(size_t i=0;i<points.size();i++) {
    gridpoint_t point;

    point.r.x=rad*points[i].x;
    point.r.y=rad*points[i].y;
    point.r.z=rad*points[i].z;
    // Displace to center
    point.r=point.r+info.cen;

    // Compute quadrature weight
    // (Becke weight not included)
    point.w=wrad*points[i].w;

    // Add point
    grid.push_back(point);
  }

  // Store number of points on this shell
  info.np=np;
}

void AngularGrid::becke_weights(double a) {
  // Compute weights of points.

  // Number of atoms in system
  const size_t Nat=basp->get_Nnuc();

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
  std::vector<nucleus_t> nuccoords=basp->get_nuclei();

  // Get nuclear distances
  arma::mat nucdist=basp->nuclear_distances();
  // Compute closest distance to other atoms
  double Rin=DBL_MAX;
  for(size_t i=0;i<info.atind;i++)
    if(nucdist(info.atind,i)<Rin)
      Rin=nucdist(info.atind,i);
  for(size_t i=info.atind+1;i<Nat;i++)
    if(nucdist(info.atind,i)<Rin)
      Rin=nucdist(info.atind,i);

  double scrthr=std::pow(0.5*(1-a)*Rin,2);

  // Loop over grid
  for(size_t ip=0;ip<grid.size();ip++) {
    // Coordinates of the point are
    coords_t coord_p=grid[ip].r;

    // Prescreen - is the weight unity?
    if(normsq(nuccoords[info.atind].r-coord_p) < scrthr)
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
    grid[ip].w*=atom_weight(info.atind)/arma::sum(atom_weight);
  }
}

void AngularGrid::hirshfeld_weights(const Hirshfeld & hirsh) {
  // Compute weights of points.
  for(size_t ip=0;ip<grid.size();ip++)
    // The Hirshfeld weight is
    grid[ip].w*=hirsh.get_weight(info.atind,grid[ip].r);
}

void AngularGrid::prune_points() {
  // Prune points with small weight.
  for(size_t i=grid.size()-1;i<grid.size();i--)
    if(grid[i].w<=info.tol)
      grid.erase(grid.begin()+i);

  // Update amont of points
  info.np=grid.size();
}

void AngularGrid::free() {
  // Free integration points
  grid.clear();
  w.clear();

  // Free basis lists
  pot_shells.clear();
  pot_bf_ind.clear();
  shells.clear();
  bf_ind.clear();
  bf_potind.clear();

  // Free values of basis functions
  bf.clear();
  bf_x.clear();
  bf_y.clear();
  bf_z.clear();
  bf_lapl.clear();

  // Free density stuff
  Pv.clear();
  Pv_x.clear();
  Pv_y.clear();
  Pv_z.clear();
  Pav.clear();
  Pav_x.clear();
  Pav_y.clear();
  Pav_z.clear();
  Pbv.clear();
  Pbv_x.clear();
  Pbv_y.clear();
  Pbv_z.clear();

  // Free LDA stuff
  rho.clear();
  exc.clear();
  vxc.clear();

  // Free GGA stuff
  grho.clear();
  sigma.clear();
  vsigma.clear();

  // Free mGGA stuff
  lapl.clear();
  vlapl.clear();
  tau.clear();
  vtau.clear();

  // Free VV10 stuff
  VV10_arr.clear();
}

arma::uvec AngularGrid::screen_density(double thr) const {
  // List of points
  std::vector<size_t> idx;
  // Loop over grid
  if(!polarized) {
    for(size_t i=0;i<grid.size();i++)
      if(rho(i)>=thr)
	idx.push_back(i);
  } else {
    for(size_t i=0;i<grid.size();i++)
      if(rho(2*i)+rho(2*i+1)>=thr)
	idx.push_back(i);
  }

  return arma::conv_to<arma::uvec>::from(idx);
}


void AngularGrid::update_density(const arma::mat & P0) {
  // Update values of densitty

  if(!P0.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - density matrix is empty!\n");
  }

  // Non-polarized calculation.
  polarized=false;

  // Update density vector
  arma::mat P(P0.submat(bf_ind,bf_ind));
  Pv=P*bf;

  // Calculate density
  rho.zeros(1,grid.size());
  for(size_t ip=0;ip<grid.size();ip++) {
    double d=0.0;
    for(size_t j=0;j<bf.n_rows;j++)
      d+=Pv(j,ip)*bf(j,ip);
    rho(0,ip)=d;
  }

  // Calculate gradient
  if(do_grad) {
    grho.zeros(3,grid.size());
    sigma.zeros(1,grid.size());
    for(size_t ip=0;ip<grid.size();ip++) {
      // Initialize
      double gx=0.0, gy=0.0, gz=0.0;

      // Calculate
      for(size_t j=0;j<bf.n_rows;j++) {
	gx+=Pv(j,ip)*bf_x(j,ip);
	gy+=Pv(j,ip)*bf_y(j,ip);
	gz+=Pv(j,ip)*bf_z(j,ip);
      }

      // Store values, including the missing factor 2
      grho(0,ip)=2.0*gx;
      grho(1,ip)=2.0*gy;
      grho(2,ip)=2.0*gz;
      // Compute sigma as well
      sigma(0,ip)=4.0*(gx*gx + gy*gy + gz*gz);
    }
  }

  // Calculate laplacian and kinetic energy density
  if(do_lapl) {
    // Adjust size of grid
    lapl.zeros(1,grid.size());
    tau.zeros(1,grid.size());

    // Update helpers
    Pv_x=P*bf_x;
    Pv_y=P*bf_y;
    Pv_z=P*bf_z;

    // Calculate values
    for(size_t ip=0;ip<grid.size();ip++) {
      // Laplacian term
      double lap=0.0;
      // Gradient term
      double grad=0.0;
      // Calculate dot products
      for(size_t j=0;j<bf_lapl.n_rows;j++) {
	lap+=Pv(j,ip)*bf_lapl(j,ip);
	grad+=Pv_x(j,ip)*bf_x(j,ip) + Pv_y(j,ip)*bf_y(j,ip) + Pv_z(j,ip)*bf_z(j,ip);
      }

      // Store values
      lapl(0,ip)=2.0*(lap+grad);
      tau(0,ip)=0.5*grad;
    }
  }
}

void AngularGrid::update_density(const arma::mat & Pa0, const arma::mat & Pb0) {
  if(!Pa0.n_elem || !Pb0.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - density matrix is empty!\n");
  }

  // Polarized calculation.
  polarized=true;

  // Update density vector
  arma::mat Pa(Pa0.submat(bf_ind,bf_ind));
  arma::mat Pb(Pb0.submat(bf_ind,bf_ind));

  Pav=Pa*bf;
  Pbv=Pb*bf;

  // Calculate density
  rho.zeros(2,grid.size());
  for(size_t ip=0;ip<grid.size();ip++) {
    double da=0.0, db=0.0;
    for(size_t j=0;j<bf.n_rows;j++) {
      da+=Pav(j,ip)*bf(j,ip);
      db+=Pbv(j,ip)*bf(j,ip);
    }
    rho(0,ip)=da;
    rho(1,ip)=db;

    double na=compute_density(Pa0,*basp,grid[ip].r);
    double nb=compute_density(Pb0,*basp,grid[ip].r);
    if(fabs(da-na)>1e-6 || fabs(db-nb)>1e-6)
      printf("Density at point % .3f % .3f % .3f: %e vs %e, %e vs %e\n",grid[ip].r.x,grid[ip].r.y,grid[ip].r.z,da,na,db,nb);
  }

  // Calculate gradient
  if(do_grad) {
    grho.zeros(6,grid.size());
    sigma.zeros(3,grid.size());
    for(size_t ip=0;ip<grid.size();ip++) {
      // Initialize
      double gax=0.0, gay=0.0, gaz=0.0;
      double gbx=0.0, gby=0.0, gbz=0.0;

      // Calculate
      for(size_t j=0;j<bf.n_rows;j++) {
	gax+=Pav(j,ip)*bf_x(j,ip);
	gay+=Pav(j,ip)*bf_y(j,ip);
	gaz+=Pav(j,ip)*bf_z(j,ip);

	gbx+=Pbv(j,ip)*bf_x(j,ip);
	gby+=Pbv(j,ip)*bf_y(j,ip);
	gbz+=Pbv(j,ip)*bf_z(j,ip);
      }

      // Store values ang put in the missing factor 2
      grho(0,ip)=2.0*gax;
      grho(1,ip)=2.0*gay;
      grho(2,ip)=2.0*gaz;
      grho(3,ip)=2.0*gbx;
      grho(4,ip)=2.0*gby;
      grho(5,ip)=2.0*gbz;

      // Compute sigma as well
      sigma(0,ip)=4.0*(gax*gax + gay*gay + gaz*gaz);
      sigma(1,ip)=4.0*(gax*gbx + gay*gby + gaz*gbz);
      sigma(2,ip)=4.0*(gbx*gbx + gby*gby + gbz*gbz);
    }
  }

  // Calculate laplacian and kinetic energy density
  if(do_lapl) {
    // Adjust size of grid
    lapl.zeros(2,grid.size());
    tau.resize(2,grid.size());

    // Update helpers
    Pav_x=Pa*bf_x;
    Pav_y=Pa*bf_y;
    Pav_z=Pa*bf_z;

    Pbv_x=Pb*bf_x;
    Pbv_y=Pb*bf_y;
    Pbv_z=Pb*bf_z;

    // Calculate values
    for(size_t ip=0;ip<grid.size();ip++) {
      // Laplacian term
      double lapa=0.0, lapb=0.0;
      // Gradient term
      double grada=0.0, gradb=0.0;
      // Calculate dot products
      for(size_t j=0;j<bf_lapl.n_rows;j++) {
	lapa+=Pav(j,ip)*bf_lapl(j,ip);
	grada+=Pav_x(j,ip)*bf_x(j,ip) + Pav_y(j,ip)*bf_y(j,ip) + Pav_z(j,ip)*bf_z(j,ip);

	lapb+=Pbv(j,ip)*bf_lapl(j,ip);
	gradb+=Pbv_x(j,ip)*bf_x(j,ip) + Pbv_y(j,ip)*bf_y(j,ip) + Pbv_z(j,ip)*bf_z(j,ip);
      }

      // Store values
      lapl(0,ip)=2.0*(lapa+grada);
      lapl(1,ip)=2.0*(lapb+gradb);
      tau(0,ip)=0.5*grada;
      tau(1,ip)=0.5*gradb;
    }
  }
}

void AngularGrid::update_density(const arma::cx_vec & C0) {
  // Update values of densitty

  if(!C0.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - coefficient vector is empty!\n");
  }

  // Polarized calculation.
  polarized=true;

  // Compute value of orbital
  arma::cx_vec C(bf_ind.n_elem);
  for(size_t i=0;i<bf_ind.n_elem;i++)
    C(i)=C0(bf_ind(i));

  arma::cx_rowvec Cv=arma::strans(C)*bf;
  // Store densities
  rho.zeros(2,grid.size());
  for(size_t ip=0;ip<grid.size();ip++)
    // Compute densities
    rho(0,ip)=std::norm(Cv(ip));

  if(do_grad) {
    grho.zeros(6,grid.size());
    sigma.zeros(3,grid.size());

    // Compute orbital gradient
    arma::cx_rowvec Cv_x=arma::strans(C)*bf_x;
    arma::cx_rowvec Cv_y=arma::strans(C)*bf_y;
    arma::cx_rowvec Cv_z=arma::strans(C)*bf_z;

    // Gradient is
    for(size_t ip=0;ip<grid.size();ip++) {
      grho(0,ip)=2.0*std::real(Cv_x(ip) * std::conj(Cv(ip)));
      grho(1,ip)=2.0*std::real(Cv_y(ip) * std::conj(Cv(ip)));
      grho(2,ip)=2.0*std::real(Cv_z(ip) * std::conj(Cv(ip)));

      // Compute values of sigma
      sigma(0,ip) =std::pow(grho(0,ip),2) + std::pow(grho(1,ip),2) + std::pow(grho(2,ip),2);
    }

    if(do_lapl) {
      // Adjust size of grid
      lapl.zeros(2,grid.size());
      tau.zeros(2,grid.size());

      // Compute orbital laplacian
      arma::cx_rowvec Cv_lapl=arma::strans(C)*bf_lapl;

      for(size_t ip=0;ip<grid.size();ip++) {
	// Laplacian term
	double lap=std::real(Cv_lapl(ip)*std::conj(Cv(ip)));
	// Gradient term
	double grad=std::norm(Cv_x(ip)) + std::norm(Cv_y(ip)) + std::norm(Cv_z(ip));

	// Laplacian is (including degeneracy factors)
	lapl(0,ip)=2.0*(lap+grad);
	// Kinetic energy density is
	tau(0,ip)=0.5*grad;
      }
    }
  }
}

void AngularGrid::get_density(std::vector<dens_list_t> & list) const {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("get_density() is supposed to be called with a non-polarized grid!\n");
  }

  for(size_t ip=0;ip<grid.size();ip++) {
    dens_list_t hlp;
    hlp.d=rho(0,ip);
    hlp.w=w(ip);
    list.push_back(hlp);
  }
}

double AngularGrid::compute_Nel() const {
  double nel=0.0;

  if(!polarized)
    for(size_t ip=0;ip<grid.size();ip++)
      nel+=w(ip)*rho(0,ip);
  else
    for(size_t ip=0;ip<grid.size();ip++)
      nel+=w(ip)*(rho(0,ip)+rho(1,ip));

  return nel;
}

void AngularGrid::print_grid() const {
  for(size_t ip=0;ip<grid.size();ip++)
    printf("%5i % f % f % f %e\n",(int) ip+1,grid[ip].r.x,grid[ip].r.y,grid[ip].r.z,grid[ip].w);
}

void AngularGrid::init_xc() {
  // Size of grid.
  const size_t N=grid.size();

  // Check allocation of arrays.
  exc.zeros(N);

  if(!polarized) {
    // Restricted case
    vxc.zeros(1,N);
    if(do_grad)
      vsigma.zeros(1,N);
    if(do_lapl) {
      vtau.zeros(1,N);
      vlapl.zeros(1,N);
    }
  } else {
    // Unrestricted case
    vxc.zeros(2,N);
    if(do_grad)
      vsigma.zeros(3,N);
    if(do_lapl) {
      vlapl.zeros(2,N);
      vtau.zeros(2,N);
    }
  }

  // Initial values
  do_gga=false;
  do_mgga=false;
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

void AngularGrid::compute_xc(int func_id, bool pot) {
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
  arma::vec exc_wrk(exc);
  arma::mat vxc_wrk(vxc);
  arma::mat vsigma_wrk(vsigma);
  arma::mat vlapl_wrk(vlapl);
  arma::mat vtau_wrk(vtau);

  exc_wrk.zeros();
  vxc_wrk.zeros();
  vsigma_wrk.zeros();
  vlapl_wrk.zeros();
  vtau_wrk.zeros();

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
  if(has_exc(func_id)) {
    if(pot) {
      if(mgga) // meta-GGA
	xc_mgga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(), lapl.memptr(), tau.memptr(), exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr(), vlapl_wrk.memptr(), vtau_wrk.memptr());
      else if(gga) // GGA
	xc_gga_exc_vxc(&func, N, rho.memptr(), sigma.memptr(), exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
      else // LDA
	xc_lda_exc_vxc(&func, N, rho.memptr(), exc_wrk.memptr(), vxc_wrk.memptr());
    } else {
      if(mgga) // meta-GGA
	xc_mgga_exc(&func, N, rho.memptr(), sigma.memptr(), lapl.memptr(), tau.memptr(), exc_wrk.memptr());
      else if(gga) // GGA
	xc_gga_exc(&func, N, rho.memptr(), sigma.memptr(), exc_wrk.memptr());
      else // LDA
	xc_lda_exc(&func, N, rho.memptr(), exc_wrk.memptr());
    }

  } else {
    if(pot) {
      if(mgga) // meta-GGA
	xc_mgga_vxc(&func, N, rho.memptr(), sigma.memptr(), lapl.memptr(), tau.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr(), vlapl_wrk.memptr(), vtau_wrk.memptr());
      else if(gga) // GGA
	xc_gga_vxc(&func, N, rho.memptr(), sigma.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
      else // LDA
	xc_lda_vxc(&func, N, rho.memptr(), vxc_wrk.memptr());
    }
  }

  // Sum to total arrays containing both exchange and correlation
  exc+=exc_wrk;
  if(pot) {
    vxc+=vxc_wrk;
    vsigma+=vsigma_wrk;
    vlapl+=vlapl_wrk;
    vtau+=vtau_wrk;
  }

  // Free functional
  xc_func_end(&func);
}

void AngularGrid::init_VV10(double b, double C, bool pot) {
  if(!do_grad)
    throw std::runtime_error("Invalid do_grad setting for VV10!\n");
  if(do_lapl)
    throw std::runtime_error("Invalid do_lapl setting for VV10!\n");
  do_gga=true;
  do_mgga=false;
  VV10_thr=1e-8;

  if(rho.size() != grid.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Grid size is " << grid.size() << " but there are only " << rho.size() << " density values!\n";
    throw std::runtime_error(oss.str());
  }
  if(sigma.size() != grid.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Grid size is " << grid.size() << " but there are only " << sigma.size() << " reduced gradient values!\n";
    throw std::runtime_error(oss.str());
  }
  if(b <= 0.0 || C <= 0.0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "VV10 parameters b = " << b << ", C = " << C << " are not valid.\n";
    throw std::runtime_error(oss.str());
  }

  if(pot) {
    // Plug in the constant energy density
    double beta=1.0/32.0 * std::pow(3.0/(b*b), 3.0/4.0);
    for(size_t i=0;i<grid.size();i++) {
      exc(i)+=beta;
      vxc(0,i)+=beta;
    }
  }
}

void VV10_Kernel(const arma::mat & xc, const arma::mat & nl, arma::mat & ret) {
  // Input arrays contain grid[i].r, omega0(i), kappa(i) (and grid[i].w, rho[i] for nl)
  // Return array contains: nPhi, U, and W

  if(xc.n_cols !=5) {
    ERROR_INFO();
    throw std::runtime_error("xc matrix has the wrong size.\n");
  }
  if(nl.n_cols !=7) {
    ERROR_INFO();
    throw std::runtime_error("nl matrix has the wrong size.\n");
  }
  if(ret.n_rows != xc.n_rows || ret.n_cols != 3) {
    throw std::runtime_error("Error - invalid size output array!\n");
  }

  // Loop
  for(size_t i=0;i<xc.n_rows;i++) {
    double nPhi=0.0, U=0.0, W=0.0;

    for(size_t j=0;j<nl.n_rows;j++) {
      // Distance between the grid points
      double dx=xc(i,0)-nl(j,0);
      double dy=xc(i,1)-nl(j,1);
      double dz=xc(i,2)-nl(j,2);
      double Rsq=dx*dx + dy*dy + dz*dz;

      // g factors
      double gi=xc(i,3)*Rsq + xc(i,4);
      double gj=nl(j,3)*Rsq + nl(j,4);
      // Sum of the factors
      double gs=gi+gj;
      // Reciprocal sum
      double rgis=1.0/gi + 1.0/gs;

      // Integral kernel
      double Phi = - 3.0 / ( 2.0 * gi * gj * gs);
      // Absorb grid point weight and density into kernel
      Phi *= nl(j,5) * nl(j,6);

      // Increment nPhi
      nPhi += Phi;
      // Increment U
      U    -= Phi * rgis;
      // Increment W
      W    -= Phi * rgis * Rsq;
    }

    // Store output
    ret(i,0)+=nPhi;
    ret(i,1)+=U;
    ret(i,2)+=W;
  }
}

void VV10_Kernel_F(const arma::mat & xc, const arma::mat & nl, arma::mat & ret) {
  // Input arrays contain grid[i].r, omega0(i), kappa(i) (and grid[i].w, rho[i] for nl)
  // Return array contains: nPhi, U, W, and fx, fy, fz

  if(xc.n_cols !=5) {
    ERROR_INFO();
    throw std::runtime_error("xc matrix has the wrong size.\n");
  }
  if(nl.n_cols !=7) {
    ERROR_INFO();
    throw std::runtime_error("nl matrix has the wrong size.\n");
  }
  if(ret.n_rows != xc.n_rows || ret.n_cols != 6) {
    throw std::runtime_error("Error - invalid size output array!\n");
  }

  // Loop
  for(size_t i=0;i<xc.n_rows;i++) {
    double nPhi=0.0, U=0.0, W=0.0;
    double fpx=0.0, fpy=0.0, fpz=0.0;

    for(size_t j=0;j<nl.n_rows;j++) {
      // Distance between the grid points
      double dx=xc(i,0)-nl(j,0);
      double dy=xc(i,1)-nl(j,1);
      double dz=xc(i,2)-nl(j,2);
      double Rsq=dx*dx + dy*dy + dz*dz;

      // g factors
      double gi=xc(i,3)*Rsq + xc(i,4);
      double gj=nl(j,3)*Rsq + nl(j,4);
      // Sum of the factors
      double gs=gi+gj;
      // Reciprocal sum
      double rgis=1.0/gi + 1.0/gs;

      // Integral kernel
      double Phi = - 3.0 / ( 2.0 * gi * gj * gs);
      // Absorb grid point weight and density into kernel
      Phi *= nl(j,5) * nl(j,6);

      // Increment nPhi
      nPhi += Phi;
      // Increment U
      U    -= Phi * rgis;
      // Increment W
      W    -= Phi * rgis * Rsq;

      // Q factor
      double Q = -2.0 * Phi * (xc(i,3)/gi + nl(j,3)/gj + (xc(i,3)+nl(j,3))/gs );
      // Increment force
      fpx += Q * dx;
      fpy += Q * dy;
      fpz += Q * dz;
    }

    // Store output
    ret(i,0)+=nPhi;
    ret(i,1)+=U;
    ret(i,2)+=W;
    ret(i,3)+=fpx;
    ret(i,4)+=fpy;
    ret(i,5)+=fpz;
  }
}

void AngularGrid::collect_VV10(arma::mat & data, std::vector<size_t> & idx, double b, double C, bool nl) const {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("collect_VV10 can only be called in a non-polarized grid!\n");
  }

  // Create list of points with significant densities
  idx.clear();
  for(size_t i=0;i<grid.size();i++)
    if(rho(0,i) >= VV10_thr)
      idx.push_back(i);

  // Create input datas
  if(nl)
    data.zeros(idx.size(),7);
  else
    data.zeros(idx.size(),5);

  // Constants for omega and kappa
  const double oc=4.0*M_PI/3.0;
  const double kc=(3.0*M_PI*b)/2.0*std::pow(9.0*M_PI,-1.0/6.0);
  for(size_t ii=0;ii<idx.size();ii++) {
    size_t i=idx[ii];
    data(ii,0)=grid[i].r.x;
    data(ii,1)=grid[i].r.y;
    data(ii,2)=grid[i].r.z;
    // omega_0[i]
    data(ii,3)=sqrt(C * std::pow(sigma(0,i)/(rho(0,i)*rho(0,i)),2) + oc*rho(0,i));
    // kappa[i]
    data(ii,4)=kc * cbrt(sqrt(rho(0,i)));
  }
  if(nl) {
    for(size_t ii=0;ii<idx.size();ii++) {
      size_t i=idx[ii];
      data(ii,5)=w(i);
      data(ii,6)=rho(0,i);
    }
  }
}

void AngularGrid::compute_VV10(const std::vector<arma::mat> & nldata, double b, double C) {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("compute_VV10 should be run in non-polarized mode!\n");
  }

  Timer t;

  // Create list of points with significant densities and input data
  std::vector<size_t> idx;
  arma::mat xc;
  collect_VV10(xc, idx, b, C, false);

  /*
  for(size_t i=0;i<xc.n_rows;i++) {
    printf("% .3f % .3f % .3f",grid[i].r.x,grid[i].r.y,grid[i].r.z);
    for(size_t j=0;j<xc.n_cols;j++)
      printf(" % e",xc(i,j));
    printf("\n");
  }
  */

  // Calculate integral kernel
  VV10_arr.zeros(xc.n_rows,3);
  for(size_t i=0;i<nldata.size();i++)
    VV10_Kernel(xc,nldata[i],VV10_arr);

  // Collect data
  for(size_t ii=0;ii<idx.size();ii++) {
    // Index of grid point is
    size_t i=idx[ii];

    // Increment the energy density
    exc[i] += 0.5 * VV10_arr(ii,0);

    // Increment LDA and GGA parts of potential.
    double ri=rho(0,i);
    double ri4=std::pow(ri,4);
    double si=sigma(0,i);
    double w0=xc(ii,3);
    double dkdn  = xc(ii,4)/(6.0*ri); // d kappa / d n
    double dw0ds = C*si / ( w0 * ri4); // d omega0 / d sigma
    double dw0dn = 2.0/w0 * ( M_PI/3.0 - C*si*si / (ri*ri4)); // d omega0 / d n
    vxc(0,i) += VV10_arr(ii,0) + ri *( dkdn * VV10_arr(ii,1) + dw0dn * VV10_arr(ii,2));
    vsigma(0,i) += ri * dw0ds * VV10_arr(ii,2);
  }

  /*
  for(size_t i=0;i<VV10_arr.n_rows;i++) {
    printf("% .3f % .3f % .3f",grid[i].r.x,grid[i].r.y,grid[i].r.z);
    for(size_t j=0;j<VV10_arr.n_cols;j++)
      printf(" % e",VV10_arr(i,j));
    printf("\n");
  }
  */
}

arma::vec AngularGrid::compute_VV10_F(const std::vector<arma::mat> & nldata, const std::vector<angshell_t> & nlgrids, double b, double C) {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("compute_VV10_F should be run in non-polarized mode!\n");
  }

  // Create list of points with significant densities and input data
  std::vector<size_t> idx;
  arma::mat xc;
  collect_VV10(xc, idx, b, C, false);

  // Calculate integral kernels
  VV10_arr.zeros(xc.n_rows,6);
  for(size_t i=0;i<nldata.size();i++)
    if(nlgrids[i].atind==info.atind) {
      // No Q contribution!
      //VV10_Kernel(xc,nldata[i],VV10_arr.cols(0,2));

      arma::mat Kat(xc.n_rows,3);
      Kat.zeros();
      VV10_Kernel(xc,nldata[i],Kat);
      VV10_arr.cols(0,2)+=Kat;
    } else
      // Full contribution
      VV10_Kernel_F(xc,nldata[i],VV10_arr);

  // Evaluate force contribution
  double fx=0.0, fy=0.0, fz=0.0;

  for(size_t ii=0;ii<idx.size();ii++) {
    // Index of grid point is
    size_t i=idx[ii];

    // Increment the energy density
    exc[i] += 0.5 * VV10_arr(ii,0);

    // Increment LDA and GGA parts of potential.
    double ri=rho(0,i);
    double ri4=std::pow(ri,4);
    double si=sigma(0,i);
    double w0=xc(ii,3);
    double dkdn  = xc(ii,4)/(6.0*ri); // d kappa / d n
    double dw0ds = C*si / ( w0 * ri4); // d omega0 / d sigma
    double dw0dn = 2.0/w0 * ( M_PI/3.0 - C*si*si / (ri*ri4)); // d omega0 / d n
    vxc(0,i) += VV10_arr(ii,0) + ri *( dkdn * VV10_arr(ii,1) + dw0dn * VV10_arr(ii,2));
    vsigma(0,i) += ri * dw0ds * VV10_arr(ii,2);

    // Increment total force
    fx += grid[i].w*rho(0,i)*VV10_arr(ii,3);
    fy += grid[i].w*rho(0,i)*VV10_arr(ii,4);
    fz += grid[i].w*rho(0,i)*VV10_arr(ii,5);
  }

  arma::vec F(3);
  F(0)=fx;
  F(1)=fy;
  F(2)=fz;

  return F;
}

void AngularGrid::print_density(FILE *f) const {
  // Loop over grid points
  for(size_t i=0;i<grid.size();i++) {
    // Get data in point
    libxc_dens_t d=get_dens(i);

    // Print out data
    fprintf(f,"% .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e\n",d.rhoa,d.rhob,d.sigmaaa,d.sigmaab,d.sigmabb,d.lapla,d.laplb,d.taua,d.taub);
  }
}

void AngularGrid::print_potential(int func_id, FILE *f) const {
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

void AngularGrid::get_weights() {
  if(!grid.size())
    return;
  w.zeros(grid.size());
  for(size_t pi=0;pi<grid.size();pi++)
    w(pi)=grid[pi].w;
}

libxc_dens_t AngularGrid::get_dens(size_t idx) const {
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

  if(polarized) {
    ret.rhoa=rho(0,idx);
    ret.rhob=rho(1,idx);
  } else {
    ret.rhoa=ret.rhob=rho(0,idx)/2.0;
  }

  if(do_gga) {
    if(polarized) {
      ret.sigmaaa=sigma(0,idx);
      ret.sigmaab=sigma(1,idx);
      ret.sigmabb=sigma(2,idx);
    } else {
      ret.sigmaaa=ret.sigmaab=ret.sigmabb=sigma(0,idx)/4.0;
    }
  }

  if(do_mgga) {
    if(polarized) {
      ret.lapla=lapl(0,idx);
      ret.laplb=lapl(1,idx);
      ret.taua=tau(0,idx);
      ret.taub=tau(1,idx);
    } else {
      ret.lapla=ret.laplb=lapl(0,idx)/2.0;
      ret.taua=ret.taub=tau(0,idx)/2.0;
    }
  }

  return ret;
}

libxc_debug_t AngularGrid::get_data(size_t idx) const {
  libxc_debug_t d;
  d.dens=get_dens(idx);
  d.pot=get_pot(idx);
  return d;
}

libxc_pot_t AngularGrid::get_pot(size_t idx) const {
  libxc_pot_t ret;

  // Alpha and beta density
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

  if(polarized) {
    ret.vrhoa=vxc(0,idx);
    ret.vrhob=vxc(1,idx);
  } else {
    ret.vrhoa=ret.vrhob=vxc(0,idx)/2.0;
  }

  if(do_gga) {
    if(polarized) {
      ret.vsigmaaa=vsigma(0,idx);
      ret.vsigmaab=vsigma(1,idx);
      ret.vsigmabb=vsigma(2,idx);
    } else {
      ret.vsigmaaa=ret.vsigmaab=ret.vsigmabb=vsigma(0,idx)/4.0;
    }
  }

  if(do_mgga) {
    if(polarized) {
      ret.vlapla=vlapl(0,idx);
      ret.vlaplb=vlapl(1,idx);
      ret.vtaua=vtau(0,idx);
      ret.vtaub=vtau(1,idx);
    } else {
      ret.vlapla=ret.vlaplb=vlapl(0,idx)/2.0;
      ret.vtaua=ret.vtaub=vtau(0,idx)/2.0;
    }
  }

  return ret;
}

double AngularGrid::eval_Exc() const {
  double Exc=0.0;

  if(!polarized)
    for(size_t i=0;i<grid.size();i++)
      Exc+=w(i)*exc(i)*rho(0,i);
  else
    for(size_t i=0;i<grid.size();i++)
      Exc+=w(i)*exc(i)*(rho(0,i)+rho(1,i));

  return Exc;
}

void AngularGrid::eval_overlap(arma::mat & So) const {
  // Calculate in subspace
  arma::mat S(bf_ind.n_elem,bf_ind.n_elem);
  S.zeros();
  increment_lda<double>(S,w,bf);
  // Increment
  So.submat(bf_ind,bf_ind)+=S;
}

void AngularGrid::eval_diag_overlap(arma::vec & S) const {
  for(size_t ip=0;ip<grid.size();ip++) {
    for(size_t j=0;j<bf.n_rows;j++)
      S(bf_potind(j))+=w(ip)*bf(j,ip)*bf(j,ip);
  }
}

void AngularGrid::eval_Fxc(arma::mat & Ho) const {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
  }

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());

  // Work matrix
  arma::mat H(bf_ind.n_elem,bf_ind.n_elem);
  H.zeros();

  {
    // LDA potential
    arma::rowvec vrho(vxc.row(0));
    // Multiply weights into potential
    vrho%=w;
    // Increment matrix
    increment_lda<double>(H,vrho,bf,screen);
  }

  if(do_gga) {
    // Get vsigma
    arma::rowvec vs(vsigma.row(0));
    // Get grad rho
    arma::uvec idx(arma::linspace<arma::uvec>(0,2,3));
    arma::mat gr(arma::trans(grho.rows(idx)));
    // Multiply grad rho by vsigma and the weights
    for(size_t i=0;i<gr.n_rows;i++)
      for(size_t ic=0;ic<gr.n_cols;ic++)
	gr(i,ic)=2.0*w(i)*vs(i)*gr(i,ic);
    // Increment matrix
    increment_gga<double>(H,gr,bf,bf_x,bf_y,bf_z,screen);
  }

  if(do_mgga) {
    // Get vtau and vlapl
    arma::rowvec vt(vtau.row(0));
    arma::rowvec vl(vlapl.row(0));
    // Scale both with weights
    vt%=w;
    vl%=w;

    // Evaluate kinetic contribution
    increment_mgga_kin<double>(H,0.5*vt + 2.0*vl,bf_x,bf_y,bf_z,screen);

    // Evaluate laplacian contribution. Get function laplacian
    increment_mgga_lapl<double>(H,vl,bf,bf_lapl,screen);
  }

  Ho(bf_ind,bf_ind)+=H;
}

void AngularGrid::eval_diag_Fxc(arma::vec & H) const {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
  }

  {
    // LDA potential
    arma::rowvec vrho(vxc.row(0));
    // Multiply weights into potential
    vrho%=w;
    // Increment matrix
    for(size_t ip=0;ip<grid.size();ip++)
      for(size_t j=0;j<bf.n_rows;j++)
	H(bf_potind(j))+=vrho(ip)*bf(j,ip)*bf(j,ip);
  }

  if(do_gga) {
    // Get vsigma
    arma::rowvec vs(vsigma.row(0));
    // Get grad rho
    arma::uvec idx(arma::linspace<arma::uvec>(0,2,3));
    arma::mat gr(arma::trans(grho.rows(idx)));
    // Multiply grad rho by vsigma and the weights
    for(size_t i=0;i<gr.n_rows;i++)
      for(size_t ic=0;ic<gr.n_cols;ic++)
	gr(i,ic)=2.0*w(i)*vs(i)*gr(i,ic);
    // Increment matrix
    for(size_t ip=0;ip<grid.size();ip++)
      for(size_t j=0;j<bf.n_rows;j++)
	H(bf_potind(j))+=2.0 * (gr(ip,0)*bf_x(j,ip) + gr(ip,1)*bf_y(j,ip) + gr(ip,2)*bf_z(j,ip)) * bf(j,ip);

    if(do_mgga) {
      // Get vtau and vlapl
      arma::rowvec vt(vtau.row(0));
      arma::rowvec vl(vlapl.row(0));
      // Scale both with weights
      vt%=w;
      vl%=w;

      // Evaluate kinetic contribution
      for(size_t ip=0;ip<grid.size();ip++)
	for(size_t j=0;j<bf.n_rows;j++)
	  H(bf_potind(j))+=(0.5*vt(ip)+2.0*vl(ip))*(bf_x(j,ip)*bf_x(j,ip) + bf_y(j,ip)*bf_y(j,ip) + bf_z(j,ip)*bf_z(j,ip));

      // Evaluate laplacian contribution.
      for(size_t ip=0;ip<grid.size();ip++)
	for(size_t j=0;j<bf.n_rows;j++)
	  H(bf_potind(j))+=2.0*vl(ip)*bf(j,ip)*bf_lapl(j,ip);
    }
  }
}

void AngularGrid::eval_Fxc(arma::mat & Hao, arma::mat & Hbo, bool beta) const {
  if(!polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
  }

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());

  arma::mat Ha(bf_ind.n_elem,bf_ind.n_elem);
  Ha.zeros();
  arma::mat Hb;
  if(beta)
    Hb.zeros(bf_ind.n_elem,bf_ind.n_elem);

  {
    // LDA potential
    arma::rowvec vrhoa(vxc.row(0));
    // Multiply weights into potential
    vrhoa%=w;
    // Increment matrix
    increment_lda<double>(Ha,vrhoa,bf,screen);

    if(beta) {
      arma::rowvec vrhob(vxc.row(1));
      vrhob%=w;
      increment_lda<double>(Hb,vrhob,bf,screen);
    }
  }

  if(do_gga) {
    // Get vsigma
    arma::rowvec vs_aa(vsigma.row(0));
    arma::rowvec vs_ab(vsigma.row(1));

    // Get grad rho
    arma::uvec idxa(arma::linspace<arma::uvec>(0,2,3));
    arma::uvec idxb(arma::linspace<arma::uvec>(3,5,3));
    arma::mat gr_a0(arma::trans(grho.rows(idxa)));
    arma::mat gr_b0(arma::trans(grho.rows(idxb)));

    // Multiply grad rho by vsigma and the weights
    arma::mat gr_a(gr_a0);
    for(size_t i=0;i<gr_a.n_rows;i++)
      for(size_t ic=0;ic<gr_a.n_cols;ic++)
	gr_a(i,ic)=w(i)*(2.0*vs_aa(i)*gr_a0(i,ic) + vs_ab(i)*gr_b0(i,ic));
    // Increment matrix
    increment_gga<double>(Ha,gr_a,bf,bf_x,bf_y,bf_z,screen);

    if(beta) {
      arma::rowvec vs_bb(vsigma.row(2));
      arma::mat gr_b(gr_b0);
      for(size_t i=0;i<gr_b.n_rows;i++)
	for(size_t ic=0;ic<gr_b.n_cols;ic++)
	  gr_b(i,ic)=w(i)*(2.0*vs_bb(i)*gr_b0(i,ic) + vs_ab(i)*gr_a0(i,ic));
      increment_gga<double>(Hb,gr_b,bf,bf_x,bf_y,bf_z,screen);
    }
  }

  if(do_mgga) {
    // Get vtau and vlapl
    arma::rowvec vt_a(vtau.row(0));
    arma::rowvec vl_a(vlapl.row(0));

    // Scale both with weights
    vt_a%=w;
    vl_a%=w;

    // Evaluate kinetic contribution
    increment_mgga_kin<double>(Ha,0.5*vt_a + 2.0*vl_a,bf_x,bf_y,bf_z,screen);

    // Evaluate laplacian contribution. Get function laplacian
    increment_mgga_lapl<double>(Ha,vl_a,bf,bf_lapl,screen);

    if(beta) {
      arma::rowvec vt_b(vtau.row(1));
      arma::rowvec vl_b(vlapl.row(1));
      vt_b%=w;
      vl_b%=w;
      increment_mgga_kin<double>(Hb,0.5*vt_b + 2.0*vl_b,bf_x,bf_y,bf_z,screen);
      increment_mgga_lapl<double>(Hb,vl_b,bf,bf_lapl,screen);
    }
  }

  Hao(bf_ind,bf_ind)+=Ha;
  if(beta)
    Hbo(bf_ind,bf_ind)+=Hb;
}

void AngularGrid::eval_diag_Fxc(arma::vec & Ha, arma::vec & Hb) const {
  if(!polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
  }

  {
    // LDA potential
    arma::rowvec vrhoa(vxc.row(0));
    // Multiply weights into potential
    vrhoa%=w;
    arma::rowvec vrhob(vxc.row(1));
    vrhob%=w;
    // Increment matrix
    for(size_t ip=0;ip<grid.size();ip++)
      for(size_t j=0;j<bf.n_rows;j++) {
	Ha(bf_potind(j))+=vrhoa(ip)*bf(j,ip)*bf(j,ip);
	Hb(bf_potind(j))+=vrhob(ip)*bf(j,ip)*bf(j,ip);
      }
  }

  if(do_gga) {
    // Get vsigma
    arma::rowvec vs_aa(vsigma.row(0));
    arma::rowvec vs_ab(vsigma.row(1));
    arma::rowvec vs_bb(vsigma.row(2));
    // Get grad rho
    arma::uvec idxa(arma::linspace<arma::uvec>(0,2,3));
    arma::uvec idxb(arma::linspace<arma::uvec>(3,5,3));
    arma::mat gra0(arma::trans(grho.rows(idxa)));
    arma::mat grb0(arma::trans(grho.rows(idxb)));

    // Multiply grad rho by vsigma and the weights
    arma::mat gra(gra0);
    for(size_t i=0;i<gra.n_rows;i++)
      for(size_t ic=0;ic<gra.n_cols;ic++)
	gra(i,ic)=w(i)*(2.0*vs_aa(i)*gra0(i,ic)+vs_ab(i)*grb0(i,ic));
    // Increment matrix
    for(size_t ip=0;ip<grid.size();ip++)
      for(size_t j=0;j<bf.n_rows;j++)
	Ha(bf_potind(j))+=2.0 * (gra(ip,0)*bf_x(j,ip) + gra(ip,1)*bf_y(j,ip) + gra(ip,2)*bf_z(j,ip)) * bf(j,ip);

    arma::mat grb(grb0);
    for(size_t i=0;i<grb.n_rows;i++)
      for(size_t ic=0;ic<grb.n_cols;ic++)
	grb(i,ic)=w(i)*(2.0*vs_bb(i)*grb0(i,ic)+vs_ab(i)*gra0(i,ic));
    for(size_t ip=0;ip<grid.size();ip++)
      for(size_t j=0;j<bf.n_rows;j++)
	Hb(bf_potind(j))+=2.0 * (grb(ip,0)*bf_x(j,ip) + grb(ip,1)*bf_y(j,ip) + grb(ip,2)*bf_z(j,ip)) * bf(j,ip);

    if(do_mgga) {
      // Get vtau and vlapl
      arma::rowvec vta(vtau.row(0));
      arma::rowvec vla(vlapl.row(0));
      arma::rowvec vtb(vtau.row(1));
      arma::rowvec vlb(vlapl.row(1));
      // Scale both with weights
      vta%=w;
      vla%=w;
      vtb%=w;
      vlb%=w;

      // Evaluate kinetic contribution
      for(size_t ip=0;ip<grid.size();ip++)
	for(size_t j=0;j<bf.n_rows;j++) {
	  Ha(bf_potind(j))+=(0.5*vta(ip)+2.0*vla(ip))*(bf_x(j,ip)*bf_x(j,ip) + bf_y(j,ip)*bf_y(j,ip) + bf_z(j,ip)*bf_z(j,ip));
	  Hb(bf_potind(j))+=(0.5*vtb(ip)+2.0*vlb(ip))*(bf_x(j,ip)*bf_x(j,ip) + bf_y(j,ip)*bf_y(j,ip) + bf_z(j,ip)*bf_z(j,ip));
	}

      // Evaluate laplacian contribution.
      for(size_t ip=0;ip<grid.size();ip++)
	for(size_t j=0;j<bf.n_rows;j++) {
	  Ha(bf_potind(j))+=2.0*vla(ip)*bf(j,ip)*bf_lapl(j,ip);
	  Hb(bf_potind(j))+=2.0*vlb(ip)*bf(j,ip)*bf_lapl(j,ip);
	}
    }
  }
}

arma::vec AngularGrid::eval_force(const arma::mat & Pa, const arma::mat & Pb) const {
  if(!polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted force with restricted density.\n");
  }

  arma::rowvec f(3*basp->get_Nnuc());
  f.zeros();

  // Get functions in basis set
  std::vector<GaussianShell> gshells=basp->get_shells();

  // Loop over nuclei
  for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {

    // LDA part. Loop over grid
    for(size_t ip=0;ip<grid.size();ip++) {

      // Grad rho in current point
      arma::rowvec gradrhoa(3);
      gradrhoa.zeros();
      arma::rowvec gradrhob(3);
      gradrhob.zeros();

      // Loop over shells on the nucleus
      for(size_t iish=0;iish<shells.size();iish++) {
	if(basp->get_shell_center_ind(shells[iish]) != inuc)
	  continue;
	size_t ish=shells[iish];

	// First function on shell is
	size_t mu0=gshells[ish].get_first_ind();

	// Evaluate the gradient in the current grid point
	arma::mat grad=gshells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	// Increment sum. Loop over mu
	for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	  // Function index is
	  size_t mu=mu0+imu;

	  // Loop over the functions on the grid point
	  for(size_t inu=0;inu<bf_ind.size();inu++) {
	    // Get index of function
	    size_t nu(bf_ind(inu));

	    gradrhoa+=Pa(mu,nu)*grad.row(imu)*bf(inu,ip);
	    gradrhob+=Pb(mu,nu)*grad.row(imu)*bf(inu,ip);
	  }
	}
      }
      // Plug in the factor 2 to get the total gradient
      gradrhoa*=2.0;
      gradrhob*=2.0;

      // Increment total force
      f.subvec(3*inuc,3*inuc+2)+=w(ip)*(vxc(0,ip)*gradrhoa + vxc(1,ip)*gradrhob);
    }
  }

  // GGA part
  if(do_gga) {

    // Loop over nuclei
    for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {
      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// X in the current grid point
	arma::mat Xa(3,3);
	Xa.zeros();

	arma::mat Xb(3,3);
	Xb.zeros();

	// Loop over shells on the nucleus
	for(size_t iish=0;iish<shells.size();iish++) {
	  if(basp->get_shell_center_ind(shells[iish]) != inuc)
	    continue;
	  size_t ish=shells[iish];

	  // First function on shell is
	  size_t mu0=gshells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=gshells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=gshells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    for(size_t inu=0;inu<bf_ind.size();inu++) {
	      // Get index of function
	      size_t nu=bf_ind(inu);

	      double glist[3];
	      glist[0]=bf_x(inu,ip);
	      glist[1]=bf_y(inu,ip);
	      glist[2]=bf_z(inu,ip);

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  Xa(ic,jc)+=Pa(mu,nu)*(bf(inu,ip)*hess(imu,3*ic+jc) + glist[jc]*grad(imu,ic));
	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  Xb(ic,jc)+=Pb(mu,nu)*(bf(inu,ip)*hess(imu,3*ic+jc) + glist[jc]*grad(imu,ic));
	    }
	  }
	}
	Xa*=2.0;
	Xb*=2.0;

	// The xc "vector" is
	arma::vec xca(3);
	for(int ic=0;ic<3;ic++)
	  xca(ic)=2.0*vsigma(0,ip)*grho(ic,ip) + vsigma(1,ip)*grho(ic+3,ip);
	arma::vec xcb(3);
	for(int ic=0;ic<3;ic++)
	  xcb(ic)=2.0*vsigma(2,ip)*grho(ic+3,ip) + vsigma(1,ip)*grho(ic,ip);

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*arma::trans(Xa*xca+Xb*xcb);
      }
    }
  }

  // meta-GGA part
  if(do_mgga) {

    // First part. Loop over nuclei
    for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {
      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// y in the current grid point
	arma::vec ya(3);
	ya.zeros();
	arma::vec yb(3);
	yb.zeros();

	// Loop over shells on the nucleus
	for(size_t iish=0;iish<shells.size();iish++) {
	  if(basp->get_shell_center_ind(shells[iish]) != inuc)
	    continue;
	  size_t ish=shells[iish];

	  // First function on shell is
	  size_t mu0=gshells[ish].get_first_ind();

	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=gshells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    for(size_t inu=0;inu<bf_ind.size();inu++) {
	      // Get index of function
	      size_t nu(bf_ind(inu));

	      // Collect grad nu
	      arma::vec gnu(3);
	      gnu(0)=bf_x(inu,ip);
	      gnu(1)=bf_y(inu,ip);
	      gnu(2)=bf_z(inu,ip);

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++) {
		  ya(ic)+=Pa(mu,nu)*hess(imu,3*ic+jc)*gnu(jc);
		  yb(ic)+=Pb(mu,nu)*hess(imu,3*ic+jc)*gnu(jc);
		}
	    }
	  }
	}

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=w(ip)*arma::trans((vtau(0,ip)+2.0*vlapl(0,ip))*ya + (vtau(1,ip)+2.0*vlapl(1,ip))*yb);
      }
    }

    // Second part. Loop over nuclei
    for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {
      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// z in the current grid point
	arma::rowvec za(3);
	za.zeros();
	arma::rowvec zb(3);
	zb.zeros();

	// Loop over shells on the nucleus
	for(size_t iish=0;iish<shells.size();iish++) {
	  if(basp->get_shell_center_ind(shells[iish]) != inuc)
	    continue;
	  size_t ish=shells[iish];

	  // First function on shell is
	  size_t mu0=gshells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=gshells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the laplacian of the gradient in the current grid point
	  arma::mat laplgrad=gshells[ish].eval_laplgrad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    for(size_t inu=0;inu<bf_ind.n_elem;inu++) {
	      // Get index of function
	      size_t nu(bf_ind(inu));

	      za+=Pa(mu,nu)*(bf_lapl(inu,ip)*grad.row(imu)+bf(inu,ip)*laplgrad.row(imu));
	      zb+=Pb(mu,nu)*(bf_lapl(inu,ip)*grad.row(imu)+bf(inu,ip)*laplgrad.row(imu));
	    }
	  }
	}
	za*=2.0;
	zb*=2.0;

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=w(ip)*(vlapl(0,ip)*za + vlapl(1,ip)*zb);
      }
    }
  }


  return arma::trans(f);
}


arma::vec AngularGrid::eval_force(const arma::mat & P) const {
  if(polarized) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted force with unrestricted density.\n");
  }

  // Initialize force
  arma::rowvec f(3*basp->get_Nnuc());
  f.zeros();

  // Get functions centered on the atom
  std::vector<GaussianShell> gshells=basp->get_shells();

  // Loop over nuclei
  for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {
    // LDA part. Loop over grid
    for(size_t ip=0;ip<grid.size();ip++) {

      // Grad rho in current point
      arma::rowvec gradrho(3);
      gradrho.zeros();

      // Loop over shells on the nucleus
      for(size_t iish=0;iish<shells.size();iish++) {
	if(basp->get_shell_center_ind(shells[iish]) != inuc)
	  continue;
	size_t ish=shells[iish];

	// First function on shell is
	size_t mu0=gshells[ish].get_first_ind();

	// Evaluate the gradient in the current grid point
	arma::mat grad=gshells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	// Increment sum. Loop over mu
	for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	  // Function index is
	  size_t mu=mu0+imu;

	  for(size_t inu=0;inu<bf_ind.n_elem;inu++) {
	    // Get index of function
	    size_t nu(bf_ind(inu));

	    gradrho+=P(mu,nu)*grad.row(imu)*bf(inu,ip);
	  }
	}
      }
      // Plug in the factor 2 to get the total gradient
      gradrho*=2.0;

      // Increment total force
      f.subvec(3*inuc,3*inuc+2)+=w(ip)*vxc(0,ip)*gradrho;
    }
  }

  // GGA part
  if(do_gga) {
    // Loop over nuclei
    for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {

      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// X in the current grid point
	arma::mat X(3,3);
	X.zeros();

	// Loop over shells on the nucleus
	for(size_t iish=0;iish<shells.size();iish++) {
	  if(basp->get_shell_center_ind(shells[iish]) != inuc)
	    continue;
	  size_t ish=shells[iish];

	  // First function on shell is
	  size_t mu0=gshells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=gshells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=gshells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    for(size_t inu=0;inu<bf_ind.n_elem;inu++) {
	      // Get index of function
	      size_t nu(bf_ind(inu));
	      double glist[3];
	      glist[0]=bf_x(inu,ip);
	      glist[1]=bf_y(inu,ip);
	      glist[2]=bf_z(inu,ip);

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  X(ic,jc)+=P(mu,nu)*(bf(inu,ip)*hess(imu,3*ic+jc) + glist[jc]*grad(imu,ic));
	    }
	  }
	}
	X*=2.0;

	// The xc "vector" is
	arma::vec xc(3);
	for(int ic=0;ic<3;ic++)
	  xc(ic)=2.0*vsigma(ip)*grho(ic,ip);

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=w(ip)*arma::trans(X*xc);
      }
    }
  }

  // meta-GGA part
  if(do_mgga) {

    // First part. Loop over nuclei
    for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {
      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// y in the current grid point
	arma::vec y(3);
	y.zeros();

	// Loop over shells on the nucleus
	for(size_t iish=0;iish<shells.size();iish++) {
	  if(basp->get_shell_center_ind(shells[iish]) != inuc)
	    continue;
	  size_t ish=shells[iish];

	  // First function on shell is
	  size_t mu0=gshells[ish].get_first_ind();

	  // Evaluate the Hessian in the current grid point
	  arma::mat hess=gshells[ish].eval_hess(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    for(size_t inu=0;inu<bf_ind.n_elem;inu++) {
	      // Get index of function
	      size_t nu(bf_ind(inu));

	      // Collect grad nu
	      arma::vec gnu(3);
	      gnu(0)=bf_x(inu,ip);
	      gnu(1)=bf_y(inu,ip);
	      gnu(2)=bf_z(inu,ip);

	      for(int ic=0;ic<3;ic++)
		for(int jc=0;jc<3;jc++)
		  y(ic)+=P(mu,nu)*hess(imu,3*ic+jc)*gnu(jc);
	    }
	  }
	}

	// Increment total force
	f.subvec(3*inuc,3*inuc+2)+=grid[ip].w*(vtau(0,ip)+2.0*vlapl(0,ip))*arma::trans(y);
      }
    }

    // Second part. Loop over nuclei
    for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {
      // Loop over the grid
      for(size_t ip=0;ip<grid.size();ip++) {

	// z in the current grid point
	arma::rowvec z(3);
	z.zeros();

	// Loop over shells on the nucleus
	for(size_t iish=0;iish<shells.size();iish++) {
	  if(basp->get_shell_center_ind(shells[iish]) != inuc)
	    continue;
	  size_t ish=shells[iish];

	  // First function on shell is
	  size_t mu0=gshells[ish].get_first_ind();

	  // Evaluate the gradient in the current grid point
	  arma::mat grad=gshells[ish].eval_grad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	  // Evaluate the laplacian of the gradient in the current grid point
	  arma::mat laplgrad=gshells[ish].eval_laplgrad(grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);

	  // Increment sum. Loop over mu
	  for(size_t imu=0;imu<gshells[ish].get_Nbf();imu++) {
	    // Function index is
	    size_t mu=mu0+imu;

	    for(size_t inu=0;inu<bf_ind.n_elem;inu++) {
	      // Get index of function
	      size_t nu(bf_ind(inu));

	      z+=P(mu,nu)*(bf_lapl(inu,ip)*grad.row(imu)+bf(inu,ip)*laplgrad.row(imu));
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

void AngularGrid::check_grad_lapl(int x_func, int c_func) {
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

void AngularGrid::get_grad_lapl(bool & grad, bool & lap) const {
  grad=do_grad;
  lap=do_lapl;
}

void AngularGrid::set_grad_lapl(bool grad, bool lap) {
  do_grad=grad;
  do_lapl=lap;
}

// Fixed size shell
angshell_t AngularGrid::construct() {
  // Form the grid.
  form_grid();
  // Return the updated info structure, holding the amount of grid
  // points and function values
  return info;
}

angshell_t AngularGrid::construct(const arma::mat & P, double ftoler, int x_func, int c_func) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Köster et al for specifics.

  // Start with
  info.l=3;

  if(x_func == 0 && c_func == 0) {
    // No exchange or correlation!
    return info;
  }

  // Update shell list size
  form_grid();
  if(!info.nfunc)
    // No points!
    return info;

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(ftoler));

  // Old and new diagonal elements of Hamiltonian
  arma::vec Hold, Hnew;

  // Initialize vectors
  Hold.zeros(pot_bf_ind.n_elem);
  Hnew.zeros(pot_bf_ind.n_elem);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual rule
  do {
    // Form the grid using the current settings
    form_grid();
    // Compute density
    update_density(P);
    // Compute exchange and correlation.
    init_xc();
    // Compute the functionals
    if(x_func>0)
      compute_xc(x_func,true);
    if(c_func>0)
      compute_xc(c_func,true);
    // Construct the Fock matrix
    Hnew.zeros();
    eval_diag_Fxc(Hnew);

    // Compute maximum difference of diagonal elements of Fock matrix
    maxdiff=arma::max(arma::abs(Hold-Hnew));

    // Switch contents
    std::swap(Hold,Hnew);

    // Increment order if tolerance not achieved.
    if(maxdiff>ftoler) {
      if(use_lobatto)
	info.l+=2;
      else {
	// Need to determine what is next order of Lebedev
	// quadrature that is supported.
	info.l=next_lebedev(info.l);
      }
    }
  } while(maxdiff>ftoler && info.l<=lmax);

  // Free memory
  free();

  return info;
}

angshell_t AngularGrid::construct(const arma::mat & Pa, const arma::mat & Pb, double ftoler, int x_func, int c_func) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Köster et al for specifics.

  // Start with
  info.l=3;

  if(x_func == 0 && c_func == 0) {
    // No exchange or correlation!
    return info;
  }

  // Update shell list size
  form_grid();
  if(!info.nfunc)
    // No points!
    return info;

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(ftoler));

  // Old and new diagonal elements of Hamiltonian
  arma::vec Haold, Hanew, Hbold, Hbnew;
  // Initialize vectors
  Haold.zeros(pot_bf_ind.n_elem);
  Hanew.zeros(pot_bf_ind.n_elem);
  Hbold.zeros(pot_bf_ind.n_elem);
  Hbnew.zeros(pot_bf_ind.n_elem);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits
  do {
    form_grid();
    // Compute density
    update_density(Pa,Pb);

    // Compute exchange and correlation.
    init_xc();
    // Compute the functionals
    if(x_func>0)
      compute_xc(x_func,true);
    if(c_func>0)
      compute_xc(c_func,true);
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
    if(maxdiff>ftoler) {
      if(use_lobatto) {
	info.l+=2;
      } else {
	// Need to determine what is next order of Lebedev
	// quadrature that is supported.
	info.l=next_lebedev(info.l);
      }
    }
  } while(maxdiff>ftoler && info.l<=lmax);

  // Free memory
  free();

  return info;
}

angshell_t AngularGrid::construct(const arma::cx_vec & C, double ftoler, int x_func, int c_func) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Köster et al for specifics.

  // Start with
  info.l=3;

  if(x_func == 0 && c_func == 0) {
    // No exchange or correlation!
    return info;
  }

  // Update shell list size
  form_grid();
  if(!info.nfunc)
    // No points!
    return info;

  // Determine limit for angular quadrature
  int lmax=(int) ceil(5.0-6.0*log10(ftoler));

  // Old and new diagonal elements of Hamiltonian
  arma::vec Hold, Hnew, Hdum;
  Hold.zeros(pot_bf_ind.n_elem);
  Hnew.zeros(pot_bf_ind.n_elem);
  Hdum.zeros(pot_bf_ind.n_elem);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits
  do {
    // Form the grid
    form_grid();
    // Compute density
    update_density(C);

    // Compute exchange and correlation.
    init_xc();
    // Compute the functionals
    if(x_func>0)
      compute_xc(x_func,true);
    if(c_func>0)
      compute_xc(c_func,true);
    // and construct the Fock matrices
    Hnew.zeros();
    eval_diag_Fxc(Hnew,Hdum);

    // Compute maximum difference of diagonal elements of Fock matrix
    maxdiff=arma::max(arma::abs(Hold-Hnew));

    // Copy contents
    std::swap(Hold,Hnew);

    // Increment order if tolerance not achieved.
    if(maxdiff>ftoler) {
      if(use_lobatto)
	info.l+=2;
      else {
	// Need to determine what is next order of Lebedev
	// quadrature that is supported.
	info.l=next_lebedev(info.l);
      }
    }
  } while(maxdiff>ftoler && info.l<=lmax);

  // Free memory once more
  free();

  return info;
}

angshell_t AngularGrid::construct_becke(double otoler) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Krack 1998 for details

  // Start with
  info.l=3;

  // Update shell list size
  form_grid();
  if(!info.nfunc)
    // No points!
    return info;

  // Determine limit for angular quadrature
  int lmax=(int) ceil(2.0-3.0*log10(otoler));

  // Old and new diagonal elements of overlap
  arma::vec Sold(pot_bf_ind.n_elem), Snew(pot_bf_ind.n_elem);
  Sold.zeros();

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits
  do {
    // Form grid
    form_grid();

    // Compute new overlap
    Snew.zeros();
    eval_diag_overlap(Snew);

    // Compute maximum difference of diagonal elements
    maxdiff=arma::max(arma::abs(Snew-Sold));

    // Copy contents
    std::swap(Snew,Sold);

    // Increment order if tolerance not achieved.
    if(maxdiff>otoler) {
      if(use_lobatto)
	info.l+=2;
      else {
	// Need to determine what is next order of Lebedev
	// quadrature that is supported.
	info.l=next_lebedev(info.l);
      }
    }
  } while(maxdiff>otoler && info.l<=lmax);

  // Free memory once more
  free();

  return info;
}

angshell_t AngularGrid::construct_hirshfeld(const Hirshfeld & hirsh, double otoler) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Krack 1998 for details

  // Start with
  info.l=3;

  // Update shell list size
  form_grid();
  if(!info.nfunc)
    // No points!
    return info;

  // Determine limit for angular quadrature
  int lmax=(int) ceil(2.0-3.0*log10(otoler));

  // Old and new diagonal elements of overlap
  arma::vec Sold(pot_bf_ind.n_elem), Snew(pot_bf_ind.n_elem);
  Sold.zeros();

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits
  do {
    // Form the grid
    form_hirshfeld_grid(hirsh);

    // Compute new overlap
    Snew.zeros();
    eval_diag_overlap(Snew);

    // Compute maximum difference of diagonal elements
    maxdiff=arma::max(arma::abs(Snew-Sold));

    // Copy contents
    std::swap(Snew,Sold);

    // Increment order if tolerance not achieved.
    if(maxdiff>otoler) {
      if(use_lobatto)
	info.l+=2;
      else {
	// Need to determine what is next order of Lebedev
	// quadrature that is supported.
	info.l=next_lebedev(info.l);
      }
    }
  } while(maxdiff>otoler && info.l<=lmax);

  // Free memory once more
  free();

  return info;
}

void AngularGrid::form_grid() {
  // Clear anything that already exists
  free();

  // Add grid points
  if(use_lobatto)
    lobatto_shell();
  else
    lebedev_shell();

  // Do Becke weights
  becke_weights();
  // Prune points with small weight
  prune_points();
  // Collect weights
  get_weights();

  // Update shell list
  update_shell_list();
  // Compute basis functions
  compute_bf();
}

void AngularGrid::form_hirshfeld_grid(const Hirshfeld & hirsh) {
  // Clear anything that already exists
  free();

  // Add grid points
  if(use_lobatto)
    lobatto_shell();
  else
    lebedev_shell();

  // Do Becke weights
  hirshfeld_weights(hirsh);
  // Prune points with small weight
  prune_points();
  // Collect weights
  get_weights();

  // Update shell list
  update_shell_list();
  // Compute basis functions
  compute_bf();
}

void AngularGrid::update_shell_list() {
  // Form list of important basis functions. Shell ranges
  std::vector<double> shran=basp->get_shell_ranges();
  // Distances to other nuclei
  std::vector<double> nucdist=basp->get_nuclear_distances(info.atind);

  // Current radius
  double rad=info.R;
  // Shells that might contribute, and the amount of functions
  pot_shells.clear();
  size_t Nbf=0;
  for(size_t inuc=0;inuc<basp->get_Nnuc();inuc++) {
    // Closest distance of shell to nucleus. Covers both nucleus
    // inside shell, and nucleus outside the shell.
    double dist=fabs(nucdist[inuc]-rad);
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=basp->get_shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++) {
      // Shell is relevant if range is larger than minimal distance
      if(dist<=shran[shellinds[ish]]) {
	// Add shell to list of shells to compute
	pot_shells.push_back(shellinds[ish]);
	// Increment amount of important functions
	Nbf+=basp->get_Nbf(shellinds[ish]);
      }
    }
  }

  // Store indices of functions
  pot_bf_ind.zeros(Nbf);
  size_t ioff=0;
  for(size_t i=0;i<pot_shells.size();i++) {
    // Amount of functions on shell is
    size_t Nsh=basp->get_Nbf(pot_shells[i]);
    // Shell offset
    size_t sh0=basp->get_first_ind(pot_shells[i]);
    // Indices
    arma::uvec ls=(arma::linspace<arma::uvec>(sh0,sh0+Nsh-1,Nsh));
    pot_bf_ind.subvec(ioff,ioff+Nsh-1)=ls;
    ioff+=Nsh;
  }
}

void AngularGrid::compute_bf() {
  // Create list of shells that actually contribute. Shell ranges
  std::vector<double> shran=basp->get_shell_ranges();

  shells.clear();
  size_t Nbf=0;
  for(size_t is=0;is<pot_shells.size();is++) {
    // Shell center is
    coords_t cen(basp->get_shell_center(pot_shells[is]));
    // Shell range is
    double rangesq(std::pow(shran[pot_shells[is]],2));

    // Check if the function is important on at least one point in the
    // pruned grid
    for(size_t ip=0;ip<grid.size();ip++) {
      if(normsq(grid[ip].r-cen)<=rangesq) {
	// Shell is important!
	shells.push_back(pot_shells[is]);
	Nbf+=basp->get_Nbf(pot_shells[is]);
	break;
      }
    }
  }

  // Store indices of functions
  bf_ind.zeros(Nbf);
  size_t ioff=0;
  for(size_t i=0;i<shells.size();i++) {
    // Amount of functions on shell is
    size_t Nsh=basp->get_Nbf(shells[i]);
    // Shell offset
    size_t sh0=basp->get_first_ind(shells[i]);
    // Indices
    arma::uvec ls=(arma::linspace<arma::uvec>(sh0,sh0+Nsh-1,Nsh));
    bf_ind.subvec(ioff,ioff+Nsh-1)=ls;
    ioff+=Nsh;
  }

  // Store indices of functions on the potentials list
  bf_potind.zeros(Nbf);
  size_t j=0;
  ioff=0;
  size_t joff=0;
  for(size_t i=0;i<pot_shells.size() && j<shells.size();i++) {
    // Amount of functions on shell is
    size_t Nsh=basp->get_Nbf(pot_shells[i]);
    // Store indices?
    if(pot_shells[i]==shells[j]) {
      arma::uvec ls=(arma::linspace<arma::uvec>(joff,joff+Nsh-1,Nsh));
      bf_potind.subvec(ioff,ioff+Nsh-1)=ls;
      ioff+=Nsh;
      j++;
    }
    joff+=Nsh;
  }

  // Store number of function values
  info.nfunc=Nbf*grid.size();

  bf.zeros(bf_ind.n_elem,grid.size());
  // Loop over points
  for(size_t ip=0;ip<grid.size();ip++) {
    // Loop over shells. Offset
    ioff=0;
    for(size_t ish=0;ish<shells.size();ish++) {
      arma::vec fval=basp->eval_func(shells[ish],grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
      bf.submat(ioff,ip,ioff+fval.n_elem-1,ip)=fval;
      ioff+=fval.n_elem;
    }
  }

  if(do_grad) {
    bf_x.zeros(bf_ind.n_elem,grid.size());
    bf_y.zeros(bf_ind.n_elem,grid.size());
    bf_z.zeros(bf_ind.n_elem,grid.size());
    // Loop over points
    for(size_t ip=0;ip<grid.size();ip++) {
      // Loop over shells. Offset
      ioff=0;
      for(size_t ish=0;ish<shells.size();ish++) {
	arma::mat gval=basp->eval_grad(shells[ish],grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	bf_x.submat(ioff,ip,ioff+gval.n_rows-1,ip)=gval.col(0);
	bf_y.submat(ioff,ip,ioff+gval.n_rows-1,ip)=gval.col(1);
	bf_z.submat(ioff,ip,ioff+gval.n_rows-1,ip)=gval.col(2);
	ioff+=gval.n_rows;
      }
    }
  }

  if(do_lapl) {
    bf_lapl.zeros(bf_ind.n_elem,grid.size());
    // Loop over points
    for(size_t ip=0;ip<grid.size();ip++) {
      // Loop over shells. Offset
      ioff=0;
      for(size_t ish=0;ish<shells.size();ish++) {
	arma::vec lval=basp->eval_lapl(shells[ish],grid[ip].r.x,grid[ip].r.y,grid[ip].r.z);
	bf_lapl.submat(ioff,ip,ioff+lval.n_elem-1,ip)=lval;
	ioff+=lval.n_elem;
      }
    }
  }
}

DFTGrid::DFTGrid() {
}

DFTGrid::DFTGrid(const BasisSet * bas, bool ver, bool lobatto) {
  basp=bas;
  verbose=ver;

  // Allocate atomic grids
  grids.resize(bas->get_Nnuc());

  // Allocate work grids
#ifdef _OPENMP
  int nth=omp_get_max_threads();
  for(int i=0;i<nth;i++)
    wrk.push_back(AngularGrid(lobatto));
#else
  wrk.push_back(AngularGrid(lobatto));
#endif

  // Set basis set
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_basis(*bas);
}

DFTGrid::~DFTGrid() {
}

void DFTGrid::prune_shells() {
  for(size_t i=grids.size()-1;i<grids.size();i--)
    if(!grids[i].np || !grids[i].nfunc)
      grids.erase(grids.begin()+i);
}

void DFTGrid::construct(int nrad, int lmax, int x_func, int c_func, bool strict) {
  // Check necessity of gradients and laplacians
  bool grad, lapl;
  wrk[0].check_grad_lapl(x_func,c_func);
  wrk[0].get_grad_lapl(grad,lapl);
  construct(nrad,lmax,grad,lapl,strict,false);
}

void DFTGrid::construct(int nrad, int lmax, bool grad, bool lapl, bool strict, bool nl) {
  if(verbose) {
    if(nl)
      printf("Constructing static nrad=%i lmax=%i NL grid.\n",nrad,lmax);
    else
      printf("Constructing static nrad=%i lmax=%i XC grid.\n",nrad,lmax);
    fflush(stdout);
  }

  // Set necessity of gradienst and laplacian and grid
  for(size_t i=0;i<wrk.size();i++) {
    wrk[i].set_grad_lapl(grad,lapl);
  }

  // Set grid point screening tolerances
  double tol=strict ? 0.0 : DBL_EPSILON;

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev_jac(nrad,rad,wrad);
  nrad=rad.size(); // Sanity check

  // Construct grids
  size_t Nat=basp->get_Nnuc();
  grids.clear();
  // Loop over atoms
  for(size_t iat=0;iat<Nat;iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp->get_nuclear_coords(iat);
    sh.tol=tol;
    sh.np=0;
    sh.nfunc=0;

    // Loop over radii
    for(int irad=0;irad<nrad;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      sh.l=lmax;
      grids.push_back(sh);
    }
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifndef _OPENMP
    const int ith=0;
#else
    const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      wrk[ith].set_grid(grids[i]);
      grids[i]=wrk[ith].construct();
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose) {
    std::string met=nl ? "NL" : "XC";
    print_grid(met);
  }
}


void DFTGrid::construct(const arma::mat & P, double ftoler, int x_func, int c_func) {
  // Add all atoms
  if(verbose) {
    printf("Constructing adaptive XC grid with tolerance %e.\n",ftoler);
    fflush(stdout);
  }

  Timer t;

  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());

  // Form radial shells
  for(size_t iat=0;iat<basp->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp->get_nuclear_coords(iat);
    sh.tol=ftoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=std::max(20,(int) round(-5*(3*log10(ftoler)+6-element_row[basp->get_Z(iat)])));
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

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      wrk[ith].set_grid(grids[i]);
      grids[i]=wrk[ith].construct(P,ftoler/nrad[grids[i].atind],x_func,c_func);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    print_grid();
    fflush(stdout);
  }

}

void DFTGrid::construct(const arma::mat & Pa, const arma::mat & Pb, double ftoler, int x_func, int c_func) {
  // Add all atoms
  if(verbose) {
    printf("Constructing adaptive XC grid with tolerance %e.\n",ftoler);
    fflush(stdout);
  }

  Timer t;

  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());

  // Form radial shells
  for(size_t iat=0;iat<basp->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp->get_nuclear_coords(iat);
    sh.tol=ftoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=std::max(20,(int) round(-5*(3*log10(ftoler)+6-element_row[basp->get_Z(iat)])));
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

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      wrk[ith].set_grid(grids[i]);
      grids[i]=wrk[ith].construct(Pa,Pb,ftoler/nrad[grids[i].atind],x_func,c_func);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    print_grid();
    fflush(stdout);
  }
}

void DFTGrid::construct(const arma::cx_mat & Ctilde, double ftoler, int x_func, int c_func) {
  // Add all atoms
  if(verbose) {
    printf("Constructing adaptive XC grid with tolerance %e.\n",ftoler);
    fflush(stdout);
  }

  Timer t;

  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());

  // Form radial shells
  for(size_t iat=0;iat<basp->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp->get_nuclear_coords(iat);
    sh.tol=ftoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=std::max(20,(int) round(-5*(3*log10(ftoler)+6-element_row[basp->get_Z(iat)])));
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

  // Orbital grids
  std::vector< std::vector<angshell_t> > orbgrid(grids.size(),grids);
  for(size_t i=0;i<orbgrid.size();i++)
    orbgrid[i].resize(Ctilde.n_cols);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();

    // Collapse statement introduced in OpenMP 3.0 in May 2008
#if _OPENMP >= 200805
#pragma omp for schedule(dynamic,1) collapse(2)
#else
#pragma omp for schedule(dynamic,1)
#endif
#endif
    for(size_t ig=0;ig<grids.size();ig++)
      for(size_t iorb=0;iorb<Ctilde.n_cols;iorb++)
	{
	  wrk[ith].set_grid(grids[ig]);
	  orbgrid[ig][iorb]=wrk[ith].construct(Ctilde.col(iorb),ftoler/nrad[grids[ig].atind],x_func,c_func);
	}
  }

  // Update l values
  for(size_t ig=0;ig<grids.size();ig++) {
    grids[ig].l=orbgrid[ig][0].l;
    for(size_t io=1;io<orbgrid[ig].size();io++)
      grids[ig].l=std::max(orbgrid[ig][io].l,grids[ig].l);
  }

  // Rerun construction to update function value sizes
#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifndef _OPENMP
    const int ith=0;
#else
    const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      wrk[ith].set_grid(grids[i]);
      grids[i]=wrk[ith].construct();
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    print_grid();
    fflush(stdout);
  }
}

void DFTGrid::construct_becke(double otoler) {
  if(verbose) {
    printf("Constructing adaptive Becke grid with tolerance %e.\n",otoler);
    fflush(stdout);
  }

  // Only need function values
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_grad_lapl(false,false);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());

  Timer t;

  // Form radial shells
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

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      wrk[ith].set_grid(grids[i]);
      grids[i]=wrk[ith].construct_becke(otoler/nrad[grids[i].atind]);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose) {
    printf("Becke grid constructed in %s.\n",t.elapsed().c_str());
    print_grid("Becke");
    fflush(stdout);
  }
}

void DFTGrid::construct_hirshfeld(const Hirshfeld & hirsh, double otoler) {
  if(verbose) {
    printf("Constructing adaptive Hirshfeld grid with tolerance %e.\n",otoler);
    fflush(stdout);
  }

  // Only need function values
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_grad_lapl(false,false);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());

  Timer t;

  // Form radial shells
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

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      wrk[ith].set_grid(grids[i]);
      grids[i]=wrk[ith].construct_hirshfeld(hirsh,otoler/nrad[grids[i].atind]);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose) {
    printf("Hirshfeld grid constructed in %s.\n",t.elapsed().c_str());
    print_grid("Hirshfeld");
    fflush(stdout);
  }
}

size_t DFTGrid::get_Npoints() const {
  size_t np=0;
  for(size_t i=0;i<grids.size();i++)
    np+=grids[i].np;
  return np;
}

size_t DFTGrid::get_Nfuncs() const {
  size_t nf=0;
  for(size_t i=0;i<grids.size();i++)
    nf+=grids[i].nfunc;
  return nf;
}

arma::mat DFTGrid::eval_overlap() {
  // Amount of basis functions
  size_t N=basp->get_Nbf();

  // Returned matrix
  arma::mat S(N,N);
  S.zeros();

#ifndef _OPENMP
  int ith=0;
#else
  int ith=omp_get_thread_num();
#pragma omp parallel
#endif
  {
    // Work array
    arma::mat Swrk(S);
    Swrk.zeros();

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();
      // Evaluate overlap
      wrk[ith].eval_overlap(Swrk);

      // Free memory
      wrk[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    S+=Swrk;
  }

  return S;
}

arma::mat DFTGrid::eval_overlap(size_t inuc) {
  // Amount of basis functions
  size_t N=basp->get_Nbf();

  // Returned matrix
  arma::mat Sat(N,N);
  Sat.zeros();

#ifndef _OPENMP
  int ith=0;
#else
  int ith=omp_get_thread_num();
#pragma omp parallel
#endif
  {
    // Work array
    arma::mat Swrk(Sat);
    Swrk.zeros();

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      if(grids[i].atind!=inuc)
	continue;

      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();
      // Evaluate overlap
      wrk[ith].eval_overlap(Swrk);

      // Free memory
      wrk[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    Sat+=Swrk;
  }

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

    // Temporary matrix
    arma::mat Swrk(N,N);
    Swrk.zeros();

    // Add atomic contributions
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();

      // Evaluate overlap
      wrk[ith].eval_overlap(Swrk);
#ifdef _OPENMP
#pragma omp critical
#endif
      Sat[grids[i].atind]+=Swrk;

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
  for(size_t i=0;i<grids.size();i++) {
    if(grids[i].atind!=inuc)
      continue;

    wrk[ith].set_grid(grids[i]);
    wrk[ith].form_hirshfeld_grid(hirsh);
    // Evaluate overlap
    wrk[ith].eval_overlap(Sat);
    // Free memory
    wrk[ith].free();
  }

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

    // Work array
    arma::mat Swrk(Sat[0]);
    Swrk.zeros();

    // Add atomic contributions
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_hirshfeld_grid(hirsh);
      // Evaluate overlap
      wrk[ith].eval_overlap(Swrk);
      // Free memory
      wrk[ith].free();

      // Increment total matrix
#ifdef _OPENMP
#pragma omp critical
#endif
      Sat[grids[i].atind]+=Swrk;
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
#else
    int ith=0;
#endif

    // Temporary list
    std::vector<dens_list_t> hlp;

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    // Loop over integral grid
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();
      // Update density
      wrk[ith].update_density(P);

      // Compute helper
      hlp.clear();
      wrk[ith].get_density(hlp);

#ifdef _OPENMP
#pragma omp critical
#endif
      // Add to full list
      list.insert(list.end(),hlp.begin(),hlp.end());

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
#pragma omp for schedule(dynamic,1)
#endif
    // Loop over integral grid
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();
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
#pragma omp for schedule(dynamic,1)
#endif
    // Loop over integral grid
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();
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
#pragma omp for schedule(dynamic,1)
#endif
    // Loop over integral grid
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();
      // Update density
      wrk[ith].update_density(P);
      // Integrate electrons
      double dN=wrk[ith].compute_Nel();

#ifdef _OPENMP
#pragma omp critical
#endif
      Nel[grids[i].atind]+=dN;

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
#pragma omp for schedule(dynamic,1)
#endif
    // Loop over integral grid
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_hirshfeld_grid(hirsh);
      // Update density
      wrk[ith].update_density(P);
      // Integrate electrons
      double dN=wrk[ith].compute_Nel();
#ifdef _OPENMP
#pragma omp critical
#endif
      Nel[grids[i].atind]+=dN;
      // Free memory
      wrk[ith].free();
    }
  }

  return Nel;
}

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
#else
    int ith=0;
#endif

    // Loop over atoms
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();

      // Update density
      Timer tp;
      wrk[ith].update_density(P);
      // Update number of electrons
      Nel+=wrk[ith].compute_Nel();

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk[ith].compute_xc(x_func,true);
      if(c_func>0)
	wrk[ith].compute_xc(c_func,true);

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
#else
    int ith=0;
#endif

    // Loop over atoms
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {

      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(Pa,Pb);
      // Update number of electrons
      Nel+=wrk[ith].compute_Nel();

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk[ith].compute_xc(x_func,true);
      if(c_func>0)
	wrk[ith].compute_xc(c_func,true);

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

void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::cx_mat & CW, std::vector<arma::mat> & H, std::vector<double> & Exc, std::vector<double> & Nel, bool fock) {
  size_t nocc=CW.n_cols;

  // Allocate memory
  if(fock) {
    H.resize(nocc);
    // Hamiltonians computed in full space
    for(size_t ip=0;ip<nocc;ip++)
      H[ip].zeros(CW.n_rows,CW.n_rows);
  }

  // Clear exchange-correlation energy
  Exc.assign(nocc,0.0);
  // Clear number of electrons
  Nel.assign(nocc,0.0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Work matrix
    arma::mat Hwrk;
    // Dummy matrix
    arma::mat Hdum;

    if(fock)
      // Initialize memory
      Hwrk.zeros(CW.n_rows,CW.n_rows);

#ifndef _OPENMP
    int ith=0;
#else
    // Current thread is
    int ith=omp_get_thread_num();
#endif

    // Loop over grids
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {

      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();

      // Loop over densities
      for(size_t ip=0;ip<nocc;ip++) {
	// Update density
	wrk[ith].update_density(CW.col(ip));
	// Update number of electrons
#ifdef _OPENMP
#pragma omp atomic
#endif
	Nel[ip]+=wrk[ith].compute_Nel();

	// Initialize the arrays
	wrk[ith].init_xc();
	// Compute the functionals
	if(x_func>0)
	  wrk[ith].compute_xc(x_func,fock);
	if(c_func>0)
	  wrk[ith].compute_xc(c_func,fock);

	// Evaluate the energy
#ifdef _OPENMP
#pragma omp atomic
#endif
	Exc[ip]+=wrk[ith].eval_Exc();
	// and construct the Fock matrices
	if(fock) {
	  Hwrk.zeros(); // need to clear this here

	  wrk[ith].eval_Fxc(Hwrk,Hdum,false);

#ifdef _OPENMP
#pragma omp critical
#endif
	  H[ip]+=Hwrk;
	}
      }
    }

    // Free memory
    wrk[ith].free();

  } // End parallel region
}

void DFTGrid::eval_VV10(DFTGrid & nl, double b, double C, const arma::mat & P, arma::mat & H, double & Enl_, bool fock) {
  // Reset energy
  double Enl=0.0;

  // Original gradient and laplacian settings
  bool grad, lapl;
  wrk[0].get_grad_lapl(grad,lapl);

  // Collect nl grid data
  std::vector<arma::mat> nldata(nl.grids.size());
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    const int ith=0;
#else
    // Current thread is
    const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<nl.grids.size();i++) {
      // Change atom
      wrk[ith].set_grid(nl.grids[i]);
      wrk[ith].set_grad_lapl(true,false);
      // Create grid
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(P);

      // Collect the data
      wrk[ith].init_VV10(b,C,false);
      std::vector<size_t> idx;
      wrk[ith].collect_VV10(nldata[i],idx,b,C,true);

      wrk[ith].free();
    }
  } // End parallel region

  if(verbose) {
    size_t n=0;
    for(size_t i=0;i<nldata.size();i++)
      n+=nldata[i].n_rows;

    printf("%i points ... ",(int) n);
    fflush(stdout);
  }

  /*
  for(size_t ii=0;ii<nldata.size();ii++)
    for(size_t i=0;i<nldata[ii].n_rows;i++) {
      for(size_t j=0;j<nldata[ii].n_cols;j++)
	printf(" % e",nldata[ii](i,j));
      printf("\n");
    }
  */

  // Loop over grids
#ifdef _OPENMP
#pragma omp parallel reduction(+:Enl)
#endif
  {
    arma::mat Hwrk(H);
    Hwrk.zeros();

#ifndef _OPENMP
      const int ith=0;
#else
      const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      // Change atom
      wrk[ith].set_grid(grids[i]);
      wrk[ith].set_grad_lapl(true,false);
      // Initialize worker
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(P);
      // Initialize the arrays
      wrk[ith].init_xc();

      // Calculate the integral over the nl grid
      wrk[ith].init_VV10(b,C,true);
      wrk[ith].compute_VV10(nldata,b,C);

      // Evaluate the energy
      Enl+=wrk[ith].eval_Exc();
      // and construct the Fock matrices
      if(fock)
	wrk[ith].eval_Fxc(Hwrk);

      // Free memory
      wrk[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    if(fock)
      H+=Hwrk;
  }

  // Set return variable
  Enl_=Enl;

  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_grad_lapl(grad,lapl);
}

arma::vec DFTGrid::eval_force(int x_func, int c_func, const arma::mat & P) {
  arma::vec f(3*basp->get_Nnuc());
  f.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region

#ifndef _OPENMP
    const int ith=0;
#else
    // Current thread is
    const int ith=omp_get_thread_num();

    // Helper
    arma::vec fwrk(f);

#pragma omp for schedule(dynamic,1)
#endif
    // Loop over atoms
    for(size_t iat=0;iat<grids.size();iat++) {
      // Change atom and create grid
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(P);

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk[ith].compute_xc(x_func,true);
      if(c_func>0)
	wrk[ith].compute_xc(c_func,true);

      // Calculate the force on the atom
#ifdef _OPENMP
      fwrk+=wrk[ith].eval_force(P);
#else
      f+=wrk[ith].eval_force(P);
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
    const int ith=0;
#else
    // Current thread is
    const int ith=omp_get_thread_num();

    // Helper
    arma::vec fwrk(f);

#pragma omp for schedule(dynamic,1)
#endif
    // Loop over atoms
    for(size_t iat=0;iat<grids.size();iat++) {
      // Change atom and create grid
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(Pa,Pb);

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk[ith].compute_xc(x_func,true);
      if(c_func>0)
	wrk[ith].compute_xc(c_func,true);

      // Calculate the force on the atom
#ifdef _OPENMP
      fwrk+=wrk[ith].eval_force(Pa,Pb);
#else
      f+=wrk[ith].eval_force(Pa,Pb);
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

arma::vec DFTGrid::eval_VV10_force(DFTGrid & nl, double b, double C, const arma::mat & P) {
  // Forces on atoms
  arma::vec f(3*basp->get_Nnuc());
  f.zeros();

  // Original gradient and laplacian settings
  bool grad, lapl;
  wrk[0].get_grad_lapl(grad,lapl);

  // Collect nl grid data
  std::vector<arma::mat> nldata(nl.grids.size());
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    const int ith=0;
#else
    // Current thread is
    const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<nl.grids.size();i++) {
      // Need gradient but no laplacian
      wrk[ith].set_grad_lapl(true,false);
      // Change atom and create grid
      wrk[ith].set_grid(nl.grids[i]);
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(P);

      // Collect the data
      wrk[ith].init_VV10(b,C,false);
      std::vector<size_t> idx;
      wrk[ith].collect_VV10(nldata[i],idx,b,C,true);

      // Free memory
      wrk[ith].free();

    } // End nl grid loop
  } // End parallel region


  // Loop over grids
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::vec fwrk(f);
    fwrk.zeros();

#ifndef _OPENMP
    const int ith=0;
#else
    const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      // Need gradient but no laplacian
      wrk[ith].set_grad_lapl(true,false);
      // Change atom and create grid
      wrk[ith].form_grid();

      // Update density
      Timer tp;
      wrk[ith].update_density(P);
      // Initialize the arrays
      wrk[ith].init_xc();

      // Calculate kappa and omega and plug in constant energy density
      wrk[ith].init_VV10(b,C,true);
      // Evaluate the VV10 energy and potential and get the grid contribution
      fwrk.subvec(3*grids[i].atind,3*grids[i].atind+2)+=wrk[ith].compute_VV10_F(nldata,nl.grids,b,C);
      // and now evaluate the forces on the atoms
      fwrk+=wrk[ith].eval_force(P);

      // Free memory
      wrk[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    f+=fwrk;
  }

  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_grad_lapl(grad,lapl);

  return f;
}

void DFTGrid::print_density(const arma::mat & P, std::string densname) {
  // Open output files
  FILE *dens=fopen(densname.c_str(),"w");

  fprintf(dens,"%i\n",(int) get_Npoints());

  Timer t;
  if(verbose) {
    printf("\nSaving density data in %s ... ",densname.c_str());
    fflush(stdout);
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region

#ifdef _OPENMP
    // Current thread is
    const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#else
    const int ith=0;
#endif
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {

      // Change atom and create grid
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(P);

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk[ith].print_density(dens);
      }

      // Free memory
      wrk[ith].free();
    }
  } // End parallel region

    // Close output files
  fclose(dens);

  printf("done (%s)\n",t.elapsed().c_str());
}

void DFTGrid::print_potential(int func_id, const arma::mat & Pa, const arma::mat & Pb, std::string potname) {
  // Open output files
  FILE *pot=fopen(potname.c_str(),"w");
  fprintf(pot,"%i\n",(int) get_Npoints());

  Timer t;
  if(verbose) {
    printf("\nSaving potential data in %s ... ",potname.c_str());
    fflush(stdout);
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region

#ifdef _OPENMP
    // Current thread is
    const int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#else
    const int ith=0;
#endif
    // Loop over atoms
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].form_grid();

      // Update density
      wrk[ith].update_density(Pa,Pb);

      // Initialize the arrays
      wrk[ith].init_xc();
      // Compute the functionals
      if(func_id>0)
	wrk[ith].compute_xc(func_id,true);

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk[ith].print_potential(func_id,pot);
      }

      // Free memory
      wrk[ith].free();
    }
  } // End parallel region

    // Close output files
  fclose(pot);

  printf("done (%s)\n",t.elapsed().c_str());
}

void DFTGrid::print_grid(std::string met) const {
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
  /*
  printf("Grid info:\n");
  for(size_t i=0;i<grids.size();i++)
    printf("%6i %4i %8.3e %e %2i %7i %10i\n",(int) i, (int) grids[i].atind, grids[i].R, grids[i].w, grids[i].l, (int) grids[i].np, (int) grids[i].nfunc);
  */

  printf("Composition of %s grid:\n %7s %7s %10s\n",met.c_str(),"atom","Npoints","Nfuncs");
  for(size_t i=0;i<basp->get_Nnuc();i++)
    printf(" %4i %-2s %7i %10i\n",(int) i+1, basp->get_symbol(i).c_str(), (int) np(i), (int) nf(i));
}
