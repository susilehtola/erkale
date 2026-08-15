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

#include "settings.h"
#include "stringutil.h"
#include "timer.h"

// OpenMP parallellization for XC calculations
#ifdef _OPENMP
#include <omp.h>
#endif

extern Settings settings;

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

static int adaptive_l0() {
  return 3;
}

static int krack_nrad(double stol, int Z) {
  return std::max(20,(int) round(-5*(3*log10(stol)+8-(1+element_row[Z]))));
}

static int krack_lmax(double stol) {
  /*
    The default parameters in the paper

    return (int) ceil(2.0-3.0*log10(stol));

    give a very poor grid for e.g. CCl4: an 1e-5 tolerance yields a
    grid with -3.7 electrons error even in def2-SVP. As a quick hack,
    the parameters are re-chosen as the average of the old ones and of
    the ones from Koster which are meant for integrating the Fock
    matrix.
  */

  return (int) ceil(3.5-4.5*log10(stol));
}

static int koster_nrad(double ftol, int Z) {
  return std::max(20,(int) round(-5*(3*log10(ftol)+6-(1+element_row[Z]))));
}

static int koster_lmax(double ftol) {
  return (int) ceil(5.0-6.0*log10(ftol));
}

AngularGrid::AngularGrid(bool lobatto_) : use_lobatto_(lobatto_) {
  do_grad_=false;
  do_tau_=false;
  do_lapl_=false;
  do_hess_=false;
  do_lgrad_=false;
}

AngularGrid::~AngularGrid() {
}

void AngularGrid::basis(const BasisSet & basis) {
  basp_=&basis;
}

void AngularGrid::set_shell(const angshell_t & sh) {
  info_=sh;
}

std::vector<gridpoint_t> AngularGrid::grid() const {
  return grid_;
}

void AngularGrid::lobatto_shell() {
  // Add points on ind:th radial shell.

  // Get parameters of shell
  const double rad(info_.R);
  const int l(info_.l);
  const double wrad(info_.w);

  // Number of points in theta
  int nth=(l+3)/2;

  // Get corresponding Lobatto quadrature rule points in theta
  std::vector<double> xl, wl;
  lobatto_compute(nth,xl,wl);

  // Clear grid
  grid_.clear();

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
    point.w_=2.0*M_PI*wl[ith]/nphi*wrad;

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
      point.r=point.r+info_.cen;

      // Add point
      grid_.push_back(point);
      // Increment number of points
      np++;
    }
  }

  // Store number of points on this shell
  info_.np=np;
}

void AngularGrid::lebedev_shell() {
  // Parameters of shell
  const double rad(info_.R);
  const int l(info_.l);
  const double wrad(info_.w);

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
    point.r=point.r+info_.cen;

    // Compute quadrature weight
    // (Becke weight not included)
    point.w_=wrad*points[i].w;

    // Add point
    grid_.push_back(point);
  }

  // Store number of points on this shell
  info_.np=np;
}

void AngularGrid::becke_weights(double a) {
  // Compute weights of points.

  // Number of atoms in system
  const size_t Nat=basp_->get_Nnuc();

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
  std::vector<nucleus_t> nuccoords=basp_->get_nuclei();

  // Get nuclear distances
  arma::mat nucdist=basp_->nuclear_distances();
  // Compute closest distance to other atoms
  double Rin=DBL_MAX;
  for(size_t i=0;i<info_.atind;i++)
    if(nucdist(info_.atind,i)<Rin)
      Rin=nucdist(info_.atind,i);
  for(size_t i=info_.atind+1;i<Nat;i++)
    if(nucdist(info_.atind,i)<Rin)
      Rin=nucdist(info_.atind,i);

  double scrthr=std::pow(0.5*(1-a)*Rin,2);

  // Loop over grid
  for(size_t ip=0;ip<grid_.size();ip++) {
    // Coordinates of the point are
    coords_t coord_p=grid_[ip].r;

    // Prescreen - is the weight unity?
    if(normsq(nuccoords[info_.atind].r-coord_p) < scrthr)
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
    grid_[ip].w_*=atom_weight(info_.atind)/arma::sum(atom_weight);
  }
}

void AngularGrid::hirshfeld_weights(const Hirshfeld & hirsh) {
  // Compute weights of points.
  for(size_t ip=0;ip<grid_.size();ip++)
    // The Hirshfeld weight is
    grid_[ip].w_*=hirsh.weight(info_.atind,grid_[ip].r);
}

void AngularGrid::prune_points() {
  // Prune points with small weight.
  for(size_t i=grid_.size()-1;i<grid_.size();i--)
    if(grid_[i].w_<=info_.tol)
      grid_.erase(grid_.begin()+i);

  // Update amont of points
  info_.np=grid_.size();
}

void AngularGrid::free() {
  // Free integration points
  grid_.clear();
  w_.clear();

  // Free basis lists
  pot_shells_.clear();
  pot_bf_ind_.clear();
  shells_.clear();
  bf_ind_.clear();
  bf_potind_.clear();

  // Free values of basis functions
  bf_.clear();
  bf_x_.clear();
  bf_y_.clear();
  bf_z_.clear();
  bf_lapl_.clear();

  // Free density stuff
  Pv.clear();
  Pv_x.clear();
  Pv_y.clear();
  Pv_z_.clear();
  Pav.clear();
  Pav_x.clear();
  Pav_y.clear();
  Pav_z_.clear();
  Pbv.clear();
  Pbv_x.clear();
  Pbv_y.clear();
  Pbv_z_.clear();

  // Free LDA stuff
  rho_.clear();
  exc_.clear();
  vxc_.clear();

  // Free GGA stuff
  grho_.clear();
  sigma_.clear();
  vsigma_.clear();

  // Free mGGA stuff
  lapl_.clear();
  vlapl_.clear();
  tau_.clear();
  vtau_.clear();

  // Free VV10 stuff
  VV10_arr.clear();
}

arma::uvec AngularGrid::screen_density() const {
  double thr=settings.get_double("DFTDensityThr");

  // List of points
  std::vector<size_t> idx;
  // Loop over grid
  if(!polarized_) {
    if(rho_.size() != grid_.size()) throw std::logic_error("Wrong size density!\n");
    for(size_t i=0;i<grid_.size();i++)
      if(rho_(i)>=thr)
	idx.push_back(i);
  } else {
    if(rho_.size() != 2*grid_.size()) throw std::logic_error("Wrong size density!\n");
    for(size_t i=0;i<grid_.size();i++)
      if((rho_(2*i)+rho_(2*i+1))>=thr)
	idx.push_back(i);
  }

  return arma::conv_to<arma::uvec>::from(idx);
}


void AngularGrid::update_density(const arma::mat & P0, bool force) {
  // Update values of density

  if(!P0.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - density matrix is empty!\n");
  }

  // Non-polarized calculation.
  polarized_=false;

  // Update density vector
  arma::mat P(P0.submat(bf_ind_,bf_ind_));
  Pv=P*bf_;
  if(force && do_lapl_)
    Plapl=P*bf_lapl_;

  // Calculate density. Each per-grid-point dot product is replaced
  // with a single column-wise reduction (Pv % bf), which arma can
  // dispatch as a fused vectorised pass over contiguous memory rather
  // than N separate length-Nbf dots.
  rho_.zeros(1,grid_.size());
  rho_.row(0) = arma::sum(Pv % bf_, 0);

  // Calculate gradient
  if(do_grad_) {
    grho_.zeros(3,grid_.size());
    sigma_.zeros(1,grid_.size());
    grho_.row(0) = 2.0 * arma::sum(Pv % bf_x_, 0);
    grho_.row(1) = 2.0 * arma::sum(Pv % bf_y_, 0);
    grho_.row(2) = 2.0 * arma::sum(Pv % bf_z_, 0);
    sigma_.row(0) = arma::square(grho_.row(0)) + arma::square(grho_.row(1)) + arma::square(grho_.row(2));
  }

  // Calculate laplacian and kinetic energy density
  if(do_tau_ && do_lapl_) {
    // Adjust size of grid
    lapl_.zeros(1,grid_.size());
    tau_.zeros(1,grid_.size());

    // Update helpers
    Pv_x=P*bf_x_;
    Pv_y=P*bf_y_;
    Pv_z_=P*bf_z_;

    // Vectorised column-wise reductions
    const arma::rowvec lap_v(arma::sum(Pv % bf_lapl_, 0));
    const arma::rowvec grad_v(arma::sum(Pv_x % bf_x_ + Pv_y % bf_y_ + Pv_z_ % bf_z_, 0));
    lapl_.row(0) = 2.0 * (lap_v + grad_v);
    tau_.row(0) = 0.5 * grad_v;
  } else if(do_tau_) {
    // Adjust size of grid
    tau_.zeros(1,grid_.size());

    // Update helpers
    Pv_x=P*bf_x_;
    Pv_y=P*bf_y_;
    Pv_z_=P*bf_z_;

    // Vectorised column-wise reduction
    tau_.row(0) = 0.5 * arma::sum(Pv_x % bf_x_ + Pv_y % bf_y_ + Pv_z_ % bf_z_, 0);
  } else if(do_lapl_) {
    // Adjust size of grid
    lapl_.zeros(1,grid_.size());

    // Update helpers
    Pv_x=P*bf_x_;
    Pv_y=P*bf_y_;
    Pv_z_=P*bf_z_;

    // Vectorised column-wise reductions
    const arma::rowvec lap_v(arma::sum(Pv % bf_lapl_, 0));
    const arma::rowvec grad_v(arma::sum(Pv_x % bf_x_ + Pv_y % bf_y_ + Pv_z_ % bf_z_, 0));
    lapl_.row(0) = 2.0 * (lap_v + grad_v);
  }
}

void AngularGrid::update_density(const arma::mat & Pa0, const arma::mat & Pb0, bool force) {
  if(!Pa0.n_elem || !Pb0.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - density matrix is empty!\n");
  }

  // Polarized calculation.
  polarized_=true;

  // Update density vector
  arma::mat Pa(Pa0.submat(bf_ind_,bf_ind_));
  arma::mat Pb(Pb0.submat(bf_ind_,bf_ind_));

  Pav=Pa*bf_;
  Pbv=Pb*bf_;
  if(force && do_lapl_) {
    Palapl=Pa*bf_lapl_;
    Pblapl_=Pb*bf_lapl_;
  }

  // Calculate density
  rho_.zeros(2,grid_.size());
  rho_.row(0) = arma::sum(Pav % bf_, 0);
  rho_.row(1) = arma::sum(Pbv % bf_, 0);

  // Calculate gradient
  if(do_grad_) {
    grho_.zeros(6,grid_.size());
    sigma_.zeros(3,grid_.size());
    grho_.row(0) = 2.0 * arma::sum(Pav % bf_x_, 0);
    grho_.row(1) = 2.0 * arma::sum(Pav % bf_y_, 0);
    grho_.row(2) = 2.0 * arma::sum(Pav % bf_z_, 0);
    grho_.row(3) = 2.0 * arma::sum(Pbv % bf_x_, 0);
    grho_.row(4) = 2.0 * arma::sum(Pbv % bf_y_, 0);
    grho_.row(5) = 2.0 * arma::sum(Pbv % bf_z_, 0);
    sigma_.row(0) = arma::square(grho_.row(0)) + arma::square(grho_.row(1)) + arma::square(grho_.row(2));
    sigma_.row(1) = grho_.row(0)%grho_.row(3) + grho_.row(1)%grho_.row(4) + grho_.row(2)%grho_.row(5);
    sigma_.row(2) = arma::square(grho_.row(3)) + arma::square(grho_.row(4)) + arma::square(grho_.row(5));
  }

  // Calculate laplacian and kinetic energy density
  if(do_tau_ || do_lapl_) {
    // Adjust size of grid
    if(do_lapl_)
      lapl_.zeros(2,grid_.size());
    if(do_tau_)
      tau_.resize(2,grid_.size());

    // Update helpers
    Pav_x=Pa*bf_x_;
    Pav_y=Pa*bf_y_;
    Pav_z_=Pa*bf_z_;

    Pbv_x=Pb*bf_x_;
    Pbv_y=Pb*bf_y_;
    Pbv_z_=Pb*bf_z_;

    // Vectorised column-wise reductions across all grid points,
    // replacing the per-ip arma::dot loops above.
    if(do_tau_ && do_lapl_) {
      const arma::rowvec lapa(arma::sum(Pav % bf_lapl_, 0));
      const arma::rowvec lapb(arma::sum(Pbv % bf_lapl_, 0));
      const arma::rowvec grada(arma::sum(Pav_x % bf_x_ + Pav_y % bf_y_ + Pav_z_ % bf_z_, 0));
      const arma::rowvec gradb(arma::sum(Pbv_x % bf_x_ + Pbv_y % bf_y_ + Pbv_z_ % bf_z_, 0));
      lapl_.row(0) = 2.0 * (lapa + grada);
      lapl_.row(1) = 2.0 * (lapb + gradb);
      tau_.row(0)  = 0.5 * grada;
      tau_.row(1)  = 0.5 * gradb;
    } else if(do_tau_) {
      tau_.row(0) = 0.5 * arma::sum(Pav_x % bf_x_ + Pav_y % bf_y_ + Pav_z_ % bf_z_, 0);
      tau_.row(1) = 0.5 * arma::sum(Pbv_x % bf_x_ + Pbv_y % bf_y_ + Pbv_z_ % bf_z_, 0);
    } else if(do_lapl_) {
      const arma::rowvec lapa(arma::sum(Pav % bf_lapl_, 0));
      const arma::rowvec lapb(arma::sum(Pbv % bf_lapl_, 0));
      const arma::rowvec grada(arma::sum(Pav_x % bf_x_ + Pav_y % bf_y_ + Pav_z_ % bf_z_, 0));
      const arma::rowvec gradb(arma::sum(Pbv_x % bf_x_ + Pbv_y % bf_y_ + Pbv_z_ % bf_z_, 0));
      lapl_.row(0) = 2.0 * (lapa + grada);
      lapl_.row(1) = 2.0 * (lapb + gradb);
    }
  }
}

void AngularGrid::update_density(const arma::cx_vec & C0) {
  // Update values of density

  if(!C0.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Error - coefficient vector is empty!\n");
  }

  // Polarized calculation.
  polarized_=true;

  // Compute value of orbital
  arma::cx_vec C(bf_ind_.n_elem);
  for(size_t i=0;i<bf_ind_.n_elem;i++)
    C(i)=C0(bf_ind_(i));

  arma::cx_rowvec Cv=arma::strans(C)*bf_;
  // Store densities
  rho_.zeros(2,grid_.size());
  for(size_t ip=0;ip<grid_.size();ip++)
    // Compute densities
    rho_(0,ip)=std::norm(Cv(ip));

  if(do_grad_) {
    grho_.zeros(6,grid_.size());
    sigma_.zeros(3,grid_.size());

    // Compute orbital gradient
    arma::cx_rowvec Cv_x=arma::strans(C)*bf_x_;
    arma::cx_rowvec Cv_y=arma::strans(C)*bf_y_;
    arma::cx_rowvec Cv_z=arma::strans(C)*bf_z_;

    // Gradient is
    for(size_t ip=0;ip<grid_.size();ip++) {
      grho_(0,ip)=2.0*std::real(Cv_x(ip) * std::conj(Cv(ip)));
      grho_(1,ip)=2.0*std::real(Cv_y(ip) * std::conj(Cv(ip)));
      grho_(2,ip)=2.0*std::real(Cv_z(ip) * std::conj(Cv(ip)));

      // Compute values of sigma
      sigma_(0,ip) =std::pow(grho_(0,ip),2) + std::pow(grho_(1,ip),2) + std::pow(grho_(2,ip),2);
    }

    if(do_tau_ && do_lapl_) {
      // Adjust size of grid
      lapl_.zeros(2,grid_.size());
      tau_.zeros(2,grid_.size());

      // Compute orbital laplacian
      arma::cx_rowvec Cv_lapl=arma::strans(C)*bf_lapl_;

      for(size_t ip=0;ip<grid_.size();ip++) {
	// Laplacian term
	double lap=std::real(Cv_lapl(ip)*std::conj(Cv(ip)));
	// Gradient term
	double grad=std::norm(Cv_x(ip)) + std::norm(Cv_y(ip)) + std::norm(Cv_z(ip));

	// Laplacian is (including degeneracy factors)
	lapl_(0,ip)=2.0*(lap+grad);
	// Kinetic energy density is
	tau_(0,ip)=0.5*grad;
      }
    } else if(do_tau_) {
      // Adjust size of grid
      tau_.zeros(2,grid_.size());

      for(size_t ip=0;ip<grid_.size();ip++) {
	// Gradient term
	double grad=std::norm(Cv_x(ip)) + std::norm(Cv_y(ip)) + std::norm(Cv_z(ip));
	// Kinetic energy density is
	tau_(0,ip)=0.5*grad;
      }
    } else if(do_lapl_) {
      // Adjust size of grid
      lapl_.zeros(2,grid_.size());

      // Compute orbital laplacian
      arma::cx_rowvec Cv_lapl=arma::strans(C)*bf_lapl_;

      for(size_t ip=0;ip<grid_.size();ip++) {
	// Laplacian term
	double lap=std::real(Cv_lapl(ip)*std::conj(Cv(ip)));
	// Gradient term
	double grad=std::norm(Cv_x(ip)) + std::norm(Cv_y(ip)) + std::norm(Cv_z(ip));

	// Laplacian is (including degeneracy factors)
	lapl_(0,ip)=2.0*(lap+grad);
      }
    }
  }
}

void AngularGrid::density(std::vector<dens_list_t> & list) const {
  if(polarized_) {
    ERROR_INFO();
    throw std::runtime_error("get_density() is supposed to be called with a non-polarized grid!\n");
  }

  for(size_t ip=0;ip<grid_.size();ip++) {
    dens_list_t hlp;
    hlp.d=rho_(0,ip);
    hlp.w=w_(ip);
    list.push_back(hlp);
  }
}

double AngularGrid::compute_Nel() const {
  double nel=0.0;

  if(!polarized_)
    for(size_t ip=0;ip<grid_.size();ip++)
      nel+=w_(ip)*rho_(0,ip);
  else
    for(size_t ip=0;ip<grid_.size();ip++)
      nel+=w_(ip)*(rho_(0,ip)+rho_(1,ip));

  return nel;
}

void AngularGrid::print_grid() const {
  for(size_t ip=0;ip<grid_.size();ip++)
    printf("%5i % f % f % f %e\n",(int) ip+1,grid_[ip].r.x,grid_[ip].r.y,grid_[ip].r.z,grid_[ip].w_);
}

void AngularGrid::init_xc() {
  // Size of grid.
  const size_t N=grid_.size();

  // Zero energy
  zero_Exc();

  if(!polarized_) {
    // Restricted case
    vxc_.zeros(1,N);
    if(do_grad_)
      vsigma_.zeros(1,N);
    if(do_tau_)
      vtau_.zeros(1,N);
    if(do_lapl_)
      vlapl_.zeros(1,N);
  } else {
    // Unrestricted case
    vxc_.zeros(2,N);
    if(do_grad_)
      vsigma_.zeros(3,N);
    if(do_tau_)
      vtau_.zeros(2,N);
    if(do_lapl_) {
      vlapl_.zeros(2,N);
    }
  }

  // Initial values
  do_gga_=false;
  do_mgga_l_=false;
  do_mgga_t_=false;
}

void AngularGrid::zero_Exc() {
  exc_.zeros(grid_.size());
}

void AngularGrid::check_xc() {
  size_t nan=0;

  for(arma::uword i=0;i<exc_.n_elem;i++)
    if(std::isnan(exc_[i])) {
      nan++;
      exc_[i]=0.0;
    }

  for(arma::uword i=0;i<vxc_.n_elem;i++)
    if(std::isnan(vxc_[i])) {
      nan++;
      vxc_[i]=0.0;
    }

  for(arma::uword i=0;i<vsigma_.n_elem;i++)
    if(std::isnan(vsigma_[i])) {
      nan++;
      vsigma_[i]=0.0;
    }

  for(arma::uword i=0;i<vlapl_.n_elem;i++)
    if(std::isnan(vlapl_[i])) {
      nan++;
      vlapl_[i]=0.0;
    }


  for(arma::uword i=0;i<vtau_.n_elem;i++)
    if(std::isnan(vtau_[i])) {
      nan++;
      vtau_[i]=0.0;
    }

  if(nan) {
    printf("Warning - %i NaNs found in xc energy / potential.\n",(int) nan);
  }
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
  bool gga, mgga_t, mgga_l;
  is_gga_mgga(func_id,gga,mgga_t,mgga_l);

  // Update controlling flags for eval_Fxc (exchange and correlation
  // parts might be of different type)
  do_gga_=do_gga_ || gga || mgga_t || mgga_l;
  do_mgga_t_=do_mgga_t_ || mgga_t;
  do_mgga_l_=do_mgga_l_ || mgga_l;

  // Amount of grid points
  const size_t N=grid_.size();

  // Work arrays - exchange and correlation are computed separately
  arma::vec exc_wrk;
  arma::mat vxc_wrk;
  arma::mat vsigma_wrk;
  arma::mat vlapl_wrk;
  arma::mat vtau_wrk;

  if(has_exc(func_id))
    exc_wrk.zeros(exc_.n_elem);
  if(pot) {
    vxc_wrk.zeros(vxc_.n_rows,vxc_.n_cols);
    if(gga || mgga_t || mgga_l)
      vsigma_wrk.zeros(vsigma_.n_rows,vsigma_.n_cols);
    if(mgga_t)
      vtau_wrk.zeros(vtau_.n_rows,vtau_.n_cols);
    if(mgga_l)
      vlapl_wrk.zeros(vlapl_.n_rows,vlapl_.n_cols);
  }

  // Spin variable for libxc
  int nspin;
  if(!polarized_)
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
  // Set density threshold
  double thr=settings.get_double("DFTDensityThr");
  xc_func_set_dens_threshold(&func, thr);

  // Set parameters
  arma::vec pars;
  std::string functype;
  if(is_exchange(func_id)) {
    pars=settings.get_vec("DFTXpars");
    functype="exchange";
  } else if(is_correlation(func_id)) {
    pars=settings.get_vec("DFTCpars");
    functype="correlation";
  }
  if(pars.n_elem) {
    size_t npars = xc_func_info_get_n_ext_params((xc_func_info_type*) func.info);
    if(npars != pars.n_elem) {
      std::ostringstream oss;
      oss << "Inconsistent number of parameters for the " << functype << " functional.\n";
      oss << "Expected " << npars << ", got " << pars.n_elem << ".\n";
      throw std::logic_error(oss.str());
    }
    xc_func_set_ext_params(&func, pars.memptr());
  }

  // Evaluate functionals.
  if(has_exc(func_id)) {
    if(pot) {
      if(mgga_t || mgga_l) {// meta-GGA
	double * laplp = mgga_l ? lapl_.memptr() : NULL;
	double * taup = mgga_t ? tau_.memptr() : NULL;
	double * vlaplp = mgga_l ? vlapl_wrk.memptr() : NULL;
	double * vtaup = mgga_t ? vtau_wrk.memptr() : NULL;
	xc_mgga_exc_vxc(&func, N, rho_.memptr(), sigma_.memptr(), laplp, taup, exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr(), vlaplp, vtaup);
      } else if(gga) // GGA
	xc_gga_exc_vxc(&func, N, rho_.memptr(), sigma_.memptr(), exc_wrk.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
      else // LDA
	xc_lda_exc_vxc(&func, N, rho_.memptr(), exc_wrk.memptr(), vxc_wrk.memptr());
    } else {
      if(mgga_t || mgga_l) { // meta-GGA
	double * laplp = mgga_l ? lapl_.memptr() : NULL;
	double * taup = mgga_t ? tau_.memptr() : NULL;
	xc_mgga_exc(&func, N, rho_.memptr(), sigma_.memptr(), laplp, taup, exc_wrk.memptr());
      } else if(gga) // GGA
	xc_gga_exc(&func, N, rho_.memptr(), sigma_.memptr(), exc_wrk.memptr());
      else // LDA
	xc_lda_exc(&func, N, rho_.memptr(), exc_wrk.memptr());
    }

  } else {
    if(pot) {
      if(mgga_t || mgga_l) { // meta-GGA
	double * laplp = mgga_l ? lapl_.memptr() : NULL;
	double * taup = mgga_t ? tau_.memptr() : NULL;
	double * vlaplp = mgga_l ? vlapl_wrk.memptr() : NULL;
	double * vtaup = mgga_t ? vtau_wrk.memptr() : NULL;
	xc_mgga_vxc(&func, N, rho_.memptr(), sigma_.memptr(), laplp, taup, vxc_wrk.memptr(), vsigma_wrk.memptr(), vlaplp, vtaup);
      } else if(gga) // GGA
	xc_gga_vxc(&func, N, rho_.memptr(), sigma_.memptr(), vxc_wrk.memptr(), vsigma_wrk.memptr());
      else // LDA
	xc_lda_vxc(&func, N, rho_.memptr(), vxc_wrk.memptr());
    }
  }

  // Sum to total arrays containing both exchange and correlation
  if(has_exc(func_id))
    exc_+=exc_wrk;
  if(pot) {
    if(mgga_l)
      vlapl_+=vlapl_wrk;
    if(mgga_t)
      vtau_+=vtau_wrk;
    if(mgga_t || mgga_l || gga)
      vsigma_+=vsigma_wrk;
    vxc_+=vxc_wrk;
  }

  // Free functional
  xc_func_end(&func);
}

void AngularGrid::init_VV10(double b, double C, bool pot) {
  if(!do_grad_)
    throw std::runtime_error("Invalid do_grad setting for VV10!\n");
  do_gga_=true;
  do_mgga_t_=false;
  do_mgga_l_=false;
  VV10_thr=1e-8;

  if(rho_.size() != grid_.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Grid size is " << grid_.size() << " but there are only " << rho_.size() << " density values!\n";
    throw std::runtime_error(oss.str());
  }
  if(sigma_.size() != grid_.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Grid size is " << grid_.size() << " but there are only " << sigma_.size() << " reduced gradient values!\n";
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
    for(size_t i=0;i<grid_.size();i++) {
      exc_(i)+=beta;
      vxc_(0,i)+=beta;
    }
  }
}

// Both VV10_Kernel and VV10_Kernel_F do an O(N_xc * N_nl) double loop
// with ~12 fused-mul-adds plus three divides and 7 stride-1 reads of
// per-(nl-point) data per inner iteration. With the original
// arma::mat layout (5 rows for xc, 7 rows for nl, column-major), the
// nl reads inside the j loop walked stride-7 column starts in memory
// (nl.colptr(j) advances by 7 doubles per j) which compilers refuse
// to auto-vectorise.
//
// Below we transpose xc/nl once at the top of each kernel so each
// component (x, y, z, "a", "b", w, rho) is contiguous in memory.
// The cost is O(N_xc + N_nl), negligible against the O(N_xc * N_nl)
// inner work. With contiguous reads, gcc / clang vectorise the inner
// loop under #pragma omp simd reduction; the divisions become the
// remaining bottleneck and are typical 4-wide on AVX2 / 8-wide on
// AVX-512.

void VV10_Kernel(const arma::mat & xc, const arma::mat & nl, arma::mat & ret) {
  // Input arrays contain grid[i].r, omega0(i), kappa(i) (and grid[i].w, rho[i] for nl)
  // Return array contains: nPhi, U, and W

  if(xc.n_rows !=5) {
    ERROR_INFO();
    throw std::runtime_error("xc matrix has the wrong size.\n");
  }
  if(nl.n_rows !=7) {
    ERROR_INFO();
    throw std::runtime_error("nl matrix has the wrong size.\n");
  }
  if(ret.n_cols != xc.n_cols || ret.n_rows != 3) {
    throw std::runtime_error("Error - invalid size output array!\n");
  }

  const arma::mat xc_t(xc.t());
  const arma::mat nl_t(nl.t());

  const double* xc_x = xc_t.colptr(0);
  const double* xc_y = xc_t.colptr(1);
  const double* xc_z = xc_t.colptr(2);
  const double* xc_a = xc_t.colptr(3);
  const double* xc_b = xc_t.colptr(4);

  const double* nl_x   = nl_t.colptr(0);
  const double* nl_y   = nl_t.colptr(1);
  const double* nl_z   = nl_t.colptr(2);
  const double* nl_a   = nl_t.colptr(3);
  const double* nl_b   = nl_t.colptr(4);
  const double* nl_w   = nl_t.colptr(5);
  const double* nl_rho = nl_t.colptr(6);

  const size_t Nxc = xc.n_cols;
  const size_t Nnl = nl.n_cols;

  for(size_t i=0;i<Nxc;i++) {
    const double xi = xc_x[i];
    const double yi = xc_y[i];
    const double zi = xc_z[i];
    const double ai = xc_a[i];
    const double bi = xc_b[i];

    double nPhi=0.0, U=0.0, W=0.0;

#pragma omp simd reduction(+:nPhi,U,W)
    for(size_t j=0;j<Nnl;j++) {
      const double dx = xi - nl_x[j];
      const double dy = yi - nl_y[j];
      const double dz = zi - nl_z[j];
      const double Rsq = dx*dx + dy*dy + dz*dz;

      const double gi = ai*Rsq + bi;
      const double gj = nl_a[j]*Rsq + nl_b[j];
      const double gs = gi + gj;
      const double rgis = 1.0/gi + 1.0/gs;

      const double Phi = -3.0/(2.0*gi*gj*gs) * nl_w[j] * nl_rho[j];

      nPhi += Phi;
      U    -= Phi * rgis;
      W    -= Phi * rgis * Rsq;
    }

    ret(0,i) += nPhi;
    ret(1,i) += U;
    ret(2,i) += W;
  }
}

void VV10_Kernel_F(const arma::mat & xc, const arma::mat & nl, arma::mat & ret) {
  // Input arrays contain grid[i].r, omega0(i), kappa(i) (and grid[i].w, rho[i] for nl)
  // Return array contains: nPhi, U, W, and fx, fy, fz

  if(xc.n_rows !=5) {
    ERROR_INFO();
    throw std::runtime_error("xc matrix has the wrong size.\n");
  }
  if(nl.n_rows !=7) {
    ERROR_INFO();
    throw std::runtime_error("nl matrix has the wrong size.\n");
  }
  if(ret.n_cols != xc.n_cols || ret.n_rows != 6) {
    throw std::runtime_error("Error - invalid size output array!\n");
  }

  const arma::mat xc_t(xc.t());
  const arma::mat nl_t(nl.t());

  const double* xc_x = xc_t.colptr(0);
  const double* xc_y = xc_t.colptr(1);
  const double* xc_z = xc_t.colptr(2);
  const double* xc_a = xc_t.colptr(3);
  const double* xc_b = xc_t.colptr(4);

  const double* nl_x   = nl_t.colptr(0);
  const double* nl_y   = nl_t.colptr(1);
  const double* nl_z   = nl_t.colptr(2);
  const double* nl_a   = nl_t.colptr(3);
  const double* nl_b   = nl_t.colptr(4);
  const double* nl_w   = nl_t.colptr(5);
  const double* nl_rho = nl_t.colptr(6);

  const size_t Nxc = xc.n_cols;
  const size_t Nnl = nl.n_cols;

  for(size_t i=0;i<Nxc;i++) {
    const double xi = xc_x[i];
    const double yi = xc_y[i];
    const double zi = xc_z[i];
    const double ai = xc_a[i];
    const double bi = xc_b[i];

    double nPhi=0.0, U=0.0, W=0.0;
    double fpx=0.0, fpy=0.0, fpz=0.0;

#pragma omp simd reduction(+:nPhi,U,W,fpx,fpy,fpz)
    for(size_t j=0;j<Nnl;j++) {
      const double dx = xi - nl_x[j];
      const double dy = yi - nl_y[j];
      const double dz = zi - nl_z[j];
      const double Rsq = dx*dx + dy*dy + dz*dz;

      const double aj = nl_a[j];
      const double gi = ai*Rsq + bi;
      const double gj = aj*Rsq + nl_b[j];
      const double gs = gi + gj;
      const double rgis = 1.0/gi + 1.0/gs;

      const double Phi = -3.0/(2.0*gi*gj*gs) * nl_w[j] * nl_rho[j];

      nPhi += Phi;
      U    -= Phi * rgis;
      W    -= Phi * rgis * Rsq;

      const double Q = -2.0 * Phi * (ai/gi + aj/gj + (ai+aj)/gs);
      fpx += Q * dx;
      fpy += Q * dy;
      fpz += Q * dz;
    }

    ret(0,i) += nPhi;
    ret(1,i) += U;
    ret(2,i) += W;
    ret(3,i) += fpx;
    ret(4,i) += fpy;
    ret(5,i) += fpz;
  }
}

void AngularGrid::collect_VV10(arma::mat & data, std::vector<size_t> & idx, double b, double C, bool nl) const {
  if(polarized_) {
    ERROR_INFO();
    throw std::runtime_error("collect_VV10 can only be called in a non-polarized grid!\n");
  }

  // Create list of points with significant densities
  idx.clear();
  for(size_t i=0;i<grid_.size();i++)
    if(rho_(0,i) >= VV10_thr)
      idx.push_back(i);

  // Create input datas
  if(nl)
    data.zeros(7,idx.size());
  else
    data.zeros(5,idx.size());

  // Constants for omega and kappa
  const double oc=4.0*M_PI/3.0;
  const double kc=(3.0*M_PI*b)/2.0*std::pow(9.0*M_PI,-1.0/6.0);
  for(size_t ii=0;ii<idx.size();ii++) {
    size_t i=idx[ii];
    data(0,ii)=grid_[i].r.x;
    data(1,ii)=grid_[i].r.y;
    data(2,ii)=grid_[i].r.z;
    // omega_0[i]
    data(3,ii)=sqrt(C * std::pow(sigma_(0,i)/(rho_(0,i)*rho_(0,i)),2) + oc*rho_(0,i));
    // kappa[i]
    data(4,ii)=kc * cbrt(sqrt(rho_(0,i)));
  }
  if(nl) {
    for(size_t ii=0;ii<idx.size();ii++) {
      size_t i=idx[ii];
      data(5,ii)=w_(i);
      data(6,ii)=rho_(0,i);
    }
  }
}

void AngularGrid::compute_VV10(const std::vector<arma::mat> & nldata, double b, double C) {
  if(polarized_) {
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
  VV10_arr.zeros(3,xc.n_cols);
  for(size_t i=0;i<nldata.size();i++)
    VV10_Kernel(xc,nldata[i],VV10_arr);

  // Collect data
  for(size_t ii=0;ii<idx.size();ii++) {
    // Index of grid point is
    size_t i=idx[ii];

    // Increment the energy density
    exc_[i] += 0.5 * VV10_arr(0,ii);

    // Increment LDA and GGA parts of potential.
    double ri=rho_(0,i);
    double ri4=std::pow(ri,4);
    double si=sigma_(0,i);
    double w0=xc(3,ii);
    double dkdn  = xc(4,ii)/(6.0*ri); // d kappa / d n
    double dw0ds = C*si / ( w0 * ri4); // d omega0 / d sigma
    double dw0dn = 2.0/w0 * ( M_PI/3.0 - C*si*si / (ri*ri4)); // d omega0 / d n
    vxc_(0,i) += VV10_arr(0,ii) + ri *( dkdn * VV10_arr(1,ii) + dw0dn * VV10_arr(2,ii));
    vsigma_(0,i) += ri * dw0ds * VV10_arr(2,ii);
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
  if(polarized_) {
    ERROR_INFO();
    throw std::runtime_error("compute_VV10_F should be run in non-polarized mode!\n");
  }

  // Create list of points with significant densities and input data
  std::vector<size_t> idx;
  arma::mat xc;
  collect_VV10(xc, idx, b, C, false);

  // Calculate integral kernels
  VV10_arr.zeros(6,xc.n_cols);
  for(size_t i=0;i<nldata.size();i++)
    if(nlgrids[i].atind==info_.atind) {
      // No Q contribution!
      //VV10_Kernel(xc,nldata[i],VV10_arr.cols(0,2));

      arma::mat Kat(3,xc.n_rows);
      Kat.zeros();
      VV10_Kernel(xc,nldata[i],Kat);
      VV10_arr.rows(0,2)+=Kat;
    } else
      // Full contribution
      VV10_Kernel_F(xc,nldata[i],VV10_arr);

  // Evaluate force contribution
  double fx=0.0, fy=0.0, fz=0.0;

  for(size_t ii=0;ii<idx.size();ii++) {
    // Index of grid point is
    size_t i=idx[ii];

    // Increment the energy density
    exc_[i] += 0.5 * VV10_arr(0,ii);

    // Increment LDA and GGA parts of potential.
    double ri=rho_(0,i);
    double ri4=std::pow(ri,4);
    double si=sigma_(0,i);
    double w0=xc(3,ii);
    double dkdn  = xc(4,ii)/(6.0*ri); // d kappa / d n
    double dw0ds = C*si / ( w0 * ri4); // d omega0 / d sigma
    double dw0dn = 2.0/w0 * ( M_PI/3.0 - C*si*si / (ri*ri4)); // d omega0 / d n
    vxc_(0,i) += VV10_arr(0,ii) + ri *( dkdn * VV10_arr(1,ii) + dw0dn * VV10_arr(2,ii));
    vsigma_(0,i) += ri * dw0ds * VV10_arr(2,ii);

    // Increment total force
    fx += grid_[i].w_*rho_(0,i)*VV10_arr(3,ii);
    fy += grid_[i].w_*rho_(0,i)*VV10_arr(4,ii);
    fz += grid_[i].w_*rho_(0,i)*VV10_arr(5,ii);
  }

  arma::vec F(3);
  F(0)=fx;
  F(1)=fy;
  F(2)=fz;

  return F;
}

void AngularGrid::print_density(FILE *f) const {
  // Loop over grid points
  for(size_t i=0;i<grid_.size();i++) {
    // Get data in point
    libxc_dens_t d=dens(i);

    // Print out data
    fprintf(f,"% .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e\n",d.rhoa,d.rhob,d.sigmaaa,d.sigmaab,d.sigmabb,d.lapla,d.laplb,d.taua,d.taub);
  }
}

void AngularGrid::print_potential(int func_id, FILE *f) const {
  // Loop over grid points
  for(size_t i=0;i<grid_.size();i++) {
    // Get data in point
    libxc_pot_t d=pot(i);
    double e=exc_(i);

    int nspin;
    if(polarized_)
      nspin=XC_POLARIZED;
    else
      nspin=XC_POLARIZED;

    // Print out data
    fprintf(f, "%3i %2i % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e\n",func_id,nspin,e,d.vrhoa,d.vrhob,d.vsigmaaa,d.vsigmaab,d.vsigmabb,d.vlapla,d.vlaplb,d.vtaua,d.vtaub);
  }
}

void AngularGrid::check_potential(FILE *f) const {
  // Loop over grid points
  for(size_t i=0;i<grid_.size();i++) {
    // Get data in point
    libxc_pot_t v=pot(i);
    double e=exc_(i);
    if(std::isnan(e) || std::isnan(v.vrhoa) || std::isnan(v.vrhob) || std::isnan(v.vsigmaaa) || std::isnan(v.vsigmaab) || std::isnan(v.vsigmabb) || std::isnan(v.vlapla) || std::isnan(v.vlaplb) || std::isnan(v.vtaua) || std::isnan(v.vtaub)) {
      libxc_dens_t d=dens(i);
      fprintf(f,"***\n");
      fprintf(f,"% .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e\n",d.rhoa,d.rhob,d.sigmaaa,d.sigmaab,d.sigmabb,d.lapla,d.laplb,d.taua,d.taub);
      fprintf(f,"% .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e % .16e\n",e,v.vrhoa,v.vrhob,v.vsigmaaa,v.vsigmaab,v.vsigmabb,v.vlapla,v.vlaplb,v.vtaua,v.vtaub);
    }
  }
  fflush(f);
}

void AngularGrid::compute_weights() {
  if(!grid_.size())
    return;
  w_.zeros(grid_.size());
  for(size_t pi=0;pi<grid_.size();pi++)
    w_(pi)=grid_[pi].w_;
}

libxc_dens_t AngularGrid::dens(size_t idx) const {
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

  if(polarized_) {
    ret.rhoa=rho_(0,idx);
    ret.rhob=rho_(1,idx);
  } else {
    ret.rhoa=ret.rhob=rho_(0,idx)/2.0;
  }

  if(do_gga_) {
    if(polarized_) {
      ret.sigmaaa=sigma_(0,idx);
      ret.sigmaab=sigma_(1,idx);
      ret.sigmabb=sigma_(2,idx);
    } else {
      ret.sigmaaa=ret.sigmaab=ret.sigmabb=sigma_(0,idx)/4.0;
    }
  }

  if(do_mgga_t_) {
    if(polarized_) {
      ret.taua=tau_(0,idx);
      ret.taub=tau_(1,idx);
    } else {
      ret.taua=ret.taub=tau_(0,idx)/2.0;
    }
  }
  if(do_mgga_l_) {
    if(polarized_) {
      ret.lapla=lapl_(0,idx);
      ret.laplb=lapl_(1,idx);
    } else {
      ret.lapla=ret.laplb=lapl_(0,idx)/2.0;
    }
  }

  return ret;
}

libxc_debug_t AngularGrid::data(size_t idx) const {
  libxc_debug_t d;
  d.dens=dens(idx);
  d.pot=pot(idx);
  d.e=exc_[idx];
  return d;
}

libxc_pot_t AngularGrid::pot(size_t idx) const {
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

  if(polarized_) {
    ret.vrhoa=vxc_(0,idx);
    ret.vrhob=vxc_(1,idx);
  } else {
    ret.vrhoa=ret.vrhob=vxc_(0,idx)/2.0;
  }

  if(do_gga_) {
    if(polarized_) {
      ret.vsigmaaa=vsigma_(0,idx);
      ret.vsigmaab=vsigma_(1,idx);
      ret.vsigmabb=vsigma_(2,idx);
    } else {
      ret.vsigmaaa=ret.vsigmaab=ret.vsigmabb=vsigma_(0,idx)/4.0;
    }
  }

  if(do_mgga_t_) {
    if(polarized_) {
      ret.vtaua=vtau_(0,idx);
      ret.vtaub=vtau_(1,idx);
    } else {
      ret.vtaua=ret.vtaub=vtau_(0,idx)/2.0;
    }
  }

  if(do_mgga_l_) {
    if(polarized_) {
      ret.vlapla=vlapl_(0,idx);
      ret.vlaplb=vlapl_(1,idx);
    } else {
      ret.vlapla=ret.vlaplb=vlapl_(0,idx)/2.0;
    }
  }

  return ret;
}

double AngularGrid::eval_Exc() const {
  arma::uvec screen(screen_density());

  arma::rowvec dens(rho_.row(0));
  if(polarized_)
    dens+=rho_.row(1);

  // Fast path when nothing was screened: skip the indexed view
  // copies. Note `exc` is stored as arma::vec (column) while w and
  // dens are arma::rowvec, so transpose exc to align orientations
  // before the element-wise multiply.
  if(screen.n_elem == w_.n_elem)
    return arma::sum(w_ % exc_.t() % dens);
  return arma::sum(w_(screen)%exc_(screen)%dens(screen));
}

void AngularGrid::eval_overlap(arma::mat & So) const {
  // Calculate in subspace
  arma::mat S(bf_ind_.n_elem,bf_ind_.n_elem);
  S.zeros();
  increment_lda<double>(S,w_,bf_);
  // Increment
  So.submat(bf_ind_,bf_ind_)+=S;
}

void AngularGrid::eval_diag_overlap(arma::vec & S) const {
  S.zeros(pot_bf_ind_.n_elem);

  arma::mat bft(bf_.t());
  for(size_t j=0;j<bf_.n_rows;j++)
    S(bf_potind_(j))=arma::sum(arma::square(bft.col(j))%w_.t());
}

static arma::mat calculate_rho(const arma::cx_mat & Cocc, const arma::mat & bf) {
  // Transpose C to (norb,nbf)
  arma::cx_mat C(arma::strans(Cocc));
  // Calculate values of orbitals at grid points: (norb, ngrid)
  arma::cx_mat orbvals(C*bf);

  // Orbital densities at grid points (norb, ngrid)
  return arma::real(orbvals%arma::conj(orbvals));
}

void AngularGrid::eval_overlap(const arma::cx_mat & Cocc, size_t io, double k, arma::mat & So, double thr) const {
  // Calculate in subspace
  arma::mat S(bf_ind_.n_elem,bf_ind_.n_elem);
  S.zeros();

  // Orbital densities at grid points (norb, ngrid)
  arma::mat orbdens(calculate_rho(Cocc.rows(bf_ind_),bf_));

  // Calculate weightings
  arma::rowvec ww(w_);
  for(size_t ip=0;ip<grid_.size();ip++) {
    // Orbital density is
    double rhois(orbdens(io,ip));
    // Total density
    double rhotot(arma::sum(orbdens.col(ip)));

    // Screen for bad behavior
    if(rhotot>=thr)
      ww(ip)*=std::pow(rhois/rhotot,k);
    else
      ww(ip)=0.0;
  }

  increment_lda<double>(S,ww,bf_);
  // Increment
  So.submat(bf_ind_,bf_ind_)+=S;
}

void AngularGrid::eval_overlap(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, arma::mat & So, double thr) const {
  // Calculate in subspace
  arma::mat S(bf_ind_.n_elem,bf_ind_.n_elem);
  S.zeros();

  // Orbital densities at grid points (norb, ngrid)
  arma::mat orbdens(calculate_rho(Cocc.rows(bf_ind_),bf_));
  arma::mat orbdenspk(arma::pow(orbdens,k));

  // Calculate weightings
  arma::rowvec ww(w_);
  for(size_t ip=0;ip<grid_.size();ip++) {
    // Total density
    double rhotot(arma::sum(orbdens.col(ip)));

    // Screen for bad behavior
    if(rhotot>=thr)
      ww(ip)*=arma::dot(Esi,orbdenspk.col(ip))/std::pow(rhotot,k);
    else
      ww(ip)=0.0;
  }

  increment_lda<double>(S,ww,bf_);
  // Increment
  So.submat(bf_ind_,bf_ind_)+=S;
}

static void calculate_tau_grho_tauw(const arma::cx_mat & Cocc, const arma::mat & bf, const arma::mat & bf_x, const arma::mat & bf_y, const arma::mat & bf_z, arma::vec & tau, arma::mat & grho, arma::vec & tau_w) {
  // Density matrix
  arma::cx_mat P(Cocc*arma::trans(Cocc));
  arma::cx_mat Pvec(P*bf);
  arma::cx_mat Pvec_x(P*bf_x);
  arma::cx_mat Pvec_y(P*bf_y);
  arma::cx_mat Pvec_z(P*bf_z);

  // Kinetic energy density
  tau.zeros(bf.n_cols);
  tau_w.zeros(bf.n_cols);
  grho.zeros(3,bf.n_cols);

  for(size_t ip=0;ip<tau.n_elem;ip++) {
    double kinx(std::real(arma::dot(Pvec_x.col(ip),bf_x.col(ip))));
    double kiny(std::real(arma::dot(Pvec_y.col(ip),bf_y.col(ip))));
    double kinz(std::real(arma::dot(Pvec_z.col(ip),bf_z.col(ip))));
    tau(ip)=0.5*(kinx+kiny+kinz);

    // Density is
    double n=std::real(arma::dot(Pvec.col(ip),bf.col(ip)));
    // Density gradient
    grho(0,ip)=2.0*std::real(arma::dot(Pvec.col(ip),bf_x.col(ip)));
    grho(1,ip)=2.0*std::real(arma::dot(Pvec.col(ip),bf_y.col(ip)));
    grho(2,ip)=2.0*std::real(arma::dot(Pvec.col(ip),bf_z.col(ip)));
    double g=arma::dot(grho.col(ip),grho.col(ip));
    tau_w(ip)=g/(8*n);
  }

  // Transpose grho
  grho=arma::trans(grho);
}

void AngularGrid::eval_tau_overlap(const arma::cx_mat & Cocc, double k, arma::mat & So, double thr) const {
  // Calculate in subspace
  arma::mat S(bf_ind_.n_elem,bf_ind_.n_elem);
  S.zeros();

  if(!do_grad_) throw std::logic_error("Must have gradients enabled to calculate tau overlap!\n");

  // Get kinetic and Weiszäcker kinetic energy densities
  arma::mat gr;
  arma::vec t, tw;
  calculate_tau_grho_tauw(Cocc.rows(bf_ind_),bf_,bf_x_,bf_y_,bf_z_,t,gr,tw);

  // Calculate weightings
  arma::rowvec ww(w_);
  for(size_t ip=0;ip<grid_.size();ip++) {
    // Screen for bad behavior
    if(t(ip)>=thr)
      ww(ip)*=std::pow(tw(ip)/t(ip),k);
    else
      ww(ip)=0.0;
  }

  increment_lda<double>(S,ww,bf_);
  // Increment
  So.submat(bf_ind_,bf_ind_)+=S;
}

void AngularGrid::eval_tau_overlap_deriv(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, arma::mat & So, double thr) const {
  if(!do_grad_) throw std::logic_error("Must have gradients enabled to calculate tau overlap!\n");

  // Get orbital densities
  arma::mat orbdens(calculate_rho(Cocc.rows(bf_ind_),bf_));

  // Get kinetic and Weiszäcker kinetic energy densities
  arma::mat gr;
  arma::vec t, tw;
  calculate_tau_grho_tauw(Cocc.rows(bf_ind_),bf_,bf_x_,bf_y_,bf_z_,t,gr,tw);

  // Calculate in subspace
  arma::mat S(bf_ind_.n_elem,bf_ind_.n_elem);
  S.zeros();

  // LDA part
  {
    // Calculate weightings
    arma::rowvec ww(w_);
    for(size_t ip=0;ip<grid_.size();ip++) {
      // Screen for bad behavior
      if(t(ip)>=thr)
	ww(ip)*=-k*std::pow(tw(ip)/t(ip),k)*(arma::dot(Esi,orbdens.col(ip))/arma::sum(orbdens.col(ip)));
      else
	ww(ip)=0.0;
    }

    increment_lda<double>(S,ww,bf_);
  }

  // meta-GGA part
  {
    // Calculate weightings
    arma::rowvec ww(w_);
    for(size_t ip=0;ip<grid_.size();ip++) {
      // Screen for bad behavior
      if(t(ip)>=thr)
	ww(ip)*=-k/2*std::pow(tw(ip)/t(ip),k)*(arma::dot(Esi,orbdens.col(ip))/t(ip));
      else
	ww(ip)=0.0;
    }

    increment_mgga_kin<double>(S,ww,bf_x_,bf_y_,bf_z_);
  }

  // GGA part
  {
    // Multiply grad rho by the weights and the integrand
    for(size_t ip=0;ip<gr.n_rows;ip++)
      for(size_t ic=0;ic<gr.n_cols;ic++)
	if(t(ip)>=thr)
	  gr(ip,ic)*=w_(ip)*k/4*std::pow(tw(ip)/t(ip),k-1)*arma::dot(Esi,orbdens.col(ip))/(arma::sum(orbdens.col(ip))*t(ip));
	else
	  gr(ip,ic)=0;

    increment_gga<double>(S,gr,bf_,bf_x_,bf_y_,bf_z_);
  }

  // Increment
  So.submat(bf_ind_,bf_ind_)+=S;
}

void AngularGrid::eval_Fxc(arma::mat & Ho) const {
  if(polarized_) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
  }

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());
  // No important grid points, return
  if(!screen.n_elem)
    return;

  // Work matrix
  arma::mat H(bf_ind_.n_elem,bf_ind_.n_elem);
  H.zeros();

  {
    // LDA potential
    arma::rowvec vrho(vxc_.row(0));
    // Multiply weights into potential
    vrho%=w_;
    // Increment matrix
    increment_lda<double>(H,vrho,bf_,screen);
  }

  if(do_gga_) {
    // Get vsigma
    arma::rowvec vs(vsigma_.row(0));
    // Get grad rho
    arma::uvec idx(arma::linspace<arma::uvec>(0,2,3));
    arma::mat gr(arma::trans(grho_.rows(idx)));
    // Multiply grad rho by vsigma and the weights
    for(size_t i=0;i<gr.n_rows;i++)
      for(size_t ic=0;ic<gr.n_cols;ic++)
	gr(i,ic)=2.0*w_(i)*vs(i)*gr(i,ic);
    // Increment matrix
    increment_gga<double>(H,gr,bf_,bf_x_,bf_y_,bf_z_,screen);
  }

  if(do_mgga_t_ && do_mgga_l_) {
    // Get vtau and vlapl
    arma::rowvec vt(vtau_.row(0));
    arma::rowvec vl(vlapl_.row(0));
    // Scale both with weights
    vt%=w_;
    vl%=w_;

    // Evaluate kinetic contribution
    increment_mgga_kin<double>(H,0.5*vt + 2.0*vl,bf_x_,bf_y_,bf_z_,screen);

    // Evaluate laplacian contribution. Get function laplacian
    increment_mgga_lapl<double>(H,vl,bf_,bf_lapl_,screen);

  } else if(do_mgga_t_) {
    arma::rowvec vt(vtau_.row(0));
    vt%=w_;
    increment_mgga_kin<double>(H,0.5*vt,bf_x_,bf_y_,bf_z_,screen);

  } else if(do_mgga_l_) {
    arma::rowvec vl(vlapl_.row(0));
    vl%=w_;
    increment_mgga_kin<double>(H,2.0*vl,bf_x_,bf_y_,bf_z_,screen);
    increment_mgga_lapl<double>(H,vl,bf_,bf_lapl_,screen);
  }

  Ho(bf_ind_,bf_ind_)+=H;
}

void AngularGrid::eval_diag_Fxc(arma::vec & H) const {
  if(polarized_) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted Fock matrix with unrestricted density.\n");
  }

  // Initialize memory
  H.zeros(pot_bf_ind_.n_elem);

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());
  // No important grid points, return
  if(!screen.n_elem)
    return;

  {
    // LDA potential
    arma::rowvec vrho(vxc_.row(0));
    // Multiply weights into potential
    vrho%=w_;
    // Increment matrix
    for(size_t iip=0;iip<screen.n_elem;iip++) {
      size_t ip(screen(iip));
      for(size_t j=0;j<bf_.n_rows;j++)
	H(bf_potind_(j))+=vrho(ip)*bf_(j,ip)*bf_(j,ip);
    }
  }

  if(do_gga_) {
    // Get vsigma
    arma::rowvec vs(vsigma_.row(0));
    // Get grad rho
    arma::uvec idx(arma::linspace<arma::uvec>(0,2,3));
    arma::mat gr(arma::trans(grho_.rows(idx)));
    // Multiply grad rho by vsigma and the weights
    for(size_t i=0;i<gr.n_rows;i++)
      for(size_t ic=0;ic<gr.n_cols;ic++)
	gr(i,ic)=2.0*w_(i)*vs(i)*gr(i,ic);
    // Increment matrix
    for(size_t iip=0;iip<screen.n_elem;iip++) {
      size_t ip(screen(iip));
      for(size_t j=0;j<bf_.n_rows;j++)
	H(bf_potind_(j))+=2.0 * (gr(ip,0)*bf_x_(j,ip) + gr(ip,1)*bf_y_(j,ip) + gr(ip,2)*bf_z_(j,ip)) * bf_(j,ip);
    }

    if(do_mgga_t_ && do_mgga_l_) {
      // Get vtau and vlapl
      arma::rowvec vt(vtau_.row(0));
      arma::rowvec vl(vlapl_.row(0));
      // Scale both with weights
      vt%=w_;
      vl%=w_;

      // Evaluate kinetic contribution
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++)
	  H(bf_potind_(j))+=(0.5*vt(ip)+2.0*vl(ip))*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
      }

      // Evaluate laplacian contribution.
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++)
	  H(bf_potind_(j))+=2.0*vl(ip)*bf_(j,ip)*bf_lapl_(j,ip);
      }

    } else if(do_mgga_t_) {
      arma::rowvec vt(vtau_.row(0));
      vt%=w_;

      // Evaluate kinetic contribution
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++)
	  H(bf_potind_(j))+=0.5*vt(ip)*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
      }

    } else if(do_mgga_l_) {
      arma::rowvec vl(vlapl_.row(0));
      vl%=w_;

      // Evaluate kinetic contribution
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++)
	  H(bf_potind_(j))+=2.0*vl(ip)*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
      }

      // Evaluate laplacian contribution.
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++)
	  H(bf_potind_(j))+=2.0*vl(ip)*bf_(j,ip)*bf_lapl_(j,ip);
      }
    }
  }
}

void AngularGrid::eval_Fxc(arma::mat & Hao, arma::mat & Hbo, bool beta) const {
  if(!polarized_) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
  }

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());
  // No important grid points, return
  if(!screen.n_elem)
    return;

  arma::mat Ha, Hb;
  Ha.zeros(bf_ind_.n_elem,bf_ind_.n_elem);
  if(beta)
    Hb.zeros(bf_ind_.n_elem,bf_ind_.n_elem);

  {
    // LDA potential
    arma::rowvec vrhoa(vxc_.row(0));
    // Multiply weights into potential
    vrhoa%=w_;
    // Increment matrix
    increment_lda<double>(Ha,vrhoa,bf_,screen);

    if(beta) {
      arma::rowvec vrhob(vxc_.row(1));
      vrhob%=w_;
      increment_lda<double>(Hb,vrhob,bf_,screen);
    }
  }
  if(Ha.has_nan() || (beta && Hb.has_nan()))
    throw std::logic_error("NaN encountered!\n");

  if(do_gga_) {
    // Get vsigma
    arma::rowvec vs_aa(vsigma_.row(0));
    arma::rowvec vs_ab(vsigma_.row(1));

    // Get grad rho
    arma::uvec idxa(arma::linspace<arma::uvec>(0,2,3));
    arma::uvec idxb(arma::linspace<arma::uvec>(3,5,3));
    arma::mat gr_a0(arma::trans(grho_.rows(idxa)));
    arma::mat gr_b0(arma::trans(grho_.rows(idxb)));

    // Multiply grad rho by vsigma and the weights
    arma::mat gr_a(gr_a0);
    for(size_t i=0;i<gr_a.n_rows;i++)
      for(size_t ic=0;ic<gr_a.n_cols;ic++)
	gr_a(i,ic)=w_(i)*(2.0*vs_aa(i)*gr_a0(i,ic) + vs_ab(i)*gr_b0(i,ic));
    // Increment matrix
    increment_gga<double>(Ha,gr_a,bf_,bf_x_,bf_y_,bf_z_,screen);

    if(beta) {
      arma::rowvec vs_bb(vsigma_.row(2));
      arma::mat gr_b(gr_b0);
      for(size_t i=0;i<gr_b.n_rows;i++)
	for(size_t ic=0;ic<gr_b.n_cols;ic++)
	  gr_b(i,ic)=w_(i)*(2.0*vs_bb(i)*gr_b0(i,ic) + vs_ab(i)*gr_a0(i,ic));
      increment_gga<double>(Hb,gr_b,bf_,bf_x_,bf_y_,bf_z_,screen);
    }
  }

  if(do_mgga_t_ && do_mgga_l_) {
    // Get vtau and vlapl
    arma::rowvec vt_a(vtau_.row(0));
    arma::rowvec vl_a(vlapl_.row(0));

    // Scale both with weights
    vt_a%=w_;
    vl_a%=w_;

    // Evaluate kinetic contribution
    increment_mgga_kin<double>(Ha,0.5*vt_a + 2.0*vl_a,bf_x_,bf_y_,bf_z_,screen);

    // Evaluate laplacian contribution. Get function laplacian
    increment_mgga_lapl<double>(Ha,vl_a,bf_,bf_lapl_,screen);

    if(beta) {
      arma::rowvec vt_b(vtau_.row(1));
      arma::rowvec vl_b(vlapl_.row(1));
      vt_b%=w_;
      vl_b%=w_;
      increment_mgga_kin<double>(Hb,0.5*vt_b + 2.0*vl_b,bf_x_,bf_y_,bf_z_,screen);
      increment_mgga_lapl<double>(Hb,vl_b,bf_,bf_lapl_,screen);
    }
  } else if(do_mgga_t_) {
    arma::rowvec vt_a(vtau_.row(0));
    vt_a%=w_;
    increment_mgga_kin<double>(Ha,0.5*vt_a,bf_x_,bf_y_,bf_z_,screen);
    if(beta) {
      arma::rowvec vt_b(vtau_.row(1));
      vt_b%=w_;
      increment_mgga_kin<double>(Hb,0.5*vt_b,bf_x_,bf_y_,bf_z_,screen);
    }
  } else if(do_mgga_l_) {
    arma::rowvec vl_a(vlapl_.row(0));
    vl_a%=w_;
    increment_mgga_kin<double>(Ha,2.0*vl_a,bf_x_,bf_y_,bf_z_,screen);
    increment_mgga_lapl<double>(Ha,vl_a,bf_,bf_lapl_,screen);

    if(beta) {
      arma::rowvec vl_b(vlapl_.row(1));
      vl_b%=w_;
      increment_mgga_kin<double>(Hb,2.0*vl_b,bf_x_,bf_y_,bf_z_,screen);
      increment_mgga_lapl<double>(Hb,vl_b,bf_,bf_lapl_,screen);
    }
  }

  Hao(bf_ind_,bf_ind_)+=Ha;
  if(beta)
    Hbo(bf_ind_,bf_ind_)+=Hb;
}

void AngularGrid::eval_diag_Fxc(arma::vec & Ha, arma::vec & Hb) const {
  if(!polarized_) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted Fock matrix with restricted density.\n");
  }

  // Initialize memory
  Ha.zeros(pot_bf_ind_.n_elem);
  Hb.zeros(pot_bf_ind_.n_elem);

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());
  // No important grid points, return
  if(!screen.n_elem)
    return;

  {
    // LDA potential
    arma::rowvec vrhoa(vxc_.row(0));
    // Multiply weights into potential
    vrhoa%=w_;
    arma::rowvec vrhob(vxc_.row(1));
    vrhob%=w_;
    // Increment matrix
    for(size_t iip=0;iip<screen.n_elem;iip++) {
      size_t ip(screen(iip));
      for(size_t j=0;j<bf_.n_rows;j++) {
	Ha(bf_potind_(j))+=vrhoa(ip)*bf_(j,ip)*bf_(j,ip);
	Hb(bf_potind_(j))+=vrhob(ip)*bf_(j,ip)*bf_(j,ip);
      }
    }
  }

  if(do_gga_) {
    // Get vsigma
    arma::rowvec vs_aa(vsigma_.row(0));
    arma::rowvec vs_ab(vsigma_.row(1));
    arma::rowvec vs_bb(vsigma_.row(2));
    // Get grad rho
    arma::uvec idxa(arma::linspace<arma::uvec>(0,2,3));
    arma::uvec idxb(arma::linspace<arma::uvec>(3,5,3));
    arma::mat gra0(arma::trans(grho_.rows(idxa)));
    arma::mat grb0(arma::trans(grho_.rows(idxb)));

    // Multiply grad rho by vsigma and the weights
    arma::mat gra(gra0);
    for(size_t i=0;i<gra.n_rows;i++)
      for(size_t ic=0;ic<gra.n_cols;ic++)
	gra(i,ic)=w_(i)*(2.0*vs_aa(i)*gra0(i,ic)+vs_ab(i)*grb0(i,ic));
    // Increment matrix
    for(size_t iip=0;iip<screen.n_elem;iip++) {
      size_t ip(screen(iip));
      for(size_t j=0;j<bf_.n_rows;j++)
	Ha(bf_potind_(j))+=2.0 * (gra(ip,0)*bf_x_(j,ip) + gra(ip,1)*bf_y_(j,ip) + gra(ip,2)*bf_z_(j,ip)) * bf_(j,ip);
    }

    arma::mat grb(grb0);
    for(size_t i=0;i<grb.n_rows;i++)
      for(size_t ic=0;ic<grb.n_cols;ic++)
	grb(i,ic)=w_(i)*(2.0*vs_bb(i)*grb0(i,ic)+vs_ab(i)*gra0(i,ic));
    for(size_t iip=0;iip<screen.n_elem;iip++) {
      size_t ip(screen(iip));
      for(size_t j=0;j<bf_.n_rows;j++)
	Hb(bf_potind_(j))+=2.0 * (grb(ip,0)*bf_x_(j,ip) + grb(ip,1)*bf_y_(j,ip) + grb(ip,2)*bf_z_(j,ip)) * bf_(j,ip);
    }

    if(do_mgga_t_ && do_mgga_l_) {
      // Get vtau and vlapl
      arma::rowvec vta(vtau_.row(0));
      arma::rowvec vla(vlapl_.row(0));
      arma::rowvec vtb(vtau_.row(1));
      arma::rowvec vlb(vlapl_.row(1));
      // Scale both with weights
      vta%=w_;
      vla%=w_;
      vtb%=w_;
      vlb%=w_;

      // Evaluate kinetic contribution
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++) {
	  Ha(bf_potind_(j))+=(0.5*vta(ip)+2.0*vla(ip))*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
	  Hb(bf_potind_(j))+=(0.5*vtb(ip)+2.0*vlb(ip))*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
	}
      }

      // Evaluate laplacian contribution.
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++) {
	  Ha(bf_potind_(j))+=2.0*vla(ip)*bf_(j,ip)*bf_lapl_(j,ip);
	  Hb(bf_potind_(j))+=2.0*vlb(ip)*bf_(j,ip)*bf_lapl_(j,ip);
	}
      }

    } else if(do_mgga_t_) {
      arma::rowvec vta(vtau_.row(0));
      arma::rowvec vtb(vtau_.row(1));
      vta%=w_;
      vtb%=w_;

      // Evaluate kinetic contribution
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++) {
	  Ha(bf_potind_(j))+=0.5*vta(ip)*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
	  Hb(bf_potind_(j))+=0.5*vtb(ip)*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
	}
      }

    } else if(do_mgga_l_) {
      arma::rowvec vla(vlapl_.row(0));
      arma::rowvec vlb(vlapl_.row(1));
      vla%=w_;
      vlb%=w_;

      // Evaluate kinetic contribution
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++) {
	  Ha(bf_potind_(j))+=2.0*vla(ip)*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
	  Hb(bf_potind_(j))+=2.0*vlb(ip)*(bf_x_(j,ip)*bf_x_(j,ip) + bf_y_(j,ip)*bf_y_(j,ip) + bf_z_(j,ip)*bf_z_(j,ip));
	}
      }

      // Evaluate laplacian contribution.
      for(size_t iip=0;iip<screen.n_elem;iip++) {
	size_t ip(screen(iip));
	for(size_t j=0;j<bf_.n_rows;j++) {
	  Ha(bf_potind_(j))+=2.0*vla(ip)*bf_(j,ip)*bf_lapl_(j,ip);
	  Hb(bf_potind_(j))+=2.0*vlb(ip)*bf_(j,ip)*bf_lapl_(j,ip);
	}
      }
    }
  }
}

arma::vec AngularGrid::eval_force_u() const {
  if(!polarized_) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute unrestricted force with restricted density.\n");
  }

  // Initialize force
  arma::vec f(3*basp_->get_Nnuc());
  f.zeros();

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());
  // No important grid points, return
  if(!screen.n_elem)
    return f;

  // Loop over nuclei
  for(size_t inuc=0;inuc<basp_->get_Nnuc();inuc++) {
    // Grad rho in grid points wrt functions centered on nucleus
    arma::mat gradrhoa(3,grid_.size());
    gradrhoa.zeros();
    arma::mat gradrhob(3,grid_.size());
    gradrhob.zeros();
    for(size_t iish=0;iish<shells_.size();iish++)
      if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	// Increment grad rho.
	for(size_t iip=0;iip<screen.n_elem;iip++) {
	  size_t ip(screen(iip));
	  // Loop over functions on shell
	  for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
	    gradrhoa(0,ip)+=bf_x_(mu,ip)*Pav(mu,ip);
	    gradrhoa(1,ip)+=bf_y_(mu,ip)*Pav(mu,ip);
	    gradrhoa(2,ip)+=bf_z_(mu,ip)*Pav(mu,ip);

	    gradrhob(0,ip)+=bf_x_(mu,ip)*Pbv(mu,ip);
	    gradrhob(1,ip)+=bf_y_(mu,ip)*Pbv(mu,ip);
	    gradrhob(2,ip)+=bf_z_(mu,ip)*Pbv(mu,ip);
	  }
	}
      }

    // LDA potential
    arma::rowvec vrhoa(vxc_.row(0));
    arma::rowvec vrhob(vxc_.row(1));
    // Multiply weights into potential
    vrhoa%=w_;
    vrhob%=w_;

    // Force is
    f.subvec(3*inuc,3*inuc+2) += 2.0 * (gradrhoa*arma::trans(vrhoa) + gradrhob*arma::trans(vrhob));

    if(do_gga_) {
      // Calculate X_ij = 2 \sum_{u'v} P(uv) [ x(v) d_ij x(u) + (d_i x(u)) (d_j x(v)) ]
      //                = 2 \sum_u' Pv(u) d_ij x(u) + 2 \sum Pv_i(v) d_j x(v)
      arma::mat Xa(9,grid_.size());
      Xa.zeros();
      arma::mat Xb(9,grid_.size());
      Xb.zeros();

      for(size_t iish=0;iish<shells_.size();iish++)
	if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	  // First contribution
	  for(size_t iip=0;iip<screen.n_elem;iip++) {
	    size_t ip(screen(iip));
	    for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
	      for(int c=0;c<9;c++) {
		Xa(c,ip)+=bf_hess_(9*mu+c,ip)*Pav(mu,ip);
		Xb(c,ip)+=bf_hess_(9*mu+c,ip)*Pbv(mu,ip);
	      }
	    }
	  }
	  // Second contribution
	  for(size_t iip=0;iip<screen.n_elem;iip++) {
	    size_t ip(screen(iip));
	    for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
	      // X is stored in column order: xx, yx, zx, xy, yy, zy, xz, yz, zz; but it's symmetric
	      Xa(0,ip)+=Pav_x(mu,ip)*bf_x_(mu,ip);
	      Xa(1,ip)+=Pav_x(mu,ip)*bf_y_(mu,ip);
	      Xa(2,ip)+=Pav_x(mu,ip)*bf_z_(mu,ip);

	      Xa(3,ip)+=Pav_y(mu,ip)*bf_x_(mu,ip);
	      Xa(4,ip)+=Pav_y(mu,ip)*bf_y_(mu,ip);
	      Xa(5,ip)+=Pav_y(mu,ip)*bf_z_(mu,ip);

	      Xa(6,ip)+=Pav_z_(mu,ip)*bf_x_(mu,ip);
	      Xa(7,ip)+=Pav_z_(mu,ip)*bf_y_(mu,ip);
	      Xa(8,ip)+=Pav_z_(mu,ip)*bf_z_(mu,ip);

	      Xb(0,ip)+=Pbv_x(mu,ip)*bf_x_(mu,ip);
	      Xb(1,ip)+=Pbv_x(mu,ip)*bf_y_(mu,ip);
	      Xb(2,ip)+=Pbv_x(mu,ip)*bf_z_(mu,ip);

	      Xb(3,ip)+=Pbv_y(mu,ip)*bf_x_(mu,ip);
	      Xb(4,ip)+=Pbv_y(mu,ip)*bf_y_(mu,ip);
	      Xb(5,ip)+=Pbv_y(mu,ip)*bf_z_(mu,ip);

	      Xb(6,ip)+=Pbv_z_(mu,ip)*bf_x_(mu,ip);
	      Xb(7,ip)+=Pbv_z_(mu,ip)*bf_y_(mu,ip);
	      Xb(8,ip)+=Pbv_z_(mu,ip)*bf_z_(mu,ip);
	    }
	  }
	}
      // Plug in factor
      Xa*=2.0;
      Xb*=2.0;

      // Get potential
      arma::rowvec vs_aa(vsigma_.row(0));
      arma::rowvec vs_ab(vsigma_.row(1));
      arma::rowvec vs_bb(vsigma_.row(2));
      // Get grad rho
      arma::uvec idxa(arma::linspace<arma::uvec>(0,2,3));
      arma::uvec idxb(arma::linspace<arma::uvec>(3,5,3));
      arma::mat gr_a0(arma::trans(grho_.rows(idxa)));
      arma::mat gr_b0(arma::trans(grho_.rows(idxb)));
      // Multiply grad rho by vsigma and the weights
      arma::mat gr_a(gr_a0);
      arma::mat gr_b(gr_b0);
      for(size_t i=0;i<gr_a0.n_rows;i++)
	for(size_t ic=0;ic<gr_a0.n_cols;ic++) {
	  gr_a(i,ic)=w_(i)*(2.0*vs_aa(i)*gr_a0(i,ic) + vs_ab(i)*gr_b0(i,ic));
	  gr_b(i,ic)=w_(i)*(2.0*vs_bb(i)*gr_b0(i,ic) + vs_ab(i)*gr_a0(i,ic));
	}

      // f_x <- X_xx * g_x + X_xy * g_y + X_xz * g_z
      f(3*inuc  )+=arma::as_scalar(Xa.row(0)*gr_a.col(0) + Xa.row(3)*gr_a.col(1) + Xa.row(6)*gr_a.col(2));
      f(3*inuc  )+=arma::as_scalar(Xb.row(0)*gr_b.col(0) + Xb.row(3)*gr_b.col(1) + Xb.row(6)*gr_b.col(2));
      // f_y <- X_yx * g_x + X_yy * g_y + X_yz * g_z
      f(3*inuc+1)+=arma::as_scalar(Xa.row(1)*gr_a.col(0) + Xa.row(4)*gr_a.col(1) + Xa.row(7)*gr_a.col(2));
      f(3*inuc+1)+=arma::as_scalar(Xb.row(1)*gr_b.col(0) + Xb.row(4)*gr_b.col(1) + Xb.row(7)*gr_b.col(2));
      // f_z <- X_zx * g_x + X_zy * g_y + X_zz * g_z
      f(3*inuc+2)+=arma::as_scalar(Xa.row(2)*gr_a.col(0) + Xa.row(5)*gr_a.col(1) + Xa.row(8)*gr_a.col(2));
      f(3*inuc+2)+=arma::as_scalar(Xb.row(2)*gr_b.col(0) + Xb.row(5)*gr_b.col(1) + Xb.row(8)*gr_b.col(2));

      if(do_mgga_t_ || do_mgga_l_) {
	// Kinetic energy and Laplacian terms

	// Y_i = P_uv (d_i d_j x(u)) d_j x(v)
	arma::mat Ya(3,grid_.size());
	Ya.zeros();
	arma::mat Yb(3,grid_.size());
	Yb.zeros();
	for(size_t iish=0;iish<shells_.size();iish++)
	  if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	    for(size_t iip=0;iip<screen.n_elem;iip++) {
	      size_t ip(screen(iip));
	      for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
		// Y_x =  H_xx g_x + H_xy g_y + H_xz g_z
		Ya(0,ip) += bf_hess_(9*mu  ,ip) * Pav_x(mu,ip) + bf_hess_(9*mu+3,ip) * Pav_y(mu,ip) + bf_hess_(9*mu+6,ip) * Pav_z_(mu,ip);
		Ya(1,ip) += bf_hess_(9*mu+1,ip) * Pav_x(mu,ip) + bf_hess_(9*mu+4,ip) * Pav_y(mu,ip) + bf_hess_(9*mu+7,ip) * Pav_z_(mu,ip);
		Ya(2,ip) += bf_hess_(9*mu+2,ip) * Pav_x(mu,ip) + bf_hess_(9*mu+5,ip) * Pav_y(mu,ip) + bf_hess_(9*mu+8,ip) * Pav_z_(mu,ip);

		Yb(0,ip) += bf_hess_(9*mu  ,ip) * Pbv_x(mu,ip) + bf_hess_(9*mu+3,ip) * Pbv_y(mu,ip) + bf_hess_(9*mu+6,ip) * Pbv_z_(mu,ip);
		Yb(1,ip) += bf_hess_(9*mu+1,ip) * Pbv_x(mu,ip) + bf_hess_(9*mu+4,ip) * Pbv_y(mu,ip) + bf_hess_(9*mu+7,ip) * Pbv_z_(mu,ip);
		Yb(2,ip) += bf_hess_(9*mu+2,ip) * Pbv_x(mu,ip) + bf_hess_(9*mu+5,ip) * Pbv_y(mu,ip) + bf_hess_(9*mu+8,ip) * Pbv_z_(mu,ip);
	      }
	    }
	  }

	// Z_i = 2 P_uv (lapl x_v d_i x_u + x_v lapl (d_i x_u))
	arma::mat Za, Zb;
	if(do_mgga_l_) {
	  Za.zeros(3,grid_.size());
	  Zb.zeros(3,grid_.size());
	  for(size_t iish=0;iish<shells_.size();iish++)
	    if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	      for(size_t iip=0;iip<screen.n_elem;iip++) {
		size_t ip(screen(iip));
		for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
		  // Z_x =
		  Za(0,ip) += bf_x_(mu,ip)*Palapl(mu,ip) + Pav(mu,ip)*bf_lx_(mu,ip);
		  Za(1,ip) += bf_y_(mu,ip)*Palapl(mu,ip) + Pav(mu,ip)*bf_ly_(mu,ip);
		  Za(2,ip) += bf_z_(mu,ip)*Palapl(mu,ip) + Pav(mu,ip)*bf_lz_(mu,ip);

		  Zb(0,ip) += bf_x_(mu,ip)*Pblapl_(mu,ip) + Pbv(mu,ip)*bf_lx_(mu,ip);
		  Zb(1,ip) += bf_y_(mu,ip)*Pblapl_(mu,ip) + Pbv(mu,ip)*bf_ly_(mu,ip);
		  Zb(2,ip) += bf_z_(mu,ip)*Pblapl_(mu,ip) + Pbv(mu,ip)*bf_lz_(mu,ip);
		}
	      }
	    }
	  // Put in the factor 2
	  Za*=2.0;
	  Zb*=2.0;
	}

	if(do_mgga_t_ && do_mgga_l_) {
	  // Get vtau and vlapl
	  arma::rowvec vt_a(vtau_.row(0));
	  arma::rowvec vt_b(vtau_.row(1));
	  arma::rowvec vl_a(vlapl_.row(0));
	  arma::rowvec vl_b(vlapl_.row(1));
	  // Scale both with weights
	  vt_a%=w_;
	  vt_b%=w_;
	  vl_a%=w_;
	  vl_b%=w_;

	  // Increment force
	  f.subvec(3*inuc, 3*inuc+2) += Ya*arma::trans(vt_a+4*vl_a) + Za*arma::trans(vl_a);
	  f.subvec(3*inuc, 3*inuc+2) += Yb*arma::trans(vt_b+4*vl_b) + Zb*arma::trans(vl_b);

	} else if(do_mgga_t_) {
	  arma::rowvec vt_a(vtau_.row(0));
	  arma::rowvec vt_b(vtau_.row(1));
	  vt_a%=w_;
	  vt_b%=w_;

	  // Increment force
	  f.subvec(3*inuc, 3*inuc+2) += Ya*arma::trans(vt_a);
	  f.subvec(3*inuc, 3*inuc+2) += Yb*arma::trans(vt_b);

	} else if(do_mgga_l_) {
	  arma::rowvec vl_a(vlapl_.row(0));
	  arma::rowvec vl_b(vlapl_.row(1));
	  vl_a%=w_;
	  vl_b%=w_;
	  f.subvec(3*inuc, 3*inuc+2) += Ya*arma::trans(4*vl_a) + Za*arma::trans(vl_a);
	  f.subvec(3*inuc, 3*inuc+2) += Yb*arma::trans(4*vl_b) + Zb*arma::trans(vl_b);
	}
      }
    }
  }

  return f;

}

arma::vec AngularGrid::eval_force_r() const {
  if(polarized_) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to compute restricted force with unrestricted density.\n");
  }

  // Initialize force
  arma::vec f(3*basp_->get_Nnuc());
  f.zeros();

  // Screen quadrature points by small densities
  arma::uvec screen(screen_density());
  // No important grid points, return
  if(!screen.n_elem)
    return f;

  // Loop over nuclei
  for(size_t inuc=0;inuc<basp_->get_Nnuc();inuc++) {
    // Grad rho in grid points wrt functions centered on nucleus
    arma::mat gradrho(3,grid_.size());
    gradrho.zeros();
    for(size_t iish=0;iish<shells_.size();iish++)
      if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	// Increment grad rho.
	for(size_t iip=0;iip<screen.n_elem;iip++) {
	  size_t ip(screen(iip));
	  // Loop over functions on shell
	  for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
	    gradrho(0,ip)+=bf_x_(mu,ip)*Pv(mu,ip);
	    gradrho(1,ip)+=bf_y_(mu,ip)*Pv(mu,ip);
	    gradrho(2,ip)+=bf_z_(mu,ip)*Pv(mu,ip);
	  }
	}
      }

    // LDA potential
    arma::rowvec vrho(vxc_.row(0));
    // Multiply weights into potential
    vrho%=w_;

    // Force is
    f.subvec(3*inuc,3*inuc+2) += 2.0 * gradrho*arma::trans(vrho);

    if(do_gga_) {
      // Calculate X = 2 \sum_{u'v} P(uv) [ x(v) d_ij x(u) + (d_i x(u)) (d_j x(v)) ]
      //             = 2 \sum_u' Pv(u) d_ij x(u) + 2 \sum Pv_i(v) d_j x(v)
      arma::mat X(9,grid_.size());
      X.zeros();

      for(size_t iish=0;iish<shells_.size();iish++)
	if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	  // First contribution
	  for(size_t iip=0;iip<screen.n_elem;iip++) {
	    size_t ip(screen(iip));
	    for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
	      for(int c=0;c<9;c++) {
		X(c,ip)+=bf_hess_(9*mu+c,ip)*Pv(mu,ip);
	      }
	    }
	  }
	  // Second contribution
	  for(size_t iip=0;iip<screen.n_elem;iip++) {
	    size_t ip(screen(iip));
	    for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
	      // X is stored in column order: xx, yx, zx, xy, yy, zy, xz, yz, zz; but it's symmetric
	      X(0,ip)+=Pv_x(mu,ip)*bf_x_(mu,ip);
	      X(1,ip)+=Pv_x(mu,ip)*bf_y_(mu,ip);
	      X(2,ip)+=Pv_x(mu,ip)*bf_z_(mu,ip);

	      X(3,ip)+=Pv_y(mu,ip)*bf_x_(mu,ip);
	      X(4,ip)+=Pv_y(mu,ip)*bf_y_(mu,ip);
	      X(5,ip)+=Pv_y(mu,ip)*bf_z_(mu,ip);

	      X(6,ip)+=Pv_z_(mu,ip)*bf_x_(mu,ip);
	      X(7,ip)+=Pv_z_(mu,ip)*bf_y_(mu,ip);
	      X(8,ip)+=Pv_z_(mu,ip)*bf_z_(mu,ip);
	    }
	  }
	}
      // Plug in factor
      X*=2.0;

      // Get potential
      arma::rowvec vs(vsigma_.row(0));
      // Get grad rho
      arma::uvec idx(arma::linspace<arma::uvec>(0,2,3));
      arma::mat gr(arma::trans(grho_.rows(idx)));
      // Multiply grad rho by vsigma and the weights
      for(size_t i=0;i<gr.n_rows;i++)
	for(size_t ic=0;ic<gr.n_cols;ic++)
	  gr(i,ic)=2.0*w_(i)*vs(i)*gr(i,ic);

      // f_x <- X_xx * g_x + X_xy * g_y + X_xz * g_z
      f(3*inuc  )+=arma::as_scalar(X.row(0)*gr.col(0) + X.row(3)*gr.col(1) + X.row(6)*gr.col(2));
      // f_y <- X_yx * g_x + X_yy * g_y + X_yz * g_z
      f(3*inuc+1)+=arma::as_scalar(X.row(1)*gr.col(0) + X.row(4)*gr.col(1) + X.row(7)*gr.col(2));
      // f_z <- X_zx * g_x + X_zy * g_y + X_zz * g_z
      f(3*inuc+2)+=arma::as_scalar(X.row(2)*gr.col(0) + X.row(5)*gr.col(1) + X.row(8)*gr.col(2));

      if(do_mgga_t_ || do_mgga_l_) {
	// Kinetic energy and Laplacian terms

	// Y_i = P_uv (d_i d_j x(u)) d_j x(v)
	arma::mat Y(3,grid_.size());
	Y.zeros();
	for(size_t iish=0;iish<shells_.size();iish++)
	  if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	    for(size_t iip=0;iip<screen.n_elem;iip++) {
	      size_t ip(screen(iip));
	      for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
		// Y_x =  H_xx g_x + H_xy g_y + H_xz g_z
		Y(0,ip) += bf_hess_(9*mu  ,ip) * Pv_x(mu,ip) + bf_hess_(9*mu+3,ip) * Pv_y(mu,ip) + bf_hess_(9*mu+6,ip) * Pv_z_(mu,ip);
		Y(1,ip) += bf_hess_(9*mu+1,ip) * Pv_x(mu,ip) + bf_hess_(9*mu+4,ip) * Pv_y(mu,ip) + bf_hess_(9*mu+7,ip) * Pv_z_(mu,ip);
		Y(2,ip) += bf_hess_(9*mu+2,ip) * Pv_x(mu,ip) + bf_hess_(9*mu+5,ip) * Pv_y(mu,ip) + bf_hess_(9*mu+8,ip) * Pv_z_(mu,ip);
	      }
	    }
	  }

	// Z_i = 2 P_uv (lapl x_v d_i x_u + x_v lapl (d_i x_u))
	arma::mat Z;
	if(do_mgga_l_) {
	  Z.zeros(3,grid_.size());
	  for(size_t iish=0;iish<shells_.size();iish++)
	    if(basp_->get_shell_center_ind(shells_[iish])==inuc) {
	      for(size_t iip=0;iip<screen.n_elem;iip++) {
		size_t ip(screen(iip));
		for(size_t mu=bf_i0_(iish);mu<bf_i0_(iish)+bf_N_(iish);mu++) {
		  // Z_x =
		  Z(0,ip) += bf_x_(mu,ip)*Plapl(mu,ip) + Pv(mu,ip)*bf_lx_(mu,ip);
		  Z(1,ip) += bf_y_(mu,ip)*Plapl(mu,ip) + Pv(mu,ip)*bf_ly_(mu,ip);
		  Z(2,ip) += bf_z_(mu,ip)*Plapl(mu,ip) + Pv(mu,ip)*bf_lz_(mu,ip);
		}
	      }
	    }
	  // Put in the factor 2
	  Z*=2.0;
	}

	if(do_mgga_t_ && do_mgga_l_) {
	  // Get vtau and vlapl
	  arma::rowvec vt(vtau_.row(0));
	  arma::rowvec vl(vlapl_.row(0));
	  // Scale both with weights
	  vt%=w_;
	  vl%=w_;

	  // Increment force
	  f.subvec(3*inuc, 3*inuc+2) += Y*arma::trans(vt+4*vl) + Z*arma::trans(vl);

	} else if(do_mgga_t_) {
	  arma::rowvec vt(vtau_.row(0));
	  vt%=w_;
	  f.subvec(3*inuc, 3*inuc+2) += Y*arma::trans(vt);

	} else if(do_mgga_l_) {
	  arma::rowvec vl(vlapl_.row(0));
	  vl%=w_;
	  f.subvec(3*inuc, 3*inuc+2) += Y*arma::trans(4*vl) + Z*arma::trans(vl);
	}
      }
    }
  }

  return f;
}

void AngularGrid::check_grad_tau_lapl(int x_func, int c_func) {
  // Do we need gradients?
  do_grad_=false;
  if(x_func>0)
    do_grad_=do_grad_ || gradient_needed(x_func);
  if(c_func>0)
    do_grad_=do_grad_ || gradient_needed(c_func);

  // Do we need laplacians?
  do_tau_=false;
  if(x_func>0)
    do_tau_=do_tau_ || tau_needed(x_func);
  if(c_func>0)
    do_tau_=do_tau_ || tau_needed(c_func);

  // Do we need laplacians?
  do_lapl_=false;
  if(x_func>0)
    do_lapl_=do_lapl_ || laplacian_needed(x_func);
  if(c_func>0)
    do_lapl_=do_lapl_ || laplacian_needed(c_func);
}

void AngularGrid::grad_tau_lapl(bool & grad_, bool & tau_, bool & lap_) const {
  grad_=do_grad_;
  tau_=do_tau_;
  lap_=do_lapl_;
}

void AngularGrid::set_grad_tau_lapl(bool grad_, bool tau_, bool lap_) {
  do_grad_=grad_;
  do_tau_=tau_;
  do_lapl_=lap_;
}

void AngularGrid::set_hess_lgrad(bool hess, bool lgrad) {
  do_hess_=hess;
  do_lgrad_=lgrad;
}

// Fixed size shell
angshell_t AngularGrid::construct() {
  // Form the grid.
  form_grid();
  // Return the updated info structure, holding the amount of grid
  // points and function values
  return info_;
}

void AngularGrid::next_grid() {
  if(use_lobatto_)
    info_.l+=2;
  else {
    // Need to determine what is next order of Lebedev
    // quadrature that is supported.
    info_.l=next_lebedev(info_.l);
  }
}

angshell_t AngularGrid::construct(const arma::mat & P, double ftoler, int x_func, int c_func) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Köster et al for specifics.

  // Start with
  info_.l=adaptive_l0();

  // Determine limit for angular quadrature
  int lmax=koster_lmax(ftoler);

  if(x_func == 0 && c_func == 0) {
    // No exchange or correlation!
    return info_;
  }

  // Update shell list size
  form_grid();
  if(!pot_bf_ind_.n_elem)
    // No points!
    return info_;

  // Old and new diagonal elements of Hamiltonian
  arma::vec Hold, Hnew;
  // Compute density
  update_density(P);
  // Compute exchange and correlation.
  init_xc();
  // Compute the functionals
  if(x_func>0)
    compute_xc(x_func,true);
  if(c_func>0)
    compute_xc(c_func,true);
  // Clean up xc
  check_xc();
  // Construct the Fock matrix
  eval_diag_Fxc(Hold);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual rule
  int l_enough=info_.l;
  do {
    // Increment grid
    next_grid();
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
    // Clean up xc
    check_xc();
    // Construct the Fock matrix
    eval_diag_Fxc(Hnew);

    // Compute maximum difference of diagonal elements of Fock matrix
    maxdiff=arma::max(arma::abs(Hold-Hnew));

    // Switch contents
    std::swap(Hold,Hnew);

    // Increment order if tolerance not achieved.
    if(maxdiff>ftoler) {
      l_enough = info_.l;
    }
  } while(maxdiff>ftoler && info_.l<=lmax);

  // This is the value we vant
  info_.l = l_enough;

  // Free memory
  free();

  return info_;
}

angshell_t AngularGrid::construct(const arma::mat & Pa, const arma::mat & Pb, double ftoler, int x_func, int c_func) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Köster et al for specifics.

  // Start with
  info_.l=adaptive_l0();

  if(x_func == 0 && c_func == 0) {
    // No exchange or correlation!
    return info_;
  }

  // Update shell list size
  form_grid();
  if(!pot_bf_ind_.n_elem)
    // No points!
    return info_;

  // Old and new diagonal elements of Hamiltonian
  arma::vec Haold, Hanew, Hbold, Hbnew;

  // Compute density
  update_density(Pa,Pb);

  // Compute exchange and correlation.
  init_xc();
  // Compute the functionals
  if(x_func>0)
    compute_xc(x_func,true);
  if(c_func>0)
    compute_xc(c_func,true);
  // Clean up xc
  check_xc();
  // and construct the Fock matrices
  eval_diag_Fxc(Haold,Hbold);

  // Determine limit for angular quadrature
  int lmax=koster_lmax(ftoler);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits
  int l_enough=info_.l;
  do {
    // Increment grid
    next_grid();
    // Compute grid
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
    // Clean up xc
    check_xc();
    // and construct the Fock matrices
    eval_diag_Fxc(Hanew,Hbnew);

    // Compute maximum difference of diagonal elements of Fock matrix
    maxdiff=std::max(arma::max(arma::abs(Haold-Hanew)),arma::max(arma::abs(Hbold-Hbnew)));

    // Copy contents
    std::swap(Haold,Hanew);
    std::swap(Hbold,Hbnew);

    // Increment order if tolerance not achieved.
    if(maxdiff>ftoler) {
      l_enough=info_.l;
    }
  } while(maxdiff>ftoler && info_.l<=lmax);

  // This is the value we vant
  info_.l = l_enough;

  // Free memory
  free();

  return info_;
}

angshell_t AngularGrid::construct(const arma::cx_vec & C, double ftoler, int x_func, int c_func) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Köster et al for specifics.

  // Start with
  info_.l=adaptive_l0();

  if(x_func == 0 && c_func == 0) {
    // No exchange or correlation!
    return info_;
  }

  // Update shell list size
  form_grid();
  if(!pot_bf_ind_.n_elem)
    // No points!
    return info_;

  // Determine limit for angular quadrature
  int lmax=koster_lmax(ftoler);

  // Old and new diagonal elements of Hamiltonian
  arma::vec Hold, Hnew, Hdum;
  // Compute density
  update_density(C);

  // Compute exchange and correlation.
  init_xc();
  // Compute the functionals
  if(x_func>0)
    compute_xc(x_func,true);
  if(c_func>0)
    compute_xc(c_func,true);
  // Clean up xc
  check_xc();
  // and construct the Fock matrices
  eval_diag_Fxc(Hold,Hdum);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits
  int l_enough=info_.l;
  do {
    // Increment grid
    next_grid();
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
    // Clean up xc
    check_xc();
    // and construct the Fock matrices
    eval_diag_Fxc(Hnew,Hdum);

    // Compute maximum difference of diagonal elements of Fock matrix
    maxdiff=arma::max(arma::abs(Hold-Hnew));

    // Copy contents
    std::swap(Hold,Hnew);

    // Increment order if tolerance not achieved.
    if(maxdiff>ftoler) {
      l_enough = info_.l;
    }
  } while(maxdiff>ftoler && info_.l<=lmax);

  // This is the value we want
  info_.l = l_enough;

  // Free memory once more
  free();

  return info_;
}

angshell_t AngularGrid::construct_becke(double otoler) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Krack 1998 for details

  // Start with
  info_.l=adaptive_l0();

  // Determine limit for angular quadrature
  int lmax=krack_lmax(otoler);

  // Update shell list size
  form_grid();
  if(!pot_bf_ind_.n_elem)
    // No points!
    return info_;

  // Old and new diagonal elements of overlap
  arma::vec Sold, Snew;
  // Initial value
  eval_diag_overlap(Sold);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;

  // Now, determine actual quadrature limits
  int l_enough=info_.l;
  do {
    // Increase quadrature
    next_grid();
    // Form grid
    form_grid();

    // Compute new overlap
    eval_diag_overlap(Snew);

    // Compute maximum difference of diagonal elements
    maxdiff=arma::max(arma::abs(Snew-Sold));

    // Copy contents
    std::swap(Snew,Sold);

    // Increment order if tolerance not achieved.
    if(maxdiff>otoler) {
      // Update value
      l_enough=info_.l;
    }
  } while(maxdiff>otoler && info_.l<=lmax);

  // This is the value we want
  info_.l=l_enough;

  // Free memory once more
  free();

  return info_;
}

angshell_t AngularGrid::construct_hirshfeld(const Hirshfeld & hirsh, double otoler) {
  // Construct a grid centered on (x0,y0,z0)
  // with nrad radial shells
  // See Krack 1998 for details

  // Start with
  info_.l=adaptive_l0();

  // Determine limit for angular quadrature
  int lmax=krack_lmax(otoler);

  // Old and new diagonal elements of overlap
  arma::vec Sold, Snew;

  // Form the grid
  form_hirshfeld_grid(hirsh);
  if(!pot_bf_ind_.n_elem)
    // No points!
    return info_;

  // Initial overlap
  eval_diag_overlap(Sold);

  // Maximum difference of diagonal elements of Hamiltonian
  double maxdiff;
  // Now, determine actual quadrature limits
  int l_enough=info_.l;
  do {
    // Increment grid
    next_grid();
    // Form the grid
    form_hirshfeld_grid(hirsh);

    // Compute new overlap
    eval_diag_overlap(Snew);

    // Compute maximum difference of diagonal elements
    maxdiff=arma::max(arma::abs(Snew-Sold));

    // Copy contents
    std::swap(Snew,Sold);

    // Increment order if tolerance not achieved.
    if(maxdiff>otoler) {
      l_enough=info_.l;
    }
  } while(maxdiff>otoler && info_.l<=lmax);

  // This is the value we want
  info_.l=l_enough;

  // Free memory once more
  free();

  return info_;
}

void AngularGrid::form_grid() {
  // Clear anything that already exists
  free();

  // Add grid points
  if(use_lobatto_)
    lobatto_shell();
  else
    lebedev_shell();

  // Do Becke weights
  becke_weights();
  // Prune points with small weight
  prune_points();
  // Collect weights
  compute_weights();

  // Update shell list
  update_shell_list();
  // Compute basis functions
  compute_bf();
}

void AngularGrid::form_hirshfeld_grid(const Hirshfeld & hirsh) {
  // Clear anything that already exists
  free();

  // Add grid points
  if(use_lobatto_)
    lobatto_shell();
  else
    lebedev_shell();

  // Do Becke weights
  hirshfeld_weights(hirsh);
  // Prune points with small weight
  prune_points();
  // Collect weights
  compute_weights();

  // Update shell list
  update_shell_list();
  // Compute basis functions
  compute_bf();
}

void AngularGrid::update_shell_list() {
  // Form list of important basis functions. Shell ranges
  std::vector<double> shran=basp_->get_shell_ranges();
  if(shran.size() != basp_->get_Nshells())
    throw std::logic_error("Shell ranges not initialized\n");
  // Distances to other nuclei
  std::vector<double> nucdist=basp_->get_nuclear_distances(info_.atind);

  // Current radius
  double rad=info_.R;
  // Shells that might contribute, and the amount of functions
  pot_shells_.clear();
  size_t Nbf=0;
  for(size_t inuc=0;inuc<basp_->get_Nnuc();inuc++) {
    // Closest distance of shell to nucleus. Covers both nucleus
    // inside shell, and nucleus outside the shell.
    double dist=fabs(nucdist[inuc]-rad);
    // Get indices of shells centered on nucleus
    std::vector<size_t> shellinds=basp_->get_shell_inds(inuc);

    // Loop over shells on nucleus
    for(size_t ish=0;ish<shellinds.size();ish++) {
      // Shell is relevant if range is larger than minimal distance
      if(dist<=shran[shellinds[ish]]) {
	// Add shell to list of shells to compute
	pot_shells_.push_back(shellinds[ish]);
	// Increment amount of important functions
	Nbf+=basp_->get_Nbf(shellinds[ish]);
      }
    }
  }

  // Store indices of functions
  pot_bf_ind_.zeros(Nbf);
  size_t ioff=0;
  for(size_t i=0;i<pot_shells_.size();i++) {
    // Amount of functions on shell is
    size_t Nsh=basp_->get_Nbf(pot_shells_[i]);
    // Shell offset
    size_t sh0=basp_->get_first_ind(pot_shells_[i]);
    // Indices
    arma::uvec ls=(arma::linspace<arma::uvec>(sh0,sh0+Nsh-1,Nsh));
    pot_bf_ind_.subvec(ioff,ioff+Nsh-1)=ls;
    ioff+=Nsh;
  }
}

void AngularGrid::compute_bf() {
  // Create list of shells that actually contribute. Shell ranges
  std::vector<double> shran=basp_->get_shell_ranges();
  if(shran.size() != basp_->get_Nshells())
    throw std::logic_error("Shell ranges not initialized\n");

  shells_.clear();
  size_t Nbf=0;
  for(size_t is=0;is<pot_shells_.size();is++) {
    // Shell center is
    coords_t cen(basp_->get_shell_center(pot_shells_[is]));
    // Shell range is
    double rangesq(std::pow(shran[pot_shells_[is]],2));

    // Check if the function is important on at least one point in the
    // pruned grid
    for(size_t ip=0;ip<grid_.size();ip++) {
      if(normsq(grid_[ip].r-cen)<=rangesq) {
	// Shell is important!
	shells_.push_back(pot_shells_[is]);
	Nbf+=basp_->get_Nbf(pot_shells_[is]);
	break;
      }
    }
  }

  bf_i0_.zeros(shells_.size());
  bf_N_.zeros(shells_.size());

  // Store indices of functions
  bf_ind_.zeros(Nbf);
  size_t ioff=0;
  for(size_t i=0;i<shells_.size();i++) {
    // Amount of functions on shell is
    size_t Nsh=basp_->get_Nbf(shells_[i]);
    bf_N_(i)=Nsh;
    // Shell offset
    size_t sh0=basp_->get_first_ind(shells_[i]);
    // Local offset
    bf_i0_(i)=ioff;
    // Indices
    arma::uvec ls=(arma::linspace<arma::uvec>(sh0,sh0+Nsh-1,Nsh));
    bf_ind_.subvec(ioff,ioff+Nsh-1)=ls;
    ioff+=Nsh;
  }

  // Store indices of functions on the potentials list
  bf_potind_.zeros(Nbf);
  size_t j=0;
  ioff=0;
  size_t joff=0;
  for(size_t i=0;i<pot_shells_.size() && j<shells_.size();i++) {
    // Amount of functions on shell is
    size_t Nsh=basp_->get_Nbf(pot_shells_[i]);
    // Store indices?
    if(pot_shells_[i]==shells_[j]) {
      arma::uvec ls=(arma::linspace<arma::uvec>(joff,joff+Nsh-1,Nsh));
      bf_potind_.subvec(ioff,ioff+Nsh-1)=ls;
      ioff+=Nsh;
      j++;
    }
    joff+=Nsh;
  }

  // Store number of function values
  info_.nfunc=Nbf*grid_.size();

  bf_.zeros(bf_ind_.n_elem,grid_.size());
  if(do_grad_) {
    bf_x_.zeros(bf_ind_.n_elem,grid_.size());
    bf_y_.zeros(bf_ind_.n_elem,grid_.size());
    bf_z_.zeros(bf_ind_.n_elem,grid_.size());
  }
  if(do_lapl_)
    bf_lapl_.zeros(bf_ind_.n_elem,grid_.size());
  if(do_hess_)
    bf_hess_.zeros(9*bf_ind_.n_elem,grid_.size());
  if(do_lgrad_) {
    bf_lx_.zeros(bf_ind_.n_elem,grid_.size());
    bf_ly_.zeros(bf_ind_.n_elem,grid_.size());
    bf_lz_.zeros(bf_ind_.n_elem,grid_.size());
  }

  // One pass over (ip, ish) that fuses the func / grad / lapl / hess /
  // lgrad evaluations: the fused shell-level routine reuses the
  // contracted exponentials and the power tables across all requested
  // outputs, saving the redundant exp / power builds per shell-point
  // that the separate eval_* siblings would have performed.
  arma::vec fval, lval;
  arma::mat gval, hval, lgval;
  for(size_t ip=0;ip<grid_.size();ip++) {
    ioff=0;
    for(size_t ish=0;ish<shells_.size();ish++) {
      basp_->eval_bf_derivs(shells_[ish],
                           grid_[ip].r.x, grid_[ip].r.y, grid_[ip].r.z,
                           fval, gval, lval, hval, lgval,
                           do_grad_, do_lapl_, do_hess_, do_lgrad_);
      const size_t Nf = fval.n_elem;
      bf_.submat(ioff, ip, ioff+Nf-1, ip) = fval;
      if(do_grad_) {
        bf_x_.submat(ioff, ip, ioff+Nf-1, ip) = gval.col(0);
        bf_y_.submat(ioff, ip, ioff+Nf-1, ip) = gval.col(1);
        bf_z_.submat(ioff, ip, ioff+Nf-1, ip) = gval.col(2);
      }
      if(do_lapl_)
        bf_lapl_.submat(ioff, ip, ioff+Nf-1, ip) = lval;
      if(do_hess_) {
        // hval is (Nf, 9) row-major (3x3); bf_hess is (9*Nbf, npts)
        // packed as (cart, c) -> 9*ioff + 9*f + c.
        for(size_t f=0;f<Nf;f++)
          for(int c=0;c<9;c++)
            bf_hess_(9*ioff + 9*f + c, ip) = hval(f, c);
      }
      if(do_lgrad_) {
        bf_lx_.submat(ioff, ip, ioff+Nf-1, ip) = lgval.col(0);
        bf_ly_.submat(ioff, ip, ioff+Nf-1, ip) = lgval.col(1);
        bf_lz_.submat(ioff, ip, ioff+Nf-1, ip) = lgval.col(2);
      }
      ioff += Nf;
    }
  }
}

void AngularGrid::eval_SAP(const SAP & sap, arma::mat & Vo) const {
  // List of nuclei
  std::vector<nucleus_t> nuclei(basp_->get_nuclei());

  // Form the potential at every grid point
  arma::rowvec vsap(grid_.size());
  vsap.zeros();
  for(size_t inuc=0;inuc<nuclei.size();inuc++) {
    // No potential from ghost atoms
    if(nuclei[inuc].bsse)
      continue;
    for(size_t ip=0;ip<grid_.size();ip++) {
      // Distance from grid point is
      double r=norm(nuclei[inuc].r-grid_[ip].r);
      // Increment potential
      vsap(ip)+=sap.get(nuclei[inuc].Z,r);
    }
  }
  vsap%=w_;

  // Calculate in subspace
  arma::mat V(bf_ind_.n_elem,bf_ind_.n_elem);
  V.zeros();
  increment_lda<double>(V,vsap,bf_);
  // Increment
  Vo.submat(bf_ind_,bf_ind_)+=V;
}

DFTGrid::DFTGrid() {
}

DFTGrid::DFTGrid(const BasisSet * bas, bool ver, bool lobatto) {
  basp_=bas;
  verbose_=ver;

  // Allocate atomic grids
  grids_.resize(bas->get_Nnuc());

  // Allocate work grids
#ifdef _OPENMP
  int nth=omp_get_max_threads();
  for(int i=0;i<nth;i++)
    wrk_.push_back(AngularGrid(lobatto));
#else
  wrk_.push_back(AngularGrid(lobatto));
#endif

  // Set basis set
  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].basis(*bas);
}

DFTGrid::~DFTGrid() {
}

void DFTGrid::verbose(bool ver) {
  verbose_=ver;
}

void DFTGrid::prune_shells() {
  for(size_t i=grids_.size()-1;i<grids_.size();i--)
    if(!grids_[i].np || !grids_[i].nfunc)
      grids_.erase(grids_.begin()+i);
}

void DFTGrid::construct(int nrad, int lmax, int x_func, int c_func) {
  // Check necessity of gradients and laplacians
  bool grad, tau, lapl;
  wrk_[0].check_grad_tau_lapl(x_func,c_func);
  wrk_[0].grad_tau_lapl(grad,tau,lapl);
  construct(nrad,lmax,grad,tau,lapl,false);
}

void DFTGrid::construct(int nrad, int lmax, bool grad, bool tau, bool lapl, bool nl) {
  if(verbose_) {
    if(nl)
      printf("Constructing static nrad=%i lmax=%i NL grid.\n",nrad,lmax);
    else
      printf("Constructing static nrad=%i lmax=%i XC grid.\n",nrad,lmax);
    fflush(stdout);
  }

  // Set necessity of gradienst and laplacian and grid
  for(size_t i=0;i<wrk_.size();i++) {
    wrk_[i].set_grad_tau_lapl(grad,tau,lapl);
  }

  // Set grid point screening tolerances
  double tol=settings.get_double("DFTQuadThresh");

  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev_jac(nrad,rad,wrad);
  nrad=rad.size(); // Sanity check

  // Construct grids
  size_t Nat=basp_->get_Nnuc();
  grids_.clear();
  // Loop over atoms
  for(size_t iat=0;iat<Nat;iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp_->get_nuclear_coords(iat);
    sh.tol=tol;
    sh.np=0;
    sh.nfunc=0;

    // Loop over radii
    for(int irad=0;irad<nrad;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      sh.l=lmax;
      grids_.push_back(sh);
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
    for(size_t i=0;i<grids_.size();i++) {
      wrk_[ith].set_shell(grids_[i]);
      grids_[i]=wrk_[ith].construct();
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose_) {
    std::string met=nl ? "NL" : "XC";
    print_grid(met);
  }
}


void DFTGrid::construct(const arma::mat & P, double ftoler, int x_func, int c_func) {
  // Add all atoms
  if(verbose_) {
    printf("Constructing adaptive XC grid with tolerance %e.\n",ftoler);
    koster_grid_info(ftoler);
    fflush(stdout);
  }

  Timer t;

  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].check_grad_tau_lapl(x_func,c_func);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp_->get_Nnuc());

  // Form radial shells
  for(size_t iat=0;iat<basp_->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp_->get_nuclear_coords(iat);
    sh.tol=ftoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=koster_nrad(ftoler,basp_->get_Z(iat));

    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev_jac(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;

    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      grids_.push_back(sh);
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
    for(size_t i=0;i<grids_.size();i++) {
      wrk_[ith].set_shell(grids_[i]);
      grids_[i]=wrk_[ith].construct(P,ftoler/nrad[grids_[i].atind],x_func,c_func);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose_) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    print_grid();
    fflush(stdout);
  }

}

void DFTGrid::construct(const arma::mat & Pa, const arma::mat & Pb, double ftoler, int x_func, int c_func) {
  // Add all atoms
  if(verbose_) {
    printf("Constructing adaptive XC grid with tolerance %e.\n",ftoler);
    koster_grid_info(ftoler);
    fflush(stdout);
  }

  Timer t;

  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].check_grad_tau_lapl(x_func,c_func);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp_->get_Nnuc());

  // Form radial shells
  for(size_t iat=0;iat<basp_->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp_->get_nuclear_coords(iat);
    sh.tol=ftoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=koster_nrad(ftoler,basp_->get_Z(iat));
    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev_jac(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;

    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      grids_.push_back(sh);
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
    for(size_t i=0;i<grids_.size();i++) {
      wrk_[ith].set_shell(grids_[i]);
      grids_[i]=wrk_[ith].construct(Pa,Pb,ftoler/nrad[grids_[i].atind],x_func,c_func);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose_) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    print_grid();
    fflush(stdout);
  }
}

void DFTGrid::construct(const arma::cx_mat & Ctilde, double ftoler, int x_func, int c_func) {
  // Add all atoms
  if(verbose_) {
    printf("Constructing adaptive XC grid with tolerance %e.\n",ftoler);
    koster_grid_info(ftoler);
    fflush(stdout);
  }

  Timer t;

  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].check_grad_tau_lapl(x_func,c_func);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp_->get_Nnuc());

  // Form radial shells
  for(size_t iat=0;iat<basp_->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp_->get_nuclear_coords(iat);
    sh.tol=ftoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=koster_nrad(ftoler,basp_->get_Z(iat));
    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev_jac(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;

    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      grids_.push_back(sh);
    }
  }

  // Orbital grids
  std::vector< std::vector<angshell_t> > orbgrid(grids_.size(),grids_);
  for(size_t i=0;i<orbgrid.size();i++)
    orbgrid[i].resize(Ctilde.n_cols);

  // Intel compiler complains about collapse...
  const size_t Ngrid(grids_.size());
  const size_t Norb(Ctilde.n_cols);

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
    for(size_t ig=0;ig<Ngrid;ig++)
      for(size_t iorb=0;iorb<Norb;iorb++)
	{
	  wrk_[ith].set_shell(grids_[ig]);
	  orbgrid[ig][iorb]=wrk_[ith].construct(Ctilde.col(iorb),ftoler/nrad[grids_[ig].atind],x_func,c_func);
	}
  }

  // Update l values
  for(size_t ig=0;ig<grids_.size();ig++) {
    grids_[ig].l=orbgrid[ig][0].l;
    for(size_t io=1;io<orbgrid[ig].size();io++)
      grids_[ig].l=std::max(orbgrid[ig][io].l,grids_[ig].l);
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
    for(size_t i=0;i<grids_.size();i++) {
      wrk_[ith].set_shell(grids_[i]);
      grids_[i]=wrk_[ith].construct();
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose_) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    print_grid();
    fflush(stdout);
  }
}

void DFTGrid::krack_grid_info(double otoler) const {
  printf("Maximal composition of Krack adaptive grid\n");
  printf("%3s %3s %4s %4s\n","idx","sym","nrad","lmax");
  for(size_t iat=0;iat<basp_->get_Nnuc();iat++) {
    int nr=krack_nrad(otoler,basp_->get_Z(iat));
    int nl=krack_lmax(otoler);
    printf("%3i %-3s %4i %4i\n",(int) iat+1,basp_->get_symbol(iat).c_str(),nr,nl);
  }
}

void DFTGrid::koster_grid_info(double otoler) const {
  printf("Maximal composition of Koster adaptive grid\n");
  printf("%3s %3s %4s %4s\n","idx","sym","nrad","lmax");
  for(size_t iat=0;iat<basp_->get_Nnuc();iat++) {
    int nr=koster_nrad(otoler,basp_->get_Z(iat));
    int nl=koster_lmax(otoler);
    printf("%3i %-3s %4i %4i\n",(int) iat+1,basp_->get_symbol(iat).c_str(),nr,nl);
  }
}

void DFTGrid::construct_becke(double otoler) {
  if(verbose_) {
    printf("Constructing adaptive Becke grid with tolerance %e.\n",otoler);
    krack_grid_info(otoler);
    fflush(stdout);
  }

  // Only need function values
  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].set_grad_tau_lapl(false,false,false);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp_->get_Nnuc());

  Timer t;

  // Form radial shells
  for(size_t iat=0;iat<basp_->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp_->get_nuclear_coords(iat);
    sh.tol=otoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=krack_nrad(otoler,basp_->get_Z(iat));

    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev_jac(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;

    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      grids_.push_back(sh);
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
    for(size_t i=0;i<grids_.size();i++) {
      wrk_[ith].set_shell(grids_[i]);
      grids_[i]=wrk_[ith].construct_becke(otoler/nrad[grids_[i].atind]);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose_) {
    printf("Becke grid constructed in %s.\n",t.elapsed().c_str());
    print_grid("Becke");
    fflush(stdout);
  }
}

void DFTGrid::construct_hirshfeld(const Hirshfeld & hirsh, double otoler) {
  if(verbose_) {
    printf("Constructing adaptive Hirshfeld grid with tolerance %e.\n",otoler);
    krack_grid_info(otoler);
    fflush(stdout);
  }

  // Only need function values
  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].set_grad_tau_lapl(false,false,false);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp_->get_Nnuc());

  Timer t;

  // Form radial shells
  for(size_t iat=0;iat<basp_->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp_->get_nuclear_coords(iat);
    sh.tol=otoler*PRUNETHR;

    // Compute necessary number of radial points for atom
    size_t nr=krack_nrad(otoler,basp_->get_Z(iat));

    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev_jac(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;

    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad];
      grids_.push_back(sh);
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
    for(size_t i=0;i<grids_.size();i++) {
      wrk_[ith].set_shell(grids_[i]);
      grids_[i]=wrk_[ith].construct_hirshfeld(hirsh,otoler/nrad[grids_[i].atind]);
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose_) {
    printf("Hirshfeld grid constructed in %s.\n",t.elapsed().c_str());
    print_grid("Hirshfeld");
    fflush(stdout);
  }
}

size_t DFTGrid::Npoints() const {
  size_t np=0;
  for(size_t i=0;i<grids_.size();i++)
    np+=grids_[i].np;
  return np;
}

size_t DFTGrid::Nfuncs() const {
  size_t nf=0;
  for(size_t i=0;i<grids_.size();i++)
    nf+=grids_[i].nfunc;
  return nf;
}

arma::mat DFTGrid::eval_overlap() {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrix
  arma::mat S(N,N);
  S.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#endif

    // Work array
    arma::mat Swrk(S);
    Swrk.zeros();

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Evaluate overlap
      wrk_[ith].eval_overlap(Swrk);

      // Free memory
      wrk_[ith].free();
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
  size_t N=basp_->get_Nbf();

  // Returned matrix
  arma::mat Sat(N,N);
  Sat.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#endif

    // Work array
    arma::mat Swrk(Sat);
    Swrk.zeros();

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids_.size();i++) {
      if(grids_[i].atind!=inuc)
	continue;

      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Evaluate overlap
      wrk_[ith].eval_overlap(Swrk);

      // Free memory
      wrk_[ith].free();
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
  size_t N=basp_->get_Nbf();

  // Returned matrices
  std::vector<arma::mat> Sat(basp_->get_Nnuc());
  for(size_t inuc=0;inuc<basp_->get_Nnuc();inuc++)
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Evaluate overlap
      wrk_[ith].eval_overlap(Swrk);
#ifdef _OPENMP
#pragma omp critical
#endif
      Sat[grids_[i].atind]+=Swrk;

      // Free memory
      wrk_[ith].free();
    }
  }

  return Sat;
}

arma::mat DFTGrid::eval_overlap(const arma::cx_mat & Cocc, size_t io, double k, double thr) {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrices
  arma::mat S(N,N);
  S.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int ith=omp_get_thread_num();

    // Temporary matrix
    arma::mat Swrk(N,N);
    Swrk.zeros();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Evaluate overlap
#ifdef _OPENMP
      wrk_[ith].eval_overlap(Cocc,io,k,Swrk,thr);
#else
      wrk_[ith].eval_overlap(Cocc,io,k,S,thr);
#endif
      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    S+=Swrk;
#endif
  }

  return S;
}

arma::mat DFTGrid::eval_overlap(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, double thr) {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrices
  arma::mat S(N,N);
  S.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int ith=omp_get_thread_num();

    // Temporary matrix
    arma::mat Swrk(N,N);
    Swrk.zeros();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Evaluate overlap
#ifdef _OPENMP
      wrk_[ith].eval_overlap(Cocc,Esi,k,Swrk,thr);
#else
      wrk_[ith].eval_overlap(Cocc,Esi,k,S,thr);
#endif
      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    S+=Swrk;
#endif
  }

  return S;
}

arma::mat DFTGrid::eval_tau_overlap(const arma::cx_mat & Cocc, double k, double thr) {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrices
  arma::mat S(N,N);
  S.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int ith=omp_get_thread_num();

    // Temporary matrix
    arma::mat Swrk(N,N);
    Swrk.zeros();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].set_grad_tau_lapl(true,false,false);
      wrk_[ith].form_grid();
      // Evaluate overlap
#ifdef _OPENMP
      wrk_[ith].eval_tau_overlap(Cocc,k,Swrk,thr);
#else
      wrk_[ith].eval_tau_overlap(Cocc,k,S,thr);
#endif
      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    S+=Swrk;
#endif
  }

  return S;
}

arma::mat DFTGrid::eval_tau_overlap_deriv(const arma::cx_mat & Cocc, const arma::vec & Esi, double k, double thr) {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrices
  arma::mat S(N,N);
  S.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int ith=omp_get_thread_num();

    // Temporary matrix
    arma::mat Swrk(N,N);
    Swrk.zeros();
#else
    int ith=0;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].set_grad_tau_lapl(true,false,false);
      wrk_[ith].form_grid();
      // Evaluate overlap
#ifdef _OPENMP
      wrk_[ith].eval_tau_overlap_deriv(Cocc,Esi,k,Swrk,thr);
#else
      wrk_[ith].eval_tau_overlap_deriv(Cocc,Esi,k,S,thr);
#endif
      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    S+=Swrk;
#endif
  }

  return S;
}

arma::mat DFTGrid::eval_hirshfeld_overlap(const Hirshfeld & hirsh, size_t inuc) {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrices
  arma::mat Sat(N,N);
  Sat.zeros();

#ifdef _OPENMP
  int ith=omp_get_thread_num();
#else
  int ith=0;
#endif

  // Change atom and create grid
  for(size_t i=0;i<grids_.size();i++) {
    if(grids_[i].atind!=inuc)
      continue;

    wrk_[ith].set_shell(grids_[i]);
    wrk_[ith].form_hirshfeld_grid(hirsh);
    // Evaluate overlap
    wrk_[ith].eval_overlap(Sat);
    // Free memory
    wrk_[ith].free();
  }

  return Sat;
}

std::vector<arma::mat> DFTGrid::eval_hirshfeld_overlaps(const Hirshfeld & hirsh) {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrices
  std::vector<arma::mat> Sat(basp_->get_Nnuc());
  for(size_t inuc=0;inuc<basp_->get_Nnuc();inuc++)
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_hirshfeld_grid(hirsh);
      // Evaluate overlap
      wrk_[ith].eval_overlap(Swrk);
      // Free memory
      wrk_[ith].free();

      // Increment total matrix
#ifdef _OPENMP
#pragma omp critical
#endif
      Sat[grids_[i].atind]+=Swrk;
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Update density
      wrk_[ith].update_density(P);

      // Compute helper
      hlp.clear();
      wrk_[ith].density(hlp);

#ifdef _OPENMP
#pragma omp critical
#endif
      // Add to full list
      list.insert(list.end(),hlp.begin(),hlp.end());

      // Free memory
      wrk_[ith].free();
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Update density
      wrk_[ith].update_density(P);
      // Integrate electrons
      Nel+=wrk_[ith].compute_Nel();
      // Free memory
      wrk_[ith].free();
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Update density
      wrk_[ith].update_density(Pa,Pb);
      // Integrate electrons
      Nel+=wrk_[ith].compute_Nel();
      // Free memory
      wrk_[ith].free();
    }
  }

  return Nel;
}

arma::vec DFTGrid::compute_atomic_Nel(const arma::mat & P) {
  arma::vec Nel(basp_->get_Nnuc());
  Nel.zeros();

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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Update density
      wrk_[ith].update_density(P);

      // Integrate electron density
      double dN(wrk_[ith].compute_Nel());
#ifdef _OPENMP
#pragma omp critical
#endif
      Nel(grids_[i].atind)+=dN;

      // Free memory
      wrk_[ith].free();
    }
  }

  return Nel;
}


arma::vec DFTGrid::compute_atomic_Nel(const Hirshfeld & hirsh, const arma::mat & P) {
  arma::vec Nel(basp_->get_Nnuc());
  Nel.zeros();

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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_hirshfeld_grid(hirsh);
      // Update density
      wrk_[ith].update_density(P);
      // Integrate electrons
      double dN=wrk_[ith].compute_Nel();
#ifdef _OPENMP
#pragma omp critical
#endif
      Nel[grids_[i].atind]+=dN;
      // Free memory
      wrk_[ith].free();
    }
  }

  return Nel;
}

void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::mat & P, arma::mat & H, double & Excv, double & Nelv) {
  // Clear Hamiltonian
  H.zeros(P.n_rows,P.n_cols);
  // Clear exchange-correlation energy
  double Ex=0.0, Ec=0.0;
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

#pragma omp parallel shared(Hwrk) reduction(+:Nel,Ex,Ec)
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      Timer tp;
      wrk_[ith].update_density(P);
      // Update number of electrons
      Nel+=wrk_[ith].compute_Nel();

      // Initialize the arrays
      wrk_[ith].init_xc();
      // Compute the functionals
      if(x_func>0) {
	wrk_[ith].compute_xc(x_func,true);
        wrk_[ith].check_xc();
	// Increment exchange energy
	Ex+=wrk_[ith].eval_Exc();
	// Zero out array
	wrk_[ith].zero_Exc();
      }
      if(c_func>0) {
	wrk_[ith].compute_xc(c_func,true);
        wrk_[ith].check_xc();
	// Increment exchange energy
	Ec+=wrk_[ith].eval_Exc();
	// Zero out array
	wrk_[ith].zero_Exc();
      }

      // and construct the Fock matrices

#ifdef _OPENMP
      wrk_[ith].eval_Fxc(Hwrk[ith]);
#else
      wrk_[ith].eval_Fxc(H);
#endif

      // Free memory
      wrk_[ith].free();
    }
  } // End parallel region

#ifdef _OPENMP
  // Sum results
  for(int i=0;i<maxt;i++)
    H+=Hwrk[i];
#endif

  //printf("\nExchange    energy % .10f\n",Ex);
  //printf("Correlation energy % .10f\n",Ec);

  Excv=Ex+Ec;
  Nelv=Nel;
}


void DFTGrid::eval_Fxc(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb, arma::mat & Ha, arma::mat & Hb, double & Excv, double & Nelv) {
  // Clear Hamiltonian
  Ha.zeros(Pa.n_rows,Pa.n_cols);
  Hb.zeros(Pb.n_rows,Pb.n_cols);
  // Clear exchange-correlation energy
  double Ex=0.0, Ec=0.0;
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

#pragma omp parallel shared(Hawrk,Hbwrk) reduction(+:Nel,Ex,Ec)
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
    for(size_t i=0;i<grids_.size();i++) {

      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(Pa,Pb);
      // Update number of electrons
      Nel+=wrk_[ith].compute_Nel();

      // Initialize the arrays
      wrk_[ith].init_xc();
      // Compute the functionals
      if(x_func>0) {
	wrk_[ith].compute_xc(x_func,true);
        wrk_[ith].check_xc();
	// Evaluate the energy
	Ex+=wrk_[ith].eval_Exc();
	wrk_[ith].zero_Exc();
      }
      if(c_func>0) {
	wrk_[ith].compute_xc(c_func,true);
        wrk_[ith].check_xc();
	// Evaluate the energy
	Ec+=wrk_[ith].eval_Exc();
	wrk_[ith].zero_Exc();
      }

      // and construct the Fock matrices
#ifdef _OPENMP
      wrk_[ith].eval_Fxc(Hawrk[ith],Hbwrk[ith]);
#else
      wrk_[ith].eval_Fxc(Ha,Hb);
#endif

      // Free memory
      wrk_[ith].free();
    }
  } // End parallel region

#ifdef _OPENMP
  // Sum results
  for(int i=0;i<maxt;i++) {
    Ha+=Hawrk[i];
    Hb+=Hbwrk[i];
  }
#endif

  //printf("\nExchange    energy % .10f\n",Ex);
  //printf("Correlation energy % .10f\n",Ec);

  Excv=Ex+Ec;
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
    for(size_t i=0;i<grids_.size();i++) {

      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Loop over densities
      for(size_t ip=0;ip<nocc;ip++) {
	// Update density
	wrk_[ith].update_density(CW.col(ip));
	// Update number of electrons
#ifdef _OPENMP
#pragma omp atomic
#endif
	Nel[ip]+=wrk_[ith].compute_Nel();

	// Initialize the arrays
	wrk_[ith].init_xc();
	// Compute the functionals
	if(x_func>0)
	  wrk_[ith].compute_xc(x_func,fock);
	if(c_func>0)
	  wrk_[ith].compute_xc(c_func,fock);
        wrk_[ith].check_xc();

	// Evaluate the energy
#ifdef _OPENMP
#pragma omp atomic
#endif
	Exc[ip]+=wrk_[ith].eval_Exc();
	// and construct the Fock matrices
	if(fock) {
	  Hwrk.zeros(); // need to clear this here

	  wrk_[ith].eval_Fxc(Hwrk,Hdum,false);

#ifdef _OPENMP
#pragma omp critical
#endif
	  H[ip]+=Hwrk;
	}
      }
    }

    // Free memory
    wrk_[ith].free();

  } // End parallel region
}

void DFTGrid::eval_VV10(DFTGrid & nl, double b, double C, const arma::mat & P, arma::mat & H, double & Enl_, bool fock) {
  // Reset energy
  double Enl=0.0;

  // Original gradient and laplacian settings
  bool grad, tau, lapl;
  wrk_[0].grad_tau_lapl(grad,tau,lapl);

  // Collect nl grid data
  std::vector<arma::mat> nldata(nl.grids_.size());
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
    for(size_t i=0;i<nl.grids_.size();i++) {
      // Change atom
      wrk_[ith].set_shell(nl.grids_[i]);
      wrk_[ith].set_grad_tau_lapl(true,false,false);
      // Create grid
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(P);

      // Collect the data
      wrk_[ith].init_VV10(b,C,false);
      std::vector<size_t> idx;
      wrk_[ith].collect_VV10(nldata[i],idx,b,C,true);

      wrk_[ith].free();
    }
  } // End parallel region

  if(nl.verbose_) {
    size_t n=0;
    for(size_t i=0;i<nldata.size();i++)
      n+=nldata[i].n_cols;

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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].set_grad_tau_lapl(true,false,false);
      // Initialize worker
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(P);
      // Initialize the arrays
      wrk_[ith].init_xc();

      // Calculate the integral over the nl grid
      wrk_[ith].init_VV10(b,C,true);
      wrk_[ith].compute_VV10(nldata,b,C);

      // Evaluate the energy
      Enl+=wrk_[ith].eval_Exc();
      // and construct the Fock matrices
      if(fock)
	wrk_[ith].eval_Fxc(Hwrk);

      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    if(fock)
      H+=Hwrk;
  }

  // Set return variable
  Enl_=Enl;

  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].set_grad_tau_lapl(grad,tau,lapl);
}

arma::vec DFTGrid::eval_force(int x_func, int c_func, const arma::mat & P) {
  arma::vec f(3*basp_->get_Nnuc());
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
    for(size_t iat=0;iat<grids_.size();iat++) {
      bool grad, tau, lapl;
      wrk_[ith].grad_tau_lapl(grad,tau,lapl);
      // We need gradients for the LDA terms and Laplacian terms for the GGA terms (Pv_i really)
      wrk_[ith].set_grad_tau_lapl(true,grad,grad);
      // Need bf Hessian for GGA and laplacian gradient for MGGA
      wrk_[ith].set_hess_lgrad(grad,lapl);

      // Change atom and create grid
      wrk_[ith].set_shell(grids_[iat]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(P,true);

      // Initialize the arrays
      wrk_[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk_[ith].compute_xc(x_func,true);
      if(c_func>0)
	wrk_[ith].compute_xc(c_func,true);
      wrk_[ith].check_xc();

      // Calculate the force on the atom
#ifdef _OPENMP
      fwrk+=wrk_[ith].eval_force_r();
#else
      f+=wrk_[ith].eval_force_r();
#endif

      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    f+=fwrk;
#endif
  } // End parallel region

  return f;
}

arma::vec DFTGrid::eval_force(int x_func, int c_func, const arma::mat & Pa, const arma::mat & Pb) {
  arma::vec f(3*basp_->get_Nnuc());
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
    for(size_t iat=0;iat<grids_.size();iat++) {
      // We need gradients for the LDA terms
      bool grad, tau, lapl;
      wrk_[ith].grad_tau_lapl(grad,tau,lapl);
      wrk_[ith].set_grad_tau_lapl(true,grad,grad);
      // Need bf Hessian for GGA and laplacian gradient for MGGA
      wrk_[ith].set_hess_lgrad(grad,lapl);

      // Change atom and create grid
      wrk_[ith].set_shell(grids_[iat]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(Pa,Pb,true);

      // Initialize the arrays
      wrk_[ith].init_xc();
      // Compute the functionals
      if(x_func>0)
	wrk_[ith].compute_xc(x_func,true);
      if(c_func>0)
	wrk_[ith].compute_xc(c_func,true);
      wrk_[ith].check_xc();

      // Calculate the force on the atom
#ifdef _OPENMP
      fwrk+=wrk_[ith].eval_force_u();
#else
      f+=wrk_[ith].eval_force_u();
#endif

      // Free memory
      wrk_[ith].free();
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
  arma::vec f(3*basp_->get_Nnuc());
  f.zeros();

  // Original gradient and laplacian settings
  bool grad, tau, lapl;
  wrk_[0].grad_tau_lapl(grad,tau,lapl);

  // Collect nl grid data
  std::vector<arma::mat> nldata(nl.grids_.size());
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
    for(size_t i=0;i<nl.grids_.size();i++) {
      // Need gradient for VV10 but no laplacian
      wrk_[ith].set_grad_tau_lapl(true,false,false);
      // No need for Hessian or laplacian of gradient
      wrk_[ith].set_hess_lgrad(false,false);
      // Change atom and create grid
      wrk_[ith].set_shell(nl.grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(P);

      // Collect the data
      wrk_[ith].init_VV10(b,C,false);
      std::vector<size_t> idx;
      wrk_[ith].collect_VV10(nldata[i],idx,b,C,true);

      // Free memory
      wrk_[ith].free();

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
    for(size_t i=0;i<grids_.size();i++) {
      // Need gradient for VV10 and laplacian for VV10 gradient
      wrk_[ith].set_grad_tau_lapl(true,true,true);
      // Need hessian for VV10 gradient
      wrk_[ith].set_hess_lgrad(true,false);
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      Timer tp;
      wrk_[ith].update_density(P);
      // Initialize the arrays
      wrk_[ith].init_xc();

      // Calculate kappa and omega and plug in constant energy density
      wrk_[ith].init_VV10(b,C,true);
      // Evaluate the VV10 energy and potential and get the grid contribution
      fwrk.subvec(3*grids_[i].atind,3*grids_[i].atind+2)+=wrk_[ith].compute_VV10_F(nldata,nl.grids_,b,C);
      // and now evaluate the forces on the atoms
      fwrk+=wrk_[ith].eval_force_r();

      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    f+=fwrk;
  }

  for(size_t i=0;i<wrk_.size();i++)
    wrk_[i].set_grad_tau_lapl(grad,tau,lapl);

  return f;
}

void DFTGrid::print_density(const arma::mat & P, std::string densname) {
  // Open output files
  FILE *dens=fopen(densname.c_str(),"w");

  fprintf(dens,"%i\n",(int) Npoints());

  Timer t;
  if(verbose_) {
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(P);

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk_[ith].print_density(dens);
      }

      // Free memory
      wrk_[ith].free();
    }
  } // End parallel region

    // Close output files
  fclose(dens);

  printf("done (%s)\n",t.elapsed().c_str());
}

void DFTGrid::print_density(const arma::mat & Pa, const arma::mat & Pb, std::string densname) {
  // Open output files
  FILE *dens=fopen(densname.c_str(),"w");

  fprintf(dens,"%i\n",(int) Npoints());

  Timer t;
  if(verbose_) {
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(Pa,Pb);

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk_[ith].print_density(dens);
      }

      // Free memory
      wrk_[ith].free();
    }
  } // End parallel region

    // Close output files
  fclose(dens);

  printf("done (%s)\n",t.elapsed().c_str());
}

void DFTGrid::print_potential(int func_id, const arma::mat & Pa, const arma::mat & Pb, std::string potname) {
  // Open output files
  FILE *pot=fopen(potname.c_str(),"w");
  fprintf(pot,"%i\n",(int) Npoints());

  Timer t;
  if(verbose_) {
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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(Pa,Pb);

      // Initialize the arrays
      wrk_[ith].init_xc();
      // Compute the functionals
      if(func_id>0)
	wrk_[ith].compute_xc(func_id,true);
      wrk_[ith].check_xc();

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk_[ith].print_potential(func_id,pot);
      }

      // Free memory
      wrk_[ith].free();
    }
  } // End parallel region

    // Close output files
  fclose(pot);

  printf("done (%s)\n",t.elapsed().c_str());
}

void DFTGrid::check_potential(int func_id, const arma::mat & Pa, const arma::mat & Pb, std::string potname) {
  // Open output files
  FILE *pot=fopen(potname.c_str(),"w");

  Timer t;
  if(verbose_) {
    printf("\nRunning potential check. Saving output to %s ... ",potname.c_str());
    fflush(stdout);
  }

  fprintf(pot,"%23s %23s %23s %23s %23s %23s %23s %23s %23s\n","rhoa","rhob","sigmaaa","sigmaab","sigmabb","lapla","laplb","taua","taub");
  fprintf(pot,"%23s %23s %23s %23s %23s %23s %23s %23s %23s %23s\n","exc","vrhoa","vrhob","vsigmaaa","vsigmaab","vsigmabb","vlapla","vlaplb","vtaua","vtaub");

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
    for(size_t i=0;i<grids_.size();i++) {
      // Change atom and create grid
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();

      // Update density
      wrk_[ith].update_density(Pa,Pb);

      // Initialize the arrays
      wrk_[ith].init_xc();
      // Compute the functionals
      if(func_id>0)
	wrk_[ith].compute_xc(func_id,true);
      wrk_[ith].check_xc();

      // Write out density and potential data
#ifdef _OPENMP
#pragma omp critical
#endif
      {
	wrk_[ith].check_potential(pot);
      }

      // Free memory
      wrk_[ith].free();
    }
  } // End parallel region

  // Close output files
  fclose(pot);

  printf("done (%s)\n",t.elapsed().c_str());
}

void DFTGrid::print_grid(std::string met) const {
  // Amount of integration points
  arma::uvec np(basp_->get_Nnuc());
  np.zeros();
  // Amount of function values
  arma::uvec nf(basp_->get_Nnuc());
  nf.zeros();

  for(size_t i=0;i<grids_.size();i++) {
    np(grids_[i].atind)+=grids_[i].np;
    nf(grids_[i].atind)+=grids_[i].nfunc;
  }
  /*
  printf("Grid info:\n");
  for(size_t i=0;i<grids.size();i++)
    printf("%6i %4i %8.3e %e %2i %7i %10i\n",(int) i, (int) grids[i].atind, grids[i].R, grids[i].w, grids[i].l, (int) grids[i].np, (int) grids[i].nfunc);
  */

  printf("Composition of %s grid:\n %7s %7s %10s\n",met.c_str(),"atom","Npoints","Nfuncs");
  for(size_t i=0;i<basp_->get_Nnuc();i++)
    printf(" %4i %-2s %7i %10i\n",(int) i+1, basp_->get_symbol(i).c_str(), (int) np(i), (int) nf(i));
}

double DFTGrid::density_threshold(const arma::mat & P, double thr) {
  // Get the list of orbital density values
  std::vector<dens_list_t> list=eval_dens_list(P);

  // Get cutoff
  double itg=0.0;
  size_t idx=0;
  while(itg<thr && idx<list.size()) {
    // Increment integral
    itg+=list[idx].d*list[idx].w;
    // Increment index
    idx++;
  }

  // Cutoff is thus between idx and idx-1.
  return (list[idx].d + list[idx-1].d)/2.0;
}

arma::mat DFTGrid::eval_SAP() {
  // Amount of basis functions
  size_t N=basp_->get_Nbf();

  // Returned matrices
  arma::mat V(N,N);
  V.zeros();

  // SAPs
  SAP sap;

  // Change atom and create grid
#ifdef _OPENMP
#pragma omp parallel
#endif
  { // Begin parallel region

#ifdef _OPENMP
    int ith=omp_get_thread_num();
#else
    int ith=0;
#endif

#ifdef _OPENMP
    arma::mat Vwrk(N,N);
    Vwrk.zeros();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids_.size();i++) {
      wrk_[ith].set_shell(grids_[i]);
      wrk_[ith].form_grid();
      // Evaluate overlap
#ifdef _OPENMP
      wrk_[ith].eval_SAP(sap,Vwrk);
#else
      wrk_[ith].eval_SAP(sap,V);
#endif
      // Free memory
      wrk_[ith].free();
    }

#ifdef _OPENMP
#pragma omp critical
    V+=Vwrk;
#endif
  }

  return V;
}
