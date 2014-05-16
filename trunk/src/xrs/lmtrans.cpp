/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
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
#include "mathf.h"
#include "lmtrans.h"
#include "timer.h"
#include "spherical_harmonics.h"

// Tolerance for differing value of Q in Bessel table
#define QTOL (1.01*q*DBL_EPSILON)

lmtrans::lmtrans() {
}

lmtrans::lmtrans(const arma::mat & C, const BasisSet & bas, const coords_t & cen, size_t Nrad, int l, int lquad) {
  exp=expand_orbitals(C,bas,cen,true,Nrad,l,lquad);

  // Compute maximum value of l
  lmax=0;
  while(lmind(lmax,lmax)<exp.clm[0].size())
    lmax++;
  // We ran over, so decrease value
  lmax--;

  // Compute table of Gaunt coefficients
  gaunt=Gaunt(2*lmax,lmax,lmax);
}

lmtrans::~lmtrans() {
}

arma::cx_mat lmtrans::radial_integral(size_t i, size_t f, int l, const bessel_t & bes) const {
  // Returned matrix
  arma::cx_mat ret(exp.clm[0].size(),exp.clm[0].size());
  ret.zeros();

 // Loop over initial angular momentum
  for(int li=0;li<=lmax;li++)
    for(int mi=-li;mi<=li;mi++) {
      // Index of initial state in output
      size_t iind=lmind(li,mi);

      // Loop over final angular momentum
      for(int lf=0;lf<=lmax;lf++)
	for(int mf=-lf;mf<=lf;mf++) {
	  // Index of final state in output
	  size_t find=lmind(lf,mf);

	  // Value of integral
	  std::complex<double> itg;

	  // Loop over quadrature points
	  for(size_t ip=0;ip<exp.grid.size();ip++) {
	    // Increment integral. Quadrature weight already includes r^2!
	    itg+=conj(exp.clm[f][find][ip])*exp.clm[i][iind][ip]*bes.jl[l][ip]*exp.grid[ip].w;
	  }

	  // Store the result, plugging in the factor 4 \pi i^l
	  ret(iind,find)=4.0*M_PI*itg*pow(std::complex<double>(0.0,1.0),l);
	}
    }

  return ret;
}

rad_int_t lmtrans::compute_radial_integrals(size_t i, size_t f, const bessel_t & bes) const {
  // Returned integrals
  rad_int_t rad;
  // Allocate storage
  rad.itg.resize(2*lmax+1);
  // Compute all of the integrals
  for(int l=0;l<=2*lmax;l++)
    rad.itg[l]=radial_integral(i,f,l,bes);

  return rad;
}

bessel_t lmtrans::compute_bessel(double q) const {
  /*
    Timer t;

    printf("Computing Bessel functions for q=%e ... ",q);
    fflush(stdout);
  */

  // Returned array
  bessel_t bes;

  // Store value of q
  bes.q=q;
  // Resize array
  bes.jl.resize(2*lmax+1);
  for(int l=0;l<=2*lmax;l++)
    bes.jl[l].resize(exp.grid.size());

  // Compute values of Bessel function
  for(int l=0;l<=2*lmax;l++)
    for(size_t ip=0;ip<exp.grid.size();ip++)
      bes.jl[l][ip]=bessel_jl(l,q*exp.grid[ip].r);

  /*
  printf("done (%s)\n",t.elapsed().c_str());
  printf("%lu were fully computed and %lu were determined small.\n",ncomp,nser);
  */

  return bes;
}


arma::cx_mat lmtrans::transition_amplitude(const rad_int_t & rad, double qx, double qy, double qz) const {
  // Compute cos theta and phi
  double q=sqrt(qx*qx+qy*qy+qz*qz);
  double cth=qz/q;
  double phi=atan2(qy,qx);

  // Returned array
  arma::cx_mat A(lmax+1,lmax+1);

  // Compute spherical harmonics.
  std::vector< std::complex<double> > conj_sphharm(lmind(2*lmax,2*lmax)+1);
  for(int l=0;l<=2*lmax;l++)
    for(int m=-l;m<=l;m++)
      conj_sphharm[lmind(l,m)]=conj(spherical_harmonics(l,m,cth,phi));

  // Loop over l_i, l_f
  for(int li=0;li<=lmax;li++)
    for(int lf=0;lf<=lmax;lf++) {

      // A(l_i,l_f)
      std::complex<double> aif=0.0;

      // Loop over l
      for(int l=abs(li-lf);l<=li+lf;l++)
	for(int mi=-li;mi<=li;mi++)
	  for(int mf=-lf;mf<=lf;mf++) {
	    // m has to be
	    int m=mi+mf;
	    // but it can't be larger than l
	    if(abs(m)>l)
	      continue;
	    // in order for the Gaunt coefficient not to vanish.

	    // Increment result.
	    aif+=rad.itg[l](lmind(li,mi),lmind(lf,mf))*gaunt.coeff(lf,mf,l,m,li,mi)*conj_sphharm[lmind(l,m)];
	  }

      // Store result
      A(li,lf)=aif;
    }

  return A;
}

void lmtrans::print_info() const {
  // Get decomposition
  arma::mat dec=::weight_decomposition(exp,true);
  for(size_t io=0;io<exp.clm.size();io++) {
    printf("Orbital %3i: ",(int) io+1);
    for(int l=0;l<=lmax;l++)
      printf(" %.2e",dec(io,l));
    printf(" norm %e\n",dec(io,lmax+1));
  }
}

std::vector<double> lmtrans::transition_velocity(size_t is, size_t fs, const bessel_t & bes) const {
  // Compute radial integrals
  rad_int_t rad=compute_radial_integrals(is,fs,bes);

  // Get angular mesh for directional average. Even though in
  // principle this should be 2*lmax to be exact, we probably don't
  // need to go that far.
  //  std::vector<angular_grid_t> mesh=form_angular_grid(lmax);
  std::vector<angular_grid_t> mesh=form_angular_grid(2*lmax);

  // We normalize the weights so that for purely dipolar transitions we
  // get the same output as with using the dipole matrix.
  for(size_t ig=0;ig<mesh.size();ig++) {
    // Dipole integral is only wrt theta - divide off phi part.
    mesh[ig].w/=2.0*M_PI;
  }

  // Transition velocities.
  std::vector<double> t(lmax+2);
  for(size_t i=0;i<t.size();i++)
    t[i]=0.0;

  // Perform directional average
  for(size_t iq=0;iq<mesh.size();iq++) {
    // Compute x, y and z component of q
    double qx=mesh[iq].r.x;
    double qy=mesh[iq].r.y;
    double qz=mesh[iq].r.z;
    // Get transition speed for current (direction of) q
    arma::cx_mat A=transition_amplitude(rad,qx,qy,qz);
    // and sum over all elements
    std::complex<double> Atot=accu(A);

    // Increment spherical average of transition velocity
    t[0]+=mesh[iq].w*norm(Atot);

    // Compute final state stuff:
    arma::cx_mat fsm=A*conj(A);
    for(int l=0;l<=lmax;l++)
      t[1+l]+=mesh[iq].w*fsm(l,l).real();
  }

  return t;
}

void lmtrans::write_prob(size_t o, const std::string &fname) const {
  // Extra parameters to print: radius, probability density and
  // cumulative density
  const int npar=3;

  // Output array.
  arma::mat arr(exp.grid.size(),npar+lmax+1);
  arr.zeros();

  // Loop over radii
  for(size_t ir=0;ir<exp.grid.size();ir++) {
    // Radius is
    arr(ir,0)=exp.grid[ir].r;

    // Compute probabilities of different values of l
    for(int l=0;l<=lmax;l++) {
      for(int m=-l;m<=l;m++)
	arr(ir,npar+l)+=norm(exp.clm[o][lmind(l,m)][ir]);
      // Increment probability density
      arr(ir,1)+=arr(ir,npar+l);
    }
  }

  // Compute total probability density, weight already includes r^2.
  for(size_t ir=1;ir<exp.grid.size();ir++) {
    arr(ir,2)=arr(ir-1,2)+exp.grid[ir].w*arr(ir,1);
  }

  // Print output: radius and total probability
  FILE *out=fopen(fname.c_str(),"w");
  for(size_t i=0;i<arr.n_rows;i++) {
    for(size_t j=0;j<arr.n_cols;j++)
      fprintf(out,"%e\t",arr(i,j));
    fprintf(out,"\n");
  }
  fclose(out);
}
