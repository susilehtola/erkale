/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "global.h"
#include "basis.h"
#include "mathf.h"
#include "checkpoint.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

typedef struct {
  coords_t r;
  bool newline;
} cubecoord_t;


void density_cube(const BasisSet & bas, const arma::mat & P, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr, std::string fname, double & norm, bool sic) {
  // Open output file.
  fname=fname+".cube";
  FILE *out=fopen(fname.c_str(),"w");

  // Compute the norm (assumes evenly spaced grid!)
  norm=0.0;

  // Compute the density in batches, allowing
  // parallellization.
#ifdef _OPENMP
  // The number of points per batch
  const size_t Nbatch_p=100*omp_get_max_threads();
#else
  const size_t Nbatch_p=100;
#endif

  // The total number of point is
  const size_t N=x_arr.size()*y_arr.size()*z_arr.size();
  // The necessary amount of batches is
  size_t Nbatch=N/Nbatch_p;
  if(N%Nbatch_p!=0)
    Nbatch++;

  // Write out comment fields
  Timer t;
  fprintf(out,"ERKALE electron density output\n");
  fprintf(out,"Generated on %s.\n",t.current_time().c_str());

  // Spacing
  double dx=0.0;
  if(x_arr.size()>1)
    dx=(x_arr[x_arr.size()-1]-x_arr[0])/(x_arr.size()-1);
  double dy=0.0;
  if(y_arr.size()>1)
    dy=(y_arr[y_arr.size()-1]-y_arr[0])/(y_arr.size()-1);
  double dz=0.0;
  if(z_arr.size()>1)
    dz=(z_arr[z_arr.size()-1]-z_arr[0])/(z_arr.size()-1);

  // Write out starting point
  fprintf(out,"%7i % e % e % e\n",(int) bas.get_Nnuc(),x_arr[0],y_arr[0],z_arr[0]);
  // Print amount of points and step sizes in the directions
  fprintf(out,"%7i % e % e % e\n",(int) x_arr.size(),dx,0.0,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) y_arr.size(),0.0,dy,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) z_arr.size(),0.0,0.0,dz);
  // Print out atoms
  for(size_t i=0;i<bas.get_Nnuc();i++) {
    nucleus_t nuc=bas.get_nucleus(i);
    fprintf(out,"%7i %e % e % e % e\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
  }

  // The points in the batch
  cubecoord_t r[Nbatch_p];
  // The values of the density in the batch
  double rho[Nbatch_p];

  // Number of points to compute in the batch
  size_t np;
  // Total number of points computed
  size_t ntot=0;
  // Indices of x, y and z
  size_t xind=0, yind=0, zind=0;

  // Index of points written
  size_t idx=0;

  // Loop over batches.
  for(size_t ib=0;ib<Nbatch;ib++) {
    // Zero amount of points in current batch.
    np=0;

    // Form list of points to compute.
    while(np<Nbatch_p && ntot+np<N) {
      r[np].r.x=x_arr[xind];
      r[np].r.y=y_arr[yind];
      r[np].r.z=z_arr[zind];
      r[np].newline=false;

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	// z coordinate changes, break line here
	zind=0;
	r[np-1].newline=true;

	if(yind+1<y_arr.size())
	  yind++;
	else {
	  yind=0;
	  xind++;
	}
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    // Loop over the points in the batch
    for(size_t ip=0;ip<np;ip++)
      rho[ip]=compute_density(P,bas,r[ip].r);

    // Save density values
    for(size_t ip=0;ip<np;ip++) {
      norm+=rho[ip];

      if(sic)
	// For consistency with real case, take sqrt of complex orbitals
	fprintf(out," % .5e",sqrt(rho[ip]));
      else
	fprintf(out," % .5e",rho[ip]);
      idx++;
      if(idx==6 || r[ip].newline) {
	idx=0;
	fprintf(out,"\n");
      }
    }

    // Increment number of computed points
    ntot+=np;
  }

  // Close output file.
  if(idx!=0)
    fprintf(out,"\n");
  fclose(out);

  // Plug in the spacing in the integral
  norm*=dx*dy*dz;
}

void densitydiff_cube(const BasisSet & bas, const arma::mat & P, const BasisSet & basref, const arma::mat & Pref, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr, std::string fname, double & norm) {
  // Open output file.
  fname=fname+".cube";
  FILE *out=fopen(fname.c_str(),"w");

  // Compute the norm (assumes evenly spaced grid!)
  norm=0.0;

  // Compute the density in batches, allowing
  // parallellization.
#ifdef _OPENMP
  // The number of points per batch
  const size_t Nbatch_p=100*omp_get_max_threads();
#else
  const size_t Nbatch_p=100;
#endif

  // The total number of point is
  const size_t N=x_arr.size()*y_arr.size()*z_arr.size();
  // The necessary amount of batches is
  size_t Nbatch=N/Nbatch_p;
  if(N%Nbatch_p!=0)
    Nbatch++;

  // Write out comment fields
  Timer t;
  fprintf(out,"ERKALE electron density output\n");
  fprintf(out,"Generated on %s.\n",t.current_time().c_str());

  // Spacing
  double dx=0.0;
  if(x_arr.size()>1)
    dx=(x_arr[x_arr.size()-1]-x_arr[0])/(x_arr.size()-1);
  double dy=0.0;
  if(y_arr.size()>1)
    dy=(y_arr[y_arr.size()-1]-y_arr[0])/(y_arr.size()-1);
  double dz=0.0;
  if(z_arr.size()>1)
    dz=(z_arr[z_arr.size()-1]-z_arr[0])/(z_arr.size()-1);

  // Write out starting point
  fprintf(out,"%7i % e % e % e\n",(int) basref.get_Nnuc(),x_arr[0],y_arr[0],z_arr[0]);
  // Print amount of points and step sizes in the directions
  fprintf(out,"%7i % e % e % e\n",(int) x_arr.size(),dx,0.0,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) y_arr.size(),0.0,dy,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) z_arr.size(),0.0,0.0,dz);
  // Print out atoms
  for(size_t i=0;i<bas.get_Nnuc();i++) {
    nucleus_t nuc=bas.get_nucleus(i);
    fprintf(out,"%7i %g % e % e % e\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
  }

  // The points in the batch
  cubecoord_t r[Nbatch_p];
  // The values of the density in the batch
  double rho[Nbatch_p];

  // Number of points to compute in the batch
  size_t np;
  // Total number of points computed
  size_t ntot=0;
  // Indices of x, y and z
  size_t xind=0, yind=0, zind=0;

  // Index of points written
  size_t idx=0;

  // Loop over batches.
  for(size_t ib=0;ib<Nbatch;ib++) {
    // Zero amount of points in current batch.
    np=0;

    // Form list of points to compute.
    while(np<Nbatch_p && ntot+np<N) {
      r[np].r.x=x_arr[xind];
      r[np].r.y=y_arr[yind];
      r[np].r.z=z_arr[zind];
      r[np].newline=false;

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	// z coordinate changes, break line here
	zind=0;
	r[np-1].newline=true;

	if(yind+1<y_arr.size())
	  yind++;
	else {
	  yind=0;
	  xind++;
	}
      }
    }

    if(bas == basref)
#ifdef _OPENMP
#pragma omp parallel for
#endif
      // Loop over the points in the batch
      for(size_t ip=0;ip<np;ip++)
	rho[ip]=compute_density(P-Pref,bas,r[ip].r);
    else
#ifdef _OPENMP
#pragma omp parallel for
#endif
      // Loop over the points in the batch
      for(size_t ip=0;ip<np;ip++)
	rho[ip]=compute_density(P,bas,r[ip].r)-compute_density(Pref,basref,r[ip].r);

    // Save density values
    for(size_t ip=0;ip<np;ip++) {
      norm+=rho[ip];

      fprintf(out," % .5e",rho[ip]);
      idx++;
      if(idx==6 || r[ip].newline) {
	idx=0;
	fprintf(out,"\n");
      }
    }

    // Increment number of computed points
    ntot+=np;
  }

  // Close output file.
  if(idx!=0)
    fprintf(out,"\n");
  fclose(out);

  // Plug in the spacing in the integral
  norm*=dx*dy*dz;
}

void potential_cube(const BasisSet & bas, const arma::mat & P, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr, std::string fname) {
  // Open output file.
  fname=fname+".cube";
  FILE *out=fopen(fname.c_str(),"w");

  // Compute the density in batches, allowing
  // parallellization.
#ifdef _OPENMP
  // The number of points per batch
  const size_t Nbatch_p=100*omp_get_max_threads();
#else
  const size_t Nbatch_p=100;
#endif

  // The total number of point is
  const size_t N=x_arr.size()*y_arr.size()*z_arr.size();
  // The necessary amount of batches is
  size_t Nbatch=N/Nbatch_p;
  if(N%Nbatch_p!=0)
    Nbatch++;

  // Write out comment fields
  Timer t;
  fprintf(out,"ERKALE electrostatic potential output\n");
  fprintf(out,"Generated on %s.\n",t.current_time().c_str());

  // Spacing
  double dx=0.0;
  if(x_arr.size()>1)
    dx=(x_arr[x_arr.size()-1]-x_arr[0])/(x_arr.size()-1);
  double dy=0.0;
  if(y_arr.size()>1)
    dy=(y_arr[y_arr.size()-1]-y_arr[0])/(y_arr.size()-1);
  double dz=0.0;
  if(z_arr.size()>1)
    dz=(z_arr[z_arr.size()-1]-z_arr[0])/(z_arr.size()-1);

  // Write out starting point
  fprintf(out,"%7i % e % e % e\n",(int) bas.get_Nnuc(),x_arr[0],y_arr[0],z_arr[0]);
  // Print amount of points and step sizes in the directions
  fprintf(out,"%7i % e % e % e\n",(int) x_arr.size(),dx,0.0,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) y_arr.size(),0.0,dy,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) z_arr.size(),0.0,0.0,dz);
  // Print out atoms
  for(size_t i=0;i<bas.get_Nnuc();i++) {
    nucleus_t nuc=bas.get_nucleus(i);
    fprintf(out,"%7i %g % e % e % e\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
  }

  // The points in the batch
  cubecoord_t r[Nbatch_p];
  // The values of the density in the batch
  double rho[Nbatch_p];

  // Number of points to compute in the batch
  size_t np;
  // Total number of points computed
  size_t ntot=0;
  // Indices of x, y and z
  size_t xind=0, yind=0, zind=0;

  // Index of points written
  size_t idx=0;

  // Loop over batches.
  for(size_t ib=0;ib<Nbatch;ib++) {
    // Zero amount of points in current batch.
    np=0;

    // Form list of points to compute.
    while(np<Nbatch_p && ntot+np<N) {
      r[np].r.x=x_arr[xind];
      r[np].r.y=y_arr[yind];
      r[np].r.z=z_arr[zind];
      r[np].newline=false;

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	// z coordinate changes, break line here
	zind=0;
	r[np-1].newline=true;

	if(yind+1<y_arr.size())
	  yind++;
	else {
	  yind=0;
	  xind++;
	}
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    // Loop over the points in the batch
    for(size_t ip=0;ip<np;ip++)
      rho[ip]=compute_potential(P,bas,r[ip].r);

    // Save density values
    for(size_t ip=0;ip<np;ip++) {
      fprintf(out," % .5e",rho[ip]);
      idx++;
      if(idx==6 || r[ip].newline) {
	idx=0;
	fprintf(out,"\n");
      }
    }

    // Increment number of computed points
    ntot+=np;
  }

  // Close output file.
  if(idx!=0)
    fprintf(out,"\n");
  fclose(out);
}

void elf_cube(const BasisSet & bas, const arma::mat & P, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr, std::string fname) {
  // Open output file.
  fname=fname+".cube";
  FILE *out=fopen(fname.c_str(),"w");

  // Compute the density in batches, allowing
  // parallellization.
#ifdef _OPENMP
  // The number of points per batch
  const size_t Nbatch_p=100*omp_get_max_threads();
#else
  const size_t Nbatch_p=100;
#endif

  // The total number of point is
  const size_t N=x_arr.size()*y_arr.size()*z_arr.size();
  // The necessary amount of batches is
  size_t Nbatch=N/Nbatch_p;
  if(N%Nbatch_p!=0)
    Nbatch++;

  // Write out comment fields
  Timer t;
  fprintf(out,"ERKALE electron localization function output\n");
  fprintf(out,"Generated on %s.\n",t.current_time().c_str());

  // Spacing
  double dx=0.0;
  if(x_arr.size()>1)
    dx=(x_arr[x_arr.size()-1]-x_arr[0])/(x_arr.size()-1);
  double dy=0.0;
  if(y_arr.size()>1)
    dy=(y_arr[y_arr.size()-1]-y_arr[0])/(y_arr.size()-1);
  double dz=0.0;
  if(z_arr.size()>1)
    dz=(z_arr[z_arr.size()-1]-z_arr[0])/(z_arr.size()-1);

  // Write out starting point
  fprintf(out,"%7i % e % e % e\n",(int) bas.get_Nnuc(),x_arr[0],y_arr[0],z_arr[0]);
  // Print amount of points and step sizes in the directions
  fprintf(out,"%7i % e % e % e\n",(int) x_arr.size(),dx,0.0,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) y_arr.size(),0.0,dy,0.0);
  fprintf(out,"%7i % e % e % e\n",(int) z_arr.size(),0.0,0.0,dz);
  // Print out atoms
  for(size_t i=0;i<bas.get_Nnuc();i++) {
    nucleus_t nuc=bas.get_nucleus(i);
    fprintf(out,"%7i %g % e % e % e\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
  }

  // The points in the batch
  cubecoord_t r[Nbatch_p];
  // The values of the localization function in the batch
  double locf[Nbatch_p];

  // Number of points to compute in the batch
  size_t np;
  // Total number of points computed
  size_t ntot=0;
  // Indices of x, y and z
  size_t xind=0, yind=0, zind=0;

  // Index of points written
  size_t idx=0;

  // Loop over batches.
  for(size_t ib=0;ib<Nbatch;ib++) {
    // Zero amount of points in current batch.
    np=0;

    // Form list of points to compute.
    while(np<Nbatch_p && ntot+np<N) {
      r[np].r.x=x_arr[xind];
      r[np].r.y=y_arr[yind];
      r[np].r.z=z_arr[zind];
      r[np].newline=false;

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	// z coordinate changes, break line here
	zind=0;
	r[np-1].newline=true;

	if(yind+1<y_arr.size())
	  yind++;
	else {
	  yind=0;
	  xind++;
	}
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    // Loop over the points in the batch
    for(size_t ip=0;ip<np;ip++)
      locf[ip]=compute_elf(P,bas,r[ip].r);

    // Save data values
    for(size_t ip=0;ip<np;ip++) {
      fprintf(out," % .5e",locf[ip]);
      idx++;
      if(idx==6 || r[ip].newline) {
	idx=0;
	fprintf(out,"\n");
      }
    }

    // Increment number of computed points
    ntot+=np;
  }

  // Close output file.
  if(idx!=0)
    fprintf(out,"\n");
  fclose(out);
}

void orbital_cube(const BasisSet & bas, const arma::mat & C, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr, const std::vector<size_t> & orbidx, std::string fname, bool split, arma::vec & norms) {
  // Check that C is orthonormal
  arma::mat S=bas.overlap();
  check_orth(C,S,false);

  // Output file(s)
  std::vector<FILE *> out;
  if(split) {
    out.resize(orbidx.size());
    for(size_t io=0;io<orbidx.size();io++) {
      // File name will be
      char orbname[80];
      sprintf(orbname,"%s.%i.cube",fname.c_str(),(int) orbidx[io]);
      out[io]=fopen(orbname,"w");
    }
  } else {
    out.resize(1);
    std::string orbname=fname+".cube";
    out[0]=fopen(orbname.c_str(),"w");
  }

  // Compute the molecular orbital values in batches, allowing
  // parallellization.
#ifdef _OPENMP
  // The number of points per batch
  const size_t Nbatch_p=100*omp_get_max_threads();
#else
  const size_t Nbatch_p=100;
#endif

  // The total number of point is
  const size_t N=x_arr.size()*y_arr.size()*z_arr.size();
  // The necessary amount of batches is
  size_t Nbatch=N/Nbatch_p;
  if(N%Nbatch_p!=0)
    Nbatch++;

  // Write out comment fields
  Timer t;
  for(size_t io=0;io<out.size();io++) {
    fprintf(out[io],"ERKALE molecular orbital output\n");
    fprintf(out[io],"Generated on %s.\n",t.current_time().c_str());
  }

  // Spacing
  double dx=0.0;
  if(x_arr.size()>1)
    dx=(x_arr[x_arr.size()-1]-x_arr[0])/(x_arr.size()-1);
  double dy=0.0;
  if(y_arr.size()>1)
    dy=(y_arr[y_arr.size()-1]-y_arr[0])/(y_arr.size()-1);
  double dz=0.0;
  if(z_arr.size()>1)
    dz=(z_arr[z_arr.size()-1]-z_arr[0])/(z_arr.size()-1);

  for(size_t io=0;io<out.size();io++) {
    // Write out starting point. Because orbitals, amount of atoms is printed out negatively
    fprintf(out[io],"%5i % 11.6f % 11.6f % 11.6f\n",-((int) bas.get_Nnuc()),x_arr[0],y_arr[0],z_arr[0]);
    // Print amount of points and step sizes in the directions
    fprintf(out[io],"%5i % 11.6f % 11.6f % 11.6f\n",(int) x_arr.size(),dx,0.0,0.0);
    fprintf(out[io],"%5i % 11.6f % 11.6f % 11.6f\n",(int) y_arr.size(),0.0,dy,0.0);
    fprintf(out[io],"%5i % 11.6f % 11.6f % 11.6f\n",(int) z_arr.size(),0.0,0.0,dz);
    // Print out atoms
    for(size_t inuc=0;inuc<bas.get_Nnuc();inuc++) {
      nucleus_t nuc=bas.get_nucleus(inuc);
      fprintf(out[io],"%5i % 11.6f % 11.6f % 11.6f % 11.6f\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
    }
  }

  size_t idx=1;

  // Print out orbital indices
  if(split) {
    // Only one orbital, since we're splitting
    for(size_t io=0;io<orbidx.size();io++)
      fprintf(out[io],"%5i %5i\n",1,(int) orbidx[io]);
    // Reset idx
    idx=0;

  } else {
    fprintf(out[0],"%5i",(int) orbidx.size());
    for(size_t io=0;io<orbidx.size();io++) {
      fprintf(out[0],"%5i",(int) orbidx[io]);
      idx++;
      if(idx==10) {
	idx=0;
	fprintf(out[0],"\n");
      } else if(io+1 != orbidx.size())
	fprintf(out[0]," ");
    }
    if(idx!=0) {
      fprintf(out[0],"\n");
      idx=0;
    }
  }

  // The points in the batch
  cubecoord_t r[Nbatch_p];
  // The values of the orbitals in the batch
  arma::mat orbs(Nbatch_p,orbidx.size());

  // Collect orbitals
  arma::mat Cwrk(C.n_rows,orbidx.size());
  for(size_t io=0;io<orbidx.size();io++) {
    // Convert to C++ indexing
    size_t jo=orbidx[io]-1;
    // Sanity check
    if(jo>=C.n_cols) {
      std::ostringstream oss;
      oss << "There are only " << C.n_cols << " orbitals: orbital " << orbidx[io] << " does not exist!\n";
      throw std::logic_error(oss.str());
    }
    Cwrk.col(io)=C.col(jo);
  }
  // Orbital norms
  norms.zeros(orbidx.size());

  // Number of points to compute in the batch
  size_t np;
  // Total number of points computed
  size_t ntot=0;
  // Indices of x, y and z
  size_t xind=0, yind=0, zind=0;

  // Loop over batches.
  for(size_t ib=0;ib<Nbatch;ib++) {
    // Zero amount of points in current batch.
    np=0;

    // Form list of points to compute.
    while(np<Nbatch_p && ntot+np<N) {
      r[np].r.x=x_arr[xind];
      r[np].r.y=y_arr[yind];
      r[np].r.z=z_arr[zind];
      r[np].newline=false;

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	// z coordinate changes, break line here
	zind=0;
	r[np-1].newline=true;

	if(yind+1<y_arr.size())
	  yind++;
	else {
	  yind=0;
	  xind++;
	}
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    // Loop over the points in the batch
    for(size_t ip=0;ip<np;ip++) {
      orbs.row(ip)=arma::trans(compute_orbitals(Cwrk,bas,r[ip].r));
    }

    // Increment norms
    for(size_t ip=0;ip<np;ip++)
      for(size_t io=0;io<norms.n_elem;io++)
	norms(io)+=orbs(ip,io)*orbs(ip,io);

    // Save computed values
    if(split) {

      for(size_t ip=0;ip<np;ip++) {
	// Increment point index
	idx++;

	// Print out orbital values
	for(size_t io=0;io<Cwrk.n_cols;io++) {
	  fprintf(out[io]," % .5e",orbs(ip,io));
	}

	// Do we need a newline?
	if(idx==6 || r[ip].newline) {
	  idx=0;
	  for(size_t io=0;io<Cwrk.n_cols;io++)
	    fprintf(out[io],"\n");
	}
      }

    } else {
      for(size_t ip=0;ip<np;ip++)
	for(size_t io=0;io<Cwrk.n_cols;io++) {
	  fprintf(out[0]," % .5e",orbs(ip,io));
	  idx++;
	  if(idx==6 || r[ip].newline) {
	    idx=0;
	    fprintf(out[0],"\n");
	  }
	}
    }

    // Increment number of computed points
    ntot+=np;
  }
  // Close output file.
  for(size_t io=0;io<out.size();io++)
    fclose(out[io]);

  // Norms
  for(size_t io=0;io<norms.size();io++)
    norms(io)*=dx*dy*dz;
}

Settings settings;

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Cubes from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Cubes from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  // Parse settings
  settings.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  settings.add_string("RefChk","Checkpoint file to load reference density from","");
  settings.add_string("Cube", "Cube to use, e.g. -10:.3:10 -5:.2:4 -2:.1:3", "Auto");
  settings.add_double("AutoBuffer","Buffer zone in Å to add on each side of the cube",2.5);
  settings.add_double("AutoSpacing","Spacing in Å to use",0.1);
  settings.add_bool("Density", "Compute density on the cube?", false);
  settings.add_bool("SpinDensity", "Compute spin density on the cube?", false);
  settings.add_string("OrbIdx", "Indices of orbitals to compute, e.g. 1-10 1-2", "");
  settings.add_bool("SplitOrbs", "Split orbital plots into different files?", false);
  settings.add_bool("Potential", "Compute electrostatic potential on the cube?", false);
  settings.add_bool("SICOrbs", "Compute PZ-SIC orbitals on the cube?", false);
  settings.add_bool("ELF", "Compute electron localization function?", false);

  if(argc==2)
    settings.parse(argv[1]);
  else
    printf("Using default settings.\n");

  // Print settings
  settings.print();

  // Load checkpoint
  Checkpoint chkpt(settings.get_string("LoadChk"),false);

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Restricted calculation?
  bool restr;
  chkpt.read("Restricted",restr);

  arma::mat P, Pa, Pb;
  arma::mat Ca, Cb;

  chkpt.read("P",P);
  if(restr) {
    chkpt.read("C",Ca);
  } else {
    chkpt.read("Ca",Ca);
    chkpt.read("Pa",Pa);

    chkpt.read("Cb",Cb);
    chkpt.read("Pb",Pb);
  }

  // Form grid in p space.
  std::vector<double> x, y, z;
  if(stricmp(settings.get_string("Cube"),"Auto")==0) {
    // Automatical formation. Spacing to use
    double spacing=settings.get_double("AutoSpacing")*ANGSTROMINBOHR;

    // Get coordinate matrix
    arma::mat coords=basis.get_nuclear_coords();

    // Buffer to put in each side
    double extra=settings.get_double("AutoBuffer")*ANGSTROMINBOHR;

    // Minimum and maximum
    arma::rowvec minc=arma::min(coords)-extra;
    arma::rowvec maxc=arma::max(coords)+extra;

    // Round to spacing
    for(int ic=0;ic<3;ic++) {
      minc(ic)=floor(minc(ic)/spacing)*spacing;
      maxc(ic)=ceil(maxc(ic)/spacing)*spacing;
    }

    // Dimensions to use
    std::ostringstream dims;
    dims << " " << minc(0) << ":" << spacing << ":" << maxc(0); // x
    dims << " " << minc(1) << ":" << spacing << ":" << maxc(1); // y
    dims << " " << minc(2) << ":" << spacing << ":" << maxc(2); // z
    settings.set_string("Cube",dims.str());

    printf("Grid is %s.\n",dims.str().c_str());
  }
  // Parse cube
  parse_cube(settings.get_string("Cube"),x,y,z);

  Timer t;

  // Calculate orbitals on cube
  if(settings.get_string("OrbIdx").length()) {
    // Get ranges
    std::vector<std::string> ranges=splitline(settings.get_string("OrbIdx"));

    // Split orbitals into many files?
    bool split=settings.get_bool("SplitOrbs");
    // Orbital norms
    arma::vec orbnorm;

    if((ranges.size()==2 && restr) || (ranges.size()==1 && ranges[0]!="*" && !restr))
      throw std::runtime_error("Invalid orbital range specified.\n");

    std::vector< std::vector<size_t> > orbidx;
    std::vector<std::string> legend;
    if(restr) {
      orbidx.resize(1);
      legend.resize(1);
    } else {
      orbidx.resize(2);
      legend.resize(2);
      legend[0]="alpha ";
      legend[1]="beta ";
    }

    // Orbital indices, NOT IN C++ INDEXING!
    if(ranges.size()==1 && ranges[0]=="*") {
      for(size_t is=0;is<orbidx.size();is++) {
	size_t Norb = Ca.n_cols;
	std::vector<size_t> idx(Norb);
	for(size_t i=0;i<Norb;i++)
	  idx[i]=i+1;
	orbidx[is]=idx;
      }
    } else {
      for(size_t is=0;is<ranges.size();is++)
	orbidx[is]=parse_range(ranges[is],false);
    }

    for(size_t is=0;is<orbidx.size();is++) {
      printf("Calculating %sorbitals ... ",legend[is].c_str());
      fflush(stdout); t.set();

      const arma::mat * Cp;
      std::string fnam;
      if(restr) {
	Cp = &Ca;
	fnam="orbital";
      } else {
	Cp = is ? &Cb : &Ca;
	fnam = is ? "orbital-b" : "orbital-a";
      }

      orbital_cube(basis,*Cp,x,y,z,orbidx[is],fnam,split,orbnorm);
      printf("done (%s)\n",t.elapsed().c_str());

      printf("Orbital norms on grid\n");
      for(size_t io=0;io<orbidx.size();io++)
	printf("%4i %e\n",(int) orbidx[is][io],orbnorm(io));
    }
  }

  // Calculate density on cube
  if(settings.get_bool("Density")) {

    double norm;
    printf("Calculating total electron density ... ");
    fflush(stdout); t.set();
    density_cube(basis,P,x,y,z,"density",norm,false);
    printf("done (%s).\nNorm of total density is %e.\n",t.elapsed().c_str(),norm);

    if(!restr) {
      printf("Calculating alpha electron density ... ");
      fflush(stdout); t.set();
      density_cube(basis,Pa,x,y,z,"density-a",norm,false);
      printf("done (%s).\nNorm of alpha density is %e.\n",t.elapsed().c_str(),norm);

      printf("Calculating beta  electron density ... ");
      fflush(stdout); t.set();
      density_cube(basis,Pb,x,y,z,"density-b",norm,false);
      printf("done (%s).\nNorm of beta  density is %e.\n",t.elapsed().c_str(),norm);
    }
  }

  // Calculate spin density on cube
  if(settings.get_bool("SpinDensity")) {
    double norm;
    printf("Calculating electron spin density ... ");
    fflush(stdout); t.set();
    density_cube(basis,Pa-Pb,x,y,z,"spindensity",norm,false);
    printf("done (%s).\nNorm of spin density is %e.\n",t.elapsed().c_str(),norm);
  }

  // Calculate density on cube
  if(settings.get_bool("SICOrbs")) {
    if(restr) {
      // Load orbitals
      arma::cx_mat CW;
      chkpt.cread("CW",CW);

      // Loop over orbitals
      if(rms_norm(arma::imag(CW))>DBL_EPSILON) {
	for(size_t io=0;io<CW.n_cols;io++) {
	  std::ostringstream fname;
	  fname << "sicorb." << io+1;

	  printf("Orbital %i ... ",(int) io+1);
	  fflush(stdout);

	  Timer to;

	  // Density matrix
	  arma::mat Po=arma::real(CW.col(io)*arma::trans(CW.col(io)));

	  double norm;
	  density_cube(basis,Po,x,y,z,fname.str(),norm,true);

	  printf("norm is %e. (%s)\n",norm,to.elapsed().c_str());
	  fflush(stdout);
	}
      } else {
	// Real orbitals
	arma::mat CWr(arma::real(CW));
	arma::vec orbnorm;

	// Orbital indices, NOT IN C++ INDEXING!
	std::vector<size_t> orbidx(CW.n_cols);
	for(size_t i=0;i<CW.n_cols;i++)
	  orbidx[i]=i+1;

	printf("Calculating orbitals ... ");
	fflush(stdout); t.set();
	orbital_cube(basis,CWr,x,y,z,orbidx,"sicorb",true,orbnorm);
	printf("done (%s)\n",t.elapsed().c_str());
	printf("Orbital norms on grid\n");
	for(size_t io=0;io<orbidx.size();io++)
	  printf("%4i %e\n",(int) orbidx[io],orbnorm(io));
      }

    } else { // Unrestricted calculation
      // Load orbitals
      arma::cx_mat CWa, CWb;
      chkpt.cread("CWa",CWa);
      chkpt.cread("CWb",CWb);

      // Loop over orbitals
      if(rms_norm(arma::imag(CWa))>DBL_EPSILON || rms_norm(arma::imag(CWb))>DBL_EPSILON ) {
	for(size_t io=0;io<CWa.n_cols;io++) {
	  std::ostringstream fname;
	  fname << "sicorba." << io+1;

	  printf("Alpha orbital %i ... ",(int) io+1);
	  fflush(stdout);

	  Timer to;

	  // Density matrix
	  arma::mat Po=arma::real(CWa.col(io)*arma::trans(CWa.col(io)));

	  double norm;
	  density_cube(basis,Po,x,y,z,fname.str(),norm,true);

	  printf("norm is %e. (%s)\n",norm,to.elapsed().c_str());
	  fflush(stdout);
	}
	for(size_t io=0;io<CWb.n_cols;io++) {
	  std::ostringstream fname;
	  fname << "sicorbb." << io+1;

	  printf("Beta  orbital %i ... ",(int) io+1);
	  fflush(stdout);

	  Timer to;

	  // Density matrix
	  arma::mat Po=arma::real(CWb.col(io)*arma::trans(CWb.col(io)));

	  double norm;
	  density_cube(basis,Po,x,y,z,fname.str(),norm,true);

	  printf("norm is %e. (%s)\n",norm,to.elapsed().c_str());
	  fflush(stdout);
	}
      } else {
	// Real orbitals
	arma::mat CWra(arma::real(CWa));
	arma::mat CWrb(arma::real(CWb));
	arma::vec orbnorm;

	// Orbital indices, NOT IN C++ INDEXING!
	std::vector<size_t> orbidxa(CWa.n_cols);
	for(size_t i=0;i<CWa.n_cols;i++)
	  orbidxa[i]=i+1;
	std::vector<size_t> orbidxb(CWb.n_cols);
	for(size_t i=0;i<CWb.n_cols;i++)
	  orbidxb[i]=i+1;

	printf("Calculating alpha orbitals ... ");
	fflush(stdout); t.set();
	orbital_cube(basis,CWra,x,y,z,orbidxa,"sicorb-a",true,orbnorm);
	printf("done (%s)\n",t.elapsed().c_str());
	printf("Orbital norms on grid\n");
	for(size_t io=0;io<orbidxa.size();io++)
	  printf("%4i %e\n",(int) orbidxa[io],orbnorm(io));

	printf("Calculating beta orbitals ... ");
	fflush(stdout); t.set();
	orbital_cube(basis,CWrb,x,y,z,orbidxb,"sicorb-b",true,orbnorm);
	printf("done (%s)\n",t.elapsed().c_str());
	printf("Orbital norms on grid\n");
	for(size_t io=0;io<orbidxb.size();io++)
	  printf("%4i %e\n",(int) orbidxb[io],orbnorm(io));
      }
    }
  }

  // Calculate electrostatic potential on cube
  if(settings.get_bool("Potential")) {
    printf("Calculating electrostatic potential ... ");
    fflush(stdout); t.set();
    potential_cube(basis,P,x,y,z,"potential");
    printf("done (%s).\n",t.elapsed().c_str());
  }

  // Calculate localization function on cube
  if(settings.get_bool("ELF")) {
    if(restr) {
      printf("Calculating electron localization function ... ");
      fflush(stdout); t.set();
      elf_cube(basis,P/2.0,x,y,z,"elf");
      printf("done (%s).\n",t.elapsed().c_str());
    } else {
      printf("Calculating alpha localization function ... ");
      fflush(stdout); t.set();
      elf_cube(basis,Pa,x,y,z,"elf-a");
      printf("done (%s).\n",t.elapsed().c_str());

      printf("Calculating beta  localzation function ... ");
      fflush(stdout); t.set();
      elf_cube(basis,Pb,x,y,z,"elf-b");
      printf("done (%s).\n",t.elapsed().c_str());
    }
  }

  // Difference density?
  std::string refstr(settings.get_string("RefChk"));
  if(refstr.size()) {
    Checkpoint refchk(refstr,false);
    BasisSet refbas;
    refchk.read(refbas);

    arma::mat refP;
    refchk.read("P",refP);

    double norm;
    printf("Calculating difference density ... ");
    fflush(stdout); t.set();
    densitydiff_cube(basis,P,refbas,refP,x,y,z,"densitydiff",norm);
    printf("done (%s).\nNorm of difference is %e.\n",t.elapsed().c_str(),norm);
  }

  return 0;
}

int main(int argc, char **argv) {
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
