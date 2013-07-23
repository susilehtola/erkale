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
#include "checkpoint.h"
#include "stringutil.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void density_cube(const BasisSet & bas, const arma::mat & P, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr, std::string fname, double & norm) {
  // Open output file.
  fname=fname+".cube";
  FILE *out=fopen(fname.c_str(),"w");

  // Compute the norm (assumes evenly spaced grid!)
  norm=0.0;

  // Compute the momentum densities in batches, allowing
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
  fprintf(out,"%7i % g % g % g\n",(int) bas.get_Nnuc(),x_arr[0],y_arr[0],z_arr[0]);
  // Print amount of points and step sizes in the directions
  fprintf(out,"%7i % g % g % g\n",(int) x_arr.size(),dx,0.0,0.0);
  fprintf(out,"%7i % g % g % g\n",(int) y_arr.size(),0.0,dy,0.0);
  fprintf(out,"%7i % g % g % g\n",(int) z_arr.size(),0.0,0.0,dz);
  // Print out atoms
  for(size_t i=0;i<bas.get_Nnuc();i++) {
    nucleus_t nuc=bas.get_nucleus(i);
    fprintf(out,"%7i %g % g % g % g\n",nuc.Z,1.0*nuc.Z,nuc.r.x,nuc.r.y,nuc.r.z);
  }

  // The points in the batch
  coords_t r[Nbatch_p];
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
      r[np].x=x_arr[xind];
      r[np].y=y_arr[yind];
      r[np].z=z_arr[zind];

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	zind=0;

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
      rho[ip]=compute_density(P,bas,r[ip]);

    // Save density values
    for(size_t ip=0;ip<np;ip++) {
      norm+=rho[ip];
      
      fprintf(out," % .5e",rho[ip]);
      idx++;
      if(idx==6) {
	idx=0;
	fprintf(out,"\n");
      }
    }

    // Increment number of computed points
    ntot+=np;
  }

  // Close output file.
  if(idx!=0.0)
    fprintf(out,"\n");
  fclose(out);

  // Plug in the spacing in the integral
  norm*=dx*dy*dz;
}

void orbital_cube(const BasisSet & bas, const arma::mat & C, const std::vector<double> & x_arr, const std::vector<double> & y_arr, const std::vector<double> & z_arr, const std::vector<size_t> & orbidx, std::string fname, bool split) {
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

  // Compute the momentum densities in batches, allowing
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

  // Print out orbital indices
  if(split)
    for(size_t io=0;io<orbidx.size();io++)
      fprintf(out[io],"%5i",1);
  else
    fprintf(out[0],"%5i",(int) orbidx.size());

  size_t idx=1;

  if(split) {
    idx++;
    for(size_t io=0;io<orbidx.size();io++) {
      fprintf(out[io],"%5i",(int) orbidx[io]);
      if(idx==10) {
	fprintf(out[io],"\n");
      } else if(io+1 != orbidx.size())
	fprintf(out[io]," ");
    }
    // Reset idx
    if(idx==10)
      idx=0;
    // Do we need a new line?
    if(idx!=0) {
      for(size_t io=0;io<orbidx.size();io++)
	fprintf(out[io],"\n");
      idx=0;
    }
  } else {
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
  coords_t r[Nbatch_p];
  // The values of the orbitals in the batch
  arma::mat orbs(Nbatch_p,orbidx.size());

  // Collect orbitals
  arma::mat Cwrk(C.n_rows,orbidx.size());
  for(size_t io=0;io<orbidx.size();io++)
    Cwrk.col(io)=C.col(orbidx[io]-1); // Convert to C++ indexing

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
      r[np].x=x_arr[xind];
      r[np].y=y_arr[yind];
      r[np].z=z_arr[zind];

      // Increment number of points
      np++;

      // Determine next point.
      if(zind+1<z_arr.size())
	zind++;
      else {
	zind=0;

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
      orbs.row(ip)=arma::trans(compute_orbitals(Cwrk,bas,r[ip]));

    // Save computed values
    if(split) {
      for(size_t ip=0;ip<np;ip++) {
	idx++;
	for(size_t io=0;io<Cwrk.n_cols;io++) {
	  fprintf(out[io]," % .5e",orbs(ip,io));
	}

	if(idx==6) {
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
	  if(idx==6) {
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
}


int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Cubes from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Cubes from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();

  // Parse settings
  Settings set;
  set.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  set.add_string("Cube", "Cube to use, e.g. -10:.3:10 -5:.2:4 -2:.1:3", "");
  set.add_bool("Density", "Compute density on the cube?", false);
  set.add_string("OrbIdx", "Indices of orbitals to compute, e.g. 1-10 1-2", "");
  set.add_bool("SplitOrbs", "Split orbital plots into different files?", false);

  if(argc==2)
    set.parse(argv[1]);
  else
    printf("Using default settings.\n");

  // Print settings
  set.print();

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);

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
  parse_cube(set.get_string("Cube"),x,y,z);

  Timer t;

  // Calculate orbitals on cube
  if(set.get_string("OrbIdx").length()) {
    // Get ranges
    std::vector<std::string> ranges=splitline(set.get_string("OrbIdx"));

    // Split orbitals into many files?
    bool split=set.get_bool("SplitOrbs");

    if((ranges.size()==2 && restr) || (ranges.size()==1 && !restr))
      throw std::runtime_error("Invalid orbital range specified.\n");
    
    if(restr) {
      // Orbital indices, NOT IN C++ INDEXING!
      std::vector<size_t> orbidx=parse_range(ranges[0]);

      printf("Calculating orbitals ... ");
      fflush(stdout); t.set();
      orbital_cube(basis,Ca,x,y,z,orbidx,"orbital",split);
      printf("done (%s)\n",t.elapsed().c_str());

    } else {
      // Orbital indices, NOT IN C++ INDEXING!
      std::vector<size_t> orbidxa=parse_range(ranges[0]);
      std::vector<size_t> orbidxb=parse_range(ranges[1]);

      printf("Calculating alpha orbitals ... ");
      fflush(stdout); t.set();
      orbital_cube(basis,Ca,x,y,z,orbidxa,"orbital-a",split);
      printf("done (%s)\n",t.elapsed().c_str());

      printf("Calculating beta orbitals ... ");
      fflush(stdout); t.set();
      orbital_cube(basis,Cb,x,y,z,orbidxb,"orbital-b",split);
      printf("done (%s)\n",t.elapsed().c_str());
    }
  }

  // Calculate density on cube
  if(set.get_bool("Density")) {

    double norm;
    printf("Calculating total electron density ... ");
    fflush(stdout); t.set();
    density_cube(basis,P,x,y,z,"density",norm);
    printf("done (%s).\nNorm of total density is %e.\n",t.elapsed().c_str(),norm);

    if(!restr) {
      printf("Calculating alpha electron density ... ");
      fflush(stdout); t.set();
      density_cube(basis,Pa,x,y,z,"density-a",norm);
      printf("done (%s).\nNorm of alpha density is %e.\n",t.elapsed().c_str(),norm);

      printf("Calculating beta  electron density ... ");
      fflush(stdout); t.set();
      density_cube(basis,Pb,x,y,z,"density-b",norm);
      printf("done (%s).\nNorm of beta  density is %e.\n",t.elapsed().c_str(),norm);
    }
  }

  return 0;
}
