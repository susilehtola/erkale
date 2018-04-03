/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "checkpoint.h"
#include "hirshfeld.h"
#include "elements.h"
#include "guess.h"
#include "mathf.h"
#include "lebedev.h"

HirshfeldAtom::HirshfeldAtom() {
  dr=0.0;
}

HirshfeldAtom::HirshfeldAtom(const BasisSet & basis, const arma::mat & P, double drv) {
  // Set spacing
  dr=drv;

  if(basis.get_Nnuc()>1) {
    ERROR_INFO();
    fprintf(stderr,"Warning - more than one nucleus in system!\n");
  }
  if(basis.get_Nnuc()==0) {
    throw std::runtime_error("No nucleus in system!\n");
  }
  // Get coordinates of nucleus
  coords_t nuc=basis.get_nuclear_coords(0);

  // Maximum component that can appear in density is 2L.
  int lmax=next_lebedev(2*basis.get_max_am());

  // Get Lebedev rule
  std::vector<lebedev_point_t> ang=lebedev_sphere(lmax);

  /// Fill out grid
  while(true) {
    // Compute radius
    double r=rho.size()*dr;

    // Compute spherical average
    double d=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:d)
#endif
    for(size_t iang=0;iang<ang.size();iang++) {
      // Helper
      coords_t hlp;
      hlp.x=r*ang[iang].x;
      hlp.y=r*ang[iang].y;
      hlp.z=r*ang[iang].z;
      // Compute density (relative to nucleus)
      d+=ang[iang].w*compute_density(P,basis,hlp-nuc);
    }
    // Add to stack
    rho.push_back(d);
    // Stop iteration?
    if(d==0.0) {
      break;
    }
  }
}

HirshfeldAtom::HirshfeldAtom(double drv, const std::vector<double> & rhov) : dr(drv), rho(rhov) {
}

HirshfeldAtom::~HirshfeldAtom() {
}

double HirshfeldAtom::get(double r) const {
  if(dr==0.0)
    return 0.0;

  // Linear interpolation.
  double rdr=r/dr;
  // Index of entry is
  size_t i=(size_t) floor(rdr);

  // Check limit
  if(i>=rho.size()-1)
    return 0.0;

  // Perform linear intepolation
  return rho[i] + (rho[i+1]-rho[i])*(rdr-i);
}

double HirshfeldAtom::get_spacing() const {
  return dr;
}

std::vector<double> HirshfeldAtom::get_rho() const {
  return rho;
}

double HirshfeldAtom::get_range() const {
  if(rho.size())
    return (rho.size()-1)*dr;
  else
    return 0.0;
}

double HirshfeldAtom::compute_moment(int k) const {
  double m=0.0;
  for(size_t i=0;i<rho.size();i++)
    m+=std::pow(i*dr,k+2)*rho[i];

  return m*dr;
}

Hirshfeld::Hirshfeld() {
}

void Hirshfeld::compute(const BasisSet & basis, std::string method) {
  // Store atomic centers.
  cen.resize(basis.get_Nnuc());
  for(size_t i=0;i<cen.size();i++)
    cen[i]=basis.get_nucleus(i).r;

  // Reserve memory for atomic densities
  atoms.resize(basis.get_Nnuc());

  // Get list of identical nuclei
  std::vector< std::vector<size_t> > idnuc=identical_nuclei(basis);

  // Loop over list of identical nuclei
  for(size_t i=0;i<idnuc.size();i++) {
    // Perform guess
    arma::vec atE;
    arma::mat atP;
    BasisSet atbas;
    std::vector<size_t> shellidx;
    // Don't drop polarization shells but do occupation smearing. Charge is 0
    atomic_guess(basis,idnuc[i][0],method,shellidx,atbas,atE,atP,false,true,0);

    // Construct atom
    HirshfeldAtom at(atbas,atP);
    // and store it
    for(size_t j=0;j<idnuc[i].size();j++)
      atoms[idnuc[i][j]]=at;
  }
}

void Hirshfeld::load(const BasisSet & basis) {
  // Store atomic centers.
  cen.resize(basis.get_Nnuc());
  for(size_t i=0;i<cen.size();i++)
    cen[i]=basis.get_nucleus(i).r;

  // Reserve memory for atomic densities
  atoms.resize(basis.get_Nnuc());

  // Get list of nuclei
  std::vector<nucleus_t> nuc=basis.get_nuclei();
  // Get list of elements in system
  std::vector< std::vector<size_t> > Zv(maxZ+1);
  for(size_t i=0;i<nuc.size();i++) {
    if(nuc[i].bsse)
      continue;
    Zv[nuc[i].Z].push_back(i);
  }

  // Loop over elements
  for(size_t Z=0;Z<Zv.size();Z++)
    if(Zv[Z].size()) {
      // Load checkpoint
      std::string chkname=element_symbols[Z]+"_0.chk";
      Checkpoint chkpt(chkname,false);

      // Load basis set and density matrix
      BasisSet bas;
      chkpt.read(bas);

      arma::mat P;
      chkpt.read("P",P);

      // Check norm of density matrix
      double Nel=arma::trace(P*bas.overlap());
      if(fabs(Nel-Z)>1e-3) {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Loaded density matrix for " << element_symbols[Z] << " contains " << Nel << " electrons!\n";
	throw std::runtime_error(oss.str());
      }

      // Construct atom
      HirshfeldAtom at(bas,P);
      // and store it
      for(size_t j=0;j<Zv[Z].size();j++)
	atoms[Zv[Z][j]]=at;
    }
}

Hirshfeld::~Hirshfeld() {
}

double Hirshfeld::get_density(size_t inuc, const coords_t & r) const {
  // Compute distance and get density
  coords_t rd=r-cen[inuc];
  return atoms[inuc].get(norm(rd));
}

double Hirshfeld::get_weight(size_t inuc, const coords_t & r) const {
  if(atoms.size()!=cen.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "There are " << atoms.size() << " atoms but " << cen.size() << " centers!\n";
    throw std::runtime_error(oss.str());
  }

  // Compute atomic weights
  arma::vec atw(atoms.size());
  for(size_t iat=0;iat<atoms.size();iat++) {
    // Convert coordinates relative to nucleus
    coords_t rd=r-cen[iat];
    // Return the density
    atw(iat)=atoms[iat].get(norm(rd));
  }

  // Compute total sum
  double sum=arma::sum(atw);

  // Check for contingency (far away points)
  if(sum==0.0)
    return 0.0;
  else
    // Sum is nonzero.
    return atw(inuc)/sum;
}

double Hirshfeld::get_range(size_t inuc) const {
  return atoms[inuc].get_range();
}

double Hirshfeld::compute_moment(size_t inuc, int n) const {
  return atoms[inuc].compute_moment(n);
}

void Hirshfeld::print_densities() const {
  // Print out atom densities
  for(size_t i=0;i<atoms.size();i++) {
    std::ostringstream fname;
    fname << "hirshfeld_" << i << ".dat";
    FILE *out=fopen(fname.str().c_str(),"w");

    // Spacing to use
    double dr=0.001;
    // Amount of points
    size_t N=1+ (size_t) round(atoms[i].get_range()/dr);
    for(size_t ir=0;ir<=N;ir++)
      fprintf(out,"%e %e\n",ir*dr,atoms[i].get(ir*dr));
    fclose(out);
  }
}

void Hirshfeld::set(const std::vector<coords_t> & cenv, double dr, const std::vector< std::vector<double> > & rho) {
  if(cenv.size()!=rho.size()) {
    ERROR_INFO();
    throw std::runtime_error("Size of centers does not size of densities!\n");
  }

  // Store centers
  cen=cenv;

  // Store atoms
  atoms.resize(rho.size());
  for(size_t i=0;i<rho.size();i++) {
    atoms[i]=HirshfeldAtom(dr,rho[i]);
  }
}

std::vector< std::vector<double> > Hirshfeld::get_rho() const {
  std::vector< std::vector<double> > ret(atoms.size());
  for(size_t i=0;i<ret.size();i++)
    ret[i]=atoms[i].get_rho();

  return ret;
}
