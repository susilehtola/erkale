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
  dr_=0.0;
}

HirshfeldAtom::HirshfeldAtom(const BasisSet & basis, const arma::mat & P, double drv) {
  // Set spacing
  dr_=drv;

  if(basis.Nnuc()>1) {
    ERROR_INFO();
    fprintf(stderr,"Warning - more than one nucleus in system!\n");
  }
  if(basis.Nnuc()==0) {
    throw std::runtime_error("No nucleus in system!\n");
  }
  // Get coordinates of nucleus
  coords_t nuc=basis.nuclear_coords(0);

  // Maximum component that can appear in density is 2L.
  int lmax=next_lebedev(2*basis.max_am());

  // Get Lebedev rule
  std::vector<lebedev_point_t> ang=lebedev_sphere(lmax);

  /// Fill out grid
  while(true) {
    // Compute radius
    double r=rho_.size()*dr_;

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
    rho_.push_back(d);
    // Stop iteration?
    if(d==0.0) {
      break;
    }
  }
}

HirshfeldAtom::HirshfeldAtom(double drv, const std::vector<double> & rhov) : dr_(drv), rho_(rhov) {
}

HirshfeldAtom::~HirshfeldAtom() {
}

double HirshfeldAtom::density(double r) const {
  if(dr_==0.0)
    return 0.0;

  // Linear interpolation.
  double rdr=r/dr_;
  // Index of entry is
  size_t i=(size_t) floor(rdr);

  // Check limit. Test against rho_.size() rather than rho_.size()-1 so
  // that an empty rho_ (size 0) doesn't wrap on the unsigned subtraction.
  if(i+1>=rho_.size())
    return 0.0;

  // Perform linear intepolation
  return rho_[i] + (rho_[i+1]-rho_[i])*(rdr-i);
}

double HirshfeldAtom::spacing() const {
  return dr_;
}

std::vector<double> HirshfeldAtom::rho() const {
  return rho_;
}

double HirshfeldAtom::range() const {
  if(rho_.size())
    return (rho_.size()-1)*dr_;
  else
    return 0.0;
}

double HirshfeldAtom::moment(int k) const {
  double m=0.0;
  for(size_t i=0;i<rho_.size();i++)
    m+=std::pow(i*dr_,k+2)*rho_[i];

  return m*dr_;
}

Hirshfeld::Hirshfeld() {
}

void Hirshfeld::compute(const BasisSet & basis, std::string method) {
  // Store atomic centers.
  cen_.resize(basis.Nnuc());
  for(size_t i=0;i<cen_.size();i++)
    cen_[i]=basis.nucleus(i).r;

  // Reserve memory for atomic densities
  atoms_.resize(basis.Nnuc());

  // Get list of identical nuclei
  std::vector< std::vector<size_t> > idnuc=basis.find_identical_nuclei();

  // Loop over list of identical nuclei
  for(size_t i=0;i<idnuc.size();i++) {
    // Perform guess
    arma::vec atE;
    arma::mat atC;
    arma::mat atP;
    arma::mat atF;
    BasisSet atbas;
    std::vector<size_t> shellidx;
    // Don't drop polarization shells but do occupation smearing. Charge is 0
    atomic_guess(basis,idnuc[i][0],method,shellidx,atbas,atE,atC,atP,atF,0);

    // Construct atom
    HirshfeldAtom at(atbas,atP);
    // and store it
    for(size_t j=0;j<idnuc[i].size();j++)
      atoms_[idnuc[i][j]]=at;
  }
}

void Hirshfeld::load(const BasisSet & basis) {
  // Store atomic centers.
  cen_.resize(basis.Nnuc());
  for(size_t i=0;i<cen_.size();i++)
    cen_[i]=basis.nucleus(i).r;

  // Reserve memory for atomic densities
  atoms_.resize(basis.Nnuc());

  // Get list of nuclei
  std::vector<nucleus_t> nuc=basis.nuclei();
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
	atoms_[Zv[Z][j]]=at;
    }
}

Hirshfeld::~Hirshfeld() {
}

double Hirshfeld::density(size_t inuc, const coords_t & r) const {
  // Compute distance and get density
  coords_t rd=r-cen_[inuc];
  return atoms_[inuc].density(norm(rd));
}

double Hirshfeld::weight(size_t inuc, const coords_t & r) const {
  if(atoms_.size()!=cen_.size()) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "There are " << atoms_.size() << " atoms but " << cen_.size() << " centers!\n";
    throw std::runtime_error(oss.str());
  }

  // Compute atomic weights
  arma::vec atw(atoms_.size());
  for(size_t iat=0;iat<atoms_.size();iat++) {
    // Convert coordinates relative to nucleus
    coords_t rd=r-cen_[iat];
    // Return the density
    atw(iat)=atoms_[iat].density(norm(rd));
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

double Hirshfeld::range(size_t inuc) const {
  return atoms_[inuc].range();
}

double Hirshfeld::moment(size_t inuc, int n) const {
  return atoms_[inuc].moment(n);
}

void Hirshfeld::print_densities() const {
  // Print out atom densities
  for(size_t i=0;i<atoms_.size();i++) {
    std::ostringstream fname;
    fname << "hirshfeld_" << i << ".dat";
    FILE *out=fopen(fname.str().c_str(),"w");
    if(!out) {
      std::ostringstream oss;
      oss << "Could not open \"" << fname.str() << "\" for writing.\n";
      throw std::runtime_error(oss.str());
    }

    // Spacing to use
    double dr=0.001;
    // Amount of points
    size_t N=1+ (size_t) round(atoms_[i].range()/dr);
    // Iterate ir=0..N-1; the previous `<=N` upper bound walked one
    // point past the radial range, where atoms_[i].density returns 0.
    for(size_t ir=0;ir<N;ir++)
      fprintf(out,"%e %e\n",ir*dr,atoms_[i].density(ir*dr));
    fclose(out);
  }
}

void Hirshfeld::set_atoms(const std::vector<coords_t> & cenv, double dr, const std::vector< std::vector<double> > & rho) {
  if(cenv.size()!=rho.size()) {
    ERROR_INFO();
    throw std::runtime_error("Size of centers does not size of densities!\n");
  }

  // Store centers
  cen_=cenv;

  // Store atoms
  atoms_.resize(rho.size());
  for(size_t i=0;i<rho.size();i++) {
    atoms_[i]=HirshfeldAtom(dr,rho[i]);
  }
}

std::vector< std::vector<double> > Hirshfeld::rho() const {
  std::vector< std::vector<double> > ret(atoms_.size());
  for(size_t i=0;i<ret.size();i++)
    ret[i]=atoms_[i].rho();

  return ret;
}
