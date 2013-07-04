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

#include "hirshfeld.h"
#include "guess.h"
#include "mathf.h"

HirshfeldAtom::HirshfeldAtom(const BasisSet & basis, const arma::mat & P) {
  // Value of density
  double d;

  /// Fill out grid
  while(true) {
    // Compute radius
    double r=rad.size()*dr;

    // Helper
    coords_t hlp;
    hlp.x=hlp.y=0.0;
    hlp.z=r;

    // Compute density
    d=compute_density(P,basis,hlp);

    // Add to stack
    if(d>0.0) {
      rad.push_back(r);
      logrho.push_back(log(d));
    } else {
      //      printf("%i points used, max radius %e.\n",(int) rad.size(),rad[rad.size()-1]);
      break;
    }
  }
}

HirshfeldAtom::~HirshfeldAtom() {
}

double HirshfeldAtom::get(double r) const {
  /// Limiting case
  if(r>rad[rad.size()-1])
    return 0.0;

  // Otherwise, perform spline interpolation
  return exp(spline_interpolation(rad,logrho,r));
}

std::vector<double> HirshfeldAtom::get_rad() const {
  return rad;
}

std::vector<double> HirshfeldAtom::get_rho() const {
  std::vector<double> rho(logrho);
  for(size_t i=0;i<rho.size();i++)
    rho[i]=exp(rho[i]);
  return rho;
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
  atomstorage.clear();
  atomstorage.reserve(idnuc.size());

  // Get list of identical nuclei
  idnuc=identical_nuclei(basis);

  // Loop over list of identical nuclei
  for(size_t i=0;i<idnuc.size();i++) {
    // Perform guess
    arma::vec atE;
    arma::mat atP;
    BasisSet atbas;
    std::vector<size_t> shellidx;
    atomic_guess(basis,idnuc[i][0],method,shellidx,atbas,atE,atP,false);

    // Compute atomic density
    atomstorage.push_back(HirshfeldAtom(atbas,atP));
  }

  // Update the pointers
  update_pointers();
}

Hirshfeld::~Hirshfeld() {
}

void Hirshfeld::update_pointers() {
  // Reset pointers to null
  atoms.assign(atoms.size(),NULL);

  // Make links to all identical atoms
  for(size_t i=0;i<idnuc.size();i++)
    for(size_t j=0;j<idnuc[i].size();j++)
      atoms[idnuc[i][j]]=&atomstorage[i];
}

double Hirshfeld::get_density(size_t inuc, const coords_t & r) const {
  // Check for ghost nucleus
  if(atoms[inuc]==NULL)
    return 0.0;

  // Otherwise compute distance and get density
  coords_t rd=r-cen[inuc];
  return atoms[inuc]->get(norm(rd));
}

double Hirshfeld::get_weight(size_t inuc, const coords_t & r) const {
  // Compute atomic weights
  arma::vec atw(atoms.size());
  for(size_t iat=0;iat<atoms.size();iat++) {
    // Convert coordinates relative to nucleus
    coords_t rd=r-cen[iat];
    // Return the density
    atw(iat)=atoms[iat]->get(norm(rd));
  }

  // Compute total sum
  double sum=arma::sum(atw);

  // Check for contingency (far away points)
  if(sum==0.0)
    return 0.0;
  else
    // Sum is nonzero.
    return atw(inuc)/arma::sum(atw);
}
