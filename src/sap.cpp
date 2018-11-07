/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2018
 * Copyright (c) 2018, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "sap.h"
#include "elements.h"

SAP::SAP() {
  // Allocate memory
  atoms.resize(sizeof(element_symbols)/sizeof(element_symbols[0]));

  // Location to get the files from
  std::string loc;
  char * libloc=getenv("ERKALE_SAP_LIBRARY");
  if(libloc!=NULL) {
    loc=libloc;
  }

  // Read atoms
  for(size_t Z=1;Z<atoms.size();Z++) {
    std::string fname=loc+"/v_"+element_symbols[Z]+".dat";
    atoms[Z].load(fname,arma::raw_ascii);
  }
}

SAP::~SAP() {
}

double SAP::get(int Z, double r) const {
  if(Z<0 || Z>=(int) atoms.size())
    throw std::logic_error("Z outside SAP library size.\n");
  if(atoms[Z].n_rows == 0)
    throw std::logic_error("No data for atom in SAP library!\n");

  // Pointers to memory
  arma::vec rv(atoms[Z].col(0));
  arma::vec nv(atoms[Z].col(1));

  // Asymptotic potential?
  if(r>=rv(rv.n_elem-1))
    return Z;
  
  // Table lookup
  for(size_t i=0;i<rv.n_elem-1;i++)
    if(r>=rv(i) && r<=rv(i+1)) {
      // Linear correction
      double corr=(r-rv(i))/(rv(i+1)-rv(i))*(nv(i+1)-nv(i));
      // Value at wanted point is
      double P=nv(i) + corr;
      // so the value of the potential is
      double sappot=P/r;
      if(!std::isnormal(sappot))
        sappot=0.0;
      return sappot;
    }

  throw std::logic_error("Something went awry!\n");
}
