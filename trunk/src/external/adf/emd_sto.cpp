/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "emd_sto.h"
#include "mathf.h"
#include <algorithm>

RadialSlater::RadialSlater(int nv, int l, double zetav) : RadialFourier(l) {
  n=nv;
  zeta=zetav;
}

RadialSlater::~RadialSlater() {
}

void RadialSlater::print() const {
  printf("n=%i, l=%i, zeta=%e\n",n,l,zeta);
}

int RadialSlater::getn() const {
  return n;
}

double RadialSlater::getzeta() const {
  return zeta;
}

double wknl(int n, int l, int k, double zeta) {
  return pow(-1.0/(4.0*zeta*zeta),k)*fact(n-k)/(fact(k)*fact(n-l-2*k));
}

std::complex<double> RadialSlater::get(double p) const {

  double sum=0.0;
  for(int k=0;k<=(n-l)/2;k++)
    sum+=wknl(n,l,k,zeta)/pow(zeta*zeta+p*p,n+1-k);

  return pow(2.0*M_PI,1.5)*pow(2.0,n-1)*fact(n-l)/(M_PI*M_PI)*pow(std::complex<double>(0.0,-p),l)*pow(zeta,n-l)*pow(2*zeta,n+0.5)/sqrt(fact(2*n))*sum;
}


SlaterEMDEvaluator::SlaterEMDEvaluator(const SlaterEMDEvaluator & rhs) {
  *this=rhs;
}

SlaterEMDEvaluator::SlaterEMDEvaluator(const std::vector< std::vector<RadialSlater> > & radfv, const std::vector< std::vector<size_t> > & idfuncsv, const std::vector< std::vector<ylmcoeff_t> > & clm, const std::vector<size_t> & locv, const std::vector<coords_t> & coord, const arma::mat & Pv) : EMDEvaluator(idfuncsv,clm,locv,coord,Pv) {
  // Set the radial functions
  radf=radfv;
  // and assign the necessary pointers
  update_pointers();
  // Check the norms
  //  check_norm();
}

SlaterEMDEvaluator::~SlaterEMDEvaluator() {
}


SlaterEMDEvaluator & SlaterEMDEvaluator::operator=(const SlaterEMDEvaluator & rhs) {
  // Assign superclass part
  EMDEvaluator::operator=(rhs);
  // Copy radial functions
  radf=rhs.radf;
  // Update the pointers
  update_pointers();

  return *this;
}

void SlaterEMDEvaluator::update_pointers() {
  rad.resize(radf.size());
  for(size_t i=0;i<radf.size();i++) {
    rad[i].resize(radf[i].size());
    for(size_t j=0;j<radf[i].size();j++)
      rad[i][j]=&radf[i][j];
  }
}
