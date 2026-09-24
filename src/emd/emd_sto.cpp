/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "emd_sto.h"
#include "../mathf.h"
#include <algorithm>

RadialSlater::RadialSlater(int nv, int lv, double zetav) : RadialFourier(lv) {
  n_=nv;
  zeta_=zetav;
}

RadialSlater::~RadialSlater() {
}

void RadialSlater::print() const {
  printf("n=%i, l=%i, zeta=%e\n",n_,l_,zeta_);
}

int RadialSlater::n() const {
  return n_;
}

double RadialSlater::zeta() const {
  return zeta_;
}

double wknl(int n, int l, int k, double zeta) {
  return pow(-1.0/(4.0*zeta*zeta),k)*fact(n-k)/(fact(k)*fact(n-l-2*k));
}

std::complex<double> RadialSlater::eval(double p) const {

  double sum=0.0;
  for(int k=0;k<=(n_-l_)/2;k++)
    sum+=wknl(n_,l_,k,zeta_)/pow(zeta_*zeta_+p*p,n_+1-k);

  return pow(2.0*M_PI,1.5)*pow(2.0,n_-1)*fact(n_-l_)/(M_PI*M_PI)*pow(std::complex<double>(0.0,-p),l_)*pow(zeta_,n_-l_)*pow(2*zeta_,n_+0.5)/sqrt(fact(2*n_))*sum;
}

SlaterEMDEvaluator::SlaterEMDEvaluator(const std::vector< std::vector<RadialSlater> > & radfv, const std::vector< std::vector<size_t> > & idfuncsv, const std::vector< std::vector<ylmcoeff_t> > & clm, const std::vector<size_t> & locv, const std::vector<coords_t> & coord, const arma::cx_mat & Pv) : EMDEvaluator(idfuncsv,clm,locv,coord,Pv) {
  // Set the radial functions
  radf_=radfv;
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
  radf_=rhs.radf_;
  // Update the pointers
  update_pointers();

  return *this;
}

void SlaterEMDEvaluator::update_pointers() {
  rad_.resize(radf_.size());
  for(size_t i=0;i<radf_.size();i++) {
    rad_[i].resize(radf_[i].size());
    for(size_t j=0;j<radf_[i].size();j++)
      rad_[i][j]=&radf_[i][j];
  }
}
