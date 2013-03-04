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


#ifndef ERKALE_EMDSTO
#define ERKALE_EMDSTO

#include "basis.h"
#include "emd/emd.h"
#include "external/storage.h"

#include <complex>
#include <vector>

/// Slater-type radial function
class RadialSlater: public RadialFourier {
  /// n value
  int n;
  /// l value is already in RadialFourier

  /// Exponent
  double zeta;
  
 public:
  /// Constructor
  RadialSlater(int n, int l, double zeta);
  /// Destructor
  ~RadialSlater();

  /// Print expansion
  void print() const;

  /// Get n value
  int getn() const;
  /// Get zeta
  double getzeta() const;

  /// Evaluate function at p
  std::complex<double> get(double p) const;
};

/// EMD in Slater basis set
class SlaterEMDEvaluator : public EMDEvaluator {
  /// The radial functions
  std::vector< std::vector<RadialSlater> > radf;

  /// Update the pointer lists
  void update_pointers();
 public:
  /// Constructor
  SlaterEMDEvaluator(const std::vector< std::vector<RadialSlater> > & radf, const std::vector< std::vector<size_t> > & idfuncsv, const std::vector< std::vector<ylmcoeff_t> > & clm, const std::vector<size_t> & locv, const std::vector<coords_t> & coord, const arma::mat & Pv);
  /// Destructor
  ~SlaterEMDEvaluator();

  /**
   * Assignment operator. This is necessary since EMDEvaluator
   * contains pointers to the memory locations of the radial
   * functions, which change whenever assignment takes place.
   */
  SlaterEMDEvaluator & operator=(const SlaterEMDEvaluator & rhs);
};

#endif
