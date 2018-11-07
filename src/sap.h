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

#ifndef SAP_POT_H
#define SAP_POT_H

#include "global.h"
#include <armadillo>
#include <vector>

class SAP {
  /// Atomic data
  std::vector<arma::mat> atoms;
 public:
  /// Constructor
  SAP();
  /// Destructor
  ~SAP();

  /// Get potential
  double get(int Z, double r) const;
};

#endif

