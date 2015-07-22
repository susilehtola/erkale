/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_BOYS
#define ERKALE_BOYS

#include "global.h"
#include <armadillo>

namespace BoysTable {
  /// Table holding Boys function values
  extern arma::mat bfdata;
  /// Table holding exp(-x) values for recursion relations
  extern arma::vec expdata;
  /// Table holding the constant prefactor for the asymptotic formula
  extern arma::vec prefac;

  /// Maximum m value
  extern int mmax;
  /// Order of Boys function expansion
  extern int bforder;
  /// Order of exp(-x) expansion
  extern int exporder;
  /// Tabulation interval
  extern double dx;
  /// Upper limit of table
  extern double xmax;

  /// Fill table
  void fill(int mmax, int order=6, double dx=0.001, double xmax=40.0);
  
  /// Evaluate
  double eval(int m, double x);
  /// Evaluate a bunch of values
  void eval(int mmax, double x, arma::vec & Fval);
};

#endif
