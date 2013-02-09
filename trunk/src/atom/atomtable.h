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

#ifndef ATOMTABLE_H
#define ATOMTABLE_H

#include "../eritable.h"
#include "integrals.h"

class AtomTable : public ERItable {
  /// Amount of functions
  size_t Nbf;

  /// Calculate index in integral table
  size_t idx(size_t i, size_t j, size_t k, size_t l) const;
 public:
  /// Consructor
  AtomTable();
  /// Destructor
  ~AtomTable();

  /// Fill table
  void fill(const std::vector<bf_t> & bas, bool verbose);
};

#endif
