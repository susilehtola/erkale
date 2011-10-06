/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_CHECKPOINT
#define ERKALE_CHECKPOINT

#include "global.h"
#include "basis.h"

#include <armadillo>
#include <string>

// Use C routines, since C++ routines don't seem to add any ease of use.
extern "C" {
#include <hdf5.h>
}

/// Checkpointing class.
class Checkpoint {
  /// The checkpoint file
  hid_t file;

 public:
  /// Create checkpoint file
  Checkpoint(const std::string & filename, bool write=0);
  /// Destructor
  ~Checkpoint();

  /// Save matrix
  void write(const std::string & name, const arma::mat & mat);
  /// Read matrix
  void read(const std::string & name, arma::mat & mat) const;

  /// Save array
  void write(const std::string & name, const std::vector<double> & v);
  /// Load array
  void read(const std::string & name, std::vector<double> & v) const;

  /// Save basis set
  void write(const BasisSet & basis);
};


#endif
