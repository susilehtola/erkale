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

/// Fixed-length data for shell
typedef struct {
  /// Angular momentum of shell
  int am;
  /// Are spherical harmonics used?
  hbool_t uselm;
  /// Index of center
  hsize_t cenind;
  /// First function on shell
  hsize_t indstart;
} shell_data_t;

/// Length of symbol
#define SYMLEN 10

/// Nucleus type
typedef struct {
  /// Index of nucleus
  hsize_t ind;
  /// x coordinate
  double rx;
  /// y coordinate
  double ry;
  /// z coordinate
  double rz;
  /// Counterpoise?
  hbool_t bsse;
  /// Charge
  int Z;
  /// Type of nucleus
  char sym[SYMLEN];
} nuc_t;

/// Checkpointing class.
class Checkpoint {
  /// The checkpoint file
  hid_t file;

  /// Is file open for writing?
  bool writemode;

  /// Save value
  void write(const std::string & name, hbool_t val);
  /// Read value
  void read(const std::string & name, hbool_t & val) const;

 public:
  /// Create checkpoint file
  Checkpoint(const std::string & filename, bool write);
  /// Destructor
  ~Checkpoint();

  /// Remove entry if exists
  void remove(const std::string & name);

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
  /// Load basis set
  void read(BasisSet & basis) const;

  /// Save value
  void write(const std::string & name, double val);
  /// Read value
  void read(const std::string & name, double & val) const;

  /// Save value
  void write(const std::string & name, int val);
  /// Read value
  void read(const std::string & name, int & val) const;

  /// Save value
  void write(const std::string & name, bool val);
  /// Read value
  void read(const std::string & name, bool & val) const;
};


#endif
