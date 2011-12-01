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

/// Checkpoint version.
#define ERKALE_CHKVER 1

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
  /// Name of the file
  std::string filename;
  /// Is file open for writing?
  bool writemode;

  /// Is the file open
  bool opend;
  /// The checkpoint file
  hid_t file;

  // *** Helper functions ***

  /// Save value
  void write(const std::string & name, hbool_t val);
  /// Read value
  void read(const std::string & name, hbool_t & val);

 public:
  /// Create checkpoint file
  Checkpoint(const std::string & filename, bool write);
  /// Destructor
  ~Checkpoint();

  /// Open the file
  void open();
  /// Close the file
  void close();

  /// Is the file open?
  bool is_open() const;
  /// Does the entry exist in the file?
  bool exist(const std::string & name);


  /**
   * Remove entry if exists. File needs to be opened beforehand. HDF5
   * doesn't reclaim any used space after the file has been closed, so
   * this is only useful when you want to replace an existing entry
   * with something new.
   */
  void remove(const std::string & name);

  /**
   * Access routines.
   *
   * An open file will be left open, a closed file will be left closed.
   */

  /// Save matrix
  void write(const std::string & name, const arma::mat & mat);
  /// Read matrix
  void read(const std::string & name, arma::mat & mat);

  /// Save array
  void write(const std::string & name, const std::vector<double> & v);
  /// Load array
  void read(const std::string & name, std::vector<double> & v);

  /// Save basis set
  void write(const BasisSet & basis);
  /// Load basis set
  void read(BasisSet & basis);

  /// Save value
  void write(const std::string & name, double val);
  /// Read value
  void read(const std::string & name, double & val);

  /// Save value
  void write(const std::string & name, int val);
  /// Read value
  void read(const std::string & name, int & val);

  /// Save value
  void write(const std::string & name, bool val);
  /// Read value
  void read(const std::string & name, bool & val);
};

/// Check for existence of file
bool file_exists(const std::string & name);

#endif
