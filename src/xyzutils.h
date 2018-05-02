/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#ifndef ERKALE_XYZ
#define ERKALE_XYZ

#include <string>
#include <vector>

/// A structure for reading in atoms
typedef struct {
  /// Type of atom
  std::string el;
  /// Atom number
  size_t num;
  /// x coordinate;
  double x;
  /// y coordinate
  double y;
  /// z coordinate
  double z;
  /// Charge
  int Q;
} atom_t;

/// Load atoms from xyz file, return list of atoms, converting angstrom to au. Returns everything in atomic units.
std::vector<atom_t> load_xyz(std::string filename, bool convert);

/// Save atoms to xyz file
void save_xyz(const std::vector<atom_t> & at, const std::string & comment, const std::string & fname, bool append=false);

/// Print xyz
void print_xyz(const std::vector<atom_t> & at);

#endif
