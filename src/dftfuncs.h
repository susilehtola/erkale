/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#ifndef ERKALE_DFTFUNCS
#define ERKALE_DFTFUNCS

/// Struct for a functional
typedef struct {
  /// Name of functional
  std::string name;
  /// Number of functional
  int func_id;
} func_t;

/// Print keyword corresponding to functional.
std::string get_keyword(int func_id);

/// Find out ID of functional
int find_func(std::string name);

#endif
