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

#ifndef ERKALE_DFTFUNCS
#define ERKALE_DFTFUNCS

#include <string>

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

/// Parse exchange and correlation functional from exchange-correlation
void parse_xc_func(int & x_func, int & c_func, const std::string & xc);

/// Print information about used functionals
void print_info(int x_func, int c_func);
/// Print full information about a functional
void print_info(int func_id);

/// Is functional for exchange?
bool is_exchange(int func_id);
/// Is functional for correlation?
bool is_correlation(int func_id);
/// Is functional for both exchange and correlation?
bool is_exchange_correlation(int func_id);
/// Is the functional for kinetic energy? (Not used in ERKALE)
bool is_kinetic(int func_id);


/// Get fraction of exact exchange
double exact_exchange(int func_id);

/// Is gradient necessary to use given functional?
bool gradient_needed(int func);
/// Is laplacian necessary to use given functional?
bool laplacian_needed(int func);

#endif
