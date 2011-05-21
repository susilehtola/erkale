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



#include "global.h"

#ifndef ERKALE_STRUTIL
#define ERKALE_STRUTIL

#include <armadillo>
#include <vector>
#include <string>
#include <istream>

// Convert string to lower or upper case
std::string tolower(const std::string & in);
std::string toupper(const std::string & in);

// Read line from input (skip empty lines)
std::string readline(std::istream & in);

// Split line into words
std::vector<std::string> splitline(std::string line);

// Read integer
int readint(std::string num);

// Read a number in double precision
double readdouble(std::string num);

// Print the first n elements of a vector
void print_E(size_t n, const arma::vec & E);

// Convert memory requirement to readable text
std::string memory_size(size_t size);

#endif
