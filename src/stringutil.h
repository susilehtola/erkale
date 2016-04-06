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



#include "global.h"

#ifndef ERKALE_STRUTIL
#define ERKALE_STRUTIL

#include <armadillo>
#include <vector>
#include <string>
#include <istream>

/// Convert string to lower case
std::string tolower(const std::string & in);
/// Convert string to upper case
std::string toupper(const std::string & in);

/// Case insensitive comparison of two strings
int stricmp(const std::string & str1, const std::string & str2);
/// Case sensitive comparison of two strings
int strcmp(const std::string & str1, const std::string & str2);

/// Read line from input (skip empty lines)
std::string readline(std::istream & in, bool skipempty=true, const std::string & cchars="#!");
/// Check if a line is blank
bool isblank(const std::string & line);

/// Get a line from the file (not skipping anything)
std::string readline(FILE *in);

/// Split line into words
std::vector<std::string> splitline(const std::string & line);

/// Trim the line (remove whitespace from start and end)
std::string trim(const std::string & line);

/// Get rid of double white space.
std::string rem_dbl_whitespace(const std::string & line);

/// Read integer
int readint(std::string num);

/// Read a number in double precision
double readdouble(std::string num);

/// Print the energies wrt occupancies
void print_E(const arma::vec & E, const std::vector<double> & occ, bool all=false);

/// Convert memory requirement to readable text
std::string memory_size(size_t size, bool approx=false);

/// Pretty-print symmetric matrix
void print_symmat(const arma::mat &mat, bool floatformat=0, double cutoff=1e-3);
/// Print matrix
void print_mat(const arma::mat & mat, const char *fmt=" % .e");

/// Print orbital energies and coefficients
void print_orb(const arma::mat & C, const arma::vec & E);

/// Parse string into pieces separated with a character in the separator
std::vector<std::string> parse(std::string in, const std::string & separator);

/// Parse input for a range of indices, e.g. 0,2-5,9,11-15,20
std::vector<size_t> parse_range(const std::string & in, bool convert=false);

/// Parse input for a range of double precision numbers, e.g. 0:.01:10,20:30
std::vector<double> parse_range_double(const std::string & in);

/// Form wanted cube
void parse_cube(const std::string & sizes, std::vector<double> & x, std::vector<double> & y, std::vector<double> & z);

/// Add spaces in number
std::string space_number(int num);

/// Pretty-print bar
std::string print_bar(std::string msg, char pad='*', int width=80, bool toupper=true);

#endif
