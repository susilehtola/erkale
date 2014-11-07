/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_STORAGE
#define ERKALE_STORAGE

#include "global.h"

#include <string>
#include <vector>

/// Double precision number
typedef struct {
  /// Name of entry
  std::string name;
  /// Value
  double val;
} double_st_t;

/// Array of doubles
typedef struct {
  /// Name of entry
  std::string name;
  /// Value
  std::vector<double> val;
} double_vec_st_t;

/// Integer value
typedef struct {
  /// Name of entry
  std::string name;
  /// Value
  int val;
} int_st_t;

/// Array of integers
typedef struct {
  /// Name of entry
  std::string name;
  /// Value
  std::vector<int> val;
} int_vec_st_t;

/// String value
typedef struct {
  /// Name of entry
  std::string name;
  /// Value
  std::string val;
} string_st_t;

/// Class for storing input data
class Storage {
  /// Stack of integers
  std::vector<int_st_t> ints;
  /// Stack of doubles
  std::vector<double_st_t> doubles;
  /// Stack of integer arrays
  std::vector<int_vec_st_t> intvec;
  /// Stack of double precision arrays
  std::vector<double_vec_st_t> doublevec;
  /// String array
  std::vector<string_st_t> strings;

 public:
  /// Constructor
  Storage();
  /// Destructor
  ~Storage();

  /// Add integer value
  void add(const int_st_t & val);
  /// Add double precision value
  void add(const double_st_t & val);
  /// Add integer vector
  void add(const int_vec_st_t & val);
  /// Add double precision vector
  void add(const double_vec_st_t & val);
  /// Add string
  void add(const string_st_t & val);

  /// Get integer value
  int get_int(const std::string & name) const;
  /// Get double precision value
  double get_double(const std::string & name) const;
  /// Get integer vector
  std::vector<int> get_int_vec(const std::string & name) const;
  /// Get double precision vector
  std::vector<double> get_double_vec(const std::string & name) const;
  /// Get string
  std::string get_string(const std::string & name) const;

  /// Set integer value
  void set_int(const std::string & name, int val);
  /// Set double precision value
  void set_double(const std::string & name, double val);
  /// Set integer vector
  void set_int_vec(const std::string & name, const std::vector<int> & val);
  /// Get double precision vector
  void set_double_vec(const std::string & name, const std::vector<double> & val);
  /// Get string
  void set_string(const std::string & name, const std::string & val);

  /// Get possible integer keywords
  std::vector<std::string> find_int(const std::string & name) const;
  /// Get possible double keywords
  std::vector<std::string> find_double(const std::string & name) const;
  /// Get possible integer vector keywords
  std::vector<std::string> find_int_vec(const std::string & name) const;
  /// Get possible double vector keywords
  std::vector<std::string> find_double_vec(const std::string & name) const;
  /// Get possible string keywords
  std::vector<std::string> find_string(const std::string & name) const;


  /// Print contents (vector values, too?)
  void print(bool vals=false) const;
};

#endif
