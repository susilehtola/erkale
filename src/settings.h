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



#ifndef ERKALE_SETTINGS
#define ERKALE_SETTINGS

#include "global.h"
#include <armadillo>
#include <vector>
#include <string>

/// Setting with a double-type value
typedef struct {
  /// Name of setting
  std::string name;
  /// A more verbose explanation what the setting does
  std::string comment;
  /// The value of the setting
  double val;

  /// Is the value allowed to be negative?
  bool negative;
} doubleset_t;

/// Setting with a boolean value
typedef struct {
  /// Name of setting
  std::string name;
  /// A more verbose explanation what the setting does
  std::string comment;
  /// The value of the setting
  bool val;
} boolset_t;

/// Setting with an integer value
typedef struct {
  /// Name of setting
  std::string name;
  /// A more verbose explanation what the setting does
  std::string comment;
  /// The value of the setting
  int val;

  /// Is the value allowed to be negative?
  bool negative;
} intset_t;

/// Setting with a string value
typedef struct {
  /// Name of setting
  std::string name;
  /// A more verbose explanation what the setting does
  std::string comment;
  /// The value of the setting
  std::string val;
} stringset_t;

/// Settings used for a calculations
class Settings {
  /// Double precision number value settings
  std::vector<doubleset_t> dset;
  /// Boolean value settings
  std::vector<boolset_t> bset;
  /// Integer value settings
  std::vector<intset_t> iset;
  /// String value settings
  std::vector<stringset_t> sset;

 public:
  /// Constructor
  Settings();
  /// Destructor
  ~Settings();

  /// Add SCF related settings
  void add_scf_settings();
  /// Add DFT related settings
  void add_dft_settings();

  /// Add a double valued setting
  void add_double(std::string name, std::string comment, double val, bool negative=false);
  /// Add a boolean valued setting
  void add_bool(std::string name, std::string comment, bool val);
  /// Add an integer valued setting
  void add_int(std::string name, std::string comment, int val, bool negative=false);
  /// Add a string valued setting
  void add_string(std::string name, std::string comment, std::string val);

  /// Set a double valued setting
  void set_double(std::string name, double val);
  /// Set a boolean valued setting
  void set_bool(std::string name, bool val);
  /// Set an integer valued setting
  void set_int(std::string name, int val);
  /// Set a string valued setting
  void set_string(std::string name, std::string val);

  /// Get a double valued setting
  double get_double(std::string name) const;
  /// Get a boolean valued setting
  bool get_bool(std::string name) const;
  /// Get an integer valued setting
  int get_int(std::string name) const;
  /// Get a string valued setting
  std::string get_string(std::string name) const;

  /// Get a string setting and parse it as a vector
  arma::vec  get_vec(std::string name)  const;
  /// Get a string setting and parse it as an integer vector
  arma::ivec get_ivec(std::string name) const;
  /// Get a string setting and parse it as an unsigned integer vector
  arma::uvec get_uvec(std::string name) const;

  /// Is "name" a setting of double type? Returns index + 1 if found, else 0.
  size_t is_double(std::string name) const;
  /// Is "name" a setting of boolean type? Returns index + 1 if found, else 0.
  size_t is_bool(std::string name) const;
  /// Is "name" a setting of integer type? Returns index + 1 if found, else 0.
  size_t is_int(std::string name) const;
  /// Is "name" a setting of string type? Returns index + 1 if found, else 0.
  size_t is_string(std::string name) const;

  /// Parse file containing settings to use. SCF indicates special handling for the method keyword
  void parse(std::string filename, bool scf=false);

  /// Print current settings
  void print() const;
};

/// Generate setting of double value
doubleset_t gend(std::string name, std::string comment, double val, bool negative);
/// Generate setting of boolean value
boolset_t genb(std::string name, std::string comment, bool val);
/// Generate setting of integer value
intset_t geni(std::string name, std::string comment, int val, bool negative);
/// Generate setting of string value
stringset_t gens(std::string name, std::string comment, std::string val);

#endif
