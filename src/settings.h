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



#ifndef ERKALE_SETTINGS
#define ERKALE_SETTINGS

#include <vector>
#include <string>

#define ISDOUBLE 0
#define ISINT 1
#define ISSTRING 2

/// Setting with a double-type value
typedef struct {
  /// Name of setting
  std::string name;
  /// A more verbose explanation what the setting does
  std::string comment;
  /// The value of the setting
  double val;
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

  /// DFT run?
  bool dft;

 public:
  /// Constructor
  Settings();
  /// Destructor
  ~Settings();

  /// Add DFT related settings
  void add_dft_settings();
  /// Remove DFT related settings
  void remove_dft_settings();
  /// Are these settings meant for a DFT run?
  bool dft_enabled() const;

  /// Add a double valued setting
  void add_double(std::string name, std::string comment, double val);
  /// Add a boolean valued setting
  void add_bool(std::string name, std::string comment, bool val);
  /// Add an integer valued setting
  void add_int(std::string name, std::string comment, int val);
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

  /// Is "name" a setting of double type?
  bool is_double(std::string name) const;
  /// Is "name" a setting of boolean type?
  bool is_bool(std::string name) const;
  /// Is "name" a setting of integer type?
  bool is_int(std::string name) const;
  /// Is "name" a setting of string type?
  bool is_string(std::string name) const;

  /// Parse file containing settings to use
  void parse(std::string filename);

  /// Print current settings
  void print() const;
};

/// Generate setting of double value
doubleset_t gend(std::string name, std::string comment, double val);
/// Generate setting of boolean value
boolset_t genb(std::string name, std::string comment, bool val);
/// Generate setting of integer value
intset_t geni(std::string name, std::string comment, int val);
/// Generate setting of string value
stringset_t gens(std::string name, std::string comment, std::string val);

#endif
