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

/**
 * A table of named settings that share a single value type T.
 *
 * This collapses the former per-type quadruplication (a separate struct,
 * vector, and add/set/get/is loop for each of double/bool/int/string) into
 * one implementation. Lookups follow the historical semantics: add/set/is
 * match case-insensitively, get matches case-sensitively.
 */
template<typename T>
class SettingTable {
  /// One named setting
  struct Entry {
    /// Name of setting
    std::string name;
    /// A more verbose explanation of what the setting does
    std::string comment;
    /// The value of the setting
    T val;
    /// Is the value allowed to be negative? (consulted only for numeric T)
    bool negative;
  };
  /// The settings, in insertion order
  std::vector<Entry> entries;
  /// Human-readable type label used in diagnostic messages
  std::string typelabel;

  /// Case-insensitive lookup: returns index + 1 if found, else 0
  size_t find_ci(const std::string & name) const;
  /// Case-sensitive lookup: returns index + 1 if found, else 0
  size_t find_cs(const std::string & name) const;

 public:
  /// Constructor. label names the type in diagnostics ("double type", ...)
  explicit SettingTable(const std::string & label);

  /// Add a setting; throws if one of the same name already exists
  void add(const std::string & name, const std::string & comment, const T & val, bool negative=false);
  /// Set the value of an existing setting; throws if absent or, for numeric T, sign-invalid
  void set(const std::string & name, const T & val);
  /// Get the value of a setting; throws if absent
  T get(const std::string & name) const;
  /// Is "name" present? Returns index + 1 if found, else 0
  size_t is(const std::string & name) const;

  /// Number of settings in the table
  size_t size() const;
  /// Name of the i:th setting
  const std::string & name(size_t i) const;
  /// Print the i:th setting in this type's format
  void print_entry(size_t i) const;
};

/// Settings used for a calculation
class Settings {
  /// Double precision number value settings
  SettingTable<double> dset;
  /// Boolean value settings
  SettingTable<bool> bset;
  /// Integer value settings
  SettingTable<int> iset;
  /// String value settings
  SettingTable<std::string> sset;

 public:
  /// Constructor
  Settings();
  /// Destructor
  ~Settings();

  /// Add SCF related settings
  void add_scf_settings();
  /// Add the settings consumed by JKBuilder::configure (J/K build method,
  /// integral thresholds, fitting basis). Called by add_scf_settings; called
  /// directly by tools that drive a JKBuilder without the full SCF settings.
  void add_jk_settings();

  /// Add a double valued setting
  void add_double(const std::string & name, const std::string & comment, double val, bool negative=false);
  /// Add a boolean valued setting
  void add_bool(const std::string & name, const std::string & comment, bool val);
  /// Add an integer valued setting
  void add_int(const std::string & name, const std::string & comment, int val, bool negative=false);
  /// Add a string valued setting
  void add_string(const std::string & name, const std::string & comment, const std::string & val);

  /// Set a double valued setting
  void set_double(const std::string & name, double val);
  /// Set a boolean valued setting
  void set_bool(const std::string & name, bool val);
  /// Set an integer valued setting
  void set_int(const std::string & name, int val);
  /// Set a string valued setting
  void set_string(const std::string & name, const std::string & val);

  /// Get a double valued setting
  double get_double(const std::string & name) const;
  /// Get a boolean valued setting
  bool get_bool(const std::string & name) const;
  /// Get an integer valued setting
  int get_int(const std::string & name) const;
  /// Get a string valued setting
  std::string get_string(const std::string & name) const;

  /// Get a string setting and parse it as a vector
  arma::vec  get_vec(const std::string & name)  const;
  /// Get a string setting and parse it as an integer vector
  arma::ivec get_ivec(const std::string & name) const;
  /// Get a string setting and parse it as an unsigned integer vector
  arma::uvec get_uvec(const std::string & name) const;

  /// Is "name" a setting of double type? Returns index + 1 if found, else 0.
  size_t is_double(const std::string & name) const;
  /// Is "name" a setting of boolean type? Returns index + 1 if found, else 0.
  size_t is_bool(const std::string & name) const;
  /// Is "name" a setting of integer type? Returns index + 1 if found, else 0.
  size_t is_int(const std::string & name) const;
  /// Is "name" a setting of string type? Returns index + 1 if found, else 0.
  size_t is_string(const std::string & name) const;

  /// Parse file containing settings to use. SCF indicates special handling for the method keyword
  void parse(std::string filename, bool scf=false);

  /// Print current settings
  void print() const;
};

#endif
