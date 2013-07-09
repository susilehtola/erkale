/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010
 * Copyright (c) 2010, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 */

#include "global.h"

#ifndef ERKALE_BASISLIB
#define ERKALE_BASISLIB

#include <armadillo>
#include <vector>
#include <string>

#include "basis.h"

/// Find angular momentum
int find_am(char am);

/// Find basis set file
std::string find_basis(const std::string & filename, bool verbose=true);

/// System default location for basis sets
#ifndef ERKALE_SYSTEM_LIBRARY
#define ERKALE_SYSTEM_LIBRARY "/usr/share/erkale/basis"
#endif

/**
 * \class FunctionShell
 *
 * \brief A shell of functions
 *
 * This class defines a shell of functions of the same angular
 * momentum, used in the ElementBasisSet class.
 *
 * \author Susi Lehtola
 * \date 2011/05/08 17:10
 */

class FunctionShell {
  /// Angular momentum
  int am;
  /// Exponential contraction
  std::vector<contr_t> C;

 public:
  /// Construct a shell with angular momentum am
  FunctionShell(int am=-1);
  /// Construct a shell with angular momentum am and given contraction
  FunctionShell(int am, const std::vector<contr_t> & c);
  /// Destructor
  ~FunctionShell();

  /// Add exponent into contraction
  void add_exponent(double C, double z);
  /// Comparison operator for ordering in decreasing angular momentum and exponent
  bool operator<(const FunctionShell &rhs) const;

  /// Get angular momentum
  int get_am() const;

  /// Get contraction coefficients
  std::vector<contr_t> get_contr() const;

  /// Sort exponents in decreasing order
  void sort();
  /// Normalize coefficients
  void normalize();
  /// Print out info
  void print() const;

  friend class BasisSetLibrary;
};

/**
 * \class ElementBasisSet
 *
 * \brief Basis set for an element
 *
 * This class defines a basis set for an element, used by the
 * BasisSetLibrary class.
 *
 * \author Susi Lehtola
 * \date 2011/05/08 17:10
 */

class ElementBasisSet {

  /// Symbol of element
  std::string symbol;
  /// Atom id for which the basis is for (0 for all atoms)
  size_t number;
  /// List of shells
  std::vector<FunctionShell> bf;

 public:
  /// Dummy constructor
  ElementBasisSet();
  /// Constructor
  ElementBasisSet(std::string sym, size_t number=0);
  /// Destructor
  ~ElementBasisSet();

  /// Add a shell of functions to the basis
  void add_function(FunctionShell f);
  /// Sort the shells in decreasing angular momentum
  void sort();
  /// Print out basis set
  void print() const;

  /// Get the symbol of the element
  std::string get_symbol() const;
  /// Get the number
  size_t get_number() const;
  /// Set the number
  void set_number(size_t num);

  /// Comparison operator for sorting
  bool operator<(const ElementBasisSet &rhs) const;

  /// Get the shells
  std::vector<FunctionShell> get_shells() const;

  /// Get exponents and contraction coefficients of angular momentum shell am
  void get_primitives(std::vector<double> & exps, arma::mat & coeffs, int am) const;

  /// Get maximum angular momentum used in the shells
  int get_max_am() const;
  /// Get angular momentum of i:th shell
  int get_am(size_t ind) const;

  /// Normalize coefficients
  void normalize();

  /// Decontract set
  void decontract();

  /// Augment the basis
  void augment(int naug);

  friend class BasisSetLibrary;
};


/**
 * \class BasisSetLibrary
 *
 * \brief Basis set library class
 *
 * This class defines a basis set library class that can be used
 * e.g. to read in basis sets from files, or to save basis sets to
 * files.
 *
 * \author Susi Lehtola
 * \date 2011/05/08 17:10
 */

class BasisSetLibrary {

  /// Name of basis set
  std::string name;
  /// List of elements included in basis set
  std::vector<ElementBasisSet> elements;

 public:
  /// Constructor
  BasisSetLibrary();
  /// Destructor
  ~BasisSetLibrary();

  /// Load basis set from file in Gaussian'94 format
  void load_gaussian94(const char * filename, bool verbose=true);
  /// Load basis set from file in Gaussian'94 format
  void load_gaussian94(const std::string & filename, bool verbose=true);

  /// Save basis set to file in Gaussian'94 format
  void save_gaussian94(const char * filename, bool append=false) const;
  /// Save basis set to file in Gaussian'94 format
  void save_gaussian94(const std::string & filename, bool append=false) const;

  /// Save basis set to file in Dalton format
  void save_dalton(const char * filename, bool append=false) const;
  /// Save basis set to file in Dalton format
  void save_dalton(const std::string & filename, bool append=false) const;

  /// Add element to basis set
  void add_element(const ElementBasisSet & el);

  /// Sort library
  void sort();

  /// Get number of elements
  size_t get_Nel() const;
  /// Get symbol of ind'th element
  std::string get_symbol(size_t ind) const;
  /// Get elements
  std::vector<ElementBasisSet> get_elements() const;

  /// Get maximum angular momentum used in basis set
  int get_max_am() const;

  /// Get basis set for wanted element
  ElementBasisSet get_element(std::string el, size_t number=0) const;

  /// Normalize coefficients
  void normalize();

  /// Decontract basis set
  void decontract();

  /// Augment the basis
  void augment(int naug);

  /// Print out library
  void print() const;
};


#endif