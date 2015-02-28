/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_ERIDIGEST
#define ERKALE_ERIDIGEST

#include "basis.h"
class dERIWorker;

/// Integral digestor
class IntegralDigestor {
 public:
  /// Constructor
  IntegralDigestor();
  /// Destructor
  virtual ~IntegralDigestor();
  /// Digest integral block
  virtual void digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff)=0;
};

/// Coulomb matrix digestor
class JDigestor: public IntegralDigestor {
  /// Density matrix
  arma::mat P;
  /// Coulomb matrix
  arma::mat J;
 public:
  /// Construct digestor
  JDigestor(const arma::mat & P);
  /// Destruct digestor
  ~JDigestor();
  
  /// Digest integrals
  void digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff);
  /// Get output
  arma::mat get_J() const;
};

/// Exchange matrix digestor
class KDigestor: public IntegralDigestor {
  /// Density matrix
  arma::mat P;
  /// Exchange matrix
  arma::mat K;
 public:
  /// Construct digestor
  KDigestor(const arma::mat & P);
  /// Destruct digestor
  ~KDigestor();
  
  /// Digest integrals
  void digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff);
  /// Get output
  arma::mat get_K() const;
};

/// Complex exchange matrix digestor
class cxKDigestor: public IntegralDigestor {
  /// Density matrix
  arma::cx_mat P;
  /// Exchange matrix
  arma::cx_mat K;
 public:
  /// Construct digestor
  cxKDigestor(const arma::cx_mat & P);
  /// Destruct digestor
  ~cxKDigestor();
  
  /// Digest integrals
  void digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff);
  /// Get output
  arma::cx_mat get_K() const;
};

/// Force digestor
class ForceDigestor {
 public:
  /// Constructor
  ForceDigestor();
  /// Destructor
  virtual ~ForceDigestor();
  /// Digest derivative block
  virtual void digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, dERIWorker & deriw, arma::vec & f)=0;
};

/// Coulomb force digestor
class JFDigestor: public ForceDigestor {
  /// Density matrix
  arma::mat P;
 public:
  /// Construct digestor
  JFDigestor(const arma::mat & P);
  /// Destruct digestor
  ~JFDigestor();
  
  /// Digest integrals
  void digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, dERIWorker & deriw, arma::vec & f);
};

/// Exchange force digestor
class KFDigestor: public ForceDigestor {
  /// Density matrix
  arma::mat P;
  /// Fraction of exact exchange
  double kfrac;
  /// Degeneracy factor
  double fac;
  
 public:
  /// Construct digestor
  KFDigestor(const arma::mat & P, double kfrac, bool restr);
  /// Destruct digestor
  ~KFDigestor();
  
  /// Digest integrals
  void digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, dERIWorker & deriw, arma::vec & f);
};

#endif
