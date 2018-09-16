#ifndef SAP_POT_H
#define SAP_POT_H

#include "global.h"
#include <armadillo>
#include <vector>

class SAP {
  /// Atomic data
  std::vector<arma::mat> atoms;
 public:
  /// Constructor
  SAP();
  /// Destructor
  ~SAP();

  /// Get potential
  double get(int Z, double r) const;
};

#endif

