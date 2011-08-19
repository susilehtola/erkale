#ifndef ERKALE_OPTCOMP
#define ERKALE_OPTCOMP

#include <vector>

extern "C" {
#include <gsl/gsl_vector.h>
}

/// Parameters for completeness scan
typedef struct {
  /// Angular momentum of shell
  int am;
  /// Scanning exponents
  std::vector<double> scanexp;
} completeness_scan_t;

/// Helper function - evaluate completeness. v holds logarithms of exponents, params is pointer to completeness_scan_t
double evaluate_completeness(const gsl_vector *v, void *params);
/// Wrapper for the above
double evaluate_completeness(const std::vector<double> & v, completeness_scan_t p);

/// Find out exponents in completeness optimized basis set.
std::vector<double> optimize_completeness(int am, double min, double max, int Nf);
/// Same, using algorithms in GSL
std::vector<double> optimize_completeness_gsl(int am, double min, double max, int Nf);


#endif
