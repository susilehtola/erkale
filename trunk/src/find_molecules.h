#ifndef ERKALE_FINDMOL
#define ERKALE_FINDMOL

#include <vector>

#include "xyzutils.h"

/// Find probable molecules in input
std::vector< std::vector<size_t> > find_molecules(const std::vector<atom_t> & atoms);

/// Check if atoms are probably bonded together, units are Ångström
bool check_bonds(double d, const std::string & atom1, const std::string & atom2, double tol=0.3);
/// Check pair for bond, units are Ångström
bool check_bond(const std::string & atom1, const std::string & atom2, double d, const std::string & test1, const std::string & test2, double maxlen);

#endif
