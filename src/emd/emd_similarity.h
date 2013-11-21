#ifndef ERKALE_EMDSIM
#define ERKALE_EMDSIM

#include "global.h"
#include "basis.h"

/**
 * Compute similarity measures between momentum densities.
 * Returns an 8-element vector, containing
 * D_{AB} (-1) D_{AB} (0) D_{AB} (1) D_{AB} (2)
 * D0_{AB} (-1) D0_{AB} (0) D0_{AB} (1) D0_{AB} (2)
 *
 * For the used equations, see J. Vandenbussche, G. Acke, and Patrick
 * Bultinck, "Performance of DFT Methods in Momentum Space: Quantum
 * Similarity Measures versus Moments of Momentum", JCTC 9 (2013),
 * 3908.
 */
arma::mat emd_similarity(const BasisSet & bas_a, const arma::mat & P_a, const BasisSet & bas_b, const arma::mat & P_b, int nrad=500, int lmax=77, bool verbose=true);


#endif
