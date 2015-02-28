/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_EMDSIM
#define ERKALE_EMDSIM

#include "../global.h"
#include <armadillo>
class BasisSet;


/**
 * Compute EMD overlap integrals
 * Returns a cube [k][corr][iso] containing
 * k values in -1, 0, 1, 2
 * corr: S_{AA} S_{BB} S_{AB}
 * iso: 0 for general 3D integral, 1 for spherical averaged
 *
 * This algorithm uses a seminumerical routine to perform the integrals -
 * the momentum densities are expanded in spherical harmonics on a radial grid.
 */
arma::cube emd_overlap_semi(const BasisSet & bas_a, const arma::mat & P_a, const BasisSet & bas_b, const arma::mat & P_b, int nrad=500, int lmax=6, bool verbose=true);


/**
 * Compute EMD overlap integrals
 * Returns a cube [k][corr][iso] containing
 * k values in -1, 0, 1, 2
 * corr: S_{AA} S_{BB} S_{AB}
 * iso: 0 for general 3D integral, 1 for spherical averaged
 *
 * For the used equations, see J. Vandenbussche, G. Acke, and Patrick
 * Bultinck, "Performance of DFT Methods in Momentum Space: Quantum
 * Similarity Measures versus Moments of Momentum", JCTC 9 (2013),
 * 3908.
 */
arma::cube emd_overlap(const BasisSet & bas_a, const arma::mat & P_a, const BasisSet & bas_b, const arma::mat & P_b, int nrad=500, int lmax=77, bool verbose=true);

/**
 * Compute shape function similarity from EMD overlap integrals
 * Returns a cube [k][corr][iso] containing
 * k values in -1, 0, 1, 2
 * corr: S_{AA} S_{BB} S_{AB} I_{AA} I_{BB} I_{AB} D_{AB}
 * iso: 0 for general 3D integral, 1 for spherical averaged
 *
 * For the used equations, see J. Vandenbussche, G. Acke, and Patrick
 * Bultinck, "Performance of DFT Methods in Momentum Space: Quantum
 * Similarity Measures versus Moments of Momentum", JCTC 9 (2013),
 * 3908.
 */
arma::cube emd_similarity(const arma::cube & emd, int Nela, int Nelb);

#endif
