/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_POPULATION
#define ERKALE_POPULATION

#include "basis.h"

/**
 * The population analysis stuff is based on the book "Simple
 * theorems, proofs, and derivations in Quantum Chemistry" by István
 * Mayer (IM), Kluwer Academic 2003.
 */



/**
 * Computes Mulliken's overlap population
 * \f$ d_{AB} = \sum_{\mu \in A} \sum_{\nu in B} P_{\mu \nu} S_{\mu \nu} \f$
 * (IM) eqn 7.17
 */
arma::mat mulliken_overlap(const BasisSet & basis, const arma::mat & P);

/**
 * Compute bond order index
 * \f$ B_{AB} = \sum_{\mu \in A} \sum_{\nu in B} ({\mathbf PS})_{\mu \nu} \f$
 * (IM) eqn 7.35
 */
arma::mat bond_order(const BasisSet & basis, const arma::mat & P);

/**
 * Compute bond order index for open-shell case
 * (IM) eqn 7.36
 */
arma::mat bond_order(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb);

/**
 * Compute electron density at nuclei
 */
arma::vec nuclear_density(const BasisSet & basis, const arma::mat & P);


/**
 * Do all of the above.
 */
void population_analysis(const BasisSet & basis, const arma::mat & P);
void population_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb);

/**
 * Get Darwin one-electron term.
 */
double darwin_1e(const BasisSet & basis, const arma::mat & P);

/**
 * Get mass-velocity term.
 */
double mass_velocity(const BasisSet & basis, const arma::mat & P);

#endif
