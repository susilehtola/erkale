/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_CROSSBASIS
#define ERKALE_CROSSBASIS

#include "global.h"
#include "basis.h"

#include <vector>

/**
 * Two-electron matrices in one basis set (the target) from a density
 * given in another (the source), with no projection between the bases:
 * the Fock matrix of a density from a different basis, as in the
 * projection-free initial guess on a basis change or the
 * inter-particle Coulomb interaction of multicomponent (NEO)
 * calculations. The interaction is alpha/r + beta erfc(omega r)/r;
 * the defaults give the plain Coulomb operator.
 */

/// Coulomb matrix J_uv = sum_ls (uv|ls) P_ls; u,v in the target, l,s in the source basis
arma::mat cross_basis_J(const BasisSet & target, const BasisSet & source, const arma::mat & Psource, double thr, double omega=0.0, double alpha=1.0, double beta=0.0);

/// Exchange matrix K_uv = sum_ls (ul|vs) P_ls of a symmetric source density
arma::mat cross_basis_K(const BasisSet & target, const BasisSet & source, const arma::mat & Psource, double thr, double omega=0.0, double alpha=1.0, double beta=0.0);

/**
 * Density-fitted variants on the auxiliary basis aux: the cost of one
 * density-fitted Fock build. The fit uses the same operator as the
 * integrals; fitthr is the eigenvalue cutoff of the fitting metric.
 */
/// Density-fitted Coulomb matrix in the target basis of Psource
arma::mat cross_basis_J_df(const BasisSet & target, const BasisSet & source, const BasisSet & aux, const arma::mat & Psource, double fitthr);
/// Density-fitted exchange matrices K_uv = sum_i (ui|vi) in the target
/// basis of the orbitals C[k] in the source basis, scaled by the square
/// roots of their occupations, one matrix per set of orbitals
std::vector<arma::mat> cross_basis_K_df(const BasisSet & target, const BasisSet & source, const BasisSet & aux, const std::vector<arma::mat> & C, double fitthr, double omega=0.0, double alpha=1.0, double beta=0.0);

#endif
