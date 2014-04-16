/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                     HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/*
 * This file contains routines for parsing formatted checkpoint files.
 */


#ifndef ERKALE_FCHKPT
#define ERKALE_FCHKPT

#include "basis.h"
#include "checkpoint.h"
#include "mathf.h"
#include "storage.h"
#include "stringutil.h"
#include "timer.h"

/// Parse formatted checkpoint file
Storage parse_fchk(const std::string & name);

/// Form the basis set from the checkpoint file
BasisSet form_basis(const Storage & stor);

/// Form the density matrix
arma::mat form_density(const Storage & stor, const std::string & kw);

/// Form the density matrix. spin toggles spin density, scf toggles reading scf density (post-HF by default if available)
arma::mat form_density(const Storage & stor, bool spin=false, bool scf=false);

/// Get the orbital coefficient matrix
arma::mat form_orbital_C(const Storage & stor, const std::string & name);

/// Get the orbital energies
arma::vec form_orbital_E(const Storage & stor, const std::string & name);

/*
 * The below are routines needed by the ones above.
 */

/// Form the ERKALE to Gaussian index conversion array
std::vector<size_t> eg_indarr(const std::vector<int> shtype, size_t Nbf);

/// Form the ERKALE to Gaussian index conversion array
std::vector<size_t> eg_indarr(const Storage & stor);

/// Form the Gaussian to ERKALE index conversion array
std::vector<size_t> ge_indarr(const std::vector<int> shtype, size_t Nbf);

/// Form the Gaussian to ERKALE index conversion array
std::vector<size_t> ge_indarr(const Storage & stor);

#endif
