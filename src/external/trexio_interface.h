/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_TREXIO_INTERFACE
#define ERKALE_TREXIO_INTERFACE

#include <string>

/**
 * TREXIO (https://trex-coe.github.io/trexio/) interoperability.
 *
 * Converts between ERKALE's HDF5 checkpoint (.chk) and the TREXIO
 * wavefunction format. The wavefunction groups -- metadata, nucleus,
 * electron, basis, ao, mo -- are mapped both ways. The export also
 * writes the one-electron integrals (overlap, kinetic, electron-nucleus
 * potential, core Hamiltonian, dipole) in the AO and MO bases, and
 * optionally the AO electron repulsion integrals; the import ignores
 * integrals.
 *
 * Shells are cartesian or spherical, flagged for the whole basis
 * (ao.cartesian) or shell by shell (ao.cartesian_shell, which needs a
 * libtrexio newer than 2.6.1). The export writes the global flag unless
 * the basis mixes cartesian and spherical d+ shells. The convention
 * subtleties are the per-shell ordering of the real spherical
 * harmonics -- ERKALE stores them as m = -l..+l, TREXIO as
 * m = 0,+1,-1,+2,-2,..., and the MO-coefficient rows are permuted
 * accordingly -- and the normalization: ERKALE's functions, cartesian
 * ones included, are unit-normalized, while the import rescales the MO
 * coefficients by the norms of the functions the file defines. The
 * export is self-checked by comparing TREXIO's computed AO overlap
 * against ERKALE's, which also catches any normalization mismatch.
 */

/// Write the wavefunction in the ERKALE checkpoint chkfile to a TREXIO
/// file (HDF5 back end), including the one-electron integrals and, if
/// eri is set, the AO electron repulsion integrals. Overwrites
/// trexiofile if it exists.
void chk_to_trexio(const std::string & chkfile, const std::string & trexiofile, bool eri=false, bool verbose=true);

/// Read a TREXIO wavefunction and write it as an ERKALE checkpoint.
void trexio_to_chk(const std::string & trexiofile, const std::string & chkfile, bool verbose=true);

#endif
