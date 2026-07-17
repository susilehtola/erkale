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
 * electron, basis, ao, mo -- are mapped both ways; integrals are not
 * (yet) exported.
 *
 * The one convention subtlety is the per-shell ordering of the real
 * spherical harmonics: ERKALE stores them as m = -l..+l, TREXIO as
 * m = 0,+1,-1,+2,-2,...; the MO-coefficient rows are permuted
 * accordingly. The export is self-checked by comparing TREXIO's
 * computed AO overlap against ERKALE's, which also catches any
 * normalization mismatch.
 */

/// Write the wavefunction in the ERKALE checkpoint chkfile to a TREXIO
/// file (HDF5 back end). Overwrites trexiofile if it exists.
void chk_to_trexio(const std::string & chkfile, const std::string & trexiofile, bool verbose=true);

/// Read a TREXIO wavefunction and write it as an ERKALE checkpoint.
void trexio_to_chk(const std::string & trexiofile, const std::string & chkfile, bool verbose=true);

#endif
