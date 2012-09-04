/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/*! \mainpage ERKALE - DFT from Hel
 * 
 * \section intro_sec Introduction
 * 
 * ERKALE is a code for Hartree-Fock and density-functional theory
 * calculations for atoms and molecules. It uses a Gaussian basis set
 * for representing the molecular orbitals and the electron density.
 * 
 * ERKALE is written in C++ and uses the Armadillo template library
 * for linear algebra operations.
 *
 * ERKALE is designed to be as easily maintained and user friendly as
 * possible, but also to try to be reasonably fast.
 *
 *
 * \section why_another_code Why yet another code?
 *
 * I wanted to do some research on the modeling of inelastic x-ray
 * scattering. This would require low-level access to a quantum
 * chemical code.
 *
 * The code would need to have a gentle learning curve, be reasonably
 * fast for "production" use - and be free, so that I and others could
 * use the code anywhere they liked.
 *
 * As I did not find suitable existing programs on the market, I
 * decided to write my own program. This would mean spending more time
 * in development, although with the benefit of getting to grips with
 * low level stuff. The decision was made a lot simpler due to the
 * availability of fast, free libraries for computing electron
 * repulsion integrals (libint) and exchange-correlation functionals
 * (libxc), which meant that most of the dull stuff was already done
 * by others.
 *
 * Being free is also important because scientific results need to be
 * reproducible. The path from equations to results is often very long
 * in computational science; a working code used to implement the
 * equations is (at least!) as important as the equations or the
 * algorithms themselves. To guarantee that the code stays available,
 * I have chosen the GNU General public license, which is commonly
 * used in other scientific software as well.
 *
 * \author Jussi Lehtola
 * \date 2011/11/25 13:16
 */



#ifndef ERKALE_GLOBAL
#define ERKALE_GLOBAL

#include <cstdio>

// Disable bounds checking in Armadillo.
#define ARMA_NO_DEBUG

// Ångström in atomic units
#define ANGSTROMINBOHR 1.8897261
// Atomic unit in eV, http://physics.nist.gov/cgi-bin/cuu/Value?threv
#define HARTREEINEV 27.21138505
// Atomic unit in debye, http://en.wikipedia.org/wiki/Debye
#define AUINDEBYE 0.393430307
// Fine structure constant
#define FINESTRUCT 7.2973525540510957E-3

// Initial tolerance, when density-based screening is used
#define ROUGHTOL 1e-9
// Tolerance when screening is only wrt absolute value of integrals
#define STRICTTOL 1e-16

// Threshold for linear independence
#define LINTHRES 1e-5



// Error info
#define ERROR_INFO() printf("\nError in function %s (file %s, near line %i)\n",__FUNCTION__,__FILE__,__LINE__)

// Check that matrix is of wanted size
#define MAT_SIZE_CHECK(M,NR,NC,MSG) if(M.n_rows != NR || M.n_cols != NC) { \
    ERROR_INFO(); throw std::runtime_error(MSG);}
// Resize matrix if necessary
#define MAT_RESIZE(M,NR,NC) if(M.n_rows != NR || M.n_cols != NC) { M=arma::mat(NR,NC);}

#define print_license() \
  printf("\n%s%s%s%s\n",							\
	 "This program is free software; you can redistribute it and/or modify\n", \
	 "it under the terms of the GNU General Public License as published by\n", \
	 "the Free Software Foundation; either version 2 of the License, or\n", \
	 "(at your option) any later version.\n")

#endif
