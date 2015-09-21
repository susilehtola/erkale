/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
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
 * \author Susi Lehtola
 * \date 2011/11/25 13:16
 */



#ifndef ERKALE_GLOBAL
#define ERKALE_GLOBAL

// Disable bounds checking in Armadillo.
#define ARMA_NO_DEBUG
// We need BLAS
#define ARMA_USE_BLAS
// and LAPACK
#define ARMA_USE_LAPACK

// Ångström in atomic units
#define ANGSTROMINBOHR 1.8897261
// Atomic unit in eV, http://physics.nist.gov/cgi-bin/cuu/Value?threv
#define HARTREEINEV 27.21138505
// Atomic unit in debye, http://en.wikipedia.org/wiki/Debye
#define AUINDEBYE 0.393430307
// Fine structure constant
#define FINESTRUCT 7.2973525540510957E-3

// Degree in radians
#define DEGINRAD (M_PI/180.0)

// Initial tolerance, when density-based screening is used
#define ROUGHTOL 1e-9
// Fine tolance after initial convergence has been achieved (minimum)
#define FINETOL  1e-10
// When to switch to FINETOL (wrt. rms difference of density matrices)
#define TOLCHANGE 1e-5

// Tolerance when screening is only wrt absolute value of integrals
#define STRICTTOL 1e-16

// Threshold for linear independence
#define LINTHRES 1e-5

// Shorthands
#define COMPLEX1 std::complex<double>(1.0,0.0)
#define COMPLEXI std::complex<double>(0.0,1.0)

// Error info
#define ERROR_INFO() printf("\nError in function %s (file %s, near line %i)\n",__FUNCTION__,__FILE__,__LINE__);
// Print out location
#define LOC_INFO() {printf("Hello from function %s (file %s, near line %i)\n",__FUNCTION__,__FILE__,__LINE__); fflush(stdout);}

// Check that matrix is of wanted size
#define MAT_SIZE_CHECK(M,NR,NC) if(M.n_rows != NR || M.n_cols != NC) { \
    std::ostringstream oss;						\
    oss << #M << " should be " << NR << " x " << NC << " but is " << M.n_rows << " x " << M.n_cols << "!\n"; \
      throw std::runtime_error(oss.str());}
// Resize matrix if necessary
#define MAT_RESIZE(M,NR,NC) if(M.n_rows != NR || M.n_cols != NC) { M.zeros(NR,NC);}

#define print_copyright() \
  printf("(c) Susi Lehtola, 2010-2015.\n");

#define print_license() \
  printf("\n%s%s%s%s\n",							\
	 "This program is free software; you can redistribute it and/or modify\n", \
	 "it under the terms of the GNU General Public License as published by\n", \
	 "the Free Software Foundation; either version 2 of the License, or\n", \
	 "(at your option) any later version.\n")

#define print_hostname()				\
  {char _hname[4096];					\
  int _herr=gethostname(_hname,4096);			\
  if(! _herr) printf("Running on host %s.\n\n",_hname);	\
  else fprintf(stderr,"Error getting hostname.\n");}

#endif
