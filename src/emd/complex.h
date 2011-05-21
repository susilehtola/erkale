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



#ifndef ERKALE_COMPLEX
#define ERKALE_COMPLEX

/// Complex number
typedef struct {
  /// Real part
  double re;
  /// Imaginary part
  double im;
} complex;

/// Compute complex conjugate \f$ z^* \f$
complex cconj(const complex no);
/// Compute \f$ -z \f$
complex cneg(const complex no);
/// Compute complex exponential \f$ e^z \f$
complex cexp(const complex no);

/// Compute \f$ \sin(z) \f$
complex csin(const complex no);
/// Compute \f$ \cos(z) \f$
complex ccos(const complex no);
/// Compute \f$ \sinh(z) \f$
complex csinh(const complex no);
/// Compute \f$ \cosh(z) \f$
complex ccosh(const complex no);

/// Complex multiplication
complex cmult(const complex lhs, const complex rhs);
/// Complex division
complex cdiv(const complex lhs, const complex rhs);
/// Complex addition
complex cadd(const complex lhs, const complex rhs);
/// Complex substraction
complex csub(const complex lhs, const complex rhs);
/// Compute \f$ z^m \f$
complex cpow(complex no, int m);

/// Compute \f$ a^* b \f$
complex cconjmult(const complex a, const complex b);

/// Compute \f$ |z|^2 \f$
double cnormsq(const complex no);
/// Compute \f$ |z| \f$
double cnorm(const complex no);

/// Scale number with a factor fac
complex cscale(const complex no, const double fac);

/// Print number
void cprint(const complex no);

#endif
