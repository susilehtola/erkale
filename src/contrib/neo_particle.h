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

#ifndef ERKALE_NEO_PARTICLE
#define ERKALE_NEO_PARTICLE

#include "../settings.h"
#include "../stringutil.h"

#include <cstdio>

/// Mass of the proton in atomic units
#define PROTON_MASS 1836.15267389

/**
 * The quantum particle treated by the multicomponent codes. It is a
 * proton by default; a deuteron, a triton, a muon or a positron is
 * obtained by giving its mass and charge.
 */
typedef struct {
  /// Mass in atomic units (the electron mass is one)
  double m;
  /// Charge in atomic units (the electron charge is minus one)
  double q;
} quantum_particle_t;

/// Register the settings that define the quantum particle
inline void add_particle_settings(Settings & set) {
  set.add_double("ParticleMass", "Mass of the quantum particle (proton by default)", PROTON_MASS);
  set.add_double("ParticleCharge", "Charge of the quantum particle (proton by default); negative values are allowed", 1.0, true);
  // Deprecated name of ParticleMass, kept so that old input files run
  set.add_double("ProtonMass", "Deprecated: use ParticleMass", PROTON_MASS);
}

/// Get the quantum particle defined by the settings
inline quantum_particle_t get_particle(const Settings & set) {
  quantum_particle_t p;
  p.m = set.get_double("ParticleMass");
  p.q = set.get_double("ParticleCharge");

  // The deprecated setting wins if it was given, so that old input
  // files keep working
  const double pm = set.get_double("ProtonMass");
  if(pm != PROTON_MASS) {
    printf("Warning: the setting ProtonMass is deprecated, use ParticleMass instead.\n");
    fflush(stdout);
    p.m = pm;
  }

  if(p.m <= 0.0)
    throw std::runtime_error("The mass of the quantum particle must be positive.\n");
  if(p.q == 0.0)
    throw std::runtime_error("The charge of the quantum particle must be non-zero: a neutral particle does not interact with the electrons.\n");

  return p;
}

/// Print out the quantum particle
inline void print_particle(const quantum_particle_t & p) {
  printf("Quantum particle: mass % .8f, charge % .4f atomic units.\n", p.m, p.q);
  fflush(stdout);
}

#endif
