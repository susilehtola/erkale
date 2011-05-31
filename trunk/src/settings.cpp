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



#include "settings.h"
#include "stringutil.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>


Settings::Settings(bool usedft) {
  // Set default Settings

  dft=usedft;

  // Use spherical harmonics.
  bset.push_back(genb("UseLM", "Use a spherical harmonics basis set by default?", 1));

  // Use DIIS.
  bset.push_back(genb("UseDIIS", "Use Pulay's Direct Inversion in the Iterative Subspace?", 1));
  // Use old version of DIIS?
  bset.push_back(genb("C1-DIIS", "Use older version of DIIS (C1-DIIS instead of C2-DIIS)?", 0));
  // Number of DIIS matrices to use?
  iset.push_back(geni("DIISOrder", "How many DIIS iterations to keep in memory?", 20));
  // DIIS threshold
  dset.push_back(gend("DIISThr", "DIIS error threshold for DIIS updates", 0.05));

  // Use ADIIS?
  bset.push_back(genb("UseADIIS", "Use ADIIS for Fock matrix interpolation? (DIIS takes preference)", 1));
  // Use Broyden mixing?
  bset.push_back(genb("UseBroyden", "Use Broyden mixing of Fock matrices?", 0));
  
  // Mix density matrices
  bset.push_back(genb("MixDensity", "Mix density matrices?", 0));
  // Dynamically update mixing factor?
  bset.push_back(genb("DynamicMixing","Dynamically change mixing factor", 1));
  
  // Total charge of system
  iset.push_back(geni("Charge", "Total charge of system", 0));
  // Multiplicity
  iset.push_back(geni("Multiplicity", "Spin multiplicity", 1));

  // Verbose run?
  bset.push_back(genb("Verbose", "Verbose calculation?", 1));

  // Direct calculation?
  bset.push_back(genb("Direct", "Calculate two-electron integrals (or density fitting) on-the-fly?", 0));

  // Freeze core orbitals?
  bset.push_back(genb("FrozenCore", "Freeze core orbitals? (For XRS calculations)", 0));

  // Default orthogonalization method
  sset.push_back(gens("BasisOrth", "Method of orthonormalization of basis set", "Can"));

  // Default cutoff for orthogonalization
  dset.push_back(gend("BasisLinTol", "Cutoff for linearly dependent basis functions", 1e-5));

  // Density matrix convergence criteria
  dset.push_back(gend("DeltaPrms", "Maximum allowed RMS difference of density matrix", 1e-8));
  dset.push_back(gend("DeltaPmax", "Maximum allowed maximum difference of density matrix", 1e-6));
  dset.push_back(gend("DeltaEmax", "Maximum allowed change of energy", 1e-6));

  // Maximum iterations
  iset.push_back(geni("MaxIter", "Maximum number of iterations in SCF cycle", 100));

  // Calculate EMD properties?
  bset.push_back(genb("DoEMD", "Perform EMD calculation (moments of EMD, Compton profile)", 0));

  if(!usedft) {
    // Initialize HF calculation with a SVWN calculation
    sset.push_back(gens("InitMethod","Method of initializing calculation","lda_x-lda_c_vwn"));
  } else {
    // We probably don't want to initialize a DFT calculation (core guess is fine)
    sset.push_back(gens("InitMethod","Method of initializing calculation","none"));

    // Store full DFT grid in memory?
    bset.push_back(genb("DFTDirect", "Save memory by not storing values of basis functions in memory", 0));
    // Store full DFT grid in memory?
    bset.push_back(genb("DFTLobatto", "Use Lobatto quadrature instead of Lebedev quadrature?", 0));
    
    // Initial and final tolerances of DFT grid
    dset.push_back(gend("DFTInitialTol", "Tolerance of initial DFT grid", 1e-3));
    dset.push_back(gend("DFTFinalTol", "Tolerance of final DFT grid", 5e-5));
    // When to switch to final grid?
    dset.push_back(gend("DFTSwitch", "When to switch to final grid (relative to deltaE, deltaP)?", 50.0));
    
    // Default DFT exchange and correlation functionals
    sset.push_back(gens("DFT_XC", "DFT exchange and correlation (or exchange-correlation) functional", "gga_x_rpbe-gga_c_pbe"));
    
    // Use density fitting if possible?
    bset.push_back(genb("DensityFitting", "Use density fitting if possible? (Pure DFT functionals)", 1));
  }
}


Settings::~Settings() {
}

void Settings::set_double(std::string name, double val) {
  // Find setting in table
  for(size_t i=0;i<dset.size();i++)
    if(name==dset[i].name) {
      dset[i].val=val;
      return;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());
}

void Settings::set_bool(std::string name, bool val) {
  // Find setting in table
  for(size_t i=0;i<bset.size();i++)
    if(name==bset[i].name) {
      bset[i].val=val;
      return;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());
}

void Settings::set_int(std::string name, int val) {
  // Find setting in table
  for(size_t i=0;i<iset.size();i++)
    if(name==iset[i].name) {
      iset[i].val=val;
      return;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());
}


void Settings::set_string(std::string name, std::string val) {
  // Find setting in table
  for(size_t i=0;i<sset.size();i++)
    if(name==sset[i].name) {
      sset[i].val=val;
      return;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());
}



double Settings::get_double(std::string name) const {
  // Find setting in table
  for(size_t i=0;i<dset.size();i++)
    if(name==dset[i].name) {
      return dset[i].val;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  return 0.0;
}

bool Settings::get_bool(std::string name) const {
  // Find setting in table
  for(size_t i=0;i<bset.size();i++)
    if(name==bset[i].name) {
      return bset[i].val;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  return 0;
}



int Settings::get_int(std::string name) const {
  // Find setting in table
  for(size_t i=0;i<iset.size();i++)
    if(name==iset[i].name) {
      return iset[i].val;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  return 0;
}

std::string Settings::get_string(std::string name) const {
 // Find setting in table
  for(size_t i=0;i<sset.size();i++)
    if(name==sset[i].name) {
      return sset[i].val;
    }

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nThe setting "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  return "";
}

bool Settings::dft_enabled() const {
  return dft;
}

bool Settings::is_double(std::string name) const {
  for(size_t i=0;i<dset.size();i++)
    if(name==dset[i].name)
      return 1;

  return 0;
}

bool Settings::is_int(std::string name) const {
  for(size_t i=0;i<iset.size();i++)
    if(name==iset[i].name)
      return 1;

  return 0;
}

bool Settings::is_bool(std::string name) const {
  for(size_t i=0;i<bset.size();i++)
    if(name==bset[i].name)
      return 1;
  return 0;
}

bool Settings::is_string(std::string name) const {
  for(size_t i=0;i<sset.size();i++)
    if(name==sset[i].name)
      return 1;
  return 0;
}


void Settings::parse(std::string filename) {
  // Input file
  std::ifstream in(filename.c_str());
  
  if(in.good()) {
    // OK, file was succesfully opened.
    
    while(in.good()) {
      // Read line and split it into words
      std::string line=readline(in);
      std::vector<std::string> words=splitline(line);

      if(words.size()) {
	// Parse keywords
	if(is_double(words[0]))
	  set_double(words[0],readdouble(words[1]));
	else if(is_int(words[0]))
	  set_int(words[0],readint(words[1]));
	else if(is_bool(words[0]))
	  set_bool(words[0],readint(words[1]));
	else if(is_string(words[0]))
	  set_string(words[0],words[1]);
	else {
	  ERROR_INFO();
	  std::ostringstream oss;
	  oss << "\nCannot recognize keyword "<<words[0]<<"!\n"; 
	  throw std::runtime_error(oss.str());
	}
      }
    }
  }
}

void Settings::print() const {
  printf("\nCurrent Settings used by ERKALE:\n");

  std::string bvals[]={"false","true"};

  for(size_t i=0;i<bset.size();i++)
    printf("%20s\t%20s\t%s\n",bset[i].name.c_str(),bvals[bset[i].val].c_str(),bset[i].comment.c_str());
  
  for(size_t i=0;i<iset.size();i++)
    printf("%20s\t%20i\t%s\n",iset[i].name.c_str(),iset[i].val,iset[i].comment.c_str());
  
  for(size_t i=0;i<dset.size();i++)
    printf("%20s\t%20.3e\t%s\n",dset[i].name.c_str(),dset[i].val,dset[i].comment.c_str());
  
  for(size_t i=0;i<sset.size();i++)
    printf("%20s\t%20s\t%s\n",sset[i].name.c_str(),sset[i].val.c_str(),sset[i].comment.c_str());

  printf("\n");
}

doubleset_t gend(std::string name, std::string comment, double val) {
  doubleset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  return ret;
}

boolset_t genb(std::string name, std::string comment, bool val) {
  boolset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  return ret;
}

intset_t geni(std::string name, std::string comment, int val) {
  intset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  return ret;
}

stringset_t gens(std::string name, std::string comment, std::string val) {
  stringset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  return ret;
}
