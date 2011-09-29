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


Settings::Settings() {
  // Set default Settings

  // Dummy functional: this will be set to HF or a X-C combination
  sset.push_back(gens("Method", "Method used in calculation (HF or a DFT functional)", "Dummy"));

  // Default basis set
  sset.push_back(gens("Basis", "Basis set used in calculation", "aug-cc-pVTZ"));

  // Input system
  sset.push_back(gens("System", "System as an xyz file", "atoms.xyz"));

  // Log file
  sset.push_back(gens("Logfile", "File to print out full information, stdout for screen", "erkale.log"));

  // Use spherical harmonics.
  bset.push_back(genb("UseLM", "Use a spherical harmonics basis set by default?", 1));
  
  // Decontract basis set?
  bset.push_back(genb("Decontract","Decontract basis set?",0));

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
  // Occupancies
  sset.push_back(gens("Occupancies", "Orbital occupancies", ""));

  // Verbose run?
  bset.push_back(genb("Verbose", "Verbose calculation?", 1));

  // Direct calculation?
  bset.push_back(genb("Direct", "Calculate two-electron integrals (or density fitting) on-the-fly?", 0));

  // Default orthogonalization method
  sset.push_back(gens("BasisOrth", "Method of orthonormalization of basis set", "Can"));

  // Default cutoff for orthogonalization
  dset.push_back(gend("BasisLinTol", "Cutoff for linearly dependent basis functions", 1e-5));

  // Convergence criteria
  dset.push_back(gend("DeltaPrms", "Maximum allowed RMS difference of density matrix", 1e-8));
  dset.push_back(gend("DeltaPmax", "Maximum allowed maximum difference of density matrix", 1e-6));
  dset.push_back(gend("DeltaEmax", "Maximum allowed change of energy", 1e-6));
  // Relative factor for initialization
  dset.push_back(gend("DeltaInit", "When to switch to final calculation (mostly DFT), relative to Delta parameters", 100.0));

  // Maximum iterations
  iset.push_back(geni("MaxIter", "Maximum number of iterations in SCF cycle", 100));

  // Calculate EMD properties?
  bset.push_back(genb("DoEMD", "Perform calculation of isotropic EMD (moments of EMD, Compton profile)", 0));
  // Calculate EMD on a cube?
  sset.push_back(gens("EMDCube", "Calculate EMD on a cube? e.g. -10:.3:10 -5:.2:4 -2:.1:3", ""));

  // How to initialize calculation
  sset.push_back(gens("InitMethod","Method of initializing calculation","none"));

#ifdef DFT_ENABLED
  // No DFT settings by default.
  dft=0;
#endif
}


Settings::~Settings() {
}

#ifdef DFT_ENABLED
void Settings::add_dft_settings() {
  // DFT settings
  dft=1;

  // Store full DFT grid in memory?
  bset.push_back(genb("DFTDirect", "Save memory by not storing values of basis functions in memory", 0));
  // Store full DFT grid in memory?
  bset.push_back(genb("DFTLobatto", "Use Lobatto quadrature instead of Lebedev quadrature?", 0));
  
  // Initial and final tolerances of DFT grid
  dset.push_back(gend("DFTInitialTol", "Tolerance of initial DFT grid", 1e-3));
  dset.push_back(gend("DFTFinalTol", "Tolerance of final DFT grid", 5e-5));
  
  // Use density fitting if possible?
  bset.push_back(genb("DFTFitting", "Use density fitting if possible? (Pure DFT functionals)", 1));
}

void Settings::remove_dft_settings() {
  dft=0;

  // Remove all settings that contain DFT in the keyword
  for(size_t i=dset.size()-1;i<dset.size();i--)
    if(dset[i].name.find("DFT")!=std::string::npos)
      dset.erase(dset.begin()+i);

  for(size_t i=bset.size()-1;i<bset.size();i--)
    if(bset[i].name.find("DFT")!=std::string::npos)
      bset.erase(bset.begin()+i);

  for(size_t i=iset.size()-1;i<iset.size();i--)
    if(iset[i].name.find("DFT")!=std::string::npos)
      iset.erase(iset.begin()+i);

  for(size_t i=sset.size()-1;i<sset.size();i--)
    if(sset[i].name.find("DFT")!=std::string::npos)
      sset.erase(sset.begin()+i);
}
#endif
  
void Settings::add_double(std::string name, std::string comment, double val) {
  dset.push_back(gend(name,comment,val));
}

void Settings::add_bool(std::string name, std::string comment, bool val) {
  bset.push_back(genb(name,comment,val));
}

void Settings::add_int(std::string name, std::string comment, int val) {
  iset.push_back(geni(name,comment,val));
}

void Settings::add_string(std::string name, std::string comment, std::string val) {
  sset.push_back(gens(name,comment,val));
}

void Settings::set_double(std::string name, double val) {
  if(val<0.0) {
    std::ostringstream oss;
    oss << "Error: settings must have positive value.\n";
    throw std::runtime_error(oss.str());
  }

  // Find setting in table
  for(size_t i=0;i<dset.size();i++)
    if(stricmp(name,dset[i].name)==0) {
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
    if(stricmp(name,bset[i].name)==0) {
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
    if(stricmp(name,iset[i].name)==0) {
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
    if(stricmp(name,sset[i].name)==0) {
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

#ifdef DFT_ENABLED
bool Settings::dft_enabled() const {
  return dft;
}
#endif

bool Settings::is_double(std::string name) const {
  for(size_t i=0;i<dset.size();i++)
    if(stricmp(name,dset[i].name)==0)
      return 1;

  return 0;
}

bool Settings::is_int(std::string name) const {
  for(size_t i=0;i<iset.size();i++)
    if(stricmp(name,iset[i].name)==0)
      return 1;

  return 0;
}

bool Settings::is_bool(std::string name) const {
  for(size_t i=0;i<bset.size();i++)
    if(stricmp(name,bset[i].name)==0)
      return 1;
  return 0;
}

bool Settings::is_string(std::string name) const {
  for(size_t i=0;i<sset.size();i++)
    if(stricmp(name,sset[i].name)==0)
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

	if(words.size()==1) {
	  ERROR_INFO();
	  std::ostringstream oss;
	  oss << "\nParse error: "<<words[0]<<" has no value!\n"; 
	  throw std::runtime_error(oss.str());
	}
	
	if(stricmp(words[0],"Method")==0) {
	  // Hartree-Fock or DFT?
	  if(stricmp(words[1],"Hartree-Fock")==0 || stricmp(words[1],"HF")==0)
	    set_string("Method","HF");
	  else if(stricmp(words[1],"ROHF")==0)
	    set_string("Method","ROHF");
#ifdef DFT_ENABLED
	  else {
	    // Add dft related settings
	    add_dft_settings();
	    set_string("Method",words[1]);
	  }
#endif

	} else {
	  if(is_double(words[0])) {
	    set_double(words[0],readdouble(words[1]));
	  } else if(is_int(words[0])) {
	    set_int(words[0],readint(words[1]));
	  } else if(is_bool(words[0])) {
	    set_bool(words[0],readint(words[1]));
	  } else if(is_string(words[0])) {
	    // Concatenate value
	    std::string val=words[1];
	    for(size_t i=2;i<words.size();i++)
	      val+=" "+words[i];
	    // Store value
	    set_string(words[0],val);
	  } else {
	    ERROR_INFO();
	    std::ostringstream oss;
	    oss << "\nCannot recognize keyword "<<words[0]<<"!\n"; 
	    throw std::runtime_error(oss.str());
	  }
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
