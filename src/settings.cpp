/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#include "dftfuncs.h"
#include "settings.h"
#include "stringutil.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>


Settings::Settings() {
  // Set default Settings
}


Settings::~Settings() {
}

void Settings::add_scf_settings() {
  // Dummy functional: this will be set to HF or a X-C combination
  add_string("Method", "Method used in calculation (HF or a DFT functional)", "Dummy");
  add_string("AtomGuess", "Method used for atomic guess (Auto for same as method)", "Auto");

  // Default basis set
  add_string("Basis", "Basis set used in calculation", "aug-cc-pVTZ");
  // Rotate basis set to drop out redundant functions?
  add_bool("BasisRotate", "Rotate basis set to remove redundant functions?", false);
  // Cutoff for redundant functions
  add_double("BasisCutoff", "Cutoff for dropping out small primitives from contraction", 1e-8);

  // Input system
  add_string("System", "System as an xyz file", "atoms.xyz");
  add_bool("InputBohr", "Use atomic units as input units instead of angstrom?", false);

  // Electric field
  add_string("EField", "Electric field", "0.0 0.0 0.0");

  // Log file
  add_string("Logfile", "File to print out full information, stdout for screen", "erkale.log");

  // Use spherical harmonics.
  add_bool("UseLM", "Use a spherical harmonics basis set by default?", true);
  // Optimized harmonics?
  add_bool("OptLM", "If spherical harmonics used, use cartesian s and p functions?", true);

  // Specialized dimer calculation?
  add_bool("LinearSymmetry", "Do special calculation on linear molecule along z axis", false);
  add_bool("LinearFreeze", "If using linear symmetry, freeze symmetry to input guess", false);
  add_int("LinearOccupations", "Read in occupations for linear molecule calculations?", 0, true);
  add_string("LinearOccupationFile", "File to read linear occupations from", "linoccs.dat");

  // Decontract basis set?
  add_string("Decontract","Indices of atoms to decontract basis set for","");

  // Use DIIS.
  add_bool("UseDIIS", "Use Pulay's Direct Inversion in the Iterative Subspace?", true);
  // Number of DIIS matrices to use?
  add_int("DIISOrder", "How many DIIS iterations to keep in memory?", 10);
  // DIIS threshold
  add_double("DIISEps", "Start mixing in DIIS when error is", 0.1);
  // DIIS threshold
  add_double("DIISThr", "DIIS error threshold for DIIS updates", 0.01);
  // DIIS threshold
  add_bool("DIISComb", "Combine alpha and beta errors in unrestricted calcs?", false);
  // Use ADIIS?
  add_bool("UseADIIS", "Use ADIIS for Fock matrix interpolation?", true);

  // Use Broyden mixing?
  add_bool("UseBroyden", "Use Broyden mixing of Fock matrices?", false);
  // Use Trust-Region Roothaan-Hall?
  add_bool("UseTRRH", "Use Trust-Region Roothaan-Hall?", false);
  // TRRH minimal overlap
  add_double("TRRHminS", "Trust-Region Roothaan-Hall minimal occupied orbital overlap", 0.975);

  // Total charge of system
  add_int("Charge", "Total charge of system", 0, true);
  // Multiplicity
  add_int("Multiplicity", "Spin multiplicity", 1);
  // Occupancies
  add_string("Occupancies", "Orbital occupancies", "");

  // Use core guess? Default is atomic.
  add_string("Guess","Used guess: SAD (default), SADNO, core, or GWH","SAD");
  add_double("Kgwh","Scaling constant for GWH",1.75);

  // Verbose run?
  add_bool("Verbose", "Verbose calculation?", true);

  // Direct calculation?
  add_bool("Direct", "Calculate two-electron integrals (or density fitting) on-the-fly?", false);
  // Compute Fock matrix in decontracted basis
  add_bool("DecFock", "Use decontracted basis to calculate Fock matrix (direct HF)", false);
  // Strict integrals?
  add_bool("StrictIntegrals", "Use strict integrals?", false);
  // Integral threshold
  add_double("IntegralThresh", "Integral screening threshold", 1e-10);

  // Default orthogonalization method
  add_string("BasisOrth", "Method of orthonormalization of basis set", "Auto");
  // Linear dependence threshold
  add_double("LinDepThresh", "Basis set linear dependency threshold", 1e-5);
  // Cholesky orthogonalization threshold
  add_double("CholDepThresh", "Partial Cholesky decomposition threshold", 1e-7);

  // Convergence criterion
  add_double("ConvThr", "Orbital gradient convergence threshold", 1e-6);

  // Maximum iterations
  add_int("MaxIter", "Maximum number of iterations in SCF cycle", 100);
  // Level shift
  add_double("Shift", "Level shift to use in Hartree", 0.0);

  // Use density fitting if possible?
  add_bool("DensityFitting", "Use density fitting / RI?", false);
  // Use Cholesky?
  add_bool("Cholesky", "Use Cholesky decomposition?", true);
  add_double("CholeskyThr", "Cholesky decomposition threshold", 1e-7);
  add_double("CholeskyShThr", "Cholesky cache threshold", 0.01);
  add_double("CholeskyNAFThr", "Cholesky natural auxiliary function threshold", 0.0);
  add_int("CholeskyMode", "Save/load integrals? 0 no, 1 save, -1 load", 0, true);
  // Which basis to use as density fitting basis
  add_string("FittingBasis", "Basis to use for density fitting / RI (Auto for automatic)","Auto");
  // How much memory to allow for density fitting
  add_int("FittingMemory", "Amount of memory in MB to use for exchange fitting",1000);
  // Threshold for screening eigenvectors
  add_double("FittingThreshold", "Linear dependence threshold for Coulomb integrals in density fitting",1e-8);

  // SAP basis
  add_string("SAPBasis", "Tabulated atomic effective potential \"basis set\"","sap_potential.gbs");
  // Use Lobatto quadrature?
  add_bool("DFTLobatto", "Use Lobatto quadrature instead of Lebedev quadrature?", false);

  // Grid to use
  add_string("DFTGrid", "DFT integration grid to use: nrad lmax or Auto for adaptive", "50 -194");
  add_string("SAPGrid", "SAP integration grid to use: nrad lmax or leave empty", "");
  // Initial and final tolerances of DFT grid
  add_double("DFTInitialTol", "Tolerance of initial DFT grid", 1e-4);
  add_double("DFTFinalTol", "Tolerance of final DFT grid", 1e-5);
  // Relative factor for initialization
  add_double("DFTDelta", "Switch to final DFT grid has converged within factor X", 1e2);
  // Override parameters of XC functional
  add_string("DFTXpars", "Override parameters of exchange functional (expert)", "");
  add_string("DFTCpars", "Override parameters of correlation functional (expert)", "");

  // VV10?
  add_string("VV10","Use VV10 non-local correlation?","Auto");
  add_string("NLGrid", "Integration grid to use for nonlocal correlation: nrad lmax", "50 -194");
  add_string("VV10Pars","VV10 parameters: b C","");

  // Use Perdew-Zunger self-interaction correction?
  add_double("PZw", "Weight for Perdew-Zunger self-interaction correction", 1.0);
  add_string("PZscale", "Scaling for PZ: Constant, Density or Kinetic", "Constant");
  add_double("PZscaleExp", "Exponent in the dynamic scaling equation", 1.0);
  // Perturbative SIC?
  add_bool("PZ", "Perform Perdew-Zunger self-interaction correction?",false);
  add_int("PZprec", "Precondition OV block? 0: no, 1: unified, 2: orbital",1);
  add_bool("PZoo", "Optimize OO block?",true);
  add_bool("PZov", "Optimize OV block?",true);
  add_double("PZIthr", "Threshold for initialization convergence (not too small!)",1e-2);
  add_double("PZOOthr", "Gradient threshold for OO optimization",1e-4);
  add_double("PZOVthr", "Gradient threshold for OV optimization",1e-5);
  add_double("PZNRthr", "Threshold for use of NR method in OO optimization",0.0);
  add_double("PZEthr", "Threshold for energy convergence",1e-10);
  // Initialize PZ-SIC with localized orbitals?
  add_string("PZloc", "Initial localization before SIC calculation?", "Auto");
  add_string("PZlocmet", "Initial localization method (recommend FB or IAO)", "FB");
  // Run stability analysis for PZ-SIC?
  add_int("PZstab", "Stability analysis for PZ-SIC? 1 or -1 for OO, 2 or -2 for OO+OV", 0, true);
  add_double("PZstabThr", "Instability threshold (interpreted as -thr)", 1e-3);
  add_string("PZimag", "Imaginary degrees of freedom in PZ?", "Auto");
  // Mode to use PZ-SIC
  add_string("PZmode", "Apply PZ to the operators (in addition to J): X C D", "XC");
  // PZ-SIC maximum number of iterations in self-consistency cycle
  add_int("PZiter", "Max number of iterations in self-consistency iteration", 20);
  // PZ-SIC seed number
  add_int("PZseed", "Seed number for randomized matrices?", 0);
}

void Settings::add_double(std::string name, std::string comment, double val, bool negative) {
  // Check that setting does not exist
  if(is_double(name)) {
    std::ostringstream oss;
    oss << "Error in add_double: setting " << name << " already exists!";
    throw std::runtime_error(oss.str());
  }

  dset.push_back(gend(name,comment,val,negative));
}

void Settings::add_bool(std::string name, std::string comment, bool val) {
  // Check that setting does not exist
  if(is_bool(name)) {
    std::ostringstream oss;
    oss << "Error in add_bool: setting " << name << " already exists!";
    throw std::runtime_error(oss.str());
  }

  bset.push_back(genb(name,comment,val));
}

void Settings::add_int(std::string name, std::string comment, int val, bool negative) {
  // Check that setting does not exist
  if(is_int(name)) {
    std::ostringstream oss;
    oss << "Error in add_int: setting " << name << " already exists!";
    throw std::runtime_error(oss.str());
  }

  iset.push_back(geni(name,comment,val,negative));
}

void Settings::add_string(std::string name, std::string comment, std::string val) {
  // Check that setting does not exist
  if(is_string(name)) {
    std::ostringstream oss;
    oss << "Error in add_string: setting " << name << " already exists!";
    throw std::runtime_error(oss.str());
  }
  sset.push_back(gens(name,comment,val));
}

void Settings::set_double(std::string name, double val) {
  // Find setting in table
  for(size_t i=0;i<dset.size();i++)
    if(stricmp(name,dset[i].name)==0) {
      if(val<0.0 && !dset[i].negative) {
	std::ostringstream oss;
	oss << "Error: setting " << name << " must have non-negative value.\n";
	throw std::runtime_error(oss.str());
      }
      dset[i].val=val;
      return;
    }

  std::ostringstream oss;
  oss << "\nThe double type setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());
}

void Settings::set_bool(std::string name, bool val) {
  // Find setting in table
  for(size_t i=0;i<bset.size();i++)
    if(stricmp(name,bset[i].name)==0) {
      bset[i].val=val;
      return;
    }

  std::ostringstream oss;
  oss << "\nThe boolean setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());
}

void Settings::set_int(std::string name, int val) {
  // Find setting in table
  for(size_t i=0;i<iset.size();i++)
    if(stricmp(name,iset[i].name)==0) {
      if(val<0 && !iset[i].negative) {
	std::ostringstream oss;
	oss << "Error: setting " << name << " must have non-negative value.\n";
	throw std::runtime_error(oss.str());
      }
      iset[i].val=val;
      return;
    }

  std::ostringstream oss;
  oss << "\nThe integer setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());
}


void Settings::set_string(std::string name, std::string val) {
  // Find setting in table
  for(size_t i=0;i<sset.size();i++)
    if(stricmp(name,sset[i].name)==0) {
      sset[i].val=val;
      return;
    }

  std::ostringstream oss;
  oss << "\nThe string setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());
}



double Settings::get_double(std::string name) const {
  // Find setting in table
  for(size_t i=0;i<dset.size();i++)
    if(name==dset[i].name) {
      return dset[i].val;
    }

  std::ostringstream oss;
  oss << "\nThe double type setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());

  return 0.0;
}

bool Settings::get_bool(std::string name) const {
  // Find setting in table
  for(size_t i=0;i<bset.size();i++)
    if(name==bset[i].name) {
      return bset[i].val;
    }

  std::ostringstream oss;
  oss << "\nThe boolean setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());

  return 0;
}

int Settings::get_int(std::string name) const {
  // Find setting in table
  for(size_t i=0;i<iset.size();i++)
    if(name==iset[i].name) {
      return iset[i].val;
    }

  std::ostringstream oss;
  oss << "\nThe integer setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());

  return 0;
}

std::string Settings::get_string(std::string name) const {
 // Find setting in table
  for(size_t i=0;i<sset.size();i++)
    if(name==sset[i].name) {
      return sset[i].val;
    }

  std::ostringstream oss;
  oss << "\nThe string setting "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());

  return "";
}

arma::vec Settings::get_vec(std::string name) const {
  std::vector<std::string> v(splitline(get_string(name)));

  arma::vec x(v.size());
  for(size_t i=0;i<x.n_elem;i++)
    x(i)=readdouble(v[i]);

  return x;
}

arma::ivec Settings::get_ivec(std::string name) const {
  std::vector<std::string> v(splitline(get_string(name)));

  arma::ivec x(v.size());
  for(size_t i=0;i<x.n_elem;i++)
    x(i)=readint(v[i]);

  return x;
}

arma::uvec Settings::get_uvec(std::string name) const {
  return arma::conv_to<arma::uvec>::from(get_ivec(name));
}

size_t Settings::is_double(std::string name) const {
  for(size_t i=0;i<dset.size();i++)
    if(stricmp(name,dset[i].name)==0)
      return i+1;

  return 0;
}

size_t Settings::is_int(std::string name) const {
  for(size_t i=0;i<iset.size();i++)
    if(stricmp(name,iset[i].name)==0)
      return i+1;

  return 0;
}

size_t Settings::is_bool(std::string name) const {
  for(size_t i=0;i<bset.size();i++)
    if(stricmp(name,bset[i].name)==0)
      return i+1;

  return 0;
}

size_t Settings::is_string(std::string name) const {
  for(size_t i=0;i<sset.size();i++)
    if(stricmp(name,sset[i].name)==0)
      return i+1;

  return 0;
}


void Settings::parse(std::string filename, bool scf) {
  // Input file
  std::ifstream in(filename.c_str());

  if(!in.good()) {
    std::ostringstream oss;
    oss << "Input file "<<filename<<" not found!";
    throw std::runtime_error(oss.str());
  }

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

      if(scf && stricmp(words[0],"Method")==0) {
	// Hartree-Fock or DFT?
	if(stricmp(words[1],"Hartree-Fock")==0 || stricmp(words[1],"HF")==0) {
	  set_string("Method","HF");
	  // Turn of density fitting by default
	  set_bool("DensityFitting",false);
	} else if(stricmp(words[1],"ROHF")==0) {
	  set_string("Method","ROHF");
	  set_bool("DensityFitting",false);
	} else {
	  set_string("Method",words[1]);

	  // Hybrid functional? Do we turn off density fitting by default?
	  int xfunc, cfunc;
	  parse_xc_func(xfunc,cfunc,words[1]);
	  if(exact_exchange(xfunc)!=0.0 || is_range_separated(xfunc))
	    set_bool("DensityFitting",false);
	}

      } else {
	if(is_double(words[0])) {
	  set_double(words[0],readdouble(words[1]));
	} else if(is_int(words[0])) {
	  set_int(words[0],readint(words[1]));
	} else if(is_bool(words[0])) {
	  // Was the value given as a number or as a string?
	  if(isalpha(words[1][0])) {

	    // As a string - parse it
	    bool value;
	    if(stricmp(words[1],"true")==0)
	      value=true;
	    else if(stricmp(words[1],"false")==0)
	      value=false;
	    else if(stricmp(words[1],"on")==0)
	      value=true;
	    else if(stricmp(words[1],"off")==0)
	      value=false;
	    else if(stricmp(words[1],"yes")==0)
	      value=true;
	    else if(stricmp(words[1],"no")==0)
	      value=false;
	    else {
	      value=false;

	      std::ostringstream oss;
	      oss << "Could not parse the truth value " << words[1] << " for setting "<<words[0]<<"!\n";
	      throw std::runtime_error(oss.str());
	    }

	    set_bool(words[0],value);
	  } else
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
	  print();
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

  const std::string bvals[]={"false","true"};

  // First, sort the keywords alphabetically.
  std::vector<std::string> kw;
  for(size_t i=0;i<bset.size();i++)
    kw.push_back(bset[i].name);
  for(size_t i=0;i<iset.size();i++)
    kw.push_back(iset[i].name);
  for(size_t i=0;i<dset.size();i++)
    kw.push_back(dset[i].name);
  for(size_t i=0;i<sset.size();i++)
    kw.push_back(sset[i].name);
  std::stable_sort(kw.begin(),kw.end());

  // and then print the list in alphabetic order.
  for(size_t i=0;i<kw.size();i++) {
    size_t is=is_string(kw[i]);
    size_t id=is_double(kw[i]);
    size_t ii=is_int(kw[i]);
    size_t ib=is_bool(kw[i]);

    if(is>0)
      // Is string!
      printf("%5s%-15s\t%20s\t%s\n","",sset[is-1].name.c_str(),sset[is-1].val.c_str(),sset[is-1].comment.c_str());
    if(id>0)
      // Is double!
      printf("%5s%-15s\t%20.3e\t%s\n","",dset[id-1].name.c_str(),dset[id-1].val,dset[id-1].comment.c_str());
    if(ii>0)
      // Is integer!
      printf("%5s%-15s\t%20i\t%s\n","",iset[ii-1].name.c_str(),iset[ii-1].val,iset[ii-1].comment.c_str());
    if(ib>0)
      // Is boolean!
      printf("%5s%-15s\t%20s\t%s\n","",bset[ib-1].name.c_str(),bvals[bset[ib-1].val].c_str(),bset[ib-1].comment.c_str());
  }
  printf("\n");
}

doubleset_t gend(std::string name, std::string comment, double val, bool negative) {
  doubleset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  ret.negative=negative;
  return ret;
}

boolset_t genb(std::string name, std::string comment, bool val) {
  boolset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  return ret;
}

intset_t geni(std::string name, std::string comment, int val, bool negative) {
  intset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  ret.negative=negative;
  return ret;
}

stringset_t gens(std::string name, std::string comment, std::string val) {
  stringset_t ret;
  ret.name=name;
  ret.comment=comment;
  ret.val=val;
  return ret;
}
