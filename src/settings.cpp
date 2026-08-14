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

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <cfloat>

template<typename T>
SettingTable<T>::SettingTable(const std::string & label) : typelabel(label) {
}

template<typename T>
size_t SettingTable<T>::find_ci(const std::string & name) const {
  for(size_t i=0;i<entries.size();i++)
    if(stricmp(name,entries[i].name)==0)
      return i+1;
  return 0;
}

template<typename T>
size_t SettingTable<T>::find_cs(const std::string & name) const {
  for(size_t i=0;i<entries.size();i++)
    if(name==entries[i].name)
      return i+1;
  return 0;
}

template<typename T>
void SettingTable<T>::add(const std::string & name, const std::string & comment, const T & val, bool negative) {
  // Check that setting does not already exist
  if(find_ci(name)) {
    std::ostringstream oss;
    oss << "Error: the " << typelabel << " setting " << name << " already exists!";
    throw std::runtime_error(oss.str());
  }
  entries.push_back({name,comment,val,negative});
}

template<typename T>
void SettingTable<T>::set(const std::string & name, const T & val) {
  size_t idx=find_ci(name);
  if(!idx) {
    std::ostringstream oss;
    oss << "\nThe " << typelabel << " setting "<<name<<" was not found!\n";
    throw std::runtime_error(oss.str());
  }
  Entry & e=entries[idx-1];
  // Only the numeric types carry a sign constraint.
  if constexpr (std::is_same_v<T,double> || std::is_same_v<T,int>) {
    if(val<T(0) && !e.negative) {
      std::ostringstream oss;
      oss << "Error: setting " << name << " must have non-negative value.\n";
      throw std::runtime_error(oss.str());
    }
  }
  e.val=val;
}

template<typename T>
T SettingTable<T>::get(const std::string & name) const {
  // Historical behaviour: get matches case-sensitively.
  size_t idx=find_cs(name);
  if(!idx) {
    std::ostringstream oss;
    oss << "\nThe " << typelabel << " setting "<<name<<" was not found!\n";
    throw std::runtime_error(oss.str());
  }
  return entries[idx-1].val;
}

template<typename T>
size_t SettingTable<T>::is(const std::string & name) const {
  return find_ci(name);
}

template<typename T>
size_t SettingTable<T>::size() const {
  return entries.size();
}

template<typename T>
const std::string & SettingTable<T>::name(size_t i) const {
  return entries[i].name;
}

template<typename T>
void SettingTable<T>::print_entry(size_t i) const {
  const Entry & e=entries[i];
  if constexpr (std::is_same_v<T,std::string>) {
    printf("%5s%-15s\t%20s\t%s\n","",e.name.c_str(),e.val.c_str(),e.comment.c_str());
  } else if constexpr (std::is_same_v<T,double>) {
    printf("%5s%-15s\t%20.3e\t%s\n","",e.name.c_str(),e.val,e.comment.c_str());
  } else if constexpr (std::is_same_v<T,int>) {
    printf("%5s%-15s\t%20i\t%s\n","",e.name.c_str(),e.val,e.comment.c_str());
  } else if constexpr (std::is_same_v<T,bool>) {
    static const char * bvals[]={"false","true"};
    printf("%5s%-15s\t%20s\t%s\n","",e.name.c_str(),bvals[e.val],e.comment.c_str());
  }
  fflush(stdout);
}

// The four value types actually used by Settings.
template class SettingTable<double>;
template class SettingTable<bool>;
template class SettingTable<int>;
template class SettingTable<std::string>;

Settings::Settings()
  : dset("double type"), bset("boolean"), iset("integer"), sset("string") {
  // Set default Settings
}


Settings::~Settings() {
}

void Settings::add_jk_settings() {
  // Every setting consumed by JKBuilder::configure, so any tool that drives
  // a JKBuilder registers them with one call (add_scf_settings does this too).
  add_bool("Direct", "Calculate two-electron integrals (or density fitting) on-the-fly?", false);
  add_bool("DecFock", "Use decontracted basis to calculate Fock matrix (direct HF)", false);
  add_double("IntegralThresh", "Integral screening threshold", 1e-10);
  // Density-weighted Fock-contribution screening threshold. Bounds the
  // contribution of each ERI quartet to J/K by the integral times the
  // largest coupled density element. Set to 0 to disable.
  add_double("ScreeningThresh", "Density-weighted Fock-contribution screening threshold", 1e-10);
  // How to build the Coulomb and exchange matrices.
  add_string("JKMethod", "Coulomb/exchange build method: 4index (exact ERIs, in-core or Direct), RI (density fitting with a Gaussian auxiliary basis -- set FittingBasis), Cholesky (two-step pivoted Cholesky decomposition; Folkestad/Kjonstad/Koch JCP 150, 194112 (2019); exact at threshold) or CDFit (density fitting on a per-atom CD-derived auxiliary basis; Lehtola JCTC 17, 6886 (2021); exact only as the orbital basis becomes complete).", "Cholesky");
  add_bool("OccRIK", "Use the occ-RI-K algorithm (Manzer et al, JCP 143, 024113 (2015)) for density-fitted/Cholesky exchange? Gives the exact (RI) SCF energy and density but only approximate virtual orbital energies. Ignored for the four-index method.", false);
  add_double("CholeskyThr", "Cholesky decomposition threshold", 1e-7);
  add_double("CholeskyShThr", "Cholesky cache threshold", 0.01);
  add_int("CholeskyMode", "Save/load the DF/CD integral cache? 0 no, 1 save after fill, -1 load before fill (falls back to fill on mismatch). Useful for repeated runs that share orbital + auxiliary basis. Ignored when Direct=true.", 0, true);
  add_string("CholeskyFile", "Filename for the DF/CD integral cache (used when CholeskyMode != 0). Plain and range-separated entries coexist in the same file under distinct keys.", "cholesky.chk");
  // Which basis to use as density fitting basis
  add_string("FittingBasis", "Basis to use for density fitting / RI: a basis-set name, Auto (CD-derived auto-aux, Lehtola JCTC 17, 6886 (2021); uncontracted, lmax-pruned per FittingLmaxInc) or AutoABS (Eichkorn-style automatic aux, J-only)","def2-universal-jkfit");
  add_int("FittingLmaxInc", "Angular-momentum pruning increment for the CD-derived auto-aux: keep l <= max(2*l_occ, l_obs+l_occ+FittingLmaxInc) (Lehtola JCTC 19, 6242 (2023)); negative keeps all shells", 1, true);
  add_double("FittingThreshold", "Linear dependence threshold for Coulomb integrals in density fitting",1e-7);
  add_double("FittingCholeskyThreshold", "Linear dependence threshold for pivoted Cholesky of Coulomb integrals in density fitting",1e-8);
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
  // Confinement potential
  add_string("Confinement", "Confinement potential V(r) = V_i x_i^2", "0.0 0.0 0.0");

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
  add_double("LinearB", "Magnetic field along bond axis", 0.0, true);
  add_double("LinearE", "Electric field along bond (z) axis", 0.0, true);
  add_double("HarmonicConfinement", "Harmonic confinement strength k (potential 1/2 k r^2), 0 = off", 0.0, true);

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
  add_string("Guess","Used guess: SAD (default), SADNO, core, GWH, SAP, or SAPfit","SAD");
  add_double("Kgwh","Scaling constant for GWH",1.75);

  // Verbose run?
  add_bool("Verbose", "Verbose calculation?", true);

  // Coulomb/exchange build + integral-screening settings (everything
  // JKBuilder::configure consumes).
  add_jk_settings();

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

  // SAP basis
  add_string("SAPBasis", "Tabulated atomic effective potential \"basis set\"","helfem_large.gbs");
  // Use Lobatto quadrature?
  add_bool("DFTLobatto", "Use Lobatto quadrature instead of Lebedev quadrature?", false);

  // Grid to use
  add_string("DFTGrid", "DFT integration grid to use: nrad lmax or Auto for adaptive", "75 -302");
  add_double("DFTQuadThresh", "Threshold for pruning points with small quadrature weight", DBL_EPSILON);
  add_string("SAPGrid", "SAP integration grid to use: nrad lmax or leave empty", "");
  // Initial and final tolerances of DFT grid
  add_double("DFTInitialTol", "Tolerance of initial DFT grid", 1e-4);
  add_double("DFTFinalTol", "Tolerance of final DFT grid", 1e-5);
  // Relative factor for initialization
  add_double("DFTDelta", "Switch to final DFT grid has converged within factor X", 1e2);
  // Override parameters of XC functional
  add_string("DFTXpars", "Override parameters of exchange functional (expert)", "");
  add_string("DFTCpars", "Override parameters of correlation functional (expert)", "");
  // Basis set value threshold
  add_double("DFTBasisThr", "Threshold for screening basis functions on grid", 1e-10);
  // Density threshold
  add_double("DFTDensityThr", "Threshold for screening density on grid", 1e-10);

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
  add_bool("PZrand", "Apply random rotation in PZ initialization? (Should be true)",true);
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

void Settings::add_double(const std::string & name, const std::string & comment, double val, bool negative) {
  dset.add(name,comment,val,negative);
}

void Settings::add_bool(const std::string & name, const std::string & comment, bool val) {
  bset.add(name,comment,val);
}

void Settings::add_int(const std::string & name, const std::string & comment, int val, bool negative) {
  iset.add(name,comment,val,negative);
}

void Settings::add_string(const std::string & name, const std::string & comment, const std::string & val) {
  sset.add(name,comment,val);
}

void Settings::set_double(const std::string & name, double val) {
  dset.set(name,val);
}

void Settings::set_bool(const std::string & name, bool val) {
  bset.set(name,val);
}

void Settings::set_int(const std::string & name, int val) {
  iset.set(name,val);
}

void Settings::set_string(const std::string & name, const std::string & val) {
  sset.set(name,val);
}

double Settings::get_double(const std::string & name) const {
  return dset.get(name);
}

bool Settings::get_bool(const std::string & name) const {
  return bset.get(name);
}

int Settings::get_int(const std::string & name) const {
  return iset.get(name);
}

std::string Settings::get_string(const std::string & name) const {
  return sset.get(name);
}

arma::vec Settings::get_vec(const std::string & name) const {
  std::vector<std::string> v(splitline(get_string(name)));

  arma::vec x(v.size());
  for(size_t i=0;i<x.n_elem;i++)
    x(i)=readdouble(v[i]);

  return x;
}

arma::ivec Settings::get_ivec(const std::string & name) const {
  std::vector<std::string> v(splitline(get_string(name)));

  arma::ivec x(v.size());
  for(size_t i=0;i<x.n_elem;i++)
    x(i)=readint(v[i]);

  return x;
}

arma::uvec Settings::get_uvec(const std::string & name) const {
  return arma::conv_to<arma::uvec>::from(get_ivec(name));
}

size_t Settings::is_double(const std::string & name) const {
  return dset.is(name);
}

size_t Settings::is_int(const std::string & name) const {
  return iset.is(name);
}

size_t Settings::is_bool(const std::string & name) const {
  return bset.is(name);
}

size_t Settings::is_string(const std::string & name) const {
  return sset.is(name);
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
	// Normalise the method name (Hartree-Fock or DFT).
	if(stricmp(words[1],"Hartree-Fock")==0 || stricmp(words[1],"HF")==0)
	  set_string("Method","HF");
	else if(stricmp(words[1],"ROHF")==0)
	  set_string("Method","ROHF");
	else
	  set_string("Method",words[1]);

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
	} else if(stricmp(words[0],"DensityFitting")==0 || stricmp(words[0],"Cholesky")==0 || stricmp(words[0],"CholeskyAlgorithm")==0) {
	  // The method selectors were unified into JKMethod.
	  std::ostringstream oss;
	  oss << "\nThe keyword " << words[0] << " has been removed. Choose the Coulomb/exchange\n"
	      << "build method with JKMethod instead: 4index, RI, Cholesky or CDFit.\n"
	      << "  DensityFitting true        -> JKMethod RI\n"
	      << "  Cholesky false             -> JKMethod 4index\n"
	      << "  CholeskyAlgorithm CDFit    -> JKMethod CDFit\n"
	      << "  (the default is JKMethod Cholesky)\n";
	  throw std::runtime_error(oss.str());
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
  fflush(stdout);

  // First, collect the keywords and sort them alphabetically.
  std::vector<std::string> kw;
  for(size_t i=0;i<bset.size();i++)
    kw.push_back(bset.name(i));
  for(size_t i=0;i<iset.size();i++)
    kw.push_back(iset.name(i));
  for(size_t i=0;i<dset.size();i++)
    kw.push_back(dset.name(i));
  for(size_t i=0;i<sset.size();i++)
    kw.push_back(sset.name(i));
  std::stable_sort(kw.begin(),kw.end());

  // and then print the list in alphabetic order. Each keyword lives in
  // exactly one type table; the per-type probes below are defensive.
  for(size_t i=0;i<kw.size();i++) {
    size_t is=sset.is(kw[i]);
    size_t id=dset.is(kw[i]);
    size_t ii=iset.is(kw[i]);
    size_t ib=bset.is(kw[i]);

    if(is>0) sset.print_entry(is-1);
    if(id>0) dset.print_entry(id-1);
    if(ii>0) iset.print_entry(ii-1);
    if(ib>0) bset.print_entry(ib-1);
  }
  printf("\n");
  fflush(stdout);
}
