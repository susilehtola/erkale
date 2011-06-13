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



#include "basislibrary.h"
#include "elements.h"
#include "stringutil.h"

#include <algorithm>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
// For exceptions
#include <sstream>
#include <stdexcept>


int find_am(char am) {
  for(int i=0;i<=maxam;i++)
    if(shell_types[i]==am)
      return i;

  ERROR_INFO();
  std::ostringstream oss;
  oss << "Angular momentum "<<am<<" not found!\n";
  throw std::runtime_error(oss.str());

  return -1;
}

std::string find_basis(const std::string & basisname) {
  // Directories where the basis set file might be found
  std::vector<std::string> dirs;

  // First, check if there is an environmental variable called
  // ERKALE_LIBRARY
  char * libloc=getenv("ERKALE_LIBRARY");
  if(libloc!=NULL) {
    // Variable exists! Add location to array
    dirs.push_back(libloc+std::string("/"));
  }

  // Next, try local directory.
  dirs.push_back("");
  // Finally, try system wide directory.
  dirs.push_back(ERKALE_SYSTEM_LIBRARY + std::string("/"));

  // Trial names
  std::vector<std::string> trialnames;
  // Try without extension
  trialnames.push_back(basisname);
  // Try with extension
  trialnames.push_back(basisname+".gbs");

  // Loop over directories.
  for(size_t id=0;id<dirs.size();id++) {
    // Loop over trial names
    for(size_t it=0;it<trialnames.size();it++) {
      // Full file name is
      std::string fname=dirs[id]+trialnames[it];
      // Try to open file for reading
      //printf("Trying %s\n",fname.c_str());
      std::ifstream in(fname.c_str());
      if(in.is_open()) {
	// Found basis set!
	printf("Basis set ""%s"" found in file %s in %s.\n",basisname.c_str(),trialnames[it].c_str(),dirs[id].c_str());
	return fname;
      }
    }
  }

  // Error handling
  std::ostringstream oss;
  ERROR_INFO();
  oss << "Could not find basis set " << basisname << "!\n";
  throw std::runtime_error(oss.str());
}

FunctionShell::FunctionShell(int amval) {
  am=amval;
}

FunctionShell::~FunctionShell() {
}

void FunctionShell::add_exponent(double Cv, double zv) {
  C.push_back(Cv);
  z.push_back(zv);
}

void FunctionShell::sort() {
  // Sort exponents in decreasing order

  size_t i, j;

  for(i=1;i<z.size();i++) {
    double chelp=C[i];
    double zhelp=z[i];
    for(j=i;j>=2;j--) {
      if(z[j-1] >= zhelp)
	break;
      z[j]=z[j-1];
      C[j]=C[j-1];
    }
    z[j]=zhelp;
    C[j]=chelp;
  }
}

void FunctionShell::print() const {
  printf("\tam = %i, %i functions\n",am, (int) C.size());
  for(size_t i=0;i<C.size();i++)
    printf("\t\t%e\t%e\n",C[i],z[i]);
}

bool FunctionShell::operator<(const FunctionShell & rhs) const {
  // First, check if angular momentum is lower.
  if(am!=rhs.am)
    return am<rhs.am;
  else
    // Same angular momentum, sort by first exponent.
    return z[0]>rhs.z[0];
}

int FunctionShell::get_am() const {
  return am;
}

std::vector<double> FunctionShell::get_exps() const {
  return z;
}

std::vector<double> FunctionShell::get_contr() const {
  return C;
}

ElementBasisSet::ElementBasisSet() {
}

ElementBasisSet::ElementBasisSet(std::string sym, size_t num) {
  symbol=sym;
  number=num;
}

ElementBasisSet::~ElementBasisSet() {
}

void ElementBasisSet::add_function(FunctionShell f) {
  bf.push_back(f);
}

void ElementBasisSet::sort() {
  // First, sort out the exponents on the shells
  for(size_t i=0;i<bf.size();i++)
    bf[i].sort();
  // Then, sort the shells
  stable_sort(bf.begin(),bf.end());
}

void ElementBasisSet::print() const {
  printf("%s:\n",symbol.c_str());
  for(size_t i=0;i<bf.size();i++) {
    bf[i].print();
  }
  printf("\n\n");
}

std::string ElementBasisSet::get_symbol() const {
  return symbol;
}

size_t ElementBasisSet::get_number() const {
  return number;
}

bool ElementBasisSet::operator<(const ElementBasisSet &rhs) const {
  return get_Z(symbol)<get_Z(rhs.symbol);
}

size_t ElementBasisSet::get_Nshells() const {
  return bf.size();
}

std::vector<FunctionShell> ElementBasisSet::get_shells() const {
  return bf;
}

void ElementBasisSet::get_primitives(std::vector<double> & exps, arma::mat & coeffs, int am) const {
  // Count number of exponents and shells that have angular momentum am
  int nsh=0;
  // Clear current exponents
  exps.clear();
  
  for(size_t ish=0;ish<bf.size();ish++)
    if(bf[ish].get_am()==am) {
      // Increment number of shells
      nsh++;

      // Get exponents on shell
      std::vector<double> shexp=bf[ish].get_exps();

      // Loop over exponents
      for(size_t iexp=0;iexp<shexp.size();iexp++) {
	// First, check if exponent is already on list
	bool found=0;
	for(size_t i=0;i<exps.size();i++)
	  if(exps[i]==shexp[iexp]) {
	    found=1;
	    break;
	  }
	
	// If exponent was not found, add it to the list.
	if(!found)
	  exps.push_back(shexp[iexp]);
      }
    }

  // Allocate returned contractions
  coeffs=arma::mat(exps.size(),nsh);

  // Collect contraction coefficients. Loop over exponents
  for(size_t iexp=0;iexp<exps.size();iexp++) {
    int iish=0;
    // Loop over shells
    for(size_t ish=0;ish<bf.size();ish++)
      if(bf[ish].get_am()==am) {

	// Get exponents and contraction on shell
	std::vector<double> shexp=bf[ish].get_exps();
	std::vector<double> shc=bf[ish].get_contr();

	// Find current exponent
	bool found=0;
	for(size_t i=0;i<shexp.size();i++)
	  if(shexp[i]==exps[iexp]) {
	    // Found exponent!
	    found=1;
	    // Store contraction coefficient.
	    coeffs(iexp,iish)=shc[i];
	    // Exit for loop
	    break;
	  }

	if(!found)
	  // Exponent not used on this shell.
	  coeffs(iexp,iish)=0.0;

	// Increment shell index
	iish++;
      }
  }
}

int ElementBasisSet::get_max_am() const {
  int maxam=0;
  for(size_t i=0;i<bf.size();i++)
    if(bf[i].get_am()>maxam)
      maxam=bf[i].get_am();
  return maxam;
}

int ElementBasisSet::get_am(size_t ind) const {
  return bf[ind].get_am();
}

BasisSetLibrary::BasisSetLibrary() {
}

BasisSetLibrary::~BasisSetLibrary() {
}

void BasisSetLibrary::load_gaussian94(const char * filename) {
  load_gaussian94(std::string(filename));
}

void BasisSetLibrary::load_gaussian94(std::string basis) {
  // First, find out file where basis set is
  std::string filename=find_basis(basis);

  // Input file
  std::ifstream in(filename.c_str());

  if(in.is_open()) {
    // OK, file was succesfully opened.
    std::string line;
    bool useold=0;

    while(in.good()) {

      // Basis set for a given element
      ElementBasisSet el;

      // Do we need a new input line?
      if(useold) {
	useold=0;
      } else {
	// Get next line in file
	line=readline(in);
      }

      // Check for magical entry "****" marking the beginning of an element entry
      //      if(line.compare(0,4,"****")==0) {

      std::vector<std::string> line_split;
      line_split=splitline(line);

      if(line_split.size()==2) {
	// OK, found an element entry.

	// The symbol of the element is
	std::string sym=line_split[0];
	// and the atom number the basis is for is
	size_t num=readint(line_split[1]);

	// Check that there is no duplicate entry
	bool found=0;
	for(size_t i=0;i<elements.size();i++)
	  if( (elements[i].get_symbol()==sym) && (elements[i].get_number()==num))
	    found=1;
	if(found) {
	  std::ostringstream oss;
	  ERROR_INFO();
	  oss << "Error: multiple basis set definitions found for element " << sym << " in file " << filename << "!\n";
	  throw std::runtime_error(oss.str());
	}

	// Create basis set structure for the element
	el=ElementBasisSet(sym,num);

	// Now, proceed by reading in the basis functions
	while(1) {
	  // Get next line
	  line=readline(in);
	  line_split=splitline(line);
	  // If we have run into the separator "****" then there are no more shells.
	  //	  if(line.compare(0,4,"****")==0) {
	  if(line_split.size()==1 && line_split[0]=="****") {
	    // Add element to list
	    el.sort();
	    elements.push_back(el);
	    // Use the same line
	    useold=1;
	    // Break loop
	    break;
	  } else {
	    // Nope, there is a shell.
	    std::vector<std::string> words=splitline(line);

	    // The shell type is
	    std::string shelltype=words[0];
	    // The amount of exponents is
	    int nc=readint(words[1]);

	    if(shelltype.size()==2) {
	      // This is an SP shell!
	      FunctionShell S(0), P(1);
	      
	      // Read the exponents
	      for(int i=0;i<nc;i++) {
		line=readline(in);
		// Numbers
		std::vector<std::string> nums=splitline(line);
		// Add functions
		S.add_exponent(readdouble(nums[1]),readdouble(nums[0]));
		P.add_exponent(readdouble(nums[2]),readdouble(nums[0]));
	      }
	      el.add_function(S);
	      el.add_function(P);
	    } else {
	      // This is a normal shell
	      int am=find_am(shelltype[0]);
	      FunctionShell sh(am);
	      // Read the exponents
	      for(int i=0;i<nc;i++) {
		line=readline(in);
		// Numbers
		std::vector<std::string> nums=splitline(line);
		// Add functions
		sh.add_exponent(readdouble(nums[1]),readdouble(nums[0]));
	      }
	      el.add_function(sh);
	    }
	  }
	}
      }
    }
  } else {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Could not open basis library file " << filename << "!\n";
    throw std::runtime_error(oss.str());
  }
}

void BasisSetLibrary::save_gaussian94(const char * filename) const {
  FILE *out;
  out=fopen(filename,"w");

  // Loop over elements
  for(size_t iel=0;iel<elements.size();iel++) {
    // Write out name of element
    fprintf(out,"%s\t0\n",elements[iel].symbol.c_str());
    // Loop over shells
    for(size_t ish=0;ish<elements[iel].bf.size();ish++) {
      // Print out type and length of shell
      fprintf(out,"%c   %i   1.00\n",shell_types[elements[iel].bf[ish].am],(int) elements[iel].bf[ish].C.size());
      // Print out contraction
      for(size_t iexp=0;iexp<elements[iel].bf[ish].C.size();iexp++)
	fprintf(out,"\t%.16e\t\t%.16e\n",elements[iel].bf[ish].z[iexp],elements[iel].bf[ish].C[iexp]);
    }
    // Close entry
    fprintf(out,"****\n");
  }
  
  fclose(out);
}

void BasisSetLibrary::add_element(const ElementBasisSet & el) {
  elements.push_back(el);
}

void BasisSetLibrary::sort() {
  // Sort shells in elements
  for(size_t i=0;i<elements.size();i++)
    elements[i].sort();
  // Sort order of elements
  stable_sort(elements.begin(),elements.end());
}

size_t BasisSetLibrary::get_Nel() const {
  return elements.size();
}

std::string BasisSetLibrary::get_symbol(size_t ind) const {
  return elements[ind].get_symbol();
}

int BasisSetLibrary::get_max_am() const {
  int maxam=elements[0].get_max_am();
  for(size_t i=1;i<elements.size();i++)
    if(elements[i].get_max_am()>maxam)
      maxam=elements[i].get_max_am();
  return maxam;
}

void BasisSetLibrary::print() const {
  for(size_t i=0;i<elements.size();i++)
    elements[i].print();
}

ElementBasisSet BasisSetLibrary::get_element(std::string el, size_t number) const {
  // Get element from library
  
  // Go through library to find element
  for(size_t i=0;i<elements.size();i++)
    if(elements[i].get_symbol()==el && elements[i].get_number()==number)
      return elements[i];

  // If we are still here, it means the element was not found.
  //  ERROR_INFO(); // Don't print info, since we normally catch the error.
  std::ostringstream oss;
  oss << "Could not find basis for element " << el << " with atom number " << number << " in library!\n";
  throw std::runtime_error(oss.str());
  
  // Dummy return clause
  return ElementBasisSet();
}
