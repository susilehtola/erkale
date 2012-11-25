/**
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


#include "basislibrary.h"
#include "elements.h"
#include "mathf.h"
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
  for(int i=0;i<=max_am;i++)
    if(shell_types[i]==am)
      return i;

  ERROR_INFO();
  std::ostringstream oss;
  oss << "Angular momentum "<<am<<" not found!\n";
  throw std::runtime_error(oss.str());

  return -1;
}

std::string find_basis(const std::string & basisname, bool verbose) {
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
	if(verbose)
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

FunctionShell::FunctionShell(int amval, const std::vector<contr_t> & c) {
  am=amval;
  C=c;
}

FunctionShell::~FunctionShell() {
}

void FunctionShell::add_exponent(double Cv, double zv) {
  contr_t tmp;
  tmp.c=Cv;
  tmp.z=zv;
  C.push_back(tmp);
  sort();
}

void FunctionShell::sort() {
  // Sort exponents in decreasing order

  std::stable_sort(C.begin(),C.end());
}

void FunctionShell::normalize() {
  // If there's a single function on the shell, its coefficient is unity.
  if(C.size()==1) {
    C[0].c=1.0;
    return;
  }

  // Calculate overlap of normalized functions
  double S=0.0;
  for(size_t i=0;i<C.size();i++)
    for(size_t j=0;j<C.size();j++)
      S+=C[i].c*C[j].c*std::pow(4*C[i].z*C[j].z/std::pow(C[i].z+C[j].z,2),am/2.0+3.0/4.0);

  // The coefficients must be scaled by 1/sqrt(S)
  S=sqrt(S);
  for(size_t i=0;i<C.size();i++)
    C[i].c/=S;

  // Check sign of coefficient with maximum absolute value
  double maxfabs=0.0;
  for(size_t i=0;i<C.size();i++)
    if(fabs(C[i].c)>fabs(maxfabs))
      maxfabs=C[i].c;
  if(maxfabs<0.0)
    for(size_t i=0;i<C.size();i++)
      C[i].c*=-1.0;
}

void FunctionShell::print() const {
  printf("\tam = %i, %i functions\n",am, (int) C.size());
  for(size_t i=0;i<C.size();i++)
    printf("\t\t%e\t%e\n",C[i].c,C[i].z);
}

bool FunctionShell::operator<(const FunctionShell & rhs) const {
  // First, check if angular momentum is lower.
  if(am!=rhs.am)
    return am<rhs.am;
  else
    // Same angular momentum, sort by first exponent.
    return C[0]<rhs.C[0];
}

int FunctionShell::get_am() const {
  return am;
}

std::vector<contr_t> FunctionShell::get_contr() const {
  return C;
}

ElementBasisSet::ElementBasisSet() {
  // Default values
  symbol="";
  number=0;
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

void ElementBasisSet::normalize() {
  for(size_t i=0;i<bf.size();i++)
    bf[i].normalize();
}

void ElementBasisSet::print() const {
  printf("%s %i:\n",symbol.c_str(),(int) number);
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

void ElementBasisSet::set_number(size_t num) {
  number=num;
}

bool ElementBasisSet::operator<(const ElementBasisSet &rhs) const {
  return get_Z(symbol)<get_Z(rhs.symbol);
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

      // Get contraction on shell
      std::vector<contr_t> shc=bf[ish].get_contr();

      // Loop over exponents
      for(size_t iexp=0;iexp<shc.size();iexp++) {
	// First, check if exponent is already on list
	bool found=0;
	for(size_t i=0;i<exps.size();i++)
	  if(exps[i]==shc[iexp].z) {
	    found=1;
	    break;
	  }

	// If exponent was not found, add it to the list.
	if(!found)
	  exps.push_back(shc[iexp].z);
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
	std::vector<contr_t> shc=bf[ish].get_contr();

	// Find current exponent
	bool found=0;
	for(size_t i=0;i<shc.size();i++)
	  if(shc[i].z==exps[iexp]) {
	    // Found exponent!
	    found=1;
	    // Store contraction coefficient.
	    coeffs(iexp,iish)=shc[i].c;
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

void ElementBasisSet::decontract() {
  // Create list of exponents: zeta[l][1,..,nx]
  std::vector< std::vector<double> > zeta;
  zeta.resize(get_max_am()+1);

  // Loop over shells
  for(size_t ish=0;ish<bf.size();ish++) {
    // Angular momentum of the shell is
    int am=bf[ish].get_am();

    // Get exponents
    std::vector<contr_t> c=bf[ish].get_contr();

    // See if these need to be added to the list.
    for(size_t iexp=0;iexp<c.size();iexp++) {
      if(zeta[am].size()==0)
	zeta[am].push_back(c[iexp].z);
      else {
	// Get upper bound
	std::vector<double>::iterator high;
	high=std::upper_bound(zeta[am].begin(),zeta[am].end(),c[iexp].z);

	// Corresponding index is
	size_t ind=high-zeta[am].begin();

	if(ind>0 && zeta[am][ind-1]==c[iexp].z)
	  // Already on list. Don't do anything.
	  ;
	else {
	  // Term does not exist, add it
	  zeta[am].insert(high,c[iexp].z);
	}
      }
    }
  }

  // Create new basis set
  ElementBasisSet decontr(symbol);
  for(int am=0;am<=get_max_am();am++)
    for(size_t iexp=0;iexp<zeta[am].size();iexp++) {
      // Create new shell
      FunctionShell tmp(am);
      tmp.add_exponent(1.0,zeta[am][iexp]);
      decontr.add_function(tmp);
    }
  decontr.sort();

  // Change to decontracted set
  *this=decontr;
}


BasisSetLibrary::BasisSetLibrary() {
}

BasisSetLibrary::~BasisSetLibrary() {
}

void BasisSetLibrary::load_gaussian94(const char * filename, bool verbose) {
  load_gaussian94(std::string(filename),verbose);
}

void BasisSetLibrary::load_gaussian94(const std::string & basis, bool verbose) {
  // First, find out file where basis set is
  std::string filename=find_basis(basis,verbose);

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
	bool dupl=0, numfound=0;
	for(size_t i=0;i<elements.size();i++) {

	  if(elements[i].get_number()==num) {
	    // Already an element with this index!
	    numfound=1;

	    if(elements[i].get_symbol()==sym)
	      // And the entry is even duplicate!
	      dupl++;
	  }
	}

	if(dupl) {
	  std::ostringstream oss;
	  ERROR_INFO();
	  oss << "Error: multiple basis set definitions found for element " << sym << " in file " << filename << "!\n";
	  throw std::runtime_error(oss.str());
	} else if(num>0 && numfound) {
	  std::ostringstream oss;
	  ERROR_INFO();
	  oss << "Error: a special basis set given multiple times for center " << num;
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

	    if(strcmp(shelltype.c_str(),"SP")==0) {
	      // SP shell
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
	    } else if(shelltype.size()==1) {
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
	    } else {
	      // AM given with L=%i

	      if(shelltype.size()<3)
		throw std::runtime_error("Unrecognized shell type!\n");

	      // Check beginning
	      if(stricmp(shelltype.substr(0,2),"L=")!=0)
		throw std::runtime_error("Could not parse shell type.\n");

	      // Now get the shell type
	      int am=readint(shelltype.substr(2));

	      // and add the exponents
              FunctionShell sh(am);
              for(int i=0;i<nc;i++) {
                line=readline(in);
		std::vector<std::string> nums=splitline(line);
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

void BasisSetLibrary::save_gaussian94(const char * filename, bool append) const {
  FILE *out;
  if(append)
    out=fopen(filename,"a");
  else
    out=fopen(filename,"w");

  // Loop over elements
  for(size_t iel=0;iel<elements.size();iel++) {
    // Write out name of element
    fprintf(out,"%-2s %i\n",elements[iel].symbol.c_str(),(int) elements[iel].get_number());
    // Loop over shells
    for(size_t ish=0;ish<elements[iel].bf.size();ish++) {
      // Print out type and length of shell
      if(elements[iel].bf[ish].am<7)
	fprintf(out,"%c   %i   1.00\n",shell_types[elements[iel].bf[ish].am],(int) elements[iel].bf[ish].C.size());
      else
	fprintf(out,"L=%i %i   1.00\n",elements[iel].bf[ish].am,(int) elements[iel].bf[ish].C.size());
      // Print out contraction
      for(size_t iexp=0;iexp<elements[iel].bf[ish].C.size();iexp++)
	fprintf(out,"  %.10e  % .10e\n",elements[iel].bf[ish].C[iexp].z,elements[iel].bf[ish].C[iexp].c);
    }
    // Close entry
    fprintf(out,"****\n");
  }

  fclose(out);
}

void BasisSetLibrary::save_gaussian94(const std::string & filename, bool append) const {
  save_gaussian94(filename.c_str(),append);
}

void BasisSetLibrary::save_dalton(const char * filename, bool append) const {
  FILE *out;
  if(append)
    out=fopen(filename,"a");
  else {
    out=fopen(filename,"w");
    fprintf(out,"$ Supported elements\n$");
    for(size_t i=0;i<elements.size();i++)
      fprintf(out," %s",elements[i].get_symbol().c_str());
    fprintf(out,"\n");

    fprintf(out,"************************************************************************\n");
  }

  // Loop over elements
  for(size_t iel=0;iel<elements.size();iel++) {
    // Get element
    ElementBasisSet el=elements[iel];
    // Print element
    fprintf(out,"a %i\n",get_Z(el.get_symbol()));
    // Loop over angular momentum
    for(int l=0;l<=el.get_max_am();l++) {
      // Get exponents and contraction coefficients
      std::vector<double> exps;
      arma::mat coeffs;
      el.get_primitives(exps,coeffs,l);

      // Print label
      fprintf(out,"$ %s\n",toupper(element_names[get_Z(el.get_symbol())]).c_str());
      fprintf(out,"$ %c-TYPE FUNCTIONS\n",toupper(shell_types[l]));
      // Print element, number of exponents and contracted functions
      fprintf(out,"%4i %4i %4i\n",(int) exps.size(),(int) coeffs.n_cols,0);

      // Loop over exponents
      for(size_t iexp=0;iexp<exps.size();iexp++) {
	// Print exponent
	fprintf(out,"% 14.8f",exps[iexp]);
	// and contraction scheme
	int np=1; // amount of printed entries
	for(size_t ic=0;ic<coeffs.n_cols;ic++) {
	  if(np==0)
	    fprintf(out,"% 14.8f",coeffs(iexp,ic));
	  else
	    fprintf(out," % .8f",coeffs(iexp,ic));
	  np++;
	  if(np==7) {
	    fprintf(out,"\n");
	    np=0;
	  }
	}
	if(np!=0)
	  fprintf(out,"\n");
      }
    }
  }
  fclose(out);
}

void BasisSetLibrary::save_dalton(const std::string & filename, bool append) const {
  save_dalton(filename.c_str(),append);
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

void BasisSetLibrary::normalize() {
  // Normalize coefficients
  for(size_t i=0;i<elements.size();i++)
    elements[i].normalize();
}

size_t BasisSetLibrary::get_Nel() const {
  return elements.size();
}

std::string BasisSetLibrary::get_symbol(size_t ind) const {
  return elements[ind].get_symbol();
}

std::vector<ElementBasisSet> BasisSetLibrary::get_elements() const {
  return elements;
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

  if(number==0) {
    // General basis requested
    for(size_t i=0;i<elements.size();i++)
      if((elements[i].get_number()==number) && (elements[i].get_symbol()==el))
	return elements[i];
  } else {
    // Special basis requested.
    for(size_t i=0;i<elements.size();i++)
      if(elements[i].get_number()==number) {
	// Check that this is actually of the wanted type!
	if(elements[i].get_symbol()==el)
	  return elements[i];
	else {
	  // The wanted index, but a nucleus of the wrong type!
	  std::ostringstream oss;
	  oss << "Requested basis for nucleus " << el << " with index " <<number<<" but in the basis definition the given element is " << elements[i].get_symbol() << "!\n";
	  throw std::runtime_error(oss.str());
	}
      }
  }


  // If we are still here, it means the element was not found.
  //  ERROR_INFO(); // Don't print info, since we normally catch the error.
  std::ostringstream oss;
  oss << "Could not find basis for element " << el << " with atom number " << number << " in library!\n";
  throw std::runtime_error(oss.str());

  // Dummy return clause
  return ElementBasisSet();
}

void BasisSetLibrary::decontract(){
  name="Decontracted "+name;
  for(size_t iel=0;iel<elements.size();iel++)
    elements[iel].decontract();
}
