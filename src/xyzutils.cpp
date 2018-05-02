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


#include "elements.h"
#include "global.h"
#include "xyzutils.h"
#include "stringutil.h"
#include "zmatrix.h"

#include <sstream>
#include <stdexcept>

#include <fstream>
#include <cstdio>


std::vector<atom_t> load_xyz(std::string filename, bool convert) {
  // Check if input is actually Z-Matrix
  {
    FILE *in=fopen(filename.c_str(),"r");
    if(in==NULL)
      throw std::runtime_error("Error opening geometry file \"" + filename + "\".\n");
    std::vector<std::string> words(splitline(readline(in)));
    fclose(in);
    if(stricmp(words[0],"#ZMATRIX")==0)
      return load_zmat(filename, convert);
  }

  // Input file
  std::ifstream in(filename.c_str());
  // Returned array
  std::vector<atom_t> atoms;

  if(in.good()) {
    // OK, file was succesfully opened.

    // Read the first line to get the number of atoms
    std::string line=readline(in);
    std::vector<std::string> words=splitline(line);
    int Nat=readint(words[0]);

    // Reserve enough memory.
    atoms.reserve(Nat);

    // The next line contains the comment, skip through it. Also
    // accept an empty line.
    line=readline(in,false);

    // Now, proceed with reading in the atoms.
    for(int i=0;i<Nat;i++) {
      // Helper structure
      atom_t tmp;

      if(!in.good()) {
	std::ostringstream oss;
	oss << "File \""<<filename<<"\" ended unexpectedly!\n";
	throw std::runtime_error(oss.str());
      }

      // Get line containing the input
      line=readline(in);
      // and split it to words
      words=splitline(line);

      if(!words.size()) {
	std::ostringstream oss;
	oss << "File \""<<filename<<"\" ended unexpectedly!\n";
	throw std::runtime_error(oss.str());
      }
      if(words.size()<4) {
	std::ostringstream oss;
	oss << "Malformed xyz file \"" << filename << "\"!\n";
	throw std::runtime_error(oss.str());
      }

      // and extract the information.

      // Was element given as number or symbol?
      if(isdigit(words[0][0])) {
	int Z=readint(words[0]);
	if(Z> (int) (sizeof(element_symbols)/sizeof(element_symbols[0]))) {
	  std::ostringstream oss;
	  oss << "Too heavy atom Z =" << Z << " requested in xyz file.\n";
	  throw std::runtime_error(oss.str());
	} else if(Z<0) {
	  throw std::runtime_error("Can't have nucleus with negative charge.\n");
	}

	tmp.el=element_symbols[Z];
      } else
	// Given as symbol
	tmp.el=words[0];

      tmp.num=i; // Number of atom
      tmp.x=readdouble(words[1]);
      tmp.y=readdouble(words[2]);
      tmp.z=readdouble(words[3]);
      if(convert) {
        tmp.x*=ANGSTROMINBOHR;
        tmp.y*=ANGSTROMINBOHR;
        tmp.z*=ANGSTROMINBOHR;
      }
      // Charge defined?
      tmp.Q=words.size()==5 ? readint(words[4]) : 0;
      // and add the atom to the list.
      atoms.push_back(tmp);
    }
  } else {
    std::ostringstream oss;
    oss << "Could not open xyz file \""<<filename<<"\"!\n";
    throw std::runtime_error(oss.str());
  }

  if(atoms.size()==0) {
    std::ostringstream oss;
    oss << "File \""<<filename<<"\" contains no atoms!\n";
    throw std::runtime_error(oss.str());
  }

  return atoms;
}

void save_xyz(const std::vector<atom_t> & at, const std::string & comment, const std::string & fname, bool append) {
  // Output file
  FILE *out;

  if(append)
    out=fopen(fname.c_str(),"a");
  else
    out=fopen(fname.c_str(),"w");

  // Print out number of atoms
  fprintf(out,"%u\n",(unsigned int) at.size());
  // Print comment
  fprintf(out,"%s\n",comment.c_str());
  // Print atoms
  for(size_t i=0;i<at.size();i++)
    fprintf(out,"%-4s  % 10.5f  % 10.5f  % 10.5f\n",at[i].el.c_str(),at[i].x/ANGSTROMINBOHR,at[i].y/ANGSTROMINBOHR,at[i].z/ANGSTROMINBOHR);
  fclose(out);
}

void print_xyz(const std::vector<atom_t> & at) {
  for(size_t i=0;i<at.size();i++)
    printf("%4i %-4s  % 10.5f  % 10.5f  % 10.5f\n",(int) i+1, at[i].el.c_str(),at[i].x/ANGSTROMINBOHR,at[i].y/ANGSTROMINBOHR,at[i].z/ANGSTROMINBOHR);
}
