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



#include "stringutil.h"

#include <cstdio>
#include <sstream>

std::string tolower(const std::string & in) {
  std::string ret=in;
  for(size_t i=0;i<ret.size();i++)
    ret[i]=tolower(ret[i]);
  return ret;
}

std::string toupper(const std::string & in) {
  std::string ret=in;
  for(size_t i=0;i<ret.size();i++)
    ret[i]=toupper(ret[i]);
  return ret;
}

std::string readline(std::istream & in) {
  std::string ret;

  // Get line from input
  while(in.good()) {
    // Get line from file
    getline(in,ret);
    // Check that line is not a comment
    if(ret.size() && !(ret[0]=='!' || ret[0]=='#')) {
      // Check if there is something on the line
      for(size_t i=0;i<ret.size();i++)
	if(!isblank(ret[i]))
	  return ret;
    }
  }

  // Reached end of file
  return std::string();
}
    

std::vector<std::string> splitline(std::string line) {
  // Split line into words.

  // The words found on the line
  std::vector<std::string> words;

  // Loop over input line
  for(size_t ind=0;ind<line.size();ind++) {
    if(!isblank(line[ind])) {
      // Found word.
      size_t start=ind;
      size_t end=ind;
      // Find out where it ends
      while(end<line.size() && !isblank(line[end]))
	end++;
      words.push_back(line.substr(start,end-start));
      ind=end;
    }
  }

  return words;
}

int readint(std::string num) {
  int no;
  std::stringstream(num) >> no;
  return no;
}

double readdouble(std::string num) {
  // Read double precision number from string

  // Change all D's into E's
  for(size_t ind=0;ind<num.size();ind++)
    if(num[ind]=='D')
      num[ind]='E';

  double no;
  std::stringstream(num) >> no;

  return no;
}

void print_E(size_t N, const arma::vec & E) {
  // Print first N elements of E
  
  // Print nelem entries per line
  size_t nelem=5;
  // Total amount of lines to print. Always print one additional line
  // of energies.
  size_t Ntot=(size_t) ceil(N*1.0/nelem+1)*nelem;

  // Skip additional line at the end?
  bool skipline=0; 

  // Safety check:
  if(E.n_elem<Ntot) {
    Ntot=E.n_elem;
    if(E.n_elem%nelem!=0)
      skipline=1;
  }

  char fmt[]="% 13.6f*";
  char fmtv[]="% 13.6f ";

  if(N<E.n_elem) {
    // Compute gap
    double gap=E(N)-E(N-1);
    // Convert it into eV
    gap*=HARTREEINEV;

    printf("Band gap is %7.2f eV. ",gap);
  } 
  
  printf("Energies of lowest lying states:\n");

  // Loop over states
  for(size_t i=0;i<Ntot;i++) {
    if(i<N)
      printf(fmt,E(i));
    else
      printf(fmtv,E(i));
    // Return line if necessary
    if(i%nelem==nelem-1)
      printf("\n");
  }
  if(skipline)
    printf("\n");
}

std::string memory_size(size_t size) {
  std::ostringstream ret;
  
  int kilo=1024;
  int mega=kilo*kilo;
  int giga=mega*kilo;

  // Number of gigabytes
  int gigs=size/giga;
  if(gigs>0) {
    size-=gigs*giga;
    ret << gigs;
    ret << " Gi";
  }

  // Number of megabytes
  int megs=size/mega;
  if(megs>0) {
    size-=megs*mega;

    // Check that there is a space at the end
    std::string tmp=ret.str();
    if(tmp.size() && tmp[tmp.size()-1]!=' ')
      ret << " ";

    ret << megs << " Mi";
  }

  // Number of kilobytes
  int kilos=size/kilo;
  if(kilos>0) {
    // Check that there is a space at the end
    std::string tmp=ret.str();
    if(tmp.size() && tmp[tmp.size()-1]!=' ')
      ret << " ";

    ret << kilos << " ki";
  }

  return std::string(ret.str());
}

  
