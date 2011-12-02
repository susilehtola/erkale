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
#include "mathf.h"

#include <cstdio>
#include <cstring>
#include <sstream>

/// Maximum length of lines in input file
#define MAXLEN 1024

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

int stricmp(const std::string &str1, const std::string& str2) {
  return strcasecmp(str1.c_str(),str2.c_str());
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

std::string readline(FILE *in) {
  // Input array
  char tmp[MAXLEN];
  // Location on input line array
  size_t itmp=0;
  // Input character
  int c;

  while((c=getc(in))!=EOF) {
    if(c=='\n') {
      // Pad string with zeros
      tmp[itmp++]='\0';
      return std::string(tmp);
    } else {

      // Store the line
      tmp[itmp++]=c;
    }
  }

  if(c==EOF)
    throw std::runtime_error("End of file!\n");

  std::string ret;
  return ret;
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

void print_E(const arma::vec & E, const std::vector<double> & occ) {
  // Print first N elements of E

  // Print nelem entries per line
  size_t nelem=5;
  // Total amount of lines to print. Always print one additional line
  // of energies.
  size_t Ntot=(size_t) ceil(occ.size()*1.0/nelem+1)*nelem;

  // Skip additional line at the end?
  bool skipline=0;

  // Safety check:
  if(E.n_elem<Ntot) {
    Ntot=E.n_elem;
    if(E.n_elem%nelem!=0)
      skipline=1;
  }

  char fmt[] ="% 13.6f*";
  char fmtv[]="% 13.6f ";

  // Compute gap. Find HOMO and LUMO
  if(occ.size()) {
    size_t homo, lumo;
    for(homo=occ.size()-1;homo<occ.size();homo--)
      if(occ[homo]>0.0)
	break;
    for(lumo=0;lumo<occ.size();lumo++)
      if(occ[lumo]==0.0)
	break;

    if(homo>E.n_elem) {
      //    ERROR_INFO();
      std::ostringstream oss;
      oss << "Orbital " << homo+1 << " is occupied but only " << E.n_elem << " energies given!\n";
      throw std::runtime_error(oss.str());
    }

    if(lumo<E.n_elem) {

      double gap=E(lumo)-E(homo);
      // Convert it into eV
      gap*=HARTREEINEV;

      printf("HOMO-LUMO gap is %7.2f eV. ",gap);
    }
  }

  printf("Energies of lowest lying states:\n");

  // Loop over states
  for(size_t i=0;i<Ntot;i++) {
    // Is state occupied?
    if(i<occ.size() && occ[i]>0.0)
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

std::string memory_size(size_t memsize) {
  std::ostringstream ret;

  // We need long unsigned integers, since 2GB is already over INT_MAX.
  const long unsigned int kilo=1024;
  const long unsigned int mega=kilo*kilo;
  const long unsigned int giga=mega*kilo;
  long unsigned int size=(long unsigned int) memsize;

  // Number of gigabytes
  long unsigned int gigs=size/giga;
  if(gigs>0) {
    size-=gigs*giga;
    ret << gigs;
    ret << " Gi";
  }

  // Number of megabytes
  long unsigned int megs=size/mega;
  if(megs>0) {
    size-=megs*mega;

    // Check that there is a space at the end
    std::string tmp=ret.str();
    if(tmp.size() && tmp[tmp.size()-1]!=' ')
      ret << " ";

    ret << megs << " Mi";
  }

  // Number of kilobytes
  long unsigned int kilos=size/kilo;
  if(kilos>0) {
    // Check that there is a space at the end
    std::string tmp=ret.str();
    if(tmp.size() && tmp[tmp.size()-1]!=' ')
      ret << " ";

    ret << kilos << " ki";
  }

  return std::string(ret.str());
}

void print_sym(const arma::mat &mat, bool floatformat, double cutoff) {
  // Determine cutoff
  cutoff*=max_abs(mat);

  if(!floatformat) {
    for(size_t row=0;row<mat.n_rows;row++) {
      printf("%7i ",(int) row+1);
      for(size_t col=0;col<=row;col++) {
	if(fabs(mat(row,col))>=cutoff)
	  printf(" % 13.5e",mat(row,col));
	else
	  printf(" % 13.5e",0.0);
      }
      printf("\n");
    }
  } else {
    for(size_t row=0;row<mat.n_rows;row++) {
      printf("%7i ",(int) row+1);
      for(size_t col=0;col<=row;col++) {
	if(fabs(mat(row,col))>=cutoff)
	  printf(" % 8.5f",mat(row,col));
	else
	  printf(" % 8.5f",0.0);
      }
      printf("\n");
    }
  }
}

void print_orb(const arma::mat & C, const arma::vec & E) {
  // How many orbitals to print per block?
  const int norb=5;
  // Amount of blocks to print
  const int nblocks=(int) ceil(E.n_elem*1.0/norb);

  for(int iblock=0;iblock<nblocks;iblock++) {
    // Get number of orbitals in this block
    int no=std::min(norb,(int) E.n_elem-iblock*norb);

    // Print orbital indices
    //    printf("%11s ","Orbital");
    printf("%11s ","");
    for(int io=0;io<no;io++)
      printf("% 12i ",iblock*norb+io+1);
    printf("\n");

    // Print eigenvalues
    printf("%11s ","Eigenvalue");
    for(int io=0;io<no;io++)
      printf("% 12.5e ",E(iblock*norb+io));
    printf("\n");

    // Print coefficients
    for(size_t ibf=0;ibf<C.n_rows;ibf++) {
      // Index of basis function
      printf("%11i ",(int) ibf+1);
      for(int io=0;io<no;io++)
	printf("% 12.5f ",C(ibf,iblock*norb+io));
      printf("\n");
    }
  }
}

std::vector<std::string> parse(std::string in, const std::string & separator) {
  // Parse the input for separator

  // Returned variable
  std::vector<std::string> ret;

  size_t ind;
  while((ind=in.find_first_of(separator))!=std::string::npos) {
    // Add it to the stack
    ret.push_back(in.substr(0,ind));
    // and remove that part from the string
    in=in.substr(ind+1,in.size()-ind-1);
  }

  // If there is still something in in, add it to the stack.
  if(in.size())
    ret.push_back(in);

  return ret;
}


std::vector<size_t> parse_range(const std::string & in) {
  std::vector<size_t> ret;

  // First break this wrt commas
  std::vector<std::string> comma=parse(in,",");
  for(size_t ic=0;ic<comma.size();ic++) {
    // Now we can break wrt dashes.
    std::vector<std::string> dash=parse(comma[ic],"-");

    if(dash.size()>2) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Only one dash allowed in number range.\n";
      throw std::runtime_error(oss.str());
    }

    if(dash.size()==1) {
      // Only a single number given.
      ret.push_back(readint(dash[0]));
    } else {
      // We are dealing with a range. Read upper and lower limits
      size_t low=readint(dash[0]);
      size_t up=readint(dash[1]);

      if(low>up) {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Range is not monotonically increasing!\n";
	throw std::runtime_error(oss.str());
      }

      // Fill the range.
      for(size_t i=low;i<=up;i++)
	ret.push_back(i);
    }
  }

  // Sort in increasing order
  sort(ret.begin(),ret.end());

  return ret;
}

std::vector<double> parse_range_double(const std::string & in) {
  std::vector<double> ret;

  // First break this wrt commas
  std::vector<std::string> comma=parse(in,",");
  for(size_t ic=0;ic<comma.size();ic++) {
    // Now we can break wrt the semicolon
    std::vector<std::string> lim=parse(comma[ic],":");

    if(lim.size()==1) {
      // Only single number given, add it to the stack.
      ret.push_back(readdouble(lim[0]));
    } else {

      if(lim.size()!=3) {
	std::ostringstream oss;
	oss << "The given input with " << lim.size() << "entries is not a valid range of numbers.\n";
	ERROR_INFO();
	throw std::runtime_error(oss.str());
      }

      // Read minimum, maximum and spacing.
      double min=readdouble(lim[0]);
      double dx=readdouble(lim[1]);
      double max=readdouble(lim[2]);

      if(dx<=0.0) {
	ERROR_INFO();
	throw std::runtime_error("Grid spacing must be positive.\n");
      }
      if(max<min) {
	ERROR_INFO();
	throw std::runtime_error("Grid maximum cannot be smaller than minimum!\n");
      }

      // Form the points
      size_t N=(size_t) ((max-min)/dx)+1;

      for(size_t i=0;i<N;i++)
	ret.push_back(min+i*dx);
    }
  }

  // Sort in increasing order
  sort(ret.begin(),ret.end());

  return ret;
}

void parse_cube(const std::string & sizes, std::vector<double> & x, std::vector<double> & y, std::vector<double> & z) {
  // Clear the arrays of any existing content.
  x.clear();
  y.clear();
  z.clear();

  // Split output in x, y and z
  std::vector<std::string> info=splitline(sizes);
  // If only one specification was given, triple it.
  if(info.size()==1) {
    info.push_back(info[0]);
    info.push_back(info[0]);
  }

  // Check that we have the right amount of info
  if(info.size()!=3) {
    std::ostringstream oss;
    oss << "The given input \"" << sizes << "\" is not a valid cube definition.\n";
    ERROR_INFO();
    throw std::runtime_error(oss.str());
  }

  // Read data.
  x=parse_range_double(info[0]);
  y=parse_range_double(info[1]);
  z=parse_range_double(info[2]);
}
