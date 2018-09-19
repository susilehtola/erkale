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

int strcmp(const std::string &str1, const std::string& str2) {
  return strcmp(str1.c_str(),str2.c_str());
}

std::string readline(std::istream & in, bool skipempty, const std::string & cchars) {
  std::string ret;

  // Get line from input
  while(in.good()) {
    // Get line from file
    getline(in,ret);

    // Skip empty lines?
    if(!skipempty)
      return ret;

    // Check that line is not a comment
    if(ret.size()) {
      bool cmt=false;
      for(size_t j=0;j<cchars.size();j++)
	if(ret[0]==cchars[j])
	  cmt=true;

      if(!cmt && !isblank(ret))
	return ret;
    }
  }

  // Reached end of file
  return std::string();
}

bool isblank(const std::string & line) {
  bool blank(true);
  for(size_t i=0;i<line.size();i++)
    if(!isblank(line[i]))
      blank=false;

  return blank;
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

std::vector<std::string> splitline(const std::string & line) {
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

std::string trim(const std::string & line) {
  // Find start.
  size_t start;
  for(start=0;start<line.size();start++)
    if(!isblank(line[start]))
      break;

  // Find end.
  size_t end;
  for(end=line.size()-1;end<line.size();end--)
    if(!isblank(line[end]))
      break;

  if(end>line.size()) {
    // Empty line!
    std::string ret;
    return ret;
  } else {
    // Get substring.
    return line.substr(start,end-start+1);
  }
}

std::string rem_dbl_whitespace(const std::string & line) {
  std::string ret;

  // Did we already encounter whitespace?
  bool white=false;
  for(size_t i=0;i<line.size();i++) {
    if(isblank(line[i]) && !white) {
      // We did not encounter whitespace yet. Add one.
      ret+=" ";
      white=true;
    } else if(!isblank(line[i])) {
      white=false;
      ret+=line[i];
    }
  }

  return ret;
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

void print_E(const arma::vec & E, const std::vector<double> & occ, bool all) {
  // Print first N elements of E

  // Print nelem entries per line
  size_t nelem=5;
  // Total amount of lines to print. Always print one additional line
  // of energies.
  size_t Ntot=all ? E.n_elem : (size_t) ceil(occ.size()*1.0/nelem+1)*nelem;

  // Safety check:
  if(E.n_elem<Ntot){
    Ntot=E.n_elem;
  }

  // Skip additional line at the end?
  bool skipline= Ntot%nelem ? true : false;

  // Printout format
  const char fmt_occ[] ="% 13.4f*";
  const char fmt_half[]="% 13.4fo";
  const char fmt_virt[]="% 13.4f ";
  // Energy cutoff, determined from above
  double cutoff=1e7;
  const char fmt_cut[]="************* ";

  // Compute gap. Find HOMO and LUMO
  if(occ.size()) {
    size_t homo, lumo;
    for(homo=occ.size()-1;homo<occ.size();homo--)
      if(occ[homo]>0.0)
	break;
    for(lumo=0;lumo<occ.size();lumo++)
      if(occ[lumo]==0.0)
	break;
    bool occpd=false;
    for(size_t i=0;i<occ.size();i++)
      if(occ[i]>0.0)
        occpd=true;

    if(occpd) {
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
  }

  if(all)
    printf("Orbital energies:\n");
  else
    printf("Energies of lowest lying orbitals:\n");

  // Loop over states
  for(size_t i=0;i<Ntot;i++) {
    // Is state occupied?
    if(E(i)>=cutoff)
      printf("%s",fmt_cut);
    else if(i<occ.size() && occ[i]>=1.0)
      printf(fmt_occ,E(i));
    else if(i<occ.size() && occ[i]==0.5)
      printf(fmt_half,E(i));
    else
      printf(fmt_virt,E(i));
    // Return line if necessary
    if(i%nelem==nelem-1) {
      printf("\n");
    }
  }
  if(skipline)
    printf("\n");
}

std::string memory_size(size_t memsize, bool approx) {
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
    if(approx)
      return ret.str();
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
    if(approx)
      return ret.str();
  }

  // Number of kilobytes
  long unsigned int kilos=size/kilo;
  if(kilos>0) {
    // Check that there is a space at the end
    std::string tmp=ret.str();
    if(tmp.size() && tmp[tmp.size()-1]!=' ')
      ret << " ";

    ret << kilos << " ki";
    if(approx)
      return ret.str();
  }

  return ret.str();
}

void print_symmat(const arma::mat &mat, bool floatformat, double cutoff) {
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

void print_mat(const arma::mat & mat, const char *fmt) {
  for(size_t ir=0;ir<mat.n_rows;ir++) {
    for(size_t ic=0;ic<mat.n_cols;ic++)
      printf(fmt,mat(ir,ic));
    printf("\n");
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


std::vector<size_t> parse_range(const std::string & in, bool convert) {
  std::vector<size_t> ret;

  // First break this wrt commas
  std::vector<std::string> comma=parse(in,",");
  for(size_t ic=0;ic<comma.size();ic++) {
    // Now we can break wrt semicolons
    std::vector<std::string> dash=parse(comma[ic],":");

    if(dash.size()>2) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Only one dash allowed in number range.\n";
      throw std::runtime_error(oss.str());
    }

    if(dash.size()==0) {
      // No number given.
      continue;
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

  if(convert)
    // Convert to C indexing
    for(size_t i=0;i<ret.size();i++)
      ret[i]--;

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

std::string space_number(int numb) {
  // Convert number to string representation
  std::ostringstream numo;
  numo << numb;
  std::string num=numo.str();

  // Print out space after number?
  bool space[num.size()];
  for(size_t i=0;i<num.size();i++)
    space[i]=false;

  // Determine where to plug in spaces
  size_t ip=0;
  for(size_t i=num.size()-1;i<num.size();i--) {
    ip++; // Printed numbers
    if(ip%3==0 && i>0) {
      space[i-1]=true;
      ip=0;
    }
  }

  // Collect number
  std::ostringstream out;
  for(size_t i=0;i<num.size();i++) {
    out << num[i];
    if(space[i])
      out << " ";
  }

  return out.str();
}

std::string print_bar(std::string msg, char pad, int width, bool upper) {
  // Trim message
  msg=trim(msg);

  // Upper case?
  if(upper)
    msg=toupper(msg);

  // Calculate padding
  int lpad, rpad;
  // Length of message (including white spaces)
  int msglen=(int) msg.size() + 2;

  rpad = (width - msglen)/2;
  lpad = width - msglen - rpad;

  std::ostringstream oss;
  for(int i=0;i<lpad;i++)
    oss << pad;
  oss << " " << msg << " ";
  for(int i=0;i<rpad;i++)
    oss << pad;

  return oss.str();
}
