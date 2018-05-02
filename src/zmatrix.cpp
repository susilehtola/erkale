/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#include "zmatrix.h"
#include "stringutil.h"
#include <cstdio>
#include <vector>

static std::string concatenate_string(const std::vector<std::string> & words) {
  std::string ret(words[0]);
  for(size_t i=1;i<words.size();i++)
    ret+=" "+words[i];
  return ret;
}

static int parseind(const std::vector<std::string> & words, const std::vector<atom_t> & atoms, int idx) {
  int ind(readint(words[idx])-1);

  if(ind<0 || (size_t) ind >= atoms.size()) {
    std::ostringstream oss;
    oss << "Invalid reference atom on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
    throw std::runtime_error(oss.str());
  }

  return ind;
}

static int parseRind(const std::vector<std::string> & words, const std::vector<atom_t> & atoms) {
  return parseind(words,atoms,1);
}

static int parsethind(const std::vector<std::string> & words, const std::vector<atom_t> & atoms) {
  return parseind(words,atoms,3);
}

static int parsephiind(const std::vector<std::string> & words, const std::vector<atom_t> & atoms) {
  return parseind(words,atoms,5);
}

static double parseR(const std::vector<std::string> & words, bool convert) {
  double R(readdouble(words[2]));
  if(R<0.0) {
    std::ostringstream oss;
    oss << "Invalid bond length on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
    throw std::runtime_error(oss.str());
  }

  // Convert to Bohr
  if(convert)
    R*=ANGSTROMINBOHR;

  return R;
}

static double parseang(const std::vector<std::string> & words, int idx) {
  double A(readdouble(words[idx]));
  if(A<-180.0 || A>180.0) {
    std::ostringstream oss;
    oss << "Invalid bond length on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
    throw std::runtime_error(oss.str());
  }

  // Convert angle to radians
  return A*DEGINRAD;
}

static double parseth(const std::vector<std::string> & words) {
  return parseang(words,4);
}

static double parsephi(const std::vector<std::string> & words) {
  return parseang(words,6);
}

static arma::vec get_coords(const std::vector<atom_t> & atoms, int idx) {
  arma::vec ret(3);
  ret(0)=atoms[idx].x;
  ret(1)=atoms[idx].y;
  ret(2)=atoms[idx].z;
  return ret;
}

static void parse_line(const std::vector<std::string> & words, std::vector<atom_t> & atoms, bool convert) {
  // Helper
  atom_t hlp;

  if(atoms.size()==0) {
    // First atom

    if(words.size()!=1) {
      std::ostringstream oss;
      oss << "Error on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
      throw std::runtime_error(oss.str());
    }

    hlp.el=words[0];
    hlp.num=atoms.size();
    hlp.x=hlp.y=hlp.z=0.0;
    hlp.Q=0;
    atoms.push_back(hlp);

  } else if(atoms.size()==1) {
    // Second atom

    if(words.size() < 3) {
      std::ostringstream oss;
      oss << "Error on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
      throw std::runtime_error(oss.str());
    }

    // Index of reference atom is
    int Rind(parseRind(words,atoms));
    (void) Rind;
    // Distance to atom is
    double R(parseR(words,convert));

    hlp.el=words[0];
    hlp.num=atoms.size();
    hlp.x=0.0;
    hlp.y=0.0;
    hlp.z=R;
    hlp.Q=0;
    atoms.push_back(hlp);

  } else if(atoms.size()==2) {
    // Third atom

    if(words.size() < 5) {
      std::ostringstream oss;
      oss << "Error on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
      throw std::runtime_error(oss.str());
    }

    // Index of reference atom is
    int Rind(parseRind(words,atoms));
    // Distance to atom is
    double R(parseR(words,convert));
    // Index of reference atom is
    int thind(parsethind(words,atoms));
    // Distance to atom is
    double th(parseth(words));

    if(Rind == thind) {
      std::ostringstream oss;
      oss << "Error on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
      oss << "Same reference atom cannot be used twice!\n";
      throw std::runtime_error(oss.str());
    }

    // Coordinates of atom
    arma::vec D(3);
    D(0) = R * sin(th);
    D(1) = 0.0;
    D(2) = (thind==0) ? atoms[1].z - R * cos(th) : R * cos(th);

    // Add atom
    hlp.el=words[0];
    hlp.num=atoms.size();
    hlp.x=D(0);
    hlp.y=D(1);
    hlp.z=D(2);
    hlp.Q=0;
    atoms.push_back(hlp);

  } else {
    // General case

    if(words.size() < 7) {
      std::ostringstream oss;
      oss << "Error on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
      throw std::runtime_error(oss.str());
    }

    // Index of reference atom is
    int Rind(parseRind(words,atoms));
    // Distance to atom is
    double R(parseR(words,convert));
    // Index of reference atom is
    int thind(parsethind(words,atoms));
    // Distance to atom is
    double th(parseth(words));
    // Index of reference atom is
    int phiind(parsephiind(words,atoms));
    // Distance to atom is
    double phi(parsephi(words));

    if(Rind == thind || Rind == phiind || thind == phiind) {
      std::ostringstream oss;
      oss << "Error on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
      oss << "Same reference atom cannot be used twice!\n";
      throw std::runtime_error(oss.str());
    }

    // Reference coordinates
    arma::vec C(get_coords(atoms,Rind));
    arma::vec B(get_coords(atoms,thind));
    arma::vec A(get_coords(atoms,phiind));

    // Reference vectors
    arma::vec x1(arma::normalise(B-A,2));
    arma::vec x2(arma::normalise(C-B,2));

    arma::vec u(x1 - arma::dot(x1,x2)*x2);
    double unorm=arma::norm(u,2);
    if(unorm<1e-5) {
      std::ostringstream oss;
      oss << "Error on Z-Matrix line \"" << concatenate_string(words) << "\"!\n";
      oss << "Problem with dihedral, atoms on a line!\n";
      throw std::runtime_error(oss.str());
    }
    u/=unorm;

    // Helper
    arma::vec v(arma::cross(u,x2));

    // Position
    arma::vec D(C);
    D -= R * (cos(th) * x2 + sin(th) * ( cos(phi) * u - sin(phi) * v ));

    // Add atom
    hlp.el=words[0];
    hlp.num=atoms.size();
    hlp.x=D(0);
    hlp.y=D(1);
    hlp.z=D(2);
    hlp.Q=0;
    atoms.push_back(hlp);
  }
}

std::vector<atom_t> load_zmat(std::string filename, bool convert) {
  // Input file
  std::ifstream in(filename.c_str());
  // Returned array
  std::vector<atom_t> atoms;

  if(in.good()) {
    // Read input lines into a table. First, skip the ZMATRIX identifier line
    readline(in,false);

    // First, the atoms
    std::vector< std::vector<std::string> > zmat;
    while(!in.eof()) {
      std::string line(readline(in,false));
      if(line.size() && !isblank(line))
	zmat.push_back(splitline(line));
      else
	break;
    }

    // Then, the variables
    std::vector<std::string> vars;
    while(!in.eof()) {
      std::string line(readline(in,false));
      if(line.size() && !isblank(line))
	vars.push_back(line);
      else
	break;
    }

    // Search and replace variables
    for(size_t i=0;i<vars.size();i++) {
      // Modify assignment
      for(size_t j=0;j<vars[i].size();j++)
	if(vars[i][j]=='=')
	  vars[i][j]=' ';

      // Split line
      std::vector<std::string> ass(splitline(vars[i]));
      if(ass.size()!=2)
	throw std::runtime_error("Error parsing variable assignments.\n");

      // Search for variable in Z-Matrix
      for(size_t j=1;j<zmat.size();j++) {
	if(stricmp(zmat[j][2],ass[0])==0)
	  zmat[j][2]=ass[1];

	if(zmat[j].size()>=5 && stricmp(zmat[j][4],ass[0])==0)
	  zmat[j][4]=ass[1];

	if(zmat[j].size()>=7 && stricmp(zmat[j][6],ass[0])==0)
	  zmat[j][6]=ass[1];
      }
    }

    // Parse the z-matrix
    for(size_t i=0;i<zmat.size();i++)
      parse_line(zmat[i],atoms,convert);

  } else {
    std::ostringstream oss;
    oss << "Unable to open Z-Matrix file \"" << filename << "\"!\n";
    throw std::runtime_error(oss.str());
  }

  // Get rid of dummy atoms
  for(size_t i=atoms.size()-1;i<atoms.size();i--)
    if(atoms[i].el == "X")
      atoms.erase(atoms.begin()+i);

  return atoms;
}
