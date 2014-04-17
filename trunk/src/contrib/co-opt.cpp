/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

/**
 * This file contains routines for the automatical formation of
 * application-specific basis sets.
 *
 * The algorithm performs completeness-optimization that was introduced in
 *
 * P. Manninen and J. Vaara, "Systematic Gaussian Basis-Set Limit
 * Using Completeness-Optimized Primitive Sets. A Case for Magnetic
 * Properties", J. Comp. Chem. 27, 434 (2006).
 *
 *
 * The automatic algorithm for generating primitive basis sets was
 * introduced in
 *
 * J. Lehtola, P. Manninen, M. Hakala, and K. Hämäläinen,
 * "Completeness-optimized basis sets: Application to ground-state
 * electron momentum densities", J. Chem. Phys. 137, 104105 (2012).
 *
 * and it was generalized to contracted basis sets in
 *
 * S. Lehtola, P. Manninen, M. Hakala, and K. Hämäläinen, "Contraction
 * of completeness-optimized basis sets: Application to ground-state
 * electron momentum densities", J. Chem. Phys. 138, 044109 (2013).
 */

#include "co-opt.h"
#include "elements.h"

int maxam(const std::vector<coprof_t> & cpl) {
  for(size_t i=cpl.size()-1;i<cpl.size();i--)
    if(cpl[i].exps.size()>0)
      return (int) i;
  
  // Dummy statement
  return -1;
}
  
int get_nfuncs(const std::vector<coprof_t> & cpl) {
  int n=0;
  for(int am=0;am<=maxam(cpl);am++)
    n+=(2*am+1)*cpl[am].exps.size();
  return n;
}
  
void print_limits(const std::vector<coprof_t> & cpl, const char *msg) { 
  if(msg)
    printf("%s\n",msg);
  for(int am=0;am<=maxam(cpl);am++)
    printf("%c % .3f % .3f %e %2i\n",shell_types[am],cpl[am].start,cpl[am].end,cpl[am].tol,(int) cpl[am].exps.size());
  printf("Totalling %i functions.\n\n",get_nfuncs(cpl));
  fflush(stdout);
}

void print_scheme(const BasisSetLibrary & baslib, int len) {
  // Get contraction scheme
  ElementBasisSet el=baslib.get_elements()[0];

  // Number of exponents and contractions
  std::vector<int> nc, nx;

  for(int am=0;am<=el.get_max_am();am++) {
    arma::vec exps;
    arma::mat coeffs;
    el.get_primitives(exps,coeffs,am);
    nx.push_back(coeffs.n_rows);
    nc.push_back(coeffs.n_cols);
  }

  // Is the set contracted?
  bool contr=false;
  for(size_t i=0;i<nx.size();i++)
    if(nx[i]!=nc[i])
      contr=true;

  std::string out;
  char tmp[180];

  if(contr) {
    out="[";
    for(int l=0;l<=el.get_max_am();l++) {
      sprintf(tmp,"%i%c",nx[l],tolower(shell_types[l]));
      out+=std::string(tmp);
    }
    out+="|";
    for(int l=0;l<=el.get_max_am();l++)
      if(nc[l]!=nx[l]) {
	sprintf(tmp,"%i%c",nc[l],tolower(shell_types[l]));
	out+=std::string(tmp);
      }
    out+="]";
  } else {
    out="(";
    for(int l=0;l<=el.get_max_am();l++) {
      sprintf(tmp,"%i%c",nx[l],tolower(shell_types[l]));
      out+=std::string(tmp);
    }
    out+=")";
  }

  if(len==0)
    printf("%s",out.c_str());
  else {
    // Format specifier
    sprintf(tmp,"%%-%is",len);
    printf(tmp,out.c_str());
  }
}

arma::vec maxwidth_exps_table(int am, double tol, size_t nexp, double *width, int n) {
    
  // Optimized exponents
  static std::vector<co_exps_t> opt(max_am+1);
    
  // Check if we already have the exponents in store
  if(opt[am].tol!=tol || opt[am].exps.size()!=nexp) {
    opt[am].tol=tol;
    opt[am].exps=maxwidth_exps(am,tol,nexp,&opt[am].w,n);
  }
    
  *width=opt[am].w;

  if(opt[am].exps.size() != nexp) {
    std::ostringstream oss;
    oss << "Required " << nexp << " completeness-optimized primitives but got " << opt[am].exps.size() << "!\n";
    throw std::runtime_error(oss.str());
  }

  return opt[am].exps;
}

arma::vec move_exps(const arma::vec & exps, double x) {
  return exps*std::pow(10.0,x);
}

ElementBasisSet get_element_library(const std::string & el, const std::vector<coprof_t> & cpl) {
  ElementBasisSet elbas(el);
  for(size_t am=0;am<cpl.size();am++)
    for(size_t ish=0;ish<cpl[am].exps.size();ish++) {
      FunctionShell sh(am);
      sh.add_exponent(1.0,cpl[am].exps[ish]);
      elbas.add_function(sh);
    }
  elbas.sort();

  return elbas;
}

std::vector<coprof_t> augment_basis(const std::vector<coprof_t> & cplo, int ndiffuse, int ntight) {
  // New profile
  std::vector<coprof_t> cpl(cplo);
  if(ndiffuse==0 && ntight==0)
    // Nothing to do
    return cpl;

  for(int am=0;am<=maxam(cpl);am++) {
    // Require at least two exponents per augmented shell.
    if(cpl[am].exps.size()<2)
      continue;

    // Current width
    double width=cpl[am].end-cpl[am].start;

    // New width and exponents
    double w;
    arma::vec exps=maxwidth_exps_table(am,cpl[am].tol,cpl[am].exps.size()+ndiffuse+ntight,&w,OPTMOMIND);
    
    // Additional width is
    double dw=w-width;

    // Update
    cpl[am].start -= ndiffuse*dw/(ndiffuse+ntight);
    cpl[am].end   += ntight*dw/(ndiffuse+ntight);
    cpl[am].exps=move_exps(exps,cpl[am].start);
  }

  return cpl;
}


BasisSetLibrary get_library(const std::vector<coprof_t> & cpl) {
  // Construct the basis set library
  BasisSetLibrary baslib;
  for(int Z=1;Z<=maxZ;Z++)
    baslib.add_element(get_element_library(element_symbols[Z],cpl));

  return baslib;
}

