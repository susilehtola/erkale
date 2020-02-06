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
#include "linalg.h"
#include "timer.h"
#include "erifit.h"

#include <algorithm>
#include <fstream>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>
// For exceptions
#include <sstream>
#include <stdexcept>


/// Compute overlap of normalized Gaussian primitives
arma::mat overlap(const arma::vec & iexps, const arma::vec & jexps, int am) {
  arma::mat S(iexps.size(),jexps.size());

  switch(am) {
  case(-1):
    {
      for(size_t i=0;i<iexps.n_elem;i++)
	for(size_t j=0;j<jexps.n_elem;j++) {
	  // Sum of exponents
	  double zeta=iexps(i) + jexps(j);
	  // Helpers
	  double eta=4.0*iexps(i)*jexps(j)/(zeta*zeta);
	  double s_eta=sqrt(eta);
	  S(i,j)=sqrt(s_eta);
	}
    }
    break;

  case(0):
    {
      for(size_t i=0;i<iexps.n_elem;i++)
	for(size_t j=0;j<jexps.n_elem;j++) {
	  // Sum of exponents
	  double zeta=iexps(i) + jexps(j);
	  // Helpers
	  double eta=4.0*iexps(i)*jexps(j)/(zeta*zeta);
	  double s_eta=sqrt(eta);
	  S(i,j)=s_eta*sqrt(s_eta);
	}
    }
    break;

  case(1):
    {
      for(size_t i=0;i<iexps.n_elem;i++)
	for(size_t j=0;j<jexps.n_elem;j++) {
	  // Sum of exponents
	  double zeta=iexps(i) + jexps(j);
	  // Helpers
	  double eta=4.0*iexps(i)*jexps(j)/(zeta*zeta);
	  double s_eta=sqrt(eta);
	  S(i,j)=s_eta*s_eta*sqrt(s_eta);
	}
    }
    break;

  case(2):
    {
      for(size_t i=0;i<iexps.n_elem;i++)
	for(size_t j=0;j<jexps.n_elem;j++) {
	  // Sum of exponents
	  double zeta=iexps(i) + jexps(j);
	  // Helpers
	  double eta=4.0*iexps(i)*jexps(j)/(zeta*zeta);
	  double s_eta=sqrt(eta);
	  S(i,j)=s_eta*s_eta*s_eta*sqrt(s_eta);
	}
    }
    break;

  default:
    for(size_t i=0;i<iexps.n_elem;i++)
      for(size_t j=0;j<jexps.n_elem;j++) {
	// Sum of exponents
	double zeta=iexps(i) + jexps(j);
	// Helpers
	double eta=4.0*iexps(i)*jexps(j)/(zeta*zeta);
	double s_eta=sqrt(eta);
	double q_eta=sqrt(s_eta);

	// Compute overlap
	// S(i,j)=pow(eta,am/2.0+0.75)

	// Calls pow(double,int) which should be pretty fast.
	S(i,j)=pow(s_eta,am+1)*q_eta;
      }
  }

  return S;
}


int find_am(char am) {
  for(int i=0;i<=max_am;i++)
    if(shell_types[i]==toupper(am))
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

  for(size_t i=0;i<c.size();i++) {
    if(C[i].z<=0.0) {
      std::ostringstream oss;
      oss << "Negative gaussian exponent " << C[i].z << " in basis set!\n";
      throw std::runtime_error(oss.str());
    }

    if(!std::isnormal(C[i].z)) {
      std::ostringstream oss;
      oss << "Abnormal gaussian exponent " << C[i].z << " in basis set!\n";
      throw std::runtime_error(oss.str());
    }

    if(!std::isnormal(C[i].c)) {
      std::ostringstream oss;
      oss << "Abnormal contraction coefficient " << C[i].c << " in basis set!\n";
      throw std::runtime_error(oss.str());
    }
  }
}

FunctionShell::~FunctionShell() {
}

void FunctionShell::add_exponent(double Cv, double zv) {
  if(zv<=0.0) {
    std::ostringstream oss;
    oss << "Negative gaussian exponent " << zv << " in basis set!\n";
    throw std::runtime_error(oss.str());
  }

  if(!std::isnormal(zv)) {
    std::ostringstream oss;
    oss << "Abnormal gaussian exponent " << zv << " in basis set!\n";
    throw std::runtime_error(oss.str());
  }

  if(!std::isnormal(Cv)) {
    std::ostringstream oss;
    oss << "Abnormal contraction coefficient " << Cv << " in basis set!\n";
    throw std::runtime_error(oss.str());
  }


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
    printf("\t\t% e\t%e\n",C[i].c,C[i].z);
}

bool FunctionShell::operator<(const FunctionShell & rhs) const {
  // First, check if angular momentum is lower.
  if(am!=rhs.am)
    return am<rhs.am;

  // Then, sort by decreasing first exponent
  if(C[0].z != rhs.C[0].z)
    return C[0].z > rhs.C[0].z;

  // Last, sort by decreasing contraction depth
  //  if(C.size() != rhs.C.size())
  return C.size() > rhs.C.size();
}

bool FunctionShell::operator==(const FunctionShell & rhs) const {
  // First, check angular momentum
  if(am!=rhs.am)
    return false;

  if(C.size() != rhs.C.size())
    return false;

  for(size_t i=0;i<C.size();i++)
    if(C[i].z != rhs.C[i].z || C[i].c != rhs.C[i].c)
      return false;

  return true;
}

int FunctionShell::get_am() const {
  return am;
}

size_t FunctionShell::get_Ncontr() const {
  return C.size();
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
  // Check that function doesn't exist yet
  bool found=false;
  for(size_t i=0;i<bf.size();i++)
    if(bf[i]==f)
      found=true;
  if(!found)
    bf.push_back(f);
  else
    fprintf(stderr,"Duplicate %c shell removed in %s basis set\n",shell_types[f.get_am()],symbol.c_str());
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
  // First sort by increasing atom number; then the special basis sets end up at the bottom.
  if(number < rhs.number)
    return true;
  else if(number > rhs.number)
    return false;

  return get_Z(symbol)<get_Z(rhs.symbol);
}

std::vector<FunctionShell> ElementBasisSet::get_shells() const {
  return bf;
}

std::vector<FunctionShell> ElementBasisSet::get_shells(int am) const {
  std::vector<FunctionShell> ret;
  for(size_t i=0;i<bf.size();i++)
    if(bf[i].get_am() == am)
      ret.push_back(bf[i]);

  return ret;
}

int ElementBasisSet::get_Nbf() const {
  int n=0;
  for(size_t i=0;i<bf.size();i++)
    n+=2*bf[i].get_am()+1; // 2l+1 degeneracy
  return n;
}

void ElementBasisSet::get_primitives(arma::vec & expsv, arma::mat & coeffs, int am) const {
  // Count number of exponents and shells that have angular momentum am
  int nsh=0;
  // Helper
  std::vector<double> exps;

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

  // Allocate room for exponents
  expsv.zeros(exps.size());
  for(size_t iexp=0;iexp<exps.size();iexp++)
    expsv(iexp)=exps[iexp];
  // Sort in descending order
  expsv=arma::sort(expsv,"descend");

  // Allocate returned contractions
  coeffs.zeros(exps.size(),nsh);
  if((size_t) nsh > exps.size()) {
    std::ostringstream oss;
    oss << "Basis set has duplicate functions on the " << shell_types[am] << " shell: got " << nsh << " shells but only " << exps.size() << " exponents!\n";
    throw std::runtime_error(oss.str());
  }

  // Collect contraction coefficients. Loop over exponents
  for(size_t iexp=0;iexp<expsv.n_elem;iexp++) {
    int iish=0;
    // Loop over shells
    for(size_t ish=0;ish<bf.size();ish++)
      if(bf[ish].get_am()==am) {

	// Get exponents and contraction on shell
	std::vector<contr_t> shc=bf[ish].get_contr();

	// Find current exponent
	bool found=0;
	for(size_t i=0;i<shc.size();i++)
	  if(shc[i].z==expsv(iexp)) {
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

size_t ElementBasisSet::get_max_Ncontr() const {
  size_t Ncontr=0;
  for(size_t i=0;i<bf.size();i++)
    Ncontr=std::max(Ncontr, bf[i].get_Ncontr());
  return Ncontr;
}

int ElementBasisSet::get_am(size_t ind) const {
  return bf[ind].get_am();
}

void ElementBasisSet::decontract() {
  // Create new basis set
  ElementBasisSet decontr(symbol);
  for(int am=0;am<=get_max_am();am++) {
    // Get contraction style
    arma::vec exps;
    arma::mat coeffs;
    get_primitives(exps,coeffs,am);

    for(size_t iexp=0;iexp<exps.n_elem;iexp++) {
      // Create new shell
      FunctionShell tmp(am);
      tmp.add_exponent(1.0,exps(iexp));
      decontr.add_function(tmp);
    }
  }

  // Sort basis
  decontr.sort();

  // Change to decontracted set
  *this=decontr;
}

struct candidate_t {
  // Exponent
  double z;
  // Angular momentum
  int am;
};

bool operator<(const candidate_t & lhs, const candidate_t & rhs) {
  return lhs.z < rhs.z;
}

ElementBasisSet ElementBasisSet::density_fitting(int lmaxinc, double fsam) const {
  // Get primitives
  std::vector<arma::vec> prims(get_max_am()+1);
  for(size_t iam=0;iam<prims.size();iam++) {
    arma::mat contr;
    get_primitives(prims[iam],contr,iam);
  }

  // Element is
  int Z=get_Z(get_symbol());
  // Atomic am is
  int lval;
  if(Z<3)
    lval=0;
  else if(Z<19)
    lval=1;
  else if(Z<55)
    lval=2;
  else
    lval=3;

  // Generate candidates
  std::vector<candidate_t> cand;
  for(size_t iam=0;iam<prims.size();iam++)
    for(size_t ix=0;ix<prims[iam].n_elem;ix++) {
      // AM of generated function is (step 5)
      int am=std::min(2*iam,(size_t) (lval+lmaxinc));
      // and the exponent is (step 6)
      double z=2*prims[iam](ix);
      // Add to list of candidates (step 4)
      bool found=false;
      for(size_t j=0;j<cand.size();j++)
	if(cand[j].z==z && cand[j].am==am)
	  found=true;
      if(!found) {
	candidate_t c;
	c.z=z;
	c.am=iam;
	cand.push_back(c);
      }
    }

  // Sort candidates in order of exponents
  std::stable_sort(cand.begin(),cand.end());

  // Final set
  std::vector<candidate_t> set;

  // While candidates are remaining
  while(cand.size()) {
    // Generate trial set.
    std::vector<candidate_t> trial;

    // Candidate with largest exponent is moved to trial set (step 7)
    trial.push_back(cand[cand.size()-1]);
    cand.erase(cand.begin()+cand.size()-1);

    // as well as other functions that have a similar exponent (step 8)
    for(size_t i=cand.size()-1;i<cand.size();i--)
      // Since exponents are in decreasing order, ratio is always greater than one..
      if(trial[0].z/cand[i].z <= fsam) {
	trial.push_back(cand[i]);
	cand.erase(cand.begin()+i);
      }

    // Geometric average of exponents is calculated (step 9)
    double z=0.0;
    for(size_t i=0;i<trial.size();i++)
      z+=log(trial[i].z);
    z=std::exp(z/trial.size());

    // and the angular momentum is determined (step 9)
    int maxam=trial[0].am;
    for(size_t i=1;i<trial.size();i++)
      if(trial[i].am>maxam)
	maxam=trial[i].am;
    for(size_t i=0;i<set.size();i++)
      if(set[i].am>maxam)
	maxam=set[i].am;

    // Function is added to final set
    candidate_t f;
    f.z=z;
    f.am=maxam;
    set.push_back(f);
  }

  // Generate final set
  ElementBasisSet ret(*this);
  ret.bf.clear();
  for(size_t i=0;i<set.size();i++)
    for(int am=0;am<=set[i].am;am++) {
      // Contraction is
      std::vector<contr_t> c(1);
      c[0].c=1.0;
      c[0].z=set[i].z;
      FunctionShell sh(am,c);
      ret.add_function(sh);
    }
  // Sort the set
  ret.sort();

  return ret;
}

ElementBasisSet ElementBasisSet::product_set(int lmaxinc, double fsam) const {
  // Get primitives
  std::vector<arma::vec> prims(get_max_am()+1);
  for(size_t iam=0;iam<prims.size();iam++) {
    arma::mat contr;
    get_primitives(prims[iam],contr,iam);
  }

  // Element is
  int Z=get_Z(get_symbol());
  // Atomic am is
  int lval;
  if(Z<3)
    lval=0;
  else if(Z<19)
    lval=1;
  else if(Z<55)
    lval=2;
  else
    lval=3;

  // Candidate exponents
  std::vector< std::vector<double> > cand;

  // Loop over primitive ams
  for(size_t iam=0;iam<prims.size();iam++)
    for(size_t jam=0;jam<prims.size();jam++) {
      // AM of generated function is
      size_t am=std::min(iam+jam,(size_t) (lval+lmaxinc));
      if(am>=cand.size())
	cand.resize(am+1);
      // Add exponents
      for(size_t ix=0;ix<prims[iam].n_elem;ix++)
	for(size_t jx=0;jx<prims[jam].n_elem;jx++) {
	  // Exponent is
	  double zeta=prims[iam](ix)+prims[jam](jx);

	  // Loop over angular momentum
	  for(size_t i=0;i<=am;i++) {
	    // Check if exponent is on the list
	    bool found=false;
	    for(size_t j=0;j<cand[i].size();j++)
	      if(cand[i][j]==zeta)
		found=true;
	    if(!found)
	      cand[i].push_back(zeta);
	  }
	}
    }

  // Sort candidates
  for(size_t iam=0;iam<cand.size();iam++)
    std::stable_sort(cand[iam].begin(),cand[iam].end());

  // Form final exponents by doing geometric averages
  std::vector< std::vector<double> > fitexp(cand.size());
  for(size_t iam=cand.size()-1;iam<cand.size();iam--)
    while(cand[iam].size()) {
      // Trial exponents
      std::vector<double> trexp;
      trexp.push_back(cand[iam][cand[iam].size()-1]);
      cand[iam].erase(cand[iam].begin()+cand[iam].size()-1);

      // Candidates
      for(size_t i=cand[iam].size()-1;i<cand[iam].size();i--)
	// Since exponents are in decreasing order, ratio is always greater than one..
	if(trexp[0]/cand[iam][i] <= fsam) {
	  trexp.push_back(cand[iam][i]);
	  cand[iam].erase(cand[iam].begin()+i);
	}

      // Do geometric average
      double z=0.0;
      for(size_t i=0;i<trexp.size();i++)
	z+=log(trexp[i]);
      z=std::exp(z/trexp.size());

      // Add to set
      fitexp[iam].push_back(z);
    }

  // Sort fit exponents
  for(size_t iam=0;iam<fitexp.size();iam++)
    std::stable_sort(fitexp[iam].begin(),fitexp[iam].end());

  // Generate product set
  ElementBasisSet ret(*this);
  ret.bf.clear();
  for(size_t am=0;am<fitexp.size();am++)
    for(size_t ix=0;ix<fitexp[am].size();ix++) {
      // Contraction is
      std::vector<contr_t> c(1);
      c[0].c=1.0;
      c[0].z=fitexp[am][ix];
      FunctionShell sh(am,c);
      ret.add_function(sh);
    }
  ret.sort();

  return ret;
}

ElementBasisSet ElementBasisSet::cholesky_set(double thr, int maxam, double ovlthr) const {
  ElementBasisSet orbbas(*this);
  orbbas.decontract();

  // Fitting set
  ElementBasisSet fitel(orbbas.get_symbol());

  // Loop over angular momentum
  for(int iam=0;iam<=orbbas.get_max_am();iam++)
    for(int jam=0;jam<=iam;jam++) {
      if(iam+jam>maxam)
	break;

      // Get the T matrix
      arma::mat T;
      arma::vec exps;
      arma::ivec am;
      ERIfit::compute_cholesky_T(orbbas,iam,jam,T,exps);

      // Find out significant exponent pairs by a pivoted Cholesky decomposition of T
      arma::uvec sigexpidx;
      pivoted_cholesky(T,thr,sigexpidx);

      // Significant exponents
      arma::vec sigexp(sigexpidx.size());
      for(size_t ii=0;ii<sigexpidx.size();ii++) {
	sigexp(ii)=exps(sigexpidx[ii]);
      }
      sigexp=arma::sort(sigexp,"descend");

      // Create the fitting set
      for(arma::uword i=0;i<sigexp.n_elem;i++) {
	std::vector<contr_t> c(1);
	c[0].c=1.0;
	c[0].z=sigexp(i);
	fitel.add_function(FunctionShell(iam+jam,c));
      }
    }

  fitel.prune(ovlthr,true);

  return fitel;
}

void ElementBasisSet::get_primitives(arma::vec & zfree, arma::vec & zgen, arma::mat & cgen, int am) const {
  // Get current contraction style
  arma::vec exps;
  arma::mat coeffs;
  get_primitives(exps,coeffs,am);

  // Find indices of free functions
  std::vector<size_t> freex, freec;
  for(size_t iexp=0;iexp<exps.n_elem;iexp++) {
    // List contractions exponents is in
    std::vector<size_t> icontr;
    for(size_t ic=0;ic<coeffs.n_cols;ic++)
      if(coeffs(iexp,ic)!=0.0)
	icontr.push_back(ic);

    // Loop over contractions to check for free functions
    for(size_t ic=0;ic<icontr.size();ic++) {
      // Check if any other exponents appear in the contraction
      arma::vec ch=arma::abs(coeffs.col(icontr[ic]));
      // Set to zero and check norm
      ch(iexp)=0.0;
      if(arma::dot(ch,ch) == 0.0) {
	// Exponent is free.
	freex.push_back(iexp);
	freec.push_back(icontr[ic]);
	continue;
      }
    }
  }
  if(freex.size()>exps.size())
    throw std::runtime_error("Something has gone awry.\n");

  // Collect free functions
  zfree.zeros(freex.size());
  for(size_t i=0;i<freex.size();i++)
    zfree(i)=exps(freex[i]);

  // Collect generally contracted exponents and their coefficients
  zgen.zeros(exps.n_elem-freex.size());
  cgen.zeros(zgen.n_elem,coeffs.n_cols-freec.size());

  size_t ix=0;
  for(size_t iexp=0;iexp<exps.n_elem;iexp++) {
    // Check if exponent is free
    bool free=false;
    for(size_t i=0;i<freex.size();i++)
      if(iexp == freex[i])
	free=true;
    if(free) continue;

    // Store exponent
    zgen(ix)=exps(iexp);

    // Find coefficients for exponent
    size_t ic=0;
    for(size_t icontr=0;icontr<coeffs.n_cols;icontr++) {
      // If contraction free?
      free=false;
      for(size_t i=0;i<freec.size();i++)
	if(icontr == freec[i])
	  free=true;
      if(free) continue;

      cgen(ix,ic++)=coeffs(iexp,icontr);
    }

    // Increment exponent
    ix++;
  }

  /*
  arma::trans(exps).print("All exponents");
  coeffs.print("Contraction scheme");

  arma::trans(zfree).print("Free exponents: ");
  arma::trans(zgen).print("General exponents");
  cgen.print("General contractions");
  */
}

void ElementBasisSet::orthonormalize() {
  // Helper: orthonormalized basis
  ElementBasisSet orthbas(symbol);

  // Loop over am
  for(int am=0;am<=get_max_am();am++) {
    // Get the current contraction pattern
    arma::vec freex, genx;
    arma::mat genc;
    get_primitives(freex,genx,genc,am);

    // Get overlap of primitives
    arma::mat S=overlap(genx,genx,am);

    // Normalize contractions
    for(size_t i=0;i<genc.n_cols;i++)
      genc.col(i)/=sqrt(arma::as_scalar(arma::trans(genc.col(i))*S*genc.col(i)));

    // Compute overlap of contractions
    arma::mat Sovl=arma::trans(genc)*S*genc;
    // Do symmetric orthonormalization
    arma::mat Sinvh=SymmetricOrth(Sovl);
    // and adapt contraction coefficients
    genc=genc*Sinvh;

    // Store functions
    for(size_t ic=0;ic<genc.n_cols;ic++) {
      // Create new shell
      FunctionShell tmp(am);
      for(size_t iexp=0;iexp<genx.n_elem;iexp++) {
	tmp.add_exponent(genc(iexp,ic),genx(iexp));
      }
      orthbas.add_function(tmp);
    }
    // and free primitives
    for(size_t iexp=0;iexp<freex.n_elem;iexp++) {
      FunctionShell tmp(am);
      tmp.add_exponent(1.0,freex(iexp));
      orthbas.add_function(tmp);
    }
  }

  *this=orthbas;
}

double P_innerprod_inout(const arma::vec & ai, const arma::mat & S, const arma::vec & aj, size_t P) {
  return arma::as_scalar( arma::trans(ai.subvec(0,P)) * S.submat(0,0,P,P) * aj.subvec(0,P) );
}

double P_innerprod_outin(const arma::vec & ai, const arma::mat & S, const arma::vec & aj, size_t P) {
  size_t N=ai.n_elem-1;

  return arma::as_scalar( arma::trans(ai.subvec(N-P,N)) * S.submat(N-P,N-P,N,N) * aj.subvec(N-P,N) );
}

size_t count_shared(const arma::vec & ai, const arma::vec & aj) {
  size_t nshared=0;
  for(size_t fi=0;fi<ai.n_elem;fi++)
    if(ai(fi)!=0.0 && aj(fi)!=0.0)
      nshared++;
  return nshared;
}

bool treated_inout(const arma::mat & c, size_t i, size_t j) {
  // Sanity check - is block of tight functions already empty?
  bool empty=true;
  for(size_t fi=0;fi<=i;fi++) {
    if(c(fi,j)!=0.0)
      empty=false;
  }

  return empty;
}

bool treated_outin(const arma::mat & c, size_t i, size_t j) {
  // Sanity check - is block of diffuse functions already empty?
  bool empty=true;
  for(size_t fi=c.n_rows-c.n_cols+i;fi<c.n_rows;fi++) {
    if(c(fi,j)!=0.0)
      empty=false;
  }

  return empty;
}

void ElementBasisSet::P_orthogonalize(double cutoff, double Cortho) {
  // Helper: orthogonalized basis
  ElementBasisSet orthbas(symbol);

  // Always use a nonzero cutoff because of truncation errors
  cutoff=std::max(cutoff,100*DBL_EPSILON);

  // Loop over am
  for(int am=0;am<=get_max_am();am++) {
    // Get the current contraction pattern
    arma::vec freex, genx;
    arma::mat genc;
    get_primitives(freex,genx,genc,am);

    if(genc.n_cols) {
      // Get overlap of primitives
      arma::mat S=overlap(genx,genx,am);

      // Compute overlap of functions. First normalize
      for(size_t i=0;i<genc.n_cols;i++)
	genc.col(i)/=sqrt(arma::as_scalar(arma::trans(genc.col(i))*S*genc.col(i)));

      /*
      arma::mat covl=arma::trans(genc)*S*genc;
      covl.print("Contraction overlap");
      */

      // Intermediate normalization
      for(size_t i=0;i<genc.n_cols;i++) {
	// Find maximum coefficient
	arma::vec hlp=arma::abs(genc.col(i));
	arma::uword ind;
	hlp.max(ind);
	// and normalize it to unity
	genc.col(i)/=genc(ind,i);
      }

      // arma::trans(genx).print("Exponents");
      // genc.print("Contraction scheme");

      // Inside-out purification
      for(size_t i=0;i<genc.n_cols-1;i++) {
	// P values
	arma::uvec Pval(genc.n_cols-1-i);
	size_t iP=0;

	for(size_t j=i+1;j<genc.n_cols;j++) {
	  // Initially set P = M
	  size_t P=genx.n_rows-1;
	  //	  printf("in-out P: i = %3i, j = %3i\n",(int) i, (int) j);

	  // Sanity check - is block of tight functions already empty?
	  if(treated_inout(genc,i,j)) {
	    continue;
	  }
	  // Sanity check - do the functions even share any exponents?
	  if(!count_shared(genc.col(i),genc.col(j))) {
	    Pval(iP++)=P;
	    continue;
	  }

	  while(true) {
	    // eqn (7): inner products
	    double aii=P_innerprod_inout(genc.col(i),S,genc.col(i),P);
	    double aij=P_innerprod_inout(genc.col(i),S,genc.col(j),P);
	    double ajj=P_innerprod_inout(genc.col(j),S,genc.col(j),P);

	    // Compute linear dependency
	    double Op=fabs(aij)/sqrt(aii*ajj);
	    //	    printf("\tP = %3i, Op = %e\n",(int) P, Op);

	    if(1-Op <= Cortho)
	      // Reached sufficient value of P
	      break;

	    else if(P==0) {
	      // arma::trans(genc.col(i)).print("ai");
	      // arma::trans(genc.col(j)).print("aj");
	      //throw std::runtime_error("Error in P-orthogonalization in-out routine.\n");
	      fprintf(stderr,"Warning - problem in Corth for %s\n",symbol.c_str());
	      break;

	    } else {
	      // Decrement P
	      P--;
	    }
	  }

	  // Store P
	  Pval(iP++)=P;
	}

	// Orthogonalize
	size_t P=Pval.min();
	for(size_t j=i+1;j<genc.n_cols;j++) {
	  // Sanity check - do the functions even share any exponents?
	  if(!count_shared(genc.col(i),genc.col(j))) {
	    continue;
	  }
	  // Sanity check - is block of tight functions already empty?
	  if(treated_inout(genc,i,j)) {
	    continue;
	  }

	  // eqn (7): inner products
	  double aii=P_innerprod_inout(genc.col(i),S,genc.col(i),P);
	  double aij=P_innerprod_inout(genc.col(i),S,genc.col(j),P);

	  // Factor to use in elimination, eqn (8)
	  double xP=aij/aii;

	  // Orthogonalize: eqn (4)
	  genc.col(j) -= xP*genc.col(i);
	}
      }

      // genc.print("Inside-out purified");

      // Outside-in purification
      for(size_t i=genc.n_cols-1;i>0;i--) {

	// P values
	arma::uvec Pval(i);
	size_t iP=0;

	for(size_t j=i-1;j<genc.n_cols;j--) {
	  // Initially set P = M
	  size_t P=genx.n_rows-1;

	  // Sanity check - do the functions even share any exponents?
	  if(!count_shared(genc.col(i),genc.col(j))) {
	    Pval(iP++)=P;
	    continue;
	  }
	  // Sanity check - is block of diffuse functions already empty?
	  if(treated_outin(genc,i,j)) {
	    continue;
	  }

	  while(true) {
	    // eqn (7): inner products
	    double aii=P_innerprod_outin(genc.col(i),S,genc.col(i),P);
	    double aij=P_innerprod_outin(genc.col(i),S,genc.col(j),P);
	    double ajj=P_innerprod_outin(genc.col(j),S,genc.col(j),P);

	    // Compute linear dependency
	    double Op=fabs(aij)/sqrt(aii*ajj);
	    //printf("\tP = %3i, Op = %e\n",(int) P, Op);

	    if(1-Op <= Cortho) {
	      // Reached sufficient value of P
	      break;

	    } else if(P==0) {
	      // arma::trans(genc.col(i)).print("ai");
	      // arma::trans(genc.col(j)).print("aj");
	      throw std::runtime_error("Error in P-orthogonalization out-in routine.\n");

	    } else {
	      // Decrement P
	      P--;
	    }
	  }

	  // Store value
	  Pval(iP++)=P;
	}

	// Orthogonalize
	size_t P=Pval.min();
	for(size_t j=i-1;j<genc.n_cols;j--) {
	  // Sanity check - do the functions even share any exponents?
	  if(!count_shared(genc.col(i),genc.col(j))) {
	    continue;
	  }
	  // Sanity check - is block of diffuse functions already empty?
	  if(treated_outin(genc,i,j)) {
	    continue;
	  }

	  // eqn (7): inner products
	  double aii=P_innerprod_outin(genc.col(i),S,genc.col(i),P);
	  double aij=P_innerprod_outin(genc.col(i),S,genc.col(j),P);

	  // Factor to use in elimination, eqn (8)
	  double xP=aij/aii;

	  // Orthogonalize: eqn (4)
	  genc.col(j) -= xP*genc.col(i);
	}
      }

      // Intermediate normalization
      for(size_t i=0;i<genc.n_cols;i++) {
	// Find maximum coefficient
	arma::vec hlp=arma::abs(genc.col(i));
	arma::uword ind;
	hlp.max(ind);
	// and normalize it to unity
	genc.col(i)/=genc(ind,i);
      }

      //      genc.print("Refined scheme");
    }

    // Add contracted functions
    for(size_t ic=0;ic<genc.n_cols;ic++) {
      // Create new shell
      FunctionShell tmp(am);
      for(size_t iexp=0;iexp<genx.n_elem;iexp++) {
	if(fabs(genc(iexp,ic))>0.0 && fabs(genc(iexp,ic))>=cutoff)
	  tmp.add_exponent(genc(iexp,ic),genx(iexp));
      }
      orthbas.add_function(tmp);
    }

    // and free primitives
    for(size_t iexp=0;iexp<freex.n_elem;iexp++) {
      FunctionShell tmp(am);
      tmp.add_exponent(1.0,freex(iexp));
      orthbas.add_function(tmp);
    }
  }

  // Switch to orthogonalized basis
  *this=orthbas;
}

void ElementBasisSet::augment_steep(int naug) {
  // Loop over am
  for(int am=0;am<=get_max_am();am++) {
    // Get the current contraction pattern
    arma::vec exps;
    arma::mat coeffs;
    get_primitives(exps,coeffs,am);

    // If only one exponent, no augmentation
    if(exps.size()<2)
      continue;

    // Compute the new exponents
    double es=exps(0);
    double el=exps(1);
    for(int i=0;i<naug;i++) {
      double aug=es*pow(es/el,i+1);

      // Add the exponent
      FunctionShell sh(am);
      sh.add_exponent(1.0,aug);
      add_function(sh);
    }
  }

  sort();
}

void ElementBasisSet::augment_diffuse(int naug) {
  // Loop over am
  for(int am=0;am<=get_max_am();am++) {
    // Get the current contraction pattern
    arma::vec exps;
    arma::mat coeffs;
    get_primitives(exps,coeffs,am);

    // If only one exponent, no augmentation
    if(exps.size()<2)
      continue;

    // Compute the new exponents
    double ed=exps(exps.n_elem-1);
    double el=exps(exps.n_elem-2);
    for(int i=0;i<naug;i++) {
      double aug=el/pow(el/ed,i+2);

      // Add the exponent
      FunctionShell sh(am);
      sh.add_exponent(1.0,aug);
      add_function(sh);
    }
  }

  sort();
}

void ElementBasisSet::prune(double cutoff, bool coulomb) {
  // Pruned exponents
  std::vector< std::vector<double> > pruned(get_max_am()+1);

  for(int am=0;am<=get_max_am();am++) {
    // Get exponents
    arma::vec exps;
    arma::mat contr;
    get_primitives(exps,contr,am);
    if(!exps.n_elem)
      continue;

    // Compute overlap matrix
    int Sam = coulomb ? am-1 : am;
    arma::mat S=overlap(exps,exps,Sam);

    // Prune the exponents
    size_t ioff=0;
    while(ioff<S.n_cols) {
      // Determine how many exponents are similar
      size_t joff(ioff+1);
      while(joff < S.n_cols && S(ioff,joff)>cutoff)
	joff++;

      // Compute the geometric average of the current block
      double ave=exp(arma::mean(arma::log(exps.subvec(ioff,joff-1))));

      // Add it to the pruned set
      pruned[am].push_back(ave);

      // Increment offset
      ioff=joff;
    }
  }

  // Replace shells
  bf.clear();
  for(size_t am=0;am<pruned.size();am++)
    for(size_t ix=0;ix<pruned[am].size();ix++) {
      FunctionShell sh(am);
      sh.add_exponent(1.0,pruned[am][ix]);
      add_function(sh);
    }
}

void ElementBasisSet::merge(double cutoff, bool verbose, bool coulomb) {
  // Pruned exponents
  std::vector<arma::vec> exps(get_max_am()+1);
  for(int am=0;am<=get_max_am();am++) {
    // Get exponents
    arma::mat contr;
    get_primitives(exps[am],contr,am);
    if(!exps[am].n_elem)
      continue;

    // Get contractions
    arma::vec zfree, zgen;
    arma::mat cgen;
    get_primitives(zfree,zgen,cgen,am);

    // Original exponents
    const arma::vec E0(exps[am]);

    // Prune the exponents
    while(true) {
      // Compute overlap matrix
      int Sam = coulomb ? am-1 : am;
      arma::mat S=overlap(exps[am],exps[am],Sam);
      // Remove diagonal part
      S-=arma::eye(S.n_rows,S.n_cols);

      // Find maximum element
      arma::uword irow, icol;
      double Smax=S.max(irow,icol);

      // Break loop?
      if(Smax<cutoff)
	break;

      // Too large overlap. Check for originality of exponents
      bool roworig=false, colorig=false;
      arma::uword rowix=0, colix=0;
      for(size_t ix=0;ix<E0.n_elem;ix++) {
	if(E0(ix) == exps[am](irow)) {
	  roworig=true;
	  rowix=ix;
	}
	if(E0(ix) == exps[am](icol)) {
	  colorig=true;
	  colix=ix;
	}
      }

      bool dropped=false;

      // Check if one was contracted
      if(roworig && colorig) {
	bool rowcontr=false, colcontr=false;
	for(size_t ix=0;ix<zgen.n_elem;ix++) {
	  if(zgen(ix)==E0(rowix))
	    rowcontr=true;
	  if(zgen(ix)==E0(colix))
	    colcontr=true;
	}

	if(rowcontr && !colcontr) {
	  // Drop col idx
	  if(verbose) {
	    printf("%-2s: %c exponents %e and %e with overlap %e, dropped free primitive %e.\n",get_symbol().c_str(),shell_types[am],exps[am](irow),exps[am](icol),Smax,exps[am](icol));
	    fflush(stdout);
	  }

	  std::vector<double> merged=arma::conv_to< std::vector<double> >::from(exps[am]);
	  merged.erase(merged.begin()+icol);
	  exps[am]=arma::conv_to<arma::vec>::from(merged);
	  dropped=true;
	} else if(!rowcontr && colcontr) {
	  // Drop row idx
	  if(verbose) {
	    printf("%-2s: %c exponents %e and %e with overlap %e, dropped free primitive %e.\n",get_symbol().c_str(),shell_types[am],exps[am](irow),exps[am](icol),Smax,exps[am](irow));
	    fflush(stdout);
	  }

	  std::vector<double> merged=arma::conv_to< std::vector<double> >::from(exps[am]);
	  merged.erase(merged.begin()+irow);
	  exps[am]=arma::conv_to<arma::vec>::from(merged);
	  dropped=true;
	}

      } else if(roworig && !colorig) {
	if(verbose) {
	  printf("%-2s: %c exponents %e and %e with overlap %e, dropped merged primitive %e.\n",get_symbol().c_str(),shell_types[am],exps[am](irow),exps[am](icol),Smax,exps[am](icol));
	  fflush(stdout);
	}

	std::vector<double> merged=arma::conv_to< std::vector<double> >::from(exps[am]);
	merged.erase(merged.begin()+icol);
	exps[am]=arma::conv_to<arma::vec>::from(merged);
	dropped=true;

      } else if(!roworig && colorig) {
	// Drop row idx
	if(verbose) {
	  printf("%-2s: %c exponents %e and %e with overlap %e, dropped merged primitive %e.\n",get_symbol().c_str(),shell_types[am],exps[am](irow),exps[am](icol),Smax,exps[am](irow));
	  fflush(stdout);
	}

	std::vector<double> merged=arma::conv_to< std::vector<double> >::from(exps[am]);
	merged.erase(merged.begin()+irow);
	exps[am]=arma::conv_to<arma::vec>::from(merged);
	dropped=true;
      }

      if(!dropped) {
	// Merge exponents
	std::vector<double> merged=arma::conv_to< std::vector<double> >::from(exps[am]);
	merged[irow]=sqrt(exps[am](irow)*exps[am](icol));
	merged[icol]=merged[irow];
	if(verbose) {
	  printf("%-2s: merged %c exponents %e and %e with overlap %e to %e.\n",get_symbol().c_str(),shell_types[am],exps[am](irow),exps[am](icol),Smax,merged[icol]);
	  fflush(stdout);
	}

	// Remove second value
	merged.erase(merged.begin()+irow);
	exps[am]=arma::conv_to<arma::vec>::from(merged);
      }
    }
  }

  // Replace shells
  bf.clear();
  for(size_t am=0;am<exps.size();am++)
    for(size_t ix=0;ix<exps[am].n_elem;ix++) {
      FunctionShell sh(am);
      sh.add_exponent(1.0,exps[am](ix));
      add_function(sh);
    }
}

BasisSetLibrary::BasisSetLibrary() {
}

BasisSetLibrary::~BasisSetLibrary() {
}

static std::string pople_hydrogen_to_heavy(const std::string & in) {
  std::string out(in);
  for(size_t i=0;i<out.size();i++) {
    if(out[i]=='p')
      out[i]='d';
    else if(out[i]=='d')
      out[i]='f';
  }
  return out;
}

static std::string pople_heavy_to_hydrogen(const std::string & in) {
  std::string out(in);
  for(size_t i=0;i<out.size();i++) {
    if(out[i]=='d')
      out[i]='p';
    else if(out[i]=='f')
      out[i]='d';
  }
  return out;
}

static BasisSetLibrary combine_pople_basis(const BasisSetLibrary & hbas, const BasisSetLibrary & heavybas) {
  BasisSetLibrary ret;
  std::vector<ElementBasisSet> els;

  // Hydrogen
  els=hbas.get_elements();
  for(size_t i=0;i<els.size();i++)
    if(els[i].get_symbol().compare("H")==0 || els[i].get_symbol().compare("h")==0)
      ret.add_element(els[i]);

  // Heavy atoms
  els=heavybas.get_elements();
  for(size_t i=0;i<els.size();i++)
    if(!(els[i].get_symbol().compare("H")==0 || els[i].get_symbol().compare("h")==0))
      ret.add_element(els[i]);

  return ret;
}

void BasisSetLibrary::load_basis(const std::string & basis0, bool verbose) {
  std::string basis(basis0);

  if(basis.size()>4 && basis.substr(0,4).compare("6-31")==0) {
    // First, check if there is a * or ** polarization part and
    // replace it with (d) or (d,p)
    size_t spos(basis.find_first_of('*'));
    if(spos!=std::string::npos) {
      std::string polpart;
      if(basis.size()>spos+1 && basis[spos+1]=='*') {
	polpart="(d,p)";
      } else
	polpart="(d)";

      std::string newbasis(basis.substr(0,spos)+polpart);
      basis=newbasis;
    }

    // Check the polarization part
    size_t ppos=basis.find_first_of("(");
    size_t pepos=basis.find_first_of(")");
    if(ppos!=std::string::npos) {
      if(pepos==std::string::npos)
	throw std::logic_error("Error parsing Pople style basis set speficication \"" + basis + "\".\n");

      // A polarization has been specified. Original basis
      std::string obas(basis.substr(0,ppos));
      // Polarization
      std::string ppart(basis.substr(ppos+1,pepos-ppos-1));

      // Is there a comma?
      size_t cpos=ppart.find_first_of(",");
      if(cpos!=std::string::npos) {
	// Heavy atoms basis
	std::string apart(ppart.substr(0,cpos));
	BasisSetLibrary heavybas;
	heavybas.load_gaussian94(obas+"("+apart+","+pople_heavy_to_hydrogen(apart)+")");
	// Hydrogen basis
	std::string hpart(ppart.substr(cpos+1));
	BasisSetLibrary hbas;
	hbas.load_gaussian94(obas+"("+pople_hydrogen_to_heavy(hpart)+","+hpart+")");
	*this=combine_pople_basis(hbas,heavybas);
      } else {
	// Heavy atoms basis
	BasisSetLibrary heavybas;
	heavybas.load_gaussian94(obas+"("+ppart+","+pople_heavy_to_hydrogen(ppart)+")");
	// Hydrogen basis: original
	BasisSetLibrary hbas;
	hbas.load_gaussian94(obas,verbose);
	*this=combine_pople_basis(hbas,heavybas);
      }

    } else
      load_gaussian94(basis,verbose);

  } else
    load_gaussian94(basis,verbose);
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
	// Check element type
	sym=element_symbols[get_Z(sym)];

	// and the atom number the basis is for is
	size_t num=readint(line_split[1]);

	// Check that there is no duplicate entry
	int dupl=0, numfound=0;
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
	    if(words.size()!=2 && words.size()!=3) {
	      std::ostringstream oss;
	      oss << "Error parsing input line \"" << line << "\".\nExpected a shell type and amount of functions.\n";
	      throw std::runtime_error(oss.str());
	    }

	    // The shell type is
	    std::string shelltype=words[0];
	    // The amount of exponents is
	    int nc=readint(words[1]);

	    if(strcmp(shelltype,"SP")==0) {
	      // SP shell
	      FunctionShell S(0), P(1);

	      // Read the exponents
	      for(int i=0;i<nc;i++) {
		line=readline(in);
		// Numbers
		std::vector<std::string> nums=splitline(line);
		if(nums.size()!=3) {
                  std::ostringstream oss;
                  oss << "Invalid specification \"" << line << "\" for SP shell!\n";
		  throw std::runtime_error(oss.str());
		}
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
		if(nums.size()!=2) {
                  std::ostringstream oss;
                  oss << "Invalid specification \"" << line << "\" for shell!\n";
		  throw std::runtime_error(oss.str());
                }
		// Add functions
		sh.add_exponent(readdouble(nums[1]),readdouble(nums[0]));
	      }
	      el.add_function(sh);
	    } else {
	      // AM given with L=%i

	      if(shelltype.size()<3) {
		std::ostringstream oss;
		oss << "Unrecognized shell type \"" << shelltype << "\"!\n";
		throw std::runtime_error(oss.str());
	      }

	      // Check beginning
	      if(stricmp(shelltype.substr(0,2),"L=")!=0) {
		std::ostringstream oss;
		oss << "Could not parse shell type: \"" << shelltype << "\"!\n";
		throw std::runtime_error(oss.str());
	      }

	      // Now get the shell type
	      int am=readint(shelltype.substr(2));
	      if(am<0 || am>=max_am) {
		std::ostringstream oss;
		oss << "Invalid value " << am << "for shell angular momentum!\n";
		throw std::runtime_error(oss.str());
	      }

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

void BasisSetLibrary::save_gaussian94(const std::string & filename, bool append) const {
  FILE *out;
  if(append)
    out=fopen(filename.c_str(),"a");
  else
    out=fopen(filename.c_str(),"w");
  if(!out) {
    std::ostringstream oss;
    oss << "Error opening basis set output file \"" << filename << "\".\n";
    throw std::runtime_error(oss.str());
  }

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

void BasisSetLibrary::save_cfour(const std::string & filename, const std::string & basname, bool newformat, bool append) const {
  FILE *out;
  if(append) {
    out=fopen(filename.c_str(),"a");
    if(!out) {
      std::ostringstream oss;
      oss << "Error opening basis set output file \"" << filename << "\".\n";
      throw std::runtime_error(oss.str());
    }
  } else {
    out=fopen(filename.c_str(),"w");

    if(!out) {
      std::ostringstream oss;
      oss << "Error opening basis set output file \"" << filename << "\".\n";
      throw std::runtime_error(oss.str());
    }
  }

  // Group free exponents in shells of nfsh exponents
  int nfrsh=3;

  // Loop over elements
  for(size_t iel=0;iel<elements.size();iel++) {
    // Get element
    ElementBasisSet el=elements[iel];

    // Exponents and contractions
    std::vector<arma::vec> exps(el.get_max_am()+1);
    std::vector<arma::vec> frexps(el.get_max_am()+1);
    std::vector<arma::mat> coeffs(el.get_max_am()+1);
    for(int am=0;am<=el.get_max_am();am++)
      el.get_primitives(frexps[am],exps[am],coeffs[am],am);

    // Count non-trivial shells
    int nsh=0;
    for(int am=0;am<=el.get_max_am();am++) {
      if(exps[am].n_elem)
	nsh++;

      // Amount of free shells
      size_t nfr=frexps[am].n_elem/nfrsh;
      if(frexps[am].n_elem%nfrsh)
	nfr++;

      nsh+=nfr;
    }

    // Element and name of the basis set
    fprintf(out,"%s:%s\n",toupper(el.get_symbol()).c_str(),basname.c_str());

    // Comment line
    Timer t;
    std::ostringstream oss;
    oss << "Generated by ERKALE on " << t.current_time() << ".";
    fprintf(out,"%-80s\n",oss.str().c_str());

    // Blank line
    fprintf(out,"\n");

    // Number of shells in the basis set
    fprintf(out,"%3i\n",nsh);

    // Did we just change the line?
    bool cl;
    // Amount of printed entries
    size_t np;
    // Change line every n entries
    const size_t nint=10;
    const size_t ndbl=5;

    // Angular momentum for each shell
    np=0;
    cl=true;
    for(int am=0;am<=el.get_max_am();am++) {
      // Contracted exponents
      if(exps[am].n_elem) {
	fprintf(out,"%5i",am);
	np++;
	cl=false;

	if(np%nint==0) {
	  fprintf(out,"\n");
	  np=0;
	  cl=true;
	}
      }

      // Free exponent shells
      size_t nfr=frexps[am].n_elem/nfrsh;
      if(frexps[am].n_elem%nfrsh)
	nfr++;
      for(size_t i=0;i<nfr;i++) {
	fprintf(out,"%5i",am);
	np++;
	cl=false;

	if(np%nint==0) {
	  fprintf(out,"\n");
	  np=0;
	  cl=true;
	}
      }
    }
    if(!cl)
      fprintf(out,"\n");

    // Number of contracted basis functions for each shell
    np=0;
    cl=true;
    for(int am=0;am<=el.get_max_am();am++) {
      // Contracted exponents
      if(exps[am].n_elem) {
	fprintf(out,"%5i",(int) coeffs[am].n_cols);

	np++;
	cl=false;

	if(np%nint==0) {
	  fprintf(out,"\n");
	  np=0;
	  cl=true;
	}
      }

      // Free exponents
      size_t ifr=0;
      while(ifr<frexps[am].n_elem) {
	// Upper limit
	size_t ufr=std::min(ifr+nfrsh,(size_t) frexps[am].n_elem);
	fprintf(out,"%5i",(int) (ufr-ifr));
	np++;
	cl=false;

	if(np%nint==0) {
	  fprintf(out,"\n");
	  np=0;
	  cl=true;
	}

	// Switch limits
	ifr=ufr;
      }
    }
    if(!cl)
      fprintf(out,"\n");

    // Number of primitive basis functions for each shell
    np=0;
    cl=true;
    for(int am=0;am<=el.get_max_am();am++) {
      // Contracted exponents
      if(exps[am].n_elem) {
	fprintf(out,"%5i",(int) coeffs[am].n_rows);

	np++;
	cl=false;

	if(np%nint==0) {
	  fprintf(out,"\n");
	  np=0;
	  cl=true;
	}
      }

      // Free exponents
      size_t ifr=0;
      while(ifr<frexps[am].n_elem) {
	// Upper limit
	size_t ufr=std::min(ifr+nfrsh,(size_t) frexps[am].n_elem);
	fprintf(out,"%5i",(int) (ufr-ifr));
	np++;
	cl=false;

	if(np%nint==0) {
	  fprintf(out,"\n");
	  np=0;
	  cl=true;
	}

	// Switch limits
	ifr=ufr;
      }
    }
    if(!cl)
      fprintf(out,"\n");

    // Blank line
    fprintf(out,"\n");

    // Loop over shells
    np=0;
    cl=true;
    for(int am=0;am<=el.get_max_am();am++) {
      if(exps[am].n_elem) {
	// Print exponents
	cl=false;
	np=0;
	for(size_t ix=0;ix<exps[am].n_elem;ix++) {
	  if(newformat)
	    fprintf(out," % .10e",exps[am](ix));
	  else
	    fprintf(out," %14.7f",exps[am](ix));
	  cl=false;
	  np++;

	  if(np%ndbl==0) {
	    fprintf(out,"\n");
	    np=0;
	    cl=true;
	  }
	}
	if(!cl)
	  fprintf(out,"\n");

	// Blank line
	fprintf(out,"\n");

	// Coefficients - contractions in columns
	for(size_t ix=0;ix<coeffs[am].n_rows;ix++) {
	  for(size_t ic=0;ic<coeffs[am].n_cols;ic++) {
	    if(newformat)
	      fprintf(out," % .10e",coeffs[am](ix,ic));
	    else
	      fprintf(out," % 10.7f",coeffs[am](ix,ic));
	  }
	  fprintf(out,"\n");
	}

	// Blank line
	fprintf(out,"\n");
      }


      // Free exponents
      size_t ifr=0;
      while(ifr<frexps[am].n_elem) {
	// Upper limit
	size_t ufr=std::min(ifr+nfrsh,(size_t) frexps[am].n_elem);

	// Print exponents
	cl=false;
	np=0;
	for(size_t ix=ifr;ix<ufr;ix++) {
	  if(newformat)
	    fprintf(out," % .10e",frexps[am](ix));
	  else
	    fprintf(out," %14.7f",frexps[am](ix));
	  cl=false;
	  np++;

	  if(np%ndbl==0) {
	    fprintf(out,"\n");
	    np=0;
	    cl=true;
	  }
	}
	if(!cl)
	  fprintf(out,"\n");

	// Blank line
	fprintf(out,"\n");

	// Coefficients - contractions in columns
	for(size_t ix=ifr;ix<ufr;ix++) {
	  for(size_t ic=ifr;ic<ufr;ic++) {
	    if(newformat) {
	      if(ix==ic)
		fprintf(out," % .10e",1.0);
	      else
		fprintf(out," % .10e",0.0);
	    } else {
	      if(ix==ic)
		fprintf(out," % 10.7f",1.0);
	      else
		fprintf(out," % 10.7f",0.0);
	    }
	  }
	  fprintf(out,"\n");
	}

	// Blank line
	fprintf(out,"\n");

	// Switch limits
	ifr=ufr;
      }
    }
  }

  fclose(out);
}

void BasisSetLibrary::save_dalton(const std::string & filename, bool append) const {
  FILE *out;
  if(append) {
    out=fopen(filename.c_str(),"a");
    if(!out) {
      std::ostringstream oss;
      oss << "Error opening basis set output file \"" << filename << "\".\n";
      throw std::runtime_error(oss.str());
    }
  } else {
    out=fopen(filename.c_str(),"w");

    if(!out) {
      std::ostringstream oss;
      oss << "Error opening basis set output file \"" << filename << "\".\n";
      throw std::runtime_error(oss.str());
    }

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
      arma::vec exps;
      arma::mat coeffs;
      el.get_primitives(exps,coeffs,l);

      // Print label
      fprintf(out,"$ %s\n",toupper(element_names[get_Z(el.get_symbol())]).c_str());
      fprintf(out,"$ %c-TYPE FUNCTIONS\n",toupper(shell_types[l]));
      // Print element, number of exponents and contracted functions
      fprintf(out,"%4i %4i %4i\n",(int) exps.size(),(int) coeffs.n_cols,0);

      // Loop over exponents
      for(size_t iexp=0;iexp<exps.n_elem;iexp++) {
	// Print exponent
	fprintf(out,"% 18.8f",exps(iexp));
	// and contraction scheme
	int np=1; // amount of printed entries
	for(size_t ic=0;ic<coeffs.n_cols;ic++) {
	  if(np==0)
	    // Sync with exponent style
	    fprintf(out,"% 18.8f",coeffs(iexp,ic));
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

void BasisSetLibrary::save_molpro(const std::string & filename, bool append) const {
  FILE *out;
  if(append) {
    out=fopen(filename.c_str(),"a");
    if(!out) {
      std::ostringstream oss;
      oss << "Error opening basis set output file \"" << filename << "\".\n";
      throw std::runtime_error(oss.str());
    }
  } else {
    out=fopen(filename.c_str(),"w");

    if(!out) {
      std::ostringstream oss;
      oss << "Error opening basis set output file \"" << filename << "\".\n";
      throw std::runtime_error(oss.str());
    }
  }

  // Loop over elements
  for(size_t iel=0;iel<elements.size();iel++) {
    // Get element
    ElementBasisSet el=elements[iel];
    // Loop over angular momentum
    for(int am=0;am<=el.get_max_am();am++) {
      // Get exponents and contraction coefficients
      arma::vec exps;
      arma::mat coeffs;
      el.get_primitives(exps,coeffs,am);

      // Print am and element
      fprintf(out,"%c,%s",shell_types[am],el.get_symbol().c_str());
      // Print primitives
      for(size_t iexp=0;iexp<exps.n_elem;iexp++) {
	fprintf(out,",%.10e",exps(iexp));
      }
      fprintf(out,";\n");

      // Print contractions
      for(size_t ic=0;ic<coeffs.n_cols;ic++) {
	// Coefficient vector
	arma::rowvec cx=coeffs.col(ic);
	// Find first and last contracted exponent
	size_t ifirst=0;
	while(cx(ifirst)==0)
	  ifirst++;
	size_t ilast=cx.n_elem-1;
	while(cx(ilast)==0.0)
	  ilast--;

	fprintf(out,"c,%i,%i",(int) ifirst+1,(int) ilast+1);

	// Coefficients
	for(size_t ix=ifirst;ix<=ilast;ix++)
	  fprintf(out,",%.10e",coeffs(ix,ic));
	fprintf(out,";\n");
      }
    }
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

size_t BasisSetLibrary::get_max_Ncontr() const {
  size_t Ncontr=0;
  for(size_t i=0;i<elements.size();i++)
    Ncontr=std::max(Ncontr, elements[i].get_max_Ncontr());
  return Ncontr;
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
      if((elements[i].get_number()==number) && (stricmp(elements[i].get_symbol(),el)==0) )
	return elements[i];
  } else {
    // Special basis requested.
    for(size_t i=0;i<elements.size();i++)
      if(elements[i].get_number()==number) {
	// Check that this is actually of the wanted type!
	if(stricmp(elements[i].get_symbol(),el)==0)
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

BasisSetLibrary BasisSetLibrary::density_fitting(int lvalinc, double fsam) const {
  BasisSetLibrary ret(*this);
  ret.name="Density fitting "+name;
  for(size_t iel=0;iel<elements.size();iel++)
    ret.elements[iel]=elements[iel].density_fitting(lvalinc,fsam);
  return ret;
}

BasisSetLibrary BasisSetLibrary::product_set(int lvalinc, double fsam) const {
  BasisSetLibrary ret(*this);
  ret.name="Product set "+name;
  for(size_t iel=0;iel<elements.size();iel++)
    ret.elements[iel]=elements[iel].product_set(lvalinc,fsam);
  return ret;
}

BasisSetLibrary BasisSetLibrary::cholesky_set(double thr, int maxam, double ovlthr) const {
  BasisSetLibrary ret(*this);
  ret.name="Product set "+name;
  for(size_t iel=0;iel<elements.size();iel++)
    ret.elements[iel]=elements[iel].cholesky_set(thr,maxam,ovlthr);
  return ret;
}

void BasisSetLibrary::orthonormalize() {
  for(size_t iel=0;iel<elements.size();iel++)
    elements[iel].orthonormalize();
}

void BasisSetLibrary::P_orthogonalize(double Cortho, double cutoff) {
  for(size_t iel=0;iel<elements.size();iel++)
    elements[iel].P_orthogonalize(Cortho, cutoff);
}

void BasisSetLibrary::augment_steep(int naug){
  char tmp[80];
  sprintf(tmp," with %i augmentation functions",naug);
  name=name+tmp;
  for(size_t iel=0;iel<elements.size();iel++)
    elements[iel].augment_steep(naug);
}

void BasisSetLibrary::augment_diffuse(int naug){
  char tmp[80];
  sprintf(tmp," with %i augmentation functions",naug);
  name=name+tmp;
  for(size_t iel=0;iel<elements.size();iel++)
    elements[iel].augment_diffuse(naug);
}

void BasisSetLibrary::merge(double cutoff, bool verbose) {
  for(size_t iel=0;iel<elements.size();iel++)
    elements[iel].merge(cutoff,verbose);
}
