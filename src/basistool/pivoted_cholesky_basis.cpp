/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2019
 * Copyright (c) 2010-2019, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include <armadillo>

#include "../basislibrary.h"
#include "../basis.h"
#include "../linalg.h"
#include "../timer.h"
#include "../settings.h"

std::vector<size_t> shell_pivoted_cholesky(const arma::mat & S, double eps, const arma::uvec & shellidx) {
  if(shellidx.n_elem != S.n_rows)
    throw std::logic_error("Shell index does not correspond to basis!\n");

  // Error vector
  arma::vec d(arma::diagvec(S));
  // Remaining error
  double error(arma::max(d));

  // Cholesky matrix
  arma::mat B;
  // Allocate memory
  B.zeros(S.n_rows,S.n_rows);

  // Initialize pivot with Gershgorin theorem: off-diagonal S
  arma::mat odS(arma::abs(S));
  odS.diag().zeros();
  arma::vec odSs(arma::sum(odS).t());
  // Pivot index: start with functions with small off-diagonal
  arma::uvec pi=arma::sort_index(odSs,"ascend");

  // Loop index
  size_t m(0);

  // Shell pivot indices
  std::vector<size_t> shpivot;

  Timer t;

  while(error>eps && m<d.n_elem) {
    // Update the pivot index
    {
      // Remaining pivot is
      arma::uvec pileft(pi.subvec(m,d.n_elem-1));
      // Remaining errors in pivoted order
      arma::vec errs(d(pileft));
      // Sort the remaining errors so that largest one is first
      arma::uvec idx=arma::stable_sort_index(errs,"descend");
      // Store updated pivot
      pi.subvec(m,d.n_elem-1)=pileft(idx);
    }

    // Pivot index to use is
    size_t pim=pi(m);

    // Shell corresponding to pivot is
    arma::uword pivotshell(shellidx[pim]);
    shpivot.push_back(pivotshell);
    // Indices of functions on the shell
    arma::uvec bfs(arma::find(shellidx == pivotshell));

    // Get the overlaps for the pivot
    arma::mat A(S.cols(bfs));

    size_t nb=0;
    while(true) {
      // Did we already treat everything in the block?
      if(nb==A.n_cols)
	break;
      // Remaining pivot is
      arma::uvec pileft(pi.subvec(m,d.n_elem-1));
      // Remaining errors in pivoted order
      arma::vec errs(d(pileft));
      // and the largest error within the current block
      size_t blockind=0;
      double blockerr=0.0;
      size_t Aind=0;
      for(size_t ii=0;ii<bfs.n_elem;ii++) {
        // Function index is
        size_t ind=bfs(ii);
        if(d(ind)>blockerr) {
          // Check that the index is not in the old pivots
          bool found=false;
          for(size_t i=0;i<m;i++)
            if(pi(i)==ind)
              found=true;
          if(!found) {
            Aind=ii;
            blockind=ind;
            blockerr=d(ind);
          }
        }
      }

      // Increment amount of vectors in the block
      nb++;

      // Switch the pivot
      if(pi(m)!=blockind) {
	bool found=false;
	for(size_t i=m+1;i<pi.n_elem;i++)
	  if(pi(i)==blockind) {
	    found=true;
	    std::swap(pi(i),pi(m));
	    break;
	  }
	if(!found) {
	  pi.t().print("Pivot");
	  fflush(stdout);
	  std::ostringstream oss;
	  oss << "Pivot index " << blockind << " not found, m = " << m << " !\n";
	  throw std::logic_error(oss.str());
	}
      }

      pim=pi(m);

      // Compute diagonal element
      B(m,pim)=sqrt(d(pim));

      // Off-diagonal elements
      if(m==0) {
	// No B contribution here; avoid if clause in for loop
        for(size_t i=m+1;i<d.n_elem;i++) {
	  size_t pii=pi(i);
	  // Compute element
	  B(m,pii)=A(pii,Aind)/B(m,pim);
	  // Update d
	  d(pii)-=B(m,pii)*B(m,pii);
	}
      } else {
	for(size_t i=m+1;i<d.n_elem;i++) {
	  size_t pii=pi(i);
	  // Compute element
	  B(m,pii)=(A(pii,Aind) - arma::as_scalar(arma::trans(B.submat(0,pim,m-1,pim))*B.submat(0,pii,m-1,pii)))/B(m,pim);
	  // Update d
	  d(pii)-=B(m,pii)*B(m,pii);
	}
      }

      // Update error
      error=(m+1<=pi.n_elem-1) ? arma::max(d(pi.subvec(m+1,pi.n_elem-1))) : 0.0;
      // Increase m
      m++;
    }
    t.set();
  }

  // Return shell pivot
  return shpivot;
}

extern Settings settings;

arma::mat canonical_orth(const arma::vec & e, const arma::mat & v) {
  arma::uvec oidx(arma::find(e>=settings.get_double("LinDepThresh")));
  return v.cols(oidx)*arma::diagmat(arma::pow(e(oidx),-0.5));
}

BasisSetLibrary pivoted_cholesky_basis(const std::vector<atom_t> & atoms, const BasisSetLibrary & orig, double thresh) {
  // Construct a specific basis set for each element in the system
  std::vector<ElementBasisSet> fullels(atoms.size());
  std::vector<ElementBasisSet> redels(atoms.size());
  // Full list of shells
  std::vector<FunctionShell> shells;
  std::vector<size_t> atmap;
  for(size_t i=0;i<atoms.size();i++) {
    std::string el(atoms[i].el);
    if(el.size()>3 && el.substr(el.size()-3,3)=="-Bq") {
      el=el.substr(0,el.size()-3);
    }
    // Get the full element library
    fullels[i]=orig.get_element(el);
    fullels[i].set_number(i+1);
    // and initialize the reduced one
    redels[i]=ElementBasisSet(el,i+1);
    // Add shells to full list
    std::vector<FunctionShell> shs(fullels[i].get_shells());
    shells.insert(shells.end(),shs.begin(),shs.end());
    // Save the shell-to-atom mapping
    for(size_t ish=0;ish<shs.size();ish++)
      atmap.push_back(i);
  }

  // Full, element-specific basis library
  BasisSetLibrary fulllib;
  for(size_t i=0;i<atoms.size();i++)
    fulllib.add_element(fullels[i]);

  // Construct molecular basis set
  BasisSet basis;
  construct_basis(basis,atoms,fulllib);
  // and get the overlap matrix
  arma::mat S(basis.overlap());

  arma::vec oe;
  arma::mat oc;
  arma::eig_sym(oe,oc,S);
  arma::mat oX(canonical_orth(oe,oc));
  printf("Original basis has %i shells and %i functions, smallest overlap eigenvalue % e yielding %i linearly independent\n",(int) basis.get_shells().size(),(int) basis.get_Nbf(),arma::min(oe),(int) oX.n_cols);
  printf("Reciprocal condition number of original basis is %e\n",arma::rcond(S));

  // Perform pivoted Cholesky decomposition
  std::vector<size_t> shidx=shell_pivoted_cholesky(S,thresh,basis.shell_indices());

  // Add the retained shells to the reduced basis
  for(size_t i=0;i<shidx.size();i++) {
    // Shell index is
    size_t ish=shidx[i];
    // Atom index is
    size_t iat=atmap[ish];
    // Add shell to atom
    redels[iat].add_function(shells[ish]);
  }

  // Reduced element-specific basis library
  BasisSetLibrary redlib;
  for(size_t i=0;i<atoms.size();i++)
    redlib.add_element(redels[i]);
  redlib.sort();

  // Construct reduced molecular basis set
  BasisSet redbasis;
  construct_basis(redbasis,atoms,redlib);

  arma::mat redS(redbasis.overlap());
  arma::vec re;
  arma::mat rc;
  arma::eig_sym(re,rc,redS);
  arma::mat rX(canonical_orth(re,rc));
  printf("Reduced basis has %i shells and %i functions, smallest overlap eigenvalue % e yielding %i linearly independent\n",(int) redbasis.get_shells().size(),(int) redbasis.get_Nbf(),arma::min(re),(int) rX.n_cols);
  printf("Reciprocal condition number of reduced basis is %e\n",arma::rcond(redS));

  return redlib;
}
