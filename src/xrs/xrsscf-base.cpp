/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */


#include <cfloat>
#include "xrsscf.h"
#include "../diis.h"
#include "../broyden.h"
#include "../dftfuncs.h"
#include "../dftgrid.h"
#include "../linalg.h"
#include "../mathf.h"
#include "../stringutil.h"
#include "../solidharmonics.h"
#include "../elements.h"
#include "../timer.h"
#include "../settings.h"
#include "../trrh.h"
#include "../lmgrid.h"

extern Settings settings;

XRSSCF::XRSSCF(const BasisSet & basis, Checkpoint & chkpt, bool sp) : SCF(basis,chkpt) {
  spin=sp;

  // Get number of alpha and beta electrons
  get_Nel_alpha_beta(basis.Ztot()-settings.get_int("Charge"),settings.get_int("Multiplicity"),nocca,noccb);
}

XRSSCF::~XRSSCF() {
}

void XRSSCF::set_core(const arma::vec & c) {
  coreorb=c;
}

arma::vec XRSSCF::get_core() const {
  return coreorb;
}

/// Get excited atom from atomlist
size_t get_excited_atom_idx(std::vector<atom_t> & at) {
  // Indices of atoms with frozen core
  size_t ind=0;
  int found=0;

  for(size_t i=0;i<at.size();i++) {
    if(at[i].el.size()>3 && at[i].el.substr(at[i].el.size()-3,3)=="-Xc") {
      // Excited atom.
      at[i].el=at[i].el.substr(0,at[i].el.size()-3);
      ind=i;
      found++;
    }
  }

  if(found==0) {
    throw std::runtime_error("Need an atom to excite for XRS calculation!\n");
    return 0;
  } else if(found==1) {
    // Found a single excited atom.
    return(ind);
  } else {
    // Many atoms excited
    throw std::runtime_error("Error - cannot excite many atoms!\n");
    return 0;
  }
}

/// Find excited core orbital
size_t find_excited_orb(const BasisSet & basis, const arma::vec & xco, const arma::mat & C, int nocc) {
  // Overlap matrix
  arma::mat S(basis.overlap());

  // Determine overlap with current orbitals
  arma::rowvec ovl=arma::abs(arma::trans(xco)*S*C.cols(0,nocc-1));
  // Convert to probabilities by squaring the amplitudes
  ovl=arma::pow(ovl,2);

  //  print_mat(ovl," % .3f");

  // Determine maximal overlap
  arma::uword maxind;
  ovl.max(maxind);

  // Return index of orbital
  return maxind;
}

/// Normal Aufbau occupation
std::vector<double> norm_occ(size_t nocc) {
  std::vector<double> ret(nocc);
  for(size_t i=0;i<nocc;i++) {
    ret[i]=1.0;
  }

  return ret;
}

/// Set fractional occupation on excited orbital
std::vector<double> tp_occ(size_t excited, size_t nocc) {
  std::vector<double> ret(nocc);
  for(size_t i=0;i<nocc;i++) {
    ret[i]=1.0;
  }
  ret[excited]=0.5;

  return ret;
}

/// First excited state; core orbital is not occupied
std::vector<double> xch_occ(size_t excited, size_t nocc) {
  std::vector<double> ret(nocc+1);
  for(size_t i=0;i<nocc+1;i++) {
    ret[i]=1.0;
  }
  ret[excited]=0;

  return ret;
}

std::vector<double> fch_occ(size_t excited, size_t nocc) {
  std::vector<double> ret(nocc);
  for(size_t i=0;i<nocc;i++) {
    ret[i]=1.0;
  }
  ret[excited]=0;

  return ret;
}

std::vector<size_t> atom_list(const BasisSet & basis, size_t xcatom, bool verbose) {
  // Localize on all the atoms of the same type than the excited atom
  std::vector<ovl_sort_t> locind;
  for(size_t i=0;i<basis.get_Nnuc();i++)
    if(!basis.get_nucleus(i).bsse && stricmp(basis.get_symbol(i),basis.get_symbol(xcatom))==0) {
      ovl_sort_t tmp;
      tmp.idx=i;
      tmp.S=norm(basis.get_nuclear_coords(i)-basis.get_nuclear_coords(xcatom));
      locind.push_back(tmp);
    }
  // Sort in increasing distance
  std::stable_sort(locind.begin(),locind.end());
  std::reverse(locind.begin(),locind.end());

  std::vector<size_t> list(locind.size());
  for(size_t i=0;i<locind.size();i++)
    list[i]=locind[i].idx;

  if(verbose) {
    printf("\nDistances of atoms from the center\n");
    for(size_t i=0;i<locind.size();i++)
      printf("%i\t%e\n",(int) locind[i].idx+1,locind[i].S);
  }

  return list;
}

size_t localize(const BasisSet & basis, int nocc, size_t xcatom, arma::mat & C, const std::string & state, int iorb) {
  // Check orthonormality
  arma::mat S=basis.overlap();
  check_orth(C,S,false);

  // Decrypt state index
  int nxc;
  {
    char tmp[2];
    tmp[0]=state[0];
    tmp[1]='\0';
    nxc=readint(tmp);
  }
  int lxc=find_am(state[1]);

  if(iorb<0 || iorb>=(2*lxc+1))
    throw std::runtime_error("Invalid number of initial orbital.\n");

  // Charge of atom is
  int Z=basis.get_nucleus(xcatom).Z;
  // Pad to next noble atom to account for multiplicity and state
  for(size_t i=0;i<sizeof(magicno)/sizeof(magicno[0])-1;i++) {
    if(Z==magicno[i])
      break;
    else if(Z>magicno[i] && Z<=magicno[i+1]) {
      Z=magicno[i+1];
      break;
    }
  }

  // Determine amount of orbitals to localize on atom.
  int nloc=0;
  for(size_t i=0;i<sizeof(shell_order)/sizeof(shell_order[0]);i++) {
    // Still electrons left.
    int l=shell_order[i];
    int nsh=2*l+1; // Degeneracy

    // Account for spin
    if(2*(nloc+nsh) < Z)
      nloc+=nsh;
    else
      break;
  }

  // Sanity check
  if(nloc>nocc)
    nloc=nocc;

  // The atom is located at
  coords_t cen=basis.get_nuclear_coords(xcatom);

  // Compute moment integrals around the nucleus
  std::vector<arma::mat> momstack=basis.moment(2,cen.x,cen.y,cen.z);
  // Get matrix which transforms into non-localized block of occupied MO basis
  arma::mat locblock=C.cols(0,nocc-1);

  // Sum together to get x^2 + y^2 + z^2
  arma::mat rsqmat_AO=momstack[getind(2,0,0)]+momstack[getind(0,2,0)]+momstack[getind(0,0,2)];
  // and transform into the occupied MO basis
  arma::mat rsqmat(arma::trans(locblock)*rsqmat_AO*locblock);

  // Diagonalize rsq_mo
  arma::vec reig;
  arma::mat rvec;
  eig_sym_ordered(reig,rvec,rsqmat);

  // and rotate orbitals to the new basis
  C.cols(0,nocc-1)=locblock*rvec;

  // Orbitals to consider
  arma::mat Cc(C.cols(0,nloc-1));

  // Run lm decomposition of orbitals
  const real_expansion_t orbexp(expand_orbitals_real(Cc,basis,cen,false));
  const arma::mat dec=weight_decomposition(orbexp,false); // Don't include total norm

  // Orbital angular momenta
  arma::uvec lval(nloc);
  for(int io=0;io<nloc;io++) {
    arma::rowvec orbl=dec.row(io);
    orbl.max(lval(io));

    // Sanity check
    if(dec(io,lval(io))<0.7)
      printf("Warning - %c orbital %i has small norm %e\n",shell_types[lval(io)],(int) io+1,dec(io,lval(io)));
  }
  //  lval.subvec(0,nloc-1).t().print("Angular momentum");

  // Orbital indices
  arma::ivec orbidx;
  {
    int lmax=4;
    orbidx.zeros(lmax);
    for(int l=0;l<lmax;l++)
      orbidx(l)=l+1;
  }

  // Index of wanted state
  int ixc=0;

  printf("\nLocalizing %i orbitals on center %i.\n",nloc,(int) xcatom+1);

  printf("\t%2s %1s  %8s %7s\n","sh","i","occ","R [Å]");
  // Loop over orbitals
  int iloc=0;
  while(iloc<nloc) {
    // Angular momentum is
    int am=lval(iloc);

    // Degeneracy is
    double deg=2*am+1;
    // Calculate amount of orbitals
    int norb;
    for(norb=0;norb<deg;norb++)
      if(lval(iloc+norb)!=lval(iloc)) {
	break;
      }

    // Store index of initial orbital?
    if(am == lxc && orbidx(am) == nxc) {
      ixc=iloc+iorb;
      for(int i=0;i<norb;i++) {
	if(i==iorb)
	  printf("\t%i%c %1i* %8.6f %6.3f\n",(int) orbidx(am),tolower(shell_types[am]),(int) i+1,dec(iloc+i,lval(iloc+i)),reig(iloc+i)/ANGSTROMINBOHR);
	else
	  printf("\t%i%c %1i  %8.6f %6.3f\n",(int) orbidx(am),tolower(shell_types[am]),(int) i+1,dec(iloc+i,lval(iloc+i)),reig(iloc+i)/ANGSTROMINBOHR);
      }
    } else {
      for(int i=0;i<norb;i++)
	printf("\t%i%c %1i  %8.6f %6.3f\n",(int) orbidx(am),tolower(shell_types[am]),(int) i+1,dec(iloc+i,lval(iloc+i)),reig(iloc+i)/ANGSTROMINBOHR);
    }

    // Increment shell count
    orbidx(am)++;

    // Increment localization count
    iloc+=norb;
  }

  // Check orthonormality
  check_orth(C,S,false);

  printf("\n");
  fflush(stdout);

  // Return index of localized orbital
  return ixc;
}
