/*
 * This file is written by Arto Sakko and Jussi Lehtola, 2011.
 * Copyright (c) 2011, Arto Sakko and Jussi Lehtola
 *
 *
 *
 *                   This file is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Erkale is written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "casida.h"
#include "casida_grid.h"
#include "stringutil.h"
#include "timer.h"

#include <cstdio>
#include <cstdlib>
#include <cfloat>

Casida::Casida(const Settings & set, const BasisSet & basis, const arma::vec & Ev, const arma::mat & Cv, const arma::mat & Pv) {
  E.push_back(Ev);
  C.push_back(Cv);
  P.push_back(Pv);

  // Parse parameters and form K
  parse_args(set, basis, Cv.n_cols);
  printf("Casida calculation has %u pairs.\n",(unsigned int) pairs[0].size());
  fprintf(stderr,"Casida calculation has %u pairs.\n",(unsigned int) pairs[0].size());

  // Calculate K matrix
  calc_K(set,basis);
}

Casida::Casida(const Settings & set, const BasisSet & basis, const arma::vec & Ea, const arma::vec & Eb, const arma::mat & Ca, const arma::mat & Cb, const arma::mat & Pa, const arma::mat & Pb) {

  E.push_back(Ea);
  E.push_back(Eb);
  C.push_back(Ca);
  C.push_back(Cb);
  P.push_back(Pa);
  P.push_back(Pb);

  // Parse parameters and form K
  parse_args(set, basis, Ca.n_cols);
  printf("Casida calculation has %u spin up and %u spin down pairs.\n",(unsigned int) pairs[0].size(),(unsigned int) pairs[1].size());
  fprintf(stderr,"Casida calculation has %u spin up and %u spin down pairs.\n",(unsigned int) pairs[0].size(),(unsigned int) pairs[1].size());

  // Calculate K matrix
  calc_K(set,basis);
}

void Casida::parse_args(const Settings & set, const BasisSet & basis, size_t Norbs) {
  // Form pairs and occupations
  form_pairs(set,basis,Norbs,C.size()==2); // polarized calculation?
  // Form dipole matrix
  calc_dipole(basis);

  // Determine coupling
  switch(set.get_int("CasidaCoupling")) {
  case(0):
    // IPA.
    coupling=IPA;
    break;
    
  case(1):
    // RPA.
    coupling=RPA;
    break;

  case(2):
    // LDAXC, add xc contribution.
    coupling=TDLDA;
    break;

  default:
    throw std::runtime_error("Unknown coupling!\n");
  }
}

void Casida::calc_K(const Settings & set, const BasisSet & basis) {
  // Exchange and correlation functionals
  int x_func=set.get_int("CasidaX");
  int c_func=set.get_int("CasidaC");
  double tol=set.get_double("CasidaTol");

  // Do we need to form K?
  if(coupling!=IPA) {
    
    // Allocate memory
    K.resize(pairs.size());
    for(size_t is=0;is<pairs.size();is++)
      K[is].zeros(pairs[is].size(),pairs[is].size());
    
    // Compute Coulomb coupling
    Kcoul(basis);
    
    // Compute XC coupling if necessary
    if(coupling==TDLDA)
      Kxc(basis,tol,x_func,c_func);
  }
}

Casida::~Casida() {
};

void Casida::calc_dipole(const BasisSet & bas) {
  // Dipole matrix elements in AO basis
  std::vector<arma::mat> dm=bas.moment(1);
  
  dipmat.resize(C.size());
  for(size_t ispin=0;ispin<C.size();ispin++) {
    dipmat[ispin].resize(3);
    // Loop over cartesian directions
    for(int ic=0;ic<3;ic++)
      // Compute dipole matrix in MO basis
      dipmat[ispin][ic]=arma::trans(C[ispin])*dm[ic]*C[ispin];
  }
}


void Casida::form_pairs(const Settings & set, const BasisSet & bas, size_t Norb, bool pol) {
  if(pol) {
    // Polarized calculation. Get number of alpha and beta electrons.
    int Nel_alpha, Nel_beta;
    get_Nel_alpha_beta(bas.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),Nel_alpha,Nel_beta);
    
    // Amount of occupied and virtual states is
    nocc.push_back(Nel_alpha);
    nvirt.push_back(Norb-nocc[0]);

    // Amount of occupied and virtual states is
    nocc.push_back(Nel_beta);
    nvirt.push_back(Norb-nocc[1]);

    // Store occupation numbers
    f.push_back(arma::zeros(Norb));
    f[0].subvec(0,Nel_alpha-1)=arma::ones(Nel_alpha);
    
    f.push_back(arma::zeros(Norb));
    f[1].subvec(0,Nel_beta-1)=arma::ones(Nel_beta);
    
  } else {
    // Amount of occupied states is
    nocc.push_back((bas.Ztot()-set.get_int("Charge"))/2);
    // Amount of virtual states is
    nvirt.push_back(Norb-nocc[0]);

    // Store occupation numbers
    f.push_back(arma::zeros(Norb));
    f[0].subvec(0,nocc[0]-1)=2.0*arma::ones(nocc[0]);
  }

  // Resize pairs
  pairs.resize(nocc.size());

  // What orbitals are included in the calculation?
  std::vector<std::string> states=splitline(set.get_string("CasidaStates"));
  if(states.size()!=nocc.size()) {
    // Include all pairs in the calculation.
    for(size_t ispin=0;ispin<nocc.size();ispin++) {
      for(size_t iocc=0;iocc<nocc[ispin];iocc++)
	for(size_t ivirt=0;ivirt<nvirt[ispin];ivirt++) {
	  states_pair_t tmp;
	  tmp.i=iocc;
	  tmp.f=nocc[ispin]+ivirt;
	  pairs[ispin].push_back(tmp);
	}
    }
  } else {
    // Loop over spins
    for(size_t ispin=0;ispin<nocc.size();ispin++) {

      // Indices of orbitals to include.
      std::vector<size_t> idx=parse_range(states[ispin]);

      // Check that we don't run over states
      if(idx[idx.size()-1]>Norb) {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Orbital " << idx[idx.size()-1] << " was requested in calculation, but only " << Norb << " orbitals exist!\n";
	throw std::runtime_error(oss.str());
      }

      // Convert to C++ indexing
      for(size_t i=0;i<idx.size();i++)
	idx[i]--;	

      // Loop over indices
      for(size_t iiocc=0;iiocc<idx.size();iiocc++) {
	// Index of orbital is
	size_t iocc=idx[iiocc];
	// Check that it truly is occupied.
	if(iocc>=nocc[ispin])
	  continue;

	for(size_t jjvirt=iiocc+1;jjvirt<idx.size();jjvirt++) {
	  // Index of virtual is
	  size_t jvirt=idx[jjvirt];
	  // Check that it truly is virtual.
	  if(jvirt<nocc[ispin])
	    continue;

	  // Create state pair
	  states_pair_t tmp;
          tmp.i=iocc;
          tmp.f=jvirt;

          pairs[ispin].push_back(tmp);
        }
      }
    }
  }
}

double Casida::esq(states_pair_t ip, bool ispin) const {
  double dE=E[ispin](ip.f)-E[ispin](ip.i);
  return dE*dE;
}

double Casida::fe(states_pair_t ip, bool ispin) const {
  return sqrt((f[ispin](ip.i)-f[ispin](ip.f))*(E[ispin](ip.f)-E[ispin](ip.i)));
}


void Casida::solve() {
  Timer t;

  w_i.resize(pairs.size());
  F_i.resize(pairs.size());

  for(size_t ispin=0;ispin<pairs.size();ispin++) {
    // Generate the coupling matrix (eqn 2.11)
    arma::mat Omega(pairs[ispin].size(),pairs[ispin].size());
    Omega.zeros();

    for(size_t ip=0;ip<pairs[ispin].size();ip++) {
      if(coupling!=IPA) {
	// Plug in K terms
	
	double term;
	// Do off-diagonal first
	for(size_t jp=0;jp<ip;jp++) {
	  term=2.0*fe(pairs[ispin][ip],ispin)*K[ispin](ip,jp)*fe(pairs[ispin][jp],ispin);
	  Omega(ip,jp)+=term;
	  Omega(jp,ip)+=term;
	}
	// Plug in diagonal
	Omega(ip,ip)+=2.0*fe(pairs[ispin][ip],ispin)*K[ispin](ip,ip)*fe(pairs[ispin][ip],ispin);
      }
      
      // Add IPA contribution to diagonal
      Omega(ip,ip)+=esq(pairs[ispin][ip],ispin);
    }

    // Solve eigenvalues and eigenvectors using direct linear algebraic methods
    eig_sym_ordered(w_i[ispin], F_i[ispin], Omega);

    // The eigenvalues are the squares of the excitation energies
    for(size_t i=0;i<w_i[ispin].n_elem;i++)
      w_i[ispin](i) = sqrt(w_i[ispin](i));
  }

  printf("Casida equations solved in %s.\n",t.elapsed().c_str());
  fprintf(stderr,"Solution %s.\n",t.elapsed().c_str());
}

// This calculates the photoabsorption transition rates
void Casida::absorption() const {
  for(size_t ispin=0;ispin<C.size();ispin++) {
    
    printf("\n ******* Casida Photoabsorption Spectrum, spin %i ********\n",(int) ispin);
    
    // Transition rates for every transition
    arma::mat tr(pairs[ispin].size(),3);
    tr.zeros();
    
    // Loop over transitions
    for(size_t it=0;it<pairs[ispin].size();it++)
      // Loop over cartesian coordinates
      for(size_t ic=0;ic<3;ic++) {
	
	// Loop over coupled transitions
	for(size_t jt=0;jt<pairs[ispin].size();jt++)
	  // Compute |x| = x^T S^{-1/2} F_i
	  tr(it,ic)+=dipmat[ispin][ic](pairs[ispin][jt].i,pairs[ispin][jt].f)*F_i[ispin](jt,it)/fe(pairs[ispin][jt],ispin);
	
	// Normalize to get \lf$ \left\langle \Psi_0 \left| \hat{x}
	// \right| \right\rangle \lf$ , see Eq. 4.40 of Casida (1994),
	// or compare Eqs. 2.14 and 2.16 in Jamorski et al (1996).
	tr(it,ic)/=sqrt(w_i[ispin](it));
      }
    
    // Oscillator strengths, 2/3 * E * ( |x|^2 + |y|^2 + |z|^2 )
    arma::vec osc(pairs[ispin].size());
    for(size_t it=0; it<pairs[ispin].size();it++)
      osc(it) = 2.0/3.0 * w_i[ispin](it) * arma::dot(tr.row(it),tr.row(it));
    
    // Write output
    printf(  " Photoabsorption transition energies and rates\n");
    printf(  " %6s   %12s   %12s   %12s %12s %12s\n", "nn", "E [eV]", "osc.str.", "<x>", "<y>", "<z>");
    for(size_t it=0; it<pairs[ispin].size(); it++) {
      printf(" %6i    %12.6f   %12.6f   %12.6f %12.6f %12.6f\n", (int) it+1, w_i[ispin](it)*HARTREEINEV, osc(it), tr(it, 0), tr(it, 1), tr(it, 2));
    }

    char fname[80];
    sprintf(fname,"casida%i.dat",(int) ispin);
    FILE *out=fopen(fname,"w");
    for(size_t it=0; it<pairs[ispin].size(); it++)
      fprintf(out,"%e %e % e % e % e\n",w_i[ispin](it)*HARTREEINEV, osc(it), tr(it, 0), tr(it, 1), tr(it, 2));
    fclose(out);
  }
}

void Casida::Kcoul(const BasisSet & basis) {

  Timer t;
  
  // Form density fitting basis
  BasisSet dfitbas=basis.density_fitting();
  DensityFit dfit;
  // Compute all integrals in memory.
  dfit.fill(basis,dfitbas,0);
  
  const size_t Nbf=basis.get_Nbf();
  const size_t Naux=dfitbas.get_Nbf();

  if(!C.size())
    throw std::runtime_error("Error - no orbitals!\n");
  const size_t Norb=C[0].n_cols;
  
  // The [\mu \nu|I] matrix in Jamorski (4.16).
  arma::mat munu_I(Norb*Norb,Naux);
  // Work memory
  arma::mat tmp(Nbf*Norb,Naux);
  // Inverse Coulomb overlap matrix of fitting basis
  arma::mat ab_inv=dfit.get_ab_inv();

  for(size_t ispin=0;ispin<C.size();ispin++) {
    // We need to calculate the integrals in the MO basis 
    // (which is different for alpha and beta electrons)
    
    // First transform integrals wrt nu.
    tmp.zeros();
    for(size_t imu=0;imu<Nbf;imu++)
      for(size_t inu=0;inu<Nbf;inu++)
	for(size_t nu=0;nu<Norb;nu++) 
	  for(size_t iaux=0;iaux<Naux;iaux++)
	    tmp(imu*Norb+nu,iaux)+=C[ispin](inu,nu)*dfit.get_a_munu(iaux,imu,inu);
    
    // and then wrt mu.
    munu_I.zeros();
    for(size_t nu=0;nu<Norb;nu++)
      for(size_t mu=0;mu<Norb;mu++)
	for(size_t imu=0;imu<Nbf;imu++)
	  for(size_t iaux=0;iaux<Naux;iaux++)
	    munu_I(mu*Norb+nu,iaux)+=C[ispin](imu,mu)*tmp(imu*Norb+nu,iaux);

    // Now we can calculate K.
    for(size_t ip=0;ip<pairs[ispin].size();ip++) {
      for(size_t jp=0;jp<pairs[ispin].size();jp++) {
	double tmp=arma::as_scalar(munu_I.row(pairs[ispin][ip].i*Norb+pairs[ispin][ip].f)*ab_inv*arma::trans(munu_I.row(pairs[ispin][jp].i*Norb+pairs[ispin][jp].f)));
	K[ispin](ip,jp)+=tmp;
	K[ispin](jp,ip)+=tmp;
      }
      K[ispin](ip,ip)+=arma::as_scalar(munu_I.row(pairs[ispin][ip].i*Norb+pairs[ispin][ip].f)*ab_inv*arma::trans(munu_I.row(pairs[ispin][ip].i*Norb+pairs[ispin][ip].f)));
    }
  }

  printf("Coulomb coupling matrix computed in %s.\n",t.elapsed().c_str());
  fprintf(stderr,"KCoul %s.\n",t.elapsed().c_str());
}

void Casida::Kxc(const BasisSet & bas, double tol, int x_func, int c_func) {
  Timer t;

  // Make grid
  CasidaGrid grid(&bas);
  // Evaluate Kxc
  grid.Kxc(P,tol,x_func,c_func,C,pairs,K);

  printf("XC coupling matrix computed in %s.\n",t.elapsed().c_str());
  fprintf(stderr,"KXC %s.\n",t.elapsed().c_str());
}
