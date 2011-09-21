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

// \delta parameter in Eichkorn et al
#define DELTA 1e-9

Casida::Casida(const Settings & set, const BasisSet & basis, const arma::vec & Ev, const arma::mat & Cv, const arma::mat & Pv) {
  E.push_back(Ev);
  C.push_back(Cv);
  P.push_back(Pv);

  printf("\n*** Warning! The Casida implementation is still experimental. ***\n");
  fprintf(stderr,"\n*** Warning! The Casida implementation is still experimental. ***\n");

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

  printf("\n*** Warning! The Casida implementation is still experimental. ***\n");
  fprintf(stderr,"\n*** Warning! The Casida implementation is still experimental. ***\n");

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

  // Allocate memory
  if(pairs.size()==1)
    // Restricted case
    K.zeros(pairs[0].size(),pairs[0].size());
  else
    // Unrestricted case
    K.zeros(pairs[0].size()+pairs[1].size(),pairs[0].size()+pairs[1].size());

  // Do we need to form K?
  if(coupling!=IPA) {
    // Compute Coulomb coupling
    Kcoul(basis);

    // Compute XC coupling if necessary
    if(coupling==TDLDA) {
      Kxc(basis,tol,x_func,c_func);
    }
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
  } else {
    // Amount of occupied states is
    nocc.push_back((bas.Ztot()-set.get_int("Charge"))/2);
    // Amount of virtual states is
    nvirt.push_back(Norb-nocc[0]);
  }

  // Resize pairs
  pairs.resize(nocc.size());
  // Resize occupation numbers
  f.resize(nocc.size());

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

    // Form f.
    for(size_t ispin=0;ispin<nocc.size();ispin++) {
      f[ispin].zeros(nocc[ispin]+nvirt[ispin]);
      for(size_t iocc=0;iocc<nocc[ispin];iocc++)
	f[ispin](iocc)=pol ? 1.0 : 2.0;
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

      // Get active orbitals and energies
      arma::mat newC(C[ispin].n_rows,idx.size());
      for(size_t i=0;i<idx.size();i++)
	newC.col(i)=C[ispin].col(idx[i]);
      C[ispin]=newC;

      arma::vec newE(C[ispin].n_elem);
      for(size_t i=0;i<idx.size();i++)
	newE(i)=E[ispin](idx[i]);
      E[ispin]=newE;

      // Form f
      f[ispin].zeros(idx.size());
      for(size_t i=0;i<idx.size();i++)
	if(idx[i]<nocc[ispin])
	  // Occupied orbital
	  f[ispin](i)=pol ? 1.0 : 2.0;
	else
	  break;
      
      // Loop over indices
      for(size_t iocc=0;iocc<idx.size();iocc++) {
	// Check that it truly is occupied.
	if(idx[iocc]>=nocc[ispin])
	  continue;

	for(size_t jvirt=iocc+1;jvirt<idx.size();jvirt++) {
	  // Check that it truly is virtual.
	  if(idx[jvirt]<nocc[ispin])
	    continue;

	  // Create state pair (no idx needed here since we have
	  // already dropped inactive orbitals from C)
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

  w_i.zeros(K.n_rows);
  F_i.zeros(K.n_rows,K.n_cols);

  // Generate the coupling matrix (eqn 2.11), but use the K array to
  // save memory.

  if(coupling!=IPA) {
    // Add relevant factors to Coulomb / exchange-correlation terms
    for(size_t ispin=0;ispin<pairs.size();ispin++)
      for(size_t jspin=0;jspin<pairs.size();jspin++) {
	// Offset in i
	const size_t ioff=ispin*pairs[0].size();
	// Offset in j
	const size_t joff=jspin*pairs[0].size();

	for(size_t ip=0;ip<pairs[ispin].size();ip++)
	  for(size_t jp=0;jp<pairs[jspin].size();jp++)
	    K(ioff+ip,joff+jp)*=2.0*fe(pairs[ispin][ip],ispin)*fe(pairs[jspin][jp],jspin);
      }
  }


  // Add IPA contribution to diagonal
  for(size_t ispin=0;ispin<pairs.size();ispin++) {
    // Offset in i
    const size_t ioff=ispin*pairs[0].size();
    for(size_t ip=0;ip<pairs[ispin].size();ip++)
      K(ioff+ip,ioff+ip)+=esq(pairs[ispin][ip],ispin);
  }

  // Solve eigenvalues and eigenvectors using direct linear algebraic methods
  eig_sym_ordered(w_i, F_i, K);

  // The eigenvalues are the squares of the excitation energies
  for(size_t i=0;i<w_i.n_elem;i++)
    w_i(i) = sqrt(w_i(i));

  printf("Casida equations solved in %s.\n",t.elapsed().c_str());
  fprintf(stderr,"Solution %s.\n",t.elapsed().c_str());
}

// This calculates the photoabsorption transition rates
void Casida::absorption() const {
  printf("\n ******* Casida Photoabsorption Spectrum ********\n");

  // Transition rates for every transition
  arma::mat tr(w_i.n_elem,3);
  tr.zeros();

  // Loop over transitions
  for(size_t it=0;it<w_i.n_elem;it++) {
    // Loop over cartesian coordinates
    for(size_t ic=0;ic<3;ic++) {

      // Loop over spins
      for(size_t jspin=0;jspin<pairs.size();jspin++) {
	// Offset in F
	size_t joff=jspin*pairs[0].size();
	// Loop over pairs
	for(size_t jp=0;jp<pairs[jspin].size();jp++) {

	  // Compute |x| = x^T S^{-1/2} F_i
	  tr(it,ic)+=dipmat[jspin][ic](pairs[jspin][jp].i,pairs[jspin][jp].f)*F_i(joff+jp,it)/fe(pairs[jspin][jp],jspin);
	}
      }

      // Normalize to get \lf$ \left\langle \Psi_0 \left| \hat{x}
      // \right| \right\rangle \lf$ , see Eq. 4.40 of Casida (1994),
      // or compare Eqs. 2.14 and 2.16 in Jamorski et al (1996).
      tr(it,ic)/=sqrt(w_i(it));
    }
  }

  // Oscillator strengths, 2/3 * E * ( |x|^2 + |y|^2 + |z|^2 )
  arma::vec osc(w_i.n_elem);
  for(size_t it=0; it<w_i.n_elem;it++)
    osc(it) = 2.0/3.0 * w_i(it) * arma::dot(tr.row(it),tr.row(it));

  // Write output
  printf(  " Photoabsorption transition energies and rates\n");
  printf(  " %6s   %12s   %12s   %12s %12s %12s\n", "nn", "E [eV]", "osc.str.", "<x>", "<y>", "<z>");
  for(size_t it=0; it<osc.n_elem; it++) {
    printf(" %6i    %12.6f   %12.6f   %12.6f %12.6f %12.6f\n", (int) it+1, w_i(it)*HARTREEINEV, osc(it), tr(it, 0), tr(it, 1), tr(it, 2));
  }

  FILE *out=fopen("casida.dat","w");
  for(size_t it=0; it<osc.n_elem; it++)
    fprintf(out,"%e %e % e % e % e\n",w_i(it)*HARTREEINEV, osc(it), tr(it, 0), tr(it, 1), tr(it, 2));
  fclose(out);
}

void Casida::coulomb_fit(const BasisSet & basis, std::vector<arma::mat> & munu, arma::mat & ab_inv) const {
  // Get density fitting basis
  BasisSet dfitbas=basis.density_fitting();
  // Amount of auxiliary functions
  const size_t Naux=dfitbas.get_Nbf();

  // Get the shells
  std::vector<GaussianShell> orbshells=basis.get_shells();
  std::vector<GaussianShell> auxshells=dfitbas.get_shells();

  // Dummy shell, helper for computing ERIs
  coords_t cen={0.0, 0.0, 0.0};
  std::vector<double> Cd, zd;
  Cd.push_back(1.0);
  zd.push_back(0.0);
  GaussianShell dummyshell(0,0,0,0,cen,Cd,zd);

  // First, compute the two-center integrals
  arma::mat ab(Naux,Naux);
  ab.zeros();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
  for(size_t is=0;is<auxshells.size();is++) {
    for(size_t js=0;js<=is;js++) {
      // Compute (a|b)
      std::vector<double> eris=ERI(&auxshells[is],&dummyshell,&auxshells[js],&dummyshell);

      // Store integrals
      for(size_t ii=0;ii<auxshells[is].get_Nbf();ii++)
        for(size_t jj=0;jj<auxshells[js].get_Nbf();jj++) {
          ab(auxshells[is].get_first_ind()+ii,auxshells[js].get_first_ind()+jj)=eris[ii*auxshells[js].get_Nbf()+jj];
          ab(auxshells[js].get_first_ind()+jj,auxshells[is].get_first_ind()+ii)=eris[ii*auxshells[js].get_Nbf()+jj];
        }
    }
  }

  // Form ab_inv
  ab_inv=arma::inv(ab+DELTA);

  // Allocate memory for the three-center integrals.
  munu.resize(C.size());
  for(size_t ispin=0;ispin<C.size();ispin++) {
    munu[ispin].zeros(C[ispin].n_cols*C[ispin].n_cols,Naux);
  }

  // Compute the three-center integrals.
#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifdef _OPENMP
    // Worker stack for each thread
    std::vector<arma::mat> munu_wrk=munu;

#pragma omp for schedule(dynamic,1)
#endif
    for(size_t ia=0;ia<auxshells.size();ia++) {
      // Amount of functions on shell
      size_t Na=auxshells[ia].get_Nbf();
      // Index of first function on shell
      size_t a0=auxshells[ia].get_first_ind();

      for(size_t imu=0;imu<orbshells.size();imu++) {
	// Amount of functions on shell
	size_t Nmu=orbshells[imu].get_Nbf();
	// Index of first function on shell
	size_t mu0=orbshells[imu].get_first_ind();

	for(size_t inu=0;inu<=imu;inu++) {
	  // Amount of functions on shell
	  size_t Nnu=orbshells[inu].get_Nbf();
	  // Index of first function on shell
	  size_t nu0=orbshells[inu].get_first_ind();

	  // Compute the integral over the AOs
	  std::vector<double> eris=ERI(&auxshells[ia],&dummyshell,&orbshells[imu],&orbshells[inu]);

	  // Transform integrals to spin orbitals.
	  for(size_t ispin=0;ispin<C.size();ispin++) {
	    // Amount of active orbitals with current spin.
	    size_t Norb=C[ispin].n_cols;

	    size_t indmu, indnu, inda;
	    // Loop over orbitals
	    for(size_t mu=0;mu<Norb;mu++)
	      for(size_t nu=0;nu<=mu;nu++) {
		// Loop over functions
		for(size_t muf=0;muf<Nmu;muf++) {
		  indmu=mu0+muf;
		  for(size_t nuf=0;nuf<Nnu;nuf++) {
		    indnu=nu0+nuf;
		    
		    // Coefficient of integral is
		    double c=C[ispin](indmu,mu)*C[ispin](indnu,nu);
		    if(imu!=inu)
		      // inu<imu, use symmetry of ERIs
		      c+=C[ispin](indmu,nu)*C[ispin](indnu,mu);
		    
		    // Loop over auxiliary functions
		    for(size_t af=0;af<Na;af++) {
		      inda=a0+af;
		      
#ifdef _OPENMP
		      munu_wrk[ispin](mu*Norb+nu,inda)+=c*eris[(af*Nmu+muf)*Nnu+nuf];
#else
		      munu[ispin](mu*Norb+nu,inda)+=c*eris[(af*Nmu+muf)*Nnu+nuf];
#endif
		    }
		  }
		}
	      }
	    
	  } // end loop over spins
	}
      }
    }
    
#ifdef _OPENMP
#pragma omp critical
    // Sum the results together
    for(size_t ispin=0;ispin<C.size();ispin++)
      munu[ispin]+=munu_wrk[ispin];
#endif
  } // end parallel region

  // Symmetrize munu
  for(size_t ispin=0;ispin<C.size();ispin++) {
    size_t Norb=C[ispin].n_cols;
    for(size_t mu=0;mu<Norb;mu++)
      for(size_t nu=0;nu<=mu;nu++)
	munu[ispin].row(nu*Norb+mu)=munu[ispin].row(mu*Norb+nu);
  }
}

void Casida::Kcoul(const BasisSet & basis) {
  Timer t;

  if(!C.size())
    throw std::runtime_error("Error - no orbitals!\n");

  // Inverse Coulomb overlap matrix of fitting basis
  arma::mat ab_inv;
  // The [\mu \nu|I] matrices in Jamorski (4.16).
  std::vector<arma::mat> munu_I;

  // Get density fitting integrals
  coulomb_fit(basis,munu_I,ab_inv);

  // Construct K
  for(size_t ispin=0;ispin<C.size();ispin++)
    for(size_t jspin=0;jspin<=ispin;jspin++) {
      // Amount of active orbitals
      const size_t Norbi=C[ispin].n_cols;
      const size_t Norbj=C[jspin].n_cols;

      // Offset in i
      const size_t ioff=ispin*pairs[0].size();
      // Offset in j
      const size_t joff=jspin*pairs[0].size();

      if(ispin==jspin) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(size_t ip=0;ip<pairs[ispin].size();ip++) {
	  // Off-diagonal, symmetrization is done later
	  for(size_t jp=0;jp<ip;jp++) {
	    double tmp=arma::as_scalar(munu_I[ispin].row(pairs[ispin][ip].i*Norbi+pairs[ispin][ip].f)*ab_inv*arma::trans(munu_I[ispin].row(pairs[ispin][jp].i*Norbi+pairs[ispin][jp].f)));
	    K(ioff+ip,joff+jp)+=tmp;
	    K(joff+jp,ioff+ip)+=tmp;
	  }
	  // Diagonal
	  K(ioff+ip,ioff+ip)+=arma::as_scalar(munu_I[ispin].row(pairs[ispin][ip].i*Norbi+pairs[ispin][ip].f)*ab_inv*arma::trans(munu_I[ispin].row(pairs[ispin][ip].i*Norbi+pairs[ispin][ip].f)));
	}
      } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(size_t ip=0;ip<pairs[ispin].size();ip++)
	  for(size_t jp=0;jp<pairs[jspin].size();jp++) {
	    // Offset in i
	    const size_t ioff=ispin*pairs[0].size();
	    // Offset in j
	    const size_t joff=jspin*pairs[0].size();

	    K(ioff+ip,joff+jp)=arma::as_scalar(munu_I[ispin].row(pairs[ispin][ip].i*Norbi+pairs[ispin][ip].f)*ab_inv*arma::trans(munu_I[jspin].row(pairs[jspin][jp].i*Norbj+pairs[jspin][jp].f)));
	  }
      }

      if(ispin!=jspin) {
	// Symmetrize
	K.submat(joff,ioff,joff+pairs[jspin].size()-1,ioff+pairs[ispin].size()-1)=arma::trans(K.submat(ioff,joff,ioff+pairs[ispin].size()-1,joff+pairs[jspin].size()-1));
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
