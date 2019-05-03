/*
 * This file is written by Arto Sakko and Susi Lehtola, 2011.
 * Copyright (c) 2011, Arto Sakko and Susi Lehtola
 *
 *
 *
 *                   This file is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Erkale is written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "casida.h"
#include "casida_grid.h"
#include "../lmgrid.h"
#include "../settings.h"
#include "../xrs/fourierprod.h"
#include "../linalg.h"
#include "../eriworker.h"
#include "../stringutil.h"
#include "../timer.h"

#include <cstdio>
#include <cstdlib>
#include <cfloat>

// \delta parameter in Eichkorn et al
#define DELTA 1e-9

// Screen Coulomb integrals?
#define SCREENING
// Screening threshold
#define SCREENTHR 1e-10

extern Settings settings;

Casida::Casida() {
}

Casida::Casida(const BasisSet & basis, const arma::vec & Ev, const arma::mat & Cv, const arma::mat & Pv, const std::vector<double> & occs) {
  E.push_back(Ev);
  C.push_back(Cv);
  P.push_back(Pv);

  // Form pairs
  std::vector< std::vector<double> > occ;
  occ.push_back(occs);
  form_pairs(occ);
  // Sanity check
  if(pairs[0].size()==0)
    throw std::runtime_error("No pairs for Casida calculation! Please check your input.\n");
  printf("Casida calculation has %u pairs.\n",(unsigned int) pairs[0].size());

  // Parse coupling mode
  parse_coupling();

  // Calculate K matrix
  calc_K(basis);
  // and solve Casida equation
  solve();
}

Casida::Casida(const BasisSet & basis, const arma::vec & Ea, const arma::vec & Eb, const arma::mat & Ca, const arma::mat & Cb, const arma::mat & Pa, const arma::mat & Pb, const std::vector<double> & occa, const std::vector<double> & occb) {

  E.push_back(Ea);
  E.push_back(Eb);
  C.push_back(Ca);
  C.push_back(Cb);
  P.push_back(Pa);
  P.push_back(Pb);

  // Form pairs
  std::vector< std::vector<double> > occ;
  occ.push_back(occa);
  occ.push_back(occb);
  form_pairs(occ);
  // Sanity check
  if(pairs[0].size()==0 && pairs[1].size()==0)
    throw std::runtime_error("No pairs for Casida calculation! Please check your input.\n");
  printf("Casida calculation has %u spin up and %u spin down pairs.\n",(unsigned int) pairs[0].size(),(unsigned int) pairs[1].size());

  // Parse coupling mode
  parse_coupling();

  // Calculate K matrix
  calc_K(basis);
  // and solve Casida equation
  solve();
}

void Casida::parse_coupling() {
  // Determine coupling
  switch(settings.get_int("CasidaCoupling")) {
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

void Casida::calc_K(const BasisSet & basis) {
  // Exchange and correlation functionals
  int x_func=settings.get_int("CasidaXfunc");
  int c_func=settings.get_int("CasidaCfunc");
  double tol=settings.get_double("CasidaTol");

  // Allocate memory
  if(pairs.size()==1)
    // Restricted case
    K.zeros(pairs[0].size(),pairs[0].size());
  else
    // Unrestricted case
    K.zeros(pairs[0].size()+pairs[1].size(),pairs[0].size()+pairs[1].size());

  printf("\n");

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

arma::mat Casida::matrix_transform(bool ispin, const arma::mat & m) const {
  return arma::trans(C[ispin])*m*C[ispin];
}

arma::cx_mat Casida::matrix_transform(bool ispin, const arma::cx_mat & m) const {
  return arma::trans(C[ispin])*m*C[ispin];
}

void Casida::form_pairs(const std::vector< std::vector<double> > occs) {
  // First, determine amount of occupied and virtual states.

  nocc.resize(occs.size());
  nvirt.resize(occs.size());

  for(size_t ispin=0;ispin<nocc.size();ispin++) {
    // Count number of occupied states.
    nocc[ispin]=0;
    while(occs[ispin][nocc[ispin]]>0)
      nocc[ispin]++;

    // Check that all values are equal.
    for(size_t i=0;i<nocc[ispin];i++)
      if(occs[ispin][i]!=occs[ispin][0]) {
	ERROR_INFO();
	throw std::runtime_error("Error - occupancies of occupied orbitals differ!\n");
      }

    // Count number of unoccupied states.
    nvirt[ispin]=occs[ispin].size()-nocc[ispin];
    for(size_t i=nocc[ispin];i<occs[ispin].size();i++)
      if(occs[ispin][i]!=0.0) {
	ERROR_INFO();
	throw std::runtime_error("Gaps in occupancy not allowed!\n");
      }
  }

  // Resize pairs
  pairs.resize(nocc.size());
  // Resize occupation numbers
  f.resize(nocc.size());

  // What orbitals are included in the calculation?
  std::vector<std::string> states=splitline(settings.get_string("CasidaStates"));
  if(states.size()==0) {
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

    // Form f. Polarized calculation?
    bool pol=(nocc.size() == 2);
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
      if(idx[idx.size()-1]>nocc[ispin]+nvirt[ispin]) {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Orbital " << idx[idx.size()-1] << " was requested in calculation, but only " << nocc[ispin]+nvirt[ispin] << " orbitals exist!\n";
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

      // Form f. Polarized calculation?
      bool pol=(nocc.size()==2);
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
}

arma::mat Casida::transition(const std::vector<arma::mat> & m) const {
  // Transition rates for every transition
  arma::mat tr(w_i.n_elem,3);
  tr.zeros();

  // Loop over transitions
  for(size_t it=0;it<w_i.n_elem;it++) {
    // Loop over spins
    for(size_t jspin=0;jspin<pairs.size();jspin++) {
      // Offset in F
      size_t joff=jspin*pairs[0].size();
      // Loop over pairs
      for(size_t jp=0;jp<pairs[jspin].size();jp++) {
	// Compute |x| = x^T S^{-1/2} F_i
	tr(it)+=m[jspin](pairs[jspin][jp].i,pairs[jspin][jp].f)*F_i(joff+jp,it)*fe(pairs[jspin][jp],jspin);
      }
    }

    // Normalize to get \f$ \left\langle \Psi_0 \left| \hat{x}
    // \right| \right\rangle \f$ , see Eq. 4.40 of Casida (1994),
    // or compare Eqs. 2.14 and 2.16 in Jamorski et al (1996).
    tr(it)/=sqrt(w_i(it));
  }

  // Transition energies and oscillator strengths
  arma::mat osc(w_i.n_elem,2);
  for(size_t it=0; it<w_i.n_elem;it++) {
    osc(it,0) = w_i(it);
    osc(it,1) = tr(it)*tr(it);
  }

  return osc;
}

arma::mat Casida::transition(const std::vector<arma::cx_mat> & m) const {
  // Transition rates for every transition
  arma::cx_vec tr(w_i.n_elem,3);
  tr.zeros();

  // Loop over transitions
  for(size_t it=0;it<w_i.n_elem;it++) {
    // Loop over spins
    for(size_t jspin=0;jspin<pairs.size();jspin++) {
      // Offset in F
      size_t joff=jspin*pairs[0].size();
      // Loop over pairs
      for(size_t jp=0;jp<pairs[jspin].size();jp++) {
	  // Compute |x| = x^T S^{-1/2} F_i
	tr(it)+=m[jspin](pairs[jspin][jp].i,pairs[jspin][jp].f)*F_i(joff+jp,it)*fe(pairs[jspin][jp],jspin);
      }
    }

    // Normalize to get \f$ \left\langle \Psi_0 \left| \hat{x}
    // \right| \right\rangle \f$ , see Eq. 4.40 of Casida (1994),
    // or compare Eqs. 2.14 and 2.16 in Jamorski et al (1996).
    tr(it)/=sqrt(w_i(it));
  }

  // Transition energies and oscillator strengths
  arma::mat osc(w_i.n_elem,2);
  for(size_t it=0; it<w_i.n_elem;it++) {
    osc(it,0) = w_i(it);
    osc(it,1) = std::norm(tr(it));
  }

  return osc;
}

arma::mat Casida::dipole_transition(const BasisSet & bas) const {
  // Form dipole matrix
  std::vector<arma::mat> dm=bas.moment(1);

  // and convert it to the MO basis
  std::vector< std::vector<arma::mat> > dip(3);
  for(int ic=0;ic<3;ic++)
    for(size_t ispin=0;ispin<C.size();ispin++) {
      dip[ic].resize(C.size());
      dip[ic][ispin]=matrix_transform(ispin,dm[ic]);
    }

  // Compute the oscillator strengths.
  arma::mat osc(w_i.n_elem,2);
  osc.zeros();
  for(int ic=0;ic<3;ic++) {
    // Compute the transitions in the current direction
    arma::mat hlp=transition(dip[ic]);

    // Store the energies
    osc.col(0)=hlp.col(0);
    // and increment the transition speeds
    osc.col(1)+=2.0/3.0*hlp.col(1);
  }

  return osc;
}

arma::mat Casida::transition(const BasisSet & basis, const arma::vec & q) const {
  if(q.n_elem!=3) {
    ERROR_INFO();
    throw std::runtime_error("Momentum transfer should have 3 coordinates!\n");
  }

  // Form products of basis functions.
  const size_t Nbf=basis.get_Nbf();
  std::vector<prod_gaussian_3d> bfprod=compute_products(basis);

  // and their Fourier transforms
  std::vector<prod_fourier> bffour=fourier_transform(bfprod);

  // Get the momentum transfer matrix
  arma::cx_mat momtrans=momentum_transfer(bffour,Nbf,q);
  // and transform it to the MO basis
  std::vector< arma::cx_mat > mtrans(C.size());
  for(size_t ispin=0;ispin<C.size();ispin++)
    mtrans[ispin]=matrix_transform(ispin,momtrans);

  // Compute the transitions
  return transition(mtrans);
}

arma::mat Casida::transition(const BasisSet & basis, double qr) const {
  // Form products of basis functions.
  const size_t Nbf=basis.get_Nbf();
  std::vector<prod_gaussian_3d> bfprod=compute_products(basis);

  // and their Fourier transforms
  std::vector<prod_fourier> bffour=fourier_transform(bfprod);

  // Get the grid for computing the spherical averages.
  std::vector<angular_grid_t> grid=form_angular_grid(2*basis.get_max_am());
  // We normalize the weights so that for purely dipolar transitions we
  // get the same output as with using the dipole matrix.
  for(size_t i=0;i<grid.size();i++) {
    // Dipole integral is only wrt theta - divide off phi part.
    grid[i].w/=2.0*M_PI;
  }

  // Transition energies and oscillator strengths
  arma::mat osc(w_i.n_elem,2);
  osc.zeros();

  // Loop over the angular mesh
  for(size_t ig=0;ig<grid.size();ig++) {
    // Current value of q is
    arma::vec q(3);
    q(0)=qr*grid[ig].r.x;
    q(1)=qr*grid[ig].r.y;
    q(2)=qr*grid[ig].r.z;
    // and the weight is
    double w=grid[ig].w;

    // Get the momentum transfer matrix
    arma::cx_mat momtrans=momentum_transfer(bffour,Nbf,q);
    // and transform it to the MO basis
    std::vector< arma::cx_mat > mtrans(C.size());
    for(size_t ispin=0;ispin<C.size();ispin++)
      mtrans[ispin]=matrix_transform(ispin,momtrans);

    // Compute the transitions
    arma::mat hlp=transition(mtrans);
    // Store the energies
    osc.col(0)=hlp.col(0);
    // and increment the transition speeds
    osc.col(1)+=w*hlp.col(1);
  }

  return osc;
}

void Casida::coulomb_fit(const BasisSet & basis, std::vector<arma::mat> & munu, arma::mat & ab_inv) const {
  // Get density fitting basis
  BasisSet dfitbas;

  if(stricmp(settings.get_string("FittingBasis"),"Auto")==0)
    dfitbas=basis.density_fitting();
  else {
    // Load basis library
    BasisSetLibrary fitlib;
    fitlib.load_basis(settings.get_string("FittingBasis"));

    // Construct fitting basis
    construct_basis(dfitbas,basis.get_nuclei(),fitlib);
  }

  // Amount of auxiliary functions
  const size_t Naux=dfitbas.get_Nbf();

  // Get the shells
  std::vector<GaussianShell> orbshells=basis.get_shells();
  std::vector<GaussianShell> auxshells=dfitbas.get_shells();

  // Get list of pairs
  std::vector<shellpair_t> orbpairs=basis.get_unique_shellpairs();
  std::vector<shellpair_t> auxpairs=dfitbas.get_unique_shellpairs();

  // Dummy shell, helper for computing ERIs
  GaussianShell dummy=dummyshell();

  // First, compute the two-center integrals
  arma::mat ab(Naux,Naux);
  ab.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker eri(dfitbas.get_max_am(),dfitbas.get_max_Ncontr());
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<auxpairs.size();ip++) {
      // Shells in question are
      size_t is=auxpairs[ip].is;
      size_t js=auxpairs[ip].js;

      // Compute (a|b)
      eri.compute(&auxshells[is],&dummy,&auxshells[js],&dummy);
      erip=eri.getp();

      // Store integrals
      for(size_t ii=0;ii<auxshells[is].get_Nbf();ii++)
	for(size_t jj=0;jj<auxshells[js].get_Nbf();jj++) {
	  ab(auxshells[is].get_first_ind()+ii,auxshells[js].get_first_ind()+jj)=(*erip)[ii*auxshells[js].get_Nbf()+jj];
	  ab(auxshells[js].get_first_ind()+jj,auxshells[is].get_first_ind()+ii)=(*erip)[ii*auxshells[js].get_Nbf()+jj];
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

#ifdef SCREENING
  // Screen the integrals.
  arma::mat screen(orbshells.size(),orbshells.size());
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker eri(basis.get_max_am(),basis.get_max_Ncontr());
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      // The shells in question are
      size_t is=orbpairs[ip].is;
      size_t js=orbpairs[ip].js;

      // Compute (*Erip)
      eri.compute(&orbshells[is],&orbshells[js],&orbshells[is],&orbshells[js]);
      erip=eri.getp();

      // Find out maximum value
      double max=0.0;
      for(size_t i=0;i<(*erip).size();i++)
	if(fabs((*erip)[i])>max)
	  max=(*erip)[i];
      max=sqrt(max);

      // Store value
      screen(is,js)=max;
      screen(js,is)=max;
    }
  }
#endif

  // Compute the three-center integrals.
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker eri(std::max(basis.get_max_am(),dfitbas.get_max_am()),std::max(basis.get_max_Ncontr(),dfitbas.get_max_Ncontr()));
    const std::vector<double> * erip;


#ifdef _OPENMP
    // Worker stack for each thread
    std::vector<arma::mat> munu_wrk=munu;

#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      // Shells in question are
      size_t imu=orbpairs[ip].is;
      size_t inu=orbpairs[ip].js;

#ifdef SCREENING
      // Do we need to compute the integral?
      if(screen(imu,inu)<SCREENTHR)
	continue;
#endif

      // Amount of functions on shell
      size_t Nmu=orbshells[imu].get_Nbf();
      // Index of first function on shell
      size_t mu0=orbshells[imu].get_first_ind();

      // Amount of functions on shell
      size_t Nnu=orbshells[inu].get_Nbf();
      // Index of first function on shell
      size_t nu0=orbshells[inu].get_first_ind();

      for(size_t ia=0;ia<auxshells.size();ia++) {
	// Amount of functions on shell
	size_t Na=auxshells[ia].get_Nbf();
	// Index of first function on shell
	size_t a0=auxshells[ia].get_first_ind();

	// Compute the integral over the AOs
	eri.compute(&auxshells[ia],&dummy,&orbshells[imu],&orbshells[inu]);
	erip=eri.getp();

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
		  double c= (imu!=inu) ? C[ispin](indmu,mu)*C[ispin](indnu,nu) + C[ispin](indmu,nu)*C[ispin](indnu,mu) : C[ispin](indmu,mu)*C[ispin](indnu,nu);

		  // Loop over auxiliary functions
		  for(size_t af=0;af<Na;af++) {
		    inda=a0+af;

#ifdef _OPENMP
		    munu_wrk[ispin](mu*Norb+nu,inda)+=c*(*erip)[(af*Nmu+muf)*Nnu+nuf];
#else
		    munu[ispin](mu*Norb+nu,inda)+=c*(*erip)[(af*Nmu+muf)*Nnu+nuf];
#endif
		  }
		}
	      }
	    }

	} // end loop over spins
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
	    K(ioff+ip,joff+jp)=arma::as_scalar(munu_I[ispin].row(pairs[ispin][ip].i*Norbi+pairs[ispin][ip].f)*ab_inv*arma::trans(munu_I[jspin].row(pairs[jspin][jp].i*Norbj+pairs[jspin][jp].f)));
	  }
      }

      if(ispin!=jspin) {
	// Symmetrize
	K.submat(joff,ioff,joff+pairs[jspin].size()-1,ioff+pairs[ispin].size()-1)=arma::trans(K.submat(ioff,joff,ioff+pairs[ispin].size()-1,joff+pairs[jspin].size()-1));
      }
    }

  printf("Coulomb coupling matrix computed in %s.\n",t.elapsed().c_str());
}

void Casida::Kxc(const BasisSet & bas, double tol, int x_func, int c_func) {
  Timer t;

  // Make grid
  CasidaGrid grid(&bas);
  // Evaluate Kxc
  grid.Kxc(P,tol,x_func,c_func,C,pairs,K);

  printf("XC coupling matrix computed in %s.\n",t.elapsed().c_str());
}
