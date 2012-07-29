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



#include "density_fitting.h"

// \delta parameter in Eichkorn et al
#define DELTA 1e-9
// THR parameter in Eichkorn et al
#define THR 1e-10

// Screen integrals? (Direct calculations)
#define SCREENING


DensityFit::DensityFit() {
}

DensityFit::~DensityFit() {
}


void DensityFit::fill(const BasisSet & orbbas, const BasisSet & auxbas, bool dir) {
  // Construct density fitting basis

  // Store amount of functions
  Norb=orbbas.get_Nbf();
  Naux=auxbas.get_Nbf();
  direct=dir;

  // Fill index helper
  iidx=i_idx(Norb);
  // Fill list of shell pairs
  orbpairs=orbbas.get_unique_shellpairs();

  // Form total basis set
  totbas=orbbas;
  // Form indices of orbital shells
  orbind.resize(orbbas.get_Nshells());
  for(size_t i=0;i<orbbas.get_Nshells();i++)
    orbind[i]=i;

  // Add auxiliary functions to total basis set
  auxind.resize(auxbas.get_Nshells());
  std::vector<GaussianShell> auxsh=auxbas.get_shells();
  for(size_t i=0;i<auxsh.size();i++) {
    totbas.add_shell(auxsh[i].get_center_ind(),auxsh[i],false);
    auxind[i]=i+orbbas.get_Nshells();
  }

  // Finally, add dummy shell to basis set
  GaussianShell dummy=dummyshell();
  totbas.add_shell(0,dummy,false);
  dummyind=orbbas.get_Nshells()+auxbas.get_Nshells();

  // Finalize total basis set
  totbas.finalize();

  // First, compute the two-center integrals
  ab=arma::mat(Naux,Naux);
  ab.zeros();
  
  // Get list of unique auxiliary shell pairs
  std::vector<shellpair_t> auxpairs=auxbas.get_unique_shellpairs();
  
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ip=0;ip<auxpairs.size();ip++) {
    // The shells in question are
    size_t is=auxpairs[ip].is;
    size_t js=auxpairs[ip].js;
    
    // Compute (a|b)
    std::vector<double> eris=totbas.ERI(auxind[is],dummyind,auxind[js],dummyind);
    
    // Store integrals
    size_t Ni=totbas.get_Nbf(auxind[is]);
    size_t Nj=totbas.get_Nbf(auxind[js]);
    for(size_t ii=0;ii<Ni;ii++) {
      // Account for orbital functions at the beginning of the basis set
      size_t ai=totbas.get_first_ind(auxind[is])+ii-Norb;
      for(size_t jj=0;jj<Nj;jj++) {
	// Account for orbital functions at the beginning of the basis set
	size_t aj=totbas.get_first_ind(auxind[js])+jj-Norb;

	ab(ai,aj)=eris[ii*Nj+jj];
	ab(aj,ai)=eris[ii*Nj+jj];
      }
    }
  }

  // Form ab_inv
  ab_inv=arma::inv(ab+DELTA);

#ifdef SCREENING
  // Then, form the screening matrix
  if(direct) {
    screen=arma::mat(orbind.size(),orbind.size());
    screen.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      // The shells in question are
      size_t is=orbpairs[ip].is;
      size_t js=orbpairs[ip].js;
      
      // Compute ERIs
      std::vector<double> eris=totbas.ERI(orbind[is],orbind[js],orbind[is],orbind[js]);
      
      // Find out maximum value
      double max=0.0;
      for(size_t i=0;i<eris.size();i++)
	if(fabs(eris[i])>max)
	  max=eris[i];
      max=sqrt(max);
      
      // Store value
      screen(is,js)=max;
      screen(js,is)=max;
    }
  }
#endif

  // Then, compute the three-center integrals
  if(!direct) {
    a_munu.resize(Naux*Norb*(Norb+1)/2);
    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(size_t ia=0;ia<auxind.size();ia++)
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	// Shells in question are
	size_t imu=orbpairs[ip].is;
	size_t inu=orbpairs[ip].js;
	
	// Amount of functions
	
	size_t Na=totbas.get_Nbf(auxind[ia]);
	size_t Nmu=totbas.get_Nbf(orbind[imu]);
	size_t Nnu=totbas.get_Nbf(orbind[inu]);
	
	// Compute (a|mn)
	std::vector<double> eris=totbas.ERI(auxind[ia],dummyind,orbind[imu],orbind[inu]);
	
	// Store integrals
	for(size_t af=0;af<Na;af++) {
	  // Account for orbital functions at the beginning of the basis set
	  size_t inda=totbas.get_first_ind(auxind[ia])+af-Norb;
	  
	  for(size_t muf=0;muf<Nmu;muf++) {
	    size_t indmu=totbas.get_first_ind(orbind[imu])+muf;
	    
	    for(size_t nuf=0;nuf<Nnu;nuf++) {
	      size_t indnu=totbas.get_first_ind(orbind[inu])+nuf;
	      
	      a_munu[idx(inda,indmu,indnu)]=eris[(af*Nmu+muf)*Nnu+nuf];
	    }
	  }
	}
      }
  }
  
}


size_t DensityFit::idx(size_t ia, size_t imu, size_t inu) const {
  if(imu<inu)
    std::swap(imu,inu);

  return Naux*(iidx[imu]+inu)+ia;
}

size_t DensityFit::memory_estimate(const BasisSet & orbbas, const BasisSet & auxbas, bool dir) const {

  // Amount of orbital basis functions
  size_t No=orbbas.get_Nbf();
  // Amount of auxiliary functions (for representing the electron density)
  size_t Na=auxbas.get_Nbf();
  // Amount of memory required for calculation
  size_t Nmem=0;

  // Memory taken up by index helper
  Nmem+=No*sizeof(size_t);
  // Memory taken up by  ( \alpha | \mu \nu)
  if(!dir)
    Nmem+=(Na*No*(No+1)/2)*sizeof(double);
#ifdef SCREENING
  else {
    // Memory taken up by screening matrix
    size_t Nsh=orbbas.get_Nshells();
    Nmem+=Nsh*Nsh*sizeof(double);
  }
#endif

  // Memory taken by (\alpha | \beta) and its inverse
  Nmem+=2*Na*Na*sizeof(double);
  // Memory taken by gamma and expansion coefficients
  Nmem+=2*Na*sizeof(double);

  return Nmem;
}

arma::vec DensityFit::compute_expansion(const arma::mat & P) const {
  arma::vec gamma(Naux);
  gamma.zeros();

  // Compute gamma
  if(!direct) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(size_t ia=0;ia<Naux;ia++) {
      gamma(ia)=0.0;
      
      for(size_t imu=0;imu<Norb;imu++) {
	// Off-diagonal
	for(size_t inu=0;inu<imu;inu++)
	  gamma(ia)+=2.0*a_munu[idx(ia,imu,inu)]*P(imu,inu);
	// Diagonal
	gamma(ia)+=a_munu[idx(ia,imu,imu)]*P(imu,imu);
      }
    }
  } else {

#ifdef _OPENMP
#pragma omp parallel
#endif
    {

#ifdef _OPENMP
      // Worker stack for each matrix
      arma::vec gammawrk(gamma);

#pragma omp for schedule(dynamic)
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	size_t imus=orbpairs[ip].is;
	size_t inus=orbpairs[ip].js;
	
#ifdef SCREENING
	// Do we need to compute the integral?
	if(screen(imus,inus)<THR)
	  continue;
#endif
	
	size_t Nmu=totbas.get_Nbf(orbind[imus]);
	size_t Nnu=totbas.get_Nbf(orbind[inus]);
	
	for(size_t ias=0;ias<auxind.size();ias++) {
	  
	  size_t Na=totbas.get_Nbf(auxind[ias]);
	  
	  // Compute (a|mn)
	  std::vector<double> eris=totbas.ERI(auxind[ias],dummyind,orbind[imus],orbind[inus]);
	  
	  // Increment gamma
	  for(size_t iia=0;iia<Na;iia++) {
	  // Account for orbital functions at the beginning of the basis set
	    size_t ia=totbas.get_first_ind(auxind[ias])+iia-Norb;
	    
	    for(size_t iimu=0;iimu<Nmu;iimu++) {
	      size_t imu=totbas.get_first_ind(orbind[imus])+iimu;
	      
	      for(size_t iinu=0;iinu<Nnu;iinu++) {
		size_t inu=totbas.get_first_ind(orbind[inus])+iinu;

		// The contracted integral
		double res=eris[(iia*Nmu+iimu)*Nnu+iinu]*P(imu,inu);

		// If imus==inus, we need to take care that we count
		// every term only once; on the off-diagonal we get
		// every term twice.
#ifdef _OPENMP
		gammawrk(ia)+= (imus==inus) ? res : 2.0*res;
#else
		gamma(ia)+= (imus==inus) ? res : 2.0*res;
#endif
	      }
	    }
	  }
	}
      }
      
#ifdef _OPENMP
#pragma omp critical
      // Sum results together
      gamma+=gammawrk;
#endif
    } // end parallel section
  }
  
  // Compute x0
  arma::vec x0=ab_inv*gamma;
  // Compute and return c
  return x0+ab_inv*(gamma-ab*x0);
}

arma::mat DensityFit::calc_J(const arma::mat & P) const {
  // Get the expansion coefficients
  arma::vec c=compute_expansion(P);

  arma::mat J(Norb,Norb);
  J.zeros();

  if(!direct) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(size_t imu=0;imu<Norb;imu++)
      for(size_t inu=0;inu<=imu;inu++) {
	J(imu,inu)=0.0;
	
	for(size_t ia=0;ia<Naux;ia++)
	  J(imu,inu)+=a_munu[idx(ia,imu,inu)]*c(ia);
	
	J(inu,imu)=J(imu,inu);
      }

  } else {

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++)
      for(size_t ias=0;ias<auxind.size();ias++) {
	
	size_t imus=orbpairs[ip].is;
	size_t inus=orbpairs[ip].js;
	
	
#ifdef SCREENING
	// Do we need to compute the integral?
	if(screen(imus,inus)<THR)
	  continue;
#endif
	
	size_t Na=totbas.get_Nbf(auxind[ias]);
	size_t Nmu=totbas.get_Nbf(orbind[imus]);
	size_t Nnu=totbas.get_Nbf(orbind[inus]);
	
	// Compute (a|mn)
	std::vector<double> eris=totbas.ERI(auxind[ias],dummyind,orbind[imus],orbind[inus]);
	
	// Increment J
	for(size_t iia=0;iia<Na;iia++) {
	  // Account for orbital functions at the beginning of the basis set
	  size_t ia=totbas.get_first_ind(auxind[ias])+iia-Norb;
	  
	  for(size_t iimu=0;iimu<Nmu;iimu++) {
	    size_t imu=totbas.get_first_ind(orbind[imus])+iimu;
	    
	    for(size_t iinu=0;iinu<Nnu;iinu++) {
	      size_t inu=totbas.get_first_ind(orbind[inus])+iinu;

	      // Contract result
	      double tmp=eris[(iia*Nmu+iimu)*Nnu+iinu]*c(ia);

	      J(imu,inu)+=tmp;
	      // Need to symmetrize?
	      J(inu,imu)+= (imus==inus) ? 0.0 : tmp;
	    }
	  }
	}
      }
  }
  
  return J;
}

size_t DensityFit::get_Naux() const {
  return Naux;
}

double DensityFit::get_a_munu(size_t ia, size_t imu, size_t inu) const {
  if(!direct)
    return a_munu[idx(ia,imu,inu)]; 
  else {
    ERROR_INFO();
    throw std::runtime_error("get_a_munu not implemented for direct calculations!\n");
  }

  // Dummy return clause
  return 0.0;
}

arma::mat DensityFit::get_ab_inv() const {
  return ab_inv;
}
