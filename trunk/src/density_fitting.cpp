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
  
  // Dummy shell, helper for computing ERIs
  GaussianShell dummy=dummyshell();

  // Store shell data
  auxshells=auxbas.get_shells();
  orbshells=orbbas.get_shells();

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
    std::vector<double> eris=ERI(&auxshells[is],&dummy,&auxshells[js],&dummy);
    
    // Store integrals
    for(size_t ii=0;ii<auxshells[is].get_Nbf();ii++)
      for(size_t jj=0;jj<auxshells[js].get_Nbf();jj++) {
	ab(auxshells[is].get_first_ind()+ii,auxshells[js].get_first_ind()+jj)=eris[ii*auxshells[js].get_Nbf()+jj];
	ab(auxshells[js].get_first_ind()+jj,auxshells[is].get_first_ind()+ii)=eris[ii*auxshells[js].get_Nbf()+jj];
      }
  }
  
  // Form ab_inv
  ab_inv=arma::inv(ab+DELTA);

#ifdef SCREENING
  // Then, form the screening matrix
  if(direct) {
    screen=arma::mat(orbshells.size(),orbshells.size());
    screen.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      // The shells in question are
      size_t is=orbpairs[ip].is;
      size_t js=orbpairs[ip].js;
      
      // Compute ERIs
      std::vector<double> eris=ERI(&orbshells[is],&orbshells[js],&orbshells[is],&orbshells[js]);
      
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
    for(size_t ia=0;ia<auxshells.size();ia++)
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	// Shells in question are
	size_t imu=orbpairs[ip].is;
	size_t inu=orbpairs[ip].js;
	
	// Amount of functions
	
	size_t Na=auxshells[ia].get_Nbf();
	size_t Nmu=orbshells[imu].get_Nbf();
	size_t Nnu=orbshells[inu].get_Nbf();
	
	// Compute (a|mn)
	std::vector<double> eris=ERI(&auxshells[ia],&dummy,&orbshells[imu],&orbshells[inu]);
	
	// Store integrals
	for(size_t af=0;af<Na;af++) {
	  size_t inda=auxshells[ia].get_first_ind()+af;
	  
	  for(size_t muf=0;muf<Nmu;muf++) {
	    size_t indmu=orbshells[imu].get_first_ind()+muf;
	    
	    for(size_t nuf=0;nuf<Nnu;nuf++) {
	      size_t indnu=orbshells[inu].get_first_ind()+nuf;
	      
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
    // Dummy shell, helper for computing ERIs
    GaussianShell dummy=dummyshell();

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
	
	size_t Nmu=orbshells[imus].get_Nbf();
	size_t Nnu=orbshells[inus].get_Nbf();
	
	for(size_t ias=0;ias<auxshells.size();ias++) {
	  
	  size_t Na=auxshells[ias].get_Nbf();
	  
	  // Compute (a|mn)
	  std::vector<double> eris=ERI(&auxshells[ias],&dummy,&orbshells[imus],&orbshells[inus]);
	  
	  // Increment gamma
	  for(size_t iia=0;iia<Na;iia++) {
	    size_t ia=auxshells[ias].get_first_ind()+iia;
	    
	    for(size_t iimu=0;iimu<Nmu;iimu++) {
	      size_t imu=orbshells[imus].get_first_ind()+iimu;
	      
	      for(size_t iinu=0;iinu<Nnu;iinu++) {
		size_t inu=orbshells[inus].get_first_ind()+iinu;

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
    // Dummy shell, helper for computing ERIs
    GaussianShell dummy=dummyshell();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++)
      for(size_t ias=0;ias<auxshells.size();ias++) {
	
	size_t imus=orbpairs[ip].is;
	size_t inus=orbpairs[ip].js;
	
	
#ifdef SCREENING
	// Do we need to compute the integral?
	if(screen(imus,inus)<THR)
	  continue;
#endif
	
	size_t Na=auxshells[ias].get_Nbf();
	size_t Nmu=orbshells[imus].get_Nbf();
	size_t Nnu=orbshells[inus].get_Nbf();
	
	// Compute (a|mn)
	std::vector<double> eris=ERI(&auxshells[ias],&dummy,&orbshells[imus],&orbshells[inus]);
	
	// Increment J
	for(size_t iia=0;iia<Na;iia++) {
	  size_t ia=auxshells[ias].get_first_ind()+iia;
	  
	  for(size_t iimu=0;iimu<Nmu;iimu++) {
	    size_t imu=orbshells[imus].get_first_ind()+iimu;
	    
	    for(size_t iinu=0;iinu<Nnu;iinu++) {
	      size_t inu=orbshells[inus].get_first_ind()+iinu;

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
