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

#ifndef ERKALE_CO_OPT
#define ERKALE_CO_OPT

#include <cstdio>
#include <unistd.h>
#include <iostream>
#include "completeness/optimize_completeness.h"
#include "completeness/completeness_profile.h"
#include "external/fchkpt_tools.h"
#include "basislibrary.h"
#include "global.h"
#include "timer.h"
#include "linalg.h"
#include "stringutil.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/// Maximum amount of functions
#define NFMAX 70
/// Moment to optimize: area
#define OPTMOMIND 1

/// Structure for completeness-optimized basis set
typedef struct {
  /// Start of completeness profile
  double start;
  /// End of completeness profile
  double end;
  /// Tolerance
  double tol;

  /// The completeness-optimized exponents
  arma::vec exps;
} coprof_t;

/// Get maximum angular momentum in profile
int maxam(const std::vector<coprof_t> & cpl);
/// Count amount of functions in profile
int get_nfuncs(const std::vector<coprof_t> & cpl);
/// Print out the limits of the profile
void print_limits(const std::vector<coprof_t> & cpl, const char *msg=NULL);

/// Save limits to file
void save_limits(const std::vector<coprof_t> & cpl, const std::string & fname);
/// Load limits from file
void load_limits(std::vector<coprof_t> & cpl, const std::string & fname);

/// Augment basis set with additional diffuse and/or tight functions
std::vector<coprof_t> augment_basis(const std::vector<coprof_t> & cplo, int ndiffuse, int ntight);

// Construct elemental library
ElementBasisSet get_element_library(const std::string & el, const std::vector<coprof_t> & cpl);
// Construct universal library
BasisSetLibrary get_library(const std::vector<coprof_t> & cpl);

/// Print out the contraction scheme
void print_scheme(const BasisSetLibrary & baslib, int len=0);

/// Helper for maxwidth_exps_table
typedef struct {
  // Completeness
  double tol;
  // Width
  double w;
  // Exponents
  arma::vec exps;
} co_exps_t;

/// Store optimized exponents
arma::vec maxwidth_exps_table(int am, double tol, size_t nexp, double & width, int n=OPTMOMIND);
/// Get exponents with wanted tolerance that span at least the wanted width. nx gives starting point for search (nx+1 is the first one tried)
arma::vec span_width(int am, double tol, double & width, int nx=0, int n=OPTMOMIND);
/// Move exponents in logarithmic scale to start at x instead of 0.0
arma::vec move_exps(const arma::vec & exps, double start);

/// Completeness optimizer class
template<typename ValueType> 
class CompletenessOptimizer {
 private:
  /// Compute the energy
  virtual double compute_energy(const std::vector<coprof_t> & cpl)=0;
  /// Form a basis set
  virtual BasisSetLibrary form_basis(const std::vector<coprof_t> & cpl)=0;
  /// Form a contracted basis set. Porth toggles P-orthogonalization
  virtual BasisSetLibrary form_basis(const std::vector<coprof_t> & cpl, std::vector<size_t> contract, bool Porth=true) {
    /// Dummy declarations
    (void) contract;
    (void) Porth;
    return form_basis(cpl);
  }
  /// Update reference wave function
  virtual void update_reference(const std::vector<coprof_t> & cpl) {
    /// Dummy declaration
    (void) cpl;
  };


 public:  
  /// Constructor
  CompletenessOptimizer() {
  }
  /// Destructor
  ~CompletenessOptimizer() {
  }

  /**
   * Generate initial profile by minimizing the energy.
   * Place nexp functions per occupied electronic shell, given with nel argument.
   */
  std::vector<coprof_t> initial_profile(const std::vector<int> & nshells, int nexp, double cotol=1e-4) {
    std::vector<size_t> nfuncs(nshells.begin(),nshells.end());
    for(size_t i=0;i<nfuncs.size();i++)
      nfuncs[i]*=nexp;
    return initial_profile(nfuncs,cotol);
  }

  /**
   * Generate initial profile by minimizing the energy.
   * Amount of functions of each angular momentum given in nfuncs.
   */
  std::vector<coprof_t> initial_profile(const std::vector<size_t> & nfuncs, double cotol=1e-4) {
    Timer t;
    
    // Initialize profile
    std::vector<coprof_t> cpl(nfuncs.size());
    for(size_t i=0;i<cpl.size();i++) {
      cpl[i].start=0.0;
      cpl[i].end=0.0;
      cpl[i].tol=cotol;
    }

    // Exponents
    std::vector<arma::vec> exps(cpl.size());
    arma::vec widths(cpl.size());
    
    // Generate initial profile
    printf("\nStarting composition:\n");
    for(size_t am=0;am<cpl.size();am++) {
      exps[am]=maxwidth_exps_table(am,cpl[am].tol,nfuncs[am],widths[am],OPTMOMIND);
      cpl[am].start=0.0;
      cpl[am].end=widths[am];
      cpl[am].exps=exps[am];
      
      printf("%c % .3f % .3f %e %2i\n",shell_types[am],cpl[am].start,cpl[am].end,cpl[am].tol,(int) cpl[am].exps.size());
      fflush(stdout);
    }
    
    // Widths normalized by amount of exponents
    arma::vec relw(widths);
    for(size_t am=0;am<relw.size();am++)
      relw(am)/=exps[am].size();
    
    // Convergence loop
    const double h0=1e-3; // Initial step size in relative width
    size_t iiter=0;
    
    // Old gradient
    arma::vec gold;

    while(true) {
      Timer tstep;

      // Conjugate gradient step?
      bool cg=iiter%cpl.size()!=0;
      if(cg)
	printf("\nConjugate gradient iteration %i\n",(int) iiter+1);
      else
	printf("\nSteepest descent   iteration %i\n",(int) iiter+1);
      fflush(stdout);

      // Compute finite difference gradient
      arma::vec g(cpl.size());
      for(size_t am=0;am<cpl.size();am++) {
	// Step size to use for shell
	double h=h0*relw(am);

	std::vector<coprof_t> lcpl(cpl);
	lcpl[am].start-=h;
	lcpl[am].end-=h;
	lcpl[am].exps=move_exps(exps[am],lcpl[am].start);

	std::vector<coprof_t> rcpl(cpl);
	rcpl[am].start+=h;
	rcpl[am].end+=h;
	rcpl[am].exps=move_exps(exps[am],rcpl[am].start);

	// Energies
	double El=compute_energy(lcpl);
	double Er=compute_energy(rcpl);
	
	// Finite difference gradient. Absorb relative width here
	g(am)=(Er-El)/(2*h) * relw(am);
	printf("\t-grad(%c)  = % e\n", shell_types[am], -g(am));
	fflush(stdout);
      }
      
      {
	// Print bar
	size_t Nm=26;
	char minus[Nm];
	memset(minus,'-',Nm);
	minus[Nm-1]='\0';
	printf("\t%s\n",minus);
      }
      printf("\tgrad norm = % e\n",arma::norm(g,2));
      fflush(stdout);

      // Normalize gradient
      g/=arma::norm(g,2);

      // Orthogonalize wrt old gradient?
      if(cg) {
	g-=arma::dot(g,gold)*gold;
	g/=arma::norm(g,2);
      }

      // Store old gradient
      gold=g;
    
      // Line search
      const size_t Ntr=20; // This is ridiculously large to prevent runover
      arma::vec h(Ntr);
      arma::vec E(Ntr);
      E.zeros();

      // Step sizes
      h(0)=0.0;
      double hs=10*h0;
      for(size_t i=1;i<Ntr;i++)
	h(i)=hs*std::pow(2.0,i-1);

      printf("\n\tLine search:\n");
      printf("\t%5s %12s %13s %13s\n","trial","step","E","dE");
      fflush(stdout);

      size_t itr;
      for(itr=0;itr<Ntr;itr++) {
	// Profile
	std::vector<coprof_t> trcpl(cpl);
	for(size_t am=0;am<g.n_elem;am++) {
	  trcpl[am].start-=h(itr)*g(am);
	  trcpl[am].end-=h(itr)*g(am);
	  trcpl[am].exps=move_exps(exps[am],trcpl[am].start);
	}
      
	// Energy
	E(itr)=compute_energy(trcpl);

	if(itr==0)
	  printf("\t%2i/%-2i %e % e\n",(int) (itr+1), (int) Ntr, h(itr), E(itr));
	else
	  printf("\t%2i/%-2i %e % e % e\n",(int) (itr+1), (int) Ntr, h(itr), E(itr), E(itr)-E(0));
	fflush(stdout);

	// Check if energy increased
	if(itr>0 && E(itr)>E(itr-1))
	  break;
      }

      // Find minimal energy
      arma::uword ind;
      E.min(ind);
      double step=h(ind);
    
      if(ind!=0) {
	printf("\tMinimum with step %e, decrease by % e.\n",h(ind),E(ind)-E(0));
      	fflush(stdout);

	if(ind>=1) {
	  // Perform cubic interpolation.

	  arma::vec x=h.subvec(ind-1,ind+1);
	  arma::vec y=E.subvec(ind-1,ind+1);

	  // Fit polynomial
	  arma::vec c=fit_polynomial(x,y);
	  // Derivative coefficients
	  arma::vec dc=derivative_coefficients(c);
	  // Get smallest positive root
	  double st=smallest_positive(solve_roots(dc));

	  printf("\tInterpolation gives step %e, ",st);
	  fflush(stdout);
		
	  // Trial profile
	  std::vector<coprof_t> trcpl(cpl);
	  for(size_t am=0;am<g.n_elem;am++) {
	    trcpl[am].start-=st*g(am);
	    trcpl[am].end-=st*g(am);
	    trcpl[am].exps=move_exps(exps[am],trcpl[am].start);
	  }
	
	  // Compute new trial value
	  double Etr=compute_energy(trcpl);
	
	  if(Etr<E.min()) {
	    step=st;
	    printf("further decrease by % e, accepted.\n",Etr-E.min());
	  } else
	    printf("further increase by % e, rejected.\n",Etr-E.min());
	  fflush(stdout);
	}

	// Update profile limits
	for(size_t am=0;am<g.n_elem;am++) {
	  cpl[am].start-=step*g(am);
	  cpl[am].end-=step*g(am);
	  cpl[am].exps=move_exps(exps[am],cpl[am].start);
	}

	printf("Iteration took %s.\n",tstep.elapsed().c_str());
	print_limits(cpl,"\nCurrent limits");
      	fflush(stdout);

      } else
	break;

      iiter++;
    }

    print_limits(cpl,"Initial composition");
    printf("Generation of initial profile took %s.\n",t.elapsed().c_str());
    fflush(stdout);

    return cpl;
  }

  /// Compute value with given basis set
  virtual ValueType compute_value(const BasisSetLibrary & baslib)=0;
  /// Compute value with given basis set
  ValueType compute_value(const std::vector<coprof_t> & cpl) {
    return compute_value(form_basis(cpl));
  }
  
  /// Compute distance of two values
  virtual double compute_mog(const ValueType & val, const ValueType & ref)=0;

  /**
   * Compute the mog of a new polarization shell. Returns maximal mog. cpl and curval are modified.
   *
   * Parameters
   * am_max:    maximal angular momentum allowed
   * minpol:    ~ minimal allowed value for polarization exponent (log10 scale)
   * maxpol:    ~ maximal allowed value for polarization exponent (log10 scale)
   * dpol:      amount of points used per completeness range
   * polinterp: toggle interpolation of mog to find maximum
   * cotol:     deviation from completeness
   * nx:        amounts of exponents to place on the shell
   */
  double check_polarization(std::vector<coprof_t> & cpl, ValueType & curval, int am_max, double minpol, double maxpol, double dpol, bool polinterp, double cotol, int nx) {

    printf("\n\n%s\n",print_bar("POLARIZATION SHELLS").c_str());
    fflush(stdout);
	
    // Check necessity of additional polarization shell.
    Timer t;
    const int max=maxam(cpl);
    if(max>=am_max) {
      printf("Won't add another polarization shell, MaxAM=%i reached.\n",am_max);
      fflush(stdout);
      return 0.0;
    }

    Timer tpol;

    // Width of completeness
    double pw;

    // Exponent to add: l=max+1, tolerance is cotol, 1 function.
    const int addam=max+1;
    arma::vec pexp=maxwidth_exps_table(addam,cotol,nx,pw,OPTMOMIND);

    if(dpol<=0) {
      std::ostringstream oss;
      oss << "Invalid value " << dpol << " for DPol.\n";
      throw std::runtime_error(oss.str());
    }

    // Compute spacing
    double sp;
    maxwidth_exps_table(addam,cotol,pexp.size()+1,sp,OPTMOMIND);
    sp/=dpol*pexp.size();
    // Wanted width is
    double ww=maxpol-minpol;
    // Amount of points to use
    size_t Np=1+round(ww/sp);
    // Actual width is
    double width=sp*Np;

    // so actual minimum and maximum are
    double minp=minpol;
    double maxp=maxpol+(width-ww);

    // Starting points
    arma::vec startp=arma::linspace(minp,maxp,Np);

    // Trials
    std::vector< std::vector<coprof_t> > cpls(startp.n_elem);
    // Values
    std::vector<ValueType> vals(startp.n_elem);

    // Resulting mogs
    arma::vec mogs(startp.n_elem);
    mogs.zeros();
    
    printf("Scanning for optimal placement of %i exponents on %c shell.\n",(int) pexp.size(), shell_types[addam]);
    printf("Using %.3f points per exponent interval, so spacing between points is %.5f.\n\n",dpol,sp);
    
    printf("%11s %8s %12s %8s\n","trial","exponent","mog","t (s)");
    fflush(stdout);
    
    static size_t iter=0;

    // Allocate sufficient memory
    if(cpl.size() <= (size_t) addam) {
      cpl.resize(addam+1);
      // Set tolerance
      cpl[addam].tol=cotol;
    }

    // Fill in exponents and mogs
    Timer tptot;
    for(size_t iexp=0;iexp<startp.n_elem;iexp++) {
      Timer tp;

      // Completeness profile
      std::vector<coprof_t> trcpl(cpl);
      trcpl[addam].start=startp(iexp)-pw/2.0;
      trcpl[addam].end=startp(iexp)+pw/2.0;
      trcpl[addam].exps=move_exps(pexp,trcpl[addam].start);

      // Compute trial value
      ValueType trval;
      try {
	trval=compute_value(trcpl);
	// and store the mog
	mogs(iexp)=compute_mog(trval,curval);

      } catch(...) {
	// Calculation failed, set zero mog.
	mogs(iexp)=0.0;
      }
    
      printf("%5i/%-5i % 7.5f %e %8.2f\n",(int) iexp+1, (int) startp.n_elem, startp(iexp), mogs(iexp), tp.get());
      fflush(stdout);

      // Store exponent and mog
      cpls[iexp]=trcpl;
      vals[iexp]=trval;
    }

    { // Save out the results of the mog scan
      // Filename to use
      std::ostringstream fname;
      fname << "polmog_" << shell_types[addam] << "_" << iter << ".dat";
      
      arma::mat savemog(mogs.n_rows,2);
      savemog.col(0)=startp;
      savemog.col(1)=mogs;
      savemog.save(fname.str(),arma::raw_ascii);
      
      iter++;
    }

    // Find maximum mog
    arma::uword imax;
    double maxmog=mogs.max(imax);

    printf("\n");
    printf("%11s % 7.5f %e\n","max mog",startp(imax),maxmog);
    fflush(stdout);

    if(polinterp) {
      // Interpolate mogs to find more accurate maximum. Spacing to use
      double spacing=std::min(1e-6, sp/10.0);
      arma::vec finergrid=arma::linspace(minp,maxp,1+ceil((maxp-minp)/spacing));
      arma::vec moginterp=spline_interpolation(startp,mogs,finergrid);

      arma::uword iintmean;
      double intmean;
      intmean=moginterp.max(iintmean);

      printf("\n");
      printf("%11s % 7.5f %e\n","interp mog",finergrid(iintmean),intmean);
      fflush(stdout);

      if(intmean>maxmog) {
	// Trial profile
	std::vector<coprof_t> intcpl(cpl);
	intcpl[addam].start=finergrid(iintmean)-pw/2.0;
	intcpl[addam].end=finergrid(iintmean)+pw/2.0;
	intcpl[addam].exps=move_exps(pexp,intcpl[addam].start);

	// Calculate real mog
	double intmog;
	ValueType intval;
	try {
	  // Compute value
	  intval=compute_value(intcpl);
	  // and mog
	  intmog=compute_mog(intval,curval);
	} catch(...) {
	  // Interpolation calculation failed.
	  intmog=0.0;
	}

	printf("%11s %8s %e\n","real value","",intmog);

	// Did interpolation yield a better value?
	if(intmog > maxmog) {
	  printf("Interpolation accepted.\n");

	  // Switch values.
	  cpl=intcpl;
	  curval=intval;
	} else {
	  printf("Interpolation rejected.\n");
	  
	  // Update values
	  cpl=cpls[imax];
	  curval=vals[imax];
	}
	fflush(stdout);

      } else {
	printf("Interpolation wouldn't result in a better value.\n");
	fflush(stdout);
	
	// Update values
	cpl=cpls[imax];
	curval=vals[imax];
      }
    } else {
      // Update values
      cpl=cpls[imax];
      curval=vals[imax];
    }
    
    printf("\nOptimal %c shell is: % .3f ... % .3f (%i funcs) with tolerance %e, mog = %e. (%s)\n\n",shell_types[addam],cpl[addam].start,cpl[addam].end,(int) cpl[addam].exps.size(),cpl[addam].tol,maxmog,tpol.elapsed().c_str());
    fflush(stdout);
    
    return maxmog;
  }
  
  /**
   * Scan stability of an existing shell by placing in an additional exponent.
   *
   * Parameters
   * scanam:    angular momentum to scan
   * minpol:    ~ minimal allowed value for polarization exponent (log10 scale)
   * maxpol:    ~ maximal allowed value for polarization exponent (log10 scale)
   * dpol:      amount of points used per completeness range
   * polinterp: toggle interpolation of mog to find maximum
   * cotol:     deviation from completeness
   * nx:        amounts of exponents to place on the shell
   */
  double scan_profile(std::vector<coprof_t> & cpl, ValueType & curval, int scanam, double minpol, double maxpol, double dpol) {
   
    Timer t;
    Timer tpol;

    if(dpol<=0) {
      std::ostringstream oss;
      oss << "Invalid value " << dpol << " for DPol.\n";
      throw std::runtime_error(oss.str());
    }

    // Compute step size
    double step;
    {
      double nextw;
      maxwidth_exps_table(scanam,cpl[scanam].tol,cpl[scanam].exps.size()+1,nextw,OPTMOMIND);
      double curw;
      maxwidth_exps_table(scanam,cpl[scanam].tol,cpl[scanam].exps.size(),curw,OPTMOMIND);
      step=nextw-curw;
    }

    // Is there anything to do?
    if(minpol > cpl[scanam].start-step && maxpol < cpl[scanam].end+step) {
      printf("%c shell already spans scanning limits.\n",shell_types[scanam]);
      return 0.0;
    }

    // Width to check is
    double ww=maxpol-minpol;
    // Amount of points to use
    size_t Np=1+round(ww/step);
    // Actual width is
    double width=step*Np;

    // so actual minimum and maximum are
    double minp=minpol;
    double maxp=maxpol+(width-ww);

    // Starting points
    arma::vec startp=arma::linspace(minp,maxp,Np);

    // Trial mogs
    arma::vec mogs(startp.n_elem);
    mogs.zeros();
    
    printf("Scanning for stability of %c shell.\n", shell_types[scanam]);
    printf("Spacing between points is %.5f.\n\n",step);
    
    printf("%11s %8s %12s %8s\n","trial","exponent","mog","t (s)");
    fflush(stdout);
    
    static size_t iter=0;

    // Get elemental libraries
    const std::vector<ElementBasisSet> els=form_basis(cpl).get_elements();

    // Perform scan
    Timer tptot;
    for(size_t iexp=0;iexp<startp.n_elem;iexp++) {
      Timer tp;

      // Check if we are inside filled completeness profile
      bool inside=true;
      if(startp(iexp)+step < cpl[scanam].start)
	inside=false;
      if(startp(iexp)-step > cpl[scanam].end)
	inside=false;
      if(inside) {
	mogs(iexp)=0.0;
	printf("%5i/%-5i % 7.5f %e %8.2f\n",(int) iexp+1, (int) startp.n_elem, startp(iexp), mogs(iexp), tp.get());
	fflush(stdout);
	continue;
      }

      // Trial basis set
      BasisSetLibrary trbas;
      // Add the function
      for(size_t iel=0;iel<els.size();iel++) {
	ElementBasisSet elbas(els[iel]);

	FunctionShell sh(scanam);
	sh.add_exponent(1.0,startp(iexp));
	elbas.add_function(sh);
	
	// and the element to the new library
	trbas.add_element(elbas);
      }	

      // Compute trial value
      ValueType trval;
      try {
	trval=compute_value(trbas);
	// and store the mog
	mogs(iexp)=compute_mog(trval,curval);

      } catch(...) {
	// Calculation failed, set zero mog.
	mogs(iexp)=0.0;
      }
    
      printf("%5i/%-5i % 7.5f %e %8.2f\n",(int) iexp+1, (int) startp.n_elem, startp(iexp), mogs(iexp), tp.get());
      fflush(stdout);
    }

    { // Save out the results of the mog scan
      // Filename to use
      std::ostringstream fname;
      fname << "scanmog_" << shell_types[scanam] << "_" << iter << ".dat";
      
      arma::mat savemog(mogs.n_rows,2);
      savemog.col(0)=startp;
      savemog.col(1)=mogs;
      savemog.save(fname.str(),arma::raw_ascii);
      
      iter++;
    }

    // Find maximum mog
    arma::uword imax;
    double maxmog=mogs.max(imax);

    printf("\n");
    printf("%11s % 7.5f %e\n","max mog",startp(imax),maxmog);
    fflush(stdout);

    double moved;

    if(maxmog>0.0) {
      // Adjust profile.
      if(startp(imax) < cpl[scanam].start) {
	// Current width is
	double curw=cpl[scanam].end-cpl[scanam].start;
	// Necessary width is
	double nw=cpl[scanam].end-startp(imax);
	// Get real width
	double realw(nw);
	arma::vec exps=span_width(scanam,cpl[scanam].tol,realw,cpl[scanam].exps.size());
	
	// Adjust profile
	cpl[scanam].start-=realw-curw;
	cpl[scanam].exps=move_exps(exps,cpl[scanam].start);
	moved=-(realw-curw);

      } else if(startp(imax) > cpl[scanam].end) {
	// Current width is
	double curw=cpl[scanam].end-cpl[scanam].start;
	// Necessary width is
	double nw=startp(imax)-cpl[scanam].start;
	// Get real width
	double realw(nw);
	arma::vec exps=span_width(scanam,cpl[scanam].tol,realw,cpl[scanam].exps.size());
	
	// Adjust profile
	cpl[scanam].end+=realw-curw;
	cpl[scanam].exps=move_exps(exps,cpl[scanam].start);
	moved=+(realw-curw);

      } else {
	throw std::runtime_error("Possible bug in scan_limits - maximum inside profile!\n");
      }

      if(moved>0.0)
	printf("\n%c upper limit should be moved by % .3f (% .3f steps), mog = %e. (%s)\n\n",shell_types[scanam],moved,moved/step,maxmog,tpol.elapsed().c_str());
      else
	printf("\n%c lower limit should be moved by % .3f (% .3f steps), mog = %e. (%s)\n\n",shell_types[scanam],-moved,-moved/step,maxmog,tpol.elapsed().c_str());
      fflush(stdout);
      
      // Update current value
      curval=compute_value(cpl);
    }
    
    return maxmog;
  }
  
  /// Extend the current shells until mog < tau. Returns maximal mog
  double extend_profile(std::vector<coprof_t> & cpl, ValueType & curval, double tau, bool domiddle=true, bool strict=true, int nxadd=1) {
    printf("\n\n%s\n",print_bar("PROFILE EXTENSION").c_str());
    printf("Final tolerance is %e.\n",tau);
    fflush(stdout);

    if(tau<=0.0)
      throw std::runtime_error("Tolerance must be positive!\n");

    // Trials, consisting of either the move of the starting or the ending point
    std::vector< std::vector<coprof_t> > trials(maxam(cpl)+1);
    
    // Trial values
    std::vector<ValueType> trvals(maxam(cpl)+1);
    
    // Mogs
    arma::vec trmog=DBL_MAX*arma::ones(1,maxam(cpl)+1);
    arma::ivec dir(maxam(cpl)+1);
    dir.zeros();

    // Use old mog?
    arma::ivec oldmog(maxam(cpl)+1);
    oldmog.zeros();
    
    while(true) {
      // Recalculate everything every iteration
      if(strict)
	oldmog.zeros();
      
      // Was mog computed?
      arma::ivec computed(maxam(cpl)+1);
      computed.zeros();
      
      Timer ttot;

      // Separator
      printf("\n");
      fflush(stdout);

      for(int am=0;am<=maxam(cpl);am++) {
	Timer t;

	// Form trials
	std::vector<coprof_t> left(cpl);
	std::vector<coprof_t> middle(cpl);
	std::vector<coprof_t> right(cpl);

	// Get exponents
	printf("Determining exponents for %c shell ... ",shell_types[am]);
	fflush(stdout);

	double step;

	double width;
	arma::vec exps=maxwidth_exps_table(am,cpl[am].tol,cpl[am].exps.size()+nxadd,width,OPTMOMIND);
      
	step=width-(cpl[am].end-cpl[am].start);
	left[am].start=cpl[am].start-step;
	left[am].exps=move_exps(exps,left[am].start);
      
	middle[am].start=cpl[am].start-step/2.0;
	middle[am].end=cpl[am].end+step/2.0;
	middle[am].exps=move_exps(exps,middle[am].start);

	right[am].end=cpl[am].end+step;
	right[am].exps=move_exps(exps,right[am].start);
      
	printf("step size is %7.5f (%s).\n",step,t.elapsed().c_str());
	fflush(stdout);
	t.set();

	if(!oldmog(am)) {
	  
	  printf("Computing extension of %c shell (%2i funcs)",shell_types[am],(int) cpl[am].exps.size());
	  fflush(stdout);

	  // Compute values
	  ValueType lval(curval);
	  ValueType mval(curval);
	  ValueType rval(curval);

	  bool lvalfail=false, mvalfail=false, rvalfail=false;
      
	  try {
	    lval=compute_value(left);
	  } catch(std::runtime_error) {
	    lvalfail=true;
	  }

	  if(domiddle) {
	    try {
	      mval=compute_value(middle);
	    } catch(std::runtime_error) {
	      mvalfail=true;
	    }
	  } else
	    mvalfail=true;
	
	  try {
	    rval=compute_value(right);
	  } catch(std::runtime_error) {
	    rvalfail=true;
	  }	  
      
	  // and mogs
	  double lmog=0.0, mmog=0.0, rmog=0.0;
	
	  if(lvalfail) {
	    lmog=0.0;
	  } else {
	    lmog=compute_mog(lval,curval);
	  }

	  if(mvalfail) {
	    mmog=0.0;
	  } else {
	    mmog=compute_mog(mval,curval);
	  }

	  if(rvalfail) {
	    rmog=0.0;
	  } else {
	    rmog=compute_mog(rval,curval);
	  }

	  double ammog=std::max(std::max(lmog,rmog),mmog);

	  // Check which trial to try.
	  if(lmog==ammog) {
	    // Move left end
	    trmog[am]=lmog;
	    trvals[am]=lval;
	    trials[am]=left;
	    dir[am]=-1;
	  } else if(mmog==ammog) {
	    // Move symmerically
	    trmog[am]=mmog;
	    trvals[am]=mval;
	    trials[am]=middle;
	    dir[am]=0;
	  } else {
	    // Move right end
	    trmog[am]=rmog;
	    trvals[am]=rval;
	    trials[am]=right;
	    dir[am]=1;
	  }
	
	  printf(", mog is %e (%s)\n",trmog[am],t.elapsed().c_str());
	  computed(am)=true;

	} else {
	  printf("Reusing   extension of %c shell (%2i funcs)",shell_types[am],(int) cpl[am].exps.size());
	  printf(", mog is %e (%s)\n",trmog[am],t.elapsed().c_str());
	}

	fflush(stdout);
      }
      
      // Figure out maximal mog
      int maxind=maxam(cpl);
      // Do search manually to favor large am here if degeneracies occur
      double maxmog=trmog(maxind);
      for(int am=maxam(cpl)-1;am>=0;am--)
	if(trmog(am)>maxmog) {
	  maxind=am;
	  maxmog=trmog(maxind);
	}

      // Sanity check - did we compute the maximal value now?
      if(!computed(maxind)) {
	// No - it's an old value. Reuse values for what we've
	// calculated now and redo the others.
	oldmog=computed;
	printf("Maximal mog for %c shell, recomputing.\n",shell_types[maxind]);
	fflush(stdout);
	continue;
      } else {
	// Update oldmog
	oldmog+=computed;
	// but maxind is reset
	oldmog(maxind)=false;
      }
      
      // Converged?
      if(maxmog < tau) {
	printf("Maximal mog is %e, converged.\n\n",maxmog);
	fflush(stdout);

	if(!strict && arma::min(computed)==0) {
	  // Do a final iteration in strict mode
	  strict=true;
	  continue;
	}

	print_limits(cpl,"Final limits");

	return maxmog;
	
      } else {
	// Still stuff to add
	
	// Update profile
	cpl=trials[maxind];
	// Update current value
	curval=trvals[maxind];
	
	if(dir[maxind]==1)
	  printf("Moved   ending point of %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog %e (%s).\n",shell_types[maxind],cpl[maxind].start,cpl[maxind].end,(int) cpl[maxind].exps.size(),cpl[maxind].tol,maxmog,ttot.elapsed().c_str());
	else if(dir[maxind]==-1)
	  printf("Moved starting point of %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog %e (%s).\n",shell_types[maxind],cpl[maxind].start,cpl[maxind].end,(int) cpl[maxind].exps.size(),cpl[maxind].tol,maxmog,ttot.elapsed().c_str());
	else if(dir[maxind]==0)
	  printf("Moved    both limits of %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog %e (%s).\n",shell_types[maxind],cpl[maxind].start,cpl[maxind].end,(int) cpl[maxind].exps.size(),cpl[maxind].tol,maxmog,ttot.elapsed().c_str());
	fflush(stdout);

	// Update references
	update_reference(cpl);
      }
    }
  }


  /// Tighten the shells until mog < tau. Returns maximal mog.
  double tighten_profile(std::vector<coprof_t> & cpl, ValueType & curval, double tau, bool strict=true, int nxadd=1) {
    printf("\n\n%s\n",print_bar("PROFILE TIGHTENING").c_str());
    printf("Final tolerance is %e.\n",tau);
    fflush(stdout);

    // Trials for each am
    std::vector< std::vector<coprof_t> > trials(maxam(cpl)+1);
    
    // Trial values
    std::vector<ValueType> trvals(maxam(cpl)+1);
    
    // Mogs
    arma::vec trmog=DBL_MAX*arma::ones(1,maxam(cpl)+1);    

    // Use old mog?
    arma::ivec oldmog(maxam(cpl)+1);
    oldmog.zeros();
    
    while(true) {
      // Recalculate everything every iteration
      if(strict)
	oldmog.zeros();
      
      // Was mog computed?
      arma::ivec computed(maxam(cpl)+1);
      computed.zeros();

      Timer ttot;

      // Separator
      printf("\n");
      fflush(stdout);

      for(int am=0;am<=maxam(cpl);am++) {
	Timer t;

	// Form trials
	std::vector<coprof_t> add(cpl);

	// Get exponents
	printf("Determining exponents for %c shell ... ",shell_types[am]);
	fflush(stdout);

	add[am].exps=optimize_completeness(am,add[am].start,add[am].end,add[am].exps.size()+nxadd,OPTMOMIND,false,&add[am].tol);
	bool tryadd=(add[am].tol>=MINTAU);

	printf("(%s)\n",t.elapsed().c_str());
	t.set();
      
	printf("Computing tightening of %c shell (%2i funcs)",shell_types[am],(int) cpl[am].exps.size());
	fflush(stdout);
	  
	if(!oldmog(am)) {
	  

	  // Compute values
	  ValueType aval(curval);
	  bool avalfail=false;

	  if(tryadd) {
	    try {
	      aval=compute_value(add);
	    } catch(std::runtime_error) {
	      avalfail=true;
	    }
	  }
	  
	  // and mogs
	  double amog;
      
	  if(avalfail) {
	    amog=0.0;
	  } else {
	    amog=compute_mog(aval,curval);
	  }
	  
	  trmog[am]=amog;
	  trvals[am]=aval;
	  trials[am]=add;

	  computed(am)=true;
	  printf(", mog is %e (%s)\n",trmog[am],t.elapsed().c_str());

	} else {
	  printf("Reusing   tightening of %c shell (%2i funcs)",shell_types[am],(int) cpl[am].exps.size());
	  printf(", mog is %e (%s)\n",trmog[am],t.elapsed().c_str());
	}

	fflush(stdout);
      }
    
      // Figure out maximal mog
      arma::uword maxind;
      double maxmog=trmog.max(maxind);

      // Sanity check - did we compute the maximal value now?
      if(!computed(maxind)) {
	// No - it's an old value. Reuse values for what we've
	// calculated now and redo the others.
	oldmog=computed;
	printf("Maximal mog for %c shell, recomputing.\n",shell_types[maxind]);
	fflush(stdout);
	continue;
      } else {
	// Update oldmog
	oldmog+=computed;
	// but maxind is reset
	oldmog(maxind)=false;
      }
    
      // Converged?
      if(maxmog < tau) {
	printf("Maximal mog is %e, converged.\n\n",maxmog);
	fflush(stdout);

	if(!strict && arma::min(computed)==0) {
	  // Do a final iteration in strict mode
	  strict=true;
	  continue;
	}

	print_limits(cpl,"Final tolerances");
	return maxmog;
      
      } else {
	// Still stuff to add
      
	// Update profile
	cpl=trials[maxind];
	// Update current value
	curval=trvals[maxind];
      
	printf("Added exponent to %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog = %e (%s).\n",shell_types[maxind],cpl[maxind].start,cpl[maxind].end,(int) cpl[maxind].exps.size(),cpl[maxind].tol,maxmog,ttot.elapsed().c_str());
	fflush(stdout);
      }
    
      // Update references
      update_reference(cpl);
    }
  }

  /// Get SCF free-atom angular momentum
  virtual int atom_am()=0;

  /// Print value
  virtual void print_value(const ValueType & value, std::string msg)=0;

  /// Extend the basis set till the CBS limit.
  void find_cbs_limit(std::vector<coprof_t> & cpl, ValueType & curval, double cotol, double minpol, double maxpol, double dpol, bool domiddle=true, bool strict=true, bool scan=true, bool polinterp=true, int nxpol=1, bool doadd=true, int nxext=1, int am_max=max_am, double delta=0.9) {
    // Amount of polarization shells
    int npol=maxam(cpl)-atom_am();
    
    // Compute initial value
    curval=compute_value(form_basis(cpl));
    print_value(curval,"Starting value");
	
    // Compute mog of next polarization shell
    double tau;
    {
      std::vector<coprof_t> polcpl(cpl);
      ValueType polval(curval);
      tau=check_polarization(polcpl,polval,am_max,minpol,maxpol,dpol,polinterp,cotol,nxpol);
    }

    while(true) {
      // Extend existing shells
      double extmog=extend_profile(cpl,curval,tau,domiddle,strict,nxext);
      // Tighten existing shells
      if(doadd) {
	double amog=tighten_profile(cpl,curval,tau,strict,1);
	extmog=std::max(extmog,amog);
      }
     
      // Compute mog of further polarization shell
      std::vector<coprof_t> polcpl(cpl);
      arma::mat polval(curval);
      double polmog=check_polarization(polcpl,polval,am_max,minpol,maxpol,dpol,polinterp,cotol,nxpol);

      // What is the maximum mog? Because we are here, std::max(monomog,anmog)<=tau.
      if(extmog >= polmog) {
	// Need to expand existing shells.
	tau=exp((1.0-delta)*log(extmog) + delta*log(polmog));

      } else {
	// Here extmog < polmog.
	// Going to add new polarization shell in next iteration; we are converged here.

	if(scan) {
	  printf("\n\n%s\n",print_bar("SHELL STABILITY").c_str());
	  fflush(stdout);

	  // Before adding new polarization shell, scan the stability of the existing shells.
	  std::vector< std::vector<coprof_t> > scancpl(maxam(cpl)+1);
	  std::vector<ValueType> scanval(maxam(cpl)+1);
	  arma::vec scanmog(maxam(cpl)+1);
	  for(int am=0;am<=maxam(cpl);am++) {
	    scancpl[am]=cpl;
	    scanval[am]=curval;
	    scanmog[am]=scan_profile(scancpl[am],scanval[am],am,minpol,maxpol,dpol);
	  }

	  // Get maximum value
	  arma::uword maxind;
	  double scanm=scanmog.max(maxind);

	  if(scanm>=polmog) {
	    // Unstability detected.
	    cpl=scancpl[maxind];
	    curval=scanval[maxind];
	    printf("\n\nUnstability detected on %c shell, restarting extension.\n",shell_types[maxind]);

	    print_limits(cpl,"Current limits");
	    curval.print("Current value");

	    continue;
	  }
	}

	// Switch to polarized basis.
	cpl=polcpl;
	curval=polval;
	npol++;

	printf("\n\n%s\n",print_bar("CONVERGENCE ACHIEVED",'#').c_str());
	printf("\nConverged: extension mog was %e, polarization mog %e.\n",extmog,polmog);

	printf("\nFinal composition for %i polarization shells (tau = %e):\n",npol,polmog);
	print_limits(cpl);
	print_value(curval,"Current value");

	// Save basis set
	{
	  BasisSetLibrary baslib=form_basis(cpl);
	  char fname[180];
	  sprintf(fname,"un-co-ref-%i.gbs",npol);
	  baslib.save_gaussian94(fname);

	  // Save completeness range
	  sprintf(fname,"completeness-%i.dat",npol);
	  save_limits(cpl,fname);
	}
	
	// Converged?
	if(npol==am_max) {
	  // Check shell extension
	  extend_profile(cpl,curval,polmog,domiddle,nxext);
	  if(doadd)
	    tighten_profile(cpl,curval,polmog);
	  break;
	}
	
	// Reduce tau.
	tau=std::min(tau,extmog);
      }
    }

    print_limits(cpl,"\nFinal composition:");

    // Save basis set
    {
      BasisSetLibrary baslib=form_basis(cpl);
      char fname[180];
      sprintf(fname,"un-co-ref.gbs");
      baslib.save_gaussian94(fname);
    }
    
    // Save completeness range
    save_limits(cpl,"completeness.dat");
  }

  /// Reduce the profile until only a single function is left on the highest am shell (polarization / correlation consistence)
  double reduce_profile(std::vector<coprof_t> & cpl, ValueType & curval, const ValueType & refval, double tol=0.0, bool domiddle=true, bool strict=true) {
    printf("\n\n%s\n",print_bar("PROFILE REDUCTION").c_str());
    fflush(stdout);
	
    if(tol==0.0)
      printf("Running until would drop last function of %c shell.\n",shell_types[maxam(cpl)]);
    else
      printf("Running until mog >= %e.\n",tol);
    fflush(stdout);

    // Do the reduction.
    double minmog=0.0;

    // Trials, consisting of either the move of the starting or the ending point or the removal of an exponent
    std::vector< std::vector<coprof_t> > trials(maxam(cpl)+1);
    
    // Trial values
    std::vector<ValueType> trvals(maxam(cpl)+1);
    
    // Mogs
    arma::vec trmog=DBL_MAX*arma::ones(1,maxam(cpl)+1);
    arma::ivec dir(maxam(cpl)+1);
    dir.zeros();

    // Use old mog?
    arma::ivec oldmog(maxam(cpl)+1);
    oldmog.zeros();

    while(true) {
      // Recalculate everything every iteration
      if(strict)
	oldmog.zeros();
      
      // Was mog computed?
      arma::ivec computed(maxam(cpl)+1);
      computed.zeros();
      
      Timer ttot;

      // Separator
      printf("\n");
      fflush(stdout);

      minmog=0.0;
      for(int am=0;am<=maxam(cpl);am++) {
	Timer t;

	// Form trials
	std::vector<coprof_t> left(cpl);
	std::vector<coprof_t> middle(cpl);
	std::vector<coprof_t> right(cpl);
	std::vector<coprof_t> del(cpl);

	// Get exponents
	printf("Determining exponents for %c shell ... ",shell_types[am]);
	fflush(stdout);

	double step;

	if(cpl[am].exps.size()>1) {
	  double width;
	  arma::vec exps=maxwidth_exps_table(am,cpl[am].tol,cpl[am].exps.size()-1,width,OPTMOMIND);

	  step=cpl[am].end-cpl[am].start-width;
	  left[am].start=cpl[am].start+step;
	  left[am].exps=move_exps(exps,left[am].start);

	  middle[am].start=cpl[am].start+step/2.0;
	  middle[am].end=cpl[am].end-step/2.0;
	  middle[am].exps=move_exps(exps,middle[am].start);

	  right[am].end=cpl[am].end-step;
	  right[am].exps=move_exps(exps,right[am].start);

	  del[am].exps=optimize_completeness(am,del[am].start,del[am].end,del[am].exps.size()-1,OPTMOMIND,false,&del[am].tol);

	} else {
	  step=left[am].end-left[am].start;

	  // Dummy tries
	  left[am].start=0.0;
	  left[am].end=0.0;
	  left[am].exps.clear();

	  middle[am].start=0.0;
	  middle[am].end=0.0;
	  middle[am].exps.clear();

	  right[am].start=0.0;
	  right[am].end=0.0;
	  right[am].exps.clear();

	  del[am].start=0.0;
	  del[am].end=0.0;
	  del[am].exps.clear();
	}
	printf("step size is %7.5f (%s).\n",step,t.elapsed().c_str());
	t.set();

	if(!oldmog(am)) {
	  printf("Computing reduction of %c shell (%2i funcs)",shell_types[am],(int) cpl[am].exps.size());
	  fflush(stdout);

	  // Compute values
	  ValueType lval(curval);
	  ValueType mval(curval);
	  ValueType rval(curval);
	  ValueType dval(curval);

	  bool lvalfail=false, mvalfail=false, rvalfail=false, dvalfail=false;
	
	  if(cpl[am].exps.size()>1) {
	    try {
	      lval=compute_value(left);
	    } catch(std::runtime_error) {
	      lvalfail=true;
	    }

	    if(domiddle) {
	      try {
		mval=compute_value(middle);
	      } catch(std::runtime_error) {
		mvalfail=true;
	      }
	    } else
	      mvalfail=true;

	    try {
	      rval=compute_value(right);
	    } catch(std::runtime_error) {
	      rvalfail=true;
	    }
	  
	  } else {
	    // Trying to drop last exponent, make these dummy tries
	    // since they will be equal to the one below
	    lvalfail=true;
	    mvalfail=true;
	    rvalfail=true;
	  }
	
	  // Safeguard
	  if(cpl[am].exps.size())
	    try {
	      dval=compute_value(del);
	    } catch(std::runtime_error) {
	    dvalfail=true;
	  }
	  else dvalfail=true;
	
	  // and mogs
	  double lmog, mmog, rmog, dmog;
      
	  if(lvalfail) {
	    lmog=DBL_MAX;
	  } else {
	    lmog=compute_mog(lval,refval);
	  }

	  if(mvalfail) {
	    mmog=DBL_MAX;
	  } else {
	    mmog=compute_mog(mval,refval);
	  }

	  if(rvalfail) {
	    rmog=DBL_MAX;
	  } else {
	    rmog=compute_mog(rval,refval);
	  }

	  if(dvalfail) {
	    dmog=DBL_MAX;
	  } else {
	    dmog=compute_mog(dval,refval);
	  }

	  minmog=std::min(std::min(lmog,rmog),std::min(mmog,dmog));

	  // Check which trial to try.
	  if(lmog==minmog) {
	    // Move left end
	    trmog[am]=lmog;
	    trvals[am]=lval;
	    trials[am]=left;
	    dir[am]=1;
	  } else if(mmog==minmog) {
	    // Move symmerically
	    trmog[am]=mmog;
	    trvals[am]=mval;
	    trials[am]=middle;
	    dir[am]=2;
	  } else if(dmog==minmog) {
	    // Drop a function.
	    trmog[am]=dmog;
	    trvals[am]=dval;
	    trials[am]=del;
	    dir[am]=0;
	  } else {
	    // Move right end
	    trmog[am]=rmog;
	    trvals[am]=rval;
	    trials[am]=right;
	    dir[am]=-1;
	  }
      
	  printf(", mog is %e (%s)\n",trmog[am],t.elapsed().c_str());
	  computed(am)=true;
	  fflush(stdout);

	} else {
	  printf("Reusing   reduction of %c shell (%2i funcs)",shell_types[am],(int) cpl[am].exps.size());
	  printf(", mog is %e (%s)\n",trmog[am],t.elapsed().c_str());
	  fflush(stdout);
	}
      }
      
      // Figure out minimal mog
      arma::uword minind;
      minmog=trmog.min(minind);

      // Sanity check - did we compute the minimal value now?
      if(!computed(minind)) {
	// No - it's an old value. Reuse values for what we've
	// calculated now and redo the others.
	oldmog=computed;
	printf("Minimal mog for %c shell, recomputing.\n",shell_types[minind]);
	fflush(stdout);
	continue;
      } else {
	// Update oldmog
	oldmog+=computed;
	// but maxind is reset
	oldmog(minind)=false;
      }
    
      // Converged?
      if(tol==0.0 && ((int) minind == maxam(cpl)) && (cpl[minind].exps.size()==1)) {
	printf("Would remove last exponent of %c shell with mog %e, converged.\n\n",shell_types[minind],minmog);

	if(!strict && arma::min(computed)==0) {
	  // Do a final iteration in strict mode
	  strict=true;
	  continue;
	}

	print_limits(cpl,"Final limits");
	fflush(stdout);

	return minmog;

      } else if(tol>0.0 && minmog>tol) {
      
	printf("Minimal mog is %e for %c shell, converged.\n\n",minmog,shell_types[minind]);
	print_limits(cpl,"Final limits");
	fflush(stdout);
	return minmog;
      
      } else {
	// Still stuff to remove.
      
	// Update profile
	cpl=trials[minind];
	// Update current value
	curval=trvals[minind];
      
	if(dir[minind]==-1)
	  printf("Moved   ending point of %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog %e (%s).\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());
	else if(dir[minind]==1)
	  printf("Moved starting point of %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog %e (%s).\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());
	else if(dir[minind]==2)
	  printf("Moved    both limits of %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog %e (%s).\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());
	else if(dir[minind]==0)
	  printf("Dropped exponent   from %c shell, range is now % .3f ... % .3f (%2i funcs), tol = %e, mog %e (%s).\n",shell_types[minind],cpl[minind].start,cpl[minind].end,(int) cpl[minind].exps.size(),cpl[minind].tol,minmog,ttot.elapsed().c_str());

	fflush(stdout);
      }

      // Update references
      update_reference(cpl);
    }

    return minmog;
  }

  void reduce_basis(const std::vector<coprof_t> & cbscpl, std::vector<coprof_t> & cpl, bool strict=true, bool domiddle=true, bool docontr=true, bool restr=true, double nelcutoff=0.01) {
    // Reference value
    const ValueType cbsval(compute_value(form_basis(cbscpl)));

    print_limits(cbscpl,"CBS limit basis");
    print_limits(cpl,"Starting point");

    // Amount of polarization functions
    int npol=maxam(cpl)-atom_am();

    // Current value
    ValueType curval(compute_value(form_basis(cpl)));

    while(npol>=1) {
      // Reduce the profile
      double tau=reduce_profile(cpl,curval,cbsval,0.0,domiddle,strict);
      
      printf("Final composition for %i polarization shells (mog = %e):\n",npol,tau);
      print_limits(cpl);
      curval.print("Current value");

      // Save basis set
      {
	char fname[180];
	sprintf(fname,"un-co-%i.gbs",npol);
	form_basis(cpl).save_gaussian94(fname);

	sprintf(fname,"reduced-%i.dat",npol);
	save_limits(cpl,fname);
      }

      // Contract the basis
      if(docontr) {
	// Use CBS value as reference for contraction
	// ValueType contrref(cbsval);

	// Use current value as reference for contraction
	ValueType contrref(curval);

	// Contract the basis, compute mog using P-orthogonalization
	BasisSetLibrary contrbas=contract_basis(cpl,contrref,tau,nelcutoff,true,restr);

	char fname[180];
	// General contractions
	sprintf(fname,"co-general-%i.gbs",npol);
	contrbas.save_gaussian94(fname);

	// Segmented contractions
	contrbas.P_orthogonalize();
	sprintf(fname,"co-segmented-%i.gbs",npol);
	contrbas.save_gaussian94(fname);
      }
      
      // Erase polarization shell
      int delam=maxam(cpl);
      cpl[delam].start=0.0;
      cpl[delam].end=0.0;
      cpl[delam].exps.clear();
      npol--;
    }
  }



  /// Update the contraction coefficients, return the amount of electrons in each am shell.
  virtual std::vector<size_t> update_contraction(const std::vector<coprof_t> & cpl, double cutoff)=0;

  /// Contract basis set. nel holds amount of states of each angular momentum. P toggles intermediate P-orthogonalization; returned basis is always generally contracted.
  BasisSetLibrary contract_basis(const std::vector<coprof_t> & cpl, const ValueType & refval, double tol, double nelcutoff, bool Porth=true, bool restr=true) {

    printf("\n\n%s\n",print_bar("BASIS CONTRACTION").c_str());
    fflush(stdout);

    // Update the coefficients, get amount of electrons in each am shell
    std::vector<size_t> nel=update_contraction(cpl,nelcutoff);

    Timer ttot;

    // How many functions to contract, counting from the tightest functions.
    // Initialize with amount of contractions on each shell; this
    // doesn't yet affect the variational freedom of the basis set.
    std::vector<size_t> contract(nel);
    
    // Do the contraction.
    double minmog=0.0;

    while(true) {
      Timer tc;

      // Compute mogs of contracting one more function on each shell
      arma::vec mogs(contract.size());
      for(size_t am=0;am<contract.size();am++) {
	// Check if there are free functions left
	bool free=true;

	// No free functions available.
	if(contract[am] >= cpl[am].exps.size())
	  free=false;

	if(restr && am+1<cpl.size()) {
	  // Check that the shell has more functions than the one above.
	  int Ncur=cpl[am].exps.size()-contract[am];
	  int Nnext=cpl[am+1].exps.size();
	  if(am+1<contract.size())
	    Nnext-=contract[am+1];

	  // If the amount of functions is the same or smaller, don't try to contract.
	  if(Ncur<=Nnext) {
	    printf("Ncur = %i, Nnext = %i\n",Ncur,Nnext);
	    printf("%c shell limited due to %c shell.\n",shell_types[am],shell_types[am+1]);
	    free=false;
	  }
	}

	// We still have free functions left, calculate mog
	if(free) {
	  Timer tl;

	  // Trial contraction
	  std::vector<size_t> trial(contract);
	  trial[am]++;
	  // Trial basis
	  BasisSetLibrary baslib=form_basis(cpl,trial,Porth);	 

	  // Trial value
	  ValueType trval;
	  double trmog;
	  try {
	    trval=compute_value(baslib);
	    trmog=compute_mog(refval,trval);
	  } catch(std::runtime_error) {
	    trmog=DBL_MAX;
	  }

	  // Store mog
	  mogs[am]=trmog;

	  print_scheme(baslib,40);
	  printf("%e %s\n",mogs[am],tl.elapsed().c_str());
	  fflush(stdout);

	} else
	  // No free primitives left.
	  mogs[am]=DBL_MAX;
      }
    
      // Determine minimal mog
      minmog=DBL_MAX;
      for(size_t am=0;am<mogs.size();am++)
	minmog=std::min(minmog,mogs[am]);
    
      // Accept move?
      if(minmog<=tol) {
	for(size_t am=mogs.size()-1;am<mogs.size();am--)
	  if(mogs[am]==minmog) {
	    printf("Contracting a %c function, mog is %e (%s).\n\n",shell_types[am],minmog,tc.elapsed().c_str());
	    fflush(stdout);
	    contract[am]++;
	    break;
	  }
      } else {
	// Converged
	if(minmog==DBL_MAX)
	  printf("No functions left to contract. Scheme is ");
	else
	  printf("Minimal mog is %e, converged (%s). Scheme is ",minmog,tc.elapsed().c_str());
	break;
      }
    }
  
    // Compile library
    BasisSetLibrary ret=form_basis(cpl,contract,false);
  
    print_scheme(ret);
    printf(".\nContraction took %s.\n\n",ttot.elapsed().c_str());
    fflush(stdout);
  
    return ret;
  }
};

#endif