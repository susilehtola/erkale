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



/* Routines for constructing energy-optimized basis sets. */

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include "basis.h"
#include "elements.h"
#include "xyzutils.h"
#include "scf.h"
#include "stringutil.h"
#include "tempered.h"
#include "timer.h"

// Minimization routines
extern "C" {
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
}


/// Define a shell of tempered exponents
typedef struct {
  /// Angular momentum
  int am;
  /// Number of tempered exponents
  int Nf;
} tempered_t;

/// Define basis set for an atom
typedef struct {
  /// Element type
  std::string atomtype;
  /// Shells of tempered exponents
  std::vector<tempered_t> shells;
} element_t;

/// Define basis set for system
typedef struct {
  /// List of elements
  std::vector<element_t> el;
  /// List of atoms
  std::vector<atom_t> atoms;
} basis_t;

// Method to use? 0=even-tempered, 1=well-tempered, 2=full optimization
int method;
// Use DIIS?
bool usediis=1;
// Maximum step size
double maxstepsize=4;
// Maximum print out frequency 
double maxprintfreq=0.5;

// Lower and upper limit for allowed exponents
double lower_limit=1e-2;
double upper_limit=1e9;

// Clean exponents from too small or too large values
void clean_exponents(std::vector<double> & expn) {
  size_t erased=0;

  for(size_t i=expn.size()-1;i<expn.size();i--)
    if(expn[i]<lower_limit || expn[i]>upper_limit) {
      erased++;
      expn.erase(expn.begin()+i);
    }

  if(erased)
    printf("Removed %lu exponents since allowed limits were breached.\n",erased);

  if(erased && !expn.size()) {
    // Safe-guard - add one exponent
    expn.push_back(1.0);
  }
}

// Compute number of degrees of freedom
size_t compute_ndf(const basis_t & bas) {
  size_t ndf=0;
  
  // Loop over elements
  for(size_t iel=0;iel<bas.el.size();iel++) {
    if(method==0) {
      // Even-tempered
      // Two parameters for each shell, if non-empty
      for(size_t ish=0;ish<bas.el[iel].shells.size();ish++)
	if(bas.el[iel].shells[ish].Nf)
	  ndf+=2;
    } else if(method==1) {
      // Well-tempered
      // Four parameters for each shell, if non-empty
      for(size_t ish=0;ish<bas.el[iel].shells.size();ish++)
	if(bas.el[iel].shells[ish].Nf)
	  ndf+=4;
    } else {
      // Full optimization.
      for(size_t ish=0;ish<bas.el[iel].shells.size();ish++)
	ndf+=bas.el[iel].shells[ish].Nf;
    }
  }
  
  return ndf;
}

// Construct initial starting point for optimization
void starting_point(gsl_vector *v, const basis_t & bas) {
  size_t ind=0;

  int amind, lastam=INT_MAX;

  // Loop over elements
  for(size_t iel=0;iel<bas.el.size();iel++) {
    
    // Reset exponent counter
    amind=0;

    // Loop over the shells of the element
    for(size_t ish=0;ish<bas.el[iel].shells.size();ish++) {

      // Sanity check - only do something if shell has functions
      if(bas.el[iel].shells[ish].Nf==0)
	continue;

      // Reset exponent counter?
      if(bas.el[iel].shells[ish].am!=lastam)
	amind=0;
      lastam=bas.el[iel].shells[ish].am;
      
      if(method==0) {
	// Even-tempered
	// If only one exponent, no need to use two parameters.
	gsl_vector_set(v,ind,amind); // alpha
	gsl_vector_set(v,ind+1,amind/2); // beta

	amind++; // Increase exponent index
	ind+=2; // Increase parameter index
      } else if(method==1) {
	// Well-tempered

	gsl_vector_set(v,ind,amind); // alpha
	gsl_vector_set(v,ind+1,amind/2); // beta
	gsl_vector_set(v,ind+2,0.0); // gamma
	gsl_vector_set(v,ind+3,0.0); // delta
	
	amind++;
	ind+=4;
      } else {
	// Full optimization
	for(int ifunc=0;ifunc<bas.el[iel].shells[ish].Nf;ifunc++) {
	  gsl_vector_set(v,ind,2*amind);
	  ind++;
	  amind++;
	}
      }
    }
  }
}



// Construct starting point for full optimization
void starting_point_full(gsl_vector *v, const gsl_vector *par, const basis_t & bas) {
  size_t ind=0;
  size_t vind=0;

  // Loop over elements
  for(size_t iel=0;iel<bas.el.size();iel++) {
    // Loop over the shells of the element
    for(size_t ish=0;ish<bas.el[iel].shells.size();ish++) {
      // Get exponents for the shell

      std::vector<double> exps;
      // Number of exponents
      size_t Nf=bas.el[iel].shells[ish].Nf;
      if(Nf>0) {
	if(method==0) {
	  // Even-tempered
	  double alpha=exp(gsl_vector_get(par,ind));
	  double beta=exp(gsl_vector_get(par,ind+1));
	  
	  exps=eventempered_set(alpha,beta,Nf);
	  vind+=2;
	} else if(method==1) {
	  // Well-tempered
	  double alpha=exp(gsl_vector_get(par,ind));
	  double beta=exp(gsl_vector_get(par,ind+1));
	  double gamma=exp(gsl_vector_get(par,ind+2));
	  double delta=exp(gsl_vector_get(par,ind+3));
	  vind+=4;
	  
	  exps=welltempered_set(alpha,beta,gamma,delta,Nf);
	}
	
	// Now, add the exponents
	for(size_t iexp=0;iexp<Nf;iexp++) {
	  gsl_vector_set(v,ind,log(exps[iexp]));
	  ind++;
	}
      }
    }
  }
}


Settings get_settings() {
  // Settings to use:
  Settings set(1);
  // Non-verbose operation.
  set.set_bool("Verbose",0);
  // We need a high accuracy in energy.
  set.set_double("DeltaEmax",DBL_EPSILON);
  // Use DIIS?
  set.set_bool("UseDIIS",usediis);

  return set;
}  

// Construct a basis set with given arguments
BasisSetLibrary construct_library(const gsl_vector *v, void *params) {
  // Library to return
  BasisSetLibrary lib;

  // Get parameters
  basis_t *parp=(basis_t *)params;
  basis_t par=*parp;

  // Number of atom types is
  size_t Nat=par.atoms.size();

  // Settings to use
  Settings set=get_settings();

  // Index in v
  size_t vind=0;

  // Add atom types to basis set library.
  for(size_t i=0;i<Nat;i++) {

    // Check if the element is already in the library.
    try {
      ElementBasisSet el=lib.get_element(par.atoms[i].el);
    } catch(std::runtime_error) {
      
      // Was the element found?
      bool found=0;
      
      // Now, loop over the parameters to find out what kind of shell
      // structure we want for the atom.
      for(size_t j=0;j<par.el.size();j++) {
	if(par.atoms[i].el==par.el[j].atomtype) {
	  // Found atom type!
	  found=1;
	  
	  // Construct basis set for this element.
	  ElementBasisSet element(par.atoms[i].el);
	  
	  // Loop over shells:
	  for(size_t k=0;k<par.el[j].shells.size();k++) {
	    // Get number of functions to add
	    int Nf=par.el[j].shells[k].Nf;
	    
	    std::vector<double> exps;
	    
	    // Get exponents
	    if(Nf>0) {
	      if(method==0) {
		// Even-tempered
		double alpha=exp(gsl_vector_get(v,vind));
		double beta=exp(gsl_vector_get(v,vind+1));
		
		exps=eventempered_set(alpha,beta,Nf);
		vind+=2;
	      } else if(method==1) {
		// Well-tempered
		double alpha=exp(gsl_vector_get(v,vind));
		double beta=exp(gsl_vector_get(v,vind+1));
		double gamma=exp(gsl_vector_get(v,vind+2));
		double delta=exp(gsl_vector_get(v,vind+3));
		vind+=4;
		
		exps=welltempered_set(alpha,beta,gamma,delta,Nf);
	      } else {
		// Full optimization
		for(int ifunc=0;ifunc<par.el[j].shells[k].Nf;ifunc++)
		  exps.push_back(exp(gsl_vector_get(v,vind++)));
	      }

	      // Clean out exponents
	      clean_exponents(exps);
	      
	      // Add functions to basis set
	      for(size_t l=0;l<exps.size();l++) {
		// Shell for element
		FunctionShell shell(par.el[j].shells[k].am);
		// Add function
		shell.add_exponent(1.0,exps[l]);
		// Add shell to basis
		element.add_function(shell);
	      }
	    }
	  }
	  
	  // Add element to library
	  lib.add_element(element);
	}
      }

      if(!found) {
	std::ostringstream oss;
	oss << "Could not find element " << par.atoms[i].el << " in basis set!\n";
	throw std::runtime_error(oss.str());
      }
    }
    
  }

  // Print basis
  /*
  lib.sort();
  lib.save_gaussian94("lib.gbs");
  */

  return lib;
}

// Contruct a basis set from the given library
BasisSet construct_basis(const BasisSetLibrary & lib, const basis_t & bas) {
  // Number of atoms is
  size_t Nat=bas.atoms.size();

  // Settings to use
  Settings set=get_settings();

  // Create basis set
  BasisSet basis(Nat,set);

  // and add atoms to basis set
  for(size_t i=0;i<Nat;i++) {
    // Get center
    coords_t cen;
    cen.x=bas.atoms[i].x;
    cen.y=bas.atoms[i].y;
    cen.z=bas.atoms[i].z;

    // Add the nucleus
    basis.add_nucleus(i,cen,get_Z(bas.atoms[i].el),bas.atoms[i].el);
    // and its basis functions
    basis.add_functions(i,cen,lib.get_element(bas.atoms[i].el));
  }
  
  // Finalize basis set
  basis.finalize();

  return basis;
}  

// Construct a basis set with given arguments
BasisSet construct_basis(const gsl_vector *v, void *params) {
  // Get parameters
  basis_t *parp=(basis_t *)params;
  basis_t par=*parp;

  // Construct basis set library
  BasisSetLibrary lib=construct_library(v,params);

  return construct_basis(lib,par);
}


// Evaluate energy of set with parameters (alpha,beta) in v
double eval_tempered(const gsl_vector *v, void *params) {

  // Construct basis set
  BasisSet basis=construct_basis(v,params);

  // Settings to use
  Settings set=get_settings();

  // Get number of electrons
  int Nel=basis.Ztot()-set.get_int("Charge");
  // Set multiplicity
  if(Nel%2==1)
    set.set_int("Multiplicity",2);

  // Solver
  SCF solver(basis,set);
  
  // Energy
  double Etot;
  
  if(set.get_int("Multiplicity")==1 && Nel%2==0) {
    // Closed shell case
    arma::mat C;
    arma::vec E;
    Etot=solver.RHF(C,E);
  } else {
    arma::mat Ca, Cb;
    arma::vec Ea, Eb;
    Etot=solver.UHF(Ca,Cb,Ea,Eb);
  }    

  //  printf("Energy is %.6f\n",Etot);
  // Return energy
  return Etot;
}

bool operator<(const tempered_t & lhs, const tempered_t & rhs) {
  return lhs.am<rhs.am;
}

void sort_shells(std::vector<tempered_t> & shells) {
  stable_sort(shells.begin(),shells.end());
}

std::vector<element_t> load_elements(std::string filename) {
  // Input file
  std::ifstream in(filename.c_str());
  // Returned array
  std::vector<element_t> els;

  while(in.good()) {
    // Read line.
    std::string line=readline(in);

    if(!in.good())
      break;

    // Split it into words
    std::vector<std::string> words=splitline(line);

    // Read shells
    std::vector<tempered_t> shells;
    // Loop over shells
    for(size_t i=1;i<words.size();i++) {

      // Length of entry
      size_t len=words[i].length();

      tempered_t shell;
      // Angular momentum is
      shell.am=find_am(toupper(words[i][len-1]));
      // Number of functions is
      shell.Nf=atoi(words[i].substr(0,len-1).c_str());

      shells.push_back(shell);
    }

    // Sort the shells in order of angular momentum
    sort_shells(shells);

    // Element
    element_t ele;
    ele.atomtype=words[0];
    ele.shells=shells;

    // Add to returned array
    els.push_back(ele);
  }

  return els;
}


BasisSetLibrary optimize_basis(const basis_t & bas, gsl_vector *ss, gsl_vector *x, int maxiter, int maxitergrad) {
  if(method==0)
    printf("Using an even-tempered set.\n");
  else if(method==1)
    printf("Using a well-tempered set.\n");
  else
    printf("Performing full optimization.\n");

  // Number of degrees of freedom
  size_t Ndf=compute_ndf(bas);
  printf("There are %lu degrees of freedom.\n\n",Ndf);

  // Minimized function
  gsl_multimin_function minfunc;
  minfunc.n=Ndf;
  minfunc.f=eval_tempered;
  minfunc.params=(void *) &bas;

  // Minimizer
  gsl_multimin_fminimizer *min;
  // Allocate minimizer
  min=gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2,Ndf);
  printf("Minimizer allocated.\n");

  // Set minimizer
  gsl_multimin_fminimizer_set(min, &minfunc, x, ss);
  printf("Minimizer set.\n\n");


  // Iterate
  int iter=0, itergrad=0;
  int status;
  double size;
  double E=0.0, oldE=0.0;

  Timer t;

  printf("Iteration\tEnergy\t\t\tChange\t\t\tStep\n");
  do {
    iter++;
    
    // Simplex
    status = gsl_multimin_fminimizer_iterate(min);
    if (status) 
      break;

    // Store old energy
    oldE=E;
    // Get current energy
    E=min->fval;
    
    // Get convergence estimate.
    size = gsl_multimin_fminimizer_size (min);
    
    // Are we converged?
    status = gsl_multimin_test_size (size, DBL_EPSILON);
    if (status == GSL_SUCCESS)
      {
	printf ("converged to minimum at\n");
      }

    if(E!=oldE)
      printf("%i\t\t%.10f\t%.16e\t%e\n",iter,E,E-oldE,size);

  } while (status == GSL_CONTINUE && iter < maxiter);
  
  // Copy minimized vector to x
  gsl_vector_memcpy(x,min->x);

  // Free minimizer
  gsl_multimin_fminimizer_free(min);

  // Print converged basis set library
  BasisSetLibrary baslib=construct_library(x,minfunc.params);
  baslib.sort();
  //  baslib.print();

  return baslib;
}

BasisSetLibrary construct_contractions(const BasisSetLibrary & uncontr, const basis_t & bas) {

  // Construct basis set
  BasisSet basis=construct_basis(uncontr,bas);

  // Settings to use
  Settings set=get_settings();

  // Get number of electrons
  int Nel=basis.Ztot()-set.get_int("Charge");
  // Set multiplicity
  if(Nel%2==1)
    set.set_int("Multiplicity",2);

  // Returned, contracted basis set library
  BasisSetLibrary contr;

  // Solver
  SCF solver(basis,set);
  // Overlap matrix
  arma::mat S=basis.overlap();

  // Energy
  double Etot;
  
  if(set.get_int("Multiplicity")==1 && Nel%2==0) {
    // Closed shell case
    arma::mat C;
    arma::vec E;
    // Solve SCF
    Etot=solver.RHF(C,E);

    // Loop over orbitals
    for(size_t iorb=0;iorb<C.n_cols;iorb++) {
      printf("******* Orbital %lu, energy %e ********\n",iorb,E[iorb]);

      // Loop over angular momentum
      for(int am=0;am<=uncontr.get_max_am();am++) {
	printf("am = %i\n",am);
	
	// Loop over elements
	for(size_t iel=0;iel<uncontr.get_Nel();iel++) {
	  // Get the symbol of this element
	  std::string sym=uncontr.get_symbol(iel);

	  // Uncontracted basis set for the element
	  ElementBasisSet el_uncontr=uncontr.get_element(sym);
	  
	  if(el_uncontr.get_max_am()<am)
	    continue;

	  // Determine indices of nuclei of this type
	  std::vector<size_t> nucind;
	  for(size_t i=0;i<basis.get_Nnuc();i++)
	    if(basis.get_symbol(i)==sym) {
	      nucind.push_back(i);
	    }
	  
	  // Loop over nuclei
	  for(size_t i=0;i<nucind.size();i++) {
	    // Index of nucleus is
	    size_t inuc=nucind[i];
	    printf("Nucleus %lu\t",inuc);
	    
	    // Loop over shells
	    for(size_t ish=0;ish<basis.get_Nshells();ish++)
	      if(basis.get_center_ind(ish)==inuc && basis.get_am(ish)==am)
		// Yes, we are on an atom of the wanted element. Print coefficients
		for(size_t ibf=basis.get_first_ind(ish);ibf<=basis.get_last_ind(ish);ibf++)
		  printf(" %e",C(iorb,ibf));
	    printf("\n");
	  }
	}
      }
    }
  }
  
  /* else {
    arma::mat Ca, Cb;
    arma::vec Ea, Eb;
    Etot=solver.UHF(Ca,Cb,Ea,Eb);
  } 
  */   

  return contr;
}  

int main(int argc, char **argv) {

  if(argc!=6) {
    printf("Usage: %s atoms.xyz shells.dat method usediis maxiters\n",argv[0]);
    printf("method is:\n\t0 for even-tempered\n\t1 for well-tempered\n\t2 for full optimization\n");
    return 0;
  }

  // Load atoms
  std::vector<atom_t> atoms=load_xyz(std::string(argv[1]));

  // Construct elements
  std::vector<element_t> els=load_elements(std::string(argv[2]));

  // Basis set structure
  basis_t bas;
  bas.el=els;
  bas.atoms=atoms;

  // Set well-temperedness
  method=atoi(argv[3]);
  // Use DIIS?
  usediis=atoi(argv[4]);
  // Maximum number of iterations
  int maxiter=atoi(argv[5]);
  int maxitergrad=10*maxiter;

  // Optimized basis set library
  BasisSetLibrary baslib;

  if(method==2) {
    // Full optimization. 
    size_t Ndf_full=compute_ndf(bas);

    // Initialize with a tempered set.
    method=0;

    size_t Ndf=compute_ndf(bas);
    // (Initial) step size and starting point
    gsl_vector *ss, *x;
    ss=gsl_vector_alloc(Ndf);
    x=gsl_vector_alloc(Ndf);
    gsl_vector_set_all(ss,1.0);
    // Form starting point
    starting_point(x,bas);
    // Optimize parameters
    //BasisSetLibrary hlp=
    optimize_basis(bas,ss,x,maxiter,maxitergrad);

    // OK, now we know the parameters.
    // Starting point for full optimization
    gsl_vector *ssfull, *xfull;
    ssfull=gsl_vector_alloc(Ndf_full);
    xfull=gsl_vector_alloc(Ndf_full);
    gsl_vector_set_all(ssfull,1.0);
    // Get exponents
    starting_point_full(xfull,x,bas);

    // Free tempered parameters
    gsl_vector_free(x);
    gsl_vector_free(ss);  

    // Run full optimization.
    method=2;
    optimize_basis(bas,ssfull,xfull,maxiter,maxitergrad);

    // Free memory
    gsl_vector_free(xfull);
    gsl_vector_free(ssfull);  
  } else {
    size_t Ndf=compute_ndf(bas);
    // (Initial) step size and starting point
    gsl_vector *ss, *x;
    ss=gsl_vector_alloc(Ndf);
    x=gsl_vector_alloc(Ndf);
    gsl_vector_set_all(ss,1.0);
    // Form starting point
    starting_point(x,bas);

    // Optimize parameters
    baslib=optimize_basis(bas,ss,x,maxiter,maxitergrad);

    // Free memory
    gsl_vector_free(x);
    gsl_vector_free(ss);  
  }

  //  construct_contractions(baslib, bas);
  
  // Save library
  baslib.save_gaussian94("optbas.gbs");

  return 0;
}
