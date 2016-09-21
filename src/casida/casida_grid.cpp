/*
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

#include "casida_grid.h"
#include "../chebyshev.h"
#include "../elements.h"
#include "../timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

CasidaShell::CasidaShell(bool lobatto) : AngularGrid(lobatto) {
}

CasidaShell::~CasidaShell() {
}

void CasidaShell::compute_orbs(const std::vector<arma::mat> & C) {
  if(C.size()==1 && polarized) {
    ERROR_INFO();
    throw std::runtime_error("Trying to calculate restricted orbitals with unrestricted grid.\n");
  } else if(C.size()>1 && !polarized) {
    ERROR_INFO();
    throw std::runtime_error("Trying to calculate unrestricted orbitals with restricted grid.\n");
  }

  orbs.resize(C.size());
  for(size_t ispin=0;ispin<C.size();ispin++)
    // Resize to number of grid points
    orbs[ispin].resize(grid.size());

  // Loop over grid points
  for(size_t ip=0;ip<grid.size();ip++) {
    // Initialize orbital values
    for(size_t ispin=0;ispin<C.size();ispin++) {
      orbs[ispin][ip].resize(C[ispin].n_cols);
      for(size_t io=0;io<C[ispin].n_cols;io++)
	orbs[ispin][ip][io]=0.0;
    }
  }

  for(size_t ispin=0;ispin<C.size();ispin++) {
    // Orbital values are
    arma::mat Cval=arma::trans(C[ispin].rows(bf_ind))*bf;
    // Store values
    for(size_t ip=0;ip<grid.size();ip++)
      for(size_t io=0;io<Cval.n_rows;io++)
	orbs[ispin][ip][io]=Cval(io,ip);
  }
}

void CasidaShell::eval_fxc(int x_func, int c_func) {
  int nspin;
  if(!polarized)
    nspin=XC_UNPOLARIZED;
  else
    nspin=XC_POLARIZED;

  // Allocate memory for fx and fc
  if(!polarized) {
    if(fx.size()!=grid.size())
      fx.resize(grid.size());
    if(fc.size()!=grid.size())
      fc.resize(grid.size());
  } else {
    if(fc.size()!=3*grid.size())
      fc.resize(3*grid.size());
    if(fx.size()!=3*grid.size())
      fx.resize(3*grid.size());
  }

  // Correlation and exchange functionals
  xc_func_type xfunc;
  if(x_func>0) {
    if(xc_func_init(&xfunc, x_func, nspin) != 0) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<x_func<<" not found!";
      throw std::runtime_error(oss.str());
    }

    // Initialize the functional
    xc_func_init(&xfunc, x_func, nspin);

    if(xfunc.info->family!=XC_FAMILY_LDA) {
      ERROR_INFO();
      throw std::runtime_error("Casida only supports LDA functionals.\n");
    }

    // Evaluate fx
    xc_lda_fxc(&xfunc, grid.size(), &rho[0], &fx[0]);

    // Free the functional
    xc_func_end(&xfunc);
  } else {
    // No exchange.
    for(size_t i=0;i<fx.size();i++)
      fx[i]=0.0;
  }

  if(c_func>0) {
    xc_func_type cfunc;
    if(xc_func_init(&cfunc, c_func, nspin) != 0) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<c_func<<" not found!";
      throw std::runtime_error(oss.str());
    }

    xc_func_init(&cfunc, c_func, nspin);
    if(cfunc.info->family!=XC_FAMILY_LDA) {
      ERROR_INFO();
      throw std::runtime_error("Casida only supports LDA functionals.\n");
    }

    // Evaluate fx and fc
    xc_lda_fxc(&cfunc, grid.size(), &rho[0], &fc[0]);

    // Free the functionals
    xc_func_end(&cfunc);
  } else {
    // No correlation.
    for(size_t i=0;i<fc.size();i++)
      fc[i]=0.0;
  }
}

void CasidaShell::Kxc(const std::vector< std::vector<states_pair_t> > & pairs, arma::mat & K) const {
  double wxc;

  if(polarized && pairs.size()!=2) {
    ERROR_INFO();
    throw std::runtime_error("Running with polarized grid but non-polarized pairs!\n");
  } else if(!polarized && pairs.size()==2) {
    ERROR_INFO();
    throw std::runtime_error("Running with unpolarized grid but polarized pairs!\n");
  }

  // Perform the integration. Loop over spins
  for(size_t ispin=0;ispin<pairs.size();ispin++)
    for(size_t jspin=0;jspin<=ispin;jspin++) {

      // Offset in i
      const size_t ioff=ispin*pairs[0].size();
      // Offset in j
      const size_t joff=jspin*pairs[0].size();

      if(ispin==jspin) {
	// Loop over grid points
	for(size_t ip=0;ip<grid.size();ip++) {
	  // Factor in common for all orbitals. First case is polarized (up-up or down-down), second case is unpolarized
	  wxc=polarized ? grid[ip].w*(fx[3*ip+2*ispin]+fc[3*ip+2*ispin]) : grid[ip].w*(fx[ip]+fc[ip]);

	  // Loop over pairs
	  for(size_t ipair=0;ipair<pairs[ispin].size();ipair++) {
	    for(size_t jpair=0;jpair<ipair;jpair++) {

	      double term=wxc*orbs[ispin][ip][pairs[ispin][ipair].i]*orbs[ispin][ip][pairs[ispin][ipair].f]
		*orbs[ispin][ip][pairs[jspin][jpair].i]*orbs[ispin][ip][pairs[jspin][jpair].f];
	      K(ioff+ipair,joff+jpair)+=term;
	      K(joff+jpair,ioff+ipair)+=term;
	    }
	    K(ioff+ipair,ioff+ipair)+=wxc*orbs[ispin][ip][pairs[ispin][ipair].i]*orbs[ispin][ip][pairs[ispin][ipair].f]
	      *orbs[ispin][ip][pairs[ispin][ipair].i]*orbs[ispin][ip][pairs[ispin][ipair].f];
	  }
	}
      } else {
	// Loop over grid points
	for(size_t ip=0;ip<grid.size();ip++) {
	  // Factor in common for all orbitals
	  wxc=grid[ip].w*(fx[3*ip+1]+fc[3*ip+1]); // up-down and down-up

	  // Loop over pairs
	  for(size_t ipair=0;ipair<pairs[ispin].size();ipair++) {
	    for(size_t jpair=0;jpair<ipair;jpair++) {
	      double term=wxc*orbs[ispin][ip][pairs[ispin][ipair].i]*orbs[ispin][ip][pairs[ispin][ipair].f]
		*orbs[ispin][ip][pairs[jspin][jpair].i]*orbs[ispin][ip][pairs[jspin][jpair].f];
	      K(ioff+ipair,joff+jpair)+=term;
	      K(joff+jpair,ioff+ipair)+=term;
	    }
	    K(ioff+ipair,ioff+ipair)+=wxc*orbs[ispin][ip][pairs[ispin][ipair].i]*orbs[ispin][ip][pairs[ispin][ipair].f]
	      *orbs[ispin][ip][pairs[ispin][ipair].i]*orbs[ispin][ip][pairs[ispin][ipair].f];
	  }
	}
      }
    }
}

void CasidaShell::free() {
  AngularGrid::free();
  orbs.clear();
  fx.clear();
  fc.clear();
}


CasidaGrid::CasidaGrid(const BasisSet * bas, bool lobatto, bool ver) {
  basp=bas;
  verbose=ver;

  // Allocate work grids
#ifdef _OPENMP
  int nth=omp_get_max_threads();
  for(int i=0;i<nth;i++)
    wrk.push_back(CasidaShell(lobatto));
#else
  wrk.push_back(CasidaShell(lobatto));
#endif

  for(size_t i=0;i<wrk.size();i++)
    wrk[i].set_basis(*bas);
}

CasidaGrid::~CasidaGrid() {
}

void CasidaGrid::construct(const std::vector<arma::mat> & P, double ftoler, int x_func, int c_func) {
  // Add all atoms
  if(verbose) {
    printf("Constructing XC grid.\n");
    printf("\t%4s  %7s  %10s  %s\n","atom","Npoints","Nfuncs","t");
    fflush(stdout);
  }

  Timer t;
  
  // Check necessity of gradients and laplacians
  for(size_t i=0;i<wrk.size();i++)
    wrk[i].check_grad_lapl(x_func,c_func);

  // Amount of radial shells on the atoms
  std::vector<size_t> nrad(basp->get_Nnuc());
  
  // Form radial shells
  for(size_t iat=0;iat<basp->get_Nnuc();iat++) {
    angshell_t sh;
    sh.atind=iat;
    sh.cen=basp->get_nuclear_coords(iat);
    sh.tol=ftoler*PRUNETHR;
    
    // Compute necessary number of radial points for atom
    size_t nr=std::max(20,(int) round(-5*(3*log10(ftoler)+6-element_row[basp->get_Z(iat)])));
    // Get Chebyshev nodes and weights for radial part
    std::vector<double> rad, wrad;
    radial_chebyshev(nr,rad,wrad);
    nr=rad.size(); // Sanity check
    nrad[iat]=nr;
  
    // Loop over radii
    for(size_t irad=0;irad<nr;irad++) {
      sh.R=rad[irad];
      sh.w=wrad[irad]*sh.R*sh.R;
      grids.push_back(sh);
    }
  }
    
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      wrk[ith].set_grid(grids[i]);
      if(P.size()==1)
	grids[i]=wrk[ith].construct(P[0],ftoler/nrad[grids[i].atind],x_func,c_func);
      else if(P.size()==2)
	grids[i]=wrk[ith].construct(P[0],P[1],ftoler/nrad[grids[i].atind],x_func,c_func);
      else {
	ERROR_INFO();
	throw std::runtime_error("Problem in Casida routine.\n");
      }
    }
  }

  // Prune empty shells
  prune_shells();

  if(verbose) {
    printf("DFT XC grid constructed in %s.\n",t.elapsed().c_str());
    print_grid();
    fflush(stdout);
  }

}

void CasidaGrid::prune_shells() {
  for(size_t i=grids.size()-1;i<grids.size();i--)
    if(!grids[i].np || !grids[i].nfunc)
      grids.erase(grids.begin()+i);
}

void CasidaGrid::Kxc(const std::vector<arma::mat> & P, double tol, int x_func, int c_func, const std::vector<arma::mat> & C, const std::vector< std::vector<states_pair_t> > & pairs, arma::mat & Kx) {
  // First, we need to construct the grid.
  construct(P,tol,x_func,c_func);

  /* Consistency checks */
  if(P.size()!=C.size()) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of P and C are inconsistent!\n");
  }
  if(P.size()!=pairs.size()) {
    ERROR_INFO();
    throw std::runtime_error("Sizes of P and pairs are inconsistent!\n");
  }

  // Now, loop over the atoms.
#ifdef _OPENMP
#pragma omp parallel
#endif
  {

#ifndef _OPENMP
    int ith=0;
#else
    int ith=omp_get_thread_num();
    
#pragma omp for schedule(dynamic,1)
#endif
    for(size_t i=0;i<grids.size();i++) {
      // Change atom and create grid
      wrk[ith].set_grid(grids[i]);
      wrk[ith].form_grid();
      
      // Update the density
      if(P.size()==1)
	wrk[ith].update_density(P[0]);
      else
	wrk[ith].update_density(P[0],P[1]);
      // and compute fxc
      wrk[ith].eval_fxc(x_func,c_func);
      
      // Compute the values of the orbitals
      wrk[ith].compute_orbs(C);
      
      // Compute Kxc's
#ifdef _OPENMP
#pragma omp critical
#endif
      wrk[ith].Kxc(pairs, Kx);
      
      // Free the memory
      wrk[ith].free();
    }
  }

  // Symmetrize K if necessary
  for(size_t ispin=0;ispin<pairs.size();ispin++)
    for(size_t jspin=0;jspin<=ispin;jspin++) {
      // Offset in i
      const size_t ioff=ispin*pairs[0].size();
      // Offset in j
      const size_t joff=jspin*pairs[0].size();

      if(ispin!=jspin) {
	Kx.submat(joff,ioff,joff+pairs[jspin].size()-1,ioff+pairs[ispin].size()-1)=arma::trans(Kx.submat(ioff,joff,ioff+pairs[ispin].size()-1,joff+pairs[jspin].size()-1));
      }
    }
}

void CasidaGrid::print_grid() const {
  // Amount of integration points
  arma::uvec np(basp->get_Nnuc());
  np.zeros();
  // Amount of function values
  arma::uvec nf(basp->get_Nnuc());
  nf.zeros();

  for(size_t i=0;i<grids.size();i++) {
    np(grids[i].atind)+=grids[i].np;
    nf(grids[i].atind)+=grids[i].nfunc;
  }

  printf("Composition of %s grid:\n %7s %7s %10s\n","XC","atom","Npoints","Nfuncs");
  for(size_t i=0;i<basp->get_Nnuc();i++)
    printf(" %4i %-2s %7i %10i\n",(int) i+1, basp->get_symbol(i).c_str(), (int) np(i), (int) nf(i));
}
