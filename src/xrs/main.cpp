/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "bfprod.h"
#include "fourierprod.h"

#include "basis.h"
#include "basislibrary.h"
#include "dftgrid.h"
#include "dftfuncs.h"
#include "elements.h"
#include "density_fitting.h"
#include "linalg.h"
#include "lmgrid.h"
#include "lmtrans.h"
#include "mathf.h"
#include "momentum_series.h"
#include "settings.h"
#include "stringutil.h"
#include "tempered.h"
#include "timer.h"
#include "xrsscf.h"
#include "xyzutils.h"

#include <algorithm>
#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <istream>

#ifdef _OPENMP
#include <omp.h>
#endif

/// Struct for orbitals.
typedef struct {
  arma::vec C;
  double E;
} orbital_t;

/// Tool for sorting orbitals in order of increasing energy
bool operator<(const orbital_t & lhs, const orbital_t & rhs) {
  return lhs.E<rhs.E;
}

/**
 * Double basis set method
 *
 * Augment the basis with diffuse functions and diagonalize the Fock
 * matrix in the unoccupied space. The occupied orbitals and their
 * energies stay the same in the approximation.
 */
void augmented_solution(const BasisSet & basis, const Settings & set, const uscf_t & sol, size_t xcatom, size_t & ixc_orb, size_t nocca, size_t noccb, dft_t dft, BasisSet & augbas, arma::mat & Caug, arma::vec & Eaug, bool spin) {
  // Get indices of atoms to augment
  std::vector<size_t> augind=parse_range(splitline(set.get_string("XRSAugment"))[0]);
  // Convert to C++ indexing
  for(size_t i=0;i<augind.size();i++) {
    if(augind[i]==0)
      throw std::runtime_error("Error - nuclear index must be positive!\n");
    augind[i]--;
  }

  // Form augmented basis
  augbas=basis;

  // Loop over excited atoms
  for(size_t iaug=0;iaug<augind.size();iaug++) {
    // Index of atom is
    const size_t ind=augind[iaug];
    // and its charge is
    const int Z=basis.get_Z(ind);

    // Determine which augmentation to use for atom in question.
    double alpha=0.0029;
    double beta=1.4;
    int nf;

    if(Z>2 && Z<11) {
      // X-FIRST by L. G. M. Pettersson, taken from StoBe basis
      nf=19;
    } else if(Z>10 && Z<19) {
      // X-SECOND by L. G. M. Pettersson, taken from StoBe basis
      nf=25;
    } else {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Augmentation basis not defined for " << basis.get_symbol(ind) << "!\n";
      throw std::runtime_error(oss.str());
    }

    // Get exponents
    std::vector<double> exps=eventempered_set(alpha,beta,nf);

    // Add functions
    std::vector<contr_t> C(1);
    C[0].c=1.0;
    for(size_t iexp=0;iexp<exps.size();iexp++) {
      C[0].z=exps[iexp];
      for(int am=0;am<=2;am++)
	augbas.add_shell(ind,am,C,false);
    }
  }

  // Finalize augmentation basis
  augbas.finalize();

  // Amount of functions in original basis set is
  const size_t Nbf=basis.get_Nbf();
  // Total number of functions in augmented set is
  const size_t Ntot=augbas.get_Nbf();
  // Amount of augmentation functions is
  const size_t Naug=Ntot-Nbf;

  printf("\nAugmented original basis (%i functions) with %i diffuse functions.\n",(int) Nbf,(int) Naug);

  Timer taug;
  
  // Overlap matrix in augmented basis
  arma::mat S=augbas.overlap();
  arma::vec Sval;
  arma::mat Svec;
  eig_sym_ordered(Sval,Svec,S);

  printf("Condition number of overlap matrix is %e.\n",Sval(0)/Sval(Sval.n_elem-1));

  printf("Diagonalization of basis took %s.\n",taug.elapsed().c_str());
  taug.set();

  // Count number of independent functions
  size_t Nind=0;
  for(size_t i=0;i<Ntot;i++)
    if(Sval(i)>=1e-5)
      Nind++;

  printf("Augmented basis has %i linearly independent and %i dependent functions.\n",(int) Nind,(int) (Ntot-Nind));

  // Drop linearly dependent ones.
  Sval=Sval.subvec(Sval.n_elem-Nind,Sval.n_elem-1);
  Svec=Svec.submat(0,Svec.n_cols-Nind,Svec.n_rows-1,Svec.n_cols-1);

  // Form the matrix which takes from the AO basis to an orthonormal basis.
  arma::mat AOtoO(Ntot,Nind);
  AOtoO.zeros();

  // The first nocc vectors are simply the occupied states.
  size_t nocc;
  if(!spin) {
    nocc=nocca;
    AOtoO.submat(0,0,Nbf-1,nocc-1)=sol.Ca.submat(0,0,Nbf-1,nocc-1);
  } else {
    nocc=noccb;
    AOtoO.submat(0,0,Nbf-1,nocc-1)=sol.Cb.submat(0,0,Nbf-1,nocc-1);
  }

  // Do a Gram-Schmidt orthogonalization to find the rest of the
  // orthonormal vectors. But first we need to drop the eigenvectors
  // of S with the largest projection to the occupied orbitals, in
  // order to avoid linear dependency problems with the Gram-Schmidt
  // method.

  // Indices to keep in the treatment
  std::vector<size_t> keepidx;
  for(size_t i=0;i<Nind;i++)
    keepidx.push_back(i);
  
  // Drop the nocc largest eigenvalues
  for(size_t j=0;j<nocc;j++) {
    // Find maximum overlap
    double maxovl=0.0;
    size_t maxind=-1;

    // Helper vector
    arma::vec hlp=S*AOtoO.col(j);
    
    for(size_t ii=0;ii<keepidx.size();ii++) {
      // Index of eigenvector is
      size_t i=keepidx[ii];
      // Compute projection
      double ovl=fabs(arma::dot(Svec.col(i),hlp))/sqrt(Sval(i));
      // Check if it has the maximal value
      if(fabs(ovl>maxovl)) {
	maxovl=ovl;
	maxind=ii;
      }
    }
    
    // Delete the index
    printf("Deleted function %i with overlap %e.\n",(int) keepidx[maxind],maxovl);
    keepidx.erase(keepidx.begin()+maxind);
  }
  
  // Fill in the rest of the vectors
  for(size_t i=0;i<keepidx.size();i++) {
    // The index of the vector to use is
    size_t ind=keepidx[i];
    // Normalize it, too
    AOtoO.col(nocc+i)=Svec.col(ind)/sqrt(Sval(ind));
  }

  // Run the orthonormalization of the set
  for(size_t i=0;i<Nind;i++) {
    double norm=arma::as_scalar(arma::trans(AOtoO.col(i))*S*AOtoO.col(i));
    // printf("Initial norm of vector %i is %e.\n",(int) i,norm);
    
    // Remove projections of already orthonormalized set
    for(size_t j=0;j<i;j++) {
      double proj=arma::as_scalar(arma::trans(AOtoO.col(j))*S*AOtoO.col(i));

      //    printf("%i - %i was %e\n",(int) i, (int) j, proj);
      AOtoO.col(i)-=proj*AOtoO.col(j);
    }
    
    norm=arma::as_scalar(arma::trans(AOtoO.col(i))*S*AOtoO.col(i));
    // printf("Norm of vector %i is %e.\n",(int) i,norm);
    
    // and normalize
    AOtoO.col(i)/=sqrt(norm);
  }

  /*
  arma::mat ovl=arma::trans(AOtoO)*S*AOtoO;
  printf("MO overlap\n");
  ovl.print();
  */

  // Form density matrix.
  arma::mat Paaug(Ntot,Ntot), Pbaug(Ntot,Ntot), Paug(Ntot,Ntot);

  Paaug.zeros();
  Paaug.submat(0,0,Nbf-1,Nbf-1)=sol.Pa;

  Pbaug.zeros();
  Pbaug.submat(0,0,Nbf-1,Nbf-1)=sol.Pb;

  Paug=Paaug+Pbaug;

  // Form Fock matrix.
  taug.set();
  arma::mat T=augbas.kinetic();
  arma::mat V=augbas.nuclear();
  printf("Hcore formed in %s.\n",taug.elapsed().c_str());

  // Coulomb matrix
  taug.set();
  arma::mat J;
  {
    // We use the original basis' density fitting basis, since it's
    // enough to represent the density.
    BasisSet dfitbas=basis.density_fitting();
    DensityFit dfit;
    // We do the formation directly.
    dfit.fill(augbas,dfitbas,true);
    J=dfit.calc_J(Paug);
  }
  printf("J formed in %s.\n",taug.elapsed().c_str());


  // DFT grid. Get used tolerance
  dft.gridtol=set.get_double("XRSGridTol");
  taug.set();
  arma::mat XCa, XCb;
  {
    double Exc, Nelnum;
    DFTGrid grid(&augbas,true,true,false);
    grid.construct(Paaug,Pbaug,dft.gridtol,dft.x_func,dft.c_func);
    printf("XC grid constructed in %s.\n",taug.elapsed().c_str());
    taug.set();
    grid.eval_Fxc(dft.x_func,dft.c_func,Paaug,Pbaug,XCa,XCb,Exc,Nelnum);
  }
  printf("XC evaluated in %s.\n",taug.elapsed().c_str());

  // Form Fock operator  
  arma::mat H;
  if(!spin)
    H=T+V+J+XCa;
  else
    H=T+V+J+XCb;

  // Amount of virtual orbitals
  //size_t Nvirt=Nind-nocc;

  // Convert Fock operator to unoccupied MO basis.
  taug.set();
  arma::mat H_MO=arma::trans(AOtoO.submat(0,nocc,Ntot-1,Nind-1))*H*AOtoO.submat(0,nocc,Ntot-1,Nind-1);
  printf("H_MO formed in %s.\n",taug.elapsed().c_str());
  
  // Diagonalize Fockian to find orbitals and energies
  taug.set();
  arma::vec Eval;
  arma::mat Evec;
  eig_sym_ordered(Eval,Evec,H_MO);
  printf("H_MO diagonalized in unoccupied space in %s.\n",taug.elapsed().c_str());
  
  // Store energies
  Eaug.zeros(Nind);
  // Occupied orbitals
  if(!spin)
    Eaug.subvec(0,nocc-1)=sol.Ea.subvec(0,nocc-1);
  else
    Eaug.subvec(0,nocc-1)=sol.Eb.subvec(0,nocc-1);
  // Virtuals
  Eaug.subvec(nocc,Nind-1)=Eval;

  // Back-transform orbitals to AO basis
  Caug.zeros(Ntot,Nind);
  // Occupied orbitals, padded with zeros
  Caug.submat(0,0,Nbf-1,nocc-1)=AOtoO.submat(0,0,Nbf-1,nocc-1);
  // Unoccupied orbitals
  Caug.submat(0,nocc,Ntot-1,Nind-1)=AOtoO.submat(0,nocc,Ntot-1,Nind-1)*Evec;
}

typedef struct {
  // Transition energy
  double E;
  // Total oscillator strength
  double w;
  // Decomposition of strength
  std::vector<double> wdec;
} spectrum_t;

/// Compute transitions to unoccupied states.
std::vector<spectrum_t> compute_transitions(const BasisSet & basis, const arma::mat & C, const arma::vec & E, size_t iat, size_t ixc, size_t nocc, bool abs=1) {
  // Returned array
  std::vector<spectrum_t> ret;

  // Coordinates of excited atom
  coords_t xccen=basis.get_coords(iat);
  // Dipole moment matrix
  std::vector<arma::mat> mom1=basis.moment(1,xccen.x,xccen.y,xccen.z);
  // Compute RHS of transition
  arma::vec rhs_x=mom1[getind(1,0,0)]*C.col(ixc);
  arma::vec rhs_y=mom1[getind(0,1,0)]*C.col(ixc);
  arma::vec rhs_z=mom1[getind(0,0,1)]*C.col(ixc);

  for(size_t ix=nocc;ix<C.n_cols;ix++) {
    spectrum_t sp;
    // Transition energy is
    sp.E=(E[ix]-E[ixc]);

    // Compute oscillator strengths in x, y and z
    double wx=arma::dot(rhs_x,C.col(ix));
    double wy=arma::dot(rhs_y,C.col(ix));
    double wz=arma::dot(rhs_z,C.col(ix));

    // Spherical average of oscillator strength in XRS is
    sp.w=2.0/3.0*(wx*wx + wy*wy + wz*wz);
    // Store decomposition
    sp.wdec.push_back(wx);
    sp.wdec.push_back(wy);
    sp.wdec.push_back(wz);

    ret.push_back(sp);
  }

  return ret;
}

std::vector< std::vector<spectrum_t> > compute_qdep_transitions_series(const BasisSet & basis, const arma::mat & C, const arma::vec & E, size_t ixc, size_t nocc, std::vector<double> qval) \
{
  Timer t;

  // Get the grid for computing the spherical averages.
  std::vector<angular_grid_t> grid=form_angular_grid(2*basis.get_max_am());
  // We normalize the weights so that for purely dipolar transitions we
  // get the same output as with using the dipole matrix.
  for(size_t i=0;i<grid.size();i++) {
    // Dipole integral is only wrt theta - divide off phi part.
    grid[i].w/=2.0*M_PI;
  }

  // Amount of transitions is
  size_t Ntrans=C.n_cols-nocc;

  // Initialize return array
  std::vector< std::vector<spectrum_t> > ret(qval.size());
  for(size_t iq=0;iq<qval.size();iq++) {
    ret[iq].resize(Ntrans);
    for(size_t ix=nocc;ix<C.n_cols;ix++) {
      // Transition energy is
      ret[iq][ix-nocc].E=(E[ix]-E[ixc]);
      // Transition speed is
      ret[iq][ix-nocc].w=0.0;
    }
  }

  // Copy the orbitals to complex form.
  arma::cx_mat Cm(C.n_rows,Ntrans);
  for(size_t ir=0;ir<C.n_rows;ir++)
    for(size_t ix=nocc;ix<C.n_cols;ix++)
      Cm(ir,ix-nocc)=C(ir,ix);

  printf("Computing transitions using series method.\n");
  arma::cx_vec tmp;

  // Evaluator
  momentum_transfer_series ser(&basis);

  // Loop over q values.
  for(size_t iq=0;iq<qval.size();iq++) {
    printf("\tq = %8.3f ... ",qval[iq]);
    fflush(stdout);
    Timer tq;

    // Loop over angular mesh
    for(size_t ig=0;ig<grid.size();ig++) {

      // Current q is
      arma::vec q(3);
      q(0)=qval[iq]*grid[ig].r.x;
      q(1)=qval[iq]*grid[ig].r.y;
      q(2)=qval[iq]*grid[ig].r.z;
      // and the weight is
      double w=grid[ig].w;

      // Get momentum transfer matrix
      arma::cx_mat momtrans=ser.get(q,10*DBL_EPSILON,10*DBL_EPSILON);
      // Contract with initial state
      tmp=momtrans*C.col(ixc);

      // Compute matrix elements for all transitions
#ifdef _OPENMP
#pragma omp parallel for
#endif
      // Loop over transitions
      for(size_t it=0;it<Ntrans;it++) {
	// The matrix element is
	std::complex<double> el=arma::dot(tmp,Cm.col(it));
	// so the transition velocity is increased by
	ret[iq][it].w+=w*std::norm(el);
      }
    }

    printf("done (%s)\n",tq.elapsed().c_str());
  }

  return ret;
}


std::vector< std::vector<spectrum_t> > compute_qdep_transitions_fourier(const BasisSet & basis, const arma::mat & C, const arma::vec & E, size_t ixc, size_t nocc, std::vector<double> qval) \
{
  Timer t;

  printf("Computing transitions using Fourier method.\n");

  // Form products of basis functions.
  const size_t Nbf=basis.get_Nbf();

  std::vector<prod_gaussian_3d> bfprod=compute_products(basis);

  printf("Products done in %s.\n",t.elapsed().c_str());
  t.set();

  // Form Fourier transforms of the products.
  std::vector<prod_fourier> bffour=fourier_transform(bfprod);
  printf("Fourier transform done in %s.\n",t.elapsed().c_str());
  t.set();

  // Get the grid for computing the spherical averages.
  std::vector<angular_grid_t> grid=form_angular_grid(2*basis.get_max_am());
  // We normalize the weights so that for purely dipolar transitions we
  // get the same output as with using the dipole matrix.
  for(size_t i=0;i<grid.size();i++) {
    // Dipole integral is only wrt theta - divide off phi part.
    grid[i].w/=2.0*M_PI;
  }

  // Amount of transitions is
  size_t Ntrans=C.n_cols-nocc;

  // Initialize return array
  std::vector< std::vector<spectrum_t> > ret(qval.size());
  for(size_t iq=0;iq<qval.size();iq++) {
    ret[iq].resize(Ntrans);
    for(size_t ix=nocc;ix<C.n_cols;ix++) {
      // Transition energy is
      ret[iq][ix-nocc].E=(E[ix]-E[ixc]);
      // Transition speed is
      ret[iq][ix-nocc].w=0.0;
    }
  }

  // Copy the orbitals to complex form.
  arma::cx_mat Cm(C.n_rows,Ntrans);
  for(size_t ir=0;ir<C.n_rows;ir++)
    for(size_t ix=nocc;ix<C.n_cols;ix++)
      Cm(ir,ix-nocc)=C(ir,ix);

  // Loop over q values.
  arma::cx_vec tmp;

  for(size_t iq=0;iq<qval.size();iq++) {
    printf("\tq = %8.3f ... ",qval[iq]);
    fflush(stdout);
    Timer tq;

    // Loop over angular mesh
    for(size_t ig=0;ig<grid.size();ig++) {

      // Current q is
      arma::vec q(3);
      q(0)=qval[iq]*grid[ig].r.x;
      q(1)=qval[iq]*grid[ig].r.y;
      q(2)=qval[iq]*grid[ig].r.z;
      // and the weight is
      double w=grid[ig].w;

      // Get momentum transfer matrix
      arma::cx_mat momtrans=momentum_transfer(bffour,Nbf,q);

      // Contract with initial state
      tmp=momtrans*C.col(ixc);

      // Compute matrix elements for all transitions
#ifdef _OPENMP
#pragma omp parallel for
#endif
      // Loop over transitions
      for(size_t it=0;it<Ntrans;it++) {
	// The matrix element is
	std::complex<double> el=arma::dot(tmp,Cm.col(it));
	// so the transition velocity is increased by
	ret[iq][it].w+=w*std::norm(el);
      }
    }

    printf("done (%s)\n",tq.elapsed().c_str());
  }

  return ret;
}


std::vector< std::vector<spectrum_t> > compute_qdep_transitions_local(const BasisSet & basis, const Settings & set, const arma::mat & C, const arma::vec & E, size_t iat, size_t ixc, size_t nocc, std::vector<double> q) {
  // Get wanted quadrature info
  int Nrad=set.get_int("XRSNrad");
  int Lmax=set.get_int("XRSLmax");
  int Lquad=set.get_int("XRSLquad");

  if(Lquad<Lmax) {
    Lquad=Lmax;
    fprintf(stderr,"Increasing the quadrature order to Lmax.\n");
  }

  // Do lm expansion of orbitals
  lmtrans lm(C,basis,basis.get_coords(iat),Nrad,Lmax,Lquad);

  printf("\n");
  lm.print_info();
  printf("\n");

  // Save radial distribution of excited orbital
  lm.write_prob(ixc,"excited_orb.dat");

  // Returned array
  std::vector< std::vector<spectrum_t> > ret(q.size());
  for(size_t iq=0;iq<q.size();iq++)
    ret[iq].resize(C.n_cols-nocc);

  // Loop over q
  printf("Computing transitions.\n");
  for(size_t iq=0;iq<q.size();iq++) {
    printf("\tq = %8.3f ... ",q[iq]);
    fflush(stdout);
    Timer tq;

    // Compute Bessel functions
    bessel_t bes=lm.compute_bessel(q[iq]);

    // Loop over transitions
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(size_t ix=nocc;ix<C.n_cols;ix++) {
      Timer ttrans;

      spectrum_t sp;
      // Transition energy is
      sp.E=(E[ix]-E[ixc]);

      // Get oscillator strength
      std::vector<double> osc=lm.transition_velocity(ixc,ix,bes);
      // Total strength is
      sp.w=osc[0];
      // Store the rest
      osc.erase(osc.begin());
      sp.wdec=osc;

      // Store result
      ret[iq][ix-nocc]=sp;

      //      printf("\t%3i -> %3i (%s)\n",(int) ixc+1,(int) ix+1,ttrans.elapsed().c_str());
    }

    printf("done (%s)\n",tq.elapsed().c_str());
  }


  return ret;
}

void save_spectrum(const std::vector<spectrum_t> & sp, const char *fname="dipole.dat") {
  // Compute transitions
  FILE *out=fopen(fname,"w");

  for(size_t i=0;i<sp.size();i++) {
    // Print out energy in eV oscillator strength and its components
    fprintf(out,"%e\t%e",sp[i].E*HARTREEINEV,sp[i].w);
    for(size_t j=0;j<sp[i].wdec.size();j++)
      fprintf(out,"\t%e",sp[i].wdec[j]);
    fprintf(out,"\n");
  }
  fclose(out);
}

bool load(const BasisSet & basis, const Settings & set, Checkpoint & chkpt, uscf_t & sol) {
  // Was the load a success?
  bool ok=true;

  // Basis set used in the checkpoint file
  BasisSet loadbas;

  try {
    chkpt.read("Ca",sol.Ca);
    chkpt.read("Cb",sol.Cb);
    chkpt.read("Ea",sol.Ea);
    chkpt.read("Eb",sol.Eb);
    chkpt.read("Pa",sol.Pa);
    chkpt.read("Pb",sol.Pb);

    chkpt.read(loadbas);
  } catch(std::runtime_error err) {
    ok=false;
    fprintf(stderr,"Loading failed due to \"%s\".\n",err.what());
  }

  if(ok) {
    // Check consistency
    if(!(basis==loadbas)) {
      ok=false;
      fprintf(stderr,"Basis sets differ!\n");
    }
  }

  if(ok) {
    // Get number of basis functions
    size_t Nbf=basis.get_Nbf();
    
    if(sol.Ca.n_rows != Nbf)
      ok=false;

    if(sol.Cb.n_rows != Nbf)
      ok=false;

    if(sol.Ea.n_elem != sol.Ca.n_cols)
      ok=false;

    if(sol.Eb.n_elem != sol.Cb.n_cols)
      ok=false;

    if(sol.Ea.n_elem != sol.Eb.n_elem)
      ok=false;

    if(sol.Pa.n_rows != Nbf || sol.Pa.n_cols != Nbf)
      ok=false;

    if(sol.Pb.n_rows != Nbf || sol.Pb.n_cols != Nbf)
      ok=false;
    
    if(!ok)
      fprintf(stderr,"Dimensions do not match!\n");
  }

  // Check consistency of spin and hole
  if(ok) {
    bool hole;
    chkpt.read("XRSFullhole",hole);
    if(hole!=set.get_bool("XRSFullhole")) {
      ok=false;
      fprintf(stderr,"Hole character does not match.\n");
    }
  }

  if(ok) {    
    bool spin;
    chkpt.read("XRSSpin",spin);
    if(spin!=set.get_bool("XRSSpin")) {
      ok=false;
      fprintf(stderr,"Excited spin does not match.\n");
    }
  }
  
  if(!ok) {
    // Failed to load or solution was not consistent.
    sol.Ca=arma::mat();
    sol.Cb=arma::mat();
    sol.Ea=arma::vec();
    sol.Eb=arma::vec();
    sol.Pa=arma::mat();
    sol.Pb=arma::mat();
  }

  if(ok) {
    // Was the calculation converged?

    bool conv;
    chkpt.read("Converged",conv);
    ok=conv;

    if(!ok)
      fprintf(stderr,"Calculation was not converged.\n");
  }

  return ok;
}

int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - XRS from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - XRS from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");

  print_license();

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_scf_settings();

  // Change defaults
  set.set_bool("UseDIIS",0);
  set.set_bool("UseADIIS",0);
  set.set_bool("UseBroyden",1);
  set.set_string("Logfile","erkale_xrs.log");

  // Add xrs specific settings
  set.add_string("LoadChk","Initialize with ground state calculation from file","");
  set.add_string("SaveChk","Save results to ","erkale_xrs.chk");

  set.add_bool("XRSLocalize","Localize and freeze orbitals? (Needs ground-state calculation)",0);

  set.add_bool("XRSSpin","Spin to excite (0 for alpha, 1 for beta)",0); 
  set.add_bool("XRSFullhole","Run full core-hole calculation",0);
  set.add_string("XRSAugment","Which atoms to augment with diffuse functions? E.g. 1,3-5,10","");
  set.add_double("XRSGridTol","DFT grid tolerance in double basis set calculation",1e-4);

  set.add_string("XRSQval","List or range of Q values to compute","");
  set.add_string("XRSQMethod","Method of computing momentum transfer matrix: Local, Fourier or Series","Fourier");

  set.add_int("XRSNrad","Local: how many point to use in radial integration",200);
  set.add_int("XRSLmax","Local: expand orbitals up to Lmax",5);
  set.add_int("XRSLquad","Local: perform angular expansion using quadrature of Lquad order",30);

  set.parse(std::string(argv[1]));

  // Redirect output?
  std::string logfile=set.get_string("Logfile");
  if(stricmp(logfile,"stdout")!=0) {
    // Redirect stdout to file
    FILE *outstream=freopen(logfile.c_str(),"w",stdout);
    if(outstream==NULL) {
      ERROR_INFO();
      throw std::runtime_error("Unable to redirect output!\n");
    } else
      fprintf(stderr,"\n");
  }

  // Get used settings
  const bool verbose=set.get_bool("Verbose");
  const bool fullhole=set.get_bool("XRSFullhole");
  const bool spin=set.get_bool("XRSSpin");

  // Print out settings
  if(verbose)
    set.print();

  // Read in atoms.
  std::vector<atom_t> atoms;
  std::string atomfile=set.get_string("System");
  atoms=load_xyz(atomfile);

  // Get index of excited atom
  size_t xcatom=get_excited_atom_idx(atoms);

  // Read in basis set
  BasisSetLibrary baslib;
  std::string basfile=set.get_string("Basis");
  baslib.load_gaussian94(basfile);

  // Construct basis set
  BasisSet basis=construct_basis(atoms,baslib,set);

  // Get exchange and correlation functionals
  dft_t dft; // Final tolerance
  parse_xc_func(dft.x_func,dft.c_func,set.get_string("Method"));
  dft.gridtol=set.get_double("DFTFinalTol");

  dft_t dft_init(dft); // Initial tolerance
  dft_init.gridtol=set.get_double("DFTInitialTol");

  // Final convergence settings
  convergence_t conv;
  // Make initialization parameters more relaxed
  conv.deltaEmax=set.get_double("DeltaEmax");
  conv.deltaPmax=set.get_double("DeltaPmax");
  conv.deltaPrms=set.get_double("DeltaPrms");

  // Convergence settings for initialization
  convergence_t init_conv(conv);
  // Make initialization parameters more relaxed
  double initfac=set.get_double("DFTDelta");
  init_conv.deltaEmax*=initfac;
  init_conv.deltaPmax*=initfac;
  init_conv.deltaPrms*=initfac;

  // TP solution
  uscf_t sol;
  sol.en.E=0.0;

  // Index of excited orbital
  size_t xcorb;

  // Number of occupied states
  int nocca, noccb;
  get_Nel_alpha_beta(basis.Ztot()-set.get_int("Charge"),set.get_int("Multiplicity"),nocca,noccb);

  // Try to load orbitals and energies
  bool loadok=false;
  if(file_exists(set.get_string("SaveChk"))) {
    Checkpoint testload(set.get_string("SaveChk"),false);
    loadok=load(basis,set,testload,sol);
    if(loadok) fprintf(stderr,"Loaded existing checkpoint file.\n");
  }

  // No existing calculation found or system was different => perform calculation
  if(!loadok) {
    Checkpoint chkpt(set.get_string("SaveChk"),true);
    chkpt.write(basis);
    chkpt.write("XRSSpin",set.get_bool("XRSSpin"));
    chkpt.write("XRSFullhole",set.get_bool("XRSFullhole"));

    // Amount of (orbital rotation) localized orbitals (nloc-1 are then frozen)
    size_t nloc=0;
      
    // Initialize calculation with ground state if necessary
    if(stricmp(set.get_string("LoadChk"),"")!=0) {
      printf("Initializing with calculation from %s.\n",set.get_string("LoadChk").c_str());
      
      // Read checkpoint file
      Checkpoint load(set.get_string("LoadChk"),false);

      // Restricted calculation?
      bool restr;
      load.read("Restricted",restr);

      // Load basis
      BasisSet oldbas;
      load.read(oldbas);

      // Read orbitals
      if(restr) {
	arma::mat Cold;
	arma::vec Eold;

	load.read("C",Cold);
	load.read("E",Eold);

	// Project
	basis.projectMOs(oldbas,Eold,Cold,sol.Ea,sol.Ca);
	sol.Eb=sol.Ea;
	sol.Cb=sol.Ca;
      } else {
        // Load energies and orbitals
	arma::vec Eaold, Ebold;
	arma::mat Caold, Cbold;
        load.read("Ca",Caold);
        load.read("Ea",Eaold);
        load.read("Cb",Cbold);
        load.read("Eb",Ebold);
	
        // Project to new basis.
        basis.projectMOs(oldbas,Eaold,Caold,sol.Ea,sol.Ca);
        basis.projectMOs(oldbas,Ebold,Cbold,sol.Eb,sol.Cb);
      }

      if(set.get_bool("XRSLocalize")) {
	if(spin)
	  nloc=localize(basis,noccb,xcatom,sol.Cb);
	else
	  nloc=localize(basis,nocca,xcatom,sol.Ca);
      }
      
      // Find excited orbital 
      size_t ixc_orb;
      lmtrans lmground;
      if(spin) {
	ixc_orb=find_excited_orb(sol.Cb,basis,xcatom,noccb);
	// Do local expansion
	lmground=lmtrans(sol.Cb.submat(0,ixc_orb,sol.Cb.n_rows,ixc_orb),basis,basis.get_coords(xcatom));
      }
      else {
	ixc_orb=find_excited_orb(sol.Ca,basis,xcatom,nocca);
	// Do local expansion
	lmground=lmtrans(sol.Ca.submat(0,ixc_orb,sol.Ca.n_rows,ixc_orb),basis,basis.get_coords(xcatom));
      }
      // Save localized orbital
      lmground.write_prob(0,"ground_orb.dat");
    } else {
      if(set.get_bool("XRSLocalize"))
	throw std::runtime_error("Need a ground-state calculation in LoadChk to perform localization!\n");
    }

    // Proceed with TP calculation. Initialize solver
    XRSSCF solver(basis,set,chkpt,spin);

    // Set frozen orbitals
    if(nloc>0) {
      if(spin)
	solver.set_frozen(sol.Cb.submat(0,1,sol.Cb.n_rows-1,nloc-1),0);
      else
	solver.set_frozen(sol.Ca.submat(0,1,sol.Ca.n_rows-1,nloc-1),0);
    }
    
    // Do TP calculation.
    if(fullhole) {
      xcorb=solver.full_hole(xcatom,sol,init_conv,dft_init);
      xcorb=solver.full_hole(xcatom,sol,init_conv,dft_init);
    } else {      
      xcorb=solver.half_hole(xcatom,sol,init_conv,dft_init);
      xcorb=solver.half_hole(xcatom,sol,conv,dft);
    }

    printf("\n\n");
  } else {
    if(spin)
      xcorb=find_excited_orb(sol.Cb,basis,xcatom,noccb);
    else
      xcorb=find_excited_orb(sol.Ca,basis,xcatom,nocca);
  }

  // Augment the solutions if necessary
  BasisSet augbas;
  arma::mat C_aug;
  arma::vec E_aug;

  if(stricmp(set.get_string("XRSAugment"),"")!=0)
    augmented_solution(basis,set,sol,xcatom,xcorb,nocca,noccb,dft,augbas,C_aug,E_aug,spin);
  else {
    // No augmentation necessary, just copy the solutions
    augbas=basis;
    if(spin) {
      C_aug=sol.Cb;
      E_aug=sol.Eb;
    } else {
      C_aug=sol.Ca;
      E_aug=sol.Ea;
    }
  }

  // Number of occupied states
  size_t nocc;
  if(spin)
    nocc=noccb;
  else
    nocc=nocca;


  // Compute dipole transitions
  std::vector<spectrum_t> sp=compute_transitions(augbas,C_aug,E_aug,xcatom,xcorb,nocc);
  // Save spectrum
  save_spectrum(sp);

  // Get values of q to compute for
  std::vector<double> qvals=parse_range_double(set.get_string("XRSQval"));

  // The q-dependent spectra
  std::vector< std::vector<spectrum_t> > qsp;
  // The filename
  std::string spname;

  if(qvals.size()) {
    // Series method
    if(stricmp(set.get_string("XRSQMethod"),"Series")==0) {
      qsp=compute_qdep_transitions_series(augbas,C_aug,E_aug,xcorb,nocc,qvals);
      spname="trans_ser";
    }
    
    // Fourier method
    if(stricmp(set.get_string("XRSQMethod"),"Fourier")==0) {
      qsp=compute_qdep_transitions_fourier(augbas,C_aug,E_aug,xcorb,nocc,qvals);
      spname="trans_four";
    }
    
    // Local method (Sakko et al)
    if(stricmp(set.get_string("XRSQMethod"),"Local")==0) {
      qsp=compute_qdep_transitions_local(augbas,set,C_aug,E_aug,xcatom,xcorb,nocc,qvals);
      spname="trans_loc";
    }
    
    // Save transitions
    for(size_t i=0;i<qvals.size();i++) {
      char fname[80];
      sprintf(fname,"%s-%.2f.dat",spname.c_str(),qvals[i]);
      save_spectrum(qsp[i],fname);
    }
  }

  if(verbose) {
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
    t.print_time();
  }

  return 0;
}
