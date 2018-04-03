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

#include "bfprod.h"
#include "fourierprod.h"
#include "lmtrans.h"
#include "momentum_series.h"
#include "../basis.h"
#include "../basislibrary.h"
#include "../dftgrid.h"
#include "../dftfuncs.h"
#include "../elements.h"
#include "../density_fitting.h"
#include "../linalg.h"
#include "../lmgrid.h"
#include "../mathf.h"
#include "../scf.h"
#include "../settings.h"
#include "../stringutil.h"
#include "../tempered.h"
#include "../timer.h"
#include "../xyzutils.h"
#include "xrsscf.h"

// Needed for libint init
#include "../eriworker.h"

#include <algorithm>
#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <istream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

/**
 * Was loading a success?
 *
 * LOAD_FAIL: Failed to load. (start from scratch)
 * LOAD_SUCC: Succesfully loaded completed calculation. (no calculation needed)
 * LOAD_DIFF: Succesfully loaded calculation of a different type. (use as starting point)
 */
enum loadresult {LOAD_FAIL, LOAD_SUCC, LOAD_DIFF};


enum xrs_method parse_method(const std::string & method) {
  enum xrs_method met;
  if(stricmp(method,"TP")==0)
    met=TP;
  else if(stricmp(method,"FCH")==0)
    met=FCH;
  else if(stricmp(method,"XCH")==0)
    met=XCH;
  else throw std::runtime_error("Unrecognized method.\n");

  return met;
}


/// Augment the basis set with diffuse functions
BasisSet augment_basis(const BasisSet & basis, const Settings & set) {
  // Get indices of atoms to augment
  std::vector<size_t> augind=parse_range(splitline(set.get_string("XRSAugment"))[0]);
  // Convert to C++ indexing
  for(size_t i=0;i<augind.size();i++) {
    if(augind[i]==0)
      throw std::runtime_error("Error - nuclear index must be positive!\n");
    augind[i]--;
  }

  bool verbose=set.get_bool("Verbose");
  if(verbose) {
    printf("\nAugmenting basis set with diffuse functions.\n");
    fflush(stdout);
  }

  // Form augmented basis
  BasisSet augbas(basis);

  // Basis set for augmentation functions
  BasisSetLibrary augbaslib;
  augbaslib.load_basis(set.get_string("XRSDoubleBasis"));

  // Loop over excited atoms
  for(size_t iaug=0;iaug<augind.size();iaug++) {
    // Index of atom is
    const size_t ind=augind[iaug];

    // The symbol of the atom
    std::string el=basis.get_symbol(ind);

    // The basis to use for the atom.
    ElementBasisSet elbas;

    try {
      // Check first if a special set is wanted for given center
      elbas=augbaslib.get_element(el,ind+1);
    } catch(std::runtime_error & err) {
      // Did not find a special basis, use the general one instead.
      elbas=augbaslib.get_element(el,0);
    }

    // Get original number of shells
    size_t Nsh_orig=augbas.get_Nshells();
    // Add shells, no sorting.
    augbas.add_shells(ind,elbas,false);
    // Convert contractions on the added shells
    for(size_t ish=Nsh_orig;ish<augbas.get_Nshells();ish++)
      augbas.convert_contraction(ish);
  }

  // Finalize augmentation basis
  augbas.finalize();

  return augbas;
}

/**
 * Double basis set method
 *
 * Augment the basis with diffuse functions and diagonalize the Fock
 * matrix in the unoccupied space. The occupied orbitals and their
 * energies stay the same in the approximation.
 */
void augmented_solution(const BasisSet & basis, const Settings & set, const uscf_t & sol, size_t nocca, size_t noccb, dft_t dft, BasisSet & augbas, arma::mat & Caug, arma::vec & Eaug, bool spin, enum xrs_method method) {
  Timer ttot;

  augbas=augment_basis(basis,set);
  // Need to update pointers in augbas
  augbas.update_nuclear_shell_list();


  // Amount of functions in original basis set is
  const size_t Nbf=basis.get_Nbf();
  // Total number of functions in augmented set is
  const size_t Ntot=augbas.get_Nbf();
  // Amount of augmentation functions is
  const size_t Naug=Ntot-Nbf;

  bool verbose=set.get_bool("Verbose");

  if(verbose) {
    printf("\nAugmented original basis (%i functions) with %i diffuse functions.\n",(int) Nbf,(int) Naug);
    printf("Computing unoccupied orbitals.\n");
    fflush(stdout);

    fprintf(stderr,"Calculating unoccupied orbitals in augmented basis.\n");
    fflush(stderr);
  }


  // Augmented solution
  uscf_t augsol(sol);

  // Form density matrix.
  arma::mat Paaug(Ntot,Ntot), Pbaug(Ntot,Ntot), Paug(Ntot,Ntot);

  augsol.Pa.zeros(Ntot,Ntot);
  augsol.Pa.submat(0,0,Nbf-1,Nbf-1)=sol.Pa;

  augsol.Pb.zeros(Ntot,Ntot);
  augsol.Pb.submat(0,0,Nbf-1,Nbf-1)=sol.Pb;

  augsol.P=augsol.Pa+augsol.Pb;

  // Checkpoint
  std::string augchk=set.get_string("AugChk");
  bool delchk=false;
  if(stricmp(augchk,"")==0) {
    delchk=true;
    augchk=tempname();
  }

  {
    // Copy base checkpoint to augmented checkpoint
    std::string chk=set.get_string("SaveChk");
    char cmd[chk.size()+augchk.size()+5];
    sprintf(cmd,"cp %s %s",chk.c_str(),augchk.c_str());
    int cperr=system(cmd);
    if(cperr) {
      ERROR_INFO();
      throw std::runtime_error("Error copying checkpoint file.\n");
    }
  }

  Timer taug;

  // Open augmented checkpoint file in write mode, don't truncate it!
  Checkpoint chkpt(augchk,true,false);
  // Update the basis set
  chkpt.write(augbas);

  {
    // Initialize solver
    XRSSCF solver(augbas,set,chkpt,spin);
    // The fitting basis set from the non-augmented basis is enough,
    // since the density is restricted there.
    BasisSet fitbas=basis.density_fitting();
    solver.set_fitting(fitbas);

    // Load occupation number vectors
    std::vector<double> occa, occb;
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);

    // Construct Fock matrix
    DFTGrid grid(&augbas,verbose,dft.lobatto);
    DFTGrid nlgrid(&augbas,verbose,dft.lobatto);
    if(dft.adaptive) {
      // Form DFT quadrature grid
      grid.construct(augsol.Pa,augsol.Pb,dft.gridtol,dft.x_func,dft.c_func);
    } else {
      // Fixed size grid
      bool strictint(set.get_bool("StrictIntegrals"));
      grid.construct(dft.nrad,dft.lmax,dft.x_func,dft.c_func,strictint);
      if(dft.nl)
	nlgrid.construct(dft.nlnrad,dft.nllmax,true,false,false,strictint,true);
    }

    switch(method) {
    case(TP):
      solver.Fock_half_hole(augsol,dft,occa,occb,grid,nlgrid);
      break;

    case(FCH):
    case(XCH):
      solver.Fock_full_hole(augsol,dft,occa,occb,grid,nlgrid,method==XCH);
    }
  }

  if(verbose) {
    printf("Fock operator constructed in augmented space in %s.\n",taug.elapsed().c_str());
    fprintf(stderr,"Constructed augmented Fock operator (%s).\n",taug.elapsed().c_str());
  }

  // Diagonalize unoccupied space. Loop over spin
  for(size_t ispin=0;ispin<2;ispin++) {

    // Form Fock operator
    arma::mat H;
    if(!ispin)
      H=augsol.Ha;
    else
      H=augsol.Hb;

    arma::mat AOtoO;
    if(!ispin)
      AOtoO=project_orbitals(sol.Ca,basis,augbas);
    else
      AOtoO=project_orbitals(sol.Cb,basis,augbas);

    // Amount of occupied orbitals
    size_t nocc;
    if(!ispin)
      nocc=nocca;
    else
      nocc=noccb;

    // Convert Fock operator to unoccupied MO basis.
    taug.set();
    arma::mat H_MO=arma::trans(AOtoO.cols(nocc,AOtoO.n_cols-1))*H*AOtoO.cols(nocc,AOtoO.n_cols-1);
    if(verbose) {
      printf("H_MO formed in %s.\n",taug.elapsed().c_str());
      fflush(stdout);
    }

    // Diagonalize Fockian to find orbitals and energies
    taug.set();
    arma::vec Eval;
    arma::mat Evec;
    eig_sym_ordered(Eval,Evec,H_MO);

    if(verbose) {
      printf("H_MO diagonalized in unoccupied subspace in %s.\n",taug.elapsed().c_str());
      fflush(stdout);
    }

    // Store energies
    arma::vec Ea;
    Ea.zeros(AOtoO.n_cols);
    // Occupied orbital energies
    if(!spin)
      Ea.subvec(0,nocc-1)=sol.Ea.subvec(0,nocc-1);
    else
      Ea.subvec(0,nocc-1)=sol.Eb.subvec(0,nocc-1);
    // Virtuals
    Ea.subvec(nocc,AOtoO.n_cols-1)=Eval;

    // Back-transform orbitals to AO basis
    arma::mat Ca;
    Ca.zeros(Ntot,AOtoO.n_cols);
    // Occupied orbitals, padded with zeros
    Ca.submat(0,0,Nbf-1,nocc-1)=AOtoO.submat(0,0,Nbf-1,nocc-1);
    // Unoccupied orbitals
    Ca.cols(nocc,AOtoO.n_cols-1)=AOtoO.cols(nocc,AOtoO.n_cols-1)*Evec;

    // Save results
    if(!ispin) {
      augsol.Ca=Ca;
      augsol.Ea=Ea;
    } else {
      augsol.Cb=Ca;
      augsol.Eb=Ea;
    }

    // Return variables
    if((bool) ispin==spin) {
      Eaug=Ea;
      Caug=Ca;
    }
  }

  // Write out updated solution vectors
  chkpt.write("Ca",augsol.Ca);
  chkpt.write("Cb",augsol.Cb);
  chkpt.write("Ea",augsol.Ea);
  chkpt.write("Eb",augsol.Eb);
  // and the density matrix
  chkpt.write("Pa",augsol.Pa);
  chkpt.write("Pb",augsol.Pb);
  chkpt.write("P",augsol.P);
  // and the energy
  chkpt.write(augsol.en);

  if(delchk) {
    int err=remove(augchk.c_str());
    if(err) {
      ERROR_INFO();
      throw std::runtime_error("Error removing temporary file.\n");
    }
  }

  if(verbose) {
    printf("Augmentation took %s.\n",ttot.elapsed().c_str());
    fprintf(stderr,"Augmentation done (%s).\n",taug.elapsed().c_str());
    fflush(stdout);
    fflush(stderr);
  }
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
std::vector<spectrum_t> compute_transitions(const BasisSet & basis, const arma::mat & C, const arma::vec & E, size_t iat, size_t ixc, size_t nocc) {
  // Returned array
  std::vector<spectrum_t> ret;

  // Coordinates of excited atom
  coords_t xccen=basis.get_nuclear_coords(iat);
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

std::vector< std::vector<spectrum_t> > compute_qdep_transitions_series(const BasisSet & basis, const arma::mat & C, const arma::vec & E, size_t ixc, size_t nocc, std::vector<double> qval) {
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
  lmtrans lm(C,basis,basis.get_nuclear_coords(iat),Nrad,Lmax,Lquad);

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
    fprintf(out,"%e %e",sp[i].E*HARTREEINEV,sp[i].w);
    for(size_t j=0;j<sp[i].wdec.size();j++)
      fprintf(out," % e",sp[i].wdec[j]);
    fprintf(out,"\n");
  }
  fclose(out);
}

enum loadresult load(const BasisSet & basis, const Settings & set, Checkpoint & chkpt, uscf_t & sol, arma::vec & core) {
  // Was the load a success?
  enum loadresult ok=LOAD_SUCC;

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
  } catch(std::runtime_error & err) {
    ok=LOAD_FAIL;
    //    fprintf(stderr,"Loading failed due to \"%s\".\n",err.what());
  }

  if(ok) {
    // Check consistency
    if(!(basis==loadbas)) {
      ok=LOAD_FAIL;
      //      fprintf(stderr,"Basis sets differ!\n");
    }
  }

  if(ok) {
    // Get number of basis functions
    size_t Nbf=basis.get_Nbf();

    if(sol.Ca.n_rows != Nbf)
      ok=LOAD_FAIL;

    if(sol.Cb.n_rows != Nbf)
      ok=LOAD_FAIL;

    if(sol.Ea.n_elem != sol.Ca.n_cols)
      ok=LOAD_FAIL;

    if(sol.Eb.n_elem != sol.Cb.n_cols)
      ok=LOAD_FAIL;

    if(sol.Ea.n_elem != sol.Eb.n_elem)
      ok=LOAD_FAIL;

    if(sol.Pa.n_rows != Nbf || sol.Pa.n_cols != Nbf)
      ok=LOAD_FAIL;

    if(sol.Pb.n_rows != Nbf || sol.Pb.n_cols != Nbf)
      ok=LOAD_FAIL;

    //    if(!ok) fprintf(stderr,"Dimensions do not match!\n");
  }

  // Was the calculation converged?
  if(ok) {
    bool conv;
    chkpt.read("Converged",conv);

    if(!conv) {
      //      fprintf(stderr,"Calculation was not converged.\n");
      ok=LOAD_FAIL;
    }
  }

  // Check consistency of spin
  if(ok) {
    bool spin;
    chkpt.read("XRSSpin",spin);
    if(spin!=set.get_bool("XRSSpin")) {
      //      fprintf(stderr,"Excited spin does not match.\n");
      ok=LOAD_FAIL;
    }
  }

  // Check consistency of method
  if(ok) {
    std::string method;
    chkpt.read("XRSMethod",method);
    if(stricmp(method,set.get_string("XRSMethod"))!=0) {
      ok=LOAD_DIFF;
      //      fprintf(stderr,"Calculation methods do not match.\n");
    }
  }

  if(ok) {
    // Everything is OK, finish by loading initial state core orbital
    if(chkpt.exist("Ccore")) {
      printf("Loaded initial state orbital from checkpoint.\n");
      chkpt.read("Ccore",core);
    } else
      // Failed to read core orbital
      ok=LOAD_FAIL;

  } else {
    // Failed to load or solution was not consistent.
    sol.Ca=arma::mat();
    sol.Cb=arma::mat();
    sol.Ea=arma::vec();
    sol.Eb=arma::vec();
    sol.Pa=arma::mat();
    sol.Pb=arma::mat();
  }

  return ok;
}

void print_header() {
#ifdef _OPENMP
  printf("ERKALE - XRS from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - XRS from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

int main(int argc, char **argv) {
  print_header();

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
  // Need to add DFT settings so that DFTDelta setting is available
  set.add_dft_settings();

  // Change default log file
  set.set_string("Logfile","erkale_xrs.log");

  // Use convergence settings similar to StoBe. XCH calculations
  // are hard to converge to the otherwise default settings.
  set.set_double("DeltaEmax",1e-6);
  set.set_double("DeltaPmax",1e-5);
  set.set_double("DeltaPrms",1e-6);
  set.set_double("DFTDelta",100);

  // Add xrs specific settings
  set.add_string("LoadChk","Initialize with ground state calculation from file","");
  set.add_string("SaveChk","Try initializing with and save results to file","erkale_xrs.chk");
  set.add_string("AugChk","Save augmented calculation to file (if applicable)","erkale_xrs_aug.chk");

  set.add_string("XRSDoubleBasis","The augmentation basis to use for double-basis set calculations","X-AUTO");

  set.add_bool("XRSSpin","Spin to excite (false for alpha, true for beta)",false);
  set.add_string("XRSInitialState","Initial atomic state to excite","1s");
  set.add_int("XRSInitialOrbital","Index of orbital in state to excite", 1);
  set.add_string("XRSMethod", "Which kind of calculation to perform: TP, XCH or FCH","TP");

  set.add_string("XRSAugment","Which atoms to augment with diffuse functions? E.g. 1,3:5,10","");
  set.add_double("XRSGridTol","DFT grid tolerance in double basis set calculation",1e-4);

  set.add_string("XRSQval","List or range of Q values to compute","");
  set.add_string("XRSQMethod","Method of computing momentum transfer matrix: Local, Fourier or Series","Fourier");

  set.add_int("XRSNrad","Local: how many point to use in radial integration",200);
  set.add_int("XRSLmax","Local: expand orbitals up to Lmax",5);
  set.add_int("XRSLquad","Local: perform angular expansion using quadrature of Lquad order",30);

  set.parse(std::string(argv[1]),true);

  // Redirect output?
  std::string logfile=set.get_string("Logfile");
  if(stricmp(logfile,"stdout")!=0) {
    // Redirect stdout to file
    FILE *outstream=freopen(logfile.c_str(),"w",stdout);
    if(outstream==NULL) {
      ERROR_INFO();
      throw std::runtime_error("Unable to redirect output!\n");
    }

    // Print out the header in the logfile too
    print_header();
    fprintf(stderr,"\n");
  }

  // Get used settings
  const bool verbose=set.get_bool("Verbose");
  const bool spin=set.get_bool("XRSSpin");

  // Parse method
  enum xrs_method method=parse_method(set.get_string("XRSMethod"));

  // Print out settings
  if(verbose)
    set.print();

  // Read in atoms.
  std::vector<atom_t> atoms;
  std::string atomfile=set.get_string("System");
  if(file_exists(atomfile))
    atoms=load_xyz(atomfile);
  else {
    // Check if a directory has been set
    char * libloc=getenv("ERKALE_SYSDIR");
    if(libloc) {
      std::string filename=std::string(libloc)+"/"+atomfile;
      if(file_exists(filename))
	atoms=load_xyz(filename);
      else
	throw std::runtime_error("Unable to open xyz input file!\n");
    } else
      throw std::runtime_error("Unable to open xyz input file!\n");
  }

  // Get index of excited atom
  size_t xcatom=get_excited_atom_idx(atoms);

  // Read in basis set
  BasisSetLibrary baslib;
  std::string basfile=set.get_string("Basis");
  baslib.load_basis(basfile);

  // Construct basis set
  BasisSet basis;
  construct_basis(basis,atoms,baslib,set);

  // Get exchange and correlation functionals and grid settings
  dft_t dft_init(parse_dft(set,true));
  dft_t dft(parse_dft(set,false));

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

  // Ground state energy
  energy_t gsen;
  bool didgs=false;

  // Initial state
  std::string state=set.get_string("XRSInitialState");
  // Initial orbital
  int iorb=set.get_int("XRSInitialOrbital")-1;

  // Try to load orbitals and energies
  arma::vec core;
  enum loadresult loadok=LOAD_FAIL;
  if(file_exists(set.get_string("SaveChk"))) {
    Checkpoint testload(set.get_string("SaveChk"),false);
    loadok=load(basis,set,testload,sol,core);

    if(loadok==LOAD_SUCC)
      fprintf(stderr,"Loaded converged calculation from checkpoint file.\n");
    else if(loadok==LOAD_DIFF) {
      std::string met;
      testload.read("XRSMethod",met);
      fprintf(stderr,"Initializing with previous %s calculation.\n",met.c_str());
    } else
      fprintf(stderr,"No converged calculation found in SaveChk.\n");

    // Can we get a ground state energy from the checkpoint file?
    if(loadok==LOAD_DIFF && testload.exist("E_GS")) {
      testload.read("E_GS",gsen.E);
      didgs=true;
    }
  }

  // No existing calculation found or system was different => perform calculation
  if(loadok==LOAD_DIFF || loadok==LOAD_FAIL) {

    // Create checkpoint
    Checkpoint chkpt(set.get_string("SaveChk"),true);
    chkpt.write(basis);
    chkpt.write("XRSSpin",set.get_bool("XRSSpin"));
    chkpt.write("XRSMethod",set.get_string("XRSMethod"));

    // Index of core orbital
    int icore=-1;

    // Initialize calculation with ground state if necessary
    if(loadok==LOAD_FAIL) {
      if(set.get_string("LoadChk").size()==0)
	throw std::runtime_error("Need a ground-state calculation in LoadChk to do calculation!\n");

      printf("Initializing with ground-state calculation from %s.\n",set.get_string("LoadChk").c_str());

      // Read checkpoint file
      Checkpoint load(set.get_string("LoadChk"),false);

      // Restricted calculation?
      bool restr;
      load.read("Restricted",restr);

      // Load ground-state energy
      load.read(gsen);
      didgs=true;

      // Load basis
      BasisSet oldbas;
      load.read(oldbas);

      // Check that basis sets are the same
      if(!(basis == oldbas))
	throw std::runtime_error("Must use the same basis for all phases of XRS calculation!\n");

      // Read orbitals
      if(restr) {
	load.read("C",sol.Ca);
	load.read("E",sol.Ea);
	load.read("H",sol.Ha);
	load.read("P",sol.P);
	sol.Eb=sol.Ea;
	sol.Cb=sol.Ca;
	sol.Hb=sol.Ha;
	sol.Pa=sol.Pb=sol.P/2.0;
      } else {
	// Load energies and orbitals
	load.read("Ca",sol.Ca);
	load.read("Ea",sol.Ea);
	load.read("Ha",sol.Ha);
	load.read("Pa",sol.Pa);
	load.read("Cb",sol.Cb);
	load.read("Eb",sol.Eb);
	load.read("Hb",sol.Hb);
	load.read("Pb",sol.Pb);
	load.read("P",sol.P);
      }

      // Determine orbitals, update energies and densities
      if(spin) {
	icore=localize(basis,noccb,xcatom,sol.Cb,state,iorb);
	sol.Eb=arma::diagvec(arma::trans(sol.Cb)*sol.Hb*sol.Cb);

	std::vector<double> occb;
	if(method==XCH)
	  occb=xch_occ(icore,noccb);
	else if(method==FCH)
	  occb=fch_occ(icore,noccb);
	else if(method==TP)
	  occb=tp_occ(icore,noccb);
	sol.Pb=form_density(sol.Cb,occb);

      } else {
	icore=localize(basis,nocca,xcatom,sol.Ca,state,iorb);
	sol.Ea=arma::diagvec(arma::trans(sol.Ca)*sol.Ha*sol.Ca);

	std::vector<double> occa;
	if(method==XCH)
	  occa=xch_occ(icore,nocca);
	else if(method==FCH)
	  occa=fch_occ(icore,nocca);
	else if(method==TP)
	  occa=tp_occ(icore,nocca);
	sol.Pa=form_density(sol.Ca,occa);
      }
      sol.P=sol.Pa+sol.Pb;
    }

    // Proceed with TP calculation. Initialize solver
    XRSSCF solver(basis,set,chkpt,spin);

    // Set initial state core orbital
    if(loadok==LOAD_FAIL) { // LOAD_FAIL
      if(spin)
	solver.set_core(sol.Cb.col(icore));
      else
	solver.set_core(sol.Ca.col(icore));
    } else
      // LOAD_DIFF
      solver.set_core(core);

    // Write number of electrons to file
    chkpt.write("Nel",nocca+noccb);
    chkpt.write("Nel-a",nocca);
    chkpt.write("Nel-b",noccb);

    // Store core orbital
    chkpt.write("Ccore",solver.get_core());

    // Do we have the ground state energy?
    if(didgs)
      chkpt.write("E_GS",gsen.E);

    // Do calculation
    if(method==FCH || method==XCH) {
      if(dft.adaptive)
	solver.full_hole(sol,init_conv,dft_init,method==XCH);
      xcorb=solver.full_hole(sol,conv,dft,method==XCH);

      // Get excited state energy
      energy_t excen;
      chkpt.read(excen);

      // Do we have a ground state energy?
      if(didgs) {
	if(method==XCH) {
	  printf("\nAbsolute energy correction: first peak should be at %.2f eV.\n",(excen.E-gsen.E)*HARTREEINEV);
	  fprintf(stderr,"\nAbsolute energy correction: first peak should be at %.2f eV.\n",(excen.E-gsen.E)*HARTREEINEV);
	} else {
	  printf("Vertical photoionization energy is %.2f eV.\n",(excen.E-gsen.E)*HARTREEINEV);
	  fprintf(stderr,"Vertical photoionization energy is %.2f eV.\n",(excen.E-gsen.E)*HARTREEINEV);
	}
      } else {
	printf("Cannot estimate excitation energy without a ground state calculation.\n");
	fprintf(stderr,"Cannot estimate excitation energy without a ground state calculation.\n");
      }
    } else {
      if(dft.adaptive)
	solver.half_hole(sol,init_conv,dft_init);
      xcorb=solver.half_hole(sol,conv,dft);
    }
    printf("\n\n");

  } else {
    // Loaded calculation, just find excited state

    if(spin)
      xcorb=find_excited_orb(basis,core,sol.Cb,noccb);
    else
      xcorb=find_excited_orb(basis,core,sol.Ca,nocca);
  }

  // Augment the solutions if necessary
  BasisSet augbas;
  arma::mat C_aug;
  arma::vec E_aug;

  if(stricmp(set.get_string("XRSAugment"),"")!=0)
    augmented_solution(basis,set,sol,nocca,noccb,dft,augbas,C_aug,E_aug,spin,method);
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
