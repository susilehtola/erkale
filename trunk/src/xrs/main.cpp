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

// Do fractional occupation on alpha spin
#define FRACOCC

// Mix Fock matrices (otherwise: mix density)
#define MIXFOCK

// Precision for energy shift: 0.01 eV
#define ESHIFTTOL 3.675e-4

#include "bfprod.h"
#include "fourierprod.h"

#include "basis.h"
#include "basislibrary.h"
#include "dftfuncs.h"
#include "elements.h"
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



/// Use converged SCF potential Ha to solve orbitals in augmented basis.
void augmented_solution(const BasisSet & basis, const Settings & set, const arma::mat & H, arma::mat & C, arma::vec & E, size_t xcatom, size_t & ixc_orb, size_t nocc) {
  // Get indices of atoms to augment
  std::vector<size_t> augind=parse_range(splitline(set.get_string("XRSAugment"))[0]);
  // Convert to C++ indexing
  for(size_t i=0;i<augind.size();i++) {
    if(augind[i]==0)
      throw std::runtime_error("Error - nuclear index must be positive!\n");
    augind[i]--;
  }

  // Form augmented basis
  BasisSet augbas(basis);

  printf("\n\nAugmenting\n");

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
	augbas.add_shell(ind,am,C);
    }
  }

  augbas.sort();

  // Finalize augmentation basis
  augbas.finalize();

  // Amount of independent functions in augmented basis set is
  const size_t Naug=augbas.get_Nbf();
  printf("Original basis had %i functions, augmented basis has %i.\n",(int) basis.get_Nbf(),(int) augbas.get_Nbf());

  // Overlap matrix in original basis
  arma::mat ss=basis.overlap();
  // Overlap matrix between normal and augmented basis
  arma::mat Ss=augbas.overlap(basis);
  // Overlap matrix in augmented basis
  arma::mat SS=augbas.overlap();

  arma::vec sval;
  arma::mat svec;
  eig_sym_ordered(sval,svec,ss);
  arma::mat sinv(ss);
  sinv.zeros();
  for(size_t i=0;i<sval.n_elem;i++)
    sinv+=svec.col(i)*arma::trans(svec.col(i))/sval(i);
  arma::mat ssinv=ss*sinv;
  for(size_t i=0;i<ssinv.n_rows;i++)
    for(size_t j=0;j<ssinv.n_cols;j++)
      if(i!=j) {
        if(fabs(ssinv(i,j))>=1e-12)
          printf("ssinv(%i,%i)=%e\n",(int) i,(int) j,ssinv(i,j));
      } else if(fabs(ssinv(i,j)-1.0)>=1e-9)
        printf("ssinv(%i,%i)=%e\n",(int) i,(int) j,ssinv(i,j));

  // Do eigendecomposition of S
  arma::vec Sval;
  arma::mat Svec;
  eig_sym_ordered(Sval,Svec,SS);
  printf("Condition number of overlap matrix is %e.\n",Sval(0)/Sval(Sval.n_elem-1));

  // Count the number of linearly independent functions
  size_t Nind=0;
  for(size_t i=0;i<Naug;i++)
    if(Sval(i)>=1e-5)
      Nind++;
  // ... and get rid of the linearly dependent ones. The eigenvalues
  // and vectors are in the order of increasing eigenvalue, so we want
  // the tail.
  Sval=Sval.subvec(Sval.n_elem-Nind,Sval.n_elem-1);
  Svec=Svec.submat(0,Svec.n_cols-Nind,Svec.n_rows-1,Svec.n_cols-1);

  // Form inverse matrix
  arma::mat Sinv(Naug,Naug);
  Sinv.zeros();
  for(size_t i=0;i<Sval.n_elem;i++)
    Sinv+=Svec.col(i)*arma::trans(Svec.col(i))/Sval(i);

  printf("Augmented basis has %i linearly independent and %i dependent functions.\n",(int) Nind,(int) (Naug-Nind));

  // Amount of orbitals is
  const size_t Norb=C.n_cols;

  // Project SCF orbitals to orthogonal augmented basis
  arma::mat Caug(Naug,Nind);
  Caug.submat(0,0,Naug-1,Norb-1)=Sinv*Ss*C;
  // Initialize the rest of the orbitals
  Caug.submat(0,Norb,Naug-1,Nind-1).ones();


  // Test inverse matrix
  arma::mat SSinv=SS*Sinv;
  for(size_t i=0;i<SSinv.n_rows;i++)
    for(size_t j=0;j<SSinv.n_cols;j++)
      if(i!=j) {
	if(fabs(SSinv(i,j))>=1e-9)
	  printf("SSinv(%i,%i)=%e\n",(int) i,(int) j,SSinv(i,j));
      } else
	if(fabs(SSinv(i,j)-1.0)>=1e-9)
	  printf("SSinv(%i,%i)=%e\n",(int) i,(int) j,SSinv(i,j));


  /*
  // Form scaled vectors
  arma::mat Sm(Svec);
  arma::mat Sd(Svec);
  for(size_t i=0;i<Sval.n_elem;i++) {
    double ss=sqrt(Sval(i));
    Sm.col(i)*=ss;
    Sd.col(i)/=ss;
  }
  */

  /* Transform orbital coefficients to augmented basis */

  // Form the rest of the orbitals with Gram-Schmidt method
  for(size_t io=0;io<Nind;io++) {
    // Compute product with S
    arma::vec tmp=SS*Caug.col(io);

    // Orthogonalize with respect to all preceding orbitals.
    for(size_t jo=0;jo<io;jo++)
      Caug.col(io)-=arma::dot(tmp,Caug.col(jo))*Caug.col(jo);

    // Compute norm
    double norm=arma::as_scalar(arma::trans(C.col(io))*SS*C.col(io));
    printf("Orbital %i norm %e.\n",(int) io,norm);

    // and normalize.
    Caug.col(io)/=sqrt(norm);
  }

  // Check orthonormality
  arma::mat Saugorb=arma::trans(Caug)*SS*Caug;
  for(size_t i=0;i<Nind;i++)
    for(size_t j=0;j<Nind;j++)
      if(i!=j) {
	if(fabs(Saugorb(i,j))>=1e-14)
	  printf("Saugorb(%i,%i)=%e\n",(int) i,(int) j,Saugorb(i,j));
      } else if(fabs(Saugorb(i,j)-1.0)>=1e-9)
	printf("Saugorb(%i,%i)=%e\n",(int) i,(int) j,Saugorb(i,j));

  /*
  // Estimate energies of the orbitals.
  arma::vec Evals=arma::trans(Caug)*Haug*Caug;
  for(size_t i=0;i<Caug.n_cols;i++)
    printf("Orbital %i: energy % .10e\testimated % .10e\n",(int) i+1,Eaug(i),Evals(i,i));

  // Find excited orbital
  //  ixc_orb=find_excited_orb(C,basis,xcatom,nocc);
  */
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
    sp.w=1.0/3.0*(wx*wx + wy*wy + wz*wz);
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
  for(size_t i=0;i<grid.size();i++)
    // Renormalize weights by 4\pi, since otherwise sum of weights is 4\pi
    grid[i].w/=4.0*M_PI;

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

  // Form products of basis functions.
  const size_t Nbf=basis.get_Nbf();

  std::vector<prod_gaussian_3d> bfprod=compute_products(basis);

  printf("Products done in %s.\n",t.elapsed().c_str());
  t.set();

  // Check the products
  arma::mat S=basis.overlap();
  for(size_t i=0;i<Nbf;i++)
    for(size_t j=0;j<=i;j++) {
      double s=S(i,j);
      double intg=bfprod[(i*(i+1))/2+j].integral();

      if(fabs(s-intg)>std::max(DBL_EPSILON,10*DBL_EPSILON*std::max(fabs(s),fabs(intg))))
	fprintf(stderr,"Products differ by %e at (%i,%i): %e %e\n",fabs(s-intg),(int) i, (int) j, s,intg);
    }


  // Form Fourier transforms of the products.
  std::vector<prod_fourier> bffour=fourier_transform(bfprod);
  printf("Fourier transform done in %s.\n",t.elapsed().c_str());
  t.set();

  // Get the grid for computing the spherical averages.
  std::vector<angular_grid_t> grid=form_angular_grid(2*basis.get_max_am());
  for(size_t i=0;i<grid.size();i++)
    // Renormalize weights by 4\pi, since otherwise sum of weights is 4\pi
    grid[i].w/=4.0*M_PI;

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
  printf("Computing transitions using Fourier method.\n");
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

bool load(arma::mat & Ca, arma::mat & Cb, arma::vec & Ea, arma::vec & Eb, arma::mat & Ha, arma::mat & Hb) {
  bool caok=Ca.quiet_load(".erkale_Ca",arma::arma_binary);
  bool cbok=Cb.quiet_load(".erkale_Cb",arma::arma_binary);
  bool Eaok=Ea.quiet_load(".erkale_Ea",arma::arma_binary);
  bool Ebok=Eb.quiet_load(".erkale_Eb",arma::arma_binary);
  bool Haok=Ha.quiet_load(".erkale_Ha",arma::arma_binary);
  bool Hbok=Hb.quiet_load(".erkale_Hb",arma::arma_binary);

  return caok && cbok && Eaok && Ebok && Haok && Hbok;
}

void save(const arma::mat & Ca, const arma::mat & Cb, const arma::vec & Ea, const arma::vec & Eb, const arma::mat & Ha, const arma::mat & Hb) {
  Ca.save(".erkale_Ca",arma::arma_binary);
  Cb.save(".erkale_Cb",arma::arma_binary);
  Ea.save(".erkale_Ea",arma::arma_binary);
  Eb.save(".erkale_Eb",arma::arma_binary);
  Ha.save(".erkale_Ha",arma::arma_binary);
  Hb.save(".erkale_Hb",arma::arma_binary);
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

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_bool("XRSInit","Do ground state calculation first?",1);
  set.add_bool("XRSEnergyCor","Correct for excitation energy",0);
  set.add_string("XRSAugment","(EXPERIMENTAL) Which atoms to augment with diffuse functions? E.g. 1,3-5,10","");

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
  const bool enercor=set.get_bool("XRSEnergyCor");
  bool init=set.get_bool("XRSInit");

  if(enercor && !init) {
    fprintf(stderr,"Energy correction requested. Turning on ground state calculation.\n");
    set.set_bool("XRSInit",1);
    init=1;
  }

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
  if(set.get_bool("Decontract"))
    baslib.decontract();
  printf("\n");

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
  double initfac=set.get_double("DeltaInit");
  init_conv.deltaEmax*=initfac;
  init_conv.deltaPmax*=initfac;
  init_conv.deltaPrms*=initfac;

  // TPA solution
  uscf_t sol;

  // Try to load orbitals and energies
  bool loadok=load(sol.Ca,sol.Cb,sol.Ea,sol.Eb,sol.Ha,sol.Hb);

  if(loadok) {
    printf("Loaded orbitals from file.\n");

    // Check sizes of matrices
    double Sratio;
    arma::mat Sinvh=BasOrth(basis.overlap(),set,Sratio);
    // Number of orbitals
    size_t Norb=Sinvh.n_cols;
    // Number of basis functions
    size_t Nbf=Sinvh.n_rows;

    bool Eok=(sol.Ea.n_elem==Norb) && (sol.Eb.n_elem==Norb);
    bool Caok=(sol.Ca.n_rows==Nbf) && (sol.Ca.n_cols==Norb);
    bool Cbok=(sol.Cb.n_rows==Nbf) && (sol.Cb.n_cols==Norb);
    bool Haok=(sol.Ha.n_rows==Nbf) && (sol.Ha.n_cols==Nbf);
    bool Hbok=(sol.Hb.n_rows==Nbf) && (sol.Hb.n_cols==Nbf);

    loadok=Eok && Caok && Cbok && Haok && Hbok;
    if(!loadok)
      printf("Inconsistency in loaded orbitals. Performing calculation.\n");
  }

  // Index of excited orbital
  size_t xcorb;

  // Ground state
  rscf_t gsol;
  // 1st excited state
  uscf_t esol;

  if(!loadok) {
    // Initialize solver
    XRSSCF solver(basis,set);

    // Initialize calculation with ground state if necessary
    if(init) {
      // Get occupancies
      std::vector<double> occs=get_restricted_occupancy(set,basis);
      // Solve ground state
      solver.RDFT(gsol,occs,init_conv,dft_init);

      // Copy solution
      sol.Ca=gsol.C;
      sol.Cb=gsol.C;
      sol.Ea=gsol.E;
      sol.Eb=gsol.E;

      // Find localized orbital
      size_t ixc_orb=find_excited_orb(gsol.C,basis,xcatom,basis.Ztot()/2);
      // Expand localized orbital
      lmtrans lmground(gsol.C.submat(0,ixc_orb,gsol.C.n_rows,ixc_orb),basis,basis.get_coords(xcatom));
      // and save it
      lmground.write_prob(0,"ground_orb.dat");

      // Do energy correction?
      if(enercor) {
	esol.Ca=gsol.C;
	esol.Cb=gsol.C;
	esol.Ea=gsol.E;
	esol.Eb=gsol.E;

	solver.full_hole(xcatom,esol,conv,dft);
      }
    }

    // Proceed with TPA calculation
    xcorb=solver.half_hole(xcatom,sol,init_conv,dft_init);
    xcorb=solver.half_hole(xcatom,sol,conv,dft);

    // Save orbitals and energies
    save(sol.Ca,sol.Cb,sol.Ea,sol.Eb,sol.Ha,sol.Hb);

    printf("\n\n");
  } else {
    int nocc=basis.Ztot()/2;
    xcorb=find_excited_orb(sol.Ca,basis,xcatom,nocc);
  }

  /*
  // Print info about alpha orbitals
  print_info(sol.Ca,sol.Ea,basis);
  printf("\n\n");
  */

  // Number of occupied states is
  size_t nocc=basis.Ztot()/2.0;

  // Augment the solutions if necessary
  if(stricmp(set.get_string("XRSAugment"),"")!=0)
    augmented_solution(basis,set,sol.Ha,sol.Ca,sol.Ea,xcatom,xcorb,nocc);

  // Compute dipole transitions
  std::vector<spectrum_t> sp=compute_transitions(basis,sol.Ca,sol.Ea,xcatom,xcorb,nocc);
  // Save spectrum
  save_spectrum(sp);

  double shift=0.0;
  // Correct energies if necessary
  if(enercor) {
    shift=(esol.en.E-gsol.en.E)-sp[0].E;
    printf("Energy shift is %e.\n",shift);

    // Compute shifted energies
    std::vector<spectrum_t> sp_cor(sp);
    for(size_t i=0;i<sp.size();i++)
      sp_cor[i].E+=shift;
    save_spectrum(sp_cor,"dipole_cor.dat");
  }

  // Get values of q to compute for
  std::vector<double> qvals=parse_range_double(set.get_string("XRSQval"));

  // The q-dependent spectra
  std::vector< std::vector<spectrum_t> > qsp;
  // The filename
  std::string spname;

  // Series method
  if(stricmp(set.get_string("XRSQMethod"),"Series")==0) {
    qsp=compute_qdep_transitions_series(basis,sol.Ca,sol.Ea,xcorb,nocc,qvals);
    spname="trans_ser";
  }

  // Fourier method
  if(stricmp(set.get_string("XRSQMethod"),"Fourier")==0) {
    qsp=compute_qdep_transitions_fourier(basis,sol.Ca,sol.Ea,xcorb,nocc,qvals);
    spname="trans_four";
  }

  // Local method (Sakko et al)
  if(stricmp(set.get_string("XRSQMethod"),"Local")==0) {
    qsp=compute_qdep_transitions_local(basis,set,sol.Ca,sol.Ea,xcatom,xcorb,nocc,qvals);
    spname="trans_loc";
  }


  // Save transitions
  for(size_t i=0;i<qvals.size();i++) {
    char fname[80];
    sprintf(fname,"%s-%.2f.dat",spname.c_str(),qvals[i]);
    save_spectrum(qsp[i],fname);

    // and the energy corrected one if necessary
    if(enercor) {
      sprintf(fname,"%s_cor-%.2f.dat",spname.c_str(),qvals[i]);
      std::vector<spectrum_t> tmp(qsp[i]);
      for(size_t j=0;j<tmp.size();j++)
        tmp[j].E+=shift;
      save_spectrum(tmp,fname);
    }
  }

  if(verbose) {
    printf("\nRunning program took %s.\n",t.elapsed().c_str());
    t.print_time();
  }

  return 0;
}
