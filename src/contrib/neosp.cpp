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

#include "basislibrary.h"
#include "basis.h"
#include <memory>
#include "checkpoint.h"
#include "dftgrid.h"
#include "elements.h"
#include "find_molecules.h"
#include "guess.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
#include "sap.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"
#include "erichol.h"

// Needed for libint init
#include "eriworker.h"

#include "openorbitaloptimizer/scfsolver.hpp"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

//#include "openorbital/scfsolver.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

void print_header() {
#ifdef _OPENMP
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

Settings settings;

int main_guarded(int argc, char **argv) {
  print_header();

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  Timer t;
  t.print_time();

  settings.add_scf_settings();
  settings.add_string("ProtonBasis", "Protonic basis set", "");
  settings.add_string("QuantumProtons", "Indices of protons to make quantum", "");
  settings.add_double("ProtonMass", "Protonic mass", 1836.15267389);
  settings.add_string("LoadChk", "Checkpoint file to load from", "");
  settings.add_bool("FiniteProton", "Use a finite proton model", false);
  settings.add_int("NAvg", "Number of states to average over", 2);

  // Parse settings
  settings.parse(std::string(argv[1]),true);
  settings.print();
  int Q = settings.get_int("Charge");
  int M = settings.get_int("Multiplicity");
  int maxiter = settings.get_int("MaxIter");
  double proton_mass = settings.get_double("ProtonMass");
  double intthr = settings.get_double("IntegralThresh");
  bool verbose = settings.get_bool("Verbose");
  size_t fitmem = 1000000*settings.get_int("FittingMemory");
  std::string loadchk = settings.get_string("LoadChk");
  bool finiteproton = settings.get_bool("FiniteProton");
  int navg = settings.get_int("NAvg");

  // Read in basis set
  BasisSetLibrary pbaslib;
  if(settings.get_string("ProtonBasis").size())
    pbaslib.load_basis(settings.get_string("ProtonBasis"));

  auto atoms=load_xyz(settings.get_string("System"),!settings.get_bool("InputBohr"));
  std::vector<size_t> proton_indices;
  if(stricmp(settings.get_string("QuantumProtons"),"")!=0) {
    // Check for '*'
    std::string str=settings.get_string("QuantumProtons");
    if(str.size()==1 && str[0]=='*') {
      for(size_t i=0;i<atoms.size();i++)
        if(atoms[i].el.compare("H")==0)
          proton_indices.push_back(i);
    } else {
      // Parse and convert to C++ indexing
      proton_indices = parse_range(settings.get_string("QuantumProtons"),true);
    }
  }

  // Collect quantum protons
  std::vector<atom_t> quantum_protons;
  for(auto idx: proton_indices) {
    quantum_protons.push_back(atoms[idx]);
  }
  for(size_t i=0; i<quantum_protons.size(); i++)
    quantum_protons[i].num=i;

  // Collect classical nuclei
  std::vector<std::tuple<int,double,double,double>> classical_nuclei;
  for(size_t i=0;i<atoms.size();i++) {
    // Ghost nucleus
    if(atoms[i].el.size()>3 && atoms[i].el.substr(atoms[i].el.size()-3,3)=="-Bq")
      continue;
    // Skip over quantum nuclei
    bool quantum=false;
    for(auto proton_idx: proton_indices)
      if(proton_idx==i)
        quantum=true;
    if(quantum)
      continue;
    // Add to list
    classical_nuclei.push_back(std::make_tuple(get_Z(atoms[i].el), atoms[i].x, atoms[i].y, atoms[i].z));
  }

  // Set up electronic guess
  if(loadchk == "") {
    throw std::logic_error("Need to have a checkpoint to load\n");
  }
  Checkpoint load(loadchk,false);
  BasisSet basis;
  load.read(basis);

  // Construct the basis set
  BasisSet pbasis;
  construct_basis(pbasis,quantum_protons,pbaslib);

  // Classical nucleus repulsion energy
  double Ecnucr=0.0;
  for(size_t i=0;i<classical_nuclei.size();i++) {
    auto [Qi, xi, yi, zi] = classical_nuclei[i];
    for(size_t j=0;j<i;j++) {
      auto [Qj, xj, yj, zj] = classical_nuclei[j];
      Ecnucr+=Qi*Qj/sqrt(std::pow(xi-xj,2)+std::pow(yi-yj,2)+std::pow(zi-zj,2));
    }
  }

  // Construct linearly independent basis
  arma::mat Sp(pbasis.overlap());
  arma::mat Xp(BasOrth(Sp));

  // Set up ERI evaluator data
  double alpha = 1.0, beta = 0.0, omega = 0.0;
  if(finiteproton) {
    const double fwhm = 1.5900e-5; // 0.8414 fm = 1.5900e-5 bohr
    double sigma = fwhm/sqrt(8.0*log(2.0));

    alpha = 1.0;
    beta = -1.0;
    omega = 1.0/(sqrt(2.0)*sigma);
    printf("Using finite protonic model with fwhm = %e bohr => omega = %e.\n",fwhm,omega);
  }

  // Calculate matrices
  arma::mat Vpc, Tp;
  std::vector<arma::mat> pr;
  if(Sp.n_elem) {
    Vpc=-pbasis.nuclear(classical_nuclei);
    Tp=pbasis.kinetic()/proton_mass;
    pr=pbasis.moment(1);
  }

  std::function<arma::mat(const BasisSet &, const arma::mat &, const BasisSet &)> multicomponent_coulomb_tei = [&](const BasisSet & source_basis, const arma::mat & source_density, const BasisSet & target_basis) {
    // Shells in the two basis sets
    std::vector<GaussianShell> sshells=source_basis.get_shells();
    std::vector<GaussianShell> tshells=target_basis.get_shells();

    // Get shellpairs
    arma::mat Qs, Qt, Ms, Mt;
    double shtol=settings.get_double("IntegralThresh");
    bool verbose=false;
    auto spairs=source_basis.get_eripairs(Qs,Ms,shtol,omega,alpha,beta,verbose);
    auto tpairs=target_basis.get_eripairs(Qt,Mt,shtol,omega,alpha,beta,verbose);

    // Sanity check
    if(source_density.n_rows != source_basis.get_Nbf() or source_density.n_cols != source_basis.get_Nbf())
      throw std::logic_error("Density matrix does not correspond to basis set!\n");
    // Target matrix
    arma::mat Jt(target_basis.get_Nbf(), target_basis.get_Nbf(), arma::fill::zeros);

    // Compute integrals
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // ERI worker. Wrap in unique_ptr so a throw inside the loop
      // doesn't leak the allocation; .get() is used at every call
      // site inside the loop (legacy raw-pointer API).
      auto maxam = std::max(source_basis.get_max_am(),target_basis.get_max_am());
      auto maxncontr = std::max(source_basis.get_max_Ncontr(), target_basis.get_max_Ncontr());
      std::unique_ptr<ERIWorker> eri_owner((omega==0.0 && alpha==1.0 && beta==0.0) ? new ERIWorker(maxam, maxncontr) : new ERIWorker_srlr(maxam,maxncontr,omega,alpha,beta));
      ERIWorker *eri = eri_owner.get();

#ifndef _OPENMP
      int ith=0;
#else
      int ith(omp_get_thread_num());
#pragma omp for schedule(dynamic)
#endif
      for(size_t tp=0;tp<tpairs.size();tp++) {
        // Shells on first pair
        size_t it=tpairs[tp].is;
        size_t jt=tpairs[tp].js;
        // First functions on the first pair is
        size_t it0=tpairs[tp].i0;
        size_t jt0=tpairs[tp].j0;
        // Number of functions
        size_t Nti=tpairs[tp].Ni;
        size_t Ntj=tpairs[tp].Nj;

        // Target integral shell
        arma::mat Jtij(Nti,Ntj,arma::fill::zeros);

        for(size_t sp=0;sp<spairs.size();sp++) {
          // and those on the second pair are analogously
          size_t ks=spairs[sp].is;
          size_t ls=spairs[sp].js;
          size_t ks0=spairs[sp].i0;
          size_t ls0=spairs[sp].j0;
          size_t Nsk=spairs[sp].Ni;
          size_t Nsl=spairs[sp].Nj;

          // Schwarz screening estimate
          double QQ=Qt(it,jt)*Qs(ks,ls);
          if(QQ<shtol) {
            // Skip due to small value of integral. Because the
            // integrals have been ordered wrt Q, all the next ones
            // will be small as well!
            break;
          }

          // Compute integrals
          eri->compute(&tshells[it],&tshells[jt],&sshells[ks],&sshells[ls]);

          // Extract density
          arma::mat Pskl = source_density.submat(ks0,ls0,ks0+Nsk-1,ls0+Nsl-1);

          // Degeneracy factor
          double fac=1.0;
          if(ks!=ls)
            fac=2.0;

          // J_ij = (ij|kl) P_kl using matmul
          arma::mat tei((double *)(eri->getp()->data()),Nsk*Nsl,Nti*Ntj,false,true);
          arma::mat incr = fac*arma::vectorise(Pskl.t()).t()*tei;
          incr.reshape(Ntj,Nti);
          Jtij += incr.t();
        }

        // Now that we've computed the block, store it in the full matrix
        Jt.submat(it0,jt0,it0+Nti-1,jt0+Ntj-1) = Jtij;
        if(it!=jt)
          Jt.submat(jt0,it0,jt0+Ntj-1,it0+Nti-1) = arma::trans(Jtij);
      }

      // eri_owner releases automatically.
    }

    return Jt;
  };

  std::function<arma::mat(const arma::mat & P)> electron_proton_coulomb = [&](const arma::mat & Pe) {
    arma::mat J=-multicomponent_coulomb_tei(basis, Pe, pbasis);
    return J;
  };

  int Nel = basis.Ztot()-Q;
  int Nela = (Nel+M-1)/2;
  int Nelb = Nel-Nela;
  int Np = (int) proton_indices.size();
  printf("Nela = %i Nelb = %i\n",Nela,Nelb);
  if(Nela<0 or Nelb<0)
    throw std::logic_error("Negative number of electrons!\n");
  if(Nelb>Nela)
    throw std::logic_error("Nelb > Nela, check your charge and multiplicity!\n");

  arma::mat Pe;

  if(M==1) {
    arma::mat C;
    load.read("C",C);
    if(Nela>0)
      C = C.cols(0,Nela-1);
    else
      C.set_size(C.n_rows, 0);
    Pe = 2 * C * C.t();

  } else {
    arma::mat Ca, Cb;
    load.read("Ca",Ca);
    load.read("Cb",Cb);
    if(Nela>0)
      Ca = Ca.cols(0,Nela-1);
    else
      Ca.set_size(Ca.n_rows, 0);
    if(Nelb>0)
      Cb = Cb.cols(0,Nelb-1);
    else
      Cb.set_size(Cb.n_rows, 0);
    Pe = Ca*Ca.t() + Cb*Cb.t();
  }

  arma::mat Jep = electron_proton_coulomb(Pe);
  arma::mat Fp = Xp.t() * (Tp + Vpc + Jep) * Xp;

  arma::vec E;
  arma::mat C;
  arma::eig_sym(E, C, Fp);
  E.print("Protonic orbital energies");

  if(E.n_elem >= navg)
    printf("State average of %i first states is % .14e\n",(int) navg,arma::mean(E.subvec(0,navg-1)));
  else
    printf("State average of %i first states is % .14e\n",(int) E.n_elem,arma::mean(E));

  printf("\nRunning program took %s.\n",t.elapsed().c_str());
  return 0;
}

int main(int argc, char **argv) {
#ifdef CATCH_EXCEPTIONS
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
#else
  return main_guarded(argc, argv);
#endif
}
