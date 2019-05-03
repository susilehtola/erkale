/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "../checkpoint.h"
#include "../mathf.h"
#include "../emd/emd_gto.h"
#include "../stringutil.h"
#include "../settings.h"

/// Absolute tolerance for normalization of basis functions
const double normtol=1e-10;

/// Absolute tolerance in total energy
const double Etol=1e-5;
/// Absolute tolerance for density matrix difference
const double dPtol=1e-6;
/// Absolute tolerance for orbital matrix difference
const double dCtol=1e-5;
/// Relative tolerance in orbital energies
const double dEtol=1e-5;

/// Tolerance for the number of electrons estimate
const double P0TOL=50.0;
/// Tolerance for the EMD error estimate
const double P2TOL=50.0;
/// Orbitals are deemed degenerate if the energy difference is less than
const double odegthr=1e-4;

// Check proper normalization of basis
void check_norm(const BasisSet & bas) {
  size_t Nbf=bas.get_Nbf();
  arma::mat S=bas.overlap();

  for(size_t i=0;i<Nbf;i++)
    if(fabs(S(i,i)-1.0)>=normtol) {
      std::ostringstream oss;
      ERROR_INFO();
      fflush(stdout);
      oss << "Function " << i+1 << " is not normalized: norm is " << S(i,i) << "!.\n";
      throw std::runtime_error(oss.str());
    }
}

void fix_signs(arma::mat & C, const arma::mat & Cr, const arma::mat & S) {
  for(size_t io=0;io<Cr.n_cols;io++) {
    // Get dot product
    double dp=arma::as_scalar(arma::trans(C.col(io))*S*Cr.col(io));
    // Invert sign?
    if(dp<0.0)
      C.col(io)*=-1.0;
  }
}

double C_diff(const arma::mat & Cr, const arma::mat & S, arma::mat & Cc, const arma::vec & E) {
  size_t io=0;
  while(io<Cc.n_cols-1) {
    // Check for degenerate orbitals
    size_t jo=io;
    while(jo+1 < Cc.n_cols && fabs(E(io)-E(jo+1))<odegthr)
      jo++;
    if(jo!=io) {
      // Calculate projection of orbitals
      arma::mat Op(arma::trans(Cc.cols(io,jo))*S*Cr.cols(io,jo));
      // Rotate orbitals to match original ones
      Cc.cols(io,jo)=Cc.cols(io,jo)*Op;
    }

    // Next orbital
    io=jo+1;
  }

  // Get norm
  return rms_norm(Cc-Cr);
}

double E_diff(const arma::mat & Er, const arma::vec & Ec) {
  // Normalization vector
  arma::vec Enorm(arma::abs(Er));
  Enorm=arma::max(Enorm,arma::ones(Er.n_elem));

  return arma::max(arma::abs((Er-Ec)/Enorm));
}

Settings settings;

int main(int argc, char ** argv) {
  // Get reference directory
  char * refdir=getenv("ERKALE_REFDIR");
  if(!refdir)
    throw std::runtime_error("Need to set reference directory!\n");

  fprintf(stderr,"Reference directory is set to \"%s\".\n",refdir);
  fflush(stderr);

  if(argc != 2) {
    std::ostringstream oss;
    oss << "Usage: " << argv[0] << " chkpt\n";
    throw std::runtime_error(oss.str());
  }

  // Current file
  std::string curfile(argv[1]);
  if(!file_exists(curfile)) {
    std::ostringstream oss;
    oss << "Current checkpoint file \"" << curfile << "\" does not exist!\n";
    throw std::runtime_error(oss.str());
  }
  Checkpoint cur(curfile,false);

  // Reference file
  std::string reffile=std::string(refdir)+"/"+curfile;
  if(!file_exists(reffile)) {
    std::ostringstream oss;
    oss << "Reference checkpoint file \"" << reffile << "\" does not exist!\n";
    throw std::runtime_error(oss.str());
  }
  Checkpoint ref(reffile,false);

  // Check consistency of run type
  int rrestr, crestr;
  ref.read("Restricted",rrestr);
  cur.read("Restricted",crestr);
  if(rrestr != crestr)
    throw std::runtime_error("Run types don't match!\n");

  // Basis sets
  BasisSet bref, bcur;
  ref.read(bref);
  cur.read(bcur);

  // Check that basis sets match
  if( ! (bref == bcur) )
    throw std::runtime_error("Basis sets don't match!\n");

  // Check normalization of basis
  check_norm(bref);
  check_norm(bcur);

  // Overlap matrix
  arma::mat S(bcur.overlap());

  if(rrestr) {
    int Nelr, Nelc;
    ref.read("Nel",Nelr);
    cur.read("Nel",Nelc);
    if(Nelr != Nelc) throw std::runtime_error("Amount of electrons doesn't match!\n");

   // Total energies
    energy_t Er, Ec;
    ref.read(Er);
    cur.read(Ec);
    double dE=Ec.E-Er.E;
    printf("Total energy difference %e\n",dE);
    fflush(stdout);
    if(fabs(dE) > Etol) throw std::runtime_error("Total energies don't match!\n");

    // Densities
    arma::cx_mat Pref, Pcur;
    arma::mat Prefr, Prefi, Pcurr, Pcuri;
    ref.read("P",Prefr);
    if(ref.exist("P_im")) {
      ref.read("P_im",Prefi);
      Pref=Prefr*COMPLEX1 + Prefi*COMPLEXI;
    } else
      Pref=Prefr*COMPLEX1;

    cur.read("P",Pcurr);
    if(cur.exist("P_im")) {
      cur.read("P_im",Pcuri);
      Pcur=Pcurr*COMPLEX1 + Pcuri*COMPLEXI;
    } else
      Pcur=Pcurr*COMPLEX1;

    // Check norm
    double Nelnum=std::real(arma::trace(Pcur*S));
    double dNel=Nelnum-Nelc;
    printf("Electron count difference %e\n",dNel);
    fflush(stdout);
    if(fabs(dNel)>dPtol) throw std::runtime_error("Norm of density matrix is wrong.\n");

    // Check difference
    double dP=rms_cnorm(Pcur-Pref);
    printf("Density matrix difference %e\n",dP);
    if(dP>dPtol) throw std::runtime_error("Density matrices differ!\n");

    // Orbitals
    arma::mat Cref, Ccur;
    ref.read("C",Cref);
    cur.read("C",Ccur);
    if(Cref.n_cols != Ccur.n_cols) throw std::runtime_error("Amount of orbitals doesn't match!\n");

    // Fix orbital phase signs
    fix_signs(Ccur,Cref,S);

    // Orbital energies
    arma::vec Eref, Ecur;
    ref.read("E",Eref);
    cur.read("E",Ecur);

    // Calculate differences of nondegenerate orbitals
    double dC=C_diff(Cref,S,Ccur,Ecur);
    printf("Orbital matrix difference %e\n",dC);
    fflush(stdout);
    //    if(dC>dCtol) throw std::runtime_error("Orbital coefficients differ!\n");

    // Orbital energy differences
    double dEo=E_diff(Eref,Ecur);
    printf("Orbital energy difference %e\n",dEo);
    fflush(stdout);
    if(dEo > dEtol) throw std::runtime_error("Orbital energies differ!\n");

    // Check EMD
    GaussianEMDEvaluator eval(bcur,Pcur);
    EMD emd(&eval, &eval, Nelc, 0, 0);
    emd.initial_fill(false);
    emd.find_electrons(false);
    emd.optimize_moments(false);

    // Get moments
    arma::mat mom=emd.moments();

    // Compare <p^2> with T and <p^0> with tr(P*S)
    double p0diff=mom(2,1)-std::real(arma::trace(Pcur*S));
    double p0err=mom(2,2);
    printf("<p^0> - N = % e, d<p^0> = %e\n",p0diff,p0err);
    double p2diff=mom(4,1)-2.0*std::real(arma::trace(Pcur*bcur.kinetic()));
    double p2err=mom(4,2);
    printf("<p^2> - T = % e, d<p^2> = %e\n",p2diff,p2err);
    fflush(stdout);
    if(fabs(p0diff)>=P0TOL*p0err || fabs(p2diff)>P2TOL*p2err)
      throw std::runtime_error("EMD failed.\n");

  } else {
    int Nelar, Nelbr, Nelr;
    int Nelac, Nelbc, Nelc;
    ref.read("Nel-a",Nelar);
    ref.read("Nel-b",Nelbr);
    ref.read("Nel",Nelr);

    cur.read("Nel-a",Nelac);
    cur.read("Nel-b",Nelbc);
    cur.read("Nel",Nelc);
    if(Nelar != Nelac || Nelbr != Nelbc || Nelr != Nelc)
      throw std::runtime_error("Amount of electrons doesn't match!\n");

    // Total energies
    energy_t Er, Ec;
    ref.read(Er);
    cur.read(Ec);
    double dE=Ec.E-Er.E;
    printf("Total energy difference %e\n",dE);
    fflush(stdout);
    if(fabs(dE) > Etol) throw std::runtime_error("Total energies don't match!\n");

    // Densities
    arma::mat Paref, Pbref, Prefr, Prefi;
    arma::mat Pacur, Pbcur, Pcurr, Pcuri;
    arma::cx_mat Pcur, Pref;
    ref.read("Pa",Paref);
    ref.read("Pb",Pbref);
    ref.read("P",Prefr);
    if(ref.exist("P_im")) {
      ref.read("P_im",Prefi);
      Pref=Prefr*COMPLEX1 + Prefi*COMPLEXI;
    } else
      Pref=Prefr*COMPLEX1;

    cur.read("Pa",Pacur);
    cur.read("Pb",Pbcur);
    cur.read("P",Pcurr);
    if(cur.exist("P_im")) {
      cur.read("P_im",Pcuri);
      Pcur=Pcurr*COMPLEX1 + Pcuri*COMPLEXI;
    } else
      Pcur=Pcurr*COMPLEX1;

    // Check norm
    double Nelanum=arma::trace(Pacur*S);
    double Nelbnum=arma::trace(Pbcur*S);
    double Nelnum=std::real(arma::trace(Pcur*S));

    // XRS calculation?
    bool xrs, xrsspin;
    std::string xrsmethod;
    try {
      cur.read("XRSMethod",xrsmethod);
      cur.read("XRSSpin",xrsspin);
      xrs=true;
    } catch(std::runtime_error &) {
      xrs=false;
    }

    // Alpha electron count error
    double dNela;
    if(xrs && stricmp(xrsmethod,"TP")==0  && !xrsspin)
      dNela=Nelanum + 0.5 - Nelac;
    else if(xrs && stricmp(xrsmethod,"FCH")==0 && !xrsspin)
      dNela=Nelanum + 1.0 - Nelac;
    else
      dNela=Nelanum - Nelac;
    printf("Alpha electron count difference %e\n",dNela);
    fflush(stdout);
    if(fabs(dNela)>dPtol) throw std::runtime_error("Norm of alpha density matrix is wrong.\n");

    // Beta electron count error
    double dNelb;
    if(xrs && stricmp(xrsmethod,"TP")==0  && xrsspin)
      dNelb=Nelbnum + 0.5 - Nelbc;
    else if(xrs && stricmp(xrsmethod,"FCH")==0 && xrsspin)
      dNelb=Nelbnum + 1.0 - Nelbc;
    else
      dNelb=Nelbnum - Nelbc;
    printf("Beta  electron count difference %e\n",dNela);
    fflush(stdout);
    if(fabs(dNelb)>dPtol) throw std::runtime_error("Norm of beta  density matrix is wrong.\n");

    double dNel;
    if(xrs && stricmp(xrsmethod,"TP")==0)
      dNel=Nelnum + 0.5 - Nelc;
    else if(xrs && stricmp(xrsmethod,"FCH")==0)
      dNel=Nelnum + 1.0 - Nelc;
    else
      dNel=Nelnum - Nelc;
    printf("Total electron count difference %e\n",dNela);
    fflush(stdout);
    if(fabs(dNel)>dPtol)
      throw std::runtime_error("Norm of total density matrix is wrong.\n");

    // Check differences
    double dPa=rms_norm(Pacur-Paref);
    printf("Alpha density matrix difference %e\n",dPa);
    fflush(stdout);
    if(dPa>dPtol) throw std::runtime_error("Alpha density matrices differ!\n");
    double dPb=rms_norm(Pbcur-Pbref);
    printf("Beta  density matrix difference %e\n",dPb);
    fflush(stdout);
    if(dPb>dPtol) throw std::runtime_error("Density matrices differ!\n");

    double dP=rms_cnorm(Pcur-Pref);
    printf("Total density matrix difference %e\n",dP);
    fflush(stdout);
    if(dP >dPtol) throw std::runtime_error("Total density matrices differ!\n");

    // Orbitals
    arma::mat Caref, Cbref, Cacur, Cbcur;
    ref.read("Ca",Caref);
    ref.read("Cb",Cbref);
    cur.read("Ca",Cacur);
    cur.read("Cb",Cbcur);

    if(Caref.n_cols != Cacur.n_cols)
      throw std::runtime_error("Amount of alpha orbitals doesn't match!\n");
    if(Cbref.n_cols != Cbcur.n_cols)
      throw std::runtime_error("Amount of beta  orbitals doesn't match!\n");

    // Fix orbital phase signs
    fix_signs(Cacur,Caref,S);
    fix_signs(Cbcur,Cbref,S);

    // Orbital energies
    arma::vec Earef, Ebref, Eacur, Ebcur;
    ref.read("Ea",Earef);
    ref.read("Eb",Ebref);
    cur.read("Ea",Eacur);
    cur.read("Eb",Ebcur);

    // Check differences
    double dCa=C_diff(Caref,S,Cacur,Eacur);
    printf("Alpha orbital matrix difference %e\n",dCa);
    // if(dCa>dCtol) throw std::runtime_error("Alpha orbital coefficients differ!\n");
    double dCb=C_diff(Cbref,S,Cbcur,Ebcur);
    printf("Beta  orbital matrix difference %e\n",dCb);
    // if(dCb>dCtol) throw std::runtime_error("Beta  orbital coefficients differ!\n");

    double dEoa=E_diff(Earef,Eacur);
    printf("Alpha orbital energy difference %e\n",dEoa);
    fflush(stdout);
    if(dEoa > dEtol) throw std::runtime_error("Alpha orbital energies differ!\n");
    double dEob=E_diff(Ebref,Ebcur);
    printf("Beta  orbital energy difference %e\n",dEob);
    fflush(stdout);
    if(dEob > dEtol) throw std::runtime_error("Beta orbital energies differ!\n");

    // Amount of electrons in density matrix
    double Pnel(Nelc);
    if(xrs) {
      if(stricmp(xrsmethod,"FCH")==0)
	Pnel-=1.0;
      else if(stricmp(xrsmethod,"TP")==0)
	Pnel-=0.5;
    }

    // Check EMD
    GaussianEMDEvaluator eval(bcur,Pcur);
    EMD emd(&eval, &eval, Pnel, 0, 0);
    emd.initial_fill(false);
    emd.find_electrons(false);
    emd.optimize_moments(false);
    // Get moments
    arma::mat mom=emd.moments();

    // Compare <p^2> with T and <p^0> with tr(P*S)
    double p0diff=mom(2,1)-std::real(arma::trace(Pcur*S));
    double p0err=mom(2,2);
    printf("<p^0> - N = %e, d<p^0> = %e\n",p0diff,p0err);
    double p2diff=mom(4,1)-2.0*std::real(arma::trace(Pcur*bcur.kinetic()));
    double p2err=mom(4,2);
    printf("<p^2> - T = %e, d<p^2> = %e\n",p2diff,p2err);
    fflush(stdout);
    if(fabs(p0diff)>=P0TOL*p0err || fabs(p2diff)>P2TOL*p2err)
      throw std::runtime_error("EMD failed.\n");
  }
}
