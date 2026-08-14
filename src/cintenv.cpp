/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "cintenv.h"
#include "mathf.h"

#include <algorithm>
#include <stdexcept>

extern "C" {
#include <cint.h>
#include <cint_funcs.h>

// The plain three- and two-center kernels are exported by the library
// but missing from qcint's cint_funcs.h
extern CINTIntegralFunction int3c2e_cart;
extern CINTIntegralFunction int3c2e_sph;
extern CINTIntegralFunction int2c2e_cart;
extern CINTIntegralFunction int2c2e_sph;
extern CINTOptimizerFunction int3c2e_optimizer;
extern CINTOptimizerFunction int2c2e_optimizer;
}
// cint.h defines function-like atm() and bas() accessor macros, which
// mangle any same-named variable that is followed by a parenthesis
#undef atm
#undef bas



int cint_1e_ncomp(cint_1e_kernel_t kernel) {
  switch(kernel) {
  case CINT1E_OVLP:
  case CINT1E_KIN:
  case CINT1E_RINV:
    return 1;
  case CINT1E_OVLPIP:
  case CINT1E_IPOVLP:
  case CINT1E_IPKIN:
  case CINT1E_KINIP:
  case CINT1E_IPRINV:
  case CINT1E_R:
    return 3;
  case CINT1E_RR:
    return 9;
  case CINT1E_RRR:
    return 27;
  case CINT1E_RRRR:
    return 81;
  default:
    ERROR_INFO();
    throw std::logic_error("Unknown one-electron kernel!\n");
  }
}

CintEnv::CintEnv() : Nsh_orb_(0), max_Nbf_(0), lm_(true) {
}

CintEnv::CintEnv(const BasisSet & basis, bool build_opts) {
  const std::vector<GaussianShell> & sh = basis.get_shells_ref();
  build(sh, sh.size(), build_opts);
}

CintEnv::CintEnv(const BasisSet & basis, const BasisSet & aux, bool build_opts) {
  std::vector<GaussianShell> sh(basis.get_shells());
  const std::vector<GaussianShell> & auxsh = aux.get_shells_ref();
  const size_t Norb = sh.size();
  sh.insert(sh.end(), auxsh.begin(), auxsh.end());
  build(sh, Norb, build_opts);
}

CintEnv::CintEnv(const std::vector<GaussianShell> & sh, bool build_opts) {
  build(sh, sh.size(), build_opts);
}

CintEnv::CintEnv(const std::vector<GaussianShell> & sh, size_t Nsh_orbital, bool build_opts) {
  build(sh, Nsh_orbital, build_opts);
}

CintEnv::OptSet::~OptSet() {
  for(size_t i=0;i<opts.size();i++)
    if(opts[i]) {
      CINTOpt * o=(CINTOpt *) opts[i];
      CINTdel_optimizer(&o);
      opts[i]=nullptr;
    }
}

void CintEnv::build(const std::vector<GaussianShell> & sh, size_t Nsh_orbital, bool build_opts) {
  Nsh_orb_=Nsh_orbital;
  shells_=sh;

  if(!shells_.size())
    throw std::logic_error("CintEnv: no shells to build an environment for!\n");

  // ERKALE's optlm keeps the s and p shells cartesian even when the rest
  // of the basis is spherical; since libcint's spherical s and p shells
  // coincide with the cartesian ones, such a basis is evaluated with the
  // spherical kernels. Only shells with l >= 2 decide the mode, and they
  // all have to agree.
  bool have_lm=false, have_cart=false;
  for(size_t is=0;is<shells_.size();is++) {
    if(shells_[is].get_am()<2)
      continue;
    if(shells_[is].lm_in_use())
      have_lm=true;
    else
      have_cart=true;
  }
  if(have_lm && have_cart)
    throw std::runtime_error("CintEnv: the basis mixes spherical and cartesian shells of l >= 2, which libcint cannot evaluate in a single call.\n");
  // A basis of only s and p shells is the same either way; use the
  // spherical kernels, as they are what the rest of ERKALE defaults to.
  lm_=!have_cart;

  // Collect the distinct centers
  std::vector<coords_t> centers;
  std::vector<size_t> shell_center(shells_.size());
  for(size_t is=0;is<shells_.size();is++) {
    const coords_t cen=shells_[is].get_center();
    size_t icen;
    for(icen=0;icen<centers.size();icen++)
      if(centers[icen]==cen)
        break;
    if(icen==centers.size())
      centers.push_back(cen);
    shell_center[is]=icen;
  }

  // Fill the tables
  cint_atm_.assign(ATM_SLOTS*centers.size(), 0);
  cint_bas_.assign(BAS_SLOTS*shells_.size(), 0);
  cint_env_.assign(PTR_ENV_START, 0.0);

  for(size_t icen=0;icen<centers.size();icen++) {
    cint_atm_[icen*ATM_SLOTS+CHARGE_OF]=0;
    cint_atm_[icen*ATM_SLOTS+NUC_MOD_OF]=POINT_NUC;
    cint_atm_[icen*ATM_SLOTS+PTR_COORD]=(int) cint_env_.size();
    cint_env_.push_back(centers[icen].x);
    cint_env_.push_back(centers[icen].y);
    cint_env_.push_back(centers[icen].z);
  }

  shell_Nbf_.resize(shells_.size());
  shell_first_.resize(shells_.size());
  fnorm_.resize(shells_.size());
  max_Nbf_=0;

  size_t ibf=0;
  for(size_t is=0;is<shells_.size();is++) {
    const GaussianShell & sh=shells_[is];
    const int l=sh.get_am();
    const size_t nprim=sh.get_Ncontr();
    const size_t nctr=sh.get_Nctr();

    cint_bas_[is*BAS_SLOTS+ATOM_OF]=(int) shell_center[is];
    cint_bas_[is*BAS_SLOTS+ANG_OF]=l;
    cint_bas_[is*BAS_SLOTS+NPRIM_OF]=(int) nprim;
    cint_bas_[is*BAS_SLOTS+NCTR_OF]=(int) nctr;
    cint_bas_[is*BAS_SLOTS+KAPPA_OF]=0;

    // Shared primitive exponents
    cint_bas_[is*BAS_SLOTS+PTR_EXP]=(int) cint_env_.size();
    {
      const std::vector<contr_t> c0=sh.get_contr_normalized(0);
      for(size_t ip=0;ip<nprim;ip++)
        cint_env_.push_back(c0[ip].z);
    }

    // The coefficient columns, one contraction after the other, each
    // over normalized primitives (libcint contracts normalized primitives)
    cint_bas_[is*BAS_SLOTS+PTR_COEFF]=(int) cint_env_.size();
    for(size_t ic=0;ic<nctr;ic++) {
      const std::vector<contr_t> cc=sh.get_contr_normalized(ic);
      for(size_t ip=0;ip<nprim;ip++)
        cint_env_.push_back(cc[ip].c*CINTgto_norm(l,cc[ip].z));
    }

    // Number of functions: nctr angular blocks. Spherical mode
    // evaluates every shell in the spherical basis (s and p coincide
    // with the cartesian ones).
    shell_Nbf_[is]= nctr * (lm_ ? (size_t) (2*l+1) : (size_t) ((l+1)*(l+2)/2));
    shell_first_[is]=ibf;
    ibf+=shell_Nbf_[is];
    max_Nbf_=std::max(max_Nbf_,shell_Nbf_[is]);
  }

  // Measure the normalization of the basis functions against ERKALE's:
  // the two conventions describe the same functions and so differ by a
  // diagonal scaling. This is what carries the Coulomb normalization of
  // an auxiliary basis, which rescales the functions after the shells
  // were built. ERKALE's norms are evaluated in closed form here rather
  // than with the overlap integrals of BasisSet, which are themselves
  // evaluated through an environment.
  CINTIntegralFunction * ovlp = lm_ ? int1e_ovlp_sph : int1e_ovlp_cart;
  unit_norm_=true;
  std::vector<double> buf;
  for(size_t is=0;is<shells_.size();is++) {
    const size_t Nbf=shell_Nbf_[is];
    fnorm_[is].assign(Nbf,1.0);

    int shls[2]={(int) is, (int) is};
    buf.resize(Nbf*Nbf);
    if(!ovlp(buf.data(),NULL,shls,cint_atm_.data(),(int) centers.size(),cint_bas_.data(),(int) shells_.size(),cint_env_.data(),NULL,NULL))
      throw std::runtime_error("CintEnv: failed to evaluate the self-overlap of a shell.\n");

    const arma::vec Serk=shells_[is].function_norms();
    if(Serk.n_elem != Nbf)
      throw std::logic_error("CintEnv: the shell has an unexpected number of functions.\n");

    for(size_t i=0;i<Nbf;i++) {
      const double scint=buf[i*Nbf+i];
      if(scint<=0.0)
        throw std::runtime_error("CintEnv: a basis function has a non-positive norm.\n");
      fnorm_[is][i]=sqrt(Serk(i)/scint);
      if(std::abs(fnorm_[is][i]-1.0)>1e-12)
        unit_norm_=false;
    }
  }

  // Build the integral optimizers. They cache the primitive pair data,
  // which is the whole point of holding on to the environment.
  if(!build_opts)
    return;

  opts_=std::make_shared<OptSet>();
  opts_->opts.assign(CINT_NKERNEL, nullptr);
  int * atmp=cint_atm_.data();
  int * basp=cint_bas_.data();
  double * envp=cint_env_.data();
  const int natm=(int) centers.size();
  const int nbas=(int) shells_.size();

  CINTOptimizerFunction * const optfun[CINT_NKERNEL]={
    int2e_optimizer, int2e_ip1_optimizer, int2e_ip2_optimizer,
    int3c2e_optimizer, int3c2e_ip1_optimizer, int3c2e_ip2_optimizer,
    int2c2e_optimizer, int2c2e_ip1_optimizer};
  for(int ik=0;ik<CINT_NKERNEL;ik++) {
    CINTOpt * o=nullptr;
    optfun[ik](&o, atmp, natm, basp, nbas, envp);
    opts_->opts[ik]=(void *) o;
  }
}

bool CintEnv::is_filled() const {
  return cint_bas_.size()!=0;
}

size_t CintEnv::Nsh() const {
  return shell_Nbf_.size();
}

const GaussianShell & CintEnv::shell(size_t ish) const {
  return shells_[ish];
}

size_t CintEnv::Nsh_orb() const {
  return Nsh_orb_;
}

size_t CintEnv::Nbf(size_t ish) const {
  return shell_Nbf_[ish];
}

size_t CintEnv::first_ind(size_t ish) const {
  return shell_first_[ish];
}

size_t CintEnv::max_Nbf() const {
  return max_Nbf_;
}

bool CintEnv::lm_in_use() const {
  return lm_;
}

const std::vector<double> & CintEnv::fnorm(size_t ish) const {
  return fnorm_[ish];
}

bool CintEnv::has_unit_norm() const {
  return unit_norm_;
}

int * CintEnv::atm() const {
  return const_cast<int *>(cint_atm_.data());
}

int CintEnv::natm() const {
  return (int) (cint_atm_.size()/ATM_SLOTS);
}

int * CintEnv::bas() const {
  return const_cast<int *>(cint_bas_.data());
}

int CintEnv::nbas() const {
  return (int) (cint_bas_.size()/BAS_SLOTS);
}

const std::vector<double> & CintEnv::env() const {
  return cint_env_;
}

void * CintEnv::opt(cint_kernel_t kernel) const {
  // An environment built without the optimizers passes NULL, which
  // libcint accepts (at the cost of recomputing the pair data)
  return opts_ ? opts_->opts[kernel] : nullptr;
}
