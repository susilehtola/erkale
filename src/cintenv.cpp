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

CintEnv::CintEnv() : Nsh_orb(0), max_Nbf(0), lm(true) {
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

CintEnv::OptSet::~OptSet() {
  for(size_t i=0;i<opts.size();i++)
    if(opts[i]) {
      CINTOpt * o=(CINTOpt *) opts[i];
      CINTdel_optimizer(&o);
      opts[i]=nullptr;
    }
}

void CintEnv::build(const std::vector<GaussianShell> & sh, size_t Nsh_orbital, bool build_opts) {
  Nsh_orb=Nsh_orbital;
  shells=sh;

  if(!shells.size())
    throw std::logic_error("CintEnv: no shells to build an environment for!\n");

  // ERKALE's optlm keeps the s and p shells cartesian even when the rest
  // of the basis is spherical; since libcint's spherical s and p shells
  // coincide with the cartesian ones, such a basis is evaluated with the
  // spherical kernels. Only shells with l >= 2 decide the mode, and they
  // all have to agree.
  bool have_lm=false, have_cart=false;
  for(size_t is=0;is<shells.size();is++) {
    if(shells[is].get_am()<2)
      continue;
    if(shells[is].lm_in_use())
      have_lm=true;
    else
      have_cart=true;
  }
  if(have_lm && have_cart)
    throw std::runtime_error("CintEnv: the basis mixes spherical and cartesian shells of l >= 2, which libcint cannot evaluate in a single call.\n");
  // A basis of only s and p shells is the same either way; use the
  // spherical kernels, as they are what the rest of ERKALE defaults to.
  lm=!have_cart;

  // Collect the distinct centers
  std::vector<coords_t> centers;
  std::vector<size_t> shell_center(shells.size());
  for(size_t is=0;is<shells.size();is++) {
    const coords_t cen=shells[is].get_center();
    size_t icen;
    for(icen=0;icen<centers.size();icen++)
      if(centers[icen]==cen)
        break;
    if(icen==centers.size())
      centers.push_back(cen);
    shell_center[is]=icen;
  }

  // Fill the tables
  cint_atm.assign(ATM_SLOTS*centers.size(), 0);
  cint_bas.assign(BAS_SLOTS*shells.size(), 0);
  cint_env.assign(PTR_ENV_START, 0.0);

  for(size_t icen=0;icen<centers.size();icen++) {
    cint_atm[icen*ATM_SLOTS+CHARGE_OF]=0;
    cint_atm[icen*ATM_SLOTS+NUC_MOD_OF]=POINT_NUC;
    cint_atm[icen*ATM_SLOTS+PTR_COORD]=(int) cint_env.size();
    cint_env.push_back(centers[icen].x);
    cint_env.push_back(centers[icen].y);
    cint_env.push_back(centers[icen].z);
  }

  shell_Nbf.resize(shells.size());
  shell_first.resize(shells.size());
  fnorm.resize(shells.size());
  max_Nbf=0;

  size_t ibf=0;
  for(size_t is=0;is<shells.size();is++) {
    const GaussianShell & sh=shells[is];
    const int l=sh.get_am();
    const std::vector<contr_t> c=sh.get_contr_normalized();

    cint_bas[is*BAS_SLOTS+ATOM_OF]=(int) shell_center[is];
    cint_bas[is*BAS_SLOTS+ANG_OF]=l;
    cint_bas[is*BAS_SLOTS+NPRIM_OF]=(int) c.size();
    cint_bas[is*BAS_SLOTS+NCTR_OF]=1;
    cint_bas[is*BAS_SLOTS+KAPPA_OF]=0;

    cint_bas[is*BAS_SLOTS+PTR_EXP]=(int) cint_env.size();
    for(size_t ip=0;ip<c.size();ip++)
      cint_env.push_back(c[ip].z);

    // libcint contracts normalized primitives
    cint_bas[is*BAS_SLOTS+PTR_COEFF]=(int) cint_env.size();
    for(size_t ip=0;ip<c.size();ip++)
      cint_env.push_back(c[ip].c*CINTgto_norm(l,c[ip].z));

    // Number of functions: spherical mode evaluates every shell in the
    // spherical basis (s and p coincide with the cartesian ones)
    shell_Nbf[is]= lm ? (size_t) (2*l+1) : (size_t) ((l+1)*(l+2)/2);
    shell_first[is]=ibf;
    ibf+=shell_Nbf[is];
    max_Nbf=std::max(max_Nbf,shell_Nbf[is]);
  }

  // Measure the normalization of the basis functions against ERKALE's
  // own overlap integrals: the two bases differ by a diagonal scaling,
  // since they are the same functions with different normalization
  // conventions. This is also what carries the Coulomb normalization of
  // an auxiliary basis, which rescales the functions after the shells
  // were constructed.
  CINTIntegralFunction * ovlp = lm ? int1e_ovlp_sph : int1e_ovlp_cart;
  unit_norm=true;
  std::vector<double> buf;
  for(size_t is=0;is<shells.size();is++) {
    const size_t Nbf=shell_Nbf[is];
    fnorm[is].assign(Nbf,1.0);

    int shls[2]={(int) is, (int) is};
    buf.resize(Nbf*Nbf);
    if(!ovlp(buf.data(),NULL,shls,cint_atm.data(),(int) centers.size(),cint_bas.data(),(int) shells.size(),cint_env.data(),NULL,NULL))
      throw std::runtime_error("CintEnv: failed to evaluate the self-overlap of a shell.\n");

    const arma::mat S=shells[is].overlap(shells[is]);
    if(S.n_rows != Nbf)
      throw std::logic_error("CintEnv: the shell has an unexpected number of functions.\n");

    for(size_t i=0;i<Nbf;i++) {
      const double scint=buf[i*Nbf+i];
      if(scint<=0.0)
        throw std::runtime_error("CintEnv: a basis function has a non-positive norm.\n");
      fnorm[is][i]=sqrt(S(i,i)/scint);
      if(std::abs(fnorm[is][i]-1.0)>1e-12)
        unit_norm=false;
    }
  }

  // Build the integral optimizers. They cache the primitive pair data,
  // which is the whole point of holding on to the environment.
  if(!build_opts)
    return;

  opts=std::make_shared<OptSet>();
  opts->opts.assign(CINT_NKERNEL, nullptr);
  int * atmp=cint_atm.data();
  int * basp=cint_bas.data();
  double * envp=cint_env.data();
  const int natm=(int) centers.size();
  const int nbas=(int) shells.size();

  CINTOptimizerFunction * const optfun[CINT_NKERNEL]={
    int2e_optimizer, int2e_ip1_optimizer, int2e_ip2_optimizer,
    int3c2e_optimizer, int3c2e_ip1_optimizer, int3c2e_ip2_optimizer,
    int2c2e_optimizer, int2c2e_ip1_optimizer};
  for(int ik=0;ik<CINT_NKERNEL;ik++) {
    CINTOpt * o=nullptr;
    optfun[ik](&o, atmp, natm, basp, nbas, envp);
    opts->opts[ik]=(void *) o;
  }
}

bool CintEnv::is_filled() const {
  return cint_bas.size()!=0;
}

size_t CintEnv::get_Nsh() const {
  return shell_Nbf.size();
}

const GaussianShell & CintEnv::get_shell(size_t ish) const {
  return shells[ish];
}

size_t CintEnv::get_Nsh_orb() const {
  return Nsh_orb;
}

size_t CintEnv::get_Nbf(size_t ish) const {
  return shell_Nbf[ish];
}

size_t CintEnv::get_first_ind(size_t ish) const {
  return shell_first[ish];
}

size_t CintEnv::get_max_Nbf() const {
  return max_Nbf;
}

bool CintEnv::lm_in_use() const {
  return lm;
}

const std::vector<double> & CintEnv::get_fnorm(size_t ish) const {
  return fnorm[ish];
}

bool CintEnv::has_unit_norm() const {
  return unit_norm;
}

int * CintEnv::get_atm() const {
  return const_cast<int *>(cint_atm.data());
}

int CintEnv::get_natm() const {
  return (int) (cint_atm.size()/ATM_SLOTS);
}

int * CintEnv::get_bas() const {
  return const_cast<int *>(cint_bas.data());
}

int CintEnv::get_nbas() const {
  return (int) (cint_bas.size()/BAS_SLOTS);
}

const std::vector<double> & CintEnv::get_env() const {
  return cint_env;
}

void * CintEnv::get_opt(cint_kernel_t kernel) const {
  // An environment built without the optimizers passes NULL, which
  // libcint accepts (at the cost of recomputing the pair data)
  return opts ? opts->opts[kernel] : nullptr;
}
