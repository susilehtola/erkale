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

#include "eriworker.h"
#include "integrals.h"
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

namespace {
  /// The libcint optimizer constructor for the given operator
  CINTOptimizerFunction * optimizer_function(cint_kernel_t kernel) {
    switch(kernel) {
    case CINT_ERI:
      return int2e_optimizer;
    case CINT_ERI_IP1:
      return int2e_ip1_optimizer;
    case CINT_ERI_IP2:
      return int2e_ip2_optimizer;
    case CINT_3C2E:
      return int3c2e_optimizer;
    case CINT_3C2E_IP1:
      return int3c2e_ip1_optimizer;
    case CINT_3C2E_IP2:
      return int3c2e_ip2_optimizer;
    case CINT_2C2E:
      return int2c2e_optimizer;
    case CINT_2C2E_IP1:
      return int2c2e_ip1_optimizer;
    default:
      ERROR_INFO();
      throw std::logic_error("Unknown integral kernel!\n");
    }
  }

  /// The libcint kernel for the given one-electron operator and basis
  CINTIntegralFunction * kernel_1e_function(cint_1e_kernel_t kernel, bool lm) {
    switch(kernel) {
    case CINT1E_OVLP:
      return lm ? int1e_ovlp_sph : int1e_ovlp_cart;
    case CINT1E_KIN:
      return lm ? int1e_kin_sph : int1e_kin_cart;
    case CINT1E_RINV:
      return lm ? int1e_rinv_sph : int1e_rinv_cart;
    case CINT1E_OVLPIP:
      return lm ? int1e_ovlpip_sph : int1e_ovlpip_cart;
    case CINT1E_IPOVLP:
      return lm ? int1e_ipovlp_sph : int1e_ipovlp_cart;
    case CINT1E_IPKIN:
      return lm ? int1e_ipkin_sph : int1e_ipkin_cart;
    case CINT1E_KINIP:
      return lm ? int1e_kinip_sph : int1e_kinip_cart;
    case CINT1E_IPRINV:
      return lm ? int1e_iprinv_sph : int1e_iprinv_cart;
    case CINT1E_R:
      return lm ? int1e_r_sph : int1e_r_cart;
    case CINT1E_RR:
      return lm ? int1e_rr_sph : int1e_rr_cart;
    case CINT1E_RRR:
      return lm ? int1e_rrr_sph : int1e_rrr_cart;
    case CINT1E_RRRR:
      return lm ? int1e_rrrr_sph : int1e_rrrr_cart;
    default:
      ERROR_INFO();
      throw std::logic_error("Unknown one-electron kernel!\n");
    }
  }

  /// The libcint kernel for the given operator and basis
  CINTIntegralFunction * kernel_function(cint_kernel_t kernel, bool lm) {
    switch(kernel) {
    case CINT_ERI:
      return lm ? int2e_sph : int2e_cart;
    case CINT_ERI_IP1:
      return lm ? int2e_ip1_sph : int2e_ip1_cart;
    case CINT_ERI_IP2:
      return lm ? int2e_ip2_sph : int2e_ip2_cart;
    case CINT_3C2E:
      return lm ? int3c2e_sph : int3c2e_cart;
    case CINT_3C2E_IP1:
      return lm ? int3c2e_ip1_sph : int3c2e_ip1_cart;
    case CINT_3C2E_IP2:
      return lm ? int3c2e_ip2_sph : int3c2e_ip2_cart;
    case CINT_2C2E:
      return lm ? int2c2e_sph : int2c2e_cart;
    case CINT_2C2E_IP1:
      return lm ? int2c2e_ip1_sph : int2c2e_ip1_cart;
    default:
      ERROR_INFO();
      throw std::logic_error("Unknown integral kernel!\n");
    }
  }
}

IntegralWorker::IntegralWorker(const CintEnv & cenv, double omega, double alpha, double beta) {
  envp=&cenv;
  // The kernels take the data array as a non-const pointer, and the
  // range separation constant lives in it, so each worker holds a copy
  env=cenv.get_env();
  rs_omega=omega;
  rs_alpha=alpha;
  rs_beta=beta;

  sr_opts.assign(CINT_NKERNEL, nullptr);
}

IntegralWorker::~IntegralWorker() {
  for(size_t i=0;i<sr_opts.size();i++)
    if(sr_opts[i]) {
      CINTOpt * o=(CINTOpt *) sr_opts[i];
      CINTdel_optimizer(&o);
    }
}

void * IntegralWorker::get_opt(cint_kernel_t kernel, double omega) {
  // The environment's optimizers are built for the full-range kernels
  if(omega==0.0)
    return envp->get_opt(kernel);

  // The attenuated kernels need an optimizer of their own, built with
  // the range separation constant in place. The environment data array
  // already carries it when this is called.
  if(!sr_opts[kernel]) {
    CINTOpt * o=nullptr;
    optimizer_function(kernel)(&o, envp->get_atm(), envp->get_natm(), envp->get_bas(), envp->get_nbas(), env.data());
    sr_opts[kernel]=(void *) o;
  }
  return sr_opts[kernel];
}

void IntegralWorker::evaluate(cint_kernel_t kernel, int nsh, const int * shls_in, int ncomp, size_t N,
                              std::vector<double> & out) {
  CINTIntegralFunction * intor=kernel_function(kernel, envp->lm_in_use());

  int shls[4]={0, 0, 0, 0};
  for(int i=0;i<nsh;i++)
    shls[i]=shls_in[i];

  const size_t Ntot=ncomp*N;
  out.resize(Ntot);

  // Evaluate the kernel with the given range separation constant. The
  // constant must be in place already for the scratch size query, since
  // the attenuated kernels need a bigger cache than the full-range ones.
  auto evaluate_omega=[&](double omega, std::vector<double> & buf) {
    env[PTR_RANGE_OMEGA]=omega;
    CINTOpt * opt=(CINTOpt *) get_opt(kernel,omega);
    const size_t csize=intor(NULL,NULL,shls,envp->get_atm(),envp->get_natm(),envp->get_bas(),envp->get_nbas(),env.data(),NULL,NULL);
    if(csize>cache.size())
      cache.resize(csize);
    if(!intor(buf.data(),NULL,shls,envp->get_atm(),envp->get_natm(),envp->get_bas(),envp->get_nbas(),env.data(),opt,cache.data()))
      std::fill(buf.begin(),buf.end(),0.0);
  };

  // Full-range Coulomb component
  if(rs_alpha!=0.0)
    evaluate_omega(0.0,out);
  else
    std::fill(out.begin(),out.end(),0.0);

  // Short-range erfc(omega r12)/r12 component: a negative omega selects
  // the complementary error function attenuation in libcint
  if(rs_beta!=0.0) {
    srbuf.resize(Ntot);
    evaluate_omega(-rs_omega,srbuf);
    for(size_t i=0;i<Ntot;i++)
      out[i]=rs_alpha*out[i]+rs_beta*srbuf[i];
  } else if(rs_alpha!=1.0)
    for(size_t i=0;i<Ntot;i++)
      out[i]*=rs_alpha;
}

void IntegralWorker::normalize(const size_t * shls, int nsh, int ncomp, std::vector<double> & out) const {
  if(envp->has_unit_norm())
    return;

  // Number of functions on each shell, and the stride of each index
  size_t Nbf[4], stride[4];
  size_t N=1;
  for(int i=nsh-1;i>=0;i--) {
    Nbf[i]=envp->get_Nbf(shls[i]);
    stride[i]=N;
    N*=Nbf[i];
  }

  for(size_t idx=0;idx<N;idx++) {
    double norm=1.0;
    for(int i=0;i<nsh;i++)
      norm*=envp->get_fnorm(shls[i])[(idx/stride[i])%Nbf[i]];
    for(int ic=0;ic<ncomp;ic++)
      out[ic*N+idx]*=norm;
  }
}

/// Remap a three-center block from libcint's layout, which runs the
/// first shell fastest, to ERKALE's, which runs the last one fastest.
/// The auxiliary shell has to stay last in the call, so unlike the two-
/// and four-center integrals the tuple cannot simply be reversed.
void IntegralWorker::remap_3c(const std::vector<double> & in, int ncomp,
                              size_t Ni, size_t Nj, size_t Nk, std::vector<double> & out) const {
  const size_t N=Ni*Nj*Nk;
  out.resize(ncomp*N);
  for(int ic=0;ic<ncomp;ic++) {
    const double * ip=in.data()+ic*N;
    double * op=out.data()+ic*N;
    for(size_t k=0;k<Nk;k++)
      for(size_t j=0;j<Nj;j++)
        for(size_t i=0;i<Ni;i++)
          op[(i*Nj+j)*Nk+k]=ip[i+Ni*(j+Nj*k)];
  }
}

ERIWorker::ERIWorker(const CintEnv & cenv, double omega, double alpha, double beta) : IntegralWorker(cenv,omega,alpha,beta) {
}

ERIWorker::~ERIWorker() {
}

void ERIWorker::compute(size_t is, size_t js, size_t ks, size_t ls) {
  // libcint runs the first shell of the tuple fastest, whereas ERKALE
  // stores the last index fastest: evaluating the reversed quartet
  // (lk|ji), which by the permutational symmetry of the integrals is the
  // same integral, gives the result directly in ERKALE's layout.
  const int shls[4]={(int) ls, (int) ks, (int) js, (int) is};
  const size_t N=envp->get_Nbf(is)*envp->get_Nbf(js)*envp->get_Nbf(ks)*envp->get_Nbf(ls);

  evaluate(CINT_ERI,4,shls,1,N,ints);

  const size_t erk[4]={is, js, ks, ls};
  normalize(erk,4,1,ints);
}

void ERIWorker::compute_3c(size_t is, size_t js, size_t ks) {
  const int shls[3]={(int) is, (int) js, (int) ks};
  const size_t Ni=envp->get_Nbf(is), Nj=envp->get_Nbf(js), Nk=envp->get_Nbf(ks);

  evaluate(CINT_3C2E,3,shls,1,Ni*Nj*Nk,tmp);
  remap_3c(tmp,1,Ni,Nj,Nk,ints);

  const size_t erk[3]={is, js, ks};
  normalize(erk,3,1,ints);
}

void ERIWorker::compute_2c(size_t is, size_t js) {
  const int shls[2]={(int) js, (int) is};
  const size_t N=envp->get_Nbf(is)*envp->get_Nbf(js);

  evaluate(CINT_2C2E,2,shls,1,N,ints);

  const size_t erk[2]={is, js};
  normalize(erk,2,1,ints);
}

void ERIWorker::compute_debug(size_t is, size_t js, size_t ks, size_t ls) {
  const GaussianShell & shi=envp->get_shell(is);
  const GaussianShell & shj=envp->get_shell(js);
  const GaussianShell & shk=envp->get_shell(ks);
  const GaussianShell & shl=envp->get_shell(ls);
  const GaussianShell * shs[4]={&shi, &shj, &shk, &shl};

  const std::vector<shellf_t> & ci=shi.get_cart_ref();
  const std::vector<shellf_t> & cj=shj.get_cart_ref();
  const std::vector<shellf_t> & ck=shk.get_cart_ref();
  const std::vector<shellf_t> & cl=shl.get_cart_ref();

  const std::vector<contr_t> coni(shi.get_contr());
  const std::vector<contr_t> conj(shj.get_contr());
  const std::vector<contr_t> conk(shk.get_contr());
  const std::vector<contr_t> conl(shl.get_contr());

  const coords_t Ri(shi.get_center()), Rj(shj.get_center()), Rk(shk.get_center()), Rl(shl.get_center());

  // Cartesian integrals over the Huzinaga routines
  std::vector<double> cart(ci.size()*cj.size()*ck.size()*cl.size(),0.0);
  for(size_t ic=0;ic<ci.size();ic++)
    for(size_t jc=0;jc<cj.size();jc++)
      for(size_t kc=0;kc<ck.size();kc++)
        for(size_t lc=0;lc<cl.size();lc++) {
          double el=0.0;
          for(size_t xi=0;xi<coni.size();xi++)
            for(size_t xj=0;xj<conj.size();xj++)
              for(size_t xk=0;xk<conk.size();xk++)
                for(size_t xl=0;xl<conl.size();xl++)
                  el+=coni[xi].c*conj[xj].c*conk[xk].c*conl[xl].c*
                    ERI_int(ci[ic].l,ci[ic].m,ci[ic].n,Ri.x,Ri.y,Ri.z,coni[xi].z,
                            cj[jc].l,cj[jc].m,cj[jc].n,Rj.x,Rj.y,Rj.z,conj[xj].z,
                            ck[kc].l,ck[kc].m,ck[kc].n,Rk.x,Rk.y,Rk.z,conk[xk].z,
                            cl[lc].l,cl[lc].m,cl[lc].n,Rl.x,Rl.y,Rl.z,conl[xl].z);

          cart[((ic*cj.size()+jc)*ck.size()+kc)*cl.size()+lc]=
            ci[ic].relnorm*cj[jc].relnorm*ck[kc].relnorm*cl[lc].relnorm*el;
        }

  // Transform the indices into the basis the environment evaluates in.
  // A cartesian shell, and any shell in a cartesian basis, transforms
  // with the identity.
  std::vector<double> in(cart), out;
  size_t Ncart[4], Nbf[4];
  for(int q=0;q<4;q++) {
    Ncart[q]=shs[q]->get_Ncart();
    Nbf[q]=envp->get_Nbf(q==0 ? is : (q==1 ? js : (q==2 ? ks : ls)));
  }

  // The transform is done one index at a time; the index being
  // transformed is rotated to the front and back again.
  for(int q=0;q<4;q++) {
    const bool trans=envp->lm_in_use() && shs[q]->lm_in_use();
    const arma::mat T(trans ? shs[q]->get_trans() : arma::mat());

    // Sizes of the indices, with the ones before q already transformed
    size_t Nbefore=1, Nafter=1;
    for(int r=0;r<q;r++)
      Nbefore*=Nbf[r];
    for(int r=q+1;r<4;r++)
      Nafter*=Ncart[r];

    if(!trans) {
      continue;
    }

    out.assign(Nbefore*Nbf[q]*Nafter,0.0);
    for(size_t b=0;b<Nbefore;b++)
      for(size_t n=0;n<Nbf[q];n++)
        for(size_t c=0;c<Ncart[q];c++) {
          const double t=T(n,c);
          if(t==0.0)
            continue;
          for(size_t a=0;a<Nafter;a++)
            out[(b*Nbf[q]+n)*Nafter+a]+=t*in[(b*Ncart[q]+c)*Nafter+a];
        }
    in.swap(out);
  }

  ints=in;
}

std::vector<double> ERIWorker::get() const {
  return ints;
}

std::vector<double> & ERIWorker::rget() {
  return ints;
}

const std::vector<double> * ERIWorker::getp() const {
  return &ints;
}

dERIWorker::dERIWorker(const CintEnv & cenv, double omega, double alpha, double beta) : IntegralWorker(cenv,omega,alpha,beta) {
  N=0;
  nsh=0;
}

dERIWorker::~dERIWorker() {
}

void dERIWorker::compute(size_t is, size_t js, size_t ks, size_t ls) {
  const size_t Ni=envp->get_Nbf(is), Nj=envp->get_Nbf(js), Nk=envp->get_Nbf(ks), Nl=envp->get_Nbf(ls);
  N=Ni*Nj*Nk*Nl;
  nsh=4;
  dR.assign(12*N,0.0);

  // As in ERIWorker::compute, the reversed quartet gives ERKALE's
  // layout. libcint's ip1 differentiates the first shell of the tuple
  // and ip2 the third, so the reversed quartet yields the derivatives
  // with respect to the fourth and the second shell; the third shell
  // derivative comes from the quartet (kl|ji), whose layout needs a
  // remap. The first shell derivative follows from translational
  // invariance. libcint differentiates the electron coordinate, which is
  // minus the derivative with respect to the shell center.
  const int rev[4]={(int) ls, (int) ks, (int) js, (int) is};
  const int kfirst[4]={(int) ks, (int) ls, (int) js, (int) is};

  evaluate(CINT_ERI_IP1,4,rev,3,N,tmp);
  for(size_t i=0;i<3*N;i++)
    dR[9*N+i]=-tmp[i];

  evaluate(CINT_ERI_IP2,4,rev,3,N,tmp);
  for(size_t i=0;i<3*N;i++)
    dR[3*N+i]=-tmp[i];

  evaluate(CINT_ERI_IP1,4,kfirst,3,N,tmp);
  for(int ic=0;ic<3;ic++) {
    const double * ip=tmp.data()+ic*N;
    double * op=dR.data()+(6+ic)*N;
    for(size_t ii=0;ii<Ni;ii++)
      for(size_t jj=0;jj<Nj;jj++)
        for(size_t kk=0;kk<Nk;kk++)
          for(size_t ll=0;ll<Nl;ll++)
            op[((ii*Nj+jj)*Nk+kk)*Nl+ll]=-ip[kk+Nk*(ll+Nl*(jj+Nj*ii))];
  }

  for(size_t i=0;i<3*N;i++)
    dR[i]=-(dR[3*N+i]+dR[6*N+i]+dR[9*N+i]);

  const size_t erk[4]={is, js, ks, ls};
  normalize(erk,4,12,dR);
}

void dERIWorker::compute_3c(size_t is, size_t js, size_t ks) {
  const size_t Ni=envp->get_Nbf(is), Nj=envp->get_Nbf(js), Nk=envp->get_Nbf(ks);
  N=Ni*Nj*Nk;
  nsh=3;
  dR.assign(9*N,0.0);

  // ip1 differentiates the first shell and ip2 the auxiliary one; the
  // second shell follows from translational invariance
  const int shls[3]={(int) is, (int) js, (int) ks};

  evaluate(CINT_3C2E_IP1,3,shls,3,N,tmp);
  remap_3c(tmp,3,Ni,Nj,Nk,scr);
  for(size_t i=0;i<3*N;i++)
    dR[i]=-scr[i];

  evaluate(CINT_3C2E_IP2,3,shls,3,N,tmp);
  remap_3c(tmp,3,Ni,Nj,Nk,scr);
  for(size_t i=0;i<3*N;i++)
    dR[6*N+i]=-scr[i];

  for(size_t i=0;i<3*N;i++)
    dR[3*N+i]=-(dR[i]+dR[6*N+i]);

  const size_t erk[3]={is, js, ks};
  normalize(erk,3,9,dR);
}

void dERIWorker::compute_2c(size_t is, size_t js) {
  const size_t Ni=envp->get_Nbf(is), Nj=envp->get_Nbf(js);
  N=Ni*Nj;
  nsh=2;
  dR.assign(6*N,0.0);

  // The reversed pair gives ERKALE's layout, and ip1 then differentiates
  // the second shell; the first one follows from translational
  // invariance
  const int rev[2]={(int) js, (int) is};

  evaluate(CINT_2C2E_IP1,2,rev,3,N,tmp);
  for(size_t i=0;i<3*N;i++) {
    dR[3*N+i]=-tmp[i];
    dR[i]=tmp[i];
  }

  const size_t erk[2]={is, js};
  normalize(erk,2,6,dR);
}

std::vector<double> dERIWorker::get(int idx) {
  getp(idx);
  return ints;
}

const std::vector<double> * dERIWorker::getp(int idx) {
  if(idx<0 || idx>=3*nsh) {
    ERROR_INFO();
    throw std::runtime_error("Invalid derivative index requested!\n");
  }

  ints.assign(dR.begin()+idx*N, dR.begin()+(idx+1)*N);
  return &ints;
}

ERIWorker_srlr::ERIWorker_srlr(const CintEnv & cenv, double omega, double alpha, double beta) : ERIWorker(cenv,omega,alpha,beta) {
}

ERIWorker_srlr::~ERIWorker_srlr() {
}

dERIWorker_srlr::dERIWorker_srlr(const CintEnv & cenv, double omega, double alpha, double beta) : dERIWorker(cenv,omega,alpha,beta) {
}

dERIWorker_srlr::~dERIWorker_srlr() {
}

Int1eWorker::Int1eWorker(const CintEnv & cenv) : IntegralWorker(cenv) {
}

Int1eWorker::~Int1eWorker() {
}

void Int1eWorker::compute(cint_1e_kernel_t kernel, size_t is, size_t js,
                          const double * rinv_orig, const double * common_orig) {
  CINTIntegralFunction * intor=kernel_1e_function(kernel, envp->lm_in_use());
  const int ncomp=cint_1e_ncomp(kernel);

  // The operator origins live in the data array, of which the worker
  // holds a private copy
  if(rinv_orig)
    for(int i=0;i<3;i++)
      env[PTR_RINV_ORIG+i]=rinv_orig[i];
  if(common_orig)
    for(int i=0;i<3;i++)
      env[PTR_COMMON_ORIG+i]=common_orig[i];

  const size_t Ni=envp->get_Nbf(is), Nj=envp->get_Nbf(js);
  const size_t N=Ni*Nj;

  // The one-electron integrals are cheap, so they are evaluated without
  // an optimizer
  int shls[2]={(int) is, (int) js};
  tmp.resize(ncomp*N);
  const size_t csize=intor(NULL,NULL,shls,envp->get_atm(),envp->get_natm(),envp->get_bas(),envp->get_nbas(),env.data(),NULL,NULL);
  if(csize>cache.size())
    cache.resize(csize);
  if(!intor(tmp.data(),NULL,shls,envp->get_atm(),envp->get_natm(),envp->get_bas(),envp->get_nbas(),env.data(),NULL,cache.data()))
    std::fill(tmp.begin(),tmp.end(),0.0);

  // libcint runs the first shell fastest; ERKALE runs the last index
  // fastest
  ints.resize(ncomp*N);
  for(int ic=0;ic<ncomp;ic++) {
    const double * ip=tmp.data()+ic*N;
    double * op=ints.data()+ic*N;
    for(size_t i=0;i<Ni;i++)
      for(size_t j=0;j<Nj;j++)
        op[i*Nj+j]=ip[i+Ni*j];
  }

  const size_t erk[2]={is, js};
  normalize(erk,2,ncomp,ints);
}

const std::vector<double> * Int1eWorker::getp() const {
  return &ints;
}

arma::mat Int1eWorker::get_mat(int ic, size_t is, size_t js) const {
  const size_t Ni=envp->get_Nbf(is), Nj=envp->get_Nbf(js);
  arma::mat M(Ni,Nj);
  const double * ip=ints.data()+ic*Ni*Nj;
  for(size_t i=0;i<Ni;i++)
    for(size_t j=0;j<Nj;j++)
      M(i,j)=ip[i*Nj+j];
  return M;
}
