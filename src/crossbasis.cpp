/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "crossbasis.h"
#include "cintenv.h"
#include "eriworker.h"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

arma::mat cross_basis_J(const BasisSet & target, const BasisSet & source, const arma::mat & Psource, double thr, double omega, double alpha, double beta) {
  if(Psource.n_rows != source.Nbf() || Psource.n_cols != source.Nbf())
    throw std::logic_error("Density matrix does not correspond to the source basis set!\n");

  // Shell pairs of the two bases, sorted by their Schwarz estimate
  const ScreeningData s_scr(source.compute_screening(thr, omega, alpha, beta));
  const ScreeningData t_scr(target.compute_screening(thr, omega, alpha, beta));
  const std::vector<eripair_t> & spairs(s_scr.shpairs);
  const std::vector<eripair_t> & tpairs(t_scr.shpairs);

  arma::mat J(target.Nbf(), target.Nbf(), arma::fill::zeros);

  // libcint environment: the target shells, followed by the source shells
  CintEnv cenv(target, source);
  const size_t Nsh_tgt(cenv.Nsh_orb());

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri_owner(make_eri_worker(cenv, omega, alpha, beta));
    ERIWorker * eri(eri_owner.get());

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t tp=0;tp<tpairs.size();tp++) {
      const size_t it=tpairs[tp].is, jt=tpairs[tp].js;
      const size_t it0=tpairs[tp].i0, jt0=tpairs[tp].j0;
      const size_t Nti=tpairs[tp].Ni, Ntj=tpairs[tp].Nj;

      arma::mat Jtij(Nti,Ntj,arma::fill::zeros);
      for(size_t sp=0;sp<spairs.size();sp++) {
        const size_t ks=spairs[sp].is, ls=spairs[sp].js;
        const size_t ks0=spairs[sp].i0, ls0=spairs[sp].j0;
        const size_t Nsk=spairs[sp].Ni, Nsl=spairs[sp].Nj;

        // Schwarz screening; the pairs are sorted, so all the rest are
        // small as well
        if(t_scr.Q(it,jt)*s_scr.Q(ks,ls)<thr)
          break;

        eri->compute(it,jt,Nsh_tgt+ks,Nsh_tgt+ls);
        const arma::mat Pkl(Psource.submat(ks0,ls0,ks0+Nsk-1,ls0+Nsl-1));
        // The source pair stands for both orderings
        const double fac = (ks!=ls) ? 2.0 : 1.0;
        // (ij|kl) P_kl
        const arma::mat tei(const_cast<double *>(eri->getp()->data()),Nsk*Nsl,Nti*Ntj,false,true);
        arma::mat incr(fac*arma::vectorise(Pkl.t()).t()*tei);
        incr.reshape(Ntj,Nti);
        Jtij += incr.t();
      }

      J.submat(it0,jt0,it0+Nti-1,jt0+Ntj-1) = Jtij;
      if(it!=jt)
        J.submat(jt0,it0,jt0+Ntj-1,it0+Nti-1) = Jtij.t();
    }
  }

  return J;
}

arma::mat cross_basis_K(const BasisSet & target, const BasisSet & source, const arma::mat & Psource, double thr, double omega, double alpha, double beta) {
  if(Psource.n_rows != source.Nbf() || Psource.n_cols != source.Nbf())
    throw std::logic_error("Density matrix does not correspond to the source basis set!\n");

  const size_t Nt(target.Nshells()), Ns(source.Nshells());
  arma::mat K(target.Nbf(), target.Nbf(), arma::fill::zeros);

  // libcint environment: the target shells, followed by the source shells
  CintEnv cenv(target, source);
  const size_t Nsh_tgt(cenv.Nsh_orb());

  // Largest density element of each pair of source shells
  arma::mat Pmax(Ns, Ns);
  for(size_t l=0;l<Ns;l++)
    for(size_t s=0;s<Ns;s++)
      Pmax(l,s)=arma::abs(Psource.submat(source.first_ind(l), source.first_ind(s), source.last_ind(l), source.last_ind(s))).max();

  // Schwarz estimates of the mixed pairs, Q(u,l) = max sqrt((ul|ul))
  arma::mat Q(Nt, Ns);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri_owner(make_eri_worker(cenv, omega, alpha, beta));
    ERIWorker * eri(eri_owner.get());
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(dynamic)
#endif
    for(size_t u=0;u<Nt;u++)
      for(size_t l=0;l<Ns;l++) {
        eri->compute(u, Nsh_tgt+l, u, Nsh_tgt+l);
        const std::vector<double> & ints(*eri->getp());
        const size_t Nu(target.Nbf(u)), Nl(source.Nbf(l));
        double m=0.0;
        // Diagonal elements (ul|ul) at index ((iu*Nl+il)*Nu+iu)*Nl+il
        for(size_t iu=0;iu<Nu;iu++)
          for(size_t il=0;il<Nl;il++)
            m=std::max(m, std::abs(ints[((iu*Nl+il)*Nu+iu)*Nl+il]));
        Q(u,l)=std::sqrt(m);
      }
  }

  // K_uv = sum_ls (ul|vs) P_ls over the target shell pairs u <= v; K is
  // symmetric for a symmetric density
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri_owner(make_eri_worker(cenv, omega, alpha, beta));
    ERIWorker * eri(eri_owner.get());
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t uv=0;uv<Nt*(Nt+1)/2;uv++) {
      // Unpack u <= v
      size_t v=0;
      while((v+1)*(v+2)/2<=uv)
        v++;
      const size_t u=uv-v*(v+1)/2;
      const size_t u0(target.first_ind(u)), v0(target.first_ind(v));
      const size_t Nu(target.Nbf(u)), Nv(target.Nbf(v));

      arma::mat Kuv(Nu, Nv, arma::fill::zeros);
      for(size_t l=0;l<Ns;l++) {
        if(Q(u,l)*Q.row(v).max()*Pmax.row(l).max()<thr)
          continue;
        const size_t l0(source.first_ind(l)), Nl(source.Nbf(l));
        for(size_t s=0;s<Ns;s++) {
          if(Q(u,l)*Q(v,s)*Pmax(l,s)<thr)
            continue;
          const size_t s0(source.first_ind(s)), Nsf(source.Nbf(s));
          eri->compute(u, Nsh_tgt+l, v, Nsh_tgt+s);
          const std::vector<double> & ints(*eri->getp());
          // (ul|vs) at index ((iu*Nl+il)*Nv+iv)*Ns+is
          for(size_t iu=0;iu<Nu;iu++)
            for(size_t il=0;il<Nl;il++)
              for(size_t iv=0;iv<Nv;iv++) {
                const double * row(&ints[((iu*Nl+il)*Nv+iv)*Nsf]);
                double el=0.0;
                for(size_t is=0;is<Nsf;is++)
                  el+=row[is]*Psource(l0+il, s0+is);
                Kuv(iu,iv)+=el;
              }
        }
      }

      K.submat(u0,v0,u0+Nu-1,v0+Nv-1) = Kuv;
      if(u!=v)
        K.submat(v0,u0,v0+Nv-1,u0+Nu-1) = Kuv.t();
    }
  }

  return K;
}

namespace {
  /// Inverse square root of the fitting metric (a|b) of the auxiliary
  /// shells, which follow Nsh_orb orbital shells in the environment
  arma::mat metric_invh(const CintEnv & cenv, const BasisSet & aux, double fitthr, double omega, double alpha, double beta) {
    const size_t Nsh_orb(cenv.Nsh_orb());
    const size_t Nash(aux.Nshells());
    arma::mat M(aux.Nbf(), aux.Nbf(), arma::fill::zeros);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      auto eri_owner(make_eri_worker(cenv, omega, alpha, beta));
      ERIWorker * eri(eri_owner.get());
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for(size_t a=0;a<Nash;a++)
        for(size_t b=0;b<=a;b++) {
          eri->compute_2c(Nsh_orb+a, Nsh_orb+b);
          const size_t Na(aux.Nbf(a)), Nb(aux.Nbf(b)), a0(aux.first_ind(a)), b0(aux.first_ind(b));
          // (a|b) at index ia*Nb+ib
          const arma::mat blk(const_cast<double *>(eri->getp()->data()), Nb, Na, false, true);
          M.submat(a0, b0, a0+Na-1, b0+Nb-1) = blk.t();
          M.submat(b0, a0, b0+Nb-1, a0+Na-1) = blk;
        }
    }
    arma::vec lambda;
    arma::mat V;
    arma::eig_sym(lambda, V, M);
    const arma::uvec keep(arma::find(lambda > fitthr));
    return V.cols(keep)*arma::diagmat(1.0/arma::sqrt(lambda(keep)))*V.cols(keep).t();
  }

  /// libcint environment of the target shells, the source shells and the
  /// auxiliary shells
  CintEnv cross_env(const BasisSet & target, const BasisSet & source, const BasisSet & aux) {
    std::vector<GaussianShell> shells(target.shells());
    const std::vector<GaussianShell> sshells(source.shells()), ashells(aux.shells());
    shells.insert(shells.end(), sshells.begin(), sshells.end());
    const size_t Nsh_orb(shells.size());
    shells.insert(shells.end(), ashells.begin(), ashells.end());
    return CintEnv(shells, Nsh_orb);
  }

  /// Contract the (pair|aux) integrals of all pairs of shells of bas,
  /// shells offset by sh0 in the environment: the fitting vector
  /// d_a = sum_ls (ls|a) P_ls if fit, else J_ls += sum_a (ls|a) gamma_a
  void three_index_pairs(const CintEnv & cenv, const BasisSet & bas, size_t sh0, const BasisSet & aux, bool fit, const arma::mat & P, arma::vec & d, const arma::vec & gamma, arma::mat & J) {
    const size_t Nsh(bas.Nshells()), Nash(aux.Nshells()), Nsh_orb(cenv.Nsh_orb());
    if(fit)
      d.zeros(aux.Nbf());
    else
      J.zeros(bas.Nbf(), bas.Nbf());
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      auto eri_owner(make_eri_worker(cenv, 0.0, 1.0, 0.0));
      ERIWorker * eri(eri_owner.get());
      arma::vec dth(fit ? aux.Nbf() : 0, arma::fill::zeros);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for(size_t l=0;l<Nsh;l++)
        for(size_t s=0;s<=l;s++) {
          const size_t Nl(bas.Nbf(l)), Ns(bas.Nbf(s)), l0(bas.first_ind(l)), s0(bas.first_ind(s));
          // Both orderings of the pair
          const double fac = (l==s) ? 1.0 : 2.0;
          arma::mat Jls(Nl, Ns, arma::fill::zeros);
          for(size_t a=0;a<Nash;a++) {
            eri->compute_3c(sh0+l, sh0+s, Nsh_orb+a);
            const std::vector<double> & ints(*eri->getp());
            const size_t Na(aux.Nbf(a)), a0(aux.first_ind(a));
            // (ls|a) at index (il*Ns+is)*Na+ia
            for(size_t il=0;il<Nl;il++)
              for(size_t is=0;is<Ns;is++) {
                const double * row(&ints[(il*Ns+is)*Na]);
                if(fit) {
                  const double Pls(fac*P(l0+il, s0+is));
                  for(size_t ia=0;ia<Na;ia++)
                    dth(a0+ia)+=row[ia]*Pls;
                } else {
                  double el=0.0;
                  for(size_t ia=0;ia<Na;ia++)
                    el+=row[ia]*gamma(a0+ia);
                  Jls(il,is)+=el;
                }
              }
          }
          if(!fit) {
            J.submat(l0, s0, l0+Nl-1, s0+Ns-1) = Jls;
            J.submat(s0, l0, s0+Ns-1, l0+Nl-1) = Jls.t();
          }
        }
      if(fit) {
#ifdef _OPENMP
#pragma omp critical
#endif
        d+=dth;
      }
    }
  }
}

arma::mat cross_basis_J_df(const BasisSet & target, const BasisSet & source, const BasisSet & aux, const arma::mat & Psource, double fitthr) {
  if(Psource.n_rows != source.Nbf() || Psource.n_cols != source.Nbf())
    throw std::logic_error("Density matrix does not correspond to the source basis set!\n");

  const CintEnv cenv(cross_env(target, source, aux));
  const arma::mat Minvh(metric_invh(cenv, aux, fitthr, 0.0, 1.0, 0.0));

  // Fit the source density, then assemble in the target basis
  arma::vec d, gamma;
  arma::mat J;
  three_index_pairs(cenv, source, target.Nshells(), aux, true, Psource, d, gamma, J);
  gamma=Minvh*(Minvh*d);
  three_index_pairs(cenv, target, 0, aux, false, Psource, d, gamma, J);
  return J;
}

std::vector<arma::mat> cross_basis_K_df(const BasisSet & target, const BasisSet & source, const BasisSet & aux, const std::vector<arma::mat> & C, double fitthr, double omega, double alpha, double beta) {
  for(const arma::mat & Ck : C)
    if(Ck.n_rows != source.Nbf())
      throw std::logic_error("Orbital coefficients do not correspond to the source basis set!\n");

  const CintEnv cenv(cross_env(target, source, aux));
  const arma::mat Minvh(metric_invh(cenv, aux, fitthr, omega, alpha, beta));
  const size_t Nt(target.Nshells()), Ns(source.Nshells()), Nash(aux.Nshells()), Naux(aux.Nbf());
  const size_t Nsh_src(Nt), Nsh_aux(cenv.Nsh_orb());

  // Half-transformed integrals B_k(i*Naux+a, u) = sum_l (u l|a) C_k(l,i),
  // built in parallel over the target shells, which own their columns
  std::vector<arma::mat> B(C.size());
  for(size_t k=0;k<C.size();k++)
    B[k].zeros(C[k].n_cols*Naux, target.Nbf());

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto eri_owner(make_eri_worker(cenv, omega, alpha, beta));
    ERIWorker * eri(eri_owner.get());
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t u=0;u<Nt;u++) {
      const size_t Nu(target.Nbf(u)), u0(target.first_ind(u));
      for(size_t l=0;l<Ns;l++) {
        const size_t Nl(source.Nbf(l)), l0(source.first_ind(l));
        for(size_t a=0;a<Nash;a++) {
          eri->compute_3c(u, Nsh_src+l, Nsh_aux+a);
          const size_t Na(aux.Nbf(a)), a0(aux.first_ind(a));
          // (ul|a) at index (iu*Nl+il)*Na+ia, as (Na x Nl) per iu
          for(size_t iu=0;iu<Nu;iu++) {
            const arma::mat ula(const_cast<double *>(&(*eri->getp())[iu*Nl*Na]), Na, Nl, false, true);
            for(size_t k=0;k<C.size();k++) {
              const arma::mat X(ula*C[k].rows(l0, l0+Nl-1));
              for(size_t i=0;i<C[k].n_cols;i++)
                B[k].col(u0+iu).subvec(i*Naux+a0, i*Naux+a0+Na-1) += X.col(i);
            }
          }
        }
      }
    }
  }

  // Fit and contract: K = sum_i (Minvh B_i)^T (Minvh B_i)
  std::vector<arma::mat> K(C.size());
  for(size_t k=0;k<C.size();k++) {
    for(size_t i=0;i<C[k].n_cols;i++)
      B[k].rows(i*Naux, (i+1)*Naux-1) = Minvh*B[k].rows(i*Naux, (i+1)*Naux-1);
    K[k]=B[k].t()*B[k];
  }
  return K;
}
