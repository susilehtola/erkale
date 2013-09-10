/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2013
 * Copyright (c) 2010-2013, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "gaunt.h"
#include "linalg.h"
#include "mathf.h"
#include "integrals.h"
#include "basis.h"
#include "slaterfit/form_exponents.h"

/// A. Kumar and P. C. Mishra, Pramana 29 (1987), pp. 385-390.
double Ul(int l, int n12, int n34, double z12, double z34) {
  // Prefactor
  double prefac=fact(n34+l)/pow(z34,n34+l+1);

  // First term
  double term1=fact(n12-l-1)/pow(z12,n12-l);

  // Second term
  double term2=0.0;
  for(int lp=1;lp<=n34+l+1;lp++)
    term2-=pow(z34,n34+l-lp+1)/fact(n34+l-lp+1) * fact(n12+n34-lp)/pow(z12+z34,n12+n34-lp+1);

  // Third term
  double term3=0.0;
  for(int lp=1;lp<=n34-l;lp++)
    term3+=pow(z34,n34+l-lp+1)*fact(n12+n34-lp) / (fact(n34-l-lp)*pow(z12+z34,n12+n34-lp+1));
  term3*=fact(n34-l-1)/fact(n34+l);

  return prefac*(term1+term2+term3);
}

/// Gaunt factor. E. U. Condon and G. H. Shortley, The theory of atomic spectra, Cambridge University Press 1963, page 176.
double gaunt(int l, int m, int lp, int mp, int L) {
  // Cannot couple if
  if((l+lp+L)%2==1)
    return 0.0;

  int g=(l+lp+L)/2;
  if(g<l || g<lp || g<L)
    return 0.0;

  //  printf("1fact with arguments %i, %i, %i, %i, %i, %i\n",2*g-2*lp,g,g-l,g-lp,g-L,2*g+1);
  double fac1=pow(-1.0,g-l-mp)*fact(2*g-2*lp)*fact(g)/(fact(g-l)*fact(g-lp)*fact(g-L)*fact(2*g+1));
  //  printf("fac1 computed\n"); fflush(stdout);

  //  printf("2fact with arguments %i, %i, %i, %i, %i, %i\n",L-m-mp,l+m,lp+mp,lp-mp,L+m+mp,l-m);
  double fac2=sqrt((2*l+1)*(2*lp+1)*(2*L+1)*fact(L-m-mp)*fact(l+m)*fact(lp+mp)*fact(lp-mp)/(2*(fact(L+m+mp)*fact(l-m))));
  //  printf("fac2 computed\n"); fflush(stdout);

  // Third term,
  double fac3=0.0;

  // Minimum of t
  int tmin=0;
  int tmin1=-(L+m+mp);
  int tmin2=-(l-lp+m+mp);
  if(tmin<tmin1)
    tmin=tmin1;
  if(tmin<tmin2)
    tmin=tmin2;

  int tmax=lp-mp;
  int tmax2=l+lp-m-mp;
  int tmax3=L-m-mp;
  if(tmax>tmax2)
    tmax=tmax2;
  if(tmax>tmax3)
    tmax=tmax3;

  for(int t=tmin;t<=tmax;t++) {
    //    printf("3fact with arguments %i, %i, %i, %i, %i, %i\n",L+m+mp+t,l+lp-m-mp-t,L-m-mp-t,l-lp+m+mp+t,lp-mp-t,t);
    //    fflush(stdout);
    fac3+=pow(-1.0,t)*fact(L+m+mp+t)*fact(l+lp-m-mp-t)/(fact(L-m-mp-t)*fact(l-lp+m+mp+t)*fact(lp-mp-t)*fact(t));
  }
  return fac1*fac2*fac3;
}

// Angular factor, Condon-Shortley page 175
double ck(int k, int l, int m, int lp, int mp) {
  if(k<abs(m-mp))
    return 0.0;

  return sqrt(2.0/(2*k+1))*gaunt(k,m-mp,lp,mp,l);
}

// Normalization coefficient
double normalization(int n, double z) {
  return sqrt(pow(2.0*z,2*n+1)/fact(2*n));
}

// Primitive overlap integral
double overlap_primitive(int nab, double zab) {
  return fact(nab)/pow(zab,nab+1);
}

// Repulsion integral (ab|cd) = \int \phi_a(r_1) \phi_b(r_1) r^{-1}_{12} \phi_c(r_2) \phi_d(r_2) d^3 r_1 d^3 r_2
double ERI_unnormalized(int na, int nb, int nc, int nd, double za, double zb, double zc, double zd, int la, int ma, int lb, int mb, int lc, int mc, int ld, int md) {
  // Remember complex conjugation in ERI
  if(mb-ma!=md-mc) {
    //    printf("m different - ma=%i, mb=%i => %i, mc=%i, md=%i => %i\n",ma,mb,mb-ma,mc,md,md-mc);
    return 0.0;
  }

  // What is maximum angular momentum we can couple to?
  //  int lmax=la+lc;
  /*
  if(lb+ld<lmax)
    lmax=lb+ld;
  */

  // The functions la and lb can couple to |la-lb| <= l <= la+lb
  // but we also must require that         |lc-ld| <= l <= lc+ld
  // and |m|<=l
  int lmin1=std::abs(la-lb);
  int lmin2=std::abs(lc-ld);
  int lmin3=abs(mb-ma);
  int lmin=std::max(lmin1,std::max(lmin2,lmin3));

  int lmax=std::min(la+lb,lc+ld);

  double eri=0.0;

  for(int l=lmin;l<=lmax;l++)
    eri+=Ul(l,na+nb,nc+nd,za+zb,zc+zd)*ck(l,la,ma,lb,mb)*ck(l,lc,mc,ld,md);

  return eri;
}

double ERI(int na, int nb, int nc, int nd, double za, double zb, double zc, double zd, int la, int ma, int lb, int mb, int lc, int mc, int ld, int md) {
  double eri=ERI_unnormalized(na,nb,nc,nd,za,zb,zc,zd,la,ma,lb,mb,lc,mc,ld,md);
  // Plug in normalization
  eri*=normalization(na,za)*normalization(nb,zb)*normalization(nc,zc)*normalization(nd,zd);
  return eri;
}

double coulomb_overlap(const bf_t & a, const bf_t & b) {
  bf_t dummy;
  dummy.zeta=0.0;
  dummy.n=1;
  dummy.l=0;
  dummy.m=0;

  // Overlap wrt overlap-normalized functions
  //  return ERI_unnormalized(a,dummy,b,dummy)*normalization(a.n,a.zeta)*normalization(b.n,b.zeta);

  // Overlap wrt ERI-normalized functions
  return ERI_unnormalized(a,dummy,b,dummy)/sqrt(ERI_unnormalized(a,dummy,a,dummy)*ERI_unnormalized(b,dummy,b,dummy));
}

double three_overlap(int na, int nc, int nd, int la, int ma, int lc, int mc, int ld, int md, double za, double zc, double zd) {
  if(ma!=md-mc) {
    return 0.0;
  }
  // The functions lc and ld can couple to |lc-ld| <= l <= lc+ld
  // which must thus be the same as la
  if(la<abs(lc-ld) || la>lc+ld)
    return 0.0;

  //return overlap_primitive(na+nc+nd,za+zc+zd)*gaunt_coefficient(la,ma,lc,mc,ld,md)*normalization(na,za)*normalization(nc,zc)*normalization(nd,zd);
  return overlap_primitive(na+nc+nd,za+zc+zd)*ck(la,lc,mc,ld,md)*normalization(na,za)*normalization(nc,zc)*normalization(nd,zd);
}


// Do Gaussian expansion of ERI
double gaussian_ERI(int la, int ma, int lb, int mb, int lc, int mc, int ld, int md, double za, double zb, double zc, double zd, int nfit) {
  ERIWorker eri(std::max(la,lb),nfit);
  std::vector<double> eris;

  std::vector<contr_t> ca=slater_fit(za,la,nfit,false);
  std::vector<contr_t> cb=slater_fit(zb,lb,nfit,false);
  std::vector<contr_t> cc=slater_fit(zc,lc,nfit,false);
  std::vector<contr_t> cd=slater_fit(zd,ld,nfit,false);

  GaussianShell ash(la,true,ca);
  GaussianShell bsh(lb,true,cb);
  GaussianShell csh(lc,true,cc);
  GaussianShell dsh(ld,true,cd);

  coords_t cen={0.0, 0.0, 0.0};

  ash.set_first_ind(0);
  ash.set_center(cen,0);

  bsh.set_first_ind(ash.get_last_ind()+1);
  bsh.set_center(cen,0);

  csh.set_first_ind(bsh.get_last_ind()+1);
  csh.set_center(cen,0);

  dsh.set_first_ind(csh.get_last_ind()+1);
  dsh.set_center(cen,0);


  ash.convert_contraction();
  ash.normalize();
  bsh.convert_contraction();
  bsh.normalize();
  csh.convert_contraction();
  csh.normalize();
  dsh.convert_contraction();
  dsh.normalize();

  eri.compute(&ash,&bsh,&csh,&dsh);
  eris=eri.get();

  // Lengths
  int bn=2*lb+1;
  int cn=2*lc+1;
  int dn=2*ld+1;

  // Indices
  int ai=la+ma;
  int bi=lb+mb;
  int ci=lc+mc;
  int di=ld+md;

  return eris[((ai*bn+bi)*cn+ci)*dn+di];
}

// Repulsion integral (ab|cd) = \int \phi_a(r_1) \phi_b(r_1) r^{-1}_{12} \phi_c(r_2) \phi_d(r_2) d^3 r_1 d^3 r_2
double overlap(int na, int nb, double za, double zb, int la, int ma, int lb, int mb) {
  if(la!=lb || ma!=mb)
    return 0.0;

  return normalization(na,za)*normalization(nb,zb)*overlap_primitive(na+nb,za+zb);
}

// Nuclear attraction integral
double nuclear(int na, int nb, double za, double zb, int la, int ma, int lb, int mb) {
  if(la!=lb || ma!=mb)
    return 0.0;

  return -normalization(na,za)*normalization(nb,zb)*overlap_primitive(na+nb-1,za+zb);
}

// Kinetic energy integral
double kinetic(int na, int nb, double za, double zb, int la, int ma, int lb, int mb) {
  if(la!=lb || ma!=mb)
    return 0.0;

  // First term is
  double term1=(lb*(lb+1)-nb*(nb-1))*overlap_primitive(na+nb-2,za+zb);
  // Second term is
  double term2=2*zb*nb*overlap_primitive(na+nb-1,za+zb);
  // Third term is
  double term3=-zb*zb*overlap_primitive(na+nb,za+zb);

  return 0.5*normalization(na,za)*normalization(nb,zb)*(term1+term2+term3);
}

/// Form overlap matrix
arma::mat overlap(const std::vector<bf_t> & basis) {
  arma::mat S(basis.size(),basis.size());
  S.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=overlap(basis[i].n,basis[j].n,basis[i].zeta,basis[j].zeta,basis[i].l,basis[i].m,basis[j].l,basis[j].m);
      S(i,j)=el;
      S(j,i)=el;
    }

  return S;
}

arma::mat kinetic(const std::vector<bf_t> & basis) {
  arma::mat T(basis.size(),basis.size());
  T.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=kinetic(basis[i].n,basis[j].n,basis[i].zeta,basis[j].zeta,basis[i].l,basis[i].m,basis[j].l,basis[j].m);
      T(i,j)=el;
      T(j,i)=el;
    }

  return T;
}

arma::mat nuclear(const std::vector<bf_t> & basis, int Z) {
  arma::mat V(basis.size(),basis.size());
  V.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=nuclear(basis[i].n,basis[j].n,basis[i].zeta,basis[j].zeta,basis[i].l,basis[i].m,basis[j].l,basis[j].m);
      V(i,j)=el;
      V(j,i)=el;
    }

  return Z*V;
}

arma::mat coulomb(const std::vector<bf_t> & basis, const arma::mat & P) {
  arma::mat J(basis.size(),basis.size());
  J.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=0.0;

      for(size_t k=0;k<basis.size();k++)
	for(size_t l=0;l<basis.size();l++)
	  el+=P(k,l)*ERI(basis[i],basis[j],basis[k],basis[l]);

      J(i,j)=el;
      J(j,i)=el;
    }

  return J;
}

arma::vec integral(const std::vector<bf_t> & fitbas, bool coulomb=false) {
  arma::vec r(fitbas.size());
  r.zeros();
  for(size_t i=0;i<fitbas.size();i++)
    if(fitbas[i].l==0) {
      // Integral is
      r(i)=sqrt(4.0*M_PI)*pow(fitbas[i].zeta,-fitbas[i].n-2)*fact(fitbas[i].n+1);

      if(!coulomb) {
	r(i)*=normalization(fitbas[i].n,fitbas[i].zeta);
      } else {
	bf_t dummy;
	dummy.zeta=0.0;
	dummy.n=1;
	dummy.l=0;
	dummy.m=0;

	r(i)/=sqrt(ERI_unnormalized(fitbas[i],dummy,fitbas[i],dummy));
      }
    }

  return r;
}

arma::mat coulomb_coul(const std::vector<bf_t> & basis, const std::vector<bf_t> & fitbas, const arma::mat & P) {
  bf_t dummy;
  dummy.zeta=0.0;
  dummy.n=1;
  dummy.l=0;
  dummy.m=0;

  // Coulomb overlap matrix
  arma::mat V(fitbas.size(),fitbas.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<fitbas.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=ERI_unnormalized(fitbas[i],dummy,fitbas[j],dummy)/sqrt(ERI_unnormalized(fitbas[i],dummy,fitbas[i],dummy)*ERI_unnormalized(fitbas[j],dummy,fitbas[j],dummy));
      V(i,j)=el;
      V(j,i)=el;
    }  
  arma::mat Vinvh=BasOrth(V,false);
  arma::mat Vinv=Vinvh*Vinvh;

  // Three-center integrals
  arma::cube thr(fitbas.size(),basis.size(),basis.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<fitbas.size();i++)
    for(size_t k=0;k<basis.size();k++)
      for(size_t l=0;l<basis.size();l++) {
	thr(i,k,l)=ERI_unnormalized(fitbas[i],dummy,basis[k],basis[l])/sqrt(ERI_unnormalized(fitbas[i],dummy,fitbas[i],dummy))*normalization(basis[k].n,basis[k].zeta)*normalization(basis[l].n,basis[l].zeta);
      }

  // Compute fitting coefficients
  arma::vec gamma(fitbas.size());
  gamma.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<fitbas.size();i++)
    for(size_t k=0;k<basis.size();k++)
      for(size_t l=0;l<basis.size();l++) {
	gamma(i)+=P(k,l)*thr(i,k,l);
      }

  //  printf("Norm of expanded density is %.12f (coul).\n",arma::dot(gamma,integral(fitbas,true)));
  //printf("Norm of expanded density is %.12f (ovl).\n",arma::dot(gamma,integral(fitbas,false)));

  // Translate coefficients
  gamma=Vinv*gamma;
  //  printf("Norm of expanded density is %.12f (coul).\n",arma::dot(gamma,integral(fitbas,true)));
  //  printf("Norm of expanded density is %.12f (ovl).\n",arma::dot(gamma,integral(fitbas,false)));

  // Collect Coulomb matrix
  arma::mat J(basis.size(),basis.size());
  J.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=0.0;

      for(size_t k=0;k<fitbas.size();k++)
	el+=gamma(k)*thr(k,i,j);

      J(i,j)=el;
      J(j,i)=el;
    }

  return J;
}

arma::mat coulomb_ovl(const std::vector<bf_t> & basis, const std::vector<bf_t> & fitbas, const arma::mat & P) {
  // Form overlap matrix
  arma::mat S(fitbas.size(),fitbas.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<fitbas.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=overlap(fitbas[i],fitbas[j]);
      S(i,j)=el;
      S(j,i)=el;
    }

  // Inverse overlap
  arma::mat Sinvh=BasOrth(S,false);
  arma::mat Sinv=Sinvh*Sinvh;

  // Coulomb overlap matrix
  arma::mat V(fitbas.size(),fitbas.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<fitbas.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=coulomb_overlap(fitbas[i],fitbas[j]);
      V(i,j)=el;
      V(j,i)=el;
    }  

  // Three-center integrals
  arma::cube thr(fitbas.size(),basis.size(),basis.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<fitbas.size();i++)
    for(size_t k=0;k<basis.size();k++)
      for(size_t l=0;l<basis.size();l++) {
	thr(i,k,l)=three_overlap(fitbas[i],basis[k],basis[l]);
      }

  // Compute fitting coefficients
  arma::vec gamma(fitbas.size());
  gamma.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<fitbas.size();i++)
    for(size_t k=0;k<basis.size();k++)
      for(size_t l=0;l<basis.size();l++) {
	gamma(i)+=P(k,l)*thr(i,k,l);
      }
  //  printf("Norm of expanded density was %.12f.\n",arma::dot(gamma,integral(fitbas)));
  gamma=Sinv*gamma;

  //  printf("Norm of expanded density was %.12f.\n",arma::dot(gamma,integral(fitbas)));

  /*
    double Nelfit=arma::dot(gamma,integral(fitbas));
    double Nel=arma::trace(P*overlap(basis));
    // Renormalize
    gamma*=Nel/Nelfit;
  */

  // Translate coefficients
  gamma=Sinv*V*gamma;
  
  // Collect Coulomb matrix
  arma::mat J(basis.size(),basis.size());
  J.zeros();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=0.0;

      for(size_t k=0;k<fitbas.size();k++)
	el+=gamma(k)*thr(k,i,j);

      J(i,j)=el;
      J(j,i)=el;
    }

  return J;
}

arma::mat coulomb(const std::vector<bf_t> & basis, const std::vector<bf_t> & fitbas, const arma::mat & P) {
  return coulomb_coul(basis,fitbas,P);
  //  return coulomb_ovl(basis,fitbas,P);
}

arma::mat exchange(const std::vector<bf_t> & basis, const arma::mat & P) {
  arma::mat K(basis.size(),basis.size());
  K.zeros();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t i=0;i<basis.size();i++)
    for(size_t j=0;j<=i;j++) {
      double el=0.0;

      for(size_t k=0;k<basis.size();k++)
	for(size_t l=0;l<basis.size();l++)
	  el+=P(k,l)*ERI(basis[i],basis[k],basis[j],basis[l]);

      K(i,j)=el;
      K(j,i)=el;
    }

  return K;
}
