/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_ERIDIGEST
#define ERKALE_ERIDIGEST

#define digest_J(shpairs,ip,jp,P,ints,ioff,J)				\
  {									\
									\
    /* Amount of functions on the first pair is */			\
    size_t Ni=shpairs[ip].Ni;						\
    size_t Nj=shpairs[ip].Nj;						\
    /* and on second pair is */						\
    size_t Nk=shpairs[jp].Ni;						\
    size_t Nl=shpairs[jp].Nj;						\
									\
    /* First functions on the first pair is */				\
    size_t i0=shpairs[ip].i0;						\
    size_t j0=shpairs[ip].j0;						\
    /* Second pair is */						\
    size_t k0=shpairs[jp].i0;						\
    size_t l0=shpairs[jp].j0;						\
									\
    /* J_ij = (ij|kl) P_kl */						\
    {									\
      /* Work matrix */							\
      arma::mat Jij(Ni,Nj);						\
      Jij.zeros();							\
      arma::mat Pkl=P.submat(k0,l0,k0+Nk-1,l0+Nl-1);			\
									\
      /* Degeneracy factor */						\
      double fac=1.0;							\
      if(ks!=ls)							\
	fac=2.0;							\
									\
      /* Increment matrix */						\
      for(size_t ii=0;ii<Ni;ii++)					\
	for(size_t jj=0;jj<Nj;jj++) {					\
									\
	  /* Matrix element */						\
	  double el=0.0;						\
	  for(size_t kk=0;kk<Nk;kk++)					\
	    for(size_t ll=0;ll<Nl;ll++)					\
	      el+=Pkl(kk,ll)*ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll];	\
									\
	  /* Set the element */						\
	  Jij(ii,jj)+=fac*el;						\
	}								\
									\
      /* Store the matrix element */					\
      J.submat(i0,j0,i0+Ni-1,j0+Nj-1)+=Jij;				\
      if(is!=js)							\
	J.submat(j0,i0,j0+Nj-1,i0+Ni-1)+=arma::trans(Jij);		\
    }									\
									\
    /* Permutation:							\
       J_kl = (ij|kl) P_ij */						\
    if(ip!=jp) {							\
      /* Work matrix */							\
      arma::mat Jkl(Nk,Nl);						\
      Jkl.zeros();							\
      arma::mat Pij=P.submat(i0,j0,i0+Ni-1,j0+Nj-1);			\
									\
      /* Degeneracy factor */						\
      double fac=1.0;							\
      if(is!=js)							\
	fac=2.0;							\
									\
      /* Increment matrix */						\
      for(size_t kk=0;kk<Nk;kk++)					\
	for(size_t ll=0;ll<Nl;ll++) {					\
									\
	  /* Matrix element */						\
	  double el=0.0;						\
	  for(size_t ii=0;ii<Ni;ii++) {					\
	    for(size_t jj=0;jj<Nj;jj++) {				\
	      el+=Pij(ii,jj)*ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll];	\
	    }								\
	  }								\
									\
	  /* Set the element */						\
	  Jkl(kk,ll)+=fac*el;						\
	}								\
									\
      /* Store the matrix element */					\
      J.submat(k0,l0,k0+Nk-1,l0+Nl-1)+=Jkl;				\
      if(ks!=ls)							\
	J.submat(l0,k0,l0+Nl-1,k0+Nk-1)+=arma::trans(Jkl);		\
    }									\
  }

#define digest_K(shpairs,ip,jp,T,P,ints,ioff,K)				\
  {									\
  /* Amount of functions on the first pair is */			\
  size_t Ni=shpairs[ip].Ni;						\
  size_t Nj=shpairs[ip].Nj;						\
  /* and on second pair is */						\
  size_t Nk=shpairs[jp].Ni;						\
  size_t Nl=shpairs[jp].Nj;						\
									\
  /* First functions on the first pair is */				\
  size_t i0=shpairs[ip].i0;						\
  size_t j0=shpairs[ip].j0;						\
  /* Second pair is */							\
  size_t k0=shpairs[jp].i0;						\
  size_t l0=shpairs[jp].j0;						\
									\
  {									\
  /* When all indices are different, the following integrals are	\
     equivalent:							\
     									\
									(ij|kl) (ij|lk) (ji|kl) (ji|lk)	\
									(kl|ij) (kl|ji) (lk|ij) (lk|ji)	\
									\
									This translates to \
									\
									K(i,k) += (ij|kl) P(j,l) // always \
									K(j,k) += (ij|kl) P(i,l) // if (is!=js)	\
									K(i,l) += (ij|kl) P(j,k) // if (ls!=ks)	\
									K(j,l) += (ij|kl) P(i,k) // if (is!=js) and (ls!=ks) \
									\
									and for ij != kl \
									\
									K(k,i) += (ij|kl) P(j,l) // always \
									K(k,j) += (ij|kl) P(i,l) // if (is!=js)	\
									K(l,i) += (ij|kl) P(j,k) // if (ks!=ls)	\
									K(l,j) += (ij|kl) P(i,k) // if (is!=js) and (ks!=ls) \
									\
									However, the latter four permutations just make the \
									exchange matrix symmetric. So the only thing we need to do \
									is do the first four permutations, and at the end we sum up \
									K_ij and K_ji for j>i and set K_ij and K_ji to this \
									value. This makes things a *lot* easier. So: \
									\
									We just need to check if the shells are different, in which \
									case K will get extra increments. \
  */									\
									\
    /* First, do the ik part:						\
       K(i,k) += (ij|kl) P(j,l) */					\
    {									\
      arma::Mat<T> Kik(Ni,Nk);						\
      Kik.zeros();							\
      arma::Mat<T> Pjl =P.submat(j0,l0,j0+Nj-1,l0+Nl-1);		\
									\
      /* Increment Kik */						\
      for(size_t ii=0;ii<Ni;ii++)					\
	for(size_t kk=0;kk<Nk;kk++)					\
	  for(size_t ll=0;ll<Nl;ll++)					\
	    for(size_t jj=0;jj<Nj;jj++)					\
	      {								\
		Kik (ii,kk)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pjl (jj,ll); \
	      }								\
									\
      /* Set elements */						\
      K.submat(i0,k0,i0+Ni-1,k0+Nk-1)+=Kik;				\
      /* Symmetrize if necessary */					\
      if(ip!=jp)							\
	K.submat(k0,i0,k0+Nk-1,i0+Ni-1)+=arma::trans(Kik);		\
    }									\
									\
									\
    /* Then, the second part						\
       K(j,k) += (ij|kl) P(i,l) */					\
    if(is!=js) {							\
      arma::Mat<T> Kjk(Nj,Nk);						\
      Kjk.zeros();							\
      arma::Mat<T> Pil=P.submat(i0,l0,i0+Ni-1,l0+Nl-1);			\
									\
      /* Increment Kjk */						\
      for(size_t jj=0;jj<Nj;jj++)					\
	for(size_t kk=0;kk<Nk;kk++)					\
	  for(size_t ll=0;ll<Nl;ll++)					\
	    for(size_t ii=0;ii<Ni;ii++) {				\
	      Kjk(jj,kk)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pil(ii,ll); \
	    }								\
									\
      /* Set elements */						\
      K.submat(j0,k0,j0+Nj-1,k0+Nk-1)+=Kjk;				\
      /* Symmetrize if necessary (take care about possible overlap with next routine) */ \
      if(ip!=jp) {							\
	K.submat(k0,j0,k0+Nk-1,j0+Nj-1)+=arma::trans(Kjk);		\
      }									\
    }									\
    									\
    /* Third part: K(i,l) += (ij|kl) P(j,k) */				\
    if(ks!=ls) {							\
      arma::Mat<T> Kil(Ni,Nl);						\
      Kil.zeros();							\
      arma::Mat<T> Pjk=P.submat(j0,k0,j0+Nj-1,k0+Nk-1);			\
      									\
      /* Increment Kil */						\
      for(size_t ii=0;ii<Ni;ii++)					\
	for(size_t ll=0;ll<Nl;ll++)					\
	  for(size_t jj=0;jj<Nj;jj++)					\
	    for(size_t kk=0;kk<Nk;kk++) {				\
	      Kil(ii,ll)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pjk(jj,kk); \
	    }								\
      									\
      /* Set elements */						\
      K.submat(i0,l0,i0+Ni-1,l0+Nl-1)+=Kil;				\
      /* Symmetrize if necessary */					\
      if(ip!=jp)							\
	K.submat(l0,i0,l0+Nl-1,i0+Ni-1)+=arma::trans(Kil);		\
    }									\
									\
    /* Last permutation: K(j,l) += (ij|kl) P(i,k) */			\
    if(is!=js && ks!=ls) {						\
      arma::Mat<T> Kjl(Nj,Nl);						\
      Kjl.zeros();							\
      arma::Mat<T> Pik=P.submat(i0,k0,i0+Ni-1,k0+Nk-1);			\
									\
      /* Increment Kjl */						\
      for(size_t jj=0;jj<Nj;jj++)					\
	for(size_t ll=0;ll<Nl;ll++)					\
	  for(size_t ii=0;ii<Ni;ii++)					\
	    for(size_t kk=0;kk<Nk;kk++) {				\
	      Kjl(jj,ll)+=ints[ioff+((ii*Nj+jj)*Nk+kk)*Nl+ll]*Pik(ii,kk);	\
	    }								\
									\
      /* Set elements */						\
      K.submat(j0,l0,j0+Nj-1,l0+Nl-1)+=Kjl;				\
      /* Symmetrize if necessary */					\
      if (ip!=jp)							\
	K.submat(l0,j0,l0+Nl-1,j0+Nj-1)+=arma::trans(Kjl);		\
    }									\
  }									\
  }
#endif
