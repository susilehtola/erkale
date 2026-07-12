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
#include "linalg.h"
#include "mathf.h"
#include "integrals.h"
#include <cfloat>

extern "C" {
#include <cint.h>
#include <cint_funcs.h>
}
// cint.h defines function-like atm() and bas() accessor macros, which
// mangle any same-named variable that is followed by a parenthesis
#undef atm
#undef bas

IntegralWorker::IntegralWorker() {
  input=&arrone;
  output=&arrtwo;

  // Plain Coulomb integrals by default
  rs_omega=0.0;
  rs_alpha=1.0;
  rs_beta=0.0;

  // libcint tables for a single shell quartet; the environment grows
  // on demand in setup_cint_env
  cint_atm.resize(4*ATM_SLOTS);
  cint_bas.resize(4*BAS_SLOTS);
  cint_env.resize(PTR_ENV_START);
}

IntegralWorker::~IntegralWorker() {
}

void IntegralWorker::spherical_transform(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  const bool is_lm=is->lm_in_use();
  const bool js_lm=js->lm_in_use();
  const bool ks_lm=ks->lm_in_use();
  const bool ls_lm=ls->lm_in_use();

  const int am_i=is->get_am();
  const int am_j=js->get_am();
  const int am_k=ks->get_am();
  const int am_l=ls->get_am();

  const size_t Nj_tgt=js->get_Nbf();
  const size_t Nk_tgt=ks->get_Nbf();
  const size_t Nl_tgt=ls->get_Nbf();

  const size_t Ni_cart=is->get_Ncart();
  const size_t Nj_cart=js->get_Ncart();
  const size_t Nk_cart=ks->get_Ncart();

  // Do transforms
  if(ls_lm)
    transform_l(am_l,Ni_cart,Nj_cart,Nk_cart);
  if(ks_lm)
    transform_k(am_k,Ni_cart,Nj_cart,Nl_tgt);
  if(js_lm)
    transform_j(am_j,Ni_cart,Nk_tgt,Nl_tgt);
  if(is_lm)
    transform_i(am_i,Nj_tgt,Nk_tgt,Nl_tgt);
}

ERIWorker::ERIWorker(int maxam, int maxcontr) {
  // The generated spherical transform routines cover angular momenta
  // up to ERKALE's own limit
  if(maxam>max_am) {
    ERROR_INFO();
    throw std::domain_error("The spherical transform tables don't support this angular momentum.\n");
  }
  (void) maxcontr;
}

ERIWorker::~ERIWorker() {
}

dERIWorker::dERIWorker(int maxam, int maxcontr) {
  // The generated spherical transform routines cover angular momenta
  // up to ERKALE's own limit
  if(maxam>max_am) {
    ERROR_INFO();
    throw std::domain_error("The spherical transform tables don't support this angular momentum.\n");
  }
  (void) maxcontr;
}

dERIWorker::~dERIWorker() {
}

void IntegralWorker::setup_cint_env(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  const GaussianShell *shs[4]={is, js, ks, ls};

  // Make sure the environment is large enough: the worker is sized for
  // the orbital basis, but e.g. basis set normalization uses helper
  // shells that may hold more primitives
  {
    size_t need=PTR_ENV_START;
    for(int q=0;q<4;q++)
      need+=3+2*shs[q]->get_Ncontr();
    if(need>cint_env.size())
      cint_env.resize(need);
  }

  // Zero the global parameter block (range separation is set in compute_cartesian)
  std::fill(cint_env.begin(),cint_env.begin()+PTR_ENV_START,0.0);

  int off=PTR_ENV_START;
  for(int q=0;q<4;q++) {
    const GaussianShell *sh=shs[q];

    // Each shell sits on its own dummy atom; nuclear charges don't
    // enter the two-electron integrals
    cint_atm[q*ATM_SLOTS+CHARGE_OF]=0;
    cint_atm[q*ATM_SLOTS+PTR_COORD]=off;
    cint_atm[q*ATM_SLOTS+NUC_MOD_OF]=POINT_NUC;
    cint_atm[q*ATM_SLOTS+PTR_ZETA]=0;
    coords_t cen=sh->get_center();
    cint_env[off++]=cen.x;
    cint_env[off++]=cen.y;
    cint_env[off++]=cen.z;

    // libcint scales cartesian s and p shells by Y_00 = 1/sqrt(4pi)
    // resp. |Y_1m| = sqrt(3/(4pi)) (its common_fac_sp convention) and
    // leaves l>=2 alone. Compensating in the contraction coefficients
    // makes the output plain integrals over the bare primitives; the
    // relnorm factors applied in compute_cartesian then take care of
    // all normalization.
    const int l=sh->get_am();
    double fl=1.0;
    if(l==0)
      fl=2.0*sqrt(M_PI);
    else if(l==1)
      fl=sqrt(4.0*M_PI/3.0);

    const std::vector<contr_t> & c=sh->get_contr_ref();
    cint_bas[q*BAS_SLOTS+ATOM_OF]=q;
    cint_bas[q*BAS_SLOTS+ANG_OF]=l;
    cint_bas[q*BAS_SLOTS+NPRIM_OF]=(int) c.size();
    cint_bas[q*BAS_SLOTS+NCTR_OF]=1;
    cint_bas[q*BAS_SLOTS+KAPPA_OF]=0;
    cint_bas[q*BAS_SLOTS+PTR_EXP]=off;
    for(size_t ip=0;ip<c.size();ip++)
      cint_env[off++]=c[ip].z;
    cint_bas[q*BAS_SLOTS+PTR_COEFF]=off;
    for(size_t ip=0;ip<c.size();ip++)
      cint_env[off++]=c[ip].c*fl;
  }
}

void IntegralWorker::cint_int2e(cint_eri_t kernel, int ncomp, const int *shls_in, size_t N, std::vector<double> & out) {
  CINTIntegralFunction *intor;
  switch(kernel) {
  case CINT_ERI:
    intor=int2e_cart;
    break;
  case CINT_ERI_IP1:
    intor=int2e_ip1_cart;
    break;
  case CINT_ERI_IP2:
    intor=int2e_ip2_cart;
    break;
  default:
    ERROR_INFO();
    throw std::logic_error("Unknown integral kernel!\n");
  }
  int shls[4]={shls_in[0], shls_in[1], shls_in[2], shls_in[3]};

  out.resize(ncomp*N);

  // Evaluate the kernel with the given range separation parameter.
  // The omega value must be set in the environment *before* the
  // scratch size query, since the attenuated integrals need a larger
  // cache than the full-range ones.
  auto evaluate=[&](double omega, std::vector<double> & buf) {
    cint_env[PTR_RANGE_OMEGA]=omega;
    // Make sure the scratch area is large enough (out=NULL is a size query)
    size_t csize=intor(NULL,NULL,shls,cint_atm.data(),4,cint_bas.data(),4,cint_env.data(),NULL,NULL);
    if(csize>cint_cache.size())
      cint_cache.resize(csize);
    if(!intor(buf.data(),NULL,shls,cint_atm.data(),4,cint_bas.data(),4,cint_env.data(),NULL,cint_cache.data()))
      std::fill(buf.begin(),buf.end(),0.0);
  };

  // Full-range Coulomb component
  if(rs_alpha!=0.0)
    evaluate(0.0,out);
  else
    std::fill(out.begin(),out.end(),0.0);

  // Short-range erfc(omega r12)/r12 component: negative omega selects
  // the complementary error function attenuation in libcint
  if(rs_beta!=0.0) {
    cint_sr.resize(ncomp*N);
    evaluate(-rs_omega,cint_sr);
    for(size_t i=0;i<ncomp*N;i++)
      out[i]=rs_alpha*out[i]+rs_beta*cint_sr[i];
  } else if(rs_alpha!=1.0)
    for(size_t i=0;i<ncomp*N;i++)
      out[i]*=rs_alpha;
}

void ERIWorker::compute_cartesian(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  // Fill the libcint tables for this quartet
  setup_cint_env(is,js,ks,ls);

  // libcint runs the first shell fastest in its output buffer, whereas
  // ERKALE stores the last index fastest: computing the reversed
  // quartet (lk|ji), which by the permutational symmetry of the
  // integrals equals (ij|kl), yields the result directly in ERKALE's
  // layout. libcint places no restrictions on the angular momentum
  // order within the quartet, so no swaps are needed either.
  int shls[4]={3, 2, 1, 0};

  size_t N=is->get_Ncart()*js->get_Ncart()*ks->get_Ncart()*ls->get_Ncart();
  cint_int2e(CINT_ERI,1,shls,N,cint_out);

  // Collect the results, plugging in the normalization factors
  size_t ind_ij, ind_ijk, ind;
  double norm_i, norm_ij, norm_ijk, norm;

  // Numbers of functions on each shell
  const std::vector<shellf_t> & ci=is->get_cart_ref();
  const std::vector<shellf_t> & cj=js->get_cart_ref();
  const std::vector<shellf_t> & ck=ks->get_cart_ref();
  const std::vector<shellf_t> & cl=ls->get_cart_ref();
  (*input).resize(N);

  for(size_t ii=0;ii<ci.size();ii++) {
    norm_i=ci[ii].relnorm;
    for(size_t ji=0;ji<cj.size();ji++) {
      ind_ij=ii*cj.size()+ji;
      norm_ij=cj[ji].relnorm*norm_i;
      for(size_t ki=0;ki<ck.size();ki++) {
	ind_ijk=ind_ij*ck.size()+ki;
	norm_ijk=ck[ki].relnorm*norm_ij;
	for(size_t li=0;li<cl.size();li++) {
	  // Index in computed integrals table
	  ind=ind_ijk*cl.size()+li;
	  // Total norm factor
	  norm=cl[li].relnorm*norm_ijk;
	  // Store scaled integral
	  (*input)[ind]=norm*cint_out[ind];
	}
      }
    }
  }
}

void dERIWorker::compute_cartesian() {
  // Fill the libcint tables for this quartet
  setup_cint_env(is,js,ks,ls);

  const size_t Ni=is->get_Ncart();
  const size_t Nj=js->get_Ncart();
  const size_t Nk=ks->get_Ncart();
  const size_t Nl=ls->get_Ncart();
  const size_t N=Ni*Nj*Nk*Nl;

  // As in ERIWorker::compute_cartesian, the reversed quartet (lk|ji)
  // yields the libcint buffer directly in ERKALE's layout. int2e_ip1
  // differentiates the first shell of the quartet and int2e_ip2 the
  // third, so the reversed quartet gives the derivatives with respect
  // to the 4th and 2nd shell; the 3rd shell derivative comes from the
  // quartet (kl|ji), whose buffer needs a layout remap. The 1st shell
  // derivative follows from translational invariance in get_idx().
  int rev[4]={3, 2, 1, 0};
  int kfirst[4]={2, 3, 1, 0};

  // Derivatives with respect to a dummy shell (zero exponent; the
  // density fitting routines express two- and three-center integrals
  // as four-center ones with dummy padding) vanish identically, but
  // libcint does not produce clean zeros for them. Zero those blocks
  // explicitly: this also keeps the translational invariance
  // reconstruction of the first shell derivative in get_idx() valid.
  auto isdummy=[](const GaussianShell *sh) {
    const std::vector<contr_t> & c=sh->get_contr_ref();
    for(size_t ip=0;ip<c.size();ip++)
      if(c[ip].z!=0.0)
	return false;
    return true;
  };

  if(isdummy(ls))
    cint_dL.assign(3*N,0.0);
  else
    cint_int2e(CINT_ERI_IP1,3,rev,N,cint_dL);

  if(isdummy(js))
    cint_dJ.assign(3*N,0.0);
  else
    cint_int2e(CINT_ERI_IP2,3,rev,N,cint_dJ);

  cint_dK.resize(3*N);
  if(isdummy(ks))
    std::fill(cint_dK.begin(),cint_dK.end(),0.0);
  else {
    cint_int2e(CINT_ERI_IP1,3,kfirst,N,cint_scr);
    // Remap the 3rd shell derivative into ERKALE layout
    for(int cmp=0;cmp<3;cmp++)
      for(size_t ii=0;ii<Ni;ii++)
	for(size_t jj=0;jj<Nj;jj++)
	  for(size_t kk=0;kk<Nk;kk++)
	    for(size_t ll=0;ll<Nl;ll++)
	      cint_dK[cmp*N+((ii*Nj+jj)*Nk+kk)*Nl+ll]=cint_scr[cmp*N+(kk+Nk*(ll+Nl*(jj+Nj*ii)))];
  }

  // Plug in the normalization factors, and the sign: libcint's ip is
  // the derivative with respect to the electron coordinate, which is
  // minus the derivative with respect to the shell center.
  const std::vector<shellf_t> & ci=is->get_cart_ref();
  const std::vector<shellf_t> & cj=js->get_cart_ref();
  const std::vector<shellf_t> & ck=ks->get_cart_ref();
  const std::vector<shellf_t> & cl=ls->get_cart_ref();

  size_t ind_i, ind_ij, ind_ijk, ind;
  double norm_i, norm_ij, norm_ijk, norm;
  for(size_t ii=0;ii<ci.size();ii++) {
    ind_i=ii*cj.size();
    norm_i=ci[ii].relnorm;
    for(size_t ji=0;ji<cj.size();ji++) {
      ind_ij=(ind_i+ji)*ck.size();
      norm_ij=cj[ji].relnorm*norm_i;
      for(size_t ki=0;ki<ck.size();ki++) {
        ind_ijk=(ind_ij+ki)*cl.size();
        norm_ijk=ck[ki].relnorm*norm_ij;
        for(size_t li=0;li<cl.size();li++) {
          ind=ind_ijk+li;
          norm=-cl[li].relnorm*norm_ijk;
          for(int cmp=0;cmp<3;cmp++) {
            cint_dJ[cmp*N+ind]*=norm;
            cint_dK[cmp*N+ind]*=norm;
            cint_dL[cmp*N+ind]*=norm;
          }
        }
      }
    }
  }
}

void dERIWorker::get_idx(int idx) {
  // Amount of integrals is
  size_t N=(is->get_Ncart())*(js->get_Ncart())*(ks->get_Ncart())*(ls->get_Ncart());
  (*input).resize(N);

  if(idx<0 || idx>=12) {
    ERROR_INFO();
    throw std::runtime_error("Invalid derivative index requested!\n");
  }

  const int cmp=idx%3;
  const double *dJ=cint_dJ.data()+cmp*N;
  const double *dK=cint_dK.data()+cmp*N;
  const double *dL=cint_dL.data()+cmp*N;

  switch(idx/3) {
  case 0:
    // Derivative wrt the first center from translational invariance:
    // d/dA_i = - d/dB_i - d/dC_i - d/dD_i
    for(size_t i=0;i<N;i++)
      (*input)[i]=-dJ[i]-dK[i]-dL[i];
    break;
  case 1:
    for(size_t i=0;i<N;i++)
      (*input)[i]=dJ[i];
    break;
  case 2:
    for(size_t i=0;i<N;i++)
      (*input)[i]=dK[i];
    break;
  case 3:
    for(size_t i=0;i<N;i++)
      (*input)[i]=dL[i];
    break;
  }

  // Transform to the spherical basis
  spherical_transform(is,js,ks,ls);
}

std::vector<double> dERIWorker::get(int idx) {
  // Get the integrals
  get_idx(idx);

#ifdef DEBUGDERIV
  // Evaluate other integrals
  std::vector<double> eris=get_debug(idx);

  // Check the integrals
  size_t Nfail=0;
  for(size_t i=0;i<N;i++)
    if(fabs(ints[i]-eris[i])>1e-6) {
      Nfail++;
      printf("(%c %c %c %c) integral, idx = %i, first functions (%i %i %i %i)\n",shell_types[is->get_am()],shell_types[js->get_am()],shell_types[ks->get_am()],shell_types[ls->get_am()],idx,(int) is->get_first_ind(),(int) js->get_first_ind(),(int) ks->get_first_ind(),(int) ls->get_first_ind());
      printf("%i % e % e % e\n",(int) i,ints[i],eris[i],ints[i]-eris[i]);
    }

  if(Nfail) {
    /*
      is->print();
      js->print();
      ks->print();
      ls->print();
    */

    /*
      for(size_t i=0;i<N;i++)
      printf("%i % e % e % e\n",(int) i,ints[i],eris[i],ints[i]-eris[i]);
    */
    printf("%i integrals failed\n",(int) Nfail);
  }

#endif

  return *input;
}

const std::vector<double> * dERIWorker::getp(int idx) {
  get_idx(idx);
  return input;
}


std::vector<double> dERIWorker::get_debug(int idx) {
  // Amount of integrals
  size_t N=(is->get_Ncart())*(js->get_Ncart())*(ks->get_Ncart())*(ls->get_Ncart());
  (*input).resize(N);

  // Form cartesian shells
  GaussianShell isc(is->get_am(),false,is->get_contr());
  isc.set_center(is->get_center(),is->get_center_ind());
  isc.set_first_ind(is->get_first_ind());
  isc.normalize(); // Need to normalize, otherwise cartesian relative factors aren't initalized

  GaussianShell jsc(js->get_am(),false,js->get_contr());
  jsc.set_center(js->get_center(),js->get_center_ind());
  jsc.set_first_ind(js->get_first_ind());
  jsc.normalize(); // Need to normalize, otherwise cartesian relative factors aren't initalized

  GaussianShell ksc(ks->get_am(),false,ks->get_contr());
  ksc.set_center(ks->get_center(),ks->get_center_ind());
  ksc.set_first_ind(ks->get_first_ind());
  ksc.normalize(); // Need to normalize, otherwise cartesian relative factors aren't initalized

  GaussianShell lsc(ls->get_am(),false,ls->get_contr());
  lsc.set_center(ls->get_center(),ls->get_center_ind());
  lsc.set_first_ind(ls->get_first_ind());
  lsc.normalize(); // Need to normalize, otherwise cartesian relative factors aren't initalized

  // Sizes
  size_t Ni=isc.get_Ncart();
  size_t Nj=jsc.get_Ncart();
  size_t Nk=ksc.get_Ncart();
  size_t Nl=lsc.get_Ncart();

  // ERI evaluator
  ERIWorker eri(std::max(std::max(isc.get_am(),jsc.get_am()),std::max(ksc.get_am(),lsc.get_am()))+1,std::max(std::max(isc.get_Ncontr(),jsc.get_Ncontr()),std::max(ksc.get_Ncontr(),lsc.get_Ncontr())));

  // Zero out (*input) array
  (*input).assign(N,0.0);

  // Which derivative are we taking?
  if(idx>=0 && idx<3) {
    // Get normalized contraction.
    std::vector<contr_t> ic=isc.get_contr_normalized();

    // Get cartesians
    std::vector<shellf_t> icart=isc.get_cart();

    // Loop over contraction
    for(size_t iic=0;iic<ic.size();iic++) {
      // Dummy contraction
      std::vector<contr_t> dumcontr(1);
      dumcontr[0].c=1.0;
      dumcontr[0].z=ic[iic].z;

      // Form helpers
      GaussianShell icp(is->get_am()+1,false,dumcontr);
      icp.set_center(is->get_center(),is->get_center_ind());
      icp.set_first_ind(is->get_first_ind());
      icp.normalize();

      // Evaluate ERI
      const std::vector<double> * erip;
      eri.compute(&icp,&jsc,&ksc,&lsc);
      erip=eri.getp();

      // Collect terms
      for(size_t ica=0;ica<icart.size();ica++) {
	int l=icart[ica].l;
	int m=icart[ica].m;
	int n=icart[ica].n;

	int il=l;
	int im=m;
	int in=n;

	double fac=ic[iic].c*sqrt(ic[iic].z);
	if(idx==0) {
	  fac*=sqrt(2*l+1);
	  il++;
	} else if(idx==1) {
	  fac*=sqrt(2*m+1);
	  im++;
	} else if(idx==2) {
	  fac*=sqrt(2*n+1);
	  in++;
	}

	// i index of target integral is
	size_t idt=getind(l,m,n);
	// i index of source integral is
	size_t idp=getind(il,im,in);

	// Loop over target integrals
	for(size_t jj=0;jj<Nj;jj++)
	  for(size_t kk=0;kk<Nk;kk++)
	    for(size_t ll=0;ll<Nl;ll++)
	      (*input)[((idt*Nj+jj)*Nk+kk)*Nl+ll]+=fac*(*erip)[((idp*Nj+jj)*Nk+kk)*Nl+ll];
      }

      if(is->get_am()>0) {
	GaussianShell icm=GaussianShell(is->get_am()-1,false,dumcontr);
	icm.set_center(is->get_center(),is->get_center_ind());
	icm.set_first_ind(is->get_first_ind());
	icm.normalize();

	// Evaluate ERI
	eri.compute(&icm,&jsc,&ksc,&lsc);
	erip=eri.getp();

	// Collect terms
	for(size_t ica=0;ica<icart.size();ica++) {
	  int l=icart[ica].l;
	  int m=icart[ica].m;
	  int n=icart[ica].n;

	  // Skip nonexistent integrals
	  if(idx==0 && l==0)
	    continue;
	  if(idx==1 && m==0)
	    continue;
	  if(idx==2 && n==0)
	    continue;

	  int il=l;
	  int im=m;
	  int in=n;

	  double fac=-2*ic[iic].c*sqrt(ic[iic].z);
	  if(idx==0) {
	    fac*=l/sqrt(2*l-1);
	    il--;
	  } else if(idx==1) {
	    fac*=m/sqrt(2*m-1);
	    im--;
	  } else if(idx==2) {
	    fac*=n/sqrt(2*n-1);
	    in--;
	  }

	  // i index of target integral is
	  size_t idt=getind(l,m,n);
	  // i index of source integral is
	  size_t idm=getind(il,im,in);

	  // Loop over target integrals
	  for(size_t jj=0;jj<Nj;jj++)
	    for(size_t kk=0;kk<Nk;kk++)
	      for(size_t ll=0;ll<Nl;ll++)
		(*input)[((idt*Nj+jj)*Nk+kk)*Nl+ll]+=fac*(*erip)[((idm*Nj+jj)*Nk+kk)*Nl+ll];
	}
      }
    }

  } else if(idx>=3 && idx<6) {
    // Get normalized contraction.
    std::vector<contr_t> jc=jsc.get_contr_normalized();

    // Get cartesians
    std::vector<shellf_t> jcart=jsc.get_cart();

    // Loop over contraction
    for(size_t jic=0;jic<jc.size();jic++) {
      // Dummy contraction
      std::vector<contr_t> dumcontr(1);
      dumcontr[0].c=1.0;
      dumcontr[0].z=jc[jic].z;

      // Form helpers
      GaussianShell jcp(js->get_am()+1,false,dumcontr);
      jcp.set_center(js->get_center(),js->get_center_ind());
      jcp.set_first_ind(js->get_first_ind());
      jcp.normalize();
      size_t Njp=jcp.get_Ncart();

      // Evaluate ERI
      const std::vector<double> * erip;
      eri.compute(&isc,&jcp,&ksc,&lsc);
      erip=eri.getp();

      // Collect terms
      for(size_t jca=0;jca<jcart.size();jca++) {
	int l=jcart[jca].l;
	int m=jcart[jca].m;
	int n=jcart[jca].n;

	int il=l;
	int im=m;
	int in=n;

	double fac=jc[jic].c*sqrt(jc[jic].z);
	if(idx==3) {
	  fac*=sqrt(2*l+1);
	  il++;
	} else if(idx==4) {
	  fac*=sqrt(2*m+1);
	  im++;
	} else if(idx==5) {
	  fac*=sqrt(2*n+1);
	  in++;
	}

	// j index of target integral is
	size_t jdt=getind(l,m,n);
	// j index of source integral is
	size_t jdp=getind(il,im,in);

	// Loop over target integrals
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t kk=0;kk<Nk;kk++)
	    for(size_t ll=0;ll<Nl;ll++)
	      (*input)[((ii*Nj+jdt)*Nk+kk)*Nl+ll]+=fac*(*erip)[((ii*Njp+jdp)*Nk+kk)*Nl+ll];
      }

      if(js->get_am()>0) {
	GaussianShell jcm=GaussianShell(js->get_am()-1,false,dumcontr);
	jcm.set_center(js->get_center(),js->get_center_ind());
	jcm.set_first_ind(js->get_first_ind());
	jcm.normalize();
	size_t Njm=jcm.get_Ncart();

	// Evaluate ERI
	eri.compute(&isc,&jcm,&ksc,&lsc);
	erip=eri.getp();

	// Collect terms
	for(size_t jca=0;jca<jcart.size();jca++) {
	  int l=jcart[jca].l;
	  int m=jcart[jca].m;
	  int n=jcart[jca].n;

	  // Skip nonexistent integrals
	  if(idx==3 && l==0)
	    continue;
	  if(idx==4 && m==0)
	    continue;
	  if(idx==5 && n==0)
	    continue;

	  int il=l;
	  int im=m;
	  int in=n;

	  double fac=-jc[jic].c*sqrt(jc[jic].z);
	  if(idx==3) {
	    fac*=2*l/sqrt(2*l-1);
	    il--;
	  } else if(idx==4) {
	    fac*=2*m/sqrt(2*m-1);
	    im--;
	  } else if(idx==5) {
	    fac*=2*n/sqrt(2*n-1);
	    in--;
	  }

	  // j index of target integral is
	  size_t jdt=getind(l,m,n);
	  // j index of source integral is
	  size_t jdm=getind(il,im,in);

	  // Loop over target integrals
	  for(size_t ii=0;ii<Ni;ii++)
	    for(size_t kk=0;kk<Nk;kk++)
	      for(size_t ll=0;ll<Nl;ll++)
		(*input)[((ii*Nj+jdt)*Nk+kk)*Nl+ll]+=fac*(*erip)[((ii*Njm+jdm)*Nk+kk)*Nl+ll];
	}
      }
    }

  } else if(idx>=6 && idx<9) {
    // Get normalized contraction.
    std::vector<contr_t> kc=ksc.get_contr_normalized();

    // Get cartesians
    std::vector<shellf_t> kcart=ksc.get_cart();

    // Loop over contraction
    for(size_t kic=0;kic<kc.size();kic++) {
      // Dummy contraction
      std::vector<contr_t> dumcontr(1);
      dumcontr[0].c=1.0;
      dumcontr[0].z=kc[kic].z;

      // Form helpers
      GaussianShell kcp(ks->get_am()+1,false,dumcontr);
      kcp.set_center(ks->get_center(),ks->get_center_ind());
      kcp.set_first_ind(ks->get_first_ind());
      kcp.normalize();
      size_t Nkp=kcp.get_Ncart();

      // Evaluate ERI
      const std::vector<double> * erip;
      eri.compute(&isc,&jsc,&kcp,&lsc);
      erip=eri.getp();

      // Collect terms
      for(size_t kca=0;kca<kcart.size();kca++) {
	int l=kcart[kca].l;
	int m=kcart[kca].m;
	int n=kcart[kca].n;

	int il=l;
	int im=m;
	int in=n;

	double fac=kc[kic].c*sqrt(kc[kic].z);
	if(idx==6) {
	  fac*=sqrt(2*l+1);
	  il++;
	} else if(idx==7) {
	  fac*=sqrt(2*m+1);
	  im++;
	} else if(idx==8) {
	  fac*=sqrt(2*n+1);
	  in++;
	}

	// k index of target integral is
	size_t kdt=getind(l,m,n);
	// k index of source integral is
	size_t kdp=getind(il,im,in);

	// Loop over target integrals
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t jj=0;jj<Nj;jj++)
	    for(size_t ll=0;ll<Nl;ll++)
	      (*input)[((ii*Nj+jj)*Nk+kdt)*Nl+ll]+=fac*(*erip)[((ii*Nj+jj)*Nkp+kdp)*Nl+ll];
      }

      if(ks->get_am()>0) {
	GaussianShell kcm=GaussianShell(ks->get_am()-1,false,dumcontr);
	kcm.set_center(ks->get_center(),ks->get_center_ind());
	kcm.set_first_ind(ks->get_first_ind());
	kcm.normalize();
	size_t Nkm=kcm.get_Ncart();

	// Evaluate ERI
	eri.compute(&isc,&jsc,&kcm,&lsc);
	erip=eri.getp();

	// Collect terms
	for(size_t kca=0;kca<kcart.size();kca++) {
	  int l=kcart[kca].l;
	  int m=kcart[kca].m;
	  int n=kcart[kca].n;

	  // Skip nonexistent integrals
	  if(idx==6 && l==0)
	    continue;
	  if(idx==7 && m==0)
	    continue;
	  if(idx==8 && n==0)
	    continue;

	  int il=l;
	  int im=m;
	  int in=n;

	  double fac=-kc[kic].c*sqrt(kc[kic].z);
	  if(idx==6) {
	    fac*=2*l/sqrt(2*l-1);
	    il--;
	  } else if(idx==7) {
	    fac*=2*m/sqrt(2*m-1);
	    im--;
	  } else if(idx==8) {
	    fac*=2*n/sqrt(2*n-1);
	    in--;
	  }

	  // k index of target integral is
	  size_t kdt=getind(l,m,n);
	  // k index of source integral is
	  size_t kdm=getind(il,im,in);

	  // Loop over target integrals
	  for(size_t ii=0;ii<Ni;ii++)
	    for(size_t jj=0;jj<Nj;jj++)
	      for(size_t ll=0;ll<Nl;ll++)
		(*input)[((ii*Nj+jj)*Nk+kdt)*Nl+ll]+=fac*(*erip)[((ii*Nj+jj)*Nkm+kdm)*Nl+ll];
	}
      }
    }
  } else if(idx>=9 && idx<12) {
    // Get normalized contraction.
    std::vector<contr_t> lc=lsc.get_contr_normalized();

    // Get cartesians
    std::vector<shellf_t> lcart=lsc.get_cart();

    // Loop over contraction
    for(size_t lic=0;lic<lc.size();lic++) {
      // Dummy contraction
      std::vector<contr_t> dumcontr(1);
      dumcontr[0].c=1.0;
      dumcontr[0].z=lc[lic].z;

      // Form helpers
      GaussianShell lcp(ls->get_am()+1,false,dumcontr);
      lcp.set_center(ls->get_center(),ls->get_center_ind());
      lcp.set_first_ind(ls->get_first_ind());
      lcp.normalize();
      size_t Nlp=lcp.get_Ncart();

      // Evaluate ERI
      const std::vector<double> * erip;
      eri.compute(&isc,&jsc,&ksc,&lcp);
      erip=eri.getp();

      // Collect terms
      for(size_t lca=0;lca<lcart.size();lca++) {
	int l=lcart[lca].l;
	int m=lcart[lca].m;
	int n=lcart[lca].n;

	int il=l;
	int im=m;
	int in=n;

	double fac=lc[lic].c*sqrt(lc[lic].z);
	if(idx==9) {
	  fac*=sqrt(2*l+1);
	  il++;
	} else if(idx==10) {
	  fac*=sqrt(2*m+1);
	  im++;
	} else if(idx==11) {
	  fac*=sqrt(2*n+1);
	  in++;
	}

	// l index of target integral is
	size_t ldt=getind(l,m,n);
	// l index of source integral is
	size_t ldp=getind(il,im,in);

	// Loop over target integrals
	for(size_t ii=0;ii<Ni;ii++)
	  for(size_t jj=0;jj<Nj;jj++)
	    for(size_t kk=0;kk<Nk;kk++)
	      (*input)[((ii*Nj+jj)*Nk+kk)*Nl+ldt]+=fac*(*erip)[((ii*Nj+jj)*Nk+kk)*Nlp+ldp];
      }

      if(ls->get_am()>0) {
	GaussianShell lcm=GaussianShell(ls->get_am()-1,false,dumcontr);
	lcm.set_center(ls->get_center(),ls->get_center_ind());
	lcm.set_first_ind(ls->get_first_ind());
	lcm.normalize();
	size_t Nlm=lcm.get_Ncart();

	// Evaluate ERI
	eri.compute(&isc,&jsc,&ksc,&lcm);
	erip=eri.getp();

	// Collect terms
	for(size_t lca=0;lca<lcart.size();lca++) {
	  int l=lcart[lca].l;
	  int m=lcart[lca].m;
	  int n=lcart[lca].n;

	  // Skip nonexistent integrals
	  if(idx==9 && l==0)
	    continue;
	  if(idx==10 && m==0)
	    continue;
	  if(idx==11 && n==0)
	    continue;

	  int il=l;
	  int im=m;
	  int in=n;

	  double fac=-lc[lic].c*sqrt(lc[lic].z);
	  if(idx==9) {
	    fac*=2*l/sqrt(2*l-1);
	    il--;
	  } else if(idx==10) {
	    fac*=2*m/sqrt(2*m-1);
	    im--;
	  } else if(idx==11) {
	    fac*=2*n/sqrt(2*n-1);
	    in--;
	  }

	  // l index of target integral is
	  size_t ldt=getind(l,m,n);
	  // l index of source integral is
	  size_t ldm=getind(il,im,in);

	  // Loop over target integrals
	  for(size_t ii=0;ii<Ni;ii++)
	    for(size_t jj=0;jj<Nj;jj++)
	      for(size_t kk=0;kk<Nk;kk++)
		(*input)[((ii*Nj+jj)*Nk+kk)*Nl+ldt]+=fac*(*erip)[((ii*Nj+jj)*Nk+kk)*Nlm+ldm];
	}
      }
    }
  }

  // Convert the integrals to the shperical harmonics basis
  spherical_transform(is,js,ks,ls);

  // Return the integrals
  return *input;
}



void ERIWorker::compute(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  // Calculate ERIs and transform them to spherical harmonics basis, if necessary.

  // Get the cartesian ERIs (libcint places no restrictions on the
  // angular momentum order, so no shell swaps are needed)
  compute_cartesian(is,js,ks,ls);
  // and transform them into the spherical basis
  spherical_transform(is,js,ks,ls);
}

void ERIWorker::compute_debug(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  // Get the cartesian ERIs
  compute_cartesian_debug(is,js,ks,ls);
  // and transform them into the spherical basis
  spherical_transform(is,js,ks,ls);
}

void ERIWorker::compute_cartesian_debug(const GaussianShell *is, const GaussianShell *js, const GaussianShell *ks, const GaussianShell *ls) {
  std::vector<shellf_t> carti(is->get_cart());
  std::vector<shellf_t> cartj(js->get_cart());
  std::vector<shellf_t> cartk(ks->get_cart());
  std::vector<shellf_t> cartl(ls->get_cart());

  std::vector<contr_t> contri(is->get_contr());
  std::vector<contr_t> contrj(js->get_contr());
  std::vector<contr_t> contrk(ks->get_contr());
  std::vector<contr_t> contrl(ls->get_contr());

  coords_t Ri(is->get_center());
  coords_t Rj(js->get_center());
  coords_t Rk(ks->get_center());
  coords_t Rl(ls->get_center());

  input->assign(carti.size()*cartj.size()*cartk.size()*cartl.size(),0.0);

  for(size_t ic=0;ic<carti.size();ic++)
    for(size_t jc=0;jc<cartj.size();jc++)
      for(size_t kc=0;kc<cartk.size();kc++)
	for(size_t lc=0;lc<cartl.size();lc++) {
	  int li(carti[ic].l);
	  int mi(carti[ic].m);
	  int ni(carti[ic].n);
	  double reli(carti[ic].relnorm);

	  int lj(cartj[jc].l);
	  int mj(cartj[jc].m);
	  int nj(cartj[jc].n);
	  double relj(cartj[jc].relnorm);

	  int lk(cartk[kc].l);
	  int mk(cartk[kc].m);
	  int nk(cartk[kc].n);
	  double relk(cartk[kc].relnorm);

	  int ll(cartl[lc].l);
	  int ml(cartl[lc].m);
	  int nl(cartl[lc].n);
	  double rell(cartl[lc].relnorm);

	  double el=0.0;

	  for(size_t xi=0;xi<contri.size();xi++)
	    for(size_t xj=0;xj<contrj.size();xj++)
	      for(size_t xk=0;xk<contrk.size();xk++)
		for(size_t xl=0;xl<contrl.size();xl++) {
		  double zi(contri[xi].z);
		  double zj(contrj[xj].z);
		  double zk(contrk[xk].z);
		  double zl(contrl[xl].z);

		  double ci(contri[xi].c);
		  double cj(contrj[xj].c);
		  double ck(contrk[xk].c);
		  double cl(contrl[xl].c);

		  el+=ci*cj*ck*cl*ERI_int(li,mi,ni,Ri.x,Ri.y,Ri.z,zi,	\
					  lj,mj,nj,Rj.x,Rj.y,Rj.z,zj,	\
					  lk,mk,nk,Rk.x,Rk.y,Rk.z,zk,	\
					  ll,ml,nl,Rl.x,Rl.y,Rl.z,zl);
		}

	  (*input)[((ic*cartj.size()+jc)*cartk.size()+kc)*cartl.size()+lc]=reli*relj*relk*rell*el;
	}
}

std::vector<double> ERIWorker::get() const {
  return *input;
}

std::vector<double> & ERIWorker::rget() const {
  return *input;
}

const std::vector<double> * ERIWorker::getp() const {
  return input;
}

void dERIWorker::compute(const GaussianShell *is_in, const GaussianShell *js_in, const GaussianShell *ks_in, const GaussianShell *ls_in) {
  // Store the shells (libcint places no restrictions on the angular
  // momentum order, so no swaps are needed)
  is=is_in;
  js=js_in;
  ks=ks_in;
  ls=ls_in;

  // Calculate the derivatives
  compute_cartesian();
}


ERIWorker_srlr::ERIWorker_srlr(int maxam, int maxcontr, double w, double a, double b) : ERIWorker(maxam,maxcontr) {
  rs_omega=w;
  rs_alpha=a;
  rs_beta=b;
}

ERIWorker_srlr::~ERIWorker_srlr() {
}

dERIWorker_srlr::dERIWorker_srlr(int maxam, int maxcontr, double w, double a, double b) : dERIWorker(maxam,maxcontr) {
  rs_omega=w;
  rs_alpha=a;
  rs_beta=b;
}

dERIWorker_srlr::~dERIWorker_srlr() {
}

