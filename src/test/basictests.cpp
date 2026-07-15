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

#include "../solidharmonics.h"
#include "../checkpoint.h"
#include "../linalg.h"
#include "../mathf.h"
#include "../settings.h"
#include "../basis.h"
#include "../basislibrary.h"
#include "../cintenv.h"
#include "../eriworker.h"

/// Check orthogonality of spherical harmonics up to
const int Lmax=10;
/// Tolerance for orthonormality
const double orthtol=500*DBL_EPSILON;

/// Test indices
void testind() {
  for(int am=0;am<max_am;am++) {
    int idx=0;
    for(int ii=0;ii<=am;ii++)
      for(int jj=0;jj<=ii;jj++) {
	int l=am-ii;
	int m=ii-jj;
	int n=jj;

	int ind=getind(l,m,n);
	if(ind!=idx) {
	  ERROR_INFO();
	  printf("l=%i, m=%i, n=%i, ind=%i, idx=%i.\n",l,m,n,ind,idx);
	  throw std::runtime_error("Indexing error.\n");
	}

	idx++;
      }
  }

  printf("Indices OK.\n");
}

// Check normalization of spherical harmonics
double cartint(int l, int m, int n) {
  // J. Comput. Chem. 27, 1009-1019 (2006)
  // \int x^l y^m z^n d\Omega =
  // 4 \pi (l-1)!! (m-1)!! (n-1)!! / (l+m+n+1)!! if l,m,n even,
  // 0 otherwise

  if(l%2==1 || m%2==1 || n%2==1)
    return 0.0;

  return 4.0*M_PI*doublefact(l-1)*doublefact(m-1)*doublefact(n-1)/doublefact(l+m+n+1);
}

// Check norm of Y_{l,m}.
void check_sph_orthonorm(int lmax) {

  // Left hand value of l
  for(int ll=0;ll<=lmax;ll++)
    // Right hand value of l
    for(int lr=ll;lr<=lmax;lr++) {

      // Loop over m values
      for(int ml=-ll;ml<=ll;ml++) {
	// Get the coefficients
	std::vector<double> cl=calcYlm_coeff(ll,ml);

	// Form the list of cartesian functions
	std::vector<shellf_t> cartl(((ll+1)*(ll+2))/2);
	size_t n=0;
	for(int i=0; i<=ll; i++) {
	  int nx = ll - i;
	  for(int j=0; j<=i; j++) {
	    int ny = i-j;
	    int nz = j;

	    cartl[n].l=nx;
	    cartl[n].m=ny;
	    cartl[n].n=nz;
	    cartl[n].relnorm=cl[n];
	    n++;
	  }
	}

	for(int mr=-lr;mr<=lr;mr++) {
	  // Get the coefficients
	  std::vector<double> cr=calcYlm_coeff(lr,mr);

	  // Form the list of cartesian functions
	  std::vector<shellf_t> cartr(((lr+1)*(lr+2))/2);
	  size_t N=0;
	  for(int i=0; i<=lr; i++) {
	    int nx = lr - i;
	    for(int j=0; j<=i; j++) {
	      int ny = i-j;
	      int nz = j;

	      cartr[N].l=nx;
	      cartr[N].m=ny;
	      cartr[N].n=nz;
	      cartr[N].relnorm=cr[N];
	      N++;
	    }
	  }

	  // Compute dot product
	  double norm=0.0;
	  for(size_t i=0;i<cartl.size();i++)
	    for(size_t j=0;j<cartr.size();j++)
	      norm+=cartl[i].relnorm*cartr[j].relnorm*cartint(cartl[i].l+cartr[j].l,cartl[i].m+cartr[j].m,cartl[i].n+cartr[j].n);

	  if( (ll==lr) && (ml==mr) ) {
	    if(fabs(norm-1.0)>orthtol) {
	      fprintf(stderr,"Square norm of (%i,%i) is %e, deviation %e from unity!\n",ll,ml,norm,norm-1.0);
	      throw std::runtime_error("Wrong norm.\n");
	    }
	  } else {
	    if(fabs(norm)>orthtol) {
	      fprintf(stderr,"Inner product of (%i,%i) and (%i,%i) is %e!\n",ll,ml,lr,mr,norm);
	      throw std::runtime_error("Functions not orthogonal.\n");
	    }
	  }
	}
      }
    }
}

/// Test checkpoints
void test_checkpoint() {
  // Temporary file name
  std::string tmpfile=tempname();

  {
    // Dummy checkpoint
    Checkpoint chkpt(tmpfile,true);

    // Size of vectors and matrices
    size_t N=5000, M=300;

    /* Vectors */

    // Random vector
    arma::vec randvec=randu_mat(N,1);
    chkpt.write("randvec",randvec);

    arma::vec randvec_load;
    chkpt.read("randvec",randvec_load);

    double vecnorm=arma::norm(randvec-randvec_load,"fro")/N;
    if(vecnorm>DBL_EPSILON) {
      printf("Vector read/write norm %e.\n",vecnorm);
      ERROR_INFO();
      throw std::runtime_error("Error in vector read/write.\n");
    }

    // Complex vector
    arma::cx_vec crandvec=randu_mat(N,1)+std::complex<double>(0.0,1.0)*randu_mat(N,1);
    chkpt.cwrite("crandvec",crandvec);
    arma::cx_vec crandvec_load;
    chkpt.cread("crandvec",crandvec_load);

    double cvecnorm=arma::norm(crandvec-crandvec_load,"fro")/N;
    if(cvecnorm>DBL_EPSILON) {
      printf("Complex vector read/write norm %e.\n",cvecnorm);
      ERROR_INFO();
      throw std::runtime_error("Error in complex vector read/write.\n");
    }

    /* Matrices */
    arma::mat randmat=randn_mat(N,M);
    chkpt.write("randmat",randmat);
    arma::mat randmat_load;
    chkpt.read("randmat",randmat_load);

    double matnorm=arma::norm(randmat-randmat_load,"fro")/(M*N);
    if(matnorm>DBL_EPSILON) {
      printf("Matrix read/write norm %e.\n",matnorm);
      ERROR_INFO();
      throw std::runtime_error("Error in matrix read/write.\n");
    }

  }

  remove(tmpfile.c_str());
}

Settings settings;

/// Check that a natively generally contracted shell (built by merging
/// two same-exponent contractions) reproduces the two separate
/// segmented shells, both when its functions are evaluated in real
/// space and when its integrals are computed.
void check_general_contraction() {
  // Two d-shells sharing three exponents but with different
  // contraction coefficients: a generally contracted block, built in
  // memory so the test needs no basis-set library.
  std::vector<contr_t> c0(3), c1(3);
  const double z[3]={4.0, 1.1, 0.35};
  const double a0[3]={0.20, 0.55, 0.35};
  const double a1[3]={-0.09, 0.31, 0.82};
  for(int i=0;i<3;i++) {
    c0[i].z=z[i]; c0[i].c=a0[i];
    c1[i].z=z[i]; c1[i].c=a1[i];
  }

  const coords_t orig{0.0,0.0,0.0};
  GaussianShell s0(2,true,c0), s1(2,true,c1);
  s0.set_center(orig,0); s1.set_center(orig,0);
  s0.normalize(); s1.normalize();
  s0.set_first_ind(0); s1.set_first_ind(0);

  GaussianShell gc=s0;
  gc.merge_contraction(s1);
  gc.set_first_ind(0);

  if(gc.get_Nctr()!=2 || gc.get_Nbf()!=2*(2*2+1))
    throw std::runtime_error("check_general_contraction: merged shell has the wrong shape.\n");

  // Real-space evaluation
  const double px=0.3, py=-0.2, pz=0.15;
  const arma::vec fgc=gc.eval_func(px,py,pz);
  const arma::vec fstack=arma::join_cols(s0.eval_func(px,py,pz), s1.eval_func(px,py,pz));
  if(arma::abs(fgc-fstack).max() > 1e-12)
    throw std::runtime_error("check_general_contraction: eval_func of the generally contracted shell disagrees.\n");

  // Integrals: the native-GC self-quartet against the segmented one
  std::vector<GaussianShell> gcv{gc};
  CintEnv egc(gcv,false); ERIWorker wgc(egc);
  wgc.compute(0,0,0,0);
  const std::vector<double> igc(*wgc.getp());

  std::vector<GaussianShell> segv{s0,s1};
  CintEnv eseg(segv,false); ERIWorker wseg(eseg);
  const size_t nb=gc.get_Nbf();
  const size_t nlm=nb/gc.get_Nctr();
  double maxd=0.0;
  for(int ci=0;ci<2;ci++)for(int cj=0;cj<2;cj++)for(int ck=0;ck<2;ck++)for(int cl=0;cl<2;cl++) {
    wseg.compute(ci,cj,ck,cl);
    const std::vector<double> * p=wseg.getp();
    for(size_t fi=0;fi<nlm;fi++)for(size_t fj=0;fj<nlm;fj++)for(size_t fk=0;fk<nlm;fk++)for(size_t fl=0;fl<nlm;fl++) {
      const size_t gi=ci*nlm+fi, gj=cj*nlm+fj, gk=ck*nlm+fk, gl=cl*nlm+fl;
      const double vgc=igc[((gi*nb+gj)*nb+gk)*nb+gl];
      const double vseg=(*p)[((fi*nlm+fj)*nlm+fk)*nlm+fl];
      maxd=std::max(maxd,std::fabs(vgc-vseg));
    }
  }
  if(maxd > 1e-10)
    throw std::runtime_error("check_general_contraction: generally contracted integrals disagree with the segmented ones.\n");
}

int main(void) {
  settings.add_scf_settings();
  // Test indices
  testind();
  // Then, check norms of spherical harmonics.
  check_sph_orthonorm(Lmax);
  printf("Solid harmonics OK.\n");
  // Then, check checkpoint utilities
  test_checkpoint();
  printf("Checkpointing OK.\n");
  // Generally contracted shells
  check_general_contraction();
  printf("General contraction OK.\n");
  // Test lapack thread safety
  try {
    check_lapack_thread();
  } catch(std::runtime_error &) {
    throw std::runtime_error("LAPACK library is not thread safe!\nThis might cause problems in some parts of ERKALE.\n");
  }
}
