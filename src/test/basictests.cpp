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
#include "../xyzutils.h"
#include "../dftgrid.h"
#include "../elements.h"

#include <cstdio>
#include <fstream>
#include <sstream>

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
  s0.first_ind(0); s1.first_ind(0);

  GaussianShell gc=s0;
  gc.merge_contraction(s1);
  gc.first_ind(0);

  if(gc.Nctr()!=2 || gc.Nbf()!=2*(2*2+1))
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
  const size_t nb=gc.Nbf();
  const size_t nlm=nb/gc.Nctr();
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

/// Load minimal BSE-format JSON and verify (1) a segmented STO-3G
/// hydrogen shell parses to the canonical contraction and dispatches
/// through load_basis, and (2) a two-row electron_shell sharing its
/// exponents loads as a single generally contracted FunctionShell and
/// round-trips through save_bse_json bit-for-bit.
void test_bse_json() {
  // STO-3G hydrogen: one s shell, three primitives (segmented). Plus a
  // synthetic carbon s shell with two coefficient rows over the same
  // three exponents -- a generally contracted block.
  const std::string json_str = R"JSON({
  "molssi_bse_schema": {"schema_type":"complete","schema_version":"0.1"},
  "name": "test-mixed",
  "elements": {
    "1": {
      "electron_shells": [
        {
          "function_type": "gto", "region": "valence",
          "angular_momentum": [0],
          "exponents": ["3.42525091", "0.62391373", "0.16885540"],
          "coefficients": [["0.15432897", "0.53532814", "0.44463454"]]
        }
      ]
    },
    "6": {
      "electron_shells": [
        {
          "function_type": "gto", "region": "valence",
          "angular_momentum": [0],
          "exponents": ["4.0", "1.1", "0.35"],
          "coefficients": [["0.20", "0.55", "0.35"], ["-0.09", "0.31", "0.82"]]
        }
      ]
    }
  }
})JSON";

  const std::string tmpname = "bse_test_mixed";
  const std::string tmpfile = tmpname + ".json";
  {
    std::ofstream of(tmpfile);
    of << json_str;
  }

  // Direct API and the load_basis dispatch (a "<name>.json" in the cwd
  // must win over any legacy .gbs entry).
  BasisSetLibrary lib;
  lib.load_bse_json(tmpfile, false);
  BasisSetLibrary lib_dispatch;
  lib_dispatch.load_basis(tmpname, false);
  remove(tmpfile.c_str());

  if(lib.get_Nel() != 2)
    throw std::runtime_error("BSE JSON: expected 2 elements.\n");

  // Segmented hydrogen shell
  std::vector<FunctionShell> Hsh = lib.get_element("H").get_shells();
  if(Hsh.size() != 1 || Hsh[0].get_am() != 0 || Hsh[0].get_Nctr() != 1)
    throw std::runtime_error("BSE JSON: expected one segmented s shell on H.\n");
  const std::vector<contr_t> Hc = Hsh[0].get_contr();
  const double z_ref[] = {3.42525091, 0.62391373, 0.16885540};
  const double c_ref[] = {0.15432897, 0.53532814, 0.44463454};
  if(Hc.size() != 3)
    throw std::runtime_error("BSE JSON: expected 3 primitives in the H s shell.\n");
  for(size_t k=0; k<3; k++)
    if(std::abs(Hc[k].z - z_ref[k]) > DBL_EPSILON*std::abs(z_ref[k]) ||
       std::abs(Hc[k].c - c_ref[k]) > DBL_EPSILON*std::abs(c_ref[k]))
      throw std::runtime_error("BSE JSON: H primitive mismatch.\n");

  // Generally contracted carbon shell: one shell, two contractions
  std::vector<FunctionShell> Csh = lib.get_element("C").get_shells();
  if(Csh.size() != 1 || Csh[0].get_am() != 0 || Csh[0].get_Nctr() != 2)
    throw std::runtime_error("BSE JSON: expected one generally contracted (nctr=2) s shell on C.\n");
  const arma::mat cf = Csh[0].get_coefs();
  const double cf_ref[3][2] = {{0.20,-0.09},{0.55,0.31},{0.35,0.82}};
  if(cf.n_rows != 3 || cf.n_cols != 2)
    throw std::runtime_error("BSE JSON: C coefficient matrix has the wrong shape.\n");
  for(size_t i=0;i<3;i++)
    for(size_t jc=0;jc<2;jc++)
      if(std::abs(cf(i,jc)-cf_ref[i][jc]) > DBL_EPSILON*std::abs(cf_ref[i][jc]))
        throw std::runtime_error("BSE JSON: C generally contracted coefficient mismatch.\n");

  // Dispatch path agrees
  if(lib_dispatch.get_element("C").get_shells().at(0).get_Nctr() != 2)
    throw std::runtime_error("BSE JSON: load_basis dispatch lost the general contraction.\n");

  // Round-trip: save and reload, and require the generally contracted
  // carbon shell to come back bit-for-bit (writer uses %.17g).
  const std::string rt = "bse_test_roundtrip.json";
  lib.save_bse_json(rt);
  BasisSetLibrary lib_rt;
  lib_rt.load_bse_json(rt, false);
  remove(rt.c_str());
  const arma::mat cf_rt = lib_rt.get_element("C").get_shells().at(0).get_coefs();
  if(cf_rt.n_rows != cf.n_rows || cf_rt.n_cols != cf.n_cols)
    throw std::runtime_error("BSE JSON: round-trip changed the C shell shape.\n");
  for(size_t i=0;i<cf.n_rows;i++)
    for(size_t jc=0;jc<cf.n_cols;jc++)
      if(cf_rt(i,jc) != cf(i,jc))
        throw std::runtime_error("BSE JSON: generally contracted round-trip is not bit-exact.\n");

  printf("BSE JSON reader OK.\n");
}

/// A BSE JSON basis that carries an effective core potential on an
/// element must be flagged, and construct_basis must refuse it -- but
/// only when that element is actually used (ERKALE is all-electron).
void test_bse_json_ecp() {
  // Hydrogen with a plain s shell; sodium with a valence s shell *and*
  // an ecp_potentials block (as a def2-style ECP set would carry).
  const std::string json_str = R"JSON({
  "name": "ecp-test",
  "elements": {
    "1": {
      "electron_shells": [
        {"angular_momentum":[0],"exponents":["1.0"],"coefficients":[["1.0"]]}
      ]
    },
    "11": {
      "electron_shells": [
        {"angular_momentum":[0],"exponents":["0.5"],"coefficients":[["1.0"]]}
      ],
      "ecp_electrons": 10,
      "ecp_potentials": [
        {"angular_momentum":[0],"r_exponents":[2],
         "gaussian_exponents":["1.0"],"coefficients":[["1.0"]]}
      ]
    }
  }
})JSON";

  const std::string tmpfile = "bse_test_ecp.json";
  {
    std::ofstream of(tmpfile);
    of << json_str;
  }
  BasisSetLibrary lib;
  lib.load_bse_json(tmpfile, false);
  remove(tmpfile.c_str());

  // The ECP is detected on Na, and not on H
  if(lib.get_element("H").has_ecp())
    throw std::runtime_error("BSE JSON: H wrongly flagged as carrying an ECP.\n");
  if(!lib.get_element("Na").has_ecp())
    throw std::runtime_error("BSE JSON: the Na effective core potential was not detected.\n");

  // A molecule that does not use the ECP element builds fine
  {
    std::vector<atom_t> atoms(1);
    atoms[0].el="H"; atoms[0].num=0; atoms[0].x=atoms[0].y=atoms[0].z=0.0; atoms[0].Q=0;
    BasisSet bas;
    construct_basis(bas, atoms, lib); // must not throw
  }

  // ...but a molecule that uses the ECP element must be refused
  {
    std::vector<atom_t> atoms(1);
    atoms[0].el="Na"; atoms[0].num=0; atoms[0].x=atoms[0].y=atoms[0].z=0.0; atoms[0].Q=0;
    BasisSet bas;
    bool threw=false;
    try {
      construct_basis(bas, atoms, lib);
    } catch(std::runtime_error &) {
      threw=true;
    }
    if(!threw)
      throw std::runtime_error("BSE JSON: construct_basis accepted an element carrying an ECP.\n");
  }

  printf("BSE JSON ECP rejection OK.\n");
}

void check_spherical_order() {
  // The spherical functions ERKALE evaluates itself (on the DFT grid,
  // in the reference integrals) come from GaussianShell::transmat,
  // whereas libcint evaluates the integrals. The two must agree on the
  // spherical functions and their order for every l: the libcint
  // overlap in the spherical basis must equal the one transformed from
  // the cartesian basis with transmat.
  std::vector<contr_t> c(3);
  const double z[3]={3.0, 0.9, 0.25};
  const double a[3]={0.30, 0.50, 0.40};
  for(int i=0;i<3;i++) {
    c[i].z=z[i];
    c[i].c=a[i];
  }

  BasisSet bsph, bcart;
  for(size_t inuc=0;inuc<2;inuc++) {
    nucleus_t nuc;
    nuc.ind=inuc;
    // A general geometry, so that no symmetry hides a wrong order
    nuc.r.x=0.3*inuc; nuc.r.y=-0.7*inuc; nuc.r.z=1.1*inuc;
    nuc.bsse=false;
    nuc.symbol="C";
    nuc.Z=6;
    nuc.Q=0;
    bsph.add_nucleus(nuc);
    bcart.add_nucleus(nuc);
    for(int am=0;am<=4;am++) {
      bsph.add_shell(inuc, am, true, c, false);
      bcart.add_shell(inuc, am, false, c, false);
    }
  }
  bsph.finalize();
  bcart.finalize();

  // Block-diagonal transformation from the cartesian to the spherical basis
  arma::mat T(bsph.Nbf(), bcart.Nbf(), arma::fill::zeros);
  for(size_t is=0;is<bsph.Nshells();is++) {
    // transmat acts on the bare monomials, whereas the cartesian basis
    // functions carry their own normalization factors
    const std::vector<shellf_t> cart(bcart.shells()[is].cart());
    arma::vec rn(cart.size());
    for(size_t ic=0;ic<cart.size();ic++)
      rn(ic)=cart[ic].relnorm;
    T.submat(bsph.first_ind(is), bcart.first_ind(is), bsph.first_ind(is)+bsph.Nbf(is)-1, bcart.first_ind(is)+bcart.Nbf(is)-1)=bsph.shells()[is].transmat()*arma::diagmat(1.0/rn);
  }

  const arma::mat Ssph(bsph.overlap());
  arma::mat Stra(T*bcart.overlap()*T.t());
  // transmat carries the solid-harmonic prefactors, and the functions
  // are normalized afterwards; compare the normalized overlaps
  const arma::vec n(1.0/arma::sqrt(arma::diagvec(Stra)));
  Stra=arma::diagmat(n)*Stra*arma::diagmat(n);
  const double d=arma::abs(Ssph-Stra).max();
  if(d>1e-10) {
    std::ostringstream oss;
    oss << "check_spherical_order: the spherical overlap from libcint and from transmat differ by " << d << ".\n";
    throw std::runtime_error(oss.str());
  }
}

void check_m_values() {
  // m labels for linear symmetry. Two nuclei on the z axis carry
  // generally contracted s, p and d shells, with cartesian s and p
  // functions (the OptLM default) as well as spherical ones. The labels
  // are checked against the functions themselves: in a linear molecule,
  // functions with different m do not overlap.
  BasisSet basis;
  for(size_t inuc=0;inuc<2;inuc++) {
    nucleus_t nuc;
    nuc.ind=inuc;
    nuc.r.x=0.0; nuc.r.y=0.0; nuc.r.z=2.1*inuc;
    nuc.bsse=false;
    nuc.symbol="N";
    nuc.Z=7;
    nuc.Q=0;
    basis.add_nucleus(nuc);
  }
  // Two contractions over the same exponents form one generally
  // contracted shell
  const double z[3]={4.0, 1.1, 0.35};
  const double a[2][3]={{0.20, 0.55, 0.35}, {-0.09, 0.31, 0.82}};
  for(size_t inuc=0;inuc<2;inuc++)
    for(int am=0;am<=2;am++)
      for(int lm=0;lm<=1;lm++) {
        // Cartesian functions only carry m for s and p
        if(!lm && am>1)
          continue;
        for(int ic=0;ic<2;ic++) {
          std::vector<contr_t> c(3);
          for(int i=0;i<3;i++) {
            c[i].z=z[i];
            c[i].c=a[ic][i];
          }
          basis.add_shell(inuc, am, lm, c, false);
        }
      }
  basis.finalize();

  const arma::ivec m(basis.m_values());
  const arma::mat S(basis.overlap());
  if(m.n_elem != S.n_rows)
    throw std::runtime_error("check_m_values: wrong number of m values.\n");
  if(arma::abs(m).max() > 2)
    throw std::runtime_error("check_m_values: m value outside the angular momenta of the basis.\n");
  for(size_t i=0;i<S.n_rows;i++)
    for(size_t j=0;j<S.n_cols;j++)
      if(m(i)!=m(j) && std::abs(S(i,j))>1e-10) {
        std::ostringstream oss;
        oss << "check_m_values: functions " << i << " (m=" << m(i) << ") and " << j << " (m=" << m(j) << ") overlap by " << S(i,j) << ".\n";
        throw std::runtime_error(oss.str());
      }
}

void check_becke_weight_derivative() {
  // The analytic nuclear derivative of the Becke-Stratmann quadrature
  // weights, against central differences: the quadrature of a smooth
  // function attached to the points, which ride on their parent atoms,
  // changes only through the weights.
  BasisSet basis;
  const double r[4][3]={{0.0, 0.0, 0.0}, {1.43, 1.10, 0.10}, {-1.40, 1.02, -0.20}, {0.3, -1.9, 1.2}};
  const int Z[4]={8, 1, 1, 7};
  for(size_t inuc=0;inuc<4;inuc++) {
    nucleus_t nuc;
    nuc.ind=inuc;
    nuc.r.x=r[inuc][0]; nuc.r.y=r[inuc][1]; nuc.r.z=r[inuc][2];
    nuc.bsse=false;
    nuc.symbol=element_symbols[Z[inuc]];
    nuc.Z=Z[inuc];
    nuc.Q=0;
    basis.add_nucleus(nuc);
    std::vector<contr_t> c(1);
    c[0].z=1.0;
    c[0].c=1.0;
    basis.add_shell(inuc, 0, true, c, false);
  }
  basis.finalize();
  const arma::mat R0(basis.nuclear_coords());

  // Smooth function of the position of the point relative to its atom
  auto h = [](const arma::vec & d) {
    return std::exp(-0.3*arma::dot(d,d))*(1.0 + 0.4*d(0) - 0.2*d(1)*d(2));
  };
  // Quadrature of h on a shell of atom A, and its weight derivative
  auto shell = [&](const BasisSet & bas, size_t A, double rad, arma::vec & dQ) {
    angshell_t sh;
    sh.atind=A;
    sh.cen=bas.nuclear_coords(A);
    sh.R=rad;
    sh.w=1.0;
    sh.l=17;
    sh.tol=0.0;
    sh.np=0;
    sh.nfunc=0;
    AngularGrid grid;
    grid.basis(bas);
    grid.set_shell(sh);
    grid.form_grid();
    const std::vector<gridpoint_t> pts(grid.grid());
    arma::vec hv(pts.size());
    double Q=0.0;
    for(size_t ip=0;ip<pts.size();ip++) {
      hv(ip)=h(coords_to_vec(pts[ip].r-sh.cen));
      Q+=pts[ip].w_*hv(ip);
    }
    dQ=grid.becke_weight_derivative()*hv;
    return Q;
  };

  const double step=1e-4;
  double maxd=0.0, maxref=0.0;
  for(size_t A=0;A<4;A++)
    for(double rad : {0.4, 1.0, 1.7, 2.8}) {
      arma::vec dQ, dum;
      shell(basis, A, rad, dQ);
      arma::vec fd(dQ.n_elem);
      for(size_t i=0;i<dQ.n_elem;i++) {
        // Richardson-extrapolated central difference
        auto Q = [&](double t) {
          arma::mat Rt(R0);
          Rt(i/3, i%3)+=t;
          BasisSet bas(basis);
          bas.set_nuclear_coords(Rt);
          return shell(bas, A, rad, dum);
        };
        const double d1=(Q(step)-Q(-step))/(2*step), d2=(Q(0.5*step)-Q(-0.5*step))/step;
        fd(i)=(4.0*d2-d1)/3.0;
      }
      maxd=std::max(maxd, arma::abs(dQ-fd).max());
      maxref=std::max(maxref, arma::abs(fd).max());
    }
  printf("Becke weight derivative: max deviation %.1e (max derivative %.1e)\n", maxd, maxref);
  fflush(stdout);
  if(maxd > 1e-8*maxref)
    throw std::runtime_error("check_becke_weight_derivative: analytic and finite-difference derivatives differ.\n");
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
  // Spherical functions: transmat against libcint
  check_spherical_order();
  printf("Spherical harmonic order OK.\n");
  // m labels of the functions for linear symmetry
  check_m_values();
  printf("Linear-symmetry m values OK.\n");
  // Nuclear derivative of the quadrature weights
  check_becke_weight_derivative();
  printf("Becke weight derivative OK.\n");
  // BSE JSON basis-set reader / writer
  test_bse_json();
  // BSE JSON effective-core-potential rejection
  test_bse_json_ecp();
  // Test lapack thread safety
  try {
    check_lapack_thread();
  } catch(std::runtime_error &) {
    throw std::runtime_error("LAPACK library is not thread safe!\nThis might cause problems in some parts of ERKALE.\n");
  }
}
