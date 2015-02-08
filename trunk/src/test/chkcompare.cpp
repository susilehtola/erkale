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

#include "solidharmonics.h"
#include "checkpoint.h"
#include "mathf.h"
#include "../emd/emd_gto.h"

/// Absolute tolerance in total energy
const double Etol=1e-6;
/// Absolute tolerance for density matrix difference
const double dPtol=1e-7;
/// Absolute olerance for orbital matrix difference
const double dCtol=1e-7;
/// Relative tolerance in orbital energies
const double dEtol=1e-5;

/// Absolute tolerance for normalization of basis functions
const double normtol=1e-10;

/// Tolerance for the EMD error estimate
const double P2TOL=50.0;

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
  char *tmpfile=tempnam("./",".chk");

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

  remove(tmpfile);
  free(tmpfile);
}


/// Test checkpoints
void test_checkpoint_basis(const BasisSet & bas) {
  // Temporary file name
  char *tmpfile=tempnam("./",".chk");

  {
    // Dummy checkpoint
    Checkpoint chkpt(tmpfile,true);
    
    // Write basis
    chkpt.write(bas);

    // Read basis
    BasisSet loadbas;
    chkpt.read(loadbas);

    // Get overlap matrices
    arma::mat S=bas.overlap();
    arma::mat Sload=loadbas.overlap();

    double matnorm=rms_norm(S-Sload);
    if(matnorm>DBL_EPSILON) {
      printf("Basis set read-write error %e.\n",matnorm);
      ERROR_INFO();
      throw std::runtime_error("Error in basis set read/write.\n");
    }
  }

  remove(tmpfile);
  free(tmpfile);
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

  // Test indices
  testind();
  // Then, check norms of spherical harmonics.
  check_sph_orthonorm(Lmax);
  printf("Solid harmonics OK.\n");
  // Then, check checkpoint utilities
  test_checkpoint();
  printf("Checkpointing OK.\n");
  
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
  // and checkpoints
  test_checkpoint_basis(bref);
  test_checkpoint_basis(bcur);

  // Overlap matrix
  arma::mat S(bcur.overlap());

  if(rrestr) {
    int Nelr, Nelc;
    ref.read("Nel",Nelr);
    cur.read("Nel",Nelc);
    if(Nelr != Nelc)
      throw std::runtime_error("Amount of electrons doesn't match!\n");

   // Total energies
    energy_t Er, Ec;
    ref.read(Er);
    cur.read(Ec);
    double dE=Ec.E-Er.E;
    printf("Total energy difference %e\n",dE);
    fflush(stdout);
    if(fabs(dE) > Etol)
      throw std::runtime_error("Total energies don't match!\n");
    
    // Densities
    arma::mat Pref, Pcur;
    ref.read("P",Pref);
    cur.read("P",Pcur);

    // Check norm
    double Nelnum=arma::trace(Pcur*S);
    double dNel=Nelnum-Nelc;
    printf("Electron count difference %e\n",dNel);
    fflush(stdout);
    if(fabs(dNel)>dPtol)
      throw std::runtime_error("Norm of density matrix is wrong.\n");
    
    // Check difference
    double dP=rms_norm(Pcur-Pref);
    printf("Density matrix difference %e\n",dP);
    if(dP>dPtol)
      throw std::runtime_error("Density matrices differ!\n");
    
    // Orbitals    
    arma::mat Cref, Ccur;
    ref.read("C",Cref);
    cur.read("C",Ccur);
    if(Cref.n_cols != Ccur.n_cols)
      throw std::runtime_error("Amount of orbitals doesn't match!\n");

    // Fix orbital phase signs
    fix_signs(Ccur,Cref,S);

    // Calculate difference
    double dC=rms_norm(Cref-Ccur);
    printf("Orbital matrix difference %e\n",dC);
    fflush(stdout);
    if(dC>dCtol) {
      throw std::runtime_error("Orbital coefficients differ!\n");
    }

    // Orbital energies
    arma::vec Eref, Ecur;
    ref.read("E",Eref);
    cur.read("E",Ecur);
    double dEo=arma::max(arma::abs((Ecur-Eref)/Eref));
    printf("Orbital energy difference %e\n",dEo);
    fflush(stdout);
    if(dEo > dEtol)
      throw std::runtime_error("Orbital energies differ!\n");

    // Check EMD
    GaussianEMDEvaluator eval(bcur,Pcur);
    EMD emd(&eval, &eval, Nelc, 0, 0);
    emd.initial_fill(false);
    emd.find_electrons(false);
    emd.optimize_moments(false);
    
    // Get moments
    arma::mat mom=emd.moments();
    
    // Compare <p^2> with T and <p^0> with tr(P*S)
    double p0diff=mom(2,1)-arma::trace(Pcur*S);
    double p0err=mom(2,2);
    printf("<p^0> - N = % e, d<p^0> = %e\n",p0diff,p0err);
    double p2diff=mom(4,1)-2.0*Ec.Ekin;
    double p2err=mom(4,2);
    printf("<p^2> - T = % e, d<p^2> = %e\n",p2diff,p2err);
    fflush(stdout);
    if(fabs(p0diff)>=2*p0err || fabs(p2diff)>P2TOL*p2err)
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
    if(fabs(dE) > Etol)
      throw std::runtime_error("Total energies don't match!\n");
    
    // Densities
    arma::mat Paref, Pbref, Pref;
    arma::mat Pacur, Pbcur, Pcur;
    ref.read("Pa",Paref);
    ref.read("Pb",Pbref);
    ref.read("P",Pref);
    cur.read("Pa",Pacur);
    cur.read("Pb",Pbcur);
    cur.read("P",Pcur);

    // Check norm
    double Nelanum=arma::trace(Pacur*S);
    double Nelbnum=arma::trace(Pbcur*S);
    double Nelnum=arma::trace(Pcur*S);

    // XRS calculation?
    bool xrs, xrsspin;
    std::string xrsmethod;
    try {
      cur.read("XRSMethod",xrsmethod);
      cur.read("XRSSpin",xrsspin);
      xrs=true;
    } catch(std::runtime_error) {
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
    if(fabs(dNela)>dPtol)
      throw std::runtime_error("Norm of alpha density matrix is wrong.\n");

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
    if(fabs(dNelb)>dPtol)
      throw std::runtime_error("Norm of beta  density matrix is wrong.\n");

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
    if(dPa>dPtol)
      throw std::runtime_error("Alpha density matrices differ!\n");
    double dPb=rms_norm(Pbcur-Pbref);
    printf("Beta  density matrix difference %e\n",dPb);
    if(dPb>dPtol)
      throw std::runtime_error("Density matrices differ!\n");
    double dP=rms_norm(Pcur-Pref);
    printf("Total density matrix difference %e\n",dP);
    if(dP>dPtol)
      throw std::runtime_error("Total density matrices differ!\n");

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
    
    // Check differences
    double dCa=rms_norm(Cacur-Caref);
    printf("Alpha orbital matrix difference %e\n",dCa);
    if(dCa>dCtol)
      throw std::runtime_error("Alpha orbital matrices differ!\n");
    double dCb=rms_norm(Cbcur-Cbref);
    printf("Beta  orbital matrix difference %e\n",dCb);
    if(dCb>dCtol)
      throw std::runtime_error("Beta  orbital matrices differ!\n");
    
    // Orbital energies
    arma::vec Earef, Ebref, Eacur, Ebcur;
    ref.read("Ea",Earef);
    ref.read("Eb",Ebref);
    cur.read("Ea",Eacur);
    cur.read("Eb",Ebcur);

    double dEoa=arma::max(arma::abs((Eacur-Earef)/Earef));
    printf("Alpha orbital energy difference %e\n",dEoa);
    if(dEoa > dEtol)
      throw std::runtime_error("Alpha orbital energies differ!\n");
    double dEob=arma::max(arma::abs((Ebcur-Ebref)/Ebref));
    printf("Beta  orbital energy difference %e\n",dEob);
    if(dEob > dEtol)
      throw std::runtime_error("Beta orbital energies differ!\n");

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
    double p0diff=mom(2,1)-arma::trace(Pcur*S);
    double p0err=mom(2,2);
    printf("<p^0> - N = %e, d<p^0> = %e\n",p0diff,p0err);
    double p2diff=mom(4,1)-2.0*Ec.Ekin;
    double p2err=mom(4,2);
    printf("<p^2> - T = %e, d<p^2> = %e\n",p2diff,p2err);
    if(fabs(p0diff)>=2*p0err || fabs(p2diff)>P2TOL*p2err)
      throw std::runtime_error("EMD failed.\n");
  }
}
