/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "../../emd/emd_sto.h"
#include "../../mathf.h"
#include "../../stringutil.h"
#include "../../settings.h"
#include "../../lmgrid.h"
#include "../../timer.h"
#include "../storage.h"

#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../../version.h"
#endif

extern "C" {
  /* ADF's keyed-file routines */
#include "KFc.h"
}

// The kf file
KFFile kf;

void open_kf() {
  if (openKFFile(&kf, "TAPE21") < 0)
    throw std::runtime_error("Error opening TAPE21.\n");
}

void close_kf() {
  closeKFFile (&kf);
}

std::vector<int> get_int_vec_kf(const std::string & name) {
  // Figure out the number of entries
  int N=getKFVariableLength(&kf,name.c_str());
  if(N<=0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Entry "<< name <<" in KF file has " << N << " elements!\n";;
    throw std::runtime_error(oss.str());
  }

  // Allocate memory
  std::vector<int> ret(N);

  // Read the data
  if(getKFData(&kf, name.c_str(), (void *) &ret[0]) < 0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Error reading "<< name <<" entry from KF file!";
    throw std::runtime_error(oss.str());
  }

  return ret;
}

std::vector<double> get_double_vec_kf(const std::string & name) {
  // Figure out the number of entries
  int N=getKFVariableLength(&kf,name.c_str());
  if(N<=0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Entry "<< name <<" in KF file has " << N << " elements!\n";;
    throw std::runtime_error(oss.str());
  }

  // Allocate memory
  std::vector<double> ret(N);

  // Read the data
  if(getKFData(&kf, name.c_str(), (void *) &ret[0]) < 0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Error reading "<< name <<" entry from KF file!";
    throw std::runtime_error(oss.str());
  }

  return ret;
}

std::vector< std::vector<size_t> > find_identical_functions() {
  // Get atom type list
  std::vector<int> nbptr=get_int_vec_kf("Basis%nbptr");
  // Get number of atoms of each type
  std::vector<int> nqptr=get_int_vec_kf("Geometry%nqptr");

  // The returned list is
  std::vector< std::vector<size_t> > ret;

  // Index of current function
  size_t ind=0;

  // Loop over atom types
  for(size_t i=0;i<nbptr.size()-1;i++) {
    // Index of first function on list
    size_t i0=ret.size();

    // Add space to list
    ret.resize(ret.size()+nbptr[i+1]-nbptr[i]);

    // Loop over atoms of current type
    for(int iat=0;iat<nqptr[i+1]-nqptr[i];iat++)
      // Loop over functions on current atom
      for(int ifunc=0;ifunc<nbptr[i+1]-nbptr[i];ifunc++)
	ret[i0+ifunc].push_back(ind++);
  }

  /*
      printf("Identical functions:\n");
      for(size_t ig=0;ig<ret.size();ig++) {
      printf("Group %i:",(int) ig);
      for(size_t i=0;i<ret[ig].size();i++)
      printf(" %i",(int) ret[ig][i]);
      printf("\n");
      }
  */

  return ret;
}

std::vector< std::vector<ylmcoeff_t> > form_clm() {
  // Get the exponents
  std::vector<int> kx=get_int_vec_kf("Basis%kx");
  std::vector<int> ky=get_int_vec_kf("Basis%ky");
  std::vector<int> kz=get_int_vec_kf("Basis%kz");

  // The returned expansion
  std::vector< std::vector<ylmcoeff_t> > ret;

  // Expansions of cartesian functions. ADF is currently limited to
  // f-type, so this is enough.
  CartesianExpansion cart(3);

  // Loop over functions
  for(size_t i=0;i<kx.size();i++) {
    // Get the transform
    SphericalExpansion expn=cart.get(kx[i],ky[i],kz[i]);
    // Get coefficients
    std::vector<ylmcoeff_t> c=expn.getcoeffs();
    // and normalize them
    double n=0.0;
    for(size_t ic=0;ic<c.size();ic++)
      n+=norm(c[ic].c);
    n=sqrt(n);
    for(size_t ic=0;ic<c.size();ic++)
      c[ic].c/=n;

    // and add them to the stack
    ret.push_back(c);
  }

  /*
    for(size_t i=0;i<ret.size();i++) {
    printf("*** Function %3i ***\n",(int) i +1);
    for(size_t j=0;j<ret[i].size();j++)
    printf(" (% e,% e) Y_%i^%+i",ret[i][j].c.real(),ret[i][j].c.imag(),ret[i][j].l,ret[i][j].m);
    printf("\n");
    }
  */

  return ret;
}

std::vector< std::vector<RadialSlater> > form_radial() {
  // Get the exponents
  std::vector<int> kx=get_int_vec_kf("Basis%kx");
  std::vector<int> ky=get_int_vec_kf("Basis%ky");
  std::vector<int> kz=get_int_vec_kf("Basis%kz");
  std::vector<int> kr=get_int_vec_kf("Basis%kr");
  std::vector<double> z=get_double_vec_kf("Basis%alf");

  // Returned functions
  std::vector< std::vector<RadialSlater> > ret(kx.size());

  // Loop over functions
  for(size_t i=0;i<kx.size();i++) {
    // Compute value of angular momentum
    int am=kx[i]+ky[i]+kz[i];

    // Compute value of n
    int n=am+kr[i]+1;

    // Add functions
    for(int l=am;l>=0;l-=2)
      ret[i].push_back(RadialSlater(n,l,z[i]));
  }

  return ret;
}

std::vector<size_t> get_centers() {
  // Index of center
  size_t ind=0;

  // Get atom type list
  std::vector<int> nbptr=get_int_vec_kf("Basis%nbptr");
  // Get number of atoms of each type
  std::vector<int> nqptr=get_int_vec_kf("Geometry%nqptr");

  // Get number of functions
  int nbf=get_int_vec_kf("Symmetry%nfcn")[0];

  // Allocate memory
  std::vector<size_t> ret(nbf);

  // Loop over atom types
  for(size_t i=0;i<nbptr.size()-1;i++)
    // Loop over atoms of current type
    for(int iat=nqptr[i];iat<nqptr[i+1];iat++)
      // Loop over functions on current atom
      for(int ifunc=0;ifunc<nbptr[i+1]-nbptr[i];ifunc++) {
	// Set center of function
	ret[ind++]=iat-1;
      }

  return ret;
}

SlaterEMDEvaluator get_eval(arma::mat & P) {
  // Form radial functions
  std::vector< std::vector<RadialSlater > > radf=form_radial();

  // Form identical functions
  std::vector< std::vector<size_t> > idf=find_identical_functions();

  // Form Ylm expansion of functions
  std::vector< std::vector<ylmcoeff_t> > clm=form_clm();

  // Form list of centers of functions
  std::vector<size_t> loc=get_centers();

  /*
    printf("Functions centered on atoms:\n");
    for(size_t i=0;i<loc.size();i++)
    printf("%i: %i\n",(int) i+1, (int) loc[i]+1);
  */

  // Form the list of atomic coordinates
  std::vector<coords_t> coord;
  std::vector<double> clist=get_double_vec_kf("Geometry%xyz");
  for(size_t i=0;i<clist.size();i+=3) {
    coords_t tmp;
    tmp.x=clist[i];
    tmp.y=clist[i+1];
    tmp.z=clist[i+2];
    coord.push_back(tmp);
  }

  /*
    printf("Coordinates of atoms:\n");
    for(size_t i=0;i<coord.size();i++)
    printf("%3i % f % f % f\n",(int) i+1, coord[i].x, coord[i].y, coord[i].z);
  */

  arma::cx_mat Phlp=P*COMPLEX1;
  return SlaterEMDEvaluator(radf,idf,clm,loc,coord,Phlp);
}

arma::mat form_density() {
  // Get number of orbitals
  int nmo=get_int_vec_kf("Symmetry%norb")[0];

  // Get number of basis functions
  int nbf=get_int_vec_kf("Symmetry%nfcn")[0];

  //  printf("%i functions and %i orbitals.\n",nbf,nmo);

  // Get basis function vector
  std::vector<int> npart=get_int_vec_kf("A%npart");
  // Convert to c++ indexing
  for(size_t i=0;i<npart.size();i++)
    npart[i]--;
  if((int) npart.size() != nbf)
    throw std::runtime_error("Size of npart is incorrect!\n");

  // Returned density matrix
  arma::mat P(nbf,nbf);
  P.zeros();

  // Alpha orbital part.
  // Get MO coefficients
  std::vector<double> cca=get_double_vec_kf("A%Eigen-Bas_A");
  // and occupation numbers
  std::vector<double> occa=get_double_vec_kf("A%froc_A");
  if((int) cca.size() != nbf*nmo)
    throw std::runtime_error("Size of cc is incorrect!\n");


  // Form matrix
  arma::mat C(nbf,nmo);
  for(int fi=0;fi<nbf;fi++)
    for(int io=0;io<nmo;io++) {
      C(npart[fi],io)=cca[io*nbf+fi];
    }

  // Increment density matrix
  for(size_t i=0;i<occa.size();i++)
    P+=occa[i]*C.col(i)*arma::trans(C.col(i));

  // Beta orbital part.
  if(get_int_vec_kf("General%nspin")[0]==2) {
    std::vector<double> ccb=get_double_vec_kf("A%Eigen-Bas_B");
    std::vector<double> occb=get_double_vec_kf("A%froc_B");

    C.zeros();
    for(int fi=0;fi<nbf;fi++)
      for(int io=0;io<nmo;io++)
	C(npart[fi],io)=ccb[io*nbf+fi];

    for(size_t i=0;i<occa.size();i++)
      P+=occb[i]*C.col(i)*arma::trans(C.col(i));
  }

  return P;
}

Settings settings;

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - ADF interface from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - ADF interface from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc!=2) {
    printf("Usage: %s file\n",argv[0]);
    return 1;
  }

  Timer t;
  t.print_time();

  // Open the KF file
  open_kf();

  // Construct EMD evaluator
  arma::mat P=form_density();

  SlaterEMDEvaluator eval=get_eval(P);
  //  eval.print();

  // Get number of electrons
  int Nel=get_double_vec_kf("General%electrons")[0];

  // Close the KF file
  close_kf();

  EMD emd(&eval, &eval, Nel, 0, 0);
  emd.initial_fill();
  emd.find_electrons();
  emd.optimize_moments(true,1e-7);
  emd.save("emd.txt");
  emd.moments("moments.txt");
  emd.compton_profile("compton.txt");
  emd.compton_profile_interp("compton-interp.txt");

  printf("Computing EMD properties took %s.\n",t.elapsed().c_str());

  return 0;
}

int main(int argc, char **argv) {
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
