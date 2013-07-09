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

#include "global.h"
#include "basis.h"
#include "checkpoint.h"
#include "mathf.h"
#include "stringutil.h"
#include "timer.h"
#include "linalg.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

void size_distribution(const BasisSet & basis, arma::mat & C, std::string filename, const std::vector<size_t> & printidx) {
  // Get the r_i^2 r_j^2 matrices
  std::vector<arma::mat> momstack=basis.moment(4);
  // Diagonal: x^4 + y^4 + z^4
  arma::mat rfour=momstack[getind(4,0,0)] + momstack[getind(0,4,0)] + momstack[getind(0,0,4)] \
    // Off-diagonal: 2 x^2 y^2 + 2 x^2 z^2 + 2 y^2 z^2
    +2.0*(momstack[getind(2,2,0)]+momstack[getind(2,0,2)]+momstack[getind(0,2,2)]);

  // Get R^3 matrices
  momstack=basis.moment(3);
  std::vector<arma::mat> rrsq(3);
  // x^3 + xy^2 + xz^2
  rrsq[0]=momstack[getind(3,0,0)]+momstack[getind(1,2,0)]+momstack[getind(1,0,2)];
  // x^2y + y^3 + yz^2
  rrsq[1]=momstack[getind(2,1,0)]+momstack[getind(0,3,0)]+momstack[getind(0,1,2)];
  // x^2z + y^2z + z^3
  rrsq[2]=momstack[getind(2,0,1)]+momstack[getind(0,2,1)]+momstack[getind(0,0,3)];

  // Get R^2 matrix
  momstack=basis.moment(2);
  std::vector< std::vector<arma::mat> > rr(3);
  for(int ic=0;ic<3;ic++)
    rr[ic].resize(3);

  // Diagonal
  rr[0][0]=momstack[getind(2,0,0)];
  rr[1][1]=momstack[getind(0,2,0)];
  rr[2][2]=momstack[getind(0,0,2)];

  // Off-diagonal
  rr[0][1]=momstack[getind(1,1,0)];
  rr[1][0]=rr[0][1];

  rr[0][2]=momstack[getind(1,0,1)];
  rr[2][0]=rr[0][2];

  rr[1][2]=momstack[getind(0,1,1)];
  rr[2][1]=rr[1][2];

  // and the rsq matrix
  arma::mat rsq=rr[0][0]+rr[1][1]+rr[2][2];

  // Get r matrices
  std::vector<arma::mat> rmat=basis.moment(1);

  // Output file
  FILE *out=fopen(filename.c_str(),"w");
  for(size_t i=0;i<printidx.size();i++) {
    // Orbital index is
    size_t iorb=printidx[i];
   
    // r^4 term
    double rfour_t=arma::as_scalar(arma::trans(C.col(iorb))*rfour*C.col(iorb));

    // rr^2 term
    arma::vec rrsq_t(3);
    for(int ic=0;ic<3;ic++)
      rrsq_t(ic)=arma::as_scalar(arma::trans(C.col(iorb))*rrsq[ic]*C.col(iorb));

    // rr terms
    arma::mat rr_t(3,3);
    for(int ic=0;ic<3;ic++)
      for(int jc=0;jc<=ic;jc++) {
	rr_t(ic,jc)=arma::as_scalar(arma::trans(C.col(iorb))*rr[ic][jc]*C.col(iorb));
	rr_t(jc,ic)=rr_t(ic,jc);
      }

    // <r^2> term
    double rsq_t=arma::as_scalar(arma::trans(C.col(iorb))*rsq*C.col(iorb));

    // <r> terms
    arma::vec r_t(3);
    for(int ic=0;ic<3;ic++)
      r_t(ic)=arma::as_scalar(arma::trans(C.col(iorb))*rmat[ic]*C.col(iorb));

    // Second moment is
    double SM=sqrt(rsq_t - arma::dot(r_t,r_t));
    // Fourth moment is
    double FM=sqrt(sqrt(rfour_t - 4.0*arma::dot(rrsq_t,r_t) + 2.0*rsq_t*arma::dot(r_t,r_t) + 4.0 * arma::as_scalar(arma::trans(r_t)*rr_t*r_t) - 3.0*std::pow(arma::dot(r_t,r_t),2)));

    // Print
    fprintf(out,"%i %e %e\n",(int) iorb+1,SM,FM);
  }
  fclose(out);
}

void localize_wrk(const BasisSet & basis, arma::mat & C, arma::vec & E, const std::vector<double> & occs, enum locmet method, enum unitmethod umet, enum unitacc acc, bool randomize, bool delocalize, std::string sizedist, bool size) {
  // Orbitals to localize
  std::vector<size_t> locorb;
  for(size_t io=0;io<occs.size();io++)
    if(occs[io]!=0.0)
      locorb.push_back(io);

  // Save indices
  std::vector<size_t> printidx(locorb);
    
  // Loop over orbitals
  while(locorb.size()) {
    // Orbitals to treat in this run
    std::vector<size_t> orbidx;
    
    // Occupation number
    double occno=occs[locorb[0]];
    
    for(size_t io=locorb.size()-1;io<locorb.size();io--)
      // Degeneracy in occupation?
      if(fabs(occs[locorb[io]]-occno)<1e-6) {
	// Orbitals are degenerate; add to current batch
	  orbidx.push_back(locorb[io]);
	  locorb.erase(locorb.begin()+io);
      }
    
    std::sort(orbidx.begin(),orbidx.end());
    
    // Collect orbitals
    arma::mat Cwrk(C.n_rows,orbidx.size());
    for(size_t io=0;io<orbidx.size();io++)
      Cwrk.col(io)=C.col(orbidx[io]);
    // and orbital energies
    arma::vec Ewrk(orbidx.size());
    for(size_t io=0;io<orbidx.size();io++)
      Ewrk(io)=E(orbidx[io]);    

    // Localizing matrix
    arma::cx_mat U;
    if(randomize)
      U=std::complex<double>(1.0,0.0)*real_orthogonal(orbidx.size(),orbidx.size());
    else
      U.eye(orbidx.size(),orbidx.size());
    double measure=1e-7;
    
    // Run localization
    if(delocalize)
      printf("Delocalizing orbitals:");
    else
      printf("Localizing   orbitals:");
    for(size_t io=0;io<orbidx.size();io++)
      printf(" %i",(int) orbidx[io]+1);
    printf("\n");

    orbital_localization(method,basis,Cwrk,measure,U,true,umet,acc,delocalize);
    
    // Transform orbitals
    arma::mat Cloc=arma::real(Cwrk*U);
    // and energies
    arma::vec Eloc=arma::real(arma::diagvec(arma::trans(U)*arma::diagmat(Ewrk)*U));
    // and sort them in the new energy order
    sort_eigvec(Eloc,Cloc);
    
    // Update orbitals and energies
    for(size_t io=0;io<orbidx.size();io++)
      C.col(orbidx[io])=Cloc.col(io);
    for(size_t io=0;io<orbidx.size();io++)
      E(orbidx[io])=Eloc(io);
  }

  // Compute size distribution
  if(size)
    size_distribution(basis,C,sizedist,printidx);
}


void localize(const BasisSet & basis, arma::mat & C, arma::vec & E, std::vector<double> occs, bool virt, enum locmet method, enum unitmethod umet, enum unitacc acc, bool randomize, bool delocalize, std::string sizedist, bool size) {
  // Run localization, occupied space
  localize_wrk(basis,C,E,occs,method,umet,acc,randomize,delocalize,sizedist+".o",size);

  // Run localization, virtual space
  if(virt) {
    for(size_t i=0;i<occs.size();i++)
      if(occs[i]==0.0)
	occs[i]=1.0;
      else
	occs[i]=0.0;
    localize_wrk(basis,C,E,occs,method,umet,acc,randomize,delocalize,sizedist+".v",size);
  }
}

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Localization from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Localization from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  // Initialize libint
  init_libint_base();

  Settings set;
  set.add_string("LoadChk","Checkpoint to load","erkale.chk");
  set.add_string("SaveChk","Checkpoint to save results to","erkale.chk");
  set.add_string("Method","Localization method: FB, FB2, FB3, FB4, FM, FM2, FM3, FM4, MU, LO, BE, HI, ER","FB");
  set.add_bool("Virtual","Localize virtual orbitals as well?",false);
  set.add_string("Logfile","File to store standard output in","stdout");
  set.add_string("Accelerator","Accelerator to use: SDSA, CGPR, CGFR","CGPR");
  set.add_string("LineSearch","Line search to use: poly_df, poly_fdf, armijo, fourier_df","poly_df");
  set.add_bool("Randomize","Use random starting point instead of canonical orbitals?",true);
  set.add_bool("Delocalize","Run delocalization instead of localization",false);
  set.add_string("SizeDistribution","File to save orbital size distribution in","");
  set.parse(argv[1]);
  set.print();

  // Redirect output?
  std::string logfile=set.get_string("Logfile");
  if(stricmp(logfile,"stdout")!=0) {
    // Redirect stdout to file
    FILE *outstream=freopen(logfile.c_str(),"w",stdout);
    if(outstream==NULL) {
      ERROR_INFO();
      throw std::runtime_error("Unable to redirect output!\n");
    } else
      fprintf(stderr,"\n");
  }

  bool virt=set.get_bool("Virtual");

  std::string loadname(set.get_string("LoadChk"));
  std::string savename(set.get_string("SaveChk"));
  std::string sizedist(set.get_string("SizeDistribution"));
  bool size=stricmp(sizedist,"");

  // Determine method
  enum locmet method;
  std::string mets=set.get_string("Method");
  if(stricmp(mets,"FB")==0)
    method=BOYS;
  else if(stricmp(mets,"FB2")==0)
    method=BOYS_2;
  else if(stricmp(mets,"FB3")==0)
    method=BOYS_3;
  else if(stricmp(mets,"FB4")==0)
    method=BOYS_4;
  else if(stricmp(mets,"FM")==0)
    method=FM_1;
  else if(stricmp(mets,"FM2")==0)
    method=FM_2;
  else if(stricmp(mets,"FM3")==0)
    method=FM_3;
  else if(stricmp(mets,"FM4")==0)
    method=FM_4;
  else if(stricmp(mets,"MU")==0)
    method=PIPEK_MULLIKEN;
  else if(stricmp(mets,"LO")==0)
    method=PIPEK_LOWDIN;
  else if(stricmp(mets,"BE")==0)
    method=PIPEK_BECKE;
  else if(stricmp(mets,"HI")==0)
    method=PIPEK_HIRSHFELD;
  else if(stricmp(mets,"ER")==0)
    method=EDMINSTON;
  else throw std::runtime_error("Localization method not implemented.\n");

  // Determine accelerator
  enum unitacc acc;
  std::string accs=set.get_string("Accelerator");
  if(stricmp(accs,"SDSA")==0)
    acc=SDSA;
  else if(stricmp(accs,"CGPR")==0)
    acc=CGPR;
  else if(stricmp(accs,"CGFR")==0)
    acc=CGFR;
  else throw std::runtime_error("Accelerator not implemented.\n");

  // Determine line search
  enum unitmethod umet;
  std::string umets=set.get_string("LineSearch");
  if(stricmp(umets,"poly_df")==0)
    umet=POLY_DF;
  else if(stricmp(umets,"poly_fdf")==0)
    umet=POLY_FDF;
  else if(stricmp(umets,"armijo")==0)
    umet=ARMIJO;
  else if(stricmp(umets,"fourier_df")==0)
    umet=FOURIER_DF;
  else throw std::runtime_error("Accelerator not implemented.\n");
  
  if(stricmp(loadname,savename)!=0) {
    // Copy checkpoint
    std::ostringstream oss;
    oss << "cp " << loadname << " " << savename;
    int cp=system(oss.str().c_str());
    if(cp) {
      ERROR_INFO();
      throw std::runtime_error("Failed to copy checkpoint file.\n");
    }
  }

  // Use randomized starting point?
  bool randomize=set.get_bool("Randomize");
  bool delocalize=set.get_bool("Delocalize");

  // Open checkpoint in read-write mode, don't truncate
  Checkpoint chkpt(savename,true,false);
    
  // Basis set
  BasisSet basis;
  chkpt.read(basis);

  // Restricted run?
  bool restr;
  chkpt.read("Restricted",restr);

  if(restr) {
    // Orbitals
    arma::mat C;
    chkpt.read("C",C);
    // and energies
    arma::vec E;
    chkpt.read("E",E);

    // Check orthogonality
    check_orth(C,basis.overlap(),false);

    // Occupation numbers
    std::vector<double> occs;
    chkpt.read("occs",occs);

    // Run localization
    localize(basis,C,E,occs,virt,method,umet,acc,randomize,delocalize,sizedist,size);

    chkpt.write("C",C);
    chkpt.write("E",E);

  } else {
    // Orbitals
    arma::mat Ca, Cb;
    chkpt.read("Ca",Ca);
    chkpt.read("Cb",Cb);
    // and energies
    arma::vec Ea, Eb;
    chkpt.read("Ea",Ea);
    chkpt.read("Eb",Eb);

    // Check orthogonality
    check_orth(Ca,basis.overlap(),false);
    check_orth(Cb,basis.overlap(),false);

    // Occupation numbers
    std::vector<double> occa, occb;
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);

    // Run localization
    localize(basis,Ca,Ea,occa,virt,method,umet,acc,randomize,delocalize,sizedist+".a",size);
    localize(basis,Cb,Eb,occb,virt,method,umet,acc,randomize,delocalize,sizedist+".b",size);

    chkpt.write("Ca",Ca);
    chkpt.write("Cb",Cb);
    chkpt.write("Ea",Ea);
    chkpt.write("Eb",Eb);
  }

  return 0;
}