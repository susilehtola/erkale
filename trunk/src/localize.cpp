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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

enum locmet {
  // Boys
  BOYS,
  // Pipek-Mezey
  PIPEK
};

void localize_wrk(const BasisSet & basis, arma::mat & C, const std::vector<double> & occs, enum locmet method, enum unitmethod umet, enum unitacc acc) {
  // Orbitals to localize
  std::vector<size_t> locorb;
  for(size_t io=0;io<occs.size();io++)
    if(occs[io]!=0.0)
      locorb.push_back(io);
    
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
    
    // Localizing matrix
    arma::cx_mat U(orbidx.size(),orbidx.size());
    U.eye();
    double measure=1e-7;
    
    // Run localization
    printf("Localizing orbitals:");
    for(size_t io=0;io<orbidx.size();io++)
      printf(" %i",(int) orbidx[io]+1);
    printf("\n");
    
    if(method==BOYS)
      boys_localization(basis,Cwrk,measure,U,true,umet,acc);
    else if(method==PIPEK)
      pipek_localization(basis,Cwrk,measure,U,true,umet,acc);
    
    // Transform orbitals
    arma::mat Cloc=arma::real(Cwrk*U);
    
    // Update orbitals
    for(size_t io=0;io<orbidx.size();io++)
      C.col(orbidx[io])=Cloc.col(io);
  }
}


void localize(const BasisSet & basis, arma::mat & C, std::vector<double> occs, bool virt, enum locmet method, enum unitmethod umet, enum unitacc acc) {
  // Run localization, occupied space
  localize_wrk(basis,C,occs,method,umet,acc);

  // Run localization, virtual space
  if(virt) {
    for(size_t i=0;i<occs.size();i++)
      if(occs[i]==0.0)
	occs[i]=1.0;
      else
	occs[i]=0.0;
    localize_wrk(basis,C,occs,method,umet,acc);
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

  Settings set;
  set.add_string("LoadChk","Checkpoint to load","erkale.chk");
  set.add_string("SaveChk","Checkpoint to save results to","erkale.chk");
  set.add_string("Localization","Localization method: BF, PM, ER","BF");
  set.add_bool("Virtual","Localize virtual orbitals as well?",false);
  set.add_string("Logfile","File to store standard output in","erkale_loc.log");
  set.add_string("Accelerator","Accelerator to use: SD, CGPR, CGFR","CGPR");
  set.add_string("LineSearch","Line search to use: poly_df, poly_fdf, armijo","poly_df");
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

  // Determine method
  enum locmet method;
  std::string mets=set.get_string("Localization");
  if(stricmp(mets,"BF")==0)
    method=BOYS;
  else if(stricmp(mets,"PM")==0)
    method=PIPEK;
  else throw std::runtime_error("Localization method not implemented.\n");

  // Determine accelerator
  enum unitacc acc;
  std::string accs=set.get_string("Accelerator");
  if(stricmp(accs,"SD")==0)
    acc=SD;
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

    // Occupation numbers
    std::vector<double> occs;
    chkpt.read("occs",occs);

    // Run localization
    localize(basis,C,occs,virt,method,umet,acc);

    chkpt.write("C",C);

  } else {
    // Orbitals
    arma::mat Ca, Cb;
    chkpt.read("Ca",Ca);
    chkpt.read("Cb",Cb);

    // Occupation numbers
    std::vector<double> occa, occb;
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);

    // Run localization
    localize(basis,Ca,occa,virt,method,umet,acc);
    localize(basis,Cb,occb,virt,method,umet,acc);

    chkpt.write("Ca",Ca);
    chkpt.write("Cb",Cb);
  }

  return 0;
}
