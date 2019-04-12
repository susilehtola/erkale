/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2019
 * Copyright (c) 2010-2019, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "elements.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Linear symmetries from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Linear symmetries from Hel, serial version.\n");
#endif

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }

  // Parse settings
  Settings set;
  set.add_string("Nuclei","Nuclei in molecule","");
  set.add_int("Nspin","Number of spin states to look at",1);
  set.add_int("Mmax","Maximum net angular momentum allowed",3);
  set.add_bool("saveconf","Save configurations to disk",true);
  set.add_bool("savegeom","Save geometry to disk",true);
  set.add_bool("largeactive","Use larger active space?",false);
  set.parse(std::string(argv[1]),true);
  set.print();
  int nspin=set.get_int("Nspin");
  int netmmax=set.get_int("Mmax");
  bool saveconf=set.get_bool("saveconf");
  bool savegeom=set.get_bool("savegeom");
  bool largeactive=set.get_bool("largeactive");
  std::string nucstr=set.get_string("Nuclei");
  
  std::vector<std::string> nuclei(splitline(nucstr));
  // Translate to integers
  std::vector<int> Zs(nuclei.size());
  for(size_t i=0;i<nuclei.size();i++)
    Zs[i]=get_Z(nuclei[i]);

  // Print geometry
  {
    int Zmax=(int) (sizeof(covalent_radii)/sizeof(covalent_radii[0]));
    arma::vec zval(Zs.size());
    double z=0.0;
    
    for(size_t i=0;i<Zs.size();i++) {
      if(Zs[i]>=Zmax)
        throw std::logic_error("No radius found for element\n");

      if(i>0)
        z+=covalent_radii[Zs[i]];
      // z coordinate for atom is
      zval(i)=z;
      z+=covalent_radii[Zs[i]];
    }

    if(savegeom) {
      FILE *out=fopen("atoms.xyz","w");
      fprintf(out,"%i\n%s\n",(int) Zs.size(),nucstr.c_str());
      for(size_t i=0;i<Zs.size();i++) 
        fprintf(out,"%-2s %.3f %.3f %.3f\n",element_symbols[Zs[i]].c_str(),0.0,0.0,zval(i));
      fclose(out);
    } else {
      printf("Input geometry\n");
      for(size_t i=0;i<Zs.size();i++) 
        printf("%-2s %.3f %.3f %.3f\n",element_symbols[Zs[i]].c_str(),0.0,0.0,zval(i));
    }
  }

  // Form frozen core and active space: sigma, pi, delta, phi
  const int mmax=3;

  arma::ivec mval(arma::linspace<arma::ivec>(-mmax,mmax,2*mmax+1));
  arma::ivec frozen(2*mmax+1);
  arma::ivec active(2*mmax+1);
  int nelectrons=0;
  frozen.fill(0);
  active.fill(0);
  
  for(size_t inuc=0;inuc<Zs.size();inuc++) {
    // Find noble gas core
    size_t imagic=0;
    while(imagic<sizeof(magicno)/sizeof(magicno[0])-1) {
      printf("imagic = %i\n",(int)imagic);
      if(Zs[inuc]<=magicno[imagic+1])
        break;
      imagic++;
    }
    int Znoble=magicno[imagic];
    int Znext=magicno[imagic+1];

    printf("Z=%i, Znoble = %i, Znext = %i\n",Zs[inuc],Znoble,Znext);

    // Fill shells up to magic number
    int nfilled=0;
    size_t ishell=0;
    if(Znoble) {
      for(;ishell<sizeof(shell_order)/sizeof(shell_order[0]);ishell++) {
        // shell am is
        int l(shell_order[ishell]);
        // increment frozen orbital occupation
        frozen.subvec(mmax-l,mmax+l)+=arma::ones<arma::ivec>(2*l+1);
        printf("After filling l=%i\n",l);
        frozen.print();      
        nfilled+=2*(2*l+1);
        if(nfilled==Znoble) {
          ishell++;
          break;
        }
      }
    }
    
    // Number of electrons to distribute
    nelectrons+=Zs[inuc]-Znoble;

    // Form active space
    int Zfill = largeactive ? Znext : Zs[inuc];
    for(;ishell<sizeof(shell_order)/sizeof(shell_order[0]);ishell++) {
      // shell am is
      int l(shell_order[ishell]);
      // increment frozen orbital occupation
      active.subvec(mmax-l,mmax+l)+=arma::ones<arma::ivec>(2*l+1);
      printf("Active after filling l=%i\n",l);
      active.print();      
      nfilled+=2*(2*l+1);
      if(nfilled>=Zfill)
        break;
    }
  }

  frozen.print("Frozen orbitals");
  active.print("Active orbitals");
  
  // Generate configurations in active space
  printf("Need to distribute %i active electrons onto %i active orbitals\n",nelectrons,(int) arma::sum(active));

  // Generate spin states
  size_t totnconf=0;
  for(int nx=0;nx<nspin;nx++) {
    int nelb=(nelectrons/2)-nx;
    int nela=nelectrons-nelb;
    int dnel=nela-nelb;

    printf("Spin state S = %i\n",1+dnel);

    // Generate list of orbitals that can be occupied
    std::vector<int> orbitals;
    for(size_t i=0;i<active.n_elem;i++)
      for(int j=0;j<active(i);j++)
        // orbital m value
        orbitals.push_back((int)i - mmax);

    // Generate list of possible beta occupations by permuting through all orbital occupations
    std::vector<arma::ivec> beta_occs;
    std::vector<int> ohelper(orbitals);
    do {
      // Calculate net value of m
      int mnet=0;
      for(int i=0;i<nelb;i++)
        mnet+=ohelper[i];
      if(mnet<0)
        // Restrict to considering positive net values of m
        continue;
      if(mnet>netmmax)
        // Too large net am
        continue;

      // Configuration is okay, generate beta occ vector
      arma::ivec bocc(2*mmax+1);
      bocc.zeros();
      for(int i=0;i<nelb;i++)
        bocc(mmax+ohelper[i])++;

      // Check that an identical one does not exist
      bool exist=false;
      for(size_t i=0;i<beta_occs.size();i++) {
        bool match=true;
        for(size_t j=0;j<bocc.n_elem;j++)
          if(beta_occs[i][j]!=bocc[j])
            match=false;
        if(match)
          exist=true;
      }
      if(!exist) {
        beta_occs.push_back(bocc);
      }
    } while(std::next_permutation(ohelper.begin(),ohelper.end()));

    // Full set of occupations
    std::vector<arma::imat> occlist;
    if(nela == nelb) {
      // Occupations are the same
      for(size_t ibeta=0;ibeta<beta_occs.size();ibeta++) {
        arma::imat occs(beta_occs[ibeta].n_elem,2);
        occs.col(0)=beta_occs[ibeta];
        occs.col(1)=beta_occs[ibeta];
        // Compute net value of m
        int mnet=0;
        for(int m=-mmax;m<=mmax;m++)
          mnet+=m*arma::sum(occs.row(m+mmax));
        if(mnet<=netmmax)
          occlist.push_back(occs);
      }
    } else {
      for(size_t ibeta=0;ibeta<beta_occs.size();ibeta++) {
        // Generate list of possible alpha occupations. List orbitals
        // that can be occupied by extra alpha electrons
        std::vector<int> aorbitals;
        arma::ivec aactive(active-beta_occs[ibeta]);
        for(size_t i=0;i<aactive.n_elem;i++)
          for(int j=0;j<aactive(i);j++)
            // orbital m value
            aorbitals.push_back((int)i - mmax);

        // Orbital helper is now
        ohelper=aorbitals;
        std::vector<arma::ivec> alpha_occs;
        do {
          // Generate alpha occ vector
          arma::ivec aocc(2*mmax+1);
          aocc.zeros();
          for(int i=0;i<dnel;i++)
            aocc(mmax+ohelper[i])++;
          // Plug in the occupied beta background
          aocc+=beta_occs[ibeta];

          // Calculate net value of m
          int mnet=0;
          for(int m=-mmax;m<=mmax;m++)
            mnet+=m*(aocc(m+mmax)+beta_occs[ibeta](m+mmax));
          if(mnet<0)
            // Restrict to considering positive net values of m
            continue;
          if(mnet>netmmax)
            // Too large net am
            continue;
                
          // Check that an identical one does not exist
          bool exist=false;
          for(size_t i=0;i<alpha_occs.size();i++) {
            bool match=true;
            for(size_t j=0;j<aocc.n_elem;j++)
              if(alpha_occs[i][j]!=aocc[j])
                match=false;
            if(match)
              exist=true;
          }
          if(!exist) {
            alpha_occs.push_back(aocc);
          }
        } while(std::next_permutation(ohelper.begin(),ohelper.end()));

        // Form total configurations
        for(size_t ialpha=0;ialpha<alpha_occs.size();ialpha++) {        
          arma::imat occs(beta_occs[ialpha].n_elem,2);
          occs.col(0)=alpha_occs[ialpha];
          occs.col(1)=beta_occs[ibeta];
          // Compute net value of m
          int mnet=0;
          for(int m=-mmax;m<=mmax;m++)
            mnet+=m*arma::sum(occs.row(m+mmax));
          if(mnet<=netmmax)
            occlist.push_back(occs);
        }
      }
    }

    // Increase total number of configurations
    totnconf+=occlist.size();

    // Check configurations are all different
    for(size_t iconf=0;iconf<occlist.size();iconf++)
      for(size_t jconf=0;jconf<iconf;jconf++) {
        bool match=true;
        for(size_t i=0;i<occlist[iconf].n_rows;i++) {
          if(occlist[iconf](i,1)!=occlist[jconf](i,1))
            match=false;
          if(occlist[iconf](i,2)!=occlist[jconf](i,2))
            match=false;
        }
        if(match)
          printf("Configurations %i and %i are the same!\n",iconf,jconf);
      }

    // Print out configurations
    for(size_t iconf=0;iconf<occlist.size();iconf++) {
      arma::imat conf(2*mmax+1,3);
      // Plug in frozen core orbitals
      conf.col(0)=occlist[iconf].col(0)+frozen;
      conf.col(1)=occlist[iconf].col(1)+frozen;
      conf.col(2)=mval;

      // Find number of electrons per m channel
      arma::ivec nelec(2*mmax+1);
      for(size_t i=0;i<nelec.n_elem;i++)
        nelec(i)=conf(i,0)+conf(i,1);

      int mnet=0;
      for(int m=-mmax;m<=mmax;m++)
        mnet+=m*nelec(m+mmax);

      // Get rid of zero rows
      arma::uvec idx(arma::find(nelec>0));
      conf=conf.rows(idx);
      
      if(saveconf) {
        std::ostringstream oss;
        oss << "linoccs_" << 1+dnel << "_" << iconf << ".dat";
        conf.save(oss.str(),arma::raw_ascii);
        printf("Configuration %i with net angular momentum M=%i saved to file %s\n",(int) iconf,mnet,oss.str().c_str());
      } else {
        printf("Configuration %i with net angular momentum M=%i\n",(int) iconf,mnet);
        conf.print();
      }
    }
  }

  printf("%i possible configurations found in total.\n",totnconf);
  
  return 0;
}
