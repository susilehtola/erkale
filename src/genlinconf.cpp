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

bool check_configuration(const arma::imat & conf) {
  int mmax=(conf.n_rows-1)/2;
  // Calculate net value of m
  int mnet=0;
  for(int m=-mmax;m<=mmax;m++)
    mnet+=m*(conf(m+mmax,0)+conf(m+mmax,1));
  if(mnet<0)
    // Restrict to considering positive net values of m
    return false;

  return true;
}

int classify_m(const arma::imat & conf) {
  int mmax=(conf.n_rows-1)/2;
  // Classify configuration
  int mval=0;
  for(int m=1;m<=mmax;m++) {
    // alpha and beta contributions to m
    int dna=conf(m+mmax,0)-conf(mmax-m,0);
    int dnb=conf(m+mmax,1)-conf(mmax-m,1);

    // dna>=0 and dnb>=0: mval=0
    // dna<0 or dnb<0 but dna+dnb>=0: mval=1
    // dna<0 and dnb<0 and dna+dnb<0: mval=2
    if((dna<0) || (dnb<0)) {
      if(dna+dnb>=0) {
        mval=std::max(mval,1);
      } else {
        mval=std::max(mval,2);
      }
    }
  }

  return mval;
}

// Only need up to phi orbitals
const int mmax=3;

void integer_occupations(const std::vector<int> & Zs, bool largeactive, int Q, bool saveconf) {
  // Form frozen core and active space: sigma, pi, delta, phi
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
      active.subvec(mmax-l,mmax+l)+=arma::ones<arma::ivec>(2*l+1);
      nfilled+=2*(2*l+1);
      if(nfilled>=Zfill)
        break;
    }
  }

  // Add in charge
  nelectrons-=Q;

  {
    arma::imat oinfo(2*mmax+1,3);
    oinfo.col(0)=mval;
    oinfo.col(1)=frozen;
    oinfo.col(2)=active;

    arma::ivec nelec(2*mmax+1);
    for(size_t i=0;i<oinfo.n_rows;i++)
      nelec(i)=oinfo(i,1)+oinfo(i,2);
    // Get rid of zero rows
    arma::uvec idx(arma::find(nelec>0));
    oinfo=oinfo.rows(idx);

    printf("Orbital information\n%2s %6s %6s\n","m","frozen","active");
    for(size_t ir=0;ir<oinfo.n_rows;ir++)
      printf("% i %6i %6i\n",(int) oinfo(ir,0),(int) oinfo(ir,1),(int) oinfo(ir,2));
  }

  // Generate configurations in active space
  printf("Need to distribute %i active electrons onto %i active orbitals\n",nelectrons,(int) arma::sum(active));

  // Check that this is sane
  if(nelectrons<0)
    throw std::logic_error("No electrons to distribute!\n");
  if(nelectrons>2*arma::sum(active))
    throw std::logic_error("Too many electrons to fit in active space!\n");

  // Generate spin states
  size_t totnconf=0;
  for(int nx=0;;nx++) {
    int nelb=(nelectrons/2)-nx;
    int nela=nelectrons-nelb;
    int dnel=nela-nelb;

    // Can we even have this spin state for this molecule?
    if(nela>arma::sum(active))
      break;
    if(nelb<0)
      break;

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
      // Generate beta occ vector
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

        // Add to list
        if(check_configuration(occs))
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
          arma::imat occs(beta_occs[ibeta].n_elem,2);
          occs.col(0)=alpha_occs[ialpha];
          occs.col(1)=beta_occs[ibeta];

          if(check_configuration(occs))
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
        for(size_t j=0;j<occlist[iconf].n_cols;j++)
          for(size_t i=0;i<occlist[iconf].n_rows;i++)
            if(occlist[iconf](i,j)!=occlist[jconf](i,j))
              match=false;
        if(match) {
          printf("Configurations %i and %i are the same!\n",(int) iconf, (int) jconf);
          throw std::logic_error("Degenerate configs!\n");
        }
      }

    // Remove degenerate configs (switch +m/-m)
    {
      std::vector<arma::imat> newlist;
      for(size_t iconf=0;iconf<occlist.size();iconf++) {
        arma::imat testconf(occlist[iconf]);
        bool duplicate=false;

        for(size_t ic=0;ic<testconf.n_cols;ic++)
          for(int m=1;m<=mmax;m++) {
            // Swap m and -m
            std::swap(testconf(m+mmax,ic),testconf(mmax-m,ic));
          }
        for(size_t jconf=0;jconf<newlist.size();jconf++) {
          bool match=true;
          for(size_t j=0;j<testconf.n_cols;j++)
            for(size_t i=0;i<testconf.n_rows;i++)
              if(testconf(i,j)!=newlist[jconf](i,j))
                match=false;
          if(match) {
            duplicate=true;
            break;
          }
        }
        if(!duplicate)
          newlist.push_back(occlist[iconf]);
      }
      std::swap(occlist,newlist);
    }

    // Print out configurations
    std::vector< std::vector<size_t> > numconf;
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
      if(mnet<0)
        throw std::logic_error("Shouldn't have config with m<0^!\n");

      int mclass=classify_m(conf);

      // File name
      std::ostringstream fname;
      fname << "linoccs_" << mclass << "_" << 1+dnel << "_" << mnet << "_";

      // Check storage space
      if(numconf.size() <= (size_t) mclass)
        numconf.resize(mclass+1);
      if(numconf[mclass].size() <= (size_t) mnet)
        numconf[mclass].resize(mnet+1,0);
      fname << numconf[mclass][mnet]++ << ".dat";

      // Get rid of zero rows
      arma::uvec idx(arma::find(nelec>0));
      conf=conf.rows(idx);

      if(saveconf) {
        conf.save(fname.str(),arma::raw_ascii);
        printf("Configuration %i with net angular momentum M=%i saved to file %s\n",(int) iconf,mnet,fname.str().c_str());
      } else {
        printf("Configuration %i with net angular momentum M=%i is of class %i\n",(int) iconf,mnet,mclass);
        conf.print();
      }
    }
  }

  printf("%i possible configurations found in total.\n",(int) totnconf);
}

void fractional_occupations(const std::vector<int> & Zs, bool largeactive, int Q, bool saveconf) {
  // Form frozen core and active space: sigma, pi, delta, phi
  arma::ivec mval(arma::linspace<arma::ivec>(0,mmax,mmax+1));
  arma::ivec frozen(mmax+1);
  arma::ivec active(mmax+1);
  int nelectrons=0;
  frozen.fill(0);
  active.fill(0);

  for(size_t inuc=0;inuc<Zs.size();inuc++) {
    // Find noble gas core
    size_t imagic=0;
    while(imagic<sizeof(magicno)/sizeof(magicno[0])-1) {
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
        frozen(0)++;
        if(l>0)
          frozen.subvec(1,l)+=arma::ones<arma::ivec>(l);
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
      active(0)++;
      if(l)
        active.subvec(1,l)+=2*arma::ones<arma::ivec>(l);
      nfilled+=2*(2*l+1);
      if(nfilled>=Zfill)
        break;
    }
  }

  // Add in charge
  nelectrons-=Q;

  {
    arma::imat oinfo(mmax+1,3);
    oinfo.col(0)=mval;
    oinfo.col(1)=frozen;
    oinfo.col(2)=active;

    arma::ivec nelec(mmax+1);
    for(size_t i=0;i<oinfo.n_rows;i++)
      nelec(i)=oinfo(i,1)+oinfo(i,2);
    // Get rid of zero rows
    arma::uvec idx(arma::find(nelec>0));
    oinfo=oinfo.rows(idx);

    printf("Orbital information\n%2s %6s %6s\n","m","frozen","active");
    for(size_t ir=0;ir<oinfo.n_rows;ir++)
      printf("% i %6i %6i\n",(int) oinfo(ir,0),(int) oinfo(ir,1),(int) oinfo(ir,2));
  }

  // Generate configurations in active space
  printf("Need to distribute %i active electrons onto %i active orbitals\n",nelectrons,(int) arma::sum(active));

  // Check that this is sane
  if(nelectrons<0)
    throw std::logic_error("No electrons to distribute!\n");
  if(nelectrons>2*arma::sum(active))
    throw std::logic_error("Too many electrons to fit in active space!\n");

  // Generate spin states
  size_t totnconf=0;
  for(int nx=0;;nx++) {
    int nelb=(nelectrons/2)-nx;
    int nela=nelectrons-nelb;
    int dnel=nela-nelb;

    // Can we even have this spin state for this molecule?
    if(nela>arma::sum(active))
      break;
    if(nelb<0)
      break;

    printf("Spin state S = %i\n",1+dnel);

    // Generate list of orbitals that can be occupied
    std::vector<int> orbitals;
    for(size_t i=0;i<active.n_elem;i++)
      for(int j=0;j<active(i);j++)
        // orbital m value
        orbitals.push_back((int) i);

    // Generate list of possible beta occupations by permuting through all orbital occupations
    std::vector<arma::ivec> beta_occs;
    std::vector<int> ohelper(orbitals);
    do {
      // Generate beta occ vector
      arma::ivec bocc(mmax+1);
      bocc.zeros();
      for(int i=0;i<nelb;i++)
        bocc(ohelper[i])++;

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

        // Add to list
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
            aorbitals.push_back((int)i);

        // Orbital helper is now
        ohelper=aorbitals;
        std::vector<arma::ivec> alpha_occs;
        do {
          // Generate alpha occ vector
          arma::ivec aocc(mmax+1);
          aocc.zeros();
          for(int i=0;i<dnel;i++)
            aocc(ohelper[i])++;
          // Plug in the occupied beta background
          aocc+=beta_occs[ibeta];

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
          arma::imat occs(beta_occs[ibeta].n_elem,2);
          occs.col(0)=alpha_occs[ialpha];
          occs.col(1)=beta_occs[ibeta];
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
        for(size_t j=0;j<occlist[iconf].n_cols;j++)
          for(size_t i=0;i<occlist[iconf].n_rows;i++)
            if(occlist[iconf](i,j)!=occlist[jconf](i,j))
              match=false;
        if(match) {
          printf("Configurations %i and %i are the same!\n",(int) iconf, (int) jconf);
          throw std::logic_error("Degenerate configs!\n");
        }
      }

    // Print out configurations
    size_t numconf=0;
    for(size_t iconf=0;iconf<occlist.size();iconf++) {
      arma::mat conf(2*mmax+1,3);
      // Plug in frozen core orbitals
      for(int m=-mmax;m<=mmax;m++) {
        if(m==0) {
          conf(m+mmax,0)=occlist[iconf](m,0)+frozen(m);
          conf(m+mmax,1)=occlist[iconf](m,1)+frozen(m);
        } else {
          conf(m+mmax,0)=frozen(std::abs(m))+0.5*occlist[iconf](std::abs(m),0);
          conf(m+mmax,1)=frozen(std::abs(m))+0.5*occlist[iconf](std::abs(m),1);
        }
        conf(m+mmax,2)=m;
      }

      // File name
      std::ostringstream fname;
      fname << "linoccs_fracocc" << "_" << 1+dnel << "_";

      // Check storage space
      fname << numconf++ << ".dat";

      // Number of electrons per m channel
      arma::vec nelec(conf.n_rows);
      for(size_t i=0;i<nelec.n_elem;i++)
        nelec(i)=conf(i,0)+conf(i,1);
      // Get rid of zero rows
      arma::uvec idx(arma::find(nelec>0.0));
      conf=conf.rows(idx);

      if(saveconf) {
        conf.save(fname.str(),arma::raw_ascii);
        printf("Configuration %i saved to file %s\n",(int) iconf,fname.str().c_str());
      } else {
        printf("Configuration %i\n",(int) iconf);
        conf.print();
      }
    }
  }

  printf("%i possible configurations found in total.\n",(int) totnconf);
}

Settings settings;

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
  settings.add_string("Nuclei","Nuclei in molecule","");
  settings.add_int("Charge","Charge state for molecule",0,true);
  settings.add_bool("saveconf","Save configurations to disk",true);
  settings.add_bool("savegeom","Save geometry to disk",true);
  settings.add_bool("largeactive","Use larger active space?",false);
  settings.add_bool("fracocc","Use fractional occupations to ensure spherically symmetric density?",false);
  settings.parse(std::string(argv[1]),true);
  settings.print();
  int Q=settings.get_int("Charge");
  bool saveconf=settings.get_bool("saveconf");
  bool savegeom=settings.get_bool("savegeom");
  bool largeactive=settings.get_bool("largeactive");
  bool fracocc=settings.get_bool("fracocc");
  std::string nucstr=settings.get_string("Nuclei");

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

  if(fracocc)
    fractional_occupations(Zs,largeactive,Q,saveconf);
  else
    integer_occupations(Zs,largeactive,Q,saveconf);

  return 0;
}
