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

/**
 * This file contains routines for reading and writing formatted
 * checkpoint files. These are provided to provide interoperability
 * with other software, e.g., for visualizing orbitals and densities
 * computed with ERKALE with other software such as Avogadro, and to
 * enable momentum density computations with calculations made with
 * other program suites.
 */

#include "storage.h"
#include "fchkpt_tools.h"
#include "../global.h"
#include "../basis.h"
#include "../checkpoint.h"
#include "../linalg.h"
#include "../mathf.h"
#include "../settings.h"
#include "../stringutil.h"
#include "../timer.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

void print(const std::string & entry, int val, FILE *out) {
  fprintf(out,"%-42s I   %3s %11i\n",entry.c_str(),"",val);
  fflush(out);
}

void print(const std::string & entry, double val, FILE *out) {
  fprintf(out,"%-42s R   %3s %11e\n",entry.c_str(),"",val);
  fflush(out);
}

void print(const std::string & entry, const std::vector<int> & val, FILE *out) {
  fprintf(out,"%-42s I   %3s %11i\n",entry.c_str(),"N=",(int) val.size());

  short int N=0;
  for(size_t i=0;i<val.size();i++) {
    fprintf(out," % 11i",val[i]);
    N++;
    if(N==6) {
      N=0;
      fprintf(out,"\n");
    }
  }
  if(N!=0)
    fprintf(out,"\n");

  fflush(out);
}

void print(const std::string & entry, const std::vector<double> & val, FILE *out) {
  fprintf(out,"%-42s R   %3s %11i\n",entry.c_str(),"N=",(int) val.size());

  short int N=0;
  for(size_t i=0;i<val.size();i++) {
    fprintf(out," % 15.8e",val[i]);
    N++;
    if(N==5) {
      N=0;
      fprintf(out,"\n");
    }
  }
  if(N!=0)
    fprintf(out,"\n");

  fflush(out);
}

std::vector<int> form_shelltypes(const BasisSet & basis) {
  // Get the shells
  std::vector<GaussianShell> shells=basis.get_shells();

  // Get shell types.
  std::vector<int> shtypes(shells.size());
  for(size_t i=0;i<shells.size();i++) {
    // Get angular momentum
    int am=shells[i].get_am();

    // Use spherical harmonics?
    if(shells[i].lm_in_use())
      shtypes[i]=-am;
    else
      shtypes[i]=am;
  }

  return shtypes;
}

void write_mo(const std::string & entry, const BasisSet & basis, const arma::mat & C, FILE *out) {
  // Print MO matrix
  fprintf(out,"%-42s R   %s %11i\n",entry.c_str(),"N=",(int) (C.n_rows*C.n_cols));

  // Get amount of basis functions
  size_t Nbf=basis.get_Nbf();
  if(Nbf!=C.n_rows) {
    ERROR_INFO();
    throw std::runtime_error("Orbitals do not correspond to basis set!\n");
  }

  // Get shell types
  std::vector<int> shtype=form_shelltypes(basis);

  // Get index converter
  std::vector<size_t> idx=ge_indarr(shtype,Nbf);

  size_t N=0;

  // Print output
  for(size_t imo=0;imo<C.n_cols;imo++)
    for(size_t ibf=0;ibf<Nbf;ibf++) {
      // Get ERKALE index
      size_t ie=idx[ibf];

      fprintf(out," % 15.8e",C(ie,imo));
      N++;
      if(N==5) {
	N=0;
	fprintf(out,"\n");
      }
    }
  if(N!=0)
    fprintf(out,"\n");

  fflush(out);
}

void write_density(const std::string & entry, const BasisSet & basis, const arma::mat & P, FILE *out) {
  // Get amount of basis functions
  size_t Nbf=basis.get_Nbf();
  if(Nbf!=P.n_rows || Nbf!=P.n_cols) {
    ERROR_INFO();
    throw std::runtime_error("Density matrix does not correspond to basis set!\n");
  }

  // Amount of non-equivalent entries is
  size_t Nind=Nbf*(Nbf+1)/2;

  // Print density matrix
  fprintf(out,"%-42s R   %s %11i\n",entry.c_str(),"N=",(int) (Nind));

  // Get shell types
  std::vector<int> shtype=form_shelltypes(basis);

  // Get index converter
  std::vector<size_t> idx=ge_indarr(shtype,Nbf);

  size_t N=0;
  size_t ntot=0;

  // Loop over functions
  for(size_t i=0;i<Nbf;i++) {
    // ERKALE index
    size_t ie=idx[i];

    for(size_t j=0;j<=i;j++) {
      // ERKALE index
      size_t je=idx[j];

      fprintf(out," % 15.8e",P(ie,je));
      N++;
      ntot++;

      if(N==5) {
	N=0;
	fprintf(out,"\n");
      }
    }
  }
  if(N!=0)
    fprintf(out,"\n");

  if(ntot!=Nind)
    throw std::runtime_error("Wrong amount!\n");

  fflush(out);
}

void write_basis(const BasisSet & basis, FILE *out) {
  // Print number of atoms.
  print("Number of atoms", (int) basis.get_Nnuc(), out);
  // Print number of basis functions.
  print("Number of basis functions", (int) basis.get_Nbf(), out);

  /* Nuclei */

  // Get the nuclei.
  std::vector<nucleus_t> nucs=basis.get_nuclei();

  // Print atomic numbers.
  std::vector<int> atnum(nucs.size());
  for(size_t i=0;i<atnum.size();i++)
    atnum[i]=nucs[i].Z;
  print("Atomic numbers",atnum,out);

  // Print coordinates of nuclei.
  std::vector<double> coords(3*nucs.size());
  for(size_t i=0;i<nucs.size();i++) {
    coords[3*i]=nucs[i].r.x;
    coords[3*i+1]=nucs[i].r.y;
    coords[3*i+2]=nucs[i].r.z;
  }
  print("Current cartesian coordinates",coords,out);

  /* Basis set */

  // Get the shells
  std::vector<GaussianShell> shells=basis.get_shells();

  // Print shell types.
  std::vector<int> shtypes=form_shelltypes(basis);
  print("Shell types",shtypes,out);

  // Print shell to atom map
  std::vector<int> shmap(shells.size());
  for(size_t i=0;i<shells.size();i++)
    shmap[i]=(int) shells[i].get_center_ind()+1;
  print("Shell to atom map",shmap,out);

  // Print the coordinates of the shells
  std::vector<double> shcoords(3*shells.size());
  for(size_t i=0;i<shells.size();i++) {
    coords_t r=shells[i].get_center();

    shcoords[3*i]=r.x;
    shcoords[3*i+1]=r.y;
    shcoords[3*i+2]=r.z;
  }
  print("Coordinates of each shell",shcoords,out);

  // Print number of primitives per shell, exponents and contraction coefficients.
  std::vector<double> exps;
  std::vector<double> contr;
  std::vector<int> nprim(shells.size());
  for(size_t i=0;i<shells.size();i++) {
    // Get the contraction of *normalized* primitives
    std::vector<contr_t> c=shells[i].get_contr_normalized();
    nprim[i]=(int) c.size();

    // Save exponents and *primitive* contraction coefficients
    for(size_t j=0;j<c.size();j++) {
      exps.push_back(c[j].z);
      contr.push_back(c[j].c);
    }
  }
  print("Number of primitives per shell",nprim,out);
  print("Primitive exponents",exps,out);
  print("Contraction coefficients",contr,out);
}

Settings settings;

void load_fchk(double tol) {
  Timer t;

  // Read in checkpoint
  printf("Read in formatted checkpoint ... ");
  fflush(stdout);
  Storage stor=parse_fchk(settings.get_string("LoadFchk"));
  printf("done (%s)\n",t.elapsed().c_str());
  //  stor.print(false);

  // Construct basis set
  BasisSet basis=form_basis(stor);
  //  basis.print(true);

  // Construct density matrix
  arma::mat P=form_density(stor);

  int Nel=stor.get_int("Number of electrons");
  int Nela=stor.get_int("Number of alpha electrons");
  int Nelb=stor.get_int("Number of beta electrons");

  // Special handling for ROHF
  try {
    if(stor.get_int("IROHF")==1) {
      P.zeros();
      arma::mat Ca=form_orbital_C(stor,"Alpha MO coefficients");

      P.zeros();
      for(int i=0;i<Nela;i++)
	P+=Ca.col(i)*arma::trans(Ca.col(i));
      for(int i=0;i<Nelb;i++)
	P+=Ca.col(i)*arma::trans(Ca.col(i));
    }
  } catch(std::runtime_error &) {};

  // Form orbitals
  arma::mat Ca, Cb;
  bool restr=false;
  Ca=form_orbital_C(stor,"Alpha MO coefficients");
  try {
    Cb=form_orbital_C(stor,"Beta MO coefficients");
  } catch(std::runtime_error &) {
    // Restricted checkpoint
    restr=true;

    // but check again if we have the ROHF bug
    try {
      if(stor.get_int("IROHF")==1) {
	restr=false;
	Cb=Ca;
      }
    } catch(std::runtime_error &) {};
  }

  // Energies
  arma::vec Ea, Eb;
  Ea=form_orbital_E(stor,"Alpha Orbital Energies");
  if(!restr)
    Eb=form_orbital_E(stor,"Beta Orbital Energies");

  // Densities?
  arma::mat Pa, Pb;
  if(!restr) {
    arma::mat Pspin=form_density(stor,true);

    Pa=(P+Pspin)/2.0;
    Pb=(P-Pspin)/2.0;
  }

  // Check that everything is OK
  t.set();
  printf("\nComputing overlap matrix ... ");
  fflush(stdout);
  arma::mat S=basis.overlap();
  printf("done (%s)\n",t.elapsed().c_str());

  double nelnum=arma::trace(P*S);
  double neldiff=nelnum-Nel;
  if(fabs(neldiff)/Nel>tol) {
    std::ostringstream oss;
    oss << "\nNumber of electrons and trace of density matrix differ by " << neldiff << "!\n";
    throw std::runtime_error(oss.str());
  }
  printf("tr PS - Nel = %.e\n",neldiff);

  if(!restr) {
    double nelnuma=arma::trace(Pa*S);
    double neldiffa=nelnuma-Nela;

    double nelnumb=arma::trace(Pb*S);
    double neldiffb=nelnumb-Nelb;

    if(fabs(neldiffa)/Nela>tol) {
      std::ostringstream oss;
      oss << "\nNumber of alpha electrons and trace of alpha density matrix differ by " << neldiffa << "!\n";
      throw std::runtime_error(oss.str());
    }
    printf("tr PaS - Nela = %.e\n",neldiffa);

    if(fabs(neldiffb)/Nelb>tol) {
      std::ostringstream oss;
      oss << "\nNumber of beta electrons and trace of beta density matrix differ by " << neldiffb << "!\n";
      throw std::runtime_error(oss.str());
    }
    printf("tr PbS - Nelb = %.e\n",neldiffb);
  }


  // Renormalize
  if(settings.get_bool("Renormalize")) {
    P*=Nel/nelnum;

    if(!restr) {
      Pa*=Nela/arma::trace(Pa*S);
      Pb*=Nelb/arma::trace(Pb*S);
    }
  }

  if(settings.get_bool("Reorthonormalize")) {
    printf("\nReorthonormalizing orbitals ... ");
    fflush(stdout);
    t.set();

    // Compute deviation from orthonormality
    double Camax=arma::max(arma::abs(arma::diagvec(arma::trans(Ca)*S*Ca)-1.0));

    // Compute Ca overlap
    Ca=orthonormalize(S,Ca);

    if(restr)
      printf("done (%s).\nMaximum deviation from orthonormality was %e.\n",t.elapsed().c_str(),Camax);

    else {
      double Cbmax=arma::max(arma::abs(arma::diagvec(arma::trans(Cb)*S*Cb)-1.0));

      Cb=orthonormalize(S,Cb);

      printf("done (%s).\nMaximum deviation from orthonormality was %e %e.\n",t.elapsed().c_str(),Camax,Cbmax);
    }
  }

  // Check the orbitals
  try {
    check_orth(Ca,S,true,tol);
    if(!restr)
      check_orth(Cb,S,true,tol);
  } catch(std::runtime_error &) {
    std::ostringstream oss;

    oss << "\nIt seems the orbitals in the checkpoint file are not orthonormal.\n";
    if(restr)
      oss << "The maximal deviation from orthonormality was found to be " << orth_diff(Ca,S) << ".\n";
    else
      oss << "The maximal deviation from orthonormality was found to be " << orth_diff(Ca,S) << " and " << orth_diff(Cb,S) << " for alpha and beta orbitals.\n";
    oss << "Increase the tolerance or run the program with the reorthonormalize option.\n";
    throw std::runtime_error(oss.str());
  }

  // Save the result
  t.set();
  Checkpoint chkpt(settings.get_string("SaveChk"),true);
  chkpt.write(basis);
  chkpt.write("P",P);

  chkpt.write("Restricted",restr);
  if(restr) {
    chkpt.write("C",Ca);
    chkpt.write("E",Ea);

    // Occupations
    std::vector<double> occs(Ca.n_cols,0.0);
    for(int io=0;io<Nel/2;io++)
      occs[io]=2.0;
    chkpt.write("occs",occs);
  } else {
    chkpt.write("Ca",Ca);
    chkpt.write("Cb",Cb);
    chkpt.write("Ea",Ea);
    chkpt.write("Eb",Eb);

    // Occupations
    std::vector<double> occa(Ca.n_cols,0.0);
    for(int io=0;io<Nela;io++)
      occa[io]=1.0;
    chkpt.write("occa",occa);
    std::vector<double> occb(Ca.n_cols,0.0);
    for(int io=0;io<Nelb;io++)
      occb[io]=1.0;
    chkpt.write("occb",occb);

    // Density matrices
    chkpt.write("Pa",Pa);
    chkpt.write("Pb",Pb);
  }

  chkpt.write("Converged",true);

  // Write number of electrons
  chkpt.write("Nel",Nel);
  chkpt.write("Nel-a",Nela);
  chkpt.write("Nel-b",Nelb);

  printf("\nERKALE checkpoint saved in %s.\n",t.elapsed().c_str());
}

void save_fchk() {
  Timer t;

  // Load checkpoint
  Checkpoint chkpt(settings.get_string("LoadChk"),false);

  // Construct basis
  BasisSet basis;
  chkpt.read(basis);
  //  basis.print(true);

  // File to save
  std::string savename=settings.get_string("SaveFchk");

  // Handle also compressed files
  std::string gzcmd="gzip ";
  bool usegz=false;
  if(strstr(savename.c_str(),".gz")!=NULL)
    usegz=true;

  std::string xzcmd="xz ";
  bool usexz=false;
  if(strstr(savename.c_str(),".xz")!=NULL)
    usexz=true;

  std::string bz2cmd="bzip2 ";
  bool usebz2=false;
  if(strstr(savename.c_str(),".bz2")!=NULL)
    usebz2=true;

  std::string lzmacmd="lzma ";
  bool uselzma=false;
  if(strstr(savename.c_str(),".lzma")!=NULL)
    uselzma=true;

  // Open output file.
  FILE *out=fopen(settings.get_string("SaveFchk").c_str(),"w");

  // Write comment
  fprintf(out,"%-80s\n","ERKALE formatted checkpoint for visualization purposes");
  char line[80];
  sprintf(line,"Created on %s.",t.current_time().c_str());
  fprintf(out,"%-80s\n",line);

  // Write the basis set info.
  write_basis(basis,out);

  // Write the orbitals
  bool restr;
  chkpt.read("Restricted",restr);
  if(restr) {
    arma::mat C;
    chkpt.read("C",C);
    write_mo("Alpha MO coefficients",basis,C,out);

    arma::vec E;
    chkpt.read("E",E);
    std::vector<double> Ev(E.n_elem);
    for(size_t i=0;i<E.n_elem;i++)
      Ev[i]=E(i);
    print("Alpha Orbital Energies",Ev,out);

    print("Number of independent functions",(int) C.n_cols,out);
  } else {
    arma::mat Ca;
    chkpt.read("Ca",Ca);
    write_mo("Alpha MO coefficients",basis,Ca,out);

    arma::vec Ea;
    chkpt.read("Ea",Ea);
    std::vector<double> Ev(Ea.n_elem);
    for(size_t i=0;i<Ea.n_elem;i++)
      Ev[i]=Ea(i);
    print("Alpha Orbital Energies",Ev,out);

    arma::mat Cb;
    chkpt.read("Cb",Cb);
    write_mo("Beta MO coefficients",basis,Cb,out);

    arma::vec Eb;
    chkpt.read("Eb",Eb);
    for(size_t i=0;i<Eb.n_elem;i++)
      Ev[i]=Eb(i);
    print("Beta Orbital Energies",Ev,out);

    print("Number of independent functions",(int) Ca.n_cols,out);
  }

  // Write the number of electrons
  int Nel;
  chkpt.read("Nel",Nel);
  print("Number of electrons",Nel,out);
  chkpt.read("Nel-a",Nel);
  print("Number of alpha electrons",Nel,out);
  chkpt.read("Nel-b",Nel);
  print("Number of beta electrons",Nel,out);

  // Save density matrix
  arma::mat P;
  chkpt.read("P",P);
  write_density("Total SCF Density",basis,P,out);

  // Save spin density
  if(!restr) {
    arma::mat Pa, Pb;
    chkpt.read("Pa",Pa);
    chkpt.read("Pb",Pb);

    arma::mat S=Pa-Pb;
    write_density("Spin SCF Density",basis,S,out);
  }

  // Close output file
  fclose(out);

  printf("Formatted checkpoint file created in %s.\n",t.elapsed().c_str());

  if(usegz || usexz || usebz2 || uselzma) {
    t.set();

    if(usegz) {
      std::string name=savename.substr(0,savename.size()-3);
      std::ostringstream cmd;
      cmd << "mv " << savename << " " << name;
      int error=system(cmd.str().c_str());
      if(error) throw std::runtime_error("Unable to rename file.\n");
      gzcmd+=name;
      error=system(gzcmd.c_str());
      if(error) throw std::runtime_error("Unable to compress file.\n");
    }

    if(usexz) {
      std::string name=savename.substr(0,savename.size()-3);
      std::ostringstream cmd;
      cmd << "mv " << savename << " " << name;
      int error=system(cmd.str().c_str());
      if(error) throw std::runtime_error("Unable to rename file.\n");
      xzcmd+=name;
      error=system(xzcmd.c_str());
      if(error) throw std::runtime_error("Unable to compress file.\n");
    }

    if(usebz2) {
      std::string name=savename.substr(0,savename.size()-4);
      std::ostringstream cmd;
      cmd << "mv " << savename << " " << name;
      int error=system(cmd.str().c_str());
      if(error) throw std::runtime_error("Unable to rename file.\n");
      bz2cmd+=name;
      error=system(bz2cmd.c_str());
      if(error) throw std::runtime_error("Unable to compress file.\n");
    }

    if(uselzma) {
      std::string name=savename.substr(0,savename.size()-5);
      std::ostringstream cmd;
      cmd << "mv " << savename << " " << name;
      int error=system(cmd.str().c_str());
      if(error) throw std::runtime_error("Unable to rename file.\n");
      lzmacmd+=name;
      error=system(lzmacmd.c_str());
      if(error) throw std::runtime_error("Unable to compress file.\n");
    }

    printf("File compressed in %s.\n",t.elapsed().c_str());
  }

}

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Gaussian(TM) interface from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Gaussian(TM) interface from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  settings.add_string("LoadFchk","Gaussian formatted checkpoint file to load","");
  settings.add_string("SaveFchk","Gaussian formatted checkpoint file to load","");
  settings.add_double("FchkTol","Tolerance for deviation in orbital orthonormality and density matrix",1e-8);
  settings.add_bool("Renormalize","Renormalize density matrix?",false);
  settings.add_bool("Reorthonormalize","Reorthonormalize orbitals?",false);
  settings.add_string("LoadChk","Save results to ERKALE checkpoint","");
  settings.add_string("SaveChk","Save results to ERKALE checkpoint","");

  // Parse settings
  settings.parse(argv[1]);
  settings.print();

  bool loadfchk=(settings.get_string("LoadFchk")!="");
  bool savefchk=(settings.get_string("SaveFchk")!="");
  bool loadchk=(settings.get_string("LoadChk")!="");
  bool savechk=(settings.get_string("SaveChk")!="");
  double tol=settings.get_double("FchkTol");

  if(loadfchk && savechk && !loadchk && !savefchk)
    load_fchk(tol);
  else if(!loadfchk && !savechk && loadchk && savefchk)
    save_fchk();
  else
    throw std::runtime_error("Conflicting settings!\n");

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
