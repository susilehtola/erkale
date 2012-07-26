/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
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

#include "basis.h"
#include "checkpoint.h"
#include "mathf.h"
#include "storage.h"
#include "stringutil.h"
#include "timer.h"
#include "fchkpt_tools.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
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


void load_fchk(const Settings & set) {
  Timer t;

  // Read in checkpoint
  Storage stor=parse_fchk(set.get_string("LoadFchk"));
  //  stor.print(false);
  printf("Read in formatted checkpoint in %s.\n",t.elapsed().c_str());

  // Construct basis set
  BasisSet basis=form_basis(stor);
  //  basis.print(true);

  // Construct density matrix
  arma::mat P=form_density(stor);

  // Form orbitals
  arma::mat Ca, Cb;
  bool restr=false;
  Ca=form_orbital(stor,"Alpha MO coefficients");
  try {
    Cb=form_orbital(stor,"Beta MO coefficients");
  } catch(std::runtime_error) {
    // Restricted checkpoint
    restr=true;
  }

  // Check that everything is OK
  t.set();
  arma::mat S=basis.overlap();
  printf("\nComputed overlap matrix in %s.\n",t.elapsed().c_str());

  int Nel=stor.get_int("Number of electrons");
  double neldiff=arma::trace(P*S);
  neldiff-=Nel;
  if(fabs(neldiff)/Nel>1e-8) {
    std::ostringstream oss;
    oss << "\nNumber of electrons and trace of density matrix differ by " << neldiff << "!\n";
    throw std::runtime_error(oss.str());
  }
  printf("tr PS - Nel = %.e\n",neldiff);

  // Save the result
  t.set();
  Checkpoint chkpt(set.get_string("SaveChk"),true);
  chkpt.write(basis);
  chkpt.write("P",P);

  chkpt.write("Restricted",restr);
  if(restr) {
    chkpt.write("C",Ca);
  } else {
    chkpt.write("Ca",Ca);
    chkpt.write("Cb",Cb);
  }

  chkpt.write("Converged",1);

  // Write number of electrons
  int Nela=stor.get_int("Number of alpha electrons");
  int Nelb=stor.get_int("Number of beta electrons");
  chkpt.write("Nel",Nel);
  chkpt.write("Nel-a",Nela);
  chkpt.write("Nel-b",Nelb);

  printf("\nERKALE checkpoint saved in %s.\n",t.elapsed().c_str());
}

void save_fchk(const Settings & set) {
  Timer t;

  // Load checkpoint
  Checkpoint chkpt(set.get_string("LoadChk"),false);

  // Construct basis
  BasisSet basis;
  chkpt.read(basis);
  //  basis.print(true);

  // Open output file.
  FILE *out=fopen(set.get_string("SaveFchk").c_str(),"w");

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
}

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2011.\n");

  print_license();

  if(argc>2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  Settings set;
  set.add_string("LoadFchk","Gaussian formatted checkpoint file to load","");
  set.add_string("SaveFchk","Gaussian formatted checkpoint file to load","");
  set.add_string("LoadChk","Save results to ERKALE checkpoint","");
  set.add_string("SaveChk","Save results to ERKALE checkpoint","");

  // Parse settings
  if(argc==2)
    set.parse(argv[1]);

  bool loadfchk=(set.get_string("LoadFchk")!="");
  bool savefchk=(set.get_string("SaveFchk")!="");
  bool loadchk=(set.get_string("LoadChk")!="");
  bool savechk=(set.get_string("SaveChk")!="");

  if(loadfchk && savechk && !loadchk && !savefchk)
    load_fchk(set);
  else if(!loadfchk && !savechk && loadchk && savefchk)
    save_fchk(set);
  else
    throw std::runtime_error("Conflicting settings!\n");

  return 0;
}
