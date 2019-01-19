/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Copyright © 2015 The Regents of the University of California
 * All Rights Reserved
 *
 * Written by Susi Lehtola, Lawrence Berkeley National Laboratory
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "../checkpoint.h"
#include "../density_fitting.h"
#include "../erichol.h"
#include "../localization.h"
#include "../timer.h"
#include "../mathf.h"
#include "../linalg.h"
#include "../eriworker.h"
#include "../settings.h"
#include "../stringutil.h"

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

Settings settings;

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - Checkpoints from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Checkpoints from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc!=2) {
    printf("Usage: $ %s (runfile)\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  Timer t;

  // Parse settings
  settings.add_string("LoadChk","Checkpoint file that contains basis set etc","erkale.chk");
  settings.add_string("SaveChk","Checkpoint to which orbitals are saved","chkpt.chk");
  settings.add_string("Model","Model from which orbitals are","");
  settings.add_bool("Binary","Use binary I/O?",true);
  settings.add_string("Deleted","Add in deleted orbitals from file","");
  settings.add_bool("NO","Are these natural orbitals?",false);

  settings.parse(argv[1]);
  settings.print();

  // Load checkpoint
  std::string loadchk(settings.get_string("LoadChk"));
  std::string savechk(settings.get_string("SaveChk"));
  std::string model(settings.get_string("Model"));
  std::string del(settings.get_string("Deleted"));
  bool binary(settings.get_bool("Binary"));
  bool NO(settings.get_bool("NO"));

  if(savechk.size()) {
    // Copy checkpoint data
    std::ostringstream cmd;
    cmd << "cp " << loadchk << " " << savechk;
    if(system(cmd.str().c_str()))
      throw std::runtime_error("Error copying checkpoint file.\n");
  }

  // Save checkpoint
  Checkpoint chkpt(savechk,true,false);

  // File type
  arma::file_type atype = binary ? arma::arma_binary : arma::raw_ascii;

  // Load orbitals
  arma::mat Ca, Cb;
  arma::mat Pa, Pb;

  if(NO) {
    // MOs are in
    Ca.load("Ca.dat",atype);
    Cb.load("Cb.dat",atype);

    // Load density matrices
    Pa.load("Pa_" + model + "_MO.dat",atype);
    Pb.load("Pb_" + model + "_MO.dat",atype);
    // Convert to AO basis
    Pa=Ca*Pa*arma::trans(Ca);
    Pb=Cb*Pb*arma::trans(Cb);

    // and the NO coefficients are in
    arma::mat Wa, Wb;
    Wa.load("Ca_"+model+"_MO.dat",atype);
    Wb.load("Cb_"+model+"_MO.dat",atype);

    // Rotate MOs to NO basis
    Ca*=Wa;
    Cb*=Wb;

  } else {
    Ca.load("Ca_"+model+".dat",atype);
    Cb.load("Cb_"+model+".dat",atype);

    // Load density matrices
    Pa.load("Pa_" + model + "_MO.dat",atype);
    Pb.load("Pb_" + model + "_MO.dat",atype);
    // Convert to AO basis
    Pa=Ca*Pa*arma::trans(Ca);
    Pb=Cb*Pb*arma::trans(Cb);
  }

  // Was earlier checkpoint restricted?
  int orestr;
  chkpt.read("Restricted",orestr);
  if(orestr) {
    chkpt.remove("C");
  } else {
    chkpt.remove("Ca");
    chkpt.remove("Cb");
  }

  // Save orbitals
  int restr;
  if(arma::norm(Ca-Cb,2)==0.0) {
    restr=1;
    chkpt.remove("Ha");
    chkpt.remove("Hb");
    chkpt.write("C",Ca);
    chkpt.write("P",Pa+Pb);
  } else {
    restr=0;
    chkpt.remove("H");
    chkpt.write("Ca",Ca);
    chkpt.write("Cb",Cb);
    chkpt.write("P",Pa+Pb);
    chkpt.write("Pa",Pa);
    chkpt.write("Pb",Pb);
  }
  chkpt.write("Restricted",restr);

  if(orestr && !restr) {
    arma::vec E;
    chkpt.read("E",E);
    chkpt.remove("E");
    chkpt.write("Ea",E);
    chkpt.write("Eb",E);
  } else if(!orestr && restr) {
    arma::vec E;
    chkpt.read("Ea",E);
    chkpt.remove("Ea");
    chkpt.remove("Eb");
    chkpt.write("E",E);
  }

  if(NO) {
    if(restr) {
      // Load occupation numbers
      arma::vec E;
      E.load("NOON_" + model + "_alpha.dat",atype);
      chkpt.write("E",E);
    } else {
      // Load occupation numbers
      arma::vec Ea, Eb;
      Ea.load("NOON_" + model + "_alpha.dat",atype);
      Eb.load("NOON_" + model + "_beta.dat",atype);
      chkpt.write("Ea",Ea);
      chkpt.write("Eb",Eb);
    }
  }

  if(del.size()) {
    // Load orbitals from file
    arma::mat Cd;
    Cd.load(del,atype);

    arma::vec zerocc(Cd.n_cols);
    zerocc.zeros();

    if(restr) {
      arma::mat C;
      chkpt.read("C",C);

      arma::vec E;
      chkpt.read("E",E);

      arma::mat Cnew(arma::join_rows(C,Cd));
      chkpt.write("C",Cnew);

      if(NO) {
	arma::vec Enew(E.n_elem+zerocc.n_elem);
	Enew.subvec(0,E.n_elem-1)=E;
	Enew.subvec(E.n_elem,Enew.n_elem-1)=zerocc;
	chkpt.write("E",Enew);
      }
    } else {
      chkpt.read("Ca",Ca);
      chkpt.read("Cb",Cb);

      arma::vec Ea, Eb;
      chkpt.read("Ea",Ea);
      chkpt.read("Eb",Eb);

      arma::mat Canew(arma::join_rows(Ca,Cd));
      chkpt.write("Ca",Canew);
      arma::mat Cbnew(arma::join_rows(Cb,Cd));
      chkpt.write("Cb",Cbnew);

      if(NO) {
	arma::vec Eanew(Ea.n_elem+zerocc.n_elem);
	Eanew.subvec(0,Ea.n_elem-1)=Ea;
	Eanew.subvec(Ea.n_elem,Eanew.n_elem-1)=zerocc;
	chkpt.write("Ea",Eanew);

      	arma::vec Ebnew(Eb.n_elem+zerocc.n_elem);
	Ebnew.subvec(0,Eb.n_elem-1)=Eb;
	Ebnew.subvec(Eb.n_elem,Ebnew.n_elem-1)=zerocc;
	chkpt.write("Eb",Ebnew);
      }
    }

    printf("Added in %i deleted virtuals from %s.\n",(int) Cd.n_cols,del.c_str());
  }

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
