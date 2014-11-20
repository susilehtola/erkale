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

#include "basis.h"
#include "checkpoint.h"
#include "fchkpt_tools.h"
#include "elements.h"
#include "mathf.h"
#include "storage.h"
#include "stringutil.h"
#include "timer.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif


/// Parse formatted checkpoint file
Storage parse_fchk(const std::string & name) {
  // Returned storage
  Storage ret;

  // Input file
  FILE *in;

  // Handle also compressed files
  const char gzcmd[]="zcat ";
  bool usegz=false;
  if(strstr(name.c_str(),".gz")!=NULL)
    usegz=true;

  const char xzcmd[]="xzcat ";
  bool usexz=false;
  if(strstr(name.c_str(),".xz")!=NULL)
    usexz=true;

  const char bz2cmd[]="bzcat ";
  bool usebz2=false;
  if(strstr(name.c_str(),".bz2")!=NULL)
    usebz2=true;

  const char lzmacmd[]="lzcat ";
  bool uselzma=false;
  if(strstr(name.c_str(),".lzma")!=NULL)
    uselzma=true;

  /* Open the file */
  if(usegz) {
    char cmd[strlen(gzcmd)+name.size()+1];
    sprintf(cmd,"%s%s",gzcmd,name.c_str());
    in=popen(cmd,"r");
  } else if(usexz) {
    char cmd[strlen(xzcmd)+name.size()+1];
    sprintf(cmd,"%s%s",xzcmd,name.c_str());
    in=popen(cmd,"r");
  } else if(usebz2) {
    char cmd[strlen(bz2cmd)+name.size()+1];
    sprintf(cmd,"%s%s",bz2cmd,name.c_str());
    in=popen(cmd,"r");
  } else if(uselzma) {
    char cmd[strlen(lzmacmd)+name.size()+1];
    sprintf(cmd,"%s%s",lzmacmd,name.c_str());
    in=popen(cmd,"r");
  } else
    in=fopen(name.c_str(),"r");

  if(in==NULL) {
    std::ostringstream oss;
    oss << "Unable to open input checkpoint file \"" << name << "\".\n";
    throw std::runtime_error(oss.str());
  }

  // Line number
  size_t iline=0;

  // The name of the entry
  std::string entryname;
  // Number of entries to read in
  size_t N=0;

  // Are in the middle of reading an integer vector
  bool intvec=false;
  std::vector<int> intv;

  // Are we in the middle of reading a double vector?
  bool doublevec=false;
  std::vector<double> dblv;

  while(true) {
    // Input line
    std::string line;

    try {
      // Get line from input file
      line=readline(in);
      // Increment line number
      iline++;
    } catch(std::runtime_error) {
      break;
    }

    // Split the input into fields
    std::vector<std::string> words=splitline(line);

    // Skip first two lines.
    if(iline<3)
      continue;

    // Read in numbers?
    if(intvec) {
      for(size_t i=0;i<words.size();i++)
	intv.push_back(readint(words[i]));

      N-=words.size();
      if(N==0) {
	intvec=false;
	// Add to stack
	int_vec_st_t help;
	help.name=entryname;
	help.val=intv;
	ret.add(help);
	// Clear vector
	intv.clear();
      }
    } else if(doublevec) {
      for(size_t i=0;i<words.size();i++)
	dblv.push_back(readdouble(words[i]));
      N-=words.size();
      if(N==0) {
	doublevec=false;
	// Add to stack
	double_vec_st_t help;
	help.name=entryname;
	help.val=dblv;
	ret.add(help);
	// Clear vector
	dblv.clear();
      }
    } else {
      // New entry. Is it a vector?

      if(words[words.size()-2]=="N=") {
	// Yes, it is. Number of entries to read in is
	N=readint(words[words.size()-1]);
	// Read integers or doubles?
	if(words[words.size()-3]=="R")
	  doublevec=true;
	else if(words[words.size()-3]=="I")
	  intvec=true;
	else {
	  ERROR_INFO();
	  throw std::runtime_error("Should not end up here!\n");
	}

	// Entry name is
	entryname="";
	for(size_t i=0;i<words.size()-3;i++) {
	  entryname+=words[i];
	  if(i<words.size()-4)
	    entryname+=" ";
	}
      } else {
	// Not reading a vector.

	// Entry name is
	entryname="";
	for(size_t i=0;i<words.size()-2;i++) {
	  entryname+=words[i];
	  if(i<words.size()-3)
	    entryname+=" ";
	}

	// Read integer or double?
	if(words[words.size()-2]=="I") {
	  // Integer.
	  int_st_t help;
	  help.val=readint(words[words.size()-1]);
	  help.name=entryname;
	  ret.add(help);
	} else if(words[words.size()-2]=="R") {
	  // Double.
	  double_st_t help;
	  help.val=readdouble(words[words.size()-1]);
	  help.name=entryname;
	  ret.add(help);
	} else {
	  ERROR_INFO();
	  throw std::runtime_error("Should have not ended up here!\n");
	}
      }
    }
  }

  // Close input
  if(usegz || usexz || usebz2 || uselzma)
    pclose(in);
  else
    fclose(in);

  return ret;
}

/// Form the ERKALE to Gaussian index conversion array
 std::vector<size_t> eg_indarr(const std::vector<int> & shtype, size_t Nbf) {
  // Returned index list
  std::vector<size_t> idx(Nbf,0);

  // Loop over shells
  size_t g_idx=0; // Index of Gaussian function
  size_t e_idx=0; // Index of ERKALE function

  for(size_t ish=0;ish<shtype.size();ish++) {
    // Determine shell type.


    // S shell
    if(shtype[ish]==0) {
      idx[e_idx++]=g_idx++;


      // (Cartesian) P shell. px, py and pz
    } else if(shtype[ish]==1) {

      for(size_t i=0;i<3;i++) {
	idx[e_idx++]=g_idx++;
      }


      // SP shell: first S, then px, py and pz
    } else if(shtype[ish]==-1) {

      for(size_t i=0;i<4;i++) {
	idx[e_idx++]=g_idx++;
      }


      // General spherical harmonics shell.
    } else if(shtype[ish]<-1) {

      // l value is
      int l=-shtype[ish];

      // Ordering in m in Gaussian is 0,+1,-1,+2,-2, ...
      // whereas in ERKALE it is -m,-m+1,...,m-1,m

      // Add m=0
      idx[e_idx+l]=g_idx++;
      // and the rest
      for(int absm=1;absm<=l;absm++)
	for(int sign=1;sign>=-1;sign-=2) {
	  // Value of m is
	  int m=sign*absm;
	  // ERKALE index is thus
	  idx[e_idx+l+m]=g_idx++;
	}

      // Increment ERKALE index
      e_idx+=2*l+1;


      // Other types of cartesian shells
    } else {

      // Create list of cartesians
      std::vector<shellf_t> sh;

      shellf_t hlp;
      hlp.relnorm=1.0;

      if(shtype[ish]==2) {
	// Cartesian D shell.

	// x^2
	hlp.l=2;
	hlp.m=0;
	hlp.n=0;
	sh.push_back(hlp);

	// y^2
	hlp.l=0;
	hlp.m=2;
	hlp.n=0;
	sh.push_back(hlp);

	// z^2
	hlp.l=0;
	hlp.m=0;
	hlp.n=2;
	sh.push_back(hlp);

	// xy
	hlp.l=1;
	hlp.m=1;
	hlp.n=0;
	sh.push_back(hlp);

	// xz
	hlp.l=1;
	hlp.m=0;
	hlp.n=1;
	sh.push_back(hlp);

	// yz
	hlp.l=0;
	hlp.m=1;
	hlp.n=1;
	sh.push_back(hlp);
      } else if(shtype[ish]==3) {
	// Cartesian F shell

	// x^3
	hlp.l=3;
	hlp.m=0;
	hlp.n=0;
	sh.push_back(hlp);

	// y^3
	hlp.l=0;
	hlp.m=3;
	hlp.n=0;
	sh.push_back(hlp);

	// z^3
	hlp.l=0;
	hlp.m=0;
	hlp.n=3;
	sh.push_back(hlp);

	// xy^2
	hlp.l=1;
	hlp.m=2;
	hlp.n=0;
	sh.push_back(hlp);

	// x^2y
	hlp.l=2;
	hlp.m=1;
	hlp.n=0;
	sh.push_back(hlp);

	// x^2z
	hlp.l=2;
	hlp.m=0;
	hlp.n=1;
	sh.push_back(hlp);

	// xz^2
	hlp.l=1;
	hlp.m=0;
	hlp.n=2;
	sh.push_back(hlp);

	// yz^2
	hlp.l=0;
	hlp.m=1;
	hlp.n=2;
	sh.push_back(hlp);

	// y^2z
	hlp.l=0;
	hlp.m=2;
	hlp.n=1;
	sh.push_back(hlp);

	// xyz
	hlp.l=1;
	hlp.m=1;
	hlp.n=1;
	sh.push_back(hlp);

      } else {
	// Order is like in ERKALE but different

	int am=shtype[ish];
	for(int nx=0;nx<=am;nx++)
	  for(int ny=0;ny<=am-nx;ny++) {
	    int nz=am-nx-ny;
	    hlp.l=nx;
	    hlp.m=ny;
	    hlp.n=nz;
	    sh.push_back(hlp);
	  }
      }

      // Add shells
      for(size_t i=0;i<sh.size();i++)
	idx[e_idx+getind(sh[i].l,sh[i].m,sh[i].n)]=g_idx++;
      // Increment ERKALE index
      e_idx+=sh.size();
    }
  }

  return idx;
}

/// Form the ERKALE to Gaussian index conversion array
std::vector<size_t> eg_indarr(const Storage & stor) {
  // Amount of basis functions
  size_t Nbf=stor.get_int("Number of basis functions");

  // Get shell types
  std::vector<int> shtype=stor.get_int_vec("Shell types");

  return eg_indarr(shtype,Nbf);
}

/// Form the Gaussian to ERKALE index conversion array
std::vector<size_t> ge_indarr(const std::vector<int> & shtype, size_t Nbf) {
  // Erkale to Gaussian
  std::vector<size_t> eg=eg_indarr(shtype, Nbf);

  // Gaussian to ERKALE
  std::vector<size_t> ret(eg.size());
  for(size_t i=0;i<eg.size();i++)
    ret[eg[i]]=i;

  return ret;
}

/// Form the Gaussian to ERKALE index conversion array
std::vector<size_t> ge_indarr(const Storage & stor) {
  // Amount of basis functions
  size_t Nbf=stor.get_int("Number of basis functions");

  // Get shell types
  std::vector<int> shtype=stor.get_int_vec("Shell types");

  return ge_indarr(shtype,Nbf);
}


arma::mat form_density(const Storage & stor, bool spin, bool scf) {
  // Check what kind of densities are available
  std::vector<std::string> keys=stor.find_double_vec("Density");
  if(spin) {
    for(size_t i=keys.size()-1;i<keys.size();i--)
      if(splitline(keys[i])[0]!="Spin")
	keys.erase(keys.begin()+i);
  } else {
    for(size_t i=keys.size()-1;i<keys.size();i--)
      if(splitline(keys[i])[0]!="Total")
	keys.erase(keys.begin()+i);
  }

  /*
  printf("Available densities\n");
  for(size_t i=0;i<keys.size();i++)
    printf("\t%s\n",keys[i].c_str());
  */

  // The density matrix
  std::string key;
  if(keys.size()==1)
    // Only single density matrix available
    key=keys[0];

  else if(keys.size()==2) {
    // Which index to pick?
    int ind;

    // Find SCF density
    if(splitline(keys[0])[1]=="SCF")
      ind=0;
    else
      ind=1;

    // Post-HF density?
    if(!scf)
      ind=!ind;

    // Store key
    key=keys[ind];

  } else {
    ERROR_INFO();
    throw std::runtime_error("Could not find density matrix to use!\n");
  }

  return form_density(stor,key);
}


arma::mat form_density(const Storage & stor, const std::string & key) {
  // Get density
  std::vector<double> dens=stor.get_double_vec(key);

  // Amount of basis functions
  size_t Nbf=stor.get_int("Number of basis functions");
  arma::mat P(Nbf,Nbf);
  P.zeros();

  // Get index converter
  std::vector<size_t> idx=ge_indarr(stor);

  // Fill matrix
  for(size_t i=0;i<Nbf;i++) {
    // ERKALE index
    size_t ie=idx[i];

    for(size_t j=0;j<=i;j++) {
      // ERKALE index
      size_t je=idx[j];

      P(ie,je)=dens[(i*(i+1))/2+j];
      P(je,ie)=P(ie,je);
    }
  }

  /*
    printf("\n\n");
    printf("Indices\nGaussian\tERKALE\n");
    for(size_t i=0;i<idx.size();i++)
    printf("\t%3i -> %3i\n",(int) i+1,(int) idx[i]+1);
  */

  return P;
}

/// Get orbital matrix
arma::mat form_orbital_C(const Storage & stor, const std::string & name) {
  // Amount of basis functions
  size_t Nbf=stor.get_int("Number of basis functions");
  // Amount of orbitals
  size_t Nmo;
  try {
    Nmo=stor.get_int("Number of independent functions");
  } catch(std::runtime_error) {
    // G03 has spelling error
    Nmo=stor.get_int("Number of independant functions");
  }

  // Get index converter
  std::vector<size_t> idx=ge_indarr(stor);

  // Get orbital coefficients
  std::vector<double> c=stor.get_double_vec(name);

  arma::mat C(Nbf,Nmo);
  C.zeros();

  if(c.size()!=Nmo*Nbf) {
    ERROR_INFO();
    throw std::runtime_error("Not the right amount of orbital coefficients!\n");
  }

  for(size_t imo=0;imo<Nmo;imo++)
    for(size_t ibf=0;ibf<Nbf;ibf++) {
      // ERKALE index of basis function is
      size_t ie=idx[ibf];

      // Store coefficient
      C(ie,imo)=c[imo*Nbf+ibf];
    }

  return C;
}

/// Get orbital energies
arma::vec form_orbital_E(const Storage & stor, const std::string & name) {
  // Amount of orbitals
  size_t Nmo;
  try {
    Nmo=stor.get_int("Number of independent functions");
  } catch(std::runtime_error) {
    // G03 has spelling error
    Nmo=stor.get_int("Number of independant functions");
  }

  // Get the energies
  std::vector<double> e=stor.get_double_vec(name);

  if(e.size()!=Nmo) {
    ERROR_INFO();
    throw std::runtime_error("Not the right amount of orbital energies!\n");
  }

  arma::vec E(Nmo);
  for(size_t imo=0;imo<Nmo;imo++)
    E(imo)=e[imo];

  return E;
}

/// Form the basis set from the checkpoint file
BasisSet form_basis(const Storage & stor) {

  // Get shell types.
  std::vector<int> shtypes=stor.get_int_vec("Shell types");
  // Get number of primitives per shell
  std::vector<int> nprim=stor.get_int_vec("Number of primitives per shell");
  // Get shell to atom map
  std::vector<int> shatom=stor.get_int_vec("Shell to atom map");

  // Get exponents
  std::vector<double> exps=stor.get_double_vec("Primitive exponents");
  // Get contraction coefficients
  std::vector<double> coeff=stor.get_double_vec("Contraction coefficients");

  // Get SP contraction coefficients
  std::vector<double> spcoeff;
  try {
    spcoeff=stor.get_double_vec("P(S=P) Contraction coefficients");
  } catch(std::runtime_error) {
    // Not using SP coefficients.
  }

  // Coordinates of shells
  std::vector<double> coords=stor.get_double_vec("Coordinates of each shell");

  // Atom numbers
  std::vector<int> nuctypes=stor.get_int_vec("Atomic numbers");
  // and coordinates
  std::vector<double> nuccoord=stor.get_double_vec("Current cartesian coordinates");

  // Returned basis set.
  BasisSet bas;

  // Add the nuclei.
  for(size_t i=0;i<nuctypes.size();i++) {
    nucleus_t nuc;
    nuc.r.x=nuccoord[3*i];
    nuc.r.y=nuccoord[3*i+1];
    nuc.r.z=nuccoord[3*i+2];
    nuc.Z=nuctypes[i];
    nuc.ind=i;
    nuc.bsse=false;
    nuc.symbol=element_symbols[nuc.Z];

    bas.add_nucleus(nuc);
  }

  // Primitive index
  size_t iprim=0;

  // Add the shells.
  for(size_t ish=0;ish<shtypes.size();ish++) {
    // Construct contraction

    // Nuclear index is
    size_t nucind=shatom[ish]-1;

    std::vector<contr_t> C(nprim[ish]);
    for(int ip=0;ip<nprim[ish];ip++) {
      C[ip].z=exps[iprim+ip];
      C[ip].c=coeff[iprim+ip];
    }

    if(shtypes[ish]==-1) {
      // SP shell.

      // p contraction coefficients are in P(S=P) block
      std::vector<contr_t> Cp(C);
      for(int ip=0;ip<nprim[ish];ip++) {
	Cp[ip].c=spcoeff[iprim+ip];
      }

      // s comes first
      bas.add_shell(nucind,0,false,C,false);
      // and then p
      bas.add_shell(nucind,1,false,Cp,false);
    } else {
      // Add shell. Angular momentum is
      int am=abs(shtypes[ish]);
      // Use spherical harmonics on shell?
      bool lm=(shtypes[ish]<-1);
      
      // Add shell
      bas.add_shell(nucind,am,lm,C,false);
    }
      
    // Increment primitive index
    iprim+=nprim[ish];
  }

  // Finalize basis set, converting contraction coefficients and normalizing.
  bas.finalize(true,true);

  // Check that we get the same amount of basis functions
  if((int) bas.get_Nbf() != stor.get_int("Number of basis functions")) {
    std::ostringstream oss;
    oss << "\nERKALE basis has " << bas.get_Nbf() << " functions while Gaussian has " << stor.get_int("Number of basis functions") << " functions!\n";
    throw std::runtime_error(oss.str());
  }

  return bas;
}
