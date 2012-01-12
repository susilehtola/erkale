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
    throw std::runtime_error("Unable to open input file.\n");
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
std::vector<size_t> eg_indarr(const std::vector<int> shtype, size_t Nbf) {
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
	// Normal ordering as in ERKALE

	int am=shtype[ish];
	for(int i=0; i<=am; i++) {
	  int nx = am - i;
	  for(int j=0; j<=i; j++) {
	    int ny = i-j;
	    int nz = j;

	    hlp.l=nx;
	    hlp.m=ny;
	    hlp.n=nz;
	    sh.push_back(hlp);
	  }
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
std::vector<size_t> ge_indarr(const std::vector<int> shtype, size_t Nbf) {
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


arma::mat form_density(const Storage & stor) {
  // Check what kind of densities are available
  std::vector<std::string> keys=stor.find_double_vec("Density");
  for(size_t i=keys.size()-1;i<keys.size();i--)
    if(splitline(keys[i])[0]!="Total")
      keys.erase(keys.begin()+i);

  /*
  printf("Available densities\n");
  for(size_t i=0;i<keys.size();i++)
    printf("\t%s\n",keys[i].c_str());
  */

  // The density matrix
  std::vector<double> dens;
  if(keys.size()==1)
    dens=stor.get_double_vec(keys[0]);
  else {
    if(splitline(keys[0])[1]!="SCF")
      dens=stor.get_double_vec(keys[0]);
    else if(splitline(keys[1])[1]!="SCF")
      dens=stor.get_double_vec(keys[1]);
    else {
      ERROR_INFO();
      throw std::runtime_error("Could not find density matrix to use!\n");
    }
  }

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
arma::mat form_orbital(const Storage & stor, const std::string & name) {
  // Amount of basis functions
  size_t Nbf=stor.get_int("Number of basis functions");
  // Amount of orbitals
  size_t Nmo=stor.get_int("Number of independent functions");

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
      // SP shell. Add S shell first.

      std::vector<contr_t> Cs(C);
      for(int ip=0;ip<nprim[ish];ip++) {
	Cs[ip].c=spcoeff[iprim+ip];
      }

      bas.add_shell(nucind,0,false,Cs,false);
    }

    // Add shell. Angular momentum is
    int am=abs(shtypes[ish]);
    // Use spherical harmonics on shell?
    bool lm=(shtypes[ish]<-1);

    GaussianShell sh(am,lm,C);
    bas.add_shell(nucind,sh,false);

    // Increment primitive index
    iprim+=nprim[ish];
  }

  // Finalize basis set, converting contraction coefficients.
  bas.finalize(true);

  // Check that we get the same amount of basis functions
  if((int) bas.get_Nbf() != stor.get_int("Number of basis functions")) {
    std::ostringstream oss;
    oss << "\nERKALE basis has " << bas.get_Nbf() << " functions while Gaussian has " << stor.get_int("Number of basis functions") << " functions!\n";
    throw std::runtime_error(oss.str());
  }

  return bas;
}

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

  double neldiff=arma::trace(P*S);
  neldiff-=stor.get_int("Number of electrons");
  if(fabs(neldiff)>1e-8) {
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
