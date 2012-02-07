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

#include "mathf.h"
#include "../storage.h"
#include "stringutil.h"
#include "lmgrid.h"
#include "timer.h"

#include "emd_sto.h"

#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAXLEN 1024

/// Parse the TAPE21 file, converted into ASCII
Storage parse_tape(const std::string & name) {
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

    // Line from input file.
    std::string line;
    
    try {
      line=readline(in);
      iline++;
    } catch(std::runtime_error) {
      break;
    }

    // and split it into fields
    std::vector<std::string> words=splitline(line);

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
      // New entry. Get its name.

      // Get next line as well, which is part of the name.
      std::string line2=readline(in);
      iline++;
	
      // The name is thus
      entryname=trim(line)+" - "+trim(line2);

      //      printf("\nline %i\n%s\n%s\n",(int) iline,line.c_str(),line2.c_str());

      // Move on. Next line contains nres, ndat and data type.
      line=readline(in);
      iline++;

      std::vector<std::string> words=splitline(line);
      if(words.size()!=3) {
	std::ostringstream oss;
	oss << "Unexpected line: " << line <<"!\n";
	throw std::runtime_error(oss.str());
      }

      //      int nres=readint(words[0]);
      int ndat=readint(words[1]);
      int type=readint(words[2]);

      if(type==3) {
	// Read in string.

	// Empty the line variable.
	line="";
	// Read in the characters.
	do {
	  line2=readline(in);
	  iline++;
	  ndat-=line2.size();
	  line+=line2;
	} while(ndat>0);

	// Store the string.
	string_st_t help;
	help.name=entryname;
	help.val=line;
	ret.add(help);
      } else if(type==2) {
	// Read in floating point values.

	doublevec=true;
	N=ndat;
      } else if(type==1) {
	// Read in integers

	intvec=true;
	N=ndat;
      } else if(type==4) {
	// Read in logical values. 
	N=ndat;

	do {
	  line=readline(in);
	  for(size_t l=0;l<line.size();l++)
	    if(isalpha(line[l])) {
	      if(line[l]=='T') {
		intv.push_back(1);
		N--;
	      } else if(line[l]=='F') {
		intv.push_back(0);
		N--;
	      } else {
		ERROR_INFO();
		throw std::runtime_error("Truth value was not recognized!\n");
	      }
	    }
	  
	  int_vec_st_t help;
	  help.name=entryname;
	  help.val=intv;
	  ret.add(help);
	  // Clear vector
	  intv.clear();
	} while(N>0);
      } else {
	throw std::runtime_error("Unknown data type.\n");
      }
    }
  }
  
  // Close input
  if(usegz || usexz || usebz2 || uselzma)
    pclose(in);
  else
    fclose(in);

  //  printf("Read in %u lines from input file.\n",(unsigned int) iline);
  
  return ret;
}

std::vector< std::vector<size_t> > find_identical_functions(const Storage & stor) {
  // Get atom type list
  std::vector<int> nbptr=stor.get_int_vec("Basis - nbptr");
  // Get number of atoms of each type
  std::vector<int> nqptr=stor.get_int_vec("Geometry - nqptr");

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

std::vector< std::vector<ylmcoeff_t> > form_clm(const Storage & stor) {
  // Get the exponents
  std::vector<int> kx=stor.get_int_vec("Basis - kx");
  std::vector<int> ky=stor.get_int_vec("Basis - ky");
  std::vector<int> kz=stor.get_int_vec("Basis - kz");

  // The returned expansion
  std::vector< std::vector<ylmcoeff_t> > ret;

  // Expansions of cartesian functions. ADF is currently limited to
  // d-type, so this is enough.
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

std::vector< std::vector<RadialSlater> > form_radial(const Storage & stor) {
  // Get the exponents
  std::vector<int> kx=stor.get_int_vec("Basis - kx");
  std::vector<int> ky=stor.get_int_vec("Basis - ky");
  std::vector<int> kz=stor.get_int_vec("Basis - kz");
  std::vector<int> kr=stor.get_int_vec("Basis - kr");
  std::vector<double> z=stor.get_double_vec("Basis - alf");

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

std::vector<size_t> get_centers(const Storage & stor) {
  // Index of center
  size_t ind=0;

  // Get atom type list
  std::vector<int> nbptr=stor.get_int_vec("Basis - nbptr");
  // Get number of atoms of each type
  std::vector<int> nqptr=stor.get_int_vec("Geometry - nqptr");

  // Get number of functions
  int nbf=stor.get_int_vec("Symmetry - nfcn")[0];

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

SlaterEMDEvaluator get_eval(const Storage & stor, const arma::mat & P) {
  // Form radial functions
  std::vector< std::vector<RadialSlater > > radf=form_radial(stor);

  // Form identical functions
  std::vector< std::vector<size_t> > idf=find_identical_functions(stor);

  // Form Ylm expansion of functions
  std::vector< std::vector<ylmcoeff_t> > clm=form_clm(stor);

  // Form list of centers of functions
  std::vector<size_t> loc=get_centers(stor);

  /*
    printf("Functions centered on atoms:\n");
    for(size_t i=0;i<loc.size();i++)
    printf("%i: %i\n",(int) i+1, (int) loc[i]+1);
  */

  // Form the list of atomic coordinates
  std::vector<coords_t> coord;
  std::vector<double> clist=stor.get_double_vec("Geometry - xyz");
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

  return SlaterEMDEvaluator(radf,idf,clm,loc,coord,P);
}

/// Read overlap matrix (needs TAPE15)
arma::mat read_overlap(const Storage & stor) {
  // Read number of basis functions
  int nbf=stor.get_int_vec("Basis - naos")[0];
  arma::mat S(nbf,nbf);

  try {
    std::vector<double> sv=stor.get_double_vec("Matrices - Smat");
    for(int i=0;i<nbf;i++)
      for(int j=0;j<=i;j++) {
	S(i,j)=sv[i*(i+1)/2 + j];
	S(j,i)=S(i,j);
      }
    
    S.print("Read in overlap matrix");
  } catch(std::runtime_error) {
    S=arma::mat(0,0);
  }

  return S;
}

/// Read density matrix (needs TAPE15)
arma::mat read_density(const Storage & stor) {
  // Read number of basis functions                                                                                                                                                           
  int nbf=stor.get_int_vec("Basis - naos")[0];

  arma::mat P(nbf,nbf);

  try {
    std::vector<double> pv=stor.get_double_vec("Pmat - Pmat_A");
    for(int i=0;i<nbf;i++)
      for(int j=0;j<=i;j++) {
	P(i,j)=pv[i*(i+1)/2 + j];
	P(j,i)=P(i,j);
      }
    
    P.print("Read in density matrix");
  } catch(std::runtime_error) {
    P=arma::mat(0,0);
  }

  return P;
}


arma::mat calc_overlap(const Storage & stor) {
  // Get number of basis functions.
  int nbf=stor.get_int_vec("Symmetry - nfcn")[0];

  // Returned matrix.
  arma::mat S(nbf,nbf);
  S.zeros();

  arma::mat dS(nbf,nbf);
  dS.zeros();

  // Get radial grid
  std::vector<radial_grid_t> grid=form_radial_grid(500);
  
  // Integrand
  for(int i=0;i<nbf;i++)
    for(int j=0;j<=i;j++) {
      // Dummy density matrix
      arma::mat P(nbf,nbf);
      P.zeros();
      P(i,j)+=0.5;
      P(j,i)+=0.5;

      // Construct EMD evaluator
      SlaterEMDEvaluator eval=get_eval(stor,P);

      char fname[80];
      sprintf(fname,"ovl-%i-%i.dat",i,j);
      FILE *out=fopen(fname,"w");

      double integ=0.0;
      for(size_t ip=0;ip<grid.size();ip++) {
	double d=eval.get(grid[ip].r);
	integ+=grid[ip].w*d;
	fprintf(out,"%e %e\n",grid[ip].r,d);
      }
      fclose(out);

      printf("%i %i %e\n",i,j,integ);
      S(i,j)=integ;
      S(j,i)=integ;
    }

  return S;  
}

/// Calculate analytic overlaps. Only works with single center.
arma::mat anal_overlap(const Storage & stor) {
  // Get number of functions
  int nbf=stor.get_int_vec("Symmetry - nfcn")[0];
  arma::mat S(nbf,nbf);
  S.zeros();
  
  // Get lm exansion
  std::vector< std::vector<ylmcoeff_t> > clm=form_clm(stor);
  // Get radial functions
  std::vector< std::vector<RadialSlater> > radf=form_radial(stor);

  // Get the centers of the functions
  std::vector<size_t> cen=get_centers(stor);

  // Find identical functions
  std::vector< std::vector<size_t> > idfunc=find_identical_functions(stor);

  // Loop over equivalent functions
  for(size_t iid=0;iid<idfunc.size();iid++)
    for(size_t jid=0;jid<=iid;jid++) {

      // Compute overlap
      std::complex<double> ovl=0.0;
      for(size_t ilm=0;ilm<clm[iid].size();ilm++)
	for(size_t jlm=0;jlm<clm[jid].size();jlm++) {
	  int l=clm[iid][ilm].l;
	  int m=clm[iid][ilm].m;
	  std::complex<double> c=clm[iid][ilm].c;

	  int lp=clm[jid][jlm].l;
	  int mp=clm[jid][jlm].m;
	  std::complex<double> cp=clm[jid][jlm].c;

	  // Get zeta and n
	  int n=-1;
	  double z=-1.0;
	  for(size_t ir=0;ir<radf[iid].size();ir++)
	    if(radf[iid][ir].getl()==l) {
	      n=radf[iid][ir].getn();
	      z=radf[iid][ir].getzeta();
	    }
	  if(n==-1)
	    throw std::runtime_error("Did not find n and z!\n");

	  int np=-1;
	  double zp=-1;
          for(size_t ir=0;ir<radf[jid].size();ir++)
            if(radf[jid][ir].getl()==lp) {
              np=radf[jid][ir].getn();
              zp=radf[jid][ir].getzeta();
            }
	  if(np==-1)
	    throw std::runtime_error("Did not find np and zp!\n");

	  if((l==lp) && (m==mp))
	    ovl+=std::conj(c)*cp*pow(2,n+np+1)*pow(z,n+0.5)*pow(zp,np+0.5)/pow(z+zp,n+np+1)*fact(n+np)/sqrt(fact(2*n)*fact(2*np));	      
	}

      if(fabs(ovl.imag())>DBL_EPSILON)
	printf("Imaginary part %e.\n",ovl.imag());

      // Set overlaps
      for(size_t iif=0;iif<idfunc[iid].size();iif++)
	for(size_t jjf=0;jjf<idfunc[jid].size();jjf++) {
	  
	  // The function indices are
	  size_t mu=idfunc[iid][iif];
	  size_t nu=idfunc[jid][jjf];
	  
	  // Check that functions are on same center
	  if(cen[mu]!=cen[nu]) {
	    S(mu,nu)=1.0/0.0;
	    S(nu,mu)=1.0/0.0;
	  } else {
	    S(mu,nu)=ovl.real();
	    S(nu,mu)=ovl.real();
	  }
	}
    }
  return S;
}


arma::mat form_density(const Storage & stor) {
  // Get number of orbitals
  int nmo=stor.get_int_vec("Symmetry - norb")[0];

  // Get number of basis functions
  int nbf=stor.get_int_vec("Symmetry - nfcn")[0];

  //  printf("%i functions and %i orbitals.\n",nbf,nmo);

  // Get basis function vector
  std::vector<int> npart=stor.get_int_vec("A - npart");
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
  std::vector<double> cca=stor.get_double_vec("A - Eigen-Bas_A");
  // and occupation numbers
  std::vector<double> occa=stor.get_double_vec("A - froc_A");
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
  if(stor.get_int_vec("General - nspin")[0]==2) {
    std::vector<double> ccb=stor.get_double_vec("A - Eigen-Bas_B");
    std::vector<double> occb=stor.get_double_vec("A - froc_B");

    C.zeros();
    for(int fi=0;fi<nbf;fi++)
      for(int io=0;io<nmo;io++)
	C(npart[fi],io)=ccb[io*nbf+fi];

    for(size_t i=0;i<occa.size();i++)
      P+=occb[i]*C.col(i)*arma::trans(C.col(i));
  }

  return P;
}

int main(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - ADF EMD from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - ADF EMD from Hel, serial version.\n");
#endif
  printf("(c) Jussi Lehtola, 2010-2012.\n");
  print_license();

  if(argc!=2) {
    printf("Usage: %s file\n",argv[0]);
    return 1;
  }

  Timer t;
  t.print_time();

  // Read in the checkpoint file.
  Storage tape=parse_tape(argv[1]);
  //  tape.print(false);

  // Construct EMD evaluator
  arma::mat P=form_density(tape);

  SlaterEMDEvaluator eval=get_eval(tape,P);
  //eval.print();

  // Get number of electrons
  int Nel=tape.get_double_vec("General - electrons")[0];
  EMD emd(&eval, Nel);
  emd.initial_fill();
  emd.find_electrons();
  emd.optimize_moments(true,1e-8);
  emd.save("emd.txt");
  emd.moments("moments.txt");
  emd.compton_profile("compton.txt");
  emd.compton_profile_interp("compton-interp.txt");

  printf("Computing EMD properties took %s.\n",t.elapsed().c_str());
  
  return 0;
}
