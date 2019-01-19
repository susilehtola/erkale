/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2012
 * Copyright (c) 2010-2012, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "emd.h"
#include "emd_gto.h"
#include "emd_similarity.h"
#include "emdcube.h"
//include "gto_fourier_ylm.h"
#include "spherical_expansion.h"
#include "../solidharmonics.h"
#include "../spherical_harmonics.h"
#include "../checkpoint.h"
#include "../settings.h"
#include "../stringutil.h"
#include "../timer.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

Settings settings;

int main_guarded(int argc, char **argv) {
#ifdef _OPENMP
  printf("ERKALE - EMD from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - EMD from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();

  if(argc!=1 && argc!=2) {
    printf("Usage: $ %s (runfile)\n",argv[0]);
    return 0;
  }

  Timer t;

  // Parse settings
  settings.add_string("LoadChk","Checkpoint file to load density from","erkale.chk");
  settings.add_bool("DoEMD", "Perform calculation of isotropic EMD (moments of EMD, Compton profile)", true);
  settings.add_double("EMDTol", "Tolerance for the numerical integration of the radial EMD",1e-8);
  settings.add_string("EMDlm", "Which projection of the radial EMD to compute","");
  settings.add_bool("EMDAdapt", "Use adaptive grid to compute EMD?", true);
  settings.add_string("EMDCube", "Calculate EMD on a cube? e.g. -10:.3:10 -5:.2:4 -2:.1:3", "");
  settings.add_string("EMDOrbitals", "Compute EMD of given orbitals, e.g. 1,2,4:6","");
  settings.add_string("Similarity", "Compute similarity measure to checkpoint","");
  settings.add_string("SimilarityGrid", "Grid to use for computing similarity integrals","500 77");
  settings.add_bool("SimilarityLM", "Seminumerical computation of similarity integrals?", false);
  settings.add_int("SimilarityLmax", "Maximum angular momentum for seminumerical computation", 6);

  if(argc==2)
    settings.parse(argv[1]);
  else
    printf("Using default settings.\n");
  settings.print();

  // Get the tolerance
  double tol=settings.get_double("EMDTol");

  // Load checkpoint
  Checkpoint chkpt(settings.get_string("LoadChk"),false);

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  // Load density matrix
  arma::cx_mat P;
  arma::mat Pr, Pi;
  chkpt.read("P",Pr);
  if(chkpt.exist("P_im")) {
    chkpt.read("P_im",Pi);
    P=Pr*COMPLEX1 + Pi*COMPLEXI;
  } else
    P=Pr*COMPLEX1;

  // The projection to calculate
  int l=0, m=0;
  std::string lmstr=settings.get_string("EMDlm");
  if(lmstr.size()) {
    // Get l and m values
    std::vector<std::string> lmval=splitline(lmstr);
    if(lmval.size()!=2)
      throw std::runtime_error("Invalid specification of l and m values.\n");
    l=readint(lmval[0]);
    m=readint(lmval[1]);
  }
  bool adaptive=settings.get_bool("EMDAdapt");

  // Compute orbital EMDs?
  if(settings.get_string("EMDOrbitals")!="") {
    // Get orbitals
    std::vector<std::string> orbs=splitline(settings.get_string("EMDOrbitals"));

    // Polarized calculation?
    bool restr;
    chkpt.read("Restricted",restr);
    if(restr!= (orbs.size()==1))
      throw std::runtime_error("Invalid occupancies for spin alpha and beta!\n");

    if(l!=0)
      printf("\nComputing the (%i %+i) projection of the orbital EMD.\n",l,m);

    for(size_t ispin=0;ispin<orbs.size();ispin++) {
      // Indices of orbitals to include.
      std::vector<size_t> idx=parse_range(orbs[ispin]);
      // Change into C++ indexing
      for(size_t i=0;i<idx.size();i++)
	idx[i]--;

      // Read orbitals
      arma::mat C;
      if(restr)
	chkpt.read("C",C);
      else {
	if(ispin==0)
	  chkpt.read("Ca",C);
	else
	  chkpt.read("Cb",C);
      }

      for(size_t i=0;i<idx.size();i++) {
	// Names of output files
        std::ostringstream emdname;
        std::ostringstream momname;
        std::ostringstream Jname;
        std::ostringstream Jintname;

        std::ostringstream suffix;
	if(l>0)
	  suffix << "_" << l << "_" << m;
        suffix << ".txt";

	if(restr) {
          emdname << "emd-" << idx[i]+1 << suffix.str();
          momname << "moments-" << idx[i]+1 << suffix.str();
          Jname << "compton-" << idx[i]+1 << suffix.str();
          Jintname << "compton-interp-" << idx[i]+1 << suffix.str();
	} else {
	  if(ispin==0) {
            emdname << "emd-a-" << idx[i]+1 << suffix.str();
            momname << "moments-a-" << idx[i]+1 << suffix.str();
            Jname << "compton-a-" << idx[i]+1 << suffix.str();
            Jintname << "compton-interp-a-" << idx[i]+1 << suffix.str();
	  } else {
            emdname << "emd-b-" << idx[i]+1 << suffix.str();
            momname << "moments-b-" << idx[i]+1 << suffix.str();
            Jname << "compton-b-" << idx[i]+1 << suffix.str();
            Jintname << "compton-interp-b-" << idx[i]+1 << suffix.str();
	  }
	}

	// Generate dummy density matrix
	arma::cx_mat Pdum=C.col(idx[i])*arma::trans(C.col(idx[i]))*COMPLEX1;

	Timer temd;

	GaussianEMDEvaluator *poseval=new GaussianEMDEvaluator(basis,Pdum,l,std::abs(m));
	GaussianEMDEvaluator *negeval;
	if(m!=0)
	  negeval=new GaussianEMDEvaluator(basis,Pdum,l,-std::abs(m));
	else
	  negeval=NULL;

	EMD emd(poseval, negeval, 1, l, m);
	if(adaptive) {
	  emd.initial_fill();
	  if(l==0 && m==0) emd.find_electrons();
	  emd.optimize_moments(true,tol);
	} else
	  emd.fixed_fill();

	emd.save(emdname.str());
	emd.moments(momname.str());
	if(l==0 && m==0) {
	  emd.compton_profile(Jname.str());
	  emd.compton_profile_interp(Jintname.str());
	}

	delete poseval;
	if(m!=0) delete negeval;
      }
    }
  }

  if(settings.get_bool("DoEMD")) {
    t.print_time();

    printf("\nCalculating EMD properties.\n");
    printf("Please read and cite the reference:\n%s\n%s\n%s\n",	\
	   "J. Lehtola, M. Hakala, J. Vaara and K. Hämäläinen",		\
	   "Calculation of isotropic Compton profiles with Gaussian basis sets", \
	   "Phys. Chem. Chem. Phys. 13 (2011), pp. 5630 - 5641.");

    if(l!=0)
      printf("\nComputing the (%i %+i) projection of the EMD.\n",l,m);
    else
      printf("\nComputing the isotropic projection of the EMD.\n");

    // Amount of electrons is
    int Nel;
    chkpt.read("Nel",Nel);

    // Construct EMD evaluators
    Timer temd;
    GaussianEMDEvaluator *poseval=new GaussianEMDEvaluator(basis,P,l,std::abs(m));
    GaussianEMDEvaluator *negeval;
    if(m!=0)
      negeval=new GaussianEMDEvaluator(basis,P,l,-std::abs(m));
    else
      negeval=NULL;

    temd.set();
    EMD emd(poseval, negeval, Nel, l, m);
    if(adaptive) {
      emd.initial_fill();
      if(l==0 && m==0) emd.find_electrons();
      emd.optimize_moments(true,tol);
    } else
      emd.fixed_fill();

    if(l==0 && m==0) {
      emd.save("emd.txt");
      emd.moments("moments.txt");
      emd.compton_profile("compton.txt");
      emd.compton_profile_interp("compton-interp.txt");
    } else {
      char fname[80];
      sprintf(fname,"emd_%i_%i.txt",l,m);
      emd.save(fname);

      sprintf(fname,"moments_%i_%i.txt",l,m);
      emd.moments(fname);
    }

    if(l==0 && m==0)
      printf("Calculating isotropic EMD properties took %s.\n",temd.elapsed().c_str());
    else
      printf("Calculating projected EMD properties took %s.\n",temd.elapsed().c_str());

    delete poseval;
    if(m!=0) delete negeval;
  }

  // Do EMD on a cube?
  if(stricmp(settings.get_string("EMDCube"),"")!=0) {
    t.print_time();
    Timer temd;

    // Form grid in p space.
    std::vector<double> px, py, pz;
    parse_cube(settings.get_string("EMDCube"),px,py,pz);

    // Calculate EMD on cube
    emd_cube(basis,P,px,py,pz);

    printf("Calculating EMD on a cube took %s.\n",temd.elapsed().c_str());
  }

  // Compute similarity?
  if(stricmp(settings.get_string("Similarity"),"")!=0) {

    // Load checkpoint
    Checkpoint simchk(settings.get_string("Similarity"),false);

    // Get grid size
    std::vector<std::string> gridsize=splitline(settings.get_string("SimilarityGrid"));
    if(gridsize.size()!=2) {
      throw std::runtime_error("Invalid grid size!\n");
    }
    int nrad=readint(gridsize[0]);
    int lmax=readint(gridsize[1]);
    int radlmax=settings.get_int("SimilarityLmax");

    // Load basis set
    BasisSet simbas;
    simchk.read(simbas);

    // Load density matrix
    arma::mat simPr, simPi;
    arma::cx_mat simP;
    simchk.read("P",simPr);
    if(simchk.exist("P_im")) {
      simchk.read("P_im",simPi);
      simP=simPr*COMPLEX1 + simPi*COMPLEXI;
    } else
      simP=simPr*COMPLEX1;

    // Compute momentum density overlap
    arma::cube ovl;
    if(settings.get_bool("SimilarityLM"))
      ovl=emd_overlap_semi(basis,P,simbas,simP,nrad,radlmax);
    else
      ovl=emd_overlap(basis,P,simbas,simP,nrad,lmax);

    // Amount of electrons
    int Nela, Nelb;
    chkpt.read("Nel", Nela);
    simchk.read("Nel", Nelb);

    // Shape function overlap
    arma::cube sh=emd_similarity(ovl,Nela,Nelb);
    sh.slice(0).save("similarity.dat",arma::raw_ascii);
    sh.slice(1).save("similarity_avg.dat",arma::raw_ascii);

    for(int s=0;s<2;s++) {
      if(s) {
	printf("%2s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n","k","S0(AA)","S0(BB)","S0(AB)","I0(AA)","I0(BB)","I0(AB)","D0(AB)");
	for(int k=-1;k<3;k++)
	  // Vandenbussche don't include p^2 in the spherical average
	  printf("%2i\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n", k+1, sh(k+1,0,s), sh(k+1,1,s), sh(k+1,2,s), sh(k+1,3,s), sh(k+1,4,s), sh(k+1,5,s), sh(k+1,6,s));
      } else {
	printf("%2s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n","k","S(AA)","S(BB)","S(AB)","I(AA)","I(BB)","I(AB)","D(AB)");
	for(int k=-1;k<3;k++)
	  printf("%2i\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n", k, sh(k+1,0,s), sh(k+1,1,s), sh(k+1,2,s), sh(k+1,3,s), sh(k+1,4,s), sh(k+1,5,s), sh(k+1,6,s));
      }
      printf("\n");
    }
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
