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

#include "elements.h"
#include "dftfuncs.h"
#include "guess.h"
#include "checkpoint.h"
#include "linalg.h"
#include "settings.h"
#include "scf.h"
#include "stringutil.h"
#include "timer.h"
#include <algorithm>

extern Settings settings;

void atomic_guess(const BasisSet & basis, size_t inuc, const std::string & method, std::vector<size_t> & shellidx, BasisSet & atbas, arma::vec & atEocc, arma::mat & atCocc, arma::mat & atP, arma::mat & atF, int Q) {
  // Nucleus is
  nucleus_t nuc=basis.get_nucleus(inuc);

  // Set number
  nuc.ind=0;
  nuc.r.x=0.0;
  nuc.r.y=0.0;
  nuc.r.z=0.0;

  // Settings to use
  Settings settings0(settings);
  settings=Settings();
  settings.add_scf_settings();
  settings.add_bool("ForcePol","Force polarized calculation",true);
  settings.add_string("SaveChk","Save calculation to","");
  settings.add_string("LoadChk","Load calculation from","");

  settings.set_string("Guess","Core");
  settings.set_int("MaxIter",200);
  settings.set_bool("DensityFitting",false);
  settings.set_bool("Verbose",false);
  settings.set_bool("Direct",false);
  settings.set_bool("DensityFitting",false);
  settings.set_bool("Cholesky",true);
  settings.set_double("CholeskyThr",1e-5);
  // Use a rather large grid to make sure the calculation converges
  // even in cases where the functional requires a large grid to be
  // used. The other way would be to pass the user settings to this
  // routine..
  settings.set_string("DFTGrid","100 -434");

  // Don't do PZ-SIC for the initial guess.
  try {
    settings.set_bool("PZ",false);
  } catch(...) {
  }
  // Also, turn off non-local correlation for initial guess
  settings.set_string("VV10","False");

  // Use default convergence settings
  settings.set_bool("UseDIIS",true);
  settings.set_int("DIISOrder",20);
  settings.set_bool("UseADIIS",true);
  settings.set_bool("UseBroyden",false);
  settings.set_bool("UseTRRH",false);
  // and default charge
  settings.set_int("Charge", Q);

  // Method
  settings.set_string("Method",method);

  // Relax convergence requirements - open shell atoms may be hard to
  // converge
  settings.set_double("ConvThr",1e-4);

  // Construct the basis set
  atbas=BasisSet(1);
  // Add the nucleus
  atbas.add_nucleus(nuc);

  std::vector<GaussianShell> shells=basis.get_funcs(inuc);
  // Indices of shells included
  shellidx.clear();
  for(size_t ish=0;ish<shells.size();ish++) {
    // Add shell on zeroth atom, don't sort
    atbas.add_shell(0,shells[ish],false);
    shellidx.push_back(ish);
  }

  // Finalize basis set
  atbas.finalize();

  // Set dummy multiplicity
  settings.set_int("Multiplicity",1);

  // Force polarized calculation
  settings.set_bool("ForcePol",true);

  // Get occupancies
  std::vector<double> occa(atomic_occupancy(0.5*(nuc.Z-Q),atbas.get_Nbf()));
  std::vector<double> occb(occa);

  std::ostringstream occs;
  for(size_t i=0;i<occa.size();i++)
    occs << occa[i] << " " << occb[i] << " ";
  settings.set_string("Occupancies",occs.str());

  // Temporary file name
  std::string tmpname(tempname());
  settings.set_string("SaveChk",tmpname);

  // Run calculation
  calculate(atbas);

  // Load energies and density matrix
  {
    // Checkpoint
    Checkpoint chkpt(tmpname,false);

    chkpt.read("P",atP);

    int restr;
    chkpt.read("Restricted",restr);
    if (restr) {
      chkpt.read("E",atEocc);
      chkpt.read("C",atCocc);
      chkpt.read("H",atF);
    } else {
      arma::mat Fa, Fb;
      chkpt.read("Ha",Fa);
      chkpt.read("Hb",Fb);
      atF=(Fa+Fb)*0.5;
      chkpt.read("Ea",atEocc);
      chkpt.read("Ca",atCocc);
    }
    // Substract core Hamiltonian to leave only Coulomb and
    // exchange-correlation part
    arma::mat Hcore;
    chkpt.read("Hcore",Hcore);
    atF-=Hcore;

    // Drop virtual orbitals
    atEocc=atEocc.subvec(0,occa.size()-1);
    atCocc=atCocc.cols(0,occa.size()-1);
  }

  // Remove temporary file
  remove(tmpname.c_str());

  // Restore global settings
  settings=settings0;
}

typedef enum {
              FORM_SAD,
              FORM_SAP,
              FORM_HUCKEL,
              FORM_MINBAS
} atomic_guess_t;

static const std::string guesstypes[]={
                                       "SAD",
                                       "SAP",
                                       "Huckel",
                                       "Minimal basis"
};

arma::mat atomic_guess_wrk(const BasisSet & basis, atomic_guess_t type, double Kgwh) {
  // First of all, we need to determine which atoms are identical in
  // the way that the basis sets coincide.

  // Get list of identical nuclei
  std::vector< std::vector<size_t> > idnuc=identical_nuclei(basis);

  // Amount of basis functions is
  size_t Nbf=basis.get_Nbf();

  // Print out info?
  bool verbose=settings.get_bool("Verbose");

  std::string method=settings.get_string("Method");
  if(stricmp(settings.get_string("AtomGuess"),"Auto")!=0)
    method=settings.get_string("AtomGuess");

  if(verbose) {
    // Parse method
    bool hf= (stricmp(method,"HF")==0);
    if(hf)
      method="HF";
    else {
      bool rohf=(stricmp(method,"ROHF")==0);
      if(rohf)
	method="ROHF";
      else {
	// Parse functional
	dft_t dft;
	parse_xc_func(dft.x_func,dft.c_func,method);
	if(dft.c_func>0) {
	  // Correlation exists.
	  method=get_keyword(dft.x_func)+"-"+get_keyword(dft.c_func);
	} else
	  method=get_keyword(dft.x_func);
      }
    }

    printf("Performing %s guess for atoms:\n",method.c_str());
    fprintf(stderr,"Calculating initial atomic guess ... ");
    fflush(stdout);
    fflush(stderr);
  }

  Timer ttot;

  // Identical nuclei
  std::vector< std::vector<size_t> > shellidx(idnuc.size());
  std::vector<BasisSet> atbas(idnuc.size());
  std::vector<arma::vec> atEocc(idnuc.size());
  std::vector<arma::mat> atCocc(idnuc.size());
  std::vector<arma::mat> atP(idnuc.size());
  std::vector<arma::mat> atF(idnuc.size());

  // Loop over list of identical nuclei
  for(size_t i=0;i<idnuc.size();i++) {
    Timer tsol;
    if(verbose) {
      printf("%-2s:",basis.get_nucleus(idnuc[i][0]).symbol.c_str());
      for(size_t iid=0;iid<idnuc[i].size();iid++)
	printf(" %i",(int) idnuc[i][iid]+1);
      fflush(stdout);
    }

    // Perform the guess
    atomic_guess(basis,idnuc[i][0],method,shellidx[i],atbas[i],atEocc[i],atCocc[i],atP[i],atF[i],basis.get_nucleus(idnuc[i][0]).Q);

    if(verbose) {
      printf(", %i orbitals (%s)\n",(int) atEocc[i].n_elem,tsol.elapsed().c_str());
      fflush(stdout);
    }
  }

  // Molecular density or Fock matrix
  arma::mat M(Nbf,Nbf);
  M.zeros();

  // Form the molecular guess
  if(type == FORM_SAD || type == FORM_SAP) {
    for(size_t i=0;i<idnuc.size();i++) {
      // Get the atomic shells
      std::vector<GaussianShell> shells=atbas[i].get_funcs(0);

      // Loop over shells
      for(size_t ish=0;ish<shells.size();ish++)
        for(size_t jsh=0;jsh<shells.size();jsh++) {

          // Loop over identical nuclei
          for(size_t iid=0;iid<idnuc[i].size();iid++) {
            // Get shells on nucleus
            std::vector<GaussianShell> idsh=basis.get_funcs(idnuc[i][iid]);

            // Store density / Fock matrix
            const arma::mat & atM = (type==FORM_SAP) ? atF[i] : atP[i];
            size_t i_global_first(idsh[shellidx[i][ish]].get_first_ind());
            size_t i_global_last(idsh[shellidx[i][ish]].get_last_ind());
            size_t j_global_first(idsh[shellidx[i][jsh]].get_first_ind());
            size_t j_global_last(idsh[shellidx[i][jsh]].get_last_ind());

            size_t i_local_first(shells[ish].get_first_ind());
            size_t i_local_last(shells[ish].get_last_ind());
            size_t j_local_first(shells[jsh].get_first_ind());
            size_t j_local_last(shells[jsh].get_last_ind());

            M.submat(i_global_first, j_global_first, i_global_last, j_global_last) = atM.submat(i_local_first, j_local_first, i_local_last, j_local_last);
          }
        }
    }

  } else if(type == FORM_HUCKEL || type == FORM_MINBAS) {
    // Number of functions on each nucleus
    arma::uvec numorb(basis.get_Nnuc());
    numorb.zeros();
    for(size_t i=0;i<idnuc.size();i++) {
      for(size_t iid=0;iid<idnuc[i].size();iid++) {
        numorb(idnuc[i][iid])=atEocc[i].n_elem;

        if(atEocc[i].n_elem != atCocc[i].n_cols) {
          std::ostringstream oss;
          oss << "Number of orbital energies does not match that of orbital coefficients!\n";
          throw std::logic_error(oss.str());
        }
      }
    }

    // Index of irreducible nucleus
    arma::uvec irrnuc(basis.get_Nnuc());
    irrnuc.ones();
    irrnuc*=-1;
    for(size_t i=0;i<idnuc.size();i++) {
      for(size_t iid=0;iid<idnuc[i].size();iid++) {
        irrnuc(idnuc[i][iid])=i;
      }
    }

    // Total dimension of Huckel basis
    size_t n(arma::sum(numorb));

    // Orbital coefficients and energies
    std::vector<arma::vec> orbEat(basis.get_Nnuc());
    std::vector<arma::mat> orbCat(basis.get_Nnuc());
    for(size_t i=0;i<idnuc.size();i++) {
      for(size_t iid=0;iid<idnuc[i].size();iid++) {
        orbEat[idnuc[i][iid]]=atEocc[i];
        orbCat[idnuc[i][iid]]=atCocc[i];
      }
    }

    // Atomic orbitals in full basis
    arma::mat HuC(Nbf,n);
    HuC.zeros();
    // Orbital energies in full basis
    arma::vec HuE(n);
    HuE.zeros();
    // Orbital index
    {
      size_t io=0;
      for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
        // Skip ghost atoms
        if(basis.get_nucleus(inuc).bsse)
          continue;

        // Get shells on nucleus
        std::vector<GaussianShell> idsh=basis.get_funcs(inuc);
        std::vector<GaussianShell> shells=atbas[irrnuc[inuc]].get_funcs(0);

        // Loop over shells
        for(size_t ish=0;ish<shells.size();ish++) {
          // Global indices
          size_t i_global_first(idsh[ish].get_first_ind());
          size_t i_global_last(idsh[ish].get_last_ind());

          // Local indices
          size_t i_local_first(shells[ish].get_first_ind());
          size_t i_local_last(shells[ish].get_last_ind());

          // Store orbitals
          HuC.submat(i_global_first, io, i_global_last, io+orbCat[inuc].n_cols-1) = orbCat[inuc].rows(i_local_first, i_local_last);
        }
        // Store energies
        HuE.subvec(io, io+orbEat[inuc].n_elem-1) = orbEat[inuc];

        // Increment orbital count
        io+=orbCat[inuc].n_cols;
      }
      if(io!=n)
        throw std::logic_error("Huckel orbital coefficient indexing error!\n");
    }

    // Overlap matrix
    arma::mat S(basis.overlap());
    // Overlap in Huckel basis
    arma::mat ShuC(S*HuC);
    arma::mat Shu(HuC.t()*ShuC);

    // Form Huckel matrix in MO basis
    arma::mat Hu(n,n);
    Hu.zeros();
    for(size_t i=0;i<n;i++) {
      Hu(i,i)=HuE(i);
      for(size_t j=0;j<i;j++) {
        Hu(i,j) = Hu(j,i) = 0.5*Kgwh*Shu(i,j)*(HuE(i)+HuE(j));
      }
    }

    // Convert Huckel matrix to full AO basis
    if(type == FORM_HUCKEL)
      M=ShuC*Hu*ShuC.t();
    else if(type == FORM_MINBAS) {
      M=ShuC;
    } else
      throw std::logic_error("Case not handled!\n");
  }

  /*
  // Check that density matrix contains the right amount of electrons
  int Neltot=basis.Ztot()-settings.get_int("Charge");
  double Nel=arma::trace(P*S);
  if(fabs(Nel-Neltot)/Neltot*100>1e-10)
  fprintf(stderr,"Nel = %i, P contains %f electrons, difference %e.\n",Neltot,Nel,Nel-Neltot);
  */

  if(verbose) {
    printf("%s guess formed in %s.\n\n",guesstypes[type].c_str(),ttot.elapsed().c_str());
    fprintf(stderr,"done (%s)\n\n",ttot.elapsed().c_str());
    fflush(stderr);
  }

  return M;
}

arma::mat sad_guess(const BasisSet & basis) {
  return atomic_guess_wrk(basis,FORM_SAD,0.0);
}

arma::mat sap_guess(const BasisSet & basis) {
  return atomic_guess_wrk(basis,FORM_SAP,0.0);
}

arma::mat huckel_guess(const BasisSet & basis, double Kgwh) {
  return atomic_guess_wrk(basis,FORM_HUCKEL,Kgwh);
}

arma::mat minimal_basis_projection(const BasisSet & basis) {
  return atomic_guess_wrk(basis,FORM_MINBAS,0.0);
}

std::vector< std::vector<size_t> > identical_nuclei(const BasisSet & basis) {
  // Returned list
  std::vector< std::vector<size_t> > ret;

  // Loop over nuclei
  for(size_t i=0;i<basis.get_Nnuc();i++) {
    // Check that nucleus isn't BSSE
    nucleus_t nuc=basis.get_nucleus(i);
    if(nuc.bsse)
      continue;

    // Get the shells on the nucleus
    std::vector<GaussianShell> shi=basis.get_funcs(i);

    // Check if there something already on the list
    bool found=false;
    for(size_t j=0;j<ret.size();j++) {
      std::vector<GaussianShell> shj=basis.get_funcs(ret[j][0]);

      // Check nuclear type
      if(basis.get_symbol(i).compare(basis.get_symbol(ret[j][0]))!=0)
	continue;
      // Check charge status
      if(basis.get_nucleus(i).Q != basis.get_nucleus(ret[j][0]).Q)
	continue;

      // Do comparison
      if(shi.size()!=shj.size())
	continue;
      else {

	bool same=true;
	for(size_t ii=0;ii<shi.size();ii++) {
	  // Check angular momentum
	  if(shi[ii].get_am()!=shj[ii].get_am()) {
	    same=false;
	    break;
	  }

	  // and exponents
	  std::vector<contr_t> lhc=shi[ii].get_contr();
	  std::vector<contr_t> rhc=shj[ii].get_contr();

	  if(lhc.size() != rhc.size()) {
	    same=false;
	    break;
	  }
	  for(size_t ic=0;ic<lhc.size();ic++) {
	    if(!(lhc[ic]==rhc[ic])) {
	      same=false;
	      break;
	    }
	  }

	  if(!same)
	    break;
	}

	if(same) {
	  // Found identical atom.
	  found=true;

	  // Add it to the list.
	  ret[j].push_back(i);
	}
      }
    }

    if(!found) {
      // Didn't find the atom, add it to the list.
      std::vector<size_t> tmp;
      tmp.push_back(i);

      ret.push_back(tmp);
    }
  }

  return ret;
}


bool operator<(const el_conf_t & lhs, const el_conf_t & rhs) {
  if(lhs.n + lhs.l < rhs.n + rhs.l)
    return true;
  else if(lhs.n + lhs.l == rhs.n + rhs.l)
    return lhs.n < rhs.n;

  return false;
}

std::vector<el_conf_t> get_occ_order(int nmax) {
  std::vector<el_conf_t> confs;
  for(int n=1;n<nmax;n++)
    for(int l=0;l<n;l++) {
      el_conf_t tmp;
      tmp.n=n;
      tmp.l=l;
      confs.push_back(tmp);
    }
  std::sort(confs.begin(),confs.end());

  return confs;
}

gs_conf_t get_ground_state(int Z) {
  // The returned configuration
  gs_conf_t ret;

  // Get the ordering of the shells
  std::vector<el_conf_t> confs=get_occ_order(8);

  // Start occupying.
  size_t i=0;
  while(Z>=2*(2*confs[i].l+1)) {
    Z-=2*(2*confs[i].l+1);
    i++;
  }

  if(Z==0) {
    // All shells are full.
    ret.mult=1;
    ret.L=0;
    ret.dJ=0;
  } else {
    // Determine how the states electrons are occupied.

    arma::imat occs(2*confs[i].l+1,2);
    occs.zeros();

    // Column to fill
    int col=0;
    do {
      // Occupy column
      for(int ml=confs[i].l;ml>=-confs[i].l;ml--)
	if(Z>0) {
	  occs(confs[i].l-ml,col)=1;
	  Z--;
	}

      // If we still have electrons left, switch to the next column
      if(Z>0) {
	if(col==0)
	  col=1;
	else {
	  ERROR_INFO();
	  throw std::runtime_error("Should not end up here!\n");
	}
      }
    } while(Z>0);

    // Compute S and L value
    int m=0, L=0, dJ=0;
    for(size_t j=0;j<occs.n_rows;j++) {
      m+=occs(j,0)-occs(j,1);
      L+=(confs[i].l-j)*(occs(j,0)+occs(j,1));
    }

    if(col==0)
      dJ=abs(2*L-m);
    else if(col==1)
      dJ=2*L+m;

    ret.mult=m+1;
    ret.L=L;
    ret.dJ=dJ;
  }

  return ret;
}
