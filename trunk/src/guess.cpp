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

#include "guess.h"
#include "checkpoint.h"
#include "scf.h"
#include "timer.h"
#include <algorithm>

void atomic_guess(const BasisSet & basis, arma::mat & C, arma::mat & E, Settings set) {
  // First of all, we need to determine which atoms are identical in
  // the way that the basis sets coincide.

  // Get list of identical nuclei
  std::vector< std::vector<size_t> > idnuc=identical_nuclei(basis);

  // Amount of basis functions is
  size_t Nbf=basis.get_Nbf();

  // Density matrix
  arma::mat P(Nbf,Nbf);
  P.zeros();

  // Print out info?
  bool verbose=set.get_bool("Verbose");
  
  // Settings to use
  set.set_string("Guess","Core");
  set.set_int("MaxIter",200);
  set.set_bool("DensityFitting",false);
  set.set_bool("ForcePol",true);
  set.set_bool("Verbose",false);
  set.set_bool("Direct",false);

  // Relax convergence requirements - open shell atoms may be hard to
  // converge
  set.set_double("DeltaPmax",1e-5);
  set.set_double("DeltaPrms",1e-6);

  if(verbose) {
    printf("Performing atomic guess for atoms:\n");
    fprintf(stderr,"Calculating initial atomic guess ... ");
    fflush(stdout);
    fflush(stderr);
  }

  Timer ttot;

  // Approximate orbital energies
  std::vector<double> orbE;
  
  // Loop over list of identical nuclei
  for(size_t i=0;i<idnuc.size();i++) {

    Timer tsol;

    if(verbose) {
      printf("\t");
      for(size_t iid=0;iid<idnuc[i].size();iid++)
	printf("%i ",(int) idnuc[i][iid]+1);
      fflush(stdout);
    }

    // Nucleus is
    nucleus_t nuc=basis.get_nucleus(idnuc[i][0]);
    // Set number
    nuc.ind=0;
    nuc.r.x=0.0;
    nuc.r.y=0.0;
    nuc.r.z=0.0;

    // Construct the basis set
    BasisSet atbas(1,set);
    // Add the nucleus
    atbas.add_nucleus(nuc);

    // Add the shells relevant for a single atom.
    int ammax;
    if(nuc.Z<3)
      // Only s electrons up to lithium
      ammax=0;
    else if(nuc.Z<21)
      // s and p electrons
      ammax=1;
    else if(nuc.Z<57)
      // s, p and d electrons
      ammax=2;
    else 
      // s, p, d and f electrons
      ammax=3;
    
    std::vector<GaussianShell> shells=basis.get_funcs(idnuc[i][0]);
    // Indices of shells included
    std::vector<size_t> shellidx;
    for(size_t ish=0;ish<shells.size();ish++) {
      // Add shell on zeroth atom, don't sort
      if(shells[ish].get_am()<=ammax) {
	atbas.add_shell(0,shells[ish],false);
	shellidx.push_back(ish);
      }
    }

    // Finalize basis set
    atbas.finalize();

    // Determine ground state
    gs_conf_t gs=get_ground_state(nuc.Z);

    // Set multiplicity
    set.set_int("Multiplicity",gs.mult);

    // Temporary file name
    char *tmpname=tempnam("./",".chk");
    set.set_string("SaveChk",tmpname);

    // Run calculation
    calculate(atbas,set);

    // Checkpoint
    Checkpoint chkpt(tmpname,false);

    // Re-get shells, in new indexing.
    shells=atbas.get_funcs(0);

    // Load energies and density matrix
    arma::vec atEa;
    arma::mat atP;

    chkpt.read("Ea",atEa);
    chkpt.read("P",atP);
    
    // Store approximate energies
    for(size_t iid=0;iid<idnuc[i].size();iid++)
      for(size_t io=0;io<atEa.size();io++)
	orbE.push_back(atEa(io));

    // Loop over shells
    for(size_t ish=0;ish<shells.size();ish++)
      for(size_t jsh=0;jsh<shells.size();jsh++) {

	// Loop over identical nuclei
	for(size_t iid=0;iid<idnuc[i].size();iid++) {
	  // Get shells on nucleus
	  std::vector<GaussianShell> idsh=basis.get_funcs(idnuc[i][iid]);
	  
	  // Store density
	  P.submat(idsh[shellidx[ish]].get_first_ind(),idsh[shellidx[jsh]].get_first_ind(),idsh[shellidx[ish]].get_last_ind(),idsh[shellidx[jsh]].get_last_ind())=atP.submat(shells[ish].get_first_ind(),shells[jsh].get_first_ind(),shells[ish].get_last_ind(),shells[jsh].get_last_ind());
	}
      }

    if(verbose) {
      printf(" (%s)\n",tsol.elapsed().c_str());
      fflush(stdout);
    }

    // Remove temporary file
    remove(tmpname);
    // Free memory
    free(tmpname);
  }

  Timer trest;
  if(verbose) {
    printf("Diagonalizing density matrix ... ");
    fflush(stdout);
  }

  // Check that density matrix contains the right amount of electrons
  arma::mat S=basis.overlap();
  int Neltot=basis.Ztot()-set.get_int("Charge");
  double Nel=arma::trace(P*S);
  if(fabs(Nel-Neltot)/Neltot*100>1e-10)
    fprintf(stderr,"Nel = %i, P contains %f electrons, difference %e.\n",Neltot,Nel,Nel-Neltot);

  // Go to natural orbitals
  arma::mat NO;
  arma::vec occs;
  arma::mat tmp;
  form_NOs(P,S,NO,tmp,occs);

  // Store orbitals, but in reverse order!
  C.zeros(NO.n_rows,NO.n_cols);
  for(size_t i=0;i<NO.n_cols;i++)
    C.col(i)=NO.col(NO.n_cols-1-i);

  // Store energies
  E.zeros(C.n_cols);
  std::sort(orbE.begin(),orbE.end());
  for(size_t i=0;i<std::min(orbE.size(),(size_t) C.n_cols);i++)
    E(i)=orbE[i];
  
  if(verbose) {
    printf("done (%s)\n",trest.elapsed().c_str());
    printf("Atomic guess formed in %s.\n\n",ttot.elapsed().c_str());
    fprintf(stderr,"done (%s)\n\n",ttot.elapsed().c_str());
    fflush(stderr);
  }
}

int atom_am(int Z) {
  int atomam;
  if(Z<5)
    atomam=0;
  else if(Z<21)
    atomam=1;
  else if(Z<57)
    atomam=2;
  else
    atomam=3;

  return atomam;
}

void molecular_guess(const BasisSet & basis, const Settings & set, std::string & chkname) {
  Timer t;
  
  // Get temporary file name
  char *tmpn=tempnam("./",".chk");
  std::string tempname=std::string(tmpn);
  free(tmpn);

  // New settings
  Settings newset(set);
  newset.set_string("LoadChk","");
  newset.set_string("SaveChk",tempname);
  newset.set_string("Guess","atomic");
  newset.set_bool("DensityFitting",false);
  newset.set_bool("Verbose",true);

  // Use relaxed convergence settings
  newset.set_double("DeltaEmax",std::max(set.get_double("DeltaEmax"),1e-6));
  newset.set_double("DeltaPmax",std::max(set.get_double("DeltaPmax"),1e-5));
  newset.set_double("DeltaPrms",std::max(set.get_double("DeltaPrms"),1e-6));

  // Construct new basis
  BasisSet newbas(basis.get_Nnuc(),newset);

  // Add the nuclei
  std::vector<nucleus_t> nuclei=basis.get_nuclei();
  for(size_t i=0;i<nuclei.size();i++)
    newbas.add_nucleus(nuclei[i]);

  // Indices of added shells
  std::vector<size_t> addedidx;
  // Indices of missing shells
  std::vector<size_t> missingidx;

  // Add the shells
  std::vector<GaussianShell> shells=basis.get_shells();
  for(size_t i=0;i<shells.size();i++) {
    // Add the shell to the minimal basis? Default is true
    bool add=true;

    // Check for polarization shells
    if(shells[i].get_am() > atom_am(nuclei[shells[i].get_center_ind()].Z))
      add=false;

    if(add) {
      // Add the shell to the basis set
      newbas.add_shell(shells[i].get_center_ind(),shells[i],false);
      // and to the list
      addedidx.push_back(i);
    } else
      // Add the shell to the missing list
      missingidx.push_back(i);
  }
  newbas.finalize();

  // Now we have built the basis set and we can proceed with the solution.
  printf("Calculating molecular guess.\nFull basis has %i functions, whereas reduced basis only has %i.\n",(int) basis.get_Nbf(),(int) newbas.get_Nbf());
  fprintf(stderr,"Calculating molecular guess.\nFull basis has %i functions, whereas reduced basis only has %i.\n",(int) basis.get_Nbf(),(int) newbas.get_Nbf());

  // Calculate the solution in the temporary file.
  calculate(newbas,newset);

  printf("\nSolving the density in the reduced basis took %s.\n",t.elapsed().c_str());
  fflush(stdout);
  fprintf(stderr,"\nSolving the density in the reduced basis took %s.\n",t.elapsed().c_str());
  fflush(stderr);
  t.set();
  
  // Get another temporary file name. This will contain the returned
  // orbitals and energies.
  tmpn=tempnam("./",".chk");
  chkname=std::string(tmpn);
  free(tmpn);  

  // Open the return file
  Checkpoint chkpt(chkname,true);

  {
    // Open the temp file
    Checkpoint load(tempname,false);
    
    bool restr;
    load.read("Restricted",restr);

    if(restr) {
      BasisSet oldbas;
      arma::mat Cold;
      arma::vec Eold;
      load.read(oldbas);
      load.read("E",Eold);
      load.read("C",Cold);

      // Project the orbitals
      arma::mat C=project_orbitals(Cold,oldbas,basis);
      chkpt.write("C",C);

      // and generate dummy energies
      arma::vec E(C.n_cols);
      E.subvec(0,Eold.n_elem-1)=Eold;
      for(size_t i=Eold.n_elem;i<E.n_elem;i++)
	E(i)=Eold(Eold.n_elem-1);
      chkpt.write("E",E);

    } else {
      BasisSet oldbas;
      arma::mat Caold, Cbold;
      arma::vec Eaold, Ebold;

      load.read(oldbas);
      load.read("Ca",Caold);
      load.read("Cb",Cbold);
      load.read("Ea",Eaold);
      load.read("Eb",Ebold);

      // Project the orbitals
      arma::mat Ca=project_orbitals(Caold,oldbas,basis);
      arma::mat Cb=project_orbitals(Cbold,oldbas,basis);
      chkpt.write("Ca",Ca);
      chkpt.write("Cb",Cb);

      // and generate dummy energies
      arma::vec Ea(Ca.n_cols);
      arma::vec Eb(Cb.n_cols);
      Ea.subvec(0,Eaold.n_elem-1)=Eaold;
      Eb.subvec(0,Eaold.n_elem-1)=Ebold;
      for(size_t i=Eaold.n_elem;i<Ea.n_elem;i++)
	Ea(i)=Eaold(Eaold.n_elem-1);
      for(size_t i=Ebold.n_elem;i<Eb.n_elem;i++)
	Eb(i)=Ebold(Ebold.n_elem-1);

      chkpt.write("Ea",Ea);
      chkpt.write("Eb",Eb);
    }
  }

  // Delete the temporary file
  remove(tempname.c_str());

  fprintf(stderr,"Projection of molecular guess took %s.\n",t.elapsed().c_str());
  fflush(stderr);
  printf("Projection of molecular guess took %s.\n",t.elapsed().c_str());
  fflush(stdout);
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
