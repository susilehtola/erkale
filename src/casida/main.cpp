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

#include "casida.h"
#include "../checkpoint.h"
#include "../settings.h"
#include "../stringutil.h"
#include "../timer.h"
#include "../dftfuncs.h"

// Needed for libint init
#include "../eriworker.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "../version.h"
#endif

void print_spectrum(const std::string & fname, const arma::mat & m) {
  FILE *out=fopen(fname.c_str(),"w");
  for(size_t it=0; it<m.n_rows; it++)
    fprintf(out,"%e %e\n",m(it,0)*HARTREEINEV, m(it,1));
  fclose(out);
}

void parse_states(const std::vector<double> & occs, std::string & input) {
    // Get HOMO and LUMO
    size_t homo;
    for(homo=occs.size()-1;homo<occs.size();homo--)
      if(occs[homo]>0)
	break;

    size_t lumo;
    for(lumo=0;lumo<occs.size();lumo++)
      if(occs[lumo]==0)
	break;

    // Convert to human indexing
    homo++;
    lumo++;

    char homostr[80];
    char lumostr[80];
    sprintf(homostr,"%i",(int) homo);
    sprintf(lumostr,"%i",(int) lumo);

    // Check if input contains homo or lumo specifier. Replace all occurrences
    while(true) {
      std::string oldinput(input);

      size_t ind=input.find("homo");
      if(ind!=std::string::npos) {
	// Found homo specifier.
	if(ind>0)
	  input=input.substr(0,ind)+std::string(homostr)+input.substr(ind+4,input.size());
	else
	  input=std::string(homostr)+input.substr(ind+4,input.size());
      }

      ind=input.find("lumo");
      if(ind!=std::string::npos) {
	// Found lumo specifier.
	if(ind>0)
	  input=input.substr(0,ind)+std::string(lumostr)+input.substr(ind+4,input.size());
	else
	  input=std::string(lumostr)+input.substr(ind+4,input.size());
      }

      // String was unchanged, we are converged.
      if(oldinput.compare(input)==0)
	break;
    }

    // Perform addition or substraction
    while(true) {
      std::string oldinput(input);

      std::vector<char> signs;
      signs.push_back('+');
      signs.push_back('-');
      signs.push_back('*');

      for(size_t is=0;is<signs.size();is++) {
	size_t ind=input.find(signs[is]);
	if(ind!=std::string::npos) {
	  // Found specifier.
	  if(ind==0 && signs[is]=='+')
	    // Ignore
	    input=input.substr(1,input.size());
	  else {
	    // Find start and end
	    size_t start;
	    std::string startsep;
	    for(start=ind-1;start<input.size();start--)
	      if(input[start]==':' || input[start] == ',') {
		// Found start
		startsep=input[start];
		break;
	      }
	    // Go back one step
	    start++;
	    if(start>input.size())
	      // Overrun
	      start=0;

	    size_t end;
	    std::string endsep;
	    for(end=ind+1;end<input.size();end++)
	      if(input[end]==':' || input[end] == ',') {
		// Found end
		endsep=input[end];
		break;
	      }
	    // Go back one step
	    end--;
	    if(end>=input.size())
	      end=input.size()-1;

	    // Break string into four parts:
	    std::string head="";
	    if(start>0)
	      head=input.substr(0,start-1);

	    std::string val1str=input.substr(start,ind-start);
	    std::string val2str=input.substr(ind+1,end-ind);

	    std::string foot;
	    if(end+2<input.size())
	      foot=input.substr(end+2,input.size());

	    // Perform addition
	    int val1=readint(val1str);
	    int val2=readint(val2str);

	    int res=0;
	    if(signs[is]=='+')
	      res=val1+val2;
	    else if(signs[is]=='-')
	      res=val1-val2;
	    else if(signs[is]=='*')
	      res=val1*val2;

	    char resstr[80];
	    sprintf(resstr,"%i",res);

	    // Replace string
	    input=head+startsep+std::string(resstr)+endsep+foot;
	  }
	}
      }

      // String was unchanged, we are converged.
      if(oldinput.compare(input)==0)
	break;
    }
}

std::string parse_states(Checkpoint & chkpt, const std::string & stateset) {
  // Get specification of states
  std::vector<std::string> states=splitline(tolower(stateset));

  // No states specified
  if(states.size()==0)
    return stateset;

  // Is the calculation restricted?
  bool restr;
  chkpt.read("Restricted",restr);

  // Sanity check
  if((states.size()==1 && !restr) || (states.size()==2 && restr) || states.size()>2)
    throw std::runtime_error("CasidaStates input not consistent with type of wavefunction!\n");

  // Parse input
  std::string state;
  if(restr) {
    std::vector<double> occs;
    chkpt.read("occs",occs);

    parse_states(occs,states[0]);
    state=states[0];
  } else {
    std::vector<double> occa, occb;
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);
    parse_states(occa,states[0]);
    parse_states(occb,states[1]);
    state=states[0] + " " +states[1];
  }

  return state;
}

void print_header() {
#ifdef _OPENMP
  printf("ERKALE - Casida from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Casida from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

Settings settings;

int main_guarded(int argc, char **argv) {
  print_header();

  if(argc!=1 && argc!=2) {
    printf("Usage: $ %s (runfile)\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();

  Timer t;
  t.print_time();

  // Parse settings
  settings.add_string("FittingBasis","Basis set to use for density fitting (Auto for automatic)","Auto");
  settings.add_string("CasidaX","Exchange functional for Casida","lda_x");
  settings.add_string("CasidaC","Correlation functional for Casida","lda_c_vwn");
  settings.add_bool("CasidaPol","Perform polarized Casida calculation (when using restricted wf)",false);
  settings.add_int("CasidaCoupling","Coupling mode: 0 for IPA, 1 for RPA and 2 for TDLDA",2);
  settings.add_double("CasidaTol","Tolerance for Casida grid",1e-4);
  settings.add_string("CasidaStates","States to include in Casida calculation, eg ""1,3:7,10,13"" ","");
  settings.add_string("CasidaQval","Values of Q to compute spectrum for","");
  settings.add_string("LoadChk","Checkpoint to load","erkale.chk");

  if(argc==2)
    settings.parse(std::string(argv[1]));
  else
    printf("\nDefault settings used.");

  // Print settings
  settings.print();

  // Get functional strings
  int xfunc=find_func(settings.get_string("CasidaX"));
  int cfunc=find_func(settings.get_string("CasidaC"));

  if(is_correlation(xfunc))
    throw std::runtime_error("Refusing to use a correlation functional as exchange.\n");
  if(is_kinetic(xfunc))
    throw std::runtime_error("Refusing to use a kinetic energy functional as exchange.\n");
  if(is_exchange(cfunc))
    throw std::runtime_error("Refusing to use an exchange functional as correlation.\n");
  if(is_kinetic(cfunc))
    throw std::runtime_error("Refusing to use a kinetic energy functional as correlation.\n");
  settings.add_int("CasidaXfunc","Internal variable",xfunc);
  settings.add_int("CasidaCfunc","Internal variable",cfunc);

  // Print information about used functionals
  print_info(xfunc,cfunc);

  // Get values of q to compute spectrum for
  std::vector<double> qvals=parse_range_double(settings.get_string("CasidaQval"));

  // Load checkpoint
  std::string fchk=settings.get_string("LoadChk");
  Checkpoint chkpt(fchk,false);

  // Check that calculation was converged
  bool conv;
  chkpt.read("Converged",conv);
  if(!conv)
    throw std::runtime_error("Refusing to run Casida calculation based on a non-converged SCF density!\n");

  // Parse input states
  std::string states=settings.get_string("CasidaStates");
  std::string newstates=parse_states(chkpt,states);
  settings.set_string("CasidaStates",newstates);
  if(states.compare(newstates)!=0)
    printf("CasidaStates has been parsed to \"%s\".\n",newstates.c_str());

  // Load basis set
  BasisSet basis;
  chkpt.read(basis);

  Casida cas;
  bool restr;
  chkpt.read("Restricted",restr);

  if(restr) {
    // Load energy and orbitals
    arma::mat C, P;
    arma::vec E;
    std::vector<double> occs;

    chkpt.read("P",P);
    chkpt.read("C",C);
    chkpt.read("E",E);
    chkpt.read("occs",occs);

    // Check orthonormality
    check_orth(C,basis.overlap(),false);

    if(settings.get_bool("CasidaPol")) {
      // Half occupancy (1.0 instead of 2.0)
      std::vector<double> hocc(occs);
      for(size_t i=0;i<hocc.size();i++)
	hocc[i]/=2.0;
      cas=Casida(basis,E,E,C,C,P/2.0,P/2.0,hocc,hocc);
    }
    else
      cas=Casida(basis,E,C,P,occs);

  } else {
    arma::mat Ca, Cb;
    arma::mat Pa, Pb;
    arma::vec Ea, Eb;
    std::vector<double> occa, occb;

    chkpt.read("Pa",Pa);
    chkpt.read("Pb",Pb);
    chkpt.read("Ca",Ca);
    chkpt.read("Cb",Cb);
    chkpt.read("Ea",Ea);
    chkpt.read("Eb",Eb);
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);

    // Check orthonormality
    check_orth(Ca,basis.overlap(),false);
    check_orth(Cb,basis.overlap(),false);

    cas=Casida(basis,Ea,Eb,Ca,Cb,Pa,Pb,occa,occb);
  }

  // Dipole transition
  print_spectrum("casida.dat",cas.dipole_transition(basis));
  // Q dependent stuff
  for(size_t iq=0;iq<qvals.size();iq++) {
    // File to save output
    char fname[80];
    sprintf(fname,"casida-%.2f.dat",qvals[iq]);

    print_spectrum(fname,cas.transition(basis,qvals[iq]));
  }

  printf("\nRunning program took %s.\n",t.elapsed().c_str());
  t.print_time();

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
