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

#include "properties.h"
#include "stringutil.h"
#include "dftgrid.h"
#include "bader.h"
#include "linalg.h"

arma::mat mulliken_overlap(const BasisSet & basis, const arma::mat & P) {
  // Amount of nuclei in basis set
  size_t Nnuc=basis.get_Nnuc();

  arma::mat ret(Nnuc,Nnuc);
  ret.zeros();

  // Get overlap
  arma::mat S=basis.overlap();

  // Loop over nuclei
  for(size_t ii=0;ii<Nnuc;ii++) {
    // Get shells on nucleus
    std::vector<GaussianShell> ifuncs=basis.get_funcs(ii);

    // Loop over nuclei
    for(size_t jj=0;jj<=ii;jj++) {
      // Get shells on nucleus
      std::vector<GaussianShell> jfuncs=basis.get_funcs(jj);

      // Initialize output
      ret(ii,jj)=0.0;

      // Loop over shells
      for(size_t fi=0;fi<ifuncs.size();fi++) {
	// First function on shell is
	size_t ifirst=ifuncs[fi].get_first_ind();
	// Last function on shell is
	size_t ilast=ifuncs[fi].get_last_ind();

	// Loop over shells
	for(size_t fj=0;fj<jfuncs.size();fj++) {
	  size_t jfirst=jfuncs[fj].get_first_ind();
	  size_t jlast=jfuncs[fj].get_last_ind();

	  // Loop over functions
	  for(size_t i=ifirst;i<=ilast;i++)
	    for(size_t j=jfirst;j<=jlast;j++)
	      ret(ii,jj)+=P(i,j)*S(i,j);
	}
      }

      // Symmetricize
      if(ii!=jj)
	ret(jj,ii)=ret(ii,jj);
    }
  }

  return ret;
}

arma::mat bond_order(const BasisSet & basis, const arma::mat & P) {
  // Amount of nuclei in basis set
  size_t Nnuc=basis.get_Nnuc();

  arma::mat ret(Nnuc,Nnuc);
  ret.zeros();

  // Get overlap
  arma::mat S=basis.overlap();

  // Form PS
  arma::mat PS=P*S;

  // Loop over nuclei
  for(size_t ii=0;ii<Nnuc;ii++) {
    // Get shells on nucleus
    std::vector<GaussianShell> ifuncs=basis.get_funcs(ii);

    // Loop over nuclei
    for(size_t jj=0;jj<=ii;jj++) {
      // Get shells on nucleus
      std::vector<GaussianShell> jfuncs=basis.get_funcs(jj);

      // Initialize output
      ret(ii,jj)=0.0;

      // Loop over shells
      for(size_t fi=0;fi<ifuncs.size();fi++) {
	// First function on shell is
	size_t ifirst=ifuncs[fi].get_first_ind();
	// Last function on shell is
	size_t ilast=ifuncs[fi].get_last_ind();

	// Loop over shells
	for(size_t fj=0;fj<jfuncs.size();fj++) {
	  size_t jfirst=jfuncs[fj].get_first_ind();
	  size_t jlast=jfuncs[fj].get_last_ind();

	  // Loop over functions
	  for(size_t i=ifirst;i<=ilast;i++)
	    for(size_t j=jfirst;j<=jlast;j++)
	      ret(ii,jj)+=PS(i,j)*PS(j,i);
	}
      }

      // Symmetricize
      if(ii!=jj)
	ret(jj,ii)=ret(ii,jj);
    }
  }

  // The factor 1/2 seems necessary.
  return ret/2.0;
}

arma::mat bond_order(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  return bond_order(basis,Pa+Pb)+bond_order(basis,Pa-Pb);
}

arma::vec lowdin_charges(const BasisSet & basis, const arma::mat & P) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Get half overlap
  arma::mat Sh, Sinvh;
  S_half_invhalf(S,Sh,Sinvh);

  // Compute ShPSh
  arma::mat SPS=Sh*P*Sh;

  arma::vec q(basis.get_Nnuc());
  q.zeros();

  // Loop over nuclei
  for(size_t ii=0;ii<basis.get_Nnuc();ii++) {
    // Get shells on nucleus
    std::vector<GaussianShell> ifuncs=basis.get_funcs(ii);
    
    // Loop over shells
    for(size_t fi=0;fi<ifuncs.size();fi++) {
      size_t ifirst=ifuncs[fi].get_first_ind();
      size_t ilast=ifuncs[fi].get_last_ind();
      
      // Loop over functions
      for(size_t i=ifirst;i<=ilast;i++)
	q(ii)+=SPS(i,i);
    }
  }

  return q;
}

arma::mat lowdin_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Get half overlap
  arma::mat Sh, Sinvh;
  S_half_invhalf(S,Sh,Sinvh);

  // Compute ShPSh
  arma::mat SPaS=Sh*Pa*Sh;
  arma::mat SPbS=Sh*Pb*Sh;

  arma::mat q(basis.get_Nnuc(),3);
  q.zeros();

  // Loop over nuclei
  for(size_t ii=0;ii<basis.get_Nnuc();ii++) {
    // Get shells on nucleus
    std::vector<GaussianShell> ifuncs=basis.get_funcs(ii);
    
    // Loop over shells
    for(size_t fi=0;fi<ifuncs.size();fi++) {
      size_t ifirst=ifuncs[fi].get_first_ind();
      size_t ilast=ifuncs[fi].get_last_ind();
      
      // Loop over functions
      for(size_t i=ifirst;i<=ilast;i++) {
	q(ii,0)+=SPaS(i,i);
	q(ii,1)+=SPbS(i,i);
      }
    }

    // Total charge
    q(ii,2)=q(ii,0)+q(ii,1);
    
  }

  return q;
}

void lowdin_analysis(const BasisSet & basis, const arma::mat & P) {
  // Get charges
  arma::vec q=lowdin_charges(basis,P);

  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      q(inuc)+=nuc.Z;
  }

  printf("\nLöwdin charges\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i));
  printf("Sum of Löwdin charges %e\n",arma::sum(q));
}

void lowdin_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  // Get charges
  arma::mat q=lowdin_charges(basis,Pa,Pb);

  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      q(inuc,2)+=nuc.Z;
  }

  printf("\nLöwdin charges: alpha, beta, total (incl. nucleus)\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i,0), q(i,1), q(i,2));
  printf("Sum of Löwdin charges %e\n",arma::sum(q.col(2)));
}

arma::vec nuclear_density(const BasisSet & basis, const arma::mat & P) {
  arma::vec ret(basis.get_Nnuc());
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++)
    ret(inuc)=compute_density(P,basis,basis.get_nuclear_coords(inuc));
  return ret;
}

void becke_analysis(const BasisSet & basis, const arma::mat & P) {
  // Get charges
  arma::vec q=becke_charges(basis,P);

  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      q(inuc)+=nuc.Z;
  }

  printf("\nBecke charges\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i));
  printf("Sum of Becke charges %e\n",arma::sum(q));
}

void becke_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  // Get charges
  arma::mat q=becke_charges(basis,Pa,Pb);

  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      q(inuc,2)+=nuc.Z;
  }

  printf("\nBecke charges: alpha, beta, total (incl. nucleus)\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i,0), q(i,1), q(i,2));
  printf("Sum of Becke charges %e\n",arma::sum(q.col(2)));
}

arma::vec becke_charges(const BasisSet & basis, const arma::mat & P) {
  arma::vec q(basis.get_Nnuc());

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,false);
  // Construct grid
  intgrid.construct_becke(1e-5);
  // Evaluate overlaps
  std::vector<arma::mat> Sat=intgrid.eval_overlaps();

  // Loop over atoms
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    // Compute charges
    q(inuc)=-arma::trace(P*Sat[inuc]);
  }

  return q;
}

arma::mat becke_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  arma::mat q(basis.get_Nnuc(),3);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,false);
  // Construct grid
  intgrid.construct_becke(1e-5);
  // Evaluate overlaps
  std::vector<arma::mat> Sat=intgrid.eval_overlaps();

  // Loop over atoms
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    // Compute charges
    q(inuc,0)=-arma::trace(Pa*Sat[inuc]);
    q(inuc,1)=-arma::trace(Pb*Sat[inuc]);
    q(inuc,2)=q(inuc,0)+q(inuc,1);
  }

  return q;
}


void hirshfeld_analysis(const BasisSet & basis, const arma::mat & P, std::string method) {
  // Get charges
  double Nelnum;
  arma::vec q=hirshfeld_charges(basis,P,method,Nelnum);

  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      q(inuc)+=nuc.Z;
  }

  printf("\nHirshfeld charges\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i));
  printf("Sum of Hirshfeld charges %e\n",arma::sum(q));
  printf("Integral over density %.8f\n",Nelnum);
}

void hirshfeld_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, std::string method) {
  // Get charges
  double Nelnum;
  arma::mat q=hirshfeld_charges(basis,Pa,Pb,method,Nelnum);

  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      q(inuc,2)+=nuc.Z;
  }

  printf("\nHirshfeld charges: alpha, beta, total (incl. nucleus)\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i,0), q(i,1), q(i,2));
  printf("Sum of Hirshfeld charges %e\n",arma::sum(q.col(2)));
  printf("Integral over density %.8f\n",Nelnum);
}

arma::vec hirshfeld_charges(const BasisSet & basis, const arma::mat & P, std::string method, double & Nelnum) {
  arma::vec q(basis.get_Nnuc(),1);

  // Hirshfeld atomic charges
  Hirshfeld hirsh;
  hirsh.compute(basis,method);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,false);
  // Construct grid
  intgrid.construct_hirshfeld(hirsh,1e-5);

  // Evaluate overlaps
  std::vector<arma::mat> Sat=intgrid.eval_hirshfeld_overlaps(hirsh);
  // Evaluate densities
  Nelnum=intgrid.compute_Nel(P);

  // Loop over atoms
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    // Compute charges
    q(inuc)=-arma::trace(P*Sat[inuc]);
  }

  return q;
}

arma::mat hirshfeld_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, std::string method, double & Nelnum) {
  arma::mat q(basis.get_Nnuc(),3);

  // Hirshfeld atomic charges
  Hirshfeld hirsh;
  hirsh.compute(basis,method);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,false);
  // Construct grid
  intgrid.construct_hirshfeld(hirsh,1e-5);

  // Evaluate overlaps
  std::vector<arma::mat> Sat=intgrid.eval_hirshfeld_overlaps(hirsh);
  // Evaluate densities
  Nelnum=intgrid.compute_Nel(Pa,Pb);

  // Loop over atoms
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    // Compute charges
    q(inuc,0)=-arma::trace(Pa*Sat[inuc]);
    q(inuc,1)=-arma::trace(Pb*Sat[inuc]);
    q(inuc,2)=q(inuc,0)+q(inuc,1);
  }

  return q;
}

void bader_analysis(const BasisSet & basis, const arma::mat & P) {
  Bader bader;
  bader.fill(basis,P);
  bader.analysis();

  arma::vec q=-bader.nuclear_charges();
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      q(inuc,0)+=nuc.Z;
  }

  printf("\nBader charges\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i));
  printf("Sum of Bader charges %e\n",arma::sum(q));
}

void population_analysis(const BasisSet & basis, const arma::mat & P) {

  // Mulliken overlap
  arma::mat mulov=mulliken_overlap(basis,P);
  // Mulliken charges
  arma::vec mulq=-sum(mulov);
  for(size_t i=0;i<basis.get_Nnuc();i++) {
    nucleus_t nuc=basis.get_nucleus(i);
    if(!nuc.bsse)
      mulq(i)+=nuc.Z;
  }

  // Bond order
  arma::mat bord=bond_order(basis,P);

  // Electron density at nuclei
  arma::vec nucd=nuclear_density(basis,P);

  printf("\nElectron density at nuclei\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), nucd(i));

  printf("\nMulliken charges\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), mulq(i));
  printf("Sum of Mulliken charges %e\n",arma::sum(mulq));

  //  becke_analysis(basis,P);

  // These generate a lot of output
  /*
  mulov.print("Mulliken overlap");
  bord.print("Bond order");
  */
}

void population_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  arma::mat P=Pa+Pb;

  // Mulliken overlap
  arma::mat mulova=mulliken_overlap(basis,Pa);
  arma::mat mulovb=mulliken_overlap(basis,Pb);
  // Mulliken charges
  arma::mat mulq(basis.get_Nnuc(),3);
  mulq.col(0)=-arma::trans(sum(mulova));
  mulq.col(1)=-arma::trans(sum(mulovb));
  mulq.col(2)=mulq.col(0)+mulq.col(1);
  for(size_t i=0;i<basis.get_Nnuc();i++) {
    nucleus_t nuc=basis.get_nucleus(i);
    if(!nuc.bsse) {
      mulq(i,2)+=nuc.Z;
    }
  }

  // Bond order
  arma::mat bord=bond_order(basis,Pa,Pb);

  // Electron density at nuclei
  arma::vec nucd_a=nuclear_density(basis,Pa);
  arma::vec nucd_b=nuclear_density(basis,Pb);

  // Total density
  arma::mat nucd(nucd_a.n_elem,3);
  nucd.col(0)=nucd_a;
  nucd.col(1)=nucd_b;
  nucd.col(2)=nucd_a+nucd_b;

  printf("\nElectron density at nuclei: alpha, beta, total\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), nucd(i,0), nucd(i,1), nucd(i,2));

  printf("\nMulliken charges: alpha, beta, total (incl. nucleus)\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), mulq(i,0), mulq(i,1), mulq(i,2));
  printf("Sum of Mulliken charges %e\n",arma::sum(mulq.col(2)));

  //  becke_analysis(basis,Pa,Pb);

  // These generate a lot of output
  /*
  mulov.print("Mulliken overlap");
  bord.print("Bond order");
  */
}


double darwin_1e(const BasisSet & basis, const arma::mat & P) {
  // Energy
  double E=0.0;
  nucleus_t nuc;

  // Loop over nuclei
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    // Get nucleus
    nuc=basis.get_nucleus(inuc);

    if(!nuc.bsse)
      // Don't do correction for BSSE nuclei
      E+=nuc.Z*compute_density(P,basis,nuc.r);
  }

  // Plug in the constant terms
  E*=0.5*M_PI*FINESTRUCT*FINESTRUCT;

  return E;
}
