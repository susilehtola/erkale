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
#include "badergrid.h"
#include "stockholder.h"
#include "hirshfeldi.h"
#include "linalg.h"
#include "settings.h"

void print_analysis(const BasisSet & basis, const std::string & msg, const arma::vec & q) {
  printf("\n%s charges\n",msg.c_str());
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i));
  printf("Sum of %s charges %e\n",msg.c_str(),arma::sum(q));
}

void print_analysis(const BasisSet & basis, const std::string & msg, const arma::mat & q) {
  printf("\n%s charges: alpha, beta, total (incl. nucleus)\n",msg.c_str());
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f % 15.6f % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), q(i,0), q(i,1), q(i,2));
  printf("Sum of %s charges %e\n",msg.c_str(),arma::sum(q.col(2)));
}

arma::vec add_nuclear_charges(const BasisSet & basis, const arma::vec & q) {
  if(basis.get_Nnuc()!=q.n_elem) {
    ERROR_INFO();
    throw std::runtime_error("Nuclear charge vector does not match amount of nuclei in system.\n");
  }

  arma::vec qr(q);
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++) {
    nucleus_t nuc=basis.get_nucleus(inuc);
    if(!nuc.bsse)
      qr(inuc)+=nuc.Z;
  }

  return qr;
}

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

void mulliken_analysis(const BasisSet & basis, const arma::mat & P) {
  // Get charges
  arma::vec q=mulliken_charges(basis,P);
  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Mulliken",q);
}

void mulliken_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  // Get charges
  arma::mat q=mulliken_charges(basis,Pa,Pb);
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Mulliken",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Mulliken spin",qs);
}

arma::vec mulliken_charges(const BasisSet & basis, const arma::mat & P) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Compute PS
  arma::mat PS=P*S;

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
	q(ii)-=PS(i,i);
    }
  }

  return q;
}

arma::mat mulliken_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Compute PS
  arma::mat PaS=Pa*S;
  arma::mat PbS=Pb*S;

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
	q(ii,0)-=PaS(i,i);
	q(ii,1)-=PbS(i,i);
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
  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Löwdin",q);
}


void lowdin_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  // Get charges
  arma::mat q=lowdin_charges(basis,Pa,Pb);
  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Löwdin",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Löwdin spin",qs);
}

extern Settings settings;

arma::vec lowdin_charges(const BasisSet & basis, const arma::mat & P) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Get half overlap
  arma::mat Sh, Sinvh;
  S_half_invhalf(S,Sh,Sinvh,settings.get_double("LinDepThresh"));

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
	q(ii)-=SPS(i,i);
    }
  }

  return q;
}

arma::mat lowdin_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Get half overlap
  arma::mat Sh, Sinvh;
  S_half_invhalf(S,Sh,Sinvh,settings.get_double("LinDepThresh"));

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
	q(ii,0)-=SPaS(i,i);
	q(ii,1)-=SPbS(i,i);
      }
    }

    // Total charge
    q(ii,2)=q(ii,0)+q(ii,1);

  }

  return q;
}

void IAO_analysis(const BasisSet & basis, const arma::mat & C, std::string minbas) {
  // Get charges
  arma::vec q=2*IAO_charges(basis,C,minbas);
  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"IAO",q);
}

void IAO_analysis(const BasisSet & basis, const arma::cx_mat & C, std::string minbas) {
  // Get charges
  arma::vec q=2*IAO_charges(basis,C,minbas);
  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"IAO",q);
}

void IAO_analysis(const BasisSet & basis, const arma::mat & Ca, const arma::mat & Cb, std::string minbas) {
  // Get charges
  arma::vec qa=IAO_charges(basis,Ca,minbas);
  arma::vec qb=IAO_charges(basis,Cb,minbas);

  arma::mat q(qa.n_elem,3);
  q.col(0)=qa;
  q.col(1)=qb;
  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(0)+q.col(1));

  print_analysis(basis,"IAO",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"IAO spin",qs);
}

void IAO_analysis(const BasisSet & basis, const arma::cx_mat & Ca, const arma::cx_mat & Cb, std::string minbas) {
  // Get charges
  arma::vec qa=IAO_charges(basis,Ca,minbas);
  arma::vec qb=IAO_charges(basis,Cb,minbas);

  arma::mat q(qa.n_elem,3);
  q.col(0)=qa;
  q.col(1)=qb;
  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(0)+q.col(1));

  print_analysis(basis,"IAO",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"IAO spin",qs);
}

arma::vec IAO_charges(const BasisSet & basis, const arma::mat & C, std::string minbas) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Get IAO orbitals
  std::vector< std::vector<size_t> > idx;
  arma::mat iao=construct_IAO(basis,C,idx,true,minbas);

  arma::vec q(basis.get_Nnuc());
  q.zeros();

  // Construct density matrix
  arma::mat P(C*arma::trans(C));

  // Loop over nuclei
  for(size_t ii=0;ii<basis.get_Nnuc();ii++)
    // Loop over functions on nucleus
    for(size_t fi=0;fi<idx[ii].size();fi++) {
      // IAO orbital index is
      size_t io=idx[ii][fi];

      q(ii)-=arma::as_scalar(arma::trans(iao.col(io))*S*P*S*iao.col(io));
    }

  return q;
}

arma::vec IAO_charges(const BasisSet & basis, const arma::cx_mat & C, std::string minbas) {
  // Get overlap matrix
  arma::mat S=basis.overlap();

  // Get IAO orbitals
  std::vector< std::vector<size_t> > idx;
  arma::cx_mat iao=construct_IAO(basis,C,idx,true,minbas);

  arma::vec q(basis.get_Nnuc());
  q.zeros();

  // Construct density matrix
  arma::cx_mat P(C*arma::trans(C));

  // Loop over nuclei
  for(size_t ii=0;ii<basis.get_Nnuc();ii++)
    // Loop over functions on nucleus
    for(size_t fi=0;fi<idx[ii].size();fi++) {
      // IAO orbital index is
      size_t io=idx[ii][fi];

      q(ii)-=std::real(arma::as_scalar(arma::trans(iao.col(io))*S*P*S*iao.col(io)));
    }

  return q;
}

arma::vec nuclear_density(const BasisSet & basis, const arma::mat & P) {
  arma::vec ret(basis.get_Nnuc());
  for(size_t inuc=0;inuc<basis.get_Nnuc();inuc++)
    ret(inuc)=compute_density(P,basis,basis.get_nuclear_coords(inuc));
  return ret;
}

void becke_analysis(const BasisSet & basis, const arma::mat & P, double tol) {
  // Get charges
  arma::vec q=becke_charges(basis,P,tol);
  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Becke",q);
}

void becke_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  // Get charges
  arma::mat q=becke_charges(basis,Pa,Pb,tol);
  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Becke",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Becke spin",qs);
}

arma::vec becke_charges(const BasisSet & basis, const arma::mat & P, double tol) {
  arma::vec q(basis.get_Nnuc());

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_becke(tol);

  return -intgrid.compute_atomic_Nel(P);
}

arma::mat becke_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  arma::mat q(basis.get_Nnuc(),3);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_becke(tol);

  q.col(0)=-intgrid.compute_atomic_Nel(Pa);
  q.col(1)=-intgrid.compute_atomic_Nel(Pb);
  q.col(2)=q.col(0)+q.col(1);

  return q;
}

void hirshfeld_analysis(const BasisSet & basis, const arma::mat & P, std::string method, double tol) {
  // Get charges
  arma::vec q=hirshfeld_charges(basis,P,method,tol);

  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Hirshfeld",q);
}

void hirshfeld_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, std::string method, double tol) {
  // Get charges
  arma::mat q=hirshfeld_charges(basis,Pa,Pb,method,tol);

  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Hirshfeld",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Hirshfeld spin",qs);
}

arma::vec hirshfeld_charges(const BasisSet & basis, const arma::mat & P, std::string method, double tol) {
  arma::vec q(basis.get_Nnuc(),1);

  // Hirshfeld atomic charges
  Hirshfeld hirsh;
  if(stricmp(method,"Load")==0)
    hirsh.load(basis);
  else
    hirsh.compute(basis,method);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_hirshfeld(hirsh,tol);

  return -intgrid.compute_atomic_Nel(hirsh,P);
}

arma::mat hirshfeld_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, std::string method, double tol) {
  arma::mat q(basis.get_Nnuc(),3);

  // Hirshfeld atomic charges
  Hirshfeld hirsh;
  if(stricmp(method,"Load")==0)
    hirsh.load(basis);
  else
    hirsh.compute(basis,method);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_hirshfeld(hirsh,tol);

  q.col(0)=-intgrid.compute_atomic_Nel(hirsh,Pa);
  q.col(1)=-intgrid.compute_atomic_Nel(hirsh,Pb);
  q.col(2)=q.col(0)+q.col(1);

  return q;
}

void iterative_hirshfeld_analysis(const BasisSet & basis, const arma::mat & P, std::string method, double tol) {
  // Get charges
  arma::vec q=iterative_hirshfeld_charges(basis,P,method,tol);

  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Iterative Hirshfeld",q);
}

void iterative_hirshfeld_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, std::string method, double tol) {
  // Get charges
  arma::mat q=iterative_hirshfeld_charges(basis,Pa,Pb,method,tol);

  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Iterative Hirshfeld",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Iterative Hirshfeld spin",qs);
}

arma::vec iterative_hirshfeld_charges(const BasisSet & basis, const arma::mat & P, std::string method, double tol) {
  arma::vec q(basis.get_Nnuc(),1);

  // Iterative Hirshfeld atomic charges
  HirshfeldI hirshi;
  if(stricmp(method,"Load")==0)
    hirshi.compute_load(basis,P,tol);
  else
    hirshi.compute(basis,P,method,tol);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_hirshfeld(hirshi.get(),tol);

  return -intgrid.compute_atomic_Nel(hirshi.get(),P);
}

arma::mat iterative_hirshfeld_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, std::string method, double tol) {
  arma::mat q(basis.get_Nnuc(),3);

  // Iterative Hirshfeld atomic charges
  HirshfeldI hirshi;
  if(stricmp(method,"Load")==0)
    hirshi.compute_load(basis,Pa+Pb,tol);
  else
    hirshi.compute(basis,Pa+Pb,method,tol);

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_hirshfeld(hirshi.get(),tol);

  q.col(0)=-intgrid.compute_atomic_Nel(hirshi.get(),Pa);
  q.col(1)=-intgrid.compute_atomic_Nel(hirshi.get(),Pb);
  q.col(2)=q.col(0)+q.col(1);

  return q;
}

void stockholder_analysis(const BasisSet & basis, const arma::mat & P, double tol) {
  // Get charges
  arma::vec q=stockholder_charges(basis,P,tol);

  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Stockholder",q);
}

void stockholder_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  // Get charges
  arma::mat q=stockholder_charges(basis,Pa,Pb,tol);

  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Stockholder",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Stockholder spin",qs);
}

arma::vec stockholder_charges(const BasisSet & basis, const arma::mat & P, double tol) {
  arma::vec q(basis.get_Nnuc(),1);

  // Stockholder atomic charges
  Stockholder stock(basis,P);
  // Helper
  Hirshfeld hirsh=stock.get();

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_hirshfeld(hirsh,tol);

  return -intgrid.compute_atomic_Nel(hirsh,P);
}

arma::mat stockholder_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  arma::mat q(basis.get_Nnuc(),3);

  // Stockholder atomic charges
  Stockholder stock(basis,Pa+Pb);
  // Helper
  Hirshfeld hirsh=stock.get();

  // Helper. Non-verbose operation
  DFTGrid intgrid(&basis,true);
  // Construct grid
  intgrid.construct_hirshfeld(hirsh,tol);

  q.col(0)=-intgrid.compute_atomic_Nel(hirsh,Pa);
  q.col(1)=-intgrid.compute_atomic_Nel(hirsh,Pb);
  q.col(2)=q.col(0)+q.col(1);

  return q;
}

void bader_analysis(const BasisSet & basis, const arma::mat & P, double tol) {
  arma::vec q=bader_charges(basis,P,tol);
  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Bader",q);
}

void bader_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  arma::mat q=bader_charges(basis,Pa,Pb,tol);

  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Bader",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Bader spin",qs);
}

arma::vec bader_charges(const BasisSet & basis, const arma::mat & P, double tol) {
  // Helper. Non-verbose operation
  BaderGrid intgrid;
  intgrid.set(basis,true);
  // Construct grid
  intgrid.construct_bader(P,tol);

  // Get nuclear charges
  return intgrid.nuclear_charges(P);
}

arma::mat bader_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  // Helper. Non-verbose operation
  BaderGrid intgrid;
  intgrid.set(basis,true);
  // Construct grid
  intgrid.construct_bader(Pa+Pb,tol);

  arma::mat q(basis.get_Nnuc(),3);
  q.col(0)=intgrid.nuclear_charges(Pa);
  q.col(1)=intgrid.nuclear_charges(Pb);
  q.col(2)=q.col(0)+q.col(1);

  return q;
}

void voronoi_analysis(const BasisSet & basis, const arma::mat & P, double tol) {
  arma::vec q=voronoi_charges(basis,P,tol);
  // Add contribution from nuclei
  q=add_nuclear_charges(basis,q);

  print_analysis(basis,"Voronoi",q);
}

void voronoi_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  arma::mat q=voronoi_charges(basis,Pa,Pb,tol);

  // Add contribution from nuclei
  q.col(2)=add_nuclear_charges(basis,q.col(2));

  print_analysis(basis,"Voronoi",q);

  // Get charges. Note that in charges above electrons have negative charge, so switch sign
  arma::vec qs=-q.col(0)+q.col(1);
  print_analysis(basis,"Voronoi spin",qs);
}

arma::vec voronoi_charges(const BasisSet & basis, const arma::mat & P, double tol) {
  // Helper. Non-verbose operation
  BaderGrid intgrid;
  intgrid.set(basis,true);
  // Construct grid
  intgrid.construct_voronoi(tol);

  // Get nuclear charges
  return intgrid.nuclear_charges(P);
}

arma::mat voronoi_charges(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb, double tol) {
  // Helper. Non-verbose operation
  BaderGrid intgrid;
  intgrid.set(basis,true);
  // Construct grid
  intgrid.construct_voronoi(tol);

  arma::mat q(basis.get_Nnuc(),3);
  q.col(0)=intgrid.nuclear_charges(Pa);
  q.col(1)=intgrid.nuclear_charges(Pb);
  q.col(2)=q.col(0)+q.col(1);

  return q;
}


void nuclear_analysis(const BasisSet & basis, const arma::mat & P) {
  // Electron density at nuclei
  arma::vec nucd=nuclear_density(basis,P);

  printf("\nElectron density at nuclei\n");
  for(size_t i=0;i<basis.get_Nnuc();i++)
    printf("%4i %-5s % 15.6f\n",(int) i+1, basis.get_symbol_hr(i).c_str(), nucd(i));
}

void nuclear_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
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
}

void population_analysis(const BasisSet & basis, const arma::mat & P) {
  mulliken_analysis(basis,P);
  nuclear_analysis(basis,P);
}

void population_analysis(const BasisSet & basis, const arma::mat & Pa, const arma::mat & Pb) {
  mulliken_analysis(basis,Pa,Pb);
  nuclear_analysis(basis,Pa,Pb);
}

double spin_S2(const BasisSet & basis, const arma::mat & Ca, const arma::mat & Cb) {
  arma::uword Na=Ca.n_cols;
  arma::uword Nb=Cb.n_cols;

  arma::mat S(basis.overlap());
  double Sz=(Na-Nb)/2.0;

  arma::mat Sab;
  if(Nb)
    Sab=arma::trans(Ca)*S*Cb;

  double S2=Sz*(Sz+1) + Nb;
  for(arma::uword i=0;i<Na;i++)
    for(arma::uword j=0;j<Nb;j++)
      S2-=std::pow(Sab(i,j),2);

  return S2;
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
