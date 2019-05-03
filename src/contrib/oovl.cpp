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

#include "../settings.h"
#include "../checkpoint.h"

std::string getlegend(size_t i, size_t n) {
  if(n==1)
    return "orbital";
  else if(n==2) {
    if(i)
      return "beta orbital";
    else
      return "alpha orbital";
  } else
    throw std::logic_error("Should have not ended up here.\n");
}

Settings settings;

int main_guarded(int argc, char **argv) {
  if(argc!=3) {
    printf("Usage: %s chkpt1 chkpt2",argv[0]);
    return 1;
  }

  Checkpoint lhchk(argv[1],false);
  Checkpoint rhchk(argv[2],false);

  BasisSet lhbas;
  lhchk.read(lhbas);

  BasisSet rhbas;
  rhchk.read(rhbas);

  // Get basis set overlaps
  arma::mat S12=lhbas.overlap(rhbas);

  // Restricted calcs?
  int lhrestr, rhrestr;
  lhchk.read("Restricted",lhrestr);
  rhchk.read("Restricted",rhrestr);

  // Orbital coefficient matrices
  std::vector<arma::cx_mat> lmat, rmat;
  if(lhrestr) {
    lmat.resize(1);
    if(lhchk.exist("CW.re"))
      lhchk.cread("CW",lmat[0]);
    else {
      arma::mat C;
      lhchk.read("C",C);
      lmat[0]=C*COMPLEX1;
    }

    int Nel;
    lhchk.read("Nel-a",Nel);
    if(lmat[0].n_cols> (size_t) Nel)
      lmat[0].shed_cols(Nel,lmat[0].n_cols-1);

  } else {
    lmat.resize(2);
    if(lhchk.exist("CWa.re")) {
      lhchk.cread("CWa",lmat[0]);
      lhchk.cread("CWb",lmat[1]);
    } else {
      arma::mat Ca, Cb;
      lhchk.read("Ca",Ca);
      lhchk.read("Cb",Cb);
      lmat[0]=Ca*COMPLEX1;
      lmat[1]=Cb*COMPLEX1;
    }

    int Nel;
    lhchk.read("Nel-a",Nel);
    if(lmat[0].n_cols> (size_t) Nel)
      lmat[0].shed_cols(Nel,lmat[0].n_cols-1);
    lhchk.read("Nel-b",Nel);
    if(lmat[1].n_cols> (size_t) Nel)
      lmat[1].shed_cols(Nel,lmat[1].n_cols-1);
  }

  if(rhrestr) {
    rmat.resize(1);
    if(rhchk.exist("CW.re"))
      rhchk.cread("CW",rmat[0]);
    else {
      arma::mat C;
      rhchk.read("C",C);
      rmat[0]=C*COMPLEX1;
    }

    int Nel;
    rhchk.read("Nel-a",Nel);
    if(rmat[0].n_cols> (size_t) Nel)
      rmat[0].shed_cols(Nel,rmat[0].n_cols-1);
  } else {
    rmat.resize(2);
    if(rhchk.exist("CWa.re")) {
      rhchk.cread("CWa",rmat[0]);
      rhchk.cread("CWb",rmat[1]);
    } else {
      arma::mat Ca, Cb;
      rhchk.read("Ca",Ca);
      rhchk.read("Cb",Cb);
      rmat[0]=Ca*COMPLEX1;
      rmat[1]=Cb*COMPLEX1;
    }

    int Nel;
    rhchk.read("Nel-a",Nel);
    if(rmat[0].n_cols> (size_t) Nel)
      rmat[0].shed_cols(Nel,rmat[0].n_cols-1);
    rhchk.read("Nel-b",Nel);
    if(rmat[1].n_cols> (size_t) Nel)
      rmat[1].shed_cols(Nel,rmat[1].n_cols-1);
  }

  for(size_t il=0;il<lmat.size();il++)
    for(size_t ir=0;ir<rmat.size();ir++) {

      if(lmat.size()==2 && rmat.size()==2 && ((il && !ir) || (ir && !il)))
	continue;

      // Orbital projections
      arma::mat proj;
      {
	// With or without complex conjugate?
	arma::cx_mat Smot(arma::trans(lmat[il])*S12*rmat[ir]);
	arma::mat prot(arma::real(Smot%arma::conj(Smot)));

	arma::cx_mat Smost(arma::strans(lmat[il])*S12*rmat[ir]);
	arma::mat prost(arma::real(Smost%arma::conj(Smost)));

	// Sums
	double sumt(arma::sum(arma::sum(prot)));
	double sumst(arma::sum(arma::sum(prost)));

	if(sumt>sumst) {
	  printf("Determinants do not differ by complex conjugation\n");
	  proj=prot;
	} else {
	  printf("Determinants differ by complex conjugation\n");
	  proj=prost;
	}
      }

      printf("%s - %s projection\n",getlegend(il,lmat.size()).c_str(),getlegend(ir,rmat.size()).c_str());
      //proj.print();

      if(lmat[il].n_cols == rmat[ir].n_cols) {
	printf("Trace of density        projection  is % 10.6f, deviation from norm is %e\n",arma::sum(arma::sum(proj)),rmat[ir].n_cols-arma::sum(arma::sum(proj)));

	// Determine ordering
	arma::uvec lorder(arma::linspace<arma::uvec>(0,lmat[il].n_cols-1,lmat[il].n_cols));
	arma::uvec rorder(arma::linspace<arma::uvec>(0,rmat[ir].n_cols-1,rmat[ir].n_cols));
	for(size_t i=0;i<lorder.n_elem;i++) {
	  // Sorted matrix
	  arma::mat sproj(proj(lorder,rorder));
	  // Take submatrix
	  sproj=sproj.submat(i,i,sproj.n_rows-1,sproj.n_cols-1);

	  // Find maximum element
	  arma::uword lind, rind;
	  sproj.max(lind,rind);
	  lind+=i;
	  rind+=i;

	  // Swap indices
	  std::swap(lorder(i),lorder(lind));
	  std::swap(rorder(i),rorder(rind));
	}

	lorder.t().print("Left  ordering");
	rorder.t().print("Right ordering");

	// Sort projection matrix
	arma::mat sproj(proj(lorder,rorder));
	sproj.print("Projection matrix");

	arma::vec osort(arma::sort(arma::diagvec(sproj),"descend"));
	printf("Sum of diagonal orbital projections is % 10.6f, deviation from norm is %e\n",arma::sum(osort),rmat[ir].n_cols-arma::sum(osort));

	osort-=arma::ones<arma::vec>(osort.n_elem);
	osort.print("Minimal orbital differences");
      } else {
	proj.print();
      }

      printf("\n");
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
