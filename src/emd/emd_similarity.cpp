/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2014
 * Copyright (c) 2010-2014, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "emd_similarity.h"
#include "lebedev.h"
#include "chebyshev.h"
#include "gto_fourier.h"
#include "emd_gto.h"
#include "timer.h"

// Debug routines?
//#define DEBUGSIM

double similarity_quadrature(const std::vector<double> & rad, const std::vector<double> & wrad, const std::vector<lebedev_point_t> & angmesh, const std::vector< std::vector<double> > & emd_a, const std::vector< std::vector<double> > & emd_b, int k, bool sphave) {

  // Similarity
  double sim=0.0;

  if(sphave) {
    // Spherical averaging. Loop over radii
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sim)
#endif
    for(size_t ir=0;ir<rad.size();ir++) {
      // Average of A
      double a_ave=0.0;
      for(size_t iang=0;iang<angmesh.size();iang++)
	a_ave+=emd_a[ir][iang]*angmesh[iang].w;
      a_ave/=4.0*M_PI;
      // Average of B
      double b_ave=0.0;
      for(size_t iang=0;iang<angmesh.size();iang++)
	b_ave+=emd_b[ir][iang]*angmesh[iang].w;
      b_ave/=4.0*M_PI;

      // Increment similarity measure
      sim+=std::pow(rad[ir],2*k+2)*a_ave*b_ave*wrad[ir];
    }

  } else {
    // No averaging
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sim)
#endif
    for(size_t ir=0;ir<rad.size();ir++) {
      
      // Contribution from current radial shell
      double c=0.0;
      for(size_t iang=0;iang<angmesh.size();iang++)
	  c+=emd_a[ir][iang]*emd_b[ir][iang]*angmesh[iang].w;
      
      // Increment measure
      sim+=std::pow(rad[ir],2*k+2)*c*wrad[ir];
    }
  }

  return sim;
}

arma::vec emd_moments(const std::vector<double> & rad, const std::vector<double> & wrad, const std::vector<lebedev_point_t> & angmesh, const std::vector< std::vector<double> > & emd) {
  // Moments
  arma::vec mom(7);
  mom.zeros();

  // Loop over radii
  for(size_t ir=0;ir<rad.size();ir++) {
    // Average of A
    double ave=0.0;
    for(size_t iang=0;iang<angmesh.size();iang++)
      ave+=emd[ir][iang]*angmesh[iang].w;
    
    // Increment moments
    for(int k=-2;k<=4;k++)
      mom(k+2)+=std::pow(rad[ir],k+2)*ave*wrad[ir];
  }

  return mom;
}

void fill_mesh(const BasisSet & basis, const arma::mat & P, const std::vector<double> & rad, const std::vector<lebedev_point_t> & angmesh, std::vector< std::vector<double> > & emd) {
  // List of identical functions
  std::vector< std::vector<size_t> > idents;
  // Fourier transforms
  std::vector< std::vector<GTO_Fourier> > fourier=fourier_expand(basis,idents);

  emd.resize(rad.size());
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t ir=0;ir<rad.size();ir++) {
    emd[ir].resize(angmesh.size());
    for(size_t iang=0;iang<angmesh.size();iang++) {
      // Coordinate is
      double px=rad[ir]*angmesh[iang].x;
      double py=rad[ir]*angmesh[iang].y;
      double pz=rad[ir]*angmesh[iang].z;
      
      emd[ir][iang]=eval_emd(basis, P, fourier, idents, px, py, pz);
    }
  }
}

arma::cube emd_overlap(const BasisSet & basis_a, const arma::mat & P_a, const BasisSet & basis_b, const arma::mat & P_b, int nrad, int lmax, bool verbose) {
  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev(nrad,rad,wrad);

  // Get angular quadrature rule
  std::vector<lebedev_point_t> angmesh=lebedev_sphere(lmax);

  Timer t;

  if(verbose) {
    printf("\n%lu point radial grid, %lu point angular grid, totalling %lu points for the similarity integrals.\n",(long unsigned) rad.size(), (long unsigned) angmesh.size(), (long unsigned) (rad.size()*angmesh.size()));
    printf("Computing reference  EMD ... ");
    fflush(stdout);
    t.set();
  }

  // Compute EMDs on mesh
  std::vector< std::vector<double> > emd_a;
  fill_mesh(basis_a,P_a,rad,angmesh,emd_a);

  if(verbose) {
    printf("done (%s).\n",t.elapsed().c_str());
    printf("Computing comparison EMD ... ");
    fflush(stdout);
    t.set();
  }
  
  std::vector< std::vector<double> > emd_b;
  fill_mesh(basis_b,P_b,rad,angmesh,emd_b);

  if(verbose) {
    printf("done (%s).\n",t.elapsed().c_str());
    fflush(stdout);

#ifdef DEBUGSIM
    // Compute moments
    arma::vec amom=emd_moments(rad,wrad,angmesh,emd_a);
    arma::vec bmom=emd_moments(rad,wrad,angmesh,emd_b);

    printf("\nMoments of the radial EMD computed on the grid (for comparison with accurate values)\n");
    printf("%4s\t%9s\t%9s\n","k","A","B");
    for(int k=-2;k<=4;k++)
      printf("%4i\t%e\t%e\n",k,amom(k+2),bmom(k+2));
    printf("\n");
    fflush(stdout);
#endif
    
    t.set();
  }

  // Get the similarity measures
  arma::cube ret(4,3,2);
  for(int k=-1;k<=2;k++)
    for(int ave=0;ave<=1;ave++) {
      ret(1+k,0,ave)=similarity_quadrature(rad,wrad,angmesh,emd_a,emd_a,k,ave); // AA
      ret(1+k,1,ave)=similarity_quadrature(rad,wrad,angmesh,emd_b,emd_b,k,ave); // BB
      ret(1+k,2,ave)=similarity_quadrature(rad,wrad,angmesh,emd_a,emd_b,k,ave); // AB

      //      printf("k = %i, ave = %i: AA = %e, BB = %e, AB = %e\n",k,ave,ret(1+k,0,ave),ret(1+k,1,ave),ret(1+k,2,ave));
    }

  if(verbose) {
    printf("Similarity moments computed in %s.\n\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  return ret;
}

std::vector<double> evaluate_projection(const BasisSet & basis, const arma::mat & P, const std::vector<double> rad, int l, int m) {
  int Nel=round(arma::trace(P*basis.overlap()));

  GaussianEMDEvaluator *poseval=new GaussianEMDEvaluator(basis,P,l,std::abs(m));
  GaussianEMDEvaluator *negeval=NULL;
  if(m!=0)
    negeval=new GaussianEMDEvaluator(basis,P,l,-std::abs(m));
  EMD emd(poseval, negeval, Nel, l, m);

  std::vector<double> ret(rad.size());
  for(size_t i=0;i<rad.size();i++)
    ret[i]=emd.eval(rad[i]);

  delete poseval;
  if(m!=0) delete negeval;

  return ret;
}

// Offset in array
int lm_offset(int l, int m) {
  return l*(l-1)/2 + (l+m);
}

double similarity_quadrature_semi(const std::vector<double> & rad, const std::vector<double> & wrad, const std::vector< std::vector<double> > & emd_a, const std::vector< std::vector<double> > & emd_b, int k, int lmax) {
  // Compute total product density
  std::vector<double> proddens;
  proddens.assign(rad.size(),0.0);
  for(int l=0;l<=lmax;l+=2)
    for(int m=-l;m<=l;m++)
      for(size_t ir=0;ir<rad.size();ir++)
	proddens[ir]+=emd_a[lm_offset(l,m)][ir]*emd_b[lm_offset(l,m)][ir];
  
  // Similarity
  double sim=0.0;
  for(size_t ir=0;ir<rad.size();ir++) {
    // Increment measure
    sim+=std::pow(rad[ir],2*k+2)*proddens[ir]*wrad[ir];
  }
  
  // Remove extra 4pi factor
  sim/=4.0*M_PI;

  return sim;
}

arma::cube emd_overlap_semi(const BasisSet & basis_a, const arma::mat & P_a, const BasisSet & basis_b, const arma::mat & P_b, int nrad, int lmax, bool verbose) {
  // Get Chebyshev nodes and weights for radial part
  std::vector<double> rad, wrad;
  radial_chebyshev(nrad,rad,wrad);

  Timer t;

  // Radial meshes of EMDs
  std::vector< std::vector<double> > emd_a(lm_offset(lmax,lmax)+1);
  std::vector< std::vector<double> > emd_b(lm_offset(lmax,lmax)+1);


  if(verbose) {
    printf("\n%lu point radial grid, coupling up to l=%i.\n",(long unsigned) rad.size(), lmax);
    fflush(stdout);
    t.set();
  }

  // Returned similarities
  arma::cube ret(4,3,2);

  // Loop over l
  for(int l=0;l<=lmax;l+=2) {
    // Compute reference
    printf("Computing l = %-2i reference  EMD ... ",l);
    fflush(stdout);
    for(int m=-l;m<=l;m++)
      emd_a[lm_offset(l,m)]=evaluate_projection(basis_a,P_a,rad,l,m);
    
    if(verbose) {
      printf("done (%s).\n",t.elapsed().c_str());
      printf("Computing l = %-2i comparison EMD ... ",l);
      fflush(stdout);
      t.set();
    }
    
    for(int m=-l;m<=l;m++)
      emd_b[lm_offset(l,m)]=evaluate_projection(basis_b,P_b,rad,l,m);
    
    if(verbose) {
      printf("done (%s).\n",t.elapsed().c_str());
      fflush(stdout);
      t.set();
    }
  }
    
  // Get the similarity measures
  for(int k=-1;k<=2;k++) {
    // Full result
    ret(1+k,0,0)=similarity_quadrature_semi(rad,wrad,emd_a,emd_a,k,lmax); // AA
    ret(1+k,1,0)=similarity_quadrature_semi(rad,wrad,emd_b,emd_b,k,lmax); // BB
    ret(1+k,2,0)=similarity_quadrature_semi(rad,wrad,emd_a,emd_b,k,lmax); // AB
    
    // Spherical average
    ret(1+k,0,1)=similarity_quadrature_semi(rad,wrad,emd_a,emd_a,k,0); // AA
    ret(1+k,1,1)=similarity_quadrature_semi(rad,wrad,emd_b,emd_b,k,0); // BB
    ret(1+k,2,1)=similarity_quadrature_semi(rad,wrad,emd_a,emd_b,k,0); // AB
    
    //      printf("k = %i, ave = %i: AA = %e, BB = %e, AB = %e\n",k,ave,ret(1+k,0,ave),ret(1+k,1,ave),ret(1+k,2,ave));
  }
  
  if(verbose) {
    printf("Similarity moments computed in %s.\n\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }
  
  return ret;
}

arma::cube emd_similarity(const arma::cube & emd, int Nela, int Nelb) {
  // Compute shape function overlap                                                                                                                                                                              
  arma::cube sh(4,7,2);
  sh.zeros();
  for(size_t s=0;s<sh.n_slices;s++)
    for(size_t k=0;k<sh.n_rows;k++) {
      sh(k,0,s)=emd(k,0,s);
      sh(k,1,s)=emd(k,1,s);
      sh(k,2,s)=emd(k,2,s);
      sh(k,3,s)=emd(k,0,s)/(Nela*Nela);
      sh(k,4,s)=emd(k,1,s)/(Nelb*Nelb);
      sh(k,5,s)=emd(k,2,s)/(Nela*Nelb);
      sh(k,6,s)=sqrt(sh(k,3,s) + sh(k,4,s) - 2*sh(k,5,s));
    }

  return sh;
}
