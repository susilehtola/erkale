#include "emd_similarity.h"
#include "lebedev.h"
#include "chebyshev.h"
#include "gto_fourier.h"
#include "timer.h"

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
  std::vector<double> xc, wc;
  chebyshev(nrad,xc,wc);

  // Compute radii
  std::vector<double> rad(nrad), wrad(nrad);
  for(size_t ir=0;ir<xc.size();ir++) {
    // Calculate value of radius
    double ixc=xc.size()-1-ir;
    rad[ir]=1.0/M_LN2*log(2.0/(1.0-xc[ixc]));
    
    // Jacobian of transformation is
    double jac=1.0/M_LN2/(1.0-xc[ixc]);
    // so total quadrature weight (excluding r^2!) is
    wrad[ir]=wc[ixc]*jac;
  }

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

    // Compute moments
    arma::vec amom=emd_moments(rad,wrad,angmesh,emd_a);
    arma::vec bmom=emd_moments(rad,wrad,angmesh,emd_b);

    printf("\nMoments of the radial EMD computed on the grid (for comparison with accurate values)\n");
    printf("%4s\t%9s\t%9s\n","k","A","B");
    for(int k=-2;k<=4;k++)
      printf("%4i\t%e\t%e\n",k,amom(k+2),bmom(k+2));
    printf("\n");
    fflush(stdout);
    
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
    printf("Similarity moments computed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  return ret;
}

arma::cube emd_similarity(const arma::cube & emd, int Nela, int Nelb) {
  // Compute shape function overlap                                                                                                                                                                              
  arma::cube sh(4,4,2);
  sh.zeros();
  for(size_t s=0;s<sh.n_slices;s++)
    for(size_t k=0;k<sh.n_rows;k++) {
      sh(k,0,s)=emd(k,0,s)/(Nela*Nela);
      sh(k,1,s)=emd(k,1,s)/(Nelb*Nelb);
      sh(k,2,s)=emd(k,2,s)/(Nela*Nelb);
      sh(k,3,s)=sqrt(sh(k,0,s) + sh(k,1,s) - 2*sh(k,2,s));
    }

  return sh;
}
