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
      // Average of B
      double b_ave=0.0;
      for(size_t iang=0;iang<angmesh.size();iang++)
	b_ave+=emd_b[ir][iang]*angmesh[iang].w;

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

arma::mat emd_similarity(const BasisSet & basis_a, const arma::mat & P_a, const BasisSet & basis_b, const arma::mat & P_b, int nrad, int lmax, bool verbose) {
  // Get evaluators
  std::vector< std::vector<size_t> > idents_a, idents_b;
  std::vector< std::vector<GTO_Fourier> > fourier_a=fourier_expand(basis_a,idents_a);
  std::vector< std::vector<GTO_Fourier> > fourier_b=fourier_expand(basis_b,idents_b);
  
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

  // EMDs
  std::vector< std::vector<double> > emd_a(rad.size()), emd_b(rad.size());
  for(size_t ir=0;ir<rad.size();ir++) {
    emd_a[ir].resize(angmesh.size());
    emd_b[ir].resize(angmesh.size());
  }

  Timer t;

  if(verbose) {
    printf("Using %lu points for the similarity integrals.\n",(long unsigned) (rad.size()*angmesh.size()));
    printf("Computing reference  EMD ... ");
    fflush(stdout);
    t.set();
  }

  // Compute EMDs on mesh
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t ir=0;ir<rad.size();ir++)
    for(size_t iang=0;iang<angmesh.size();iang++) {
      // Coordinate is
      double px=rad[ir]*angmesh[iang].x;
      double py=rad[ir]*angmesh[iang].y;
      double pz=rad[ir]*angmesh[iang].z;

      emd_a[ir][iang]=eval_emd(basis_a, P_a, fourier_a, idents_a, px, py, pz);
    }

  if(verbose) {
    printf("done (%s).\n",t.elapsed().c_str());
    printf("Computing comparison EMD ... ");
    fflush(stdout);
    t.set();
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(size_t ir=0;ir<rad.size();ir++)
    for(size_t iang=0;iang<angmesh.size();iang++) {
      // Coordinate is
      double px=rad[ir]*angmesh[iang].x;
      double py=rad[ir]*angmesh[iang].y;
      double pz=rad[ir]*angmesh[iang].z;

      emd_b[ir][iang]=eval_emd(basis_b, P_b, fourier_b, idents_b, px, py, pz);
    }

  if(verbose) {
    printf("done (%s).\n",t.elapsed().c_str());
    fflush(stdout);

    // Compute moments
    arma::vec amom=emd_moments(rad,wrad,angmesh,emd_a);
    arma::vec bmom=emd_moments(rad,wrad,angmesh,emd_b);

    printf("\nMoments of the EMD computed on the grid\n");
    printf("%4s\t%9s\t%9s\n","k","A","B");
    for(int k=-2;k<=4;k++)
      printf("%4i\t%e\t%e\n",k,amom(k+2),bmom(k+2));
    printf("\n");
    fflush(stdout);
    
    t.set();
  }

  

  // Get the similarity measures
  arma::mat ret(4,2);
  for(int k=-1;k<4;k++) {
    double AA=similarity_quadrature(rad,wrad,angmesh,emd_a,emd_a,k,false);
    double AB=similarity_quadrature(rad,wrad,angmesh,emd_a,emd_b,k,false);
    double BB=similarity_quadrature(rad,wrad,angmesh,emd_b,emd_b,k,false);
    ret(1+k,0)=sqrt(AA+BB-2*AB);

    double AAa=similarity_quadrature(rad,wrad,angmesh,emd_a,emd_a,k,true);
    double ABa=similarity_quadrature(rad,wrad,angmesh,emd_a,emd_b,k,true);
    double BBa=similarity_quadrature(rad,wrad,angmesh,emd_b,emd_b,k,true);
    ret(1+k,1)=sqrt(AAa+BBa-2*ABa);
  }

  if(verbose) {
    printf("Similarity moments computed in %s.\n",t.elapsed().c_str());
    fflush(stdout);
    t.set();
  }

  return ret;
}

