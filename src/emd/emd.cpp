/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "emd.h"
#include "../gaunt.h"
#include "../lmgrid.h"
#include "../mathf.h"
#include "../stringutil.h"
#include "../timer.h"
#include "spherical_harmonics.h"

#include <cfloat>

// Value of moment of electron density in Fourier space
#define moment(i) (pow(dens[i].p,2+mom)*dens[i].d)
#define density(i) (pow(dens[i].p,2)*dens[i].d)

// Integration rules
#define roughdens(i) ((density(i-2)+4.0*density(i)+density(i+2))*(dens[i+2].p-dens[i-2].p)/6.0)
#define finedens(i)  ((density(i-2)+4.0*density(i-1)+2.0*density(i)+4.0*density(i+1)+density(i+2))*(dens[i+2].p-dens[i-2].p)/12.0)

#define roughmom(i)  ((moment(i-2)+4.0*moment(i)+moment(i+2))*(dens[i+2].p-dens[i-2].p)/6.0)
#define finemom(i)   ((moment(i-2)+4.0*moment(i-1)+2.0*moment(i)+4.0*moment(i+1)+moment(i+2))*(dens[i+2].p-dens[i-2].p)/12.0)

// Maximum number of points allowed for converging number of electrons
#define MAXPOINTS 700

// Print out moments at maximum every N seconds
#define MAXPRINTFREQ 5
//#define MAXPRINTFREQ 1e-10

/// Index of (l,m) in tables: l^2 + l + m
#define lmind(l,m) ( ((size_t) (l))*(size_t (l)) + (size_t) (l) + (size_t) (m))

/// Number of radial points when checking norm of radial functions
#define NRAD 1000
/// Normality tolerance
#define NORMTOL 1e-10

//#define DEBUG

RadialFourier::RadialFourier(int lv) {
  l=lv;
}

RadialFourier::~RadialFourier() {
}

int RadialFourier::getl() const {
  return l;
}

bool operator<(const coupl_coeff_t & lhs, const coupl_coeff_t & rhs) {
  // Sort first by l
  if(lhs.l<rhs.l)
    return true;
  else if(lhs.l==rhs.l) {
    // Then by l'

    if(lhs.lp<rhs.lp)
      return true;
    else if(lhs.lp==rhs.lp) {

      // Then by L
      if(lhs.L<rhs.L)
	return true;

      else if(lhs.L==rhs.L)
	// Finally by M
	return lhs.M<rhs.M;
    }
  }

  return false;
}

bool operator==(const coupl_coeff_t & lhs, const coupl_coeff_t & rhs) {
  return (lhs.l==rhs.l) && (lhs.lp==rhs.lp) && (lhs.L==rhs.L) && (lhs.M==rhs.M);
}

bool operator<(const total_coupl_t & lhs, const total_coupl_t & rhs) {
  if(lhs.L<rhs.L)
    return true;
  else if(lhs.L==rhs.L)
    return lhs.M<rhs.M;

  return false;
}

bool operator==(const total_coupl_t & lhs, const total_coupl_t & rhs) {
  return (lhs.L==rhs.L) && (lhs.M==rhs.M);
}

void add_coupling_term(std::vector<total_coupl_t> & v, total_coupl_t & t) {
  if(v.size()==0) {
    v.push_back(t);
  } else {
    // Get upper bound
    std::vector<total_coupl_t>::iterator high;
    high=std::upper_bound(v.begin(),v.end(),t);

    // Corresponding index is
    size_t ind=high-v.begin();

    if(ind>0 && v[ind-1]==t)
      // Found it.
      v[ind-1].c+=t.c;
    else {
      // Term does not exist, add it
      v.insert(high,t);
    }
  }
}

EMDEvaluator::EMDEvaluator() {
}

EMDEvaluator::EMDEvaluator(const std::vector< std::vector<size_t> > & idfuncsv, const std::vector< std::vector<ylmcoeff_t> > & clm, const std::vector<size_t> & locv, const std::vector<coords_t> & coord, const arma::mat & Pv, int lp, int mp) {

  idfuncs=idfuncsv;
  loc=locv;
  P=Pv;

  if(P.n_rows!=P.n_cols)
    throw std::runtime_error("Density matrix not square!\n");

  // Compute the coupling coefficients.
  compute_coefficients(clm,lp,mp);

  // Compute the distance table (must be done after Lmax has been set)
  distance_table(coord);
}

void EMDEvaluator::distance_table(const std::vector<coords_t> & coord) {
  Nat=coord.size();
  dist.resize((Nat*(Nat+1))/2);
  YLM.resize((Nat*(Nat+1))/2);

  for(size_t i=0;i<YLM.size();i++)
    YLM[i].resize(lmind(Lmax,Lmax)+1);

  for(size_t i=0;i<coord.size();i++)
    for(size_t j=0;j<=i;j++) {

      // Index in array
      size_t ind=i*(i+1)/2+j;

      if(i==j) {
	// Same atom.
	dist[ind]=0.0;

	// Initialize array
	for(int L=0;L<=Lmax;L++)
	  for(int M=-L;M<=L;M++)
	    YLM[ind][lmind(L,M)]=0.0;
	// However, Y_0^0 is 1/sqrt(4*pi)
	YLM[ind][lmind(0,0)]=1.0/sqrt(4.0*M_PI);
      } else {
	// Displacement vector
	coords_t dr_vec=coord[i]-coord[j];

	// Compute distance
	double dr=norm(dr_vec);
	dist[ind]=dr;

	// Phi and cos(theta)
	double phi, cth;
	if(dr>0) {
	  phi=atan2(dr_vec.y,dr_vec.x);
	  cth=dr_vec.z/dr;
	} else {
	  phi=-1;
	  cth=-1;
	}

	// Loop over L and M
	for(int L=0;L<=Lmax;L++)
	  for(int M=-L;M<=L;M++)
	    YLM[ind][lmind(L,M)]=std::conj(spherical_harmonics(L,M,cth,phi));
      }
    }
}

void EMDEvaluator::compute_coefficients(const std::vector< std::vector<ylmcoeff_t> > & clm, int lp, int mp) {
  if(clm.size()!=idfuncs.size())
    throw std::runtime_error("Sizes of clm and idfuncs do not match!\n");

  // Sanity check
  if(lp<0.0 || abs(mp)>lp) {
    std::ostringstream oss;
    oss << "Illegal values for spherical harmonics l = " << lp << ", m = " << mp << "!\n";
    throw std::runtime_error(oss.str());
  }

  if(lp%2!=0) {
    std::ostringstream oss;
    oss << "Projection for l = " << lp << ", m = " << mp << ", but all projections for odd l vanish!\n";
    throw std::runtime_error(oss.str());
  }    

  // Number of nonequivalent functions
  size_t N=clm.size();

  // Resize coupling coefficient array
  //  cc.resize(N*(N+1)/2);
  cc.resize(N*N);

  // Determine the maximum value of L we need
  int lmax=0;
  for(size_t i=0;i<clm.size();i++)
    for(size_t j=0;j<clm[i].size();j++)
      if(lmax<clm[i][j].l)
	lmax=clm[i][j].l;

  // We can thus couple up to
  Lmax=2*lmax+lp;

  // Compute Gaunt coefficient table.
  Gaunt gaunt(lmax+lp,Lmax,lmax+lp);

  // Form the coefficients. Loop over groups of equivalent functions.
  for(size_t iig=0;iig<clm.size();iig++) {
    if(!clm[iig].size())
      throw std::runtime_error("clm[iig] is empty!\n");

    for(size_t jjg=0;jjg<clm.size();jjg++) {

      if(!clm[jjg].size())
	throw std::runtime_error("clm[jjg] is empty!\n");

      // Loop over l, l' and m, m'
      for(size_t iil=0;iil<clm[iig].size();iil++)
	for(size_t jjl=0;jjl<clm[jjg].size();jjl++) {
	  // l and l' are
	  int il=clm[iig][iil].l;
	  int jl=clm[jjg][jjl].l;

	  // m and m' are
	  int im=clm[iig][iil].m;
	  int jm=clm[jjg][jjl].m;

	  // and the expansion coefficients are
	  std::complex<double> cmu=clm[iig][iil].c;
	  std::complex<double> cnu=clm[jjg][jjl].c;

	  // Loop over l1
	  int m1=jm+mp;
	  for(int l1=std::max(std::abs(jl-lp),std::abs(m1));l1<=jl+lp;l1++) {
	    // Gaunt coefficient
	    double g1=gaunt.coeff(l1,m1,jl,jm,lp,mp);

	    //	    printf("(%i % i) + (%i % i) -> (%i % i) % e g1\n",jl,jm,lp,mp,l1,m1,g1);
	    //	    fflush(stdout);
	    if(g1==0.0)
	      continue;

	    // Loop over L
	    for(int L=std::max(abs(il-l1),abs(im-m1));L<=il+l1;L++) {
	      // Coupling coefficient
	      coupl_coeff_t tmp;
	      
	      // Set l indices
	      tmp.l=il;
	      tmp.lp=jl;
	      
	      // Coupled values
	      tmp.L=L;
	      tmp.M=im-m1;
	      
	      // Compute coefficient
	      double g=gaunt.coeff(il,im,tmp.L,tmp.M,l1,m1);
	      tmp.c=std::pow(4.0*M_PI,3.0/2.0)*std::conj(cmu)*cnu*g1*pow(std::complex<double>(0.0,1.0),L)*g;
	      //	      printf("(%i % i) + (%i % i) -> (%i % i) % e g\n",tmp.L,tmp.M,l1,m1,il,im,g);
	      //	      fflush(stdout);

	      // Store coefficient
	      if(std::norm(tmp.c)>0.0) {
		add_coupling(iig,jjg,tmp);
	      }
	    }
	  }
	}
    }
  }

  // Clear coefficients with zero weight
  size_t nclean=0;
  for(size_t i=0;i<cc.size();i++) {
    for(size_t j=cc[i].size()-1;j<cc[i].size();j--)
      if(norm(cc[i][j].c)==0.0) {
	cc[i].erase(cc[i].begin()+j);
	nclean++;
      }
  }
  //  printf("%i terms cleaned.\n",(int) nclean);

}

EMDEvaluator::~EMDEvaluator() {
}

std::vector<radf_val_t> EMDEvaluator::get_radial(size_t ig, double p) const {
  std::vector<radf_val_t> ret;

  for(size_t j=0;j<rad[ig].size();j++) {
    radf_val_t hlp;
    hlp.l=rad[ig][j]->getl();
    hlp.f=rad[ig][j]->get(p);
    if(norm(hlp.f)>0.0)
      ret.push_back(hlp);
  }

  return ret;
}

void EMDEvaluator::add_coupling(size_t ig, size_t jg, coupl_coeff_t t) {
  // Index is
  size_t ijdx=ig*idfuncs.size()+jg;

#ifdef DEBUG
  if(ijdx>=cc.size()) {
    std::ostringstream oss;
    oss << "Error in add_coupling: requested element " << ijdx << " while size of cc is " << cc.size() <<"!\n";
    throw std::runtime_error(oss.str());
  }
#endif

  if(cc[ijdx].size()==0) {
    cc[ijdx].push_back(t);
  } else {
    // Get upper bound
    std::vector<coupl_coeff_t>::iterator high;
    high=std::upper_bound(cc[ijdx].begin(),cc[ijdx].end(),t);

    // Corresponding index is
    size_t ind=high-cc[ijdx].begin();

    if(ind>0 && cc[ijdx][ind-1]==t)
      // Found it.
      cc[ijdx][ind-1].c+=t.c;
    else {
      // Term does not exist, add it
      cc[ijdx].insert(high,t);
    }
  }
}

std::vector<total_coupl_t> EMDEvaluator::get_coupling(size_t ig, size_t jg, int l, int lp) const {
  // Find coupling coefficients with the wanted l and l'

  // The index in the cc list is
  size_t ijidx=ig*idfuncs.size()+jg;

  /*
  // Lower limit
  coupl_coeff_t lot;
  lot.l=l;
  lot.lp=lp;
  lot.L=abs(l-lp);

  // Upper limit
  coupl_coeff_t hit;
  hit.l=l;
  hit.lp=lp;
  hit.L=l+lp;

  // Find upper limit
  std::vector<coupl_coeff_t>::iterator high;
  high=std::upper_bound(cc[ijidx].begin(),cc[ijidx].end(),hit);

  // Find lower limit
  std::vector<coupl_coeff_t>::iterator low;
  low=std::upper_bound(cc[ijidx].begin(),cc[ijidx].end(),lot);

  // Corresponding indices are
  size_t hiind=high-cc[ijidx].begin();
  size_t loind=low-cc[ijidx].begin();

  // Collect results
  std::vector<total_coupl_t> ret;
  for(size_t i=loind;i<=hiind;i++)
  if((cc[ijidx][i].l==l) && (cc[ijidx][i].lp==lp)) {

  total_coupl_t hlp;
  hlp.L=cc[ijidx][i].L;
  hlp.c=cc[ijidx][i].c;
  ret.push_back(hlp);
  }
  */

  std::vector<total_coupl_t> ret;
  for(size_t i=0;i<cc[ijidx].size();i++)
    if((cc[ijidx][i].l==l) && (cc[ijidx][i].lp==lp)) {

      total_coupl_t hlp;
      hlp.L=cc[ijidx][i].L;
      hlp.M=cc[ijidx][i].M;
      hlp.c=cc[ijidx][i].c;
      ret.push_back(hlp);
    }

  return ret;
}

std::vector<total_coupl_t> EMDEvaluator::get_total_coupling(size_t ig, size_t jg, double p) const {
  // Get the radial parts
  std::vector<radf_val_t> ri=get_radial(ig,p);
  std::vector<radf_val_t> rj=get_radial(jg,p);

  // Returned array
  std::vector<total_coupl_t> ret;

  // Loop over factors
  for(size_t il=0;il<ri.size();il++) {
    // l is
    int l=ri[il].l;

    for(size_t ilp=0;ilp<rj.size();ilp++) {
      // l' is
      int lp=rj[ilp].l;

      // Get the coupling constants
      std::vector<total_coupl_t> c=get_coupling(ig,jg,l,lp);

      // Increment total value
      for(size_t ic=0;ic<c.size();ic++) {

	// Term to add
	total_coupl_t hlp;
	hlp.L=c[ic].L;
	hlp.M=c[ic].M;
	hlp.c=std::conj(ri[il].f)*rj[ilp].f*c[ic].c;

	// Add term
	add_coupling_term(ret,hlp);
      }
    }
  }

  // Clear coefficients with zero weight
  for(size_t i=ret.size()-1;i<ret.size();i--)
    if(ret[i].c==0.0)
      ret.erase(ret.begin()+i);

#ifdef DEBUG
  printf("Total coupling between functions %i and %i at p=%e\n",(int) ig,(int) jg,p);
  for(size_t i=0;i<ret.size();i++)
    printf("\t%i\t%i\t(% e,% e)\n",ret[i].L,ret[i].M,ret[i].c.real(),ret[i].c.imag());
#endif

  return ret;
}

void EMDEvaluator::print() const {
  printf("Radial parts\n");
  for(size_t i=0;i<rad.size();i++) {
    printf("Function %i / %i\n",(int) i+1, (int) rad.size());
    for(size_t j=0;j<rad[i].size();j++) {
      printf("%2i ",(int) j);
      rad[i][j]->print();
    }
  }
}

void EMDEvaluator::check_norm() const {
  // Get radial grid
  std::vector<radial_grid_t> grid=form_radial_grid(NRAD);

  for(size_t i=0;i<rad.size();i++) {
    for(size_t j=0;j<rad[i].size();j++) {
      // Calculate norm
      double norm=0.0;
      for(size_t ip=0;ip<grid.size();ip++)
	norm+=grid[ip].w*std::norm(rad[i][j]->get(grid[ip].r));
      norm=sqrt(norm);

#ifdef DEBUG
      printf("Function %i %i has norm %e, difference by % e.\n",(int) i+1, (int) j, norm, norm-1.0);
#else
      if(fabs(norm-1.0)>=NORMTOL) {
	printf("Function %i %i has norm %e, difference by % e.\n",(int) i+1, (int) j, norm, norm-1.0);
      }
#endif
    }
  }
  printf("Norms of the functions checked.\n");
}

std::complex<double> EMDEvaluator::get(double p) const {
  // Arguments of Bessel functions are
  std::vector<double> args(dist);
  for(size_t i=0;i<args.size();i++)
    args[i]*=p;
  // Evaluate Bessel functions
  arma::mat jl=bessel_array(args,Lmax);

  // Continue by computing the radial EMD
  std::complex<double> np=0.0;

  // List of off-diagonal elements
  std::vector<noneqradf_t> offd;
  for(size_t iig=0;iig<idfuncs.size();iig++)
    for(size_t jjg=0;jjg<iig;jjg++) {
      noneqradf_t hlp;
      hlp.i=iig;
      hlp.j=jjg;
      offd.push_back(hlp);
    }

  double npre=0.0, npim=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:npre,npim)
#endif
  // Loop over groups of equivalent functions
  for(size_t iii=0;iii<offd.size();iii++) {
    size_t iig=offd[iii].i;
    size_t jjg=offd[iii].j;

    // Get the total coupling coefficient
    std::vector<total_coupl_t> totc=get_total_coupling(iig,jjg,p);
    if(totc.size()==0)
      continue;

    // Loop over the individual functions
    for(size_t ii=0;ii<idfuncs[iig].size();ii++)
      for(size_t jj=0;jj<idfuncs[jjg].size();jj++) {
	// The indices are
	size_t mu=idfuncs[iig][ii];
	size_t nu=idfuncs[jjg][jj];

	// and the functions are centered on
	size_t iat=loc[mu];
	size_t jat=loc[nu];

	// so the corresponding index in the Bessel and spherical harmonics arrays is
	size_t ibes;
	// Sign of spherical harmonics
	int ylmsign=1;

	//	  ibes=iat*Nat+jat;
	if(iat>jat)
	  ibes=iat*(iat+1)/2+jat;
	else {
	  // Reverse sign of Ylm
	  ibes=jat*(jat+1)/2+iat;
	  ylmsign=-1;
	}

	// Loop over coupling coefficient
	for(size_t ic=0;ic<totc.size();ic++) {
	  // L and M are
	  int L=totc[ic].L;
	  int M=totc[ic].M;

	  // Increment EMD; we get the increment twice since we are off-diagonal.
	  std::complex<double> incr=2.0*P(mu,nu)*totc[ic].c*YLM[ibes][lmind(L,M)]*pow(ylmsign,L)*jl(ibes,L);
	  npre+=incr.real();
	  npim+=incr.imag();
	}
      }
  }
  np+=std::complex<double>(npre,npim);

  npre=0.0; npim=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:npre,npim)
#endif
  // Then, do diagonal. Get the total coupling coefficient
  for(size_t iig=0;iig<idfuncs.size();iig++) {
    std::vector<total_coupl_t> totc=get_total_coupling(iig,iig,p);
    if(totc.size()==0)
      continue;

    // Loop over the individual functions
    for(size_t ii=0;ii<idfuncs[iig].size();ii++)
      for(size_t jj=0;jj<idfuncs[iig].size();jj++) {

	// The indices are
	size_t mu=idfuncs[iig][ii];
	size_t nu=idfuncs[iig][jj];

	// and the functions are centered on
	size_t iat=loc[mu];
	size_t jat=loc[nu];

	// so the corresponding index in the Bessel and spherical harmonics arrays is
	size_t ibes;
	// Sign of spherical harmonics
	int ylmsign=1;

	//	  ibes=iat*Nat+jat;
	if(iat>jat)
	  ibes=iat*(iat+1)/2+jat;
	else {
	  // Reverse sign of Ylm
	  ibes=jat*(jat+1)/2+iat;
	  ylmsign=-1;
	}

	// Loop over coupling coefficient
	for(size_t ic=0;ic<totc.size();ic++) {
	  // L and M are
	  int L=totc[ic].L;
	  int M=totc[ic].M;

	  // Increment EMD
	  std::complex<double> incr=P(mu,nu)*totc[ic].c*YLM[ibes][lmind(L,M)]*pow(ylmsign,L)*jl(ibes,L);
	  npre+=incr.real();
	  npim+=incr.imag();
	}
      }
  }
  np+=std::complex<double>(npre,npim);

  return np;
}


arma::mat bessel_array(const std::vector<double> & args, int lmax) {
  // Returned array.
  arma::mat j(args.size(),lmax+1);
  j.zeros();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t i=0;i<args.size();i++) {
      for(int l=0;l<=lmax;l++)
	j(i,l)=bessel_jl(l,args[i]);
    }
  }

#ifdef DEBUG
  j.print("Bessel array");
#endif

  return j;
}


EMD::EMD(const EMDEvaluator * posevalp, const EMDEvaluator * negevalp, int Nelv, int lv, int mv) {
  Nel=Nelv;
  poseval=posevalp;
  l=lv;
  m=mv;

  if(m>0) {
    negeval=negevalp;
    negcoef=M_SQRT1_2;
    poscoef=std::pow(-1.0,m)*M_SQRT1_2;
  } else if(m==0) {
    negeval=NULL;
    negcoef=0.0;
    poscoef=1.0;
  } else {
    negeval=negevalp;
    negcoef=std::complex<double>(0.0,M_SQRT1_2);
    poscoef=-std::pow(-1.0,m)*std::complex<double>(0.0,M_SQRT1_2);
  }
}

EMD::~EMD() {
}

double EMD::eval(double p) const {
  // Result
  std::complex<double> r;

  // Positive part
  if(negcoef!=0.0)
    r = poscoef*poseval->get(p) + negcoef*negeval->get(p);
  else
    r = poscoef*poseval->get(p);

  // Return real part
  return r.real();
}

void EMD::initial_fill(bool verbose) {
  if(verbose) {
    printf("\nFilling in initial grid ... ");
    fflush(stdout);
  }

  // Fill a grid with initial spacing 0.1 for q = 0 .. 1.0.
  // Multiply upper limit and spacing by factor 2 every interval.
  fixed_fill(false,0.01,1.0,2.0,2.0);
  
  if(verbose)
    printf("done.\n");
}

void EMD::add4(size_t loc) {
  // Add 4 points around loc
  emd_t integ[4];

  //  printf("Adding more points at %lu.\n",loc);

#ifdef _OPENMP
#pragma omp parallel for ordered
#endif
  for(int ipoint=2;ipoint>-2;ipoint--) {
    // Value of p is
    integ[2-ipoint].p=0.5*(dens[loc+ipoint].p+dens[loc+ipoint-1].p);
    // Value of the density at this point is
    integ[2-ipoint].d=eval(integ[2-ipoint].p);
    // Add value to the list
#ifdef _OPENMP
#pragma omp ordered
#endif
    dens.insert(dens.begin()+loc+ipoint,integ[2-ipoint]);
  }
}

void EMD::find_electrons(bool verbose, double tol) {
  // Integral and its estimated error
  double integral, error;
  // Integral slices
  double rough=0.0, fine=0.0;
  // Location of maximum error
  double maxerror=0;
  size_t maxind=0;

  if(verbose) {
    printf("Continuing fill of grid to find electrons ... ");
    fflush(stdout);
  }

  size_t iter=0;

  // Tighten grid adaptively
  while(true) {
    iter++;
    integral=0;
    error=0;

    maxerror=0;
    // Calculate integral and find maximum error
    for(size_t i=dens.size()-3;i<dens.size();i-=4) {
      // The rough value is (3-point Simpson)
      rough=roughdens(i);
      // The fine value is (5-point Simpson)
      fine=finedens(i);

      // Increment total integrals
      integral+=fine;
      error+=fabs(fine-rough)/15.0;

      // Is the maximum error here?
      if(fabs(rough-fine)>maxerror) {
        maxerror=fabs(rough-fine);
        maxind=i;
      }
    }

    if(fabs(Nel-integral)/Nel>tol) {
      // Check that the calculation will actually converge at some point..
      if(dens.size()>MAXPOINTS) {
	ERROR_INFO();
	std::ostringstream oss;
	oss << "Error in find_electrons: maximum allowed number of points reached. int=" << integral << ", Nel=" << Nel <<".\n";
	throw std::runtime_error(oss.str());
      }

      // Add points to area of maximum error
      if(maxind==0) {
	ERROR_INFO();
	save("emddump.txt");
	throw std::runtime_error("Unable to find location of maximum error!\n");
      }

      add4(maxind);
    } else
      // Converged
      break;
  }

  if(verbose)
    printf("done.\n");
}

void EMD::optimize_moments(bool verbose, double tol) {
  // Integrals
  double rough, fine;

  // Moments of density
  int mom;
  int Nmom=7;
  int moms[]={-2, -1, 0, 1, 2, 3, 4};
  double momval[Nmom];
  double momerr[Nmom];
  size_t mommaxerrloc[Nmom];

  // Temporary helper
  double maxerr;

  // Maximum relative error
  double errel;
  int errelind;

  // Timer for printouts
  Timer t;
  size_t iter=0;

  if(verbose)
    printf("Optimizing the moments of the EMD.\n");

  do {
    iter++;

    // Calculate moments and error estimates
    for(int imom=0;imom<Nmom;imom++) {
      mom=moms[imom];
      momval[imom]=0.0;
      momerr[imom]=0.0;

      maxerr=0;
      mommaxerrloc[imom]=-1;

      // Calculate <p^k>
      for(size_t i=dens.size()-3;i<dens.size();i-=4) {
	rough=roughmom(i);
	fine=finemom(i);

	momval[imom]+=fine;
	momerr[imom]+=fabs(fine-rough)/15.0;

	// Determine location of maximum error for each value of k
	if(fabs(rough-fine)>maxerr) {
	  mommaxerrloc[imom]=i;
	  maxerr=fabs(rough-fine);
	}
      }
    }

    // Find out which moment has maximum error and where it is
    errel=0;
    errelind=-1;
    for(int imom=0;imom<Nmom;imom++)
      // Moments can be negative for l>0
      if(fabs(momerr[imom]/momval[imom])>errel) {
        errel=fabs(momerr[imom]/momval[imom]);
        errelind=imom;
      }
    
    // Print out current values if necessary
    if(verbose && (iter==1 || t.get()>MAXPRINTFREQ || errel<=tol)) {
      t.set();
      if(l==0 && m==0)
	printf("\nUsing %u points, charge differs from Nel by %e.\n",(unsigned int) dens.size(),momval[2]-Nel);
      else
	printf("\nUsing %u points.\n",(unsigned int) dens.size());
      printf("Current values of moments are:\n");
      printf("\t%2s\t%13s\t%12s\t%12s\n","k","<p^k>","Abs error","Rel error");
      for(int imom=0;imom<Nmom;imom++)
        printf("\t% i\t% e\t%e\t%e\n",moms[imom],momval[imom],momerr[imom],fabs(momerr[imom]/momval[imom]));
    }
    
    // If tolerance has not been reached, add more points
    if(errel>tol)
      add4(mommaxerrloc[errelind]);
    
  } while(errel>tol);
  
  if(verbose) {
      t.set();
      if(l==0 && m==0)
	printf("\nUsed %u points, charge differs from Nel by %e.\n",(unsigned int) dens.size(),momval[2]-Nel);
      else
	printf("\nUsed %u points.\n",(unsigned int) dens.size());
      printf("Final values of moments are:\n");
      printf("\t%2s\t%13s\t%12s\t%12s\n","k","<p^k>","Abs error","Rel error");
      for(int imom=0;imom<Nmom;imom++)
        printf("\t% i\t% e\t%e\t%e\n",moms[imom],momval[imom],momerr[imom],fabs(momerr[imom]/momval[imom]));
  }
}

void EMD::fixed_fill(bool verbose, double h0, double len0, double hfac, double lfac) {
  Timer t;
  if(verbose) {
    printf("Filling the EMD grid ... ");
    fflush(stdout);
  }

  // Add the origin
  dens.resize(1);
  dens[0].p=0.0;
  dens[0].d=eval(0.0);

  // Loop over intervals
  double pmin=0.0;

  double len(len0);
  double h(h0);

  while(true) {
    // Compute the length of the interval
    size_t Nint=(size_t) round((len-pmin)/(4*h));

    // Allocate memory
    size_t i0=dens.size();
    dens.resize(dens.size()+4*Nint);

    // Loop over intervals
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(size_t i=0;i<Nint;i++)
      for(int j=0;j<4;j++) {
	dens[i0+4*i+j].p = pmin + 4*i*h + (j+1)*h;
	dens[i0+4*i+j].d = eval(dens[i0+4*i+j].p);
      }

    pmin+=Nint*4*h;
    h*=hfac;
    len*=lfac;

    // Check if density has vanished
    if(dens[dens.size()-1].d==0.0 && dens[dens.size()-2].d==0.0)
      break;
  }

  if(verbose) {
    printf("done (%s)\n",t.elapsed().c_str());
    printf("Grid filled up to p = %e.\n",dens[dens.size()-1].p);
    fflush(stdout);
  }
}

std::vector<emd_t> EMD::get() const {
  return dens;
}

void EMD::save(const char * fname) const {
  FILE *out=fopen(fname,"w");
  for(size_t i=0;i<dens.size();i++)
    fprintf(out,"%.15e\t%.15e\n",dens[i].p,dens[i].d);
  fclose(out);
}

arma::mat EMD::moments() const {
  // Three and five point Simpson
  double rough, fine;

  // Moments calculated by Hart & Thakkar
  int Nm=7;
  int momarr[]={-2, -1, 0, 1, 2, 3, 4};

  // Moments and errors
  arma::mat moms(Nm,3);

  // Helper variables
  const size_t N=dens.size();
  double p[N], integrand[N];

  // Fill in momentum grid
  for(size_t i=0;i<N;i++)
    p[i]=dens[i].p;

  // Calculate the moments
  for(int mi=0;mi<Nm;mi++) {
    // Fill in helper grid
    for(size_t i=0;i<N;i++)
      integrand[i]=pow(p[i],2+momarr[mi])*dens[i].d;

    // Zero out old values
    moms(mi,0)=momarr[mi];
    moms(mi,1)=0;
    moms(mi,2)=0;

    // Calculate moments & errors
    rough=0;
    fine=0;
    for(size_t i=N-3;i<N;i-=4) {
      // Sum from infinity to get more accurate integrals
      rough=(integrand[i-2]+4.0*integrand[i]+integrand[i+2])/6.0*(p[i+2]-p[i-2]);
      fine=(integrand[i-2]+4.0*integrand[i-1]+2.0*integrand[i]+4.0*integrand[i+1]+integrand[i+2])/12.0*(p[i+2]-p[i-2]);

      moms(mi,1)+=fine;
      moms(mi,2)+=fabs(fine-rough)/15.0;
    }
  }

  return moms;
}

void EMD::moments(const char * fname) const {
  arma::mat moms=moments();
  // Print out the moments
  FILE *out=fopen(fname,"w");
  for(size_t mi=0;mi<moms.n_rows;mi++)
    fprintf(out,"\t% i\t%.12e\t%.12e\n",(int) moms(mi,0),moms(mi,1),moms(mi,2));
  fclose(out);
}

arma::mat EMD::compton_profile() const {
  double rough, fine, Jint, Jerr;
  double integrand[dens.size()];

  size_t N=(dens.size()-1)/4;
  arma::mat J(N,3);

  // Calculate integrand
  for(size_t i=0;i<dens.size();i++)
    integrand[i]=dens[i].p*dens[i].d;

  // Calculate the Compton profile
  rough=0.0;
  fine=0.0;
  Jint=0.0;
  Jerr=0.0;
  size_t n=N-1;

  // Calculate integral
  for(size_t i=dens.size()-3;i<dens.size();i-=4) {
    rough=(integrand[i-2]+4.0*integrand[i]+integrand[i+2])/6.0*(dens[i+2].p-dens[i-2].p);
    fine=(integrand[i-2]+4.0*integrand[i-1]+2.0*integrand[i]+4.0*integrand[i+1]+integrand[i+2])/12.0*(dens[i+2].p-dens[i-2].p);

    Jint+=fine;
    Jerr+=fabs(fine-rough)/15.0;

    // Save profile
    J(n,0)=dens[i-2].p; // Must be i-1 to get J(0) = 1st moment of density / 2
    J(n,1)=0.5*Jint; // J = 1/2 \int_{|q|}^\infty
    J(n,2)=0.5*Jerr;
    n--;
  }

  return J;
}

void EMD::compton_profile(const char * fname_raw) const {
  arma::mat J=compton_profile();

  // Print out profile
  FILE *out=fopen(fname_raw,"w");
  for(size_t n=0;n<J.n_rows;n++)
    fprintf(out,"%.12e\t%.12e\t%.12e\n",J(n,0),J(n,1),J(n,2));
  fclose(out);
}

void EMD::compton_profile_interp(const char * fname_interp) const {
  // Get the profile.
  arma::mat Jv=compton_profile();

  // Momentum transfer, profile and its error
  std::vector<double> p, J, dJ;
  for(size_t i=0;i<Jv.n_rows;i++) {
    p.push_back(Jv(i,0));
    J.push_back(Jv(i,1));
    dJ.push_back(Jv(i,2));
  }

  // Interpolate profile to p = 0 .. pmax with spacing dp
  const int Nreg=2;
  double pmax[Nreg]={10.0, 40.0};
  double dp[Nreg]={0.01, 0.5};
  int Npoints[Nreg];

  Npoints[0]=(int) round(pmax[0]/dp[0]);
  for(int i=1;i<Nreg;i++)
    Npoints[i]=(int) round((pmax[i]-pmax[i-1])/dp[i]);
  // Add one final point for "safety"
  Npoints[Nreg-1]++;

  // Interpolated p values.
  std::vector<double> p_interp;
  for(int i=0;i<Nreg;i++)
    for(int j=0;j<Npoints[i];j++) {
      if(i>0)
	p_interp.push_back(pmax[i-1]+j*dp[i]);
      else
	p_interp.push_back(j*dp[i]);
    }

  // Get interpolated Compton profile and its error
  std::vector<double> J_interp=spline_interpolation(p,J,p_interp);
  std::vector<double> dJ_interp=spline_interpolation(p,dJ,p_interp);

  // Save output
  FILE *out=fopen(fname_interp,"w");
  for(size_t i=0;i<p_interp.size();i++)
    fprintf(out,"%.12e\t%.12e\t%.12e\n",p_interp[i],J_interp[i],dJ_interp[i]);
  fclose(out);
}
