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

#include "density_fitting.h"
#include "linalg.h"
#include "scf.h"
#include "stringutil.h"
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

// \delta parameter in Eichkorn et al
#define DELTA 1e-9

DensityFit::DensityFit() {
  omega=0.0;
  alpha=1.0;
  beta=0.0;
}

DensityFit::~DensityFit() {
}

void DensityFit::set_range_separation(double w, double a, double b) {
  omega=w;
  alpha=a;
  beta=b;
}

void DensityFit::get_range_separation(double & w, double & a, double & b) const {
  w=omega;
  a=alpha;
  b=beta;
}

size_t DensityFit::fill(const BasisSet & orbbas, const BasisSet & auxbas, bool dir, double erithr, double linthr, double cholthr) {
  // Construct density fitting basis

  // Store amount of functions
  Nbf=orbbas.get_Nbf();
  Naux=auxbas.get_Nbf();
  Nnuc=orbbas.get_Nnuc();
  direct=dir;

  // Fill list of shell pairs
  arma::mat Q, M;
  orbpairs=orbbas.get_eripairs(Q,M,erithr);

  // Get orbital shells, auxiliary shells and dummy shell
  orbshells=orbbas.get_shells();
  auxshells=auxbas.get_shells();
  dummy=dummyshell();

  maxorbam=orbbas.get_max_am();
  maxauxam=auxbas.get_max_am();
  maxorbcontr=orbbas.get_max_Ncontr();
  maxauxcontr=auxbas.get_max_Ncontr();
  maxam=std::max(orbbas.get_max_am(),auxbas.get_max_am());
  maxcontr=std::max(orbbas.get_max_Ncontr(),auxbas.get_max_Ncontr());

  // First, compute the two-center integrals
  ab.zeros(Naux,Naux);

  // Get list of unique auxiliary shell pairs
  std::vector<shellpair_t> auxpairs=auxbas.get_unique_shellpairs();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker *eri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(maxam,maxcontr);
    else
      eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<auxpairs.size();ip++) {
      // The shells in question are
      size_t is=auxpairs[ip].is;
      size_t js=auxpairs[ip].js;

      // Compute (a|b)
      eri->compute(&auxshells[is],&dummy,&auxshells[js],&dummy);
      erip=eri->getp();

      // Store integrals
      size_t Ni=auxshells[is].get_Nbf();
      size_t Nj=auxshells[js].get_Nbf();
      for(size_t ii=0;ii<Ni;ii++) {
	size_t ai=auxshells[is].get_first_ind()+ii;
	for(size_t jj=0;jj<Nj;jj++) {
	  size_t aj=auxshells[js].get_first_ind()+jj;

	  ab(ai,aj)=(*erip)[ii*Nj+jj];
	  ab(aj,ai)=(*erip)[ii*Nj+jj];
	}
      }
    }

    delete eri;
  }

  ab_invh = PartialCholeskyOrth(ab, cholthr, linthr);
  ab_inv = ab_invh * ab_invh.t();

  // Then, compute the three-center integrals
  if(!direct) {
    // Compute the three-center integrals
    arma::mat ints;
    three_center_integrals(ints);
    // Transform into proper B matrix
    B=ints*ab_invh;
  }

  return orbpairs.size();
}

double DensityFit::fitting_error() const {
  arma::mat error_matrix(maxorbam+1, maxorbam+1, arma::fill::zeros);

  // Loop over pairs
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    arma::mat wrk_error(error_matrix);

    ERIWorker *eri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(maxam,maxcontr);
    else
      eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      // Shells in question are
      size_t imus=orbpairs[ip].is;
      size_t inus=orbpairs[ip].js;
      // Amount of functions
      size_t Nmu=orbshells[imus].get_Nbf();
      size_t Nnu=orbshells[inus].get_Nbf();

      // Compute the (A|uv) integrals
      arma::mat auv(compute_a_munu(eri,ip));

      // This gives the density fitted (uv|uv) integrals as
      arma::mat dfit_uvuv(arma::trans(auv) * ab_inv * auv);

      // The correct integrals are, however
      eri->compute(&orbshells[inus],&orbshells[imus],&orbshells[inus],&orbshells[imus]);
      const std::vector<double> * erip(eri->getp());

      double shell_error=0.0;
      for(size_t mu=0;mu<Nmu;mu++)
        for(size_t nu=0;nu<Nnu;nu++) {
          size_t imunu = nu*Nmu+mu;
          size_t ieri = imunu*(Nmu*Nnu) + imunu;
          double delta= (*erip)[ieri] - dfit_uvuv(imunu, imunu);
          //printf("(%c%c|%c%c): (%i %i|%i %i) = %e (fit) vs %e (exact), error %e\n", shell_types[orbshells[inus].get_am()], shell_types[orbshells[imus].get_am()], shell_types[orbshells[inus].get_am()], shell_types[orbshells[imus].get_am()], (int) (nu0+nu),(int) (mu0+mu),(int) (nu0+nu),(int) (mu0+mu),dfit_uvuv(imunu, imunu),(*erip)[ieri],delta);
          shell_error += delta;
        }

      wrk_error(orbshells[imus].get_am(), orbshells[inus].get_am()) += shell_error;
      if(imus != inus)
        wrk_error(orbshells[inus].get_am(), orbshells[imus].get_am()) += shell_error;
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    error_matrix += wrk_error;
  }

  printf("\n");
  for(int iam=0;iam<=maxorbam;iam++)
    for(int jam=0;jam<=iam;jam++)
      printf("Total (%c%c|%c%c) error %e\n",shell_types[jam],shell_types[iam],shell_types[jam],shell_types[iam],error_matrix(jam,iam));

  double total_error = arma::sum(arma::sum(error_matrix));
  printf("Total error is %.15e\n",total_error);

  return total_error;
}

arma::mat DensityFit::compute_a_munu(ERIWorker *eri, size_t ip) const {
  // Shells in question are
  size_t imus=orbpairs[ip].is;
  size_t inus=orbpairs[ip].js;
  // Amount of functions
  size_t Nmu=orbshells[imus].get_Nbf();
  size_t Nnu=orbshells[inus].get_Nbf();

  // Allocate storage
  arma::mat amunu;
  amunu.zeros(Naux,Nmu*Nnu);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(size_t ia=0;ia<auxshells.size();ia++) {
    // Number of functions on shell
    size_t Na=auxshells[ia].get_Nbf();
    size_t a0=auxshells[ia].get_first_ind();

    // Compute (a|vu)
    eri->compute(&auxshells[ia],&dummy,&orbshells[inus],&orbshells[imus]);
    const std::vector<double> * erip(eri->getp());

    // Store integrals
    for(size_t a=0;a<Na;a++)
      for(size_t imunu=0;imunu<Nmu*Nnu;imunu++)
	// Use Fortran ordering so it's compatible with Armadillo
	//amunu(a0+a,nu*Nmu+mu)=(*erip)[(a*Nnu+nu)*Nmu+mu];
	amunu(a0+a,imunu)=(*erip)[a*Nnu*Nmu+imunu];
  }

  return amunu;
}

void DensityFit::digest_Jexp(const arma::mat & P, size_t ip, const arma::mat & amunu, arma::vec & gamma) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

  // Shells in question are
  size_t imus=orbpairs[ip].is;
  size_t inus=orbpairs[ip].js;
  // First function on shell
  size_t mubeg=orbshells[imus].get_first_ind();
  size_t nubeg=orbshells[inus].get_first_ind();
  // Amount of functions
  size_t muend=orbshells[imus].get_last_ind();
  size_t nuend=orbshells[inus].get_last_ind();

  // Density submatrix
  arma::vec Psub;
  if(imus != inus)
    // On the off-diagonal we're twice degenerate
    Psub=2.0*arma::vectorise(P.submat(mubeg,nubeg,muend,nuend));
  else
    Psub=arma::vectorise(P.submat(mubeg,nubeg,muend,nuend));

  // Contract over indices
  gamma+=amunu*Psub;
}

void DensityFit::digest_J(const arma::mat & gamma, size_t ip, const arma::mat & amunu, arma::mat & J) const {
  // Shells in question are
  size_t imus=orbpairs[ip].is;
  size_t inus=orbpairs[ip].js;
  // First function on shell
  size_t mu0=orbshells[imus].get_first_ind();
  size_t nu0=orbshells[inus].get_first_ind();
  // Amount of functions
  size_t Nmu=orbshells[imus].get_Nbf();
  size_t Nnu=orbshells[inus].get_Nbf();

  // vec(uv) = c_a (a,uv)
  arma::mat Jsub(arma::trans(gamma)*amunu);
  // Reshape into matrix
  Jsub.reshape(Nmu,Nnu);

  J.submat(mu0,nu0,mu0+Nmu-1,nu0+Nnu-1)=Jsub;
  J.submat(nu0,mu0,nu0+Nnu-1,mu0+Nmu-1)=arma::trans(Jsub);
}

void DensityFit::digest_K_incore(const arma::mat & C, const arma::vec & occs, arma::mat & K) const {
  if(direct)
    throw std::logic_error("Code can only be run in non-direct mode!\n");

  if(C.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", C.n_rows = " << C.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  size_t Na = B.n_cols;

  // Reshape B
  const arma::mat Breshape((double *) B.memptr(), Nbf, Nbf*Na, false, true);
  arma::mat Awork(Nbf,Na);
  
  for(size_t i=0;i<C.n_cols;i++) {
    // A(s,Q) = B(r;sQ) C(r;i)
    Awork = arma::reshape(arma::trans(Breshape)*C.col(i),Nbf,Na);
    K+=occs[i]*Awork*arma::trans(Awork);
  }
}

void DensityFit::digest_K_incore(const arma::cx_mat & C, const arma::vec & occs, arma::cx_mat & K) const {
  if(direct)
    throw std::logic_error("Code can only be run in non-direct mode!\n");

 if(C.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", C.n_rows = " << C.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // Reshape B
  const arma::mat Breshape((double *) B.memptr(), Nbf, Nbf*Naux, false, true);
  arma::cx_mat Awork(Nbf,Naux);
  
  for(size_t i=0;i<C.n_cols;i++) {
    // A(s,Q) = B(r;sQ) C(r;i)
    Awork = arma::reshape(arma::trans(Breshape)*C.col(i),Nbf,Naux);
    K+=occs[i]*arma::conj(Awork)*arma::strans(Awork);
  }
}

void DensityFit::digest_K_direct(const arma::mat & C, const arma::vec & occs, arma::mat & K) const {
  throw std::logic_error("Direct K digestion not implemented\n");
}

size_t DensityFit::memory_estimate(const BasisSet & orbbas, const BasisSet & auxbas, double thr, bool dir) const {
  // Amount of auxiliary functions (for representing the electron density)
  size_t Nbf=orbbas.get_Nbf();
  // Amount of auxiliary functions (for representing the electron density)
  size_t Na=auxbas.get_Nbf();
  // Amount of memory required for calculation
  size_t Nmem=0;

  // Memory taken up by  ( \alpha | \mu \nu)
  if(!dir) {
    // Count number of function pairs
    Nmem+=Na*Nbf*Nbf*sizeof(double);
  }

  // Memory taken by (\alpha | \beta) and its inverse
  Nmem+=2*Na*Na*sizeof(double);
  // We also have (a|b)^(-1/2)
  Nmem+=Na*Na*sizeof(double);

  // Memory taken by gamma and expansion coefficients
  Nmem+=2*Na*sizeof(double);

  return Nmem;
}

arma::vec DensityFit::compute_expansion(const arma::mat & P) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

  // Compute gamma
  if(!direct) {
    // c(P) = B(st;P) P(st)
    return arma::trans(B)*arma::vectorise(P);
  }

  arma::vec gamma(Naux);
  gamma.zeros();

#ifdef _OPENMP
#pragma omp parallel
    {
      ERIWorker *eri;
      if(omega==0.0 && alpha==1.0 && beta==0.0)
	eri=new ERIWorker(maxam,maxcontr);
      else
	eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

      arma::vec gv(gamma);
#pragma omp for schedule(dynamic)
      for(size_t ip=0;ip<orbpairs.size();ip++)
	digest_Jexp(P,ip,compute_a_munu(eri,ip),gv);
#pragma omp critical
      gamma+=gv;

      delete eri;
    }
#else
    // Sequential code

    ERIWorker *eri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(maxam,maxcontr);
    else
      eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

    for(size_t ip=0;ip<orbpairs.size();ip++)
      digest_Jexp(P,ip,compute_a_munu(eri,ip),gamma);

    delete eri;
#endif

  // Compute and return c
  return ab_inv*gamma;
}

std::vector<arma::vec> DensityFit::compute_expansion(const std::vector<arma::mat> & P) const {
  for(size_t i=0;i<P.size();i++) {
    if(P[i].n_rows != Nbf || P[i].n_cols != Nbf) {
      std::ostringstream oss;
      oss << "Error in DensityFit: Nbf = " << Nbf << ", P[" << i << "].n_rows = " << P[i].n_rows << ", P[" << i << "].n_cols = " << P[i].n_cols << "!\n";
      throw std::logic_error(oss.str());
    }
  }

  std::vector<arma::vec> gamma(P.size());
  for(size_t i=0;i<P.size();i++)
    gamma[i].zeros(Naux);

  // Compute gamma
  if(!direct) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(size_t iden=0;iden<P.size();iden++)
      // c(P) = B(st;P) P(st)
      gamma[iden] = arma::trans(B)*arma::vectorise(P[iden]);
    
  } else {
    for(size_t iden=0;iden<P.size();iden++) {
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
	ERIWorker *eri;
	if(omega==0.0 && alpha==1.0 && beta==0.0)
	  eri=new ERIWorker(maxam,maxcontr);
	else
	  eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

#ifdef _OPENMP
	// Worker stack for each matrix
	arma::vec gv(gamma[iden]);
#pragma omp for schedule(dynamic)
#endif
	for(size_t ip=0;ip<orbpairs.size();ip++) {
#ifdef _OPENMP
	  digest_Jexp(P[iden],ip,compute_a_munu(eri,ip),gv);
#else
	  digest_Jexp(P[iden],ip,compute_a_munu(eri,ip),gamma[iden]);
#endif
	}
#ifdef _OPENMP
#pragma omp critical
	gamma[iden]+=gv;
#endif

	delete eri;
      }
    }
    // Compute and return c
    for(size_t ig=0;ig<P.size();ig++)
      gamma[ig]=ab_inv*gamma[ig];
  }

  return gamma;
}

arma::mat DensityFit::calcJ(const arma::mat & P) const {
  if(P.n_rows != Nbf || P.n_cols != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", P.n_rows = " << P.n_rows << ", P.n_cols = " << P.n_cols << "!\n";
    throw std::logic_error(oss.str());
  }

  // Get the expansion coefficients
  arma::vec c=compute_expansion(P);
  return calcJ_vector(c);
}

arma::mat DensityFit::calcJ_vector(const arma::vec & c) const {
  arma::mat J(Nbf,Nbf);
  J.zeros();

  if(!direct) {
    if(c.n_elem != B.n_cols) {
      std::ostringstream oss;
      oss << "Error in DensityFit: B.n_cols = " << B.n_cols << ", c.n_elem = " << c.n_elem << "!\n";
      throw std::logic_error(oss.str());
    }
    J=arma::reshape(B*c,Nbf,Nbf);

  } else {
    if(c.n_elem != Naux) {
      std::ostringstream oss;
      oss << "Error in DensityFit: Naux = " << Naux << ", c.n_elem = " << c.n_elem << "!\n";
      throw std::logic_error(oss.str());
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      ERIWorker *eri;
      if(omega==0.0 && alpha==1.0 && beta==0.0)
	eri=new ERIWorker(maxam,maxcontr);
      else
	eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++)
	digest_J(c,ip,compute_a_munu(eri,ip),J);

      delete eri;
    }
  }

  return J;
}

std::vector<arma::mat> DensityFit::calcJ(const std::vector<arma::mat> & P) const {
  // Get the expansion coefficients
  std::vector<arma::vec> c=compute_expansion(P);

  std::vector<arma::mat> J(P.size());
  for(size_t iden=0;iden<P.size();iden++)
    J[iden].zeros(Nbf,Nbf);

  if(!direct) {
    for(size_t iden=0;iden<P.size();iden++) {
      J[iden]=arma::reshape(B*c[iden],Nbf,Nbf);
    }
    
  } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      ERIWorker *eri;
      if(omega==0.0 && alpha==1.0 && beta==0.0)
	eri=new ERIWorker(maxam,maxcontr);
      else
	eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for(size_t ip=0;ip<orbpairs.size();ip++) {
	arma::mat amunu(compute_a_munu(eri,ip));
	for(size_t iden=0;iden<P.size();iden++)
	  digest_J(c[iden],ip,amunu,J[iden]);
      }

      delete eri;
    }
  }

  return J;
}

arma::vec DensityFit::forceJ(const arma::mat & P) {
  // First, compute the expansion
  arma::vec c=compute_expansion(P);

  // The force
  arma::vec f(3*Nnuc);
  f.zeros();

  // First part: f = *#* 1/2 c_a (a|b)' c_b *#* - gamma_a' c_a
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    dERIWorker *deri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      deri=new dERIWorker(maxam,maxcontr);
    else
      deri=new dERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
    // Worker stack for each matrix
    arma::vec fwrk(f);
    fwrk.zeros();

#pragma omp for schedule(dynamic)
#endif
    for(size_t ias=0;ias<auxshells.size();ias++)
      for(size_t jas=0;jas<=ias;jas++) {

	// Symmetry factor
	double fac=0.5;
	if(ias!=jas)
	  fac*=2.0;

	size_t Na=auxshells[ias].get_Nbf();
	size_t anuc=auxshells[ias].get_center_ind();
	size_t Nb=auxshells[jas].get_Nbf();
	size_t bnuc=auxshells[jas].get_center_ind();

	if(anuc==bnuc)
	  // Contributions vanish
	  continue;

	// Compute (a|b)
	deri->compute(&auxshells[ias],&dummy,&auxshells[jas],&dummy);

	// Compute forces
	const static int index[]={0, 1, 2, 6, 7, 8};
	const size_t Nidx=sizeof(index)/sizeof(index[0]);
	double ders[Nidx];

	for(size_t iid=0;iid<Nidx;iid++) {
	  ders[iid]=0.0;
	  // Index is
	  int ic=index[iid];

	  // Increment force, anuc
	  erip=deri->getp(ic);
	  for(size_t iia=0;iia<Na;iia++) {
	    size_t ia=auxshells[ias].get_first_ind()+iia;

	    for(size_t iib=0;iib<Nb;iib++) {
	      size_t ib=auxshells[jas].get_first_ind()+iib;

	      // The integral is
	      double res=(*erip)[iia*Nb+iib];

	      ders[iid]+= res*c(ia)*c(ib);
	    }
	  }
	  ders[iid]*=fac;
	}

	// Increment forces
	for(int ic=0;ic<3;ic++) {
#ifdef _OPENMP
	  fwrk(3*anuc+ic)+=ders[ic];
	  fwrk(3*bnuc+ic)+=ders[ic+3];
#else
	  f(3*anuc+ic)+=ders[ic];
	  f(3*bnuc+ic)+=ders[ic+3];
#endif
	}
      }

#ifdef _OPENMP
#pragma omp critical
    // Sum results together
    f+=fwrk;
#endif
    delete deri;
  } // end parallel section


    // Second part: f = 1/2 c_a (a|b)' c_b *#* - gamma_a' c_a *#*
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    dERIWorker *deri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      deri=new dERIWorker(maxam,maxcontr);
    else
      deri=new dERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);
    const std::vector<double> * erip;

#ifdef _OPENMP
    // Worker stack for each matrix
    arma::vec fwrk(f);
    fwrk.zeros();

#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      size_t imus=orbpairs[ip].is;
      size_t inus=orbpairs[ip].js;

      size_t Nmu=orbshells[imus].get_Nbf();
      size_t Nnu=orbshells[inus].get_Nbf();

      size_t inuc=orbshells[imus].get_center_ind();
      size_t jnuc=orbshells[inus].get_center_ind();

      // If imus==inus, we need to take care that we count
      // every term only once; on the off-diagonal we get
      // every term twice.
      double fac=2.0;
      if(imus==inus)
	fac=1.0;

      for(size_t ias=0;ias<auxshells.size();ias++) {
	size_t Na=auxshells[ias].get_Nbf();
	size_t anuc=auxshells[ias].get_center_ind();

	if(inuc==jnuc && jnuc==anuc)
	  // Contributions vanish
	  continue;

	// Compute (a|mn)
	deri->compute(&auxshells[ias],&dummy,&orbshells[imus],&orbshells[inus]);

	// Expansion coefficients
	arma::vec ca=c.subvec(auxshells[ias].get_first_ind(),auxshells[ias].get_last_ind());

	// Compute forces
	const static int index[]={0, 1, 2, 6, 7, 8, 9, 10, 11};
	const size_t Nidx=sizeof(index)/sizeof(index[0]);
	double ders[Nidx];

	for(size_t iid=0;iid<Nidx;iid++) {
	  // Index is
	  int ic=index[iid];
	  arma::vec hlp(Na);

	  erip=deri->getp(ic);
	  hlp.zeros();
	  for(size_t iia=0;iia<Na;iia++)
	    for(size_t iimu=0;iimu<Nmu;iimu++) {
	      size_t imu=orbshells[imus].get_first_ind()+iimu;
	      for(size_t iinu=0;iinu<Nnu;iinu++) {
		size_t inu=orbshells[inus].get_first_ind()+iinu;

		// The contracted integral
		hlp(iia)+=(*erip)[(iia*Nmu+iimu)*Nnu+iinu]*P(imu,inu);
	      }
	    }
	  ders[iid]=fac*arma::dot(hlp,ca);
	}

	// Increment forces
	for(int ic=0;ic<3;ic++) {
#ifdef _OPENMP
	  fwrk(3*anuc+ic)-=ders[ic];
	  fwrk(3*inuc+ic)-=ders[ic+3];
	  fwrk(3*jnuc+ic)-=ders[ic+6];
#else
	  f(3*anuc+ic)-=ders[ic];
	  f(3*inuc+ic)-=ders[ic+3];
	  f(3*jnuc+ic)-=ders[ic+6];
#endif
	}

      }
    }

#ifdef _OPENMP
#pragma omp critical
    // Sum results together
    f+=fwrk;
#endif
    delete deri;
  } // end parallel section

  return f;
}

arma::mat DensityFit::calcK(const arma::mat & Corig, const std::vector<double> & occo, size_t fitmem) const {
  if(Corig.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", Corig.n_rows = " << Corig.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // Count number of orbitals
  size_t Nmo=0;
  for(size_t i=0;i<occo.size();i++)
    if(occo[i]>0)
      Nmo++;

  // Collect orbitals to use
  arma::mat C(Nbf,Nmo);
  arma::vec occs(Nmo);
  {
    size_t io=0;
    for(size_t i=0;i<occo.size();i++)
      if(occo[i]>0) {
	// Store orbital and occupation number
	C.col(io)=Corig.col(i);
	occs[io]=occo[i];
	io++;
      }
  }

  // Returned matrix
  arma::mat K(Nbf,Nbf);
  K.zeros();

  if(!direct) {
    digest_K_incore(C,occs,K);

  } else {
    // Determine how many orbitals we can handle at one go. Memory
    // needed for one orbital is
    size_t oneorb(sizeof(double)*Naux*Nbf);
#ifdef _OPENMP
    // Each thread needs its own storage
    oneorb*=omp_get_max_threads();
#endif

    // Number of orbitals that can be handled in one block is
    size_t blocksize(floor(fitmem*1.0/oneorb));
    if(blocksize<1) {
      std::ostringstream oss;
      oss << "Not enough fitting memory! Need at least " << memory_size(oneorb) << " per orbital!\n";
      throw std::logic_error(oss.str());
    }
    // Number of blocks is then
    size_t nblocks(ceil(C.n_cols*1.0/blocksize));

    //printf("Handling %i orbitals at once.\n",(int) blocksize);

    // Loop over blocks
    for(size_t iblock=0;iblock<nblocks;iblock++) {
      // Block starts at
      size_t iobeg(iblock*blocksize);
      size_t ioend(std::min((iblock+1)*blocksize-1,(size_t) C.n_cols-1));

      //printf("Treating orbitals %i-%i\n",(int) iobeg,(int) ioend);
      //fflush(stdout);

      digest_K_direct(C.cols(iobeg,ioend),occs.subvec(iobeg,ioend),K);
    }
  }

  return K;
}

arma::mat DensityFit::calcK(const arma::mat & Corig, const arma::vec & occo, size_t fitmem) const {
  return calcK(Corig, arma::conv_to<std::vector<double>>::from(occo), fitmem);
}

arma::cx_mat DensityFit::calcK(const arma::cx_mat & Corig, const std::vector<double> & occo, size_t fitmem) const {
  if(Corig.n_rows != Nbf) {
    std::ostringstream oss;
    oss << "Error in DensityFit: Nbf = " << Nbf << ", Corig.n_rows = " << Corig.n_rows << "!\n";
    throw std::logic_error(oss.str());
  }

  // Count number of orbitals
  size_t Nmo=0;
  for(size_t i=0;i<occo.size();i++)
    if(occo[i]>0)
      Nmo++;

  // Collect orbitals to use
  arma::cx_mat C(Nbf,Nmo);
  arma::vec occs(Nmo);
  {
    size_t io=0;
    for(size_t i=0;i<occo.size();i++)
      if(occo[i]>0) {
	// Store orbital and occupation number
	C.col(io)=Corig.col(i);
	occs[io]=occo[i];
	io++;
      }
  }

  // Returned matrix
  arma::cx_mat K(Nbf,Nbf);
  K.zeros();

  if(!direct) {
    digest_K_incore(C,occs,K);
  } else {
    (void) fitmem;
    throw std::logic_error("Direct mode hasn't been implemented for density-fitted complex exchange!\n");
  }

  return K;
}

arma::cx_mat DensityFit::calcK(const arma::cx_mat & Corig, const arma::vec & occo, size_t fitmem) const {
  return calcK(Corig, arma::conv_to<std::vector<double>>::from(occo), fitmem);
}

size_t DensityFit::get_Norb() const {
  return Nbf;
}

size_t DensityFit::get_Naux() const {
  return Naux;
}

size_t DensityFit::get_Naux_indep() const {
  return ab_invh.n_cols;
}

arma::mat DensityFit::get_ab() const {
  return ab;
}

arma::mat DensityFit::get_ab_inv() const {
  return ab_inv;
}

arma::mat DensityFit::get_ab_invh() const {
  return ab_invh;
}

void DensityFit::three_center_integrals(arma::mat & ints) const {
  if(direct)
    throw std::runtime_error("Must run in tabulated mode!\n");

  // Collect AO integrals
  ints.zeros(Nbf*Nbf,Naux);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker *eri;
    if(omega==0.0 && alpha==1.0 && beta==0.0)
      eri=new ERIWorker(maxam,maxcontr);
    else
      eri=new ERIWorker_srlr(maxam,maxcontr,omega,alpha,beta);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(size_t ip=0;ip<orbpairs.size();ip++) {
      size_t imus=orbpairs[ip].is;
      size_t inus=orbpairs[ip].js;
      size_t Nmu=orbshells[imus].get_Nbf();
      size_t Nnu=orbshells[inus].get_Nbf();
      size_t mu0=orbshells[imus].get_first_ind();
      size_t nu0=orbshells[inus].get_first_ind();

      const arma::mat amunu(compute_a_munu(eri,ip));

      for(size_t ias=0;ias<auxshells.size();ias++) {
        size_t Na=auxshells[ias].get_Nbf();
        size_t a0=auxshells[ias].get_first_ind();

        for(size_t imu=0;imu<Nmu;imu++)
          for(size_t inu=0;inu<Nnu;inu++)
            for(size_t ia=0;ia<Na;ia++) {
              size_t mu=imu+mu0;
              size_t nu=inu+nu0;
              size_t a=ia+a0;

              double el(amunu(ia+a0,inu*Nmu+imu));
              ints(mu*Nbf+nu,a)=el;
              ints(nu*Nbf+mu,a)=el;
            }
      }
    }

    delete eri;
  }
}

void DensityFit::B_matrix(arma::mat & B_) const {
  B_=B;
}
