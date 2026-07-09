/*
 * White-box finite-difference check of DensityFit::forceJ_cholesky and
 * DensityFit::forceK in two-step Cholesky (CD) mode.
 *
 * forceK / forceJ_cholesky compute the EXPLICIT (fixed-density) nuclear
 * gradient of the fitted exchange / Coulomb energy. We verify them against a
 * finite difference of the very same energy, with the density (C, occ) frozen
 * and the CD tensor rebuilt at each displaced geometry:
 *
 *   E_J = (1/2) tr(P J),   J = calcJ(P)            -> forceJ_cholesky
 *   E_K = -(1/4) tr(P K),  K = calcK(C, occ)       -> forceK
 *
 * calcK sums over all pivots, so the energy FD is insensitive to pivot
 * reordering.
 *
 * This test caught a pivot double-counting bug in the CD derivative kernels
 * (accumulate_2c_metric_force / accumulate_3c_force_CD): cd_pivot_index is
 * symmetric, so within a diagonal pivot shellpair both (kappa, lambda) and
 * (lambda, kappa) mapped to the same pivot rank and the derivative
 * contractions accumulated it twice, whereas the value-side metric / 3-index
 * builds assign (once per pivot). forceK was off by ~2.4e-3 on H2O/cc-pVDZ HF
 * regardless of CholeskyThr. forceJ_cholesky shares the kernels but appeared
 * correct on C2v water, because by symmetry the total density has zero
 * fitting coefficient on the affected (same-shell off-diagonal) pivot
 * products -- use a distorted geometry to actually exercise the J path.
 * With the fixed kernels both forces match FD to ~1e-8.
 *
 * Usage: cdforcetest <chkfile> <orbital-basis-name> [CholeskyThr=1e-12] [delta=1e-5]
 */

#include "../basis.h"
#include "../basislibrary.h"
#include "../checkpoint.h"
#include "../density_fitting.h"
#include "../eriworker.h"
#include "../settings.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

Settings settings;

static const double cholshthr = 0.01;
static const double intthr    = 1e-10;
static const double fitcholthr= 1e-14;  // full rank

static double & coordref(nucleus_t & n, int x) {
  return (x==0) ? n.r.x : (x==1 ? n.r.y : n.r.z);
}

static double energy(const std::vector<nucleus_t> & nuc, const BasisSetLibrary & baslib,
                     const arma::mat & C, const std::vector<double> & occs,
                     const arma::mat & P, double cholthr, char which) {
  BasisSet b; construct_basis(b, nuc, baslib);
  DensityFit dfit;
  dfit.fill_cholesky(b, false, cholthr, cholshthr, intthr, fitcholthr, false);
  if(which=='J') { arma::mat J=dfit.calcJ(P); return 0.5*arma::trace(P*J); }
  arma::mat K=dfit.calcK(C,occs); return -0.25*arma::trace(P*K);
}

int main(int argc, char ** argv) {
  if(argc<3) { fprintf(stderr,"usage: %s <chkfile> <basis> [CholeskyThr=1e-12] [delta=1e-5]\n",argv[0]); return 1; }
  const std::string chkf=argv[1], basisname=argv[2];
  const double cholthr=(argc>3)?atof(argv[3]):1e-12;
  const double delta  =(argc>4)?atof(argv[4]):1e-5;

  init_libint_base();
  init_libderiv_base();
  settings.add_scf_settings();

  Checkpoint chk(chkf,false);
  BasisSet basis_chk; chk.read(basis_chk);
  arma::mat C; chk.read("C",C);
  int Nela=0; chk.read("Nel-a",Nela);
  printf("Loaded %s: Nbf=%i Nel-a=%i\n",chkf.c_str(),(int)basis_chk.get_Nbf(),Nela);

  std::vector<double> occs(C.n_cols,0.0); arma::vec occv(C.n_cols,arma::fill::zeros);
  for(int i=0;i<Nela;i++){occs[i]=2.0; occv(i)=2.0;}
  const arma::mat P=C*arma::diagmat(occv)*C.t();

  BasisSetLibrary baslib; baslib.load_basis(basisname);
  const std::vector<nucleus_t> nuc0=basis_chk.get_nuclei();
  const size_t Nnuc=nuc0.size();

  BasisSet b0; construct_basis(b0,nuc0,baslib);
  if(b0.get_Nbf()!=basis_chk.get_Nbf()){ fprintf(stderr,"basis mismatch\n"); return 1; }

  DensityFit dfit0; dfit0.fill_cholesky(b0,false,cholthr,cholshthr,intthr,fitcholthr,false);
  const arma::vec fJ=dfit0.forceJ_cholesky(b0,P);
  const arma::vec fK=dfit0.forceK(b0,C,occs,1.0);

  const size_t npiv0=dfit0.find_cholesky_pivots(b0,cholthr,cholshthr,intthr,false).size();
  { std::vector<nucleus_t> nud=nuc0; coordref(nud[0],2)+=delta;
    BasisSet bd; construct_basis(bd,nud,baslib); DensityFit dfd;
    const size_t npivd=dfd.find_cholesky_pivots(bd,cholthr,cholshthr,intthr,false).size();
    printf("Pivot shellpairs: R0=%i displaced=%i => %s\n",(int)npiv0,(int)npivd,(npiv0==npivd)?"STABLE":"CHANGED"); }

  arma::vec fdJ(3*Nnuc), fdK(3*Nnuc);
  for(size_t a=0;a<Nnuc;a++) for(int x=0;x<3;x++){
    std::vector<nucleus_t> p=nuc0,m=nuc0; coordref(p[a],x)+=delta; coordref(m[a],x)-=delta;
    fdJ(3*a+x)=(energy(p,baslib,C,occs,P,cholthr,'J')-energy(m,baslib,C,occs,P,cholthr,'J'))/(2*delta);
    fdK(3*a+x)=(energy(p,baslib,C,occs,P,cholthr,'K')-energy(m,baslib,C,occs,P,cholthr,'K'))/(2*delta);
  }
  auto report=[&](const char* tag,const arma::vec& a,const arma::vec& f){
    double mm=0; for(size_t i=0;i<a.n_elem;i++) mm=std::max(mm,std::min(std::fabs(a(i)-f(i)),std::fabs(a(i)+f(i))));
    printf("%-34s sign-robust max mismatch = %.3e\n",tag,mm);
  };
  report("forceJ_cholesky (control)",fJ,fdJ);
  report("forceK (suspect)",fK,fdK);
  return 0;
}
