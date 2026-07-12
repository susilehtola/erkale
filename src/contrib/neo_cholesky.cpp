/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Shared-pivot Cholesky decomposition for multicomponent (NEO) systems.
 * See neo_cholesky.h for the construction and why it is the right object.
 */

#include "neo_cholesky.h"
#include "basis.h"
#include "density_fitting.h"
#include "eriworker.h"
#include "linalg.h"
#include "timer.h"

#include <cstdio>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

  typedef std::pair<size_t, size_t> pivpair_t;

  /// Concatenate the two orbital bases into one pivot shell list with globally
  /// unique first_ind: electronic functions occupy [0, Ne), protonic [Ne, Ne+Np).
  /// The pivot index matrix and DirectCDBlocks both address this space.
  std::vector<GaussianShell> concatenate_shells(const BasisSet & ebasis, const BasisSet & pbasis) {
    std::vector<GaussianShell> shells = ebasis.get_shells();
    const size_t Ne = ebasis.get_Nbf();
    for(GaussianShell sh: pbasis.get_shells()) {
      sh.set_first_ind(sh.get_first_ind() + Ne);
      shells.push_back(sh);
    }
    return shells;
  }

  /// (piv|piv) over the combined pivot set. Identical in structure to
  /// fill_cholesky's phase D, but the shellpairs may belong to either species,
  /// so the electron-proton cross block is filled by the same loop.
  arma::mat pivot_metric(const std::vector<GaussianShell> & shells,
                         const std::vector<pivpair_t> & piv_sp,
                         const arma::umat & piv_index,
                         arma::uword sentinel,
                         size_t Nselected) {
    arma::mat M(Nselected, Nselected, arma::fill::zeros);

    // libcint description of the combined pivot shells
    CintEnv cenv(shells);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      auto eri = make_eri_worker(cenv, 0.0, 1.0, 0.0);

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
      for(size_t ip=0; ip<piv_sp.size(); ip++) {
        const size_t is = piv_sp[ip].first;
        const size_t js = piv_sp[ip].second;
        const size_t Ni = shells[is].get_Nbf();
        const size_t Nj = shells[js].get_Nbf();
        const size_t i0 = shells[is].get_first_ind();
        const size_t j0 = shells[js].get_first_ind();
        for(size_t jp=0; jp<=ip; jp++) {
          const size_t ks = piv_sp[jp].first;
          const size_t ls = piv_sp[jp].second;
          const size_t Nk = shells[ks].get_Nbf();
          const size_t Nl = shells[ls].get_Nbf();
          const size_t k0 = shells[ks].get_first_ind();
          const size_t l0 = shells[ls].get_first_ind();

          eri->compute(is,js,ks,ls);
          const std::vector<double> * erip = eri->getp();

          for(size_t ii=0; ii<Ni; ii++)
            for(size_t jj=0; jj<Nj; jj++) {
              const arma::uword pidx = piv_index(i0+ii, j0+jj);
              if(pidx == sentinel) continue;
              for(size_t kk=0; kk<Nk; kk++)
                for(size_t ll=0; ll<Nl; ll++) {
                  const arma::uword qidx = piv_index(k0+kk, l0+ll);
                  if(qidx == sentinel) continue;
                  const double val = (*erip)[((ii*Nj+jj)*Nk+kk)*Nl+ll];
                  // Each (pidx,qidx) maps to a unique unordered pivot-shellpair
                  // pair, visited exactly once by the ip / jp<=ip loop.
                  M(pidx, qidx) = val;
                  M(qidx, pidx) = val;
                }
            }
        }
      }
    }
    return M;
  }

} // anonymous namespace

void neo_shared_cholesky(const BasisSet & ebasis, const BasisSet & pbasis,
                         bool direct,
                         double cholesky_tol,
                         double shell_reuse_thr,
                         double shell_screen_tol,
                         double fit_cholesky_thr,
                         bool verbose,
                         DensityFit & dfit, DensityFit & pfit) {
  Timer ttot;

  const size_t Ne = ebasis.get_Nbf();
  const size_t Np = pbasis.get_Nbf();
  if(!Np)
    throw std::runtime_error("neo_shared_cholesky: no protonic basis functions.\n");

  // Pivots per species, then unioned. A jointly pivoted selection would pick a
  // smaller set, but the union is bounded at the same threshold (see header)
  // and the joint orthogonalisation below prunes the redundancy.
  DensityFit selector;   // bare 1/r12; the operator must be shared by all blocks
  arma::uvec pi_e, pi_p;
  arma::umat invmap_e, invmap_p;
  std::set<pivpair_t> sp_e, sp_p;
  const size_t Nsel_e = selector.select_two_step_pivots(ebasis, cholesky_tol, shell_reuse_thr,
                                                        shell_screen_tol, verbose, pi_e, invmap_e, sp_e);
  const size_t Nsel_p = selector.select_two_step_pivots(pbasis, cholesky_tol, shell_reuse_thr,
                                                        shell_screen_tol, verbose, pi_p, invmap_p, sp_p);
  const size_t Nselected = Nsel_e + Nsel_p;

  const std::vector<GaussianShell> piv_shells = concatenate_shells(ebasis, pbasis);
  const size_t nesh = ebasis.get_shells().size();

  // Pivot shellpairs: electronic first, then protonic with shell indices shifted
  // into the concatenated list. Ranks follow the same order, so the electronic
  // pivots occupy [0, Nsel_e) of the shared vector index.
  std::vector<pivpair_t> piv_sp(sp_e.begin(), sp_e.end());
  for(const pivpair_t & q: sp_p)
    piv_sp.push_back(pivpair_t(q.first + nesh, q.second + nesh));

  const arma::uword sentinel = Nselected;
  arma::umat piv_index(Ne+Np, Ne+Np);
  piv_index.fill(sentinel);
  for(size_t p=0; p<Nsel_e; p++) {
    const arma::uword pii = pi_e(p);
    piv_index(invmap_e(0,pii), invmap_e(1,pii)) = p;
    piv_index(invmap_e(1,pii), invmap_e(0,pii)) = p;
  }
  for(size_t p=0; p<Nsel_p; p++) {
    const arma::uword pii = pi_p(p);
    piv_index(Ne+invmap_p(0,pii), Ne+invmap_p(1,pii)) = Nsel_e + p;
    piv_index(Ne+invmap_p(1,pii), Ne+invmap_p(0,pii)) = Nsel_e + p;
  }

  const int piv_max_am = std::max(ebasis.get_max_am(), pbasis.get_max_am());
  const int piv_max_contr = std::max(ebasis.get_max_Ncontr(), pbasis.get_max_Ncontr());

  Timer t;
  arma::mat M = pivot_metric(piv_shells, piv_sp, piv_index, sentinel, Nselected);
  const double t_int = t.get();

  // Orthogonalise the joint metric. Normalise to unit diagonal first: the
  // protonic pivot products are far tighter than the electronic ones, so their
  // self-energies (A|A) differ by orders of magnitude, and a threshold applied
  // to the raw metric would discard independent electronic functions purely
  // because they are small. On the unit-diagonal correlation matrix the
  // eigenvalues measure genuine linear dependence. Fold the normalisation back
  // in afterwards, so X^T M X = I on the true metric.
  t.set();
  arma::vec dinv(Nselected);
  for(arma::uword p=0; p<Nselected; p++) {
    const double dpp = M(p,p);
    dinv(p) = (dpp > 0.0) ? 1.0/std::sqrt(dpp) : 0.0;
  }
  arma::mat Mtilde(M);
  Mtilde.each_col() %= dinv;
  Mtilde.each_row() %= dinv.t();
  arma::mat X = CanonicalOrth(Mtilde, fit_cholesky_thr);
  X.each_col() %= dinv;
  const double t_chol = t.get();

  if(verbose) {
    printf("\nNEO shared-pivot Cholesky decomposition.\n");
    printf("  pivots: %i electronic + %i protonic = %i, orthogonalised to %i shared vectors.\n",
           (int) Nsel_e, (int) Nsel_p, (int) Nselected, (int) X.n_cols);
    printf("  metric build %s (integrals %3.1f %%, linear algebra %3.1f %%).\n",
           t.elapsed().c_str(),
           100*t_int/(t_int+t_chol), 100*t_chol/(t_int+t_chol));
    fflush(stdout);
  }

  dfit.fill_cholesky_shared(ebasis, piv_shells, piv_sp, piv_index, sentinel, X,
                            direct, shell_screen_tol, piv_max_am, piv_max_contr, verbose);
  pfit.fill_cholesky_shared(pbasis, piv_shells, piv_sp, piv_index, sentinel, X,
                            direct, shell_screen_tol, piv_max_am, piv_max_contr, verbose);

  if(verbose) {
    printf("NEO shared-pivot Cholesky finished in %s.\n", ttot.elapsed().c_str());
    fflush(stdout);
  }
}
