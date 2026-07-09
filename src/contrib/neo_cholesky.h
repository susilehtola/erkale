/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Shared-pivot Cholesky decomposition for multicomponent (NEO) systems.
 */

#ifndef ERKALE_NEO_CHOLESKY
#define ERKALE_NEO_CHOLESKY

#include <armadillo>

class BasisSet;
class DensityFit;

/**
 * Decompose the electronic and protonic two-particle integrals against one
 * shared set of Cholesky vectors.
 *
 * The object being decomposed is the Gram matrix of the *union of the two
 * same-species pair spaces* under the Coulomb metric,
 *
 *     M = [ (ee|ee)  (ee|pp) ]
 *         [ (pp|ee)  (pp|pp) ]
 *
 * and not the pair space of the union of the two basis sets. The latter would
 * additionally generate cross products (mu a) of one electronic and one
 * protonic function. Those have large diagonals -- protonic functions are
 * tight and sit on the electronic ones -- so they would attract pivots, and
 * they reconstruct (mu a | nu b) integrals no NEO model uses: electrons and
 * protons are distinguishable, so there is no electron-proton exchange.
 *
 * ERKALE's CD is two-step, i.e. density fitting in an auxiliary basis of pivot
 * orbital products. So this does not decompose M explicitly and slice the
 * result; it builds a pivot set spanning both pair spaces, one metric
 * M = (piv|piv) over it, and one orthogonaliser X = M^{-1/2}. Each species'
 * DensityFit then holds L = X^T (piv | mu nu) over its own orbital basis, and
 * the two therefore share a vector index P:
 *
 *     (mu nu | la si) = sum_P B_e[P,mu,nu] B_e[P,la,si]
 *     (a b   | c d)   = sum_P B_p[P,a,b]   B_p[P,c,d]
 *     (mu nu | a b)   = sum_P B_e[P,mu,nu] B_p[P,a,b]      <-- exact, shared P
 *
 * The last line is the point. With independent per-species fits it holds only
 * when both species happen to be expanded in the same auxiliary basis, and
 * never for a screened operator.
 *
 * The pivots are the *union of the two independently selected pivot sets*
 * rather than a jointly pivoted selection. The standard Cholesky bound
 * |dev (mu nu|a b)| <= sqrt(D_munu * D_ab) on the residual diagonals D bounds
 * the cross block at the same threshold as the diagonal blocks, and the joint
 * canonical orthogonalisation below removes whatever redundancy the union
 * introduces. Crucially that orthogonalisation normalises the metric to unit
 * diagonal first, so the ~10^6 disparity between tight protonic and diffuse
 * electronic pivot self-energies discards functions by genuine linear
 * dependence rather than by sheer magnitude.
 *
 * Only meaningful when all three blocks share the bare 1/r12 operator, i.e.
 * point protons. A screened (finite-proton) e-p operator is a different
 * kernel and cannot reuse these vectors; the caller must not request it.
 *
 * Independent of vpp: whether the proton-proton mean field enters the SCF Fock
 * is a separate question from whether the protonic block is decomposed. A
 * downstream correlation treatment needs B_p either way.
 *
 * On return dfit and pfit are CD-mode DensityFit objects with equal Naux.
 * CD gradients are unavailable on them (the pivot basis is not the orbital
 * basis) and throw if called.
 *
 * \param ebasis,pbasis      electronic and protonic orbital bases
 * \param direct             recompute (piv|mu nu) per block instead of caching
 * \param cholesky_tol       pivoted-CD threshold on the residual diagonal
 * \param shell_reuse_thr,shell_screen_tol  pivot-selection screening controls
 * \param fit_cholesky_thr   linear-dependence threshold of the joint metric
 * \param verbose            print the decomposition summary
 * \param dfit,pfit          filled on return
 */
void neo_shared_cholesky(const BasisSet & ebasis, const BasisSet & pbasis,
                         bool direct,
                         double cholesky_tol,
                         double shell_reuse_thr,
                         double shell_screen_tol,
                         double fit_cholesky_thr,
                         bool verbose,
                         DensityFit & dfit, DensityFit & pfit);

#endif
