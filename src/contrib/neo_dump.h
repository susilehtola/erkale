/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * NEO-SCF export for external post-SCF correlation codes.
 *
 * Writes a converged nuclear-electronic-orbital SCF to a self-describing
 * HDF5 file (see neo_dump_format.md) containing AO/MO data and the bare
 * (sign-free) two-particle integrals (electron-electron, proton-proton,
 * electron-proton) needed by a spin-orbital NEO-CCSD driver.
 */

#ifndef ERKALE_NEO_DUMP
#define ERKALE_NEO_DUMP

#include <armadillo>
#include <string>
#include <vector>

class BasisSet;
class DensityFit;

/**
 * Write a converged NEO-SCF to an HDF5 dump.
 *
 * The electron blocks are passed as vectors: size 1 for restricted (RHF),
 * size 2 (alpha,beta) for unrestricted (UHF). Coefficients are AO x MO in the
 * full AO basis; they are written verbatim (reordered occupied-first), because
 * they are the orbitals that produced the SCF density.
 *
 * No Fock matrix or orbital energy is dumped. The reference one-particle
 * operator is use-case dependent -- a correlation model without a
 * proton-proton fluctuation potential wants the self-interaction-free proton
 * Fock h_p + V_ep[D_e], one with it wants the J/K-dressed Fock -- so the
 * consumer builds whichever it needs from the core Hamiltonian, the dumped
 * two-particle factors and the occupied densities, and takes its orbital
 * energies from the diagonal of the resulting MO Fock.
 *
 * \param filename        output HDF5 file
 * \param representation  "btensor" (engine CD/RI factors) or "dense" (full rank-4)
 * \param do_verify       reconstruct the SCF energy from the dumped tensors and check
 * \param ebasis,dfit     electron basis and its (converged) density-fit/Cholesky engine
 * \param restricted_e    true=RHF (one electron block), false=UHF (two blocks)
 * \param Ce,occe         per-block electron SCF MOs and occupations
 * \param hcore_e         electron AO core Hamiltonian (spin-independent)
 * \param pbasis,pfit     proton basis and engine
 * \param Cp,occp,hcore_p proton SCF MOs, occupations, AO core Hamiltonian
 * \param n_electrons,n_protons,proton_mass  counts / mass
 * \param e_scf,e_classical  SCF total energy and classical nuclear repulsion
 * \param density_fitting omega/alpha/beta select the e-p (and p-p) operator
 * \param version         ERKALE version string for provenance
 */
void neo_dump(const std::string & filename,
              const std::string & representation,
              bool do_verify,
              // electrons
              const BasisSet & ebasis, const DensityFit & dfit,
              bool restricted_e,
              const std::vector<arma::mat> & Ce,
              const std::vector<arma::vec> & occe,
              const arma::mat & hcore_e,
              // protons
              const BasisSet & pbasis, const DensityFit & pfit,
              const arma::mat & Cp, const arma::vec & occp,
              const arma::mat & hcore_p,
              // scalars
              int n_electrons, int n_protons, double proton_mass,
              double e_scf, double e_classical,
              bool density_fitting, double omega, double alpha, double beta,
              const std::string & version);

#endif
