/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * NEO-SCF export for external post-SCF correlation codes. See
 * neo_dump_format.md for the on-disk format and the index/sign conventions.
 */

#include "neo_dump.h"
#include "basis.h"
#include "density_fitting.h"
#include "eriworker.h"

#include <hdf5.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace {

  // ---- raw HDF5 writers (C API, mirroring checkpoint.cpp's idiom) ----

  void write_str_attr(hid_t loc, const std::string & name, const std::string & val) {
    hid_t atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, val.size()+1);
    H5Tset_strpad(atype, H5T_STR_NULLTERM);
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name.c_str(), atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, atype, val.c_str());
    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);
  }

  void write_int_attr(hid_t loc, const std::string & name, int val) {
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name.c_str(), H5T_NATIVE_INT, aspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &val);
    H5Aclose(attr);
    H5Sclose(aspace);
  }

  void write_dbl_attr(hid_t loc, const std::string & name, double val) {
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name.c_str(), H5T_NATIVE_DOUBLE, aspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
    H5Sclose(aspace);
  }

  // Scalar integer dataset.
  void write_int(hid_t loc, const std::string & name, int val) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t ds = H5Dcreate(loc, name.c_str(), H5T_NATIVE_INT, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
    H5Dclose(ds);
    H5Sclose(sp);
  }

  // Double dataset from a flat C-order buffer of the given dimensions.
  void write_buffer(hid_t loc, const std::string & name, const double * data,
                    const std::vector<hsize_t> & dims) {
    hid_t sp = H5Screate_simple((int) dims.size(), dims.data(), NULL);
    hid_t ds = H5Dcreate(loc, name.c_str(), H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(sp);
  }

  // 1D vector.
  void write_vec(hid_t loc, const std::string & name, const arma::vec & v) {
    std::vector<hsize_t> dims = { (hsize_t) v.n_elem };
    write_buffer(loc, name, v.memptr(), dims);
  }

  // 2D matrix, written in C/row-major order so an h5py reader gets A[i,j]=M(i,j).
  // Armadillo is column-major, so the column-major buffer of M.t() is exactly the
  // row-major buffer of M.
  void write_mat(hid_t loc, const std::string & name, const arma::mat & M) {
    arma::mat Mt = M.t();
    std::vector<hsize_t> dims = { (hsize_t) M.n_rows, (hsize_t) M.n_cols };
    write_buffer(loc, name, Mt.memptr(), dims);
  }

  // ---- engine integral helpers ----

  // The engine B factor (Nbf*Nbf) x Naux, with rows indexed by the C-order pair
  // index mu*Nbf+nu and columns by the aux/Cholesky vector. (pq|rs) is
  // reconstructed as sum_P B(p*Nbf+q, P) B(r*Nbf+s, P).
  arma::mat B_factor(const DensityFit & fit) {
    arma::mat B;
    fit.B_matrix(B);
    return B;
  }

  // Write the B factor as a (Naux, Nbf, Nbf) C-order tensor. B.memptr()
  // (column-major (Nbf*Nbf) x Naux) IS the C-order (Naux,Nbf,Nbf) buffer with
  // element [P,mu,nu] = B(mu*Nbf+nu, P), so no transpose is needed.
  void write_B(hid_t loc, const arma::mat & B, size_t Nbf) {
    size_t Naux = B.n_cols;
    std::vector<hsize_t> dims = { (hsize_t) Naux, (hsize_t) Nbf, (hsize_t) Nbf };
    write_buffer(loc, "B", B.memptr(), dims);
  }

  // Exact electron-proton Coulomb integrals (mu nu | a b), bare (no sign),
  // with the engine's (omega,alpha,beta) operator so the finite-proton
  // (screened) model is honored. Returns G(mu*Ne+nu, a*Np+b) = (mu nu | a b).
  // Mirrors the contraction loop in neo.cpp's multicomponent_coulomb_tei, but
  // stores the full tensor instead of contracting with a density.
  arma::mat exact_ep(const BasisSet & ebasis, const BasisSet & pbasis,
                     double omega, double alpha, double beta) {
    std::vector<GaussianShell> eshells = ebasis.get_shells();
    std::vector<GaussianShell> pshells = pbasis.get_shells();
    size_t Ne = ebasis.get_Nbf();
    size_t Np = pbasis.get_Nbf();

    arma::mat G(Ne*Ne, Np*Np, arma::fill::zeros);

    // libcint environment: the electronic shells, followed by the
    // protonic ones
    CintEnv cenv(ebasis, pbasis);
    const size_t Nsh_e = cenv.get_Nsh_orb();
    auto eri_owner = make_eri_worker(cenv, omega, alpha, beta);
    ERIWorker * eri = eri_owner.get();

    for(size_t i=0;i<eshells.size();i++) {
      size_t Ni = eshells[i].get_Nbf();
      size_t i0 = eshells[i].get_first_ind();
      for(size_t j=0;j<eshells.size();j++) {
        size_t Nj = eshells[j].get_Nbf();
        size_t j0 = eshells[j].get_first_ind();
        for(size_t k=0;k<pshells.size();k++) {
          size_t Nk = pshells[k].get_Nbf();
          size_t k0 = pshells[k].get_first_ind();
          for(size_t l=0;l<pshells.size();l++) {
            size_t Nl = pshells[l].get_Nbf();
            size_t l0 = pshells[l].get_first_ind();

            eri->compute(i,j,Nsh_e+k,Nsh_e+l);
            const std::vector<double> & ints = *eri->getp();

            for(size_t ii=0;ii<Ni;ii++)
              for(size_t jj=0;jj<Nj;jj++)
                for(size_t kk=0;kk<Nk;kk++)
                  for(size_t ll=0;ll<Nl;ll++) {
                    double val = ints[((ii*Nj+jj)*Nk+kk)*Nl+ll];
                    size_t mu = i0+ii, nu = j0+jj, a = k0+kk, b = l0+ll;
                    G(mu*Ne+nu, a*Np+b) = val;
                  }
          }
        }
      }
    }

    return G;
  }

  // K[D]_pq = sum_rs (pr|qs) D_rs, from a dense G(p*N+r, q*N+s) = (pr|qs).
  arma::mat calcK_from_G(const arma::mat & G, const arma::mat & D) {
    size_t N = D.n_rows;
    arma::mat K(N, N, arma::fill::zeros);
    for(size_t p=0;p<N;p++)
      for(size_t q=0;q<N;q++) {
        double k=0.0;
        for(size_t r=0;r<N;r++)
          for(size_t s=0;s<N;s++)
            k += G(p*N+r, q*N+s) * D(r,s);
        K(p,q)=k;
      }
    return K;
  }

  // AO density from energy-ordered MOs and occupations.
  arma::mat density(const arma::mat & C, const arma::vec & occ) {
    return C * arma::diagmat(occ) * C.t();
  }

  // Reorder the columns so the occupied orbitals come first, preserving the
  // relative order within each group (arma::find returns ascending indices).
  // The dump's occupation contract is purely positional.
  void occupied_first(arma::mat & C, arma::vec & occ) {
    arma::uvec perm = arma::join_cols(arma::find(occ > 0.0), arma::find(occ <= 0.0));
    C = C.cols(perm);
    occ = occ(perm);
  }

  // The consumer reads occupations off the column order: the first nocc columns
  // carry occ_full, the rest are empty. Fractional or non-aufbau occupations
  // cannot be expressed that way, so reject them rather than dump a file whose
  // stated nocc does not reproduce the SCF density.
  size_t checked_nocc(const arma::vec & occ, double occ_full, const std::string & label) {
    size_t nocc = (size_t) arma::accu(occ > 0.0);
    for(size_t i=0;i<occ.n_elem;i++) {
      double want = (i<nocc) ? occ_full : 0.0;
      if(std::abs(occ(i)-want) > 1e-10) {
        std::ostringstream oss;
        oss << "neo_dump: " << label << " orbital " << i << " has occupation " << occ(i)
            << ", expected " << want << ".\nFractional or non-aufbau occupations cannot be"
            << " expressed by the occupied-first column convention.\n";
        throw std::runtime_error(oss.str());
      }
    }
    return nocc;
  }

  // Max-norm deviation from orthonormality in the AO overlap metric.
  double orthonormality(const arma::mat & C, const arma::mat & S) {
    arma::mat I(C.n_cols, C.n_cols, arma::fill::eye);
    return arma::abs(C.t()*S*C - I).max();
  }

  // Semicanonical orbital energies: diagonalize F separately within the
  // occupied and the virtual block. This is the operation the consumer performs
  // after rebuilding its own reference Fock. It leaves both subspaces -- and
  // hence the density -- untouched, so it is valid for any F, including one the
  // SCF orbitals do not diagonalize. The surviving occupied-virtual coupling
  // max|F_ia| is returned; it measures how far the SCF orbitals are from being
  // canonical with respect to F.
  arma::vec semicanonical_eps(const arma::mat & C, const arma::mat & F, size_t nocc,
                              double & occvir) {
    arma::mat Fmo = C.t()*F*C;
    Fmo = 0.5*(Fmo + Fmo.t());
    size_t nmo = C.n_cols;

    occvir = (nocc>0 && nocc<nmo) ? arma::abs(Fmo.submat(0,nocc,nocc-1,nmo-1)).max() : 0.0;

    arma::vec eps(nmo);
    if(nocc)
      eps.subvec(0,nocc-1) = arma::eig_sym(Fmo.submat(0,0,nocc-1,nocc-1));
    if(nocc<nmo)
      eps.subvec(nocc,nmo-1) = arma::eig_sym(Fmo.submat(nocc,nocc,nmo-1,nmo-1));
    return eps;
  }

} // anonymous namespace

void neo_dump(const std::string & filename,
              const std::string & representation,
              bool do_verify,
              const BasisSet & ebasis, const DensityFit & dfit,
              bool restricted_e,
              const std::vector<arma::mat> & Ce,
              const std::vector<arma::vec> & occe,
              const arma::mat & hcore_e,
              const BasisSet & pbasis, const DensityFit & pfit,
              const arma::mat & Cp, const arma::vec & occp,
              const arma::mat & hcore_p,
              int n_electrons, int n_protons, double proton_mass,
              double e_scf, double e_classical,
              bool shared_aux, double omega, double alpha, double beta,
              const std::string & version) {

  if(representation != "btensor" && representation != "dense") {
    std::ostringstream oss;
    oss << "neo_dump: unknown integral representation '" << representation << "' (use btensor or dense)\n";
    throw std::runtime_error(oss.str());
  }
  bool dense = (representation == "dense");

  size_t Ne = ebasis.get_Nbf();
  size_t Np = pbasis.get_Nbf();

  printf("\nWriting NEO-SCF dump to %s (%s integrals).\n", filename.c_str(), representation.c_str());
  fflush(stdout);

  // Engine B factors and the dense integrals (used for verify, and for the
  // dense representation on disk). e-e/p-p come from the SCF's own engine
  // factor. The e-p tensor matches the engine the SCF used: the shared-aux RI
  // reconstruction B_e B_p^T in density-fitting mode, or the exact engine
  // integral (same omega,alpha,beta) in Cholesky mode -- where the SCF itself
  // evaluates e-p exactly.
  arma::mat Be = B_factor(dfit);
  arma::mat Bp = B_factor(pfit);
  arma::mat Gee = Be * Be.t();
  arma::mat Gpp = Bp * Bp.t();
  arma::mat Gep;
  if(shared_aux) {
    if(Be.n_cols != Bp.n_cols)
      throw std::runtime_error("neo_dump: the electron and proton factors do not share a vector index -- cannot form the e-p expansion.\n");
    Gep = Be * Bp.t();
  } else {
    Gep = exact_ep(ebasis, pbasis, omega, alpha, beta);
  }

  // AO overlap is the metric each species' basis lives in (also dumped).
  arma::mat S_e = ebasis.overlap();
  arma::mat S_p = pbasis.overlap();

  // ---- SCF orbitals, reordered occupied-first ----
  // The coefficients are written verbatim: they are the ones that produced the
  // SCF density. They are deliberately NOT recanonicalized against some Fock,
  // because which Fock is "the" reference is use-case dependent. In particular
  // the self-interaction-free proton Fock h_p + V_ep[D_e] does not share the
  // SCF occupied subspace once there is more than one quantum proton, so
  // diagonalizing it would silently hand the consumer a density that is not the
  // SCF density. The consumer rebuilds its own reference operator instead.
  double occ_e = restricted_e ? 2.0 : 1.0;
  std::vector<arma::mat> Cc(Ce);
  std::vector<arma::vec> occc(occe);
  std::vector<size_t> nocc_e(Ce.size());
  arma::mat Cp_o(Cp);
  arma::vec occp_o(occp);

  printf("\nNEO dump orbital check:\n");
  for(size_t s=0;s<Cc.size();s++) {
    const char * lbl = (Ce.size()==1) ? "electron" : (s==0 ? "electron alpha" : "electron beta");
    occupied_first(Cc[s], occc[s]);
    nocc_e[s] = checked_nocc(occc[s], occ_e, lbl);
    double rI = orthonormality(Cc[s], S_e);
    printf("  %-14s nocc %3i / nmo %3i   ||C'SC-I||=%.2e\n",
           lbl, (int) nocc_e[s], (int) Cc[s].n_cols, rI);
    if(rI>1e-8)
      throw std::runtime_error("neo_dump: electron orbitals are not orthonormal!\n");
  }
  occupied_first(Cp_o, occp_o);
  size_t nocc_p = checked_nocc(occp_o, 1.0, "proton");
  {
    double rI = orthonormality(Cp_o, S_p);
    printf("  %-14s nocc %3i / nmo %3i   ||C'SC-I||=%.2e\n",
           "proton", (int) nocc_p, (int) Cp_o.n_cols, rI);
    if(rI>1e-8)
      throw std::runtime_error("neo_dump: proton orbitals are not orthonormal!\n");
  }
  fflush(stdout);

  // ---- SCF densities (the quantities the dumped orbitals must reproduce) ----
  arma::mat De(Ne, Ne, arma::fill::zeros);
  std::vector<arma::mat> Dspin;
  for(size_t s=0;s<Cc.size();s++) {
    Dspin.push_back(density(Cc[s], occc[s]));
    De += Dspin.back();
  }
  arma::mat Dp = density(Cp_o, occp_o);
  arma::vec de_vec = arma::vectorise(De); // symmetric => pair index mu*Ne+nu
  arma::vec dp_vec = arma::vectorise(Dp);

  // ---- diagnostic: the self-interaction-free proton spectrum ----
  // Built here from exactly the quantities that go on disk (h_p and the e-p
  // tensor contracted with the electron density), i.e. along the same path the
  // consumer takes. Nothing below is written to the file.
  {
    arma::mat Vep(-arma::reshape(Gep.t()*de_vec, Np, Np));
    arma::mat Fp_si = hcore_p + 0.5*(Vep + Vep.t());
    double occvir;
    arma::vec eps = semicanonical_eps(Cp_o, Fp_si, nocc_p, occvir);

    printf("\nSelf-interaction-free proton spectrum (h_p + V_ep[D_e], semicanonical):\n");
    printf("  highest occupied  % .6f\n", nocc_p ? eps(nocc_p-1) : 0.0);
    // Dissociation threshold for a quantum proton is 0 (free proton at rest,
    // vanishing potential at infinity); bound orbitals have eps < 0.
    const double thr = 0.0;
    if(nocc_p < (size_t) Cp_o.n_cols) {
      double lowest_virtual = eps(nocc_p);
      size_t nbound = (size_t) arma::accu(eps.subvec(nocc_p, Cp_o.n_cols-1) < thr);
      printf("  lowest virtual    % .6f  (%i bound virtual(s) below %.1f)\n",
             lowest_virtual, (int) nbound, thr);
      if(lowest_virtual >= thr)
        throw std::runtime_error("neo_dump: lowest proton virtual is unbound -- proton self-interaction not removed?\n");
    }
    // Nonzero for more than one quantum proton: the SCF orbitals diagonalize the
    // J/K-dressed proton Fock, not this one. The consumer must semicanonicalize
    // whichever reference operator it builds -- it must not assume the dumped
    // orbitals are canonical.
    printf("  max|F_ia| (occ-vir coupling) %.2e\n", occvir);
    fflush(stdout);
  }

  // ---- write the file ----
  hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file < 0)
    throw std::runtime_error("neo_dump: could not create " + filename + "\n");

  // /meta attributes on the root group
  hid_t root = H5Gopen2(file, "/", H5P_DEFAULT);
  write_str_attr(root, "units", "Hartree atomic units");
  write_str_attr(root, "integral_convention", "chemist (pq|rs)");
  write_str_attr(root, "integral_representation", representation);
  write_str_attr(root, "ao_or_mo", "AO");
  write_str_attr(root, "storage_order", "C (row-major)");
  write_int_attr(root, "n_electrons", n_electrons);
  write_int_attr(root, "restricted_electrons", restricted_e ? 1 : 0);
  write_int_attr(root, "n_quantum_protons", n_protons);
  write_str_attr(root, "proton_spin_treatment", "high-spin");
  write_dbl_attr(root, "proton_mass", proton_mass);
  if(!dense) {
    write_int_attr(root, "naux_e", (int) Be.n_cols);
    write_int_attr(root, "naux_p", (int) Bp.n_cols);
  }
  // With a shared auxiliary/pivot space the two B factors carry a common vector
  // index, so eri_ep is a product of tensors already on disk and is not stored.
  write_int_attr(root, "shared_aux", shared_aux ? 1 : 0);
  write_dbl_attr(root, "e_scf", e_scf);
  write_dbl_attr(root, "e_classical", e_classical);
  write_str_attr(root, "erkale_version", version);

  // /electron
  hid_t eg = H5Gcreate2(file, "/electron", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  write_int(eg, "nbf", (int) Ne);
  write_mat(eg, "hcore", hcore_e);
  write_mat(eg, "overlap", S_e);
  if(restricted_e) {
    write_int(eg, "nmo", (int) Cc[0].n_cols);
    write_int(eg, "nocc", (int) nocc_e[0]);
    write_mat(eg, "C", Cc[0]);
    write_vec(eg, "occ", occc[0]);
  } else {
    const char * suf[2] = {"_a", "_b"};
    for(int s=0;s<2;s++) {
      write_int(eg, std::string("nmo")+suf[s], (int) Cc[s].n_cols);
      write_int(eg, std::string("nocc")+suf[s], (int) nocc_e[s]);
      write_mat(eg, std::string("C")+suf[s], Cc[s]);
      write_vec(eg, std::string("occ")+suf[s], occc[s]);
    }
  }
  if(!dense)
    write_B(eg, Be, Ne);

  // /proton
  hid_t pg = H5Gcreate2(file, "/proton", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  write_int(pg, "nbf", (int) Np);
  write_int(pg, "nmo", (int) Cp_o.n_cols);
  write_int(pg, "nocc", (int) nocc_p);
  write_mat(pg, "C", Cp_o);
  write_vec(pg, "occ", occp_o);
  write_mat(pg, "hcore", hcore_p);
  write_mat(pg, "overlap", S_p);
  if(!dense)
    write_B(pg, Bp, Np);

  // Electron-proton tensor, bare positive (mu nu | a b). Omitted when the two B
  // factors share a vector index, since it is then exactly B_e B_p^T and storing
  // it would cost Ne^2 Np^2 for nothing. Written dense otherwise -- including in
  // "dense" representation, where no B factor is on disk to reconstruct it from.
  if(dense || !shared_aux) {
    // C-order (Ne,Ne,Np,Np): row-major of Gep is the column-major buffer of Gep.t().
    arma::mat Gept = Gep.t();
    std::vector<hsize_t> dims = { (hsize_t) Ne, (hsize_t) Ne, (hsize_t) Np, (hsize_t) Np };
    write_buffer(file, "eri_ep", Gept.memptr(), dims);
  } else {
    printf("eri_ep omitted: shared auxiliary index, (mu nu|a b) = sum_P B_e[P,mu,nu] B_p[P,a,b].\n");
    fflush(stdout);
  }

  // dense per-species tensors. Gee/Gpp are symmetric in (pair<->pair), so their
  // column-major buffer equals the C-order rank-4 buffer.
  if(dense) {
    std::vector<hsize_t> edims = { (hsize_t) Ne, (hsize_t) Ne, (hsize_t) Ne, (hsize_t) Ne };
    write_buffer(file, "eri_ee", Gee.memptr(), edims);
    std::vector<hsize_t> pdims = { (hsize_t) Np, (hsize_t) Np, (hsize_t) Np, (hsize_t) Np };
    write_buffer(file, "eri_pp", Gpp.memptr(), pdims);
  }

  H5Gclose(pg);
  H5Gclose(eg);
  H5Gclose(root);
  H5Fclose(file);

  printf("NEO-SCF dump written.\n");
  fflush(stdout);

  // ---- verify: reconstruct the SCF energy from the dumped quantities ----
  if(do_verify) {
    // The densities come from the dumped occupied orbitals, so this checks the
    // whole contract: the file reproduces the SCF energy using only the core
    // Hamiltonians, the two-particle factors and the occupied orbital columns.
    // one-particle
    double E1e = arma::trace(De * hcore_e);
    double E1p = arma::trace(Dp * hcore_p);

    // e-e Coulomb and exchange (ERKALE conventions)
    double Ecoul_ee = 0.5 * arma::dot(de_vec, Gee * de_vec);
    double Eexch_ee = 0.0;
    if(restricted_e) {
      arma::mat Ke = calcK_from_G(Gee, De);
      Eexch_ee = -0.25 * arma::trace(Ke * De);
    } else {
      for(size_t s=0;s<2;s++) {
        arma::mat Ks = calcK_from_G(Gee, Dspin[s]);
        Eexch_ee += -0.5 * arma::trace(Ks * Dspin[s]);
      }
    }

    // p-p Coulomb and exchange (high-spin protons)
    double Ecoul_pp = 0.5 * arma::dot(dp_vec, Gpp * dp_vec);
    arma::mat Kp = calcK_from_G(Gpp, Dp);
    double Eexch_pp = -0.5 * arma::trace(Kp * Dp);

    // e-p coupling: attractive sign applied here (q_e*q_p = -1)
    double E_ep = - arma::dot(de_vec, Gep * dp_vec);

    double E_recon = E1e + E1p + Ecoul_ee + Eexch_ee + Ecoul_pp + Eexch_pp + E_ep + e_classical;
    double diff = E_recon - e_scf;

    printf("\nNEO dump self-consistency check:\n");
    printf("  e one-particle    % .10f\n", E1e);
    printf("  p one-particle    % .10f\n", E1p);
    printf("  e-e Coulomb       % .10f\n", Ecoul_ee);
    printf("  e-e exchange      % .10f\n", Eexch_ee);
    printf("  p-p Coulomb       % .10f\n", Ecoul_pp);
    printf("  p-p exchange      % .10f\n", Eexch_pp);
    printf("  e-p coupling      % .10f\n", E_ep);
    printf("  classical nuc.    % .10f\n", e_classical);
    printf("  reconstructed E   % .10f\n", E_recon);
    printf("  SCF E             % .10f\n", e_scf);
    printf("  difference        % .3e\n", diff);
    fflush(stdout);

    if(std::abs(diff) > 1e-7) {
      bool screened = !(omega == 0.0 && alpha == 1.0 && beta == 0.0);
      if(shared_aux && screened)
        // Finite-proton RI uses dfit's metric for the electron-side expansion and
        // pfit's (screened) 3-center for the proton-side projection; that mixed
        // metric is not exactly B_e B_p^T, so a small residual is expected here.
        printf("  NOTE: finite-proton density-fitting run -- the e-p RI uses a mixed\n"
               "  metric, so a small residual is expected. Use the Cholesky path for\n"
               "  a machine-precision check.\n");
      else {
        std::ostringstream oss;
        oss << "neo_dump verify: reconstructed energy differs from SCF by " << diff << " (> 1e-7)!\n";
        throw std::runtime_error(oss.str());
      }
    }
    fflush(stdout);
  }
}
