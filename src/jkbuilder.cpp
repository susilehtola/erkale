/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "jkbuilder.h"
#include "settings.h"
#include "stringutil.h"
#include "basislibrary.h"
#include "dftfuncs.h"
#include "timer.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

/// Global settings object (defined in each executable's main).
extern Settings settings;

namespace {
  // CholeskyMode == -1: try to load the cached DF/CD integrals from
  // cholfile. Direct mode has no stored integrals, and a non-load
  // cholmode means "don't load" (we still may save). Returns true on a
  // successful load. The caller must have set range separation on dfit
  // beforehand if needed -- the load key depends on it.
  bool try_cache_load(DensityFit & dfit, const BasisSet & basis, const BasisSet * auxbas,
                      int cholmode, bool direct, const std::string & cholfile,
                      bool verbose, Timer & t) {
    if(direct || cholmode != -1) return false;
    if(verbose) t.set();
    bool loaded = dfit.load(basis, auxbas, cholfile);
    if(loaded && verbose) {
      printf("Loaded integrals from %s (%s).\n", cholfile.c_str(), t.elapsed().c_str());
      fflush(stdout);
    }
    return loaded;
  }

  // CholeskyMode == +1: save dfit to cholfile. Direct mode is skipped.
  void try_cache_save(const DensityFit & dfit, int cholmode, bool direct,
                      const std::string & cholfile, bool verbose) {
    if(direct || cholmode != 1) return;
    dfit.save(cholfile);
    if(verbose) {
      printf("Saved integrals to %s.\n", cholfile.c_str());
      fflush(stdout);
    }
  }

  /// Settings-derived configuration shared by every backend.
  struct JKConfig {
    bool direct=false, decfock=false, occ_rik=false;
    double intthr=1e-10, fitthr=1e-7, fitcholthr=1e-8, cholthr=1e-7, cholshthr=0.01;
    double screenthr=1e-10;
    int cholmode=0, fitlmaxinc=1;
    std::string cholfile="cholesky.chk", fittingbasis="def2-universal-jkfit";
  };

  /// Reconstruct a density matrix from occupied orbitals (for the four-index
  /// backends, whose exchange is density-based, when handed orbitals).
  template<typename T>
  T form_density(const T & C, const std::vector<double> & occ) {
    arma::Col<typename T::elem_type> o(arma::conv_to< arma::Col<typename T::elem_type> >::from(occ));
    return C * arma::diagmat(o) * C.t();
  }

  /// Four-index Coulomb + exact-exchange force gradient via the screener.
  /// ERItable has no force kernels, so both four-index backends route forces
  /// through an ERIscreen, ensured filled on the (contracted) orbital basis
  /// -- this also re-fills a decfock screener that was built on the
  /// decontracted basis. Restricted: J(+K) from the total density.
  arma::vec fourindex_force(ERIscreen & scr, ERIscreen & scr_rs, const BasisSet * basisp,
                            double intthr, bool verbose, const arma::mat & Ptot,
                            double kfull, double kshort, double omega, double tol) {
    if(scr.get_N() != basisp->get_Nbf())
      scr.fill(basisp, intthr, verbose);
    arma::vec f = (kfull != 0.0) ? scr.forceJK(Ptot, tol, kfull) : scr.forceJ(Ptot, tol);
    if(omega != 0.0) {
      scr_rs.set_range_separation({omega, 0.0, 1.0});
      if(scr_rs.get_N() != basisp->get_Nbf())
        scr_rs.fill(basisp, intthr, verbose);
      f += scr_rs.forceK(Ptot, tol, kshort);
    }
    return f;
  }
  /// Unrestricted: J from the total density; exact exchange from the spin
  /// channels.
  arma::vec fourindex_force(ERIscreen & scr, ERIscreen & scr_rs, const BasisSet * basisp,
                            double intthr, bool verbose, const arma::mat & Ptot,
                            const arma::mat & Pa, const arma::mat & Pb,
                            double kfull, double kshort, double omega, double tol) {
    if(scr.get_N() != basisp->get_Nbf())
      scr.fill(basisp, intthr, verbose);
    arma::vec f = (kfull != 0.0) ? scr.forceJK(Pa, Pb, tol, kfull) : scr.forceJ(Ptot, tol);
    if(omega != 0.0) {
      scr_rs.set_range_separation({omega, 0.0, 1.0});
      if(scr_rs.get_N() != basisp->get_Nbf())
        scr_rs.fill(basisp, intthr, verbose);
      f += scr_rs.forceK(Pa, Pb, tol, kshort);
    }
    return f;
  }
}

// --------------------------------------------------------------------------
// Polymorphic J/K backend: one subclass per integral engine. The build,
// range-separated build and (transitional) engine-accessor calls forward
// here from the JKBuilder facade; the method is chosen once, when the facade
// constructs the backend, so there is no per-call dispatch. These classes
// are defined only here (JKBackend is forward-declared in the header for the
// facade's unique_ptr); they are not part of any public interface.
// --------------------------------------------------------------------------
class JKBackend {
   protected:
    JKConfig cfg;
    bool verbose=true;
    const BasisSet * basisp=nullptr;

    [[noreturn]] static void wrong_engine(const char * eng) {
      throw std::runtime_error(std::string("JKBuilder: the configured method does not own a ")+eng+" engine.\n");
    }

   public:
    explicit JKBackend(const JKConfig & c) : cfg(c) {}
    virtual ~JKBackend() {}

    virtual void init(const BasisSet & basis, bool verb) = 0;
    virtual void init_rs(double omega) = 0;
    virtual void set_fitting(const BasisSet &) {}

    virtual bool is_densityfit() const { return false; }
    virtual bool is_cholesky() const { return false; }
    bool is_direct() const { return cfg.direct; }
    bool is_decfock() const { return cfg.decfock; }
    bool is_occ_rik() const { return cfg.occ_rik; }

    // Exact-exchange admixture, set once at fill time (set_range_separation):
    // K = kfull_*K_full + kshort_*K_short(omega_). Defaults to Hartree-Fock.
    double kfull_ = 1.0, kshort_ = 0.0, omega_ = 0.0;
    void set_range_separation(double kfull, double kshort, double omega) {
      kfull_ = kfull; kshort_ = kshort; omega_ = omega;
      if(omega_ != 0.0) init_rs(omega_);
    }
    bool has_exact_exchange() const { return kfull_ != 0.0 || kshort_ != 0.0; }

    // Coulomb + the combined exact exchange kfull_*K_full + kshort_*K_short(omega_).
    // J is always the full Coulomb operator. Builds the full-range J/K and the
    // short-range K through the per-backend primitives and combines them here,
    // so callers never see the full/short split. (Restricted / unrestricted,
    // real / complex.)
    void formJK(const arma::mat & Ptot, const arma::mat & C, const std::vector<double> & occ,
                const arma::mat & S, arma::mat & J, arma::mat & K) const {
      build_JKfull(Ptot, C, occ, S, kfull_ != 0.0, J, K);
      if(kfull_ != 0.0) K *= kfull_; else K.zeros(J.n_rows, J.n_cols);
      if(omega_ != 0.0) K += kshort_ * formKshort(Ptot, C, occ, S);
    }
    void formJK(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat & cC,
                const std::vector<double> & occ, const arma::mat & S, arma::mat & J, arma::cx_mat & K) const {
      build_JKfull(Ptot, cP, cC, occ, S, kfull_ != 0.0, J, K);
      if(kfull_ != 0.0) K *= kfull_; else K.zeros(J.n_rows, J.n_cols);
      if(omega_ != 0.0) K += kshort_ * formKshort(cP, cC, occ, S);
    }
    void formJK(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                const arma::mat & Ca, const arma::mat & Cb,
                const std::vector<double> & occa, const std::vector<double> & occb,
                const arma::mat & S, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const {
      build_JKfull(Ptot, Pa, Pb, Ca, Cb, occa, occb, S, kfull_ != 0.0, J, Ka, Kb);
      if(kfull_ != 0.0) { Ka *= kfull_; Kb *= kfull_; }
      else              { Ka.zeros(J.n_rows, J.n_cols); Kb.zeros(J.n_rows, J.n_cols); }
      if(omega_ != 0.0) {
        arma::mat Kas, Kbs;
        formKshort(Pa, Pb, Ca, Cb, occa, occb, S, Kas, Kbs);
        Ka += kshort_ * Kas; Kb += kshort_ * Kbs;
      }
    }
    void formJK(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
                const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                const std::vector<double> & occa, const std::vector<double> & occb,
                const arma::mat & S, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const {
      build_JKfull(Ptot, cPa, cPb, cCa, cCb, occa, occb, S, kfull_ != 0.0, J, Ka, Kb);
      if(kfull_ != 0.0) { Ka *= kfull_; Kb *= kfull_; }
      else              { Ka.zeros(J.n_rows, J.n_cols); Kb.zeros(J.n_rows, J.n_cols); }
      if(omega_ != 0.0) {
        arma::cx_mat Kas, Kbs;
        formKshort(cPa, cPb, cCa, cCb, occa, occb, S, Kas, Kbs);
        Ka += kshort_ * Kas; Kb += kshort_ * Kbs;
      }
    }

   protected:
    // Per-backend primitives: full-range J (+ raw full-range K when want_K),
    // and the raw short-range K. Combined by formJK above.
    virtual void build_JKfull(const arma::mat & Ptot, const arma::mat & C, const std::vector<double> & occ,
                              const arma::mat & S, bool want_K, arma::mat & J, arma::mat & K) const = 0;
    virtual void build_JKfull(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat & cC,
                              const std::vector<double> & occ, const arma::mat & S, bool want_K,
                              arma::mat & J, arma::cx_mat & K) const = 0;
    virtual void build_JKfull(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                              const arma::mat & Ca, const arma::mat & Cb,
                              const std::vector<double> & occa, const std::vector<double> & occb,
                              const arma::mat & S, bool want_K, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const = 0;
    virtual void build_JKfull(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
                              const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                              const std::vector<double> & occa, const std::vector<double> & occb,
                              const arma::mat & S, bool want_K, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const = 0;

    virtual arma::mat   formKshort(const arma::mat & P, const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const = 0;
    virtual arma::cx_mat formKshort(const arma::cx_mat & cP, const arma::cx_mat & cC, const std::vector<double> & occ, const arma::mat & S) const = 0;
    virtual void formKshort(const arma::mat & Pa, const arma::mat & Pb, const arma::mat & Ca, const arma::mat & Cb,
                            const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                            arma::mat & Ka, arma::mat & Kb) const = 0;
    virtual void formKshort(const arma::cx_mat & cPa, const arma::cx_mat & cPb, const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                            const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                            arma::cx_mat & Ka, arma::cx_mat & Kb) const = 0;
   public:

    virtual arma::mat   calcJ(const arma::mat & P) const = 0;
    virtual arma::mat   calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const = 0;
    virtual arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const = 0;
    virtual arma::mat   calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const = 0;
    virtual arma::cx_mat calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const = 0;

    // Coulomb + exact-exchange force gradient.
    virtual arma::vec formForce(const arma::mat & P, const arma::mat & C, const std::vector<double> & occ,
                                double kfull, double kshort, double omega, double tol) const = 0;
    virtual arma::vec formForce(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                                const arma::mat & Ca, const arma::mat & Cb,
                                const std::vector<double> & occa, const std::vector<double> & occb,
                                double kfull, double kshort, double omega, double tol) const = 0;

    // Transitional engine accessors (the backend that does not own the
    // requested engine throws).
    virtual ERItable & eritable() const { wrong_engine("in-core ERItable"); }
    virtual ERItable & eritable_rs() const { wrong_engine("in-core ERItable"); }
    virtual ERIscreen & eriscreen() const { wrong_engine("direct ERIscreen"); }
    virtual ERIscreen & eriscreen_rs() const { wrong_engine("direct ERIscreen"); }
    virtual DensityFit & densityfit() const { wrong_engine("density-fitting"); }
    virtual DensityFit & densityfit_rs() const { wrong_engine("density-fitting"); }
  };

  // ----------------------------------------------------------------------
  // In-core four-index ERIs (ERItable). Exchange is density-based.
  // ----------------------------------------------------------------------
  class InCoreJK : public JKBackend {
    mutable ERItable tab, tab_rs;
    /// Force-only screeners: ERItable has no force kernels, so gradients go
    /// through an ERIscreen, filled lazily on the first force evaluation.
    mutable ERIscreen fscr, fscr_rs;
   public:
    explicit InCoreJK(const JKConfig & c) : JKBackend(c) {}

    void init(const BasisSet & basis, bool verb) override {
      verbose=verb; basisp=&basis;
      Timer t;
      size_t N;
      if(verbose) {
        N=tab.N_ints(&basis,cfg.intthr);
        printf("Forming table of %u ERIs, requiring %s of memory ... ",(unsigned int) N,memory_size(N*sizeof(double)).c_str());
        fflush(stdout);
      }
      size_t Npairs=tab.fill(&basis,cfg.intthr);
      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
      }
    }
    void init_rs(double omega) override {
      const bool fill = !tab_rs.get_N() || tab_rs.get_range_separation().omega != omega;
      if(!fill) return;
      Timer t;
      if(verbose) { printf("Computing short-range repulsion integrals ... "); fflush(stdout); }
      tab_rs.set_range_separation({omega, 0.0, 1.0});
      size_t Np=tab_rs.fill(basisp,cfg.intthr);
      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        printf("%i short-range shell pairs are significant.\n",(int) Np);
        fflush(stdout);
      }
    }

    void build_JKfull(const arma::mat & Ptot, const arma::mat &, const std::vector<double> &,
                const arma::mat &, bool want_K, arma::mat & J, arma::mat & K) const override {
      J = tab.calcJ(Ptot);
      if(want_K) K = tab.calcK(Ptot);
    }
    void build_JKfull(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat &,
                const std::vector<double> &, const arma::mat &, bool want_K,
                arma::mat & J, arma::cx_mat & K) const override {
      J = tab.calcJ(Ptot);
      if(want_K) K = tab.calcK(cP);
    }
    void build_JKfull(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                const arma::mat &, const arma::mat &, const std::vector<double> &, const std::vector<double> &,
                const arma::mat &, bool want_K, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const override {
      J = tab.calcJ(Ptot);
      if(want_K) { Ka = tab.calcK(Pa); Kb = tab.calcK(Pb); }
    }
    void build_JKfull(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
                const arma::cx_mat &, const arma::cx_mat &, const std::vector<double> &, const std::vector<double> &,
                const arma::mat &, bool want_K, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const override {
      J = tab.calcJ(Ptot);
      if(want_K) { Ka = tab.calcK(cPa); Kb = tab.calcK(cPb); }
    }

    arma::mat   formKshort(const arma::mat & P, const arma::mat &, const std::vector<double> &, const arma::mat &) const override { return tab_rs.calcK(P); }
    arma::cx_mat formKshort(const arma::cx_mat & cP, const arma::cx_mat &, const std::vector<double> &, const arma::mat &) const override { return tab_rs.calcK(cP); }
    void formKshort(const arma::mat & Pa, const arma::mat & Pb, const arma::mat &, const arma::mat &,
                    const std::vector<double> &, const std::vector<double> &, const arma::mat &,
                    arma::mat & Ka, arma::mat & Kb) const override { Ka = tab_rs.calcK(Pa); Kb = tab_rs.calcK(Pb); }
    void formKshort(const arma::cx_mat & cPa, const arma::cx_mat & cPb, const arma::cx_mat &, const arma::cx_mat &,
                    const std::vector<double> &, const std::vector<double> &, const arma::mat &,
                    arma::cx_mat & Ka, arma::cx_mat & Kb) const override { Ka = tab_rs.calcK(cPa); Kb = tab_rs.calcK(cPb); }

    arma::mat   calcJ(const arma::mat & P) const override { return tab.calcJ(P); }
    arma::mat   calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat &) const override { return tab.calcK(form_density(C,occ)); }
    arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat &) const override { return tab.calcK(form_density(C,occ)); }
    arma::mat   calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat &) const override { return tab_rs.calcK(form_density(C,occ)); }
    arma::cx_mat calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat &) const override { return tab_rs.calcK(form_density(C,occ)); }

    arma::vec formForce(const arma::mat & P, const arma::mat &, const std::vector<double> &,
                        double kfull, double kshort, double omega, double tol) const override {
      return fourindex_force(fscr, fscr_rs, basisp, cfg.intthr, verbose, P, kfull, kshort, omega, tol);
    }
    arma::vec formForce(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                        const arma::mat &, const arma::mat &, const std::vector<double> &, const std::vector<double> &,
                        double kfull, double kshort, double omega, double tol) const override {
      return fourindex_force(fscr, fscr_rs, basisp, cfg.intthr, verbose, Ptot, Pa, Pb, kfull, kshort, omega, tol);
    }

    ERItable & eritable() const override { return tab; }
    ERItable & eritable_rs() const override { return tab_rs; }
  };

  // ----------------------------------------------------------------------
  // Direct four-index ERIs (ERIscreen). Single-pass calcJK; owns decfock.
  // ----------------------------------------------------------------------
  class DirectJK : public JKBackend {
    mutable ERIscreen scr, scr_rs;
    BasisSet decbas;       ///< decontracted basis (decfock)
    arma::mat decconv;     ///< contracted -> decontracted conversion

    arma::mat    decontract(const arma::mat & P) const { return decconv*P*arma::trans(decconv); }
    arma::cx_mat decontract(const arma::cx_mat & P) const { return decconv*P*arma::trans(decconv); }
    arma::mat    recontract(const arma::mat & M) const { return arma::trans(decconv)*M*decconv; }
    arma::cx_mat recontract(const arma::cx_mat & M) const { return arma::trans(decconv)*M*decconv; }

    arma::mat directJ(const arma::mat & P) const {
      return cfg.decfock ? recontract(scr.calcJ(decontract(P), cfg.intthr)) : scr.calcJ(P, cfg.intthr);
    }
    template<typename Td, typename Tk>
    void directJK(const Td & P, arma::mat & J, Tk & K) const {
      if(cfg.decfock) {
        arma::mat Jd; Tk Kd;
        scr.calcJK(decontract(P), Jd, Kd, cfg.intthr);
        J = recontract(Jd); K = recontract(Kd);
      } else
        scr.calcJK(P, J, K, cfg.intthr);
    }
    template<typename Td, typename Tk>
    void directJK(const Td & Pa, const Td & Pb, arma::mat & J, Tk & Ka, Tk & Kb) const {
      if(cfg.decfock) {
        arma::mat Jd; Tk Kad, Kbd;
        scr.calcJK(decontract(Pa), decontract(Pb), Jd, Kad, Kbd, cfg.intthr);
        J = recontract(Jd); Ka = recontract(Kad); Kb = recontract(Kbd);
      } else
        scr.calcJK(Pa, Pb, J, Ka, Kb, cfg.intthr);
    }
    template<typename T>
    T directKshort(const T & P) const {
      return cfg.decfock ? recontract(scr_rs.calcK(decontract(P), cfg.intthr)) : scr_rs.calcK(P, cfg.intthr);
    }
    template<typename T>
    void directKshort(const T & Pa, const T & Pb, T & Ka, T & Kb) const {
      if(cfg.decfock) {
        T Kad, Kbd;
        scr_rs.calcK(decontract(Pa), decontract(Pb), Kad, Kbd, cfg.intthr);
        Ka = recontract(Kad); Kb = recontract(Kbd);
      } else
        scr_rs.calcK(Pa, Pb, Ka, Kb, cfg.intthr);
    }

   public:
    explicit DirectJK(const JKConfig & c) : JKBackend(c) {}

    void init(const BasisSet & basis, bool verb) override {
      verbose=verb; basisp=&basis;
      scr.set_screen_thresh(cfg.screenthr);
      scr_rs.set_screen_thresh(cfg.screenthr);
      Timer t;
      size_t Npairs;
      if(verbose) { t.set(); printf("Forming ERI screening matrix ... "); fflush(stdout); }
      if(cfg.decfock) {
        decbas=basis.decontract(decconv);
        Npairs=scr.fill(&decbas,cfg.intthr,verbose);
      } else {
        Npairs=scr.fill(&basis,cfg.intthr,verbose);
      }
      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        const BasisSet & b = cfg.decfock ? decbas : basis;
        printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) b.get_unique_shellpairs().size());
      }
    }
    void init_rs(double omega) override {
      const bool fill = !scr_rs.get_N() || scr_rs.get_range_separation().omega != omega;
      if(!fill) return;
      Timer t;
      if(verbose) { printf("Computing short-range repulsion integrals ... "); fflush(stdout); }
      scr_rs.set_range_separation({omega, 0.0, 1.0});
      size_t Np=scr_rs.fill(basisp,cfg.intthr);
      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        printf("%i short-range shell pairs are significant.\n",(int) Np);
        fflush(stdout);
      }
    }

    void build_JKfull(const arma::mat & Ptot, const arma::mat &, const std::vector<double> &,
                const arma::mat &, bool want_K, arma::mat & J, arma::mat & K) const override {
      if(want_K) directJK(Ptot, J, K); else J = directJ(Ptot);
    }
    void build_JKfull(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat &,
                const std::vector<double> &, const arma::mat &, bool want_K,
                arma::mat & J, arma::cx_mat & K) const override {
      if(want_K) directJK(cP, J, K); else J = directJ(Ptot);
    }
    void build_JKfull(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                const arma::mat &, const arma::mat &, const std::vector<double> &, const std::vector<double> &,
                const arma::mat &, bool want_K, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const override {
      if(want_K) directJK(Pa, Pb, J, Ka, Kb); else J = directJ(Ptot);
    }
    void build_JKfull(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
                const arma::cx_mat &, const arma::cx_mat &, const std::vector<double> &, const std::vector<double> &,
                const arma::mat &, bool want_K, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const override {
      if(want_K) directJK(cPa, cPb, J, Ka, Kb); else J = directJ(Ptot);
    }

    arma::mat   formKshort(const arma::mat & P, const arma::mat &, const std::vector<double> &, const arma::mat &) const override { return directKshort(P); }
    arma::cx_mat formKshort(const arma::cx_mat & cP, const arma::cx_mat &, const std::vector<double> &, const arma::mat &) const override { return directKshort(cP); }
    void formKshort(const arma::mat & Pa, const arma::mat & Pb, const arma::mat &, const arma::mat &,
                    const std::vector<double> &, const std::vector<double> &, const arma::mat &,
                    arma::mat & Ka, arma::mat & Kb) const override { directKshort(Pa, Pb, Ka, Kb); }
    void formKshort(const arma::cx_mat & cPa, const arma::cx_mat & cPb, const arma::cx_mat &, const arma::cx_mat &,
                    const std::vector<double> &, const std::vector<double> &, const arma::mat &,
                    arma::cx_mat & Ka, arma::cx_mat & Kb) const override { directKshort(cPa, cPb, Ka, Kb); }

    arma::mat   calcJ(const arma::mat & P) const override { return directJ(P); }
    arma::mat   calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat &) const override {
      arma::mat K; const arma::mat P(form_density(C,occ));
      if(cfg.decfock) K=recontract(scr.calcK(decontract(P), cfg.intthr)); else K=scr.calcK(P, cfg.intthr);
      return K;
    }
    arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat &) const override {
      arma::cx_mat K; const arma::cx_mat P(form_density(C,occ));
      if(cfg.decfock) K=recontract(scr.calcK(decontract(P), cfg.intthr)); else K=scr.calcK(P, cfg.intthr);
      return K;
    }
    arma::mat   calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat &) const override { return directKshort(form_density(C,occ)); }
    arma::cx_mat calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat &) const override { return directKshort(form_density(C,occ)); }

    arma::vec formForce(const arma::mat & P, const arma::mat &, const std::vector<double> &,
                        double kfull, double kshort, double omega, double tol) const override {
      return fourindex_force(scr, scr_rs, basisp, cfg.intthr, verbose, P, kfull, kshort, omega, tol);
    }
    arma::vec formForce(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                        const arma::mat &, const arma::mat &, const std::vector<double> &, const std::vector<double> &,
                        double kfull, double kshort, double omega, double tol) const override {
      return fourindex_force(scr, scr_rs, basisp, cfg.intthr, verbose, Ptot, Pa, Pb, kfull, kshort, omega, tol);
    }

    ERIscreen & eriscreen() const override { return scr; }
    ERIscreen & eriscreen_rs() const override { return scr_rs; }
  };

  // ----------------------------------------------------------------------
  // Density fitting / Cholesky (DensityFit). Covers RI, two-step Cholesky
  // and CDFit; exchange is orbital-based (occ-RI-K when requested).
  // ----------------------------------------------------------------------
  class DensityFitJK : public JKBackend {
    JKBuilder::Method method;
    mutable DensityFit dfit, dfit_rs;
    BasisSet dfitbas;
    bool have_override=false;

    arma::mat   k_orb(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const {
      return cfg.occ_rik ? dfit.calcK_occ(C, occ, S) : dfit.calcK(C, occ);
    }
    arma::cx_mat k_orb(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const {
      return cfg.occ_rik ? dfit.calcK_occ(C, occ, S) : dfit.calcK(C, occ);
    }
    arma::mat   k_orb_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const {
      return cfg.occ_rik ? dfit_rs.calcK_occ(C, occ, S) : dfit_rs.calcK(C, occ);
    }
    arma::cx_mat k_orb_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const {
      return cfg.occ_rik ? dfit_rs.calcK_occ(C, occ, S) : dfit_rs.calcK(C, occ);
    }

   public:
    DensityFitJK(const JKConfig & c, JKBuilder::Method m) : JKBackend(c), method(m) {}

    bool is_densityfit() const override { return true; }
    bool is_cholesky() const override { return dfit.is_cholesky(); }
    void set_fitting(const BasisSet & fitbas) override { dfitbas=fitbas; have_override=true; }

    void init(const BasisSet & basis, bool verb) override {
      verbose=verb; basisp=&basis;
      Timer t;

      if(method==JKBuilder::Method::CholeskyTwoStep) {
        // Two-step CD: L vectors held by DensityFit with metric (a|b)=I.
        if(!try_cache_load(dfit, basis, nullptr, cfg.cholmode, cfg.direct, cfg.cholfile, verbose, t)) {
          if(verbose) { t.set(); printf("Computing repulsion integrals (two-step CD).\n"); fflush(stdout); }
          size_t Npairs = dfit.fill_cholesky(basis, cfg.direct, cfg.cholthr, cfg.cholshthr, cfg.intthr, cfg.fitcholthr, verbose);
          if(verbose) { printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size()); fflush(stdout); }
          try_cache_save(dfit, cfg.cholmode, cfg.direct, cfg.cholfile, verbose);
        }
        return;
      }

      // RI or CDFit: build a Gaussian / CD-derived auxiliary basis (unless an
      // explicit one was set via set_fitting), then fit on it.
      if(!have_override) {
        if(method==JKBuilder::Method::CDFit) {
          if(verbose) { t.set(); printf("Building auxiliary basis from pivoted Cholesky decomposition (CDFit) ... "); fflush(stdout); }
          dfitbas = basis.cholesky_aux_basis(cfg.cholthr, cfg.fitlmaxinc);
          if(verbose) { printf("done (%s)\n",t.elapsed().c_str()); printf("Auxiliary basis contains %i functions.\n",(int) dfitbas.get_Nbf()); fflush(stdout); }
        } else {
          // RI: resolve the FittingBasis keyword.
          bool rik=false;
          if(stricmp(settings.get_string("Method"),"HF")==0 || stricmp(settings.get_string("Method"),"ROHF")==0)
            rik=true;
          else {
            int xfunc, cfunc;
            parse_xc_func(xfunc,cfunc,settings.get_string("Method"));
            if(exact_exchange(xfunc)!=0.0) rik=true;
          }
          if(stricmp(cfg.fittingbasis.c_str(),"Auto")==0) {
            // CD-derived auto-aux (Lehtola 2021/2023): spans the orbital
            // products, so valid for exact exchange too.
            dfitbas=basis.cholesky_aux_basis(cfg.cholthr, cfg.fitlmaxinc);
          } else if(stricmp(cfg.fittingbasis.c_str(),"AutoABS")==0) {
            // Eichkorn-style automatic aux. J-only.
            if(rik)
              throw std::runtime_error("FittingBasis AutoABS is not implemented for exact exchange.\nUse Auto (CD-derived) or set an explicit FittingBasis.\n");
            dfitbas=basis.density_fitting();
          } else {
            BasisSetLibrary fitlib;
            fitlib.load_basis(cfg.fittingbasis);
            bool uselm=settings.get_bool("UseLM");
            settings.set_bool("UseLM",true);
            construct_basis(dfitbas,basis.get_nuclei(),fitlib);
            dfitbas.coulomb_normalize();
            settings.set_bool("UseLM",uselm);
          }
        }
      }

      if(!try_cache_load(dfit, basis, &dfitbas, cfg.cholmode, cfg.direct, cfg.cholfile, verbose, t)) {
        std::string memest=memory_size(dfit.memory_estimate(basis,dfitbas,cfg.intthr,cfg.direct));
        if(verbose) {
          if(cfg.direct) printf("Initializing density fitting calculation, requiring %s memory ... ",memest.c_str());
          else printf("Computing density fitting integrals, requiring %s memory ... ",memest.c_str());
          fflush(stdout); t.set();
        }
        size_t Npairs=dfit.fill(basis,dfitbas,cfg.direct,cfg.intthr,cfg.fitthr,cfg.fitcholthr);
        if(verbose) {
          printf("done (%s)\n",t.elapsed().c_str());
          printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
          printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
          fflush(stdout);
        }
        try_cache_save(dfit, cfg.cholmode, cfg.direct, cfg.cholfile, verbose);
      } else if(verbose) {
        printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
        fflush(stdout);
      }
    }

    void init_rs(double omega) override {
      const bool fill = !dfit_rs.get_Naux() || dfit_rs.get_range_separation().omega != omega;
      if(!fill) return;
      dfit_rs.set_range_separation({omega, 0.0, 1.0});
      const bool is_cd = dfit.is_cholesky();
      Timer t;
      if(!try_cache_load(dfit_rs, *basisp, is_cd ? nullptr : &dfitbas, cfg.cholmode, cfg.direct, cfg.cholfile, verbose, t)) {
        if(is_cd) {
          if(verbose) { printf("Computing short-range repulsion integrals (two-step CD).\n"); fflush(stdout); }
          t.set();
          size_t Npairs = dfit_rs.fill_cholesky(*basisp, cfg.direct, cfg.cholthr, cfg.cholshthr, cfg.intthr, cfg.fitcholthr, verbose);
          if(verbose) { printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basisp->get_unique_shellpairs().size()); fflush(stdout); }
        } else {
          std::string memest=memory_size(dfit.memory_estimate(*basisp,dfitbas,cfg.intthr,cfg.direct));
          if(verbose) {
            if(cfg.direct) printf("Initializing short-range density fitting calculation, requiring %s memory ... ",memest.c_str());
            else printf("Computing short-range density fitting integrals, requiring %s memory ... ",memest.c_str());
            fflush(stdout);
          }
          t.set();
          size_t Npairs=dfit_rs.fill(*basisp,dfitbas,cfg.direct,cfg.intthr,cfg.fitthr,cfg.fitcholthr);
          if(verbose) {
            printf("done (%s)\n",t.elapsed().c_str());
            printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basisp->get_unique_shellpairs().size());
            printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
            fflush(stdout);
          }
        }
        try_cache_save(dfit_rs, cfg.cholmode, cfg.direct, cfg.cholfile, verbose);
      }
    }

    void build_JKfull(const arma::mat & Ptot, const arma::mat & C, const std::vector<double> & occ,
                const arma::mat & S, bool want_K, arma::mat & J, arma::mat & K) const override {
      J = dfit.calcJ(Ptot);
      if(want_K) K = k_orb(C, occ, S);
    }
    void build_JKfull(const arma::mat & Ptot, const arma::cx_mat &, const arma::cx_mat & cC,
                const std::vector<double> & occ, const arma::mat & S, bool want_K,
                arma::mat & J, arma::cx_mat & K) const override {
      J = dfit.calcJ(Ptot);
      if(want_K) K = k_orb(cC, occ, S);
    }
    void build_JKfull(const arma::mat & Ptot, const arma::mat &, const arma::mat &,
                const arma::mat & Ca, const arma::mat & Cb, const std::vector<double> & occa, const std::vector<double> & occb,
                const arma::mat & S, bool want_K, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const override {
      J = dfit.calcJ(Ptot);
      if(want_K) { Ka = k_orb(Ca, occa, S); Kb = k_orb(Cb, occb, S); }
    }
    void build_JKfull(const arma::mat & Ptot, const arma::cx_mat &, const arma::cx_mat &,
                const arma::cx_mat & cCa, const arma::cx_mat & cCb, const std::vector<double> & occa, const std::vector<double> & occb,
                const arma::mat & S, bool want_K, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const override {
      J = dfit.calcJ(Ptot);
      if(want_K) { Ka = k_orb(cCa, occa, S); Kb = k_orb(cCb, occb, S); }
    }

    arma::mat   formKshort(const arma::mat &, const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const override { return k_orb_short(C, occ, S); }
    arma::cx_mat formKshort(const arma::cx_mat &, const arma::cx_mat & cC, const std::vector<double> & occ, const arma::mat & S) const override { return k_orb_short(cC, occ, S); }
    void formKshort(const arma::mat &, const arma::mat &, const arma::mat & Ca, const arma::mat & Cb,
                    const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                    arma::mat & Ka, arma::mat & Kb) const override { Ka = k_orb_short(Ca, occa, S); Kb = k_orb_short(Cb, occb, S); }
    void formKshort(const arma::cx_mat &, const arma::cx_mat &, const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                    const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                    arma::cx_mat & Ka, arma::cx_mat & Kb) const override { Ka = k_orb_short(cCa, occa, S); Kb = k_orb_short(cCb, occb, S); }

    arma::mat   calcJ(const arma::mat & P) const override { return dfit.calcJ(P); }
    arma::mat   calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const override { return k_orb(C, occ, S); }
    arma::cx_mat calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const override { return k_orb(C, occ, S); }
    arma::mat   calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const override { return k_orb_short(C, occ, S); }
    arma::cx_mat calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const override { return k_orb_short(C, occ, S); }

    // DF/CD gradients: algebraic Coulomb (forceJ / forceJ_cholesky) plus the
    // scaled exchange gradient. The short-range dfit_rs is filled by the SCF
    // (init_rs) before forces, as in the unscreened case.
    arma::vec formForce(const arma::mat & P, const arma::mat & C, const std::vector<double> & occ,
                        double kfull, double kshort, double omega, double /*tol*/) const override {
      arma::vec f = dfit.is_cholesky() ? dfit.forceJ_cholesky(*basisp, P) : dfit.forceJ(P);
      if(kfull != 0.0) f += dfit.forceK(*basisp, C, occ, kfull);
      if(omega != 0.0) f += dfit_rs.forceK(*basisp, C, occ, kshort);
      return f;
    }
    arma::vec formForce(const arma::mat & Ptot, const arma::mat &, const arma::mat &,
                        const arma::mat & Ca, const arma::mat & Cb,
                        const std::vector<double> & occa, const std::vector<double> & occb,
                        double kfull, double kshort, double omega, double /*tol*/) const override {
      arma::vec f = dfit.is_cholesky() ? dfit.forceJ_cholesky(*basisp, Ptot) : dfit.forceJ(Ptot);
      if(kfull != 0.0) { f += dfit.forceK(*basisp, Ca, occa, kfull); f += dfit.forceK(*basisp, Cb, occb, kfull); }
      if(omega != 0.0) { f += dfit_rs.forceK(*basisp, Ca, occa, kshort); f += dfit_rs.forceK(*basisp, Cb, occb, kshort); }
      return f;
    }

    DensityFit & densityfit() const override { return dfit; }
    DensityFit & densityfit_rs() const override { return dfit_rs; }
  };

// --------------------------------------------------------------------------
// JKBuilder facade.
// --------------------------------------------------------------------------

JKBuilder::JKBuilder() : method(Method::CholeskyTwoStep), impl(nullptr) {
}

JKBuilder::~JKBuilder() {
}

JKBuilder::Method JKBuilder::resolve_method(const Settings & set) {
  const std::string m = set.get_string("JKMethod");
  if(stricmp(m,"4index")==0 || stricmp(m,"Exact")==0)
    return Method::FourIndex;
  if(stricmp(m,"RI")==0 || stricmp(m,"DF")==0 || stricmp(m,"DensityFitting")==0)
    return Method::DensityFitting;
  if(stricmp(m,"Cholesky")==0 || stricmp(m,"TwoStep")==0)
    return Method::CholeskyTwoStep;
  if(stricmp(m,"CDFit")==0)
    return Method::CDFit;
  std::ostringstream oss;
  oss << "Unknown JKMethod '" << m << "'; expected 4index, RI, Cholesky or CDFit.\n";
  throw std::runtime_error(oss.str());
}

std::string JKBuilder::method_name(Method m) {
  switch(m) {
  case Method::FourIndex:       return "four-index";
  case Method::DensityFitting:  return "density fitting";
  case Method::CholeskyTwoStep: return "two-step Cholesky";
  case Method::CDFit:           return "CD-fit density fitting";
  }
  return "unknown";
}

void JKBuilder::configure(const Settings & set) {
  method = resolve_method(set);

  JKConfig cfg;
  cfg.direct      = set.get_bool("Direct");
  cfg.decfock     = set.get_bool("DecFock");
  cfg.occ_rik     = set.get_bool("OccRIK");
  if(cfg.occ_rik && method==Method::FourIndex)
    throw std::runtime_error("OccRIK (occ-RI-K) requires a density-fitting method (JKMethod RI, Cholesky or CDFit).\n");
  cfg.intthr      = set.get_double("IntegralThresh");
  cfg.fitthr      = set.get_double("FittingThreshold");
  cfg.fitcholthr  = set.get_double("FittingCholeskyThreshold");
  cfg.cholthr     = set.get_double("CholeskyThr");
  cfg.cholshthr   = set.get_double("CholeskyShThr");
  cfg.screenthr   = set.get_double("ScreeningThresh");
  cfg.cholmode    = set.get_int("CholeskyMode");
  cfg.cholfile    = set.get_string("CholeskyFile");
  cfg.fittingbasis= set.get_string("FittingBasis");
  cfg.fitlmaxinc  = set.get_int("FittingLmaxInc");

  // Build the backend for the resolved method (still empty; init fills it).
  switch(method) {
  case Method::FourIndex:
    if(cfg.direct) impl.reset(new DirectJK(cfg));
    else           impl.reset(new InCoreJK(cfg));
    break;
  default:
    impl.reset(new DensityFitJK(cfg, method));
    break;
  }
}

void JKBuilder::set_fitting(const BasisSet & fitbas) { impl->set_fitting(fitbas); }
void JKBuilder::init(const BasisSet & basis, bool verbose) { impl->init(basis, verbose); }
void JKBuilder::init_rs(double omega) { impl->init_rs(omega); }

bool JKBuilder::is_densityfit() const { return impl->is_densityfit(); }
bool JKBuilder::is_cholesky() const { return impl->is_cholesky(); }
bool JKBuilder::is_direct() const { return impl->is_direct(); }
bool JKBuilder::is_decfock() const { return impl->is_decfock(); }
bool JKBuilder::is_occ_rik() const { return impl->is_occ_rik(); }

arma::mat JKBuilder::calcJ(const arma::mat & P) const { return impl->calcJ(P); }
arma::mat JKBuilder::calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const { return impl->calcK(C, occ, S); }
arma::cx_mat JKBuilder::calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const { return impl->calcK(C, occ, S); }
arma::mat JKBuilder::calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const { return impl->calcK_short(C, occ, S); }
arma::cx_mat JKBuilder::calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const { return impl->calcK_short(C, occ, S); }

// The exact-exchange admixture and the full/short combination live in the
// backend (set at fill time); the facade just forwards.
void JKBuilder::set_range_separation(double kfull, double kshort, double omega) {
  impl->set_range_separation(kfull, kshort, omega);
}
bool JKBuilder::has_exact_exchange() const { return impl->has_exact_exchange(); }

void JKBuilder::formJK(const arma::mat & Ptot, const arma::mat & C, const std::vector<double> & occ,
                       const arma::mat & S, arma::mat & J, arma::mat & K) const {
  impl->formJK(Ptot, C, occ, S, J, K);
}
void JKBuilder::formJK(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat & cC,
                       const std::vector<double> & occ, const arma::mat & S,
                       arma::mat & J, arma::cx_mat & K) const {
  impl->formJK(Ptot, cP, cC, occ, S, J, K);
}
void JKBuilder::formJK(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                       const arma::mat & Ca, const arma::mat & Cb,
                       const std::vector<double> & occa, const std::vector<double> & occb,
                       const arma::mat & S, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const {
  impl->formJK(Ptot, Pa, Pb, Ca, Cb, occa, occb, S, J, Ka, Kb);
}
void JKBuilder::formJK(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
                       const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                       const std::vector<double> & occa, const std::vector<double> & occb,
                       const arma::mat & S, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const {
  impl->formJK(Ptot, cPa, cPb, cCa, cCb, occa, occb, S, J, Ka, Kb);
}

arma::vec JKBuilder::formForce(const arma::mat & P, const arma::mat & C, const std::vector<double> & occ,
                               double kfull, double kshort, double omega, double tol) const {
  return impl->formForce(P, C, occ, kfull, kshort, omega, tol);
}
arma::vec JKBuilder::formForce(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                               const arma::mat & Ca, const arma::mat & Cb,
                               const std::vector<double> & occa, const std::vector<double> & occb,
                               double kfull, double kshort, double omega, double tol) const {
  return impl->formForce(Ptot, Pa, Pb, Ca, Cb, occa, occb, kfull, kshort, omega, tol);
}

ERItable & JKBuilder::eritable() const { return impl->eritable(); }
ERItable & JKBuilder::eritable_rs() const { return impl->eritable_rs(); }
ERIscreen & JKBuilder::eriscreen() const { return impl->eriscreen(); }
ERIscreen & JKBuilder::eriscreen_rs() const { return impl->eriscreen_rs(); }
DensityFit & JKBuilder::densityfit() const { return impl->densityfit(); }
DensityFit & JKBuilder::densityfit_rs() const { return impl->densityfit_rs(); }
