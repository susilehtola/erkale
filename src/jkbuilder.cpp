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
}

JKBuilder::JKBuilder() :
  method(Method::CholeskyTwoStep), direct(false), decfock(false), occ_rik(false),
  verbose(true), intthr(1e-10), fitthr(1e-7), fitcholthr(1e-8), cholthr(1e-7),
  cholshthr(0.01), cholmode(0), cholfile("cholesky.chk"),
  fittingbasis("def2-universal-jkfit"), fitlmaxinc(1), basisp(nullptr) {
}

JKBuilder::~JKBuilder() {
}

bool JKBuilder::uses_dfit() const {
  return method != Method::FourIndex;
}

arma::mat JKBuilder::calcJ(const arma::mat & P) const {
  return dfit.calcJ(P);
}

arma::mat JKBuilder::calcK(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const {
  return occ_rik ? dfit.calcK_occ(C, occ, S) : dfit.calcK(C, occ);
}
arma::cx_mat JKBuilder::calcK(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const {
  return occ_rik ? dfit.calcK_occ(C, occ, S) : dfit.calcK(C, occ);
}
arma::mat JKBuilder::calcK_short(const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const {
  return occ_rik ? dfit_rs.calcK_occ(C, occ, S) : dfit_rs.calcK(C, occ);
}
arma::cx_mat JKBuilder::calcK_short(const arma::cx_mat & C, const std::vector<double> & occ, const arma::mat & S) const {
  return occ_rik ? dfit_rs.calcK_occ(C, occ, S) : dfit_rs.calcK(C, occ);
}

// decfock helpers: the conversion matrix decconv maps contracted ->
// decontracted, so a density transforms as D P D^T and a built matrix
// transforms back as D^T M D.
arma::mat JKBuilder::decontract(const arma::mat & P) const {
  return decconv*P*arma::trans(decconv);
}
arma::cx_mat JKBuilder::decontract(const arma::cx_mat & P) const {
  return decconv*P*arma::trans(decconv);
}
arma::mat JKBuilder::recontract(const arma::mat & M) const {
  return arma::trans(decconv)*M*decconv;
}
arma::cx_mat JKBuilder::recontract(const arma::cx_mat & M) const {
  return arma::trans(decconv)*M*decconv;
}

// --- Unified full-range J/K build (restricted) -------------------------

void JKBuilder::formJK(const arma::mat & Ptot, const arma::mat & C, const std::vector<double> & occ,
                       const arma::mat & S, bool want_K, arma::mat & J, arma::mat & K) const {
  if(uses_dfit()) {
    J = dfit.calcJ(Ptot);
    if(want_K)
      K = calcK(C, occ, S);
  } else if(direct) {
    if(want_K) {
      if(decfock) {
        arma::mat Jd, Kd;
        scr.calcJK(decontract(Ptot), Jd, Kd, intthr);
        J = recontract(Jd);
        K = recontract(Kd);
      } else
        scr.calcJK(Ptot, J, K, intthr);
    } else {
      J = decfock ? recontract(scr.calcJ(decontract(Ptot), intthr)) : scr.calcJ(Ptot, intthr);
    }
  } else {
    J = tab.calcJ(Ptot);
    if(want_K)
      K = tab.calcK(Ptot);
  }
}

void JKBuilder::formJK(const arma::mat & Ptot, const arma::cx_mat & cP, const arma::cx_mat & cC,
                       const std::vector<double> & occ, const arma::mat & S, bool want_K,
                       arma::mat & J, arma::cx_mat & K) const {
  if(uses_dfit()) {
    J = dfit.calcJ(Ptot);
    if(want_K)
      K = calcK(cC, occ, S);
  } else if(direct) {
    if(want_K) {
      if(decfock) {
        arma::mat Jd;
        arma::cx_mat Kd;
        scr.calcJK(decontract(cP), Jd, Kd, intthr);
        J = recontract(Jd);
        K = recontract(Kd);
      } else
        scr.calcJK(cP, J, K, intthr);
    } else {
      // J only: built from the real total density.
      J = decfock ? recontract(scr.calcJ(decontract(Ptot), intthr)) : scr.calcJ(Ptot, intthr);
    }
  } else {
    J = tab.calcJ(Ptot);
    if(want_K)
      K = tab.calcK(cP);
  }
}

// --- Unified full-range J/K build (unrestricted) -----------------------

void JKBuilder::formJK(const arma::mat & Ptot, const arma::mat & Pa, const arma::mat & Pb,
                       const arma::mat & Ca, const arma::mat & Cb,
                       const std::vector<double> & occa, const std::vector<double> & occb,
                       const arma::mat & S, bool want_K, arma::mat & J, arma::mat & Ka, arma::mat & Kb) const {
  if(uses_dfit()) {
    J = dfit.calcJ(Ptot);
    if(want_K) {
      Ka = calcK(Ca, occa, S);
      Kb = calcK(Cb, occb, S);
    }
  } else if(direct) {
    if(want_K) {
      if(decfock) {
        arma::mat Jd, Kad, Kbd;
        scr.calcJK(decontract(Pa), decontract(Pb), Jd, Kad, Kbd, intthr);
        J = recontract(Jd);
        Ka = recontract(Kad);
        Kb = recontract(Kbd);
      } else
        scr.calcJK(Pa, Pb, J, Ka, Kb, intthr);
    } else {
      J = decfock ? recontract(scr.calcJ(decontract(Ptot), intthr)) : scr.calcJ(Ptot, intthr);
    }
  } else {
    J = tab.calcJ(Ptot);
    if(want_K) {
      Ka = tab.calcK(Pa);
      Kb = tab.calcK(Pb);
    }
  }
}

void JKBuilder::formJK(const arma::mat & Ptot, const arma::cx_mat & cPa, const arma::cx_mat & cPb,
                       const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                       const std::vector<double> & occa, const std::vector<double> & occb,
                       const arma::mat & S, bool want_K, arma::mat & J, arma::cx_mat & Ka, arma::cx_mat & Kb) const {
  if(uses_dfit()) {
    J = dfit.calcJ(Ptot);
    if(want_K) {
      Ka = calcK(cCa, occa, S);
      Kb = calcK(cCb, occb, S);
    }
  } else if(direct) {
    if(want_K) {
      if(decfock) {
        arma::mat Jd;
        arma::cx_mat Kad, Kbd;
        scr.calcJK(decontract(cPa), decontract(cPb), Jd, Kad, Kbd, intthr);
        J = recontract(Jd);
        Ka = recontract(Kad);
        Kb = recontract(Kbd);
      } else
        scr.calcJK(cPa, cPb, J, Ka, Kb, intthr);
    } else {
      J = decfock ? recontract(scr.calcJ(decontract(Ptot), intthr)) : scr.calcJ(Ptot, intthr);
    }
  } else {
    J = tab.calcJ(Ptot);
    if(want_K) {
      Ka = tab.calcK(cPa);
      Kb = tab.calcK(cPb);
    }
  }
}

// --- Unified short-range exchange (range separation) -------------------

arma::mat JKBuilder::formKshort(const arma::mat & P, const arma::mat & C, const std::vector<double> & occ, const arma::mat & S) const {
  if(uses_dfit())
    return calcK_short(C, occ, S);
  else if(direct)
    return decfock ? recontract(scr_rs.calcK(decontract(P), intthr)) : scr_rs.calcK(P, intthr);
  else
    return tab_rs.calcK(P);
}

arma::cx_mat JKBuilder::formKshort(const arma::cx_mat & cP, const arma::cx_mat & cC, const std::vector<double> & occ, const arma::mat & S) const {
  if(uses_dfit())
    return calcK_short(cC, occ, S);
  else if(direct)
    return decfock ? recontract(scr_rs.calcK(decontract(cP), intthr)) : scr_rs.calcK(cP, intthr);
  else
    return tab_rs.calcK(cP);
}

void JKBuilder::formKshort(const arma::mat & Pa, const arma::mat & Pb, const arma::mat & Ca, const arma::mat & Cb,
                           const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                           arma::mat & Ka, arma::mat & Kb) const {
  if(uses_dfit()) {
    Ka = calcK_short(Ca, occa, S);
    Kb = calcK_short(Cb, occb, S);
  } else if(direct) {
    if(decfock) {
      arma::mat Kad, Kbd;
      scr_rs.calcK(decontract(Pa), decontract(Pb), Kad, Kbd, intthr);
      Ka = recontract(Kad);
      Kb = recontract(Kbd);
    } else
      scr_rs.calcK(Pa, Pb, Ka, Kb, intthr);
  } else {
    Ka = tab_rs.calcK(Pa);
    Kb = tab_rs.calcK(Pb);
  }
}

void JKBuilder::formKshort(const arma::cx_mat & cPa, const arma::cx_mat & cPb, const arma::cx_mat & cCa, const arma::cx_mat & cCb,
                           const std::vector<double> & occa, const std::vector<double> & occb, const arma::mat & S,
                           arma::cx_mat & Ka, arma::cx_mat & Kb) const {
  if(uses_dfit()) {
    Ka = calcK_short(cCa, occa, S);
    Kb = calcK_short(cCb, occb, S);
  } else if(direct) {
    if(decfock) {
      arma::cx_mat Kad, Kbd;
      scr_rs.calcK(decontract(cPa), decontract(cPb), Kad, Kbd, intthr);
      Ka = recontract(Kad);
      Kb = recontract(Kbd);
    } else
      scr_rs.calcK(cPa, cPb, Ka, Kb, intthr);
  } else {
    Ka = tab_rs.calcK(cPa);
    Kb = tab_rs.calcK(cPb);
  }
}

bool JKBuilder::is_cholesky() const {
  return dfit.is_cholesky();
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
  method      = resolve_method(set);
  direct      = set.get_bool("Direct");
  decfock     = set.get_bool("DecFock");
  occ_rik     = set.get_bool("OccRIK");
  if(occ_rik && method==Method::FourIndex)
    throw std::runtime_error("OccRIK (occ-RI-K) requires a density-fitting method (JKMethod RI, Cholesky or CDFit).\n");

  intthr      = set.get_double("IntegralThresh");
  fitthr      = set.get_double("FittingThreshold");
  fitcholthr  = set.get_double("FittingCholeskyThreshold");
  cholthr     = set.get_double("CholeskyThr");
  cholshthr   = set.get_double("CholeskyShThr");
  cholmode    = set.get_int("CholeskyMode");
  cholfile    = set.get_string("CholeskyFile");
  fittingbasis= set.get_string("FittingBasis");
  fitlmaxinc  = set.get_int("FittingLmaxInc");
}

void JKBuilder::set_fitting(const BasisSet & fitbas) {
  dfitbas = fitbas;
}

void JKBuilder::init(const BasisSet & basis, bool verb) {
  verbose = verb;
  basisp  = &basis;
  Timer t;

  if(method==Method::DensityFitting) {
    // Form density fitting basis.

    // Do we need RI-K, or is RI-J sufficient?
    bool rik=false;
    if(stricmp(settings.get_string("Method"),"HF")==0)
      rik=true;
    else if(stricmp(settings.get_string("Method"),"ROHF")==0)
      rik=true;
    else {
      // No Hartree-Fock; check if functional has exact exchange part
      int xfunc, cfunc;
      parse_xc_func(xfunc,cfunc,settings.get_string("Method"));
      if(exact_exchange(xfunc)!=0.0)
        rik=true;
    }

    if(stricmp(fittingbasis.c_str(),"Auto")==0) {
      // Default automatic aux basis: the per-atom CD-derived basis
      // (Lehtola JCTC 17, 6886 (2021)), uncontracted and lmax-pruned
      // (JCTC 19, 6242 (2023)) via FittingLmaxInc. Unlike the Eichkorn
      // construction it spans the orbital products, so it is valid for
      // exact exchange as well -- no RI-K restriction.
      dfitbas=basis.cholesky_aux_basis(cholthr, fitlmaxinc);
    } else if(stricmp(fittingbasis.c_str(),"AutoABS")==0) {
      // Eichkorn-style automatic auxiliary basis. J-only: not built for
      // exact exchange.
      if(rik)
        throw std::runtime_error("FittingBasis AutoABS is not implemented for exact exchange.\nUse Auto (CD-derived) or set an explicit FittingBasis.\n");
      dfitbas=basis.density_fitting();
    } else {
      // Load basis library
      BasisSetLibrary fitlib;
      fitlib.load_basis(fittingbasis);

      // Construct fitting basis
      bool uselm=settings.get_bool("UseLM");
      settings.set_bool("UseLM",true);
      construct_basis(dfitbas,basis.get_nuclei(),fitlib);
      dfitbas.coulomb_normalize();
      settings.set_bool("UseLM",uselm);
    }

    if(!try_cache_load(dfit, basis, &dfitbas, cholmode, direct, cholfile, verbose, t)) {
      std::string memest=memory_size(dfit.memory_estimate(basis,dfitbas,intthr,direct));
      if(verbose) {
        if(direct)
          printf("Initializing density fitting calculation, requiring %s memory ... ",memest.c_str());
        else
          printf("Computing density fitting integrals, requiring %s memory ... ",memest.c_str());
        fflush(stdout);
        t.set();
      }
      size_t Npairs=dfit.fill(basis,dfitbas,direct,intthr,fitthr,fitcholthr);
      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
        printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
        fflush(stdout);
      }
      try_cache_save(dfit, cholmode, direct, cholfile, verbose);
    } else if(verbose) {
      printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
      fflush(stdout);
    }

  } else if(method==Method::CDFit) {
    // Density fitting with an atom-centered aux basis built by per-atom
    // pivoted Cholesky on the orbital primitives (Lehtola JCTC 17, 6886
    // (2021)). Routes through the same DensityFit machinery as a regular
    // RI run; only valid when the orbital basis is rich enough for the
    // per-atom CD pivots to span the molecular ERI tensor faithfully.
    if(verbose) {
      t.set();
      printf("Building auxiliary basis from pivoted Cholesky decomposition (CDFit) ... ");
      fflush(stdout);
    }
    dfitbas = basis.cholesky_aux_basis(cholthr, fitlmaxinc);
    if(verbose) {
      printf("done (%s)\n",t.elapsed().c_str());
      printf("Auxiliary basis contains %i functions.\n",(int) dfitbas.get_Nbf());
      fflush(stdout);
    }

    // Drive density fitting on the CD-derived aux basis.
    if(!try_cache_load(dfit, basis, &dfitbas, cholmode, direct, cholfile, verbose, t)) {
      std::string memest=memory_size(dfit.memory_estimate(basis,dfitbas,intthr,direct));
      if(verbose) {
        if(direct)
          printf("Initializing density fitting calculation, requiring %s memory ... ",memest.c_str());
        else
          printf("Computing density fitting integrals, requiring %s memory ... ",memest.c_str());
        fflush(stdout);
        t.set();
      }
      size_t Npairs=dfit.fill(basis,dfitbas,direct,intthr,fitthr,fitcholthr);
      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
        fflush(stdout);
      }
      try_cache_save(dfit, cholmode, direct, cholfile, verbose);
    }

  } else if(method==Method::CholeskyTwoStep) {
    // TwoStep CD: routed through DensityFit on the merged path. L
    // vectors are orthonormal by construction; DensityFit holds them in
    // block-shellpair CachedBlocks layout with metric (a|b) = I so the
    // same J/K kernels handle CD and DF transparently. The pivot metric
    // is stashed for algebraic forceJ.
    if(!try_cache_load(dfit, basis, /*auxbas*/nullptr, cholmode, direct, cholfile, verbose, t)) {
      if(verbose) {
        t.set();
        printf("Computing repulsion integrals (two-step CD).\n");
        fflush(stdout);
      }
      size_t Npairs = dfit.fill_cholesky(basis, direct, cholthr, cholshthr, intthr, fitcholthr, verbose);
      if(verbose) {
        printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
        fflush(stdout);
      }
      try_cache_save(dfit, cholmode, direct, cholfile, verbose);
    }

  } else {
    // Exact four-index ERIs.
    if(direct) {
      size_t Npairs;
      // Form decontracted basis set and get the screening matrix
      if(verbose) {
        t.set();
        printf("Forming ERI screening matrix ... ");
        fflush(stdout);
      }

      if(decfock) {
        // Use decontracted basis
        decbas=basis.decontract(decconv);
        Npairs=scr.fill(&decbas,intthr,verbose);
      } else {
        // Use contracted basis
        Npairs=scr.fill(&basis,intthr,verbose);
      }

      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        if(decfock)
          printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) decbas.get_unique_shellpairs().size());
        else
          printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
      }

    } else {
      // Compute memory requirement
      size_t N;

      if(verbose) {
        N=tab.N_ints(&basis,intthr);
        printf("Forming table of %u ERIs, requiring %s of memory ... ",(unsigned int) N,memory_size(N*sizeof(double)).c_str());
        fflush(stdout);
      }
      // Don't compute small integrals
      size_t Npairs=tab.fill(&basis,intthr);

      if(verbose) {
        printf("done (%s)\n",t.elapsed().c_str());
        printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basis.get_unique_shellpairs().size());
      }
    }
  }
}

void JKBuilder::init_rs(double omega) {
  if(uses_dfit()) {
    // Compute range separated integrals if necessary
    const bool fill = !dfit_rs.get_Naux() || dfit_rs.get_range_separation().omega != omega;
    if(fill) {
      // dfit_rs.set_range_separation must happen before any load() --
      // the cache key is built from (omega, alpha, beta) so the load
      // only matches an entry stored with the exact same parameters
      // (see df_cache_prefix() in density_fitting.cpp). The Nbf / Naux
      // checks still apply on top of that.
      dfit_rs.set_range_separation({omega, 0.0, 1.0});

      const bool is_cd = dfit.is_cholesky();
      Timer t;
      if(!try_cache_load(dfit_rs, *basisp, is_cd ? nullptr : &dfitbas, cholmode, direct, cholfile, verbose, t)) {
        if(is_cd) {
          // Two-step CD on the merged path: build a fresh short-range B
          // tensor via DensityFit's CD fill. No aux basis is in play
          // (the L vectors are their own thing).
          if(verbose) {
            printf("Computing short-range repulsion integrals (two-step CD).\n");
            fflush(stdout);
          }
          t.set();
          size_t Npairs = dfit_rs.fill_cholesky(*basisp, direct, cholthr, cholshthr, intthr, fitcholthr, verbose);
          if(verbose) {
            printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basisp->get_unique_shellpairs().size());
            fflush(stdout);
          }
        } else {
          // Real density fitting: same aux basis as the unscreened dfit,
          // just with the range-separation kernel.
          std::string memest=memory_size(dfit.memory_estimate(*basisp,dfitbas,intthr,direct));
          if(verbose) {
            if(direct)
              printf("Initializing short-range density fitting calculation, requiring %s memory ... ",memest.c_str());
            else
              printf("Computing short-range density fitting integrals, requiring %s memory ... ",memest.c_str());
            fflush(stdout);
          }
          t.set();
          size_t Npairs=dfit_rs.fill(*basisp,dfitbas,direct,intthr,fitthr,fitcholthr);
          if(verbose) {
            printf("done (%s)\n",t.elapsed().c_str());
            printf("%i shell pairs out of %i are significant.\n",(int) Npairs, (int) basisp->get_unique_shellpairs().size());
            printf("Auxiliary basis contains %i functions.\n",(int) dfit.get_Naux());
            fflush(stdout);
          }
        }
        try_cache_save(dfit_rs, cholmode, direct, cholfile, verbose);
      }
    }

  } else {
    if(!direct) {
      // Compute range separated integrals if necessary
      const bool fill = !tab_rs.get_N() || tab_rs.get_range_separation().omega != omega;
      if(fill) {
        Timer t;
        if(verbose) {
          printf("Computing short-range repulsion integrals ... ");
          fflush(stdout);
        }
        tab_rs.set_range_separation({omega, 0.0, 1.0});
        size_t Np=tab_rs.fill(basisp,intthr);

        if(verbose) {
          printf("done (%s)\n",t.elapsed().c_str());
          printf("%i short-range shell pairs are significant.\n",(int) Np);
          fflush(stdout);
        }
      }
    } else {
      const bool fill = !scr_rs.get_N() || scr_rs.get_range_separation().omega != omega;
      if(fill) {
        Timer t;
        if(verbose) {
          printf("Computing short-range repulsion integrals ... ");
          fflush(stdout);
        }

        scr_rs.set_range_separation({omega, 0.0, 1.0});
        size_t Np=scr_rs.fill(basisp,intthr);

        if(verbose) {
          printf("done (%s)\n",t.elapsed().c_str());
          printf("%i short-range shell pairs are significant.\n",(int) Np);
          fflush(stdout);
        }
      }
    }
  }
}
