/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#ifndef ERKALE_CONTRIB_ELECTRONIC_XC
#define ERKALE_CONTRIB_ELECTRONIC_XC

#include "dftgrid.h"
#include "dftfuncs.h"
#include "scf.h"
#include "stringutil.h"
#include <armadillo>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Electronic exchange-correlation for the stand-alone SCF programs
 * (erkale_neo, erkale_complex_orbs), which build their own Fock
 * matrices. Parses the Method functional, gives its exact-exchange
 * admixture, refuses what these programs do not implement, builds a
 * fixed integration grid, and evaluates the XC energy and AO matrix.
 * The exact exchange is kfull K + kshort K_sr(omega), where K_sr is the
 * exchange with the short-range (erfc) interaction: a global hybrid has
 * only kfull, a range-separated one also kshort and omega. With
 * Hartree-Fock (Method HF) it is inactive: kfull is one and the XC terms
 * are zero.
 */
class ElectronicXC {
  /// Exchange and correlation functionals (0 for none)
  int x_func_=0, c_func_=0;
  /// Exact-exchange admixture: full-range and short-range fractions,
  /// and the range-separation parameter
  double kfull_=1.0, kshort_=0.0, omega_=0.0;
  /// Integration grid
  DFTGrid grid_;
  /// Is the grid constructed?
  bool have_grid_=false;

 public:
  /// Constructor
  ElectronicXC(const BasisSet & basis, bool verbose) : grid_(&basis, verbose) {}

  /**
   * Parse the functional and construct the grid (gridstr is nrad lmax or
   * a named grid). The grid is built when the functional needs it, or
   * when need_grid is set for other terms evaluated on it (e.g. the
   * electron-proton correlation in erkale_neo).
   */
  void setup(const std::string & method, const std::string & gridstr, bool need_grid=false) {
    x_func_=c_func_=0;
    if(stricmp(method,"HF")!=0)
      parse_xc_func(x_func_, c_func_, method);
    // Hartree-Fock has full exact exchange; a functional has its own
    // admixture, which is zero without an exchange functional.
    if(active())
      range_separation(x_func_, omega_, kfull_, kshort_);
    else {
      kfull_=1.0;
      kshort_=0.0;
      omega_=0.0;
    }

    double b, C;
    if((x_func_>0 && needs_VV10(x_func_, b, C)) || (c_func_>0 && needs_VV10(c_func_, b, C)))
      throw std::runtime_error("VV10 functionals are not supported in this program.\n");

    if(active() || need_grid) {
      if(stricmp(gridstr,"Auto")==0)
        throw std::runtime_error("Adaptive DFT grids are not supported in this program; give DFTGrid as nrad lmax.\n");
      dft_t griddft;
      parse_grid(griddft, gridstr, "DFT");
      grid_.construct(griddft.nrad, griddft.lmax, x_func_, c_func_);
      have_grid_=true;
    }
  }

  /// Is there an XC functional?
  bool active() const { return x_func_>0 || c_func_>0; }
  /// Full-range exact-exchange fraction
  double kfull() const { return kfull_; }
  /// Short-range exact-exchange fraction
  double kshort() const { return kshort_; }
  /// Range-separation parameter (zero without range separation)
  double omega() const { return omega_; }
  /// Does the functional depend on the kinetic energy density or the laplacian?
  bool is_meta_gga() const {
    for(int f : {x_func_, c_func_}) {
      if(f<=0)
        continue;
      bool gga, mgga_t, mgga_l;
      is_gga_mgga(f, gga, mgga_t, mgga_l);
      if(mgga_t || mgga_l)
        return true;
    }
    return false;
  }
  /// Integration grid (for other terms evaluated on it)
  DFTGrid & grid() {
    if(!have_grid_)
      throw std::logic_error("ElectronicXC grid has not been constructed.\n");
    return grid_;
  }

  /// Restricted: XC energy and matrix for the total density P
  double eval(const arma::mat & P, arma::mat & Vxc) {
    double Exc=0.0, Nel;
    if(active())
      grid_.eval_Fxc(x_func_, c_func_, P, Vxc, Exc, Nel);
    else
      Vxc.zeros(P.n_rows, P.n_cols);
    return Exc;
  }
  /// Unrestricted: XC energy and matrices for the spin densities
  double eval(const arma::mat & Pa, const arma::mat & Pb, arma::mat & Vxca, arma::mat & Vxcb) {
    double Exc=0.0, Nel;
    if(active())
      grid_.eval_Fxc(x_func_, c_func_, Pa, Pb, Vxca, Vxcb, Exc, Nel);
    else {
      Vxca.zeros(Pa.n_rows, Pa.n_cols);
      Vxcb.zeros(Pb.n_rows, Pb.n_cols);
    }
    return Exc;
  }

  /// Restricted: first-order change of the XC matrix at the density P
  /// for the symmetric perturbation D
  arma::mat response(const arma::mat & P, const arma::mat & D) {
    if(!active())
      return arma::zeros<arma::mat>(P.n_rows, P.n_cols);
    return grid_.eval_Kxc(x_func_, c_func_, P, {D})[0];
  }
  /// Unrestricted: first-order change of the XC matrices at (Pa, Pb)
  /// for the perturbation (Da, Db)
  void response(const arma::mat & Pa, const arma::mat & Pb, const arma::mat & Da, const arma::mat & Db, arma::mat & dVa, arma::mat & dVb) {
    if(!active()) {
      dVa.zeros(Pa.n_rows, Pa.n_cols);
      dVb.zeros(Pb.n_rows, Pb.n_cols);
      return;
    }
    std::vector<arma::mat> Ka, Kb;
    grid_.eval_Kxc(x_func_, c_func_, Pa, Pb, {Da}, {Db}, Ka, Kb);
    dVa = Ka[0];
    dVb = Kb[0];
  }
};

#endif
