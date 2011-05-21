/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#include <string>
#include <sstream>
#include <stdexcept>
#include "dftfuncs.h"
#include "stringutil.h"

/* List of functions defined in libxc

   generated with
   $ cat /usr/include/xc_funcs.h |awk '{n++; funcname=tolower($2); sub(/xc_/,"",funcname); printf("{\"%s\", %s}, ",funcname,$3) } END {print ""}'
*/

static const func_t funcs_list[] = {
  {"none", 0},

  {"lda_x", 1}, {"lda_c_wigner", 2}, {"lda_c_rpa", 3}, {"lda_c_hl", 4}, {"lda_c_gl", 5}, {"lda_c_xalpha", 6}, {"lda_c_vwn", 7}, {"lda_c_vwn_rpa", 8}, {"lda_c_pz", 9}, {"lda_c_pz_mod", 10}, {"lda_c_ob_pz", 11}, {"lda_c_pw", 12}, {"lda_c_pw_mod", 13}, {"lda_c_ob_pw", 14}, {"lda_c_2d_amgb", 15}, {"lda_c_2d_prm", 16}, {"lda_c_vbh", 17}, {"lda_c_1d_csc", 18}, {"lda_x_2d", 19}, {"lda_xc_teter93", 20}, {"lda_x_1d", 21}, {"lda_c_ml1", 22}, {"lda_c_ml2", 23}, {"lda_c_gombas", 24}, {"lda_k_tf", 50}, {"lda_k_lp", 51}, {"gga_x_pbe", 101}, {"gga_x_pbe_r", 102}, {"gga_x_b86", 103}, {"gga_x_herman", 104}, {"gga_x_b86_mgc", 105}, {"gga_x_b88", 106}, {"gga_x_g96", 107}, {"gga_x_pw86", 108}, {"gga_x_pw91", 109}, {"gga_x_optx", 110}, {"gga_x_dk87_r1", 111}, {"gga_x_dk87_r2", 112}, {"gga_x_lg93", 113}, {"gga_x_ft97_a", 114}, {"gga_x_ft97_b", 115}, {"gga_x_pbe_sol", 116}, {"gga_x_rpbe", 117}, {"gga_x_wc", 118}, {"gga_x_mpw91", 119}, {"gga_x_am05", 120}, {"gga_x_pbea", 121}, {"gga_x_mpbe", 122}, {"gga_x_xpbe", 123}, {"gga_x_2d_b86_mgc", 124}, {"gga_x_bayesian", 125}, {"gga_x_pbe_jsjr", 126}, {"gga_x_2d_b88", 127}, {"gga_x_2d_b86", 128}, {"gga_x_2d_pbe", 129}, {"gga_c_pbe", 130}, {"gga_c_lyp", 131}, {"gga_c_p86", 132}, {"gga_c_pbe_sol", 133}, {"gga_c_pw91", 134}, {"gga_c_am05", 135}, {"gga_c_xpbe", 136}, {"gga_c_lm", 137}, {"gga_c_pbe_jrgx", 138}, {"gga_x_optb88_vdw", 139}, {"gga_x_pbek1_vdw", 140}, {"gga_x_optpbe_vdw", 141}, {"gga_x_rge2", 142}, {"gga_c_rge2", 143}, {"gga_x_rpw86", 144}, {"gga_x_kt1", 145}, {"gga_xc_kt2", 146}, {"gga_c_wl", 147}, {"gga_c_wi", 148}, {"gga_x_lb", 160}, {"gga_xc_hcth_93", 161}, {"gga_xc_hcth_120", 162}, {"gga_xc_hcth_147", 163}, {"gga_xc_hcth_407", 164}, {"gga_xc_edf1", 165}, {"gga_xc_xlyp", 166}, {"gga_xc_b97", 167}, {"gga_xc_b97_1", 168}, {"gga_xc_b97_2", 169}, {"gga_xc_b97_d", 170}, {"gga_xc_b97_k", 171}, {"gga_xc_b97_3", 172}, {"gga_xc_pbe1w", 173}, {"gga_xc_mpwlyp1w", 174}, {"gga_xc_pbelyp1w", 175}, {"gga_xc_sb98_1a", 176}, {"gga_xc_sb98_1b", 177}, {"gga_xc_sb98_1c", 178}, {"gga_xc_sb98_2a", 179}, {"gga_xc_sb98_2b", 180}, {"gga_xc_sb98_2c", 181}, {"gga_x_lbm", 182}, {"gga_k_vw", 500}, {"gga_k_ge2", 501}, {"gga_k_golden", 502}, {"gga_k_yt65", 503}, {"gga_k_baltin", 504}, {"gga_k_lieb", 505}, {"gga_k_absr1", 506}, {"gga_k_absr2", 507}, {"gga_k_gr", 508}, {"gga_k_ludena", 509}, {"gga_k_gp85", 510}, {"gga_k_pearson", 511}, {"gga_k_ol1", 512}, {"gga_k_ol2", 513}, {"gga_k_fr_b88", 514}, {"gga_k_fr_pw86", 515}, {"hyb_gga_xc_b3pw91", 401}, {"hyb_gga_xc_b3lyp", 402}, {"hyb_gga_xc_b3p86", 403}, {"hyb_gga_xc_o3lyp", 404}, {"hyb_gga_xc_mpw1k", 405}, {"hyb_gga_xc_pbeh", 406}, {"hyb_gga_xc_b97", 407}, {"hyb_gga_xc_b97_1", 408}, {"hyb_gga_xc_b97_2", 410}, {"hyb_gga_xc_x3lyp", 411}, {"hyb_gga_xc_b1wc", 412}, {"hyb_gga_xc_b97_k", 413}, {"hyb_gga_xc_b97_3", 414}, {"hyb_gga_xc_mpw3pw", 415}, {"hyb_gga_xc_b1lyp", 416}, {"hyb_gga_xc_b1pw91", 417}, {"hyb_gga_xc_mpw1pw", 418}, {"hyb_gga_xc_mpw3lyp", 419}, {"hyb_gga_xc_sb98_1a", 420}, {"hyb_gga_xc_sb98_1b", 421}, {"hyb_gga_xc_sb98_1c", 422}, {"hyb_gga_xc_sb98_2a", 423}, {"hyb_gga_xc_sb98_2b", 424}, {"hyb_gga_xc_sb98_2c", 425}, {"mgga_x_lta", 201}, {"mgga_x_tpss", 202}, {"mgga_x_m06l", 203}, {"mgga_x_gvt4", 204}, {"mgga_x_tau_hcth", 205}, {"mgga_x_br89", 206}, {"mgga_x_bj06", 207}, {"mgga_x_tb09", 208}, {"mgga_x_rpp09", 209}, {"mgga_x_2d_prhg07", 210}, {"mgga_x_2d_prhg07_prp10", 211}, {"mgga_c_tpss", 231}, {"mgga_c_vsxc", 232}, {"lca_omc", 301}, {"lca_lch", 302}


};

// Print keyword corresponding to functional.
std::string get_keyword(int func_id) {

  // Determine number of functionals
  size_t Nfuncs=sizeof(funcs_list)/sizeof(funcs_list[0]);

  for(size_t i=0;i<Nfuncs;i++)
    if(funcs_list[i].func_id==func_id)
      return funcs_list[i].name;

  ERROR_INFO();
  std::ostringstream oss;
  oss << "\nError in dft.cpp: "<<func_id<<" is not a valid functional number!\n"; 
  throw std::runtime_error(oss.str());
}
 

// Find out ID of functional
int find_func(std::string name) {

  // Convert string to lower case
  name=tolower(name);

  // Was functional given as a number? If so, use it.
  if(isdigit(name[0]))
    return atoi(name.c_str());


  // Otherwise we need to search the list for the functional.
  size_t Nfuncs=sizeof(funcs_list)/sizeof(funcs_list[0]);
  
  // Find functional in list
  for(size_t i=0;i<Nfuncs;i++)
    if(funcs_list[i].name==name)
      return funcs_list[i].func_id;

  // If we are still here, functional was not found.
  std::ostringstream oss;
  oss << "\nError in dft.cpp: functional "<<name<<" not found!\n"; 
  throw std::runtime_error(oss.str());

  return 0;
}
