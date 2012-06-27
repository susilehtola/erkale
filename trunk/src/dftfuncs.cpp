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
#include "dftgrid.h"
#include "stringutil.h"

// LibXC
#include <xc.h>

/* List of functions defined in libxc

   generated with
   $ cat /usr/include/xc_funcs.h |awk '{n++; funcname=tolower($2); sub(/xc_/,"",funcname); printf("%s \"%s\"\n",$3,funcname) } END {print ""}' | sort -n | awk '{printf("{%s, %s}, ",$2,$1)}'
*/

static const func_t funcs_list[] = {
  {"none", 0},

  {"lda_x", 1}, {"lda_c_wigner", 2}, {"lda_c_rpa", 3}, {"lda_c_hl", 4}, {"lda_c_gl", 5}, {"lda_c_xalpha", 6}, {"lda_c_vwn", 7}, {"lda_c_vwn_rpa", 8}, {"lda_c_pz", 9}, {"lda_c_pz_mod", 10}, {"lda_c_ob_pz", 11}, {"lda_c_pw", 12}, {"lda_c_pw_mod", 13}, {"lda_c_ob_pw", 14}, {"lda_c_2d_amgb", 15}, {"lda_c_2d_prm", 16}, {"lda_c_vbh", 17}, {"lda_c_1d_csc", 18}, {"lda_x_2d", 19}, {"lda_xc_teter93", 20}, {"lda_x_1d", 21}, {"lda_c_ml1", 22}, {"lda_c_ml2", 23}, {"lda_c_gombas", 24}, {"lda_c_pw_rpa", 25}, {"lda_k_tf", 50}, {"lda_k_lp", 51}, {"gga_x_pbe", 101}, {"gga_x_pbe_r", 102}, {"gga_x_b86", 103}, {"gga_x_herman", 104}, {"gga_x_b86_mgc", 105}, {"gga_x_b88", 106}, {"gga_x_g96", 107}, {"gga_x_pw86", 108}, {"gga_x_pw91", 109}, {"gga_x_optx", 110}, {"gga_x_dk87_r1", 111}, {"gga_x_dk87_r2", 112}, {"gga_x_lg93", 113}, {"gga_x_ft97_a", 114}, {"gga_x_ft97_b", 115}, {"gga_x_pbe_sol", 116}, {"gga_x_rpbe", 117}, {"gga_x_wc", 118}, {"gga_x_mpw91", 119}, {"gga_x_am05", 120}, {"gga_x_pbea", 121}, {"gga_x_mpbe", 122}, {"gga_x_xpbe", 123}, {"gga_x_2d_b86_mgc", 124}, {"gga_x_bayesian", 125}, {"gga_x_pbe_jsjr", 126}, {"gga_x_2d_b88", 127}, {"gga_x_2d_b86", 128}, {"gga_x_2d_pbe", 129}, {"gga_c_pbe", 130}, {"gga_c_lyp", 131}, {"gga_c_p86", 132}, {"gga_c_pbe_sol", 133}, {"gga_c_pw91", 134}, {"gga_c_am05", 135}, {"gga_c_xpbe", 136}, {"gga_c_lm", 137}, {"gga_c_pbe_jrgx", 138}, {"gga_x_optb88_vdw", 139}, {"gga_x_pbek1_vdw", 140}, {"gga_x_optpbe_vdw", 141}, {"gga_x_rge2", 142}, {"gga_c_rge2", 143}, {"gga_x_rpw86", 144}, {"gga_x_kt1", 145}, {"gga_xc_kt2", 146}, {"gga_c_wl", 147}, {"gga_c_wi", 148}, {"gga_x_mb88", 149}, {"gga_x_sogga", 150}, {"gga_x_sogga11", 151}, {"gga_c_sogga11", 152}, {"gga_c_wi0", 153}, {"gga_xc_th1", 154}, {"gga_xc_th2", 155}, {"gga_xc_th3", 156}, {"gga_xc_th4", 157}, {"gga_x_c09x", 158}, {"gga_c_sogga11_x", 159}, {"gga_x_lb", 160}, {"gga_xc_hcth_93", 161}, {"gga_xc_hcth_120", 162}, {"gga_xc_hcth_147", 163}, {"gga_xc_hcth_407", 164}, {"gga_xc_edf1", 165}, {"gga_xc_xlyp", 166}, {"gga_xc_b97", 167}, {"gga_xc_b97_1", 168}, {"gga_xc_b97_2", 169}, {"gga_xc_b97_d", 170}, {"gga_xc_b97_k", 171}, {"gga_xc_b97_3", 172}, {"gga_xc_pbe1w", 173}, {"gga_xc_mpwlyp1w", 174}, {"gga_xc_pbelyp1w", 175}, {"gga_xc_sb98_1a", 176}, {"gga_xc_sb98_1b", 177}, {"gga_xc_sb98_1c", 178}, {"gga_xc_sb98_2a", 179}, {"gga_xc_sb98_2b", 180}, {"gga_xc_sb98_2c", 181}, {"gga_x_lbm", 182}, {"gga_x_ol2", 183}, {"gga_x_apbe", 184}, {"gga_k_apbe", 185}, {"gga_c_apbe", 186}, {"gga_k_tw1", 187}, {"gga_k_tw2", 188}, {"gga_k_tw3", 189}, {"gga_k_tw4", 190}, {"gga_x_htbs", 191}, {"gga_x_airy", 192}, {"gga_x_lag", 193}, {"gga_xc_mohlyp", 194}, {"gga_xc_mohlyp2", 195}, {"gga_xc_th_fl", 196}, {"gga_xc_th_fc", 197}, {"gga_xc_th_fcfo", 198}, {"gga_xc_th_fco", 199}, {"mgga_x_lta", 201}, {"mgga_x_tpss", 202}, {"mgga_x_m06l", 203}, {"mgga_x_gvt4", 204}, {"mgga_x_tau_hcth", 205}, {"mgga_x_br89", 206}, {"mgga_x_bj06", 207}, {"mgga_x_tb09", 208}, {"mgga_x_rpp09", 209}, {"mgga_x_2d_prhg07", 210}, {"mgga_x_2d_prhg07_prp10", 211}, {"mgga_c_tpss", 231}, {"mgga_c_vsxc", 232}, {"hyb_gga_xc_b3pw91", 401}, {"hyb_gga_xc_b3lyp", 402}, {"hyb_gga_xc_b3p86", 403}, {"hyb_gga_xc_o3lyp", 404}, {"hyb_gga_xc_mpw1k", 405}, {"hyb_gga_xc_pbeh", 406}, {"hyb_gga_xc_b97", 407}, {"hyb_gga_xc_b97_1", 408}, {"hyb_gga_xc_b97_2", 410}, {"hyb_gga_xc_x3lyp", 411}, {"hyb_gga_xc_b1wc", 412}, {"hyb_gga_xc_b97_k", 413}, {"hyb_gga_xc_b97_3", 414}, {"hyb_gga_xc_mpw3pw", 415}, {"hyb_gga_xc_b1lyp", 416}, {"hyb_gga_xc_b1pw91", 417}, {"hyb_gga_xc_mpw1pw", 418}, {"hyb_gga_xc_mpw3lyp", 419}, {"hyb_gga_xc_sb98_1a", 420}, {"hyb_gga_xc_sb98_1b", 421}, {"hyb_gga_xc_sb98_1c", 422}, {"hyb_gga_xc_sb98_2a", 423}, {"hyb_gga_xc_sb98_2b", 424}, {"hyb_gga_xc_sb98_2c", 425}, {"hyb_gga_x_sogga11_x", 426}, {"gga_k_vw", 500}, {"gga_k_ge2", 501}, {"gga_k_golden", 502}, {"gga_k_yt65", 503}, {"gga_k_baltin", 504}, {"gga_k_lieb", 505}, {"gga_k_absr1", 506}, {"gga_k_absr2", 507}, {"gga_k_gr", 508}, {"gga_k_ludena", 509}, {"gga_k_gp85", 510}, {"gga_k_pearson", 511}, {"gga_k_ol1", 512}, {"gga_k_ol2", 513}, {"gga_k_fr_b88", 514}, {"gga_k_fr_pw86", 515}, {"gga_k_dk", 516}, {"gga_k_perdew", 517}, {"gga_k_vsk", 518}, {"gga_k_vjks", 519}, {"gga_k_ernzerhof", 520}, {"gga_k_lc94", 521}, {"gga_k_llp", 522}, {"gga_k_thakkar", 523}

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
  // Was functional given as a number? If so, use it.
  if(isdigit(name[0]))
    return atoi(name.c_str());

  // Otherwise we need to search the list for the functional.
  size_t Nfuncs=sizeof(funcs_list)/sizeof(funcs_list[0]);
  
  // Find functional in list
  for(size_t i=0;i<Nfuncs;i++)
    if(stricmp(funcs_list[i].name,name)==0)
      return funcs_list[i].func_id;

  // If we are still here, functional was not found.
  std::ostringstream oss;
  oss << "\nError in dft.cpp: functional "<<name<<" not found!\n"; 
  throw std::runtime_error(oss.str());

  return 0;
}

void parse_xc_func(int & x_func, int & c_func, const std::string & xc) {
  // Default values - no functional used.
  x_func=0;
  c_func=0;

  // Check if there is a dash in the input.
  size_t dpos=xc.find('-',0);

  if(dpos!=std::string::npos) {
    // OK, there is a dash.

    // Exchange part is
    std::string x=xc.substr(0,dpos);
    // and correlation part is
    std::string c=xc.substr(dpos+1,xc.size()-dpos);

    // Functionals are
    x_func=find_func(x);
    c_func=find_func(c);
  } else {
    // No dash, so this should be an exchange-correlation functional.
    x_func=find_func(xc);
    c_func=0;
  }

  // Check functionals
  if(is_correlation(x_func) && !is_exchange_correlation(x_func)) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to use a correlation functional as exchange.\n");
  }
  if(is_exchange(c_func)) {
    ERROR_INFO();
    throw std::runtime_error("Refusing to use an exchange functional as correlation.\n");
  }

  // Sanity check: don't try to use kinetic energy functionals.
  if(is_kinetic(x_func)) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "The wanted functional "<< get_keyword(x_func) << " is a kinetic energy functional.\n";
    throw std::runtime_error(oss.str());
  }
  if(is_kinetic(c_func)) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "The wanted functional "<< get_keyword(c_func) << " is a kinetic energy functional.\n";
    throw std::runtime_error(oss.str());
  }

}

void print_info(int x_func, int c_func) {
  if(is_exchange_correlation(x_func)) {
    printf("Used exchange-correlation functional is %s, ",get_keyword(x_func).c_str());
    print_info(x_func);
  } else {
    if(is_exchange(x_func)) {
      printf("Used exchange functional is %s, ",get_keyword(x_func).c_str());
      print_info(x_func);
    } else
      printf("No exchange functional.\n");
    
    if(is_correlation(c_func)) {
      printf("\nUsed correlation functional is %s, ",get_keyword(c_func).c_str());
      print_info(c_func);
      printf("\n");
    } else
      printf("\nNo correlation functional.\n\n");
  }
}

void print_info(int func_id) {
  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0) {
      ERROR_INFO();
      std::ostringstream oss;
      oss << "\nFunctional "<<func_id<<" not found!\n";
      throw std::runtime_error(oss.str());
    }
    
    printf("'%s', defined in the reference(s):\n%s\n", func.info->name, func.info->refs);
    xc_func_end(&func);
  }
}

bool is_exchange(int func_id) {
  bool ans=0;
  
  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    
    switch(func.info->kind)
      {
	/* By default we assign exchange-correlation functionals as
	   the exchange functional, since we check whether the
	   exchange part includes exact exchange. */
      case XC_EXCHANGE:
      case XC_EXCHANGE_CORRELATION:
	ans=1;
	break;
      default:
	ans=0;
      }
    // Free functional
    xc_func_end(&func);
  } else
    // Dummy exchange
    ans=0;

  return ans;
}

bool is_exchange_correlation(int func_id) {
  bool ans=0;
  
  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    
    switch(func.info->kind)
      {
	/* By default we assign exchange-correlation functionals as
	   the exchange functional, since we check whether the
	   exchange part includes exact exchange. */
      case XC_EXCHANGE_CORRELATION:
	ans=1;
	break;
      default:
	ans=0;
      }
    // Free functional
    xc_func_end(&func);
  } else
    // Dummy exchange
    ans=0;

  return ans;
}

bool is_correlation(int func_id) {
  bool ans=0;
  
  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    
    switch(func.info->kind)
      {
	/* By default we assign exchange-correlation functionals as
	   the exchange functional, since we check whether the
	   exchange part includes exact exchange. */
      case XC_CORRELATION:
	ans=1;
	break;
      default:
	ans=0;
      }
    // Free functional
    xc_func_end(&func);
  } else
    // Dummy correlation
    ans=0;

  return ans;
}

bool is_kinetic(int func_id) {
  bool ans=0;
  
  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    
    switch(func.info->kind)
      {
	/* By default we assign exchange-correlation functionals as
	   the exchange functional, since we check whether the
	   exchange part includes exact exchange. */
      case XC_KINETIC:
	ans=1;
	break;
      default:
	ans=0;
      }
    // Free functional
    xc_func_end(&func);
  } else
    // Dummy correlation
    ans=0;

  return ans;
}

double exact_exchange(int func_id) {
  // Default - no exact exchange.
  double f=0.0;

  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }

    switch(func.info->family)
      {
      case XC_FAMILY_HYB_GGA:
	// TODO - FIXME
	//	f=xc_hyb_gga_exx_coef(&func);
	f=xc_hyb_gga_exx_coef(func.gga);
	break;
      }
    
    xc_func_end(&func);
  }

  //  printf("Fraction of exact exchange is %f.\n",f);

  return f;
}


bool gradient_needed(int func_id) {
  // Is gradient necessary?

  bool grad=0;

  if(func_id>0) {    
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!"; 
      throw std::runtime_error(oss.str());
    }
    
    switch(func.info->family)
      {
      case XC_FAMILY_GGA:
      case XC_FAMILY_HYB_GGA:
      case XC_FAMILY_MGGA:
	grad=1;
	break;
      }
    // Free functional
    xc_func_end(&func);
  }

  return grad;
}

bool laplacian_needed(int func_id) {
  // Is gradient necessary?

  bool lapl=0;

  if(func_id>0) {    
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!"; 
      throw std::runtime_error(oss.str());
    }
    
    switch(func.info->family)
      {
      case XC_FAMILY_MGGA:
	lapl=1;
	break;
      }
    // Free functional
    xc_func_end(&func);
  }

  return lapl;
}
