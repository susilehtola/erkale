/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
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
extern "C" {
#include <xc.h>
}

// Print keyword corresponding to functional.
std::string get_keyword(int func_id) {
  // Check if none was specified. This is internal to ERKALE.
  if(func_id==0)
    return "none";

  // Call libxc function
  char *keyword=XC(functional_get_name)(func_id);

  if(keyword==NULL) {
    // Functional id not recognized.
    std::ostringstream oss;
    oss << "\nError: libxc did not recognize functional id "<<func_id<<"!\n";
    throw std::runtime_error(oss.str());
  }

  // Get the keyword
  std::string key(keyword);
  // and free memory allocated by libxc
  free(keyword);

  return key;
}


// Find out ID of functional
int find_func(std::string name) {
  // Was functional given as a number? If so, use it.
  if(isdigit(name[0]))
    return atoi(name.c_str());

  // Check if 'none' was specified. This is internal to ERKALE
  if(stricmp(name,"none")==0)
    return 0;

  // Otherwise, call libxc function.
  char help[strlen(name.c_str())+1];
  strcpy(help,name.c_str());
  int funcid=XC(functional_get_number)(help);

  // If libxc returned -1, the functional was not found.
  if(funcid==-1) {
    std::ostringstream oss;
    oss << "\nError: libxc did not recognize functional "<<name<<"!\n";
    throw std::runtime_error(oss.str());
  } else
    return funcid;

  // Dummy return clause
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

  /*
  // Check functionals
  if(is_correlation(x_func) && !is_exchange_correlation(x_func)) {
    ERROR_INFO();
    print_info(x_func,0);
    throw std::runtime_error("Refusing to use a correlation functional as exchange.\n");
  }
  if(is_exchange(c_func)) {
    ERROR_INFO();
    print_info(c_func,0);
    throw std::runtime_error("Refusing to use an exchange functional as correlation.\n");
  }
  */

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

#if XC_MAJOR_VERSION < 3
    printf("'%s', defined in the reference(s):\n%s\n", func.info->name, func.info->refs);
#else
    printf("'%s', defined in the reference(s):\n", func.info->name);
    for(int i=0;i<5;i++)
      if(func.info->refs[i]!=NULL)
	printf("%s (DOI %s)\n",func.info->refs[i]->ref,func.info->refs[i]->doi);
#endif
    xc_func_end(&func);
  }
  if(!has_exc(func_id)) {
    printf("The functional doesn't have an energy density, so the calculated energy is incorrect.");
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

void is_gga_mgga(int func_id, bool & gga, bool & mgga) {
  // Initialize
  gga=false;
  mgga=false;

  // Correlation and exchange functionals
  xc_func_type func;
  if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0) {
    ERROR_INFO();
    std::ostringstream oss;
    oss << "Functional "<<func_id<<" not found!";
    throw std::runtime_error(oss.str());
  }  

  switch(func.info->family)
    {
    case XC_FAMILY_LDA:
      gga=false;
      mgga=false;
      break;

    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
      gga=true;
      mgga=false;
      break;

    case XC_FAMILY_MGGA:
    case XC_FAMILY_HYB_MGGA:
      gga=false;
      mgga=true;
      break;

    default:
      {
        ERROR_INFO();
	std::ostringstream oss;
        oss << "Functional family " << func.info->family << " not currently supported in ERKALE!\n";
        throw std::runtime_error(oss.str());
      }
    }

  // Free functional
  xc_func_end(&func);
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
      case XC_FAMILY_HYB_MGGA:
	// libxc prior to 2.0.0
	// f=xc_hyb_gga_exx_coef(func.gga);
	// libxc 2.0.0
	f=xc_hyb_exx_coef(&func);
	break;
      }

    xc_func_end(&func);
  }

  //  printf("Fraction of exact exchange is %f.\n",f);

  return f;
}

bool is_range_separated(int func_id) {
  bool ans=false;
  
  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    // Get flag
    ans=func.info->flags & XC_FLAGS_HYB_CAM;
    // Free functional
    xc_func_end(&func);
  }
  
  // Sanity check
  double w, a, b;
  range_separation(func_id,w,a,b);

  if(ans && w==0.0)
    fprintf(stderr,"Error in libxc detected - functional is marked range separated but with vanishing omega!\n");
  else if(!ans && w!=0.0)
    fprintf(stderr,"Error in libxc detected - functional is not marked range separated but has nonzero omega!\n");
  
  return ans;
}

void range_separation(int func_id, double & omega, double & alpha, double & beta) {
  // Defaults
  omega=0.0;
  alpha=0.0;
  beta=0.0;

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
      case XC_FAMILY_HYB_MGGA:
	XC(hyb_cam_coef(&func,&omega,&alpha,&beta));
	break;
      }

    xc_func_end(&func);
  }
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
      case XC_FAMILY_HYB_MGGA:
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
      case XC_FAMILY_HYB_MGGA:
	lapl=1;
	break;
      }
    // Free functional
    xc_func_end(&func);
  }

  return lapl;
}

bool has_exc(int func_id) {
  bool ans=false;

  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    // Get flag
    ans=func.info->flags & XC_FLAGS_HAVE_EXC;
    // Free functional
    xc_func_end(&func);
  }
  
  return ans;
}
