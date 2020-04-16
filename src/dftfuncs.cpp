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

#define ID_NONE 0
#define ID_HF -1
#define KW_NONE "none"
#define KW_HF "hyb_x_hf"

// Print keyword corresponding to functional.
std::string get_keyword(int func_id) {
  // Check if none was specified. This is internal to ERKALE.
  if(func_id==ID_NONE)
    return KW_NONE;
  else if(func_id==ID_HF)
    return KW_HF;

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
  if(stricmp(name,KW_NONE)==0)
    return ID_NONE;
  else if(stricmp(name,KW_HF)==0)
    return ID_HF;

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

  // No handling for Hartree-Fock here
  if(stricmp(xc,"HF")==0 || stricmp(xc,"ROHF")==0) {
    return;
  }

  // Check if there is a dash or plus in the input.
  size_t dpos=xc.find('-',0);
  size_t ppos=xc.find('+',0);

  size_t pos=std::string::npos;
  if(ppos!=std::string::npos)
    pos=ppos;
  if(dpos!=std::string::npos)
    pos=dpos;

  if(pos!=std::string::npos) {
    // OK, there is a dash.

    // Exchange part is
    std::string x=xc.substr(0,pos);
    // and correlation part is
    std::string c=xc.substr(pos+1,xc.size()-pos);

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

    bool gga, mgga_t, mgga_l;
    is_gga_mgga(func_id,gga,mgga_t,mgga_l);
    if(gga) printf("Functional is a GGA\n");
    if(mgga_t) printf("Functional is a tau-mGGA\n");
    if(mgga_l) printf("Functional is a lapl-mGGA\n");
  }
  if(!has_exc(func_id)) {
    printf("The functional doesn't have an energy density, so the calculated energy is incorrect.");
  }
}

bool is_exchange(int func_id) {
  bool ans=false;

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
	ans=true;
	break;
      default:
	ans=false;
      }
    // Free functional
    xc_func_end(&func);

  } else if(func_id==ID_HF) {
    ans=true;

  } else
    // Dummy exchange
    ans=false;

  return ans;
}

bool is_exchange_correlation(int func_id) {
  bool ans=false;

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
	ans=true;
	break;
      default:
	ans=false;
      }
    // Free functional
    xc_func_end(&func);
  } else
    // Dummy exchange
    ans=false;

  return ans;
}

bool is_correlation(int func_id) {
  bool ans=false;

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
	ans=true;
	break;
      default:
	ans=false;
      }
    // Free functional
    xc_func_end(&func);
  } else
    // Dummy correlation
    ans=false;

  return ans;
}

bool is_kinetic(int func_id) {
  bool ans=false;

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
	ans=true;
	break;
      default:
	ans=false;
      }
    // Free functional
    xc_func_end(&func);
  } else
    // Dummy correlation
    ans=false;

  return ans;
}

void is_gga_mgga(int func_id, bool & gga, bool & mgga_t, bool & mgga_l) {
  // Initialize
  gga=false;
  mgga_t=false;
  mgga_l=false;

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
#ifdef XC_FAMILY_HYB_LDA
    case XC_FAMILY_HYB_LDA:
#endif
      break;

    case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
    case XC_FAMILY_HYB_GGA:
#endif
      gga=true;
      break;

    case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
    case XC_FAMILY_HYB_MGGA:
#endif
      mgga_t=true;
#if defined(XC_FLAGS_NEEDS_LAPLACIAN)
      mgga_l=func.info->flags & XC_FLAGS_NEEDS_LAPLACIAN;
#else
      mgga_l=true;
#endif
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

#if XC_MAJOR_VERSION >= 6
    switch(xc_hyb_type(&func)) {
    case(XC_HYB_HYBRID):
      f=xc_hyb_exx_coef(&func);
      break;
    }
#else
    switch(func.info->family)
      {
#ifdef XC_FAMILY_HYB_LDA
    case XC_FAMILY_HYB_LDA:
#endif
#ifdef XC_FAMILY_HYB_GGA
      case XC_FAMILY_HYB_GGA:
#endif
#ifdef XC_FAMILY_HYB_MGGA
      case XC_FAMILY_HYB_MGGA:
#endif
	// libxc prior to 2.0.0
	// f=xc_hyb_gga_exx_coef(func.gga);
	// libxc 2.0.0
	f=xc_hyb_exx_coef(&func);
	break;
      }
#endif

    xc_func_end(&func);
  } else if(func_id==ID_HF)
    f=1.0;

  //  printf("Fraction of exact exchange is %f.\n",f);

  return f;
}

bool is_supported(int func_id) {
  bool support=true;

  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    // Get flag
#if XC_MAJOR_VERSION >= 6
    switch(xc_hyb_type(&func)) {
    case(XC_HYB_SEMILOCAL):
    case(XC_HYB_HYBRID):
    case(XC_HYB_CAM):
      break;

    default:
      support=false;
    }
#else
    if(func.info->flags & XC_FLAGS_HYB_CAMY)
      support=false;
    if(func.info->flags & XC_FLAGS_HYB_LCY)
      support=false;
#endif
    // Free functional
    xc_func_end(&func);
  }

  return support;
}

bool is_range_separated(int func_id, bool check) {
  bool ans=false;

  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
#if XC_MAJOR_VERSION >= 6
    ans=(xc_hyb_type(&func) == XC_HYB_CAM);
#else
    // Get flag
    ans=func.info->flags & XC_FLAGS_HYB_CAM;
#endif
    // Free functional
    xc_func_end(&func);
  }

  if(check) {
    // Sanity check
    double w, a, b;
    range_separation(func_id,w,a,b);

    if(ans && w==0.0)
      fprintf(stderr,"Error in libxc detected - functional is marked range separated but with vanishing omega!\n");
    else if(!ans && w!=0.0)
      fprintf(stderr,"Error in libxc detected - functional is not marked range separated but has nonzero omega!\n");
  }

  return ans;
}

void range_separation(int func_id, double & omega, double & alpha, double & beta, bool check) {
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

#if XC_MAJOR_VERSION >= 6
    switch(xc_hyb_type(&func)) {
    case(XC_HYB_HYBRID):
    case(XC_HYB_CAM):
      XC(hyb_cam_coef(&func,&omega,&alpha,&beta));
    break;
    }
#else

    switch(func.info->family) {
#ifdef XC_FAMILY_HYB_LDA
    case XC_FAMILY_HYB_LDA:
#endif
#ifdef XC_FAMILY_HYB_GGA
    case XC_FAMILY_HYB_GGA:
#endif
#ifdef XC_FAMILY_HYB_MGGA
    case XC_FAMILY_HYB_MGGA:
#endif
      XC(hyb_cam_coef(&func,&omega,&alpha,&beta));
      break;
    }
#endif


    xc_func_end(&func);
  } else if(func_id==ID_HF)
    alpha=1.0;

  bool ans=is_range_separated(func_id,false);
  if(check) {
    if(ans && omega==0.0) {
      fprintf(stderr,"Error in libxc detected - functional is marked range separated but with vanishing omega!\n");
      printf("Error in libxc detected - functional is marked range separated but with vanishing omega!\n");
    } else if(!ans && omega!=0.0) {
      fprintf(stderr,"Error in libxc detected - functional is not marked range separated but has nonzero omega!\n");
      printf("Error in libxc detected - functional is not marked range separated but has nonzero omega!\n");
    }
  }

  // Work around libxc bug
  if(!ans) {
    omega=0.0;
    beta=0.0;
  }
}

bool needs_VV10(int func_id, double & b, double & C) {
  b=0.0;
  C=0.0;

  bool ret=false;
#ifdef XC_FLAGS_VV10
  if(func_id>0) {
    xc_func_type func;
    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
      ERROR_INFO();
      std::ostringstream oss;
      oss << "Functional "<<func_id<<" not found!";
      throw std::runtime_error(oss.str());
    }
    // Get flag
    ret=func.info->flags & XC_FLAGS_VV10;
    if(ret)
      XC(nlc_coef)(&func, &b, &C);
    // Free functional
    xc_func_end(&func);
  }
#endif

  return ret;
}

bool gradient_needed(int func_id) {
  // Is gradient necessary?

  bool grad=false;

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
#ifdef XC_FAMILY_HYB_GGA
      case XC_FAMILY_HYB_GGA:
#endif
      case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
      case XC_FAMILY_HYB_MGGA:
#endif
	grad=true;
	break;
      }
    // Free functional
    xc_func_end(&func);
  }

  return grad;
}

bool tau_needed(int func_id) {
  bool tau=false;

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
#ifdef XC_FAMILY_HYB_MGGA
      case XC_FAMILY_HYB_MGGA:
#endif
	tau=true;
	break;
      }
    // Free functional
    xc_func_end(&func);
  }

  return tau;
}

bool laplacian_needed(int func_id) {
  bool lapl=false;

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
#ifdef XC_FAMILY_HYB_MGGA
      case XC_FAMILY_HYB_MGGA:
#endif
#ifdef XC_FLAGS_NEEDS_LAPLACIAN
	lapl=func.info->flags & XC_FLAGS_NEEDS_LAPLACIAN;
#else
	lapl=true;
#endif
	break;
      }
    // Free functional
    xc_func_end(&func);
  }

  return lapl;
}

bool has_exc(int func_id) {
  bool ans=true;

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
