/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010
 * Copyright (c) 2010, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * The
 *
 */

#include <cmath>
#include <cstdlib>
#include <cstdio>

#include <sstream>
#include <stdexcept>

#include "mathf.h"
#include "solidharmonics.h"

// Uncomment to print out the solid harmonics
//#define DEBUG

// Uncomment to compile this as the main program
//#define YLMDEBUG


arma::mat Ylm_transmat(int l) {
  // Get transformation matrix (Y_lm, cart)

  arma::mat ret(2*l+1,(l+1)*(l+2)/2);
  ret.zeros();

  std::vector<double> tmp;

  for(int m=-l;m<=l;m++) {
    tmp=calcYlm_coeff(l,m);
    for(size_t i=0;i<tmp.size();i++)
      ret(m+l,i)=tmp[i];
  }

  return ret;
}


std::vector<double> calcYlm_coeff(int l, int mval) {
  // Form list of cartesian coefficients

  // Size of returned array
  const int N=(l+1)*(l+2)/2;

  // Returned array
  std::vector<double> ret;
  ret.reserve(N);
  ret.resize(N);

  int m=abs(mval);

  // Compute prefactor
  double prefac=sqrt((2*l+1)/(4.0*M_PI))*pow(2.0,-l);
  if(m!=0)
    prefac*=sqrt(fact(l-m)*2.0/fact(l+m));

  // Calculate bar Pi contribution
  for(int k=0;k<=(l-m)/2;k++) {
    // Compute factor in front
    double ffac=pow(-1.0,k)*choose(l,k)*choose(2*(l-k),l);
    if(m!=0)
      ffac*=fact(l-2*k)/fact(l-2*k-m);
    ffac*=prefac;

    // Distribute exponents
    for(int a=0;a<=k;a++) {
      double afac=choose(k,a)*ffac;

      for(int b=0;b<=a;b++) {
	double fac=choose(a,b)*afac;

	// Current exponents
	int zexp=2*b-2*k+l-m;
	int yexp=2*(a-b);
	int xexp=2*(k-a);

	// Now, add in the contribution of A or B.
	if(mval>0) {
	  // Contribution from A_m
	  for(int p=0;p<=m;p++) {

	    // Check if term contributes
	    int cosfac;
	    switch((m-p)%4) {
	    case(0):
	      // cos(0) = 1
	      cosfac=1;
	      break;
	    case(1):
	      // cos(pi/2) = 0
	      cosfac=0;
	      break;
	    case(2):
	      // cos(pi) = -1
	      cosfac=-1;
	      break;
	    case(3):
	      // cos(3*pi/2) = 0
	      cosfac=0;
	      break;
	    default:
	      ERROR_INFO();
	      throw std::domain_error("An error occurred in Am(x,y).\n");
	    }

	    if(cosfac!=0) {
	      // OK, term contributes, store result.
	      //	      printf("Contribution to %i %i %i\n",xexp+p,yexp+absm-p,zexp);
	      ret[getind(xexp+p,yexp+m-p,zexp)]+=cosfac*choose(m,p)*fac;
	    }
	  }
	} else if(m==0) {
	  // No A_m or B_m term.
	  ret[getind(xexp,yexp,zexp)]+=fac;

	} else {
	  // B_m contributes
	  for(int p=0;p<=m;p++) {

	    // Check contribution of current term
	    int sinfac;
	    switch((m-p)%4) {
	    case(0):
	      // sin(0) = 0
	      sinfac=0;
	      break;
	    case(1):
	      // sin(pi/2) = 1
	      sinfac=1;
	      break;
	    case(2):
	      // sin(pi) = 0
	      sinfac=0;
	      break;
	    case(3):
	      // sin(3*pi/2) = -1
	      sinfac=-1;
	      break;
	    default:
	      ERROR_INFO();
	      throw std::domain_error("An error occurred in Bm(x,y).\n");
	    }

	    if(sinfac!=0) {
	      // OK, contribution is made
	      //	      printf("Contribution to %i %i %i\n",xexp+p,yexp+absm-p,zexp);
	      ret[getind(xexp+p,yexp+m-p,zexp)]+=sinfac*choose(m,p)*fac;
	    }
	  } // End loop over p
	} // End B_m clause
      } // End loop over b
    } // End loop over a
  } // End loop over k

#ifdef DEBUG
  printf("Y %i %i\n",l,mval);
  for(int ii=0; ii<=l; ii++) {
    int nx=l - ii;
    for(int jj=0; jj<=ii; jj++) {
      int ny=ii - jj;
      int nz=jj;

      if(ret[getind(nx,ny,nz)]!=0)
	printf("%e\t%i\t%i\t%i\n",ret[getind(nx,ny,nz)],nx,ny,nz);
    }
  }
  printf("\n");
#endif

  return ret;
}

#ifdef YLMDEBUG

void checkYlm() {
  for(int l=0;l<4;l++) {
    for(int m=-l;m<=l;m++) {
      calcYlm_coeff(l,m);
    }
    printf("\n\n");
  }
}

int main() {
  checkYlm();
  return 0;
}

#endif
