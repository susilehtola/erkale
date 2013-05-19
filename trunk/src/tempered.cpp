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



#include "tempered.h"
#include <cmath>

// Construct even-tempered set of exponents
std::vector<double> eventempered_set(double alpha, double beta, int Nf) {
  std::vector<double> ret;

  for(int i=0;i<Nf;i++)
    ret.push_back(alpha*pow(beta,i));

  return ret;
}

// Construct well-tempered set of exponents
std::vector<double> welltempered_set(double alpha, double beta, double gamma, double delta, size_t Nf) {
  std::vector<double> ret;

  /*
  // WT-F1
  for(int i=0;i<Nf;i++)
    ret.push_back(alpha*pow(beta,i)*(1.0 + gamma*pow(i*1.0/Nf,delta)));
  */

  // WT-F2
  if(Nf>=1)
    ret.push_back(alpha);
  if(Nf>=2)
    ret.push_back(alpha*beta);
  for(size_t i=2;i<Nf;i++)
    ret.push_back(ret[i-1]*beta*(1.0 + gamma*pow((i+1.0)/Nf,delta)));

  return ret;
}

