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



#include "elements.h"
#include "global.h"
#include <sstream>
#include <stdexcept>

int get_Z(std::string el) {
  for(int Z=1;Z<(int) (sizeof(element_symbols)/sizeof(element_symbols[0]));Z++)
    if(el==element_symbols[Z])
      return Z;

  ERROR_INFO();
  throw std::runtime_error("Element not found in element library!\n");

  // Not found, return dummy charge.
  return 0;
}

std::vector<int> shell_count(int Z) {
  // Determine how many shells we have.
  std::vector <int> ret;
  
  // Electrons in closed shells
  int n=0;
  for(size_t i=0;i<sizeof(shell_order)/sizeof(shell_order[0]);i++) {
    if(Z>n) {
      // Still electrons left.
      int l=shell_order[i];
      n+=2*(2*l+1); // Amount of electrons on this shell

      // Resize output
      while(l+1>(int) ret.size())
        ret.push_back(0);
      ret[l]++;
    } else
      break;
  }

  // Overflow?
  if(Z>n) {
    std::ostringstream oss;
    oss << "Only implemented up to Z=" << n <<".\n";
    throw std::runtime_error(oss.str());
  }

  return ret;
}
