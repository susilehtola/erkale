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
  for(int Z=1;Z<element_N;Z++)
    if(el==element_symbols[Z])
      return Z;

  ERROR_INFO();
  throw std::runtime_error("Element not found in element library!\n");

  // Not found, return dummy charge.
  return 0;
}
