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



#include "xyzutils.h"

int main(int argc, char **argv) {

  if(argc<5) {
    printf("Usage: %s infile outfile commands\n",argv[0]);
    return 1;
  }

  // Read atoms
  std::vector<atom_t> atoms=loadxyz(argv[1]);

  // Parse commands
  int cur=3;
  while(cur<argc) {
    // Selected atoms
    std::vector<size_t> atinds;

    if(argv[cur]=="move") {
      cur++;
      // Read which atoms to move
      std::string tmp=argv[cur];

      // Generate list
      atinds.clear();

      size_t i=0, j=0;
    } else if(argv[cur]=="along")
      }
}
