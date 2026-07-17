/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2026
 * Copyright (c) 2010-2026, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "trexio_interface.h"
#include "../eriworker.h"
#include "../settings.h"
#include "../stringutil.h"
#include <cstdio>
#include <string>

#ifdef SVNRELEASE
#include "../version.h"
#endif

/// Global settings object referenced by liberkale (e.g. basis.cpp).
Settings settings;

void help() {
  printf("Usage: erkale_trexio export <in.chk> <out.trexio>\n");
  printf("       erkale_trexio import <in.trexio> <out.chk>\n");
  fflush(stdout);
}

int main_guarded(int argc, char ** argv) {
  printf("ERKALE - TREXIO interface from Hel.\n");
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n", SVNREVISION);
#endif
  print_hostname();

  if(argc != 4) {
    help();
    return 1;
  }

  // basis.overlap() in the export self-check needs libint.
  init_libint_base();

  const std::string mode = argv[1];
  const std::string in   = argv[2];
  const std::string out  = argv[3];

  if(mode == "export")
    chk_to_trexio(in, out);
  else if(mode == "import")
    trexio_to_chk(in, out);
  else {
    printf("Unknown mode '%s'.\n", mode.c_str());
    help();
    return 1;
  }
  return 0;
}

int main(int argc, char ** argv) {
#ifdef CATCH_EXCEPTIONS
  try {
    return main_guarded(argc, argv);
  } catch(const std::exception & e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
#else
  return main_guarded(argc, argv);
#endif
}
