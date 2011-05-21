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



#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include <gsl/gsl_integration.h>
}

int main() {

  gsl_integration_glfixed_table *tab;
  double wtot;

  double *x, *w;
  FILE *out;
  char fname[256];

  for(int n=1;n<50;n++) {
    tab=gsl_integration_glfixed_table_alloc(n);

    // Upper limit of loop
    int ipmax=(n + 1) >> 1;

    // Allocate memory for helper arrays
    x=new double[n];
    w=new double[n];
    
    // Calculate sum of weights
    if(n%2==1) {
      // odd n
      wtot=tab->w[0];
      for(int ip=1;ip<ipmax;ip++)
	wtot+=2*tab->w[ip];
    } else {
      wtot=0;
      for(int ip=0;ip<ipmax;ip++)
	wtot+=2*tab->w[ip];
    }

    // Store nodes and weights
    if(n%2==0) {
      // Even n
      int ind=0;
      for(int ip=ipmax-1;ip>=0;ip--) {
	x[ind]=-tab->x[ip];
	w[ind]=tab->w[ip];
	ind++;
      }
      for(int ip=0;ip<ipmax;ip++) {
	x[ind]=tab->x[ip];
	w[ind]=tab->w[ip];
	ind++;
      }
    } else {
      // Odd n
      int ind=0;

      for(int ip=ipmax-1;ip>0;ip--) {
	x[ind]=-tab->x[ip];
	w[ind]=tab->w[ip];
	ind++;
      }

      // Node at origin
      x[ind]=tab->x[0];
      w[ind]=tab->w[0];
      ind++;

      for(int ip=1;ip<ipmax;ip++) {
	x[ind]=tab->x[ip];
	w[ind]=tab->w[ip];
	ind++;
      }
    }

    // Loop over points
    printf("Rule of order %i:, sum of weights is %e.\n",n,wtot);
    for(int ip=0;ip<ipmax;ip++)
      printf("%e\t%e\n",tab->x[ip],tab->w[ip]);
    printf("\n");

    // Store point arrays
    sprintf(fname,"glpoints-%i.dat",n);
    out=fopen(fname,"w");
    for(int ip=0;ip<n;ip++)
      fprintf(out,"%e\t%e\n",x[ip],w[ip]);
    fclose(out);

    // Free memory
    delete [] x;
    delete [] w;
    gsl_integration_glfixed_table_free(tab);
  }

  return 0;
}
