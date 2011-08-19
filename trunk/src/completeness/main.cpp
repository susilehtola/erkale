#include <cstdio>
#include "optimize_completeness.h"
#include "../basislibrary.h"

int main(int argc, char **argv) {

  if(argc!=5) {
    printf("Usage: %s am min max Nf\n",argv[0]);
    return 1;
  }

  // Get parameters
  int am=atoi(argv[1]);
  double min=atof(argv[2]);
  double max=atof(argv[3]);
  int Nf=atoi(argv[4]);

  // Form optimized set of primitives
  std::vector<double> exps=optimize_completeness(am,min,max,Nf);
  
  // Create a basis set out of it
  ElementBasisSet el("El");
  for(size_t i=0;i<exps.size();i++) {
    // Create shell of functions
    FunctionShell tmp(am);
    tmp.add_exponent(1.0,exps[i]);
    // and add it to the basis set
    el.add_function(tmp);
  }

  BasisSetLibrary baslib;
  baslib.add_element(el);
  baslib.save_gaussian94("optimized.gbs");


  return 0;
}
