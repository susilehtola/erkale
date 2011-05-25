#include "basis.h"
#include "basislibrary.h"
#include "completeness_profile.h"

int main(int argc, char **argv) {
  if(argc!=3) {
    printf("Usage: %s basis.gbs element\n",argv[0]);
    return 0;
  }

  // Load basis set
  BasisSetLibrary baslib;
  baslib.load_gaussian94(argv[1]);

  // Get wanted element from basis
  ElementBasisSet elbas=baslib.get_element(argv[2]);

  // Compute completeness profile
  compprof_t prof=compute_completeness(elbas);

  // Print legend
  printf("%-13s","# alpha");
  for(size_t j=0;j<prof.shells.size();j++)
    printf("\t%c%13s",shell_types[prof.shells[j].am],"");
  printf("\n");

  // Print profile
  for(size_t i=0;i<prof.lga.size();i++) {
    // Value of scanning exponent
    printf("%13e",prof.lga[i]);
    // Print completeness of shells
    for(size_t j=0;j<prof.shells.size();j++)
      printf("\t%13e",prof.shells[j].Y[i]);
    printf("\n");
  }

  return 0;
}
