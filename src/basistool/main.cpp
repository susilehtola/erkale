#include "../basislibrary.h"
#include "../stringutil.h"
#include "../completeness/completeness_profile.h"

std::string cmds[]={"completeness", "decontract", "dump"};


void help() {
  printf("Valid commands:\n");
  for(size_t i=0;i<sizeof(cmds)/sizeof(cmds[0]);i++)
    printf("\t%s\n",cmds[i].c_str());
}

int main(int argc, char **argv) {
  if(argc<3) {
    printf("Usage: %s input.gbs command\n\n",argv[0]);
    help();
    return 0;
  }

  // Get filename
  std::string filein(argv[1]);
  // Load input
  BasisSetLibrary bas;
  bas.load_gaussian94(filein);

  // Get command
  std::string cmd(argv[2]);
  // and determine what to do.
  if(stricmp(cmd,"completeness")==0) {
    // Print completeness profile.

    if(argc!=5) {
      printf("\nUsage: %s input.gbs completeness element output.dat\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);

    // Get wanted element from basis
    ElementBasisSet elbas=bas.get_element(el);
    
    // Compute completeness profile
    compprof_t prof=compute_completeness(elbas);

    // Print profile in output file
    FILE *out=fopen(fileout.c_str(),"w");
    for(size_t i=0;i<prof.lga.size();i++) {
      // Value of scanning exponent
      fprintf(out,"%13e",prof.lga[i]);
      // Print completeness of shells
      for(size_t j=0;j<prof.shells.size();j++)
	fprintf(out,"\t%13e",prof.shells[j].Y[i]);
      fprintf(out,"\n");
    }
    fclose(out);

  } else if(stricmp(cmd,"decontract")==0) {
    // Decontract basis set.

    if(argc!=4) {
      printf("\nUsage: %s input.gbs decontract output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout(argv[3]);
    bas.decontract();
    bas.save_gaussian94(fileout);
  } else if(stricmp(cmd,"dump")==0) {
    // Dump wanted element.

    if(argc!=5) {
      printf("\nUsage: %s input.gbs dump element output.gbs\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);

    // Save output
    BasisSetLibrary elbas;
    elbas.add_element(bas.get_element(el));
    elbas.save_gaussian94(fileout);
  } else {
    printf("\nInvalid command.\n");

    help();
  }

  return 0;
}
