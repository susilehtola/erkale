#include "../basislibrary.h"
#include "../stringutil.h"

std::string cmds[]={"decontract", "dump"};


void help() {
  printf("Valid commands:\n");
  for(size_t i=0;i<sizeof(cmds)/sizeof(cmds[0]);i++)
    printf("\t%s\n",cmds[i].c_str());
}

int main(int argc, char **argv) {
  if(argc<3) {
    printf("Usage: %s input.gbs command\n",argv[0]);
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
  if(stricmp(cmd,"decontract")==0) {
    // Decontract basis set.

    if(argc!=4) {
      printf("Usage: %s input.gbs decontract output.gbs\n",argv[0]);
      return 1;
    }

    std::string fileout(argv[3]);
    bas.decontract();
    bas.save_gaussian94(fileout);
  } else if(stricmp(cmd,"dump")==0) {
    // Dump wanted element.

    if(argc!=5) {
      printf("Usage: %s input.gbs dump element output.gbs\n",argv[0]);
      return 1;
    }

    std::string el(argv[3]);
    std::string fileout(argv[4]);

    // Save output
    BasisSetLibrary elbas;
    elbas.add_element(bas.get_element(el));
    elbas.save_gaussian94(fileout);
  } else
    throw std::runtime_error("Invalid command.\n");

  return 0;
}
