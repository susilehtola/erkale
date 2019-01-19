#include "../eriworker.h"
#include "../integrals.h"
#include "../checkpoint.h"
#include "../settings.h"

Settings settings;

int main(int argc, char **argv) {
  if(argc!=2) {
    printf("Usage: %s checkpoint\n",argv[0]);
    return 1;
  }

  // Initialize libint
  init_libint_base();

  // Load checkpoint
  Checkpoint chkpt(argv[1],false);

  // Read basis set
  BasisSet basis;
  chkpt.read(basis);

  // Get shells in basis set
  std::vector<GaussianShell> shells(basis.get_shells());

  // Loop over shells
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    ERIWorker eri(basis.get_max_am(),basis.get_max_Ncontr());

#ifdef _OPENMP
#pragma omp for
#endif
    for(size_t is=0;is<shells.size();is++) {
      for(size_t js=0;js<shells.size();js++) {
	for(size_t ks=0;ks<shells.size();ks++) {
	  for(size_t ls=0;ls<shells.size();ls++) {
	    // Get libint integrals
	    eri.compute(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	    std::vector<double> libint(eri.get());

	    // Get Huzinaga integrals
	    eri.compute_debug(&shells[is],&shells[js],&shells[ks],&shells[ls]);
	    std::vector<double> huzinaga(eri.get());

	    // Compare integrals
	    size_t Ni=shells[is].get_Nbf();
	    size_t Nj=shells[js].get_Nbf();
	    size_t Nk=shells[ks].get_Nbf();
	    size_t Nl=shells[ls].get_Nbf();

	    for(size_t ii=0;ii<Ni;ii++)
	      for(size_t jj=0;jj<Nj;jj++)
		for(size_t kk=0;kk<Nk;kk++)
		  for(size_t ll=0;ll<Nl;ll++) {
		    size_t i=((ii*Nj+jj)*Nk+kk)*Nl+ll;
		    if(fabs(libint[i]-huzinaga[i])>1e-6*std::max(1.0,std::max(fabs(libint[i]),fabs(huzinaga[i])))) {
		      printf("%4i %e %e %e\n",(int) i, libint[i], huzinaga[i], libint[i]-huzinaga[i]);
		      printf("is, i = %i\n",(int) ii);
		      shells[is].print();
		      printf("js, j = %i\n",(int) jj);
		      shells[js].print();
		      printf("ks, k = %i\n",(int) kk);
		      shells[ks].print();
		      printf("ls, l = %i\n",(int) ll);
		      shells[ls].print();
		      printf("ints:");
		      for(size_t j=0;j<libint.size();j++)
			printf(" % e",libint[j]);
		      printf("\n");
		      fflush(stdout);

		      throw std::runtime_error("Integrals are wrong.\n");
		    }
		  }
	    //printf("%c %c %c %c OK\n",shell_types[shells[is].get_am()],shell_types[shells[js].get_am()],shell_types[shells[ks].get_am()],shell_types[shells[ls].get_am()]);
	  }
	  //printf("%c %c %c * OK\n",shell_types[shells[is].get_am()],shell_types[shells[js].get_am()],shell_types[shells[ks].get_am()]);
	}
	printf("%c %c * * OK\n",shell_types[shells[is].get_am()],shell_types[shells[js].get_am()]);
      }
    }
  }

  return 0;
}

