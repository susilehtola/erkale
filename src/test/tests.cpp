#include "basislibrary.h"
#include "global.h"
#include "mathf.h"
#include "scf.h"
#include "timer.h"
#include "xyzutils.h"

#include <cmath>
#include <cstdio>

/// Check if \f$ | x - y | < \tau \f$
bool compare(double x, double y, double tau) {
  // Compute relative difference
  double d=(x-y)/y;

  if(fabs(d)<tau) {
    //    printf("%e vs %e, difference %e, ok\n",x,y,d);
    return 1;
  } else {
    //    printf("%e vs %e, difference %e, fail\n",x,y,d);
    return 0;
  }
}

bool compare(const arma::vec & x, const arma::vec & y, double tau, size_t & nsucc, size_t & nfail) {
  //  if(x.n_elem!=y.n_elem)
  //  printf("len(x)!=len(y)!\n");
  size_t N=min(x.n_elem,y.n_elem);

  //  printf("Nx=%i, Ny=%i, N=%i\n",(int) Nx, (int) Ny, (int) N);

  nsucc=0;
  nfail=0;

  bool ok=1;
  for(size_t i=0;i<N;i++) {
    // Use larger value as reference, since orbital energy can be zero, or small
    double r;
    if(fabs(x(i))>fabs(y(i)))
      r=x(i);
    else
      r=y(i);      
    
    // Compute relative difference
    double d=(x(i)-y(i))/r;

    if(fabs(d)>tau) {
      //      printf("%e vs %e, difference %e, fail\n",x(i),y(i),d);
      ok=0;
      nfail++;
    } else {
      //      printf("%e vs %e, difference %e, ok\n",x(i),y(i),d);
      nsucc++;
    }
  }
    
  return ok;
}

  

void set_vec(arma::vec & test, const double * arr) {
  size_t N=sizeof(arr)/sizeof(arr[0]);
  test=arma::vec(N);

  for(size_t i=0;i<N;i++)
    test(i)=arr[i];
}

atom_t convert_to_bohr(const atom_t & in) {
  atom_t ret=in;

  ret.x*=ANGSTROMINBOHR;
  ret.y*=ANGSTROMINBOHR;
  ret.z*=ANGSTROMINBOHR;

  return ret;
}


/// Run unit tests
int main(void) {
  // Load basis sets

  printf("****** Loading basis sets *******\n");

  BasisSetLibrary b3_21G;
  b3_21G.load_gaussian94("3-21G");

  BasisSetLibrary cc_pVDZ;
  cc_pVDZ.load_gaussian94("cc-pVDZ");

  BasisSetLibrary cc_pVTZ;
  cc_pVTZ.load_gaussian94("cc-pVTZ");

  BasisSetLibrary cc_pVQZ;
  cc_pVQZ.load_gaussian94("cc-pVQZ");

  BasisSetLibrary cc_pV5Z;
  cc_pV5Z.load_gaussian94("cc-pV5Z");

  BasisSetLibrary cc_pV6Z;
  cc_pV6Z.load_gaussian94("cc-pV6Z");

  BasisSetLibrary aug_cc_pVDZ;
  aug_cc_pVDZ.load_gaussian94("aug-cc-pVDZ");

  BasisSetLibrary aug_cc_pVTZ;
  aug_cc_pVTZ.load_gaussian94("aug-cc-pVTZ");

  BasisSetLibrary aug_cc_pVQZ;
  aug_cc_pVQZ.load_gaussian94("aug-cc-pVQZ");
  
  // Helper structure
  atom_t at;

  // Oxygen atom
  std::vector<atom_t> O;
  at.el="O"; at.x=0.0; at.y=0.0; at.z=0.0; O.push_back(convert_to_bohr(at));

  // Water monomer optimized at B3LYP/aug-cc-pVTZ level
  std::vector<atom_t> h2o;
  at.el="O"; at.x= 0.000000; at.y= 0.117030; at.z=0.000000; h2o.push_back(convert_to_bohr(at));
  at.el="H"; at.x= 0.763404; at.y=-0.468123; at.z=0.000000; h2o.push_back(convert_to_bohr(at));
  at.el="H"; at.x=-0.763404; at.y=-0.468123; at.z=0.000000; h2o.push_back(convert_to_bohr(at));

  // Cadmium complex
  std::vector<atom_t> cdcplx;
  at.el="Cd"; at.x= 0.000000; at.y= 0.000000; at.z= 0.000000; cdcplx.push_back(convert_to_bohr(at));
  at.el="N";  at.x= 0.000000; at.y= 0.000000; at.z=-2.260001; cdcplx.push_back(convert_to_bohr(at));
  at.el="N";  at.x=-0.685444; at.y= 0.000000; at.z=-4.348035; cdcplx.push_back(convert_to_bohr(at));
  at.el="C";  at.x= 0.676053; at.y= 0.000000; at.z=-4.385069; cdcplx.push_back(convert_to_bohr(at));
  at.el="C";  at.x= 1.085240; at.y= 0.000000; at.z=-3.091231; cdcplx.push_back(convert_to_bohr(at));
  at.el="C";  at.x=-1.044752; at.y= 0.000000; at.z=-3.060220; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x= 1.231530; at.y= 0.000000; at.z=-5.300759; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x= 2.088641; at.y= 0.000000; at.z=-2.711077; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x=-2.068750; at.y= 0.000000; at.z=-2.726515; cdcplx.push_back(convert_to_bohr(at));
  at.el="H";  at.x=-1.313170; at.y= 0.000000; at.z=-5.174718; cdcplx.push_back(convert_to_bohr(at));

  // Construct settings
  Settings set;
  set.set_bool("Verbose",0);

  // No spherical harmonics
  Settings cart=set;
  cart.set_bool("UseLM",0);

  // Settings for DFT
  Settings dftset=set;
  dftset.add_dft_settings();
  dftset.set_bool("DFTFitting",0);

  Settings dftcart=cart;
  dftcart.add_dft_settings();
  //  dftcart.set_bool("Verbose",1);

  printf("****** Running calculations *******\n");  
  Timer t;

  arma::vec E;
  arma::vec Etest;

  double Etot;
  arma::mat C;

  // Tolerance in total energy
  double tol=1e-5;
  // Relative tolerance in orbital energies
  double otol=6e-3;

  // Succeeded tests
  size_t nsucc=0, nfail=0;
  bool Eok;
  const char * stat[]={"fail","ok"};

  // Oxygen, HF, cc-pVDZ
  printf("Oxygen, HF/cc-pVDZ, ");
  fflush(stdout);
  BasisSet bas=construct_basis(O,cc_pVDZ,set);
  SCF solver=SCF(bas,set);
  Etot=solver.RHF(C,E);
  // Compare result to one obtained with Gaussian '09
  Etest="-20.69967 -1.25243 -0.57227 -0.57227 0.00953 1.13616 1.13616 1.18674 1.36560 2.86593 2.87507 2.87507 2.90263 2.90263";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-74.6652787264,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  // Cartesians
  t.set();
  printf("Oxygen, HF/cc-pVDZ cart, ");
  fflush(stdout);
  bas=construct_basis(O,cc_pVDZ,cart);
  solver=SCF(bas,cart);
  Etot=solver.RHF(C,E);
  Etest="-20.70015 -1.25253 -0.57240 -0.57240  0.00940 1.13604 1.13604 1.18661 1.20425 2.86543 2.87492 2.87492 2.90248 2.90248 5.44588";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-74.6653416523,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  // Oxygen, HF, cc-pVTZ
  t.set();
  printf("Oxygen, HF/cc-pVTZ, ");
  fflush(stdout);
  bas=construct_basis(O,cc_pVTZ,set);
  solver=SCF(bas,set);
  Etot=solver.RHF(C,E);
  Etest="-20.70108 -1.25918 -0.58231 -0.58231  -0.00610 0.74787 0.74787 0.78657 0.91580 1.67689 1.68280 1.68280 1.70640 1.70640 3.99594 4.01376 4.01376 5.43621 5.44818 5.44818 5.48443 5.48443 5.54635 5.54635 6.57353 6.59222 6.59222 6.65088 6.65088 7.50677";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-74.6842418162,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  t.set();
  printf("Oxygen, HF/cc-pVTZ cart, ");
  fflush(stdout);
  bas=construct_basis(O,cc_pVTZ,cart);
  solver=SCF(bas,cart);
  Etot=solver.RHF(C,E);
  Etest="-20.70154 -1.25951 -0.58257 -0.58257  -0.00705 0.54359 0.70313 0.70313 0.74062 1.67536 1.68252 1.68252 1.70613 1.70613 2.47225 3.26634 3.27440 3.27440 5.43462 5.44700 5.44700 5.48408 5.48408 5.54599 5.54599 6.57348 6.59184 6.59184 6.65049 6.65049 6.95154 7.04346 7.04346 10.23501 16.31527";
  compare(E,Etest,otol,nsucc,nfail);
  Eok=compare(Etot,-74.6843568531,tol);
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  // Oxygen, HF, cc-pVQZ
  t.set();
  printf("Oxygen, HF/cc-pVQZ, ");
  fflush(stdout);
  bas=construct_basis(O,cc_pVQZ,set);
  solver=SCF(bas,set);
  Etot=solver.RHF(C,E);
  Etest="-20.70379 -1.26136 -0.58514 -0.58514  -0.01278 0.55682 0.55682 0.59174 0.59206 1.19341 1.19865 1.19865 1.22015 1.22015 2.59696 2.61128 2.61128 3.33417 3.34311 3.34311 3.37043 3.37043 3.41825 3.41825 3.61833 4.01719 4.03123 4.03123 4.07884 4.07884 9.19414 9.20304 9.20304 9.22975 9.22975 9.27433 9.27433 9.33687 9.33687 10.17521 10.23960 10.23960 11.31927 11.33411 11.33411 11.37897 11.37897 11.45555 11.45555 12.35192 12.37736 12.37736 12.45454 12.45454 39.93177";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-74.6896699557,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  t.set();
  printf("Oxygen, HF/cc-pVQZ cart, ");
  fflush(stdout);
  bas=construct_basis(O,cc_pVQZ,cart);
  solver=SCF(bas,cart);
  Etot=solver.RHF(C,E);
  Etest="-20.70395 -1.26150 -0.58522 -0.58522  -0.01390 0.36164 0.48360 0.48360 0.51483 1.19066 1.19784 1.19784 1.21934 1.21934 1.53419 1.81479 1.81693 1.81693 3.33121 3.34111 3.34111 3.37031 3.37031 3.41814 3.41814 3.90729 3.90995 3.95366 3.95366 4.01554 4.02941 4.02941 4.07700 4.07700 8.61066 9.19402 9.20292 9.20292 9.22963 9.22963 9.27421 9.27421 9.33674 9.33674 10.71611 10.79199 10.79199 11.32101 11.33544 11.33544 11.37885 11.37885 11.45541 11.45541 11.98940 12.01604 12.01604 12.09874 12.09874 19.04223 21.03341 21.05805 21.05805 21.14146 21.14146 29.09364 29.20881 29.20881 35.63262 115.44523";
  compare(E,Etest,otol,nsucc,nfail);
  Eok=compare(Etot,-74.6897155194,tol);
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  // Oxygen, HF, cc-pV5Z
  t.set();
  printf("Oxygen, HF/cc-pV5Z, ");
  fflush(stdout);
  bas=construct_basis(O,cc_pV5Z,set);
  solver=SCF(bas,set);
  Etot=solver.RHF(C,E);
  Etest="-20.70511 -1.26244 -0.58632 -0.58632  -0.01839 0.35854 0.35854 0.38664 0.46293 0.96080 0.96565 0.96565 0.98555 0.98555 1.53018 1.53018 1.53451 2.32935 2.33632 2.33632 2.35751 2.35751 2.39404 2.39404 2.57605 2.95710 2.96755 2.96755 3.00597 3.00597 4.70802 4.76126 4.76126 5.89702 5.90404 5.90404 5.92511 5.92511 5.96026 5.96026 6.00960 6.00960 6.93364 6.94666 6.94666 6.98593 6.98593 7.05216 7.05216 8.03498 8.05804 8.05804 8.12937 8.12937 9.82858 14.09962 14.10617 14.10617 14.12579 14.12579 14.15844 14.15844 14.20405 14.20405 14.26249 14.26249 15.46629 15.54528 15.54528 18.31509 18.32573 18.32573 18.35762 18.35762 18.41066 18.41066 18.48468 18.48468 18.78266 18.79900 18.79900 18.84806 18.84806 18.93002 18.93002 21.73915 21.76765 21.76765 21.85343 21.85343 59.24108";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-74.6911280032,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  t.set();
  printf("Oxygen, HF/cc-pV5Z cart, ");
  fflush(stdout);
  bas=construct_basis(O,cc_pV5Z,cart);
  solver=SCF(bas,cart);
  Etot=solver.RHF(C,E);
  Etest="-20.70514 -1.26246 -0.58634 -0.58634  -0.01916 0.24647 0.30671 0.30671 0.33057 0.93947 0.94945 0.94945 0.96899 0.96899 1.01907 1.09974 1.09974 1.10895 2.12982 2.13651 2.13651 2.15633 2.15633 2.18889 2.18889 2.38580 2.54504 2.56618 2.56618 2.82425 2.83505 2.83505 2.87144 2.87144 4.56865 5.02778 5.08513 5.08513 5.51245 5.52242 5.52242 5.55250 5.55250 5.60677 5.60677 5.89699 5.90401 5.90401 5.92508 5.92508 5.96024 5.96024 6.00958 6.00958 6.89329 6.91450 6.91450 6.97965 6.97965 7.92822 9.30815 9.39985 9.39985 11.01970 11.03333 11.03333 11.07443 11.07443 11.14425 11.14425 13.20014 14.08526 14.10004 14.10655 14.10655 14.11271 14.11271 14.12605 14.12605 14.15855 14.15855 14.19904 14.19904 14.20403 14.20403 14.26247 14.26247 18.31506 18.32570 18.32570 18.35760 18.35760 18.41064 18.41064 18.48465 18.48465 20.27752 20.36487 20.36487 21.81215 21.82923 21.82923 21.87990 21.87990 21.96411 21.96411 22.28737 22.90014 23.01500 23.01500 25.40236 25.43618 25.43618 25.54124 25.54124 37.16624 37.19789 37.19789 37.28412 37.28412 39.91947 69.23639 69.33973 69.33973 71.82932 227.60232";
  compare(E,Etest,otol,nsucc,nfail);
  Eok=compare(Etot,-74.6911429425,tol);
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  // Water
  printf("\nWater, HF/cc-pVTZ, ");
  fflush(stdout);
  bas=construct_basis(h2o,cc_pVTZ,set);
  solver=SCF(bas,set);
  Etot=solver.RHF(C,E);
  Etest="-20.55528 -1.34286 -0.70829 -0.57575 -0.50391 0.14188 0.20352 0.54325 0.59754 0.66950 0.78748 0.80274 0.80481 0.85899 0.95702 1.13447 1.19282 1.52418 1.55795 2.03244 2.05947 2.06544 2.16865 2.23632 2.59095 2.95820 3.36100 3.49140 3.57419 3.64637 3.79772 3.87397 3.88245 3.95695 4.01991 4.07604 4.18620 4.30928 4.38758 4.56401 4.68180 4.85510 5.13809 5.25002 5.52756 6.04025 6.54533 6.91136 6.93662 7.00037 7.00782 7.06095 7.15980 7.22566 7.45618 7.77997 8.26537 12.80413";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-76.0568255234,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());  

  printf("Water, HF/aug-cc-pVTZ, ");
  fflush(stdout);
  bas=construct_basis(h2o,aug_cc_pVTZ,set);
  solver=SCF(bas,set);
  Etot=solver.RHF(C,E);
  Etest="-20.56777 -1.35083 -0.71636 -0.58299 -0.50982 0.02942 0.04763 0.13602 0.15865 0.17493 0.18285 0.22738 0.24476 0.26503 0.30096 0.31072 0.36560 0.43721 0.46596 0.59976 0.65741 0.70865 0.72190 0.73171 0.82943 0.84544 0.89190 0.91232 0.91963 0.92184 0.92993 0.96330 1.02075 1.04310 1.08285 1.10913 1.17508 1.18023 1.26105 1.51033 1.54187 1.58466 1.80411 1.81150 1.98618 2.11521 2.26177 2.31576 2.32571 2.41747 2.41893 2.44216 2.47762 2.67999 2.69914 2.76764 2.82401 2.88318 3.64821 3.74648 4.01301 4.08257 4.17453 4.24082 4.29774 4.37711 4.38189 4.40232 4.51073 4.71885 4.83341 5.13601 5.14861 5.25850 5.31911 5.51418 5.67201 6.16672 6.51749 6.70844 6.88448 7.13211 7.24970 7.30695 7.31358 7.31963 7.35598 7.53071 7.90764 7.91128 8.78808 15.69360";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-76.0602911220,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());  

  printf("Water, PBEPBE/cc-pVTZ, ");
  fflush(stdout);
  bas=construct_basis(h2o,cc_pVTZ,set);
  solver=SCF(bas,set);
  Etot=solver.RDFT(C,E,101,130);
  Etest="-18.73894 -0.91587 -0.47160 -0.32507 -0.24817 0.00846 0.08113 0.33807 0.37949 0.46549 0.54540 0.59072 0.59688 0.64398 0.74419 0.88307 0.97382 1.24128 1.26114 1.69991 1.72617 1.76196 1.82563 1.90028 2.18469 2.53269 2.98751 3.12603 3.18920 3.27996 3.28597 3.43994 3.51149 3.56979 3.61759 3.63634 3.69477 3.92402 3.95123 4.15577 4.19324 4.42924 4.64597 4.73039 4.98983 5.48690 5.98390 6.28438 6.29018 6.37812 6.42029 6.48118 6.53297 6.65945 6.84046 7.17250 7.62594 11.96201";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-76.3729402201,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());  

  printf("Water, PBEPBE/aug-cc-pVTZ, ");
  fflush(stdout);
  bas=construct_basis(h2o,aug_cc_pVTZ,set);
  solver=SCF(bas,set);
  Etot=solver.RDFT(C,E,101,130);
  Etest="-18.76009 -0.93087 -0.48493 -0.34193 -0.26600 -0.03436 0.01677 0.07005 0.08694 0.09434 0.10444 0.14266 0.16048 0.18183 0.21002 0.21323 0.30034 0.31377 0.32746 0.45440 0.49730 0.54936 0.56970 0.57003 0.65822 0.68674 0.70264 0.73842 0.74059 0.76799 0.78901 0.79661 0.85773 0.86385 0.88804 0.96776 1.01582 1.02802 1.07922 1.29115 1.32480 1.35099 1.52748 1.57682 1.69399 1.80870 1.93209 1.98954 2.00836 2.06610 2.11133 2.11975 2.15947 2.36758 2.37763 2.42223 2.52029 2.55683 3.24564 3.34397 3.55927 3.64908 3.81272 3.87875 3.92861 4.00025 4.01398 4.02074 4.07490 4.21634 4.36372 4.72476 4.74080 4.81900 4.83729 5.02603 5.19603 5.62127 5.95490 6.13267 6.31009 6.53461 6.61683 6.67150 6.70300 6.72563 6.73045 6.91700 7.28016 7.29703 8.14526 14.74052";
  compare(E,Etest,otol,nsucc,nfail); // Compare orbital energies
  Eok=compare(Etot,-76.3802078286,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());  
  
  // Cadmium, HF
  t.set();
  printf("\nCadmium complex, HF/3-21G, ");
  fflush(stdout);
  bas=construct_basis(cdcplx,b3_21G,cart);
  solver=SCF(bas,cart);
  Etot=solver.RHF(C,E);
  Etest="-949.85624 -141.46636 -131.19295 -131.19288 -131.19260 -27.67655 -23.23298 -23.23272 -23.23112 -16.04905 -16.04905 -16.04783 -16.04772 -16.04771 -15.60453 -15.53225 -11.29666 -11.24924 -11.23267 -4.39708 -2.89928 -2.89866 -2.89512 -1.41777 -1.23126 -1.06107 -0.87645 -0.85303 -0.81305 -0.72468 -0.71752 -0.71751 -0.71454 -0.71369 -0.65649 -0.65485 -0.64820 -0.61951 -0.51149 -0.45694 -0.36926 -0.18059    0.06931 0.07401 0.11409 0.14993 0.18267 0.19356 0.21198 0.25237 0.27656 0.28532 0.30337 0.33343 0.33689 0.39653 0.42174 0.54894 0.56114 0.68233 0.88549 0.92616 0.92671 0.96328 0.98347 0.99887 1.03645 1.08344 1.09366 1.19893 1.26177 1.28184 1.31939 1.38959 1.43089 1.47028 1.49453 1.56837 1.58225 1.62715 1.63231 1.67008 1.72945 1.83746 1.94602 1.97796 2.05689 2.24401 2.98294 3.07885 5.27574 211.21754 ";
  Eok=compare(Etot,-5663.73193295,tol); // Compare total energies
  compare(E,Etest,otol,nsucc,nfail);
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());

  // B3LYP
  t.set();
  printf("Cadmium complex, B3LYP/3-21G, ");
  fflush(stdout);
  solver=SCF(bas,dftcart);
  Etot=solver.RDFT(C,E,402,0);
  Etest="-939.84203 -137.53211 -127.75495 -127.75486 -127.75463 -25.87765 -21.69823 -21.69798 -21.69672 -14.96384 -14.96383 -14.96304 -14.96289 -14.96285 -14.36030 -14.28429 -10.22209 -10.18909 -10.17254 -3.72057 -2.37081 -2.37015 -2.36679 -1.08587 -0.93089 -0.79261 -0.66325 -0.64256 -0.60923 -0.49709 -0.48502 -0.48171 -0.47252 -0.46880 -0.46869 -0.46436 -0.46397 -0.45266 -0.33969 -0.32987 -0.27191 -0.13327    -0.00454 -0.00259 0.00790 0.03369 0.03759 0.06365 0.09551 0.13206 0.13440 0.15777 0.16229 0.18803 0.20173 0.21972 0.24645 0.36714 0.37514 0.45358 0.65944 0.67598 0.67887 0.72415 0.73375 0.73580 0.77452 0.84047 0.85272 0.92810 0.97421 1.00431 1.02562 1.07975 1.12282 1.19798 1.21563 1.28586 1.30330 1.32963 1.33687 1.38221 1.42041 1.55335 1.66768 1.67948 1.76658 1.91601 2.65978 2.75562 4.78461 208.10531 ";
  Eok=compare(Etot,-5667.68130846,tol); // Compare total energies
  compare(E,Etest,otol,nsucc,nfail);
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[Eok],(int) nsucc, (int) nfail,t.elapsed().c_str());


  return 0;
}
