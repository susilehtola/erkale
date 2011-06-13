#include "basislibrary.h"
#include "global.h"
#include "mathf.h"
#include "scf.h"
#include "timer.h"
#include "xyzutils.h"

#include <cmath>
#include <cstdio>

/// Relative tolerance in total energy
double tol=1e-7;
/// Absolute tolerance in orbital energies
double otol=8e-5;

/// To compute references instead of running tests
//#define COMPUTE_REFERENCE

double rel_diff(double x, double y) {
  return (x-y)/y;
}

/// Check if \f$ | (x - y)/y | < \tau \f$
bool compare(double x, double y, double tau) {
  // Compute relative difference
  double d=rel_diff(x,y);
  
  if(fabs(d)<tau) {
    //    printf("%e vs %e, difference %e, ok\n",x,y,d);
    return 1;
  } else {
    //    printf("%e vs %e, difference %e, fail\n",x,y,d);
    return 0;
  }
}

bool compare(const arma::vec & x, const arma::vec & y, double tau, size_t & nsucc, size_t & nfail) {
  if(x.n_elem!=y.n_elem)
    throw std::runtime_error("Error - differing amount of computed and reference orbital energies!\n");

  size_t N=min(x.n_elem,y.n_elem);

  nsucc=0;
  nfail=0;

  bool ok=1;
  for(size_t i=0;i<N;i++) {
    double d=x(i)-y(i);

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

double max_diff(const arma::vec & x, const arma::vec & y) {
  if(x.n_elem!=y.n_elem)
    throw std::runtime_error("Error - differing amount of computed and reference orbital energies!\n");
  
  double m=0;
  for(size_t i=0;i<x.n_elem;i++) {
    double d=fabs(x(i)-y(i));
    if(d>m)
      m=d;
  }

  return m;
}

atom_t convert_to_bohr(const atom_t & in) {
  atom_t ret=in;

  ret.x*=ANGSTROMINBOHR;
  ret.y*=ANGSTROMINBOHR;
  ret.z*=ANGSTROMINBOHR;

  return ret;
}

// Possible statuses
const char * stat[]={"fail","ok"};

void rhf_test(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorb, const std::string & label) {
  Timer t;

  printf("%s, ",label.c_str());
  fflush(stdout);

  arma::vec E;
  arma::mat C;

  BasisSet bas=construct_basis(at,baslib,set);
  SCF solver=SCF(bas,set);
  double Et=solver.RHF(C,E);

#ifdef COMPUTE_REFERENCE
  printf("done (%s)\n",t.elapsed().c_str());
  printf("Etot=%e;\n",Et);
  printf("Eorb=\"");
  for(size_t i=0;i<E.n_elem;i++)
    printf("%e ",E(i));
  printf("\";\n");
#else
  // Compare results
  bool ok=1;
  size_t nsucc=0, nfail=0;
  compare(E,Eorb,otol,nsucc,nfail); // Compare orbital energies
  ok=compare(Et,Etot,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[ok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, maximum difference of orbital energy is %e.\n",rel_diff(Et,Etot),max_diff(E,Eorb));
  if(nfail!=0)
    ok=0;
  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    fflush(stdout);
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }  
#endif
}

void uhf_test(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorba, const arma::vec & Eorbb, const std::string & label) {
  Timer t;

  printf("%s, ",label.c_str());
  fflush(stdout);

  arma::vec Ea, Eb;
  arma::mat Ca, Cb;

  BasisSet bas=construct_basis(at,baslib,set);
  SCF solver=SCF(bas,set);
  double Et=solver.UHF(Ca,Cb,Ea,Eb);

#ifdef COMPUTE_REFERENCE
  printf("done (%s)\n",t.elapsed().c_str());
  printf("Etot=%e;\n",Et);
  printf("Eorba=\"");
  for(size_t i=0;i<Ea.n_elem;i++)
    printf("%e ",Ea(i));
  printf("\";\n");
  printf("Eorbb=\"");
  for(size_t i=0;i<Eb.n_elem;i++)
    printf("%e ",Eb(i));
  printf("\";\n");
#else
  // Compare results
  bool ok=1;
  size_t nsucca=0, nfaila=0;
  size_t nsuccb=0, nfailb=0;
  compare(Ea,Eorba,otol,nsucca,nfaila); // Compare orbital energies
  compare(Eb,Eorbb,otol,nsuccb,nfailb); // Compare orbital energies
  size_t nsucc=nsucca+nsuccb;
  size_t nfail=nfaila+nfailb;

  ok=compare(Et,Etot,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[ok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, maximum difference of orbital energies are %e and %e.\n",rel_diff(Et,Etot),max_diff(Ea,Eorba),max_diff(Eb,Eorbb));
  if(nfail!=0)
    ok=0;
  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }  
#endif
}

void rdft_test(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorb, const std::string & label, int xfunc, int cfunc) {
  Timer t;

  printf("%s, ",label.c_str());
  fflush(stdout);

  arma::vec E;
  arma::mat C;

  BasisSet bas=construct_basis(at,baslib,set);
  SCF solver=SCF(bas,set);
  double Et=solver.RDFT(C,E,xfunc,cfunc);

#ifdef COMPUTE_REFERENCE
  printf("done (%s)\n",t.elapsed().c_str());
  printf("Etot=%e;\n",Et);
  printf("Eorb=\"");
  for(size_t i=0;i<E.n_elem;i++)
    printf("%e ",E(i));
  printf("\";\n");
#else
  // Compare results
  bool ok=1;
  size_t nsucc=0, nfail=0;
  compare(E,Eorb,otol,nsucc,nfail); // Compare orbital energies
  ok=compare(Et,Etot,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[ok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, maximum difference of orbital energy is %e.\n",rel_diff(Et,Etot),max_diff(E,Eorb));
  if(nfail!=0)
    ok=0;
  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }  
#endif
}

void udft_test(const std::vector<atom_t> & at, const BasisSetLibrary & baslib, const Settings & set, double Etot, const arma::vec & Eorba, const arma::vec & Eorbb, const std::string & label, int xfunc, int cfunc) {
  Timer t;

  printf("%s, ",label.c_str());
  fflush(stdout);

  arma::vec Ea, Eb;
  arma::mat Ca, Cb;

  BasisSet bas=construct_basis(at,baslib,set);
  SCF solver=SCF(bas,set);
  double Et=solver.UDFT(Ca,Cb,Ea,Eb,xfunc,cfunc);

#ifdef COMPUTE_REFERENCE
  printf("done (%s)\n",t.elapsed().c_str());
  printf("Etot=%e;\n",Et);
  printf("Eorba=\"");
  for(size_t i=0;i<Ea.n_elem;i++)
    printf("%e ",Ea(i));
  printf("\";\n");
  printf("Eorbb=\"");
  for(size_t i=0;i<Eb.n_elem;i++)
    printf("%e ",Eb(i));
  printf("\";\n");
#else
  // Compare results
  bool ok=1;
  size_t nsucca=0, nfaila=0;
  size_t nsuccb=0, nfailb=0;
  compare(Ea,Eorba,otol,nsucca,nfaila); // Compare orbital energies
  compare(Eb,Eorbb,otol,nsuccb,nfailb); // Compare orbital energies
  size_t nsucc=nsucca+nsuccb;
  size_t nfail=nfaila+nfailb;

  ok=compare(Et,Etot,tol); // Compare total energies
  printf("E=%f %s, orbital energies %i ok, %i failed (%s)\n",Etot,stat[ok],(int) nsucc, (int) nfail,t.elapsed().c_str());
  printf("Relative difference of total energy is %e, maximum difference of orbital energies are %e and %e.\n",rel_diff(Et,Etot),max_diff(Ea,Eorba),max_diff(Eb,Eorbb));
  if(nfail!=0)
    ok=0;
  if(!ok) {
    std::ostringstream oss;
    ERROR_INFO();
    oss << "Test " << label << " failed.\n";
    throw std::runtime_error(oss.str());
  }  
#endif
}

/// Run unit tests by comparing calculations to ones that should be OK
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

  BasisSetLibrary aug_cc_pVDZ;
  aug_cc_pVDZ.load_gaussian94("aug-cc-pVDZ");

  /*
  BasisSetLibrary cc_pV5Z;
  cc_pV5Z.load_gaussian94("cc-pV5Z");

  BasisSetLibrary cc_pV6Z;
  cc_pV6Z.load_gaussian94("cc-pV6Z");

  BasisSetLibrary aug_cc_pVTZ;
  aug_cc_pVTZ.load_gaussian94("aug-cc-pVTZ");

  BasisSetLibrary aug_cc_pVQZ;
  aug_cc_pVQZ.load_gaussian94("aug-cc-pVQZ");
  */
  
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
  Settings sph;
  sph.set_bool("Verbose",0);

  // No spherical harmonics
  Settings cart=sph;
  cart.set_bool("UseLM",0);

  // Polarized calculation
  Settings pol=sph;
  pol.set_int("Charge",1);
  pol.set_int("Multiplicity",2);
  pol.set_bool("UseADIIS",0);

#ifdef DFT_ENABLED
  // DFT tests

  // Settings for DFT
  Settings dftsph=sph; // Normal settings
  dftsph.add_dft_settings();

  Settings dftcart=cart; // Cartesian basis
  dftcart.add_dft_settings();

  Settings dftnofit=dftsph; // No density fitting
  dftnofit.set_bool("DFTFitting",0);

  Settings dftdirect=dftsph; // Direct calculation
  dftdirect.set_bool("Direct",1);
  dftdirect.set_bool("DFTDirect",1);

  Settings dftpol=pol; // Polarized calculation
  dftpol.add_dft_settings();
  dftpol.set_bool("UseADIIS",0);
  dftpol.set_bool("UseBroyden",1);
#endif

  printf("****** Running calculations *******\n");  
  Timer t;

  // Reference total energy
  double Etot;
  // Reference orbital energies
  arma::vec Eorb;
  arma::vec Eorba;
  arma::vec Eorbb;

  // Oxygen, HF, cc-pVDZ
  Etot=-7.466528e+01;
  Eorb="-2.069967e+01 -1.252434e+00 -5.722705e-01 -5.722705e-01 9.529982e-03 1.136165e+00 1.136165e+00 1.186735e+00 1.365598e+00 2.865932e+00 2.875073e+00 2.875073e+00 2.902633e+00 2.902633e+00";
  rhf_test(O,cc_pVDZ,sph,Etot,Eorb,"Oxygen, HF/cc-pVDZ");
  
  Etot=-7.466534e+01;  // Cartesians
  Eorb="-2.070015e+01 -1.252528e+00 -5.723995e-01 -5.723995e-01 9.400249e-03 1.136041e+00 1.136041e+00 1.186611e+00 1.204250e+00 2.865434e+00 2.874924e+00 2.874924e+00 2.902485e+00 2.902485e+00 5.445885e+00 ";
  rhf_test(O,cc_pVDZ,cart,Etot,Eorb,"Oxygen, HF/cc-pVDZ cart");

  // Oxygen, HF, cc-pVTZ
  Etot=-7.468427e+01;
  Eorb="-2.070108e+01 -1.259254e+00 -5.823819e-01 -5.823798e-01 -6.071613e-03 7.474772e-01 7.475098e-01 7.854909e-01 9.157364e-01 1.676828e+00 1.682725e+00 1.682727e+00 1.706306e+00 1.706342e+00 3.974074e+00 4.000171e+00 4.000547e+00 5.449769e+00 5.451510e+00 5.472909e+00 5.484357e+00 5.485845e+00 5.561510e+00 5.562078e+00 6.573465e+00 6.592133e+00 6.592135e+00 6.650773e+00 6.650790e+00 7.506801e+00 ";
  rhf_test(O,cc_pVTZ,sph,Etot,Eorb,"Oxygen, HF/cc-pVTZ");

  Etot=-7.468436e+01;
  Eorb="-2.070154e+01 -1.259514e+00 -5.825730e-01 -5.825730e-01 -7.045070e-03 5.435867e-01 7.031265e-01 7.031265e-01 7.406204e-01 1.675360e+00 1.682519e+00 1.682519e+00 1.706133e+00 1.706133e+00 2.472268e+00 3.266339e+00 3.274396e+00 3.274396e+00 5.434615e+00 5.447003e+00 5.447003e+00 5.484080e+00 5.484080e+00 5.545990e+00 5.545990e+00 6.573477e+00 6.591839e+00 6.591839e+00 6.650485e+00 6.650485e+00 6.951536e+00 7.043459e+00 7.043459e+00 1.023563e+01 1.631530e+01 ";
  rhf_test(O,cc_pVTZ,cart,Etot,Eorb,"Oxygen, HF/cc-pVTZ cart");

  // Oxygen, HF, cc-pVQZ
  Etot=-7.468967e+01;
  Eorb="-2.070382e+01 -1.261376e+00 -5.851582e-01 -5.851563e-01 -1.277311e-02 5.566162e-01 5.566306e-01 5.878852e-01 5.919101e-01 1.193309e+00 1.198640e+00 1.198645e+00 1.220123e+00 1.220123e+00 2.589096e+00 2.609706e+00 2.610298e+00 3.338956e+00 3.341771e+00 3.345877e+00 3.370418e+00 3.370910e+00 3.420239e+00 3.420463e+00 3.541432e+00 4.017591e+00 4.031230e+00 4.031279e+00 4.078743e+00 4.078766e+00 9.026756e+00 9.201356e+00 9.204180e+00 9.233385e+00 9.234572e+00 9.274699e+00 9.277822e+00 9.307010e+00 9.337121e+00 1.016940e+01 1.023868e+01 1.023938e+01 1.133699e+01 1.134156e+01 1.136611e+01 1.137920e+01 1.138372e+01 1.147806e+01 1.147985e+01 1.235232e+01 1.237765e+01 1.237767e+01 1.245503e+01 1.245535e+01 4.052264e+01 ";
  rhf_test(O,cc_pVQZ,sph,Etot,Eorb,"Oxygen, HF/cc-pVQZ");

  Etot=-7.468972e+01;
  Eorb="-2.070395e+01 -1.261496e+00 -5.852231e-01 -5.852231e-01 -1.389756e-02 3.616428e-01 4.835979e-01 4.835979e-01 5.148287e-01 1.190663e+00 1.197844e+00 1.197844e+00 1.219342e+00 1.219342e+00 1.534186e+00 1.814791e+00 1.816928e+00 1.816928e+00 3.331214e+00 3.341110e+00 3.341110e+00 3.370314e+00 3.370314e+00 3.418136e+00 3.418136e+00 3.907290e+00 3.909951e+00 3.953655e+00 3.953655e+00 4.015545e+00 4.029409e+00 4.029409e+00 4.076995e+00 4.076995e+00 8.610672e+00 9.194019e+00 9.202917e+00 9.202917e+00 9.229630e+00 9.229630e+00 9.274205e+00 9.274205e+00 9.336736e+00 9.336736e+00 1.071611e+01 1.079199e+01 1.079199e+01 1.132101e+01 1.133544e+01 1.133544e+01 1.137885e+01 1.137885e+01 1.145541e+01 1.145541e+01 1.198940e+01 1.201604e+01 1.201604e+01 1.209873e+01 1.209873e+01 1.904225e+01 2.103341e+01 2.105805e+01 2.105805e+01 2.114146e+01 2.114146e+01 2.909364e+01 2.920881e+01 2.920881e+01 3.563321e+01 1.154532e+02 ";
  rhf_test(O,cc_pVQZ,cart,Etot,Eorb,"Oxygen, HF/cc-pVQZ cart");

  // Oxygen, UHF, cc-pVTZ
  Etot=-7.418750e+01;
  Eorba="-2.141749e+01 -1.915988e+00 -1.310965e+00 -1.201094e+00 -5.546067e-01 2.817500e-01 3.202741e-01 3.431319e-01 4.373154e-01 1.136630e+00 1.138794e+00 1.163857e+00 1.186446e+00 1.191993e+00 3.316796e+00 3.383943e+00 3.402246e+00 4.834020e+00 4.840516e+00 4.882378e+00 4.883040e+00 4.924440e+00 4.942217e+00 4.945733e+00 5.904474e+00 5.905271e+00 5.964262e+00 5.981436e+00 5.997036e+00 6.882874e+00 ";
  Eorbb="-2.137179e+01 -1.734349e+00 -1.154756e+00 -5.196519e-01 -4.285624e-01 3.306610e-01 3.561771e-01 4.048465e-01 4.742327e-01 1.189318e+00 1.191518e+00 1.217296e+00 1.241786e+00 1.246929e+00 3.404171e+00 3.425119e+00 3.491997e+00 4.888889e+00 4.896563e+00 4.912510e+00 4.925368e+00 4.931613e+00 4.997593e+00 5.001098e+00 5.984460e+00 5.998798e+00 6.018105e+00 6.078332e+00 6.079842e+00 6.933469e+00 ";
  uhf_test(O,cc_pVTZ,pol,Etot,Eorba,Eorbb,"Oxygen, HF/cc-pVTZ polarized");

  printf("\n");

  // Water
  Etot=-7.605685e+01;
  Eorb="-2.055534e+01 -1.342917e+00 -7.083177e-01 -5.758163e-01 -5.039581e-01 1.418768e-01 2.035184e-01 5.432521e-01 5.974307e-01 6.690261e-01 7.861819e-01 8.026456e-01 8.047746e-01 8.589546e-01 9.569600e-01 1.136901e+00 1.192753e+00 1.523846e+00 1.556896e+00 2.032364e+00 2.059426e+00 2.065381e+00 2.168410e+00 2.235601e+00 2.593081e+00 2.955469e+00 3.361116e+00 3.490653e+00 3.574155e+00 3.646620e+00 3.809889e+00 3.872154e+00 3.880778e+00 3.956919e+00 3.998767e+00 4.073654e+00 4.180179e+00 4.309243e+00 4.376046e+00 4.571595e+00 4.634812e+00 4.859677e+00 5.140229e+00 5.287481e+00 5.546337e+00 6.042405e+00 6.533004e+00 6.901669e+00 6.936545e+00 6.980232e+00 7.018298e+00 7.133879e+00 7.215467e+00 7.225594e+00 7.423558e+00 7.721143e+00 8.269751e+00 1.275531e+01 ";
  rhf_test(h2o,cc_pVTZ,sph,Etot,Eorb,"Water, HF/cc-pVTZ");

  // Direct calculation
  Settings direct=sph;
  direct.set_bool("Direct",1);
  rhf_test(h2o,cc_pVTZ,direct,Etot,Eorb,"Water, HF/cc-pVTZ direct");

  Etot=-7.606449e+01;
  Eorb="-2.056037e+01 -1.346728e+00 -7.128851e-01 -5.800089e-01 -5.076016e-01 1.167178e-01 1.705911e-01 4.488507e-01 4.618804e-01 4.987231e-01 5.826574e-01 6.065213e-01 6.137282e-01 6.541555e-01 7.185927e-01 8.509644e-01 9.188957e-01 1.108944e+00 1.154790e+00 1.348935e+00 1.414543e+00 1.477277e+00 1.484925e+00 1.582442e+00 1.681417e+00 1.906420e+00 2.071943e+00 2.199132e+00 2.284143e+00 2.358650e+00 2.426616e+00 2.484806e+00 2.528228e+00 2.558698e+00 2.578436e+00 2.648050e+00 2.669568e+00 2.842725e+00 2.877561e+00 3.042562e+00 3.125988e+00 3.292035e+00 3.339502e+00 3.442319e+00 3.624332e+00 3.799142e+00 3.996339e+00 4.126279e+00 4.183726e+00 4.212047e+00 4.453009e+00 4.483644e+00 4.683275e+00 4.727338e+00 4.760679e+00 4.914445e+00 5.366310e+00 5.426341e+00 5.990114e+00 6.093699e+00 6.235828e+00 6.303874e+00 6.688010e+00 6.793750e+00 7.071180e+00 7.231019e+00 7.292778e+00 7.330824e+00 7.364649e+00 7.529648e+00 7.617726e+00 7.725976e+00 8.006686e+00 8.088802e+00 8.100575e+00 8.113291e+00 8.186436e+00 8.267147e+00 8.314501e+00 8.318396e+00 8.409891e+00 8.591075e+00 8.909564e+00 8.933914e+00 8.993261e+00 9.155455e+00 9.289979e+00 9.353296e+00 9.908170e+00 1.005875e+01 1.026279e+01 1.043536e+01 1.057525e+01 1.063185e+01 1.075811e+01 1.125961e+01 1.141029e+01 1.157364e+01 1.166789e+01 1.171606e+01 1.184640e+01 1.219255e+01 1.229671e+01 1.241918e+01 1.244116e+01 1.246759e+01 1.357602e+01 1.376268e+01 1.418635e+01 1.455709e+01 1.472017e+01 1.486546e+01 1.642926e+01 1.687004e+01 4.459059e+01 ";
  rhf_test(h2o,cc_pVQZ,sph,Etot,Eorb,"Water, HF/cc-pVQZ");
  // Direct calculation should yield same energies
  rhf_test(h2o,cc_pVQZ,direct,Etot,Eorb,"Water, HF/cc-pVQZ direct");

#ifdef DFT_ENABLED
  Etot=-7.637304e+01;
  Eorb="-1.873915e+01 -9.159507e-01 -4.716228e-01 -3.251310e-01 -2.482871e-01 8.427613e-03 8.112298e-02 3.380516e-01 3.794465e-01 4.650484e-01 5.446214e-01 5.906953e-01 5.968128e-01 6.439666e-01 7.440528e-01 8.849489e-01 9.737601e-01 1.241036e+00 1.260454e+00 1.699817e+00 1.726094e+00 1.761928e+00 1.825371e+00 1.899433e+00 2.186886e+00 2.529953e+00 2.987781e+00 3.124981e+00 3.189164e+00 3.280495e+00 3.290463e+00 3.437137e+00 3.507760e+00 3.569756e+00 3.599794e+00 3.636262e+00 3.687023e+00 3.923959e+00 3.946028e+00 4.157687e+00 4.164052e+00 4.433344e+00 4.649475e+00 4.762326e+00 5.004596e+00 5.488673e+00 5.972774e+00 6.282620e+00 6.290036e+00 6.385120e+00 6.390059e+00 6.555652e+00 6.591888e+00 6.659287e+00 6.808596e+00 7.113568e+00 7.630591e+00 1.191571e+01 ";
  rdft_test(h2o,cc_pVTZ,dftnofit,Etot,Eorb,"Water, PBEPBE/cc-pVTZ no fitting",101,130);

  Etot=-7.637310e+01;
  Eorb="-1.873926e+01 -9.160319e-01 -4.717088e-01 -3.252089e-01 -2.483627e-01 7.657931e-03 8.019473e-02 3.376552e-01 3.790702e-01 4.647614e-01 5.444571e-01 5.902622e-01 5.965674e-01 6.438336e-01 7.439259e-01 8.848180e-01 9.735899e-01 1.240827e+00 1.260327e+00 1.699703e+00 1.725976e+00 1.761539e+00 1.825209e+00 1.899076e+00 2.186727e+00 2.529796e+00 2.987738e+00 3.124887e+00 3.189126e+00 3.280304e+00 3.290358e+00 3.437106e+00 3.507738e+00 3.569698e+00 3.599670e+00 3.636047e+00 3.686840e+00 3.923963e+00 3.945946e+00 4.157683e+00 4.163992e+00 4.433266e+00 4.649433e+00 4.762259e+00 5.004442e+00 5.488644e+00 5.972772e+00 6.282563e+00 6.289931e+00 6.385003e+00 6.390042e+00 6.555642e+00 6.591785e+00 6.659328e+00 6.808586e+00 7.113602e+00 7.630363e+00 1.191559e+01 ";
  rdft_test(h2o,cc_pVTZ,dftsph,Etot,Eorb,"Water, PBEPBE/cc-pVTZ",101,130);
  // This should also give the same energies
  rdft_test(h2o,cc_pVTZ,dftdirect,Etot,Eorb,"Water, PBEPBE/cc-pVTZ direct",101,130);  

  Etot=-7.637468e+01;
  Eorb="-1.874167e+01 -9.179029e-01 -4.734842e-01 -3.274596e-01 -2.505488e-01 2.787613e-03 7.804009e-02 3.238311e-01 3.592367e-01 4.524231e-01 5.141262e-01 5.776675e-01 5.842403e-01 6.425336e-01 6.597412e-01 7.424130e-01 9.718689e-01 1.182204e+00 1.202320e+00 1.575909e+00 1.636077e+00 1.698210e+00 1.724507e+00 1.862853e+00 1.908125e+00 2.164186e+00 2.347307e+00 2.789322e+00 3.004930e+00 3.083164e+00 3.187665e+00 3.232810e+00 3.416909e+00 3.457948e+00 3.505027e+00 3.528260e+00 3.568314e+00 3.577224e+00 3.837898e+00 3.922650e+00 4.086715e+00 4.092685e+00 4.331080e+00 4.415458e+00 4.432276e+00 4.602742e+00 5.126604e+00 5.220077e+00 5.484037e+00 6.149445e+00 6.279956e+00 6.288526e+00 6.350245e+00 6.405876e+00 6.435835e+00 6.657002e+00 6.715284e+00 6.737213e+00 6.939869e+00 7.340654e+00 8.278955e+00 8.355181e+00 9.339051e+00 1.448007e+01 1.582274e+01 ";
  rdft_test(h2o,cc_pVTZ,dftcart,Etot,Eorb,"Water, PBEPBE/cc-pVTZ cart",101,130);

  // Polarized calculation
  Etot=-7.425450e+01;
  Eorba="-1.899994e+01 -1.389811e+00 -9.279117e-01 -7.939669e-01 -7.912421e-01 -3.905831e-01 -3.097197e-01 -2.376932e-01 -2.283773e-01 -1.897660e-01 -1.817343e-01 -1.367744e-01 -1.161584e-01 -1.127449e-01 -7.317887e-02 -6.020582e-02 8.004859e-02 1.246236e-01 1.412522e-01 2.351983e-01 3.122653e-01 4.358942e-01 4.693205e-01 4.838727e-01 6.028244e-01 6.527270e-01 7.533987e-01 8.281906e-01 1.280656e+00 1.287082e+00 1.373462e+00 1.634499e+00 1.657022e+00 1.778617e+00 1.951700e+00 2.155352e+00 2.714152e+00 2.739083e+00 2.760266e+00 3.071203e+00 3.362461e+00 ";
  Eorbb="-1.900379e+01 -1.362775e+00 -9.141688e-01 -7.755015e-01 -7.516922e-01 -3.883419e-01 -3.065892e-01 -2.408169e-01 -2.256310e-01 -1.895145e-01 -1.825064e-01 -1.360620e-01 -1.216288e-01 -1.083224e-01 -7.871656e-02 -6.372931e-02 8.036207e-02 1.266519e-01 1.426582e-01 2.342335e-01 3.094563e-01 4.421994e-01 4.731084e-01 4.878737e-01 6.068900e-01 6.579362e-01 7.570282e-01 8.340061e-01 1.282158e+00 1.286241e+00 1.367518e+00 1.639101e+00 1.658022e+00 1.791091e+00 1.953158e+00 2.165033e+00 2.752282e+00 2.777138e+00 2.781963e+00 3.099384e+00 3.375556e+00 ";
  udft_test(h2o,aug_cc_pVDZ,dftpol,Etot,Eorba,Eorbb,"Water, B3LYP/aug-cc-pVDZ polarized",402,0);
#endif

  printf("\n");
  Etot=-5.663732e+03;
  Eorb="-9.498562e+02 -1.414664e+02 -1.311929e+02 -1.311929e+02 -1.311926e+02 -2.767655e+01 -2.323298e+01 -2.323272e+01 -2.323112e+01 -1.604905e+01 -1.604905e+01 -1.604783e+01 -1.604772e+01 -1.604771e+01 -1.560453e+01 -1.553225e+01 -1.129666e+01 -1.124924e+01 -1.123267e+01 -4.397079e+00 -2.899275e+00 -2.898660e+00 -2.895119e+00 -1.417774e+00 -1.231260e+00 -1.061069e+00 -8.764530e-01 -8.530338e-01 -8.130518e-01 -7.246835e-01 -7.175226e-01 -7.175129e-01 -7.145393e-01 -7.136902e-01 -6.564951e-01 -6.548451e-01 -6.481956e-01 -6.195145e-01 -5.114928e-01 -4.569408e-01 -3.692576e-01 -1.805922e-01 6.931437e-02 7.401136e-02 1.140901e-01 1.499323e-01 1.826698e-01 1.935578e-01 2.119784e-01 2.523714e-01 2.765621e-01 2.853236e-01 3.033661e-01 3.334321e-01 3.368891e-01 3.965296e-01 4.217426e-01 5.489380e-01 5.611364e-01 6.823257e-01 8.854853e-01 9.261582e-01 9.267094e-01 9.632847e-01 9.834670e-01 9.988740e-01 1.036451e+00 1.083441e+00 1.093656e+00 1.198934e+00 1.261767e+00 1.281843e+00 1.319395e+00 1.389594e+00 1.430889e+00 1.470280e+00 1.494533e+00 1.568375e+00 1.582251e+00 1.627153e+00 1.632313e+00 1.670078e+00 1.729453e+00 1.837456e+00 1.946016e+00 1.977961e+00 2.056894e+00 2.244013e+00 2.982936e+00 3.078848e+00 5.275740e+00 2.112179e+02 ";
  rhf_test(cdcplx,b3_21G,cart,Etot,Eorb,"Cadmium complex, HF/3-21G");
#ifdef DFT_ENABLED
  Etot=-5.632262e+03;
  Eorb="-9.337814e+02 -1.355393e+02 -1.258285e+02 -1.258284e+02 -1.258282e+02 -2.512098e+01 -2.099163e+01 -2.099132e+01 -2.099035e+01 -1.437210e+01 -1.437207e+01 -1.437154e+01 -1.437151e+01 -1.437126e+01 -1.373102e+01 -1.366119e+01 -9.659560e+00 -9.634095e+00 -9.617624e+00 -3.458171e+00 -2.150194e+00 -2.149490e+00 -2.147612e+00 -9.639814e-01 -8.133654e-01 -6.817701e-01 -5.775482e-01 -5.562555e-01 -5.224229e-01 -4.291027e-01 -4.203120e-01 -4.031336e-01 -3.807796e-01 -3.717032e-01 -3.599132e-01 -3.599045e-01 -3.592276e-01 -3.568260e-01 -2.834441e-01 -2.757981e-01 -2.321579e-01 -1.298306e-01 -5.211204e-02 -4.524458e-02 -3.877380e-02 -2.182186e-02 -1.163751e-02 8.210258e-03 4.205393e-02 6.842280e-02 7.935417e-02 1.029258e-01 1.053816e-01 1.358824e-01 1.471807e-01 1.767494e-01 2.054324e-01 3.137195e-01 3.216120e-01 4.129698e-01 5.962713e-01 5.971708e-01 6.208457e-01 6.574267e-01 6.732689e-01 6.816609e-01 6.943079e-01 7.800087e-01 8.098804e-01 8.781911e-01 9.031421e-01 9.414527e-01 9.704216e-01 9.964499e-01 1.042417e+00 1.146223e+00 1.161414e+00 1.228233e+00 1.246515e+00 1.275367e+00 1.281942e+00 1.319289e+00 1.356222e+00 1.487541e+00 1.600562e+00 1.621270e+00 1.701360e+00 1.838430e+00 2.594089e+00 2.690597e+00 4.712081e+00 2.086366e+02 ";
  rdft_test(cdcplx,b3_21G,dftcart,Etot,Eorb,"Cadmium complex, B3LYP/3-21G",402,0);
#endif

  printf("****** Tests complete in %s *******\n",t.elapsed().c_str());
  
  return 0;
}
