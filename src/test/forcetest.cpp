/*
 * End-to-end check of the analytic SCF nuclear gradient against a finite
 * difference of the converged total energy.
 *
 * The SCF energy is variational in the orbitals, so its total derivative
 * along a nuclear displacement equals the analytic force projected on it:
 *
 *   dE/dlambda (R0 + lambda d) = -F . d
 *
 * Each case converges the SCF at R0 (with forces) and at R0 +- h d, where d
 * is a fixed direction without molecular symmetry, so every force component
 * enters the check with two extra SCF runs per case.
 *
 * Unlike cdforcetest, which exercises the DensityFit kernels directly, this
 * goes through calculate() and therefore also checks how the SCF combines
 * the J/K force contributions for each spin and method. The unrestricted
 * cases caught two bugs: the density-fitted / Cholesky exchange force of
 * each spin channel was taken with the restricted normalization and came
 * out at half its true value, and the four-index short-range exchange force
 * of range-separated hybrids also included a short-range Coulomb force.
 *
 * Usage: forcetest   (needs ERKALE_LIBRARY pointing to the basis library)
 */

#include "../basis.h"
#include "../basislibrary.h"
#include "../checkpoint.h"
#include "../scf.h"
#include "../settings.h"
#include "../xyzutils.h"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

Settings settings;

struct forcecase_t {
  std::string method;
  std::string jkmethod;
  int charge;
  int mult;
  // Allowed deviation |dE/dlambda + F.d| in Eh/bohr
  double tol;
};

// Distorted water, coordinates in bohr
static std::vector<atom_t> geometry(const arma::vec & x) {
  static const char * els[]={"O","H","H"};
  std::vector<atom_t> atoms(3);
  for(size_t i=0;i<atoms.size();i++) {
    atoms[i].el=els[i];
    atoms[i].num=i;
    atoms[i].x=x(3*i);
    atoms[i].y=x(3*i+1);
    atoms[i].z=x(3*i+2);
    atoms[i].Q=0;
  }
  return atoms;
}

static double run_scf(const arma::vec & x, const BasisSetLibrary & baslib, const std::string & load, const std::string & save, bool force, arma::vec & f) {
  BasisSet basis;
  construct_basis(basis,geometry(x),baslib);

  settings.set_string("LoadChk",load);
  settings.set_string("SaveChk",save);
  calculate(basis,force);

  Checkpoint chk(save,false);
  energy_t en;
  chk.read(en);
  if(force)
    chk.read("Force",f);
  return en.E;
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;

  settings.add_scf_settings();
  settings.add_string("SaveChk","File to use as checkpoint","erkale.chk");
  settings.add_string("LoadChk","File to load old results from","");
  settings.add_bool("ForcePol","Force polarized calculation",false);
  settings.set_bool("Verbose",false);
  settings.set_string("Basis","def2-SVP");
  settings.set_string("FittingBasis","def2-universal-jkfit");
  settings.set_double("ConvThr",1e-9);
  // Cholesky decomposition close to exact so that its energy is smooth
  settings.set_double("CholeskyThr",1e-10);
  // The forces omit the derivative of the DFT quadrature weights, which is
  // ~1e-5 Eh/bohr on the default grid; a dense grid takes it below 1e-7.
  settings.set_string("DFTGrid","150 -974");

  BasisSetLibrary baslib;
  baslib.load_basis(settings.get_string("Basis"));

  // Reference geometry in bohr
  arma::vec x0={0.00, 0.00, 0.12,
                1.45, 0.08, -0.95,
                -1.38, 0.11, -1.02};
  // Displacement direction
  arma::vec d={0.31, -0.12, 0.44,
               -0.53, 0.27, -0.18,
               0.22, -0.41, -0.29};
  d/=arma::norm(d,2);
  const double h=1e-3;

  // The RI energy carries ~1e-7 Eh of roundoff from the conditioning of the
  // fitting metric, which limits the finite difference accuracy to ~1e-4.
  // The bugs this guards against are orders of magnitude larger.
  const double tol=1e-6, tolri=2e-4;
  const std::vector<forcecase_t> cases={
    {"HF",               "4index",   0, 1, tol},
    {"HF",               "4index",   1, 2, tol},
    {"HF",               "Cholesky", 0, 1, tol},
    {"HF",               "Cholesky", 1, 2, tol},
    {"HF",               "RI",       0, 1, tolri},
    {"HF",               "RI",       1, 2, tolri},
    {"hyb_gga_xc_b3lyp", "Cholesky", 1, 2, tol},
    {"hyb_gga_xc_b3lyp", "RI",       1, 2, tolri},
    {"hyb_gga_xc_wb97x", "4index",   0, 1, tol},
    {"hyb_gga_xc_wb97x", "4index",   1, 2, tol},
    {"hyb_gga_xc_wb97x", "Cholesky", 1, 2, tol},
    {"hyb_gga_xc_wb97x", "RI",       1, 2, tolri},
  };



  int nfail=0;
  printf("%-18s %-9s %3s %4s %14s %14s %10s\n","method","jk","Q","mult","-F.d","dE/dlambda","error");
  for(size_t ic=0;ic<cases.size();ic++) {
    const forcecase_t & c(cases[ic]);
    settings.set_string("Method",c.method);
    settings.set_string("JKMethod",c.jkmethod);
    settings.set_int("Charge",c.charge);
    settings.set_int("Multiplicity",c.mult);

    const std::string chk0="forcetest_0.chk", chkp="forcetest_p.chk", chkm="forcetest_m.chk";
    arma::vec f, fdum;
    run_scf(x0,baslib,"",chk0,true,f);
    const double Ep=run_scf(x0+h*d,baslib,chk0,chkp,false,fdum);
    const double Em=run_scf(x0-h*d,baslib,chk0,chkm,false,fdum);
    std::remove(chk0.c_str());
    std::remove(chkp.c_str());
    std::remove(chkm.c_str());

    const double analytic=-arma::dot(f,d);
    const double numeric=(Ep-Em)/(2.0*h);
    const double err=std::abs(analytic-numeric);
    const bool ok=err<c.tol;
    if(!ok)
      nfail++;
    printf("%-18s %-9s %3i %4i % 14.8f % 14.8f %10.3e %s\n",c.method.c_str(),c.jkmethod.c_str(),c.charge,c.mult,analytic,numeric,err,ok ? "OK" : "FAIL");
    fflush(stdout);
  }

  if(nfail) {
    printf("%i force checks failed.\n",nfail);
    return 1;
  }
  printf("All force checks passed.\n");
  return 0;
}
