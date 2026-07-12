/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "basislibrary.h"
#include "basis.h"
#include "checkpoint.h"
#include "dftgrid.h"
#include "elements.h"
#include "find_molecules.h"
#include "guess.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
#include "sap.h"
#include "settings.h"
#include "neo_particle.h"
#include "stringutil.h"
#include "timer.h"

#include "eriworker.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SVNRELEASE
#include "version.h"
#endif

void print_H(const arma::mat & H, int num=10) {
  size_t N=std::min(num,(int) H.n_rows);
  for(size_t i=0;i<N;i++) {
    for(size_t j=0;j<N;j++)
      printf(" % .6f",H(i,j));
    printf("\n");
  }
}

void print_header() {
#ifdef _OPENMP
  printf("ERKALE - HF/DFT from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - HF/DFT from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

Settings settings;

#define BFIDX(ie, ip) {((ip)*e_nbf+ie)}
#define MOIDX(ie, ip) {((ip)*e_nmo+ie)}

void analyze_density(const BasisSet & basis, const arma::mat & P, const std::string & label) {
  std::vector<arma::mat> dip(basis.moment(1));
  std::vector<arma::mat> quad(basis.moment(2));

  // Compute expected position
  std::vector<double> dipval(dip.size());
  double dipsq=0.0;
  for(size_t i=0;i<dip.size();i++) {
    dipval[i]=arma::trace(P*dip[i]);
    dipsq += dipval[i]*dipval[i];
  }

  // Compute <(r-<r>)^2> = <r^2> - <r>^2
  arma::mat Rsqmat = quad[getind(0,0,2)]+quad[getind(0,2,0)]+quad[getind(2,0,0)];
  double rsq = arma::trace(P*Rsqmat);

  printf("Expected position for %s wave packet: % .6f % .6f % .6f\n",label.c_str(),dipval[0],dipval[1],dipval[2]);
  printf("Root-mean-square size of %s wave packet % .6f\n",label.c_str(),sqrt(rsq-dipsq));
}

// External one-body confining trap on the quantum proton, together with the
// Cartesian moment integrals it is assembled from. The same moments serve the
// per-state protonic moment analysis, so they are formed exactly once and about
// exactly one center. See README_TRAP.md and README_MOMENTS.md.
struct proton_trap_t {
  /// Is the trap active (NEOTrap)?
  bool enabled;
  /// Trap frequencies, in Hartree
  double wpar, wperp;
  /// Dimensionless quartic anharmonicity, and the z^2 (x^2+y^2) coefficient
  double g, lam_cross;
  /// Trap center, in bohr
  double x0, y0, z0;
  /// Cartesian moments about the trap center: M[getind(a,b,c)] is
  /// < mu | (x-x0)^a (y-y0)^b (z-z0)^c | nu >, of total degree 2 resp. 4.
  /// Empty unless the trap or the moment analysis needs them.
  std::vector<arma::mat> M2, M4;
  /// One-body trap matrix in the protonic AO basis; zero matrix if !enabled
  arma::mat V;
};

// Build the trap operator
//   V(r) = 1/2 m [ w_perp^2 (x^2+y^2) + w_par^2 z^2 ]
//        + lam_par z^4 + lam_perp (x^2+y^2)^2 + lam_cross z^2 (x^2+y^2),
// with lam_par = g m^2 w_par^3, lam_perp = g m^2 w_perp^3 from the dimensionless
// anharmonicity g. V is a zero matrix when NEOTrap is off (so TrapG=0 / trap-off
// reproduce the baseline bit-for-bit). The moment integrals are also formed when
// want_moments is set, even with the trap off, so that the protonic moments can
// be reported for an untrapped proton.
static proton_trap_t build_proton_trap(const BasisSet & pbasis, double proton_mass, bool want_moments) {
  proton_trap_t tr;
  tr.enabled = settings.get_bool("NEOTrap");
  tr.wpar = tr.wperp = tr.g = tr.lam_cross = 0.0;
  tr.x0 = tr.y0 = tr.z0 = 0.0;
  tr.V.zeros(pbasis.get_Nbf(), pbasis.get_Nbf());
  if(!tr.enabled && !want_moments)
    return tr;

  // Trap center: explicit TrapCenter, else the protonic nucleus center. This
  // is also the origin the reported protonic moments refer to.
  const std::string tc = settings.get_string("TrapCenter");
  if(tc.size()) {
    std::vector<std::string> tok = splitline(tc);
    if(tok.size() != 3)
      throw std::runtime_error("TrapCenter needs three numbers: x y z (bohr).\n");
    tr.x0=readdouble(tok[0]); tr.y0=readdouble(tok[1]); tr.z0=readdouble(tok[2]);
  } else {
    std::vector<nucleus_t> nuc = pbasis.get_nuclei();
    if(nuc.size() != 1)
      throw std::runtime_error("NEOTrap v1 supports a single quantum proton; set TrapCenter for a multi-proton trap.\n");
    tr.x0=nuc[0].r.x; tr.y0=nuc[0].r.y; tr.z0=nuc[0].r.z;
  }

  tr.M2 = pbasis.moment(2, tr.x0, tr.y0, tr.z0);
  tr.M4 = pbasis.moment(4, tr.x0, tr.y0, tr.z0);

  if(!tr.enabled) {
    printf("Protonic moments referenced to (% .4f % .4f % .4f).\n", tr.x0, tr.y0, tr.z0);
    fflush(stdout);
    return tr;
  }

  // Trap frequencies (convert cm^-1 -> Hartree if requested).
  tr.wpar  = settings.get_double("TrapOmegaPar");
  tr.wperp = settings.get_double("TrapOmegaPerp");
  const std::string unit = settings.get_string("TrapOmegaUnit");
  if(stricmp(unit, "cm-1") == 0) {
    const double cm2Eh = 4.5563352529e-6;
    tr.wpar  *= cm2Eh;
    tr.wperp *= cm2Eh;
  } else if(stricmp(unit, "Eh") != 0) {
    throw std::runtime_error("TrapOmegaUnit must be 'cm-1' or 'Eh'.\n");
  }

  // Quartic couplings from the dimensionless anharmonicity g.
  tr.g = settings.get_double("TrapG");
  const double m2 = proton_mass*proton_mass;
  const double lam_par   = tr.g * m2 * tr.wpar*tr.wpar*tr.wpar;
  const double lam_perp  = tr.g * m2 * tr.wperp*tr.wperp*tr.wperp;
  tr.lam_cross = settings.get_double("TrapLambdaCross");

  tr.V += 0.5*proton_mass*tr.wperp*tr.wperp * (tr.M2[getind(2,0,0)] + tr.M2[getind(0,2,0)]);
  tr.V += 0.5*proton_mass*tr.wpar *tr.wpar  *  tr.M2[getind(0,0,2)];
  tr.V += lam_par      *  tr.M4[getind(0,0,4)];
  tr.V += lam_perp     * (tr.M4[getind(4,0,0)] + 2.0*tr.M4[getind(2,2,0)] + tr.M4[getind(0,4,0)]);
  tr.V += tr.lam_cross * (tr.M4[getind(2,0,2)] + tr.M4[getind(0,2,2)]);

  // The moment integrals are symmetric; enforce it exactly against round-off.
  tr.V = 0.5*(tr.V + tr.V.t());

  printf("NEO proton trap: w_par = %.6e Eh, w_perp = %.6e Eh, g = %.5f, lam_cross = %.4e a.u.; center (% .4f % .4f % .4f).\n",
         tr.wpar, tr.wperp, tr.g, tr.lam_cross, tr.x0, tr.y0, tr.z0);
  fflush(stdout);
  return tr;
}

/// Protonic density moments of a single CI root, in bohr^2 / bohr^4
struct pmoments_t {
  double x2, y2, z2, x2py2;
  double x4, y4, z4, x2y2, x2z2, y2z2;
};

/// Per-root CI data reported by the multi-root analysis
struct ciroot_t {
  /// Total energy and excitation energy from the ground root, in Hartree
  double E, gap;
  /// Tr[D S_p], which must be unity for a normalized single-proton root
  double trace;
  /// Protonic density moments; zero unless the moments were requested
  pmoments_t mom;
};

/// Expectation value Tr[D M] of a symmetric operator over a symmetric density
static double expval(const arma::mat & D, const arma::mat & M) {
  return arma::accu(D % M);
}

/// Protonic density moments of the AO-basis protonic density matrix Dao
static pmoments_t proton_moments(const arma::mat & Dao, const proton_trap_t & tr) {
  pmoments_t m;
  m.x2 = expval(Dao, tr.M2[getind(2,0,0)]);
  m.y2 = expval(Dao, tr.M2[getind(0,2,0)]);
  m.z2 = expval(Dao, tr.M2[getind(0,0,2)]);
  m.x2py2 = m.x2 + m.y2;
  m.x4 = expval(Dao, tr.M4[getind(4,0,0)]);
  m.y4 = expval(Dao, tr.M4[getind(0,4,0)]);
  m.z4 = expval(Dao, tr.M4[getind(0,0,4)]);
  m.x2y2 = expval(Dao, tr.M4[getind(2,2,0)]);
  m.x2z2 = expval(Dao, tr.M4[getind(2,0,2)]);
  m.y2z2 = expval(Dao, tr.M4[getind(0,2,2)]);
  return m;
}

// Protonic one-particle density matrix of a CI root, in the protonic AO basis.
// The CI vector c_{i,I} runs over (electronic MO) x (protonic MO) products, so
// tracing out the electron gives the protonic RDM in the protonic MO basis,
//   D^p_{IJ} = sum_i c_{i,I} c_{i,J},
// which the protonic MO coefficients back-transform to the AO basis. For a
// normalized root Tr[D^p] = 1, hence Tr[D^AO S_p] = 1.
static arma::mat proton_rdm(const arma::vec & ci, size_t e_nmo, size_t p_nmo, const arma::mat & Xp_bo) {
  arma::mat civec(ci);
  civec.reshape(e_nmo, p_nmo);
  return Xp_bo * (civec.t()*civec) * Xp_bo.t();
}

// State-averaging weights over the nr lowest roots. An empty SAWeights means
// equal weights; a shorter list zero-pads (those roots simply do not
// contribute). Weights are renormalized to sum to unity.
static arma::vec sa_weights(size_t nr) {
  const std::string s = settings.get_string("SAWeights");
  arma::vec w;
  if(s.size()) {
    std::vector<std::string> tok = splitline(s);
    if(tok.size() > nr)
      throw std::runtime_error("SAWeights has more entries than there are roots.\n");
    w.zeros(nr);
    for(size_t i=0;i<tok.size();i++)
      w(i) = readdouble(tok[i]);
  } else
    w.ones(nr);

  if(arma::any(w < 0.0))
    throw std::runtime_error("SAWeights must be non-negative.\n");
  const double sum = arma::sum(w);
  if(sum <= 0.0)
    throw std::runtime_error("SAWeights must have a positive sum.\n");
  return w/sum;
}

/// Strip the directory-preserving extension off a file name
static std::string strip_extension(const std::string & fname) {
  size_t slash = fname.find_last_of("/\\");
  size_t dot = fname.find_last_of('.');
  if(dot != std::string::npos && (slash == std::string::npos || dot > slash))
    return fname.substr(0, dot);
  return fname;
}

/// Write the moments of a single state into the JSON sidecar
static void write_moments_json(FILE *out, const pmoments_t & m, bool full) {
  fprintf(out, ", \"z2\": %.17g, \"x2py2\": %.17g, \"z4\": %.17g", m.z2, m.x2py2, m.z4);
  if(full)
    fprintf(out, ",\n     \"x2\": %.17g, \"y2\": %.17g, \"x4\": %.17g, \"y4\": %.17g,"
            " \"x2y2\": %.17g, \"x2z2\": %.17g, \"y2z2\": %.17g",
            m.x2, m.y2, m.x4, m.y4, m.x2y2, m.x2z2, m.y2z2);
}

// Machine-readable sidecar with the trap parameters, the per-root data and the
// state average, so that the sweep driver parses data rather than prose.
// Energies in Hartree, moments in bohr^2 / bohr^4.
static void write_props(const std::string & fname, const proton_trap_t & tr,
                        const std::vector<ciroot_t> & roots, bool have_mom, bool full,
                        bool stateavg, const arma::vec & w, double Eavg,
                        const pmoments_t & mavg, const std::string & densfile) {
  FILE *out = fopen(fname.c_str(), "w");
  if(!out)
    throw std::runtime_error("Could not open " + fname + " for writing.\n");

  fprintf(out, "{\n");
  fprintf(out, "  \"trap\": {\"enabled\": %s, \"omega_par_au\": %.17g, \"omega_perp_au\": %.17g,\n"
          "            \"g\": %.17g, \"lambda_cross_au\": %.17g, \"center\": [%.17g, %.17g, %.17g]},\n",
          tr.enabled ? "true" : "false", tr.wpar, tr.wperp, tr.g, tr.lam_cross, tr.x0, tr.y0, tr.z0);

  fprintf(out, "  \"roots\": [\n");
  for(size_t i=0;i<roots.size();i++) {
    fprintf(out, "    {\"index\": %i, \"E\": %.17g, \"gap\": %.17g, \"trace\": %.17g",
            (int) i, roots[i].E, roots[i].gap, roots[i].trace);
    if(have_mom)
      write_moments_json(out, roots[i].mom, full);
    fprintf(out, "}%s\n", (i+1<roots.size()) ? "," : "");
  }
  fprintf(out, "  ]");

  if(stateavg) {
    fprintf(out, ",\n  \"state_average\": {\"weights\": [");
    for(size_t i=0;i<w.n_elem;i++)
      fprintf(out, "%.17g%s", w(i), (i+1<w.n_elem) ? ", " : "");
    fprintf(out, "], \"E\": %.17g", Eavg);
    if(have_mom)
      write_moments_json(out, mavg, full);
    if(densfile.size())
      fprintf(out, ", \"density_file\": \"%s\"", densfile.c_str());
    fprintf(out, "}");
  }
  fprintf(out, "\n}\n");
  fclose(out);

  printf("Wrote machine-readable properties to %s\n", fname.c_str());
  fflush(stdout);
}

int main_guarded(int argc, char **argv) {
  print_header();

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }


  Timer t;
  t.print_time();

  settings.add_scf_settings();
  settings.add_string("ProtonBasis", "Protonic basis set", "");
  add_particle_settings(settings);
  settings.add_int("Verbosity", "Verboseness level", 5);
  settings.add_bool("H2", "Run H2+ instead of H atom?", false);
  settings.add_double("H2BondLength", "Bond length for H2+ in a.u.", 2.0);
  settings.add_bool("RemoveCOM", "Remove COM terms", false);
  // External one-body confining trap on the quantum proton (see README_TRAP.md)
  settings.add_bool("NEOTrap", "Apply an external harmonic + quartic trap to the quantum proton", false);
  settings.add_string("TrapOmegaUnit", "Unit of the trap frequencies: cm-1 or Eh", "cm-1");
  settings.add_double("TrapOmegaPar", "Trap frequency along the z axis (w_par)", 0.0);
  settings.add_double("TrapOmegaPerp", "Trap frequency perpendicular to z (w_perp)", 0.0);
  settings.add_double("TrapG", "Dimensionless quartic anharmonicity g (sets lam_par, lam_perp)", 0.0);
  // lam_cross is an independent coefficient of either sign: it tilts the quartic
  // between prolate and oblate, and only the total V need stay confining.
  settings.add_double("TrapLambdaCross", "Coefficient of the z^2 (x^2+y^2) quartic, a.u.", 0.0, true);
  settings.add_string("TrapCenter", "Trap center 'x y z' in bohr (empty = quantum proton center)", "");
  // Multi-root analysis, protonic moments and state averaging (see README_MOMENTS.md)
  settings.add_int("NRoots", "Number of CI roots to solve, store and print", 1);
  settings.add_bool("PrintMoments", "Print the per-state protonic moments <z^2>, <x^2+y^2>, <z^4>", false);
  settings.add_bool("MomentsFull", "Also print the full order-2 and order-4 protonic moment set", false);
  settings.add_bool("StateAverage", "Report the weighted state-averaged energy and moments", false);
  settings.add_string("SAWeights", "Per-root state-averaging weights, renormalized (empty = equal)", "");
  settings.add_bool("SADensityOut", "Write out the state-averaged protonic 1-RDM in the AO basis", false);

  // Parse settings
  settings.parse(std::string(argv[1]),true);
  settings.print();

  // Get parameters
  double intthr=settings.get_double("IntegralThresh");
  bool verbose=settings.get_bool("Verbose");
  const quantum_particle_t particle = get_particle(settings);
  const double proton_mass = particle.m;
  const double proton_charge = particle.q;
  print_particle(particle);
  bool dimer = settings.get_bool("H2");
  double R = settings.get_double("H2BondLength");
  bool removecom = settings.get_bool("RemoveCOM");

  // Multi-root analysis. Defaults (NRoots 1, no moments, no state average)
  // reproduce the pre-existing output exactly.
  int nroots = settings.get_int("NRoots");
  bool printmom = settings.get_bool("PrintMoments");
  bool momfull = settings.get_bool("MomentsFull");
  bool stateavg = settings.get_bool("StateAverage");
  bool sadens = settings.get_bool("SADensityOut");
  if(nroots < 1)
    throw std::runtime_error("NRoots must be at least one.\n");
  // The averaged density is defined by the state-averaging weights, so asking
  // for it asks for the state average.
  if(sadens && !stateavg) {
    stateavg = true;
    printf("SADensityOut implies StateAverage.\n");
    fflush(stdout);
  }
  // Moment integrals are only formed if something is going to consume them.
  bool want_moments = printmom || stateavg;
  bool report = (nroots > 1) || printmom || stateavg;

  // Total mass of the system
  double MT = 1+proton_mass;

  // Read in basis sets
  BasisSetLibrary baslib;
  baslib.load_basis(settings.get_string("Basis"));
  BasisSetLibrary pbaslib;
  if(settings.get_string("ProtonBasis").size())
    pbaslib.load_basis(settings.get_string("ProtonBasis"));
  else {
    // Automatic basis. Get element
    ElementBasisSet orbbas(baslib.get_element("H"));
    orbbas.decontract();

    // New basis
    ElementBasisSet pbas("H", 0);
    // Loop over angular momentum
    for(int am=0;am<=orbbas.get_max_am();am++) {
      arma::vec exps;
      arma::mat coeffs;
      orbbas.get_primitives(exps,coeffs,am);

      // Scale exponents
      exps *= sqrt(proton_mass);

      for(auto expn: exps) {
        FunctionShell fn(am);
        fn.add_exponent(1.0,expn);
        pbas.add_function(fn);
      }
    }
    pbaslib.add_element(pbas);

    printf("Autogenerated protonic basis by scaling exponents by %e\n",sqrt(proton_mass));
  }

  std::vector<atom_t> atoms(1);
  atoms[0].el="H";
  atoms[0].num=0;
  atoms[0].x=atoms[0].y=atoms[0].z=0.0;
  atoms[0].Q=0;
  std::vector<atom_t> protons(atoms);

  if(dimer) {
    atoms.push_back(atoms[0]);
    atoms[1].num=1;
    atoms[1].z=R;
    printf("Placed second atom at %e bohr distance\n",R);
  }

  // Construct the orbital basis sets
  BasisSet basis;
  construct_basis(basis,atoms,baslib);
  BasisSet pbasis;
  construct_basis(pbasis,protons,pbaslib);

  printf("%i electronic and %i protonic basis functions\n", basis.get_Nbf(), pbasis.get_Nbf());
  printf("Hamiltonian is %i x %i\n", basis.get_Nbf()*pbasis.get_Nbf(), basis.get_Nbf()*pbasis.get_Nbf());

  // Get the overlap matrices
  arma::mat Se(basis.overlap());
  arma::mat Sp(pbasis.overlap());

  // Get gradient integrals
  std::vector<arma::mat> grade(basis.gradient_integral());
  std::vector<arma::mat> gradp(pbasis.gradient_integral());

  // Kinetic energy matrices
  arma::mat T, Tp;
  if(removecom) {
    T = (1.0-1.0/MT) * basis.kinetic();
    Tp = (1.0/proton_mass - 1.0/MT)* pbasis.kinetic();
  } else {
    T = basis.kinetic();
    Tp = pbasis.kinetic()/proton_mass;
  }

  // Nuclear attraction
  arma::mat V(T.n_rows, T.n_cols, arma::fill::zeros);
  arma::mat Vp(Tp.n_rows, Tp.n_cols, arma::fill::zeros);
  if(dimer) {
    std::vector<std::tuple<int, double, double, double>> classical_nuclei(1,std::make_tuple(1,atoms[1].x,atoms[1].y,atoms[1].z));
    // Classical nucleus is attractive for electrons
    V = basis.nuclear(classical_nuclei);
    // and repulsive for protons
    // The electronic routines give the potential felt by a particle of
    // charge -1
    Vp = -proton_charge*pbasis.nuclear(classical_nuclei);
  }

  // External confining trap on the quantum proton (one-body). Added to the
  // proton core Hamiltonian, so both the BO proton spectrum and the CI (which
  // read H0p below) see it. Zero unless NEOTrap is enabled.
  proton_trap_t trap = build_proton_trap(pbasis, proton_mass, want_moments);

  // Core Hamiltonian
  arma::mat H0 = T + V;
  arma::mat H0p = Tp + Vp + trap.V;

  // Orthogonalizing matrices
  arma::mat Xe(BasOrth(Se,verbose));
  arma::mat Xp(BasOrth(Sp,verbose));

  // Solve the BO problem
  //arma::mat H_bo = Xe.t() * (T + basis.nuclear()) * Xe;
  arma::mat H_bo = Xe.t() * H0 * Xe;
  arma::vec E_bo;
  arma::mat C_bo;
  arma::eig_sym(E_bo, C_bo, H_bo);
  arma::mat Xe_bo = Xe * C_bo;
  E_bo.print("Electronic spectrum without quantum proton");

  // Solve the nuclear problem
  arma::mat Hp_bo = Xp.t() * H0p * Xp;
  arma::vec Ep_bo;
  arma::mat Cp_bo;
  arma::eig_sym(Ep_bo, Cp_bo, Hp_bo);
  arma::mat Xp_bo = Xp * Cp_bo;
  Ep_bo.print("Quantum proton spectrum");
  // Low protonic roots at full precision (parseable by the trap sweep / tests).
  for(size_t i=0;i<std::min((size_t) 8, (size_t) Ep_bo.n_elem);i++) {
    printf("Proton root %2i % .10e\n", (int) i, Ep_bo(i));
    fflush(stdout);
  }

  // Compute the two-electron integrals
  size_t e_nbf = basis.get_Nbf();
  size_t p_nbf = pbasis.get_Nbf();
  size_t e_nmo = Xe.n_cols;
  size_t p_nmo = Xp.n_cols;

  // Shells
  std::vector<GaussianShell> eshells=basis.get_shells();
  std::vector<GaussianShell> pshells=pbasis.get_shells();

  // Compute shell pairs
  double omega=0.0;
  double alpha=1.0;
  double beta=0.0;
  ScreeningData e_scr=basis.compute_screening(intthr,omega,alpha,beta,verbose);
  ScreeningData p_scr=pbasis.compute_screening(intthr,omega,alpha,beta,verbose);
  const arma::mat & Qe = e_scr.Q;
  const arma::mat & Qp = p_scr.Q;
  const std::vector<eripair_t> & epairs = e_scr.shpairs;
  const std::vector<eripair_t> & ppairs = p_scr.shpairs;

  int max_am = std::max(basis.get_max_am(), pbasis.get_max_am());
  int max_Ncontr = std::max(basis.get_max_Ncontr(), pbasis.get_max_Ncontr());

  // Form AO matrix of proton-electron integrals
  arma::mat V_ao(e_nbf*p_nbf,e_nbf*p_nbf,arma::fill::zeros);
#pragma omp parallel
  {
    ERIWorker eri(max_am, max_Ncontr);

#pragma omp for collapse(2)
    for(size_t ep=0; ep<epairs.size(); ep++)
      for(size_t pp=0; pp<ppairs.size(); pp++) {
        // Shells are
        size_t is = epairs[ep].is;
        size_t js = epairs[ep].js;
        size_t Is = ppairs[pp].is;
        size_t Js = ppairs[pp].js;

        // Check screening
        double QQ = Qe(is,js)*Qp(Is,Js);
        if(QQ<intthr)
          continue;

        // Start and end
        size_t N_i = eshells[is].get_Nbf();
        size_t i_start = eshells[is].get_first_ind();

        size_t N_j = eshells[js].get_Nbf();
        size_t j_start = eshells[js].get_first_ind();

        size_t N_I = pshells[Is].get_Nbf();
        size_t I_start = pshells[Is].get_first_ind();

        size_t N_J = pshells[Js].get_Nbf();
        size_t J_start = pshells[Js].get_first_ind();

        // Compute the integrals
        eri.compute(&eshells[is],&eshells[js],&pshells[Is],&pshells[Js]);
        const std::vector<double> & eris=eri.rget();

        // Store the integrals in the array
        for(size_t ii=0;ii<N_i;ii++)
          for(size_t jj=0;jj<N_j;jj++)
            for(size_t II=0;II<N_I;II++)
              for(size_t JJ=0;JJ<N_J;JJ++) {
                size_t i = i_start+ii;
                size_t j = j_start+jj;
                size_t I = I_start+II;
                size_t J = J_start+JJ;

                // The product of the charges of the electron and of
                // the quantum particle
                double element = -proton_charge*eris[((ii*N_j+jj)*N_I+II)*N_J+JJ];
                // (ij|IJ)
                V_ao(BFIDX(i,I),BFIDX(j,J)) = element;
                // (ij|JI)
                V_ao(BFIDX(i,J),BFIDX(j,I)) = element;
                // (ji|IJ)
                V_ao(BFIDX(j,I),BFIDX(i,J)) = element;
                // (ji|JI)
                V_ao(BFIDX(j,J),BFIDX(i,I)) = element;
              }
      }
  }
  printf("Finished e-p integrals\n");
  fflush(stdout);

  if(removecom) {
    for(size_t i=0; i<e_nbf; i++)
      for(size_t j=0; j<e_nbf; j++)
        for(size_t I=0; I<p_nbf; I++)
          for(size_t J=0; J<p_nbf; J++) {
            double el = 0.0;
            for(size_t ic=0; ic<3; ic++)
              el += grade[ic](i,j) * gradp[ic](I,J);
            V_ao(BFIDX(j,J),BFIDX(i,I)) += 1.0/MT * el;
          }
    printf("Finished nabla.nabla terms\n");
    fflush(stdout);
  }

  // Compute transformation matrix to go to normalized AO basis
  arma::mat ao_to_orth(e_nbf*p_nbf,e_nmo*p_nmo,arma::fill::zeros);
#pragma omp parallel for collapse(4)
  for(size_t u=0; u<e_nbf; u++)
    for(size_t i=0; i<e_nmo; i++)
      for(size_t U=0; U<p_nbf; U++)
        for(size_t I=0; I<p_nmo; I++) {
          ao_to_orth(BFIDX(u,U), MOIDX(i,I)) = Xe_bo(u,i) * Xp_bo(U,I);
        }
  printf("Finished forming ao to orthogonal transformation\n");
  fflush(stdout);

  arma::mat V_ci = ao_to_orth.t() * V_ao * ao_to_orth;
  printf("Formed orthogonal Hamiltonian\n");
  fflush(stdout);

  // Add in one-particle terms
  arma::mat H0_mo = Xe_bo.t() * H0 * Xe_bo;
  arma::mat H0p_mo = Xp_bo.t() * H0p * Xp_bo;

  arma::mat H0_ci(V_ci.n_rows, V_ci.n_cols, arma::fill::zeros);
  for(size_t i=0; i<e_nmo; i++)
    for(size_t j=0; j<e_nmo; j++)
      for(size_t I=0; I<p_nmo; I++) {
        H0_ci(MOIDX(i,I), MOIDX(j,I)) += H0_mo(i,j);
      }
  for(size_t i=0; i<e_nmo; i++)
    for(size_t I=0; I<p_nmo; I++) {
      for(size_t J=0; J<p_nmo; J++)
        H0_ci(MOIDX(i,I), MOIDX(i,J)) += H0p_mo(I,J);
      }

  arma::mat H_ci = H0_ci + V_ci;

  arma::vec E;
  arma::mat C;
  arma::eig_sym(E,C,H_ci);
  //E.print("Eigenvalues");

  // Compute electronic and protonic density matrices
  arma::mat civec(C.col(0));
  civec.reshape(e_nmo, p_nmo);
  //civec.print("CI vector");

  arma::mat Pe = -civec*civec.t();
  arma::mat Pp = -civec.t()*civec;

  arma::vec noon_e, noon_p;
  arma::mat no_e, no_p;
  arma::eig_sym(noon_e, no_e, Pe);
  arma::eig_sym(noon_p, no_p, Pp);
  noon_e *= -1;
  noon_p *= -1;

  // Go back to AO basis
  Pe = -Xe_bo * Pe * Xe_bo.t();
  Pp = -Xp_bo * Pp * Xp_bo.t();

  noon_e.print("Electronic natural occupations");
  noon_p.print("Protonic   natural occupations");
  printf("CI energy % .15f\n",E(0));

  double Enuc_CI = arma::as_scalar(C.col(0).t() * V_ci * C.col(0));
  double Ekine=arma::trace(Pe*T);
  double Ekinp=arma::trace(Pp*Tp);
  double Enucel=arma::trace(Pe*V);
  double Enucnuc=arma::trace(Pp*Vp);
  double Enuc=arma::trace(Pp*Tp);
  double Ekin=Ekine+Ekinp;
  printf("Electronic kinetic energy (dm) %.9f\n", Ekine);
  printf("Protonic   kinetic energy (dm) %.9f\n", Ekinp);
  printf("Total      kinetic energy (dm) %.9f\n", Ekin);
  if(dimer) {
    printf("Electron-classical proton energy (dm) %.9f\n", Enucel);
    printf("Quantum-classical proton energy (dm) %.9f\n", Enucnuc);
  }
  printf("Electron-proton Coulomb energy %.9f\n", Enuc_CI);
  printf("Virial ratio %e\n",-E(0)/Ekin);

  printf("\n");
  analyze_density(basis, Pe, "electronic");
  analyze_density(pbasis, Pp, " protonic ");

  // Multi-root analysis: per-root energies, gaps and protonic density moments,
  // plus their weighted state average.
  //
  // The CI is full for these 1e + 1p systems, so the single diagonalization
  // above already determines every root and its density. FCI roots and their
  // densities are invariant to the choice of reference orbitals, so a
  // state-averaged orbital optimization would not change any number below;
  // "state averaging" here is purely the post-diagonalization weighted average
  // over the CI roots. (Were the CI ever truncated instead of full, SA orbitals
  // would then matter -- out of scope.)
  if(report) {
    size_t nr = std::min((size_t) nroots, (size_t) E.n_elem);
    if(nr < (size_t) nroots) {
      printf("\nOnly %i roots exist in this CI space; reducing NRoots from %i.\n", (int) nr, nroots);
      fflush(stdout);
    }

    arma::vec w = sa_weights(nr);

    // Per-root protonic densities and moments, accumulating the state-averaged
    // density as we go. Averaging the moments and contracting the averaged
    // density give the same numbers -- the contraction is linear in D.
    std::vector<ciroot_t> roots(nr);
    arma::mat Davg(Sp.n_rows, Sp.n_cols, arma::fill::zeros);
    pmoments_t mavg = {};
    double Eavg = 0.0;
    for(size_t I=0;I<nr;I++) {
      arma::mat Dao = proton_rdm(C.col(I), e_nmo, p_nmo, Xp_bo);
      roots[I].E = E(I);
      roots[I].gap = E(I) - E(0);
      roots[I].trace = expval(Dao, Sp);
      roots[I].mom = want_moments ? proton_moments(Dao, trap) : pmoments_t();
      Eavg += w(I)*E(I);
      Davg += w(I)*Dao;
    }
    if(want_moments)
      mavg = proton_moments(Davg, trap);

    printf("\n root         E (Eh)             gap (Eh)         Tr[D S_p]");
    if(printmom)
      printf("        <z^2>         <x^2+y^2>        <z^4>");
    printf("\n");
    fflush(stdout);
    for(size_t I=0;I<nr;I++) {
      printf(" %4i  % .12f  % .12f    %.8f", (int) I, roots[I].E, roots[I].gap, roots[I].trace);
      if(printmom)
        printf("  %14.8e %14.8e %14.8e", roots[I].mom.z2, roots[I].mom.x2py2, roots[I].mom.z4);
      printf("\n");
      fflush(stdout);
    }

    if(printmom && momfull) {
      // <x^2> == <y^2> holds per state only for a nondegenerate state of a
      // cylindrical trap; within a degenerate pi manifold only the sum over the
      // manifold is invariant, since the CI vectors there are basis dependent.
      printf("\n root        <x^2>          <y^2>          <x^4>          <y^4>"
             "        <x^2 y^2>      <x^2 z^2>      <y^2 z^2>\n");
      fflush(stdout);
      for(size_t I=0;I<nr;I++) {
        const pmoments_t & m = roots[I].mom;
        printf(" %4i %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e %14.8e\n",
               (int) I, m.x2, m.y2, m.x4, m.y4, m.x2y2, m.x2z2, m.y2z2);
        fflush(stdout);
      }
    }

    const std::string jobname = strip_extension(argv[1]);
    std::string densfile;
    if(stateavg) {
      printf("\nState average over %i roots, %s weights:", (int) nr,
             settings.get_string("SAWeights").size() ? "SAWeights" : "equal");
      for(size_t I=0;I<nr;I++)
        printf(" %.6f", w(I));
      printf("\n  E_avg = % .12f\n", Eavg);
      if(printmom)
        printf("  <z^2>_avg = %14.8e  <x^2+y^2>_avg = %14.8e  <z^4>_avg = %14.8e\n",
               mavg.z2, mavg.x2py2, mavg.z4);
      fflush(stdout);
    }
    if(sadens) {
      // The state-averaged protonic density in the AO basis: the natural target
      // density for the protonic basis-set optimizer.
      densfile = jobname + ".sadens.dat";
      Davg.save(densfile, arma::raw_ascii);
      printf("Wrote the state-averaged protonic 1-RDM (AO basis) to %s\n", densfile.c_str());
      fflush(stdout);
    }

    write_props(jobname + ".props.json", trap, roots, want_moments,
                momfull, stateavg, w, Eavg, mavg, densfile);
  }

  printf("\nRunning program took %s.\n",t.elapsed().c_str());

  return 0;
}

int main(int argc, char **argv) {
#ifdef CATCH_EXCEPTIONS
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
#else
  return main_guarded(argc, argv);
#endif
}
