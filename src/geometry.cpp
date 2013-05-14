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
#include "dftfuncs.h"
#include "elements.h"
#include "emd/emd.h"
#include "find_molecules.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
#include "scf.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"

#include <armadillo>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
#include <gsl/gsl_multimin.h>
}

// Optimization helpers
typedef struct {
  // Atoms in the system
  std::vector<atom_t> atoms;
  // Basis set library
  BasisSetLibrary baslib;
  // Settings used
  Settings set;
} opthelper_t;

enum minimizer {
  // Fletcher-Reeves conjugate gradients
  CGFR,
  // Polak-Ribiere conjugate gradient
  CGPR,
  // Broyden-Fletcher-Goldfarb-Shanno
  BFGS,
  // Broyden-Fletcher-Goldfarb-Shanno, second version
  BFGS2,
  // Steepest descent
  SD
};

// Convergence criteria
enum convergence {
  LOOSE,
  NORMAL,
  TIGHT,
  VERYTIGHT
};

std::vector<atom_t> get_atoms(const gsl_vector * x, const std::vector<atom_t> orig) {
  // Update atomic positions
  std::vector<atom_t> atoms(orig);
  for(size_t i=0;i<x->size/3;i++) {
    atoms[i].x=gsl_vector_get(x,3*i);
    atoms[i].y=gsl_vector_get(x,3*i+1);
    atoms[i].z=gsl_vector_get(x,3*i+2);
  }

  return atoms;
}

double calc_E(const gsl_vector *x, void *par) {
  // Get the helpers
  opthelper_t *p=(opthelper_t *) par;

  // Get the atomic positions
  std::vector<atom_t> atoms=get_atoms(x,p->atoms);

  // Construct basis set
  BasisSet basis=construct_basis(atoms,p->baslib,p->set);

  // Perform the electronic structure calculation
  calculate(basis,p->set,false);

  // Solution checkpoint
  Checkpoint solchk(p->set.get_string("SaveChk"),false);

  // Current energy is
  energy_t en;
  solchk.read(en);

  return en.E;
}

void calc_Ef(const gsl_vector *x, void *par, double *E, gsl_vector *g) {
  // Get the helpers
  opthelper_t *p=(opthelper_t *) par;

  // Get the atomic positions
  std::vector<atom_t> atoms=get_atoms(x,p->atoms);

  // Construct basis set
  BasisSet basis=construct_basis(atoms,p->baslib,p->set);

  // Perform the electronic structure calculation
  calculate(basis,p->set,true);

  // Solution checkpoint
  Checkpoint solchk(p->set.get_string("SaveChk"),false);

  // Energy
  energy_t en;
  solchk.read(en);
  *E=en.E;

  // Force
  arma::vec f;
  solchk.read("Force",f);

  // Set components
  size_t N=atoms.size();
  for(size_t i=0;i<N;i++) {
    gsl_vector_set(g,3*i  ,-f(3*i));
    gsl_vector_set(g,3*i+1,-f(3*i+1));
    gsl_vector_set(g,3*i+2,-f(3*i+2));
  }
}

void calc_f(const gsl_vector *x, void *par, gsl_vector *g) {
  double E;
  calc_Ef(x,par,&E,g);
}

void get_displacement(const std::vector<atom_t> & g, const std::vector<atom_t> & o, double & dmax, double & drms) {
  if(g.size()!=o.size()) {
    ERROR_INFO();
    throw std::runtime_error("Cannot compare different geometries.\n");
  }

  dmax=0.0;
  drms=0.0;

  // Compute displacements
  for(size_t i=0;i<g.size();i++) {
    double dsq=pow(g[i].x-o[i].x,2) + pow(g[i].y-o[i].y,2) + pow(g[i].z-o[i].z,2);

    drms+=dsq;
    if(dsq>dmax)
      dmax=dsq;
  }

  // Take square roots here
  drms=sqrt(drms/g.size());
  dmax=sqrt(dmax);
}

void get_forces(const gsl_vector *g, double & fmax, double & frms) {
  fmax=0.0;
  frms=0.0;

  for(size_t i=0;i<g->size/3;i++) {
    double x=gsl_vector_get(g,3*i);
    double y=gsl_vector_get(g,3*i+1);
    double z=gsl_vector_get(g,3*i+2);

    double fsq=x*x + y*y + z*z;

    frms+=fsq;
    if(fsq>fmax)
      fmax=fsq;
  }

  frms=sqrt(3*frms/g->size);
  fmax=sqrt(fmax);
}


int main(int argc, char **argv) {

#ifdef _OPENMP
  printf("ERKALE - Geometry optimization from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Geometry optimization from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();

  if(argc!=2) {
    printf("Usage: $ %s runfile\n",argv[0]);
    return 0;
  }

  // Initialize libint
  init_libint_base();
  // Initialize libderiv
  init_libderiv_base();

  Timer t;
  t.print_time();

  // Parse settings
  Settings set;
  set.add_scf_settings();
  set.add_string("SaveChk","File to use as checkpoint","erkale.chk");
  set.add_string("LoadChk","File to load old results from","");
  set.add_bool("ForcePol","Force polarized calculation",false);
  set.add_bool("FreezeCore","Freeze the atomic cores?",false);
  set.add_string("Optimizer","Optimizer to use: CGFR, CGPR, BFGS, BFGS2 (default), SD","BFGS2");
  set.add_int("MaxSteps","Maximum amount of geometry steps",256);
  set.add_string("Criterion","Convergence criterion to use: LOOSE, NORMAL, TIGHT, VERYTIGHT","NORMAL");
  set.parse(std::string(argv[1]));

  bool verbose=set.get_bool("Verbose");
  int maxiter=set.get_int("MaxSteps");

  // Interpret optimizer
  enum minimizer alg;
  std::string method=set.get_string("Optimizer");
  if(stricmp(method,"CGFR")==0)
    alg=CGFR;
  else if(stricmp(method,"CGPR")==0)
    alg=CGPR;
  else if(stricmp(method,"BFGS")==0)
    alg=BFGS;
  else if(stricmp(method,"BFGS2")==0)
    alg=BFGS2;
  else if(stricmp(method,"SD")==0)
    alg=SD;
  else {
    ERROR_INFO();
    throw std::runtime_error("Unknown optimization method.\n");
  }

  // Interpret optimizer
  enum convergence crit;
  method=set.get_string("Criterion");
  if(stricmp(method,"LOOSE")==0)
    crit=LOOSE;
  else if(stricmp(method,"NORMAL")==0)
    crit=NORMAL;
  else if(stricmp(method,"TIGHT")==0)
    crit=TIGHT;
  else if(stricmp(method,"VERYTIGHT")==0)
    crit=VERYTIGHT;
  else {
    ERROR_INFO();
    throw std::runtime_error("Unknown optimization method.\n");
  }

  // Redirect output?
  std::string logfile=set.get_string("Logfile");
  if(stricmp(logfile,"stdout")!=0) {
    // Redirect stdout to file
    FILE *outstream=freopen(logfile.c_str(),"w",stdout);
    if(outstream==NULL) {
      ERROR_INFO();
      throw std::runtime_error("Unable to redirect output!\n");
    } else
      fprintf(stderr,"\n");
  }

  // Read in atoms.
  std::string atomfile=set.get_string("System");
  const std::vector<atom_t> origgeom=load_xyz(atomfile);
  std::vector<atom_t> atoms(origgeom);

  // Read in basis set
  BasisSetLibrary baslib;
  std::string basfile=set.get_string("Basis");
  baslib.load_gaussian94(basfile);
  printf("\n");

  // Save to output
  save_xyz(atoms,"Initial configuration","optimize.xyz",false);

  // Minimizer options
  opthelper_t pars;
  pars.atoms=atoms;
  pars.baslib=baslib;
  pars.set=set;

  // GSL stuff
  int status;

  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;

  gsl_vector *x;
  gsl_multimin_function_fdf minimizer;

  minimizer.n = 3*atoms.size();
  minimizer.f = calc_E;
  minimizer.df = calc_f;
  minimizer.fdf = calc_Ef;
  minimizer.params = (void *) &pars;

  /* Starting point */
  x = gsl_vector_alloc (minimizer.n);
  for(size_t i=0;i<atoms.size();i++) {
    gsl_vector_set(x,3*i,atoms[i].x);
    gsl_vector_set(x,3*i+1,atoms[i].y);
    gsl_vector_set(x,3*i+2,atoms[i].z);
  }

  switch(alg) {

  case(CGFR):
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    if(verbose) printf("Using Fletcher-Reeves conjugate gradients.\n");
    break;

  case(CGPR):
    T = gsl_multimin_fdfminimizer_conjugate_pr;
    if(verbose) printf("Using Polak-Ribière conjugate gradients.\n");
    break;

  case(BFGS):
    T = gsl_multimin_fdfminimizer_vector_bfgs;
    if(verbose) printf("Using the BFGS minimizer.\n");
    break;

  case(BFGS2):
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    if(verbose) printf("Using the BFGS2 minimizer.\n");
    break;

  case(SD):
    T = gsl_multimin_fdfminimizer_steepest_descent;
    if(verbose) printf("Using the steepest descent minimizer.\n");
    break;
  }

  // Run an initial calculation
  double oldE=calc_E(x,minimizer.params);
  // Turn off verbose setting
  pars.set.set_bool("Verbose",false);
  // and load from old checkpoint
  pars.set.set_string("LoadChk",pars.set.get_string("SaveChk"));

  s = gsl_multimin_fdfminimizer_alloc (T, minimizer.n);

  gsl_multimin_fdfminimizer_set (s, &minimizer, x, 0.01, 1e-4);

  fprintf(stderr,"Geometry optimizer initialized in %s.\n",t.elapsed().c_str());
  fprintf(stderr,"Entering minimization loop with %s optimizer.\n",set.get_string("Optimizer").c_str());

  fprintf(stderr,"%5s %13s %10s %9s %9s %9s %9s %s\n","iter","E","deltaE","disp max","disp rms","f max","f rms", "titer");

  std::vector<atom_t> oldgeom(atoms);

  for(int iter=1;iter<=maxiter;iter++) {
    Timer titer;
    t.set();

    status = gsl_multimin_fdfminimizer_iterate (s);

    if (status) {
      fprintf(stderr,"GSL encountered error: \"%s\".\n",gsl_strerror(status));
      break;
    }

    // New geometry is
    std::vector<atom_t> geom=get_atoms(s->x,pars.atoms);

    // Calculate displacements
    double dmax, drms;
    get_displacement(geom, oldgeom, dmax, drms);

    // Switch geometries
    oldgeom=geom;

    // Get forces
    double fmax, frms;
    get_forces(s->gradient, fmax, frms);

    // Save geometry step
    char comment[80];
    sprintf(comment,"Step %i",(int) iter);
    save_xyz(get_atoms(s->x,pars.atoms),comment,"optimize.xyz",true);

    fprintf (stderr,"%5d % 08.8f % .3e %.3e %.3e %.3e %.3e %s\n", (int) iter, s->f, s->f - oldE, dmax, drms, fmax, frms, titer.elapsed().c_str());
    fflush(stderr);

    // Check convergence
    bool convd=false;

    switch(crit) {

    case(LOOSE):
      if((fmax < 2.5e-3) && (frms < 1.7e-3) && (dmax < 1.0e-2) && (drms < 6.7e-3))
	convd=true;

    case(NORMAL):
      if((fmax < 4.5e-4) && (frms < 3.0e-4) && (dmax < 1.8e-3) && (drms < 1.2e-3))
	convd=true;

    case(TIGHT):
      if((fmax < 1.5e-5) && (frms < 1.0e-5) && (dmax < 6.0e-5) && (drms < 4.0e-5))
	convd=true;

    case(VERYTIGHT):
      if((fmax < 2.0e-6) && (frms < 1.0e-6) && (dmax < 6.0e-6) && (drms < 4.0e-6))
	convd=true;
    }

    if(convd) {
      fprintf(stderr,"Converged.\n");
      break;
    }

    // Store old energy
    oldE=s->f;
  }

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);

  printf("Running program took %s.\n",t.elapsed().c_str());

  return 0;
}
