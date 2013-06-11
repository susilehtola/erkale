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
  // Indices of dofs
  std::vector<size_t> dofidx;
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

double calculate_projection(const std::vector<atom_t> & g, const std::vector<atom_t> & o, const arma::mat & f, const std::vector<size_t> & dofidx) {
  double dE=0.0;
  for(size_t i=0;i<g.size();i++) {
    size_t j=dofidx[i];
    dE+=f(i,0)*(g[j].x-o[j].x) + f(i,1)*(g[j].y-o[j].y) + f(i,2)*(g[j].z-o[j].z);
  }
  return dE;
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

std::vector<atom_t> get_atoms(const gsl_vector * x, const opthelper_t & opts) {
  // Update atomic positions
  std::vector<atom_t> atoms(opts.atoms);

  for(size_t i=0;i<opts.dofidx.size();i++) {
    size_t j=opts.dofidx[i];
    atoms[j].x=gsl_vector_get(x,3*i);
    atoms[j].y=gsl_vector_get(x,3*i+1);
    atoms[j].z=gsl_vector_get(x,3*i+2);
  }

  return atoms;
}

enum calcd {
  // Full calculation
  FULLCALC,
  // Just the forces
  FORCECALC,
  // Nothing
  NOCALC
};

enum calcd run_calc(const BasisSet & basis, Settings & set, bool force) {
  // Checkpoint file to load
  std::string loadname=set.get_string("LoadChk");

  if(stricmp(loadname,"")==0) {
    // Nothing to load - run full calculation.
    calculate(basis,set,force);
    return FULLCALC;
  }

  // Otherwise, check if the basis set is the same
  BasisSet oldbas;
  bool convd;
  {
    Checkpoint chkpt(loadname,false);
    
    chkpt.read("Converged",convd);
    
    chkpt.read(oldbas);
  }

  // Copy the checkpoint if necessary
  std::string savename=set.get_string("SaveChk");
  if(stricmp(savename,loadname)!=0) {
    // Copy the file
    std::ostringstream oss;
    oss << "cp " << loadname << " " << savename;
    int cp=system(oss.str().c_str());
    if(cp) {
      ERROR_INFO();
      throw std::runtime_error("Failed to copy checkpoint file.\n");
    }
  }
  
  if(!(basis==oldbas) || !convd) {
    // Basis set is different or calculation was not converged - need
    // to run calculation. Project Fock matrix to new basis and
    // diagonalize to get new orbitals
    {
      // Open in write mode, don't truncate
      Checkpoint chkpt(savename,true,false);

      // Restricted or unrestricted run?
      bool restr;
      chkpt.read("Restricted",restr);
      chkpt.read(oldbas);

      if(restr) {
	rscf_t sol;
	std::vector<double> occs;
	chkpt.read("H",sol.H);
	chkpt.read("occs",occs);

	// Overlap matrix
	arma::mat S, Sinvh;
	chkpt.read("S",S);
	chkpt.read("Sinvh",Sinvh);

	// Form new orbitals
	diagonalize(S,Sinvh,sol);

	// Form new density
	sol.P=form_density(sol.C,occs);

	// Write basis set, orbitals and density
	chkpt.write(basis);
	chkpt.write("C",sol.C);
	chkpt.write("E",sol.E);
	chkpt.write("P",sol.P);

      } else {
	uscf_t sol;
	std::vector<double> occa, occb;
	chkpt.read("Ha",sol.Ha);
	chkpt.read("Hb",sol.Hb);
	chkpt.read("occa",occa);
	chkpt.read("occb",occb);

	// Overlap matrix
	arma::mat S, Sinvh;
	chkpt.read("S",S);
	chkpt.read("Sinvh",Sinvh);

	// Form new orbitals
	diagonalize(S,Sinvh,sol);

	// Form new density
	sol.Pa=form_density(sol.Ca,occa);
	sol.Pb=form_density(sol.Cb,occb);
	sol.P=sol.Pa+sol.Pb;

	// Write basis set, orbitals and density
	chkpt.write(basis);
	chkpt.write("Ca",sol.Ca);
	chkpt.write("Cb",sol.Cb);
	chkpt.write("Ea",sol.Ea);
	chkpt.write("Eb",sol.Eb);
	chkpt.write("Pa",sol.Pa);
	chkpt.write("Pb",sol.Pb);
	chkpt.write("P",sol.P);
      }

      // Calculation is now not converged
      bool conv=false;
      chkpt.write("Converged",conv);
    }
    
    // Now we can do the SCF calculation
    calculate(basis,set,force);
    return FULLCALC;
  }

  if(!force)
    return NOCALC;

  // Open the checkpoint in write mode, don't truncate it
  Checkpoint chkpt(savename,true,false);

  // SCF structure
  SCF solver(basis,set,chkpt);

  // Do a plain Hartree-Fock calculation?
  bool hf= (stricmp(set.get_string("Method"),"HF")==0);
  bool rohf=(stricmp(set.get_string("Method"),"ROHF")==0);

  // Get exchange and correlation functionals
  dft_t dft;
  if(!hf && !rohf) {
    parse_xc_func(dft.x_func,dft.c_func,set.get_string("Method"));
    dft.gridtol=0.0;
    
    if(stricmp(set.get_string("DFTGrid"),"Auto")!=0) {
      std::vector<std::string> opts=splitline(set.get_string("DFTGrid"));
      if(opts.size()!=2) {
        throw std::runtime_error("Invalid DFT grid specified.\n");
      }

      dft.adaptive=false;
      dft.nrad=readint(opts[0]);
      dft.lmax=readint(opts[1]);
      if(dft.nrad<1 || dft.lmax<1) {
        throw std::runtime_error("Invalid DFT grid specified.\n");
      }
    } else {
      dft.adaptive=true;
      dft.gridtol=set.get_double("DFTFinalTol");
    }
  }

  // Force
  arma::vec f;

  double tol;
  chkpt.read("tol",tol);

  // Multiplicity
  int mult=set.get_int("Multiplicity");
  // Amount of electrons
  int Nel=basis.Ztot()-set.get_int("Charge");

  if(mult==1 && Nel%2==0 && !set.get_bool("ForcePol")) {
    rscf_t sol;
    chkpt.read("C",sol.C);
    chkpt.read("E",sol.E);
    chkpt.read("H",sol.H);
    chkpt.read("P",sol.P);

    std::vector<double> occs;
    chkpt.read("occs",occs);

    // Restricted case
    if(hf) {
      f=solver.force_RHF(sol,occs,FINETOL);
    } else {
      DFTGrid grid(&basis,set.get_bool("Verbose"),set.get_bool("DFTLobatto"));

      // Fixed grid?
      if(dft.adaptive)
	grid.construct(sol.P,dft.gridtol,dft.x_func,dft.c_func);
      else
	grid.construct(dft.nrad,dft.lmax,dft.x_func,dft.c_func);
      f=solver.force_RDFT(sol,occs,dft,grid,FINETOL);
    }
    
  } else {

    uscf_t sol;
    chkpt.read("Ca",sol.Ca);
    chkpt.read("Ea",sol.Ea);
    chkpt.read("Ha",sol.Ha);
    chkpt.read("Pa",sol.Pa);

    chkpt.read("Cb",sol.Cb);
    chkpt.read("Eb",sol.Eb);
    chkpt.read("Hb",sol.Hb);
    chkpt.read("Pb",sol.Pb);
    chkpt.read("P",sol.P);

    std::vector<double> occa, occb;
    chkpt.read("occa",occa);
    chkpt.read("occb",occb);

    if(hf) {
      f=solver.force_UHF(sol,occa,occb,tol);
    } else if(rohf) {
      int Nela, Nelb;
      chkpt.read("Nel-a",Nela);
      chkpt.read("Nel-b",Nelb);
      
      throw std::runtime_error("Not implemented!\n");
      //      f=solver.force_ROHF(sol,Nela,Nelb,tol);
    } else {
      DFTGrid grid(&basis,set.get_bool("Verbose"),set.get_bool("DFTLobatto"));

      // Fixed grid?
      if(dft.adaptive)
	grid.construct(sol.P,dft.gridtol,dft.x_func,dft.c_func);
      else
	grid.construct(dft.nrad,dft.lmax,dft.x_func,dft.c_func);
      f=solver.force_UDFT(sol,occa,occb,dft,grid,tol);
    }
  }

  chkpt.write("Force",f);
  chkpt.close();

  return FORCECALC;
}


double calc_E(const gsl_vector *x, void *par) {
  Timer t;

  // Get the helpers
  opthelper_t *p=(opthelper_t *) par;

  // Get the atomic positions
  std::vector<atom_t> atoms=get_atoms(x,*p);

  // Construct basis set
  BasisSet basis=construct_basis(atoms,p->baslib,p->set);

  // Perform the electronic structure calculation
  enum calcd mode=run_calc(basis,p->set,false);

  // Solution checkpoint
  Checkpoint solchk(p->set.get_string("SaveChk"),false);

  // Current energy is
  energy_t en;
  solchk.read(en);

  if(mode==FULLCALC)
    printf("Computed energy % .8f in %s.\n",en.E,t.elapsed().c_str());
  else
    printf("Found    energy % .8f in checkpoint file.\n",en.E);
  fflush(stdout);
  //  print_xyz(atoms);

  return en.E;
}

void calc_Ef(const gsl_vector *x, void *par, double *E, gsl_vector *g) {
  Timer t;

  // Get the helpers
  opthelper_t *p=(opthelper_t *) par;

  // Get the atomic positions
  std::vector<atom_t> atoms=get_atoms(x,*p);

  // Construct basis set
  BasisSet basis=construct_basis(atoms,p->baslib,p->set);

  // Perform the electronic structure calculation
  enum calcd mode=run_calc(basis,p->set,true);

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
  for(size_t i=0;i<p->dofidx.size();i++) {
    size_t j=p->dofidx[i];
    gsl_vector_set(g,3*i  ,-f(3*j));
    gsl_vector_set(g,3*i+1,-f(3*j+1));
    gsl_vector_set(g,3*i+2,-f(3*j+2));
  }

  double frms, fmax;
  get_forces(g,fmax,frms);

  if(mode==FULLCALC)
    printf("Computed energy % .8f and forces (max = %.3e, rms = %.3e) in %s.\n",en.E,fmax,frms,t.elapsed().c_str());
  else if(mode==FORCECALC)
    printf("Found    energy % .8f in checkpoint file and calculated forces (max = %.3e, rms = %.3e) in %s.\n",en.E,fmax,frms,t.elapsed().c_str());
  else if(mode==NOCALC)
    printf("Found energy and forces in checkpoint file.\n");
  fflush(stdout);

  fflush(stdout);
  //  print_xyz(atoms);
}

void calc_f(const gsl_vector *x, void *par, gsl_vector *g) {
  double E;
  calc_Ef(x,par,&E,g);
}

arma::mat interpret_force(const gsl_vector *x) {
  arma::mat r(x->size/3,3);
  for(size_t i=0;i<x->size/3;i++) {
    r(i,0)=gsl_vector_get(x,3*i);
    r(i,1)=gsl_vector_get(x,3*i+1);
    r(i,2)=gsl_vector_get(x,3*i+2);
  }
  return r;
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

  Timer tprog;
  tprog.print_time();

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
  set.add_string("OptMovie","xyz movie to store progress in","optimize.xyz");
  set.parse(std::string(argv[1]));

  bool verbose=set.get_bool("Verbose");
  int maxiter=set.get_int("MaxSteps");
  std::string optmovie=set.get_string("OptMovie");

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

  // Are any atoms fixed?
  std::vector<size_t> dofidx;
  for(size_t i=0;i<atoms.size();i++) {
    bool fixed=false;

    if(atoms[i].el.size()>3)
      if(stricmp(atoms[i].el.substr(atoms[i].el.size()-3),"-Fx")==0) {
	fixed=true;
	atoms[i].el=atoms[i].el.substr(0,atoms[i].el.size()-3);
      }

    // Add to degrees of freedom
    if(!fixed)
      dofidx.push_back(i);
  }

  // Read in basis set
  BasisSetLibrary baslib;
  std::string basfile=set.get_string("Basis");
  baslib.load_gaussian94(basfile);
  printf("\n");

  // Save to output
  save_xyz(atoms,"Initial configuration",optmovie,false);

  // Minimizer options
  opthelper_t pars;
  pars.atoms=atoms;
  pars.baslib=baslib;
  pars.set=set;
  pars.dofidx=dofidx;

  /* Starting point */
  gsl_vector *x = gsl_vector_alloc (3*dofidx.size());
  for(size_t i=0;i<dofidx.size();i++) {
    gsl_vector_set(x,3*i,atoms[dofidx[i]].x);
    gsl_vector_set(x,3*i+1,atoms[dofidx[i]].y);
    gsl_vector_set(x,3*i+2,atoms[dofidx[i]].z);
  }

  // GSL status
  int status;

  const gsl_multimin_fdfminimizer_type *T;
  gsl_multimin_fdfminimizer *s;

  gsl_multimin_function_fdf minimizer;

  minimizer.n = x->size;
  minimizer.f = calc_E;
  minimizer.df = calc_f;
  minimizer.fdf = calc_Ef;
  minimizer.params = (void *) &pars;

  if(alg==CGFR) {
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    if(verbose) printf("Using Fletcher-Reeves conjugate gradients.\n");
  } else if(alg==CGPR) {
    T = gsl_multimin_fdfminimizer_conjugate_pr;
    if(verbose) printf("Using Polak-Ribière conjugate gradients.\n");
  } else if(alg==BFGS) {
    T = gsl_multimin_fdfminimizer_vector_bfgs;
    if(verbose) printf("Using the BFGS minimizer.\n");
  } else if(alg==BFGS2) {
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    if(verbose) printf("Using the BFGS2 minimizer.\n");
  } else if(alg==SD) {
    T = gsl_multimin_fdfminimizer_steepest_descent;
    if(verbose) printf("Using the steepest descent minimizer.\n");
  } else {
    ERROR_INFO();
    throw std::runtime_error("Unsupported minimizer\n");
  }

  // Run an initial calculation
  double oldE=calc_E(x,minimizer.params);

  // Turn off verbose setting
  pars.set.set_bool("Verbose",false);
  // and load from old checkpoint
  pars.set.set_string("LoadChk",pars.set.get_string("SaveChk"));

  // Initialize minimizer
  s = gsl_multimin_fdfminimizer_alloc (T, minimizer.n);

  // Use initial step length of 0.02 bohr, and a line search accuracy
  // 1e-1 (recommended in the GSL manual for BFGS)
  gsl_multimin_fdfminimizer_set (s, &minimizer, x, 0.02, 1e-1);

  // Store old force
  arma::mat oldf=interpret_force(s->gradient);

  fprintf(stderr,"Geometry optimizer initialized in %s.\n",tprog.elapsed().c_str());
  fprintf(stderr,"Entering minimization loop with %s optimizer.\n",set.get_string("Optimizer").c_str());

  fprintf(stderr,"%4s %16s %10s %10s %9s %9s %9s %9s %s\n","iter","E","dE","dE/dEproj","disp max","disp rms","f max","f rms", "titer");

  std::vector<atom_t> oldgeom(atoms);

  bool convd=false;
  int iter;

  for(iter=1;iter<=maxiter;iter++) {
    printf("\nGeometry iteration %i\n",(int) iter);
    fflush(stdout);

    Timer titer;

    status = gsl_multimin_fdfminimizer_iterate (s);

    if (status) {
      fprintf(stderr,"GSL encountered error: \"%s\".\n",gsl_strerror(status));
      break;
    }

    // New geometry is
    std::vector<atom_t> geom=get_atoms(s->x,pars);

    // Calculate displacements
    double dmax, drms;
    get_displacement(geom, oldgeom, dmax, drms);

    // Calculate projected change of energy
    double dEproj=calculate_projection(geom,oldgeom,oldf,pars.dofidx);
    // Actual change of energy is
    double dE=s->f - oldE;

    // Switch geometries
    oldgeom=geom;
    // Save old force

    // Get forces
    double fmax, frms;
    get_forces(s->gradient, fmax, frms);

    // Save geometry step
    char comment[80];
    sprintf(comment,"Step %i",(int) iter);
    save_xyz(get_atoms(s->x,pars),comment,optmovie,true);

    // Check convergence
    bool fmaxconv=false, frmsconv=false;
    bool dmaxconv=false, drmsconv=false;

    switch(crit) {

    case(LOOSE):
      if(fmax < 2.5e-3)
	fmaxconv=true;
      if(frms < 1.7e-3)
	frmsconv=true;
      if(dmax < 1.0e-2)
	dmaxconv=true;
      if(drms < 6.7e-3)
	drmsconv=true;
      break;

    case(NORMAL):
      if(fmax < 4.5e-4)
	fmaxconv=true;
      if(frms < 3.0e-4)
	frmsconv=true;
      if(dmax < 1.8e-3)
	dmaxconv=true;
      if(drms < 1.2e-3)
	drmsconv=true;
      break;

    case(TIGHT):
      if(fmax < 1.5e-5)
	fmaxconv=true;
      if(frms < 1.0e-5)
	frmsconv=true;
      if(dmax < 6.0e-5)
	dmaxconv=true;
      if(drms < 4.0e-5)
	drmsconv=true;
      break;

    case(VERYTIGHT):
      if(fmax < 2.0e-6)
	fmaxconv=true;
      if(frms < 1.0e-6)
	frmsconv=true;
      if(dmax < 6.0e-6)
	dmaxconv=true;
      if(drms < 4.0e-6)
	drmsconv=true;
      break;

    default:
      ERROR_INFO();
      throw std::runtime_error("Not implemented!\n");
    }

    // Converged?
    const static char cconv[]=" *";

    fprintf(stderr,"%4d % 16.8f % .3e % .3e %.3e%c %.3e%c %.3e%c %.3e%c %s\n", (int) iter, s->f, dE, dE/dEproj, dmax, cconv[dmaxconv], drms, cconv[drmsconv], fmax, cconv[fmaxconv], frms, cconv[frmsconv], titer.elapsed().c_str());
    fflush(stderr);

    convd=dmaxconv && drmsconv && fmaxconv && frmsconv;

    if(convd) {
      fprintf(stderr,"Converged.\n");
      break;
    }

    // Store old energy
    oldE=s->f;
    // Store old force
    oldf=interpret_force(s->gradient);
  }

  gsl_multimin_fdfminimizer_free (s);

  gsl_vector_free (x);

  if(iter==maxiter && !convd) {
    printf("Geometry convergence was not achieved!\n");
  }


  printf("Running program took %s.\n",tprog.elapsed().c_str());

  return 0;
}
