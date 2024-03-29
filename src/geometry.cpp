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
#include "dftgrid.h"
#include "elements.h"
#include "emd/emd.h"
#include "find_molecules.h"
#include "lebedev.h"
#include "linalg.h"
#include "mathf.h"
#include "xyzutils.h"
#include "properties.h"
#include "scf.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"
#include "lbfgs.h"

// Needed for libint init
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

// Settings
Settings settings;

// Optimization helpers
typedef struct {
  // Atoms in the system
  std::vector<atom_t> atoms;
  // Basis set library
  BasisSetLibrary baslib;
  // Indices of dofs
  std::vector<size_t> dofidx;

  // Numeric gradient?
  bool numgrad;
  // Step size for numeric gradient
  double step;
  // Order of stencil
  int npoints;
} opthelper_t;

enum minimizer {
  // Fletcher-Reeves conjugate gradients
  gCGFR,
  // Polak-Ribiere conjugate gradient
  gCGPR,
  // Broyden-Fletcher-Goldfarb-Shanno
  gBFGS,
  // Steepest descent
  gSD
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

double calculate_projection(const std::vector<atom_t> & g, const std::vector<atom_t> & o, const arma::vec & f, const std::vector<size_t> & dofidx) {
  double dE=0.0;
  for(size_t i=0;i<g.size();i++) {
    size_t j=dofidx[i];
    dE+=f(3*i)*(g[j].x-o[j].x) + f(3*i+1)*(g[j].y-o[j].y) + f(3*i+2)*(g[j].z-o[j].z);
  }
  return dE;
}

void get_forces(const arma::vec & g, double & fmax, double & frms) {
  fmax=0.0;
  frms=0.0;

  for(size_t i=0;i<g.n_elem/3;i++) {
    double x=g(3*i);
    double y=g(3*i+1);
    double z=g(3*i+2);

    double fsq=x*x + y*y + z*z;

    frms+=fsq;
    if(fsq>fmax)
      fmax=fsq;
  }

  frms=sqrt(3*frms/g.n_elem);
  fmax=sqrt(fmax);
}

std::vector<atom_t> get_atoms(const arma::vec & x, const opthelper_t & opts) {
  // Update atomic positions
  std::vector<atom_t> atoms(opts.atoms);

  for(size_t i=0;i<opts.dofidx.size();i++) {
    size_t j=opts.dofidx[i];
    atoms[j].x=x(3*i);
    atoms[j].y=x(3*i+1);
    atoms[j].z=x(3*i+2);
  }

  return atoms;
}

enum calcd {
  // Full calculation
  FULLCALC,
  // Just the forces
  FORCECALC,
  // Just the energy
  ECALC,
  // Nothing
  NOCALC
};


void run_calc(const BasisSet & basis, bool force) {
  bool pz=false;
  try {
    pz=settings.get_bool("PZ");
  } catch(std::runtime_error &) {
  }

  if(pz)
    throw std::logic_error("Analytic forces not implemented for PZ-SIC!\n");

  Timer t;

  // Nothing to load - run full calculation.
  calculate(basis,force);
  if(force) {
    Checkpoint solchk(settings.get_string("SaveChk"),false);
    arma::vec f;
    solchk.read("Force",f);
    interpret_force(f).t().print("Analytic force");

    double fmax, frms;
    get_forces(f, fmax, frms);
    printf("Max force = %.3e, rms force = %.3e, evaluated in %s\n",fmax,frms,t.elapsed().c_str());
  }
}

void run_calc_num(const BasisSet & basis, bool force, int npoints, double h) {
  std::string savename=settings.get_string("SaveChk");
  std::string tempname=".tempchk";

  // Run calculation
  Timer t;
  calculate(basis,false);
  if(force)
    printf("Energy evaluated in %s\n",t.elapsed().c_str());

  // All we needed was the energy.
  if(!force)
    return;
  // Turn off verbose setting
  settings.set_bool("Verbose",false);

  // We have converged the energy, next compute force by finite
  // differences.
  arma::mat fm;
  fm.zeros(3,basis.get_Nnuc());

  // Nuclear coordinates. Take the transpose so that (x,y,z) are
  // stored consecutively in memory
  arma::mat nuccoord(basis.get_nuclear_coords().t());

  // Get the stencil
  if(npoints<2)
    throw std::runtime_error("Invalid stencil, must be >=2.\n");
  // Points to evaluate at
  arma::vec dx=arma::linspace<arma::vec>(-(npoints-1)/2.0,(npoints-1)/2.0,npoints);
  // Weights at the points
  arma::vec w;
  {
    arma::mat c(dx.n_elem,2);
    stencil(0.0,dx,c);
    w=c.col(1);

    // Eliminate any small weights
    for(size_t i=0;i<w.n_elem;i++)
      if(std::abs(w(i))<DBL_EPSILON*npoints) {
	dx.subvec(i,dx.n_elem-2)=dx.subvec(i+1,dx.n_elem-1);
	dx.resize(dx.n_elem-1);

	w.subvec(i,w.n_elem-2)=w.subvec(i+1,w.n_elem-1);
	w.resize(w.n_elem-1);
      }

    static bool printout=true;
    if(printout) {
      // Only print out stencil once
      printf("\n%13s %13s\n","xsten","wsten");
      for(size_t i=0;i<w.n_elem;i++)
	printf("% e % e\n",dx(i),w(i));
      printout=false;
    }

    // Put in spacing
    dx*=h;
    w/=h;
  }

  t.set();

  // Loop over degrees of freedom
  size_t Ndof=3*basis.get_Nnuc()-3;
  printf("Calculating %i displacements with %i point stencil\n",(int) Ndof,(int) dx.n_elem);
  fflush(stdout);
  for(size_t idof=0;idof<Ndof;idof++) {
    // No x or y force
    if(settings.get_bool("LinearSymmetry"))
      if(idof%3!=2)
	continue;

    Timer tdof;
    // Energies
    arma::vec E(dx.n_elem);

    for(size_t isten=0;isten<dx.n_elem;isten++) {
      // Coordinates of nuclei.
      arma::mat dcoord(nuccoord);
      dcoord[idof]+=dx(isten);
      // Take the back-transpose
      dcoord=dcoord.t();

      // Basis set
      BasisSet dbas(basis);
      dbas.set_nuclear_coords(dcoord);

      // Energy
      energy_t en;

      Settings settings0(settings);
      settings.set_string("LoadChk",savename);
      settings.set_string("SaveChk",tempname);
      {
	// Run calculation
	calculate(dbas,false);
	Checkpoint solchk(settings.get_string("SaveChk"),false);
	// Current energy is
	solchk.read(en);
	E(isten)=en.E;
      }
      remove(tempname.c_str());
      // Restore original settings
      settings=settings0;
    }

    // Calculate force: - grad E
    fm(idof)=-arma::dot(w,E);

    printf("%i/%i done in %s\n",(int) idof+1,(int) Ndof,tdof.elapsed().c_str());
    fflush(stdout);
  }

  // Force on last nucleus is just the negative of the sum of all the
  // forces on the other nuclei
  fm.col(fm.n_cols-1)=-arma::sum(fm.cols(0,fm.n_cols-2),1);
  //fm.print("Numerical forces");

  // Open the checkpoint in write mode, don't truncate it
  Checkpoint chkpt(savename,true,false);
  arma::vec f(arma::vectorise(fm));
  chkpt.write("Force",f);
  chkpt.close();

  interpret_force(f).t().print("Numerical force");
  double fmax, frms;
  get_forces(f, fmax, frms);
  printf("Max force = %.3e, rms force = %.3e, evaluated in %s\n",fmax,frms,t.elapsed().c_str());
}

void calculate(const arma::vec & x, const opthelper_t & p, double & E, arma::vec & g, bool force) {
  Timer t;

  // Get the atomic positions
  std::vector<atom_t> atoms=get_atoms(x,p);
  //  print_xyz(atoms);

  // Construct basis set
  BasisSet basis;
  construct_basis(basis,atoms,p.baslib);

  // Perform the electronic structure calculation
  if(p.numgrad)
    run_calc_num(basis,force,p.npoints,p.step);
  else
    run_calc(basis,force);

  // Solution checkpoint
  Checkpoint solchk(settings.get_string("SaveChk"),false);

  // Energy
  energy_t en;
  solchk.read(en);
  E=en.E;

  // Force
  arma::vec f;
  if(force) {
    solchk.read("Force",f);

    // Set components
    g.zeros(3*p.dofidx.size());
    for(size_t i=0;i<p.dofidx.size();i++) {
      size_t j=p.dofidx[i];
      g(3*i)=-f(3*j);
      g(3*i+1)=-f(3*j+1);
      g(3*i+2)=-f(3*j+2);
    }
  }
}

std::string getchk(size_t n) {
  std::ostringstream oss;
  oss << "geomcalc_" << getpid() << "_" << n << ".chk";
  return oss.str();
}

// Helper for line search
typedef struct {
  // Step length
  double s;
  // Energy
  double E;
  // Calculation
  size_t icalc;
} linesearch_t;

bool operator<(const linesearch_t & lh, const linesearch_t & rh) {
  return lh.s < rh.s;
}

void print_header() {
#ifdef _OPENMP
  printf("ERKALE - Geometry optimization from Hel, OpenMP version, running on %i cores.\n",omp_get_max_threads());
#else
  printf("ERKALE - Geometry optimization from Hel, serial version.\n");
#endif
  print_copyright();
  print_license();
#ifdef SVNRELEASE
  printf("At svn revision %s.\n\n",SVNREVISION);
#endif
  print_hostname();
}

void find_minimum(const std::vector<linesearch_t> & steps, double & Emin, double & smin, size_t & imin) {
  Emin=steps[0].E;
  smin=steps[0].s;
  imin=0;
  for(size_t i=1;i<steps.size();i++)
    if(steps[i].E < Emin) {
      Emin=steps[i].E;
      smin=steps[i].s;
      imin=i;
    }
}

int main_guarded(int argc, char **argv) {
  print_header();

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
  settings.add_scf_settings();
  settings.add_string("SaveChk","File to use as checkpoint","erkale.chk");
  settings.add_string("LoadChk","File to load old results from","");
  settings.add_bool("ForcePol","Force polarized calculation",false);
  settings.add_string("Optimizer","Optimizer to use: CGFR, CGPR, BFGS, SD","BFGS");
  settings.add_int("CGReset","Reset CG direction to SD every N steps", 5);
  settings.add_int("MaxSteps","Maximum amount of geometry steps",256);
  settings.add_string("Criterion","Convergence criterion to use: LOOSE, NORMAL, TIGHT, VERYTIGHT","NORMAL");
  settings.add_string("OptMovie","xyz movie to store progress in","optimize.xyz");
  settings.add_string("Result","File to save optimized geometry in","optimized.xyz");
  settings.set_string("Logfile","erkale_geom.log");
  settings.add_bool("NumGrad","Use finite-difference gradient?",false);
  settings.add_int("Stencil","Order of finite-difference stencil for numgrad",2);
  settings.add_double("Stepsize","Finite-difference stencil step size",1e-6);
  settings.add_double("LineStepFac","Line search step length factor",sqrt(10.0));
  settings.add_double("InitialStep","Initial step size in bohr",0.2);
  settings.add_double("ParaThr","Threshold for parabolic fit interval in bohr (dimer only)",1e-2);
  settings.add_double("BoundThr","Threshold for bond distance of bound molecules (dimer only)",10.0);
  settings.add_double("FuncThr","Threshold for function value change in search (dimer only)",1e-6);
  settings.parse(std::string(argv[1]),true);
  settings.print();

  // Don't try saving or loading Cholesky integrals
  settings.set_int("CholeskyMode",0);

  int maxiter=settings.get_int("MaxSteps");
  std::string optmovie=settings.get_string("OptMovie");
  std::string result=settings.get_string("Result");
  bool numgrad=settings.get_bool("NumGrad");
  int stencil=settings.get_int("Stencil");
  double step=settings.get_double("Stepsize");
  double steplen=settings.get_double("InitialStep");
  double Rcutoff=settings.get_double("BoundThr");
  double fac=settings.get_double("LineStepFac");
  double parathr=settings.get_double("ParaThr");
  double functhr=settings.get_double("FuncThr");
  int cgreset=settings.get_int("CGReset");

  // Interpret optimizer
  enum minimizer alg;
  std::string method=settings.get_string("Optimizer");
  if(stricmp(method,"CGFR")==0)
    alg=gCGFR;
  else if(stricmp(method,"CGPR")==0)
    alg=gCGPR;
  else if(stricmp(method,"BFGS")==0)
    alg=gBFGS;
  else if(stricmp(method,"SD")==0)
    alg=gSD;
  else {
    ERROR_INFO();
    throw std::runtime_error("Unknown optimization method.\n");
  }

  // Interpret optimizer
  enum convergence crit;
  method=settings.get_string("Criterion");
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
  std::string logfile=settings.get_string("Logfile");
  if(stricmp(logfile,"stdout")!=0) {
    // Redirect stdout to file
    FILE *outstream=freopen(logfile.c_str(),"w",stdout);
    if(outstream==NULL) {
      ERROR_INFO();
      throw std::runtime_error("Unable to redirect output!\n");
    }

    fprintf(stderr,"\n");
    // Print out the header in the logfile as well
    print_header();
  }

  // Read in atoms.
  std::string atomfile=settings.get_string("System");
  const std::vector<atom_t> origgeom=load_xyz(atomfile,!settings.get_bool("InputBohr"));
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
  std::string basfile=settings.get_string("Basis");
  baslib.load_basis(basfile);
  printf("\n");

  // Save to output
  save_xyz(atoms,"Initial configuration",optmovie,false);

  // Minimizer options
  opthelper_t pars;
  pars.atoms=atoms;
  pars.baslib=baslib;
  pars.dofidx=dofidx;
  pars.numgrad=numgrad;
  pars.npoints=stencil+1;
  pars.step=step;

  /* Starting point */
  arma::vec x(3*dofidx.size());
  for(size_t i=0;i<dofidx.size();i++) {
    x(3*i)=atoms[dofidx[i]].x;
    x(3*i+1)=atoms[dofidx[i]].y;
    x(3*i+2)=atoms[dofidx[i]].z;
  }

  // Steps taken in optimization, index of reference calc
  size_t ncalc=0, iref=0;
  // Stored checkpoints
  std::vector<size_t> chkstore;

  // Energy and force, as well as old energy and force
  double E=0.0, Eold;
  arma::vec f, fold;
  // Current and old search direction
  arma::vec sd, sdold;

  // Old geometry
  std::vector<atom_t> oldgeom(atoms);

  // Helper
  LBFGS bfgs;

  // Save calculation to
  Settings settings0(settings);
  std::string savechk0(settings.get_string("SaveChk"));
  settings.set_string("SaveChk",getchk(ncalc));

  if(atoms.size()==2) {
    // Specialized code for diatomic molecules
    double R0=round(dist(atoms[0],atoms[1])/steplen)*steplen;

    // Nuclear coordinates
    x.zeros(6);
    x(2)=-R0/2;
    x(5)=R0/2;

    // Calculate energy at the starting point
    calculate(x,pars,E,f,false);
    chkstore.push_back(ncalc);
    ncalc++;
    // Turn off verbose setting for any later calcs
    settings.set_bool("Verbose",false);
    try {
      // Also, don't localize, since it would screw up the converged guess
      settings.set_string("PZloc","false");
      // And don't run stability analysis, since we are only doing small displacements
      settings.set_int("PZstab",0);
    } catch(std::runtime_error &) {
    }

    // Bond distance search structure
    std::vector<linesearch_t> steps;
    linesearch_t p;
    p.s=0;
    p.icalc=ncalc-1;
    p.E=E;
    steps.push_back(p);

    printf("\nDimer calculation\n");
    printf("%8s %14s %12s\n","R (bohr)","Energy","Delta R");
    printf("%8.5f % 14.6f\n",R0,E);

    fprintf(stderr,"%8s %14s %12s\n","R (bohr)","Energy","Delta R");
    fprintf(stderr,"%8.5f % 14.6f\n",R0,E);

    // Search direction is bond length
    sd.zeros(6);
    sd(2)=-0.5;
    sd(5)=0.5;

    // Minimum energy and index
    double Emin;
    double smin;
    size_t imin;

    // Go closer
    size_t ileft=0;
    do {
      p.s=-1.0*((++ileft)*steplen);
      p.icalc=ncalc++;

      // Load reference from earlier calculation
      settings.set_string("LoadChk",getchk(p.icalc-1));
      // Save calculation to
      settings.set_string("SaveChk",getchk(p.icalc));

      calculate(x+p.s*sd,pars,p.E,f,false);
      chkstore.push_back(p.icalc);
      steps.insert(steps.begin(),p);

      printf("%8.5f % 14.6f\n",R0+p.s,p.E);
      fflush(stdout);

      fprintf(stderr,"%8.5f % 14.6f\n",R0+p.s,p.E);
      fflush(stderr);

      // Find the minimum energy
      find_minimum(steps,Emin,smin,imin);
    } while(imin < 2);

    bool bound=true;

    size_t iright=0;
    do {
      p.s=(++iright)*steplen;
      p.icalc=ncalc++;

      double Rlen=R0+p.s;
      // System is not bound
      if(Rlen>Rcutoff) {
        bound=false;
        break;
      }

      // Load reference from earlier calculation
      if(iright==1)
        settings.set_string("LoadChk",getchk(0));
      else
        settings.set_string("LoadChk",getchk(p.icalc-1));
      // Save calculation to
      settings.set_string("SaveChk",getchk(p.icalc));
      calculate(x+p.s*sd,pars,p.E,f,false);
      chkstore.push_back(p.icalc);
      steps.push_back(p);

      printf("%8.5f % 14.6f\n",R0+p.s,p.E);
      fflush(stdout);

      fprintf(stderr,"%8.5f % 14.6f\n",R0+p.s,p.E);
      fflush(stderr);

      // Find the minimum energy
      find_minimum(steps,Emin,smin,imin);
    } while(imin >= steps.size()-1);

    // Threshold
    double deltaRthr;
    switch(crit) {
    case(LOOSE):
      deltaRthr=1.0e-2;
      break;
    case(NORMAL):
      deltaRthr=1.0e-3;
      break;
    case(TIGHT):
      deltaRthr=1.0e-4;
      break;
    case(VERYTIGHT):
      deltaRthr=1.0e-5;
      break;

    default:
      throw std::logic_error("Case not handled.\n");
    }

    // Minimum is bracketed, proceed with search
    double deltaR;
    bool convd=false;
    int nrefined=0;
    if(bound) {
      while(true) {
        // Sort the steps in length
        std::sort(steps.begin(),steps.end());

        // Find minimum position
        find_minimum(steps,Emin,smin,imin);

        // Do parabolic fit
        double a=steps[imin-1].s;
        double b=steps[imin].s;
        double c=steps[imin+1].s;
        double fa=steps[imin-1].E;
        double fb=steps[imin].E;
        double fc=steps[imin+1].E;

        // Is there any variation in the function value?
        double fmin=std::min(std::min(fa,fb),fc);
        double fmax=std::max(std::max(fa,fb),fc);
        if(fmax-fmin<=functhr) {
          printf("Function value converged within threshold\n");
          convd=true;
          break;
        }

        // New position is at
        double xs = b - 0.5*((b-a)*(b-a)*(fb-fc) - (b-c)*(b-c)*(fb-fa))/((b-a)*(fb-fc) - (b-c)*(fb-fa));

        // Length of search intervals are
        double len1(b-a);
        double len2(c-b);
        deltaR=std::max(len1,len2);

        // Is this within the search interval?
        bool parafit = (deltaR <= parathr) && xs>=a && xs<=c;
        if(!parafit) {
          // No, use golden ratio search
          const double golden =(sqrt(5.0)-1.0)/2.0;
          // Split larger interval
          if(len1 > len2) {
            xs = a + golden*(b-a);
          } else {
            xs = b + golden*(c-b);
          }
        }

        // Determine point closest to the minimum
        size_t rmin=0;
        double dRmin=std::abs(steps[0].s-xs);
        for(size_t i=1;i<steps.size();i++) {
          double dR=std::abs(steps[i].s-xs);
          if(dR<dRmin) {
            rmin=i;
            dRmin=dR;
          }
        }
        settings.set_string("LoadChk",getchk(rmin));
        p.s=xs;
        p.icalc=ncalc++;
        settings.set_string("SaveChk",getchk(p.icalc));
        chkstore.push_back(p.icalc);
        calculate(x+p.s*sd,pars,p.E,f,false);
        steps.push_back(p);

        printf("%8.5f % 14.6f %e\n",R0+p.s,p.E,deltaR);
        fflush(stdout);

        fprintf(stderr,"%8.5f % 14.6f %e\n",R0+p.s,p.E,deltaR);
        fflush(stderr);

        if(deltaR < deltaRthr) {
          convd=true;
          break;
        }

        // Increment iteration count
        nrefined++;
        if(nrefined>=maxiter) {
          break;
        }
      }
    }
    if(convd)
      printf("Converged\n");
    else {
      printf("Failed to attain converge\n");
      return 1;
    }

    // Update geometry
    find_minimum(steps,Emin,smin,imin);
    x+=sd*smin;

  } else {

    // Calculate energy at the starting point
    calculate(x,pars,E,f,false);
    chkstore.push_back(ncalc);
    ncalc++;
    // Turn off verbose setting for any later calcs
    settings.set_bool("Verbose",false);
    try {
      // Also, don't localize, since it would screw up the converged guess
      settings.set_string("PZloc","false");
      // And don't run stability analysis, since we are only doing small displacements
      settings.set_int("PZstab",0);
    } catch(std::runtime_error &) {
    }

    printf("\n\nStarting geometry optimization\n");
    printf("%4s %18s %9s %9s\n","iter","E","fmax","frms");

    fprintf(stderr,"\n%3s %18s %10s %10s %10s %10s %10s %10s %s\n", "it", "E", "dE", "dEfrac", "dmax ", "drms ", "fmax ", "frms ", "t");
    fflush(stderr);

    for(int iiter=0;iiter<maxiter;iiter++) {
      Timer titer;

      // Store old values of gradient and search direction
      fold=f;
      sdold=sd;

      // Load reference from earlier calculation
      settings.set_string("LoadChk",getchk(iref));
      // Save calculation to
      settings.set_string("SaveChk",getchk(ncalc));

      // Calculate energy and force at current position
      calculate(x,pars,E,f,true);
      chkstore.push_back(ncalc);
      // Change reference
      iref=ncalc;
      // Increment step value
      ncalc++;

      // Store old value of energy
      Eold=E;

      // Save geometry step
      {
        char comment[80];
        sprintf(comment,"Step %i",(int) iiter);
        save_xyz(get_atoms(x,pars),comment,optmovie,true);
      }

      // Search direction
      sd=-f;
      std::string steptype="SD";

      if(iiter>0) {
        if((alg==gCGPR || alg==gCGFR) && (iiter%cgreset!=0)) {
          // Polak-Ribière
          double gamma;
          if(alg==gCGPR) {
            gamma=arma::dot(f,f-fold)/arma::dot(fold,fold);
            steptype="CGPR";
          } else {
            gamma=arma::dot(f,f)/arma::dot(fold,fold);
            steptype="CGFR";
          }

          arma::vec sdnew=sd+gamma*sdold;
          if(arma::dot(f,fold)>=0.2*arma::dot(fold,fold)) {
            steptype="Powell restart - SD";
          } else
            sd=sdnew;

        } else if(alg==gBFGS) {
          // Update BFGS
          bfgs.update(x,f);
          // Get search direction
          sd=-bfgs.solve();
          steptype="BFGS";
        }

        // Check we are still going downhill / BFGS direction isn't nan
        if(!(arma::dot(sd,f)<=0)) {
          steptype+=": Bad search direction. SD";
          sd=-f;
        }
      }

      // Get forces
      double fmax, frms;
      get_forces(f, fmax, frms);

      // Macroiteration status
      printf("\n%s step\n",steptype.c_str());
      printf("%4i % 18.10f %.3e %.3e\n",iiter,E,fmax,frms);
      fflush(stdout);

      // Legend
      printf("\t%2s %12s %13s\n","i","step","dE");

      // Do a line search on the search direction by bracketing the
      // minimum
      std::vector<linesearch_t> steps;

      // Initial entry is
      {
        // Step length and energy
        linesearch_t p;
        p.s=0.0;
        p.E=E;
        p.icalc=iref;
        steps.push_back(p);
      }

      // First, try the step length as is
      {
        Timer ts;

        // Step length and energy
        linesearch_t p;
        p.s=steplen;
        p.icalc=ncalc;

        double Et;
        arma::vec ft;
        settings.set_string("LoadChk",getchk(iref));
        settings.set_string("SaveChk",getchk(ncalc));
        calculate(x+p.s*sd,pars,Et,ft,false);
        iref=ncalc;
        chkstore.push_back(ncalc);
        ncalc++;

        p.E=Et;
        steps.push_back(p);

        printf("\t%2i %e % e %s\n",(int) steps.size()-1,p.s,Et-E,ts.elapsed().c_str());
        fflush(stdout);
      }

      // Minimum energy and index
      double Emin;
      size_t imin;

      while(true) {
        // Sort the steps in length
        std::sort(steps.begin(),steps.end());

        // Find the minimum energy
        double smin;
        find_minimum(steps,Emin,smin,imin);

        // Where is the minimum?
        if(imin==0 || imin==1 || imin==steps.size()-1) {
          Timer ts;

          linesearch_t p;
          if(imin==0 || (steps.size()>2 && imin==1)) {
            // Need smaller step
            p.s=steps[1].s/fac;

            if(p.s*arma::max(arma::abs(sd)) < DBL_EPSILON) {
              printf("Step length too small, stopping line search.\n");
              break;
            }
          } else {
            // Need bigger step
            p.s=steps[imin].s*fac;
          }
          p.icalc=ncalc;

          double Et;
          arma::vec ft;
          settings.set_string("LoadChk",getchk(steps[imin].icalc));
          settings.set_string("SaveChk",getchk(ncalc));
          calculate(x+p.s*sd,pars,Et,ft,false);
          chkstore.push_back(ncalc);
          ncalc++;

          p.E=Et;
          steps.push_back(p);
          printf("\t%2i %e % e %s\n",(int) steps.size()-1,p.s,Et-E,ts.elapsed().c_str());
          fflush(stdout);

        } else {
          // Optimum is somewhere in the middle
          break;
        }
      }

      // Find the minimum energy
      double smin;
      find_minimum(steps,Emin,smin,imin);
      if(imin == 0) {
        printf("Energy not decreasing.\n");
        break;
      }

      {
        // Interpolate: A b = y
        arma::mat A(3,3);
        arma::vec y(3);
        for(size_t i=0;i<3;i++) {
          A(i,0)=1.0;
          A(i,1)=steps[imin+i-1].s;
          A(i,2)=std::pow(A(i,1),2);

          y(i)=steps[imin+i-1].E;
        }

        arma::mat b;
        if(arma::solve(b,A,y) && b(2)>sqrt(DBL_EPSILON)) {
          // Success in solution and parabola gives minimum.

          // The minimum of the parabola is at
          double x0=-b(1)/(2*b(2));

          // Is this an interpolation?
          if(A(0,1) < x0 && x0 < A(2,1)) {
            Timer ts;

            // Figure out which reference is the closest
            iref=steps[imin-1].icalc;
            double dminv=std::abs(x0-A(0,1));
            for(size_t i=1;i<A.n_rows;i++) {
              double d=std::abs(x0-A(i,1));
              if(d<dminv) {
                dminv=d;
                iref=steps[imin+i-1].icalc;
              }
            }

            // Do the calculation with the interpolated step
            linesearch_t p;
            p.s=x0;
            p.icalc=ncalc;

            double Et;
            arma::vec ft;
            settings.set_string("LoadChk",getchk(iref));
            settings.set_string("SaveChk",getchk(ncalc));
            calculate(x+p.s*sd,pars,Et,ft,false);
            chkstore.push_back(ncalc);
            ncalc++;

            p.E=Et;
            steps.push_back(p);
            printf("\t%2i %e % e %s\n",(int) steps.size()-1,p.s,Et-E,ts.elapsed().c_str());
            fflush(stdout);

            // Resort the steps in length
            std::sort(steps.begin(),steps.end());

            // Find the minimum energy
            find_minimum(steps,Emin,smin,imin);
          }
        }
      }

      // Switch to the minimum geometry
      x+=steps[imin].s*sd;
      iref=steps[imin].icalc;

      // Store optimal step length
      steplen=steps[imin].s;

      printf("Best step size %e changed energy by % e\n",steplen,steps[imin].E-E);

      // Copy checkpoint file
      {
        std::ostringstream oss;
        oss << "\\cp " << getchk(iref) << " " << savechk0;
        if(system(oss.str().c_str()))
          throw std::runtime_error("Error copying checkpoint.\n");
      }

      // Erase all unnecessary calcs
      for(size_t i=0;i<chkstore.size();i++)
        if(chkstore[i]!=iref)
          remove(getchk(chkstore[i]).c_str());

      // New geometry
      std::vector<atom_t> geom=get_atoms(x,pars);

      // Calculate displacements
      double dmax, drms;
      get_displacement(geom, oldgeom, dmax, drms);
      // Calculate projected change of energy
      double dEproj=calculate_projection(geom,oldgeom,f,pars.dofidx);
      // Actual change of energy is
      double dE=steps[imin].E - Eold;

      // Store new geometry
      oldgeom=geom;

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

      double dEfrac;
      if(dEproj!=0.0)
        dEfrac=dE/dEproj;
      else
        dEfrac=0.0;

      const static char cconv[]=" *";

      fprintf(stderr,"%3i % 18.10f % .3e % .3e %.3e%c %.3e%c %.3e%c %.3e%c %s\n", iiter, E, dE, dEfrac, dmax, cconv[dmaxconv], drms, cconv[drmsconv], fmax, cconv[fmaxconv], frms, cconv[frmsconv], titer.elapsed().c_str());
      fflush(stderr);

      bool convd=dmaxconv && drmsconv && fmaxconv && frmsconv;

      if(convd) {
        fprintf(stderr,"Converged.\n");
        break;
      }
    }
  }

  // Run calculation to get final checkpoint
  settings=settings0;
  settings.set_string("LoadChk",savechk0);
  settings.set_string("SaveChk",savechk0);
  calculate(x,pars,E,f,false);
  save_xyz(get_atoms(x,pars),"Optimized configuration",result,false);

  // Remove the rest
  for(size_t i=0;i<chkstore.size();i++)
    remove(getchk(chkstore[i]).c_str());

  printf("Running program took %s.\n",tprog.elapsed().c_str());

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
