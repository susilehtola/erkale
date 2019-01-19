#include "basis.h"
#include "checkpoint.h"
#include "elements.h"
#include "linalg.h"
#include "settings.h"
#include "stringutil.h"
#include "timer.h"
#include <algorithm>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#define ZTHR 1e-6

// Diatomic basis function
typedef struct {
  int am; // l value
  double a; // exponent
  bool atom; // Atom-centered function?
} dim_bf_t;

bool operator<(const dim_bf_t & lh, const dim_bf_t & rh) {
  // First, sort by z value
  if(!lh.atom && rh.atom)
    return true;
  if(lh.atom && !rh.atom)
    return false;

  // Then by am value
  if(lh.am < rh.am)
    return true;
  if(lh.am > rh.am)
    return false;

  // Then by exponent, decreasing order
  if(lh.a > rh.a)
    return true;
  if(lh.a < rh.a)
    return false;

  // Default value
  return false;
}

int maxam(const std::vector<dim_bf_t> & funcs) {
  int maxam=-1;
  for(size_t i=0;i<funcs.size();i++)
    maxam=std::max(maxam,funcs[i].am);
  return maxam;
}

int nprim(const std::vector<dim_bf_t> & funcs, int am) {
  int nprim=0;
  for(size_t i=0;i<funcs.size();i++)
    if(funcs[i].am == am) nprim++;
  return nprim;
}

int nbf(const std::vector<dim_bf_t> & funcs) {
  int nbf=0;
  for(size_t i=0;i<funcs.size();i++)
    nbf += funcs[i].atom ? 2*(2*funcs[i].am+1) : (2*funcs[i].am+1);
  return nbf;
}

// Diatomic molecule
typedef struct {
  double R; // bond length
  int Z1; // nuclear charge at left hand
  int Z2; // nuclear charge at right hand

  // Condition number penalty?
  double gamma;
  // Number of centers
  int ncen;
} dim_sys_t;

void get_system(std::vector<nucleus_t> & nuclei, std::vector<ElementBasisSet> & els, const dim_sys_t & sys, std::vector<dim_bf_t> funcs) {
  // Map from coordinate to nuclear centers
  if(sys.ncen == 2) {
    std::vector<double> atommap(3);
    atommap[0]=-1.0;
    atommap[1]=0.0;
    atommap[2]=1.0;

    // Form list of nuclei
    nuclei.clear();
    nuclei.resize(atommap.size());
    for(size_t i=0;i<atommap.size();i++) {
      nuclei[i].r.x=0.0;
      nuclei[i].r.y=0.0;
      nuclei[i].r.z=atommap[i]*sys.R/2.0;
      nuclei[i].Q=0;
      nuclei[i].Z=0;
      nuclei[i].symbol="H";
      nuclei[i].bsse=true;
    }

    // Set real atoms
    nuclei[0].Z=sys.Z1;
    if(sys.Z1>0) {
      nuclei[0].bsse=false;
      nuclei[0].symbol=element_symbols[sys.Z1];
    }

    nuclei[nuclei.size()-1].Z=sys.Z2;
    if(sys.Z2>0) {
      nuclei[nuclei.size()-1].bsse=false;
      nuclei[nuclei.size()-1].symbol=element_symbols[sys.Z2];
    }

    // Construct basis sets
    els.clear();
    els.resize(nuclei.size());
    for(size_t i=0;i<nuclei.size();i++)
      els[i]=ElementBasisSet(nuclei[i].symbol,i+1);

    // Loop over functions
    for(size_t i=0;i<funcs.size();i++) {
      FunctionShell sh(funcs[i].am);
      sh.add_exponent(1.0,std::pow(10.0,funcs[i].a));

      if(!funcs[i].atom)
        els[1].add_function(sh);
      else {
        els[0].add_function(sh);
        els[2].add_function(sh);
      }
    }

  } else {
    std::vector<double> atommap(1);
    atommap[0]=0.0;

    // Form list of nuclei
    nuclei.clear();
    nuclei.resize(atommap.size());
    for(size_t i=0;i<atommap.size();i++) {
      nuclei[i].r.x=0.0;
      nuclei[i].r.y=0.0;
      nuclei[i].r.z=atommap[i]*sys.R/2.0;
      nuclei[i].Q=0;
      nuclei[i].Z=0;
      nuclei[i].symbol="H";
      nuclei[i].bsse=true;
    }

    // Set real atoms
    nuclei[0].Z=std::max(sys.Z1,sys.Z2);
    nuclei[0].bsse=false;
    nuclei[0].symbol=element_symbols[nuclei[0].Z];

    // Construct basis sets
    els.clear();
    els.resize(nuclei.size());
    for(size_t i=0;i<nuclei.size();i++)
      els[i]=ElementBasisSet(nuclei[i].symbol,i+1);

    // Loop over functions
    for(size_t i=0;i<funcs.size();i++) {
      FunctionShell sh(funcs[i].am);
      sh.add_exponent(1.0,std::pow(10.0,funcs[i].a));
      els[0].add_function(sh);
    }
  }
}

void construct_basis(BasisSet & basis, const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs) {
  // Get the system
  std::vector<nucleus_t> nuclei;
  std::vector<ElementBasisSet> els;
  get_system(nuclei, els, sys, funcs);

  // Construct basis set
  basis=BasisSet(nuclei.size());
  for(size_t i=0;i<els.size();i++) {
    basis.add_nucleus(nuclei[i]);
    basis.add_shells(i,els[i],true); // sort, always pure functions
  }
  basis.finalize(true);
}


Settings settings;

void get_config(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, int & Q, int & M) {
  // Get charge and multiplicity
  Q=settings.get_int("Charge");
  M=settings.get_int("Multiplicity");

  // Total number of electrons is
  int Nel=(sys.Z1+sys.Z2)-Q;
  int Nela, Nelb;
  get_Nel_alpha_beta(Nel,M,Nela,Nelb);

  // Do we have enough basis functions to describe all electrons? Note next iteration will have one function more!
  int Nbf=nbf(funcs)+1;
  if(Nbf<std::max(Nela,Nelb)) {
    // This is how many electrons we can fit in the system
    int Nocc=2*Nbf;
    // so the charge is
    Q=sys.Z1+sys.Z2-Nocc;
    // and we have a singlet
    M=1;
  }
}

double condition_number(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs) {
  // Get centers
  BasisSet basis;
  construct_basis(basis,sys,funcs);
  return arma::cond(basis.overlap());
}

double get_energy(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, int Q, int M, int useref) {
  // Get centers
  BasisSet basis;

  Settings settings0(settings);
  construct_basis(basis,sys,funcs);

  settings.set_bool("Verbose",false);
  settings.add_string("LoadChk","","");

  {
    std::ostringstream chk;
    chk << "erkale_tmp";
#ifdef _OPENMP
    chk << "_" << omp_get_thread_num();
#endif
    chk << ".chk";
    settings.add_string("SaveChk","",chk.str());
  }

  if(useref==1) {
    // Load reference calculation
    settings.set_string("LoadChk","erkale_ref.chk");
  } else if(useref==-1) {
    // Save reference calculation
    settings.set_string("SaveChk","erkale_ref.chk");
  }

  // Set charge and multiplicity
  settings.set_int("Charge",Q);
  settings.set_int("Multiplicity",M);

  // Run the calculation
  calculate(basis,false);

  // Load the result
  Checkpoint chkpt(settings.get_string("SaveChk"),false,false);
  energy_t en;
  chkpt.read(en);

  // Only look at electronic energy
  double E=en.Eel;

  // Condition number penalty?
  if(sys.gamma!=0.0)
    E+=sys.gamma*condition_number(sys,funcs);

  settings=settings0;

  return E;
}

std::vector<dim_bf_t> interpret(const arma::vec & p, const std::vector<dim_bf_t> & funcs) {
  std::vector<dim_bf_t> ret(funcs);
  if(p.n_elem != funcs.size()) throw std::logic_error("Wrong vector length!\n");
  for(size_t i=0;i<funcs.size();i++)
    ret[i].a=p(i);

  return ret;
}

arma::vec collect(const std::vector<dim_bf_t> & funcs) {
  arma::vec ret;
  ret.zeros(funcs.size());
  for(size_t i=0;i<funcs.size();i++)
    ret(i)=funcs[i].a;

  return ret;
}

arma::vec energy_gradient(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, int Q, int M, int mval) {
  // Step size
  double ss(sqrt(DBL_EPSILON));
  // Collect params
  arma::vec x(collect(funcs));

  // Gradient
  arma::vec g;
  g.zeros(x.n_elem);

  // No truly parallel HDF5!
#if 0
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
#endif
  for(size_t i=0;i<g.n_elem;i++) {
    // Zero out all other blocks
    if(mval>=0 && funcs[i].am!=mval)
      continue;

    arma::vec xlh(x), xrh(x);
    xlh(i)-=ss;
    xrh(i)+=ss;

    double ylh(get_energy(sys,interpret(xlh,funcs),Q,M,1));
    double yrh(get_energy(sys,interpret(xrh,funcs),Q,M,1));

    g(i)=(yrh-ylh)/(2.0*ss);
  }

  return g;
}

arma::mat energy_hessian(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, int Q, int M, int mval) {
  // Step size
  double ss(sqrt(DBL_EPSILON));
  // Collect params
  arma::vec x(collect(funcs));
  // Current energy
  double y0(get_energy(sys,interpret(x,funcs),Q,M,1));

  // Gradient
  arma::mat h;
  h.zeros(x.n_elem,x.n_elem);

  // No truly parallel HDF5!
#if 0
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
#endif
  for(size_t i=0;i<x.n_elem;i++) {
    if(mval>=0 && funcs[i].am!=mval)
      continue;

    for(size_t j=0;j<=i;j++) {
      if(mval>=0 && funcs[j].am!=mval)
        continue;

      arma::vec xil(x), xir(x), xjl(x), xjr;
      xil(i)-=ss;
      xir(i)+=ss;
      xjl(j)-=ss;
      xjr(j)+=ss;

      double yil(get_energy(sys,interpret(xil,funcs),Q,M,1));
      double yir(get_energy(sys,interpret(xir,funcs),Q,M,1));
      double yjl(get_energy(sys,interpret(xjl,funcs),Q,M,1));
      double yjr(get_energy(sys,interpret(xjr,funcs),Q,M,1));

      h(i,j) = h(j,i) = (yil + yir + yjl + yjr - 4.0*y0)/(ss*ss);
    }
  }

  return h;
}

double optimize_energy(const dim_sys_t & sys, std::vector<dim_bf_t> & funcs, int Q, int M, int mval=-1) {
  // Current parameters
  arma::vec x(collect(funcs));

  // Get the initial energy
  double E0(get_energy(sys,interpret(x,funcs),Q,M,-1));
  // Current energy
  double E(E0);

  int iit;
  for(iit=0;iit<10;iit++) {
    // Evaluate gradient and diagonal hessian
    arma::vec g=energy_gradient(sys,interpret(x,funcs),Q,M,mval);

    printf("Iteration %i gradient norm is %e.\n",iit,arma::norm(g,2));

    // Search direction
    arma::vec ss(-g);

    printf("Line search:\n");

    // Do line search
    std::vector<double> step, y;
    step.push_back(0.0);
    y.push_back(get_energy(sys,interpret(x,funcs),Q,M,1));
    printf("\t%e % e\n",step[step.size()-1],y[y.size()-1]);

    double h0=sqrt(DBL_EPSILON);
    step.push_back(h0);
    y.push_back(get_energy(sys,interpret(x+ss*step[step.size()-1],funcs),Q,M,1));
    printf("\t%e % e % e\n",step[step.size()-1],y[y.size()-1],y[y.size()-1]-y[y.size()-2]);

    if(y[1]>y[0]) {
      printf("Function value is not decreasing!\n");
      break;
    }

    while(y[y.size()-1]<=y[y.size()-2]) {
      step.push_back(step[step.size()-1]*2);
      y.push_back(get_energy(sys,interpret(x+ss*step[step.size()-1],funcs),Q,M,1));
      printf("\t%e % e % e\n",step[step.size()-1],y[y.size()-1],y[y.size()-1]-y[y.size()-2]);
    }

    // Update x and E
    if(y[y.size()-2]<E) {
      x=x+ss*step[step.size()-2];
      E=y[y.size()-2];
    } else
      break;
  }

  // Update functions
  funcs=interpret(x,funcs);
  std::sort(funcs.begin(),funcs.end());

  printf("Optimization converged in %i step and decreased the value by %e\n",iit,E-E0);

  return E;
}

void add_function(const arma::vec & avals, arma::uword minai, arma::uword atom, int minam, std::vector<dim_bf_t> & funcs) {
  // Add function
  dim_bf_t add;
  add.am=minam;
  add.a=avals(minai);
  add.atom=(atom!=0);
  funcs.push_back(add);
  // Sort values
  std::sort(funcs.begin(),funcs.end());
}

arma::mat scan(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, const arma::vec & avals, int Q, int M, int am, const arma::imat & allow, double E0=0.0, int useref=1) {
  arma::mat ret(avals.n_elem,sys.ncen);

#ifndef _HDF5_H
#error "HDF5 headers not included"
#endif

  // No truly parallel HDF5!
#if 0
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
#endif
  for(int iz=0;iz<sys.ncen;iz++)
    for(size_t ia=0;ia<avals.n_elem;ia++) {
      if(!allow(ia,iz)) {
        // Shortcut
        ret(ia,iz)=0.0;
        continue;
      }

      // Add function
      std::vector<dim_bf_t> testfuncs(funcs);
      add_function(avals,ia,iz,am,testfuncs);
      try {
        ret(ia,iz)=get_energy(sys,testfuncs,Q,M,useref)-E0;
      } catch (std::runtime_error & err) {
        ret(ia,iz)=DBL_MAX;
      }
    }

  return ret;
}

arma::mat scan(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, const arma::vec & avals, int Q, int M, int am, double E0=0.0, int useref=1) {
  arma::imat allow(avals.n_elem,sys.ncen);
  allow.ones();
  return scan(sys,funcs,avals,Q,M,am,allow,E0,useref);
}

void save_basis(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs) {
  // Get basis set
  std::vector<nucleus_t> nuclei;
  std::vector<ElementBasisSet> els;
  get_system(nuclei,els,sys,funcs);

  // Iteration count
  static int iter=0;

  // Output names
  std::ostringstream suffix;
  suffix << "_" << iter;
  std::string geom("opt_atoms"+suffix.str()+".xyz");
  std::string basis("opt_basis"+suffix.str()+".gbs");

  {
    FILE *out=fopen(geom.c_str(),"w");
    fprintf(out,"%i\nOptimized basis set for %s-%s with distance R=%e\n",(int) nuclei.size(),element_symbols[sys.Z1].c_str(),element_symbols[sys.Z2].c_str(),sys.R);
    for(size_t i=0;i<nuclei.size();i++) {
      std::string sym;
      if(nuclei[i].bsse)
        sym="H-Bq";
      else
        sym=element_symbols[nuclei[i].Z];

      fprintf(out,"%6s % .6f % .6f % .6f\n", sym.c_str(), nuclei[i].r.x, nuclei[i].r.y, nuclei[i].r.z);
    }
    fclose(out);
  }

  BasisSetLibrary baslib;
  for(size_t i=0;i<els.size();i++)
    baslib.add_element(els[i]);
  baslib.save_gaussian94(basis);

  iter++;
}

void update_allowed(arma::imat & allowed, const arma::mat & Eam) {
  // Update allowed sector
  double ammin(Eam.min());

  // We should be within 3 orders of magnitude of the optimal lowering
  double thresh(ammin*1e-3);

  if(allowed.n_rows != Eam.n_rows) throw std::logic_error("allowed and Eam not the same size!\n");
  if(allowed.n_cols != Eam.n_cols) throw std::logic_error("allowed and Eam not the same size!\n");

  for(size_t i=0;i<allowed.n_rows;i++)
    for(size_t j=0;j<allowed.n_cols;j++)
      // E.g. if ammin is -1e-2, thresh will be -1e-5, and we thus
      // have to require that
      if(Eam(i,j)>=thresh)
        allowed(i,j)=false;
}

int main_guarded(int argc, char ** argv) {
  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  // Initialize libint
  init_libint_base();

  settings.add_scf_settings();
  settings.add_double("Tolerance","Convergence tolerance for addition of a function",1e-6);
  settings.add_bool("ForcePol","Force polarized calculation",false);
  settings.add_bool("Optimize","Run primitive energy optimization?",false);
  settings.add_int("MinLogExp","Minimum exponent value",-4);
  settings.add_int("MaxLogExp","Maximum exponent value",9);
  settings.add_int("NExp","Number of points on exponent grid",131);
  settings.add_int("NGeom","Number of points on geometry grid",3);
  settings.add_int("ScanOutdate","Redo scans every N new functions",4);
  settings.add_double("MapOutdate","Outdate scan maps when minimum increases by factor",10.0);
  settings.add_double("Gamma","Condition number penalizer",1e-3);

  settings.parse(std::string(argv[1]),true);
  // Must use core guess
  settings.set_string("Guess","Core");
  // Use dimer symmetry
  settings.set_bool("DimerSymmetry",true);
  // .. which requires setting
  settings.set_bool("OptLM",false);
  // No basis set rotation
  settings.set_bool("BasisRotate",false);
  settings.print();

  std::vector<std::string> sysv(splitline(settings.get_string("System")));
  if(sysv.size() != 3 && sysv.size() != 4) {
    throw std::logic_error("System specification invalid: format Z1 Z2 R (angstrom)\n");
  }

  // Parse system
  dim_sys_t sys;
  sys.Z1=readint(sysv[0]);
  sys.Z2=readint(sysv[1]);
  sys.R=readdouble(sysv[2]);
  sys.gamma=settings.get_double("Gamma");
  if(sysv.size()>3) {
    std::string spec(sysv[3]);
    if(stricmp(spec,"angstrom")==0)
      sys.R*=ANGSTROMINBOHR;
    else if(stricmp(spec,"bohr")!=0)
      throw std::logic_error("Invalid unit specification " + spec + "!\n");
  }
  if(sys.Z1 == 0 || sys.Z2 == 0)
    sys.ncen=1;
  else
    sys.ncen=2;

  double thr(settings.get_double("Tolerance"));

  // Maximum and minimum exponent
  const double mina(settings.get_int("MinLogExp"));
  const double maxa(settings.get_int("MaxLogExp"));
  // Exponent grid
  const int Nexp(settings.get_int("NExp"));

  int scanoutdate(settings.get_int("ScanOutdate"));
  double mapoutdatefac(settings.get_double("MapOutdate"));
  bool optimize(settings.get_bool("Optimize"));

  if(sys.Z1 == 0 || sys.Z2 == 0)
    printf("Optimizing basis set for %s atom\n",element_symbols[std::max(sys.Z1,sys.Z2)].c_str());
  else
    printf("Optimizing basis set for %s-%s dimer with bond distance R=%e a.u.\n",element_symbols[sys.Z1].c_str(),element_symbols[sys.Z2].c_str(),sys.R);
  printf("Continuing optimization until energy drops by less than %e\n",thr);
  printf("Spacing exponents by factor %e.\n",std::pow(10.0,(maxa-mina)/Nexp));
  printf("Full exponent scans are made whenever the minimum increases by %e.\n",mapoutdatefac);
  printf("Exponent subscans are made at least every %i iterations.\n\n",scanoutdate);

  printf("Nuclear repulsion energy is %.16e\n",sys.Z1*sys.Z2/sys.R);

  // Grids
  arma::vec avals(arma::linspace<arma::vec>(mina,maxa,Nexp));
  avals.save("x_expn.dat",arma::raw_ascii);

  std::vector<dim_bf_t> funcs;

  // Charge and multiplicity
  int Q, M;
  // Current energy
  double E0;
  // Energy change
  double dE;

  // Initialize basis
  while(true) {
    // Determine charge and multiplicity
    get_config(sys,funcs,Q,M);
    // Form checkpoint
    printf("Running now for state Q=%+i M=%i\n",Q,M);
    fflush(stdout);

    // No reference energy
    E0=0.0;

    // Values to scan
    int nam(maxam(funcs)+2);
    arma::cube Egrid(avals.n_elem,sys.ncen,nam);

    for(int am=0;am<nam;am++) {
      // We don't have a reference because we don't have an old basis that's big enough
      Timer tam;
      arma::mat Eam=scan(sys,funcs,avals,Q,M,am,E0,0);

      std::ostringstream oss;
      oss << "Egrid_" << shell_types[am] << "_" << funcs.size() << ".dat";
      Eam.save(oss.str(),arma::raw_ascii);

      Egrid.slice(am)=Eam;

      printf("%c shell done in % .3f\n",shell_types[am],tam.get());
      fflush(stdout);
    }

    // What's the best function?
    arma::uword minai, mincen, minam;
    dE=Egrid.min(minai,mincen,minam);
    add_function(avals,minai,mincen,minam,funcs);

    printf("Added %c function at index %i with exponent % e, energy % .10e changed by % e. nbf = %i, K = %e\n",shell_types[minam],(int) mincen,std::pow(10.0,avals(minai)),E0+dE,dE,nbf(funcs),condition_number(sys,funcs));

    // Do we have a minimal basis now?
    if(Q==settings.get_int("Charge") && M==settings.get_int("Multiplicity"))
      break;
  }

  // Current energy
  E0=get_energy(sys,funcs,Q,M,-1);
  printf("Calculation initialized, initial energy without penalty is % .16e\n\n",E0-sys.gamma*condition_number(sys,funcs));
  fflush(stdout);

  while(true) {
    int nam(maxam(funcs)+2);

    // Update map at next iteration?
    bool mapupdate=true;
    // Minimum value seen on map
    double mapminimum=DBL_MAX;

    // Value grid
    arma::cube Egrid(avals.n_elem,sys.ncen,nam);
    // Allow scan?
    arma::icube allowed(avals.n_elem,sys.ncen,nam);
    // Are values up to date?
    std::vector<int> outdated;

    while(true) {
      Timer t;

      if(mapoutdatefac<=1.0)
        // No map used
        mapupdate=true;

      // Reset map?
      if(mapupdate) {
        allowed.ones();
        mapupdate=false;
      }

      // Scan
      for(int am=0;am<nam;am++) {
        bool calcam=false;
        if(outdated.size() <= (size_t) am) {
          outdated.resize(am+1);
          calcam=true;
        } else if(outdated[am]>=scanoutdate) {
          calcam=true;
        }

        if(calcam) {
          Timer tam;
          arma::mat Eam(scan(sys,funcs,avals,Q,M,am,allowed.slice(am),E0));

          std::ostringstream oss;
          oss << "Egrid_" << shell_types[am] << "_" << funcs.size() << ".dat";
          Eam.save(oss.str(),arma::raw_ascii);

          Egrid.slice(am)=Eam;

          printf("%i%c shell mog = %e done in % .3f (%i out of %i allowed)\n",nprim(funcs,am)+1,shell_types[am],Eam.min(),tam.get(),(int) arma::sum(arma::sum(allowed.slice(am))),(int) (allowed.n_rows*allowed.n_cols));
          fflush(stdout);
          outdated[am]=false;

          update_allowed(allowed.slice(am),Eam);
        }
      }

      // Store new map minimum
      mapminimum=std::min(mapminimum, Egrid.min());
      // Do we need an update to the map?
      if(mapoutdatefac>1.0 && mapminimum <= Egrid.min() * mapoutdatefac) { // -1e-1 <= -1e-3 * 1e1
        printf("Threshold reached, updating map\n");
        mapupdate=true;
        mapminimum=DBL_MAX;
        outdated.assign(outdated.size(),scanoutdate);
        continue;
      }

      // What's the best function?
      arma::uword minai, mincen, minam;
      while(true) {
        dE=Egrid.min(minai,mincen,minam);
        // Is value up to date?
        if(outdated[minam]==0)
          break;

        // Update minimum value
        Timer tam;
        arma::mat Eam(scan(sys,funcs,avals,Q,M,minam,allowed.slice(minam),E0));

        std::ostringstream oss;
        oss << "Egrid_" << shell_types[minam] << "_" << funcs.size() << ".dat";
        Eam.save(oss.str(),arma::raw_ascii);

        Egrid.slice(minam)=Eam;

        printf("%i%c shell mog = %e done in % .3f (%i out of %i allowed)\n",nprim(funcs,minam)+1,shell_types[minam],Eam.min(),tam.get(),(int) arma::sum(arma::sum(allowed.slice(minam))),(int) (allowed.n_rows*allowed.n_cols));
        fflush(stdout);

        outdated[minam]=0;
        // Update allowed sector
        update_allowed(allowed.slice(minam),Eam);
      }

      // Is the added function a new polarization shell?
      bool newpol(minam == (arma::uword) nam-1);
      if(newpol)
        // Yes, save the current basis set
        save_basis(sys,funcs);

      add_function(avals,minai,mincen,minam,funcs);

      // None of the values are up to date.
      for(size_t i=0;i<outdated.size();i++)
        outdated[i]++;

      printf("Added %c function at %i with exponent % e, energy % .10e changed by % e. nbf = %i, K = %e\n",shell_types[minam],(int) mincen,std::pow(10.0,avals(minai)),E0+dE,dE,nbf(funcs),condition_number(sys,funcs));
      fflush(stdout);

      // Update reference
      double Enew=get_energy(sys,funcs,Q,M,-1);
      double Eold=E0+dE;
      if(std::abs(Enew-Eold)>=1e-7)
        printf("Energy doesn't match: new E0=%e, old E0=%e, difference %e!\n",Enew,Eold,Enew-Eold);
      E0=Enew;

      printf("Scan took %s\n\n",t.elapsed().c_str());
      fflush(stdout);

      if(std::abs(dE)<thr)
        break;

      // Go to next main loop
      if(newpol)
        break;
    }

    if(std::abs(dE)<thr)
      break;
  }

  // Yes, save the current basis set
  save_basis(sys,funcs);

  // Run energy optimization?
  if(optimize) {
    printf("Energy before primitive optimization is % .16e\n",get_energy(sys,funcs,Q,M,0));
    optimize_energy(sys,funcs,Q,M);
  }

  printf("Converged. Final energy without penalty is % .16e\n",get_energy(sys,funcs,Q,M,0)-sys.gamma*condition_number(sys,funcs));

  return 0;
}

int main(int argc, char **argv) {
  try {
    return main_guarded(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
