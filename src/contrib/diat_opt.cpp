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

// No z optimization?
#define NOZOPT

// Diatomic basis function
typedef struct {
  int am; // l value
  double a; // exponent
  double z; // z value, as fraction of R
} dim_bf_t;

bool operator<(const dim_bf_t & lh, const dim_bf_t & rh) {
  // First, sort by z value
  if(lh.z < rh.z)
    return true;
  if(lh.z > rh.z)
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

int nbf(const std::vector<dim_bf_t> & funcs) {
  int nbf=0;
  for(size_t i=0;i<funcs.size();i++)
    nbf+=2*funcs[i].am+1;
  return nbf;
}

// Diatomic molecule
typedef struct {
  double R; // bond length
  int Z1; // nuclear charge at left hand
  int Z2; // nuclear charge at right hand
} dim_sys_t;

inline size_t get_index(std::vector<double> atommap, double z) {
  for(size_t i=0;i<atommap.size();i++)
    if(std::abs(atommap[i]-z)<ZTHR)
      return i;

  return std::string::npos;

  /*
    std::vector<double>::iterator it(std::upper_bound(atommap.begin(), atommap.end(), z));
    if(std::abs(*it-z)<ZTHR)
    // OK found it
    return it-atommap.begin();

    if(it!=atommap.begin()) {
    if(std::abs(*(it-1)-z)<ZTHR)
    // OK found it
    return it-1-atommap.begin();
    }

    // Not found
    return std::string::npos;
  */
}

void get_system(std::vector<nucleus_t> & nuclei, std::vector<ElementBasisSet> & els, const dim_sys_t & sys, std::vector<dim_bf_t> funcs) {
  // Map from coordinate to nuclear centers
  std::vector<double> atommap(2);
  atommap[0]=-1.0;
  atommap[1]=1.0;

  // Add nuclei to map
  std::sort(funcs.begin(),funcs.end());
  for(size_t i=0;i<funcs.size();i++) {
    size_t idx(get_index(atommap,funcs[i].z));
    if(idx == std::string::npos) {
      // Add at upper bound
      std::vector<double>::iterator it(std::upper_bound(atommap.begin(), atommap.end(), funcs[i].z));
      atommap.insert(it,funcs[i].z);
    }
  }

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
    // Which center?
    size_t idx(get_index(atommap,funcs[i].z));
    if(idx>=els.size()) {
      arma::conv_to<arma::vec>::from(atommap).print("atommap");

      std::ostringstream oss;
      oss << "Couldn't find index for z=" << funcs[i].z << ", got " << idx << "!\n";
      throw std::logic_error(oss.str());
    }

    FunctionShell sh(funcs[i].am);
    sh.add_exponent(1.0,std::pow(10.0,funcs[i].a));
    els[idx].add_function(sh);
  }
}

void construct_basis(BasisSet & basis, const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, Settings & set) {
  // Get the system
  std::vector<nucleus_t> nuclei;
  std::vector<ElementBasisSet> els;
  get_system(nuclei, els, sys, funcs);

  // Construct basis set
  basis=BasisSet(nuclei.size(),set);
  for(size_t i=0;i<els.size();i++) {
    basis.add_nucleus(nuclei[i]);
    basis.add_shells(i,els[i],true); // sort, always pure functions
  }
  basis.finalize(true);
}

void get_config(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, const Settings & set, int & Q, int & M) {
  // Get charge and multiplicity
  Q=set.get_int("Charge");
  M=set.get_int("Multiplicity");

  // Total number of electrons is
  int Nel=(sys.Z1+sys.Z2)-Q;
  int Nela, Nelb;
  get_Nel_alpha_beta(Nel,M,Nela,Nelb);

  // Do we have enough basis functions to describe all electrons?
  int Nbf=nbf(funcs);
  if(Nbf<std::max(Nela,Nelb)) {
    // This is how many electrons we can fit in the system
    int Nocc=2*Nbf;
    // so the charge is
    Q=sys.Z1+sys.Z2-Nocc;
    // and we have a singlet
    M=1;
  }
}

double get_energy(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, const Settings & set0, int Q, int M, int useref) {
  // Get centers
  BasisSet basis;
  Settings set(set0);
  construct_basis(basis,sys,funcs,set);
  
  set.set_bool("Verbose",false);
  set.add_string("LoadChk","","");

  {
    std::ostringstream chk;
    chk << "erkale_tmp";
#ifdef _OPENMP
    chk << "_" << omp_get_thread_num();
#endif
    chk << ".chk";
    set.add_string("SaveChk","",chk.str());
  }

  if(useref==1) {
    // Load reference calculation
    set.set_string("LoadChk","erkale_ref.chk");
  } else if(useref==-1) {
    // Save reference calculation
    set.set_string("SaveChk","erkale_ref.chk");
  }

  // Set charge and multiplicity
  set.set_int("Charge",Q);
  set.set_int("Multiplicity",M);

  // Run the calculation
  calculate(basis,set,false);

  // Load the result
  Checkpoint chkpt(set.get_string("SaveChk"),false,false);
  energy_t en;
  chkpt.read(en);

  return en.E;
}

arma::uword which_func(arma::uword i) {
#ifdef NOZOPT
  return i;
#else
  return i/2;
#endif
}

std::vector<dim_bf_t> interpret(const arma::vec & p, const std::vector<dim_bf_t> & funcs) {
  std::vector<dim_bf_t> ret(funcs);
#ifdef NOZOPT
  if(p.n_elem != funcs.size()) throw std::logic_error("Wrong vector length!\n");
  for(size_t i=0;i<funcs.size();i++)
    ret[i].a=p(i);
#else
  if(p.n_elem != 2*funcs.size()) throw std::logic_error("Wrong vector length!\n");
  for(size_t i=0;i<funcs.size();i++) {
    ret[i].a=p(2*i);
    ret[i].z=p(2*i+1);
  }
#endif
  return ret;
}

arma::vec collect(const std::vector<dim_bf_t> & funcs) {
#ifdef NOZOPT
  arma::vec ret(funcs.size());
  for(size_t i=0;i<funcs.size();i++)
    ret(i)=funcs[i].a;
#else
  arma::vec ret(2*funcs.size());
  for(size_t i=0;i<funcs.size();i++) {
    ret(2*i)=funcs[i].a;
    ret(2*i+1)=funcs[i].z;
  }
#endif
  
  return ret;
}

arma::vec energy_gradient(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, const Settings & set, int Q, int M, int mval) {
  // Update reference
  get_energy(sys,funcs,set,Q,M,-1);
  
  // Step size
  double h(sqrt(DBL_EPSILON));
  // Collect params
  arma::vec x(collect(funcs));
  
  // Gradient
  arma::vec g(x.n_elem);
  g.zeros();
  for(size_t i=0;i<g.n_elem;i++) {
    // Zero out all other blocks
    if(mval>=0 && funcs[which_func(i)].am!=mval)
      continue;

    arma::vec xlh(x), xrh(x);
    xlh(i)-=h;
    xrh(i)+=h;
      
    double ylh(get_energy(sys,interpret(xlh,funcs),set,Q,M,1));
    double yrh(get_energy(sys,interpret(xrh,funcs),set,Q,M,1));
      
    g(i)=(yrh-ylh)/(2.0*h);
  }

  return g;
}

double optimize_energy(const dim_sys_t & sys, std::vector<dim_bf_t> & funcs, const Settings & set, int Q, int M, int mval=-1) {
  // Current parameters
  arma::vec x(collect(funcs));

  // Get the initial energy
  double E0(get_energy(sys,interpret(x,funcs),set,Q,M,0));
  // Current energy
  double E(E0);
  
  int iit;
  for(iit=0;iit<10;iit++) {
    // Evaluate gradient
    arma::vec g(energy_gradient(sys,interpret(x,funcs),set,Q,M,mval));

    //printf("Gradient norm is %e. Line search\n",arma::norm(g,2));

    // Do line search
    std::vector<double> step, y;
    step.push_back(0.0);
    y.push_back(get_energy(sys,interpret(x,funcs),set,Q,M,1));
    //printf("\t%e % e\n",step[step.size()-1],y[y.size()-1]);

    double h0=sqrt(DBL_EPSILON);
    step.push_back(h0);
    y.push_back(get_energy(sys,interpret(x-g*step[step.size()-1],funcs),set,Q,M,1));
    //printf("\t%e % e % e\n",step[step.size()-1],y[y.size()-1],y[y.size()-1]-y[y.size()-2]);

    if(y[1]>y[0]) {
      //printf("Function value is not decreasing!\n");
      break;
    }

    while(y[y.size()-1]<=y[y.size()-2]) {
      step.push_back(step[step.size()-1]*2);
      y.push_back(get_energy(sys,interpret(x-g*step[step.size()-1],funcs),set,Q,M,1));
      //printf("\t%e % e % e\n",step[step.size()-1],y[y.size()-1],y[y.size()-1]-y[y.size()-2]);
    }

    // Update x and E
    x=x-g*step[step.size()-2];
    E=y[y.size()-2];
  }

  // Update functions
  funcs=interpret(x,funcs);
  std::sort(funcs.begin(),funcs.end());

  printf("Optimization converged in %i step and decreased the value by %e\n",iit,E-E0);

  return E;
}

void add_function(const arma::vec & avals, const arma::vec & zvals, arma::uword minai, arma::uword minzi, int minam, std::vector<dim_bf_t> & funcs) {
  // Add function
  dim_bf_t add;
  add.am=minam;
  add.a=avals(minai);
  add.z=zvals(minzi);
  funcs.push_back(add);
  // Sort values
  std::sort(funcs.begin(),funcs.end());
}

arma::mat scan(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, const Settings & set, const arma::vec & avals, const arma::vec & zvals, int Q, int M, int am, const arma::imat & allow, double E0=0.0, int useref=1) {
  arma::mat ret(avals.n_elem,zvals.n_elem);

#ifndef _HDF5_H
#error "HDF5 headers not included"
#endif
  
#ifdef H5_HAVE_THREADSAFE
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(guided)
#endif
#endif
  for(size_t iz=0;iz<zvals.n_elem;iz++)
    for(size_t ia=0;ia<avals.n_elem;ia++) {
      if(!allow(ia,iz)) {
        // Shortcut
        ret(ia,iz)=0.0;
        continue;
      }
      
      // Add function
      std::vector<dim_bf_t> testfuncs(funcs);
      add_function(avals,zvals,ia,iz,am,testfuncs);
      try {
        ret(ia,iz)=get_energy(sys,testfuncs,set,Q,M,useref)-E0;
      } catch (std::runtime_error & err) {
        ret(ia,iz)=DBL_MAX;
      }
    }

  return ret;
}

arma::mat scan(const dim_sys_t & sys, const std::vector<dim_bf_t> & funcs, const Settings & set, const arma::vec & avals, const arma::vec & zvals, int Q, int M, int am, double E0=0.0, int useref=1) {
  arma::imat allow(avals.n_elem,zvals.n_elem);
  allow.ones();
  return scan(sys,funcs,set,avals,zvals,Q,M,am,allow,E0,useref);
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


int main(int argc, char ** argv) {
  if(argc!=2) {
    printf("Usage: %s runfile\n",argv[0]);
    return 1;
  }

  // Initialize libint
  init_libint_base();

  Settings set;
  set.add_scf_settings();
  set.add_double("Tolerance","Convergence tolerance for addition of a function",1e-6);
  set.add_bool("ForcePol","Force polarized calculation",false);
  set.add_bool("Optimize","Run final energy optimization?",false);
  set.add_int("MinLogExp","Minimum exponent value",-4);
  set.add_int("MaxLogExp","Maximum exponent value",9);
  set.add_int("NExp","Number of points on exponent grid",131);
  set.add_int("ScanOutdate","Redo scans every N new functions",4);
  set.add_double("MapOutdate","Outdate scan maps when minimum increases by factor",10.0);

  set.parse(std::string(argv[1]),true);
  // Must use core guess
  set.set_string("Guess","Core");
  // Use dimer symmetry
  set.set_bool("DimerSymmetry",true);
  // .. which requires setting
  set.set_bool("OptLM",false);
  // No basis set rotation
  set.set_bool("BasisRotate",false);
  set.print();

  std::vector<std::string> sysv(splitline(set.get_string("System")));
  if(sysv.size() != 3 && sysv.size() != 4) {
    throw std::logic_error("System specification invalid: format Z1 Z2 R (angstrom)\n");
  }

  // Parse system
  dim_sys_t sys;
  sys.Z1=readint(sysv[0]);
  sys.Z2=readint(sysv[1]);
  sys.R=readdouble(sysv[2]);
  if(sysv.size()>3) {
    std::string spec(sysv[3]);
    if(stricmp(spec,"angstrom")==0)
      sys.R*=ANGSTROMINBOHR;
    else if(stricmp(spec,"bohr")!=0)
      throw std::logic_error("Invalid unit specification " + spec + "!\n");
  }
  double thr(set.get_double("Tolerance"));

  // Maximum and minimum exponent
  const double mina(set.get_int("MinLogExp"));
  const double maxa(set.get_int("MaxLogExp"));
  // Exponent grid
  const int Nexp(set.get_int("NExp"));

  int scanoutdate(set.get_int("ScanOutdate"));
  double mapoutdatefac(set.get_double("MapOutdate"));
  bool optimize(set.get_bool("Optimize"));
  
  // Geometry grid: left, middle, right
  const int Ngeom = (sys.Z1 == 0 || sys.Z2 == 0) ? 1 : 3;
  if(Ngeom == 1)
    // Fictitious bond length, set to zero so functions are centered
    // at the nucleus
    sys.R = 0.0;

  if(sys.Z1 == 0 || sys.Z2 == 0)
    printf("Optimizing basis set for %s atom\n",element_symbols[std::max(sys.Z1,sys.Z2)].c_str());
  else
    printf("Optimizing basis set for %s-%s dimer with bond distance R=%e a.u.\n",element_symbols[sys.Z1].c_str(),element_symbols[sys.Z2].c_str(),sys.R);
  printf("Continuing optimization until energy drops by less than %e\n",thr);
  printf("Spacing exponents by factor %e.\n",std::pow(10.0,(maxa-mina)/Nexp));
  printf("Full exponent scans are made whenever the minimum increases by %e.\n",mapoutdatefac);
  printf("Exponent subscans are made at least every %i iterations.\n\n",scanoutdate);

  // Grids
  arma::vec avals(arma::linspace<arma::vec>(mina,maxa,Nexp));
  avals.save("x_expn.dat",arma::raw_ascii);

  arma::vec zvals(arma::linspace<arma::vec>(-1.0,1.0,Ngeom));
  zvals.save("y_geom.dat",arma::raw_ascii);
  // Make sure values are O.K.
  zvals(0)=-1.0;
  zvals(zvals.n_elem-1)=1.0;

  std::vector<dim_bf_t> funcs;

  // Charge and multiplicity
  int Q, M;
  {
    // Dummy basis: one s function
    std::vector<dim_bf_t> dumfuncs;
    add_function(avals,zvals,0,0,0,dumfuncs);
    get_config(sys,dumfuncs,set,Q,M);
  }  

  // Current energy
  double E0;
  // Energy change
  double dE;
  
  // Initialize basis
  while(true) {
    // Form checkpoint
    printf("Running now with charge %i, multiplicity %i ",Q,M);
    fflush(stdout);
    if(funcs.size())
      E0=get_energy(sys,funcs,set,Q,M,-1);
    else
      E0=0.0;
    printf("ref energy %e\n",E0);
    fflush(stdout);
    
    // Values to scan
    int nam(maxam(funcs)+2);
    arma::cube Egrid(avals.n_elem,zvals.n_elem,nam);

    for(int am=0;am<nam;am++) {
      Timer tam;
      //arma::mat Eam=scan(sys,funcs,avals,zvals,Q,M,am,E0,(funcs.size()>0));
      arma::mat Eam=scan(sys,funcs,set,avals,zvals,Q,M,am,E0,0);
      
      std::ostringstream oss;
      oss << "Egrid_" << shell_types[am] << "_" << funcs.size() << ".dat";
      Eam.save(oss.str(),arma::raw_ascii);
      
      Egrid.slice(am)=Eam;
      
      printf("%c shell done in % .3f\n",shell_types[am],tam.get());
      fflush(stdout);
    }

    // What's the best function?
    arma::uword minai, minzi, minam;
    dE=Egrid.min(minai,minzi,minam);
    add_function(avals,zvals,minai,minzi,minam,funcs);

    printf("Added %c function at % .6f with exponent % e, energy % .10e changed by % e. nbf = %i\n",shell_types[minam],zvals(minzi),std::pow(10.0,avals(minai)),E0+dE,dE,nbf(funcs));

    // Do we have a minimal basis now?
    get_config(sys,funcs,set,Q,M);
    if(Q==set.get_int("Charge") && M==set.get_int("Multiplicity"))
      break;
  }

  // Current energy
  printf("Calculation initialized, running now with target charge %i, multiplicity %i.\n",Q,M);
  fflush(stdout);
  E0=get_energy(sys,funcs,set,Q,M,-1);
  printf("Initial energy is % .16e\n\n",E0);

  while(true) {
    int nam(maxam(funcs)+2);

    // Update map at next iteration?
    bool mapupdate=true;
    // Minimum value seen on map
    double mapminimum=DBL_MAX;

    // Value grid
    arma::cube Egrid(avals.n_elem,zvals.n_elem,nam);
    // Allow scan?
    arma::icube allowed(avals.n_elem,zvals.n_elem,nam);
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
          arma::mat Eam(scan(sys,funcs,set,avals,zvals,Q,M,am,allowed.slice(am),E0));
          
          std::ostringstream oss;
          oss << "Egrid_" << shell_types[am] << "_" << funcs.size() << ".dat";
          Eam.save(oss.str(),arma::raw_ascii);
          
          Egrid.slice(am)=Eam;
          
          printf("%c shell mog = %e done in % .3f (%i out of %i allowed)\n",shell_types[am],Eam.min(),tam.get(),(int) arma::sum(arma::sum(allowed.slice(am))),(int) (allowed.n_rows*allowed.n_cols));
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
      arma::uword minai, minzi, minam;
      while(true) {
        dE=Egrid.min(minai,minzi,minam);
        // Is value up to date?
        if(outdated[minam]==0)
          break;

        // Update minimum value
        Timer tam;
        arma::mat Eam(scan(sys,funcs,set,avals,zvals,Q,M,minam,allowed.slice(minam),E0));
        
        std::ostringstream oss;
        oss << "Egrid_" << shell_types[minam] << "_" << funcs.size() << ".dat";
        Eam.save(oss.str(),arma::raw_ascii);
        
        Egrid.slice(minam)=Eam;
        
        printf("%c shell mog = %e done in % .3f (%i out of %i allowed)\n",shell_types[minam],Eam.min(),tam.get(),(int) arma::sum(arma::sum(allowed.slice(minam))),(int) (allowed.n_rows*allowed.n_cols));
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
      
      add_function(avals,zvals,minai,minzi,minam,funcs);

      // None of the values are up to date.
      for(size_t i=0;i<outdated.size();i++)
        outdated[i]++;

      printf("Added %c function at % .6f with exponent % e, energy % .10e changed by % e. nbf = %i\n",shell_types[minam],zvals(minzi),std::pow(10.0,avals(minai)),E0+dE,dE,nbf(funcs));
      fflush(stdout);

      // Update reference
      double Enew=get_energy(sys,funcs,set,Q,M,-1);
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
  if(optimize)
    optimize_energy(sys,funcs,set,Q,M);

  printf("Converged. Final energy is % .16e\n",get_energy(sys,funcs,set,Q,M,0));
  
  return 0;
}
