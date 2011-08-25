#include <cfloat>
#include "optimize_completeness.h"
#include "completeness_profile.h"
#include "../basislibrary.h"
#include "../mathf.h"

extern "C" {
#include <gsl/gsl_multimin.h>
}

// Maximum number of iterations
#define MAXITER 500
// Tolerance for optimization
#define OPTTOL 1e-15

double evaluate_completeness(const gsl_vector *v, void *params) {
  // Create element basis set.

  // Angular momentum of shell to optimize is
  completeness_scan_t *par=(completeness_scan_t *) params;

  // Helper structure
  ElementBasisSet el;
  for(size_t i=0;i<v->size;i++) {
    // Create shell of functions
    FunctionShell tmp(par->am);
    tmp.add_exponent(1.0,pow(10.0,gsl_vector_get(v,i)));
    // and add it to the basis set
    el.add_function(tmp);
  }

  // Number of points to use in scan
  size_t Npoints;
  //  Npoints=(int) ceil((par->max-par->min)/0.01);
  Npoints=2000;
  // and it needs to be odd for the Simpson rule to work
  if(Npoints%2==0)
    Npoints++;

  // Now we can compute the completeness profile. Use Cholesky factorization.
  compprof_t prof=compute_completeness(el, par->scanexp, 1);

  // Evaluate the mean square difference from completeness
  double cpl=0.0;
  for(size_t i=1;i<Npoints-1;i+=2) {
    // Compute differences from unity
    double ld=prof.shells[par->am].Y[i-1]-1.0;
    double md=prof.shells[par->am].Y[i]-1.0;
    double rd=prof.shells[par->am].Y[i+1]-1.0;
    // Increment profile measure
    cpl+=ld*ld+4.0*md*md+rd*rd;
  }
  // Plug in normalization factors (this should also hold the length of the integration interval, but we drop it out)
  cpl/=6.0*prof.lga.size();

  return sqrt(cpl);
}

double evaluate_completeness(const std::vector<double> & v, completeness_scan_t p) {
  // Make helper variable
  gsl_vector *x = gsl_vector_alloc(v.size());
  for(size_t i=0;i<v.size();i++)
    gsl_vector_set(x,i,v[i]);
  
  // Calculate result
  double res=evaluate_completeness(x,(void *) &p);
  // Free memory
  gsl_vector_free(x);

  return res;
}  


std::vector<double> optimize_completeness(int am, double pmin, double pmax, int Nf) {
  // Logarithms of exponents
  std::vector<double> loge(Nf);
  // Step sizes
  std::vector<double> ss(Nf);

  // Parameters for minimization
  completeness_scan_t par;
  par.am=am;
  // Scanning exponents
  par.scanexp=get_scanning_exponents(pmin,pmax,2000);

  // Starting point: spread exponents evenly
  double dx=(pmax-pmin)/(Nf-1);
  for(int i=0;i<Nf;i++) {
    loge[i]=pmin+(i+0.5)*dx;
    // Initialize step size to one tenth of the spacing
    ss[i]=0.1*dx;
    //    ss[i]=1.0;
  }

  // Loop order of trials
  std::vector<size_t> order;
  order.reserve(Nf);
  for(int i=0;i<Nf/2;i++) {
    // Do the borderline exponents first, since they should be a lot
    // easier to move.
    order[2*i]=i;
    order[2*i+1]=Nf-1-i;
  }
  // Handle case of odd number of functions
  if(Nf%2==1)
    order[Nf-1]=Nf/2+1;

  // Current cost
  double cost=evaluate_completeness(loge,par);
  // Old value of cost
  double oldcost=DBL_MAX;

  // Loop until maximal step size is smaller than epsilon.
  printf("\tIter\tcost\n");
  for(unsigned int iter=0;iter<MAXITER;iter++) {
    // Compute rms step size
    double rmsstep=0;
    for(int i=0;i<Nf;i++)
      rmsstep+=ss[i]*ss[i];
    //    rmsstep=sqrt(rmsstep/Nf);
    rmsstep=sqrt(rmsstep)/Nf;
    
    // Store value of cost
    oldcost=cost;

    // Increase all step sizes.
    for(int i=0;i<Nf;i++)
      ss[i]*=1.2;

    // Loop over functions
    for(int i=0;i<Nf;i++) {
      // Exponent to optimize is
      int ix=order[i];
      
      while(ss[ix]>DBL_EPSILON) {
	// Compute trial exponents
	std::vector<double> left(loge);
	left[ix]-=ss[ix];
	
	std::vector<double> right(loge);
	right[ix]+=ss[ix];
	
	// and trial costs
	double ly;
	try {
	  // Evaluate completeness
	  ly=evaluate_completeness(left,par);
	} catch(...) {
	  // Catch errors caused by badly behaving matrix
	  ly=DBL_MAX;
	}

	double ry;
	try {
	  ry=evaluate_completeness(right,par);
	} catch(...) {
	  ry=DBL_MAX;
	}
	
	// Was optimal value already used?
	bool opt_ok=0; // Optimal
	bool opt_l=0; // Left value
	bool opt_r=0; // Right value

	/*
	// Now we can do a parabola fit.
	double my=cost;
	double lx=left[order[i]];
	double mx=loge[order[i]];
	double rx=right[order[i]];	

        // Parabola fit parameters
	double p=(mx-lx)*(my-ry);
	double q=(mx-rx)*(my-ly);

	if(fabs(p-q)>100*DBL_EPSILON) {
	  // The optimal value should be at
	  double xopt=mx-0.5*((mx-lx)*p-(mx-rx)*q)/(p-q);

	  // Compute cost at optimal exponent
	  std::vector<double> opt(loge);
	  opt[order[i]]=xopt;
	  double yopt=evaluate_completeness(opt,par);

	  // Does move result in a minimization?
	  if(yopt<cost) {
	    // Yes, accept it.
	    opt_ok=1;
	    loge=opt;
	    cost=yopt;
	  }
	}
	*/

	// Check left and right end.
	if(ly<cost) {
	  opt_l=1;
	  cost=ly;
	  loge=left;
	}

	if(ry<cost) {
	  opt_r=1;
	  cost=ry;
	  loge=right;
	}

	// Decrease step size due to failed try?
	if(!opt_ok && !opt_l && !opt_r)
	  ss[ix]/=2.0;
	else
	  // Exit while loop
	  break;
      }
    }

    // Print info
    printf("\t%u\t%e\t%e\t%e\n",iter,cost,cost-oldcost,rmsstep);	

    // Check if cost was decreased in meaningful way
    if(oldcost-cost<OPTTOL)
      break;

    // Break due to small step size
    if(max(ss)<DBL_EPSILON)
      break;
  }

  // Returned exponents
  std::vector<double> exps(Nf);
  for(int i=0;i<Nf;i++)
    exps[i]=pow(10.0,loge[i]);

  return exps;    
}


std::vector<double> optimize_completeness_gsl(int am, double min, double max, int Nf) {
  const gsl_multimin_fminimizer_type *T;
  gsl_multimin_fminimizer *s;
     
  gsl_vector *x;
  gsl_vector *step;
  gsl_multimin_function minf;

  // Parameters for minimization
  completeness_scan_t par;
  par.am=am;
  // Scanning exponents
  par.scanexp=get_scanning_exponents(min,max,2000);
     
  minf.n = Nf;
  minf.f = &evaluate_completeness;
  minf.params =(void *) &par;
     
  // Starting point: spread exponents evenly
  double dx=(max-min)/(Nf-1);
  x = gsl_vector_alloc (Nf);
  for(int i=0;i<Nf;i++) {
    // Set value of exponent to
    double expn=min+(i+0.5)*dx;
    gsl_vector_set(x,i,expn);
  }

  // Set all step sizes to a tenth of the initial spacing
  step = gsl_vector_alloc(Nf);
  for(int i=0;i<Nf;i++)
    gsl_vector_set(step,i,0.1*dx);

  // Use Nead-Miller simplex algorithm
  T = gsl_multimin_fminimizer_nmsimplex2;
  s = gsl_multimin_fminimizer_alloc (T, Nf);

  // Initialize the minimizer, use a tenth of the spacing as initial step size
  gsl_multimin_fminimizer_set (s, &minf, x, step);

  // Iteration index
  size_t iter = 0;
  // Status of algorithm
  int status;
  // "Size" of the minimizer
  double size;
  
  do {
    iter++;
    status = gsl_multimin_fminimizer_iterate (s);
    
    if (status)
      break;
    
    size = gsl_multimin_fminimizer_size (s);
    status = gsl_multimin_test_size (size, 1e-2);
  } while (status == GSL_CONTINUE && iter < MAXITER);
  
  if(status == GSL_SUCCESS)
    printf ("Minimum %e found at:\n",s->fval);
  else
    printf ("Failed to find a minimum, current guess is %e at:\n",s->fval);
  
  for(int i=0;i<Nf;i++)
    printf(" %f",gsl_vector_get(s->x,i));
  printf("\n");      

  // Get exponents
  std::vector<double> ret(Nf);
  for(int i=0;i<Nf;i++)
    ret[i]=pow(10.0,gsl_vector_get(s->x,i));
  
  // Free memory
  gsl_multimin_fminimizer_free (s);
  gsl_vector_free (x);

  return ret;
}

