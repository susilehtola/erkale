#include "lbfgs.h"

LBFGS::LBFGS(size_t nmax_) : nmax(nmax_) {
}

LBFGS::~LBFGS() {
}

void LBFGS::update(const arma::vec & x, const arma::vec & g) {
  xk.push_back(x);
  gk.push_back(g);

  if(xk.size()>nmax) {
    xk.erase(xk.begin());
    gk.erase(gk.begin());
  }
}

arma::vec LBFGS::diagonal_hessian(const arma::vec & q) const {
  arma::vec s=xk[xk.size()-1]-xk[xk.size()-2];
  arma::vec y=gk[gk.size()-1]-gk[gk.size()-2];

  return arma::dot(s,y)/arma::dot(y,y)*q;
}

arma::vec LBFGS::solve() const {
  // Algorithm 9.1 in Nocedal's book
  size_t k=gk.size()-1;
  arma::vec q=gk[k];

  // Sanity check
  if(k==0)
    return q;
  
  // Compute differences
  std::vector<arma::vec> sk(k);
  for(size_t i=0;i<k;i++)
    sk[i]=xk[i+1]-xk[i];
  std::vector<arma::vec> yk(k);
  for(size_t i=0;i<k;i++)
    yk[i]=gk[i+1]-gk[i];

  // Alpha_i
  std::vector<double> alphai(k);
  
  // First part
  for(size_t i=k-1;i<gk.size();i--) {
    // Rho_i
    double rhoi=1.0/arma::dot(yk[i],sk[i]);
    // Alpha_i
    alphai[i]=rhoi*arma::dot(sk[i],q);
    // Update q
    q-=alphai[i]*yk[i];
  }

  // Apply diagonal Hessian
  arma::vec r(diagonal_hessian(q));

  // Second part
  for(size_t i=0;i<k;i++) {
    // Rho_i
    double rhoi=1.0/arma::dot(yk[i],sk[i]);
    // Beta
    double beta=rhoi*arma::dot(yk[i],r);
    // Update r
    r+=sk[i]*(alphai[i]-beta);
  }

  return r;
}
