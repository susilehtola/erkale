/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2015
 * Copyright (c) 2010-2015, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "eri_digest.h"

IntegralDigestor::IntegralDigestor() {
}

IntegralDigestor::~IntegralDigestor() {
}

JDigestor::JDigestor(const arma::mat & P_) : P(P_) {
  J.zeros(P.n_rows,P.n_cols);
}

JDigestor::~JDigestor() {
}

void JDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;
  digest_J(shpairs,ip,jp,ints,ioff,P,J);
}

arma::mat JDigestor::get_J() const {
  return J;
}

JvDigestor::JvDigestor(const std::vector<arma::mat> & P_) : P(P_) {
  J.resize(P.size());
  for(size_t i=0;i<P.size();i++)
    J[i].zeros(P[i].n_rows,P[i].n_cols);
}

JvDigestor::~JvDigestor() {
}

void JvDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;
  for(size_t i=0;i<P.size();i++)
    digest_J(shpairs,ip,jp,ints,ioff,P[i],J[i]);
}

std::vector<arma::mat> JvDigestor::get_J() const {
  return J;
}

KDigestor::KDigestor(const arma::mat & P_) : P(P_) {
  K.zeros(P.n_rows,P.n_cols);
}

KDigestor::~KDigestor() {
}

void KDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;
  digest_K(shpairs,ip,jp,ints,ioff,double,P,K);
}

arma::mat KDigestor::get_K() const {
  return K;
}

JKDigestor::JKDigestor(const arma::mat & P_) : P(P_) {
  J.zeros(P.n_rows,P.n_cols);
  K.zeros(P.n_rows,P.n_cols);
}

JKDigestor::~JKDigestor() {
}

void JKDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;
  digest_J(shpairs,ip,jp,ints,ioff,P,J);
  digest_K(shpairs,ip,jp,ints,ioff,double,P,K);
}

arma::mat JKDigestor::get_J() const {
  return J;
}

arma::mat JKDigestor::get_K() const {
  return K;
}

KabDigestor::KabDigestor(const arma::mat & Pa_, const arma::mat & Pb_) : Pa(Pa_), Pb(Pb_) {
  Ka.zeros(Pa.n_rows,Pa.n_cols);
  Kb.zeros(Pa.n_rows,Pa.n_cols);
}

KabDigestor::~KabDigestor() {
}

void KabDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;
  digest_K(shpairs,ip,jp,ints,ioff,double,Pa,Ka);
  digest_K(shpairs,ip,jp,ints,ioff,double,Pb,Kb);
}

arma::mat KabDigestor::get_Ka() const {
  return Ka;
}

arma::mat KabDigestor::get_Kb() const {
  return Kb;
}

JKabDigestor::JKabDigestor(const arma::mat & Pa_, const arma::mat & Pb_) : P(Pa_+Pb_), Pa(Pa_), Pb(Pb_) {
  J.zeros(P.n_rows,P.n_cols);
  Ka.zeros(Pa.n_rows,Pa.n_cols);
  Kb.zeros(Pa.n_rows,Pa.n_cols);
}

JKabDigestor::~JKabDigestor() {
}

void JKabDigestor::digest(const std::vector<eripair_t> & shpairs, size_t ip, size_t jp, const std::vector<double> & ints, size_t ioff) {
  size_t is=shpairs[ip].is;
  size_t js=shpairs[ip].js;
  size_t ks=shpairs[jp].is;
  size_t ls=shpairs[jp].js;
  digest_J(shpairs,ip,jp,ints,ioff,P,J);
  digest_K(shpairs,ip,jp,ints,ioff,double,Pa,Ka);
  digest_K(shpairs,ip,jp,ints,ioff,double,Pb,Kb);
}

arma::mat JKabDigestor::get_J() const {
  return J;
}

arma::mat JKabDigestor::get_Ka() const {
  return Ka;
}

arma::mat JKabDigestor::get_Kb() const {
  return Kb;
}
