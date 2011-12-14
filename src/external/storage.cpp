/*
 *                This source code is part of
 *
 *                     E  R  K  A  L  E
 *                             -
 *                       HF/DFT from Hel
 *
 * Written by Jussi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Jussi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */

#include "storage.h"
#include <sstream>
#include <stdexcept>

Storage::Storage() {
}

Storage::~Storage() {
}

void Storage::add(const int_st_t & val) {
  ints.push_back(val);
}

void Storage::add(const double_st_t & val) {
  doubles.push_back(val);
}

void Storage::add(const int_vec_st_t & val) {
  intvec.push_back(val);
}

void Storage::add(const double_vec_st_t & val) {
  doublevec.push_back(val);
}

void Storage::add(const string_st_t & val) {
  strings.push_back(val);
}


int Storage::get_int(const std::string & name) const {
  for(size_t i=0;i<ints.size();i++)
    if(ints[i].name==name)
      return ints[i].val;

  std::ostringstream oss;
  oss << "\nThe entry "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  return 0;
}

double Storage::get_double(const std::string & name) const {
  for(size_t i=0;i<doubles.size();i++)
    if(doubles[i].name==name)
      return doubles[i].val;

  std::ostringstream oss;
  oss << "\nThe entry "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  return 0.0;
}

std::vector<int> Storage::get_int_vec(const std::string & name) const {
  for(size_t i=0;i<intvec.size();i++)
    if(intvec[i].name==name)
      return intvec[i].val;

  std::ostringstream oss;
  oss << "\nThe entry "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  std::vector<int> ret;
  return ret;
}

std::vector<double> Storage::get_double_vec(const std::string & name) const {
  for(size_t i=0;i<doublevec.size();i++)
    if(doublevec[i].name==name)
      return doublevec[i].val;

  std::ostringstream oss;
  oss << "\nThe entry "<<name<<" was not found!\n"; 
  throw std::runtime_error(oss.str());

  std::vector<double> ret;
  return ret;
}

std::string Storage::get_string(const std::string & name) const {
  for(size_t i=0;i<strings.size();i++)
    if(strings[i].name==name)
      return strings[i].val;

  std::ostringstream oss;
  oss << "\nThe entry "<<name<<" was not found!\n";
  throw std::runtime_error(oss.str());

  std::string ret;
  return ret;
}


std::vector<std::string> Storage::find_int(const std::string & name) const {
  std::vector<std::string> ret;

  for(size_t i=0;i<ints.size();i++)
    if(ints[i].name.find(name)!=std::string::npos)
      ret.push_back(ints[i].name);

  return ret;
}

std::vector<std::string> Storage::find_double(const std::string & name) const {
  std::vector<std::string> ret;

  for(size_t i=0;i<doubles.size();i++)
    if(doubles[i].name.find(name)!=std::string::npos)
      ret.push_back(doubles[i].name);

  return ret;
}

std::vector<std::string> Storage::find_int_vec(const std::string & name) const {
  std::vector<std::string> ret;

  for(size_t i=0;i<intvec.size();i++)
    if(intvec[i].name.find(name)!=std::string::npos)
      ret.push_back(intvec[i].name);

  return ret;
}

std::vector<std::string> Storage::find_double_vec(const std::string & name) const {
  std::vector<std::string> ret;

  for(size_t i=0;i<doublevec.size();i++)
    if(doublevec[i].name.find(name)!=std::string::npos)
      ret.push_back(doublevec[i].name);

  return ret;
}

std::vector<std::string> Storage::find_string(const std::string & name) const {
  std::vector<std::string> ret;

  for(size_t i=0;i<strings.size();i++)
    if(strings[i].name.find(name)!=std::string::npos)
      ret.push_back(strings[i].name);

  return ret;
}

void Storage::print(bool vals) const {
  if(ints.size())
    printf("Integers:\n");
  for(size_t i=0;i<ints.size();i++)
    printf("\t%s\t%i\n",ints[i].name.c_str(),ints[i].val);

  if(doubles.size())
    printf("\nDoubles:\n");
  for(size_t i=0;i<doubles.size();i++)
    printf("\t%s\t%e\n",doubles[i].name.c_str(),doubles[i].val);

  if(intvec.size())
    printf("\nInteger vectors:\n");
  for(size_t i=0;i<intvec.size();i++) {
    printf("\t%s\t",intvec[i].name.c_str());
    if(vals) {
      for(size_t j=0;j<intvec[i].val.size();j++)
	printf("%i ",intvec[i].val[j]);
      printf("\n");
    } else {
      printf("%i values\n",(int) intvec[i].val.size());
    }
  }

  if(doublevec.size())
    printf("\nDouble vectors:\n");
  for(size_t i=0;i<doublevec.size();i++) {
    printf("\t%s\t",doublevec[i].name.c_str());
    if(vals) {
      for(size_t j=0;j<doublevec[i].val.size();j++)
	printf("%e ",doublevec[i].val[j]);
      printf("\n");
    } else {
      printf("%i values\n",(int) doublevec[i].val.size());
    }
  }

  if(strings.size())
    printf("\nStrings:\n");
  for(size_t i=0;i<strings.size();i++) {
    printf("\t%s\t",strings[i].name.c_str());
    if(vals) {
      printf("%s ",strings[i].val.c_str());
    } else {
      printf("%i chars\n",(int) strings[i].val.size());
    }
  }

}
