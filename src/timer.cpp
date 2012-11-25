/*
 *                This source code is part of
 * 
 *                     E  R  K  A  L  E
 *                             -
 *                       DFT from Hel
 *
 * Written by Susi Lehtola, 2010-2011
 * Copyright (c) 2010-2011, Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */



#include <cmath>
#include <cstdio>
#include <string>
#include <sstream>

#include "timer.h"

Timer::Timer() {
  set();
}

Timer::~Timer() {
}

void Timer::stop() {
  struct timespec tstop;
  clock_gettime(CLOCK_REALTIME,&tstop);

  elapsd+=(tstop.tv_sec-tstart.tv_sec)+(tstop.tv_nsec-tstart.tv_nsec)*1.0e-9;
}

void Timer::cont() {
  clock_gettime(CLOCK_REALTIME,&tstart);
}

void Timer::set() {
  // Get time.
  clock_gettime(CLOCK_REALTIME,&tstart);
  elapsd=0.0;
}

void Timer::print() const {
  printf("Time elapsed is %s.\n",elapsed().c_str());
}

std::string Timer::current_time() const {
  char out[256];

  // Get time
  time_t t;
  time(&t);

  // Convert it into struct tm
  struct tm tm;
  gmtime_r(&t,&tm);

  const char * days[]={"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
  const char * months[]={"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};

  // Print time
  sprintf(out,"%s %02i %s %4i %02i:%02i:%02i",days[tm.tm_wday],tm.tm_mday,months[tm.tm_mon],1900+tm.tm_year,tm.tm_hour,tm.tm_min,tm.tm_sec);
  return std::string(out);
}

void Timer::print_time() const {
  printf("Current time is %s.\n",current_time().c_str());
}

double Timer::get() const {
  struct timespec tstop;
  clock_gettime(CLOCK_REALTIME,&tstop);

  return elapsd+(tstop.tv_sec-tstart.tv_sec)+(tstop.tv_nsec-tstart.tv_nsec)*1.0e-9;
}

std::string Timer::elapsed() const {
  std::ostringstream ret;

  // Get elapsed time
  double telapsed=get();

  // Minute is 60 sec
  size_t min=60;
  // Hour is 60 minutes
  size_t hour=60*min;
  // Day is 24 hours
  size_t day=24*hour;

  // Compute number of days
  size_t days=(size_t) trunc(telapsed/day);
  if(days) {
    telapsed-=days*day;

    ret << days << " d";
  }
  
  // Compute number of hours
  size_t hours=(size_t) trunc(telapsed/hour);
  if(hours) {
    telapsed-=hours*hour;

    // Check that there is a space at the end
    std::string tmp=ret.str();
    if(tmp.size() && tmp[tmp.size()-1]!=' ')
      ret << " ";
    
    ret << hours << " h";
  }

  // Compute number of minutes
  size_t mins=(size_t) trunc(telapsed/min);
  if(mins) {
    telapsed-=mins*min;

    // Check that there is a space at the end
    std::string tmp=ret.str();
    if(tmp.size() && tmp[tmp.size()-1]!=' ')
      ret << " ";
    
    ret << mins << " min";
  }

  // Check that there is a space at the end
  std::string tmp=ret.str();
  if(tmp.size() && tmp[tmp.size()-1]!=' ')
    ret << " ";

  char hlp[80];
  sprintf(hlp,"%.2f s",telapsed);
  ret << hlp;
  
  return ret.str();
}
