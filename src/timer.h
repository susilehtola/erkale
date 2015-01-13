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



#include "global.h"

#ifndef ERKALE_TIMER
#define ERKALE_TIMER

#include <ctime>
#include <string>

extern "C" {
#include <sys/time.h>
}

/**
 * \class Timer
 *
 * \brief A timer routine
 *
 * This class implements a timer that can be used to measure runtimes
 * of routines.
 *
 * \author Susi Lehtola
 * \date 2011/01/26 21:54
*/

class Timer {
  /// Time when timer was started
  struct timespec tstart;

  /// Elapsed time
  double elapsd;

  /// Read time
  void read(struct timespec *t) const;

 public:
  /// Constructor
  Timer();
  /// Destructor
  ~Timer();

  /// Zero timer
  void set();

  /// Stop timer
  void stop();
  /// Continue timing
  void cont();

  /// Get current time
  std::string current_time() const;

  /// Print elapsed time
  void print() const;
  /// Print current time
  void print_time() const;

  /// Get elapsed time
  double get() const;
  /// Get pretty-printed elapsed time
  std::string elapsed() const;
  /// Get pretty-printed time
  std::string parse(double t) const;
};

#endif
