#--------------------------------------------------------------------
AR=ar
CC=gcc
CXX=g++
CPP=cpp

INCLUDE=-I../armadillo-1.1.92/include -I../src
#CXXFLAGS=-O2 -ansi -g -Wall -DDFT_ENABLED -fPIC $(INCLUDE)
CXXFLAGS=-O2 -ansi -g -Wall -fPIC $(INCLUDE)
CFLAGS=$(CXXFLAGS)
#LDFLAGS=-L/usr/lib64/atlas -llapack -latlas -lgsl -lint -lxc -lprofiler -ltcmalloc -L.
LDFLAGS=-L/usr/lib64/atlas -llapack -latlas -lgsl -lint -lxc -L.

# Flag to enable OpenMP
OPENMP="-fopenmp"

#-------------- No changes necessary hereafter ----------------------

all:	dirs
	+make -C build-ser -f ../Makefile execs
	+make -C build-par -f ../Makefile execs CXXFLAGS="$(CXXFLAGS) $(OPENMP)" OMP="$(OPENMP)"

doc:	
	doxygen

dirs:
	-mkdir build-ser build-par

execs:	$(OBJS) $(EXEOBJS) $(EMDOBJS)
	+make -f ../Makefile $(ERKALE) $(BASGEN) $(SLATER) $(COPROF)

LIBERKALE_SHARED=liberkale.so
LIBERKALE_EMD_SHARED=liberkale_emd.so

LIBERKALE_STATIC=liberkale.a
LIBERKALE_EMD_STATIC=liberkale_emd.a

LIBERKALE=$(LIBERKALE_STATIC)
LIBERKALE_EMD=$(LIBERKALE_EMD_STATIC)

ERKALE=erkale.x
COPROF=coprof.x
SLATER=slaterfit.x

$(ERKALE):	main.o $(LIBERKALE) $(LIBERKALE_EMD)
#	$(CXX) $(CXXFLAGS) $(LDFLAGS) -lerkale -o $(ERKALE) main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(ERKALE) main.o liberkale.a liberkale_emd.a

$(SLATER):	form_exponents.o solve_coefficients.o tempered.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(SLATER) form_exponents.o solve_coefficients.o tempered.o

$(COPROF):	print_completeness.o $(LIBERKALE)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(COPROF) print_completeness.o liberkale.a

EXEOBJS=hf.o dft.o print_completeness.o

OBJS=basis.o basislibrary.o stringutil.o mathf.o integrals.o eritable.o eriscreen.o timer.o linalg.o obara-saika.o solidharmonics.o diis.o scf.o elements.o xyzutils.o settings.o lobatto.o dftgrid.o dftfuncs.o chebyshev.o density_fitting.o broyden.o adiis.o lebedev.o tempered.o completeness_profile.o

EMDOBJS=complex.o emd.o gto_fourier.o spherical_expansion.o spherical_harmonics.o

$(LIBERKALE_SHARED):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(LIBERKALE) -shared $(OBJS)

$(LIBERKALE_STATIC):	$(OBJS)
	$(AR) rcs $(LIBERKALE_STATIC) $(OBJS)

$(LIBERKALE_EMD_SHARED):	$(LIBERKALE_SHARED) $(EMDOBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(LIBERKALE_EMD) -lerkale -shared $(EMDOBJS)

$(LIBERKALE_EMD_STATIC):	$(EMDOBJS)
	$(AR) rcs $(LIBERKALE_EMD_STATIC) $(EMDOBJS) 

scf.o:	scf-base.cpp scf-solvers.cpp.in
	cat ../src/scf-base.cpp > scf.cpp
	$(CPP) -E ../src/scf-solvers.cpp.in >> scf.cpp
	$(CPP) -E -DDFT ../src/scf-solvers.cpp.in >> scf.cpp
	$(CPP) -E -DRESTRICTED ../src/scf-solvers.cpp.in >> scf.cpp
	$(CPP) -E -DDFT -DRESTRICTED  ../src/scf-solvers.cpp.in >> scf.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c scf.cpp -o scf.o
	\rm scf.cpp

eritable.o:	eritable-base.cpp eri-routines.cpp.in eri-routines-increment.cpp.in
	cat ../src/eritable-base.cpp > eritable.cpp
	$(CPP) -E $(INCLUDE) $(OMP) -x c++ -DERITABLE ../src/eri-routines.cpp.in >> eritable.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c eritable.cpp -o eritable.o
	#\rm eritable.cpp

eriscreen.o:	eriscreen-base.cpp eri-routines.cpp.in eri-routines-increment.cpp.in
	cat ../src/eriscreen-base.cpp > eriscreen.cpp
	$(CPP) -E $(INCLUDE) $(OMP) -DCALCJKab ../src/eri-routines.cpp.in >> eriscreen.cpp
	$(CPP) -E $(INCLUDE) $(OMP) -DCALCJK ../src/eri-routines.cpp.in >> eriscreen.cpp
	$(CPP) -E $(INCLUDE) $(OMP) -DCALCK ../src/eri-routines.cpp.in >> eriscreen.cpp
	$(CPP) -E $(INCLUDE) $(OMP) -DCALCJ ../src/eri-routines.cpp.in >> eriscreen.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c eriscreen.cpp -o eriscreen.o
	#\rm eriscreen.cpp

clean:
	rm -f build-ser/* build-par/*

# Look for source in
VPATH=../src ../src/emd ../src/slaterfit ../src/completeness

# Compilation rules
%.o: %.c %.h
	$(CC)  $(CFLAGS)   -c $< -o $@
%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
