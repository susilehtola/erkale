#!/bin/bash
# This is a script for downloading, compiling and 
# installing ERKALE with all of its prerequisite libraries and CMake.
# 2011-11-12 Susi Lehtola

# Set this to the number of cores +1
nprocs=9

# Archiver
export AR="ar"
# C compiler
export CC="gcc"
# C++ compiler
export CXX="g++"
# Fortran compiler
export F77="gfortran"
export FC="${F77}"

# C preprosessor
export CPP="${CC} -E"
# Fortran preprocessor
export FCCPP="${FC} -E"

# C flags to use. For older compilers you may need to specify the architecture
# by hand.
export CFLAGS="-Wall -g -O2 -fPIC -march=native"

# C++ flags to use
export CXXFLAGS="${CFLAGS}"
# Fortran flags to use
export FFLAGS="${CFLAGS}"
export FCFLAGS="${CFLAGS}"

# LAPACK and BLAS library to use.
LAPACK="-L/usr/lib64/atlas -llapack -lf77blas -lcblas -latlas"
BLAS="-L/usr/lib64/atlas -lf77blas -lcblas -latlas"
# Generic lapack and blas. Don't use these unless there is nothing
# else available (e.g. on Cygwin)
# LAPACK="-llapack -lblas -lgfortran"
# BLAS="-lblas -lgfortran"

# Use system packages?
system_cmake=0
system_gsl=0
system_libxc=0
system_libint=0
system_hdf5=0

# Maximum supported angular momentum (affects libint if it's compiled)
MAXAM="6"
# Maximum optimized angular momentum (affects libint if it's
# compiled). If this is very large, libint compilation will take ages
# and the resulting libraries will be HUGE.
OPTAM="4"
# Maximum angular momentum for first ERI derivatives (affects libint
# if it's compiled)
MAXDERIV="5"

# Running on cygwin?
if [[ "$CYGWIN" != "" ]]; then
    # Silence cmake warnings about changed behavior
    export CMAKE_LEGACY_CYGWIN_WIN32=0
fi

# Current versions of libraries, if they are to be compiled
export GSLVER="1.16"
export XCVER="2.2.1"
# libint 1.1.6
export INTVER="0e0ffa7887e74e6ab1fb07c89be55f776c733731"
export ARMAVER="4.650.0"
export CMAKEVER="3.1.3"
export HDF5VER="1.8.14"

############### NO CHANGES NECESSARY HEREAFTER ##################

# Current dir is
topdir="`pwd`"


srcdir=${topdir}/sources
if [ ! -d ${srcdir} ]; then
 mkdir -p ${srcdir}
fi

builddir=${topdir}/build
if [ ! -d ${builddir} ]; then
 mkdir -p ${builddir}
fi

# GSL
if(( ! ${system_gsl} )); then
    if [ ! -f ${topdir}/gsl/lib/libgsl.a ]; then
	echo -n "Compiling GSL ..."
	
	if [ ! -d ${builddir}/gsl-${GSLVER} ]; then
	    if [ ! -f ${srcdir}/gsl-${GSLVER}.tar.gz ]; then
		cd ${srcdir}
		wget -O gsl-${GSLVER}.tar.gz ftp://ftp.gnu.org/gnu/gsl/gsl-${GSLVER}.tar.gz
	    fi
	    cd ${builddir}
	    tar zxf ${srcdir}/gsl-${GSLVER}.tar.gz
	fi
	
	cd ${builddir}/gsl-${GSLVER}/
	./configure --enable-static --disable-shared --prefix=${topdir}/gsl --exec-prefix=${topdir}/gsl &>configure.log
	make -j ${nprocs} &> make.log
	make install &> install.log
	make clean &> clean.log
	echo " done"
    fi

    if [ ! -f ${topdir}/gsl/lib/libgsl.a ]; then
	echo "Error compiling GSL."
	exit
    fi
fi

# HDF5
if(( ! ${system_hdf5} )); then
    if [ ! -f ${topdir}/hdf5/lib/libhdf5.a ]; then
	echo -n "Compiling HDF5 ..."
	
	if [ ! -d ${builddir}/hdf5-${HDF5VER} ]; then
	    if [ ! -f ${srcdir}/hdf5-${HDF5VER}.tar.gz ]; then
		cd ${srcdir}
		wget -O hdf5-${HDF5VER}.tar.gz http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5VER}/src/hdf5-${HDF5VER}.tar.gz
	    fi
	    cd ${builddir}
	    tar zxf ${srcdir}/hdf5-${HDF5VER}.tar.gz
	fi
	
	cd ${builddir}/hdf5-${HDF5VER}/
	./configure --enable-static --disable-shared --prefix=${topdir}/hdf5 --exec-prefix=${topdir}/hdf5 &>configure.log
	make -j ${nprocs} VERBOSE=1 &> make.log
	make install &> install.log
	make clean &> clean.log
	echo " done"
    fi 

    if [ ! -f ${topdir}/hdf5/lib/libhdf5.a ]; then
	echo "Error compiling HDF5."
	exit
    fi
fi

# libXC
if(( ! ${system_libxc} )); then
    if [ ! -f ${topdir}/libxc/lib/libxc.a ]; then
	echo -n "Compiling libxc ..."
	
	if [ ! -d ${builddir}/libxc-${XCVER} ]; then
	    if [ ! -f ${srcdir}/libxc-${XCVER}.tar.gz ]; then
		cd ${srcdir}
		wget -O libxc-${XCVER}.tar.gz "http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-${XCVER}.tar.gz"
	    fi
	    cd ${builddir}
	    tar zxf ${srcdir}/libxc-${XCVER}.tar.gz
	fi
	
	cd ${builddir}/libxc-${XCVER}
	./configure --enable-static --disable-shared --disable-fortran --prefix=${topdir}/libxc --exec-prefix=${topdir}/libxc &>configure.log
	make -j ${nprocs} &> make.log
	make install &> install.log
	make clean &> clean.log
	echo " done"
    fi

    if [ ! -f ${topdir}/libxc/lib/libxc.a ]; then
	echo "Error compiling libxc."
	exit
    fi
fi

# libint
if(( ! ${system_libint} )); then
    if [[ ! -f ${topdir}/libint/lib/libint.a || ! -f ${topdir}/libint/lib/libderiv.a ]]; then
	echo -n "Compiling libint ..."
	
	if [ ! -d ${builddir}/libint-${INTVER} ]; then
	    if [ ! -f ${srcdir}/libint-${INTVER}.tar.gz ]; then
		cd ${srcdir}
		wget -O libint-${INTVER}.tar.gz "https://github.com/evaleev/libint/archive/${INTVER}/libint-${INTVER}.tar.gz"
	    fi
	    cd ${builddir}
	    tar zxf ${srcdir}/libint-${INTVER}.tar.gz
	fi
	
	cd ${builddir}/libint-${INTVER}
	# Use more conservative optimization flags, since libint is already highly optimized.
	export ICFLAGS=`echo ${CFLAGS} |sed 's|-O2|-O1|g'`
	export ICXXFLAGS=`echo ${CXXFLAGS} |sed 's|-O2|-O1|g'`
	aclocal -I lib/autoconf
	autoconf
	./configure --enable-static --disable-shared \
	    --prefix=${topdir}/libint --exec-prefix=${topdir}/libint \
	    --with-libint-max-am=${MAXAM} --with-libint-opt-am=${OPTAM} \
	    --with-libderiv-max-am1=${MAXDERIV}  --disable-r12 \
	    --with-cc="${CC}" --with-cxx="${CXX}" --with-ar=${AR} \
	    --with-cc-optflags="${ICFLAGS}" \
	    --with-cxx-optflags="${ICXXFLAGS}" &>configure.log

	if [[ "$cygwin" != "" ]]; then
	    # Grow stack size
	    sed -i 's| -lm | -Wl,--stack,8388608 -lm|' src/bin/MakeVars
	fi

	make -j ${nprocs} &> make.log
	make install &> install.log
	make clean &> clean.log
	echo " done"
    fi

    if [[ ! -f ${topdir}/libint/lib/libint.a || ! -f ${topdir}/libint/lib/libderiv.a ]]; then
	echo "Error compiling libint."
	exit
    fi
fi

# Armadillo
if [ ! -d ${topdir}/armadillo-${ARMAVER} ]; then
 if [ ! -f ${srcdir}/armadillo-${ARMAVER}.tar.gz ]; then
  cd ${srcdir}
  wget -O armadillo-${ARMAVER}.tar.gz http://sourceforge.net/projects/arma/files/armadillo-${ARMAVER}.tar.gz
 fi
 cd ${topdir}
 tar zxf ${srcdir}/armadillo-${ARMAVER}.tar.gz

 # Create unversioned symlink
 if [ ! -h armadillo ]; then
  ln -sf armadillo-${ARMAVER} armadillo
 fi
fi

echo "Done compiling libraries."

if(( ! ${system_cmake} )); then
    if [ ! -f ${topdir}/cmake/bin/cmake ]; then
	echo -n "Compiling CMake ..."
	if [ ! -d ${builddir}/cmake-${CMAKEVER} ]; then
	    if [ ! -f ${srcdir}/cmake-${CMAKEVER}.tar.gz ]; then
		cd ${srcdir}
		wget -O cmake-${CMAKEVER}.tar.gz http://www.cmake.org/files/v3.1/cmake-${CMAKEVER}.tar.gz
	    fi
	    cd ${builddir}
	    tar zxf ${srcdir}/cmake-${CMAKEVER}.tar.gz
	fi
	
	cd ${builddir}/cmake-${CMAKEVER}
	./bootstrap --prefix=${topdir}/cmake &> bootstrap.log
	make -j ${nprocs} &> make.log
	make install &> install.log
	echo " done"
    fi
    cmake="${topdir}/cmake/bin/cmake"

    if [ ! -f ${topdir}/cmake/bin/cmake ]; then
	echo "Error compiling CMake."
	exit
    fi
else
    cmake="cmake"
fi

# Check out ERKALE
echo "Checking out source"
cd ${builddir}
svn checkout http://erkale.googlecode.com/svn/trunk/ erkale
echo "Done"

# Generate version file
cd erkale
echo "#ifndef ERKALE_VERSION" > src/version.h
svnrev=$(svnversion)
echo "#define SVNREVISION \"$svnrev\"" >> src/version.h
echo "#endif" >> src/version.h
cd ..

### Create config files

# Armadillo
echo "set(ARMADILLO_FOUND 1)" > erkale/cmake/FindArmadillo.cmake
echo "set(ARMADILLO_INCLUDE_DIRS \"${topdir}/armadillo/include\")" >> erkale/cmake/FindArmadillo.cmake

# GSL
echo "set(GSL_FOUND 1)" > erkale/cmake/FindGSL.cmake
if(( ${system_gsl} )); then
    echo "set(GSL_INCLUDE_DIRS \"/usr/include\")" >> erkale/cmake/FindGSL.cmake
    echo "set(GSL_LIBRARY_DIRS \"/usr/lib64\")" >> erkale/cmake/FindGSL.cmake
    echo "set(GSL_LIBRARIES -lgsl)" >> erkale/cmake/FindGSL.cmake
else
    echo "set(GSL_INCLUDE_DIRS \"${topdir}/gsl/include\")" >> erkale/cmake/FindGSL.cmake
    echo "set(GSL_LIBRARY_DIRS \"${topdir}/gsl/lib\")" >> erkale/cmake/FindGSL.cmake
    echo "set(GSL_LIBRARIES ${topdir}/gsl/lib/libgsl.a)" >> erkale/cmake/FindGSL.cmake
fi

# libxc
echo "set(LIBXC_FOUND 1)" > erkale/cmake/Findlibxc.cmake
if(( ${system_libxc} )); then
    echo "set(LIBXC_INCLUDE_DIRS \"/usr/include\")" >> erkale/cmake/Findlibxc.cmake
    echo "set(LIBXC_LIBRARY_DIRS \"/usr/lib64\")" >> erkale/cmake/Findlibxc.cmake
    echo "set(LIBXC_LIBRARIES -lxc)" >> erkale/cmake/Findlibxc.cmake
else
    echo "set(LIBXC_INCLUDE_DIRS \"${topdir}/libxc/include\")" >> erkale/cmake/Findlibxc.cmake
    echo "set(LIBXC_LIBRARY_DIRS \"${topdir}/libxc/lib\")" >> erkale/cmake/Findlibxc.cmake
    echo "set(LIBXC_LIBRARIES ${topdir}/libxc/lib/libxc.a)" >> erkale/cmake/Findlibxc.cmake
fi

# HDF5
echo "set(HDF5_FOUND 1)" > erkale/cmake/FindHDF5.cmake
if(( ${system_hdf5} )); then
    echo "set(HDF5_INCLUDE_DIRS \"/usr/include\")" >> erkale/cmake/FindHDF5.cmake
    echo "set(HDF5_LIBRARY_DIRS \"/usr/lib64\")" >> erkale/cmake/FindHDF5.cmake
    echo "set(HDF5_LIBRARIES -lhdf5 -ldl -lz)" >> erkale/cmake/FindHDF5.cmake
else
    echo "set(HDF5_INCLUDE_DIRS \"${topdir}/hdf5/include\")" >> erkale/cmake/FindHDF5.cmake
    echo "set(HDF5_LIBRARY_DIRS \"${topdir}/hdf5/lib\")" >> erkale/cmake/FindHDF5.cmake
    echo "set(HDF5_LIBRARIES ${topdir}/hdf5/lib/libhdf5.a -ldl -lz)" >> erkale/cmake/FindHDF5.cmake
fi

# Libint
echo "set(LIBINT_FOUND 1)" > erkale/config/libintConfig.cmake
if(( ${system_libint} )); then
    echo "set(LIBINT_INCLUDE_DIRS \"/usr/include\")" >> erkale/config/libintConfig.cmake
    echo "set(LIBINT_LIBRARIES -lderiv -lint)"  >> erkale/config/libintConfig.cmake
else
    echo "set(LIBINT_INCLUDE_DIRS \"${topdir}/libint/include\")" >> erkale/config/libintConfig.cmake
    #echo "set(LIBINT_LIBRARY_DIRS \"${topdir}/libint/lib\")"  >> erkale/config/libintConfig.cmake
    echo "set(LIBINT_LIBRARIES ${topdir}/libint/lib/libderiv.a ${topdir}/libint/lib/libint.a)"  >> erkale/config/libintConfig.cmake
fi

## Build erkale

cd ${builddir}/erkale
export PKG_CONFIG_PATH=${topdir}/libxc/lib/pkgconfig/:${topdir}/gsl/lib/pkgconfig/:${PKG_CONFIG_PATH}

if [ ! -d openmp ]; then
 mkdir openmp
fi
cd openmp
FC=${FC} CC=${CC} CXX=${CXX} \
 FCFLAGS=${FCFLAGS} CFLAGS=${CFLAGS} CXXFLAGS=${CXXFLAGS} \
 ${cmake} .. \
 -DSVN_VERSION=ON -DUSE_OPENMP=ON \
 -DLAPACK_LIBRARIES="${LAPACK}" \
 -DBLAS_LIBRARIES="${BLAS}" \
 -DCMAKE_INSTALL_PREFIX=${topdir}/erkale
make -j ${nprocs} VERBOSE=1
make install
cd ..

if [ ! -d serial ]; then
 mkdir serial
fi
cd serial
FC=${FC} CC=${CC} CXX=${CXX} \
 FCFLAGS=${FCFLAGS} CFLAGS=${CFLAGS} CXXFLAGS=${CXXFLAGS} \
 ${cmake} .. \
 -DSVN_VERSION=ON -DUSE_OPENMP=OFF \
 -DLAPACK_LIBRARIES="${LAPACK}" \
 -DBLAS_LIBRARIES="${BLAS}" \
 -DCMAKE_INSTALL_PREFIX=${topdir}/erkale
make -j ${nprocs} VERBOSE=1
make install
cd ..
