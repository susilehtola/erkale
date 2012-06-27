#!/bin/bash
# This is a script for downloading, compiling and 
# installing ERKALE with all of its prerequisite libraries and CMake.
# 2011-11-12 Jussi Lehtola

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

# Patch libxc? Set to 1 if you use gfortran, otherwise 0.
libxc_patch=1

# C preprosessor
export CPP="${CC} -E"
# Fortran preprocessor
export FCCPP="${FC} -E"

# C flags to use. 
#For new versions of GCC on modern x86 hardware
#export CFLAGS="-Wall -O2 -funroll-loops -fPIC -march=native -msse3"
#For older versions
export CFLAGS="-Wall -O2 -funroll-loops -fPIC"

# C++ flags to use
export CXXFLAGS="${CFLAGS}"
# Fortran flags to use
export FFLAGS="${CFLAGS}"
export FCFLAGS="${CFLAGS}"

# LAPACK and BLAS library to use
LAPACK="-L/usr/lib64/atlas -llapack -latlas"
BLAS="-L/usr/lib64/atlas -latlas"

# Maximum supported angular momentum (affects libint)
# If this is very high, libint compilation will take ages and the
# resulting libraries will be HUGE.
MAXAM="6"

# Current versions of libraries
export GSLVER="1.15"
export XCVER="1.2.0"
export INTVER="1.1.4"
export ARMAVER="3.2.2"
export CMAKEVER="2.8.8"
export HDF5VER="1.8.8"

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

# HDF5
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

# libXC
if [ ! -f ${topdir}/libxc/lib/libxc.a ]; then
 echo -n "Compiling libxc ..."
 
 if [ ! -d ${builddir}/libxc-${XCVER} ]; then
  if [ ! -f ${srcdir}/libxc-${XCVER}.tar.gz ]; then
   cd ${srcdir}
   wget -O libxc-${XCVER}.tar.gz "http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-${XCVER}.tar.gz"
  fi
  cd ${builddir}
  tar zxf ${srcdir}/libxc-${XCVER}.tar.gz

  # Patch to conform to standards
  if(( $libxc_patch )); then
   patch libxc-${XCVER}/src/libxc_master.F90 &> patch.log <<EOF 
21c21
< #  define XC_F90(x) xc_s_f90_ ## x
---
> #  define XC_F90(x) xc_s_f90_/**/x  
23c23
< #  define XC_F90(x) xc_f90_ ## x
---
> #  define XC_F90(x) xc_f90_/**/x  
EOF
  fi

 fi

 cd ${builddir}/libxc-${XCVER}
 ./configure --enable-static --disable-shared --prefix=${topdir}/libxc --exec-prefix=${topdir}/libxc &>configure.log
 make -j ${nprocs} &> make.log
 make install &> install.log
 make clean &> clean.log
 echo " done"
fi

# libint
if [ ! -f ${topdir}/libint/lib/libint.a ]; then
 echo -n "Compiling libint ..."

 if [ ! -d ${builddir}/libint-${INTVER} ]; then
  if [ ! -f ${srcdir}/libint-${INTVER}.tar.gz ]; then
   cd ${srcdir}
   wget -O libint-${INTVER}.tar.gz "http://sourceforge.net/projects/libint/files/v1-releases/libint-${INTVER}.tar.gz/download"
  fi
  cd ${builddir}
  tar zxf ${srcdir}/libint-${INTVER}.tar.gz
 fi
 
 cd ${builddir}/libint-${INTVER}
 # Use more conservative optimization flags, since libint is already highly optimized.
 export ICFLAGS=`echo ${CFLAGS} |sed 's|-O2|-O1|g'`
 export ICXXFLAGS=`echo ${CXXFLAGS} |sed 's|-O2|-O1|g'`
 
 ./configure --enable-static --disable-shared \
  --prefix=${topdir}/libint --exec-prefix=${topdir}/libint \
  --disable-deriv --disable-r12 --with-libint-max-am=${MAXAM} \
  --with-cc="${CC}" --with-cxx="${CXX}" --with-ar=${AR} \
  --with-cc-optflags="${ICFLAGS}" \
  --with-cxx-optflags="${ICXXFLAGS}" &>configure.log
 make -j ${nprocs} &> make.log
 make install &> install.log
 make clean &> clean.log
 echo " done"
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

 # Patch configuration
 cd armadillo-${ARMAVER}
 sed -i 's|// #define ARMA_USE_BLAS|#define ARMA_USE_BLAS|g' include/armadillo_bits/config.hpp 
 sed -i 's|// #define ARMA_USE_LAPACK|#define ARMA_USE_LAPACK|g' include/armadillo_bits/config.hpp 
 sed -i 's|// #define ARMA_NO_DEBUG|#define ARMA_NO_DEBUG|g' include/armadillo_bits/config.hpp
fi


echo "Done compiling libraries."

if [ ! -f ${topdir}/cmake/bin/cmake ]; then
 echo -n "Compiling CMake ..."
 if [ ! -d ${builddir}/cmake-${CMAKEVER} ]; then
  if [ ! -f ${srcdir}/cmake-${CMAKEVER}.tar.gz ]; then
   cd ${srcdir}
   wget -O cmake-${CMAKEVER}.tar.gz http://www.cmake.org/files/v2.8/cmake-${CMAKEVER}.tar.gz
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


# Check out ERKALE
echo "Checking out source"
cd ${builddir}
svn checkout http://erkale.googlecode.com/svn/trunk/ erkale
echo "Done"

### Create config files

# Armadillo
echo "set(ARMADILLO_FOUND 1)" > erkale/cmake/FindArmadillo.cmake
echo "set(ARMADILLO_INCLUDE_DIRS \"${topdir}/armadillo/include\")" >> erkale/cmake/FindArmadillo.cmake

# HDF5
echo "set(HDF5_FOUND 1)" > erkale/cmake/FindHDF5.cmake
echo "set(HDF5_INCLUDE_DIRS \"${topdir}/hdf5/include\")" >> erkale/cmake/FindHDF5.cmake
echo "set(HDF5_LIBRARY_DIRS \"${topdir}/hdf5/lib\")" >> erkale/cmake/FindHDF5.cmake
echo "set(HDF5_LIBRARIES ${topdir}/hdf5/lib/libhdf5.a -lz)" >> erkale/cmake/FindHDF5.cmake

# Libint
echo "set(LIBINT_FOUND 1)" > erkale/config/libintConfig.cmake
echo "set(LIBINT_INCLUDE_DIRS \"${topdir}/libint/include\")" >> erkale/config/libintConfig.cmake
#echo "set(LIBINT_LIBRARY_DIRS \"${topdir}/libint/lib\")"  >> erkale/config/libintConfig.cmake
echo "set(LIBINT_LIBRARIES ${topdir}/libint/lib/libint.a)"  >> erkale/config/libintConfig.cmake

## Build erkale

cd ${builddir}/erkale
export PKG_CONFIG_PATH=${topdir}/libxc/lib/pkgconfig/:${topdir}/gsl/lib/pkgconfig/:${PKG_CONFIG_PATH}



if [ ! -d openmp ]; then
 mkdir openmp
fi
cd openmp
FC=${FC} CC=${CC} CXX=${CXX} \
 FCFLAGS=${FCFLAGS} CFLAGS=${CFLAGS} CXXFLAGS=${CXXFLAGS} \
 ${topdir}/cmake/bin/cmake .. \
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
 ${topdir}/cmake/bin/cmake .. \
 -DLAPACK_LIBRARIES="${LAPACK}" \
 -DBLAS_LIBRARIES="${BLAS}" \
 -DCMAKE_INSTALL_PREFIX=${topdir}/erkale \
 -DUSE_OPENMP=OFF
make -j ${nprocs} VERBOSE=1
make install
cd ..
