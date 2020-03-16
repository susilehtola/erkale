#!/bin/bash
# This is a script for downloading, compiling and
# installing ERKALE with all of its prerequisite libraries and CMake.
# 2020-03-16 Susi Lehtola

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
export CFLAGS="-Wall -g -O2 -fPIC"

# C++ flags to use
export CXXFLAGS="${CFLAGS}"
# Fortran flags to use
export FFLAGS="${CFLAGS}"
export FCFLAGS="${CFLAGS}"

### If not using system LAPACK/BLAS, libraries to link OpenMP OpenBLAS with
OMPLIBS="-fopenmp"
### If not using system LAPACK/BLAS, the OpenMP version of OpenBLAS
### needs to be compiled with -frecursive to make it thread safe.
FRECURSIVE="-frecursive"

### System LAPACK (+ BLAS) library to use.

## OpenBLAS
LAPACKOMP="-lopenblaso"
LAPACKSER="-lopenblas"

# MKL (with Intel compiler)
#LAPACKOMP="-mkl=parallel"
#LAPACKSER="-mkl=sequential"

## ATLAS, newer versions of Fedora / RHEL
#LAPACKOMP="-L/usr/lib64/atlas -lsatlas"
#LAPACKSER="-L/usr/lib64/atlas -lsatlas"

## ATLAS, older versions of Fedora / RHEL
#LAPACKOMP="-L/usr/lib64/atlas -llapack -lf77blas -lcblas -latlas"
#LAPACKSER="-L/usr/lib64/atlas -llapack -lf77blas -lcblas -latlas"

## Generic lapack and blas. Don't use these unless there is nothing
## else available (e.g. on Cygwin)
# LAPACKOMP="-llapack -lblas -lgfortran"
# LAPACKSER="-llapack -lblas -lgfortran"

# Use system packages?
system_cmake=0
system_gsl=0
system_libxc=0
system_libint=0
system_hdf5=0
system_blas=0

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
# GSL
export GSLVER="2.6"
## LibXC
#export XCVER="4.0.3"
# Use newest git snapshot
export XCVER="git"
# libint 1.1.6
export INTVER="0e0ffa7887e74e6ab1fb07c89be55f776c733731"
#export ARMAVER="9.200.6"
export ARMAVER="git"
export CMAKEVER="3.16.5"

# HDF5 version: MAJOR.MINOR
export HDF5MAJOR="1.12"
export HDF5MINOR="0"

# Version of OpenBLAS
export OPENBLASVER="0.3.9"
# You may need to disable AVX flags for OpenBLAS with older compilers
#export OPENBLASAVX="NO_AVX=1 NO_AVX2=1" # For very old
#export OPENBLASAVX="NO_AVX2=1" # For a little less old

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

	HDF5VER="${HDF5MAJOR}.${HDF5MINOR}"
        if [ ! -d ${builddir}/hdf5-${HDF5VER} ]; then
            if [ ! -f ${srcdir}/hdf5-${HDF5VER}.tar.gz ]; then
                cd ${srcdir}
		wget -O hdf5-${HDF5VER}.tar.gz http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5MAJOR}/hdf5-${HDF5VER}/src/hdf5-${HDF5VER}.tar.gz
	    fi
	    cd ${builddir}
	    tar zxf ${srcdir}/hdf5-${HDF5VER}.tar.gz
	fi

	cd ${builddir}/hdf5-${HDF5VER}/
	./configure --enable-static --disable-shared --prefix=${topdir}/hdf5 --exec-prefix=${topdir}/hdf5 --disable-hl --disable-fortran &>configure.log
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
    if [[ "$XCVER" == "git" ]]; then
	echo -n "Checking out and compiling libxc ..."
	cd $builddir
        if [[ ! -d libxc ]]; then
            git clone https://gitlab.com/libxc/libxc.git libxc
        else
            cd libxc
            git pull
            cd ..
        fi
	cd libxc
	if [[ ! -f configure ]]; then
	    autoreconf -i
	fi
	./configure --enable-static --disable-shared --disable-fortran --prefix=${topdir}/libxc --exec-prefix=${topdir}/libxc &>configure.log
	make -j ${nprocs} &> make.log
	make install &> install.log
	make clean &> clean.log
	echo " done"
    else
	if [ ! -f ${topdir}/libxc/lib/libxc.a ]; then
	    echo -n "Compiling libxc ..."
	    if [ ! -d ${builddir}/libxc-${XCVER} ]; then
		if [ ! -f ${srcdir}/libxc-${XCVER}.tar.gz ]; then
		    cd ${srcdir}
		    wget -O libxc-${XCVER}.tar.gz "http://www.tddft.org/programs/octopus/down.php?file=libxc/${XCVER}/libxc-${XCVER}.tar.gz"

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

# OpenBLAS
if(( ! ${system_blas} )); then
    if [ ! -d ${builddir}/OpenBLAS-${OPENBLASVER} ]; then
	if [ ! -f ${srcdir}/OpenBLAS-${OPENBLASVER}.tar.gz ]; then
	    cd ${srcdir}
	    wget -O OpenBLAS-${OPENBLASVER}.tar.gz https://github.com/xianyi/OpenBLAS/archive/v${OPENBLASVER}.tar.gz
	fi
	cd ${builddir}
	tar zxf ${srcdir}/OpenBLAS-${OPENBLASVER}.tar.gz
    fi
    cd ${builddir}

	# Sequential binaries
    if [[ ! -d OpenBLAS-${OPENBLASVER}-seq ]]; then
	cp -pr OpenBLAS-${OPENBLASVER} OpenBLAS-${OPENBLASVER}-seq
    fi
    if [[ ! -d OpenBLAS-${OPENBLASVER}-omp ]]; then
	cp -pr OpenBLAS-${OPENBLASVER} OpenBLAS-${OPENBLASVER}-omp
    fi

    if [ ! -f ${topdir}/openblas/lib/libopenblas.a ]; then
	echo -n "Compiling openblas, sequential ..."
        # Compile library
	make -C OpenBLAS-${OPENBLASVER}-seq TARGET=CORE2 DYNAMIC_ARCH=1 USE_THREAD=0 USE_OPENMP=0 FC=$FC CC=$CC COMMON_OPT="$CFLAGS" FCOMMON_OPT="$FFLAGS" NUM_THREADS=128 LIBPREFIX="libopenblas" INTERFACE64=0 ${OPENBLASAVX} NO_SHARED=1 &> OpenBLAS-${OPENBLASVER}-seq/make.log
	make -C OpenBLAS-${OPENBLASVER}-seq install NO_SHARED=1 PREFIX=${topdir}/openblas OPENBLAS_LIBRARY_DIR=${topdir}/openblas/lib OPENBLAS_INCLUDE_DIR=${topdir}/openblas/include OPENBLAS_BINARY_DIR=${topdir}/openblas/bin OPENBLAS_CMAKE_DIR=${topdir}/openblas/cmake
	echo " done"
    fi
    if [ ! -f ${topdir}/openblas/lib/libopenblas.a ]; then
	echo "Error building sequential OpenBLAS."
	exit
    fi

    if [ ! -f ${topdir}/openblas/lib/libopenblaso.a ]; then
	echo -n "Compiling openblas, parallel ..."
	make -C OpenBLAS-${OPENBLASVER}-omp NO_SHARED=1 TARGET=CORE2 DYNAMIC_ARCH=1 USE_THREAD=1 USE_OPENMP=1 FC=$FC CC=$CC COMMON_OPT="$CFLAGS" FCOMMON_OPT="$FFLAGS $FRECURSIVE" NUM_THREADS=128 LIBPREFIX="libopenblaso" INTERFACE64=0 ${OPENBLASAVX} EXTRALIB="${OMPLIBS}" &> OpenBLAS-${OPENBLASVER}-omp/make.log
	make -C OpenBLAS-${OPENBLASVER}-omp install NO_SHARED=1 LIBPREFIX="libopenblaso" PREFIX=${topdir}/openblas OPENBLAS_LIBRARY_DIR=${topdir}/openblas/lib OPENBLAS_INCLUDE_DIR=${topdir}/openblas/include OPENBLAS_BINARY_DIR=${topdir}/openblas/bin OPENBLAS_CMAKE_DIR=${topdir}/openblas/cmake
	echo " done"
    fi
    if [ ! -f ${topdir}/openblas/lib/libopenblaso.a ]; then
	echo "Error building parallel OpenBLAS."
	exit
    fi
fi



# Armadillo
if [ ! -d ${topdir}/armadillo-${ARMAVER} ]; then
    if [[ "$ARMAVER" == "git" ]]; then
	echo -n "Checking out Armadillo ..."
	cd $topdir
        if [[ ! -d armadillo-code ]]; then
            git clone https://gitlab.com/conradsnicta/armadillo-code.git
        else
            cd armadillo-code
            git pull
            cd ..
        fi

	# Create unversioned symlink
	if [ ! -h armadillo ]; then
	    ln -sf armadillo-code armadillo
	fi
    else
	if [ ! -f ${srcdir}/armadillo-${ARMAVER}.tar.xz ]; then
	    cd ${srcdir}
	    wget -O armadillo-${ARMAVER}.tar.xz http://sourceforge.net/projects/arma/files/armadillo-${ARMAVER}.tar.xz
	fi
	cd ${topdir}
	tar Jxf ${srcdir}/armadillo-${ARMAVER}.tar.xz

	# Create unversioned symlink
	if [ ! -h armadillo ]; then
	    ln -sf armadillo-${ARMAVER} armadillo
	fi
    fi
fi
echo "Done compiling libraries."

if(( ! ${system_cmake} )); then
    if [ ! -f ${topdir}/cmake/bin/cmake ]; then
	echo -n "Compiling CMake ..."
	if [ ! -d ${builddir}/cmake-${CMAKEVER} ]; then
	    if [ ! -f ${srcdir}/cmake-${CMAKEVER}.tar.gz ]; then
		cd ${srcdir}
                majorrel=$(echo $CMAKEVER | awk --field-separator=. '{printf("%s.%s",$1,$2)}')
		wget -O cmake-${CMAKEVER}.tar.gz http://www.cmake.org/files/v${majorrel}/cmake-${CMAKEVER}.tar.gz
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
# Check for old svn checkout
if [[ -d erkale/.svn ]]; then
    echo "A subversion checkout detected. Archiving old data."
    if [[ -d erkale.svn ]]; then
	echo "Archival directory already exists. Please remove ${builddir}/erkale or ${builddir}/erkale.svn."
	exit
    fi
    mv erkale erkale.svn
fi
# Check for old git checkout
if [[ -d erkale ]]; then
    cd erkale
    git pull
    cd ..
else
    git clone https://github.com/susilehtola/erkale.git erkale
fi
echo "Done"

# Generate version file
cd erkale
svnrev=$(git rev-list --count --first-parent HEAD)
gitversion=$(git log --pretty=format:'%H' -n 1)
gitshort=$(echo $gitversion|awk '{print substr($1,1,8)}')
echo "#ifndef ERKALE_VERSION" > src/version.h
echo "#define SVNREVISION \"$svnrev\"" >> src/version.h
echo "#define GITVERSION \"$gitversion\"" >> src/version.h
echo "#define GITSHORT \"$gitshort\"" >> src/version.h
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

# BLAS
if(( ! ${system_blas} )); then
    LAPACKSER="-L${topdir}/openblas/lib/ -lopenblas -lgfortran"
    LAPACKOMP="-L${topdir}/openblas/lib/ -lopenblaso -lgfortran"
fi

## Build erkale

cd ${builddir}/erkale
export PKG_CONFIG_PATH=${topdir}/libxc/lib/pkgconfig/:${topdir}/gsl/lib/pkgconfig/:${PKG_CONFIG_PATH}

if [ ! -d openmp ]; then
 mkdir openmp
fi
cd openmp
FC="${FC}" CC="${CC}" CXX="${CXX}" \
 FCFLAGS="${FCFLAGS}" CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" \
 ${cmake} .. \
 -DSVN_VERSION=ON -DUSE_OPENMP=ON \
 -DLAPACK_LIBRARIES="${LAPACKOMP}" \
 -DBLAS_LIBRARIES="${LAPACKOMP}" \
 -DCMAKE_INSTALL_PREFIX=${topdir}/erkale
make -j ${nprocs} VERBOSE=1
make install
cd ..

if [ ! -d serial ]; then
 mkdir serial
fi
cd serial
FC="${FC}" CC="${CC}" CXX="${CXX}" \
 FCFLAGS="${FCFLAGS}" CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" \
 ${cmake} .. \
 -DSVN_VERSION=ON -DUSE_OPENMP=OFF \
 -DLAPACK_LIBRARIES="${LAPACKSER}" \
 -DBLAS_LIBRARIES="${LAPACKSER}" \
 -DCMAKE_INSTALL_PREFIX=${topdir}/erkale
make -j ${nprocs} VERBOSE=1
make install
cd ..
