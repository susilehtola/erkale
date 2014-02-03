# The name of our project is "ERKALE".  CMakeLists files in this project can
# refer to the root source directory of the project as ${ERKALE_SOURCE_DIR} and
# to the root binary directory of the project as ${ERKALE_BINARY_DIR}.
project (ERKALE)

# Require CMake version 2.8 (might work with older ones, too)
cmake_minimum_required(VERSION 2.6)
# Enable Fortran for LAPACK
enable_language (Fortran)

# Set version
set(VERSION 0.1.0)
# Version of libraries
set(SOVERSION 0.1.0)

# Cmake modules/macros are in a subdirectory to keep the root directory cleaner
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
# Configuration where to find libint is in current directory
set(libint_DIR ${CMAKE_CURRENT_SOURCE_DIR}/config)
set(Armadillo_DIR ${CMAKE_CURRENT_SOURCE_DIR}/config)

# System default basis set repository (need not exist)
option(ERKALE_SYSTEM_LIBRARY "System default basis set repository (need not exist)" OFF)
if(ERKALE_SYSTEM_LIBRARY)
 message("Using ${ERKALE_SYSTEM_LIBRARY} as default basis set repository.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DERKALE_SYSTEM_LIBRARY=\\\"${ERKALE_SYSTEM_LIBRARY}\\\"" )
else()
 message("Using default location for basis set repository.")
endif()

# Compile SMP version?
option(USE_OPENMP "Compile OpenMP enabled version (for parallel calculations)?" ON)
# Find OpenMP support
if(USE_OPENMP)
 find_package( OpenMP )
 if(OPENMP_FOUND)
  # Add flags to CXX flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  # Set suffix
  set(SUFFIX "_omp")
 else()
  set(SUFFIX "")
 endif()
else()
 # No flags
 set(OpenMP_C_FLAGS "" )
 set(OpenMP_CXX_FLAGS "" )
 # No suffix
 set(SUFFIX "")
endif()

# Find cat
find_program(CAT cat)

# Find LAPACK
find_package(LAPACK REQUIRED)
message("LAPACK headers are in ${LAPACK_INCLUDE_DIRS}")
message("LAPACK library is ${LAPACK_LIBRARIES}")
message("")

# Find Armadillo
find_package(Armadillo REQUIRED)
message("ARMADILLO headers are in ${ARMADILLO_INCLUDE_DIRS}")
message("ARMADILLO library is ${ARMADILLO_LIBRARIES}")
message("")

# Find GSL
find_package(GSL REQUIRED)
message("GSL headers are in ${GSL_INCLUDE_DIRS}")
message("GSL library is ${GSL_LIBRARIES}")
message("")

# Find libxc
find_package(libxc REQUIRED)
message("libxc headers are in ${LIBXC_INCLUDE_DIRS}")
message("libxc linker flags are ${LIBXC_LIBRARIES}")
message("")

# Find libint
find_package(libint REQUIRED)
message("libint headers are in ${LIBINT_INCLUDE_DIRS}")
message("libint linker flags are ${LIBINT_LIBRARIES}")
message("")

# Find HDF5
find_package(HDF5 REQUIRED)
message("HDF5 headers are in ${HDF5_INCLUDE_DIRS}")
message("HDF5 linker flags are ${HDF5_LIBRARIES}")
message("")


# Find out how to preprocess C++ source code files
if(CMAKE_COMPILER_IS_GNUCXX)
 set( PPROC ${CMAKE_CXX_COMPILER} -x c++ -E ${DFTFLAGS} ${OpenMP_CXX_FLAGS})
elseif( CMAKE_CXX_COMPILER_ID strequal "PGI" )
 set( PPROC ${CMAKE_CXX_COMPILER} -E ${DFTFLAGS} ${OpenMP_CXX_FLAGS})
elseif( CMAKE_CXX_COMPILER_ID strequal "PathScale" )
 set( PPROC ${CMAKE_CXX_COMPILER} -E ${DFTFLAGS} ${OpenMP_CXX_FLAGS})
elseif( CMAKE_CXX_COMPILER_ID strequal "Intel" )
 set( PPROC ${CMAKE_CXX_COMPILER} -x c++ -E ${DFTFLAGS} ${OpenMP_CXX_FLAGS})
else()
 set( PPROC "" )
 message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} not supported." )
endif()

# Recurse into the "src" subdirectory.
add_subdirectory (src)
