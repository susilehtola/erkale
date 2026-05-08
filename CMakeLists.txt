# The name of our project is "ERKALE".  CMakeLists files in this project can
# refer to the root source directory of the project as ${ERKALE_SOURCE_DIR} and
# to the root binary directory of the project as ${ERKALE_BINARY_DIR}.
project (ERKALE)
set(PROJECT_TYPE CXX)

# CMake 3.14 is required for FetchContent_MakeAvailable, used to
# fetch libwignernj when no installed copy is detected.
cmake_minimum_required(VERSION 3.14)
# Enable Fortran for LAPACK
enable_language (Fortran)

# We use C++17 standard library facilities: std::tgamma, <random>,
# std::sph_bessel, std::sph_legendre, std::legendre.
if(NOT CMAKE_CXX_STANDARD)
 set(CMAKE_CXX_STANDARD 17)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
 set(CMAKE_CXX_EXTENSIONS OFF)
endif()

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

# Catch C++ exceptions?
option(CATCH_EXCEPTIONS "Catch C++ exceptions?" OFF)
if(CATCH_EXCEPTIONS)
  add_definitions(-DCATCH_EXCEPTIONS)
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
  # Set OMP flags for preprocessor
  set(PPROC_FLAGS "${OpenMP_CXX_FLAGS}")
 else()
   set(SUFFIX "")
   set(PPROC_FLAGS "")
 endif()
else()
 # No flags
 set(OpenMP_C_FLAGS "" )
 set(OpenMP_CXX_FLAGS "" )
 set(PPROC_FLAGS "")
 # No suffix
 set(SUFFIX "")
endif()

# Add in svn version information?
option(SVN_VERSION "Add in svn version information? (Need to generate version.h)" OFF)
if(SVN_VERSION)
  # Add flag to CXX flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSVNRELEASE")
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

# Find libwignernj for Wigner 3j symbols. If a system install is not
# present, fetch and build from source. Either path exposes the
# wignernj::wignernj imported target.
find_package(wignernj 0.5.0 QUIET CONFIG)
if(wignernj_FOUND)
 message("Found libwignernj ${wignernj_VERSION}: ${wignernj_DIR}")
else()
 message("libwignernj not found; fetching v0.5.0 from GitHub.")
 include(FetchContent)
 # Disable libwignernj's own tests / examples / extra interfaces so the
 # subbuild only produces the C library we link against.
 set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
 set(BUILD_CXX_TESTS OFF CACHE BOOL "" FORCE)
 set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
 set(BUILD_FORTRAN OFF CACHE BOOL "" FORCE)
 FetchContent_Declare(
   wignernj
   GIT_REPOSITORY https://github.com/susilehtola/libwignernj.git
   GIT_TAG        v0.5.0
 )
 FetchContent_MakeAvailable(wignernj)
 # FetchContent's add_subdirectory only exposes the bare "wignernj"
 # target; the "wignernj::wignernj" namespaced name only exists in the
 # install/export path. Provide an alias so consumers can use the same
 # name regardless of how libwignernj was acquired.
 if(NOT TARGET wignernj::wignernj)
   add_library(wignernj::wignernj ALIAS wignernj)
 endif()
endif()
message("")


# Find out how to preprocess C++ source code files
if(CMAKE_COMPILER_IS_GNUCXX)
 set( PPROC ${CMAKE_CXX_COMPILER} -x c++ -E ${DFTFLAGS} ${PPROC_FLAGS})
elseif( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" )
 set( PPROC ${CMAKE_CXX_COMPILER} -x c++ -E ${DFTFLAGS} ${PPROC_FLAGS})
elseif( ${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI" )
 set( PPROC ${CMAKE_CXX_COMPILER} -E ${DFTFLAGS} ${PPROC_FLAGS})
elseif( ${CMAKE_CXX_COMPILER_ID} STREQUAL "PathScale" )
 set( PPROC ${CMAKE_CXX_COMPILER} -E ${DFTFLAGS} ${PPROC_FLAGS})
elseif( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
 set( PPROC ${CMAKE_CXX_COMPILER} -x c++ -E ${DFTFLAGS} ${PPROC_FLAGS})
else()
 set( PPROC "" )
 message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} not supported." )
endif()

# Recurse into the "src" subdirectory.
add_subdirectory (src)

# Testing
if((${CMAKE_MAJOR_VERSION} GREATER 2) OR (${CMAKE_MAJOR_VERSION} EQUAL 2 AND (${CMAKE_MINOR_VERSION} EQUAL 8 OR ${CMAKE_MINOR_VERSION} GREATER 8)))
  enable_testing()
  add_subdirectory (tests)
  message("Can run tests.")
elseif()
  message("Version of CMake too old to run tests; at least version 2.8 required.")
endif()
