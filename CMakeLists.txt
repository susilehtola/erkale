# ERKALE - DFT from Hel.
#
# Modern CMake build. Dependencies are found on the system first and, for
# the CMake-based ones, fetched and built from source when missing
# (FetchContent + FIND_PACKAGE_ARGS). GSL and LAPACK/BLAS are system-only.
cmake_minimum_required(VERSION 3.24)
# C for the ADF reader sources and the BLAS integer-width probe; Fortran
# for LAPACK/BLAS linkage.
project(ERKALE VERSION 0.1.0 LANGUAGES C CXX Fortran)

# Version of the shared libraries
set(SOVERSION 0.1.0)

# We use C++17 standard library facilities: std::tgamma, <random>,
# std::sph_bessel, std::sph_legendre, std::legendre.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

include(FetchContent)
include(CheckCXXCompilerFlag)
include(CheckSourceCompiles)
include(CheckSourceRuns)

# ---------------------------------------------------------------- Options
# System-wide default basis-set repository (need not exist). Consumed as a
# compile definition on liberkale (see src/CMakeLists.txt).
set(ERKALE_SYSTEM_LIBRARY "" CACHE STRING
    "System-wide default basis-set repository (need not exist)")

# Catch C++ exceptions in the executables' main()?
option(CATCH_EXCEPTIONS "Catch C++ exceptions?" OFF)

# Build the OpenMP-parallel version (executables get an _omp suffix)?
option(USE_OPENMP "Compile the OpenMP-parallel version?" ON)

# 64-bit (ILP64) BLAS/LAPACK integers. AUTO probes the found BLAS; ON/OFF
# force it. Sets Armadillo's ARMA_BLAS_LONG_LONG when 64-bit.
set(ERKALE_BLAS_INT64 "AUTO" CACHE STRING
    "Use 64-bit (ILP64) BLAS/LAPACK integers: AUTO, ON or OFF")
set_property(CACHE ERKALE_BLAS_INT64 PROPERTY STRINGS AUTO ON OFF)

# ------------------------------------------------------ Compiler warnings
# Suppress GCC 13+ -Wfree-nonheap-object false-positives that fire from
# inside Armadillo's small-vector destructor path (the inlined call chain
# crosses header boundaries, so a localised pragma cannot reach it).
check_cxx_compiler_flag(-Wno-free-nonheap-object HAS_NO_FREE_NONHEAP_OBJECT)
if(HAS_NO_FREE_NONHEAP_OBJECT)
  add_compile_options(-Wno-free-nonheap-object)
endif()

# ---------------------------------------------------------------- OpenMP
if(USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    set(SUFFIX "_omp")
  else()
    set(SUFFIX "")
  endif()
else()
  set(SUFFIX "")
endif()

# Preprocessor used for the scf-*.cpp.in / xrsscf-*.cpp.in code
# generation (the C++ compiler invoked as -E). The OpenMP flags are
# included so that any _OPENMP conditionals expand consistently with the
# compiled objects.
set(PPROC_FLAGS ${OpenMP_CXX_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM|Intel|PGI|NVHPC|PathScale")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM|Intel")
    set(PPROC ${CMAKE_CXX_COMPILER} -x c++ -E ${PPROC_FLAGS})
  else()
    set(PPROC ${CMAKE_CXX_COMPILER} -E ${PPROC_FLAGS})
  endif()
else()
  message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} not supported for the code generator.")
endif()
find_program(CAT cat REQUIRED)

# ============================================================ Dependencies

# BLAS integer width: request ILP64 up front when explicitly asked, so
# find_package(BLAS) validates the 64-bit interface.
if(ERKALE_BLAS_INT64 STREQUAL "ON")
  set(BLA_SIZEOF_INTEGER 8)
endif()

# --- System-only (autotools/Make upstreams; ubiquitous) ---
find_package(LAPACK REQUIRED)   # provides LAPACK::LAPACK and BLAS::BLAS
find_package(GSL REQUIRED)       # provides GSL::gsl

# --- libxc (CMake). System LibxcConfig, else fetch. Normalise to Libxc::xc.
FetchContent_Declare(Libxc
  GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
  GIT_TAG        7.0.0
  FIND_PACKAGE_ARGS NAMES Libxc libxc CONFIG)
FetchContent_MakeAvailable(Libxc)
if(NOT TARGET Libxc::xc AND TARGET xc)
  add_library(Libxc::xc ALIAS xc)
endif()

# --- libcint (also satisfied by API-compatible qcint). No upstream CMake
# config package, so find header+library and wrap them in an imported
# target named "cint"; otherwise fetch and build (which exports "cint").
find_path(LIBCINT_INCLUDE_DIR cint.h)
find_library(LIBCINT_LIBRARY NAMES cint)
if(LIBCINT_INCLUDE_DIR AND LIBCINT_LIBRARY)
  add_library(cint UNKNOWN IMPORTED)
  set_target_properties(cint PROPERTIES
    IMPORTED_LOCATION "${LIBCINT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBCINT_INCLUDE_DIR}")
  message(STATUS "Found libcint: ${LIBCINT_LIBRARY}")
else()
  message(STATUS "libcint not found; fetching and building from source.")
  FetchContent_Declare(cint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG        v6.1.3)
  FetchContent_MakeAvailable(cint)
endif()

# --- HDF5 (C API only). System first; fetch fallback is heavy. Normalise
# to the erkale_hdf5 interface target regardless of how it was provided.
find_package(HDF5 QUIET COMPONENTS C)
if(NOT HDF5_FOUND)
  message(STATUS "HDF5 not found; fetching and building from source.")
  set(HDF5_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_HL_LIB OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(HDF5
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG        hdf5_1.14.4.3)
  FetchContent_MakeAvailable(HDF5)
endif()
add_library(erkale_hdf5 INTERFACE)
if(TARGET hdf5::hdf5)
  target_link_libraries(erkale_hdf5 INTERFACE hdf5::hdf5)
elseif(TARGET HDF5::HDF5)
  target_link_libraries(erkale_hdf5 INTERFACE HDF5::HDF5)
elseif(TARGET hdf5-static)
  target_link_libraries(erkale_hdf5 INTERFACE hdf5-static)
else()
  target_link_libraries(erkale_hdf5 INTERFACE ${HDF5_C_LIBRARIES} ${HDF5_LIBRARIES})
  target_include_directories(erkale_hdf5 INTERFACE ${HDF5_C_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
endif()

# --- Wigner 3j symbols (libwignernj). System config, else fetch. Same
# target name (wignernj::wignernj) either way.
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_CXX_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_FORTRAN OFF CACHE BOOL "" FORCE)
FetchContent_Declare(wignernj
  GIT_REPOSITORY https://github.com/susilehtola/libwignernj.git
  GIT_TAG        v0.7.0
  FIND_PACKAGE_ARGS 0.7.0 CONFIG)
FetchContent_MakeAvailable(wignernj)
# FetchContent's add_subdirectory exposes only the bare "wignernj" target;
# the namespaced alias exists only on the install/export (find) path.
if(NOT TARGET wignernj::wignernj AND TARGET wignernj)
  add_library(wignernj::wignernj ALIAS wignernj)
endif()

# --- nlohmann/json for Basis Set Exchange JSON parsing. System config,
# else fetch. Same target name (nlohmann_json::nlohmann_json) either way.
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  FIND_PACKAGE_ARGS 3.11 CONFIG)
FetchContent_MakeAvailable(nlohmann_json)

# ------------------------------------------- Armadillo + BLAS/LAPACK config
# ERKALE uses Armadillo header-only (ARMA_DONT_USE_WRAPPER): it links
# BLAS/LAPACK directly rather than the libarmadillo wrapper. So we only
# need Armadillo's headers. Find the config package (or fetch its source).
find_package(Armadillo QUIET)
if(Armadillo_FOUND OR ARMADILLO_INCLUDE_DIRS)
  set(_arma_includes "${ARMADILLO_INCLUDE_DIRS}")
else()
  message(STATUS "Armadillo not found; fetching headers from source.")
  # SOURCE_SUBDIR points at a non-existent directory so MakeAvailable
  # populates the sources but does NOT add_subdirectory (we want the
  # headers only, not Armadillo's wrapper library).
  FetchContent_Declare(armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG        14.0.x
    SOURCE_SUBDIR  do-not-build)
  FetchContent_MakeAvailable(armadillo)
  set(_arma_includes "${armadillo_SOURCE_DIR}/include")
endif()

# The interface target carrying Armadillo's headers and configuration
# macros (moved out of src/global.h) plus the BLAS/LAPACK link.
add_library(erkale_arma INTERFACE)
target_include_directories(erkale_arma INTERFACE ${_arma_includes})
target_link_libraries(erkale_arma INTERFACE LAPACK::LAPACK BLAS::BLAS)
target_compile_definitions(erkale_arma INTERFACE
  ARMA_DONT_USE_WRAPPER ARMA_USE_BLAS ARMA_USE_LAPACK
  # Disable Armadillo bounds checking except in Debug builds.
  $<$<NOT:$<CONFIG:Debug>>:ARMA_NO_DEBUG>)

# 64-bit BLAS integers -> Armadillo's ARMA_BLAS_LONG_LONG.
if(ERKALE_BLAS_INT64 STREQUAL "ON")
  target_compile_definitions(erkale_arma INTERFACE ARMA_BLAS_LONG_LONG)
elseif(ERKALE_BLAS_INT64 STREQUAL "AUTO")
  # Probe the found BLAS. First confirm we can link a ddot call at all;
  # then run it with the vector length passed through a 32-bit int. A
  # correct result (1+4+9 = 14) means a 32-bit (LP64) interface; a wrong
  # result from a program that *does* link means the library read a
  # 64-bit (ILP64) integer. A link failure is inconclusive -> assume
  # LP64 (the historical default) and let the user override.
  set(CMAKE_REQUIRED_LIBRARIES BLAS::BLAS)
  check_source_compiles(C "
    extern double ddot_(const int*, const double*, const int*, const double*, const int*);
    int main(void){ const int n=1, i=1; double x=1.0; return ddot_(&n,&x,&i,&x,&i) > 0.0 ? 0 : 1; }
    " ERKALE_BLAS_LINKS)
  if(ERKALE_BLAS_LINKS)
    check_source_runs(C "
      extern double ddot_(const int*, const double*, const int*, const double*, const int*);
      int main(void){
        const int n=3, one=1; double x[3]={1.0,2.0,3.0};
        double r = ddot_(&n,x,&one,x,&one);
        return (r>13.5 && r<14.5) ? 0 : 1;   /* 0 == LP64 */
      }" ERKALE_BLAS_LP64)
    if(NOT ERKALE_BLAS_LP64)
      message(STATUS "BLAS uses 64-bit (ILP64) integers; defining ARMA_BLAS_LONG_LONG.")
      target_compile_definitions(erkale_arma INTERFACE ARMA_BLAS_LONG_LONG)
    else()
      message(STATUS "BLAS uses 32-bit (LP64) integers.")
    endif()
  else()
    message(WARNING "Could not probe the BLAS integer width; assuming 32-bit (LP64). "
                    "Pass -DERKALE_BLAS_INT64=ON if your BLAS/LAPACK is ILP64.")
  endif()
  unset(CMAKE_REQUIRED_LIBRARIES)
endif()

# =========================================================================

# Recurse into the source tree.
add_subdirectory(src)

# Testing
enable_testing()
add_subdirectory(tests)
