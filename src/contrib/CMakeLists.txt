# Research / experimental executables. Not installed. Each links the
# completeness, EMD and base libraries; the external dependencies arrive
# transitively through liberkale.
foreach(prog dumpxc moints formchk guessbench densproj rimp2 oovl sapguess)
  add_executable(erkale_${prog} ${prog}.cpp)
  set_target_properties(erkale_${prog} PROPERTIES OUTPUT_NAME "erkale_${prog}${SUFFIX}")
  target_link_libraries(erkale_${prog} liberkale_cmp liberkale_emd liberkale)
endforeach()

# --- NEO / complex-orbital drivers ---
# These depend on the header-only, Eigen-based OpenOrbitalOptimizer
# (https://github.com/susilehtola/OpenOrbitalOptimizer). The drivers use
# its Armadillo compatibility shim, which pulls in Eigen. Find the headers
# (with a configurable hint) or fetch them; likewise for Eigen3.
set(OPENORBITAL_INCLUDE_DIR "" CACHE PATH
    "Path to the openorbitaloptimizer include dir (containing openorbitaloptimizer/scfsolver.hpp)")
find_path(OPENORBITAL_HEADER openorbitaloptimizer/scfsolver.hpp
  HINTS ${OPENORBITAL_INCLUDE_DIR})

# OpenOrbitalOptimizer and Eigen are used header-only; we deliberately
# populate their sources without running their CMakeLists (FetchContent_-
# Populate). Opt into the classic Populate semantics so newer CMake does
# not warn about the deprecated single-argument form.
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
if(NOT OPENORBITAL_HEADER)
  message(STATUS "openorbitaloptimizer not found; fetching from GitHub (header-only).")
  # Populate the sources WITHOUT running OpenOrbitalOptimizer's own
  # CMakeLists: it builds a test suite (helfem/legendre/diatomic) and
  # requires an armadillo CMake target that we do not provide. We only
  # want its headers.
  FetchContent_Declare(openorbitaloptimizer
    GIT_REPOSITORY https://github.com/susilehtola/OpenOrbitalOptimizer.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
  FetchContent_GetProperties(openorbitaloptimizer)
  if(NOT openorbitaloptimizer_POPULATED)
    FetchContent_Populate(openorbitaloptimizer)
  endif()
  set(OPENORBITAL_HEADER "${openorbitaloptimizer_SOURCE_DIR}")
endif()

find_package(Eigen3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 not found; fetching v3.4.0 (header-only).")
  FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE)
  FetchContent_GetProperties(Eigen3)
  if(NOT eigen3_POPULATED)
    FetchContent_Populate(Eigen3)
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}")
endif()

message(STATUS "Building NEO targets (openorbitaloptimizer: ${OPENORBITAL_HEADER})")

# Shared usage requirements of the NEO drivers.
add_library(erkale_neo_deps INTERFACE)
target_include_directories(erkale_neo_deps INTERFACE "${OPENORBITAL_HEADER}")
target_link_libraries(erkale_neo_deps INTERFACE
  Eigen3::Eigen liberkale_cmp liberkale_emd liberkale)

add_executable(erkale_neo neo.cpp neo_dump.cpp neo_cholesky.cpp)
set_target_properties(erkale_neo PROPERTIES OUTPUT_NAME "erkale_neo${SUFFIX}")
target_link_libraries(erkale_neo erkale_neo_deps)

add_executable(erkale_neosp neosp.cpp)
set_target_properties(erkale_neosp PROPERTIES OUTPUT_NAME "erkale_neosp${SUFFIX}")
target_link_libraries(erkale_neosp erkale_neo_deps)

add_executable(erkale_hneoci hneoci.cpp)
set_target_properties(erkale_hneoci PROPERTIES OUTPUT_NAME "erkale_hneoci${SUFFIX}")
target_link_libraries(erkale_hneoci erkale_neo_deps)

add_executable(erkale_complex_orbs complex_orbs.cpp)
set_target_properties(erkale_complex_orbs PROPERTIES OUTPUT_NAME "erkale_complex_orbs${SUFFIX}")
target_link_libraries(erkale_complex_orbs erkale_neo_deps)
