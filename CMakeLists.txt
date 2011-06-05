# The name of our project is "ERKALE".  CMakeLists files in this project can
# refer to the root source directory of the project as ${ERKALE_SOURCE_DIR} and
# to the root binary directory of the project as ${ERKALE_BINARY_DIR}.
project (ERKALE)

# Require CMake version 2.8 (might work with older ones, too)
cmake_minimum_required(VERSION 2.8)

# Compile SMP version?
option(USE_OPENMP "Compile OpenMP enabled version (for parallel calculations)?" ON)
# Find OpenMP support
if(USE_OPENMP)
 find_package( OpenMP )
else()
 # No flags
 set(OpenMP_C_FLAGS "" )
 set(OpenMP_CXX_FLAGS "" )
endif()

# Find cat
find_program(CAT cat)

# Find out how to preprocess C++ source code files
if(CMAKE_COMPILER_IS_GNUCXX)
 set( PPROC ${CMAKE_CXX_COMPILER} -x c++ -E )
else( CMAKE_CXX_COMPILER_ID strequal "PGI" )
 set( PPROC ${CMAKE_CXX_COMPILER} -E )
else( CMAKE_CXX_COMPILER_ID strequal "PathScale" )
 set( PPROC ${CMAKE_CXX_COMPILER} -E )
else( CMAKE_CXX_COMPILER_ID strequal "Intel" )
 set( PPROC ${CMAKE_CXX_COMPILER} -x c++ -E )
else()
 set( PPROC "" )
 message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} not supported." )
endif()


# Recurse into the "src" subdirectory.
add_subdirectory (src)
