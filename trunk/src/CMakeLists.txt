# make sure cmake addes the binary directory for the project to
# the include path, since some sources are generated on-the-fly
# We also want to find the electron momentum density stuff
# and the completeness profile
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# Add the armadillo include directory
INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})
# Add the GSL include directory
INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIRS})
# Add libxc include directory if necessary
if(LIBXC_FOUND)
 INCLUDE_DIRECTORIES(${LIBXC_INCLUDE_DIRS})
endif()
# Add libint include directory if necessary
if(LIBINT_FOUND)
 INCLUDE_DIRECTORIES(${LIBINT_INCLUDE_DIRS})
endif()

# Create the ERKALE base library "erkale" which includes the following sources
if(USEDFT)
add_library (erkale 
basis.cpp basislibrary.cpp stringutil.cpp mathf.cpp integrals.cpp
eritable.cpp eriscreen.cpp timer.cpp linalg.cpp obara-saika.cpp
solidharmonics.cpp diis.cpp scf.cpp elements.cpp xyzutils.cpp
settings.cpp broyden.cpp adiis.cpp tempered.cpp find_molecules.cpp
properties.cpp
# DFT stuff
lobatto.cpp dftgrid.cpp dftfuncs.cpp chebyshev.cpp
density_fitting.cpp lebedev.cpp)
else()
add_library (erkale 
basis.cpp basislibrary.cpp stringutil.cpp mathf.cpp integrals.cpp
eritable.cpp eriscreen.cpp timer.cpp linalg.cpp obara-saika.cpp
solidharmonics.cpp diis.cpp scf.cpp elements.cpp xyzutils.cpp
settings.cpp broyden.cpp adiis.cpp tempered.cpp find_molecules.cpp
properties.cpp)
endif()


# The name of the executable is
set_target_properties(erkale PROPERTIES OUTPUT_NAME "erkale${SUFFIX}")

# If we build shared libraries, they must be linked to the other libraries
if(BUILD_SHARED_LIBS)
 set_target_properties(erkale PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 target_link_libraries(erkale ${GSL_LIBRARIES})
 target_link_libraries(erkale ${LAPACK_LIBRARIES})
 if(LIBINT_FOUND)
  target_link_libraries(erkale ${LIBINT_LIBRARIES})
 endif()
 if(USEDFT)
  target_link_libraries(erkale ${LIBXC_LIBRARIES})
 endif()
endif()

# Create the ERKALE main executable
add_executable (erkale.x main.cpp)
# The name of the executable is
set_target_properties(erkale.x PROPERTIES OUTPUT_NAME "erkale${SUFFIX}")
target_link_libraries(erkale.x erkale)
target_link_libraries(erkale.x erkale_emd)


# Link libraries
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale.x ${GSL_LIBRARIES})
 target_link_libraries(erkale.x ${LAPACK_LIBRARIES})
 if(USEDFT)
  target_link_libraries(erkale.x ${LIBXC_LIBRARIES})
 endif()
 if(LIBINT_FOUND)
  target_link_libraries(erkale.x ${LIBINT_LIBRARIES})
 endif()
endif()

# Install targets
install (TARGETS erkale DESTINATION lib${LIB_SUFFIX})
install (TARGETS erkale.x DESTINATION bin)

# Install headers
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale)

# Create the ERKALE electron momentum density library
add_subdirectory(emd)
# Create the completeness profile tool
add_subdirectory(completeness)
# Create the Casida executable
add_subdirectory(casida)
# Create the Slater fit tool
add_subdirectory(slaterfit)
# Create test tool
add_subdirectory(test)

# Recipe for building ERITable
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/eritable-base.cpp > ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${PPROC} -DERITABLE ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >> ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines-increment.cpp.in
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eritable-base.cpp
)

# Recipe for building ERIScreen
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines-increment.cpp.in
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eriscreen-base.cpp

 COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/eriscreen-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCJKab ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCJK ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCJ ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCK ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
)

# Recipe for building SCF
if(USEDFT)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in
 
  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -D_ROHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DDFT -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
)
else()
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in
 
  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DROHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
)
endif()
