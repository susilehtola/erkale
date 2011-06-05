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
if(xc_FOUND)
 INCLUDE_DIRECTORIES(${xc_INCLUDE_DIRS})
endif()
# Add libint include directory if necessary
if(int_FOUND)
 INCLUDE_DIRECTORIES(${int_INCLUDE_DIRS})
endif()

# Create the ERKALE base library "erkale" which includes the following sources
add_library (erkale 
basis.cpp basislibrary.cpp stringutil.cpp mathf.cpp integrals.cpp
eritable.cpp eriscreen.cpp timer.cpp linalg.cpp obara-saika.cpp
solidharmonics.cpp diis.cpp scf.cpp elements.cpp xyzutils.cpp
settings.cpp broyden.cpp adiis.cpp tempered.cpp
completeness/completeness_profile.cpp )

# Create the ERKALE electron momentum density library
add_subdirectory(emd)
#add_library (erkale_emd
#emd/complex.cpp emd/emd.cpp emd/gto_fourier.cpp
#emd/spherical_expansion.cpp emd/spherical_harmonics.cpp )

# Create the ERKALE DFT library "erkale_dft" which contains the sources
if(USEDFT)
 add_library (erkale_dft
 lobatto.cpp dftgrid.cpp dftfuncs.cpp chebyshev.cpp
 density_fitting.cpp lebedev.cpp )
endif()

# Create the ERKALE main executable
add_executable (erkale.x main.cpp)
# The name of the executable is
set_target_properties(erkale.x PROPERTIES OUTPUT_NAME erkale)
target_link_libraries(erkale.x erkale)
target_link_libraries(erkale.x erkale_emd)
target_link_libraries(erkale.x ${GSL_LIBRARIES})
target_link_libraries(erkale.x ${LAPACK_LIBRARIES})
# Compile with DFT support?
if(USEDFT)
 target_link_libraries(erkale.x erkale_dft)
 target_link_libraries(erkale.x ${LIBXC_LIBRARIES})
endif()
# Compile with libint support?
if(LIBINT_FOUND)
 target_link_libraries(erkale.x ${LIBINT_LIBRARIES})
endif()

# Create the completeness profile tool
add_subdirectory(completeness)

# Recipe for building ERITable
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/eritable-base.cpp > ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${PPROC} -DERITABLE ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >> ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/eritable-base.cpp
)

# Recipe for building ERIScreen
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/eriscreen-base.cpp

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
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in
 
  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DDFT -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
 )
else()
 add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in
 
  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
  COMMAND ${PPROC} -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
 )
endif()

