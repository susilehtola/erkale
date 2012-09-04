# make sure cmake addes the binary directory for the project to
# the include path, since some sources are generated on-the-fly
# We also want to find the electron momentum density stuff
# and the completeness profile
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# Add the armadillo include directory
INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})
# Add the GSL include directory
INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIRS})
# Add libxc include directory
INCLUDE_DIRECTORIES(${LIBXC_INCLUDE_DIRS})
# Add libint include directory
INCLUDE_DIRECTORIES(${LIBINT_INCLUDE_DIRS})
# Add HDF5 include directory
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})

# Create the ERKALE base library "erkale" which includes the following
#sources

add_library (liberkale basis.cpp basislibrary.cpp stringutil.cpp
mathf.cpp integrals.cpp eritable.cpp eriscreen.cpp timer.cpp
linalg.cpp obara-saika.cpp solidharmonics.cpp diis.cpp scf-base.cpp
scf-solvers.cpp scf-fock.cpp scf-line.cpp elements.cpp xyzutils.cpp
settings.cpp broyden.cpp adiis.cpp tempered.cpp find_molecules.cpp
properties.cpp checkpoint.cpp guess.cpp external/storage.cpp
external/fchkpt_tools.cpp gaunt.cpp lmgrid.cpp spherical_harmonics.cpp
trrh.cpp trdsm.cpp lobatto.cpp dftgrid.cpp dftfuncs.cpp chebyshev.cpp
density_fitting.cpp lebedev.cpp)

# The name of the executable is
set_target_properties(liberkale PROPERTIES OUTPUT_NAME "erkale${SUFFIX}")

# If we build shared libraries, they must be linked to the other libraries
if(BUILD_SHARED_LIBS)
 set_target_properties(liberkale PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 target_link_libraries(liberkale ${GSL_LIBRARIES})
 target_link_libraries(liberkale ${LAPACK_LIBRARIES})
 target_link_libraries(liberkale ${LIBINT_LIBRARIES})
 target_link_libraries(liberkale ${LIBXC_LIBRARIES})
 target_link_libraries(liberkale ${HDF5_LIBRARIES})
endif()

# Create the ERKALE main executable
add_executable (erkale main.cpp)
# The name of the executable is
set_target_properties(erkale PROPERTIES OUTPUT_NAME "erkale${SUFFIX}")
target_link_libraries(erkale liberkale)
target_link_libraries(erkale liberkale_emd)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale ${GSL_LIBRARIES})
 target_link_libraries(erkale ${LAPACK_LIBRARIES})
 target_link_libraries(erkale ${LIBXC_LIBRARIES})
 target_link_libraries(erkale ${LIBINT_LIBRARIES})
 target_link_libraries(erkale ${HDF5_LIBRARIES})
endif()

# Density cube tool
add_executable (erkale_cube cube.cpp)
# The name of the executable is
set_target_properties(erkale_cube PROPERTIES OUTPUT_NAME "erkale_cube${SUFFIX}")
target_link_libraries(erkale_cube liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_cube ${GSL_LIBRARIES})
 target_link_libraries(erkale_cube ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_cube ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_cube ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_cube ${HDF5_LIBRARIES})
endif()

# Install targets
install (TARGETS liberkale DESTINATION lib${LIB_SUFFIX})
install (TARGETS erkale DESTINATION bin)
install (TARGETS erkale_cube DESTINATION bin)

# Install headers
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale)

# Create the ERKALE electron momentum density library
add_subdirectory(emd)
# Create the completeness profile tool
add_subdirectory(completeness)
# Create the Casida executable
add_subdirectory(casida)
# Create the XRS executable
add_subdirectory(xrs)

# Create the Slater fit tool
add_subdirectory(slaterfit)
# Create test tool
add_subdirectory(test)
# Create basis set tool
add_subdirectory(basistool)

# Compile support programs for external programs
add_subdirectory(external)

# Recipe for building ERITable
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/eritable-base.cpp > ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${PPROC} -DERITABLE ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >> ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${PPROC} -DERITABLE -DSTRICT ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >> ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
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
 COMMAND ${PPROC} -DCALCJKab -DSTRICT ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCJK ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCJK -DSTRICT ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCJ ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCJ -DSTRICT ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCK ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${PPROC} -DCALCK -DSTRICT ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
)

# Recipe for building SCF
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf-solvers.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in

  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in >${CMAKE_CURRENT_BINARY_DIR}/scf-solvers.cpp
  COMMAND ${PPROC} -DRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-solvers.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-solvers.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -D_ROHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-solvers.cpp
  COMMAND ${PPROC} -DRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-solvers.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-solvers.cpp
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf-fock.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-fock.cpp.in

  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in >${CMAKE_CURRENT_BINARY_DIR}/scf-fock.cpp
  COMMAND ${PPROC} -DRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-fock.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-fock.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-fock.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-fock.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -D_ROHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-fock.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-fock.cpp
  COMMAND ${PPROC} -DRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-fock.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-fock.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-fock.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-fock.cpp
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf-line.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-line.cpp.in

  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in >${CMAKE_CURRENT_BINARY_DIR}/scf-line.cpp
  COMMAND ${PPROC} -DRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-line.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-line.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-line.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-line.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -D_ROHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-line.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-line.cpp
  COMMAND ${PPROC} -DRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-line.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-line.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-line.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-line.cpp
)
