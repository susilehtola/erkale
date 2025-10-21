# make sure cmake addes the binary directory for the project to
# the include path, since some sources are generated on-the-fly
# We also want to find the electron momentum density stuff
# and the completeness profile
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# Add the armadillo include directory
INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})
# Add the GSL include directory
INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIRS})
LINK_DIRECTORIES(${GSL_LIBRARY_DIRS})
# Add libxc include directory
INCLUDE_DIRECTORIES(${LIBXC_INCLUDE_DIRS})
LINK_DIRECTORIES(${LIBXC_LIBRARY_DIRS})
# Add libint include directory
INCLUDE_DIRECTORIES(${LIBINT_INCLUDE_DIRS})
LINK_DIRECTORIES(${LIBINT_LIBRARY_DIRS})
# Add HDF5 include directory
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
LINK_DIRECTORIES(${HDF5_LIBRARY_DIRS})

# Create the ERKALE base library "erkale" which includes the following
#sources

add_library (liberkale basis.cpp basislibrary.cpp boys.cpp
stringutil.cpp mathf.cpp integrals.cpp eri_digest.cpp eritable.cpp
eriscreen.cpp timer.cpp linalg.cpp obara-saika.cpp solidharmonics.cpp
diis.cpp scf-base.cpp localization.cpp scf-solvers.cpp scf-fock.cpp
scf-force.cpp elements.cpp xyzutils.cpp zmatrix.cpp settings.cpp
broyden.cpp tempered.cpp find_molecules.cpp properties.cpp
checkpoint.cpp guess.cpp external/storage.cpp
external/fchkpt_tools.cpp gaunt.cpp lmgrid.cpp spherical_harmonics.cpp
trrh.cpp lobatto.cpp dftgrid.cpp dftfuncs.cpp chebyshev.cpp
density_fitting.cpp lebedev.cpp eriworker_routines.cpp unitary.cpp
pzstability.cpp hirshfeld.cpp hirshfeldi.cpp bader.cpp badergrid.cpp
stockholder.cpp erifit.cpp erichol.cpp lbfgs.cpp gdm.cpp
slaterfit/solve_coefficients.cpp slaterfit/form_exponents.cpp
atom/integrals.cpp atom/solvers.cpp atom/atomtable.cpp
sap.cpp)

# The name of the executable is
set_target_properties(liberkale PROPERTIES OUTPUT_NAME "erkale${SUFFIX}")

# If we build shared libraries, they must be linked to the other libraries
if(BUILD_SHARED_LIBS)
 set_target_properties(liberkale PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 if(UNIX AND NOT APPLE)
  target_link_libraries(liberkale -lrt)
 endif()
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
 if(UNIX AND NOT APPLE) 
  target_link_libraries(erkale -lrt)
 endif()
 target_link_libraries(erkale ${GSL_LIBRARIES})
 target_link_libraries(erkale ${LAPACK_LIBRARIES})
 target_link_libraries(erkale ${LIBXC_LIBRARIES})
 target_link_libraries(erkale ${LIBINT_LIBRARIES})
 target_link_libraries(erkale ${HDF5_LIBRARIES})
endif()

# Create the ERKALE geometry optimization executable
add_executable (erkale_geom geometry.cpp)
# The name of the executable is
set_target_properties(erkale_geom PROPERTIES OUTPUT_NAME "erkale_geom${SUFFIX}")
target_link_libraries(erkale_geom liberkale)
target_link_libraries(erkale_geom liberkale_emd)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)				  
  target_link_libraries(erkale_geom -lrt)
 endif()
 target_link_libraries(erkale_geom ${GSL_LIBRARIES})
 target_link_libraries(erkale_geom ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_geom ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_geom ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_geom ${HDF5_LIBRARIES})
endif()


# Create the ERKALE localization executable
add_executable (erkale_loc localize.cpp)
# The name of the executable is
set_target_properties(erkale_loc PROPERTIES OUTPUT_NAME "erkale_loc${SUFFIX}")
target_link_libraries(erkale_loc liberkale)
target_link_libraries(erkale_loc liberkale_emd)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_loc -lrt)
 endif()
 target_link_libraries(erkale_loc ${GSL_LIBRARIES})
 target_link_libraries(erkale_loc ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_loc ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_loc ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_loc ${HDF5_LIBRARIES})
endif()


# Spherical transform generator
add_executable (eriworker_gentransform eriworker_gentransform.cpp solidharmonics.cpp mathf.cpp)
target_link_libraries(eriworker_gentransform ${GSL_LIBRARIES})
target_link_libraries(eriworker_gentransform ${LAPACK_LIBRARIES})


# Density cube tool
add_executable (erkale_cube cube.cpp)
# The name of the executable is
set_target_properties(erkale_cube PROPERTIES OUTPUT_NAME "erkale_cube${SUFFIX}")
target_link_libraries(erkale_cube liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_cube -lrt)
 endif()
 target_link_libraries(erkale_cube ${GSL_LIBRARIES})
 target_link_libraries(erkale_cube ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_cube ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_cube ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_cube ${HDF5_LIBRARIES})
endif()

# Linear configurations
add_executable (erkale_genlinconf genlinconf.cpp)
# The name of the executable is
set_target_properties(erkale_genlinconf PROPERTIES OUTPUT_NAME "erkale_genlinconf${SUFFIX}")
target_link_libraries(erkale_genlinconf liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_genlinconf -lrt)
 endif()
 target_link_libraries(erkale_genlinconf ${GSL_LIBRARIES})
 target_link_libraries(erkale_genlinconf ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_genlinconf ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_genlinconf ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_genlinconf ${HDF5_LIBRARIES})
endif()

# Population analysis tool
add_executable (erkale_pop population.cpp)
# The name of the executable is
set_target_properties(erkale_pop PROPERTIES OUTPUT_NAME "erkale_pop${SUFFIX}")
target_link_libraries(erkale_pop liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_pop -lrt)
 endif()
 target_link_libraries(erkale_pop ${GSL_LIBRARIES})
 target_link_libraries(erkale_pop ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_pop ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_pop ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_pop ${HDF5_LIBRARIES})
endif()



# Install targets
install (TARGETS liberkale DESTINATION lib${LIB_SUFFIX})
install (TARGETS erkale DESTINATION bin)
install (TARGETS erkale_cube DESTINATION bin)
install (TARGETS erkale_geom DESTINATION bin)
install (TARGETS erkale_genlinconf DESTINATION bin)
install (TARGETS erkale_loc DESTINATION bin)
install (TARGETS erkale_pop DESTINATION bin)

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
# Create the atom program
add_subdirectory(atom)
# Create test tool
add_subdirectory(test)
# Create basis set tool
add_subdirectory(basistool)
# Compile support programs for external programs
add_subdirectory(external)

# Compile the extra programs
add_subdirectory(contrib)

# Recipe for compiling full ERIWorker file
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eriworker_routines.cpp
 DEPENDS eriworker_gentransform
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eriworker.cpp
 COMMAND eriworker_gentransform > ${CMAKE_CURRENT_BINARY_DIR}/eriworker_routines.cpp
 COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/eriworker.cpp >> ${CMAKE_CURRENT_BINARY_DIR}/eriworker_routines.cpp
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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf-force.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-force.cpp.in

  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in >${CMAKE_CURRENT_BINARY_DIR}/scf-force.cpp
  COMMAND ${PPROC} -DRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-force.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-force.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-force.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-force.cpp
#  COMMAND ${PPROC} -DUNRESTRICTED -D_ROHF ${CMAKE_CURRENT_SOURCE_DIR}/scf-force.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-force.cpp
  COMMAND ${PPROC} -DRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-force.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-force.cpp
  COMMAND ${PPROC} -DUNRESTRICTED -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-force.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf-force.cpp
)
