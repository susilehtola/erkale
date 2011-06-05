# Create the ERKALE base library "liberkale" which includes the following sources
add_library (erkale 
basis.cpp basislibrary.cpp stringutil.cpp mathf.cpp integrals.cpp
eritable.cpp eriscreen.cpp timer.cpp linalg.cpp obara-saika.cpp
solidharmonics.cpp diis.cpp scf.cpp elements.cpp xyzutils.cpp
settings.cpp broyden.cpp adiis.cpp tempered.cpp
completeness/completeness_profile.cpp )

# Create the ERKALE DFT library "liberkale_dft" which contains the sources
add_library (erkale_dft
lobatto.cpp dftgrid.cpp dftfuncs.cpp chebyshev.cpp
density_fitting.cpp lebedev.cpp )

# Recipe for building ERITable
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 COMMAND ${CMAKE_COMMAND} ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/eritable-base.cpp > ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
# COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >> ${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
# COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eritable.cpp
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/eritable-base.cpp
)

# Recipe for building ERIScreen
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/eriscreen-base.cpp

 COMMAND ${CMAKE_COMMAND} ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/eriscreen-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E -DCALCJKab ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E -DCALCJK ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E -DCALCJ ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E -DCALCK ${CMAKE_CURRENT_SOURCE_DIR}/eri-routines.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/eriscreen.cpp
)


# Recipe for building SCF
add_custom_command(
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in

 COMMAND ${CMAKE_COMMAND} ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/scf-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E -DDFT ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
 COMMAND ${CMAKE_COMMAND} ${CMAKE_C_COMPILER} -E -DDFT -DRESTRICTED ${CMAKE_CURRENT_SOURCE_DIR}/scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/scf.cpp
)
