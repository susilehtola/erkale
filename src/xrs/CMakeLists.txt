# Make sure the headers are found
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

add_library (liberkale_xrs bfprod.cpp fourierprod.cpp lmtrans.cpp
momentum_series.cpp xrsscf-base.cpp xrsscf-fock.cpp 
xrsscf-solvers.cpp)

# The name of the library is
set_target_properties(liberkale_xrs PROPERTIES OUTPUT_NAME "erkale_xrs${SUFFIX}")

if(BUILD_SHARED_LIBS)
 set_target_properties(liberkale_xrs PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 target_link_libraries(liberkale_xrs liberkale)
 target_link_libraries(liberkale_xrs liberkale_emd)
endif()

add_executable(erkale_xrs main.cpp)
set_target_properties(erkale_xrs PROPERTIES OUTPUT_NAME "erkale_xrs${SUFFIX}")
target_link_libraries(erkale_xrs liberkale_xrs)

if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_xrs liberkale)
 target_link_libraries(erkale_xrs liberkale_emd)
 target_link_libraries(erkale_xrs -lrt)
 target_link_libraries(erkale_xrs ${GSL_LIBRARIES})
 target_link_libraries(erkale_xrs ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_xrs ${HDF5_LIBRARIES})
 target_link_libraries(erkale_xrs ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_xrs ${LIBINT_LIBRARIES})
endif()


install (TARGETS liberkale_xrs DESTINATION lib${LIB_SUFFIX})
install (TARGETS erkale_xrs DESTINATION bin)

# Install headers
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/xrs)


add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xrsscf-fock.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../scf-fock.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../scf-includes.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/xrsscf-includes.cpp.in

  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/../scf-includes.cpp.in >${CMAKE_CURRENT_BINARY_DIR}/xrsscf-fock.cpp
  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/xrsscf-includes.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf-fock.cpp
  COMMAND ${PPROC} -DDFT -DHALFHOLE ${CMAKE_CURRENT_SOURCE_DIR}/../scf-fock.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf-fock.cpp
  COMMAND ${PPROC} -DDFT -DFULLHOLE ${CMAKE_CURRENT_SOURCE_DIR}/../scf-fock.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf-fock.cpp
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xrsscf-solvers.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../scf-solvers.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../scf-includes.cpp.in
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/xrsscf-includes.cpp.in

  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/../scf-includes.cpp.in >${CMAKE_CURRENT_BINARY_DIR}/xrsscf-solvers.cpp
  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/xrsscf-includes.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf-solvers.cpp
  COMMAND ${PPROC} -DDFT -DHALFHOLE ${CMAKE_CURRENT_SOURCE_DIR}/../scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf-solvers.cpp
  COMMAND ${PPROC} -DDFT -DFULLHOLE ${CMAKE_CURRENT_SOURCE_DIR}/../scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf-solvers.cpp
)
