# Make sure the headers are found
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})


add_library (liberkale_xrs
bfprod.cpp fourierprod.cpp gaunt.cpp lmgrid.cpp lmtrans.cpp momentum_series.cpp xrsscf.cpp )
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
 target_link_libraries(erkale_xrs ${GSL_LIBRARIES})
 target_link_libraries(erkale_xrs ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_xrs ${HDF5_LIBRARIES})
 target_link_libraries(erkale_xrs ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_xrs ${LIBINT_LIBRARIES})
endif()


install (TARGETS liberkale_xrs DESTINATION lib${LIB_SUFFIX})
install (TARGETS erkale_xrs DESTINATION bin)


add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xrsscf.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/xrsscf-base.cpp
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../scf-solvers.cpp.in

  COMMAND ${CAT} ${CMAKE_CURRENT_SOURCE_DIR}/xrsscf-base.cpp >${CMAKE_CURRENT_BINARY_DIR}/xrsscf.cpp
  COMMAND ${PPROC} -DDFT -DHALFHOLE ${CMAKE_CURRENT_SOURCE_DIR}/../scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf.cpp
  COMMAND ${PPROC} -DDFT -DFULLHOLE ${CMAKE_CURRENT_SOURCE_DIR}/../scf-solvers.cpp.in >>${CMAKE_CURRENT_BINARY_DIR}/xrsscf.cpp
)
