# Create the basis set tool
add_executable (erkale_bastool main.cpp pivoted_cholesky_basis.cpp)
set_target_properties(erkale_bastool PROPERTIES OUTPUT_NAME "erkale_bastool${SUFFIX}")
target_link_libraries(erkale_bastool liberkale)
target_link_libraries(erkale_bastool liberkale_cmp)
# Link other libraries if necessary
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_bastool -lrt)
 endif()
 target_link_libraries(erkale_bastool ${GSL_LIBRARIES})
 target_link_libraries(erkale_bastool ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_bastool ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_bastool ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_bastool ${HDF5_LIBRARIES})
endif()

install (TARGETS erkale_bastool DESTINATION bin)
