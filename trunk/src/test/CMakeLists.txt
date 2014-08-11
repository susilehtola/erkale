add_executable(erkale_tests tests.cpp)
# The name of the executable is
set_target_properties(erkale_tests PROPERTIES OUTPUT_NAME "erkale_tests${SUFFIX}")
target_link_libraries(erkale_tests liberkale_emd)
target_link_libraries(erkale_tests liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_tests -lrt)
 endif()
 target_link_libraries(erkale_tests ${GSL_LIBRARIES})
 target_link_libraries(erkale_tests ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_tests ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_tests ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_tests ${HDF5_LIBRARIES})
endif()
