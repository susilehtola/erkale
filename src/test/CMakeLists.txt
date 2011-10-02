add_executable(erkale_tests tests.cpp)
# The name of the executable is
set_target_properties(erkale_tests PROPERTIES OUTPUT_NAME "erkale_tests${SUFFIX}")

target_link_libraries(erkale_tests erkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_tests ${GSL_LIBRARIES})
 target_link_libraries(erkale_tests ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_tests ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_tests ${LIBINT_LIBRARIES})
endif()
