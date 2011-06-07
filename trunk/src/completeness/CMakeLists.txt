# Create the completeness profile plotting tool
add_executable (erkale_cprof print_completeness.cpp)
target_link_libraries(erkale_cprof erkale)

# Link other libraries if necessary
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_cprof ${GSL_LIBRARIES})
 target_link_libraries(erkale_cprof ${LAPACK_LIBRARIES})
endif()

install (TARGETS erkale_cprof DESTINATION bin)
