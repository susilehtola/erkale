# Create the completeness profile plotting tool
add_executable (erkale_completeness print_completeness.cpp)
target_link_libraries(erkale_completeness erkale)
target_link_libraries(erkale_completeness ${GSL_LIBRARIES})
target_link_libraries(erkale_completeness ${LAPACK_LIBRARIES})
