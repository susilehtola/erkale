# Create the completeness profile library
add_library(erkale_cmp completeness_profile.cpp optimize_completeness.cpp)
set_target_properties(erkale_cmp PROPERTIES OUTPUT_NAME "erkale_cmp${SUFFIX}")
if(BUILD_SHARED_LIBS)
 set_target_properties(erkale_cmp PROPERTIES
  VERSION ${VERSION} SOVERSION ${VERSION} )
 target_link_libraries(erkale_cmp ${GSL_LIBRARIES})
endif()

# Create the completeness profile plotting tool
add_executable (erkale_cprof print_completeness.cpp)
# The name of the executable is
set_target_properties(erkale_cprof PROPERTIES OUTPUT_NAME "erkale_cprof${SUFFIX}")
target_link_libraries(erkale_cprof erkale_cmp)
target_link_libraries(erkale_cprof erkale)
# Link other libraries if necessary
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_cprof ${GSL_LIBRARIES})
 target_link_libraries(erkale_cprof ${LAPACK_LIBRARIES})
endif()

# Completeness optimization tool
add_executable (erkale_copt main.cpp)
set_target_properties(erkale_copt PROPERTIES OUTPUT_NAME "erkale_copt${SUFFIX}")
target_link_libraries(erkale_copt erkale_cmp)
target_link_libraries(erkale_copt erkale)
# Link other libraries if necessary
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_copt ${GSL_LIBRARIES})
 target_link_libraries(erkale_copt ${LAPACK_LIBRARIES})
endif()

install (TARGETS erkale_cprof DESTINATION bin)
install (TARGETS erkale_copt DESTINATION bin)
