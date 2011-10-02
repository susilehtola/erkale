# Create the completeness profile library
add_library(erkale_cmp completeness_profile.cpp optimize_completeness.cpp)
set_target_properties(erkale_cmp PROPERTIES OUTPUT_NAME "erkale_cmp${SUFFIX}")
if(BUILD_SHARED_LIBS)
 set_target_properties(erkale_cmp PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 target_link_libraries(erkale_cmp ${GSL_LIBRARIES})
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
 if(LIBINT_FOUND)
  target_link_libraries(erkale_copt ${LIBINT_LIBRARIES})
 endif()
endif()

install (TARGETS erkale_copt DESTINATION bin)
install (TARGETS erkale_cmp DESTINATION lib${LIB_SUFFIX})
