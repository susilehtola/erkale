# Create the completeness profile library
add_library(liberkale_cmp completeness_profile.cpp optimize_completeness.cpp)
set_target_properties(liberkale_cmp PROPERTIES OUTPUT_NAME "erkale_cmp${SUFFIX}")
if(BUILD_SHARED_LIBS)
 set_target_properties(liberkale_cmp PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 target_link_libraries(liberkale_cmp ${GSL_LIBRARIES})
endif()

# Completeness optimization tool
add_executable (erkale_copt main.cpp)
set_target_properties(erkale_copt PROPERTIES OUTPUT_NAME "erkale_copt${SUFFIX}")
target_link_libraries(erkale_copt liberkale_cmp)
target_link_libraries(erkale_copt liberkale)
# Link other libraries if necessary
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_copt -lrt)
 endif()
 target_link_libraries(erkale_copt ${GSL_LIBRARIES})
 target_link_libraries(erkale_copt ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_copt ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_copt ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_copt ${HDF5_LIBRARIES})
endif()

# Completeness optimization tool
add_executable (erkale_copt_plateau plateau.cpp)
set_target_properties(erkale_copt_plateau PROPERTIES OUTPUT_NAME "erkale_copt_plateau${SUFFIX}")
target_link_libraries(erkale_copt_plateau liberkale_cmp)
target_link_libraries(erkale_copt_plateau liberkale)
# Link other libraries if necessary
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_copt_plateau -lrt)
 endif()
 target_link_libraries(erkale_copt_plateau ${GSL_LIBRARIES})
 target_link_libraries(erkale_copt_plateau ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_copt_plateau ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_copt_plateau ${LIBXC_LIBRARIES})
 target_link_libraries(erkale_copt_plateau ${HDF5_LIBRARIES})
endif()


install (TARGETS erkale_copt DESTINATION bin)
install (TARGETS erkale_copt_plateau DESTINATION bin)
install (TARGETS liberkale_cmp DESTINATION lib${LIB_SUFFIX})
# Install headers
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/completeness)
