add_executable(erkale_slaterfit main.cpp)
# The name of the executable is
set_target_properties(erkale_slaterfit PROPERTIES OUTPUT_NAME "erkale_slaterfit${SUFFIX}")

target_link_libraries(erkale_slaterfit liberkale)
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_slaterfit -lrt)
 target_link_libraries(erkale_slaterfit ${GSL_LIBRARIES})
 target_link_libraries(erkale_slaterfit ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_slaterfit ${LIBINT_LIBRARIES})
endif()



install (TARGETS erkale_slaterfit DESTINATION bin)
# Install headers
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/slaterfit)
