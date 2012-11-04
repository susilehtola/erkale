# Create the basis set tool
add_executable (erkale_bastool main.cpp)
target_link_libraries(erkale_bastool liberkale)
target_link_libraries(erkale_bastool liberkale_cmp)
# Link other libraries if necessary
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_bastool -lrt)
 target_link_libraries(erkale_bastool ${GSL_LIBRARIES})
 target_link_libraries(erkale_bastool ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_bastool ${LIBINT_LIBRARIES})
endif()

install (TARGETS erkale_bastool DESTINATION bin)
