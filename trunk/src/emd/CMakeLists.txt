add_library (liberkale_emd
emd.cpp gto_fourier.cpp spherical_expansion.cpp spherical_harmonics.cpp )
# The name of the library is
set_target_properties(liberkale_emd PROPERTIES OUTPUT_NAME "erkale_emd${SUFFIX}")

if(BUILD_SHARED_LIBS)
 set_target_properties(liberkale_emd PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 target_link_libraries(liberkale_emd liberkale)
endif()

install (TARGETS liberkale_emd DESTINATION lib${LIB_SUFFIX})


# EMD executable
add_executable (erkale_emd main.cpp)
set_target_properties(erkale_emd PROPERTIES OUTPUT_NAME "erkale_emd${SUFFIX}")

target_link_libraries(erkale_emd liberkale)
target_link_libraries(erkale_emd liberkale_emd)
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_emd ${GSL_LIBRARIES})
 target_link_libraries(erkale_emd ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_emd ${HDF5_LIBRARIES})
 target_link_libraries(erkale_emd ${LIBINT_LIBRARIES})
endif()

install (TARGETS erkale_emd DESTINATION bin)
