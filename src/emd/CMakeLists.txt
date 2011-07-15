add_library (erkale_emd
complex.cpp emd.cpp gto_fourier.cpp
spherical_expansion.cpp spherical_harmonics.cpp )
# The name of the library is
set_target_properties(erkale_emd PROPERTIES OUTPUT_NAME "erkale_emd${SUFFIX}")

if(BUILD_SHARED_LIBS)
 set_target_properties(erkale_emd PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
 target_link_libraries(erkale_emd erkale)
endif()

install (TARGETS erkale_emd DESTINATION lib${LIB_SUFFIX})
