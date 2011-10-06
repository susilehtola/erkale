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
