add_library (erkale_emd
complex.cpp emd.cpp gto_fourier.cpp
spherical_expansion.cpp spherical_harmonics.cpp )
if(BUILD_SHARED_LIBS)
 set_target_properties(erkale_emd PROPERTIES
  VERSION ${VERSION} SOVERSION ${SOVERSION} )
endif()

install (TARGETS erkale_emd DESTINATION lib)
