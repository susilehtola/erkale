# Gaussian EMD executable
add_executable (erkale_fchkpt fchkpt.cpp )
set_target_properties(erkale_fchkpt PROPERTIES OUTPUT_NAME "erkale_fchkpt${SUFFIX}")

target_link_libraries(erkale_fchkpt liberkale)
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_fchkpt -lrt)
 target_link_libraries(erkale_fchkpt ${GSL_LIBRARIES})
 target_link_libraries(erkale_fchkpt ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_fchkpt ${HDF5_LIBRARIES})
 target_link_libraries(erkale_fchkpt ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_fchkpt ${LIBXC_LIBRARIES})
endif()

# ADF EMD executable
add_executable (erkale_adf_emd adf/main.cpp adf/ArrayList.c adf/KFReader.c)
set_target_properties(erkale_adf_emd PROPERTIES OUTPUT_NAME "erkale_adf_emd${SUFFIX}")
target_link_libraries(erkale_adf_emd liberkale_emd)
target_link_libraries(erkale_adf_emd liberkale)
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_adf_emd -lrt)
 target_link_libraries(erkale_adf_emd ${GSL_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${HDF5_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${LIBXC_LIBRARIES})
endif()

install (TARGETS erkale_fchkpt erkale_adf_emd DESTINATION bin)
# Install headers
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/external)
