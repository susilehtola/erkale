# Gaussian EMD executable
add_executable (erkale_chkpt_gaussian gaussian_chkpt.cpp storage.cpp )
set_target_properties(erkale_chkpt_gaussian PROPERTIES OUTPUT_NAME "erkale_chkpt_gaussian${SUFFIX}")

target_link_libraries(erkale_chkpt_gaussian liberkale)
if(BUILD_SHARED_LIBS)
else()
 target_link_libraries(erkale_chkpt_gaussian ${GSL_LIBRARIES})
 target_link_libraries(erkale_chkpt_gaussian ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_chkpt_gaussian ${HDF5_LIBRARIES})
 target_link_libraries(erkale_chkpt_gaussian ${LIBINT_LIBRARIES})
endif()

install (TARGETS erkale_chkpt_gaussian DESTINATION bin)
