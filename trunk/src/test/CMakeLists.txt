add_executable(chkcompare chkcompare.cpp)
# The name of the executable is
set_target_properties(chkcompare PROPERTIES OUTPUT_NAME "chkcompare${SUFFIX}")
target_link_libraries(chkcompare liberkale_emd)
target_link_libraries(chkcompare liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
  if(UNIX AND NOT APPLE)
    target_link_libraries(chkcompare -lrt)
  endif()
  target_link_libraries(chkcompare ${GSL_LIBRARIES})
  target_link_libraries(chkcompare ${LAPACK_LIBRARIES})
  target_link_libraries(chkcompare ${LIBXC_LIBRARIES})
  target_link_libraries(chkcompare ${LIBINT_LIBRARIES})
  target_link_libraries(chkcompare ${HDF5_LIBRARIES})
endif()
