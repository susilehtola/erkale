add_executable(basictests basictests.cpp)
# The name of the executable is
set_target_properties(basictests PROPERTIES OUTPUT_NAME "basictests${SUFFIX}")
target_link_libraries(basictests liberkale_emd)
target_link_libraries(basictests liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
  if(UNIX AND NOT APPLE)
    target_link_libraries(basictests -lrt)
  endif()
  target_link_libraries(basictests ${GSL_LIBRARIES})
  target_link_libraries(basictests ${LAPACK_LIBRARIES})
  target_link_libraries(basictests ${LIBXC_LIBRARIES})
  target_link_libraries(basictests ${LIBINT_LIBRARIES})
  target_link_libraries(basictests ${HDF5_LIBRARIES})
endif()

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

add_executable(integraltest integraltest.cpp)
# The name of the executable is
set_target_properties(integraltest PROPERTIES OUTPUT_NAME "integraltest${SUFFIX}")
target_link_libraries(integraltest liberkale_emd)
target_link_libraries(integraltest liberkale)

# Link libraries
if(BUILD_SHARED_LIBS)
else()
  if(UNIX AND NOT APPLE)
    target_link_libraries(integraltest -lrt)
  endif()
  target_link_libraries(integraltest ${GSL_LIBRARIES})
  target_link_libraries(integraltest ${LAPACK_LIBRARIES})
  target_link_libraries(integraltest ${LIBXC_LIBRARIES})
  target_link_libraries(integraltest ${LIBINT_LIBRARIES})
  target_link_libraries(integraltest ${HDF5_LIBRARIES})
endif()
