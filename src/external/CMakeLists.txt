# Interop tools for external programs (Gaussian fchk, ADF).
add_executable(erkale_fchkpt fchkpt.cpp)
set_target_properties(erkale_fchkpt PROPERTIES OUTPUT_NAME "erkale_fchkpt${SUFFIX}")
target_link_libraries(erkale_fchkpt liberkale)

add_executable(erkale_adf_emd adf/main.cpp adf/ArrayList.c adf/KFReader.c)
set_target_properties(erkale_adf_emd PROPERTIES OUTPUT_NAME "erkale_adf_emd${SUFFIX}")
target_link_libraries(erkale_adf_emd liberkale_emd liberkale)

# Optional TREXIO (https://trex-coe.github.io/trexio/) interface.
# Built only if libtrexio is found; set TREXIO_DIR to point at an install.
set(TREXIO_DIR "" CACHE PATH "Path to a TREXIO install (containing include/trexio.h and lib/libtrexio)")
find_path(TREXIO_INCLUDE_DIR trexio.h HINTS ${TREXIO_DIR}/include)
find_library(TREXIO_LIBRARY NAMES trexio HINTS ${TREXIO_DIR}/lib ${TREXIO_DIR}/lib64)

set(external_targets erkale_fchkpt erkale_adf_emd)
if(TREXIO_INCLUDE_DIR AND TREXIO_LIBRARY)
  message(STATUS "TREXIO found (${TREXIO_LIBRARY}); building erkale_trexio")
  add_executable(erkale_trexio trexio_tool.cpp trexio_interface.cpp)
  set_target_properties(erkale_trexio PROPERTIES OUTPUT_NAME "erkale_trexio${SUFFIX}")
  target_include_directories(erkale_trexio PRIVATE ${TREXIO_INCLUDE_DIR})
  target_link_libraries(erkale_trexio liberkale ${TREXIO_LIBRARY})
  list(APPEND external_targets erkale_trexio)
else()
  message(STATUS "TREXIO not found; skipping erkale_trexio (set TREXIO_DIR to enable)")
endif()

install(TARGETS ${external_targets} DESTINATION bin)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/external)
