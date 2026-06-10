# Gaussian EMD executable
add_executable (erkale_fchkpt fchkpt.cpp )
set_target_properties(erkale_fchkpt PROPERTIES OUTPUT_NAME "erkale_fchkpt${SUFFIX}")

target_link_libraries(erkale_fchkpt liberkale)
if(BUILD_SHARED_LIBS)
else()
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_fchkpt -lrt)
 endif()
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
 if(UNIX AND NOT APPLE)
  target_link_libraries(erkale_adf_emd -lrt)
 endif()
 target_link_libraries(erkale_adf_emd ${GSL_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${LAPACK_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${HDF5_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${LIBINT_LIBRARIES})
 target_link_libraries(erkale_adf_emd ${LIBXC_LIBRARIES})
endif()

# Optional TREXIO (https://trex-coe.github.io/trexio/) interface.
# A system/`TREXIO_DIR` libtrexio is used if found; otherwise it is
# fetched and built from upstream (disable with ERKALE_FETCH_TREXIO=OFF).
option(ERKALE_FETCH_TREXIO "If libtrexio is not found, fetch and build it from upstream" ON)
set(TREXIO_DIR "" CACHE PATH "Path to a TREXIO install (containing include/trexio.h and lib/libtrexio)")
find_path(TREXIO_INCLUDE_DIR trexio.h
  HINTS ${TREXIO_DIR}/include /home/work/trexio/install/include /home/work/trexio/include)
find_library(TREXIO_LIBRARY NAMES trexio
  HINTS ${TREXIO_DIR}/lib ${TREXIO_DIR}/lib64 /home/work/trexio/install/lib /home/work/trexio/install/lib64)

set(ERKALE_BUILD_TREXIO OFF)
set(TREXIO_LINK "")
set(TREXIO_INCLUDES "")
set(TREXIO_HAS_HELPERS OFF)

if(TREXIO_INCLUDE_DIR AND TREXIO_LIBRARY)
  message(STATUS "TREXIO found (${TREXIO_LIBRARY}); building erkale_trexio")
  set(ERKALE_BUILD_TREXIO ON)
  set(TREXIO_LINK ${TREXIO_LIBRARY})
  set(TREXIO_INCLUDES ${TREXIO_INCLUDE_DIR})
  # The export self-check uses trexio_compute_ao_overlap /
  # trexio_check_mo_orthonormality, which are not part of upstream TREXIO.
  # Enable it only when the found libtrexio actually provides them.
  # (Use a separate cache result var: check_library_exists skips its work
  # if the named variable is already defined, and TREXIO_HAS_HELPERS is.)
  include(CheckLibraryExists)
  check_library_exists(${TREXIO_LIBRARY} trexio_compute_ao_overlap "" ERKALE_TREXIO_LIB_HAS_HELPERS)
  if(ERKALE_TREXIO_LIB_HAS_HELPERS)
    set(TREXIO_HAS_HELPERS ON)
  endif()
elseif(ERKALE_FETCH_TREXIO)
  message(STATUS "TREXIO not found; fetching v2.6.1 from upstream (set ERKALE_FETCH_TREXIO=OFF to disable).")
  include(FetchContent)
  # Build only the C library: no TREXIO tests in the subbuild.
  set(TREXIO_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    trexio
    GIT_REPOSITORY https://github.com/trex-coe/trexio.git
    GIT_TAG        v2.6.1
  )
  FetchContent_MakeAvailable(trexio)
  set(ERKALE_BUILD_TREXIO ON)
  set(TREXIO_LINK trexio)
  # The lib is not built yet at configure time, so probe the fetched
  # header for the helper declaration instead of linking against it.
  # Stock upstream lacks them today, but will pick them up if they land.
  if(EXISTS "${trexio_SOURCE_DIR}/include/trexio.h")
    file(STRINGS "${trexio_SOURCE_DIR}/include/trexio.h" TREXIO_HELPER_DECL
      REGEX "trexio_compute_ao_overlap")
    if(TREXIO_HELPER_DECL)
      set(TREXIO_HAS_HELPERS ON)
    endif()
  endif()
else()
  message(STATUS "TREXIO not found and ERKALE_FETCH_TREXIO=OFF; skipping erkale_trexio.")
endif()

if(ERKALE_BUILD_TREXIO)
  add_executable (erkale_trexio trexio_tool.cpp trexio_interface.cpp)
  set_target_properties(erkale_trexio PROPERTIES OUTPUT_NAME "erkale_trexio${SUFFIX}")
  if(TREXIO_INCLUDES)
    target_include_directories(erkale_trexio PRIVATE ${TREXIO_INCLUDES})
  endif()
  if(TREXIO_HAS_HELPERS)
    message(STATUS "  libtrexio provides overlap helpers; enabling export self-check")
    target_compile_definitions(erkale_trexio PRIVATE ERKALE_TREXIO_OVERLAP_HELPERS)
  endif()
  target_link_libraries(erkale_trexio liberkale ${TREXIO_LINK})
  if(BUILD_SHARED_LIBS)
  else()
   if(UNIX AND NOT APPLE)
    target_link_libraries(erkale_trexio -lrt)
   endif()
   target_link_libraries(erkale_trexio ${GSL_LIBRARIES})
   target_link_libraries(erkale_trexio ${LAPACK_LIBRARIES})
   target_link_libraries(erkale_trexio ${HDF5_LIBRARIES})
   target_link_libraries(erkale_trexio ${LIBINT_LIBRARIES})
   target_link_libraries(erkale_trexio ${LIBXC_LIBRARIES})
  endif()
  install (TARGETS erkale_trexio DESTINATION bin)
endif()

install (TARGETS erkale_fchkpt erkale_adf_emd DESTINATION bin)
# Install headers
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/external)
