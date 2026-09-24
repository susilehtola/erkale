# Interop tools for external programs (Gaussian fchk, ADF).
add_executable(erkale_fchkpt fchkpt.cpp)
set_target_properties(erkale_fchkpt PROPERTIES OUTPUT_NAME "erkale_fchkpt${SUFFIX}")
target_link_libraries(erkale_fchkpt liberkale)

add_executable(erkale_adf_emd adf/main.cpp adf/ArrayList.c adf/KFReader.c)
set_target_properties(erkale_adf_emd PROPERTIES OUTPUT_NAME "erkale_adf_emd${SUFFIX}")
target_link_libraries(erkale_adf_emd liberkale_emd liberkale)

include(FetchContent)

# Optional TREXIO (https://trex-coe.github.io/trexio/) interface.
# A system/TREXIO_DIR libtrexio is used if found; otherwise it is fetched
# and built from the trexio-release mirror (disable with ERKALE_FETCH_TREXIO=OFF).
option(ERKALE_FETCH_TREXIO "If libtrexio is not found, fetch and build it" ON)
set(TREXIO_DIR "" CACHE PATH "Path to a TREXIO install (containing include/trexio.h and lib/libtrexio)")
find_path(TREXIO_INCLUDE_DIR trexio.h HINTS ${TREXIO_DIR}/include)
find_library(TREXIO_LIBRARY NAMES trexio HINTS ${TREXIO_DIR}/lib ${TREXIO_DIR}/lib64)

set(external_targets erkale_fchkpt erkale_adf_emd)
set(ERKALE_BUILD_TREXIO OFF)
set(TREXIO_HAS_HELPERS OFF)
if(TREXIO_INCLUDE_DIR AND TREXIO_LIBRARY)
  message(STATUS "TREXIO found (${TREXIO_LIBRARY}); building erkale_trexio")
  set(ERKALE_BUILD_TREXIO ON)
  add_library(erkale_trexio_lib INTERFACE)
  target_include_directories(erkale_trexio_lib INTERFACE ${TREXIO_INCLUDE_DIR})
  target_link_libraries(erkale_trexio_lib INTERFACE ${TREXIO_LIBRARY})
  # The export self-check uses trexio_compute_ao_overlap /
  # trexio_check_mo_orthonormality, which are not part of upstream TREXIO.
  # Enable it only when the found libtrexio actually provides them.
  # (Use a separate cache result var: check_library_exists skips its work
  # if the named variable is already defined.)
  include(CheckLibraryExists)
  check_library_exists(${TREXIO_LIBRARY} trexio_compute_ao_overlap "" ERKALE_TREXIO_LIB_HAS_HELPERS)
  if(ERKALE_TREXIO_LIB_HAS_HELPERS)
    set(TREXIO_HAS_HELPERS ON)
  endif()
elseif(ERKALE_FETCH_TREXIO)
  message(STATUS "TREXIO not found; fetching v2.6.1 from trexio-release (set ERKALE_FETCH_TREXIO=OFF to disable).")
  # Build only the C library: no TREXIO tests in the subbuild.
  set(TREXIO_TESTS OFF CACHE BOOL "" FORCE)
  # trexio-release mirrors the release tarballs, which ship the generated
  # C sources. The upstream git repository only has the org-mode sources
  # and a .devel marker that makes TREXIO regenerate them with emacs.
  FetchContent_Declare(trexio
    GIT_REPOSITORY https://github.com/susilehtola/trexio-release.git
    GIT_TAG        v2.6.1)
  FetchContent_MakeAvailable(trexio)
  set(ERKALE_BUILD_TREXIO ON)
  add_library(erkale_trexio_lib INTERFACE)
  target_link_libraries(erkale_trexio_lib INTERFACE trexio)
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
  add_executable(erkale_trexio trexio_tool.cpp trexio_interface.cpp)
  set_target_properties(erkale_trexio PROPERTIES OUTPUT_NAME "erkale_trexio${SUFFIX}")
  if(TREXIO_HAS_HELPERS)
    message(STATUS "  libtrexio provides overlap helpers; enabling export self-check")
    target_compile_definitions(erkale_trexio PRIVATE ERKALE_TREXIO_OVERLAP_HELPERS)
  endif()
  target_link_libraries(erkale_trexio liberkale erkale_trexio_lib)
  list(APPEND external_targets erkale_trexio)

  # trexio-validate recomputes the contents of the exported files with
  # libcint; the test suite uses it to check the files erkale_trexio
  # writes. It is handed ERKALE's own TREXIO and libcint instead of
  # finding or fetching its own.
  option(ERKALE_TREXIO_VALIDATE "Validate the TREXIO export in the test suite with trexio-validate" ON)
  if(ERKALE_TREXIO_VALIDATE)
    if(TARGET trexio)
      # Already populated above, so its own fetch reuses ours.
      set(TREXIO_VALIDATE_FETCH_TREXIO ON)
    endif()
    # Its libcint lookup is skipped when these are set; the cint target
    # carries the include directories.
    set(LIBCINT_LIBRARY cint)
    if(NOT LIBCINT_INCLUDE_DIR)
      set(LIBCINT_INCLUDE_DIR "${cint_BINARY_DIR}/include")
    endif()
    FetchContent_Declare(TrexioValidate
      GIT_REPOSITORY https://github.com/susilehtola/trexio-validate.git
      GIT_TAG        33c3e1b59c8ed83698ce8d34e31f0106aa5b9d2c)
    FetchContent_MakeAvailable(TrexioValidate)
  endif()
endif()

install(TARGETS ${external_targets} DESTINATION bin)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/external)
