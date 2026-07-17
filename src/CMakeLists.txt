# liberkale: the ERKALE engine library. All dependencies are attached
# PUBLIC so their include directories and link requirements propagate
# transitively to every executable and sublibrary that links liberkale.

add_library(liberkale
  basis.cpp basislibrary.cpp
  stringutil.cpp mathf.cpp integrals.cpp eri_digest.cpp eritable.cpp
  eriscreen.cpp timer.cpp linalg.cpp solidharmonics.cpp
  diis.cpp scf-base.cpp localization.cpp scf-solvers.cpp scf-fock.cpp
  scf-force.cpp elements.cpp xyzutils.cpp zmatrix.cpp settings.cpp
  broyden.cpp tempered.cpp find_molecules.cpp properties.cpp
  checkpoint.cpp guess.cpp external/storage.cpp
  external/fchkpt_tools.cpp gaunt.cpp lmgrid.cpp spherical_harmonics.cpp
  trrh.cpp lobatto.cpp dftgrid.cpp dftfuncs.cpp chebyshev.cpp
  density_fitting.cpp jkbuilder.cpp lebedev.cpp eriworker.cpp cintenv.cpp unitary.cpp
  pzstability.cpp hirshfeld.cpp hirshfeldi.cpp bader.cpp badergrid.cpp
  stockholder.cpp b_tensor.cpp lbfgs.cpp gdm.cpp
  slaterfit/solve_coefficients.cpp slaterfit/form_exponents.cpp
  sap.cpp)

set_target_properties(liberkale PROPERTIES OUTPUT_NAME "erkale${SUFFIX}")
if(BUILD_SHARED_LIBS)
  set_target_properties(liberkale PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${SOVERSION})
endif()

# Own headers (source tree + the binary dir holding generated sources and
# version.h) plus every dependency, all PUBLIC.
target_include_directories(liberkale PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/erkale>)
target_link_libraries(liberkale PUBLIC
  erkale_arma GSL::gsl Libxc::xc cint erkale_hdf5
  wignernj::wignernj nlohmann_json::nlohmann_json)

# OpenMP: PUBLIC so the -fopenmp compile/link flags reach every consumer
# (and _OPENMP is defined consistently with the scf-*.cpp.in code generator,
# which is invoked with the same OpenMP flags).
if(OpenMP_CXX_FOUND)
  target_link_libraries(liberkale PUBLIC OpenMP::OpenMP_CXX)
endif()

# System-wide default basis-set repository, if one was configured.
if(ERKALE_SYSTEM_LIBRARY)
  target_compile_definitions(liberkale PUBLIC
    ERKALE_SYSTEM_LIBRARY="${ERKALE_SYSTEM_LIBRARY}")
endif()

# Version information. When building from a git checkout, generate
# version.h (into the binary dir, which is on liberkale's include path)
# and define SVNRELEASE so the executables print the revision. Tarball
# builds skip this silently.
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-list --count --first-parent HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_REVCOUNT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(GIT_HASH)
    string(SUBSTRING "${GIT_HASH}" 0 8 GIT_SHORT)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
                   ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)
    target_compile_definitions(liberkale PUBLIC SVNRELEASE)
  endif()
endif()

# --- Executables. Each links liberkale (+ the sublibraries it needs);
# the external dependencies arrive transitively. ---

add_executable(erkale main.cpp)
set_target_properties(erkale PROPERTIES OUTPUT_NAME "erkale${SUFFIX}")
target_link_libraries(erkale liberkale liberkale_emd)

add_executable(erkale_geom geometry.cpp)
set_target_properties(erkale_geom PROPERTIES OUTPUT_NAME "erkale_geom${SUFFIX}")
target_link_libraries(erkale_geom liberkale liberkale_emd)

add_executable(erkale_loc localize.cpp)
set_target_properties(erkale_loc PROPERTIES OUTPUT_NAME "erkale_loc${SUFFIX}")
target_link_libraries(erkale_loc liberkale liberkale_emd)

add_executable(erkale_cube cube.cpp)
set_target_properties(erkale_cube PROPERTIES OUTPUT_NAME "erkale_cube${SUFFIX}")
target_link_libraries(erkale_cube liberkale)

add_executable(erkale_genlinconf genlinconf.cpp)
set_target_properties(erkale_genlinconf PROPERTIES OUTPUT_NAME "erkale_genlinconf${SUFFIX}")
target_link_libraries(erkale_genlinconf liberkale)

add_executable(erkale_pop population.cpp)
set_target_properties(erkale_pop PROPERTIES OUTPUT_NAME "erkale_pop${SUFFIX}")
target_link_libraries(erkale_pop liberkale)

# --- Install ---
install(TARGETS liberkale DESTINATION lib${LIB_SUFFIX})
install(TARGETS erkale erkale_cube erkale_geom erkale_genlinconf erkale_loc erkale_pop
        DESTINATION bin)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale)

# --- Subdirectory libraries and tools ---
add_subdirectory(emd)          # electron momentum density
add_subdirectory(completeness) # completeness profiles
add_subdirectory(casida)       # Casida TDDFT
add_subdirectory(xrs)          # X-ray spectra
add_subdirectory(slaterfit)    # Slater -> Gaussian fits
add_subdirectory(test)         # unit / integral tests
add_subdirectory(basistool)    # basis-set manipulation
add_subdirectory(external)     # ADF / Gaussian interop
add_subdirectory(contrib)      # research / experimental tools

# ---------------------------- SCF code generation ----------------------------
# The spin/method specialisations of the SCF driver are produced by running
# the C++ preprocessor over the *.cpp.in templates with combinations of
# -DRESTRICTED/-DUNRESTRICTED and -DHF/-DDFT/-D_ROHF. scf-includes.cpp.in is
# prepended once.

# Generate <name>.cpp from <name>.cpp.in for the given -D macro combinations.
# Each element of ARGN is one space-separated set of flags (one preprocessed
# block appended per set).
function(erkale_gen_scf name)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp)
  set(tmpl ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp.in)
  set(inc ${CMAKE_CURRENT_SOURCE_DIR}/scf-includes.cpp.in)
  set(commands COMMAND ${CAT} ${inc} >${out})
  foreach(flags IN LISTS ARGN)
    separate_arguments(flaglist UNIX_COMMAND "${flags}")
    list(APPEND commands COMMAND ${PPROC} ${flaglist} ${tmpl} >>${out})
  endforeach()
  add_custom_command(OUTPUT ${out} DEPENDS ${inc} ${tmpl} ${commands})
endfunction()

erkale_gen_scf(scf-solvers
  "-DRESTRICTED -DHF" "-DUNRESTRICTED -DHF" "-DUNRESTRICTED -D_ROHF"
  "-DRESTRICTED -DDFT" "-DUNRESTRICTED -DDFT")
erkale_gen_scf(scf-fock
  "-DRESTRICTED -DHF" "-DUNRESTRICTED -DHF" "-DUNRESTRICTED -D_ROHF"
  "-DRESTRICTED -DDFT" "-DUNRESTRICTED -DDFT")
# scf-force omits the ROHF specialisation.
erkale_gen_scf(scf-force
  "-DRESTRICTED -DHF" "-DUNRESTRICTED -DHF"
  "-DRESTRICTED -DDFT" "-DUNRESTRICTED -DDFT")
