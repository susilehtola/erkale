# Each test is already designed to run in parallel so ctest should not parallellize
set(CTEST_PARALLEL_LEVEL 1)

function(run_test target)
  # Run ERKALE to get checkpoint file
  add_test(NAME ${target}_chk COMMAND erkale ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run)# &> ${target}.stdout)
  set_property(TEST ${target}_chk APPEND PROPERTY ENVIRONMENT "ERKALE_SYSDIR=${CMAKE_CURRENT_SOURCE_DIR}/xyz")
  set_property(TEST ${target}_chk APPEND PROPERTY ENVIRONMENT "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  set_property(TEST ${target}_chk APPEND PROPERTY DEPENDS erkale)
  set_property(TEST ${target}_chk APPEND PROPERTY DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run)
  
  # Run comparison against checkpoint file
  add_test(NAME ${target} COMMAND chkcompare ${target}.chk)
  set_property(TEST ${target} APPEND PROPERTY DEPENDS ${target}_chk)
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_REFDIR=${PROJECT_SOURCE_DIR}/refdata")
  set_property(TEST ${target} APPEND PROPERTY DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${target}.chk)
endfunction()

function(run_xrs target dep)
  # Run ERKALE to get checkpoint file
  add_test(NAME ${target}_chk COMMAND erkale_xrs ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run)# &> ${target}.stdout)
  set_property(TEST ${target}_chk APPEND PROPERTY ENVIRONMENT "ERKALE_SYSDIR=${CMAKE_CURRENT_SOURCE_DIR}/xyz")
  set_property(TEST ${target}_chk APPEND PROPERTY ENVIRONMENT "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  set_property(TEST ${target}_chk APPEND PROPERTY DEPENDS erkale_xrs)
  set_property(TEST ${target}_chk APPEND PROPERTY DEPENDS ${dep})
  set_property(TEST ${target}_chk APPEND PROPERTY DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run)
  
  # Run comparison against checkpoint file
  add_test(NAME ${target} COMMAND chkcompare ${target}.chk)
  set_property(TEST ${target} APPEND PROPERTY DEPENDS ${target}_chk)
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_REFDIR=${PROJECT_SOURCE_DIR}/refdata")
  set_property(TEST ${target} APPEND PROPERTY DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${target}.chk)
endfunction()

# Multicomponent (NEO) tests. The dump's self-consistency check reconstructs
# the SCF energy from the exported tensors and throws if they disagree, so the
# run validates the charge and the mass factors by itself; the expected energy
# guards against numerical drift. The multicomponent codes have no reference
# checkpoints, so they are not compared with chkcompare.
function(run_neo target energy)
  add_test(NAME ${target} COMMAND erkale_neo ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run)
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_SYSDIR=${CMAKE_CURRENT_SOURCE_DIR}/xyz")
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  set_property(TEST ${target} APPEND PROPERTY DEPENDS erkale_neo)
  set_property(TEST ${target} APPEND PROPERTY DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run)
  set_property(TEST ${target} PROPERTY PASS_REGULAR_EXPRESSION "Converged to energy ${energy}")
endfunction()

# First, compile and run the basic test
add_test(build_basictests  "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target basictests)
add_test(NAME basictests COMMAND basictests)

# To run any tests we need to check that ERKALE and the test comparison binary have been built
add_test(build_erkale     "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target erkale)
add_test(build_chkcompare "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target chkcompare)
add_test(build_erkale_neo "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target erkale_neo)

# Multicomponent tests: a proton, a particle of negative charge (which fixes
# the sign of the electron-particle coupling), a deuteron (the mass), and two
# particles of a charge whose square is not one (the particle-particle mean
# field). The last two conditions are what it takes to see the charge factors:
# they are invisible for a single proton.
run_neo(neo_proton "-75.98665")
run_neo(neo_negative "-75.32345")
run_neo(neo_deuteron "-75.99781")
run_neo(neo_two_particles "-76.34112")

# Generate list with genlist.sh
include(TestList.txt)

# TREXIO export: write TREXIO files from the checkpoints of a few of the
# tests above (restricted, unrestricted and ROHF; up to g functions) and
# validate them with trexio-validate. Each file is also imported back and
# re-exported, and the result validated, which exercises the import.
function(run_trexio target)
  set(trexio_env "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  add_test(NAME ${target}_trexio_export COMMAND erkale_trexio export ${target}.chk ${target}.h5)
  add_test(NAME ${target}_trexio_import COMMAND erkale_trexio import ${target}.h5 ${target}_trexio.chk)
  add_test(NAME ${target}_trexio_reexport COMMAND erkale_trexio export ${target}_trexio.chk ${target}_trexio.h5)
  set_tests_properties(${target}_trexio_export ${target}_trexio_import ${target}_trexio_reexport
    PROPERTIES ENVIRONMENT "${trexio_env}")
  set_tests_properties(${target}_trexio_export PROPERTIES DEPENDS ${target}_chk FIXTURES_SETUP ${target}_trexio)
  set_tests_properties(${target}_trexio_import PROPERTIES FIXTURES_REQUIRED ${target}_trexio FIXTURES_SETUP ${target}_trexio_rt)
  set_tests_properties(${target}_trexio_reexport PROPERTIES FIXTURES_REQUIRED ${target}_trexio_rt FIXTURES_SETUP ${target}_trexio_rt2)
  set(required basis nucleus_repulsion electron_count mo_orthonormality)
  trexio_validate_add_test(${target}_trexio_validate
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${target}.h5
    FIXTURES_REQUIRED ${target}_trexio REQUIRE ${required})
  trexio_validate_add_test(${target}_trexio_revalidate
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${target}_trexio.h5
    FIXTURES_REQUIRED ${target}_trexio_rt2 REQUIRE ${required})
endfunction()

if(TARGET erkale_trexio AND COMMAND trexio_validate_add_test)
  add_test(build_erkale_trexio "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target erkale_trexio trexio-validate)
  run_trexio(H2O_tz_pbe)
  run_trexio(H2O+_tz_hf)
  run_trexio(Mg+_rohf_tz)
  run_trexio(H2O_qz)
endif()
