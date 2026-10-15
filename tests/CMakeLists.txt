# Each test is already designed to run in parallel so ctest should not parallellize
set(CTEST_PARALLEL_LEVEL 1)

# The input of a test: in the source directory, or derived from that of
# another test in the build directory
function(test_input target var)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run)
    set(${var} ${CMAKE_CURRENT_SOURCE_DIR}/${target}.run PARENT_SCOPE)
  else()
    set(${var} ${CMAKE_CURRENT_BINARY_DIR}/${target}.run PARENT_SCOPE)
  endif()
endfunction()

# Derive the input of target from that of the test source: its output
# files are renamed for target, and the keyword lines in ARGN are
# appended, which take precedence over those of source.
function(derived_input source target)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/${source}.run input)
  string(REPLACE "${source}." "${target}." input "${input}")
  string(APPEND input "\n")
  foreach(line IN LISTS ARGN)
    string(APPEND input "${line}\n")
  endforeach()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${target}.run "${input}")
endfunction()

# Each test provides its checkpoint as a fixture. The checkpoint is
# compared against the reference of the same name, or of REF; a test
# that starts from the checkpoint of another test gives it in LOAD.
function(run_test target)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "REF;LOAD" "")
  if(NOT arg_REF)
    set(arg_REF ${target})
  endif()
  test_input(${target} input)

  # Run ERKALE to get checkpoint file
  add_test(NAME ${target}_chk COMMAND erkale ${input})
  set_property(TEST ${target}_chk APPEND PROPERTY ENVIRONMENT "ERKALE_SYSDIR=${CMAKE_CURRENT_SOURCE_DIR}/xyz")
  set_property(TEST ${target}_chk APPEND PROPERTY ENVIRONMENT "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  set_property(TEST ${target}_chk APPEND PROPERTY ENVIRONMENT "ERKALE_SAP_LIBRARY=${PROJECT_SOURCE_DIR}/sap")
  set_property(TEST ${target}_chk APPEND PROPERTY DEPENDS erkale)
  set_property(TEST ${target}_chk APPEND PROPERTY DEPENDS ${input})
  set_property(TEST ${target}_chk APPEND PROPERTY FIXTURES_SETUP ${target})
  if(arg_LOAD)
    set_property(TEST ${target}_chk APPEND PROPERTY FIXTURES_REQUIRED ${arg_LOAD})
  endif()

  # Run comparison against checkpoint file
  add_test(NAME ${target} COMMAND chkcompare ${target}.chk ${arg_REF}.chk)
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
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "LOAD" "")
  test_input(${target} input)
  add_test(NAME ${target} COMMAND erkale_neo ${input})
  set_property(TEST ${target} APPEND PROPERTY FIXTURES_SETUP ${target})
  if(arg_LOAD)
    set_property(TEST ${target} APPEND PROPERTY FIXTURES_REQUIRED ${arg_LOAD})
  endif()
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_SYSDIR=${CMAKE_CURRENT_SOURCE_DIR}/xyz")
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  set_property(TEST ${target} APPEND PROPERTY DEPENDS erkale_neo)
  set_property(TEST ${target} APPEND PROPERTY DEPENDS ${input})
  set_property(TEST ${target} PROPERTY PASS_REGULAR_EXPRESSION "Converged to energy ${energy}")
endfunction()

# Linear-symmetry SCF with erkale_complex_orbs, checked against the
# energy erkale gives for the same calculation.
function(run_complex_orbs target energy)
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "LOAD" "")
  test_input(${target} input)
  add_test(NAME ${target} COMMAND erkale_complex_orbs ${input})
  set_property(TEST ${target} APPEND PROPERTY FIXTURES_SETUP ${target})
  if(arg_LOAD)
    set_property(TEST ${target} APPEND PROPERTY FIXTURES_REQUIRED ${arg_LOAD})
  endif()
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_SYSDIR=${CMAKE_CURRENT_SOURCE_DIR}/xyz")
  set_property(TEST ${target} APPEND PROPERTY ENVIRONMENT "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  set_property(TEST ${target} APPEND PROPERTY DEPENDS erkale_complex_orbs)
  set_property(TEST ${target} PROPERTY PASS_REGULAR_EXPRESSION "Total energy is ${energy}")
endfunction()
# Also check the term symbol (a regular expression: escape the + of Sigma+)
function(run_complex_orbs_term target term energy)
  run_complex_orbs(${target} ${energy})
  set_property(TEST ${target} PROPERTY PASS_REGULAR_EXPRESSION "Term symbol ${term} .*Total energy is ${energy}")
endfunction()

# First, compile and run the basic test
add_test(build_basictests  "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target basictests)
add_test(NAME basictests COMMAND basictests)

# Analytic SCF forces against finite differences of the total energy
add_test(build_forcetest "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target forcetest)
add_test(NAME forcetest COMMAND forcetest)
set_property(TEST forcetest APPEND PROPERTY ENVIRONMENT "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")

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
# Multicomponent DFT with electron-proton correlation (restricted and
# unrestricted electrons); the EPC functional contributes to both the
# electronic and the protonic Fock matrices.
if(ERKALE_LIBXC_HAS_EPC)
  run_neo(neo_dft_epc "-75.84436")
  run_neo(neo_dft_epc_u "-75.38644")
endif()
# Multicomponent DFT with a range-separated hybrid for the electrons
run_neo(neo_wb97x "-76.36052")
# Density fitting in the automatic electron-proton auxiliary basis
run_neo(neo_ri "-76.01543")
# Unrestricted electrons in triplet methylene
run_neo(neo_triplet "-38.88650")
# The initial guesses: the core Hamiltonian, and the orbitals of a
# multicomponent calculation in another basis or of an electronic one of
# erkale; the triplet also from a restricted calculation of the singlet
derived_input(neo_proton neo_proton_guess_Core "Guess Core")
run_neo(neo_proton_guess_Core "-75.98665")
derived_input(neo_proton neo_proton_guess_Load "Guess Load" "LoadChk neo_ri.chk")
run_neo(neo_proton_guess_Load "-75.98665" LOAD neo_ri)
derived_input(neo_proton neo_proton_guess_Load_erkale "Guess Load" "LoadChk H2O_tz_hf.chk")
run_neo(neo_proton_guess_Load_erkale "-75.98665" LOAD H2O_tz_hf)
derived_input(neo_triplet neo_triplet_guess_Core "Guess Core")
run_neo(neo_triplet_guess_Core "-38.88650")
derived_input(neo_triplet neo_triplet_guess_Load "Guess Load" "LoadChk CH2_dz_hf.chk")
run_neo(neo_triplet_guess_Load "-38.88650" LOAD CH2_dz_hf)
derived_input(neo_triplet neo_triplet_guess_Load_restricted "Guess Load" "LoadChk CH2_dz_rhf.chk")
run_neo(neo_triplet_guess_Load_restricted "-38.88650" LOAD CH2_dz_rhf)

# Linear symmetry in the complex basis, restricted and unrestricted DFT
add_test(build_erkale_complex_orbs "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target erkale_complex_orbs)
run_complex_orbs(complex_orbs_n2_b3lyp "-109.53324")
run_complex_orbs(complex_orbs_o2_tpss "-150.35981")
# Range-separated hybrids
run_complex_orbs(complex_orbs_n2_wb97x "-109.5077902")
run_complex_orbs(complex_orbs_o2_camb3lyp "-150.2878160")
# States erkale cannot represent: boron with its p electron in the
# complex m=+1 orbital, without and with a magnetic field. The occupation
# file is read from the working directory.
configure_file(complex_orbs_b_ml.linocc ${CMAKE_CURRENT_BINARY_DIR}/complex_orbs_b_ml.linocc COPYONLY)
configure_file(complex_orbs_be_hf_saddle.linocc ${CMAKE_CURRENT_BINARY_DIR}/complex_orbs_be_hf_saddle.linocc COPYONLY)
configure_file(complex_orbs_be_hf_inversion.linocc ${CMAKE_CURRENT_BINARY_DIR}/complex_orbs_be_hf_inversion.linocc COPYONLY)
run_complex_orbs(complex_orbs_b_ml_0 "-24.3478542")
run_complex_orbs(complex_orbs_b_ml_p "-24.3943705")
run_complex_orbs(complex_orbs_b_ml_m "-24.2943705")
# Current-corrected meta-GGA (TPSS) in a field, without and with the
# trust-region solver
run_complex_orbs(complex_orbs_b_tpss_ml_m "-24.6048081")
# Cylindrical symmetry: m and -m share their orbitals and occupations
run_complex_orbs(complex_orbs_o2_b3lyp_cyl "-150.23617")
run_complex_orbs_term(complex_orbs_o2_b3lyp_triplet "3Sigma_g-" "-150.334038")
run_complex_orbs_term(complex_orbs_c_hund "3Pi_g" "-37.457485")
# The initial guesses: the core Hamiltonian, and the Fock matrices of a
# calculation in another basis, with another functional, or unrestricted
# without cylindrical symmetry for a restricted one with it
derived_input(complex_orbs_n2_b3lyp complex_orbs_n2_b3lyp_guess_Core "Guess Core")
run_complex_orbs(complex_orbs_n2_b3lyp_guess_Core "-109.53324")
derived_input(complex_orbs_n2_b3lyp complex_orbs_n2_b3lyp_tz "Basis cc-pVTZ")
run_complex_orbs(complex_orbs_n2_b3lyp_tz "-109.568317")
derived_input(complex_orbs_n2_b3lyp complex_orbs_n2_b3lyp_guess_Load "Guess Load" "LoadChk complex_orbs_n2_b3lyp_tz.chk")
run_complex_orbs(complex_orbs_n2_b3lyp_guess_Load "-109.53324" LOAD complex_orbs_n2_b3lyp_tz)
derived_input(complex_orbs_o2_tpss complex_orbs_o2_tpss_guess_Core "Guess Core")
run_complex_orbs(complex_orbs_o2_tpss_guess_Core "-150.35981")
derived_input(complex_orbs_o2_tpss complex_orbs_o2_tpss_guess_Load "Guess Load" "LoadChk complex_orbs_o2_b3lyp_triplet.chk")
run_complex_orbs(complex_orbs_o2_tpss_guess_Load "-150.35981" LOAD complex_orbs_o2_b3lyp_triplet)
derived_input(complex_orbs_o2_b3lyp_cyl complex_orbs_o2_b3lyp_cyl_guess_Load "Guess Load" "LoadChk complex_orbs_o2_b3lyp_triplet.chk")
run_complex_orbs(complex_orbs_o2_b3lyp_cyl_guess_Load "-150.23617" LOAD complex_orbs_o2_b3lyp_triplet)
# Inversion symmetry: the orbitals of each m are blocked further into
# gerade and ungerade ones
run_complex_orbs(complex_orbs_o2_b3lyp_cyl_inversion "-150.23617")
# Guess Load between calculations with and without the parity blocks
derived_input(complex_orbs_o2_b3lyp_cyl_inversion complex_orbs_o2_b3lyp_cyl_inversion_guess_Load "Guess Load" "LoadChk complex_orbs_o2_b3lyp_triplet.chk")
run_complex_orbs(complex_orbs_o2_b3lyp_cyl_inversion_guess_Load "-150.23617" LOAD complex_orbs_o2_b3lyp_triplet)
derived_input(complex_orbs_o2_b3lyp_cyl complex_orbs_o2_b3lyp_cyl_guess_Load_inversion "Guess Load" "LoadChk complex_orbs_o2_b3lyp_cyl_inversion.chk")
run_complex_orbs(complex_orbs_o2_b3lyp_cyl_guess_Load_inversion "-150.23617" LOAD complex_orbs_o2_b3lyp_cyl_inversion)

# Second-order trust-region SCF and stability analysis (OpenTrustRegion)
if(ERKALE_TRUST_REGION)
  run_complex_orbs(complex_orbs_h2_instability "-1.0027839")
  run_neo(neo_proton_tr "-75.98665")
  run_complex_orbs(complex_orbs_n2_b3lyp_tr "-109.53324")
  run_complex_orbs(complex_orbs_c_pbe_tr "-37.737104")
  run_complex_orbs(complex_orbs_b_tpss_ml_p "-24.7048081")
  run_complex_orbs(complex_orbs_fe2_pbe_tr "-2526.386959")
  run_complex_orbs(complex_orbs_be_hf_saddle "-14.57261104")
  run_complex_orbs_term(complex_orbs_h2_inversion "1Sigma_g\\+" "-0.92190859")
  run_complex_orbs_term(complex_orbs_be_hf_inversion "1Sigma_g\\+" "-14.57233763")
  if(ERKALE_LIBXC_HAS_EPC)
    run_neo(neo_dft_epc_tr "-75.84436")
  endif()
endif()

# Projection-free guess on a change of basis, restricted DFT and
# unrestricted Hartree-Fock
run_test(H2O_projfree_pbe LOAD H2O_tz_pbe)
run_test(H2O+_projfree_hf LOAD H2O+_tz_hf)

# Every initial guess for the closed-shell water molecule and triplet
# methylene, whose orbitals are not degenerate and from which every guess
# converges to the solution of the reference calculation with the default
# guess (SAD). The Load and ProjFree guesses start from a calculation in a
# smaller basis, which for methylene is also an unrestricted or a
# restricted singlet one, and Auto is Load when LoadChk is given.
foreach(ref H2O_tz_hf CH2_tz_hf)
  foreach(guess Core SADNO GWH Huckel SAP SAPfit)
    derived_input(${ref} ${ref}_guess_${guess} "Guess ${guess}")
    run_test(${ref}_guess_${guess} REF ${ref})
  endforeach()
endforeach()
foreach(guess Load ProjFree)
  derived_input(H2O_tz_hf H2O_tz_hf_guess_${guess} "Guess ${guess}" "LoadChk H2O_qz.chk")
  run_test(H2O_tz_hf_guess_${guess} REF H2O_tz_hf LOAD H2O_qz)
  derived_input(CH2_tz_hf CH2_tz_hf_guess_${guess} "Guess ${guess}" "LoadChk CH2_dz_hf.chk")
  run_test(CH2_tz_hf_guess_${guess} REF CH2_tz_hf LOAD CH2_dz_hf)
  derived_input(CH2_tz_hf CH2_tz_hf_guess_${guess}_restricted "Guess ${guess}" "LoadChk CH2_dz_rhf.chk")
  run_test(CH2_tz_hf_guess_${guess}_restricted REF CH2_tz_hf LOAD CH2_dz_rhf)
endforeach()
derived_input(CH2_tz_hf CH2_tz_hf_guess_Auto "LoadChk CH2_dz_hf.chk")
run_test(CH2_tz_hf_guess_Auto REF CH2_tz_hf LOAD CH2_dz_hf)

# Generate list with genlist.sh
include(TestList.txt)

# TREXIO export: write TREXIO files from the checkpoints of a few of the
# tests above (restricted, unrestricted and ROHF; spherical and cartesian;
# up to g functions) and
# validate them with trexio-validate. Each file is also imported back and
# re-exported, and the result validated, which exercises the import. The
# one-electron integrals are always exported and checked; with the ERI
# option, the AO electron repulsion integrals are too (trexio-validate
# checks them for at most 64 AOs).
#
# With FROM, the TREXIO file is given instead of exported from a
# checkpoint, and the round trip starts with its import.
function(run_trexio target)
  cmake_parse_arguments(PARSE_ARGV 1 arg "ERI" "FROM" "")
  set(trexio_env "ERKALE_LIBRARY=${PROJECT_SOURCE_DIR}/basis")
  set(export export)
  set(required basis nucleus_repulsion electron_count mo_orthonormality)
  foreach(op overlap kinetic potential_n_e core_hamiltonian dipole_x dipole_y dipole_z)
    list(APPEND required ao_1e_int_${op} mo_1e_int_${op})
    list(APPEND required_mo mo_1e_int_${op})
  endforeach()
  if(arg_ERI)
    list(APPEND export --eri)
    list(APPEND required ao_2e_int_eri)
  endif()

  if(arg_FROM)
    set(src ${arg_FROM})
    # A given file need not carry the AO integrals
    set(required_src basis nucleus_repulsion electron_count mo_orthonormality ${required_mo})
  else()
    set(required_src ${required})
    set(src ${CMAKE_CURRENT_BINARY_DIR}/${target}.h5)
    add_test(NAME ${target}_trexio_export COMMAND erkale_trexio ${export} ${target}.chk ${src})
    set_tests_properties(${target}_trexio_export PROPERTIES ENVIRONMENT "${trexio_env}"
      DEPENDS ${target}_chk FIXTURES_SETUP ${target}_trexio)
  endif()
  add_test(NAME ${target}_trexio_import COMMAND erkale_trexio import ${src} ${target}_trexio.chk)
  add_test(NAME ${target}_trexio_reexport COMMAND erkale_trexio ${export} ${target}_trexio.chk ${target}_trexio.h5)
  set_tests_properties(${target}_trexio_import ${target}_trexio_reexport
    PROPERTIES ENVIRONMENT "${trexio_env}")
  set_tests_properties(${target}_trexio_import PROPERTIES FIXTURES_REQUIRED ${target}_trexio FIXTURES_SETUP ${target}_trexio_rt)
  set_tests_properties(${target}_trexio_reexport PROPERTIES FIXTURES_REQUIRED ${target}_trexio_rt FIXTURES_SETUP ${target}_trexio_rt2)
  trexio_validate_add_test(${target}_trexio_validate
    FILE ${src}
    FIXTURES_REQUIRED ${target}_trexio REQUIRE ${required_src})
  trexio_validate_add_test(${target}_trexio_revalidate
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${target}_trexio.h5
    FIXTURES_REQUIRED ${target}_trexio_rt2 REQUIRE ${required})
endfunction()

if(TARGET erkale_trexio AND COMMAND trexio_validate_add_test)
  add_test(build_erkale_trexio "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target erkale_trexio trexio-validate)
  run_trexio(H2O_tz_pbe ERI)
  run_trexio(H2O+_tz_hf ERI)
  run_trexio(Mg+_rohf_tz ERI)
  run_trexio(H2O_qz)
  # Spherical p functions (OptLM false): libcint's (x, y, z) order
  run_trexio(H2O_tz_pbe_sphp ERI)
  # Cartesian d and higher functions (UseLM false); trexio-validate
  # checks the ERIs only of the smaller Ne
  run_trexio(H2O_tz_pbe_cart)
  run_trexio(Ne_tz_cart ERI)
  # A basis mixing spherical and cartesian shells, flagged per shell with
  # ao.cartesian_shell: the H2O_tz_pbe wavefunction with the oxygen f
  # shell rewritten as cartesian, in another normalization convention.
  # The file carries only the MO-basis integrals.
  if(ERKALE_TREXIO_HAS_CARTESIAN_SHELL)
    run_trexio(H2O_tz_pbe_mixed ERI FROM ${CMAKE_CURRENT_SOURCE_DIR}/H2O_tz_pbe_mixed.h5)
  endif()
endif()
