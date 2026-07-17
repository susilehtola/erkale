# X-ray spectra (XRS) library and executable.
add_library(liberkale_xrs bfprod.cpp fourierprod.cpp lmtrans.cpp
  momentum_series.cpp xrsscf-base.cpp xrsscf-fock.cpp xrsscf-solvers.cpp)
set_target_properties(liberkale_xrs PROPERTIES OUTPUT_NAME "erkale_xrs${SUFFIX}")
if(BUILD_SHARED_LIBS)
  set_target_properties(liberkale_xrs PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${SOVERSION})
endif()
# Own source dir (headers) and binary dir (generated xrsscf-*.cpp).
target_include_directories(liberkale_xrs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_link_libraries(liberkale_xrs PUBLIC liberkale liberkale_emd)

add_executable(erkale_xrs main.cpp)
set_target_properties(erkale_xrs PROPERTIES OUTPUT_NAME "erkale_xrs${SUFFIX}")
target_link_libraries(erkale_xrs liberkale_xrs)

install(TARGETS liberkale_xrs DESTINATION lib${LIB_SUFFIX})
install(TARGETS erkale_xrs DESTINATION bin)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/xrs)

# --- XRS SCF code generation ---
# The XRS SCF specialisations reuse the parent scf-*.cpp.in templates,
# preprocessed with -DHALFHOLE / -DFULLHOLE, prepended with the shared and
# the XRS-specific include headers.
set(SCFDIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(XRSINC ${CMAKE_CURRENT_SOURCE_DIR}/xrsscf-includes.cpp.in)

function(erkale_gen_xrs name tmpl)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp)
  add_custom_command(OUTPUT ${out}
    DEPENDS ${SCFDIR}/${tmpl} ${SCFDIR}/scf-includes.cpp.in ${XRSINC}
    COMMAND ${CAT} ${SCFDIR}/scf-includes.cpp.in >${out}
    COMMAND ${CAT} ${XRSINC} >>${out}
    COMMAND ${PPROC} -DDFT -DHALFHOLE ${SCFDIR}/${tmpl} >>${out}
    COMMAND ${PPROC} -DDFT -DFULLHOLE ${SCFDIR}/${tmpl} >>${out})
endfunction()

erkale_gen_xrs(xrsscf-fock    scf-fock.cpp.in)
erkale_gen_xrs(xrsscf-solvers scf-solvers.cpp.in)
