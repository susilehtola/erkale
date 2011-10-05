add_executable(erkale_slaterfit solve_coefficients.cpp form_exponents.cpp)
# The name of the executable is
set_target_properties(erkale_slaterfit PROPERTIES OUTPUT_NAME "erkale_slaterfit${SUFFIX}")

target_link_libraries(erkale_slaterfit liberkale)
# These are needed anyhow.
target_link_libraries(erkale_slaterfit ${GSL_LIBRARIES})
target_link_libraries(erkale_slaterfit ${LAPACK_LIBRARIES})

install (TARGETS erkale_slaterfit DESTINATION bin)
