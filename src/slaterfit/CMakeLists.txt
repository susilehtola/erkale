add_executable(erkale_slaterfit solve_coefficients.cpp form_exponents.cpp)
target_link_libraries(erkale_slaterfit erkale)
# These are needed anyhow.
target_link_libraries(erkale_slaterfit ${GSL_LIBRARIES})
target_link_libraries(erkale_slaterfit ${LAPACK_LIBRARIES})

install (TARGETS erkale_slaterfit DESTINATION bin)
