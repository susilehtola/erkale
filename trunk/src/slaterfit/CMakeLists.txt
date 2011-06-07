add_executable(slaterfit solve_coefficients.cpp form_exponents.cpp)
target_link_libraries(slaterfit erkale)
target_link_libraries(slaterfit ${GSL_LIBRARIES})
target_link_libraries(slaterfit ${LAPACK_LIBRARIES})

install (TARGETS slaterfit DESTINATION bin)
