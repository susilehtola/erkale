# Casida linear-response TDDFT executable.
add_executable(erkale_casida main.cpp casida.cpp casida_grid.cpp)
set_target_properties(erkale_casida PROPERTIES OUTPUT_NAME "erkale_casida${SUFFIX}")
target_link_libraries(erkale_casida liberkale_xrs liberkale_emd liberkale)

install(TARGETS erkale_casida DESTINATION bin)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/casida)
