# Basis-set manipulation tool.
add_executable(erkale_bastool main.cpp pivoted_cholesky_basis.cpp)
set_target_properties(erkale_bastool PROPERTIES OUTPUT_NAME "erkale_bastool${SUFFIX}")
target_link_libraries(erkale_bastool liberkale_cmp liberkale)

install(TARGETS erkale_bastool DESTINATION bin)
