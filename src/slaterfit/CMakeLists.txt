# Slater -> Gaussian fitting tool.
add_executable(erkale_slaterfit main.cpp)
set_target_properties(erkale_slaterfit PROPERTIES OUTPUT_NAME "erkale_slaterfit${SUFFIX}")
target_link_libraries(erkale_slaterfit liberkale)

install(TARGETS erkale_slaterfit DESTINATION bin)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/slaterfit)
