# Interop tools for external programs (Gaussian fchk, ADF).
add_executable(erkale_fchkpt fchkpt.cpp)
set_target_properties(erkale_fchkpt PROPERTIES OUTPUT_NAME "erkale_fchkpt${SUFFIX}")
target_link_libraries(erkale_fchkpt liberkale)

add_executable(erkale_adf_emd adf/main.cpp adf/ArrayList.c adf/KFReader.c)
set_target_properties(erkale_adf_emd PROPERTIES OUTPUT_NAME "erkale_adf_emd${SUFFIX}")
target_link_libraries(erkale_adf_emd liberkale_emd liberkale)

install(TARGETS erkale_fchkpt erkale_adf_emd DESTINATION bin)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/external)
