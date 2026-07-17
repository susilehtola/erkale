# Electron momentum density library and tool.
add_library(liberkale_emd emd.cpp emd_gto.cpp emd_sto.cpp emdcube.cpp
  gto_fourier.cpp spherical_expansion.cpp emd_similarity.cpp)
set_target_properties(liberkale_emd PROPERTIES OUTPUT_NAME "erkale_emd${SUFFIX}")
if(BUILD_SHARED_LIBS)
  set_target_properties(liberkale_emd PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${SOVERSION})
endif()
target_link_libraries(liberkale_emd PUBLIC liberkale)
install(TARGETS liberkale_emd DESTINATION lib${LIB_SUFFIX})

add_executable(erkale_emd main.cpp)
set_target_properties(erkale_emd PROPERTIES OUTPUT_NAME "erkale_emd${SUFFIX}")
target_link_libraries(erkale_emd liberkale_emd)
install(TARGETS erkale_emd DESTINATION bin)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/emd)
