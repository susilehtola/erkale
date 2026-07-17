# Completeness-profile library and tools.
add_library(liberkale_cmp completeness_profile.cpp optimize_completeness.cpp)
set_target_properties(liberkale_cmp PROPERTIES OUTPUT_NAME "erkale_cmp${SUFFIX}")
if(BUILD_SHARED_LIBS)
  set_target_properties(liberkale_cmp PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${SOVERSION})
endif()
target_link_libraries(liberkale_cmp PUBLIC liberkale)

add_executable(erkale_copt main.cpp)
set_target_properties(erkale_copt PROPERTIES OUTPUT_NAME "erkale_copt${SUFFIX}")
target_link_libraries(erkale_copt liberkale_cmp)

add_executable(erkale_copt_plateau plateau.cpp)
set_target_properties(erkale_copt_plateau PROPERTIES OUTPUT_NAME "erkale_copt_plateau${SUFFIX}")
target_link_libraries(erkale_copt_plateau liberkale_cmp)

install(TARGETS erkale_copt erkale_copt_plateau DESTINATION bin)
install(TARGETS liberkale_cmp DESTINATION lib${LIB_SUFFIX})
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION include/erkale/completeness)
