# In-process unit / integral / force tests. Not installed.
foreach(t basictests chkcompare integraltest cdforcetest)
  add_executable(${t} ${t}.cpp)
  set_target_properties(${t} PROPERTIES OUTPUT_NAME "${t}${SUFFIX}")
  target_link_libraries(${t} liberkale_emd liberkale)
endforeach()
