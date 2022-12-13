find_package(GTest REQUIRED)

aux_source_directory(../test DIR_TEST)
set(link_lib pthread glog gtest)

add_executable(test_kuiper_course ${DIR_TEST})
target_link_libraries(test_kuiper_course ${link_lib})