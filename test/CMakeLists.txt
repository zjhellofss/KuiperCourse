find_package(GTest REQUIRED)

aux_source_directory(../test DIR_TEST)
set(link_lib  glog gtest pthread)

add_executable(test_kuiper_course ${DIR_TEST})
link_directories( /usr/local/lib/)
target_link_libraries(test_kuiper_course ${link_lib})