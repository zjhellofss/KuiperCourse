find_package(GTest REQUIRED)

aux_source_directory(../test DIR_TEST)
set(link_lib glog gtest pthread  )
set(link_math_lib armadillo blas lapack)

aux_source_directory(../source/data DIR_DATA)
add_executable(test_kuiper_course ${DIR_TEST} ${DIR_DATA})

link_directories(/usr/local/lib/)
target_link_libraries(test_kuiper_course ${link_lib} ${link_math_lib})
