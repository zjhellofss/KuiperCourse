find_package(GTest REQUIRED)

aux_source_directory(../test DIR_TEST)
set(link_lib glog gtest pthread)
set(link_math_lib armadillo blas lapack)

aux_source_directory(../source/data DIR_DATA)
aux_source_directory(../source/ops DIR_OP)
aux_source_directory(../source/layer DIR_LAYER)
aux_source_directory(../source/factory DIR_FAC)
aux_source_directory(../source/runtime DIR_RUNTIME)

add_executable(test_kuiper_course ${DIR_TEST}  ${DIR_RUNTIME} ${DIR_DATA} ${DIR_OP} ${DIR_LAYER} ${DIR_FAC})

link_directories(/usr/local/lib/)
target_link_libraries(test_kuiper_course ${link_lib} ${link_math_lib})
