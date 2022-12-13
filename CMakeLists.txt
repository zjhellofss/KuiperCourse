cmake_minimum_required(VERSION 3.23)
project(kuiper_course)

set(CMAKE_CXX_STANDARD 17)


set(link_lib glog pthread gflags gtest)
set(link_math_lib armadillo blas lapack)
add_executable(kuiper_course main.cpp)

target_include_directories(kuiper_course PUBLIC /usr/include/armadillo_bits)
target_include_directories(kuiper_course PUBLIC ${GTEST_INCLUDE_DIRS})
