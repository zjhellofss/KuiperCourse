cmake_minimum_required(VERSION 3.16)
project(kuiper_course)

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)

set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)
add_executable(kuiper_course main.cpp)

target_include_directories(kuiper_course PUBLIC /usr/include/armadillo_bits)
target_include_directories(kuiper_course PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(kuiper_course ${link_lib} ${link_math_lib})

enable_testing()
add_subdirectory(test)