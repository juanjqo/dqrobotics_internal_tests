cmake_minimum_required(VERSION 3.5)

project(cpp_interface_qpoases_pull_1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(proxsuite REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
                      dqrobotics
                      qpOASES
                      proxsuite::proxsuite

)
