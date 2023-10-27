cmake_minimum_required(VERSION 3.17)

project(PHBF_CPP)

set(CMAKE_CXX_STANDARD 20)

include_directories("/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/")

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/utils)

aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

add_executable(PHBF_CPP ${DIR_SRCS})