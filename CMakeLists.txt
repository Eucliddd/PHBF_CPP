cmake_minimum_required(VERSION 3.17)

project(PHBF_CPP)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp -DMKL_ILP64 -m64 -std=c++2a -fconcepts")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# add_compile_options(-DMKL_ILP64  -m64)

set(MKL_LIBRARIES
    -L${MKLROOT}/lib/intel64
    -Wl,--no-as-needed
    -lmkl_intel_ilp64
    -lmkl_gnu_thread
    -lmkl_core
    -lgomp
    -lpthread
    -lm
    -ldl
)

add_subdirectory(${PROJECT_SOURCE_DIR}/HABF/util)

include_directories("/usr/local/include/eigen3")

include_directories(${MKLROOT}/include)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/utils)

include_directories(${PROJECT_SOURCE_DIR}/HABF/habf)

aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

add_executable(PHBF_CPP ${DIR_SRCS})

target_link_libraries(PHBF_CPP PRIVATE ${MKL_LIBRARIES})

target_link_libraries(PHBF_CPP PRIVATE -lstdc++fs)

target_link_libraries(PHBF_CPP PRIVATE util)