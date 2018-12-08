cmake_minimum_required(VERSION 2.7)


find_package(CUDA QUIET REQUIRED)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
list(APPEND CMAKE_CXX_FLAGS "-pthread -std=c++11 ${CMAKE_CXX_FLAGS}")

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(p31 ./src/main1.cc)

add_executable(p32 ./src/main2.cc)

cuda_add_executable(p33 ./src/main3.cu)