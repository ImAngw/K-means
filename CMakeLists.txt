cmake_minimum_required(VERSION 3.23)
project(k_means C)

set(CMAKE_C_STANDARD 99)

add_executable(k_means k_means_functions.c k_means_functions.h profiler_functions.c main.c)

find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")


if(OpenMP_C_FOUND)
    target_compile_options(k_means PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(k_means PRIVATE OpenMP::OpenMP_C)
else()
    message(FATAL_ERROR "OpenMP not found. Make sure your compiler supports OpenMP.")
endif()