cmake_minimum_required(VERSION 3.5.1)

project(CUBICS)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Available builds")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug)

set(PLATFORM "CPU" CACHE STRING "Available platforms")
set_property(CACHE PLATFORM PROPERTY STRINGS CPU GPU)

#Add headers directory
include_directories("./include" "./contrib")

#Collect source files
file(GLOB_RECURSE SOURCES "./src/*.cu" "./contrib/*.cu")

#Print informations
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform type: ${PLATFORM}")

#Set common compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g0 -O3")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")

if(${PLATFORM} STREQUAL "CPU")
    #Check GCC version
    if ((NOT CMAKE_COMPILER_IS_GNUCC) OR (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4))
        message(FATAL_ERROR "Required GCC >= 5.4")
    endif()

    #Compile .cu files as .cpp files
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCES} PROPERTIES COMPILE_FLAGS "-x c++")
    
    #Add target executable
    add_executable(cubics ${SOURCES})
    
    #Link objects as C++ objects
    set_target_properties(cubics PROPERTIES LINKER_LANGUAGE CXX)
else()
    #Check CUDA presence
    find_package(CUDA 7.5 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    
    #Define the macro GPU
    add_definitions(-DGPU)

    #Set CUDA compilation flags
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=compute_35")
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
    endif()

    #Enable compilation of CUDA code independently from C++ code
    set(CUDA_SEPARABLE_COMPILATION ON)

    #Add target executable
    cuda_add_executable(cubics ${SOURCES})

    #Bugfix for CMake 3.5.1: add CUDA Device Runtime library
    cuda_find_library_local_first(CUDA_cudadevrt_LIBRARY cudadevrt "CUDA Device Runtime library")
    target_link_libraries(cubics ${CUDA_cudadevrt_LIBRARY})
endif()
